# Copyright (c) OpenMMLab. All rights reserved.
import numbers
import os
import torchvision
import os.path as osp

import mmcv
from mmcv.runner import auto_fp16

from mmedit.core import psnr, ssim, tensor2img
from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS

import torch
import random
import torch.nn.functional as F
from .TwoHeadsNetwork import TwoHeadsNetwork
from .HYPIR.enhancer.sd2 import SD2Enhancer

import cv2
import numpy as np
from scipy import ndimage
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot as show_seg
from mmseg.core.evaluation import get_palette

CLASSES = [
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'windowpane', 'grass', 'cabinet', 'sidewalk',
    'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa',
    'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
    'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand',
    'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table',
    'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower',
    'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
    'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight',
    'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator',
    'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer',
    'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven',
    'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher',
    'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier',
    'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag'
]

@MODELS.register_module()
class BasicRestorer(BaseModel):
    """Basic model for image restoration.

    It must contain a generator that takes an image as inputs and outputs a
    restored image. It also has a pixel-wise loss for training.

    The subclasses should overwrite the function `forward_train`,
    `forward_test` and `train_step`.

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self,
                 generator,
                 pixel_loss,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()

        print('#############################################################################################################')
        print('version: huawei_hard_self_20251028')
        print('#############################################################################################################')

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # support fp16
        self.fp16_enabled = False

        # generator
        self.generator = build_backbone(generator)
        self.init_weights(pretrained)

        self.generator_ref = build_backbone(generator)
        self.generator_ref.eval()

        self.generator_inter = build_backbone(generator)
        self.generator_inter.eval()

        self.two_heads = TwoHeadsNetwork(25)
        self.two_heads.load_state_dict(torch.load('chkpts/TwoHeads.pkl', map_location=torch.device('cpu')), strict=True)
        self.two_heads.eval()

        mods = 'to_k,to_q,to_v,to_out.0,conv,conv1,conv2,conv_shortcut,conv_out,proj_in,proj_out,ff.net.2,ff.net.0.proj'
        mods = [m.strip() for m in mods.split(',') if m.strip()]
        self.hypir = SD2Enhancer(
            base_model_path='stabilityai/stable-diffusion-2-1-base',
            weight_path='chkpts/HYPIR_sd2.pth',
            lora_modules=mods,
            lora_rank=256,
            model_t=200,
            coeff_t=200,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        self.hypir.init_models()

        # loss
        self.pixel_loss = build_loss(pixel_loss)

    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.generator.init_weights(pretrained)
    
    def model_ema(self, decay=0.999):
        net_g_params = dict(self.generator.named_parameters())
        net_g_ema_params = dict(self.generator_ref.named_parameters())

        assert net_g_params.keys() == net_g_ema_params.keys()
        for k in net_g_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)    
        # "module.spynet.mean", "module.spynet.std"
        self.generator_ref.load_state_dict(net_g_ema_params, strict=False)

    def forward_reblur(self, sharp_estimated, kernels, masks):
        _, n_kernels, K, _ = kernels.shape
        N, C, H, W = sharp_estimated.shape

        padding = torch.nn.ReflectionPad2d(K // 2)
        sharp_estimated = padding(sharp_estimated)

        output_reblurred = []
        for num in range(N):
            output_c_reblurred = []
            for c in range(C):
                conv_output = F.conv2d(sharp_estimated[num:num + 1, c:c + 1, :, :], kernels[num].unsqueeze(1))
                output_c_reblurred.append(conv_output * masks[num:num + 1])
            output_c_reblurred = torch.stack(output_c_reblurred, dim=2)
            output_reblurred.append(output_c_reblurred)
        output_reblurred = torch.cat(output_reblurred, dim=0).sum(dim=1)

        return output_reblurred

    def degradation_process(self, blur_img, input_img, gamma_factor=2.2):
        output_ = input_img.clone()
        with torch.no_grad():
            blurry_tensor_to_compute_kernels = blur_img ** gamma_factor - 0.5
            kernels, masks = self.two_heads(blurry_tensor_to_compute_kernels)

        input_img_ph = input_img ** gamma_factor
        reblurred_ph = self.forward_reblur(input_img_ph, kernels, masks)
        reblurred = reblurred_ph ** (1.0 / gamma_factor)

        # reblurred = reblurred * 0.5 + output_ * 0.5
        return reblurred
    
    def calc_single_artifact_map(self, img, img2, window_size=11):
        """The proposed quantitative indicator in Equation 7.

        Args:
            img (ndarray): Images with range [0, 255] with order 'HWC'.
            img2 (ndarray): Images with range [0, 255] with order 'HWC'.

        Returns:
            float: artifact map of a single channel.
        """

        constant = (0.03 * 255)**2
        kernel = cv2.getGaussianKernel(window_size, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img, -1, window)[window_size // 2:-(window_size // 2),
                                            window_size // 2:-(window_size // 2)]  # valid mode for window size 11
        mu2 = cv2.filter2D(img2, -1, window)[window_size // 2:-(window_size // 2), window_size // 2:-(window_size // 2)]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        sigma1_sq = cv2.filter2D(img**2, -1, window)[window_size // 2:-(window_size // 2),
                                                    window_size // 2:-(window_size // 2)] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[window_size // 2:-(window_size // 2),
                                                    window_size // 2:-(window_size // 2)] - mu2_sq

        contrast_map = (2 * (sigma1_sq + 1e-8)**0.5 * (sigma2_sq + 1e-8)**0.5 + constant) / (
            sigma1_sq + sigma2_sq + constant)

        return contrast_map

    def calc_artifact_map(self, img, img2, crop_border, contrast_threshold=0.9, window_size=11, area_threshold=64):
        B, T, C, H, W = img.shape
        assert T == 1 or T == img2.shape[1], "img/img2 的时间维 T 不一致"

        pad = window_size // 2
        if pad > 0:
            img_2d = img.view(-1, C, H, W)
            img2_2d = img2.view(-1, C, H, W)
            img_2d  = F.pad(img_2d,  (pad, pad, pad, pad), mode="constant", value=0.0)
            img2_2d = F.pad(img2_2d, (pad, pad, pad, pad), mode="constant", value=0.0)
            H, W = H + 2 * pad, W + 2 * pad
            img  = img_2d.view(B, T, C, H, W)
            img2 = img2_2d.view(B, T, C, H, W)

        masks_B = []
        for b in range(B):
            masks_T = []
            for t in range(T):
                # ---- (C,H,W) -> (H,W,C) 的 numpy float32 ----
                x = img[b, t].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)   # [H,W,C]
                y = img2[b, t].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)  # [H,W,C]

                # ---- 每通道计算 indicator，并取均值得到 artifact_map: [H,W] ----
                per_ch = []
                for c in range(C):
                    ind = self.calc_single_artifact_map(x[..., c], y[..., c], window_size)  # -> [H,W]
                    per_ch.append(ind)
                artifact_map = np.mean(np.stack(per_ch, axis=0), axis=0)  # [H,W]

                # ---- 阈值化：小于阈值视为伪影(True) ----
                mask = (artifact_map < contrast_threshold)  # bool [H,W]

                # # ---- 形态学：腐蚀(1) -> 膨胀(3) -> 填洞 ----
                # k5 = np.ones((5, 5), np.uint8)
                # mask_u8 = (mask.astype(np.uint8)) * 255                # 0/255
                # eroded  = cv2.erode(mask_u8, k5, iterations=1)
                # dilated = cv2.dilate(eroded, k5, iterations=3)
                # filled  = ndimage.binary_fill_holes(dilated > 0, structure=np.ones((3, 3))).astype(np.uint8)  # 0/1
                # # 连通域面积过滤（仅用 area_threshold）
                # num, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
                # filtered = np.zeros_like(filled)
                # for (i, label) in enumerate(np.unique(labels)):
                #     if label == 0:
                #         continue
                #     if stats[i][-1] > area_threshold:
                #         filtered[labels == i] = 1
                filtered = mask

                # 回到 torch：(1,H,W)，与 img.dtype 对齐
                mask_t = torch.from_numpy(filtered).to(img.device).to(img.dtype).unsqueeze(0)  # [1,H,W]
                masks_T.append(mask_t)

            masks_B.append(torch.stack(masks_T, dim=0))  # [T,1,H,W]

        return torch.stack(masks_B, dim=0)  # [B,T,1,H,W]

    
    def save_wanted_images(self, image_list, sub_save_path):
        if self.step_counter == 0 or (self.step_counter + 1) % self.save_imgs_iter == 0:
            name = str(self.step_counter.item()).zfill(3)
            save_path = os.path.join(sub_save_path, name)
            os.makedirs(save_path, exist_ok=True)
            n, t, c, h, w = image_list[0].shape
            for ni in range(n):
                for ti in range(t):
                    for idx, img in enumerate(image_list):
                        torchvision.utils.save_image(img[ni:ni+1, ti], os.path.join(save_path, f'{idx}_n{ni}_t{ti}.png'))
    
    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        """Forward function.

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """

        if test_mode:
            return self.forward_test(lq, gt, **kwargs)

        return self.forward_train(lq, gt)

    def forward_train(self, lq, gt):
        """Training forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor.
        """
        output_ref = None
        output_inter = None

        if self.if_ref:
            if self.step_counter == self.ref_iter:
                print('##########################################################')
                print(self.step_counter, self.ref_iter)
                print('##########################################################')
                self.generator_ref.load_state_dict(self.generator.state_dict(), strict=True)
            if self.step_counter >= self.ref_iter:
                with torch.no_grad():
                    if self.if_ema:
                        self.model_ema()
                    output_ref = self.generator_ref(lq)

        if self.if_inter:
            if self.step_counter == self.inter_iter:
                print('##########################################################')
                print(self.step_counter, self.inter_iter)
                print('##########################################################')
            if self.step_counter >= self.inter_iter:
                self.generator_inter.load_state_dict(self.generator.state_dict(), strict=True)
                with torch.no_grad():
                    output_inter = self.generator_inter(lq)

                    if self.if_remove:
                        blur_vals = self.get_blur_vals(lq[:, 1:])
                        lq2, gt2, sharp_nums = self.remove_half_sharp(lq[:, 1:], gt[:, 1:], blur_vals)
                        lq_half = torch.cat([lq[:, :1], lq2], dim=1)
                        gt_half = torch.cat([gt[:, :1], gt2], dim=1)

                        lq_half_flip = torch.flip(lq_half, dims=[1])
                        gt_half_flip = torch.flip(gt_half, dims=[1])

                        lq = torch.cat([lq_half, lq_half_flip], dim=1)
                        gt = torch.cat([gt_half, gt_half_flip], dim=1)


        output = self.generator(lq)
        losses = dict()

        B, T, C, H, W = output.shape
        output_ref_one = output_ref[:, -1:].clone().detach()
        output_gt = output_inter[:, -1:].clone().detach().repeat(1, T, 1, 1, 1)

        output_f = self.degradation_process(output_gt.view(-1, C, H, W), output.view(-1, C, H, W)).view(B, T, C, H, W)
        loss_pix = self.pixel_loss(output_f, output_gt, input_=lq, iter_=self.step_counter, save_imgs_iter=self.save_imgs_iter, sub_save_path=self.sub_save_path)
        
        ref_weight = 1.0
        if self.if_aigc:
            with torch.no_grad():
                hypir_ref = self.hypir.enhance(
                    lq=output_ref_one.view(-1, C, H, W),
                    prompt="",
                ).view(B, 1, C, H, W)
            artifact_map = self.calc_artifact_map(output_ref_one, hypir_ref, crop_border=0)
            # artifact_map = torch.zeros_like(hypir_ref)
            ref_final = artifact_map * output_ref_one + (1 - artifact_map) * hypir_ref

            self.save_wanted_images([output_ref_one, hypir_ref, ref_final, artifact_map.repeat(1, 1, 3, 1, 1)], self.sub_save_path+'_grow')

            loss_pix += ref_weight * self.pixel_loss(output[:, -1:], ref_final, input_=lq, iter_=self.step_counter, save_imgs_iter=self.save_imgs_iter, sub_save_path=self.sub_save_path+'_ref')
        else:
            output_b = self.degradation_process(output_ref_one.view(-1, C, H, W), output[:, -1:].view(-1, C, H, W)).view(B, 1, C, H, W)
            loss_pix += ref_weight * self.pixel_loss(output_b, output_ref_one, input_=lq, iter_=self.step_counter, save_imgs_iter=self.save_imgs_iter, sub_save_path=self.sub_save_path+'_ref')

        losses['loss_pix'] = loss_pix
        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))
        return outputs

    def evaluate(self, output, gt):
        """Evaluation function.

        Args:
            output (Tensor): Model output with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            dict: Evaluation results.
        """
        crop_border = self.test_cfg.crop_border

        output = tensor2img(output)
        gt = tensor2img(gt)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            eval_result[metric] = self.allowed_metrics[metric](output, gt,
                                                               crop_border)
        return eval_result

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        output = self.generator(lq)
        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            mmcv.imwrite(tensor2img(output), save_path)

        return results

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        Args:
            img (Tensor): Input image.

        Returns:
            Tensor: Output image.
        """
        out = self.generator(img)
        return out

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        outputs.update({'log_vars': log_vars})
        return outputs

    def val_step(self, data_batch, **kwargs):
        """Validation step.

        Args:
            data_batch (dict): A batch of data.
            kwargs (dict): Other arguments for ``val_step``.

        Returns:
            dict: Returned output.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output
