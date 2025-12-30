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
from omegaconf import OmegaConf

from .DeblurDiff.model.cldm import ControlLDM
from .DeblurDiff.model.gaussian_diffusion import Diffusion

from .DeblurDiff.utils.common import instantiate_from_config, load_file_from_url, count_vram_usage
from .DeblurDiff.utils.pipeline import (
    Pipeline,
    bicubic_resize
)
from .DeblurDiff.utils.cond_fn import MSEGuidance, WeightedMSEGuidance

import cv2
import numpy as np
from scipy import ndimage
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot as show_seg
from mmseg.core.evaluation import get_palette

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
        print('version: huawei_hard_self_20251102')
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


        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.pipeline: Pipeline = None
        self.init_model()
        self.init_pipeline()

        # loss
        self.pixel_loss = build_loss(pixel_loss)
    
    @count_vram_usage
    def init_model(self) -> None:
        self.cldm: ControlLDM = instantiate_from_config(OmegaConf.load("mmedit/models/restorers/DeblurDiff/configs/inference/cldm.yaml"))
        self.cldm.load_state_dict(torch.load('chkpts/model.pth'))
        self.cldm.eval().to(self.device)
        ### load diffusion
        self.diffusion: Diffusion = instantiate_from_config(OmegaConf.load("mmedit/models/restorers/DeblurDiff/configs/inference/diffusion.yaml"))
        self.diffusion.to(self.device)

    def init_pipeline(self) -> None:
        self.pipeline = Pipeline(self.cldm, self.diffusion, None, self.device)

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

    def calc_artifact_map(self, img, img2, crop_border, contrast_threshold=0.7, window_size=11, area_threshold=64):
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
                print('#################################################################')
                print(artifact_map)

                # ---- 阈值化：小于阈值视为伪影(True) ----
                mask = (artifact_map < contrast_threshold)  # bool [H,W]

                # ---- 形态学：腐蚀(1) -> 膨胀(3) -> 填洞 ----
                k5 = np.ones((5, 5), np.uint8)
                mask_u8 = (mask.astype(np.uint8)) * 255                # 0/255
                eroded  = cv2.erode(mask_u8, k5, iterations=1)
                dilated = cv2.dilate(eroded, k5, iterations=3)
                filled  = ndimage.binary_fill_holes(dilated > 0, structure=np.ones((3, 3))).astype(np.uint8)  # 0/1
                # 连通域面积过滤（仅用 area_threshold）
                num, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
                filtered = np.zeros_like(filled)
                for (i, label) in enumerate(np.unique(labels)):
                    print(stats[i][-1])
                    if label == 0:
                        continue
                    if stats[i][-1] > area_threshold:
                        filtered[labels == i] = 1
                # filtered = mask

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
                hypir_ref = self.pipeline.run(
                    output_ref_one.view(-1, C, H, W), 2, 1.0, False,
                    512, 256, "", "low quality, blurry, low-resolution, noisy, unsharp, weird textures", 1.0,
                    False
                ).view(B, 1, C, H, W)
            artifact_map = self.calc_artifact_map(output_ref_one, hypir_ref, crop_border=0)
            ref_final = artifact_map * output_ref_one + (1 - artifact_map) * hypir_ref

            self.save_wanted_images([output_ref_one, hypir_ref, ref_final, artifact_map.repeat(1, 1, 3, 1, 1)], self.sub_save_path+'_grow')

            loss_pix += ref_weight * self.pixel_loss(output[:, -1:], hypir_ref, input_=lq, iter_=self.step_counter, save_imgs_iter=self.save_imgs_iter, sub_save_path=self.sub_save_path+'_ref')
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
