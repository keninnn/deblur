import os
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import json

from pwcnet import PWCNet, get_backwarp
from searaft.core.raft import RAFT
from searaft.core.utils.utils import load_ckpt

import torchvision
import torch.nn.functional as F

import cv2
from skimage import io, measure


class TrainBlurDataset(Dataset):
    def __init__(self, root='train_blur'):
        self.root_name = root.split('/')[-1]
        self.paths = sorted(
            p for p in glob.glob(os.path.join(root, "*", "*"))
            if os.path.splitext(p)[1].lower() in [".png", ".jpg", ".jpeg"]
        )
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found under {root}/*/*")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # train_blur
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        blur = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        # train_mask
        train_mask_path = path.replace(self.root_name, 'train_mask')
        train_mask_img = Image.open(train_mask_path).convert("RGB")
        train_mask = torch.from_numpy(np.array(train_mask_img)).permute(2, 0, 1).float() / 255.0

        # train_sharp_otsu_step
        misalign_sharp_path = path.replace(self.root_name, 'train_sharp_otsu_step')
        misalign_sharp_img = Image.open(misalign_sharp_path).convert("RGB")
        misalign_sharp = torch.from_numpy(np.array(misalign_sharp_img)).permute(2, 0, 1).float() / 255.0
        align_sharp_save_path = path.replace(self.root_name, 'train_sharp_otsu_step_align')
        align_sharp_mask_save_path = path.replace(self.root_name, 'train_sharp_otsu_step_align_mask')

        # deblur
        deblur_path = path.replace(self.root_name, 'deblur')
        deblur_img = Image.open(deblur_path).convert("RGB")
        deblur = torch.from_numpy(np.array(deblur_img)).permute(2, 0, 1).float() / 255.0

        # render_result
        render_result_path = path.replace(self.root_name, 'render_result')
        if os.path.exists(render_result_path):
            render_result_img = Image.open(render_result_path).convert("RGB")
            render_result = torch.from_numpy(np.array(render_result_img)).permute(2, 0, 1).float() / 255.0
            render_result_save_path = path.replace(self.root_name, 'render_result_align')
            render_result_mask_save_path = path.replace(self.root_name, 'render_result_align_mask')
            render_result_artifact_mask = path.replace(self.root_name, 'render_result_artifact_mask')

            return {
                "blur": blur,
                "train_mask": train_mask,

                "misalign_sharp": misalign_sharp,
                "align_sharp_save_path": align_sharp_save_path,
                "align_sharp_mask_save_path": align_sharp_mask_save_path,

                "render_result": render_result,
                "render_result_save_path": render_result_save_path,
                "render_result_mask_save_path": render_result_mask_save_path,
                "render_result_artifact_mask": render_result_artifact_mask,

                "deblur": deblur
            }
        else:
            return {
                "blur": blur,
                "train_mask": train_mask,

                "misalign_sharp": misalign_sharp,
                "align_sharp_save_path": align_sharp_save_path,
                "align_sharp_mask_save_path": align_sharp_mask_save_path,

                "deblur": deblur
            }


    
def make_loader(root='train_blur', batch_size=8, num_workers=4, shuffle=True):
    ds = TrainBlurDataset(root)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=True)

def get_heatmap(info_down, info_args, info_val=0.5, scale=1):
    info = F.interpolate(info_down, scale_factor=1/scale, mode='area')
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=info_args.var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=info_args.var_min, max=0)
    heatmap = (log_b * weight).sum(dim=1, keepdim=True)

    # Normalize per-frame (batch*time)
    heatmap_min = heatmap.amin(dim=(2, 3), keepdim=True)
    heatmap_max = heatmap.amax(dim=(2, 3), keepdim=True)
    heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
    heatmap = 1 - heatmap

    # Binarize by threshold
    heatmap = (heatmap >= info_val).float()
    return heatmap

def occlusion_map(pred, offset_fwd, offset_bwd, scale=1, s=0.8):
    n_, c_, h_, w_ = pred.shape
    theta = float(max(h_, w_))

    # Create normalized coordinate grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h_, device=pred.device) / theta,
        torch.arange(0, w_, device=pred.device) / theta,
        indexing='ij'
    )
    x = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).expand(n_, 2, h_, w_).type_as(pred)

    # Apply forward and backward warping
    align, _ = get_backwarp(x, offset_fwd)
    align_back, _ = get_backwarp(align, offset_bwd)

    # Compute occlusion error map
    diff = align_back - x
    result = s * theta * torch.sqrt(torch.sum(diff ** 2, dim=1, keepdim=True))
    result = torch.clamp(result, max=1.0)

    return F.interpolate(result, scale_factor=1/scale, mode='bicubic')

def get_target_mask(args, alignnet, input_down, target_down, target, scale, if_info=True):
    # searaft的输入是0-255
    origin_input_down = input_down * 255
    origin_target_down = target_down * 255

    # searaft的光流方向与pwcnet相反
    input_fwd_down = alignnet(origin_target_down, origin_input_down, iters=args.iters, test_mode=True)
    input_bwd_down = alignnet(origin_input_down, origin_target_down, iters=args.iters, test_mode=True)

    offset_input_fwd_down = input_fwd_down['flow'][-1]
    offset_input_bwd_down = input_bwd_down['flow'][-1]

    if if_info:
        info_input_fwd_down = input_fwd_down['info'][-1]
        info_input_bwd_down = input_bwd_down['info'][-1]

        info_input_mask_fwd = get_heatmap(info_input_fwd_down, args, scale=scale)
        info_input_mask_bwd = get_heatmap(info_input_bwd_down, args, scale=scale)

    input_offset = F.interpolate(offset_input_bwd_down, scale_factor=1/scale, mode='bicubic') / scale

    input_ols_map = 1 - occlusion_map(input_down, offset_input_fwd_down, offset_input_bwd_down, scale=scale)

    align_target_input, mask_origin_input = get_backwarp(target, input_offset)
    mask_input0 = mask_origin_input * input_ols_map
    if if_info:
        align_info_mask_input, _ = get_backwarp(info_input_mask_fwd, input_offset)
        mask_input = mask_input0 * info_input_mask_bwd * align_info_mask_input
    else:
        mask_input = mask_input0

    return align_target_input, mask_input, mask_input0

def get_blur_val(frame):
        frame = frame[0].clone().permute(1, 2, 0).cpu().detach().numpy()
        frame = np.clip(frame*255, 0, 255).astype(np.uint8)
        blur_val = cv2.Laplacian(frame, cv2.CV_64F).var()
        return blur_val

def refining_region(region, binary_image, threshold=0.8):
    # 想法紀錄：
    # 匡出的region會出現黑色的地方大機率是在邊界，所以檢查四邊的黑色數量是否超過閾值，如果超過就整條砍掉。
    # 依序做左、右、上、下、左、右（左右做兩次是因為上下做完要在refine一次）
    minr, minc, maxr, maxc = region.bbox
    out_minr, out_minc, out_maxr, out_maxc = minr, minc, maxr, maxc

    # left side
    i = out_minc
    while(i < out_maxc):
        left_line = binary_image[out_minr:out_maxr, i:i+1]
        if(np.sum(left_line) / (out_maxr-out_minr)) < threshold:
            out_minc = i
            i += 1
        else:
            break
    
    # right side
    i = out_maxc
    while(i > out_minc):
        right_line = binary_image[out_minr:out_maxr, i-1:i]
        if(np.sum(right_line) / (out_maxr-out_minr)) < threshold:
            out_maxc = i
            i -= 1
        else:
            break
    
    # top side
    i = minr
    while(i < maxr):
        top_line = binary_image[i:i+1, out_minc:out_maxc]
        if(np.sum(top_line) / (out_maxc-out_minc)) < threshold:
            out_minr = i
            i += 1
        else:
            break
    
    # down side
    i = maxr
    while(i > out_minr):
        down_line = binary_image[i-1:i, out_minc:out_maxc]
        if(np.sum(down_line) / (out_maxc-out_minc)) < threshold:
            out_maxr = i
            i -= 1
        else:
            break

    # left side
    i = out_minc
    while(i < out_maxc):
        left_line = binary_image[out_minr:out_maxr, i:i+1]
        if(np.sum(left_line) / (out_maxr-out_minr)) < 0.5:
            out_minc = i
            i += 1
        else:
            break
    
    # right side
    i = out_maxc
    while(i > out_minc):
        right_line = binary_image[out_minr:out_maxr, i-1:i]
        if(np.sum(right_line) / (out_maxr-out_minr)) < 0.5:
            out_maxc = i
            i -= 1
        else:
            break   

    return out_minr, out_minc, out_maxr, out_maxc

class Configurable:
    def __init__(self, json_path, model_path):
        self.cfg = json_path
        self.model = model_path
        with open(json_path, 'r') as f:
            config_data = json.load(f)
        for key, value in config_data.items():
            setattr(self, key, value)

def aaa(props, binary_image, save_path):
    try:
        max_size = 0
        max_p = props[0]
        for p in props:
            if p.bbox_area > max_size:
                max_p = p
                max_size = p.bbox_area
        start_h, start_w, end_h, end_w = refining_region(max_p, binary_image)

        H, W = binary_image.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[binary_image] = 255

        # 防越界 + 画红色矩形框（end_h/end_w按“不含”处理）
        start_h = max(0, min(start_h, H-1)); end_h = max(1, min(end_h, H))
        start_w = max(0, min(start_w, W-1)); end_w = max(1, min(end_w, W))
        rgb[start_h:end_h, start_w]   = (255, 0, 0)
        rgb[start_h:end_h, end_w-1]   = (255, 0, 0)
        rgb[start_h, start_w:end_w]   = (255, 0, 0)
        rgb[end_h-1, start_w:end_w]   = (255, 0, 0)
        Image.fromarray(rgb).save(save_path.replace('align_mask', 'align_mask0_region'))

        mask_score = (end_h - start_h) * (end_w - start_w) / (binary_image.shape[0] * binary_image.shape[1])
    except Exception as e:
        print(e)
        mask_score = 0.0
    return mask_score

if __name__ == "__main__":
    loader = make_loader("/data5/xuhonglei/data/20251201/goproshake-gs_crop/train_blur", batch_size=1, num_workers=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = Configurable('/data5/xuhonglei/data/20251201/searaft/config/eval/spring-M.json', '/data5/xuhonglei/data/20251201/searaft/models/Tartan-C-T-TSKH-spring540x960-M.pth')
    alignnet = RAFT(args)
    load_ckpt(alignnet, args.model)
    alignnet = alignnet.to(device)
    
    for i, batch in enumerate(loader):
        print(i)
        blur = batch["blur"].to(device)
        train_mask = batch["train_mask"].to(device)
        deblur = batch["deblur"].to(device)

        misalign_sharp = batch["misalign_sharp"].to(device)
        align_sharp_save_path = batch["align_sharp_save_path"][0]
        align_sharp_mask_save_path = batch["align_sharp_mask_save_path"][0]

        os.makedirs(os.path.dirname(align_sharp_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(align_sharp_mask_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(align_sharp_mask_save_path.replace('align_mask', 'align_mask0')), exist_ok=True)
        os.makedirs(os.path.dirname(align_sharp_mask_save_path.replace('align_mask', 'align_mask0_region')), exist_ok=True)

        train_label_path = align_sharp_save_path.replace('train_sharp_otsu_step_align', 'train_label')
        train_label_mask_path = align_sharp_save_path.replace('train_sharp_otsu_step_align', 'train_label_mask')
        os.makedirs(os.path.dirname(train_label_path), exist_ok=True)
        os.makedirs(os.path.dirname(train_label_mask_path), exist_ok=True)

        scale = 0.25
        blur_down = F.interpolate(blur, scale_factor=scale, mode='bicubic')
        deblur_down = F.interpolate(deblur, scale_factor=scale, mode='bicubic')
        train_mask_down = F.interpolate(train_mask, scale_factor=scale, mode='bicubic')

        misalign_sharp_down = F.interpolate(misalign_sharp, scale_factor=scale, mode='bicubic')
        align_target_input, mask_input, mask_input0 = get_target_mask(args, alignnet, deblur_down, misalign_sharp_down, misalign_sharp, scale)
        torchvision.utils.save_image(align_target_input, align_sharp_save_path)
        torchvision.utils.save_image(mask_input, align_sharp_mask_save_path)
        torchvision.utils.save_image(mask_input0, align_sharp_mask_save_path.replace('align_mask', 'align_mask0'))

        if 'render_result' in batch:
            render_result = batch["render_result"].to(device)
            render_result_save_path = batch["render_result_save_path"][0]
            render_result_mask_save_path = batch["render_result_mask_save_path"][0]
            render_result_artifact_mask = batch["render_result_artifact_mask"][0]

            os.makedirs(os.path.dirname(render_result_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(render_result_mask_save_path), exist_ok=True)
            # os.makedirs(os.path.dirname(render_result_artifact_mask), exist_ok=True)
            os.makedirs(os.path.dirname(render_result_mask_save_path.replace('align_mask', 'align_mask0')), exist_ok=True)
            os.makedirs(os.path.dirname(render_result_mask_save_path.replace('align_mask', 'align_mask0_region')), exist_ok=True)

            render_result_down = F.interpolate(render_result, scale_factor=scale, mode='bicubic')
            align_target_input_render, mask_input_render, mask_input0_render = get_target_mask(args, alignnet, deblur_down * train_mask_down, render_result_down * train_mask_down, render_result, scale)
            torchvision.utils.save_image(align_target_input_render, render_result_save_path)
            torchvision.utils.save_image(mask_input_render, render_result_mask_save_path)
            torchvision.utils.save_image(mask_input0_render, render_result_mask_save_path.replace('align_mask', 'align_mask0'))

            align_target_input_score = get_blur_val(align_target_input * mask_input * mask_input_render * train_mask)
            align_target_input_render_score = get_blur_val(align_target_input_render * mask_input_render * mask_input * train_mask)

            binary_image = mask_input0[0, 0].cpu().detach().numpy() > 0.
            binary_image_render = mask_input0_render[0, 0].cpu().detach().numpy() > 0.

            # Label different regions
            label_image = measure.label(binary_image, background=0)
            props = measure.regionprops(label_image)

            label_image_render = measure.label(binary_image_render, background=0)
            props_render = measure.regionprops(label_image_render)

            mask_score = aaa(props, binary_image, align_sharp_mask_save_path)
            mask_score_render = aaa(props_render, binary_image_render, render_result_mask_save_path)
            
            print(train_label_path)
            print(f"mask_score: {mask_score}, align_target_input_score: {align_target_input_score}")
            print(f"mask_score_render: {mask_score_render}, align_target_input_render_score: {align_target_input_render_score}")

            # render 对齐后非常扭曲，肯定是因为模糊图模糊太大了，那就不对齐了
            if mask_score_render < 0.7:
                align_target_input_render = render_result
                mask_input_render = torch.ones_like(mask_input_render)

            # 清晰帧对齐后不扭曲，且比render更清晰，就两个合到一起
            if mask_score > 0.5 and align_target_input_score > align_target_input_render_score:
                train_mask_bi = (mask_input.mean(dim=1, keepdim=True) > 0).float()
                train_label = align_target_input * mask_input + align_target_input_render * mask_input_render * (1 - train_mask_bi)
                train_label_mask = mask_input + mask_input_render * (1 - train_mask_bi)
            else:
                train_label = align_target_input_render
                train_label_mask = mask_input_render
        else:
            train_label = align_target_input * mask_input
            train_label_mask = mask_input

        torchvision.utils.save_image(train_label, train_label_path)
        torchvision.utils.save_image(train_label_mask, train_label_mask_path)
