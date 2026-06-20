import random
import lpips
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from kornia.metrics import psnr, ssim
from visualizer import *
from transformations import TransformNet
from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionInpaintPipeline
from train_utils import compute_sifid, iou


def quantization(tensor):
    x = (tensor/2 + 0.5).clamp(0,1)
    q = torch.round(x * 255)/255
    return q*2 - 1


def apply_mask_attack_ultra(generated_image, edited_image, mask):
    return generated_image * (1 - mask) + edited_image * mask

    
def apply_editing_strategy(
    strategy: str,
    wm_img: torch.Tensor,       # [B,3,H,W], [0,1], 水印图(或任何图)
    mask: torch.Tensor,         # [B,1,H,W], [0,1]
    pipe_sd=None,               # StableDiffusionInpaintPipeline
    inpainter=None,               # StableDiffusionInpaintPipeline
    pipe_lama=None,             # SimpleLama
    pipe_cn=None,               # ControlNetInpaintPipeline
    msg_decoder=None,           # Decoder_one
    localizer=None,             # Mesorch
    device: str = "cuda",
    prompt: str = "",           # 输入提示
    resolution: int = 512,      # 生成/编辑分辨率
    strength: float = None,     # inpaint强度; None表示随机(>=0.3)
    do_aug: bool = False,       # 是否进行数据增强
    augment_fn=None,            # 数据增强函数
    splice_img: torch.Tensor = None,  # splice需要的图
):
    """
    通用的“编辑 + 水印提取 + 掩码预测”函数

    参数:
    - strategy: 指定用哪种策略
        1) 'sd_inpaint' : 使用 pipe_sd 做 inpainting
        2) 'lama'       : 使用 pipe_lama (SimpleLama)
        3) 'controlnet' : 使用 pipe_cn (ControlNetInpaint)
        4) 'splice_original' : 用 splice_img 替换 mask 区域
        5) 'splice_img'      : 同上, 只是命名区分
        6) 'no_edited'       : 不做编辑
    - wm_img:   [B,3,H,W], 值域[0,1] 的图像
    - mask:     [B,1,H,W], 值域[0,1]
    - pipe_sd, pipe_lama, pipe_cn: 不同 inpaint pipeline
    - msg_decoder, localizer: 用于水印提取 + 篡改区域预测
    - device:   默认"cuda"
    - prompt:   文本提示
    - resolution: inpaint的分辨率
    - strength: inpaint强度(0~1); 若None则随机≥0.3
    - do_aug, augment_fn: 是否对编辑结果做数据增强
    - splice_img: 如果做 splice，需要这个图 [B,3,H,W],[0,1]

    返回:
    - detected_bits: [B, phi_dimension]
    - wmf:           中间特征
    - out_dict:      localizer输出, dict包含 'pred_mask'等
    - edited_img:    [B,3,H,W],最終编辑+增强之后的tensor
    """

    bsz = wm_img.shape[0]
    if strength is None:
        strength = max(random.random(), 0.3)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    if strategy == "sd_inpaint":
        results = []
        for i in range(bsz):
            in_pil  = to_pil_image(wm_img[i].clamp(0,1))
            m_pil   = to_pil_image(mask[i].clamp(0,1))
            out     = pipe_sd(
                prompt=prompt,
                image=in_pil,
                mask_image=m_pil,
                height=resolution,
                width=resolution,
                strength=strength,
                padding_mask_crop=0,
            ).images[0]
            results.append(transform(out))
        edited_img = torch.stack(results, dim=0).to(device)
    elif strategy == "sd_inpaint_all":
        results = []
        for i in range(bsz):
            in_pil  = to_pil_image(wm_img[i].clamp(0,1))
            m_pil   = to_pil_image(mask[i].clamp(0,1))
            out     = pipe_sd(
                prompt=prompt,
                image=in_pil,
                mask_image=m_pil,
                height=resolution,
                width=resolution,
                strength=strength,
                # padding_mask_crop=0,
            ).images[0]
            results.append(transform(out))
        edited_img = torch.stack(results, dim=0).to(device)
    elif strategy == "inpainter":
        inp, inp_all = inpainter.inpaint((wm_img-0.5)*2, mask)
        edited_img = inp
    elif strategy == "inpainter_all":
        inp, inp_all = inpainter.inpaint((wm_img-0.5)*2, mask)
        edited_img = inp_all
    elif strategy == "lama":
        # results = []
        edited_img = pipe_lama(wm_img.clamp(0,1), mask.clamp(0,1)).view(bsz, 3, resolution,resolution)
        edited_img = (edited_img - 0.5) * 2
        # for i in range(bsz):
            # in_pil  = to_pil_image(wm_img[i].clamp(0,1))
            # m_pil   = to_pil_image(mask[i].clamp(0,1))
            # out     = pipe_lama(wm_img[i].clamp(0,1), mask[i].clamp(0,1)).view(3, resolution,resolution)
            # results.append((out-0.5)*2)
            # results.append(transform(out))
        # edited_img = torch.stack(results, dim=0).to(device)

    elif strategy == "controlnet":
        results = []
        for i in range(bsz):
            in_pil = to_pil_image(wm_img[i].clamp(0,1).cpu())
            m_pil  = to_pil_image(mask[i].clamp(0,1).cpu())
            out    = pipe_cn(
                prompt=prompt,
                image=in_pil,
                mask_image=m_pil,
                height=resolution,
                width=resolution,
                strength=strength,
                padding_mask_crop=0,
            ).images[0]
            results.append(transform(out))
        edited_img = torch.stack(results, dim=0).to(device)

    elif strategy == "splice_original":
        local_ = wm_img * (1 - mask) + splice_img * mask
        edited_img = local_

    elif strategy == "splice_img":
        local_ = wm_img * (1 - mask) + splice_img * mask
        edited_img = local_

    elif strategy == "no_edited":
        edited_img = wm_img.clone()

    else:
        raise ValueError(f"Unknown strategy '{strategy}'!")

    if do_aug and augment_fn is not None:
        edited_img = augment_fn(edited_img,20000)

    with torch.no_grad():
        in_for_dec = edited_img
        detected_bits, wmf = msg_decoder(in_for_dec, mask, 340000)
        out_dict = localizer(in_for_dec, mask, wmf)

    return detected_bits, wmf, out_dict, (edited_img / 2 + 0.5).clamp(0, 1)

from sklearn.metrics import roc_auc_score, f1_score
import torch

def compute_mask_auc(pred_mask, gt_mask):
    pred_flat = pred_mask.detach().cpu().view(-1).numpy()
    gt_flat = gt_mask.detach().cpu().view(-1).numpy().astype(int)
    if len(set(gt_flat.tolist())) < 2:
        return float('nan')
    return roc_auc_score(gt_flat, pred_flat)


def compute_mask_f1(pred_mask_binary, gt_mask):
    pred = pred_mask_binary.detach().cpu().view(-1).numpy().astype(int)
    gt = gt_mask.detach().cpu().view(-1).numpy().astype(int)
    if len(set(gt.tolist())) < 2:
        return float('nan')
    return f1_score(gt, pred)

@torch.no_grad()
def test_strategies_aigc(
    test_loader,
    vae,
    msg_decoder,
    localizer,
    phi_dim,
    accelerator,
    global_step,
    lpips_net,
    args,
    inpainter=None,
    pipe_lama=None,
    pipe_cn=None,
    prompt="AIGC Test",
    resolution=512,
    strength=0.7,
    max_batches=10,
    best_acc=0.0,
    weight_dtype = torch.float32,
    do_aug = True,
    save_dir = None
):

    vae = accelerator.unwrap_model(vae)
    msg_decoder = accelerator.unwrap_model(msg_decoder)
    localizer = accelerator.unwrap_model(localizer)

    vae.eval()
    msg_decoder.eval()
    localizer.eval()

    strategies = ["sd_inpaint", "sd_inpaint_all", "lama", "splice_original"]

    pipe_sd = StableDiffusionInpaintPipeline.from_pretrained(
        args.pretrained_pipe_sd,
        local_files_only=True,
    ).to(accelerator.device)
    pipe_sd.set_progress_bar_config(disable=True)

    strategy_results = {
        s: {"bit_acc": 0.0, "mask_acc": 0.0, "auc": 0.0, "f1": 0.0, "iou": 0.0, "count": 0}
        for s in strategies
    }
    visual_scores = {"psnr": 0.0, "wpsnr": 0.0, "ssim": 0.0, "lpips": 0.0, "sifid": 0.0, "count": 0}

    total_correct_bits = 0
    total_bits = 0

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="🧪 [Eval-AIGC]", ncols=100, disable=not accelerator.is_local_main_process)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        pixel_values = batch["pixel_values"].to(accelerator.device)
        masks = batch["masks"].to(accelerator.device).clamp(0, 1)
        bsz = pixel_values.size(0)

        latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
        gen_img = vae.decode_plain(1 / 0.18215 * latents, return_dict=False)[0]
        gen_img_nor = (gen_img / 2 + 0.5).clamp(0, 1)

        phis = torch.bernoulli(torch.rand((bsz, phi_dim), device=accelerator.device))
        wm_img = vae.decode_wm(1 / 0.18215 * latents, phis, return_dict=False)[0]
        wm_img = quantization(wm_img)
        wm_img_nor = (wm_img / 2 + 0.5).clamp(0, 1)

        # ✅ 视觉指标只计算一次
        visual_scores["psnr"] += psnr(wm_img_nor.detach(), gen_img_nor.detach(), max_val=1.0).mean() * bsz
        visual_scores["ssim"] += ssim(wm_img_nor.detach(), gen_img_nor.detach(), window_size=5).mean() * bsz
        visual_scores["lpips"] += lpips_net(gen_img, wm_img).mean().item() * bsz
        visual_scores["sifid"] += compute_sifid(gen_img, wm_img, device=accelerator.device).mean() * bsz
        visual_scores["count"] += bsz

        transform_net = TransformNet(
            required_attack_list=['jpeg'],
            apply_many_crops=False,
            ramp=100,
            apply_required_attacks=True
        ).to(accelerator.device, dtype=weight_dtype)

        sample_dict_o = {
            "gen": wm_img_nor.detach().float().cpu(),
            "old": gen_img_nor.detach().float().cpu(),
            "mask": masks.detach().float().cpu(),
        }
        save_visual_samples(
            sample_dict_o,
            prefix=f"step{batch_idx}",
            save_dir = os.path.join(save_dir, f"val")
        )
        save_residual_images(
            img1=sample_dict_o["gen"],
            img2=sample_dict_o["old"],
            prefix=f"step{batch_idx}",
            save_dir = os.path.join(save_dir, f"val")
        )

        for strategy in strategies:
            splice_img = pixel_values if "splice" in strategy else None

            detected_bits, wmf, out_dict, aug_img = apply_editing_strategy(
                strategy=strategy,
                # wm_img=wm_img,
                wm_img=wm_img_nor,
                mask=masks,
                pipe_sd=pipe_sd,
                inpainter=inpainter,
                pipe_lama=pipe_lama,
                pipe_cn=pipe_cn,
                msg_decoder=msg_decoder,
                localizer=localizer,
                device=accelerator.device,
                prompt=prompt,
                resolution=resolution,
                strength=strength,
                splice_img=splice_img,
                augment_fn = transform_net,
                do_aug = do_aug,
            )
            sample_dict = {
                "pred_mask": (out_dict["pred_mask"].detach() > 0.5).float().cpu(),
                "aug": aug_img.detach().float().cpu(),
            }

            save_visual_samples(
                sample_dict,
                prefix=f"step{batch_idx}_{strategy}",
                save_dir = os.path.join(save_dir, f"val")
            )

            bit_acc = (detected_bits > 0.5).eq(phis > 0.5).sum().item() / detected_bits.numel()
            mask_acc = ((out_dict["pred_mask"] > 0.5).float() == masks).float().mean().item()
            auc = compute_mask_auc(out_dict["pred_mask"], masks)
            f1 = compute_mask_f1((out_dict["pred_mask"] > 0.5).float(), masks)
            iou_score = iou((out_dict["pred_mask"] > 0.5).float(), masks).mean().item()

            strategy_results[strategy]["bit_acc"] += bit_acc * bsz
            strategy_results[strategy]["mask_acc"] += mask_acc * bsz
            strategy_results[strategy]["auc"] += auc * bsz
            strategy_results[strategy]["f1"] += f1 * bsz
            strategy_results[strategy]["iou"] += iou_score * bsz
            strategy_results[strategy]["count"] += bsz

            total_correct_bits += (detected_bits > 0.5).eq(phis > 0.5).sum().item()
            total_bits += detected_bits.numel()

    mean_acc = total_correct_bits / total_bits if total_bits > 0 else 0.0
    txt_lines = [f"Step:{global_step}_Aug:{do_aug}_AIGC 编辑测试结果："]

    for s in strategies:
        c = strategy_results[s]["count"]
        if c == 0: continue
        txt_lines.append(
            f"[{s:>15}] Bit Accuracy: {strategy_results[s]['bit_acc'] / c * 100:.2f}%"
            f" | Mask Accuracy: {strategy_results[s]['mask_acc'] / c * 100:.2f}%"
            f" | AUC: {strategy_results[s]['auc'] / c:.4f}"
            f" | F1: {strategy_results[s]['f1'] / c:.4f}"
            f" | IoU: {strategy_results[s]['iou'] / c:.4f}"
        )

    if visual_scores["count"] > 0:
        txt_lines.append("\n视觉质量指标")
        txt_lines.append(
            f" | PSNR: {visual_scores['psnr'] / visual_scores['count']:.4f}"
            f" | SSIM: {visual_scores['ssim'] / visual_scores['count']:.4f}"
            f" | LPIPS: {visual_scores['lpips'] / visual_scores['count']:.4f}"
            f" | SIFID: {visual_scores['sifid'] / visual_scores['count']:.4f}"
        )

    if accelerator.is_main_process:
        accelerator.print("\n".join(txt_lines))
        vae.save_pretrained(save_dir)
        torch.save(msg_decoder.state_dict(), os.path.join(save_dir, "msg_decoder.pth"))
        torch.save(localizer.state_dict(), os.path.join(save_dir, "localizer.pth"))
        with open(os.path.join(save_dir, f"aug_{do_aug}_best_aigc_result.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))
        if mean_acc > best_acc:
            best_acc = mean_acc
    
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return mean_acc, best_acc


import os
import torch
from tqdm import tqdm

@torch.no_grad()
def test_strategies_attack(
    test_loader,
    vae,
    msg_decoder,
    localizer,
    phi_dim,
    accelerator,
    global_step,
    lpips_net,
    args,
    attack_list,
    max_batches=10,
    best_acc=0.0,
    weight_dtype=torch.float32,
    save_dir = None
):
    
    

    vae = accelerator.unwrap_model(vae)
    msg_decoder = accelerator.unwrap_model(msg_decoder)
    localizer = accelerator.unwrap_model(localizer)

    vae.eval()
    msg_decoder.eval()
    localizer.eval()

    strategy_results = {
        name: {"bit_acc": 0.0, "mask_acc": 0.0, "count": 0}
        for name in attack_list
    }
    visual_scores = {"psnr": 0.0, "wpsnr": 0.0, "ssim": 0.0, "lpips": 0.0, "sifid": 0.0, "count": 0}

    total_correct_bits = 0
    total_bits = 0

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="🧪 [Eval-Attack]", ncols=100, disable=not accelerator.is_local_main_process)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        pixel_values = batch["pixel_values"].to(accelerator.device)
        bsz = pixel_values.size(0)

        latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
        gen_img = vae.decode_plain(1 / 0.18215 * latents, return_dict=False)[0]
        gen_img_nor = (gen_img / 2 + 0.5).clamp(0, 1)

        phis = torch.bernoulli(torch.rand((bsz, phi_dim), device=accelerator.device))
        wm_img = vae.decode_wm(1 / 0.18215 * latents, phis, return_dict=False)[0]
        wm_img = quantization(wm_img)
        wm_img_nor = (wm_img / 2 + 0.5).clamp(0, 1)

        # 视觉指标统计
        visual_scores["psnr"] += psnr(wm_img_nor.detach(), gen_img_nor.detach(), max_val=1.0).mean() * bsz
        visual_scores["ssim"] += ssim(wm_img_nor.detach(), gen_img_nor.detach(), window_size=5).mean() * bsz
        visual_scores["lpips"] += lpips_net(wm_img, gen_img).mean().item() * bsz
        visual_scores["sifid"] += compute_sifid(wm_img, gen_img, net=None, device=accelerator.device).mean() * bsz
        visual_scores["count"] += bsz

        for strategy in attack_list:
            transform_net = TransformNet(
                required_attack_list=[strategy],
                apply_many_crops=False,
                ramp=1,
                apply_required_attacks=True
            ).to(accelerator.device, dtype=weight_dtype)

            aug_img = transform_net(wm_img, global_step)
            mask = torch.zeros((bsz, 1, args.resolution, args.resolution), device=accelerator.device)

            # print(strategy)
            detected_bits, wmf = msg_decoder(aug_img, mask, global_step)
            out_dict = localizer(aug_img, mask, wmf)

            bit_acc = (detected_bits > 0.5).eq(phis > 0.5).sum().item() / detected_bits.numel()
            mask_acc = ((out_dict["pred_mask"] > 0.5).float() == mask).float().mean().item()

            strategy_results[strategy]["bit_acc"] += bit_acc * bsz
            strategy_results[strategy]["mask_acc"] += mask_acc * bsz
            strategy_results[strategy]["count"] += bsz

            total_correct_bits += (detected_bits > 0.5).eq(phis > 0.5).sum().item()
            total_bits += detected_bits.numel()

    mean_acc = total_correct_bits / total_bits if total_bits > 0 else 0.0
    txt_lines = [f"Step:{global_step}_普通攻击测试结果："]

    for strategy in attack_list:
        c = strategy_results[strategy]["count"]
        if c == 0: continue
        txt_lines.append(
            f"[{strategy:>15}] Bit Accuracy: {strategy_results[strategy]['bit_acc'] / c * 100:.2f}%"
            f" | Mask Accuracy: {strategy_results[strategy]['mask_acc'] / c * 100:.2f}%"
        )

    if visual_scores["count"] > 0:
        txt_lines.append("\n视觉质量指标")
        txt_lines.append(
            f" | PSNR: {visual_scores['psnr'] / visual_scores['count']:.4f}"
            f" | SSIM: {visual_scores['ssim'] / visual_scores['count']:.4f}"
            f" | LPIPS: {visual_scores['lpips'] / visual_scores['count']:.4f}"
            f" | SIFID: {visual_scores['sifid'] / visual_scores['count']:.4f}"
        )

    if accelerator.is_main_process:
        accelerator.print("\n".join(txt_lines))
        if mean_acc > best_acc:
            best_acc = mean_acc
        with open(os.path.join(save_dir, "best_attack_result.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return mean_acc, best_acc

