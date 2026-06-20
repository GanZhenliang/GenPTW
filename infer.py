import os
import random
import torch
import numpy as np

import lpips
from torchvision.utils import save_image
from kornia.metrics import psnr, ssim

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
)
from transformers import CLIPTokenizer, CLIPTextModel
from safetensors.torch import load_file

from aaai_final_adapter import inject_wmadapter, Decoder
from Localizer.model import Localizer
from train_utils import gen_mask_mixed_shapes, compute_sifid, iou
from transformations import TransformNet
from attack_methods.attack import apply_editing_strategy


# ================= 工具函数 =================
def quantization(tensor):
    x = (tensor / 2 + 0.5).clamp(0, 1)
    q = torch.round(x * 255) / 255
    return q * 2 - 1


def compute_mask_auc(pred_mask, gt_mask):
    from sklearn.metrics import roc_auc_score

    pred_flat = pred_mask.detach().cpu().view(-1).numpy()
    gt_flat = gt_mask.detach().cpu().view(-1).numpy().astype(int)

    if len(set(gt_flat.tolist())) < 2:
        return float("nan")

    return roc_auc_score(gt_flat, pred_flat)


def compute_mask_f1(pred_mask_binary, gt_mask):
    from sklearn.metrics import f1_score

    pred = pred_mask_binary.detach().cpu().view(-1).numpy().astype(int)
    gt = gt_mask.detach().cpu().view(-1).numpy().astype(int)

    if len(set(gt.tolist())) < 2:
        return float("nan")

    return f1_score(gt, pred)


def save_tensor_image(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(tensor.detach().float().cpu().clamp(0, 1), path)


def save_residual_image(img1, img2, path, scale=10):
    residual = torch.abs(img1 - img2) * scale
    save_tensor_image(residual.clamp(0, 1), path)


def bits_to_string(bits):
    bits = bits.detach().cpu().squeeze().round().int().tolist()
    return "".join(str(int(b)) for b in bits)


def safe_mean(values):
    values = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(values)) if len(values) > 0 else float("nan")


# ================= 模型加载 =================
def load_all_models():
    print("Loading SD components...")

    scheduler = EulerDiscreteScheduler.from_pretrained(
        pretrained_path,
        subfolder="scheduler",
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_path,
        subfolder="tokenizer",
    )

    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_path,
        subfolder="text_encoder",
    ).to(device)

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_path,
        subfolder="unet",
    ).to(device)

    vae = AutoencoderKL.from_pretrained(
        pretrained_path,
        subfolder="vae",
    ).to(device)

    vae = inject_wmadapter(vae, bit_dim=phi_dimension)

    vae_sd = load_file(vae_ckpt_path, device=str(device))
    vae.load_state_dict(vae_sd, strict=False)
    vae.eval().to(dtype=weight_dtype)

    pipe = StableDiffusionPipeline(
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    ).to(device)

    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)

    pipe_sd = StableDiffusionInpaintPipeline.from_pretrained(
        inpaint_path,
        torch_dtype=weight_dtype,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    ).to(device)

    pipe_sd.set_progress_bar_config(disable=True)

    msg_decoder = Decoder(
        input_channels=3,
        output_length=phi_dimension,
    ).to(device)

    msg_decoder.load_state_dict(
        torch.load(
            os.path.join(model_path, "msg_decoder.pth"),
            map_location=device,
        )
    )
    msg_decoder.eval().to(dtype=weight_dtype)

    localizer = Localizer(
        conv_pretrain=True,
        conv_ckpt=conv_ckpt_path,
        image_size=resolution,
    ).to(device)

    localizer.load_state_dict(
        torch.load(
            os.path.join(model_path, "localizer.pth"),
            map_location=device,
        )
    )
    localizer.eval().to(dtype=weight_dtype)

    lpips_net = lpips.LPIPS(net="alex").to(device)
    lpips_net.eval()

    print("All models loaded.")

    return pipe, pipe_sd, vae, msg_decoder, localizer, lpips_net


# ================= 单张 Infer =================
@torch.no_grad()
def infer_one_prompt(
    prompt,
    pipe,
    pipe_sd,
    vae,
    msg_decoder,
    localizer,
    lpips_net,
    strategy="sd_inpaint",
    do_aug=True,
    strength=0.7,
    idx=0,
    save_dir="./infer_results",
):
    os.makedirs(save_dir, exist_ok=True)

    latents = pipe(
        prompt=prompt,
        num_inference_steps=30,
        output_type="latent",
        return_dict=True,
        height=resolution,
        width=resolution,
    )["images"]

    latents_ = latents / 0.18215

    phis = torch.bernoulli(
        torch.rand((1, phi_dimension), device=device)
    ).float()

    gen_img = vae.decode_plain(
        latents_,
        return_dict=False,
    )[0]

    gen_img_nor = (gen_img / 2 + 0.5).clamp(0, 1)

    wm_img = vae.decode_wm(
        latents_,
        phis,
        return_dict=False,
    )[0]

    wm_img_nor = (wm_img / 2 + 0.5).clamp(0, 1)
    wm_img = quantization(wm_img)

    masks = gen_mask_mixed_shapes(
        batch_size=1,
        image_size=(resolution, resolution),
        min_mask_ratio=0.15,
        max_mask_ratio=0.25,
        n_shapes=2,
        device=device,
    ).clamp(0, 1)

    psnr_val = psnr(
        wm_img_nor.detach(),
        gen_img_nor.detach(),
        max_val=1.0,
    ).mean().item()

    ssim_val = ssim(
        wm_img_nor.detach(),
        gen_img_nor.detach(),
        window_size=5,
    ).mean().item()

    lpips_val = lpips_net(
        gen_img.detach(),
        wm_img.detach(),
    ).mean().item()

    sifid_val = compute_sifid(
        gen_img.detach(),
        wm_img.detach(),
        device=device,
    ).mean().item()

    transform_net = TransformNet(
        required_attack_list=["jpeg"],
        apply_many_crops=False,
        ramp=100,
        apply_required_attacks=True,
    ).to(device, dtype=weight_dtype)

    splice_img = gen_img_nor if "splice" in strategy else None

    detected_bits, wmf, out_dict, aug_img = apply_editing_strategy(
        strategy=strategy,
        wm_img=wm_img_nor,
        mask=masks,
        pipe_sd=pipe_sd,
        inpainter=None,
        pipe_lama=None,
        pipe_cn=None,
        msg_decoder=msg_decoder,
        localizer=localizer,
        device=device,
        prompt=prompt,
        resolution=resolution,
        strength=strength,
        splice_img=splice_img,
        augment_fn=transform_net,
        do_aug=do_aug,
    )

    pred_mask = out_dict["pred_mask"]
    pred_mask_bin = (pred_mask > 0.5).float()

    bit_acc = (
        (detected_bits > 0.5)
        .eq(phis > 0.5)
        .sum()
        .item()
        / detected_bits.numel()
    )

    mask_acc = pred_mask_bin.eq(masks).float().mean().item()
    auc_val = compute_mask_auc(pred_mask, masks)
    f1_val = compute_mask_f1(pred_mask_bin, masks)
    iou_val = iou(pred_mask_bin, masks).mean().item()

    sample_dir = os.path.join(save_dir, f"{idx:04d}_{strategy}")
    os.makedirs(sample_dir, exist_ok=True)

    save_tensor_image(gen_img_nor, os.path.join(sample_dir, "gen.png"))
    save_tensor_image(wm_img_nor, os.path.join(sample_dir, "wm.png"))
    save_residual_image(wm_img_nor, gen_img_nor, os.path.join(sample_dir, "residual_x10.png"))
    save_tensor_image(masks, os.path.join(sample_dir, "mask.png"))
    save_tensor_image(aug_img, os.path.join(sample_dir, "edited.png"))
    save_tensor_image(pred_mask, os.path.join(sample_dir, "pred_mask_gray.png"))
    save_tensor_image(pred_mask_bin, os.path.join(sample_dir, "pred_mask_bin.png"))

    gt_bits_str = bits_to_string(phis)
    pred_bits_str = bits_to_string(detected_bits)

    txt_lines = [
        "=" * 80,
        f"Index: {idx}",
        f"Prompt: {prompt}",
        f"Strategy: {strategy}",
        f"Do Aug: {do_aug}",
        f"Strength: {strength}",
        "",
        "视觉质量指标",
        f"PSNR  : {psnr_val:.4f}",
        f"SSIM  : {ssim_val:.4f}",
        f"LPIPS : {lpips_val:.4f}",
        f"SIFID : {sifid_val:.4f}",
        "",
        "水印提取指标",
        f"Bit Accuracy : {bit_acc * 100:.2f}%",
        "",
        "篡改定位指标",
        f"Mask Accuracy : {mask_acc * 100:.2f}%",
        f"AUC           : {auc_val:.4f}",
        f"F1            : {f1_val:.4f}",
        f"IoU           : {iou_val:.4f}",
        "",
        "水印比特",
        f"GT   : {gt_bits_str}",
        f"Pred : {pred_bits_str}",
        "=" * 80,
    ]

    print("\n".join(txt_lines))

    with open(os.path.join(sample_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    return {
        "idx": idx,
        "prompt": prompt,
        "strategy": strategy,
        "bit_acc": bit_acc,
        "mask_acc": mask_acc,
        "auc": auc_val,
        "f1": f1_val,
        "iou": iou_val,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "lpips": lpips_val,
        "sifid": sifid_val,
        "gt_bits": gt_bits_str,
        "pred_bits": pred_bits_str,
    }


# ================= 批量 Infer =================
def infer_batch(
    prompt_list,
    num_samples,
    pipe,
    pipe_sd,
    vae,
    msg_decoder,
    localizer,
    lpips_net,
    strategy="sd_inpaint",
    do_aug=True,
    strength=0.7,
    save_dir="./infer_results",
    random_sample=True,
):
    os.makedirs(save_dir, exist_ok=True)

    all_results = []

    for i in range(num_samples):
        if random_sample:
            prompt = random.choice(prompt_list)
        else:
            prompt = prompt_list[i % len(prompt_list)]

        print("\n" + "#" * 90)
        print(f"Running [{i + 1}/{num_samples}]")
        print(f"Prompt: {prompt}")
        print("#" * 90)

        result = infer_one_prompt(
            prompt=prompt,
            pipe=pipe,
            pipe_sd=pipe_sd,
            vae=vae,
            msg_decoder=msg_decoder,
            localizer=localizer,
            lpips_net=lpips_net,
            strategy=strategy,
            do_aug=do_aug,
            strength=strength,
            idx=i,
            save_dir=save_dir,
        )

        all_results.append(result)

        torch.cuda.empty_cache()

    avg_bit = safe_mean([r["bit_acc"] for r in all_results])
    avg_mask = safe_mean([r["mask_acc"] for r in all_results])
    avg_auc = safe_mean([r["auc"] for r in all_results])
    avg_f1 = safe_mean([r["f1"] for r in all_results])
    avg_iou = safe_mean([r["iou"] for r in all_results])

    avg_psnr = safe_mean([r["psnr"] for r in all_results])
    avg_ssim = safe_mean([r["ssim"] for r in all_results])
    avg_lpips = safe_mean([r["lpips"] for r in all_results])
    avg_sifid = safe_mean([r["sifid"] for r in all_results])

    summary_lines = [
        "=" * 80,
        "FINAL AVERAGE RESULT",
        f"Samples      : {num_samples}",
        f"Strategy     : {strategy}",
        f"Do Aug       : {do_aug}",
        f"Strength     : {strength}",
        "",
        "视觉质量平均指标",
        f"PSNR         : {avg_psnr:.4f}",
        f"SSIM         : {avg_ssim:.4f}",
        f"LPIPS        : {avg_lpips:.4f}",
        f"SIFID        : {avg_sifid:.4f}",
        "",
        "水印提取平均指标",
        f"Bit Accuracy : {avg_bit * 100:.2f}%",
        "",
        "篡改定位平均指标",
        f"Mask Accuracy: {avg_mask * 100:.2f}%",
        f"AUC          : {avg_auc:.4f}",
        f"F1           : {avg_f1:.4f}",
        f"IoU          : {avg_iou:.4f}",
        "=" * 80,
    ]

    summary = "\n".join(summary_lines)

    print("\n" + summary)

    with open(os.path.join(save_dir, "final_average_result.txt"), "w", encoding="utf-8") as f:
        f.write(summary)

    detail_path = os.path.join(save_dir, "all_results.csv")
    with open(detail_path, "w", encoding="utf-8") as f:
        f.write(
            "idx,prompt,strategy,bit_acc,mask_acc,auc,f1,iou,psnr,ssim,lpips,sifid\n"
        )
        for r in all_results:
            prompt_clean = r["prompt"].replace(",", " ").replace("\n", " ")
            f.write(
                f"{r['idx']},{prompt_clean},{r['strategy']},"
                f"{r['bit_acc']:.6f},{r['mask_acc']:.6f},{r['auc']:.6f},"
                f"{r['f1']:.6f},{r['iou']:.6f},{r['psnr']:.6f},"
                f"{r['ssim']:.6f},{r['lpips']:.6f},{r['sifid']:.6f}\n"
            )

    return all_results


# ================= 基础设置 =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float32

phi_dimension = 64
resolution = 512

model_path = "..."
pretrained_path = "..."
inpaint_path = "..."
conv_ckpt_path = "..."

vae_ckpt_path = os.path.join(model_path, "diffusion_pytorch_model.safetensors")
save_dir = "./infer_results"


# ================= 主函数 =================
if __name__ == "__main__":
    pipe, pipe_sd, vae, msg_decoder, localizer, lpips_net = load_all_models()

    prompt_list = [

        "Emma Watson as a powerful mysterious sorceress, casting lightning magic, detailed flowing robe, glowing magical symbols, cinematic lighting, hyperrealistic fantasy, surreal atmosphere, full body, by Artgerm and Alphonse Mucha style",

        "a female mage standing in a ruined ancient temple, floating fire particles, intricate armor details, magical circle glowing under feet, cinematic fantasy lighting, ultra detailed, concept art",

        "a warrior princess holding a glowing sword in a storm battlefield, dynamic pose, wind blowing clothes, ultra detailed armor, dramatic lighting, hyperrealistic fantasy painting",

        "a goddess floating above clouds, golden light rays, sacred aura, extremely detailed silk fabric, fantasy oil painting style",

        "school campus in spring, students walking under sakura trees, soft sunlight, detailed petals motion, realistic anime style, highly detailed environment",

        "anime classroom scene with sunlight through windows, dust particles in air, detailed desks and books, warm tone, ultra detailed composition",

        "ink wash painting of dragon emerging from clouds, dynamic brush strokes, surreal colorful ink blending, masterpiece illustration",

        "watercolor Chinese landscape with rivers and mountains, golden sunlight, detailed brush texture, cinematic composition",

        "snow mountain range at sunrise, detailed rock textures, clouds flowing between peaks, ultra realistic landscape",

        "renaissance oil painting of a grand palace interior, candlelight shadows, extremely detailed textures, classical art style",

        "portrait of a noble woman in oil painting style, highly detailed skin texture, dramatic lighting, museum quality artwork"
    ]

    num_samples = 20

    infer_batch(
        prompt_list=prompt_list,
        num_samples=num_samples,
        pipe=pipe,
        pipe_sd=pipe_sd,
        vae=vae,
        msg_decoder=msg_decoder,
        localizer=localizer,
        lpips_net=lpips_net,
        strategy="sd_inpaint",
        do_aug=True,
        strength=0.7,
        save_dir=save_dir,
        random_sample=False,
    )

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()