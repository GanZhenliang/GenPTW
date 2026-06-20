import os
import itertools
import torch
import torch.nn.functional as F
import torchvision
import shutil
import sys
import random
import math
import numpy as np
import wandb
from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from aaai_final_adapter import inject_wmadapter, Decoder
from Localizer.model import Localizer
from data_utils import collate_fn_coco, CocoSegmentationDataset
from sifid import SIFID
from sklearn.metrics import roc_auc_score, f1_score
from transformations import TransformNet
from kornia.metrics import psnr, ssim
from datetime import datetime
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from visualizer import *

class EdgeGenerator(torch.nn.Module):
    """generate the 'edge bar' for a 0-1 mask Groundtruth of a image
    Algorithm is based on 'Morphological Dilation and Difference Reduction'

    Which implemented with fixed-weight Convolution layer with weight matrix looks like a cross,
    for example, if kernel size is 3, the weight matrix is:
        [[0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]]

    """

    def __init__(self, kernel_size=3) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def _dilate(self, image, kernel_size=3, dtype=torch.float32):
        """Doings dilation on the image

        Args:
            image (_type_): 0-1 tensor in shape (B, C, H, W)
        """
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        assert image.shape[2] > kernel_size and image.shape[3] > kernel_size, "Image must be larger than kernel size"

        kernel = torch.zeros((1, 1, kernel_size, kernel_size), dtype=dtype, device=image.device)
        kernel[0, 0, kernel_size // 2: kernel_size // 2 + 1, :] = 1
        kernel[0, 0, :, kernel_size // 2: kernel_size // 2 + 1] = 1
        kernel = kernel.float()
        # print(kernel)
        res = F.conv2d(image, kernel.view([1, 1, kernel_size, kernel_size]), stride=1, padding=kernel_size // 2)
        return (res > 0) * 1.0

    def _find_edge(self, image, kernel_size=3, return_all=False):
        """Find 0-1 edges of the image

        Args:
            image (_type_): 0-1 ndarray in shape (B, C, H, W)
        """
        # image = torch.tensor(image).float()
        shape = image.shape

        if len(shape) == 2:
            image = image.reshape([1, 1, shape[0], shape[1]])
        if len(shape) == 3:
            image = image.reshape([1, shape[0], shape[1], shape[2]])
        assert image.shape[1] == 1, "Image must be single channel"

        img = self._dilate(image, kernel_size=kernel_size)

        erosion = self._dilate(1 - image, kernel_size=kernel_size)

        diff = -torch.abs(erosion - img) + 1
        diff = (diff > 0) * 1.0
        # res = dilate(diff)
        # diff = diff.numpy()
        if return_all:
            return diff, img, erosion
        else:
            return diff

    def forward(self, x, return_all=False):
        """
        Args:
            image (_type_): 0-1 ndarray in shape (B, C, H, W)
        """
        return self._find_edge(x, self.kernel_size, return_all=return_all)


def compute_mask_auc(pred_mask, gt_mask):
    """
    pred_mask: torch.Tensor of shape [B, 1, H, W], values in [0, 1]
    gt_mask:   torch.Tensor of shape [B, 1, H, W], values in {0, 1}
    """
    pred_flat = pred_mask.detach().cpu().view(-1).numpy()
    gt_flat = gt_mask.detach().cpu().view(-1).numpy().astype(int)  # 强制转为 0/1 的 int 类型

    # AUC: 若全是同类标签，会报错，这里加个判断
    if len(set(gt_flat.tolist())) < 2:
        return float('nan')  # 无法计算AUC
    return roc_auc_score(gt_flat, pred_flat)

def compute_mask_f1(pred_mask_binary, gt_mask):
    """
    pred_mask_binary: torch.Tensor of shape [B, 1, H, W], values ∈ {0, 1}
    gt_mask:          torch.Tensor of shape [B, 1, H, W], values ∈ {0, 1}
    """
    pred = pred_mask_binary.detach().cpu().view(-1).numpy().astype(int)
    gt = gt_mask.detach().cpu().view(-1).numpy().astype(int)

    # 若全是背景或全是前景，会报错；加判断
    if len(set(gt.tolist())) < 2:
        return float('nan')

    return f1_score(gt, pred)


def iou(preds, targets, threshold=0.0, label=1):
    """
    Return IoU for a specific label (0 or 1).
    Args:
        preds (torch.Tensor): Predicted masks with shape Bx1xHxW
        targets (torch.Tensor): Target masks with shape Bx1xHxW
        label (int): The label to calculate IoU for (0 for background, 1 for foreground)
        threshold (float): Threshold to convert predictions to binary masks
    """
    preds = preds > threshold  # Bx1xHxW
    targets = targets > 0.5
    if label == 0:
        preds = ~preds
        targets = ~targets
    intersection = (preds & targets).float().sum((1,2,3))  # B
    union = (preds | targets).float().sum((1,2,3))  # B
    # avoid division by zero
    union[union == 0.0] = intersection[union == 0.0] = 1
    iou = intersection / union
    return iou


def compute_sifid(x, y, net=None,device=torch.device("cuda")):
    """
    Compute SIFID between two images.
    Args:
        x (torch.Tensor): Image tensor of shape (N, C, H, W) in range [-1, 1].
        y (torch.Tensor): Image tensor of shape (N, C, H, W) in range [-1, 1].
    Returns:
        (float): SIFID.
    """
    fn = SIFID(device=device) if net is None else net
    out = [fn(xi, yi) for xi, yi in zip(x, y)]
    return np.array(out)

def decoded_message_error_rate(message, decoded_message):
    length = message.shape[0]
    message = message.gt(0.5)
    decoded_message = decoded_message.gt(0.5)
    # 计算错误率
    # 如果有 NaN，直接返回 error_rate = 0
    if torch.isnan(message).any() or torch.isnan(decoded_message).any():
        error_rate = 0.0
    else:
        # 计算错误率
        error_rate = float(((message) == (decoded_message)).sum().item()) / length
    return error_rate

def decoded_message_error_rate_batch(messages, decoded_messages):
    error_rate = 0.0
    batch_size = len(messages)
    for i in range(batch_size):
        error_rate += decoded_message_error_rate(messages[i], decoded_messages[i])
    error_rate /= batch_size
    return error_rate


def gen_mask_mixed_shapes(
        batch_size: int,
        image_size=(512, 512),
        min_mask_ratio=0.1,
        max_mask_ratio=0.2,
        n_shapes=5,
        device="cuda"
):
    """
    生成混合形状（矩形、圆形、三角形）Mask (0=被遮挡, 1=未遮挡)，
    使用 CUDA 加速，避免 Python 循环，提高计算效率。

    参数:
    ----
    batch_size : int
        批量大小
    image_size : (H, W)
        生成的图像尺寸
    min_mask_ratio : float
        最小遮挡比例
    max_mask_ratio : float
        最大遮挡比例
    n_shapes : int
        生成形状数量
    device : str
        计算设备 (默认 "cuda"，支持 "cpu")

    返回:
    ----
    masks : torch.Tensor
        形状为 [batch_size, 1, H, W]，其中 0 表示遮挡，1 表示未遮挡。
    """

    if not (0 <= min_mask_ratio <= 1 and 0 <= max_mask_ratio <= 1):
        raise ValueError("min_mask_ratio 和 max_mask_ratio 应该在 [0, 1] 之间。")
    if min_mask_ratio > max_mask_ratio:
        raise ValueError("min_mask_ratio 不能大于 max_mask_ratio。")

    H, W = image_size
    total_area = H * W

    # 初始化 mask
    masks = torch.ones((batch_size, 1, H, W), dtype=torch.float32, device=device)
    dibao = 34
    for b in range(batch_size):
        ratio = random.uniform(min_mask_ratio, max_mask_ratio)
        target_area = int(ratio * total_area)
        accumulated_area = 0

        # for _ in range(n_shapes):
        while accumulated_area < target_area*0.8:
            if accumulated_area >= target_area:
                break

            shape_type = random.choice(["rectangle", "circle", "triangle"])

            if shape_type == "rectangle":
                # 直接用 torch randint 生成宽高
                w = torch.randint(dibao, W // 3, (1,), device=device).item()
                h = torch.randint(dibao, H // 3, (1,), device=device).item()
                shape_area = w * h

                remaining = target_area - accumulated_area
                if shape_area > remaining:
                    scale_factor = math.sqrt(remaining / shape_area)
                    w = max(1, int(w * scale_factor))
                    h = max(1, int(h * scale_factor))
                    shape_area = w * h

                if shape_area <= 0:
                    continue

                x1 = torch.randint(0, W - w, (1,), device=device).item()
                y1 = torch.randint(0, H - h, (1,), device=device).item()

                # 直接切片填充
                masks[b, 0, y1:y1 + h, x1:x1 + w] = 0
                accumulated_area += shape_area

            elif shape_type == "circle":
                r = torch.randint(dibao, min(H, W) // 6, (1,), device=device).item()
                shape_area = math.pi * (r ** 2)

                remaining = target_area - accumulated_area
                if shape_area > remaining:
                    scale_factor = math.sqrt(remaining / shape_area)
                    r = max(1, int(r * math.sqrt(scale_factor)))
                    shape_area = math.pi * (r ** 2)

                if shape_area <= 0:
                    continue

                cx = torch.randint(r, W - r, (1,), device=device).item()
                cy = torch.randint(r, H - r, (1,), device=device).item()

                # 生成网格索引并计算到圆心的距离
                yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
                mask_circle = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r ** 2)

                # 应用到 mask
                masks[b, 0][mask_circle] = 0
                accumulated_area += int(shape_area)

            else:  # triangle
                base = torch.randint(dibao, W // 3, (1,), device=device).item()
                height = torch.randint(dibao, H // 3, (1,), device=device).item()
                shape_area = 0.5 * base * height

                remaining = target_area - accumulated_area
                if shape_area > remaining:
                    scale_factor = math.sqrt(remaining / shape_area)
                    base = max(1, int(base * scale_factor))
                    height = max(1, int(height * scale_factor))
                    shape_area = 0.5 * base * height

                if shape_area <= 0:
                    continue

                x1 = torch.randint(0, W - base, (1,), device=device).item()
                y1 = torch.randint(0, H - height, (1,), device=device).item()
                p1, p2, p3 = (x1, y1), (x1 + base, y1), (x1, y1 + height)

                # 生成网格索引
                yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")

                # 计算向量叉积，判断点是否在三角形内
                def cross_product(a, b):
                    return a[0] * b[1] - a[1] * b[0]

                v0 = (p3[0] - p1[0], p3[1] - p1[1])
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (xx - p1[0], yy - p1[1])

                dot00 = v0[0] * v0[0] + v0[1] * v0[1]
                dot01 = v0[0] * v1[0] + v0[1] * v1[1]
                dot02 = v0[0] * v2[0] + v0[1] * v2[1]
                dot11 = v1[0] * v1[0] + v1[1] * v1[1]
                dot12 = v1[0] * v2[0] + v1[1] * v2[1]

                inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
                u = (dot11 * dot02 - dot01 * dot12) * inv_denom
                v = (dot00 * dot12 - dot01 * dot02) * inv_denom

                mask_triangle = (u >= 0) & (v >= 0) & (u + v <= 1)

                # 应用到 mask
                masks[b, 0][mask_triangle] = 0
                accumulated_area += int(shape_area)

    return 1-masks


def save_edge_images(edge_mask, img_save_dir, index):
    """
    计算并保存边缘掩码图像
    mask: (B, 1, H, W)
    img_save_dir: 图像保存路径
    index: 当前样本的索引 (i)
    """
    os.makedirs(img_save_dir, exist_ok=True)  # 确保保存目录存在
    edge_mask_img = edge_mask[0].detach()
    torchvision.utils.save_image(edge_mask_img, os.path.join(img_save_dir, f"edge_mask_{index}.png"))


def generate_edge_mask(mask, kernel_size=3,dtype=torch.float32):
    """
    根据二值mask生成边界mask
    mask: (B, 1, H, W) 形状的0/1张量，其中 1 表示未篡改，0 表示篡改
    kernel_size: 形态学核大小
    """
    assert kernel_size % 2 == 1, "Kernel size 必须是奇数"

    # 使 1=篡改，0=未篡改（匹配论文定义）
    mask_inv = mask

    # 形态学膨胀
    kernel = torch.zeros((1, 1, kernel_size, kernel_size), dtype=dtype, device=mask.device)
    kernel[:, :, kernel_size // 2, :] = 1  # 水平
    kernel[:, :, :, kernel_size // 2] = 1  # 竖直

    dilated = F.conv2d(mask_inv, kernel, stride=1, padding=kernel_size // 2) > 0
    eroded = F.conv2d(mask_inv, kernel, stride=1, padding=kernel_size // 2) == kernel.sum()

    # 计算边缘
    edge_mask = (eroded != dilated).float()
    # save_edge_images(edge_mask,img_save_dir, 17)

    return edge_mask


def prepare_experiment(args, train=True):
    current_time = datetime.now().strftime("%m-%d-%H-%M-%S")
    args.exp_name = f"{args.exp_name}_{current_time}"
    args.output_dir = os.path.join(args.output_dir, args.exp_name)

    if train:
        save_dir = os.path.join(args.output_dir, "train")
    else:
        save_dir = os.path.join(args.output_dir, "test")
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    current_script_path = sys.argv[0]
    script_backup_path = os.path.join(args.output_dir, os.path.basename(current_script_path))
    if os.path.isfile(current_script_path):
        shutil.copy(current_script_path, script_backup_path)
        print(script_backup_path)

    wandb.init(name=args.exp_name, project="GenPTW")
    return save_dir, logging_dir


def build_accelerator(args, logging_dir):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    return Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        kwargs_handlers=[ddp_kwargs],
    )


def get_weight_dtype(accelerator):
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def set_trainable_watermark_params(vae):
    vae.requires_grad_(False)
    vae.decoder.requires_grad_(False)
    for name, param in vae.decoder.named_parameters():
        if "watermark" in name:
            param.requires_grad_(True)
    return vae


def load_models(args, accelerator, weight_dtype):
    pretrained_model = args.pretrained_vae
    vae = AutoencoderKL.from_pretrained(os.path.join(pretrained_model), revision=args.revision)
    vae = inject_wmadapter(vae, bit_dim=args.phi_dimension,image_size=args.resolution)
    vae = set_trainable_watermark_params(vae)

    msg_decoder = Decoder(input_channels=3, output_length=args.phi_dimension,image_size=args.resolution)
    localized = Localizer(conv_pretrain=True, conv_ckpt=args.pretrained_ConvNeXt, image_size=args.resolution)

    if args.pretrained:
        from safetensors.torch import load_file

        vae_ckpt_path = os.path.join(args.pretrained_ckpt, args.vae_ckpt_name)
        vae.load_state_dict(load_file(vae_ckpt_path, device="cuda"), strict=False)
        msg_decoder.load_state_dict(torch.load(os.path.join(args.pretrained_ckpt, args.msg_decoder_ckpt_name)))
        localized.load_state_dict(torch.load(os.path.join(args.pretrained_ckpt, args.localizer_ckpt_name)))
        print(
            f"✅ Loaded pretrained models from {args.pretrained_ckpt}\n"
            f"   VAE={args.vae_ckpt_name}, "
            f"Decoder={args.msg_decoder_ckpt_name}, "
            f"Localizer={args.localizer_ckpt_name}"
        )

    msg_decoder.requires_grad_(True)
    localized.requires_grad_(True)

    vae.to(accelerator.device, dtype=weight_dtype)
    msg_decoder.to(accelerator.device, dtype=weight_dtype)
    localized.to(accelerator.device, dtype=weight_dtype)
    return vae, msg_decoder, localized


def build_optimizer(args, vae, msg_decoder, localized):
    params_to_optimize = itertools.chain(
        (p for p in vae.decoder.parameters() if p.requires_grad),
        msg_decoder.parameters(),
        localized.parameters(),
    )
    return torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


def build_dataloaders(args):
    from torch.utils.data import random_split

    full_dataset = CocoSegmentationDataset(
        img_dir=args.train_img_dir,
        ann_file=args.train_ann_file,
        resolution=args.resolution,
        target_names=args.target_names,
    )
    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(args.split_seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    loader_kwargs = dict(
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn_coco,
    )
    return (
        DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        DataLoader(val_dataset, shuffle=False, **loader_kwargs),
    )

def build_dataloaders_test(args):
    dataset = CocoSegmentationDataset(
        img_dir=args.test_img_dir,
        ann_file=args.test_ann_file,
        resolution=args.resolution,
        target_names=args.target_names,
    )

    loader_kwargs = dict(
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn_coco,
    )
    return DataLoader(dataset, shuffle=True, **loader_kwargs)

def build_lr_scheduler(args, optimizer, train_loader):
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * steps_per_epoch

    return get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.cosine_cycle * args.gradient_accumulation_steps,
    )


def build_attackers(args, device, weight_dtype):
    attack_list = ["blur", "noise", "contrast", "brightness", "saturation", "jpeg", "resize"]
    attack_groups = {
        "all": attack_list,
        "noise": ["blur", "noise"],
        "color": ["contrast", "brightness"],
        "jpeg": ["jpeg"],
    }
    attackers = {
        name: TransformNet(
            required_attack_list=attacks,
            apply_many_crops=False,
            ramp=args.attack_ramp,
            apply_required_attacks=True,
        ).to(device, dtype=weight_dtype)
        for name, attacks in attack_groups.items()
    }
    return attack_list, attackers


def maybe_empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def normalize_img(img):
    return (img / 2 + 0.5).clamp(0, 1)


def make_full_mask(batch_size, args, device):
    return torch.ones((batch_size, 1, args.resolution, args.resolution), dtype=torch.float32, device=device)


def make_empty_mask(batch_size, args, device):
    return torch.zeros((batch_size, 1, args.resolution, args.resolution), dtype=torch.float32, device=device)


def save_training_samples(save_dir, global_step, mask, out, edge, gen_img_nor, gen_img_old_nor, aug_img, hmaps, cost):
    sample_dict = {
        "mask": mask.detach().float().cpu(),
        "pred_mask": (out["pred_mask"].detach() > 0.5).float().cpu(),
        "edge": edge.detach().float().cpu(),
        "gen": gen_img_nor.detach().float().cpu(),
        "old": gen_img_old_nor.detach().float().cpu(),
        "aug": normalize_img(aug_img).detach().float().cpu(),
        "jnd": hmaps.float().cpu(),
        "cost": cost.float().cpu(),
    }
    save_visual_samples(sample_dict, prefix=f"step{global_step}", save_dir=save_dir)
    save_residual_images(
        img1=sample_dict["gen"],
        img2=sample_dict["old"],
        prefix=f"step{global_step}",
        save_dir=save_dir,
    )


def build_logs(metrics, loss, loss_key, loss_lpips, loss_rec, loss_jnd, loss_mask, lr_scheduler, loss_weights):
    return {
        "ac": metrics["error_rate"],
        "ma": metrics["mask_acc"].item(),
        "so": metrics["ssim"].item(),
        "pr": metrics["psnr"].item(),
        "iou": metrics["iou"].item(),
        "auc": metrics["auc"].item(),
        "f1": metrics["f1"].item(),
        "ls": loss.item(),
        "k": loss_key.item(),
        "fid": loss_lpips.item(),
        # "sifid": metrics["sifid"].item(),
        "lc": loss_rec.item(),
        "lj": loss_jnd.item(),
        "mk": loss_mask.item(),
        "lr": lr_scheduler.get_last_lr()[0],
        "w": str(loss_weights),
    }

def compute_metrics(phis, detected_bits, gen_img, gen_img_old, gen_img_nor, gen_img_old_nor, out, mask):
    pred_mask = (out["pred_mask"].detach() > 0.5).float()
    return {
        "error_rate": decoded_message_error_rate_batch(phis.detach(), detected_bits.detach()),
        "psnr": psnr(gen_img_nor.detach(), gen_img_old_nor.detach(), max_val=1.0),
        "ssim": torch.mean(ssim(gen_img_nor.detach(), gen_img_old_nor.detach(), window_size=5)),
        "mask_acc": (pred_mask == mask).float().sum() / mask.numel(),
        "iou": iou(pred_mask, mask).mean(),
        # "sifid": compute_sifid(gen_img.detach(), gen_img_old.detach()).mean(),
        "auc": torch.tensor(compute_mask_auc(out["pred_mask"].detach(), mask)).mean(),
        "f1": torch.tensor(compute_mask_f1(pred_mask, mask)).mean(),
    }