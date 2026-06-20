import os
import torchvision
import torch


def save_visual_samples(
    img_dict: dict,
    prefix: str,
    save_dir: str,
    clamp: bool = True
):
    """
    保存图像字典中的每一张图。

    参数：
    - img_dict: dict，例如 {"mask": tensor, "pred": tensor, "img": tensor}
    - prefix: 每张图前缀，如 step123_imgA
    - save_dir: 保存目录
    - clamp: 是否将tensor限制到[0,1]（默认True）
    """
    os.makedirs(save_dir, exist_ok=True)
    for name, img in img_dict.items():
        if img.ndim == 3:
            img = img.unsqueeze(0)  # 转成[B,C,H,W]
        img = img.detach().cpu()
        if clamp:
            img = img.clamp(0, 1)
        save_path = os.path.join(save_dir, f"{prefix}_{name}.png")
        torchvision.utils.save_image(img, save_path)


def save_residual_images(
    img1: torch.Tensor,
    img2: torch.Tensor,
    prefix: str,
    save_dir: str,
    scale_factors=(1, 10)
):
    """
    保存两张图像间的残差图。

    参数：
    - img1, img2: 两个[B,C,H,W]或[C,H,W]的图像tensor
    - prefix: 文件名前缀
    - save_dir: 保存目录
    - scale_factors: 缩放倍数，默认保存原始差值和10倍增强图
    """
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
    if img2.ndim == 3:
        img2 = img2.unsqueeze(0)

    residual = torch.abs(img1 - img2)

    for sf in scale_factors:
        scaled = (residual * sf).clamp(0, 1)
        save_visual_samples(
            {"residual": scaled},
            prefix=f"{prefix}_res{sf}",
            save_dir=save_dir
        )


def save_tensor_as_images(tensor, save_path_prefix, prefix, cmap='crest'):
    import os, matplotlib.pyplot as plt, seaborn as sns
    import torchvision.transforms.functional as TF
    from PIL import Image

    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    rgb_img = TF.to_pil_image(tensor.squeeze(0).cpu())
    rgb_img.save(f"{save_path_prefix}{prefix}.png")

    gray = TF.rgb_to_grayscale(tensor.squeeze(0)).squeeze(0).cpu().numpy()
    plt.figure(figsize=(6, 6))
    sns.heatmap(gray, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False, square=True)
    plt.axis('off')
    # plt.tight_layout()
    plt.savefig(f"{save_path_prefix}{prefix}_heatmap.png", bbox_inches='tight', dpi=300)
    plt.close()


def save_attention_heatmap_seaborn(attn, save_path_prefix, prefix, H, W, cmap='crest'):
    import os, numpy as np, matplotlib.pyplot as plt, seaborn as sns
    from PIL import Image

    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
    query_index = (H // 2) * W + (W // 2)
    attn_map = attn[0, query_index].view(H, W).detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    sns.heatmap(attn_map, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False, square=True)
    plt.axis('off')
    # plt.tight_layout()
    plt.savefig(f"{save_path_prefix}{prefix}_att_heatmap.png", bbox_inches='tight', dpi=300)
    plt.close()

    attn_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    Image.fromarray((attn_norm * 255).astype(np.uint8), mode='L').save(f"{save_path_prefix}{prefix}_att_gray.png")

def save_single_channel_heatmap(tensor, save_path_prefix, prefix, cmap='crest'):
    import os, torch, numpy as np, matplotlib.pyplot as plt, seaborn as sns
    from PIL import Image

    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.squeeze().detach().cpu().numpy()
    else:
        tensor = np.squeeze(tensor)

    norm = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    plt.figure(figsize=(6, 6))
    sns.heatmap(norm, cmap=cmap, cbar=True, xticklabels=False, yticklabels=False, square=True)
    plt.axis('off')
    # plt.tight_layout()
    plt.savefig(f"{save_path_prefix}{prefix}_heatmap.png", bbox_inches='tight', dpi=300)
    plt.close()

    Image.fromarray((norm * 255).astype(np.uint8), mode='L').save(f"{save_path_prefix}{prefix}_gray.png")

