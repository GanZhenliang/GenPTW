import torch
import random
import numpy as np
from PIL import Image
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
from pycocotools import mask as maskUtils

class CocoSegmentationDataset(CocoDetection):
    def __init__(self, img_dir, ann_file, resolution=512, target_names=None):
        super().__init__(img_dir, ann_file)
        self.resolution = resolution
        self.target_ids = None

        # 可选：只保留特定类别
        if target_names:
            cats = self.coco.loadCats(self.coco.getCatIds())
            name2id = {cat['name']: cat['id'] for cat in cats}
            self.target_ids = [name2id[n] for n in target_names if n in name2id]

        # 图像预处理（标准化到 [-1, 1]）
        self.img_transform = T.Compose([
            T.Resize(resolution),
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])

        self.mask_transform = T.Compose([
            T.Resize(resolution),
            T.CenterCrop(resolution),
            T.ToTensor(),
        ])

    def __getitem__(self, index):
        img, anns = super().__getitem__(index)
        img = img.convert("RGB")
        image = self.img_transform(img)

        width, height = img.size
        mask = np.zeros((height, width), dtype=np.uint8)

        #如果指定了 target_names → 只保留这些类别
        if self.target_ids:
            anns = [a for a in anns if a["category_id"] in self.target_ids]

        #如果没有指定类别 → 随机选一种出现的类别
        elif len(anns) > 0:
            cat_ids = list(set(a["category_id"] for a in anns))
            selected_id = random.choice(cat_ids)
            anns = [a for a in anns if a["category_id"] == selected_id]

        #合并所有目标掩码（逻辑或）
        for ann in anns:
            rle = self.coco.annToRLE(ann)
            m = maskUtils.decode(rle)
            mask = np.maximum(mask, m)  # 保证0/1

        mask = Image.fromarray(mask)
        mask = T.Resize(image.shape[1:], interpolation=T.InterpolationMode.NEAREST)(mask)
        mask = np.array(mask, dtype=np.uint8)  # 仍为 0/1
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # [1, H, W], float32, 0/1

        #判断掩码面积 0.15-0.25
        ratio = mask.sum() / mask.numel()
        if ratio < 0.15 or ratio > 0.25:
            # 如果不满足，随机重取一个 index
            return self.__getitem__(random.randint(0, len(self) - 1))

        return {
            "pixel_values": image,  # [3, H, W], float32
            "mask": mask            # [1, H, W], float32, only 0/1
        }

def collate_fn_coco(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    masks = torch.stack([x["mask"] for x in batch])
    return {"pixel_values": pixel_values, "masks": masks}
