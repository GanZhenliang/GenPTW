import os
import torch
import numpy as np
from PIL import Image
from simple_lama_inpainting.utils import prepare_img_and_mask, download_model

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
)


class SimpleLama():
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> None:
        if os.environ.get("LAMA_MODEL"):
            model_path = os.environ.get("LAMA_MODEL")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"lama torchscript model not found: {model_path}"
                )
        else:
            model_path = download_model(LAMA_MODEL_URL)

        self.model = torch.jit.load(model_path)
        # self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False  # ❗冻结权重
        self.model.train()
        self.model.to(device)
        self.device = device

    def __call__(self, image, mask):
        inpainted = self.model(image, mask)
        return inpainted