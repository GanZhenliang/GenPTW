import os
import random
import hashlib
import torch
import numpy as np
from PIL import Image
import gradio as gr
import requests
from torchvision.transforms.functional import to_pil_image

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
)
from safetensors.torch import load_file
from transformers import CLIPTokenizer, CLIPTextModel

from aaai_final_adapter import inject_wmadapter, Decoder
from Localizer.model import Localizer
from train_utils import gen_mask_mixed_shapes, compute_mask_auc, compute_mask_f1
from attack_methods.attack import apply_editing_strategy

# ================= 基础设置 =================
device = torch.device("cuda")
phi_dimension = 64

model_path = "..."
vae_ckpt_path = os.path.join(model_path, "diffusion_pytorch_model.safetensors")

pretrained_path = "..."
inpaint_path = "..."
conv_ckpt_path = "..."

# ================= 百度翻译 =================
BAIDU_APP_ID = ""
BAIDU_SECRET_KEY = ""

def translate_zh_to_en(text):
    salt = str(random.randint(10000, 99999))
    sign_str = BAIDU_APP_ID + text + salt + BAIDU_SECRET_KEY
    sign = hashlib.md5(sign_str.encode("utf-8")).hexdigest()
    params = {
        "q": text,
        "from": "zh",
        "to": "en",
        "appid": BAIDU_APP_ID,
        "salt": salt,
        "sign": sign,
    }
    try:
        res = requests.get(
            "https://fanyi-api.baidu.com/api/trans/vip/translate",
            params=params,
            timeout=10,
        )
        res.raise_for_status()
        result = res.json()
        if "trans_result" in result:
            return result["trans_result"][0]["dst"]
        return text
    except Exception as e:
        print("翻译出错:", e)
        return text

# ================= 文本 <-> 比特 =================
def text_to_bits(text, length=64):
    data = text.encode("utf-8")
    bits = []
    for b in data:
        bits.extend([(b >> i) & 1 for i in range(7, -1, -1)])
    if len(bits) < length:
        bits += [0] * (length - len(bits))
    else:
        bits = bits[:length]
    return torch.tensor(bits, dtype=torch.float32).unsqueeze(0).to(device)

def bits_to_text(bits):
    bits = bits[: (len(bits) // 8) * 8]
    byte_arr = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i + 8]
        val = 0
        for b in byte:
            val = (val << 1) | int(b)
        byte_arr.append(val)
    try:
        return bytes(byte_arr).rstrip(b"\x00").decode("utf-8", errors="ignore")
    except Exception:
        return ""

def random_bits(length=64):
    return torch.randint(0, 2, (1, length), dtype=torch.float32, device=device)

# ================= 模型加载 =================
print("Loading SD components...")

scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_path, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(pretrained_path, subfolder="text_encoder").to(device)
unet = UNet2DConditionModel.from_pretrained(pretrained_path, subfolder="unet").to(device)

vae = AutoencoderKL.from_pretrained(pretrained_path, subfolder="vae").to(device)
vae = inject_wmadapter(vae, bit_dim=phi_dimension)
vae_sd = load_file(vae_ckpt_path, device="cuda")
vae.load_state_dict(vae_sd, strict=False)
vae.eval()

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
    torch_dtype=torch.float32,
    ignore_mismatched_sizes=True,
    local_files_only=True,
).to(device)
pipe_sd.set_progress_bar_config(disable=True)

msg_decoder = Decoder(input_channels=3, output_length=phi_dimension).to(device)
msg_decoder.load_state_dict(torch.load(os.path.join(model_path, "msg_decoder.pth"), map_location=device))
msg_decoder.eval()

localized = Localizer(conv_pretrain=True, conv_ckpt=conv_ckpt_path, image_size=512).to(device)
localized.load_state_dict(torch.load(os.path.join(model_path, "localizer.pth"), map_location=device))
localized.eval()

print("All models loaded.")

# ================= 评估逻辑 =================
def evaluate_watermark_and_mask(jpeg_tensor, mask, phi, decoder, localizer, pipe_sd, output_dir, idx, do_aug=False, augment_fn=None):
    pred_bits, wmf, out_dict, edited_img = apply_editing_strategy(
        "sd_inpaint",
        jpeg_tensor,
        mask,
        pipe_sd=pipe_sd,
        msg_decoder=decoder,
        localizer=localizer,
        do_aug=do_aug,
        augment_fn=augment_fn,
    )
    pred_mask = out_dict["pred_mask"]
    pred_mask_binary = (pred_mask > 0.5).float()
    auc = torch.tensor(compute_mask_auc(pred_mask.detach(), mask)).mean().item()
    f1 = torch.tensor(compute_mask_f1(pred_mask_binary, mask)).mean().item()
    phi_binary = (phi > 0.5).float()
    extracted_binary = (pred_bits > 0.5).float()
    bit_acc = (phi_binary == extracted_binary).float().mean().item() * 100
    mask_acc = (pred_mask_binary == mask).float().mean().item() * 100
    return bit_acc, mask_acc, f1, auc, edited_img, pred_mask, pred_bits

# ================= Gradio 4 ImageEditor 输入处理 =================
def get_mask_from_image_editor(img2_value):
    if img2_value is None or isinstance(img2_value, bool):
        return None
    if isinstance(img2_value, dict):
        layers = img2_value.get("layers", None)
        if layers and len(layers) > 0 and layers[0] is not None:
            layer = layers[0].convert("RGBA").resize((512, 512))
            arr = np.array(layer)
            alpha = arr[:, :, 3]
            mask_np = (alpha > 10).astype(np.float32)
            return torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
    return None

# ================= 三步函数 =================
def step_generate(prompt, wm_message, st_wm_pil, st_phi):
    with torch.no_grad():
        prompt_en = translate_zh_to_en(prompt)

        if wm_message.strip():
            phi = text_to_bits(wm_message, length=phi_dimension)
        else:
            phi = random_bits(phi_dimension)

        latents = pipe(prompt=prompt_en, num_inference_steps=30, output_type="latent", return_dict=True)["images"]
        latents_ = latents / 0.18215
        gen_img, *_ = vae.decode_plain(latents_, return_dict=False)
        wm_img, *_ = vae.decode_wm(latents_, phi, return_dict=False)

        gen_img = (gen_img / 2 + 0.5).clamp(0, 1)
        wm_img = (wm_img / 2 + 0.5).clamp(0, 1)
        residual = torch.abs(gen_img - wm_img) * 10

        gen_pil = to_pil_image(gen_img.squeeze(0).cpu())
        wm_pil = to_pil_image(wm_img.squeeze(0).cpu())
        res_pil = to_pil_image(residual.squeeze(0).cpu())

        st_wm_pil = wm_pil
        st_phi = phi.detach().cpu()

        image_editor_value = {"background": wm_pil, "layers": [], "composite": wm_pil}
        empty_img = None
        empty_txt = ""

        return gen_pil, image_editor_value, res_pil, empty_img, empty_img, empty_img, empty_txt, empty_txt, empty_txt, empty_txt, empty_txt, st_wm_pil, st_phi

def step_inpaint(
    img2_value, wm_message, st_wm_pil, st_edit_pil, st_pred_mask_pil,
    st_bit, st_maskacc, st_f1, st_auc, st_extract_text, st_phi
):
    if st_wm_pil is None:
        return gr.update(), gr.update(), gr.update(), st_edit_pil, st_pred_mask_pil, st_bit, st_maskacc, st_f1, st_auc, st_extract_text

    mask = get_mask_from_image_editor(img2_value)
    if mask is None:
        mask = gen_mask_mixed_shapes(1, (512,512), 0.2, 0.34, 7, device)

    phi = st_phi.to(device)

    wm_tensor = torch.from_numpy(np.array(st_wm_pil.convert("RGB"))).permute(2,0,1).float().unsqueeze(0).to(device)/255.0

    bit_acc, mask_acc, f1, auc, edited_img, pred_mask, pred_bits = evaluate_watermark_and_mask(
        jpeg_tensor=wm_tensor,
        mask=mask,
        phi=phi,
        decoder=msg_decoder,
        localizer=localized,
        pipe_sd=pipe_sd,
        output_dir="demo_eval",
        idx=0,
        do_aug=False,
    )

    st_edit_pil = to_pil_image(edited_img.squeeze(0).detach().cpu())
    st_pred_mask_pil = to_pil_image((pred_mask > 0.5).float().squeeze(0).detach().cpu())

    st_bit = f"{bit_acc:.2f}%"
    st_maskacc = f"{mask_acc:.2f}%"
    st_f1 = f"{f1:.4f}"
    st_auc = f"{auc:.4f}"

    extracted_bits = pred_bits.detach().cpu().squeeze().round().int().tolist()
    if wm_message.strip():
        st_extract_text = bits_to_text(extracted_bits)
    else:
        st_extract_text = "".join(str(int(b)) for b in extracted_bits)

    mask_pil = to_pil_image(mask.squeeze(0).detach().cpu())

    return mask_pil, st_edit_pil, st_pred_mask_pil, st_edit_pil, st_pred_mask_pil, st_bit, st_maskacc, st_f1, st_auc, st_extract_text

def step_extract(st_bit, st_maskacc, st_f1, st_auc, st_extract_text):
    return st_bit or "", st_maskacc or "", st_f1 or "", st_auc or "", st_extract_text or ""

def step_inpaint_and_extract(
    img2_value, wm_message, st_wm_pil, st_edit_pil, st_pred_mask_pil,
    st_bit, st_maskacc, st_f1, st_auc, st_extract_text, st_phi
):
    mask_img, edited_img, pred_mask_img, st_edit_pil, st_pred_mask_pil, st_bit, st_maskacc, st_f1, st_auc, st_extract_text = \
        step_inpaint(img2_value, wm_message, st_wm_pil, st_edit_pil, st_pred_mask_pil,
                     st_bit, st_maskacc, st_f1, st_auc, st_extract_text, st_phi)
    bit_val, maskacc_val, f1_val, auc_val, extract_val = step_extract(st_bit, st_maskacc, st_f1, st_auc, st_extract_text)
    return mask_img, edited_img, pred_mask_img, bit_val, maskacc_val, f1_val, auc_val, extract_val, st_edit_pil, st_pred_mask_pil, st_bit, st_maskacc, st_f1, st_auc, st_extract_text

# ================= UI =================
bg_path = "./assets/bg.png"
logo_path = "./assets/logo.png"

custom_css_ui = f"""
.gradio-container {{
    background: url('file={bg_path}') !important;
    background-size: cover !important;
    background-attachment: fixed !important;
    background-position: center !important;
    color: white !important;
}}

.gradio-container::before {{
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.12);
    pointer-events: none;
    z-index: 0;
}}

.gradio-container > * {{
    position: relative;
    z-index: 1;
}}

.gradio-container h2 {{
    color: white !important;
    font-weight: 800 !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.75);
}}

.gr-button {{
    border-radius: 12px !important;
    font-weight: 800 !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.55) !important;
    background: rgba(255,255,255,0.12) !important;
    backdrop-filter: blur(8px);
}}

.gr-button:hover {{
    background: rgba(255,255,255,0.22) !important;
}}

.gr-textbox textarea,
.gr-textbox input {{
    border-radius: 10px !important;
    color: white !important;
    background: rgba(0,0,0,0.35) !important;
    border: 1px solid rgba(255,255,255,0.45) !important;
    font-weight: 500;
}}

.gr-image,
.gr-image-editor {{
    border: 2px solid rgba(255,255,255,0.8) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: rgba(0,0,0,0.25) !important;
}}

footer {{
    visibility: hidden !important;
}}

#wm_editor {{
    height: 512px !important;
}}

#wm_editor .wrap,
#wm_editor .image-container,
#wm_editor [data-testid="image"],
#wm_editor .stage-wrap {{
    width: 100% !important;
    height: 100% !important;
    max-width: none !important;
    max-height: none !important;
    overflow: hidden !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transform: translate(0,0) !important;
}}

#wm_editor canvas,
#wm_editor img,
#wm_editor .canvas {{
    width: 100% !important;
    height: 100% !important;
    max-width: none !important;
    max-height: none !important;
    object-fit: cover !important;
    top: 0 !important;
    left: 0 !important;
}}
"""

with gr.Blocks(css=custom_css_ui, theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🛡 GenPTW: Latent Image Watermarking for Provenance Tracing and Tamper Localization")

    gr.HTML(f"""
        <div style="
            position: fixed;
            right: 35px;
            bottom: 25px;
            z-index: 10;
            opacity: 0.88;
        ">
            <img src="file={logo_path}" style="width: 134px;" />
        </div>
    """)

    with gr.Row():
        prompt = gr.Textbox(label="Prompt（支持中文）", placeholder="例如：梦幻赛博朋克城市")
        wm_message = gr.Textbox(label="嵌入的水印信息", placeholder="例如：Fudan120")

    with gr.Row():
        btn_gen = gr.Button("① 生成原始图和含水印图")
        btn_inp_extract = gr.Button("② AIGC篡改并提取")

    with gr.Row():
        img1 = gr.Image(label="原始生成图", height=512, type="pil")
        img2 = gr.ImageEditor(label="嵌入水印图（可直接涂抹Mask）", type="pil", interactive=True, elem_id="wm_editor")
        img3 = gr.Image(label="残差图（×10）", height=512, type="pil")

    with gr.Row():
        img4 = gr.Image(label="掩码（mask）", height=512, type="pil")
        img5 = gr.Image(label="篡改图（inpainted）", height=512, type="pil")
        img6 = gr.Image(label="预测掩码（pred_mask）", height=512, type="pil")

    with gr.Row():
        bit = gr.Textbox(label="Bit Accuracy")
        maskacc = gr.Textbox(label="Mask Accuracy")
        f1score = gr.Textbox(label="F1 Score")
        aucscore = gr.Textbox(label="AUC")

    with gr.Row():
        extract = gr.Textbox(label="提取出的水印信息")

    st_wm_pil = gr.State(value=None)
    st_edit_pil = gr.State(value=None)
    st_pred_mask_pil = gr.State(value=None)
    st_bit = gr.State(value="")
    st_maskacc = gr.State(value="")
    st_f1 = gr.State(value="")
    st_auc = gr.State(value="")
    st_extract_text = gr.State(value="")
    st_phi = gr.State(value=None)

    # 绑定 btn_gen
    btn_gen.click(
        fn=step_generate,
        inputs=[prompt, wm_message, st_wm_pil, st_phi],
        outputs=[
            img1, img2, img3,
            img4, img5, img6,
            bit, maskacc, f1score, aucscore,
            extract,
            st_wm_pil,
            st_phi
        ],
    )

    # 绑定 btn_inp_extract
    btn_inp_extract.click(
        fn=step_inpaint_and_extract,
        inputs=[
            img2, wm_message, st_wm_pil,
            st_edit_pil, st_pred_mask_pil,
            st_bit, st_maskacc, st_f1, st_auc, st_extract_text,
            st_phi
        ],
        outputs=[
            img4, img5, img6,
            bit, maskacc, f1score, aucscore,
            extract,
            st_edit_pil, st_pred_mask_pil,
            st_bit, st_maskacc, st_f1, st_auc, st_extract_text
        ],
    )

demo.launch(
    share=True,
    server_name="0.0.0.0",
    server_port=8008,
    allowed_paths=[
        "/groupshare/NDSS/qyzzs/WOUAF/1_512_test/Vaccine_aigc_sd",
    ],
)