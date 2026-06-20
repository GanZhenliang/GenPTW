import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from Localizer.high_frequency_feature_extraction import HighDctFrequencyExtractor
from Localizer.low_frequency_feature_extraction import LowDctFrequencyExtractor
from model_utils import Upsample, ConvBNSelu

class MessageEncoder(nn.Module):
    def __init__(self, input_length=48, output_channels=4, blocks_size=32):
        super(MessageEncoder, self).__init__()
        self.blocks_size = blocks_size
        self.output_channels = output_channels

        self.fc_layers = nn.Sequential(
            nn.Linear(input_length, output_channels * blocks_size * blocks_size),
        )

        self.conv_layers = nn.Sequential(
            ConvBNSelu(4, input_length),
            nn.Conv2d(input_length, output_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, sample, m):
        batch_size = m.shape[0]

        V = self.fc_layers(m)  # (batch_size, 64*64)

        # (batch_size, 1, 32, 32)
        E = V.view(batch_size, 4, self.blocks_size, self.blocks_size)

        E = self.conv_layers(E)  # (batch_size, output_channels, 32, 32)

        return E,V

class Decoder(nn.Module):
    def __init__(self, input_channels=3, output_length=48, image_size=512):
        super(Decoder, self).__init__()
        # 提取特征的卷积层
        self.conv_layers = nn.Sequential(
            ConvBNSelu(input_channels, 64),
            ConvBNSelu(64, 128),
            ConvBNSelu(128, 256),
            ConvBNSelu(256, 512),
            ConvBNSelu(512, 256),
            ConvBNSelu(256, 128),
            ConvBNSelu(128, 64),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),  # 主要用于降维
        )
            
        self.size = image_size
        self.pool = nn.AdaptiveAvgPool2d((256, 256))  # 先池化到固定大小
        self.low_dct = LowDctFrequencyExtractor()
        self.high_dct = HighDctFrequencyExtractor()
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.size * self.size, output_length)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, Iw, mask_gt=None, step=0):

        features = self.conv_layers(Iw)
        m = self.sigmoid(self.fc_layers(features))

        return m, features


class Z_WatermarkCrossAttention(nn.Module):
    def __init__(self, wm_channels, z_channels, hidden_dim=128):
        super().__init__()
        self.q_proj = nn.Conv2d(wm_channels, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.k_proj = nn.Conv2d(z_channels, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.v_proj = nn.Conv2d(z_channels, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.out_proj = Upsample(upscale_type='nearest', in_channels=z_channels, out_channels=z_channels, up_factor=2, activation=nn.SELU)
        self.conv_layers = nn.Sequential(
            ConvBNSelu(hidden_dim + wm_channels, z_channels,kernel_size=1, stride=1, padding=0),
        )

    def forward(self, z, wm_latent):
        """
        z: [B, C, H, W]  z(1,512,32,32)
        wm_latent: [B, c', H, W]   wm(1,4,32,32)
        return: z + delta,
        """
        Q = self.q_proj(wm_latent)  # [B, d, H, W]
        K = self.k_proj(z)
        V = self.v_proj(z)

        B, d, H, W = Q.shape
        Qf = Q.flatten(2).transpose(1, 2)  # [B, HW, d]
        Kf = K.flatten(2).transpose(1, 2)
        Vf = V.flatten(2).transpose(1, 2)

        attn = F.softmax(Qf @ Kf.transpose(-2, -1) / (d ** 0.5), dim=-1)  # [B, HW, HW]
        context = attn @ Vf  # [B, HW, d]
        context = context.transpose(1, 2).view(B, d, H, W)

        wm_res = self.conv_layers(torch.concat([context, wm_latent], dim=1))
        delta = self.out_proj(wm_res)  # [B, C, H, W]

        return z + delta, delta


class WatermarkSpatialInjection(nn.Module):
    def __init__(self, wm_latent_dim=4*32*32, z_channels=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(wm_latent_dim, z_channels),
        )
        self.fuse = nn.Sequential(
            ConvBNSelu(z_channels + z_channels, z_channels),
        )

    def forward(self, z, wm_latent):
        """
        z: [B, C, H, W]
        phi: [B, D]
        return: [B, C, H, W]
        """
        B, C, H, W = z.shape
        wm_latent = wm_latent.view(B, -1)
        phi_emb = self.proj(wm_latent) # [B, C]
        phi_expanded = phi_emb.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)  # [B, C, H, W]
        z_cat = torch.cat([z, phi_expanded], dim=1)  # [B, C+C, H, W]
        return z + self.fuse(z_cat)


def inject_wmadapter(vae: AutoencoderKL, bit_dim=48, image_size=512):
    """
    将原 vae.decoder 和 vae.decode “备份”到:
        vae.decoder.old_forward
        vae.old_decode
    并分别注入新的:
        vae.decoder.forward_wm
        vae.decode_wm
        vae.decoder.forward_plain
        vae.decode_plain
    """

    # ---------------------------
    # 2.1 先保存原始方法
    # ---------------------------
    vae.decoder.old_forward = vae.decoder.forward
    vae.old_decode = vae.decode

    # ---------------------------
    # 2.2 注入模块结构
    # ---------------------------
    decoder = vae.decoder
    fuser_channels = {
        "before_conv": 4,
        "before_mid": 512,
        "before_up0": 512,
        "before_up1": 512,
        "before_up2": 512,
        "before_up3": 256
    }
    size = (image_size//8)
    decoder.watermarkEncoder = MessageEncoder(input_length=bit_dim, output_channels=4, blocks_size=size)
    decoder.watermark_0 = Z_WatermarkCrossAttention(wm_channels=4, z_channels=512, hidden_dim=512)
    decoder.watermark_1 = Z_WatermarkCrossAttention(wm_channels=512, z_channels=512, hidden_dim=512)
    decoder.watermark_2 = WatermarkSpatialInjection(wm_latent_dim=4*size*size, z_channels=256)

    # ---------------------------
    # 2.3 forward_wm（加水印）
    # ---------------------------
    def decoder_forward_wm(self_obj, z: torch.Tensor, bit_vector: torch.Tensor):
        sample = z

        wm_latent_o, wm_latent_b = self_obj.watermarkEncoder(sample, bit_vector)
        sample = sample + wm_latent_o

        sample = self_obj.conv_in(sample)
        sample = self_obj.mid_block(sample)

        z0 = self_obj.up_blocks[0](sample)
        z0, wm_latent = self_obj.watermark_0(z0, wm_latent_o)

        z1 = self_obj.up_blocks[1](z0)
        z1, wm_latent = self_obj.watermark_1(z1, wm_latent)

        z2 = self_obj.up_blocks[2](z1)
        z2 = self_obj.watermark_2(z2, wm_latent_b)

        z3 = self_obj.up_blocks[3](z2)

        sample = self_obj.conv_norm_out(z3)
        sample = self_obj.conv_act(sample)
        sample = self_obj.conv_out(sample)

        return sample, z0, z1, z2, z3

    decoder.forward_wm = decoder_forward_wm.__get__(decoder, decoder.__class__)

    # ---------------------------
    # 2.4 forward_plain（不加水印）
    # ---------------------------
    def decoder_forward_plain(self_obj, z: torch.Tensor):
        sample = z
        sample = self_obj.conv_in(sample)
        sample = self_obj.mid_block(sample)

        z0 = self_obj.up_blocks[0](sample)
        z1 = self_obj.up_blocks[1](z0)
        z2 = self_obj.up_blocks[2](z1)
        z3 = self_obj.up_blocks[3](z2)

        sample = self_obj.conv_norm_out(z3)
        sample = self_obj.conv_act(sample)
        sample = self_obj.conv_out(sample)

        return sample, z0, z1, z2, z3

    decoder.forward_plain = decoder_forward_plain.__get__(decoder, decoder.__class__)

    # ---------------------------
    # 2.5 decode_wm（加水印）
    # ---------------------------
    def decode_wm(self_obj, z: torch.FloatTensor, bit_vector: torch.Tensor, return_dict: bool = True):
        z = self_obj.post_quant_conv(z)
        dec, z0, z1, z2, z3 = self_obj.decoder.forward_wm(z, bit_vector)

        if not return_dict:
            return dec, z0, z1, z2, z3

        from diffusers.utils import BaseOutput
        class DecoderOutput(BaseOutput):
            sample: torch.FloatTensor
            z0: torch.FloatTensor
            z1: torch.FloatTensor
            z2: torch.FloatTensor
            z3: torch.FloatTensor

        return DecoderOutput(sample=dec, z0=z0, z1=z1, z2=z2, z3=z3)

    vae.decode_wm = decode_wm.__get__(vae, vae.__class__)

    # ---------------------------
    # 2.6 decode_plain（不加水印）
    # ---------------------------
    def decode_plain(self_obj, z: torch.FloatTensor, return_dict: bool = True):
        z = self_obj.post_quant_conv(z)
        dec, z0, z1, z2, z3 = self_obj.decoder.forward_plain(z)

        if not return_dict:
            return dec, z0, z1, z2, z3

        from diffusers.utils import BaseOutput
        class DecoderOutput(BaseOutput):
            sample: torch.FloatTensor
            z0: torch.FloatTensor
            z1: torch.FloatTensor
            z2: torch.FloatTensor
            z3: torch.FloatTensor

        return DecoderOutput(sample=dec, z0=z0, z1=z1, z2=z2, z3=z3)

    vae.decode_plain = decode_plain.__get__(vae, vae.__class__)

    return vae

