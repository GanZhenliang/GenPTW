import torch
import torch.nn as nn
import timm
from model_utils import GatedRes
from Localizer.high_frequency_feature_extraction import HighDctFrequencyExtractor
from Localizer.low_frequency_feature_extraction import LowDctFrequencyExtractor

class ConvNeXt(timm.models.convnext.ConvNeXt):
    def __init__(self, conv_pretrain=False, conv_ckpt=None):
        super(ConvNeXt, self).__init__(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
        if conv_pretrain:
            print("Load Convnext pretrain.")
            model = timm.create_model('convnext_tiny', pretrained=False)
            state_dict = torch.load(conv_ckpt, map_location='cuda')
            model.load_state_dict(state_dict, strict=False)
            self.load_state_dict(model.state_dict())
        original_first_layer = self.stem[0]
        new_first_layer = nn.Conv2d(4, original_first_layer.out_channels,
                                    kernel_size=original_first_layer.kernel_size, stride=original_first_layer.stride,
                                    padding=original_first_layer.padding, bias=False)
        new_first_layer.weight.data[:, :3, :, :] = original_first_layer.weight.data.clone()[:, :3, :, :]
        new_first_layer.weight.data[:, 3:, :, :] = torch.nn.init.kaiming_normal_(new_first_layer.weight[:, 3:, :, :])
        self.stem[0] = new_first_layer

    def forward_features(self, x):
        x = self.stem(x)
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        x = self.norm_pre(x)
        return x, out

    def forward(self, image, mask=None, *args, **kwargs):

        feature, out = self.forward_features(image)

        return feature, out

class MaskDecoder(nn.Module):
    def __init__(self):
        super(MaskDecoder, self).__init__()
        self.upsamplec2 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
        )
        self.upsamplec3 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
        )
        self.upsamplec4 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            GatedRes(384,96, kernel_size=1, stride=1, padding=0),
            GatedRes(96, 96, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(96, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        c2 = self.upsamplec2(c2)
        c3 = self.upsamplec3(c3)
        c4 = self.upsamplec4(c4)
        x = torch.cat([c1, c2, c3, c4], dim=1)
        pred = self.decoder(x)
        return pred

# @MODELS.register_module()
class Localizer(nn.Module):
    def __init__(self, conv_pretrain=False, conv_ckpt=None, image_size=512):
        super(Localizer, self).__init__()
        self.convnext = ConvNeXt(conv_pretrain, conv_ckpt)
        self.low_dct = LowDctFrequencyExtractor()
        self.high_dct = HighDctFrequencyExtractor()
        self.resize = nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=True)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.maskdecoder = MaskDecoder()

    def forward(self, image, mask, wm, *args, **kwargs):

        high_freq = self.high_dct(image)
        input = torch.concat([high_freq, wm.clamp(-1,1)], dim=1)
        _, outs = self.convnext(input) #(1,96,64,64)(1,192,32,32)(1,384,16,16)(1,768,8,8)
        pred_mask_64 = self.maskdecoder(outs)

        pred_mask = pred_mask_64
        pred_mask = self.resize(pred_mask)
        loss = self.loss_fn(pred_mask, mask)
        mask_pred = torch.sigmoid(pred_mask)

        output_dict = {
            "backward_loss": loss,
            "pred_mask": mask_pred,
            "pred_mask_toloss": pred_mask,
        }
        return output_dict