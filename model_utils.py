import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einops

class ConvBNSelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNSelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.SyncBatchNorm(out_channels)
        self.selu = nn.SELU()

    def forward(self, x):
        return self.selu(self.bn(self.conv(x)))

class GatedRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gate = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.skip_connection = in_channels == out_channels

    def forward(self, x):
        conv_output = self.conv(x)
        gated_output = torch.sigmoid(self.gate(x))
        output = conv_output * gated_output
        if self.skip_connection:
            output += x
        return output

class ConvTBNRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=2):
        super(ConvTBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(channels_in, channels_out, kernel_size=2, stride=stride, padding=0),
            nn.SyncBatchNorm(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class ConvTBNSelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=2):
        super(ConvTBNSelu, self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(channels_in, channels_out, kernel_size=stride, stride=stride, padding=0),
            nn.BatchNorm2d(channels_out),
            nn.SELU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)

class ConvBNSelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNSelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.SyncBatchNorm(out_channels)
        self.selu = nn.SELU()

    def forward(self, x):
        return self.selu(self.bn(self.conv(x)))


class GatedRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gate = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.skip_connection = in_channels == out_channels

    def forward(self, x):
        conv_output = self.conv(x)
        gated_output = torch.sigmoid(self.gate(x))
        output = conv_output * gated_output
        if self.skip_connection:
            output += x
        return output

class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, kernel_size=3, pad=1, stride=1):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding=pad),
            nn.SyncBatchNorm(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class ConvBNReluDrop(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvBNReluDrop, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.SyncBatchNorm(channels_out),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.02)
        )

    def forward(self, x):
        return self.layers(x)


class Resblock(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    expansion = 1

    def __init__(self, channels_in, channels_out, stride=1):
        super(Resblock, self).__init__()

        self.reslayers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1, bias=False),
            nn.SyncBatchNorm(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out * Resblock.expansion, 3, stride, padding=1, bias=False),
            nn.SyncBatchNorm(channels_out * Resblock.expansion)
        )

        self.shorcut = nn.Sequential()
        if stride != 1 or channels_in != channels_out * Resblock.expansion:
            self.shorcut = nn.Sequential(
                nn.Conv2d(channels_in, channels_out * Resblock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.SyncBatchNorm(channels_out * Resblock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.reslayers(x) + self.shorcut(x))


class SEblock(nn.Module):
    def __init__(self, channels, reduction) -> None:
        super(SEblock, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[:2]
        v = self.global_pooling(x).view(b, c)
        v = self.fc(v).view(b, c, 1, 1)
        return x * v.expand_as(x)


class SE_Resblock(nn.Module):
    expansion = 1

    def __init__(self, reduction, channels_in, channels_out, stride=1) -> None:
        super(SE_Resblock, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels_out, channels_out // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels_out // reduction, channels_out, bias=False),
            nn.Sigmoid()
        )

        self.reslayers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1, bias=False),
            nn.SyncBatchNorm(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out * SE_Resblock.expansion, 3, stride, padding=1, bias=False),
            nn.SyncBatchNorm(channels_out * SE_Resblock.expansion)
        )

        self.shorcut = nn.Sequential()
        if stride != 1 or channels_in != channels_out * Resblock.expansion:
            self.shorcut = nn.Sequential(
                nn.Conv2d(channels_in, channels_out * SE_Resblock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.SyncBatchNorm(channels_out * SE_Resblock.expansion)
            )

    def forward(self, x):
        y = x
        x = self.reslayers(x)
        b, c = x.shape[:2]
        v = self.global_pooling(x).view(b, c)
        v = self.fc(v).view(b, c, 1, 1)
        scaled = x * v.expand_as(x)
        return nn.ReLU(inplace=True)(scaled + self.shorcut(y))


class Image_SEblock(nn.Module):
    def __init__(self, channels, reduction, h=640, w=640) -> None:
        super(Image_SEblock, self).__init__()
        self.pooling = nn.AvgPool2d(kernel_size=(17, 17), stride=16, padding=8)
        self.fc = nn.Sequential(
            nn.Linear(channels * h // 16 * w // 16, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels * h // 16 * w // 16, bias=False),
            nn.Sigmoid()
        )
        self.unpool = nn.Upsample((h, w))

    def forward(self, x):
        b, c, h, w = x.shape
        v = self.pooling(x).view(b, c * h // 16 * w // 16)
        # print(v.shape)
        v = self.fc(v).view(b, c, h // 16, w // 16)
        # print(v.shape)
        v_up = self.unpool(v)
        return x * v_up


class ConvBNRelu_LN(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, H=256, W=256, kernel_size=3, pad=1, stride=1):
        super(ConvBNRelu_LN, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding=pad),
            nn.LayerNorm([channels_out, H, W]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class Resblock_LN(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    expansion = 1

    def __init__(self, channels_in, channels_out, H=256, W=256, stride=1):
        super(Resblock_LN, self).__init__()

        self.reslayers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1, bias=False),
            nn.LayerNorm([channels_out, H, W]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out * Resblock.expansion, 3, stride, padding=1, bias=False),
            nn.LayerNorm([channels_out * Resblock.expansion, H, W])
        )

        self.shorcut = nn.Sequential()
        if stride != 1 or channels_in != channels_out * Resblock.expansion:
            self.shorcut = nn.Sequential(
                nn.Conv2d(channels_in, channels_out * Resblock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.LayerNorm([channels_out * Resblock.expansion, H, W])
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.reslayers(x) + self.shorcut(x))

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
            else:
                raise NotImplementedError
        return outputs


class Upsample(nn.Module):

    def __init__(
            self,
            upscale_type: str,
            in_channels: int,
            out_channels: int,
            up_factor: int,
            activation: nn.Module,
            bias: bool = False
    ) -> None:
        """
        Build an upscaling block.
        Args:
            upscale_type (str): the type of upscaling to use
            in_channels (int): the input channel dimension
            out_channels (int): the output channel dimension
            up_factor (int): the upscaling factor
            activation (nn.Module): the type of activation to use
            bias (bool): whether to use bias in the convolution
        Returns:
            nn.Module: the upscaling block
        """
        super(Upsample, self).__init__()
        if upscale_type == 'nearest':
            upsample_block = nn.Sequential(
                nn.Upsample(scale_factor=up_factor, mode='nearest'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=bias),
                LayerNorm(out_channels, data_format="channels_first"),
                activation(),
            )
        elif upscale_type == 'bilinear':
            upsample_block = nn.Sequential(
                nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=bias),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=bias),
                LayerNorm(out_channels, data_format="channels_first"),
                activation(),
            )
        elif upscale_type == 'conv':
            upsample_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=up_factor, stride=up_factor),
                LayerNorm(out_channels, data_format="channels_first"),
                activation(),
            )
        elif upscale_type == 'pixelshuffle':
            conv = nn.Conv2d(in_channels, out_channels * up_factor ** 2, kernel_size=1, bias=False)
            upsample_block = nn.Sequential(
                conv,
                LayerNorm(out_channels * up_factor ** 2, data_format="channels_first"),
                activation(),
                nn.PixelShuffle(up_factor),
            )
            self.init_shuffle_conv_(conv, up_factor)
        else:
            raise ValueError(f"Invalid upscaling type: {upscale_type}")

        self.upsample_block = upsample_block

    def init_shuffle_conv_(self, conv, up_factor):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // (up_factor ** 2), i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = einops.repeat(conv_weight, f'o ... -> (o {up_factor ** 2}) ...')

        conv.weight.data.copy_(conv_weight)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample_block(x)


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x