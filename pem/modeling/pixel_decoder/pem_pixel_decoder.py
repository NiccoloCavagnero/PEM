import math
from detectron2.layers import DeformConv, ModulatedDeformConv

import logging
from typing import Callable, Dict, Optional, Union

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import Conv2d, DeformConv, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY


def build_pixel_decoder(cfg, in_channels):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    #if not cfg.MODEL.SEM_SEG_HEAD.USE_BISENET:
    #    return None

    # name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    name = "PEM_Pixel_Decoder"
    model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, in_channels)
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, norm='BN', groups=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False,
                              groups=groups)
        self.bn = get_norm(norm, out_chan)
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class DeformLayer(nn.Module):
    """
    Deformable Convolution Layer module.

    This module implements deformable convolutional operation followed by upsampling.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        deconv_kernel (int): Kernel size for the transposed convolution operation (default: 4).
        deconv_stride (int): Stride for the transposed convolution operation (default: 2).
        deconv_pad (int): Padding for the transposed convolution operation (default: 1).
        deconv_out_pad (int): Output padding for the transposed convolution operation (default: 0).
        num_groups (int): Number of groups for convolution operation (default: 1).
        deform_num_groups (int): Number of deformable groups for deformable convolution (default: 1).
        dilation (int): Dilation factor for deformable convolution operation (default: 1).
        norm (str): Type of normalization layer to use (default: 'BN').
    """
    def __init__(self, in_planes, out_planes, deconv_kernel=4, deconv_stride=2, deconv_pad=1, deconv_out_pad=0,
                 num_groups=1, deform_num_groups=1, dilation=1, norm="BN"):
        super(DeformLayer, self).__init__()

        deform_conv_op = ModulatedDeformConv
        # offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size
        offset_channels = 27

        self.dcn_offset = nn.Conv2d(in_planes,
                                    offset_channels * deform_num_groups,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1 * dilation,
                                    dilation=dilation)
        self.dcn = deform_conv_op(in_planes,
                                  out_planes,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1 * dilation,
                                  bias=False,
                                  groups=num_groups,
                                  dilation=dilation,
                                  deformable_groups=deform_num_groups)
        for layer in [self.dcn]:
            weight_init.c2_msra_fill(layer)

        nn.init.constant_(self.dcn_offset.weight, 0)
        nn.init.constant_(self.dcn_offset.bias, 0)

        self.dcn_bn = get_norm(norm, out_planes)

        self.up_sample = nn.ConvTranspose2d(in_channels=out_planes,
                                            out_channels=out_planes,
                                            kernel_size=deconv_kernel,
                                            stride=deconv_stride, padding=deconv_pad,
                                            output_padding=deconv_out_pad,
                                            bias=False)
        self._deconv_init()
        self.up_bn = get_norm(norm, out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x

        # DCN
        offset_mask = self.dcn_offset(out)
        offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((offset_x, offset_y), dim=1)
        mask = mask.sigmoid()
        x = self.dcn(out.to(self.dcn.weight.dtype), offset.to(self.dcn.weight.dtype), mask.to(self.dcn.weight.dtype))
        x = self.dcn_bn(x)
        x = self.relu(x)

        # Upsampling
        x_up = self.up_sample(x)
        x_up = self.up_bn(x_up)
        x_up = self.relu(x_up)
        return x, x_up

    def _deconv_init(self):
        w = self.up_sample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module for feature transformation.

    This module applies two convolutional layers with batch normalization and ReLU activation
    to transform input feature maps.

    Args:
        dim (int): The input feature dimensionality.
        factor (int): The reduction factor for the intermediate feature dimensionality.
    """
    def __init__(self, dim, factor, norm):
        super().__init__()

        self.mlp = nn.Sequential(nn.Conv2d(dim, dim // factor, 1, bias=False),
                                 get_norm(norm, dim // factor),
                                 nn.ReLU(),
                                 nn.Conv2d(dim // factor, dim, 1, bias=False),
                                 )

    def forward(self, x):
        return self.mlp(x)

class CSM(nn.Module):
    """
    Context-based Self-Modulation (CSM) module for enhancing feature representations.

    This module applies an MLP on global averaged pooled input feature to dynamically reweight
    its input in a residual fashion.

    Args:
        hidden_dim (int): The hidden dimensionality of the MLP.
        factor (int): The reduction factor for the intermediate feature dimensionality in the MLP.
    """
    def __init__(self, hidden_dim, factor, norm):
        super().__init__()
        self.mlp = nn.Sequential(MLP(hidden_dim, factor, norm),
                                 nn.Sigmoid())

    def forward(self, x):
        x_avg = x.mean(dim=[2, 3], keepdim=True)
        x_w = self.mlp(x_avg).sigmoid()
        return x + x * x_w


@SEM_SEG_HEADS_REGISTRY.register()
class PEM_Pixel_Decoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        self.norm = norm
        self.in_channels = [spec.channels for spec in input_shape.values()]

        self.in_projections = nn.ModuleList([ConvBNReLU(in_dim, conv_dim, 1, norm=norm, padding=0)
                                             for in_dim in self.in_channels])

        self.conv_avg = ConvBNReLU(self.in_channels[-1], conv_dim, 1, padding=0, norm=norm)

        self.csm = nn.ModuleList([CSM(conv_dim, factor=2, norm=norm) for _ in range(4)])
        self.dcn = nn.ModuleList([DeformLayer(conv_dim, conv_dim, norm=norm) for _ in range(3)])

        self.out = ConvBNReLU(conv_dim, mask_dim, 1, padding=0, norm=norm)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        return ret

    def forward_features(self, features):
        in_features = []
        for feature, projection in zip(features.values(), self.in_projections):
            in_features.append(projection(feature))

        conv_avg = self.conv_avg(features["res5"].mean(dim=[2, 3], keepdim=True))

        x_32 = self.csm[0](in_features[3]) + conv_avg
        x_32, x_up = self.dcn[0](x_32)

        x_16 = self.csm[1](in_features[2]) + x_up
        x_16, x_up = self.dcn[1](x_16)

        x_8 = self.csm[2](in_features[1]) + x_up
        x_8, x_up = self.dcn[2](x_8)

        x_4 = self.csm[3](in_features[0]) + x_up

        return self.out(x_4), None, [x_32, x_16, x_8]

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)
