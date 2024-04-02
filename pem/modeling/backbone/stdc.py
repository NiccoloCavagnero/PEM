# Code is adapted from the original STDC-Seg repository. https://github.com/MichaelFan01/STDC-Seg

import torch
import torch.nn as nn
from detectron2.layers import get_norm
from detectron2.modeling import Backbone, BACKBONE_REGISTRY
from torch.nn import init
import math


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, norm_layer='BN'):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = get_norm(norm_layer, out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1, norm_layer='BN'):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                get_norm(norm_layer, out_planes // 2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
                get_norm(norm_layer, in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                get_norm(norm_layer, out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1, norm_layer=norm_layer))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride, norm_layer=norm_layer))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride, norm_layer=norm_layer))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1)), norm_layer=norm_layer))
            else:
                self.conv_list.append(ConvX(out_planes // int(math.pow(2, idx)), out_planes //
                                      int(math.pow(2, idx)), norm_layer=norm_layer))

    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim=1) + x


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1, norm_layer='BN'):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                get_norm(norm_layer, out_planes // 2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1, norm_layer=norm_layer))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride, norm_layer=norm_layer))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride, norm_layer=norm_layer))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1)), norm_layer=norm_layer))
            else:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx)), norm_layer=norm_layer))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out


class STDCNet(nn.Module):
    def __init__(
            self,
            base=64,
            layers=[2, 2, 2],
            block_num=4,
            block_type="cat",
            use_conv_last=False,
            norm_layer='BN'
    ):
        super(STDCNet, self).__init__()
        if block_type == "cat":
            block = CatBottleneck
        elif block_type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block, norm_layer)

        if layers == [2, 2, 2]:
            self.x2 = nn.Sequential(self.features[:1])
            self.x4 = nn.Sequential(self.features[1:2])
            self.x8 = nn.Sequential(self.features[2:4])
            self.x16 = nn.Sequential(self.features[4:6])
            self.x32 = nn.Sequential(self.features[6:])
        elif layers == [4, 5, 3]:
            self.x2 = nn.Sequential(self.features[:1])
            self.x4 = nn.Sequential(self.features[1:2])
            self.x8 = nn.Sequential(self.features[2:6])
            self.x16 = nn.Sequential(self.features[6:11])
            self.x32 = nn.Sequential(self.features[11:])

        if self.use_conv_last:
            self.conv_last = ConvX(base * 16, max(1024, base * 16), 1, 1)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block, norm_layer):
        features = []
        features += [ConvX(3, base // 2, 3, 2, norm_layer=norm_layer)]
        features += [ConvX(base // 2, base, 3, 2, norm_layer=norm_layer)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2, norm_layer))
                elif j == 0:
                    features.append(block(base * int(math.pow(2, i + 1)), base *
                                    int(math.pow(2, i + 2)), block_num, 2, norm_layer))
                else:
                    features.append(block(base * int(math.pow(2, i + 2)), base *
                                    int(math.pow(2, i + 2)), block_num, 1, norm_layer))

        return nn.Sequential(*features)

    def forward(self, x):
        outs = {}
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        outs['res2'] = feat4

        feat8 = self.x8(feat4)
        outs['res3'] = feat8

        feat16 = self.x16(feat8)
        outs['res4'] = feat16

        feat32 = self.x32(feat16)
        outs['res5'] = feat32

        if self.use_conv_last:
            feat32 = self.conv_last(feat32)
            outs['res5'] = feat32

        return outs


@BACKBONE_REGISTRY.register()
class D2STDCnet(STDCNet, Backbone):
    def __init__(self, cfg, input_shape):
        base = 64
        layers = cfg.MODEL.STDC.LAYERS
        block_num = cfg.MODEL.STDC.BLOCK_NUM
        block_type = cfg.MODEL.STDC.BLOCK_TIPE
        use_conv_last = cfg.MODEL.STDC.USE_CONV_LAST

        super().__init__(
            base=base,
            layers=layers,
            block_num=block_num,
            block_type=block_type,
            use_conv_last=use_conv_last
        )

        self._out_features = cfg.MODEL.STDC.OUT_FEATURES

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": base,
            "res3": base * 4,
            "res4": base * 8,
            "res5": base * 16,
        }
