# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:58:14 2021

@author: angelou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pretrain.Res2Net_v1b import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s
import math
import torchvision.models as models
from model.CaraNet.lib.conv_layer import Conv, BNPReLU
from model.CaraNet.lib.axial_atten import AA_kernel
from model.CaraNet.lib.context_module import CFPModule
from model.CaraNet.lib.partial_decoder import aggregation
from model.CaraNet.lib.MAVit import MaxViTBlock
from model.CaraNet.lib.HiFormer_DLF import DLF
from typing import Type


class CaraNet_MAAtten_DLF1(nn.Module):
    def __init__(self, channel=32,
                 num_heads: int = 32,
                 grid_window_size=(8, 8),
                 attn_drop: float = 0.,
                 drop: float = 0.2,
                 drop_path: float = 0.,
                 mlp_ratio: float = 4.,
                 act_layer: Type[nn.Module] = nn.GELU,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 norm_layer_transformer: Type[nn.Module] = nn.LayerNorm):
        super().__init__()

         # ---- ResNet Backbone ----
        self.resnet = res2net101_v1b_26w_4s(pretrained=True)

        # Receptive Field Block
        self.rfb2_1 = Conv(512, 32,3,1,padding=1,bn_acti=True)
        self.rfb3_1 = Conv(1024, 32,3,1,padding=1,bn_acti=True)
        self.rfb4_1 = Conv(2048, 32,3,1,padding=1,bn_acti=True)

        # Partial Decoder
        self.agg1 = aggregation(channel)

        self.CFP_1 = CFPModule(32, d=8)
        self.CFP_2 = CFPModule(32, d=8)
        self.CFP_3 = CFPModule(32, d=8)
        ###### dilation rate 4, 62.8

        self.ra1_conv1 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra1_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)

        self.ra2_conv1 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra2_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)

        self.ra3_conv1 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv2 = Conv(32,32,3,1,padding=1,bn_acti=True)
        self.ra3_conv3 = Conv(32,1,3,1,padding=1,bn_acti=True)

        self.aa_kernel_1 = AA_kernel(32, 32)
        self.aa_kernel_2 = AA_kernel(32, 32)
        self.aa_kernel_3 = AA_kernel(32, 32)

        self.MAatten1 = MaxViTBlock(
            in_channels=64,
            out_channels=512,
            downscale=True,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            norm_layer_transformer=norm_layer_transformer
        )
        self.MAatten2 = MaxViTBlock(
            in_channels=512,
            out_channels=1024,
            downscale=True,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            norm_layer_transformer=norm_layer_transformer
        )
        self.MAatten3 = MaxViTBlock(
            in_channels=1024,
            out_channels=2048,
            downscale=True,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            norm_layer_transformer=norm_layer_transformer
        )
        self.DLF1 = DLF(num_patches=(64, 64), embed_dim=(512, 512))
        self.DLF2 = DLF(num_patches=(32, 32), embed_dim=(1024, 1024))
        self.DLF3 = DLF(num_patches=(16, 16), embed_dim=(2048, 2048))

    def forward(self, x):

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88  /bs, 64, 128, 128

        # ----------- low-level features -------------

        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88  /bs, 64, 128, 128
        x1_2 = self.resnet.layer2(x1)     # bs, 512, 44, 44  /bs, 512, 64, 64
        x1_3 = self.resnet.layer3(x1_2)     # bs, 1024, 22, 22  /bs, 1024, 32, 32
        x1_4 = self.resnet.layer4(x1_3)     # bs, 2048, 11, 11  /bs, 2048, 16, 16

        x2_2 = self.MAatten1(x)  # bs, 512, 44, 44  /bs, 512, 64, 64
        x2_3 = self.MAatten2(x2_2)  # bs, 1024, 22, 22  /bs, 1024, 32, 32
        x2_4 = self.MAatten3(x2_3)  # bs, 2048, 11, 11  /bs, 2048, 16, 16

        x1_2, x2_2 = self.DLF1(x1_2, x2_2)
        x1_3, x2_3 = self.DLF2(x1_3, x2_3)
        x1_4, x2_4 = self.DLF3(x1_4, x2_4)

        x2 = x1_2 + x2_2
        x3 = x1_3 + x2_3
        x4 = x1_4 + x2_4

        x2_rfb = self.rfb2_1(x2)  # 512 - 32  /bs, 32, 64, 64
        x3_rfb = self.rfb3_1(x3)  # 1024 - 32  /bs, 32, 32, 32
        x4_rfb = self.rfb4_1(x4)  # 2048 - 32  /bs, 32, 16, 16

        decoder_1 = self.agg1(x4_rfb, x3_rfb, x2_rfb)  # 1,44,44  /bs, 1, 64, 64
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=8, mode='bilinear')  # Sg  /bs, 1, 512, 512

        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.25, mode='bilinear')  # bs, 1, 16, 16
        cfp_out_1 = self.CFP_3(x4_rfb)  # 32 - 32  /bs, 32, 16, 16
        decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1  # bs, 1, 16, 16
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)  # bs, 32, 16, 16
        aa_atten_3_o = decoder_2_ra.expand(-1, 32, -1, -1).mul(aa_atten_3)  # bs, 32, 16, 16  /.mul函数为元素点乘

        ra_3 = self.ra3_conv1(aa_atten_3_o)  # 32 - 32  /bs, 32, 16, 16
        ra_3 = self.ra3_conv2(ra_3)  # 32 - 32  /bs, 32, 16, 16
        ra_3 = self.ra3_conv3(ra_3)  # 32 - 1  /bs, 1, 16, 16

        x_3 = ra_3 + decoder_2  # bs, 1, 16, 16
        lateral_map_2 = F.interpolate(x_3, scale_factor=32, mode='bilinear')  # bs, 1, 512, 512

        # ------------------- atten-two -----------------------
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear')  # bs, 1, 32, 32
        cfp_out_2 = self.CFP_2(x3_rfb)   # 32 - 32  /bs, 32, 32, 32
        decoder_3_ra = -1*(torch.sigmoid(decoder_3)) + 1  # bs, 1, 32, 32
        aa_atten_2 = self.aa_kernel_2(cfp_out_2)  # bs, 32, 32, 32
        aa_atten_2_o = decoder_3_ra.expand(-1, 32, -1, -1).mul(aa_atten_2)  # bs, 32, 32, 32

        ra_2 = self.ra2_conv1(aa_atten_2_o)  # 32 - 32  /bs, 32, 32, 32
        ra_2 = self.ra2_conv2(ra_2)  # 32 - 32  /bs, 32, 32, 32
        ra_2 = self.ra2_conv3(ra_2)  # 32 - 1  /bs, 1, 32, 32

        x_2 = ra_2 + decoder_3  # bs, 1, 32, 32
        lateral_map_3 = F.interpolate(x_2, scale_factor=16, mode='bilinear')  # bs, 1, 512, 512

        # ------------------- atten-three -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear')  # bs, 1, 64, 64
        cfp_out_3 = self.CFP_1(x2_rfb)  # 32 - 32  /bs, 32, 64, 64
        decoder_4_ra = -1*(torch.sigmoid(decoder_4)) + 1  # bs, 1, 64, 64
        aa_atten_1 = self.aa_kernel_1(cfp_out_3)  # bs, 32, 64, 64
        aa_atten_1_o = decoder_4_ra.expand(-1, 32, -1, -1).mul(aa_atten_1)  # bs, 32, 64, 64

        ra_1 = self.ra1_conv1(aa_atten_1_o)  # 32 - 32  /bs, 32, 64, 64
        ra_1 = self.ra1_conv2(ra_1)  # 32 - 32  /bs, 32, 64, 64
        ra_1 = self.ra1_conv3(ra_1)  # 32 - 1  /bs, 1, 64, 64

        x_1 = ra_1 + decoder_4  # bs, 1, 64, 64
        lateral_map_5 = F.interpolate(x_1, scale_factor=8, mode='bilinear')  # bs, 1, 512, 512

        return lateral_map_5, lateral_map_3, lateral_map_2, lateral_map_1


if __name__ == '__main__':
    # ras = caranet().cuda()
    ras = CaraNet_MAAtten_DLF1()
    device = torch.device('cuda:0')
    ras = ras.to(device=device)
    # input_tensor = torch.randn(1, 3, 128, 128).cuda()
    input_tensor = torch.randn(4, 3, 512, 512)
    input_tensor = input_tensor.to(device, dtype=torch.float)

    out = ras(input_tensor)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print(out[3].shape)