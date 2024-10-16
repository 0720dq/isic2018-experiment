import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
# from torchsummary import summary
# from .cross_scale_interaction import CSI
from lib.conv_layer import Conv, BNPReLU
from lib.axial_atten import AA_kernel
from lib.partial_decoder import aggregation


class expansive_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(expansive_block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, d, e=None):
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True)
        # concat

        if e is not None:
            cat = torch.cat([e, d], dim=1)
            out = self.block(cat)
        else:
            out = self.block(d)
        return out


def final_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
    )
    return block


class Resnet34_RA(nn.Module):
    def __init__(self, out_channel=1, weights=None, channel=32):
        super(Resnet34_RA, self).__init__()

        self.resnet = models.resnet34(weights=weights)
        self.layer0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            # self.resnet.maxpool
        )

        # Encode
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        # self.csi = CSI(960)

        # Receptive Field Block
        self.rfb2_1 = Conv(256, 32, 3, 1, padding=1, bn_acti=True)
        self.rfb3_1 = Conv(512, 32, 3, 1, padding=1, bn_acti=True)
        self.rfb4_1 = Conv(1024, 32, 3, 1, padding=1, bn_acti=True)
        self.agg1 = aggregation(channel)

        self.aa_kernel_1 = AA_kernel(32, 32)
        self.aa_kernel_2 = AA_kernel(32, 32)
        self.aa_kernel_3 = AA_kernel(32, 32)

        self.ra1_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.ra2_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.ra3_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        # Decode
        self.conv_decode4 = expansive_block(1024 + 512, 512, 512)
        self.conv_decode3 = expansive_block(512 + 256, 256, 256)
        self.conv_decode2 = expansive_block(256 + 128, 128, 128)
        self.conv_decode1 = expansive_block(128 + 64, 64, 64)
        self.conv_decode0 = expansive_block(64, 32, 32)
        self.final_layer = final_block(32, out_channel)

    def forward(self, x):
        x = self.layer0(x)  # [b, 64, 256, 256]

        # Encode
        encode_block1 = self.layer1(x)  # [b, 64, 256, 256]
        encode_block2 = self.layer2(encode_block1)  # [b, 128, 128, 128]
        encode_block3 = self.layer3(encode_block2)  # [b, 256, 64, 64]
        encode_block4 = self.layer4(encode_block3)  # [b, 512, 32, 32]

        # Bottleneck
        bottleneck = self.bottleneck(encode_block4)  # [b, 1024, 16, 16]
        # print('bottleneck.shape{}'.format(bottleneck.shape))

        # xx = tuple([encode_block1, encode_block2, encode_block3, encode_block4])
        # xx = self.csi(xx)
        # encode_block1, encode_block2, encode_block3, encode_block4 = xx

        x2_rfb = self.rfb2_1(encode_block3)  # [b, 32, 64, 64]
        x3_rfb = self.rfb3_1(encode_block4)  # [b, 32, 32, 32]
        x4_rfb = self.rfb4_1(bottleneck)  # [b, 32, 16, 16]

        decoder_1 = self.agg1(x4_rfb, x3_rfb, x2_rfb)  # [b, 1, 64, 64]
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=8, mode='bilinear')  # [b, 1, 512, 512]

        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.25, mode='bilinear')  # bs, 1, 16, 16

        # cfp_out_1 = self.CFP_3(x4_rfb)  # 32 - 32  /bs, 32, 16, 16
        cfp_out_1 = x4_rfb

        decoder_2_ra = -1 * (torch.sigmoid(decoder_2)) + 1  # bs, 1, 16, 16
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)  # bs, 32, 16, 16
        aa_atten_3_o = decoder_2_ra.expand(-1, 32, -1, -1).mul(aa_atten_3)  # bs, 32, 16, 16  /.mul函数为元素点乘

        ra_3 = self.ra3_conv1(aa_atten_3_o)  # 32 - 32  /bs, 32, 16, 16
        ra_3 = self.ra3_conv2(ra_3)  # 32 - 32  /bs, 32, 16, 16
        ra_3 = self.ra3_conv3(ra_3)  # 32 - 1  /bs, 1, 16, 16

        x_3 = ra_3 + decoder_2  # bs, 1, 16, 16
        lateral_map_2 = F.interpolate(x_3, scale_factor=32, mode='bilinear')  # bs, 1, 512, 512

        # ------------------- atten-two -----------------------
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear')  # bs, 1, 32, 32

        # cfp_out_2 = self.CFP_2(x3_rfb)  # 32 - 32  /bs, 32, 32, 32
        cfp_out_2 = x3_rfb

        decoder_3_ra = -1 * (torch.sigmoid(decoder_3)) + 1  # bs, 1, 32, 32
        aa_atten_2 = self.aa_kernel_2(cfp_out_2)  # bs, 32, 32, 32
        aa_atten_2_o = decoder_3_ra.expand(-1, 32, -1, -1).mul(aa_atten_2)  # bs, 32, 32, 32

        ra_2 = self.ra2_conv1(aa_atten_2_o)  # 32 - 32  /bs, 32, 32, 32
        ra_2 = self.ra2_conv2(ra_2)  # 32 - 32  /bs, 32, 32, 32
        ra_2 = self.ra2_conv3(ra_2)  # 32 - 1  /bs, 1, 32, 32

        x_2 = ra_2 + decoder_3  # bs, 1, 32, 32
        lateral_map_3 = F.interpolate(x_2, scale_factor=16, mode='bilinear')  # bs, 1, 512, 512

        # ------------------- atten-three -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear')  # bs, 1, 64, 64

        # cfp_out_3 = self.CFP_1(x2_rfb)  # 32 - 32  /bs, 32, 64, 64
        cfp_out_3 = x2_rfb

        decoder_4_ra = -1 * (torch.sigmoid(decoder_4)) + 1  # bs, 1, 64, 64
        aa_atten_1 = self.aa_kernel_1(cfp_out_3)  # bs, 32, 64, 64
        aa_atten_1_o = decoder_4_ra.expand(-1, 32, -1, -1).mul(aa_atten_1)  # bs, 32, 64, 64

        ra_1 = self.ra1_conv1(aa_atten_1_o)  # 32 - 32  /bs, 32, 64, 64
        ra_1 = self.ra1_conv2(ra_1)  # 32 - 32  /bs, 32, 64, 64
        ra_1 = self.ra1_conv3(ra_1)  # 32 - 1  /bs, 1, 64, 64

        x_1 = ra_1 + decoder_4  # bs, 1, 64, 64
        lateral_map_5 = F.interpolate(x_1, scale_factor=8, mode='bilinear')  # bs, 1, 512, 512

        return lateral_map_5, lateral_map_3, lateral_map_2, lateral_map_1

        # # Decode
        # decode_block4 = self.conv_decode4(bottleneck, encode_block4)
        # decode_block3 = self.conv_decode3(decode_block4, encode_block3)
        # decode_block2 = self.conv_decode2(decode_block3, encode_block2)
        # decode_block1 = self.conv_decode1(decode_block2, encode_block1)
        # decode_block0 = self.conv_decode0(decode_block1)
        #
        # final_layer = self.final_layer(decode_block0)
        #
        # return final_layer

def main():
    model = Resnet34_RA()
    input1 = torch.rand(4, 3, 512, 512)
    # with SummaryWriter(log_dir='logs', comment='unet_csi') as w:
    #     w.add_graph(model, (input1,))
    y = model(input1)

    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
    print(y[3].shape)
    # print(y)


if __name__ == '__main__':
    main()