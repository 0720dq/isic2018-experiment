import torch
import torch.nn as nn
import torch.nn.functional as F
from model.CaraNet.lib.MAVit import MaxViTBlock
from model.experiment.lib.Tok_Mlp import shiftedBlock
from model.CaraNet.lib.HiFormer_DLF import DLF
from model.CaraNet.lib.axial_atten import AA_kernel

class Vit_Cnn_RA(nn.Module):
    def __init__(self, channel=[3, 16, 32, 64, 128, 256],
            num_classes=1,
            ):
        super().__init__()
        self.channel = channel
        self.fc = nn.Linear(channel[0], channel[1])
        self.encoder1 = MaxViTBlock(
            in_channels=channel[1],
            out_channels=channel[2],
            downscale=True,
            grid_window_size=(8, 8)
        )
        self.encoder2 = shiftedBlock(channel[2], channel[3])
        self.encoder3 = MaxViTBlock(
            in_channels=channel[3],
            out_channels=channel[4],
            downscale=True,
            grid_window_size=(8, 8)
        )
        self.encoder4 = shiftedBlock(channel[4], channel[5])
        self.DLF1 = DLF(num_patches=(256, 64), embed_dim=(32, 128))
        self.DLF2 = DLF(num_patches=(128, 32), embed_dim=(64, 256))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(channel[5], channel[4], 1)
        self.decoder3 = shiftedBlock(channel[4], channel[3], False)
        self.decoder2 = shiftedBlock(channel[3], channel[2], False)
        self.decoder1 = shiftedBlock(channel[2], channel[1], False)
        self.conv2 = nn.Conv2d(channel[1], num_classes, 1)

        self.aa_kernel_1 = AA_kernel(128, 128)
        self.ra_conv = nn.Conv2d(channel[5], num_classes, 1)
        self.out_conv1 = nn.Conv2d(channel[4], num_classes, 1)
        self.aa_kernel_2 = AA_kernel(64, 64)
        self.out_conv2 = nn.Conv2d(channel[3], num_classes, 1)
        self.aa_kernel_3 = AA_kernel(32, 32)
        self.out_conv3 = nn.Conv2d(channel[2], num_classes, 1)
        self.aa_kernel_4 = AA_kernel(16, 16)
        self.out_conv4 = nn.Conv2d(channel[1], num_classes, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2).contiguous()
        x0 = self.fc(x)
        x0 = x0.transpose(1, 2).view(B, self.channel[1], H, W).contiguous()
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x1_1, x1_3 = self.DLF1(x1, x3)
        x1_2, x1_4 = self.DLF2(x2, x4)
        x2_4 = self.up(x1_4)
        x2_4 = self.conv1(x2_4)
        x2_3 = self.decoder3(x2_4 + x1_3)
        x2_2 = self.decoder2(x2_3 + x1_2)
        x2_1 = self.decoder1(x2_2 + x1_1)
        # x_out = self.conv2(x2_1)

        # ------------------- atten-one -----------------------
        decoder_0 = F.interpolate(self.ra_conv(x1_4), scale_factor=2, mode='bilinear')  # bs, 1, 64, 64
        revers1 = -1 * (torch.sigmoid(decoder_0)) + 1  # bs, 1, 64, 64
        atten1 = self.aa_kernel_1(x2_4)  # bs, 128, 64, 64
        ra1 = revers1.expand(-1, 128, -1, -1).mul(atten1)  # bs, 128, 64, 64  /.mul函数为元素点乘
        ra1 = self.out_conv1(ra1)  # bs, 1, 64, 64
        ra_out1 = ra1 + decoder_0  # bs, 1, 64, 64
        s1 = F.interpolate(ra_out1, scale_factor=8, mode='bilinear')  # bs, 1, 512, 512

        # ------------------- atten-two -----------------------
        decoder_1 = F.interpolate(ra_out1, scale_factor=2, mode='bilinear')  # bs, 1, 128, 128
        revers2 = -1 * (torch.sigmoid(decoder_1)) + 1  # bs, 1, 128, 128
        atten2 = self.aa_kernel_2(x2_3)  # bs, 64, 128, 128
        ra2 = revers2.expand(-1, 64, -1, -1).mul(atten2)  # bs, 64, 128, 128  /.mul函数为元素点乘
        ra2 = self.out_conv2(ra2)  # bs, 1, 128, 128
        ra_out2 = ra2 + decoder_1  # bs, 1, 128, 128
        s2 = F.interpolate(ra_out2, scale_factor=4, mode='bilinear')  # bs, 1, 512, 512

        # ------------------- atten-two -----------------------
        decoder_2 = F.interpolate(ra_out2, scale_factor=2, mode='bilinear')  # bs, 1, 256, 256
        revers3 = -1 * (torch.sigmoid(decoder_2)) + 1  # bs, 1, 256, 256
        atten3 = self.aa_kernel_3(x2_2)  # bs, 32, 256, 256
        ra3 = revers3.expand(-1, 32, -1, -1).mul(atten3)  # bs, 32, 256, 256  /.mul函数为元素点乘
        ra3 = self.out_conv3(ra3)  # bs, 1, 256, 256
        ra_out3 = ra3 + decoder_2  # bs, 1, 128, 128
        s3 = F.interpolate(ra_out3, scale_factor=2, mode='bilinear')  # bs, 1, 512, 512

        # ------------------- atten-two -----------------------
        decoder_3 = F.interpolate(ra_out3, scale_factor=2, mode='bilinear')  # bs, 1, 512, 512
        revers4 = -1 * (torch.sigmoid(decoder_3)) + 1  # bs, 1, 512, 512
        atten4 = self.aa_kernel_4(x2_1)  # bs, 16, 512, 512
        ra4 = revers4.expand(-1, 16, -1, -1).mul(atten4)  # bs, 16, 512, 512  /.mul函数为元素点乘
        ra4 = self.out_conv4(ra4)  # bs, 1, 512, 512
        s4 = ra4 + decoder_3  # bs, 1, 512, 512

        return s4, s3, s2, s1


if __name__ == '__main__':
    model = Vit_Cnn_RA()
    device = torch.device('cuda:0')
    model = model.to(device=device)
    input_tensor = torch.randn(4, 3, 512, 512)
    input_tensor = input_tensor.to(device, dtype=torch.float)

    out = model(input_tensor)
    # print(out.shape)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print(out[3].shape)