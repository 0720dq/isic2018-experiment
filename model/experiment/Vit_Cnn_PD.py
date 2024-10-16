import torch
import torch.nn as nn
from model.CaraNet.lib.MAVit import MaxViTBlock
from model.experiment.lib.Tok_Mlp import shiftedBlock
from model.CaraNet.lib.HiFormer_DLF import DLF
from model.CaraNet.lib.conv_layer import Conv
from model.CaraNet.lib.partial_decoder import aggregation

class Vit_Cnn_PD(nn.Module):
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

        self.rfb2_1 = Conv(64, 256, 3, 1, padding=1, bn_acti=True)
        self.rfb3_1 = Conv(128, 256, 3, 1, padding=1, bn_acti=True)
        self.rfb4_1 = Conv(256, 256, 3, 1, padding=1, bn_acti=True)
        self.agg1 = aggregation(256, 256)
        self.down = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)

        self.decoder4 = shiftedBlock(channel[5], channel[4], False)
        self.decoder3 = shiftedBlock(channel[4], channel[3], False)
        self.decoder2 = shiftedBlock(channel[3], channel[2], False)
        self.decoder1 = shiftedBlock(channel[2], channel[1], False)
        self.conv2 = nn.Conv2d(channel[1], num_classes, kernel_size=1)

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

        x2 = self.rfb2_1(x2)
        x3 = self.rfb3_1(x3)
        x4 = self.rfb4_1(x4)
        x = self.agg1(x4, x3, x2)
        x = self.down(x)

        x2_4 = self.decoder4(x + x1_4)
        x2_3 = self.decoder3(x2_4 + x1_3)
        x2_2 = self.decoder2(x2_3 + x1_2)
        x2_1 = self.decoder1(x2_2 + x1_1)
        x_out = self.conv2(x2_1)

        return x_out


if __name__ == '__main__':
    model = Vit_Cnn_PD()
    device = torch.device('cuda:0')
    model = model.to(device=device)
    input_tensor = torch.randn(2, 3, 512, 512)
    input_tensor = input_tensor.to(device, dtype=torch.float)

    out = model(input_tensor)
    print(out.shape)
    # print(out[0].shape)
    # print(out[1].shape)
    # print(out[2].shape)
    # print(out[3].shape)