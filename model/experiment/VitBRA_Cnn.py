import torch
import torch.nn as nn
from model.CaraNet.lib.MAVit import MaxViTBlock
from model.experiment.lib.Tok_Mlp import shiftedBlock
from model.CaraNet.lib.HiFormer_DLF import DLF
from model.experiment.lib.BiLevelRoutingAttention import BiLevelRoutingAttention

class VitBRA_Cnn(nn.Module):
    def __init__(self, channel=[3, 16, 32, 64, 128, 256],
            num_classes=1,
            ):
        super().__init__()
        self.channel = channel
        self.fc = nn.Linear(channel[0], channel[1])
        self.encoder1 = BiLevelRoutingAttention(dim=channel[1], n_win=16)
        self.encoder2 = BiLevelRoutingAttention(dim=channel[2], n_win=16)
        self.encoder3 = shiftedBlock(channel[3], channel[4])
        self.encoder4 = shiftedBlock(channel[4], channel[5])

        self.DLF1 = DLF(num_patches=(256, 64), embed_dim=(32, 128))
        self.DLF2 = DLF(num_patches=(128, 32), embed_dim=(64, 256))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = nn.Conv2d(channel[5], channel[4], kernel_size=1)
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
        x2_4 = self.up(x1_4)
        x2_4 = self.conv1(x2_4)
        x2_3 = self.decoder3(x2_4 + x1_3)
        x2_2 = self.decoder2(x2_3 + x1_2)
        x2_1 = self.decoder1(x2_2 + x1_1)
        x_out = self.conv2(x2_1)

        return x_out


if __name__ == '__main__':
    model = VitBRA_Cnn()
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