import torch
import torch.nn as nn
from model.CaraNet.lib.HiFormer_DLF import DLF
from model.CaraNet.lib.context_module import CFPModule
from model.experiment.lib.AggAtten_encoder import AAEcoder


class AggAttenFormer_CFP(nn.Module):
    def __init__(self, channel=[3, 16, 32, 64, 128, 256],
            num_classes=1,
            ):
        super().__init__()
        self.channel = channel
        self.fc = nn.Linear(channel[0], channel[1])
        self.encoder1 = AAEcoder(in_chans=channel[1], embed_dims=channel[2], input_resolution=(512, 512),
                                 out_resolution=(256, 256),
                                 num_heads=4, mlp_ratio=8, sr_ratio=8, drop_path=0.2, i=0, depth=1)
        self.encoder2 = AAEcoder(in_chans=channel[2], embed_dims=channel[3], input_resolution=(256, 256),
                                 out_resolution=(128, 128),
                                 num_heads=8, mlp_ratio=8, sr_ratio=4, drop_path=0.2, i=0, depth=1)
        self.encoder3 = AAEcoder(in_chans=channel[3], embed_dims=channel[4], input_resolution=(128, 128),
                                 out_resolution=(64, 64),
                                 num_heads=16, mlp_ratio=4, sr_ratio=2, drop_path=0.2, i=2, depth=1)
        self.encoder4 = AAEcoder(in_chans=channel[4], embed_dims=channel[5], input_resolution=(64, 64),
                                 out_resolution=(32, 32),
                                 num_heads=32, mlp_ratio=4, sr_ratio=1, drop_path=0.2, i=0, depth=1)
        self.DLF1 = DLF(num_patches=(256, 64), embed_dim=(32, 128))
        self.DLF2 = DLF(num_patches=(128, 32), embed_dim=(64, 256))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(channel[5], channel[4], kernel_size=1)

        self.decoder3 = CFPModule(channel[4], channel[3], d=8)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder2 = CFPModule(channel[3], channel[2], d=8)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder1 = CFPModule(channel[2], channel[1], d=8)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

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
        x2_3 = self.up3(x2_3)
        x2_2 = self.decoder2(x2_3 + x1_2)
        x2_2 = self.up2(x2_2)
        x2_1 = self.decoder1(x2_2 + x1_1)
        x2_1 = self.up1(x2_1)

        x_out = self.conv2(x2_1)

        return x_out


if __name__ == '__main__':
    model = AggAttenFormer_CFP()
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