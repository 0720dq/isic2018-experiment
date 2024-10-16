import torch
import torch.nn as nn
from model.experiment.lib.BiLevelRoutingAttention import BiLevelRoutingAttention
from model.u_net import Down
from model.CaraNet.lib.HiFormer_DLF import DLF


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class LGAG(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super(LGAG, self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=1,
                      bias=True),
            nn.BatchNorm2d(dim)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=1,
                      bias=True),
            nn.BatchNorm2d(dim)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x


class BRAtten_Cnn(nn.Module):
    def __init__(self,
             channel=[3, 16, 32, 64, 128, 256], num_classes=1):
        super().__init__()
        self.channel = channel
        self.fc = nn.Linear(channel[0], channel[1])

        self.BRA1 = BiLevelRoutingAttention(dim=channel[1], n_win=32)
        self.BRA2 = BiLevelRoutingAttention(dim=channel[2], n_win=32)
        self.BRA3 = BiLevelRoutingAttention(dim=channel[3], n_win=32)
        self.BRA4 = BiLevelRoutingAttention(dim=channel[4], n_win=32)

        self.cnn1 = Down(channel[1], channel[2])
        self.cnn2 = Down(channel[2], channel[3])
        self.cnn3 = Down(channel[3], channel[4])
        self.cnn4 = Down(channel[4], channel[5])

        # self.DLF1 = DLF(num_patches=(256, 256), embed_dim=(32, 32))
        # self.DLF2 = DLF(num_patches=(128, 128), embed_dim=(64, 64))
        # self.DLF3 = DLF(num_patches=(64, 64), embed_dim=(128, 128))
        # self.DLF4 = DLF(num_patches=(32, 32), embed_dim=(256, 256))

        self.LGAG1 = LGAG(32)
        self.LGAG2 = LGAG(64)
        self.LGAG3 = LGAG(128)
        # self.LGAG4 = LGAG(256)

        self.EUCB4 = EUCB(channel[5], channel[4])
        self.EUCB3 = EUCB(channel[4], channel[3])
        self.EUCB2 = EUCB(channel[3], channel[2])
        self.EUCB1 = EUCB(channel[2], channel[1])

        self.pred = nn.Conv2d(channel[1], num_classes, 1)


    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2).contiguous()
        x0 = self.fc(x)
        x0 = x0.transpose(1, 2).view(B, self.channel[1], H, W).contiguous()

        # ****************Transformer branch***************
        x1_1 = self.BRA1(x0)
        x1_2 = self.BRA2(x1_1)
        x1_3 = self.BRA3(x1_2)
        x1_4 = self.BRA4(x1_3)
        # print('x1_1.shape:', x1_1.shape)
        # print('x1_2.shape:', x1_2.shape)
        # print('x1_3.shape:', x1_3.shape)
        # print('x1_4.shape:', x1_4.shape)


        # ****************Cnn branch***************
        x2_1 = self.cnn1(x0)
        x2_2 = self.cnn2(x2_1)
        x2_3 = self.cnn3(x2_2)
        x2_4 = self.cnn4(x2_3)
        # print('x2_1.shape:', x2_1.shape)
        # print('x2_2.shape:', x2_2.shape)
        # print('x2_3.shape:', x2_3.shape)
        # print('x2_4.shape:', x2_4.shape)

        # ****************Fusion branch***************
        # x_1 = self.DLF1(x1_1, x2_1)
        # x_2 = self.DLF2(x1_2, x2_2)
        # x_3 = self.DLF3(x1_3, x2_3)
        # x_4 = self.DLF4(x1_4, x2_4)

        # ****************Decoder***************
        x1 = self.LGAG1(x1_1, x2_1)
        x2 = self.LGAG2(x1_2, x2_2)
        x3 = self.LGAG3(x1_3, x2_3)
        # x4 = self.LGAG4(x_4[0], x_4[1])

        s4 = self.EUCB4(x1_4 + x2_4)
        s3 = self.EUCB3(s4 + x3)
        s2 = self.EUCB2(s3 + x2)
        s1 = self.EUCB1(s2 + x1)

        out = self.pred(s1)

        return out


if __name__ == '__main__':
    model = BRAtten_Cnn()
    device = torch.device('cuda:0')
    model = model.to(device=device)
    input_tensor = torch.randn(4, 3, 512, 512)
    input_tensor = input_tensor.to(device, dtype=torch.float)

    out = model(input_tensor)
    print(out.shape)
