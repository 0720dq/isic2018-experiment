import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
# from torchsummary import summary
from model.cross_scale_interaction import CSI


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


class Resnet34Unet_cim(nn.Module):

    def __init__(self, in_channel=3, out_channel=1, weights=None):
        super(Resnet34Unet_cim, self).__init__()

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

        self.csi = CSI(960)

        # Decode
        self.conv_decode4 = expansive_block(1024 + 512, 512, 512)
        self.conv_decode3 = expansive_block(512 + 256, 256, 256)
        self.conv_decode2 = expansive_block(256 + 128, 128, 128)
        self.conv_decode1 = expansive_block(128 + 64, 64, 64)
        self.conv_decode0 = expansive_block(64, 32, 32)
        self.final_layer = final_block(32, out_channel)

    def forward(self, x):
        x = self.layer0(x)
        print('x.shape{}'.format(x.shape))  # [16, 64, 128, 128]
        # Encode
        encode_block1 = self.layer1(x)
        print('encode_block1.shape{}'.format(encode_block1.shape))
        encode_block2 = self.layer2(encode_block1)
        print('encode_block2.shape{}'.format(encode_block2.shape))
        encode_block3 = self.layer3(encode_block2)
        print('encode_block3.shape{}'.format(encode_block3.shape))
        encode_block4 = self.layer4(encode_block3)
        print('encode_block4.shape{}'.format(encode_block4.shape))

        # Bottleneck
        bottleneck = self.bottleneck(encode_block4)
        # print('bottleneck.shape{}'.format(bottleneck.shape))

        xx = tuple([encode_block1, encode_block2, encode_block3, encode_block4])
        xx = self.csi(xx)
        encode_block1, encode_block2, encode_block3, encode_block4 = xx

        # Decode
        decode_block4 = self.conv_decode4(bottleneck, encode_block4)
        decode_block3 = self.conv_decode3(decode_block4, encode_block3)
        decode_block2 = self.conv_decode2(decode_block3, encode_block2)
        decode_block1 = self.conv_decode1(decode_block2, encode_block1)
        decode_block0 = self.conv_decode0(decode_block1)

        final_layer = self.final_layer(decode_block0)

        return final_layer

def main():
    model = Resnet34Unet_cim()
    input1 = torch.rand(4, 3, 512, 512)
    # with SummaryWriter(log_dir='logs', comment='unet_csi') as w:
    #     w.add_graph(model, (input1,))
    y = model(input1)

    print(y.shape)


if __name__ == '__main__':
    main()