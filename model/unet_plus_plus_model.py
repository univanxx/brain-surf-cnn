""" Full assembly of the parts to form the complete network """

# import sys
# sys.path.append("/media/ssd-3t/isviridov/fmri_generation/brain-surf-cnn/model")
import torch.nn as nn
import torch
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpCat(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CW
        diffY = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, deep_supervision=True):
        super(UNetPlusPlus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.deep_supervision = deep_supervision

        self.conv0_0 = (DoubleConv(n_channels, 64))
        self.conv1_0 = (Down(64, 128))
        self.conv2_0 = (Down(128, 256))
        self.conv3_0 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.conv4_0 = (Down(512, 1024 // factor))

        self.conv0_1 = DoubleConv(64+128, 64, 64)
        self.conv1_1 = DoubleConv(128+256, 128, 128)
        self.conv2_1 = DoubleConv(256+512, 256, 256)
        self.conv3_1 = DoubleConv(512+1024 // factor, 512, 512)

        self.conv0_2 = DoubleConv(64*2+128, 64, 64)
        self.conv1_2 = DoubleConv(128*2+256, 128, 128)
        self.conv2_2 = DoubleConv(256*2+512, 256, 256)

        self.conv0_3 = DoubleConv(64*3+128, 64, 64)
        self.conv1_3 = DoubleConv(128*3+256, 128, 128)

        self.conv0_4 = DoubleConv(64*4+128, 64, 64)

        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.up_cat = UpCat()

        if self.deep_supervision:
            self.final1 = nn.Conv1d(64, n_classes, kernel_size=1)
            self.final2 = nn.Conv1d(64, n_classes, kernel_size=1)
            self.final3 = nn.Conv1d(64, n_classes, kernel_size=1)
            self.final4 = nn.Conv1d(64, n_classes, kernel_size=1)
        else:
            self.final = nn.Conv1d(64, n_classes, kernel_size=1)

    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(x0_0)
        x0_1 = self.conv0_1(self.up_cat(x1_0, x0_0))

        x2_0 = self.conv2_0(x1_0)
        x1_1 = self.conv1_1(self.up_cat(x2_0, x1_0))
        x0_2 = self.conv0_2(torch.cat([x0_0, self.up_cat(x1_1, x0_1)], 1))

        x3_0 = self.conv3_0(x2_0)
        x2_1 = self.conv2_1(self.up_cat(x3_0, x2_0))
        x1_2 = self.conv1_2(torch.cat([x1_0, self.up_cat(x2_1, x1_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, self.up_cat(x1_2, x0_2)], 1))

        x4_0 = self.conv4_0(x3_0)
        x3_1 = self.conv3_1(self.up_cat(x4_0, x3_0))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up_cat(x3_1, x2_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, self.up_cat(x2_2, x1_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, self.up_cat(x1_3, x0_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output

    def use_checkpointing(self):
        self.conv0_0 = torch.utils.checkpoint(self.conv0_0)
        self.conv1_0 = torch.utils.checkpoint(self.conv1_0)
        self.conv2_0 = torch.utils.checkpoint(self.conv2_0)
        self.conv3_0 = torch.utils.checkpoint(self.conv3_0)
        self.conv4_0 = torch.utils.checkpoint(self.conv4_0)

        self.conv0_1 = torch.utils.checkpoint(self.conv0_1)
        self.conv1_1 = torch.utils.checkpoint(self.conv1_1)
        self.conv2_1 = torch.utils.checkpoint(self.conv2_1)
        self.conv3_1 = torch.utils.checkpoint(self.conv3_1)

        self.conv0_2 = torch.utils.checkpoint(self.conv0_2)
        self.conv1_2 = torch.utils.checkpoint(self.conv1_2)
        self.conv2_2 = torch.utils.checkpoint(self.conv2_2)

        self.conv0_3 = torch.utils.checkpoint(self.conv0_3)
        self.conv1_3 = torch.utils.checkpoint(self.conv1_3)

        self.conv0_4 = torch.utils.checkpoint(self.conv0_4)

        if self.deep_supervision:
            self.final1 = torch.utils.checkpoint(self.final1)
            self.final2 = torch.utils.checkpoint(self.final2)
            self.final3 = torch.utils.checkpoint(self.final3)
            self.final4 = torch.utils.checkpoint(self.final4)
        else:
            self.final = torch.utils.checkpoint(self.final)