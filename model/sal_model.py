import torch
import torch.nn as nn
from model.densenet import *
import torch.nn.functional as f

__all__ = ['WsodDense']


class WsodDense(nn.Module):
    def __init__(self):
        super(WsodDense, self).__init__()

        # -----------------------------  decoder 1  -------------------------------- #
        self.side3_1_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.side3_2_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.sidebn3_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)

        self.side4_1_1 = nn.Conv2d(512, 128, 3, padding=1)
        self.side4_2_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.sidebn4_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)

        self.side5_1_1 = nn.Conv2d(1280, 128, 3, padding=1)
        self.side5_2_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.sidebn5_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.side5_3_1 = nn.Conv2d(64, 1, 3, padding=1)

        self.side3cat1 = nn.Conv2d(192, 64, 3, padding=1)
        self.side4cat1 = nn.Conv2d(128, 64, 3, padding=1)
        self.side3out1 = nn.Conv2d(64, 1, 3, padding=1)
        self.side4out1 = nn.Conv2d(64, 1, 3, padding=1)

        # -----------------------------  others  -------------------------------- #
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.densenet = densenet169(pretrained=True)

    def forward(self, x):
        x3, x4, x5 = self.densenet(x)

        # -----------------------------  decoder 1  -------------------------------- #
        h_side3_1 = self.sidebn3_1(self.side3_2_1(self.relu(self.side3_1_1(x3))))
        h_side4_1 = self.sidebn4_1(self.side4_2_1(self.relu(self.side4_1_1(x4))))
        h_side5_1 = self.sidebn5_1(self.side5_2_1(self.relu(self.side5_1_1(x5))))
        # upsample to same size (1/4 original size)
        h_side5_1_up2 = self.upsample(self.relu(h_side5_1))
        h_side5_1_up4 = self.upsample(h_side5_1_up2)
        h_side4_1_up2 = self.upsample(self.relu(h_side4_1))
        # fusion
        side3_1 = self.side3out1(self.side3cat1(torch.cat((h_side5_1_up4, h_side4_1_up2, self.relu(h_side3_1)), 1)))
        side4_1 = self.side4out1(self.side4cat1(torch.cat((self.relu(h_side4_1), h_side5_1_up2), 1)))
        side5_1 = self.side5_3_1(self.relu(h_side5_1))
        # upsample to original size
        side3_1 = self.sigmoid(f.interpolate(side3_1, scale_factor=4, mode='bilinear', align_corners=False))
        side4_1 = self.sigmoid(f.interpolate(side4_1, scale_factor=8, mode='bilinear', align_corners=False))
        side5_1 = self.sigmoid(f.interpolate(side5_1, scale_factor=16, mode='bilinear', align_corners=False))

        return side5_1, side4_1, side3_1
