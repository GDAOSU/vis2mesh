import torch
import torch.nn as nn


class ConvBActiv(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class ConvUNetBasicBlock(nn.Module):
    def __init__(self, input_channels=2, **kwargs):
        super().__init__()
        self.n_channels = input_channels

        self.bilinear = kwargs.get('bilinear', True)
        self.return_conf = kwargs.get('return_conf', False)
        factor = 2

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2,
                              mode='bilinear',
                              align_corners=True)

        self.upmask = nn.Upsample(scale_factor=2,
                                  mode='nearest')

        self.nb_filter = [64, 128, 256, 512, 1024]
        self.econv0_0 = ConvBActiv(input_channels, self.nb_filter[0], self.nb_filter[0])
        self.econv1_0 = ConvBActiv(self.nb_filter[0], self.nb_filter[1], self.nb_filter[1])
        self.econv2_0 = ConvBActiv(self.nb_filter[1], self.nb_filter[2], self.nb_filter[2])
        self.econv3_0 = ConvBActiv(self.nb_filter[2], self.nb_filter[3], self.nb_filter[3])
        self.econv4_0 = ConvBActiv(self.nb_filter[3], self.nb_filter[4] // factor, self.nb_filter[4] // factor)

        self.dconv3_1 = ConvBActiv(self.nb_filter[4], self.nb_filter[3] // factor,
                                  self.nb_filter[3] // factor)
        self.dconv2_2 = ConvBActiv(self.nb_filter[3], self.nb_filter[2] // factor,
                                  self.nb_filter[2] // factor)
        self.dconv1_3 = ConvBActiv(self.nb_filter[2], self.nb_filter[1] // factor,
                                  self.nb_filter[1] // factor)
        self.dconv0_4 = ConvBActiv(self.nb_filter[1], self.nb_filter[0],
                                  self.nb_filter[0])
        self.gconv3_1 = ConvBActiv(self.nb_filter[4], self.nb_filter[3] // factor,
                                   self.nb_filter[3] // factor)
        self.gconv2_2 = ConvBActiv(self.nb_filter[3], self.nb_filter[2] // factor,
                                   self.nb_filter[2] // factor)
        self.gconv1_3 = ConvBActiv(self.nb_filter[2], self.nb_filter[1] // factor,
                                   self.nb_filter[1] // factor)
        self.gconv0_4 = ConvBActiv(self.nb_filter[1], self.nb_filter[0],
                                   self.nb_filter[0])
    def forward(self, x):
        x0_0 = self.econv0_0(x)
        x1_0 = self.econv1_0(self.pool(x0_0))
        x2_0 = self.econv2_0(self.pool(x1_0))
        x3_0 = self.econv3_0(self.pool(x2_0))
        x4_0 = self.econv4_0(self.pool(x3_0))

        d3_1 = self.dconv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        d2_2 = self.dconv2_2(torch.cat([x2_0, self.up(d3_1)], 1))
        d1_3 = self.dconv1_3(torch.cat([x1_0, self.up(d2_2)], 1))
        d0_4 = self.dconv0_4(torch.cat([x0_0, self.up(d1_3)], 1))

        g3_1 = self.gconv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        g2_2 = self.gconv2_2(torch.cat([x2_0, self.up(g3_1)], 1))
        g1_3 = self.gconv1_3(torch.cat([x1_0, self.up(g2_2)], 1))
        g0_4 = self.gconv0_4(torch.cat([x0_0, self.up(g1_3)], 1))

        return d0_4, d1_3, d2_2, d3_1, g0_4, g1_3, g2_2, g3_1

class UNetEDG(ConvUNetBasicBlock):
    def __init__(self, num_classes=1, input_channels=2, **kwargs):
        super().__init__(input_channels, **kwargs)
        self.n_classes = num_classes

        self.final0 = nn.Conv2d(self.nb_filter[0], num_classes, 1)
        self.final1 = nn.Conv2d(self.nb_filter[0], num_classes, 1)

    def forward(self, x):
        d0_4, d1_3, d2_2, d3_1, g0_4, g1_3, g2_2, g3_1 = super().forward(x)

        x0 = self.final0(d0_4)
        x1 = self.final1(g0_4)
        return x0, x1


if __name__ == '__main__':
    from torchsummary import summary

    summary(UNetEDG(), (2, 64, 64))
