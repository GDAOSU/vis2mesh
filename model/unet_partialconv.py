import torch
import torch.nn as nn
from .model_parts.partialconv2d import PartialConv2d


class PConvBActiv(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, multi_channel=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = PartialConv2d(in_channels,
                                   middle_channels,
                                   kernel_size=3,
                                   padding=1,
                                   multi_channel=multi_channel,
                                   return_mask=True)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = PartialConv2d(middle_channels,
                                   out_channels,
                                   kernel_size=3,
                                   padding=1,
                                   multi_channel=multi_channel,
                                   return_mask=True)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, x_conf):
        x, x_conf = self.conv1(x, x_conf)
        x = self.bn1(x)
        x = self.relu(x)
        x, x_conf = self.conv2(x, x_conf)
        x = self.bn2(x)
        x = self.relu(x)
        return x, x_conf


class PartialConvUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=2, **kwargs):
        super().__init__()
        assert (input_channels >= 2)
        self.n_channels = input_channels
        self.n_classes = num_classes
        self.bilinear = kwargs.get('bilinear', True)
        self.return_conf = kwargs.get('return_conf', False)
        factor = 2

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2,
                              mode='bilinear',
                              align_corners=True)

        self.upmask = nn.Sequential(
            nn.Upsample(scale_factor=2,
                        mode='nearest')
        )

        nb_filter = [64, 128, 256, 512, 1024]
        self.conv0_0 = PConvBActiv(input_channels - 1, nb_filter[0], nb_filter[0])
        self.conv1_0 = PConvBActiv(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = PConvBActiv(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = PConvBActiv(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = PConvBActiv(nb_filter[3], nb_filter[4] // factor, nb_filter[4] // factor)

        self.conv3_1 = PConvBActiv(nb_filter[4], nb_filter[3] // factor,
                                   nb_filter[3] // factor)
        self.conv2_2 = PConvBActiv(nb_filter[3], nb_filter[2] // factor,
                                   nb_filter[2] // factor)
        self.conv1_3 = PConvBActiv(nb_filter[2], nb_filter[1] // factor,
                                   nb_filter[1] // factor)
        self.conv0_4 = PConvBActiv(nb_filter[1], nb_filter[0],
                                   nb_filter[0])

        self.final = PartialConv2d(nb_filter[0],
                                   num_classes,
                                   kernel_size=1,
                                   multi_channel=False,
                                   return_mask=True)

    def forward(self, x):
        datach = [i for i in range(x.shape[1]) if i != 1]
        maskch = [1]

        x_data = x[:, datach, :, :]
        x_mask = x[:, maskch, :, :]

        x0_0, m0_0 = self.conv0_0(x_data, x_mask)
        x1_0, m1_0 = self.conv1_0(self.pool(x0_0), self.pool(m0_0))
        x2_0, m2_0 = self.conv2_0(self.pool(x1_0), self.pool(m1_0))
        x3_0, m3_0 = self.conv3_0(self.pool(x2_0), self.pool(m2_0))
        x4_0, m4_0 = self.conv4_0(self.pool(x3_0), self.pool(m3_0))

        x3_1, m3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1), self.upmask(m4_0))
        x2_2, m2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1), self.upmask(m3_1))
        x1_3, m1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1), self.upmask(m2_2))
        x0_4, m0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1), self.upmask(m1_3))

        output, output_mask = self.final(x0_4, m0_4)

        if self.return_conf:
            return output, output_mask
        else:
            return output


class PartialConvUNetMT(nn.Module):
    def __init__(self, num_classes=1, input_channels=2, **kwargs):
        super().__init__()
        assert (input_channels >= 2)
        self.n_channels = input_channels
        self.n_classes = num_classes
        self.bilinear = kwargs.get('bilinear', True)
        self.return_conf = kwargs.get('return_conf', False)
        factor = 2

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2,
                              mode='bilinear',
                              align_corners=True)

        self.upmask = nn.Sequential(
            nn.Upsample(scale_factor=2,
                        mode='nearest')
        )

        nb_filter = [64, 128, 256, 512, 1024]
        self.conv0_0 = PConvBActiv(input_channels - 1, nb_filter[0], nb_filter[0], True)
        self.conv1_0 = PConvBActiv(nb_filter[0], nb_filter[1], nb_filter[1], True)
        self.conv2_0 = PConvBActiv(nb_filter[1], nb_filter[2], nb_filter[2], True)
        self.conv3_0 = PConvBActiv(nb_filter[2], nb_filter[3], nb_filter[3], True)
        self.conv4_0 = PConvBActiv(nb_filter[3], nb_filter[4] // factor, nb_filter[4] // factor, True)

        self.conv3_1 = PConvBActiv(nb_filter[4], nb_filter[3] // factor,
                                   nb_filter[3] // factor, True)
        self.conv2_2 = PConvBActiv(nb_filter[3], nb_filter[2] // factor,
                                   nb_filter[2] // factor, True)
        self.conv1_3 = PConvBActiv(nb_filter[2], nb_filter[1] // factor,
                                   nb_filter[1] // factor, True)
        self.conv0_4 = PConvBActiv(nb_filter[1], nb_filter[0],
                                   nb_filter[0], True)

        self.final = PartialConv2d(nb_filter[0],
                                   num_classes,
                                   kernel_size=1,
                                   multi_channel=True,
                                   return_mask=True)

    def forward(self, x):
        datach = [i for i in range(x.shape[1]) if i != 1]
        maskch = [1]

        x_data = x[:, datach, :, :]
        x_mask = x[:, maskch, :, :].repeat(1, x.shape[1] - 1, 1, 1)

        x0_0, m0_0 = self.conv0_0(x_data, x_mask)
        x1_0, m1_0 = self.conv1_0(self.pool(x0_0), self.pool(m0_0))
        x2_0, m2_0 = self.conv2_0(self.pool(x1_0), self.pool(m1_0))
        x3_0, m3_0 = self.conv3_0(self.pool(x2_0), self.pool(m2_0))
        x4_0, m4_0 = self.conv4_0(self.pool(x3_0), self.pool(m3_0))

        x3_1, m3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1),
                                  torch.cat([m3_0, self.upmask(m4_0)], 1))
        x2_2, m2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1),
                                  torch.cat([m2_0, self.upmask(m3_1)], 1))
        x1_3, m1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1),
                                  torch.cat([m1_0, self.upmask(m2_2)], 1))
        x0_4, m0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1),
                                  torch.cat([m0_0, self.upmask(m1_3)], 1))

        output, output_mask = self.final(x0_4, m0_4)

        if self.return_conf:
            return output, output_mask
        else:
            return output


if __name__ == '__main__':
    from torchsummary import summary

    # summary(PartialConvUNet(input_channels=4).to(torch.device('cuda')), (4, 256, 256))
    summary(PartialConvUNet(input_channels=2).cuda(), (2, 256, 256))
