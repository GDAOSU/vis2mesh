import torch.nn as nn

from .model_parts.nconv import *


class NConvBActiv(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, pos_fn=None):
        super().__init__()
        self.pos_fn = pos_fn
        self.relu = nn.ReLU(inplace=True)
        self.nconv1 = NConv2d(in_channels, middle_channels, (3, 3), pos_fn, 'p', padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.nconv2 = NConv2d(middle_channels, out_channels, (3, 3), pos_fn, 'p', padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, x_conf):
        x, x_conf = self.nconv1(x, x_conf)
        x = self.bn1(x)
        x = self.relu(x)
        x, x_conf = self.nconv2(x, x_conf)
        x = self.bn2(x)
        x = self.relu(x)
        return x, x_conf


class MyNConvUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=2, **kwargs):
        super().__init__()
        assert (input_channels >= 2)
        self.n_channels = input_channels
        self.n_classes = num_classes
        self.bilinear = kwargs.get('bilinear', True)
        self.return_conf = kwargs.get('return_conf', False)
        self.pos_fn = kwargs.get('pos_fn', None)
        factor = 2

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2,
                              mode='bilinear',
                              align_corners=True)

        self.upmask = nn.Upsample(scale_factor=2,
                                  mode='nearest')

        nb_filter = [64, 128, 256, 512, 1024]
        self.conv0_0 = NConvBActiv(input_channels - 1, nb_filter[0], nb_filter[0], self.pos_fn)
        self.conv1_0 = NConvBActiv(nb_filter[0], nb_filter[1], nb_filter[1], self.pos_fn)
        self.conv2_0 = NConvBActiv(nb_filter[1], nb_filter[2], nb_filter[2], self.pos_fn)
        self.conv3_0 = NConvBActiv(nb_filter[2], nb_filter[3], nb_filter[3], self.pos_fn)
        self.conv4_0 = NConvBActiv(nb_filter[3], nb_filter[4] // factor, nb_filter[4] // factor, self.pos_fn)

        self.conv3_1 = NConvBActiv(nb_filter[4], nb_filter[3] // factor,
                                   nb_filter[3] // factor, self.pos_fn)
        self.conv2_2 = NConvBActiv(nb_filter[3], nb_filter[2] // factor,
                                   nb_filter[2] // factor, self.pos_fn)
        self.conv1_3 = NConvBActiv(nb_filter[2], nb_filter[1] // factor,
                                   nb_filter[1] // factor, self.pos_fn)
        self.conv0_4 = NConvBActiv(nb_filter[1], nb_filter[0],
                                   nb_filter[0], self.pos_fn)
        self.final = NConv2d(nb_filter[0], num_classes, (1, 1), self.pos_fn, 'p')

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


class NConvUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=2, **kwargs):
        super().__init__()
        self.__name__ = 'NConvUNet'
        self.bilinear = kwargs.get('bilinear', True)
        self.return_conf = kwargs.get('return_conf', False)
        self.pos_fn = kwargs.get('pos_fn', None)
        num_channels = kwargs.get('mid_channels',8)

        input_channels = input_channels - 1
        self.nconv1 = NConv2d(input_channels, input_channels * num_channels, (5, 5), self.pos_fn, 'k', padding=2)
        self.nconv2 = NConv2d(input_channels * num_channels, input_channels * num_channels, (5, 5), self.pos_fn, 'k', padding=2)
        self.nconv3 = NConv2d(input_channels * num_channels, input_channels * num_channels, (5, 5), self.pos_fn, 'k', padding=2)

        self.nconv4 = NConv2d(2 * input_channels * num_channels, input_channels * num_channels, (3, 3), self.pos_fn, 'k', padding=1)
        self.nconv5 = NConv2d(2 * input_channels * num_channels, input_channels * num_channels, (3, 3), self.pos_fn, 'k', padding=1)
        self.nconv6 = NConv2d(2 * input_channels * num_channels, input_channels * num_channels, (3, 3), self.pos_fn, 'k', padding=1)

        self.nconv7 = NConv2d(input_channels * num_channels, num_classes, (1, 1), self.pos_fn, 'k')

    def forward(self, x):
        datach = [i for i in range(x.shape[1]) if i != 1]
        maskch = [1]

        x0 = x[:, datach, :, :]
        c0 = x[:, maskch, :, :].repeat(1, x.shape[1] - 1, 1, 1)

        x1, c1 = self.nconv1(x0, c0)
        x1, c1 = self.nconv2(x1, c1)
        x1, c1 = self.nconv3(x1, c1)

        # Downsample 1
        ds = 2
        c1_ds, idx = F.max_pool2d(c1, ds, ds, return_indices=True)
        x1_ds = torch.zeros(c1_ds.size()).to(x0.get_device())
        for i in range(x1_ds.size(0)):
            for j in range(x1_ds.size(1)):
                x1_ds[i, j, :, :] = x1[i, j, :, :].view(-1)[idx[i, j, :, :].view(-1)].view(idx.size()[2:])
        c1_ds /= 4

        x2_ds, c2_ds = self.nconv2(x1_ds, c1_ds)
        x2_ds, c2_ds = self.nconv3(x2_ds, c2_ds)

        # Downsample 2
        ds = 2
        c2_dss, idx = F.max_pool2d(c2_ds, ds, ds, return_indices=True)

        x2_dss = torch.zeros(c2_dss.size()).to(x0.get_device())
        for i in range(x2_dss.size(0)):
            for j in range(x2_dss.size(1)):
                x2_dss[i, j, :, :] = x2_ds[i, j, :, :].view(-1)[idx[i, j, :, :].view(-1)].view(idx.size()[2:])
        c2_dss /= 4

        x3_ds, c3_ds = self.nconv2(x2_dss, c2_dss)

        # Downsample 3
        ds = 2
        c3_dss, idx = F.max_pool2d(c3_ds, ds, ds, return_indices=True)

        x3_dss = torch.zeros(c3_dss.size()).to(x0.get_device())
        for i in range(x3_dss.size(0)):
            for j in range(x3_dss.size(1)):
                x3_dss[i, j, :, :] = x3_ds[i, j, :, :].view(-1)[idx[i, j, :, :].view(-1)].view(idx.size()[2:])
        c3_dss /= 4
        x4_ds, c4_ds = self.nconv2(x3_dss, c3_dss)

        # Upsample 1
        x4 = F.interpolate(x4_ds, c3_ds.size()[2:], mode='nearest')
        c4 = F.interpolate(c4_ds, c3_ds.size()[2:], mode='nearest')
        x34_ds, c34_ds = self.nconv4(torch.cat((x3_ds, x4), 1), torch.cat((c3_ds, c4), 1))

        # Upsample 2
        x34 = F.interpolate(x34_ds, c2_ds.size()[2:], mode='nearest')
        c34 = F.interpolate(c34_ds, c2_ds.size()[2:], mode='nearest')
        x23_ds, c23_ds = self.nconv5(torch.cat((x2_ds, x34), 1), torch.cat((c2_ds, c34), 1))

        # Upsample 3
        x23 = F.interpolate(x23_ds, x0.size()[2:], mode='nearest')
        c23 = F.interpolate(c23_ds, c0.size()[2:], mode='nearest')
        xout, cout = self.nconv6(torch.cat((x23, x1), 1), torch.cat((c23, c1), 1))

        xout, cout = self.nconv7(xout, cout)

        if self.return_conf:
            return xout, cout
        else:
            return xout

if __name__ == '__main__':
    from torchsummary import summary

    summary(NConvUNet(input_channels=3,mid_channels=16), (3, 64, 64))
