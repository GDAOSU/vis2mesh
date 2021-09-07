import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 active_fn):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        self.bias = nn.Parameter(
            torch.zeros(out_channels),
            requires_grad=True)

        self.sparsity = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        kernel = torch.FloatTensor(torch.ones([kernel_size, kernel_size])).unsqueeze(0).unsqueeze(0)

        self.sparsity.weight = nn.Parameter(
            data=kernel,
            requires_grad=False)
        self.active_fn = active_fn
        if self.active_fn:
            self.relu = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(
            kernel_size,
            stride=1,
            padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, mask):
        x = x * mask
        x = self.conv(x)
        normalizer = 1 / (self.sparsity(mask) + 1e-8)
        x = x * normalizer + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = self.bn(x)
        if self.active_fn:
            x = self.relu(x)

        mask = self.max_pool(mask)

        return x, mask


class SparseConvNet(nn.Module):

    def __init__(self, num_classes=1, input_channels=2, **kwargs):
        super().__init__()
        assert (input_channels >= 2)
        self.n_channels = input_channels
        self.n_classes = num_classes
        self.return_conf = kwargs.get('return_conf', False)

        self.SparseLayer1 = SparseConv(input_channels - 1, 16, 11, active_fn=True)
        self.SparseLayer2 = SparseConv(16, 16, 7, active_fn=True)
        self.SparseLayer3 = SparseConv(16, 16, 5, active_fn=True)
        self.SparseLayer4 = SparseConv(16, 16, 3, active_fn=True)
        self.SparseLayer5 = SparseConv(16, 16, 3, active_fn=True)
        self.SparseLayer6 = SparseConv(16, num_classes, 1, active_fn=False)

    def forward(self, x):
        datach = [i for i in range(x.shape[1]) if i != 1]
        maskch = [1]

        x_data = x[:, datach, :, :]
        x_mask = x[:, maskch, :, :]

        x, x_conf = self.SparseLayer1(x_data, x_mask)
        x, x_conf = self.SparseLayer2(x, x_conf)
        x, x_conf = self.SparseLayer3(x, x_conf)
        x, x_conf = self.SparseLayer4(x, x_conf)
        x, x_conf = self.SparseLayer5(x, x_conf)
        x, x_conf = self.SparseLayer6(x, x_conf)

        if self.return_conf:
            return x, x_conf
        else:
            return x

if __name__ == '__main__':
    from torchsummary import summary

    summary(SparseConvNet(input_channels=2), (2, 64, 64))
    summary(SparseConvNet(input_channels=5), (5, 64, 64))
