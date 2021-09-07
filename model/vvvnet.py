import torch
import torch.nn as nn
import torch.functional as F
from model import *

class VVVNet(nn.Module):
    def __init__(self, arch, trainable=[True,True,True], multitask = False):
        super().__init__()
        self.multitask = multitask
        assert(len(arch) == 3)
        self.net = nn.ModuleList()
        self.trainable = trainable
        for a in arch:
            try:
                _n = eval(a)
                # cmd = f'_n = {a}'
                # exec(cmd)
                # print(cmd)
                assert (_n is not None)
                self.net.append(_n)
            except Exception as e:
                raise RuntimeError(f'net architecture {a} init failed:\n {e}')

        assert (len(self.net) == 3)
        assert (len(self.trainable ) == 3)

    def forward(self, x):
        input_depth = x[:,0:1,:,:]
        input_mask = x[:,1:2,:,:] # second channel, the mask
        pred_vis = torch.sigmoid(self.net[0](x))
        pred_vis2 = torch.sigmoid(self.net[1](torch.cat([x, pred_vis], dim=1)))
        pred_finevis = self.net[2](torch.cat([x, pred_vis2], dim=1))

        if self.multitask:
            return pred_vis, pred_vis2, pred_finevis
        else:
            return pred_finevis

    def train(self, mode=True):
        super().train(mode)
        for i in range(len(self.net)):
            if self.trainable[i]:
                self.net[i].train(mode)
            else:
                self.net[i].eval()


def cvtCheckpoint3to1(arch,input,output):
    n = VVVNet(arch)
    statedicts = torch.load(input)
    n.net[0].load_state_dict(statedicts['net0'])
    n.net[1].load_state_dict(statedicts['net1'])
    n.net[2].load_state_dict(statedicts['net2'])
    torch.save(n.state_dict(), output)
    return n

if __name__ == '__main__':
    device = torch.device('cuda')
    arch = ['UNet(input_channels=2)', 'UNet(input_channels=3)', 'UNet(input_channels=3)']

    old_weight = "/home/sxs/GDA/cvpr21/visibility_learning_pytorch/checkpoints/VISVISVIS_['UNet(input_channels=2)', 'UNet(input_channels=3)', 'UNet(input_channels=3)']_epoch100.pth"
    new_weight = "/home/sxs/GDA/cvpr21/visibility_learning_pytorch/checkpoints/VISVISVIS_UUU_epoch100.pth"

    net = cvtCheckpoint3to1(arch,old_weight,new_weight).to(device=device)

    import torchsummary
    torchsummary.summary(net,(2,256,256))