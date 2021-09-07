#!/usr/bin/env python
import argparse
import json
import sys

from PIL import Image
import tqdm
import numpy as np
from model import *
from utils.visdataset import *
from utils.dataset_util import *

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################
class NetworkHolder:
    def __init__(self):
        self.net_ = None
        self.arch_ = None
        self.checkpoint_ = None

    def GetNetwork(self, arch, checkpoint):
        if self.net_!=None and self.arch_ == arch and self.checkpoint_ == checkpoint:
            return self.net_
        # reinitialize network
        self.net_=None
        try:
            self.net_ = eval(arch)
        except Exception as e:
            print(f'Init Network failed:\n {e}')
            return None
        assert (self.net_ is not None)
        self.net_.to(device=device)
        self.net_.load_state_dict(torch.load(checkpoint, map_location=device))
        self.arch_=arch
        self.checkpoint_ = checkpoint
        print(f'Init {arch} with {checkpoint}')
        return self.net_

    @staticmethod
    def processInput(pointdep, bgmask=False, invert=True, normalize=True, normalize_znear=False):

        dmax = pointdep.max()
        pointvalid = (pointdep > -1).astype(np.float32)
        pointdep[pointdep < 0] = dmax
        dmin = pointdep.min()
        if invert:
            pointdep = dmax - pointdep

        if normalize:
            if normalize_znear:
                pointdep = pointdep / (dmax-dmin)
            else:
                pointdep = pointdep / dmax

        return pointdep, pointvalid

    @staticmethod
    def predict_img(net, input, input_mask, device):
        net.eval()
        input = torch.from_numpy(input).reshape(1,1,*input.shape).to(device=device,dtype=torch.float32)
        input_mask = torch.from_numpy(input_mask).reshape(1,1,*input_mask.shape).to(device=device,dtype=torch.float32)
        with torch.no_grad():
            output = net(torch.cat([input,input_mask],1))
            probs = torch.sigmoid(output) * input_mask
            full_mask = probs.squeeze().cpu().numpy()

        return full_mask

nh = NetworkHolder()
def call_plugin(blockJson):
    if blockJson['Worker'] != "NETWORKVISIBILITY":
        return
    paramJson = blockJson['Param']
    input_depth = paramJson['input_depth']
    arch = paramJson['arch']
    checkpoint = paramJson['checkpoint']
    output_mask = paramJson['output_mask']

    # Get network
    net = nh.GetNetwork(arch, checkpoint)
    rawdep = readFlt(input_depth)
    ptdep, ptvalid = nh.processInput(rawdep,invert=True,normalize=True, normalize_znear=False)

    pred = nh.predict_img(net=net, input=ptdep, input_mask=ptvalid, device=device)
    pred = 255*pred

    maskImg = Image.fromarray(pred.astype(np.uint8))
    maskImg.save(output_mask)
    print(f'Output: {output_mask}')

if __name__ == '__main__':
    print(f'Using device {device}')

    argparse = argparse.ArgumentParser()
    argparse.add_argument('input_json',type=str)
    args = argparse.parse_args()
    INPUT_JSON = args.input_json

    with open(INPUT_JSON,'r') as f:
        rootJson = json.load(f)
        if not isinstance(rootJson,list):
            rootJson = [rootJson]

    for blockJson in tqdm(rootJson):
        call_plugin(blockJson)

    sys.exit(0)
