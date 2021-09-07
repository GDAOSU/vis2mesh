#!/usr/bin/env python
import argparse
import json
import sys

from PIL import Image
from tqdm import tqdm
import numpy as np

import open3d as o3d
from utils.dataset_util import *

######## Hidden Point Removal ######################################
def hpr(camK, input, input_mask, param):
    dmax = input.max()
    ptsImage = unprojImage(input, {'K': camK})
    grid_valid_mask = input_mask > 0
    camPts = ptsImage[:, grid_valid_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(camPts.T)
    # HPR
    mesh, inlier = pcd.hidden_point_removal(np.array([0, 0, 0]), dmax * (10 ** param))
    label = np.zeros(camPts.shape[1])
    label[inlier] = 1
    labelmap = np.zeros(input.shape)
    labelmap[grid_valid_mask] = label
    return labelmap

def call_plugin(blockJson):
    if blockJson['Worker'] != "O3DHPRVISIBILITY":
        return
    paramJson = blockJson['Param']
    input_depth = paramJson['input_depth']
    input_cam = paramJson['input_cam']
    radius_exp = paramJson['radius_exp']
    output_mask = paramJson['output_mask']

    rawdep = readFlt(input_depth)
    validmask = (rawdep>0).astype(np.float32)
    rawcam = readCam(input_cam)

    inliermap = hpr(np.asarray(rawcam['K']), rawdep, validmask, radius_exp)

    outliermap = (1-inliermap)*validmask
    outliermap *= 255

    maskImg = Image.fromarray(outliermap.astype(np.uint8))
    maskImg.save(output_mask)
    print(f'Output: {output_mask}')

if __name__ == '__main__':
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
