import re
from glob import glob
from os.path import basename, join, splitext, relpath, abspath, exists
import random
import imageio
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from .dataset_util import *

class BasicDataset(Dataset):
    def createOrLoadSubset(self, valPercent):
        if exists(self.allEntryPath):
            with open(self.allEntryPath, 'r') as f:
                AllEntry = f.read().splitlines()
        else:
            filelist = glob(join(self.datafolder, '**/render/cam*.json'))
            AllEntry = [relpath(i, self.datafolder) for i in filelist]
            with open(self.allEntryPath, 'w') as f:
                f.writelines([i + '\n' for i in AllEntry])

        if exists(self.trainEntryPath) and exists(self.valEntryPath):
            with open(self.trainEntryPath, 'r') as f:
                TrainEntry = f.read().splitlines()
            with open(self.valEntryPath, 'r') as f:
                ValEntry = f.read().splitlines()
        else:
            ShuffleEntry = AllEntry.copy()
            random.shuffle(ShuffleEntry)

            nDataset = len(ShuffleEntry)
            nVal = int(nDataset * valPercent)
            nTrain = nDataset - nVal
            TrainEntry = ShuffleEntry[:nTrain]
            ValEntry = ShuffleEntry[nTrain:]
            with open(self.trainEntryPath, 'w') as f:
                f.writelines([i + '\n' for i in TrainEntry])
            with open(self.valEntryPath, 'w') as f:
                f.writelines([i + '\n' for i in ValEntry])
        return AllEntry, TrainEntry, ValEntry

    def __init__(self, Datafolder, **kwargs):
        super().__init__()
        self.datafolder = Datafolder
        self.allEntryPath = join(Datafolder, 'all.txt')
        self.trainEntryPath = join(Datafolder, 'train.txt')
        self.valEntryPath = join(Datafolder, 'val.txt')

        valPercent = kwargs.get('val_percent', 0.2)
        subset = kwargs.get('subset', 'all')
        AllEntry, TrainEntry, ValEntry = self.createOrLoadSubset(valPercent)

        customizedEntryPath = join(Datafolder,f'{subset}.txt')
        entry = None
        if subset == 'train':
            entry = TrainEntry
        elif subset == 'val':
            entry = ValEntry
        elif exists(customizedEntryPath):
            with open(customizedEntryPath, 'r') as f:
                entry = f.read().splitlines()
        else:
            raise FileNotFoundError(f"Subset {subset} not found.")

        # expand
        self.record = []
        for cam in entry:
            cam = join(Datafolder, cam)
            assert (exists(cam))
            d = {
                'cam':
                    cam,
                'ptRGB':
                    re.sub(r'cam(\d+).json', lambda x: f'pt{x.group(1)}.png', cam),
                'ptDep':
                    re.sub(r'cam(\d+).json', lambda x: f'pt{x.group(1)}.flt', cam),
                'ptId':
                    re.sub(r'cam(\d+).json', lambda x: f'pt{x.group(1)}.uint',
                           cam),
                'meshDep':
                    re.sub(r'cam(\d+).json', lambda x: f'mesh{x.group(1)}.flt',
                           cam),
                'meshId':
                    re.sub(r'cam(\d+).json', lambda x: f'mesh{x.group(1)}.uint',
                           cam),
                # 'meshcullDep':
                #     re.sub(r'cam(\d+).json', lambda x: f'meshcull{x.group(1)}.flt',
                #            cam),
                # 'meshcullId':
                #     re.sub(r'cam(\d+).json',
                #            lambda x: f'meshcull{x.group(1)}.uint', cam),
                'occGT':
                    re.sub(join('render', r'cam(\d+).json'), lambda x: join('conf',f'conf{x.group(1)}.png'), cam),
            }
            for k, v in d.items():
                assert exists(v), f'{v} not exists.'
            self.record.append(d)

    def __len__(self):
        return len(self.record)

    def __getitem__(self, i):
        return self.record[i]


class CamSparseToDenseDataset(BasicDataset):
    def __init__(self, Datafolder, bgmask=False, **kwargs):
        super().__init__(Datafolder, **kwargs)
        self.normalize = kwargs.get('normalize', False)
        self.normalize_near = kwargs.get('normalize_near', False)
        self.invert = kwargs.get('invert', True)
        self.bgmask = bgmask
        self.depth_threshold = 0.5

    def __getitem__(self, i):
        rec = self.record[i]

        cam = readCam(rec['cam'])
        pointdep = readFlt(rec['ptDep'])
        meshdep = readFlt(rec['meshDep'])
        # meshculldep = readFlt(rec['meshcullDep'])

        occGT = np.asarray((imageio.imread(rec['occGT'])>0) * 1.0)

        camK = cam['K'].astype(np.float32)

        pointvalid = (pointdep > -1).astype(np.int8)
        meshvalid = (meshdep > -1).astype(np.int8)

        dmax = pointdep.max()
        pointdep[pointdep < 0] = dmax
        meshdep[meshdep < 0] = dmax
        # meshculldep[meshculldep < 0] = dmax
        dmin = pointdep.min()
        if self.bgmask:
            bgmask = pointvalid * occGT

        if self.invert:
            pointdep = dmax - pointdep
            meshdep = dmax - meshdep
            # meshculldep = dmax - meshculldep

        if self.normalize:
            if self.normalize_near:
                pointdep = pointdep / (dmax - dmin)
                meshdep = meshdep / (dmax - dmin)
                # meshculldep = meshculldep / (dmax - dmin)
            else:
                pointdep = pointdep / dmax
                meshdep = meshdep / dmax
                # meshculldep = meshculldep / dmax

        input = pointdep[np.newaxis, :, :]
        target = meshdep[np.newaxis, :, :]

        if self.bgmask:
            return camK, input, pointvalid[np.newaxis, :, :], \
                   target, meshvalid[np.newaxis, :, :], bgmask[np.newaxis, :, :]
        else:
            return camK, input, pointvalid[np.newaxis, :, :], \
                   target, meshvalid[np.newaxis, :, :]


def process_feat(rec):
    cam = readCam(rec['cam'])
    pointdep = readFlt(rec['ptDep'])
    featPath = re.sub(r'cam(\d+).json', lambda x: f'feat{x.group(1)}.npy',
                      rec['cam'])

    if exists(featPath):
        try:
            feat = np.load(open(featPath, 'rb'))
        except:
            print(f'Load error {featPath}')

        if feat.shape == (4, *pointdep.shape):
            return
        else:
            print(f'corrupt file {featPath}')

    try:
        feat = localFeatStructuredPoints(pointdep, cam)
    except Exception as e:
        print(f'Error: {featPath}')
    np.save(featPath, feat)


def local_feature_extraction(Datafolder):
    from multiprocessing import Pool, cpu_count
    dataset = BasicDataset(Datafolder, subset='all')

    with Pool(cpu_count()) as p:
        r = list(tqdm(p.imap(process_feat, dataset), total=len(dataset)))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    Datafolder = r'/home/sxs/GDA/iccv21_vis2mesh/MakeLabel/HKUST/hkust_building1'

    ds = CamSparseToDenseDataset(Datafolder, bgmask=True, normalize=False, invert=False)
    # camK, input, input_mask, target, target_mask, bgmask = ds[8]
    dl = DataLoader(ds, shuffle=False, batch_size=2)
    print(dl)
    for camK, input, input_mask, target, meshvalid, target_mask in dl:
        B, C, H, W = input.shape
        print(camK.shape)
        print(input.shape)
        print(input_mask.shape)
        print(target.shape)
        print(target_mask.shape)
        print('range',input.min(),input.max())
    
        #####################################
        ## Point Feature Extraction Section
        dmax = input.max()
        camPts = unprojTensorImage(input[:, 0:1, :, :], camK)  # Unproject to 3D
        ########################################
    
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(camPts[0, :, :])
        
        color = np.zeros_like(camPts[0, :, :])
        bglabel = target_mask[0]
        color[bglabel.view(bglabel.numel())>0,0] = 1
        pcd.colors = o3d.utility.Vector3dVector(color)

        o3d.visualization.draw_geometries([pcd])
        # break

    # for i in range(input.shape[0]):
    #     plt.figure()
    #     plt.colorbar(plt.imshow(input[i, :, :]))


    # plt.figure()
    # plt.imshow(bgmask[0,:,:])
    # plt.show( )
