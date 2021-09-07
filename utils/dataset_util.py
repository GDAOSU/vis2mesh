import numpy as np
import json
import imageio
import open3d as o3d
import torch

def readFlt(path):
    with open(path, 'rb') as f:
        shape = np.fromfile(f, np.int32, count=2)
        data = np.fromfile(f, np.float32).reshape(shape)
        return data


def readUint(path):
    with open(path, 'rb') as f:
        shape = np.fromfile(f, np.int32, count=2)
        data = np.fromfile(f, np.uint32).reshape(shape)
        return data


def readPfm(path):
    return np.array(imageio.imread(path))


def readCam(path):
    camJson = json.load(open(path))
    for k in ['K', 'C', 'R']:
        camJson[k] = np.array(camJson[k])
    return camJson


def unprojImage(depImg, cam):
    grid_xmap, grid_ymap = np.meshgrid(range(0, depImg.shape[1]),
                                       range(0, depImg.shape[0]))
    grid_zmap = depImg

    x_n_c = np.vstack((grid_xmap.ravel(), grid_ymap.ravel(),
                       np.ones_like(grid_xmap.ravel())))
    pts_3d = np.dot(np.linalg.inv(cam['K']), x_n_c * grid_zmap.ravel())
    pts_3d = pts_3d.reshape((3, *grid_zmap.shape))
    return pts_3d


def unproj(depImg, cam, returnMask=False, returnIndex=False):
    grid_valid_mask = depImg > 0

    ptsImage = unprojImage(depImg, cam)

    X_c = ptsImage[:, grid_valid_mask]

    if returnMask and not returnIndex:
        return X_c, grid_valid_mask
    elif returnIndex and not returnMask:
        return X_c, np.where(grid_valid_mask)
    elif returnMask and returnIndex:
        return X_c, grid_valid_mask, np.where(grid_valid_mask)
    else:
        return X_c


def depthMapNormalEstimate(xyzImg):
    _, height, width = xyzImg.shape
    padDepImg = np.pad(xyzImg, [[0, 0], [1, 1], [1, 1]], 'reflect')

    gradVolume = np.zeros((4, *xyzImg.shape))
    gradVolume[0, :, :, :] = padDepImg[:, :height, 1:width +
                                       1] - padDepImg[:, 1:height + 1,
                                                      1:width +
                                                      1]  # North - Center
    gradVolume[1, :, :, :] = padDepImg[:, 2:height + 2, 1:width +
                                       1] - padDepImg[:, 1:height + 1,
                                                      1:width +
                                                      1]  # South - Center
    gradVolume[
        2, :, :, :] = padDepImg[:, 1:height +
                                1, :width] - padDepImg[:, 1:height + 1,
                                                       1:width +
                                                       1]  # West - Center
    gradVolume[3, :, :, :] = padDepImg[:, 1:height + 1, 2:width +
                                       2] - padDepImg[:, 1:height + 1,
                                                      1:width +
                                                      1]  # East - Center

    depthDiffV = np.fabs(gradVolume[0, 2, :, :]) > np.fabs(gradVolume[1,
                                                                      2, :, :])
    depthDiffH = np.fabs(gradVolume[2, 2, :, :]) > np.fabs(gradVolume[3,
                                                                      2, :, :])

    def normalizeDepthMap(data):
        return data / np.linalg.norm(data, axis=2)[:, :, np.newaxis]

    #
    # def approach1(gradVolume, depthDiffV, depthDiffH):
    #     normal_map_stack = np.zeros((4, *gradVolume.shape[2:], 3))
    #     normal_map_stack[0, :, :, :] = normalizeDepthMap(
    #         np.cross(gradVolume[3, :, :, :], gradVolume[0, :, :, :], axisa=0, axisb=0))  # EN
    #     normal_map_stack[1, :, :, :] = normalizeDepthMap(
    #         np.cross(gradVolume[0, :, :, :], gradVolume[2, :, :, :], axisa=0, axisb=0))  # NW
    #     normal_map_stack[2, :, :, :] = normalizeDepthMap(
    #         np.cross(gradVolume[2, :, :, :], gradVolume[1, :, :, :], axisa=0, axisb=0))  # WS
    #     normal_map_stack[3, :, :, :] = normalizeDepthMap(
    #         np.cross(gradVolume[1, :, :, :], gradVolume[3, :, :, :], axisa=0, axisb=0))  # SE
    #
    #     normal_map = np.zeros((*gradVolume.shape[2:], 3))
    #
    #     maskEN = np.logical_and(depthDiffH, ~depthDiffV)
    #     maskNW = np.logical_and(~depthDiffH, ~depthDiffV)
    #     maskWS = np.logical_and(~depthDiffH, depthDiffV)
    #     maskSE = np.logical_and(depthDiffH, depthDiffV)
    #
    #     normal_map[maskEN] = normal_map_stack[0, maskEN, :]
    #     normal_map[maskNW] = normal_map_stack[1, maskNW, :]
    #     normal_map[maskWS] = normal_map_stack[2, maskWS, :]
    #     normal_map[maskSE] = normal_map_stack[3, maskSE, :]
    #
    #     return normal_map

    gradV = np.zeros(gradVolume.shape[1:])
    gradH = np.zeros(gradVolume.shape[1:])

    gradV[:, depthDiffV] = -gradVolume[1, :, depthDiffV].T
    gradV[:, ~depthDiffV] = gradVolume[0, :, ~depthDiffV].T
    gradH[:, depthDiffH] = -gradVolume[3, :, depthDiffH].T
    gradH[:, ~depthDiffH] = gradVolume[2, :, ~depthDiffH].T

    return normalizeDepthMap(np.cross(gradV, gradH, axisa=0, axisb=0))


def fgmask_1(pointdep, meshdep, meshculldep, threshold_depth_diff=1):
    validmask = 255 * (np.abs(pointdep - meshdep) < threshold_depth_diff) * (
        pointdep > -1) * (np.abs(meshculldep - meshdep) < threshold_depth_diff)
    return validmask.astype(np.uint8)


def fgmask_2(pointdep, meshdep, meshculldep, threshold_depth_diff=1):
    height, width = meshdep.shape
    padmeshdep = np.pad(meshdep, [[1, 1], [1, 1]], mode='reflect')
    diffVol = np.zeros((9, *meshdep.shape))
    diffVol[0, :, :] = pointdep - padmeshdep[:height, :width]
    diffVol[1, :, :] = pointdep - padmeshdep[1:height + 1, :width]
    diffVol[2, :, :] = pointdep - padmeshdep[2:height + 2, :width]
    diffVol[3, :, :] = pointdep - padmeshdep[:height, 1:width + 1]
    diffVol[4, :, :] = pointdep - padmeshdep[1:height + 1, 1:width + 1]
    diffVol[5, :, :] = pointdep - padmeshdep[2:height + 2, 1:width + 1]
    diffVol[6, :, :] = pointdep - padmeshdep[:height, 2:width + 2]
    diffVol[7, :, :] = pointdep - padmeshdep[1:height + 1, 2:width + 2]
    diffVol[8, :, :] = pointdep - padmeshdep[2:height + 2, 2:width + 2]

    diffmap = np.min(np.abs(diffVol), axis=0)
    validmask = 255 * (np.abs(diffmap) < threshold_depth_diff) * (
        pointdep > -1) * (np.abs(meshculldep - meshdep) < threshold_depth_diff)
    return validmask.astype(np.uint8)


def bgmask_1(pointdep, meshdep, meshculldep, threshold_depth_diff=1):
    fgmask = fgmask_1(pointdep, meshdep, meshculldep, threshold_depth_diff)
    return ((255 - fgmask) * (pointdep > -1)).astype(np.uint8)


def bgmask_2(pointdep, meshdep, meshculldep, threshold_depth_diff=1):
    fgmask = fgmask_2(pointdep, meshdep, meshculldep, threshold_depth_diff)
    return ((255 - fgmask) * (pointdep > -1)).astype(np.uint8)


def localFeatStructuredPoints(pointdep, cam, K=6):
    pointXYZ = unprojImage(pointdep, cam)
    vecXYZ = pointXYZ.reshape(3, -1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vecXYZ.T)

    feat = np.zeros((4, vecXYZ.shape[1]), dtype=np.float)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for i in range(vecXYZ.shape[1]):
        p = vecXYZ[:, i]
        k, idx, radius = kdtree.search_knn_vector_3d(p, K + 1)
        ck = np.mean(vecXYZ[:, idx], axis=1)
        r = np.mean(radius[1:])
        v = ck - p
        feat[0, i] = r
        feat[1:4, i] = v
    feat = feat.reshape((4, *pointdep.shape))
    return feat

def unprojTensorImage(input, camK):
    B, C, H, W = input.shape
    grid_xmap, grid_ymap = torch.meshgrid(torch.arange(0, input.shape[-1]),
                                          torch.arange(0, input.shape[-2]))
    xarr = grid_xmap.repeat(B,1,1).reshape(B, C, -1).to(input.device)
    yarr = grid_ymap.repeat(B,1,1).reshape(B, C, -1).to(input.device)
    zarr = input.reshape(B, C, -1)

    imgPts = torch.cat([xarr * zarr, yarr * zarr, zarr], dim=1)
    camPts = torch.einsum("bij,bjk->bik", torch.inverse(camK), imgPts)

    return camPts.permute([0,2,1])


def depthMapNormalTensorEstimate(xyzImg):
    B, C, height, width = xyzImg.shape
    padDepImg = np.pad(xyzImg, [[0, 0], [1, 1], [1, 1]], 'reflect')

    gradVolume = np.zeros((4, *xyzImg.shape))
    gradVolume[0, :, :, :] = padDepImg[:, :height, 1:width +
                                       1] - padDepImg[:, 1:height + 1,
                                                      1:width +
                                                      1]  # North - Center
    gradVolume[1, :, :, :] = padDepImg[:, 2:height + 2, 1:width +
                                       1] - padDepImg[:, 1:height + 1,
                                                      1:width +
                                                      1]  # South - Center
    gradVolume[
        2, :, :, :] = padDepImg[:, 1:height +
                                1, :width] - padDepImg[:, 1:height + 1,
                                                       1:width +
                                                       1]  # West - Center
    gradVolume[3, :, :, :] = padDepImg[:, 1:height + 1, 2:width +
                                       2] - padDepImg[:, 1:height + 1,
                                                      1:width +
                                                      1]  # East - Center

    depthDiffV = np.fabs(gradVolume[0, 2, :, :]) > np.fabs(gradVolume[1,
                                                                      2, :, :])
    depthDiffH = np.fabs(gradVolume[2, 2, :, :]) > np.fabs(gradVolume[3,
                                                                      2, :, :])

    def normalizeDepthMap(data):
        return data / np.linalg.norm(data, axis=2)[:, :, np.newaxis]

    #
    # def approach1(gradVolume, depthDiffV, depthDiffH):
    #     normal_map_stack = np.zeros((4, *gradVolume.shape[2:], 3))
    #     normal_map_stack[0, :, :, :] = normalizeDepthMap(
    #         np.cross(gradVolume[3, :, :, :], gradVolume[0, :, :, :], axisa=0, axisb=0))  # EN
    #     normal_map_stack[1, :, :, :] = normalizeDepthMap(
    #         np.cross(gradVolume[0, :, :, :], gradVolume[2, :, :, :], axisa=0, axisb=0))  # NW
    #     normal_map_stack[2, :, :, :] = normalizeDepthMap(
    #         np.cross(gradVolume[2, :, :, :], gradVolume[1, :, :, :], axisa=0, axisb=0))  # WS
    #     normal_map_stack[3, :, :, :] = normalizeDepthMap(
    #         np.cross(gradVolume[1, :, :, :], gradVolume[3, :, :, :], axisa=0, axisb=0))  # SE
    #
    #     normal_map = np.zeros((*gradVolume.shape[2:], 3))
    #
    #     maskEN = np.logical_and(depthDiffH, ~depthDiffV)
    #     maskNW = np.logical_and(~depthDiffH, ~depthDiffV)
    #     maskWS = np.logical_and(~depthDiffH, depthDiffV)
    #     maskSE = np.logical_and(depthDiffH, depthDiffV)
    #
    #     normal_map[maskEN] = normal_map_stack[0, maskEN, :]
    #     normal_map[maskNW] = normal_map_stack[1, maskNW, :]
    #     normal_map[maskWS] = normal_map_stack[2, maskWS, :]
    #     normal_map[maskSE] = normal_map_stack[3, maskSE, :]
    #
    #     return normal_map

    gradV = np.zeros(gradVolume.shape[1:])
    gradH = np.zeros(gradVolume.shape[1:])

    gradV[:, depthDiffV] = -gradVolume[1, :, depthDiffV].T
    gradV[:, ~depthDiffV] = gradVolume[0, :, ~depthDiffV].T
    gradH[:, depthDiffH] = -gradVolume[3, :, depthDiffH].T
    gradH[:, ~depthDiffH] = gradVolume[2, :, ~depthDiffH].T

    return normalizeDepthMap(np.cross(gradV, gradH, axisa=0, axisb=0))

if __name__ == '__main__':
    from os.path import join
    import open3d as o3d
    import matplotlib.pyplot as plt
    dataset_folder = r'dataset/hkust_building4/'
    camPath = join(dataset_folder, r'render/cam100.json')
    meshdepPath = join(dataset_folder, r'render/mesh100.flt')
    meshculldepPath = join(dataset_folder,
                           r'render/meshcull100.flt')
    pointdepPath = join(dataset_folder, r'render/pt100.flt')
    pointidPath = join(dataset_folder, r'render/pt100.uint')

    pointdep = readFlt(pointdepPath)
    pointid = readUint(pointidPath)
    cam = readCam(camPath)
    meshdep = readFlt(meshdepPath)
    meshculldep = readFlt(meshculldepPath)

    meshdepXYZMap = unprojImage(meshdep, cam)
    # meshNormalMap = depthMapNormalEstimate(meshdepXYZMap)

    # pointXYZ, pointXYZMask = unproj(meshdepXYZMap, cam, returnMask=True)
    # normalXYZ = meshNormalMap[pointXYZMask, :]
    # normalXYZ = meshNormalMap[pointXYZMask, :]


    bg_target = bgmask_2(pointdep,meshdep,meshculldep)
    cleandep = pointdep* (bg_target<1)
    plt.figure();
    plt.imshow(-meshdep)
    plt.figure();plt.imshow(-pointdep)
    plt.figure();plt.imshow(bg_target)
    plt.figure();plt.imshow(-cleandep)
    plt.show()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(meshdepXYZMap.reshape(3,-1).T)
    # pcd.normals = o3d.utility.Vector3dVector(normalXYZ)
    # # pcd.normalize_normals()
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)


