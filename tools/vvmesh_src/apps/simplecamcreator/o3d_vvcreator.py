#!/usr/bin/env python
import os
import os.path as osp
import sys
import json
import math
import argparse
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('cloud', type=str, help='input point cloud')
    parser.add_argument('-il', '--input_list', help='initial camera json list')
    parser.add_argument('-ol', '--output_list', help='output camera json list')

    parser.add_argument('--focal', type=float, default=300, help='focal length')
    parser.add_argument('--width', type=int, default=512, help='camera width')
    parser.add_argument('--height', type=int, default=512, help='camera height')
    parser.add_argument('--cameraheight', type=float, default=50, help='camera pose generator default height')
    parser.add_argument('--camgenoverlap', type=float, default=0, help='camera pose generator default overlap rate')
    args = parser.parse_args()
    return args


def setFOV(ctr, fov):
    if fov < math.pi:
        fov_deg = fov * 180 / math.pi
    else:
        fov_deg = fov
    cur_fovy = ctr.get_field_of_view()
    diff_fov = fov_deg - cur_fovy
    step_fov = diff_fov / 5.0
    ctr.change_field_of_view(step=step_fov)
    return ctr


if __name__ == '__main__':
    args = parseArgs()
    print(args)

    print("Instruction")
    print("-----------")
    print(" [SPC] Record Current View")

    WHRatio = args.width / args.height
    prin_x = args.width / 2.0
    prin_y = args.height / 2.0

    K = np.array([[args.focal, 0, prin_x],
                  [0, args.focal, prin_y],
                  [0, 0, 1]])
    invK = np.linalg.inv(K)
    fovy = 2.0 * math.atan2(args.height, 2.0 * args.focal)
    nearClipping = 0.1
    farClipping = 1000

    ### data
    pcd = o3d.io.read_point_cloud(args.cloud)
    camjson = []

    if args.input_list and osp.exists(args.input_list):
        with open(args.input_list) as f:
            ondisk = json.load(f)
            if 'imgs' in ondisk:
                camjson = ondisk['imgs']
                print(f'Read in {len(camjson)} virtual views.')
    ### Vis and Interaction
    fig = plt.figure()
    fig.canvas.set_window_title(f"Num of Virtual Views: {len(camjson)}")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=args.width + 1, height=args.height + 1)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    setFOV(ctr, fovy)
    ctr.set_constant_z_near(nearClipping)
    ctr.set_constant_z_far(farClipping)
    print("Set z near %.2f" % nearClipping)
    print("Set z far %.2f" % farClipping)
    print("Set field of view %.2f" % ctr.get_field_of_view())


    def mkCamera(K, R, C):
        return {'width': args.width,
                'height': args.height,
                'C': [C[0], C[1], C[2]],
                'K': [[K[0, 0], K[0, 1], K[0, 2]],
                      [K[1, 0], K[1, 1], K[1, 2]],
                      [K[2, 0], K[2, 1], K[2, 2]]],
                'R': [[R[0, 0], R[0, 1], R[0, 2]],
                      [R[1, 0], R[1, 1], R[1, 2]],
                      [R[2, 0], R[2, 1], R[2, 2]]]
                }


    def capture_depth_and_img(vis):
        depth = np.asarray(vis.capture_depth_float_buffer())
        # image = vis.capture_screen_float_buffer()
        # plt.imshow(np.asarray(image))
        plt.imshow(depth)
        fig.canvas.set_window_title(f"Num of Virtual Views: {len(camjson)}")
        plt.draw()
        plt.pause(0.001)
        plt.pause(0.001)


    def capture_pose(vis):
        ctr = vis.get_view_control()
        camparam = ctr.convert_to_pinhole_camera_parameters()
        intrinsic = camparam.intrinsic.intrinsic_matrix
        extrinsic = camparam.extrinsic

        R = extrinsic[:3, :3]
        C = -R.T.dot(extrinsic[:3, 3])
        camjson.append(mkCamera(K, R, C))
        capture_depth_and_img(vis)

    vis.register_key_callback(ord(' '), capture_pose)
    vis.run()
    vis.destroy_window()

    # Output part
    content = json.dumps({'imgs': camjson}, indent=2)
    if args.output_list:
        with open(args.output_list, 'w') as f:
            f.write(content)
    else:
        print(content)
