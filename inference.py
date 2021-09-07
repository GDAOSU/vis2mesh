#!/usr/bin/env python
import os
import sys
import glob
import argparse
from os.path import join,dirname,abspath,exists,basename
import pathlib
import subprocess
import logging
import json
import re
import datetime
import shutil
import stat

# logname = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.log'
logging.basicConfig(level = logging.INFO)
############################################## Utils
def next_camname_in_folder(folder):
    cams = glob.glob(join(folder,'cam*.json'))
    maxid=-1
    for c in cams:
        res=re.search(r'cam(\d+)\.json',c)
        if len(res.groups())>0:
            maxid=max(maxid,int(res.group(1)))
    return join(folder,f'cam{maxid+1}.json')

def findToolset(dep_tools = ['vvtool','o3d_vvcreator.py','ReconstructMesh']):
    Toolset=dict()
    print('#Available Toolset#')
    for t in dep_tools:
        Toolset[t] = t
        assert(Toolset[t]!=None)
        print(f'{t}: {Toolset[t]}')
    return Toolset

################### End Utils

def Setup(args):
    Toolset = findToolset()
    ########
    ## Config IO
    ########
    INPUT_CLOUD = args.cloud
    if not os.path.isabs(INPUT_CLOUD):
        INPUT_CLOUD = os.path.abspath(INPUT_CLOUD)
    WORK_FOLDER = f'{basename(INPUT_CLOUD)}_{args.workfolder}'
    if not os.path.isabs(WORK_FOLDER):
        WORK_FOLDER = join(dirname(INPUT_CLOUD),WORK_FOLDER)
    BASE_CAM = args.basecam
    INPUT_MESH = args.mesh
    INPUT_TEXMESH = args.texmesh
    INPUT_RGBMESH = args.rgbmesh
    INPUT_CAM = args.cam
    if args.subdir == None:
        TOOLCHAIN = f'{args.vis}.{args.shader}_{args.rec}'
    else:
        TOOLCHAIN = args.subdir

    if BASE_CAM=='' or BASE_CAM == None:
        BASE_CAM = ''
    elif not os.path.isabs(BASE_CAM):
        if BASE_CAM.endswith('.json'):
            BASE_CAM = os.path.abs(BASE_CAM)
        else:
            BASE_CAM = join(WORK_FOLDER,basename(BASE_CAM)+'.json')
        if not exists(BASE_CAM):
            BASE_CAM = ''

    if INPUT_CAM=='' or INPUT_CAM == None:
        INPUT_CAM = next_camname_in_folder(WORK_FOLDER)
    elif not os.path.isabs(INPUT_CAM):
        if INPUT_CAM.endswith('.json'):
            INPUT_CAM = os.path.abs(INPUT_CAM)
        else:
            INPUT_CAM = join(WORK_FOLDER,basename(INPUT_CAM)+'.json')

    CAM_NAME = os.path.basename(INPUT_CAM).strip('.json')
    if args.vis == 'NET':
        SUBDIR_FOLDER = join(WORK_FOLDER,f'VDVNet_{CAM_NAME}')

    RENDER_FOLDER = join(SUBDIR_FOLDER,'render')
    VISCONF_FOLDER = join(SUBDIR_FOLDER,'conf')
    OUTPUT_FOLDER = join(SUBDIR_FOLDER,'out')

    pathlib.Path(WORK_FOLDER).mkdir(parents=True, exist_ok=True)
    pathlib.Path(SUBDIR_FOLDER).mkdir(parents=True, exist_ok=True)
    pathlib.Path(RENDER_FOLDER).mkdir(parents=True, exist_ok=True)
    pathlib.Path(VISCONF_FOLDER).mkdir(parents=True, exist_ok=True)
    pathlib.Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    # Clean Output and conf
    for c in glob.glob(join(OUTPUT_FOLDER,'*')):
        if os.path.isdir(c):
            shutil.rmtree(c)
        else:
            os.remove(c)
    for c in glob.glob(join(VISCONF_FOLDER,'*')):
        if os.path.isdir(c):
            shutil.rmtree(c)
        else:
            os.remove(c)

    for c in glob.glob(join(RENDER_FOLDER,'*')):
        if os.path.isdir(c):
            shutil.rmtree(c)
        else:
            os.remove(c)

    print('#Project Settings#')
    print(f'Input Cloud: {INPUT_CLOUD} Exists: {exists(INPUT_CLOUD)}')
    print(f'Input Mesh: {INPUT_MESH} Exists: {exists(INPUT_MESH)}')
    print(f'Input Textured Mesh: {INPUT_TEXMESH} Exists: {exists(INPUT_TEXMESH)}')
    print(f'Input Textured Mesh: {INPUT_RGBMESH} Exists: {exists(INPUT_RGBMESH)}')
    print(f'Base Cam: {BASE_CAM} Exists: {exists(BASE_CAM)}')
    print(f'Input Cam: {INPUT_CAM} Exists: {exists(INPUT_CAM)}')
    print(f'Work Folder: {WORK_FOLDER} Exists: {exists(WORK_FOLDER)}')
    print(f'Cam: {CAM_NAME}    ToolChain: {TOOLCHAIN}')
    print(f'ToolChain Work Folder: {SUBDIR_FOLDER} Exists:{exists(SUBDIR_FOLDER)}')
    print(f'Output Folder: {OUTPUT_FOLDER} Exists:{exists(OUTPUT_FOLDER)}')

    if not exists(INPUT_CLOUD):
        logging.critical('input cloud not found')
        sys.exit(1)

    return {'INPUT_CLOUD': INPUT_CLOUD,
            'INPUT_MESH': INPUT_MESH,
            'INPUT_TEXMESH': INPUT_TEXMESH,
            'INPUT_RGBMESH': INPUT_RGBMESH,
            'BASE_CAM': BASE_CAM,
            'INPUT_CAM': INPUT_CAM,
            'WORK_FOLDER': WORK_FOLDER,
            'CAM_NAME': CAM_NAME,
            'TOOLCHAIN': TOOLCHAIN,
            'SUBDIR_FOLDER': SUBDIR_FOLDER,
            'OUTPUT_FOLDER': OUTPUT_FOLDER,
            'RENDER_FOLDER': RENDER_FOLDER,
            'VISCONF_FOLDER': VISCONF_FOLDER,
            'Toolset': Toolset,
            'imagewidth':args.imagewidth,
            'imageheight':args.imageheight,
            'imagefocal':args.imagefocal,
            # 'camgenheight':args.camgenheight,
            # 'camgenoverlap':args.camgenoverlap,
            'render_shader':args.shader,
            'render_radius_k':args.radius_k,
            'vis':args.vis,
            'vis_arch':args.arch,
            'vis_checkpoint':args.checkpoint,
            'vis_gt_tolerance': args.gttolerance,
            'vis_conf_threshold':args.conf_threshold,
            'recon_param': args.recon_param}

def CreateCam(config):
    ## Ensure Cam.json available
    ImageHeight = config['imageheight']
    ImageWidth = config['imagewidth']
    ImageFocal = config['imagefocal']
    # CamGenHeight= config['camgenheight']
    # CamGenOverlap= config['camgenoverlap']

    camcommand = [config['Toolset']['o3d_vvcreator.py'],f'--output_list={config["INPUT_CAM"]}',f'--width={ImageWidth}',f'--height={ImageHeight}',config["INPUT_CLOUD"]]
    if config['BASE_CAM'] != '':
        camcommand.append(f'--il={config["BASE_CAM"]}')
    if not exists(config['INPUT_CAM']):
        pathlib.Path(dirname(config['INPUT_CAM'])).mkdir(parents=True,exist_ok=True)
        p = subprocess.Popen(camcommand,stdout=subprocess.PIPE)
        outs, errs = p.communicate()

    print('-- CamCreator Cmd:')
    print(' '.join(camcommand))
    if not exists(config['INPUT_CAM']):
        print("Operation canceled.")
        sys.exit(0)

    # Backup camera list
    shutil.copy(config['INPUT_CAM'], join(config['SUBDIR_FOLDER'],basename(config['INPUT_CAM'])))

    config['ImageHeight']=ImageHeight
    config['ImageWidth']=ImageWidth
    config['ImageFocal'] = ImageFocal
    return config

def CreateTaskJson(config):
    ## Create Task Json
    ## Render Segment
    RENDER_SEG_PATH = join(config['SUBDIR_FOLDER'],'render_seg.json')
    camJson = json.load(open(config['INPUT_CAM']))
    camNum = len(camJson['imgs'])
    logging.info(f'Valid Cams:{camNum}')
    render_block={'Worker':'GLRENDER',
                  'Param':{
                      'input_cloud': config['INPUT_CLOUD'],
                      'input_mesh': config['INPUT_MESH'],
                      'input_texmesh': config['INPUT_TEXMESH'],
                      'input_rgbmesh': config['INPUT_RGBMESH'],
                      'input_cam_source':'external',
                      'input_cam': config['INPUT_CAM'],
                      'shader': config['render_shader'],
                      'radius_k': config['render_radius_k'],
                      'render_cull': False,
                      'output_folder':config['RENDER_FOLDER'],
                      'output_clean':False
                  }}

    render_seg = [render_block]
    with open(RENDER_SEG_PATH,'w') as f:
        f.write(json.dumps(render_seg,indent=1))


    ## Vis Estimator
    HASTEXTURE = exists(config['INPUT_TEXMESH'])
    HASRGB = exists(config['INPUT_RGBMESH'])
    HASMESH = exists(config['INPUT_MESH'])
    VIS_SEG_PATH = join(config['SUBDIR_FOLDER'],'vis_seg.json')

    VISEstimator = config['vis']

    vis_seg = []
    if VISEstimator == 'NET':
        for i in range(camNum):
            input_rgb = join(config['RENDER_FOLDER'],f'pt{i}.png')
            input_depth = join(config['RENDER_FOLDER'],f'pt{i}.flt')
            input_instance = join(config['RENDER_FOLDER'],f'pt{i}.uint')
            input_cam = join(config['RENDER_FOLDER'],f'cam{i}.json')
            input_texmesh_rgb = join(config['RENDER_FOLDER'],f'mesh{i}.png') if HASTEXTURE or HASRGB else ''
            vis_block={'Worker':'NETWORKVISIBILITY',
                        'Param':{
                            'input_rgb': input_rgb,
                            'input_depth': input_depth,
                            'input_instance': input_instance,
                            'input_cam': input_cam,
                            'input_texmesh_rgb': input_texmesh_rgb,
                            'arch': config['vis_arch'],
                            'checkpoint': config['vis_checkpoint'],
                            'output_mask':join(config['VISCONF_FOLDER'],f'conf{i}.png')
                        }}
            vis_seg.append(vis_block)
    elif VISEstimator == 'GT':
        for i in range(camNum):
            input_rgb = join(config['RENDER_FOLDER'],f'pt{i}.png')
            input_depth = join(config['RENDER_FOLDER'],f'pt{i}.flt')
            input_instance = join(config['RENDER_FOLDER'],f'pt{i}.uint')
            input_cam = join(config['RENDER_FOLDER'],f'cam{i}.json')
            input_mesh = config['INPUT_TEXMESH'] if HASTEXTURE else None
            input_mesh = config['INPUT_RGBMESH'] if HASRGB else input_mesh
            input_mesh = config['INPUT_MESH'] if HASMESH else input_mesh
            input_texmesh_rgb = join(config['RENDER_FOLDER'],f'mesh{i}.png') if HASTEXTURE or HASRGB else ''
            assert exists(input_mesh), "Cannot find mesh"
            tolerance = config['vis_gt_tolerance']
            vis_block={'Worker':'RAYGTFILTER',
                       'Param':{
                           'input_rgb': input_rgb,
                           'input_depth': input_depth,
                           'input_instance': input_instance,
                           'input_cam': input_cam,
                           'input_texmesh_rgb': input_texmesh_rgb,
                           'input_cloud': config['INPUT_CLOUD'],
                           'input_mesh': input_mesh,
                           'tolerance': tolerance,
                           'output_mask':join(config['VISCONF_FOLDER'],f'conf{i}.png')
                       }}
            vis_seg.append(vis_block)

    with open(VIS_SEG_PATH,'w') as f:
        f.write(json.dumps(vis_seg,indent=1))

    ## Bundle Assembler
    BUNDLE_SEG_PATH = join(config['SUBDIR_FOLDER'],'bundle_seg.json')
    bundle_seg=[]

    bundle_input_rays = []
    for i in range(camNum):
        bundle_input_rays.append({
            'id': vis_seg[i]['Param']['input_instance'],
            'rgb': vis_seg[i]['Param']['input_texmesh_rgb'],
            'confidence': vis_seg[i]['Param']['output_mask'],
            'cam': vis_seg[i]['Param']['input_cam']
        })
    bundle_block={'Worker':'CLOUD_BUNDLE',
                  'Param':{
                      'input_cloud': config['INPUT_CLOUD'],
                      'input_raybundle': bundle_input_rays,
                      'conf_threshold' : config['vis_conf_threshold'],
                      'compress': False,
                      'output_bundle': join(config['SUBDIR_FOLDER'],'bundle.mva'),
                      'normalize_K': True
                  }}
    bundle_seg.append(bundle_block)

    with open(BUNDLE_SEG_PATH,'w') as f:
        f.write(json.dumps(bundle_seg,indent=1))

    config['RENDER_SEG_PATH']=RENDER_SEG_PATH
    config['render_seg'] = render_seg
    config['VIS_SEG_PATH'] = VIS_SEG_PATH
    config['vis_seg'] = vis_seg
    config['BUNDLE_SEG_PATH'] = BUNDLE_SEG_PATH
    config['bundle_seg']=bundle_seg

    return config

def RunProcess(config):
    ## Run vvtool commands
    p = subprocess.Popen([config['Toolset']['vvtool'],config['RENDER_SEG_PATH']],stdout=subprocess.PIPE)
    outs, errs = p.communicate()
    p = subprocess.Popen(['python','tools/lib/network_predict.py',config['VIS_SEG_PATH']],stdout=subprocess.PIPE)
    # p = subprocess.Popen([config['Toolset']['vvtool'],config['VIS_SEG_PATH']],stdout=subprocess.PIPE)
    outs, errs = p.communicate()
    p = subprocess.Popen([config['Toolset']['vvtool'],config['BUNDLE_SEG_PATH']],stdout=subprocess.PIPE)
    outs, errs = p.communicate()
    
    openmvsMeshRecon = config['Toolset']['ReconstructMesh']
    cmdArr = None
    if config['recon_param'] == '':
        cmdArr = [openmvsMeshRecon,'-w',config['SUBDIR_FOLDER'],'-i','bundle.mva']
    else:
        cmdArr = [openmvsMeshRecon,'-w',config['SUBDIR_FOLDER'],'-i','bundle.mva',*config['recon_param'].split(' ')]
    p = subprocess.Popen(cmdArr, stdout=subprocess.PIPE)
    outs, errs = p.communicate()

    result_file = join(config['SUBDIR_FOLDER'],'bundle_mesh.ply')

    if exists(result_file):
        print("*************** ")
        print("*** Success *** ")
        print(result_file)
        os.chmod(result_file, stat.S_IROTH|stat.S_IWOTH|stat.S_IXOTH)
        result_copy_file = config['INPUT_CLOUD'][:-4]+"_vis2mesh.ply"
        cnt=1
        while exists(result_copy_file):
            result_copy_file = config['INPUT_CLOUD'][:-4]+f"_vis2mesh{cnt}.ply"
            cnt += 1
        shutil.copy(result_file, result_copy_file)
    else:
        print("*** Generation Failed *** ")
    return config

def parseArgs():
    parser = argparse.ArgumentParser(description='process point recon')
    parser.add_argument('cloud', type=str)
    parser.add_argument('workfolder', nargs='?',default='WORK',type=str)
    parser.add_argument('subdir',nargs='?', default=None,type=str)
    parser.add_argument('--basecam','-b', type=str,help='[optional] input base camera list, create one if not exist')
    parser.add_argument('--cam','-c',type=str, help='[optional] input camera list, create one if not exist')
    parser.add_argument('--mesh',type=str, default='', help='[optional] mesh obj file')
    parser.add_argument('--texmesh',type=str, default='', help='[optional] textured mesh obj file')
    parser.add_argument('--rgbmesh',type=str, default='', help='[optional] rgb colored mesh obj file')
    parser.add_argument('--rec',default='DELAY',type=str,help='Recon {DELAY}')

    camgroup = parser.add_argument_group('camcreator')
    camgroup.add_argument('--imagewidth',default=512,type=int)
    camgroup.add_argument('--imageheight',default=512,type=int)
    camgroup.add_argument('--imagefocal',default=300,type=float)
    # camgroup.add_argument('--camgenheight',default=10,type=float)
    # camgroup.add_argument('--camgenoverlap',default=0.5,type=float)

    rendgroup = parser.add_argument_group("GLRender")
    rendgroup.add_argument('--shader',default='POINT',type=str,help='shader {POINT|RECTANGLE|DOT|DIAMOND}')
    rendgroup.add_argument('--radius_k',default=6,type=int,help='number of neighbor points to estimate radius')
    visgroup = parser.add_argument_group("VisEstimator")
    visgroup.add_argument('--vis',default='NET',type=str,help='VisEstimator {NET|GT}')
    visgroup.add_argument('--arch',default="CascadeNet(['PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=3)'])",type=str)
    visgroup.add_argument('--checkpoint',default='/workspace/checkpoints/VDVNet_CascadePPP_epoch30.pth',type=str)
    visgroup.add_argument('--gttolerance',default='0.05',type=float)
    visgroup.add_argument('--conf_threshold',default='0.5',type=float)

    recongroup = parser.add_argument_group("Reconstructor")
    recongroup.add_argument('--recon_param',default='-d 1',type=str)
    return parser.parse_args()


def main():
    config=Setup(parseArgs())
    config=CreateCam(config)
    config=CreateTaskJson(config)
    config=RunProcess(config)
    print('Process Done')


if __name__ == '__main__':
    main()
