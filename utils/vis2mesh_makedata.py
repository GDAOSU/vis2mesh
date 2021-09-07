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

# logname = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.log'
logging.basicConfig(level = logging.INFO)
############################################## Utils
def search_file(filename):
    cur_dir = dirname(__file__)
    work_dir = abspath(os.getcwd())
    search_paths =[abspath(join(cur_dir,'..')),
                  abspath(join(work_dir,'..'))]

    for search_path in search_paths:
        res = glob.glob(join(search_path,f'**release/bin/{filename}'))
        if len(res)>0:
            return res[0]
    return None

def next_camname_in_folder(folder):
    cams = glob.glob(join(folder,'cam*.json'))
    maxid=-1
    for c in cams:
        res=re.search(r'cam(.*)\.json',c)
        if len(res.groups())>0:
            maxid=max(maxid,int(res.group(1)))
    return join(folder,f'cam{maxid+1}.json')

def findToolset(dep_tools = ['vvtool','simplecamcreator']):
    Toolset=dict()
    print('#Available Toolset#')
    for t in dep_tools:
        Toolset[t] = search_file(t)
        assert(Toolset[t]!=None)
        print(f'{t}: {Toolset[t]}')
    Toolset['netvis'] = join(dirname(__file__),'../apps/dl_visibility/','dl_vis_estimator.py')
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
    INPUT_MESH = args.mesh
    if not os.path.isabs(INPUT_MESH):
        INPUT_MESH = os.path.abspath(INPUT_MESH)
    WORK_FOLDER = dirname(INPUT_CLOUD)
    INPUT_CAM = args.cam
    if INPUT_CAM=='' or INPUT_CAM == None:
        INPUT_CAM = next_camname_in_folder(WORK_FOLDER)
    elif not os.path.isabs(INPUT_CAM):
        INPUT_CAM = join(dirname(INPUT_CLOUD),basename(INPUT_CAM)+'.json')

    CAM_NAME = os.path.basename(INPUT_CAM).strip('.json')
    RENDER_FOLDER = join(WORK_FOLDER,'render')
    VISCONF_FOLDER = join(WORK_FOLDER,'conf')

    pathlib.Path(RENDER_FOLDER).mkdir(parents=True, exist_ok=True)
    pathlib.Path(VISCONF_FOLDER).mkdir(parents=True, exist_ok=True)
    # Clean Output and conf
    for c in glob.glob(join(RENDER_FOLDER,'*')):
        if os.path.isdir(c):
            shutil.rmtree(c)
        else:
            os.remove(c)
    for c in glob.glob(join(VISCONF_FOLDER,'*')):
        if os.path.isdir(c):
            shutil.rmtree(c)
        else:
            os.remove(c)

    print('#Project Settings#')
    print(f'Input Cloud: {INPUT_CLOUD} Exists: {exists(INPUT_CLOUD)}')
    print(f'Input Mesh: {INPUT_MESH} Exists: {exists(INPUT_MESH)}')
    print(f'Input Cam: {INPUT_CAM} Exists: {exists(INPUT_CAM)}')
    print(f'Work Folder: {WORK_FOLDER} Exists: {exists(WORK_FOLDER)}')
    print(f'Render Folder: {RENDER_FOLDER} Exists:{exists(RENDER_FOLDER)}')

    if not exists(INPUT_CLOUD):
        logging.critical('input cloud not found')
        sys.exit(1)

    return {'INPUT_CLOUD':INPUT_CLOUD,
            'INPUT_MESH': INPUT_MESH,
            'INPUT_CAM': INPUT_CAM,
            'WORK_FOLDER': WORK_FOLDER,
            'CAM_NAME': CAM_NAME,
            'RENDER_FOLDER': RENDER_FOLDER,
            'VISCONF_FOLDER': VISCONF_FOLDER,
            'Toolset': Toolset,
            'imagewidth':args.imagewidth,
            'imageheight':args.imageheight,
            'imagefocal':args.imagefocal,
            'camgenheight':args.camgenheight,
            'camgenoverlap':args.camgenoverlap,
            'render_shader':args.shader,
            'render_radius_k':args.radius_k,
            'vis_gt_tolerance': args.gttolerance
            }

def CreateCam(config):
    ## Ensure Cam.json available
    ImageHeight = config['imageheight']
    ImageWidth = config['imagewidth']
    ImageFocal = config['imagefocal']
    CamGenHeight= config['camgenheight']
    CamGenOverlap= config['camgenoverlap']

    camcommand = [config['Toolset']['simplecamcreator'],f'--ip={config["INPUT_CLOUD"]}',f'--ol={config["INPUT_CAM"]}',f'--width={ImageWidth}',f'--height={ImageHeight}',f'--focal={ImageFocal}',f'--camgenheight={CamGenHeight}',f'--camgenoverlap={CamGenOverlap}']
    if not exists(config['INPUT_CAM']):
        pathlib.Path(dirname(config['INPUT_CAM'])).mkdir(parents=True,exist_ok=True)
        p = subprocess.Popen(camcommand,stdout=subprocess.PIPE)
        outs, errs = p.communicate()

    print('-- CamCreator Cmd:')
    print(' '.join(camcommand))
    if not exists(config['INPUT_CAM']):
        print("Operation canceled.")
        sys.exit(0)

    config['ImageHeight']=ImageHeight
    config['ImageWidth']=ImageWidth
    config['ImageFocal'] = ImageFocal
    return config

def CreateTaskJson(config):
    ## Create Task Json
    ## Render Segment
    RENDER_SEG_PATH = join(config['WORK_FOLDER'],'render_seg.json')
    camJson = json.load(open(config['INPUT_CAM']))
    camNum = len(camJson['imgs'])
    logging.info(f'Valid Cams:{camNum}')
    render_block={'Worker':'GLRENDER',
                  'Param':{
                      'input_cloud': config['INPUT_CLOUD'],
                      'input_mesh':config['INPUT_MESH'],
                      'input_cam_source':'external',
                      'input_cam': config['INPUT_CAM'],
                      'shader': config['render_shader'],
                      'radius_k': config['render_radius_k'],
                      'render_cull': True,
                      'output_folder':config['RENDER_FOLDER'],
                      'output_clean':False
                  }}

    render_seg = [render_block]
    with open(RENDER_SEG_PATH,'w') as f:
        f.write(json.dumps(render_seg,indent=1))

    config['RENDER_SEG_PATH']=RENDER_SEG_PATH
    config['render_seg'] = render_seg

    vis_seg = []
    for i in range(camNum):
        input_rgb = join(config['RENDER_FOLDER'],f'pt{i}.png')
        input_depth = join(config['RENDER_FOLDER'],f'pt{i}.flt')
        input_instance = join(config['RENDER_FOLDER'],f'pt{i}.uint')
        input_cam = join(config['RENDER_FOLDER'],f'cam{i}.json')
        input_mesh = config['INPUT_MESH']
        assert exists(input_mesh), "Cannot find mesh"
        tolerance = config['vis_gt_tolerance']
        vis_block={'Worker':'RAYGTFILTER',
                   'Param':{
                       'input_rgb': input_rgb,
                       'input_depth': input_depth,
                       'input_instance': input_instance,
                       'input_cam': input_cam,
                       'input_texmesh_rgb': "",
                       'input_cloud': config['INPUT_CLOUD'],
                       'input_mesh': input_mesh,
                       'tolerance': tolerance,
                       'output_mask':join(config['VISCONF_FOLDER'],f'conf{i}.png')
                   }}
        vis_seg.append(vis_block)
    VIS_SEG_PATH = join(config['WORK_FOLDER'],'vis_seg.json')
    with open(VIS_SEG_PATH,'w') as f:
        f.write(json.dumps(vis_seg,indent=1))

    config['VIS_SEG_PATH']=VIS_SEG_PATH
    config['vis_seg'] = vis_seg
    return config

def RunProcess(config):
    ## Run vvtool commands
    p = subprocess.Popen([config['Toolset']['vvtool'],config['RENDER_SEG_PATH']],stdout=subprocess.PIPE)
    outs, errs = p.communicate()
    p = subprocess.Popen([config['Toolset']['vvtool'],config['VIS_SEG_PATH']],stdout=subprocess.PIPE)
    outs, errs = p.communicate()

    return config

def parseArgs():
    parser = argparse.ArgumentParser(description='process point recon')
    parser.add_argument('cloud',type=str)
    parser.add_argument('mesh',type=str)
    parser.add_argument('workfolder',nargs='?',default='render',type=str)
    parser.add_argument('--cam','-c',type=str,help='[optional] input camera list, create one if not exist')

    camgroup = parser.add_argument_group('camcreator')
    camgroup.add_argument('--imagewidth',default=512,type=int)
    camgroup.add_argument('--imageheight',default=512,type=int)
    camgroup.add_argument('--imagefocal',default=300,type=float)
    camgroup.add_argument('--camgenheight',default=10,type=float)
    camgroup.add_argument('--camgenoverlap',default=0.5,type=float)

    rendgroup = parser.add_argument_group("GLRender")
    rendgroup.add_argument('--shader',default='POINT',type=str,help='shader {POINT|RECTANGLE|DOT|DIAMOND}')
    rendgroup.add_argument('--radius_k',default=6,type=int,help='number of neighbor points to estimate radius')


    visgroup = parser.add_argument_group("VisEstimator")
    visgroup.add_argument('--gttolerance',default='0.05',type=float)

    return parser.parse_args()


def main():
    config=Setup(parseArgs())
    config=CreateCam(config)
    config=CreateTaskJson(config)
    config=RunProcess(config)
    print('Process Done')


if __name__ == '__main__':
    main()
