// Copyright 2021 Shaun Song <sxsong1207@qq.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "NETWORKVISIBILITY_plugin.h"
#include "CONFLATION_DELAUNAY_GC_plugin.h"
#include "DELAUNAY_GC_plugin.h"

#include "Plugin/pluginmanager.h"
#include <nlohmann/json.hpp>
int main(int argc, char **argv) {

  nlohmann::json json = {
      {"Worker", "NETWORK_VISIBILITY"},
      {"Param", {
          {"input_rgb",
           "/home/sxs/GDA/ONR/ONR_DEMO_2021AU/Conflation/Dataset/Garage/AIRBORNE_LIDAR_scene_2015/AIRBORNE_LIDAR_scene_2015.ply_WORK/cam1_NET.POINT_DELAY_CascadeNet(['PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=3)'])/render/pt0.png"},
          {"input_depth",
           "/home/sxs/GDA/ONR/ONR_DEMO_2021AU/Conflation/Dataset/Garage/AIRBORNE_LIDAR_scene_2015/AIRBORNE_LIDAR_scene_2015.ply_WORK/cam1_NET.POINT_DELAY_CascadeNet(['PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=3)'])/render/pt0.flt"},
          {"input_instance",
           "/home/sxs/GDA/ONR/ONR_DEMO_2021AU/Conflation/Dataset/Garage/AIRBORNE_LIDAR_scene_2015/AIRBORNE_LIDAR_scene_2015.ply_WORK/cam1_NET.POINT_DELAY_CascadeNet(['PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=3)'])/render/pt0.uint"},
          {"input_cam",
           "/home/sxs/GDA/ONR/ONR_DEMO_2021AU/Conflation/Dataset/Garage/AIRBORNE_LIDAR_scene_2015/AIRBORNE_LIDAR_scene_2015.ply_WORK/cam1_NET.POINT_DELAY_CascadeNet(['PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=3)'])/render/cam0.json"},
          {"input_texmesh_rgb", ""},
          {"arch",
           "CascadeNet(['PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=3)'])"},
          {"checkpoint",
           "/home/sxs/GDA/iccv21_vis2mesh/visibility_learning_pytorch/checkpoints/VISDEPVIS_CascadePPP_epoch30.pth"},
          {"output_mask",
           "/home/sxs/GDA/ONR/ONR_DEMO_2021AU/Conflation/Dataset/Garage/AIRBORNE_LIDAR_scene_2015/AIRBORNE_LIDAR_scene_2015.ply_WORK/cam1_NET.POINT_DELAY_CascadeNet(['PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=2)', 'PartialConvUNet(input_channels=3)'])/conf/conf0.png"}
      }
      }
  };
  CONFLATION_DELAUNAY_GC_Plugin p1;
  DELAUNAY_GC_Plugin p2;
  NETWORKVISIBILITY_Plugin plugin;
  plugin(json);
  PluginManager pm;

  return 0;
}