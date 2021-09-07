// Copyright 2019 Shaun Song <sxsong1207@qq.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#ifndef OPENMVS_PARAM_H
#define OPENMVS_PARAM_H
#include <Eigen/Dense>
#include <string>

struct ParamsSet {
  std::string Input_pts;
  std::string Input_polymesh;
  std::string Input_texmesh;
  std::string Input_list;
  std::string Output_list;

  int ImageWidth;
  int ImageHeight;
  int WinWidth;
  int WinHeight;
  double focalLength;
  double nearClipping;
  double farClipping;
  double WHRatio;
  double ScaleRealDisplay;
  double fovy;
  float OctreeResolution;
  int SampleToDisplay;
  int RadiusK;
  float RadiusFactor;
  float RadiusConstant;

  Eigen::Matrix3d K;
  Eigen::Matrix3d InvK;
  int WWZB_KernelSize2;

  float CamGen_height;
  float CamGen_overlap;
  bool CamGen_autonadir;
  bool CamGen_autooblique;

  bool virtualize;
  bool dump;
  bool imageonly;

  void print() {
    printf("Params:\n==========\n");
    printf("Input_pts: %s\n", Input_pts.c_str());
    printf("==========\n");
    printf("ImageWidth: %d\n", ImageWidth);
    printf("ImageHeight: %d\n", ImageHeight);
    printf("WinWidth: %d\n", WinWidth);
    printf("WinHeight: %d\n", WinHeight);
    printf("==========\n");
    printf("focalLength: %f\n", focalLength);
    printf("nearClipping: %f\n", nearClipping);
    printf("farClipping: %f\n", farClipping);
    printf("WHRatio: %f\n", WHRatio);
    printf("ScaleRealDisplay: %f\n", ScaleRealDisplay);
    printf("fovy: %f\n", fovy * 180 / M_PI);
    printf("RadiusK: %d\n", RadiusK);
    printf("OctreeResolution: %f\n", OctreeResolution);
    printf("SampleToDisplay: %d\n", SampleToDisplay);
    printf("WWZB_KernelSize2: %d\n", WWZB_KernelSize2);
    printf("==========\n");
    printf("CamGen Height: %f\n", CamGen_height);
    printf("CamGen Overlap: %f\n", CamGen_overlap);
    printf("==========\n");
    printf("Virtualize: %s\n", virtualize ? "Yes" : "No");
    printf("Dump: %s\n", dump ? "Yes" : "No");
    printf("ImageOnly: %s\n", imageonly ? "Yes" : "No");
  }
};
#endif  // OPENMVS_PARAM_H
