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
#ifndef OPENMVS_MVSVIRTUALCAMERAVIEWER_H
#define OPENMVS_MVSVIRTUALCAMERAVIEWER_H

#include <pcl/octree/octree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "param.h"

struct CamPose {
  int width, height;
  Eigen::Matrix3d K, R;
  Eigen::Vector3d C;
};

class SimpleCamViewer {
  typedef pcl::PointXYZRGB PointT;
  typedef pcl::octree::OctreePointCloudPointVector<PointT> OctreeT;

 public:
  SimpleCamViewer(const ParamsSet &param);
  //! VIZ event loop
  void run();
  ~SimpleCamViewer();

 private:
  ParamsSet Params;
  // original cloud
  pcl::PointCloud<PointT>::Ptr cloud;
  // cloud which contains the voxel center
  pcl::PointCloud<PointT>::Ptr cloud_sampled;
  // polygon mesh
  pcl::PolygonMesh::Ptr polymesh;
  // texture mesh
  pcl::TextureMesh::Ptr texmesh;
  // octree
  OctreeT octree;
  // visualizer
  pcl::visualization::PCLVisualizer viz;

  int num_raw_points_;
  // MVS::Scene scene;
  // MVSA::Interface scene;
  // MVS::CameraIntern camintern;
  // std::vector<float> cloud_radius_;
  // bool radius_done_, addition_vc_done_;
  std::string message_;
  // // enum PICK_OP { NOTHING, JUMP, DISP, JUMP_AND_DISP, NUM_PICK_OP };
  // std::vector<std::string> StrPICKOP{"Noting", "Jump", "Disp",
  // "JumpAndDisp"}; int pick_op_;

  // const std::vector<std::string> PATTERN{"dot", "box", "diamond", "ellipse"};
  // int pattern_;

  // std::vector<MVS::Platform::Pose> virtualizedPoses_;
  std::vector<CamPose> cams_;

 private:
  //========================================================
  //! Render point cloud to camera of image
  // void renderImage(const MVS::Image &img, float *pDepth, uint32_t *pIndex,
  //                  uchar *pRGB, bool wwzb = true);

  // void visibilityCheckingForAppending(size_t preserve_views_ptid);
  //! Compute point cloud radius of neighbor structure
  // void computeRadius();
  //! Inject PCL pointcloud to MVS scene
  // void injectPointCloudtoMVS();
  // void fetchFixedEntities();
  //! Read external PLY point cloud and convert to MVS pointcloud
  bool loadCloud();
  bool loadMesh();
  //! Render Images and write to file
  // void lrenderImagesToFilel(const MVS::Image &img);

  // void writeMVS(std::string &filename);

  void writeCameraToFile(std::string &filename);
  void loadCameraFromFile(std::string &filename);
  //========================================================
  //! convert OpenGL MVP mat to extrinsic
  void cvtCamToExtrinsicParam(const pcl::visualization::Camera &cam,
                              Eigen::Matrix3d &R, Eigen::Vector3d &t);
  //! Create new image with visualizer's camera setting
  void createNewImage(const Eigen::Matrix3d &R, const Eigen::Vector3d &t);
  //! Set camera of viewer to be same as image
  // void vizJumpToImage(const MVS::Image &img);
  // //! Generate depth index rgb image with camera and display
  // void vizDisplayImage(const MVS::Image &img);
  // void vizCvShowImage(const int width, const int height, float *pDepth,
  //                     uint32_t *pIndex, uchar *pRGB);
  //========================================================
  //! Keyboard key event callback
  void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
                             void *);
  //! Point picking event callback
  void pointPickingEventOccurred(
      const pcl::visualization::PointPickingEvent &event, void *viewer_void);
  //! Mouse event callback
  void mouseEventOccurred(const pcl::visualization::MouseEvent &event,
                          void *viewer_void);
  //! Draw manual text on viz
  void showManual();
  //! Draw text on viz
  void showLegend();
  //! Draw message on screen
  void showMessage();
  //! Update PCL Visualizer with MVS Scene
  void update();
};

#endif  // OPENMVS_MVSVIRTUALCAMERAVIEWER_H
