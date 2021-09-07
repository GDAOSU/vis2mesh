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
#include "simple_viewer.h"

#include <pcl/common/time.h>
#include <pcl/console/time.h>
#include <pcl/filters/random_sample.h>
#include <pcl/io/auto_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree.h>
#include <pcl/visualization/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <vtkRenderWindow.h>

#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <thread>

#include "param.h"
#include "camgenerator.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

SimpleCamViewer::SimpleCamViewer(const ParamsSet &param)
    : Params(param),
      viz("SimpleCamCreator"),
      cloud(new pcl::PointCloud<PointT>()),
      cloud_sampled(new pcl::PointCloud<PointT>()),
      texmesh(new pcl::TextureMesh()),
      polymesh(new pcl::PolygonMesh()),
      num_raw_points_(0),
      octree(Params.OctreeResolution) {
  pcl::visualization::Camera cam;
  viz.getCameraParameters(cam);
  cam.clip[0] = Params.nearClipping;
  cam.clip[1] = Params.farClipping;
  cam.fovy = Params.fovy;
  cam.window_size[0] = Params.WinWidth;
  cam.window_size[1] = Params.WinHeight;
  viz.setCameraParameters(cam);

  // try to load the cloud
  if (!loadCloud()) return;
  if (!loadMesh()) return;
  if (!Params.Input_list.empty()) loadCameraFromFile(Params.Input_list);

  // register keyboard callbacks
  viz.registerKeyboardCallback(&SimpleCamViewer::keyboardEventOccurred, *this,
                               nullptr);
  viz.registerPointPickingCallback(&SimpleCamViewer::pointPickingEventOccurred,
                                   *this, nullptr);
  viz.registerMouseCallback(&SimpleCamViewer::mouseEventOccurred, *this,
                            nullptr);

  if (!Params.Input_pts.empty()) {
    // Sampling
    pcl::RandomSample<PointT> randSample;
    randSample.setSample(std::min(Params.SampleToDisplay, (int)cloud->size()));
    randSample.setInputCloud(cloud);
    randSample.filter(*cloud_sampled);
    //  // assign point cloud to octree
    //  octree.setInputCloud(cloud);
    //  // update bounding box automatically
    //  octree.defineBoundingBox();
    //  // add points from cloud to octree
    //  octree.addPointsFromInputCloud();
    viz.addPointCloud(cloud_sampled, "cloud_sampled");
  }

  if (!Params.Input_texmesh.empty()) {
    viz.addTextureMesh(*texmesh, "textured");
  }

  if(!Params.Input_polymesh.empty()) {
    viz.addPolygonMesh(*polymesh, "polymesh");
  }

  update();
}
SimpleCamViewer::~SimpleCamViewer() {}

void SimpleCamViewer::cvtCamToExtrinsicParam(
    const pcl::visualization::Camera &cam, Eigen::Matrix3d &R,
    Eigen::Vector3d &t) {
  Eigen::Vector3d pos_vec, rx, ry, rz, focal_vec;

  pos_vec << cam.pos[0], cam.pos[1], cam.pos[2];
  focal_vec << cam.focal[0], cam.focal[1], cam.focal[2];
  rz = focal_vec - pos_vec;
  //!! Invert Y axis, becase camera coordinate system, Y axis is down-ward, but
  //! in OpenGL, its up-ward
  ry << -cam.view[0], -cam.view[1], -cam.view[2];
  rx = ry.cross(rz);

  //!! Transpose/Invert of R matrix, because here is R_W_C, but require R_C_W
  R.block<1, 3>(0, 0) = rx.normalized();
  R.block<1, 3>(1, 0) = ry.normalized();
  R.block<1, 3>(2, 0) = rz.normalized();
  t = pos_vec;
}
void SimpleCamViewer::createNewImage(const Eigen::Matrix3d &R,
                                     const Eigen::Vector3d &t) {
  CamPose cp;
  cp.R = R;
  cp.C = t;

  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  K(0, 0) = Params.focalLength;
  K(1, 1) = Params.focalLength;
  K(0, 2) = Params.ImageWidth / 2;
  K(1, 2) = Params.ImageHeight / 2;
  cp.width = Params.ImageWidth;
  cp.height = Params.ImageHeight;
  cp.K = K;
  cams_.push_back(cp);
}

void SimpleCamViewer::keyboardEventOccurred(
    const pcl::visualization::KeyboardEvent &event, void *) {
  if (event.getKeySym() == "s" && event.keyDown()) {
    pcl::console::TicToc tt;
    tt.tic();
    message_ = "[s] Write JSON to " + Params.Output_list;
    writeCameraToFile(Params.Output_list);
  } else if ((event.getKeySym() == "l") && event.keyDown()) {
    pcl::console::TicToc tt;
    tt.tic();
    message_ = "[l] Read JSON from " + Params.Input_list;
    loadCameraFromFile(Params.Input_list);
  } else if ((event.getKeyCode() == 'D') && event.keyDown()) {
    message_ = "[D] Delete All Images";
    PCL_INFO("[D] Delete All Images\n");
    cams_.clear();
  } else if ((event.getKeyCode() == 'd') && event.keyDown()) {
    message_ = "[d] Delete last Images";
    PCL_INFO("[d] Delete last Images\n");
    cams_.erase(cams_.end() - 1);
  } else if ((event.getKeyCode() == ' ') && event.keyDown()) {
    message_ = "[SPC] Create Pose";
    PCL_INFO("[SPC] Create Pose\n");
    pcl::visualization::Camera cam;
    viz.getCameraParameters(cam);
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    cvtCamToExtrinsicParam(cam, R, t);
    createNewImage(R, t);
  } else if ((event.getKeyCode() == 'b') && event.keyDown()) {
    message_ = "[b] Reset Camera Center";
    PCL_INFO("[b] Reset Camera Center\n");

    if (cams_.empty()) return;
    CamPose &cp = cams_.back();
    pcl::visualization::Camera cam;
    viz.getCameraParameters(cam);
    cam.clip[0] = Params.nearClipping;
    cam.clip[1] = Params.farClipping;
    cam.fovy = Params.fovy;
    cam.window_size[0] = Params.WinWidth;
    cam.window_size[1] = Params.WinHeight;

    Eigen::Vector3d pos_vec = cp.C;
    Eigen::Matrix3d rotation = cp.R.inverse();
    Eigen::Vector3d y_axis(0.f, -1.f, 0.f);
    Eigen::Vector3d up_vec(rotation * y_axis);

    Eigen::Vector3d z_axis(0.f, 0.f, 1.f);
    Eigen::Vector3d focal_vec = pos_vec + rotation * z_axis;

    std::copy(pos_vec.data(), pos_vec.data() + 3, cam.pos);
    std::copy(focal_vec.data(), focal_vec.data() + 3, cam.focal);
    std::copy(up_vec.data(), up_vec.data() + 3, cam.view);

    viz.setCameraParameters(cam);

  } else if ((event.getKeyCode() == 'r') && event.keyDown()) {
    pcl::visualization::Camera cam;
    viz.getCameraParameters(cam);
    cam.clip[0] = Params.nearClipping;
    cam.clip[1] = Params.farClipping;
    cam.fovy = Params.fovy;
    viz.setCameraParameters(cam);
  } else if ((event.getKeyCode() == 'n') && event.keyDown()) {
    message_ = "[n] Automaitc Nadir Camera Generator :";
    PCL_INFO("[n] Automaitc Nadir Camera Generator\n");

    std::vector<Eigen::Matrix3d> Rs;
    std::vector<Eigen::Vector3d> ts;

    pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    _cloud->reserve(this->cloud->size());
    for (auto &pt : (*this->cloud)) {
      _cloud->push_back(pcl::PointXYZ(pt.x, pt.y, pt.z));
    }

    float vfov = 2 * std::atan2(Params.ImageHeight * 0.5, Params.focalLength);
    float hfov = 2 * std::atan2(Params.ImageWidth * 0.5, Params.focalLength);

    int numCams = CamPoseGenerator::GenerateNadirCameras(
        _cloud, hfov, vfov, Params.CamGen_height, Params.CamGen_overlap, Rs,
        ts);
    for (int i = 0; i < numCams; ++i) {
      createNewImage(Rs[i], ts[i]);
    }
    message_ += std::to_string(numCams);
  } else if ((event.getKeyCode() == 'N') && event.keyDown()) {
    message_ = "[N] Automaitc Oblique Camera Generator :";
    PCL_INFO("[N] Automaitc Oblique Camera Generator\n");

    std::vector<Eigen::Matrix3d> Rs;
    std::vector<Eigen::Vector3d> ts;

    pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    _cloud->reserve(cloud->size());
    for (auto &pt : (*this->cloud)) {
      _cloud->push_back(pcl::PointXYZ(pt.x, pt.y, pt.z));
    }

    float vfov = 2 * std::atan2(Params.ImageHeight * 0.5, Params.focalLength);
    float hfov = 2 * std::atan2(Params.ImageWidth * 0.5, Params.focalLength);

    int numCams = CamPoseGenerator::GenerateObliqueCameras(
        _cloud, hfov, vfov, Params.CamGen_height, Params.CamGen_overlap, Rs,
        ts);
    for (int i = 0; i < numCams; ++i) {
      createNewImage(Rs[i], ts[i]);
    }
    message_ += std::to_string(numCams);
  } else if ((event.getKeySym() == "Up") && event.keyDown()) {
    pcl::visualization::Camera cam;
    viz.getCameraParameters(cam);

    Eigen::Vector3d pos_vec, focal_vec;

    pos_vec << cam.pos[0], cam.pos[1], cam.pos[2];
    focal_vec << cam.focal[0], cam.focal[1], cam.focal[2];

    Eigen::Vector3d forward = focal_vec - pos_vec;
    focal_vec += forward.normalized() * 2;
    pos_vec += forward.normalized() * 2;
    std::copy(focal_vec.data(), focal_vec.data() + 3, cam.focal);
    std::copy(pos_vec.data(), pos_vec.data() + 3, cam.pos);
    viz.setCameraParameters(cam);
  } else if ((event.getKeySym() == "Left") && event.keyDown()) {
    pcl::visualization::Camera cam;
    viz.getCameraParameters(cam);

    Eigen::Vector3d pos_vec, focal_vec, view_vec;

    pos_vec << cam.pos[0], cam.pos[1], cam.pos[2];
    focal_vec << cam.focal[0], cam.focal[1], cam.focal[2];
    view_vec << cam.view[0], cam.view[1], cam.view[2];

    Eigen::Vector3d forward = focal_vec - pos_vec;
    Eigen::Vector3d left_vec =
        view_vec.normalized().cross(forward.normalized());
    focal_vec += left_vec * 2;
    pos_vec += left_vec * 2;
    std::copy(focal_vec.data(), focal_vec.data() + 3, cam.focal);
    std::copy(pos_vec.data(), pos_vec.data() + 3, cam.pos);
    viz.setCameraParameters(cam);
  } else if ((event.getKeySym() == "Right") && event.keyDown()) {
    pcl::visualization::Camera cam;
    viz.getCameraParameters(cam);

    Eigen::Vector3d pos_vec, focal_vec, view_vec;

    pos_vec << cam.pos[0], cam.pos[1], cam.pos[2];
    focal_vec << cam.focal[0], cam.focal[1], cam.focal[2];
    view_vec << cam.view[0], cam.view[1], cam.view[2];

    Eigen::Vector3d forward = focal_vec - pos_vec;
    Eigen::Vector3d left_vec =
        view_vec.normalized().cross(forward.normalized());
    focal_vec -= left_vec * 2;
    pos_vec -= left_vec * 2;
    std::copy(focal_vec.data(), focal_vec.data() + 3, cam.focal);
    std::copy(pos_vec.data(), pos_vec.data() + 3, cam.pos);
    viz.setCameraParameters(cam);
  } else if ((event.getKeySym() == "Down") && event.keyDown()) {
    pcl::visualization::Camera cam;
    viz.getCameraParameters(cam);

    Eigen::Vector3d pos_vec, focal_vec;

    pos_vec << cam.pos[0], cam.pos[1], cam.pos[2];
    focal_vec << cam.focal[0], cam.focal[1], cam.focal[2];

    Eigen::Vector3d forward = focal_vec - pos_vec;
    focal_vec -= forward.normalized() * 2;
    pos_vec -= forward.normalized() * 2;
    std::copy(focal_vec.data(), focal_vec.data() + 3, cam.focal);
    std::copy(pos_vec.data(), pos_vec.data() + 3, cam.pos);
    viz.setCameraParameters(cam);
  } else if (event.keyDown()) {
    PCL_INFO("Unknown Key Down:%d  Syms:%s\n", (int)event.getKeyCode(),
             event.getKeySym().c_str());
  }

  update();
}
void SimpleCamViewer::pointPickingEventOccurred(
    const pcl::visualization::PointPickingEvent &event, void *viewer_void) {
  Eigen::Vector3f pt;
  if (event.getPointIndex() == -1) {
    return;
  }
  event.getPoint(pt[0], pt[1], pt[2]);

  int idx = -1;
}
void SimpleCamViewer::mouseEventOccurred(
    const pcl::visualization::MouseEvent &event, void *viewer_void) {
  // switch (event.getType()) {
  //   case pcl::visualization::MouseEvent::MouseScrollUp:
  //   case pcl::visualization::MouseEvent::MouseScrollDown:
  //     pcl::visualization::Camera cam;
  //     viz.getCameraParameters(cam);
  //     cam.clip[0] = Params.nearClipping;
  //     cam.clip[1] = Params.farClipping;
  //     cam.fovy = Params.fovy;
  //     viz.setCameraParameters(cam);
  //     break;
  // }
}
void SimpleCamViewer::run() {
  while (!viz.wasStopped()) {
    // main loop of the visualizer
    viz.spinOnce(100);
    cv::waitKey(5);
  }
  cloud->clear();
  cloud_sampled->clear();
}
void SimpleCamViewer::showManual() {
  viz.addText("[SPC] Create Pose", 0, Params.WinHeight - 20, 15, 1., 1., 0,
              "manual1");
  // viz.addText("[b] Render last image", 0, Params.WinHeight - 40, 15, 1., 1.,
  // 0,
  //             "manual2");
  viz.addText("[d] Delete last Images", 0, Params.WinHeight - 60, 15, 1., 1., 0,
              "manual3");
  viz.addText("[D] Delete All Images", 0, Params.WinHeight - 80, 15, 1., 1., 0,
              "manual4");
  viz.addText("[s] Write To Json", 0, Params.WinHeight - 100, 15, 1., 1., 0,
              "manual5");
  viz.addText("[l] Read From Json", 0, Params.WinHeight - 120, 15, 1., 1., 0,
              "manual7");
  viz.addText("[r] Reset Camera Center", 0, Params.WinHeight - 140, 15, 1., 1.,
              0, "manual6");
  // viz.addText("[P] Swtich Pattern", 0, Params.WinHeight - 140, 15, 1., 1., 0,
  //             "manual7");
  // viz.addText("[p] Switch picking", 0, Params.WinHeight - 160, 15, 1., 1., 0,
  //             "manual8");
  // viz.addText("[c] Print view info", 0, Params.WinHeight - 180, 15, 1., 1.,
  // 0,
  //             "manual9");
  viz.addText("[f] Fly to", 0, Params.WinHeight - 200, 15, 1., 1., 0,
              "manual10");

  viz.addText("[n] Generate Nadir Pose", 0, Params.WinHeight - 240, 15, 1., 1.,
              0, "manual11");
  viz.addText("[N] Generate Oblique Pose", 0, Params.WinHeight - 260, 15, 1.,
              1., 0, "manual12");
}
void SimpleCamViewer::showLegend() {
  char dataDisplay[256];

  sprintf(dataDisplay, "Clipping: [%f, %f]", Params.nearClipping,
          Params.farClipping);
  viz.addText(dataDisplay, 0, 20, 15, 0.0, 1.0, 0.0, "clipping");

  sprintf(dataDisplay, "Input JSON %s", Params.Input_list.c_str());
  viz.addText(dataDisplay, 0, 60, 15, 0.0, 1.0, 0.0, "inlist");

  sprintf(dataDisplay, "Output JSON %s", Params.Output_list.c_str());
  viz.addText(dataDisplay, 0, 40, 15, 0.0, 1.0, 0.0, "outlist");

  sprintf(dataDisplay, "Points: %zu", cloud->size());
  viz.addText(dataDisplay, 0, 80, 15, 0.0, 1.0, 0.0, "numPts");

  sprintf(dataDisplay, "Camera Poses : %zu", cams_.size());
  viz.addText(dataDisplay, 0, 100, 15, 0.0, 1.0, 0.0, "imgcounter");
}
void SimpleCamViewer::showMessage() {
  viz.addText(message_, 400, 20, 15, 1.0, 1.0, 1.0, "message");
}
void SimpleCamViewer::update() {
  // remove existing shapes from visualizer
  viz.removeAllShapes();

  showManual();
  showLegend();
  showMessage();

  // restore clipping
  pcl::visualization::Camera cam;
  viz.getCameraParameters(cam);
  cam.clip[0] = Params.nearClipping;
  cam.clip[1] = Params.farClipping;
  viz.setCameraParameters(cam);
}
