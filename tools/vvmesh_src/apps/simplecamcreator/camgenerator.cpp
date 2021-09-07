// Copyright 2019 Shaun Song <sxsong1207@qq.com>
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
#include "camgenerator.h"
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree.h>

int CamPoseGenerator::GenerateNadirCameras(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                           const float hfov,
                                           const float vfov,
                                           const float height,
                                           const float overlaprate,
                                           std::vector<Eigen::Matrix3d>
                                           &Rs,
                                           std::vector<Eigen::Vector3d> &ts) {
  Eigen::Matrix3d NadirR;
  NadirR << 1, 0, 0,
      0, -1, 0,
      0, 0, -1;

  float GndWidth = 2 * std::tan(hfov / 2) * height;
  float GndHeight = 2 * std::tan(vfov / 2) * height;
  float GndWidthStep = (1. - overlaprate) * GndWidth;
  float GndHeightStep = (1. - overlaprate) * GndHeight;

  float GridSize = std::min(GndWidthStep, GndHeightStep) * 0.5;
  pcl::VoxelGrid<pcl::PointXYZ> grid;
  pcl::PointCloud<pcl::PointXYZ> sampledPts;
  grid.setInputCloud(cloud);
  grid.setLeafSize(GridSize, GridSize, 5);
  grid.filter(sampledPts);
  pcl::PointXYZ minpt, maxpt;
  pcl::getMinMax3D(sampledPts, minpt, maxpt);

  float MinX = minpt.x - GndWidth, MaxX = maxpt.x + GndWidth, LenX = MaxX - MinX;
  float MinY = minpt.y - GndWidth, MaxY = maxpt.y + GndWidth, LenY = MaxY - MinY;

  int CntX = std::ceil((LenX - GndWidth) / GndWidthStep + 1);
  int CntY = std::ceil((LenY - GndHeight) / GndHeightStep + 1);

  pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(0.1);
  octree.setInputCloud(sampledPts.makeShared());
  octree.addPointsFromInputCloud();
  for (int xi = 0; xi < CntX; ++xi)
    for (int yi = 0; yi < CntY; ++yi) {
      std::vector<int> indices;
      Eigen::Vector3f qminpt(MinX + xi * GndWidthStep, MinY + yi * GndHeightStep, minpt.z),
          qmaxpt(MinX + xi * GndWidthStep + GndWidth, MinY + yi * GndHeightStep + GndHeight, maxpt.z);
      if (0 == octree.boxSearch(qminpt, qmaxpt, indices)) continue;

      float maxz = -FLT_MAX;
      for (auto ind:indices) maxz = std::max(sampledPts[ind].z, maxz);

      Rs.push_back(NadirR);
      Eigen::Vector3d pos{0.5 * (qminpt[0] + qmaxpt[0]), 0.5 * (qminpt[1] + qmaxpt[1]), maxz + height};
      ts.push_back(pos);
    }

  return Rs.size();
}

int CamPoseGenerator::GenerateObliqueCameras(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                             const float hfov,
                                             const float vfov,
                                             const float height,
                                             const float overlaprate,
                                             std::vector<Eigen::Matrix3d>
                                             &Rs,
                                             std::vector<Eigen::Vector3d> &ts) {
  Eigen::Matrix3d NadirR, Ob1, Ob2, Ob3, Ob4;
  NadirR << 1, 0, 0,
      0, -1, 0,
      0, 0, -1;

  Ob1 << 0.70711, 0.00000, 0.70711,
      0.00000, -1.00000, 0.00000,
      0.70711, 0.00000, -0.70711;

  Ob2 << 0.00000, 0.70711, 0.70711,
      1.00000, 0.00000, 0.00000,
      0.00000, 0.70711, -0.70711;

  Ob3 << -0.70711, 0.00000, 0.70711,
      0.00000, 1.00000, 0.00000,
      -0.70711, 0.00000, -0.70711;

  Ob4 << 0.00000, -0.70711, 0.70711,
      -1.00000, 0.00000, 0.00000,
      0.00000, -0.70711, -0.70711;

  float GndWidth = 2 * std::tan(hfov / 2) * height;
  float GndHeight = 2 * std::tan(vfov / 2) * height;
  float GndWidthStep = (1. - overlaprate) * GndWidth;
  float GndHeightStep = (1. - overlaprate) * GndHeight;

  float GridSize = std::min(GndWidthStep, GndHeightStep) * 0.5;
  pcl::VoxelGrid<pcl::PointXYZ> grid;
  pcl::PointCloud<pcl::PointXYZ> sampledPts;
  grid.setInputCloud(cloud);
  grid.setLeafSize(GridSize, GridSize, 5);
  grid.filter(sampledPts);
  pcl::PointXYZ minpt, maxpt;
  pcl::getMinMax3D(sampledPts, minpt, maxpt);

  float MinX = minpt.x - GndWidth, MaxX = maxpt.x + GndWidth, LenX = MaxX - MinX;
  float MinY = minpt.y - GndWidth, MaxY = maxpt.y + GndWidth, LenY = MaxY - MinY;

  int CntX = std::ceil((LenX - GndWidth) / GndWidthStep + 1);
  int CntY = std::ceil((LenY - GndHeight) / GndHeightStep + 1);

  pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(0.1);
  octree.setInputCloud(sampledPts.makeShared());
  octree.addPointsFromInputCloud();
  for (int xi = 0; xi < CntX; ++xi)
    for (int yi = 0; yi < CntY; ++yi) {
      std::vector<int> indices;
      Eigen::Vector3f qminpt(MinX + xi * GndWidthStep + (0.5f - 1.5f) * GndWidth,
                             MinY + yi * GndHeightStep + (0.5f - 1.5f) * GndHeight,
                             minpt.z),
          qmaxpt(MinX + xi * GndWidthStep + (0.5f + 1.5f) * GndWidth,
                 MinY + yi * GndHeightStep + (0.5f + 1.5f) * GndHeight,
                 maxpt.z);
      if (0 == octree.boxSearch(qminpt, qmaxpt, indices)) continue;

      float maxz = -FLT_MAX;
      for (auto ind:indices) maxz = std::max(sampledPts[ind].z, maxz);

      Eigen::Vector3d pos{0.5 * (qminpt[0] + qmaxpt[0]), 0.5 * (qminpt[1] + qmaxpt[1]), maxz + height};

      Rs.push_back(NadirR);
      ts.push_back(pos);

      Rs.push_back(Ob1);
      ts.push_back(pos);

      Rs.push_back(Ob2);
      ts.push_back(pos);

      Rs.push_back(Ob3);
      ts.push_back(pos);

      Rs.push_back(Ob4);
      ts.push_back(pos);
    }

  return Rs.size();
}

int CamPoseGenerator::GenerateLiftedCameras(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                            const std::vector<Eigen::Matrix3d> seedRs,
                                            const std::vector<Eigen::Vector3d> seedts,
                                            float camSpacing,
                                            float camPitch,
                                            std::vector<Eigen::Matrix3d> &Rs,
                                            std::vector<Eigen::Vector3d> &ts) {
  pcl::PointCloud<pcl::PointXYZ> imgcenters;
  pcl::PointCloud<pcl::Normal> imgviews;
  std::vector<bool> imgmasks(seedRs.size(), true);
  Eigen::Vector3f zaxis{0, 0, 1};
  for (int i = 0; i < seedts.size(); ++i) {
    pcl::PointXYZ pt;
    pcl::Normal view;
    pt.x = seedts[i][0];
    pt.y = seedts[i][1];
    pt.z = seedts[i][2];
    view.getNormalVector3fMap() = seedRs[i].transpose().cast<float>() * zaxis;
    imgcenters.push_back(pt);
    imgviews.push_back(view);
  }

  /// Subsampling Poses
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(imgcenters.makeShared());

  std::vector<float> nndist;
  std::vector<int> nnidx;
  int gridfilterCnt = 0;
  for (int i = 0; i < imgcenters.size(); ++i) {
    if (imgmasks[i] == false) continue;
    kdtree.radiusSearch(i, camSpacing, nnidx, nndist);
    Eigen::Vector3f n1 = imgviews[i].getNormalVector3fMap();
    for (auto adj:nnidx) {
      if (adj == i) continue;
      Eigen::Vector3f n2 = imgviews[adj].getNormalVector3fMap();
      float anglen1n2deg = std::acos(n1.dot(n2));
      if (anglen1n2deg < 30) {
        if (imgmasks[adj]) gridfilterCnt++;
        imgmasks[adj] = false;
      }
    }
  }

  /// Creating new poses
  constexpr float EPSILON = 0.5;
  pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(EPSILON);
  octree.setInputCloud(cloud);
  octree.addPointsFromInputCloud();
  pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>::AlignedPointTVector voxel_center_list;
  std::vector<int> k_indices;
  for (int i = 0; i < imgcenters.size(); ++i) {
    if (imgmasks[i] == false) continue;
    octree.getIntersectedVoxelCenters(imgcenters[i].getVector3fMap(),
                                      imgviews[i].getNormalVector3fMap(),
                                      voxel_center_list,
                                      1);

    if (voxel_center_list.empty()) continue;
    std::cout << "Center :" << imgcenters[i].getVector3fMap().transpose() << std::endl;
    std::cout << "focus :" << voxel_center_list.front().getVector3fMap().transpose() << std::endl;

    Eigen::Vector3f minpt, maxpt;
    minpt = voxel_center_list.front().getVector3fMap();
    maxpt = voxel_center_list.front().getVector3fMap();
    minpt.x() -= EPSILON * 2;
    minpt.y() -= EPSILON * 2;
    minpt.z() = -FLT_MAX;
    maxpt.x() += EPSILON * 2;
    maxpt.y() += EPSILON * 2;
    maxpt.z() = FLT_MAX;
    octree.boxSearch(minpt, maxpt, k_indices);
    float height_top = voxel_center_list.front().z;
    for (int &k: k_indices) {
      height_top = std::max((*cloud)[k].z, height_top);
    }

    float heightAdj = -tan(DEG2RAD(camPitch)) *
        (imgcenters[i].getVector3fMap().topRows(2) - voxel_center_list.front().getVector3fMap().topRows(2)).norm();
    Eigen::Vector3d new_t = imgcenters[i].getVector3fMap().cast<double>();
    new_t.z() = height_top + heightAdj;
    std::cout << "new Center :" << new_t.transpose() << std::endl;

    Eigen::Vector3d focus = voxel_center_list.front().getVector3fMap().cast<double>();
    focus.z() = height_top;

    Eigen::Matrix3d
        new_R = Eigen::Quaterniond::FromTwoVectors(focus - new_t, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();

    Rs.push_back(new_R);
    ts.push_back(new_t);
  }
  return Rs.size();
}
