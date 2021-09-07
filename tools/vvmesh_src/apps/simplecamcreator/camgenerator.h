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
#ifndef OPENMVS_CAMGENERATOR_H
#define OPENMVS_CAMGENERATOR_H
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>

namespace MVSA {
class Interface;
}

class CamPoseGenerator {
 public:
  static int GenerateNadirCameras(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                  const float hfov,
                                  const float vfov,
                                  const float height /*height of generated camera*/,
                                  const float overlaprate /*0~1*/,
                                  std::vector<Eigen::Matrix3d> &Rs,
                                  std::vector<Eigen::Vector3d> &ts);
  static int GenerateObliqueCameras(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                    const float hfov,
                                    const float vfov,
                                    const float height /*height of generated camera*/,
                                    const float overlaprate /*0~1*/,
                                    std::vector<Eigen::Matrix3d> &Rs,
                                    std::vector<Eigen::Vector3d> &ts);

  static int GenerateLiftedCameras(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud,
                                   const std::vector<Eigen::Matrix3d> seedRs,
                                   const std::vector<Eigen::Vector3d> seedts,
                                   float camSpacing,
                                   float camPitch,
                                   std::vector<Eigen::Matrix3d> &Rs,
                                   std::vector<Eigen::Vector3d> &ts);
};

#endif //OPENMVS_CAMGENERATOR_H
