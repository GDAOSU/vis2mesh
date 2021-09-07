#include "simple_viewer.h"
#include "param.h"

#include <OBJFile/obj.h>

#include "filesystem/ghc/filesystem.hpp"
namespace fs = ghc::filesystem;

#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <pcl/common/centroid.h>
#include <pcl/common/time.h>
#include <pcl/console/time.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/random_sample.h>
#include <pcl/io/auto_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree.h>
#include <pcl/search/flann_search.h>

#include <nlohmann/json.hpp>

bool SimpleCamViewer::loadCloud() {
  if (!Params.Input_pts.empty()) {
    // read cloud
    if (pcl::io::load(Params.Input_pts, *cloud)) {
      return false;
    }
    // remove NaN Points
    std::vector<int> nanIndexes;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, nanIndexes);

    PCL_INFO("Loading PLY %s, %d pts.\n", Params.Input_pts.c_str(),
             cloud->size());
  }
  return true;
}

bool SimpleCamViewer::loadMesh() {
  if (!Params.Input_texmesh.empty()) {
    pcl::io::loadOBJFile(Params.Input_texmesh, *texmesh);
  }
  if (!Params.Input_polymesh.empty()) {
    pcl::io::loadOBJFile(Params.Input_polymesh, *polymesh);
  }
  return true;
}

void SimpleCamViewer::writeCameraToFile(std::string &filename) {
  nlohmann::json rootJson;
  nlohmann::json imgsJson;
  for (auto &img: cams_) {
    Eigen::Matrix3d K = img.K;
    Eigen::Matrix3d R = img.R;
    Eigen::Vector3d C = img.C;
    int width = img.width;
    int height = img.height;
    nlohmann::json imgJson;
    imgJson["width"] = width;
    imgJson["height"] = height;
    nlohmann::json Kvec, Rvec, Cvec;
    nlohmann::json row;

    row.clear();
    row.push_back(K(0, 0));
    row.push_back(K(0, 1));
    row.push_back(K(0, 2));
    Kvec.push_back(row);
    row.clear();
    row.push_back(K(1, 0));
    row.push_back(K(1, 1));
    row.push_back(K(1, 2));
    Kvec.push_back(row);
    row.clear();
    row.push_back(K(2, 0));
    row.push_back(K(2, 1));
    row.push_back(K(2, 2));
    Kvec.push_back(row);

    row.clear();
    row.push_back(R(0, 0));
    row.push_back(R(0, 1));
    row.push_back(R(0, 2));
    Rvec.push_back(row);
    row.clear();
    row.push_back(R(1, 0));
    row.push_back(R(1, 1));
    row.push_back(R(1, 2));
    Rvec.push_back(row);
    row.clear();
    row.push_back(R(2, 0));
    row.push_back(R(2, 1));
    row.push_back(R(2, 2));
    Rvec.push_back(row);

    Cvec.push_back(C[0]);
    Cvec.push_back(C[1]);
    Cvec.push_back(C[2]);

    imgJson["K"] = Kvec;
    imgJson["R"] = Rvec;
    imgJson["C"] = Cvec;
    // imgJson["Platform"] = scene.platforms[img.platformID].name;
    // imgJson["Image"] = img.name;
    imgsJson.push_back(imgJson);
  }
  rootJson["imgs"] = imgsJson;

  std::ofstream ofs;
  ofs.open(filename);
  if (ofs.is_open()) {
    ofs << std::setw(4) << rootJson << std::endl;
  } else {
    std::cerr << "Error: Cannot open output file " << filename << std::endl;
  }
  ofs.close();
}

void SimpleCamViewer::loadCameraFromFile(std::string &input) {
  if (input.empty()) return;
  std::ifstream ifs(input, std::ifstream::binary);
  if (ifs.good()) {
    nlohmann::json rootJson;
    ifs >> rootJson;
    ifs.close();

    for (auto it = rootJson["imgs"].begin(), end = rootJson["imgs"].end();
         it != end; ++it) {
      CamPose cam;

      cam.width = (*it)["width"].get<int>();
      cam.height = (*it)["height"].get<int>();

      nlohmann::json matJson = (*it)["K"];
      cam.K(0, 0) = matJson[0][0].get<double>();
      cam.K(0, 1) = matJson[0][1].get<double>();
      cam.K(0, 2) = matJson[0][2].get<double>();
      cam.K(1, 0) = matJson[1][0].get<double>();
      cam.K(1, 1) = matJson[1][1].get<double>();
      cam.K(1, 2) = matJson[1][2].get<double>();
      cam.K(2, 0) = matJson[2][0].get<double>();
      cam.K(2, 1) = matJson[2][1].get<double>();
      cam.K(2, 2) = matJson[2][2].get<double>();
      matJson = (*it)["R"];
      cam.R(0, 0) = matJson[0][0].get<double>();
      cam.R(0, 1) = matJson[0][1].get<double>();
      cam.R(0, 2) = matJson[0][2].get<double>();
      cam.R(1, 0) = matJson[1][0].get<double>();
      cam.R(1, 1) = matJson[1][1].get<double>();
      cam.R(1, 2) = matJson[1][2].get<double>();
      cam.R(2, 0) = matJson[2][0].get<double>();
      cam.R(2, 1) = matJson[2][1].get<double>();
      cam.R(2, 2) = matJson[2][2].get<double>();
      matJson = (*it)["C"];
      cam.C[0] = matJson[0].get<double>();
      cam.C[1] = matJson[1].get<double>();
      cam.C[2] = matJson[2].get<double>();

      cams_.push_back(cam);
    }
    printf("Read In %d cams.", rootJson["img"].size());
  }
}
