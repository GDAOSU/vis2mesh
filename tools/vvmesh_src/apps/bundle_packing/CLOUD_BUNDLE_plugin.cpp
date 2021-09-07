#include "CLOUD_BUNDLE_plugin.h"

#include <spdlog/spdlog.h>
#include <tinyply.h>
#include <pcl/io/auto_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/highgui.hpp>

std::string CLOUD_BUNDLE_Plugin::getWorkerName() { return WORKER_NAME; }
bool CLOUD_BUNDLE_Plugin::operator()(const nlohmann::json &blockJson) {
  return processBlock(blockJson);
}
bool CLOUD_BUNDLE_Plugin::exists_test(const std::string &name) {
  std::ifstream f(name.c_str());
  return f.good();
}
size_t CLOUD_BUNDLE_Plugin::read_ptsply_mva(const std::string &filepath,
                                            MVSA::Interface &obj) {
  MVSA::Interface::VertexArr &vertices = obj.vertices;
  MVSA::Interface::ColorArr &colors = obj.verticesColor;

#ifndef PCL_MAJOR_VERSION
  std::ifstream file_stream;
  file_stream.open(filepath, std::ios::binary);
  if (file_stream.fail()) {
    fprintf(stderr, "[%s]: open input file error.", WORKER_NAME);
    return 0;
  }
  tinyply::PlyFile file;
  file.parse_header(file_stream);

  std::shared_ptr<tinyply::PlyData> _vertices, _colors;
  try {
    _vertices = file.request_properties_from_element("vertex", {"x", "y", "z"});
  } catch (const std::exception &e) {
    fprintf(stderr, "tinyply exception: %s\n", e.what());
  }
  bool colorRead = false;
  try {
    _colors = file.request_properties_from_element("vertex",
                                                   {"red", "green", "blue"});
    colorRead = true;
  } catch (const std::exception &e) {
    fprintf(stderr, "tinyply exception: %s\n", e.what());
  }
  if (!colorRead) {
    try {
      _colors = file.request_properties_from_element("vertex", {"r", "g", "b"});
    } catch (const std::exception &e) {
      fprintf(stderr, "tinyply exception: %s\n", e.what());
    }
  }

  file.read(file_stream);
  // fprintf(stdout, "[%s] Read v (%zu) color (%zu)\n", WORKER_NAME,
  //         _vertices->count, _colors->_count);

  size_t _num = _vertices->count;
  vertices.resize(_num);
  if (colorRead)
    colors.resize(_num);
  else
    colors.clear();

  for (int i = 0; i < _num; ++i) {
    MVSA::Interface::Vertex &v = vertices[i];
    v.X.x = *((float *) _vertices->buffer.get() + 3 * i);
    v.X.y = *((float *) _vertices->buffer.get() + 3 * i + 1);
    v.X.z = *((float *) _vertices->buffer.get() + 3 * i + 2);
    if (colorRead) {
      MVSA::Interface::Color &c = colors[i];
      c.c.x = *(_colors->buffer.get() + 3 * i);
      c.c.y = *(_colors->buffer.get() + 3 * i + 1);
      c.c.z = *(_colors->buffer.get() + 3 * i + 2);
    }
  }
  return vertices.size();
#else
  pcl::PCLPointCloud2 data;
  pcl::io::load(filepath, data);
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  pcl::fromPCLPointCloud2(data, cloud);

  size_t _num = cloud.size();
  vertices.resize(_num);
  colors.resize(_num);
  for (int i = 0; i < _num; ++i) {
    MVSA::Interface::Vertex &v = vertices[i];
    v.X.x = cloud[i].x;
    v.X.y = cloud[i].y;
    v.X.z = cloud[i].z;
    MVSA::Interface::Color &c = colors[i];
    c.c.x = cloud[i].r;
    c.c.y = cloud[i].g;
    c.c.z = cloud[i].b;
  }

  return vertices.size();
#endif
}

bool CLOUD_BUNDLE_Plugin::read_RayBundles_json(
    const nlohmann::json &input_rays, std::vector<RayBundle> &raybundles) {
  for (auto &r : input_rays) {
    RayBundle _raybundle;
    _raybundle.id_img = r.value<std::string>("id", "");
    _raybundle.conf_img = r.value<std::string>("confidence", "");
    _raybundle.rgb_img = r.value<std::string>("rgb", "");
    if (!exists_test(_raybundle.id_img)) {
      fprintf(stderr, "[%s] %s not exists.\n", WORKER_NAME,
              _raybundle.id_img.c_str());
      return false;
    }
    if (!exists_test(_raybundle.conf_img)) {
      fprintf(stderr, "[%s] %s not exists.\n", WORKER_NAME,
              _raybundle.conf_img.c_str());
      return false;
    }
    if (!exists_test(_raybundle.rgb_img)) {
      _raybundle.rgb_img = "";
    }
    std::string camPath = r.value<std::string>("cam", "");
    std::ifstream ifs(camPath, std::ios::binary);
    if (ifs.fail()) {
      fprintf(stderr, "[%s] %s not exists.\n", WORKER_NAME, camPath.c_str());
      return false;
    }
    nlohmann::json _cam;
    ifs >> _cam;
    ifs.close();

    _raybundle.width = _cam["width"].get<int>();
    _raybundle.height = _cam["height"].get<int>();
    nlohmann::json matJson = _cam["K"];
    _raybundle.K(0, 0) = matJson[0][0].get<double>();
    _raybundle.K(0, 1) = matJson[0][1].get<double>();
    _raybundle.K(0, 2) = matJson[0][2].get<double>();
    _raybundle.K(1, 0) = matJson[1][0].get<double>();
    _raybundle.K(1, 1) = matJson[1][1].get<double>();
    _raybundle.K(1, 2) = matJson[1][2].get<double>();
    _raybundle.K(2, 0) = matJson[2][0].get<double>();
    _raybundle.K(2, 1) = matJson[2][1].get<double>();
    _raybundle.K(2, 2) = matJson[2][2].get<double>();
    matJson = _cam["R"];
    _raybundle.R(0, 0) = matJson[0][0].get<double>();
    _raybundle.R(0, 1) = matJson[0][1].get<double>();
    _raybundle.R(0, 2) = matJson[0][2].get<double>();
    _raybundle.R(1, 0) = matJson[1][0].get<double>();
    _raybundle.R(1, 1) = matJson[1][1].get<double>();
    _raybundle.R(1, 2) = matJson[1][2].get<double>();
    _raybundle.R(2, 0) = matJson[2][0].get<double>();
    _raybundle.R(2, 1) = matJson[2][1].get<double>();
    _raybundle.R(2, 2) = matJson[2][2].get<double>();
    matJson = _cam["C"];
    _raybundle.C[0] = matJson[0].get<double>();
    _raybundle.C[1] = matJson[1].get<double>();
    _raybundle.C[2] = matJson[2].get<double>();

    raybundles.push_back(_raybundle);
  }
  return true;
}

bool CLOUD_BUNDLE_Plugin::set_cameras_mva(
    const std::vector<RayBundle> &raybundles, MVSA::Interface &obj,
    bool normalize_K) {
  obj.platforms.resize(raybundles.size());
  obj.images.resize(raybundles.size());
  for (int i = 0; i < raybundles.size(); ++i) {
    const RayBundle &rb = raybundles[i];
    obj.platforms[i].poses.resize(1);
    obj.platforms[i].cameras.resize(1);
    MVSA::Interface::Platform::Pose &objpose = obj.platforms[i].poses.front();
    MVSA::Interface::Platform::Camera &objcam =
        obj.platforms[i].cameras.front();
    MVSA::Interface::Image &objimg = obj.images[i];
    // Create Pose
    objpose.R(0, 0) = rb.R(0, 0);
    objpose.R(0, 1) = rb.R(0, 1);
    objpose.R(0, 2) = rb.R(0, 2);
    objpose.R(1, 0) = rb.R(1, 0);
    objpose.R(1, 1) = rb.R(1, 1);
    objpose.R(1, 2) = rb.R(1, 2);
    objpose.R(2, 0) = rb.R(2, 0);
    objpose.R(2, 1) = rb.R(2, 1);
    objpose.R(2, 2) = rb.R(2, 2);
    objpose.C.x = rb.C[0];
    objpose.C.y = rb.C[1];
    objpose.C.z = rb.C[2];
    // Create Camera
    float scale = 1;
    if (normalize_K) {
      objcam.width = 0;
      objcam.height = 0;
      scale = objcam.GetNormalizationScale(rb.width, rb.height);
    } else {
      objcam.width = rb.width;
      objcam.height = rb.height;
      scale = 1;
    }
    objcam.K(0, 0) = rb.K(0, 0) / scale;
    objcam.K(0, 1) = rb.K(0, 1);
    objcam.K(0, 2) = rb.K(0, 2) / scale;
    objcam.K(1, 0) = rb.K(1, 0);
    objcam.K(1, 1) = rb.K(1, 1) / scale;
    objcam.K(1, 2) = rb.K(1, 2) / scale;
    objcam.K(2, 0) = rb.K(2, 0);
    objcam.K(2, 1) = rb.K(2, 1);
    objcam.K(2, 2) = rb.K(2, 2);
    objcam.R = MVSA::Interface::Mat33d(1, 0, 0, 0, 1, 0, 0, 0, 1);
    objcam.C = MVSA::Interface::Pos3d(0, 0, 0);

    // Create Image
    objimg.cameraID = 0;
    objimg.platformID = i;
    objimg.poseID = 0;
    objimg.width = rb.width;
    objimg.height = rb.height;
    objimg.ID = i;
    objimg.name = rb.rgb_img.empty() ? rb.conf_img : rb.rgb_img;
  }

  return true;
}

bool CLOUD_BUNDLE_Plugin::set_rays_mva(const std::vector<RayBundle> &raybundles,
                                       double conf_threshold,
                                       MVSA::Interface &obj) {
  std::ifstream ifs;
  for (int j = 0; j < raybundles.size(); ++j) {
    const auto &rb = raybundles[j];
    // read confidence map
    cv::Mat confimg = cv::imread(rb.conf_img, cv::IMREAD_GRAYSCALE);
    // read Idmap
    int width, height;
    std::vector<uint32_t> id_map;
    std::vector<float> conf_map;
    ifs.open(rb.id_img, std::ios::binary);
    ifs.read((char *) &height, sizeof(int));
    ifs.read((char *) &width, sizeof(int));
    if (width != confimg.cols || height != confimg.rows) {
      fprintf(stderr, "[%s] id_map(%d x %d) does not match conf_map(%d x %d)\n",
              WORKER_NAME, width, height, confimg.cols, confimg.rows);
      return false;
    }
    conf_map.resize(width * height);
    for (int ri = 0; ri < confimg.rows; ++ri)
      for (int ci = 0; ci < confimg.cols; ++ci) {
        conf_map[ri * width + ci] = float(confimg.at<uint8_t>(ri, ci)) / 255.f;
      }
    id_map.resize(width * height);
    ifs.read((char *) id_map.data(), sizeof(uint32_t) * width * height);
    ifs.close();
    // process confidence map and point id;
    std::unordered_map<uint32_t, std::pair<float, float>> id_set;
    for (int i = 0; i < width * height; ++i) {
      uint32_t pID = id_map[i];
      // 1<-(less possible to be correct)|(possible to be correct ray)->0
      float pConf = 1.f - conf_map[i];
      if (conf_threshold > 0 && pConf < conf_threshold) {
        continue;
      }
      if (pID == UINT32_MAX) {  // invalid ID
//        assert(pConf == 0.f);
        continue;
      }
      if (id_set.find(pID) == id_set.end()) {
        id_set[pID] = std::make_pair(0.f, 0.f);
      }
      std::pair<float, float> &pIDS = id_set[pID];
      pIDS.first += pConf;
      pIDS.second += 1.f;
    }

    for (auto &kv : id_set) {
      if (kv.first > obj.vertices.size()) {
        fprintf(stderr, "[%s] id map (%u) exceed points num %zu\n", WORKER_NAME,
                kv.first, obj.vertices.size());
        return false;
      }
      assert(kv.first < obj.vertices.size());
      MVSA::Interface::Vertex::View view;
      view.imageID = j;  // image id
      view.confidence = kv.second.first / kv.second.second;
      if (view.confidence <= 1e-5f) continue;  // Discard 0 confidence
      obj.vertices[kv.first].views.push_back(view);
    }
  }
  return true;
}
bool CLOUD_BUNDLE_Plugin::processBlock(const nlohmann::json &blockJson) {
  // Verify block
  if (blockJson["Worker"].get<std::string>() != WORKER_NAME) {
    spdlog::warn("Worker[{0}] does not match {1}.\n", blockJson["Worker"].get<std::string>(), WORKER_NAME);
    return false;
  }
  nlohmann::json paramJson = blockJson["Param"];

  // Processing Unit
  std::string input_cloud = paramJson.value<std::string>("input_cloud", "");
  std::string output_bundle = paramJson.value<std::string>("output_bundle", "");
  double conf_threshold = paramJson.value<double>("conf_threshold", 0.5);
  bool compress = paramJson.value<bool>("compress", true);
  if (!exists_test(input_cloud)) {
    fprintf(stderr, "[%s] %s not exists.\n", WORKER_NAME, input_cloud.c_str());
    return false;
  }

  nlohmann::json input_rays = paramJson.value<nlohmann::json>("input_raybundle", nlohmann::json::array());
  if (!input_rays.is_array()) {
    fprintf(stderr, "[%s] ray should be array.", WORKER_NAME);
    return false;
  }
  bool normalize_K = blockJson.value<bool>("normalize_K", true);

  MVSA::Interface obj;
  size_t num_pts = read_ptsply_mva(input_cloud, obj);

  std::vector<RayBundle> raybundles;
  if (!read_RayBundles_json(input_rays, raybundles)) return false;
  if (!set_cameras_mva(raybundles, obj, normalize_K)) return false;
  if (!set_rays_mva(raybundles, conf_threshold, obj)) return false;
  if (!MVSA::MVArchive::SerializeSave(obj, output_bundle, compress)) return false;

  return true;
}
