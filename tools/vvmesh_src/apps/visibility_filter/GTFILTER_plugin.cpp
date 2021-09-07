#include "GTFILTER_plugin.h"

#include <iostream>
#include <algorithm>
#include <numeric>

#include "FBORender/FBORender.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Dense>

#ifdef _USE_OPENMP
#include <omp.h>
#include <unordered_set>
#endif

#include <spdlog/spdlog.h>

GTFILTER_Plugin::GTFILTER_Plugin() {}

std::string GTFILTER_Plugin::getWorkerName() { return WORKER_NAME; }
bool GTFILTER_Plugin::operator()(const nlohmann::json &blockJson) {
  return processBlock(blockJson);
}

bool GTFILTER_Plugin::exists_test(const std::string &name) {
  std::ifstream f(name.c_str());
  return f.good();
}

namespace MyOperator {

class GTFilterObject {
 public:
  void filter(const unsigned char *inColor, const float *inDepth,
              const uint32_t *inID, unsigned char *outColor, float *outDepth,
              uint32_t *outID, const uint32_t width, const uint32_t height,
              const float invalid_depth, const uint32_t invalid_index,
              const float *inMeshDepth, const float *inMeshCullDepth,
              const float tolerance) {
    constexpr float EPSILON = std::numeric_limits<float>::epsilon();
    std::fill_n(outColor, 3 * width * height, 0);
    std::fill_n(outDepth, width * height, invalid_depth);
    std::fill_n(outID, width * height, invalid_index);
#pragma omp parallel for collapse(2)
    for (int ri = 0; ri < height; ++ri)
      for (int ci = 0; ci < width; ++ci) {
        int idx = ri * width + ci;
        bool isCull = inMeshCullDepth != nullptr;
        if (isCull)
          isCull =
              (std::fabs(inMeshDepth[idx] - inMeshCullDepth[idx]) > tolerance);
        bool isFG = (std::fabs(inDepth[idx] - inMeshDepth[idx]) < tolerance) &&
                    (std::fabs(inDepth[idx] - invalid_depth) > EPSILON) &&
                    !isCull;
        if (isFG) {
          std::copy_n(inColor + 3 * idx, 3, outColor + 3 * idx);
          outDepth[idx] = inDepth[idx];
          outID[idx] = inID[idx];
        }
      }
  }
};
}  // namespace MyOperator

bool GTFILTER_Plugin::processBlock(const nlohmann::json &blockJson) {
  // Verify block
  if (blockJson["Worker"].get<std::string>() != WORKER_NAME) {
    spdlog::warn("Worker[{0}] does not match {1}.\n", blockJson["Worker"].get<std::string>(), WORKER_NAME);
    return false;
  }
  nlohmann::json paramJson = blockJson["Param"];

  // Processing Unit
  std::string input_rgb = paramJson["input_rgb"].get<std::string>();
  std::string input_depth = paramJson["input_depth"].get<std::string>();
  std::string input_inst = paramJson["input_instance"].get<std::string>();
  std::string input_cam = paramJson["input_cam"].get<std::string>();
  std::string input_meshdep = paramJson["input_meshdepth"].get<std::string>();
  std::string input_meshculldep = paramJson["input_meshculldepth"].get<std::string>();
  bool enable_meshcull = false;
  if (!exists_test(input_rgb)) {
    fprintf(stderr, "[%s] File %s not exists.\n", WORKER_NAME, input_rgb.c_str());
    return false;
  }
  if (!exists_test(input_depth)) {
    fprintf(stderr, "[%s] File %s not exists.\n", WORKER_NAME, input_depth.c_str());
    return false;
  }
  if (!exists_test(input_inst)) {
    fprintf(stderr, "[%s] File %s not exists.\n", WORKER_NAME, input_inst.c_str());
    return false;
  }
  if (!exists_test(input_cam)) {
    fprintf(stderr, "[%s] File %s not exists.\n", WORKER_NAME, input_cam.c_str());
    return false;
  }
  if (!exists_test(input_meshdep)) {
    fprintf(stderr, "[%s] File %s not exists.\n", WORKER_NAME,
            input_meshdep.c_str());
    return false;
  }
  enable_meshcull = exists_test(input_meshculldep);

  bool usegl = paramJson.value<bool>("opengl", false);
  usegl = false;
  bool bgmask = paramJson.value<std::string>("mask_type", "bg") == "bg";
  float tolerance = paramJson.value<float>("tolerance", 0.5f);
  std::string output_rgb = paramJson.value<std::string>("output_rgb", "");
  std::string output_depth = paramJson.value<std::string>("output_depth", "");
  std::string output_inst = paramJson.value<std::string>("output_instance", "");
  std::string output_mask = paramJson.value<std::string>("output_mask", "");
  bool isOutRGB = !output_rgb.empty();
  bool isOutDepth = !output_depth.empty();
  bool isOutInstance = !output_inst.empty();
  bool isOutputMask = !output_mask.empty();
  //////////////////////

  cv::Mat cvrgb = cv::imread(input_rgb);
  int width = cvrgb.cols;
  int height = cvrgb.rows;
  std::vector<float> rawdepth(width * height), meshdepth(width * height),
      meshculldepth(enable_meshcull ? width * height : 0);
  std::vector<uint32_t> rawinst(width * height);
  std::ifstream ifs;
  int fileshape[2];

  ifs.open(input_depth, std::ifstream::binary);
  ifs.read((char *)fileshape, sizeof(int) * 2);
  assert(fileshape[0] == width);
  assert(fileshape[1] == height);
  ifs.read((char *)rawdepth.data(), sizeof(float) * rawdepth.size());
  ifs.close();
  ifs.open(input_inst, std::ifstream::binary);
  ifs.read((char *)fileshape, sizeof(int) * 2);
  assert(fileshape[0] == width);
  assert(fileshape[1] == height);
  ifs.read((char *)rawinst.data(), sizeof(uint32_t) * rawinst.size());
  ifs.close();
  ifs.open(input_meshdep, std::ifstream::binary);
  ifs.read((char *)fileshape, sizeof(int) * 2);
  assert(fileshape[0] == width);
  assert(fileshape[1] == height);
  ifs.read((char *)meshdepth.data(), sizeof(float) * meshdepth.size());
  ifs.close();
  if (enable_meshcull) {
    ifs.open(input_meshculldep, std::ifstream::binary);
    ifs.read((char *)fileshape, sizeof(int) * 2);
    assert(fileshape[0] == width);
    assert(fileshape[1] == height);
    ifs.read((char *)meshculldepth.data(),
             sizeof(float) * meshculldepth.size());
    ifs.close();
  }

  printf("Read Image (%d x %d) ", width, height);
  printf("Read Depth %zu\n", rawdepth.size());

  std::vector<uint8_t> rgbFil;
  std::vector<float> depFil;
  std::vector<uint32_t> instFil;
  std::vector<uint8_t> maskFil;

  if (usegl) {
    throw std::invalid_argument("HPR GL not implemented.");
  } else {
    rgbFil.resize(3 * width * height);
    depFil.resize(width * height);
    instFil.resize(width * height);

    MyOperator::GTFilterObject filter;
    filter.filter(cvrgb.data, rawdepth.data(), rawinst.data(),
                      rgbFil.data(), depFil.data(), instFil.data(), width,
                      height, -1.f, UINT32_MAX, meshdepth.data(),
                      enable_meshcull ? meshculldepth.data() : nullptr,
                      tolerance);

    if (isOutputMask) {
      maskFil.resize(width * height);
      if (bgmask) {
        for (int i = 0; i < width * height; ++i) {
          maskFil[i] = (rawinst[i] == instFil[i]) ? 0 : 255;
        }
      } else {
        for (int i = 0; i < width * height; ++i) {
          if (rawinst[i] == UINT32_MAX) {
            continue;
          }
          maskFil[i] = (rawinst[i] == instFil[i]) ? 255 : 0;
        }
      }
    }
  }

  ///////// OUTPUT SECTION
  if (isOutRGB) {
    cv::Mat cvout(height, width, CV_8UC3, rgbFil.data());
    cv::imwrite(output_rgb, cvout);
  }

  int imgsize[2] = {width, height};
  std::ofstream ofs;
  if (isOutDepth) {
    ofs.open(output_depth, std::ios::binary);
    ofs.write((char *)imgsize, sizeof(int) * 2);
    ofs.write((char *)depFil.data(), sizeof(float) * depFil.size());
    ofs.close();
  }

  if (isOutInstance) {
    ofs.open(output_inst, std::ifstream::binary);
    ofs.write((char *)imgsize, sizeof(int) * 2);
    ofs.write((char *)instFil.data(), sizeof(uint32_t) * instFil.size());
    ofs.close();
  }

  if (isOutputMask) {
    cv::Mat cvmask(height, width, CV_8UC1, maskFil.data());
    cv::imwrite(output_mask, cvmask);
  }

  return true;
}
