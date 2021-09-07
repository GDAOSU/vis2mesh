#ifndef CLOUD_BUNDLE_PLUGIN_H
#define CLOUD_BUNDLE_PLUGIN_H

#include "Plugin/plugininterface.h"

#include <nlohmann/json.hpp>

#include <iostream>

#include <Eigen/Core>

#include <MVArchive/ARInterface.h>

class DLL_API CLOUD_BUNDLE_Plugin : public PluginInterface {
 public:
  virtual std::string getWorkerName();
  virtual bool operator()(const nlohmann::json& blockJson);

 private:
  const char* WORKER_NAME = "CLOUD_BUNDLE";

  bool processBlock(const nlohmann::json& blockJson);

 private:
  bool exists_test(const std::string& name);

  struct RayBundle {
    std::string id_img;
    std::string conf_img;
    std::string rgb_img;
    Eigen::Matrix3f K;
    Eigen::Matrix3f R;
    Eigen::Vector3f C;
    int width;
    int height;
  };

  size_t read_ptsply_mva(const std::string& filepath, MVSA::Interface& obj);

  bool read_RayBundles_json(const nlohmann::json& input_rays,
                            std::vector<RayBundle>& raybundles);

  bool set_cameras_mva(const std::vector<RayBundle>& raybundles,
                       MVSA::Interface& obj, bool normalize_K);

  bool set_rays_mva(const std::vector<RayBundle>& raybundles,
                    double conf_threshold,
                    MVSA::Interface& obj);
};
#endif  // CLOUD_BUNDLE_PLUGIN_H
