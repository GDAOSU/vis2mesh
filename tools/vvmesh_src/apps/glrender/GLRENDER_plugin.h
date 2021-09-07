#ifndef GLRENDER_PLUGIN_H
#define GLRENDER_PLUGIN_H

#include <memory>
#include <nlohmann/json.hpp>

#include "Plugin/plugininterface.h"

namespace FBORender { class MultidrawFBO; };

class GLRENDER_Plugin : public PluginInterface {
 public:
  GLRENDER_Plugin(std::shared_ptr<FBORender::MultidrawFBO> pFBO);
  virtual std::string getWorkerName();
  virtual bool operator()(const nlohmann::json &blockJson);
  virtual nlohmann::json getDefaultParameters();
 private:
  const char *WORKER_NAME = "GLRENDER";
  std::shared_ptr<FBORender::MultidrawFBO> fbo;

  bool ensureFolderExist(std::string path);
  bool processBlock(const nlohmann::json &blockJson);
};
#endif  // GLRENDER_PLUGIN_H
