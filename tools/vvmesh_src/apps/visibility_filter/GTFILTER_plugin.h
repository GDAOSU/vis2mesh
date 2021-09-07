#ifndef GTFILTER_PLUGIN_H
#define GTFILTER_PLUGIN_H

#include <memory>
#include <nlohmann/json.hpp>

#include "Plugin/plugininterface.h"

namespace FBORender { class MultidrawFBO; };

class GTFILTER_Plugin : public PluginInterface {
 public:
  GTFILTER_Plugin();
  virtual std::string getWorkerName();
  virtual bool operator()(const nlohmann::json& blockJson);

 private:
  const char* WORKER_NAME = "GTFILTER";

  bool exists_test(const std::string& name);
  bool processBlock(const nlohmann::json& blockJson);
};
#endif  // GTFILTER_PLUGIN_H
