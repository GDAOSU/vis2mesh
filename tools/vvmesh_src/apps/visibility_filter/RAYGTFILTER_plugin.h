#ifndef RAYGTFILTER_PLUGIN_H
#define RAYGTFILTER_PLUGIN_H

#include <memory>
#include <nlohmann/json.hpp>

#include "Plugin/plugininterface.h"

struct RAYGTFILTER_Plugin_Cache;

class RAYGTFILTER_Plugin : public PluginInterface {
 public:
  RAYGTFILTER_Plugin();
  ~RAYGTFILTER_Plugin();
  virtual std::string getWorkerName();
  virtual bool operator()(const nlohmann::json &blockJson);

 private:
  const char *WORKER_NAME = "RAYGTFILTER";
  std::unique_ptr<RAYGTFILTER_Plugin_Cache> cache;
  bool exists_test(const std::string &name);
  bool processBlock(const nlohmann::json &blockJson);
};
#endif  // RAYGTFILTER_PLUGIN_H
