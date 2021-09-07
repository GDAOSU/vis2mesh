#ifndef PLUGININTERFACE_H
#define PLUGININTERFACE_H
#include "dllmacro.h"
#include <nlohmann/json.hpp>
#include <memory>
#include <unordered_map>

class PluginManager;

class DLL_API PluginInterface {
 public:
  PluginInterface();
  virtual std::string getWorkerName() = 0;
  virtual bool operator()(const nlohmann::json &blockJson) = 0;
  virtual nlohmann::json getDefaultParameters();
 protected:
  std::unordered_map<std::string, nlohmann::json> *context();
  std::unordered_map<std::string, std::string> *env();
  void setPluginManager(PluginManager *pm);

  PluginManager *pm_;

  friend class PluginManager;
};
#endif
