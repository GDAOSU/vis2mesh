#ifndef PLUGINMANAGER_H
#define PLUGINMANAGER_H
#include "dllmacro.h"
#include <unordered_map>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>

class PluginInterface;
class DLL_API PluginManager {
 public:
  PluginManager();
  bool registerPlugin(std::shared_ptr<PluginInterface> plugin);
  bool callPlugin(const nlohmann::json &blockJson);
  bool existPlugin(std::string queryName);
  nlohmann::json helpPlugin(std::string queryName);
  nlohmann::json statusPlugins();

  std::unordered_map<std::string, nlohmann::json> context;
  std::unordered_map<std::string, std::string> environmental_vars;
 protected:
  std::unordered_map<std::string, std::shared_ptr<PluginInterface> > plugins_;
};

#endif
