#include "pluginmanager.h"

#include "plugininterface.h"
#include "preprocess_json.h"

#include <spdlog/spdlog.h>

PluginManager::PluginManager() {}

bool PluginManager::registerPlugin(std::shared_ptr<PluginInterface> plugin) {
  std::string key = plugin->getWorkerName();
  if (plugins_.find(key) == plugins_.end()) {
    // not exists
    plugins_[key] = plugin;
    plugins_[key]->setPluginManager(this);
    spdlog::info("Register Plugin: {0} success.", key.c_str());
    return true;
  } else {
    spdlog::warn("Register Plugin: {0} failed.", key.c_str());
    return false;
  }
}

bool PluginManager::callPlugin(const nlohmann::json &blockJson) {
  std::string queryName = blockJson.value("Worker", "");
  // Make replace list from env vars
  std::unordered_map<std::string, std::string> envreplacelist;
  for (auto it = environmental_vars.begin(); it != environmental_vars.end(); ++it) {
    envreplacelist.insert(std::make_pair("[" + it->first + "]", it->second));
  }

  nlohmann::json processed_json = preprocess_str_json(blockJson, envreplacelist, false);
  processed_json = preprocess_value_json(processed_json, context);
  if (queryName.empty()) {
    spdlog::critical("Worker name is empty");
    return false;
  }
  auto pPlugin = plugins_.find(queryName);
  if (pPlugin == plugins_.end()) {
    spdlog::critical("Plugin {0} not found.", queryName.c_str());
    return false;
  }
  return (*pPlugin->second)(processed_json);
}

bool PluginManager::existPlugin(std::string queryName) {
  auto pPlugin = plugins_.find(queryName);
  return pPlugin != plugins_.end();
}

nlohmann::json PluginManager::helpPlugin(std::string queryName) {
  nlohmann::json value;
  auto pPlugin = plugins_.find(queryName);
  if (pPlugin == plugins_.end()) {
    value = "Requested Plugin[" + queryName + "] have not been registered.";
    return value;
  } else {
    return pPlugin->second->getDefaultParameters();
  }
}

nlohmann::json PluginManager::statusPlugins() {
  nlohmann::json result;
  for (auto &kv: plugins_) {
    result.push_back(kv.first);
  }
  return result;
}