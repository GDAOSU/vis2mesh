#include <iostream>

#include <nlohmann/json.hpp>

#include "env_setup.h"
#include "Plugin/pluginmanager.h"
#include "GLRENDER_plugin.h"

void printUsage() { printf("Usage: [executable] input_json\n"); }

int main(int argc, char** argv) {
  if (argc < 2) {
    printUsage();
    return 0;
  }
  std::string input_param_path = argv[1];
  if (input_param_path.empty()) {
    printUsage();
    return 0;
  }

  nlohmann::json rootJson;
  {
    std::ifstream ifs(input_param_path, std::ifstream::binary);
    ifs >> rootJson;
    ifs.close();
  }

  setup();

  PluginManager pm;
  pm.registerPlugin(std::make_shared<GLRENDER_Plugin>(fbo));
  if (rootJson.is_array()) {
    for (auto& j : rootJson) pm.callPlugin(j);
  } else {
    pm.callPlugin(rootJson);
  }
  printf("Done\n");
  return 0;
}
