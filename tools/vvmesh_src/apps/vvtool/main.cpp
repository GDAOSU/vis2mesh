#include <iostream>

#include "FBORender/FBORender.h"

#include <memory>
#include <nlohmann/json.hpp>

#include "Plugin/builtin_plugins.h"
#include "Plugin/pluginmanager.h"
#include "GLRENDER_plugin.h"
// #include "WWZBFILTER_plugin.h"
// #include "WAPPZBFILTER_plugin.h"
// #include "CONFLATION_DELAUNAY_GC_plugin.h"
// #include "HPRFILTER_plugin.h"
#include "GTFILTER_plugin.h"
#include "RAYGTFILTER_plugin.h"
#include "CLOUD_BUNDLE_plugin.h"
// #include "DELAUNAY_GC_plugin.h"
#include "NETWORKVISIBILITY_plugin.h"
//#include "O3DHPRVISIBILITY_plugin.h"

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

static std::shared_ptr<FBORender::MultidrawFBO> fbo;

inline bool setup() {
  if (!fbo) {
    fbo.reset(new FBORender::MultidrawFBO);
  }
  if (!FBORender::initGLContext())
    return false;
  if (!FBORender::GLMakeContextCurrent())
    return false;
  return true;
}
cxxopts::Options parseOptions() {
  cxxopts::Options options("vvtool", "virtual view toolbox.");
  options.add_options()
      ("i,input_json", "Input command file.json", cxxopts::value<std::string>()->default_value(""))
      ("l,list", "List all available plugins.")
      ("w,write", "Write example command to file.", cxxopts::value<std::string>()->implicit_value("example.json"))
      ("h,help", "Print usage of program")
      ("q,query", "Query usage or plugin.", cxxopts::value<std::vector<std::string>>());
  options.parse_positional("input_json");
  options.positional_help("command_file.json");
  return options;
}
PluginManager pm;
std::pair<int, int> registerAllPlugins() {
  uint32_t result;
  result += pm.registerPlugin(std::make_shared<DefEnvPlugin>()) ? 1 : 1 << 16;
  result += pm.registerPlugin(std::make_shared<DefContextPlugin>()) ? 1 : 1 << 16;
  result += pm.registerPlugin(std::make_shared<PrintPlugin>()) ? 1 : 1 << 16;
  result += pm.registerPlugin(std::make_shared<GlobPlugin>()) ? 1 : 1 << 16;
  result += pm.registerPlugin(std::make_shared<RegexReplacePlugin>()) ? 1 : 1 << 16;
  result += pm.registerPlugin(std::make_shared<GLRENDER_Plugin>(fbo)) ? 1 : 1 << 16;
  result += pm.registerPlugin(std::make_shared<GTFILTER_Plugin>()) ? 1 : 1 << 16;
  result += pm.registerPlugin(std::make_shared<RAYGTFILTER_Plugin>()) ? 1 : 1 << 16;
  result += pm.registerPlugin(std::make_shared<CLOUD_BUNDLE_Plugin>()) ? 1 : 1 << 16;
  result += pm.registerPlugin(std::make_shared<NETWORKVISIBILITY_Plugin>()) ? 1 : 1 << 16;

  return std::make_pair((int) (result & 0x0000FFFF), result >> 16);
}

int main(int argc, char **argv) {
  // Initialization
  spdlog::set_level(spdlog::level::info);
  if (!setup()) {
    spdlog::error("GL environment setup failed.");
    return EXIT_FAILURE;
  }
  auto regresult = registerAllPlugins();
  spdlog::info("Plugins registered: {0} successed {1} failed ", regresult.first, regresult.second);
  // Cmd options processing
  auto options = parseOptions();
  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    std::cout << "List of Available Plugins" << pm.statusPlugins() << std::endl;
    exit(0);
  }
  if (result.count("query")) {
    auto querynames = result["query"].as<std::vector<std::string>>();
    nlohmann::json helpinfo;
    for (auto &queryname: querynames) {
      helpinfo.push_back(pm.helpPlugin(queryname));
    }
    std::cout << helpinfo;
    if (result.count("write")) {
      std::string outputpath = result["write"].as<std::string>();
      std::ofstream ofs(outputpath);
      ofs << helpinfo;
      spdlog::info("Query result has been written into {0}", outputpath);
    }
    exit(0);
  }
  if (result.count("list")) {
    std::cout << "List of Available Plugins" << pm.statusPlugins() << std::endl;
    exit(0);
  }

  std::string input_param_path = result["input_json"].as<std::string>();
  if (input_param_path.empty()) {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  nlohmann::json rootJson;
  {
    std::ifstream ifs(input_param_path);
    if (!ifs.is_open()) {
      spdlog::error("Open file {0} failed", input_param_path);
      exit(0);
    }
    ifs >> rootJson;
    ifs.close();
  }
  // Input commands processing
  spdlog::info("Input Json: {0}", input_param_path);
  spdlog::info("=== Initializing Plugins ===");

  if (rootJson.is_array()) {
    int cnt = rootJson.size();
    spdlog::info("=== Input Process Units({0}) ===", cnt);

    int i = 0;
    for (auto &j: rootJson) {
      spdlog::info("=== Processing {0}/{1} ===", ++i, cnt);
      pm.callPlugin(j);
    };
  } else {
    spdlog::info("=== Input Process Units(1) ===");
    pm.callPlugin(rootJson);
  }

  spdlog::info("=== Process Done ===");
  return 0;
}
