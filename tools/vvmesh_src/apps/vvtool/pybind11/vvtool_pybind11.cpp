#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
namespace py = pybind11;

#include "FBORender/FBORender.h"

#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>

#include "Plugin/pluginmanager.h"
#include "GLRENDER_plugin.h"
#include "WWZBFILTER_plugin.h"
#include "WAPPZBFILTER_plugin.h"
#include "HPRFILTER_plugin.h"
#include "GTFILTER_plugin.h"
#include "CLOUD_BUNDLE_plugin.h"
#include "DELAUNAY_GC_plugin.h"

static std::shared_ptr<FBORender::MultidrawFBO> fbo;
static PluginManager pm;

bool setupFBO() {
  if (!fbo) {
    fbo.reset(new FBORender::MultidrawFBO);
  }
  FBORender::initGLContext();
  FBORender::GLMakeContextCurrent();
  return true;
}

int initModule() {
  if (!setupFBO()) {
    fprintf(stderr, "GL environment setup failed.\n");
    return EXIT_FAILURE;
  }
  pm.registerPlugin(std::make_shared<GLRENDER_Plugin>(fbo));
  pm.registerPlugin(std::make_shared<WWZBFILTER_Plugin>(fbo));
  pm.registerPlugin(std::make_shared<WAPPZBFILTER_Plugin>(fbo));
  pm.registerPlugin(std::make_shared<HPRFILTER_Plugin>());
  pm.registerPlugin(std::make_shared<GTFILTER_Plugin>());
  pm.registerPlugin(std::make_shared<CLOUD_BUNDLE_Plugin>());
  pm.registerPlugin(std::make_shared<DELAUNAY_GC_Plugin>());
  return EXIT_SUCCESS;
}

int execute_json_file(std::string filepath) {
  std::string input_param_path = filepath;
  if (input_param_path.empty()) {
    return EXIT_FAILURE;
  }

  nlohmann::json rootJson;
  {
    std::ifstream ifs(input_param_path, std::ifstream::binary);
    ifs >> rootJson;
    ifs.close();
  }

  printf("=== Input JSON: %s\n", input_param_path.c_str());
  printf("=== Initializing Plugins ===\n");

  if (rootJson.is_array()) {
    int cnt = rootJson.size();
    printf("=== Input Process Units(%d) ===\n", cnt);
    int i = 0;
    for (auto &j : rootJson) {
      printf("=== Processing %d/%d ===\n", i + 1, cnt);
      pm.callPlugin(j);
    };
  } else {
    printf("=== Input Process Units(1) ===");
    pm.callPlugin(rootJson);
  }

  printf("=== Process Done ===\n");
  return EXIT_SUCCESS;
}

int execute_json_string(std::string content) {
  nlohmann::json rootJson;


  bool parsingSuccessful = reader.parse(content.c_str(), rootJson);     //parse process
  if (!parsingSuccessful) {
    std::cerr << "Failed to parse" << reader.getFormattedErrorMessages();
    return EXIT_FAILURE;
  }

  printf("=== Input JSON from String\n");
  printf("=== Initializing Plugins ===\n");

  if (rootJson.is_array()) {
    int cnt = rootJson.size();
    printf("=== Input Process Units(%d) ===\n", cnt);
    int i = 0;
    for (auto &j : rootJson) {
      printf("=== Processing %d/%d ===\n", i + 1, cnt);
      pm.callPlugin(j);
    };
  } else {
    printf("=== Input Process Units(1) ===");
    pm.callPlugin(rootJson);
  }

  printf("=== Process Done ===\n");
  return EXIT_SUCCESS;
}

PYBIND11_MODULE(pyvvtool, m) {
  m.doc() = "Virtual view reconstruction tools python module";
  m.def("execute_json_file", &execute_json_file);
  m.def("execute_json_string", &execute_json_string);
  initModule();
}

