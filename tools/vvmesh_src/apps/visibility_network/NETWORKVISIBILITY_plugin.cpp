// Copyright 2021 Shaun Song <sxsong1207@qq.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


//================================================================
//
//  CLASS NETWORKVISIBILITY_plugin - IMPLEMENTATION
//
//================================================================

//== INCLUDES ====================================================

#include "NETWORKVISIBILITY_plugin.h"
#include "json2pydict.h"

#include "filesystem/ghc/filesystem.hpp"
namespace fs = ghc::filesystem;
#include <iostream>
#include <spdlog/spdlog.h>

#include <pybind11/embed.h>
namespace py = pybind11;
using namespace py::literals;
//== CONSTANTS ===================================================



//== IMPLEMENTATION ==============================================

NETWORKVISIBILITY_Plugin::NETWORKVISIBILITY_Plugin() : bridge(nullptr) {
}
NETWORKVISIBILITY_Plugin::~NETWORKVISIBILITY_Plugin() {
}
std::string NETWORKVISIBILITY_Plugin::getWorkerName() { return WORKER_NAME; }
bool NETWORKVISIBILITY_Plugin::operator()(const nlohmann::json &blockJson) {
  return processBlock(blockJson);
}

bool NETWORKVISIBILITY_Plugin::initModule() {
  try {
    if (!Py_IsInitialized()) {
      py::initialize_interpreter();
    }
    if (!bridge) {
      bridge = new py::module_;
      auto sys = py::module_::import("sys");
      py::print("Python Configuration:");
      py::print(sys.attr("executable"));
      py::print(sys.attr("version"));
      *bridge = py::module_::import("network_predict");
    }
  } catch (py::error_already_set &e) {
    spdlog::error(e.what());
  }
  return true;
}
bool NETWORKVISIBILITY_Plugin::releaseModule() {
  if (bridge) {
    delete bridge;
  }
  if (Py_IsInitialized()) {
    py::finalize_interpreter();
  }
  return true;
}
bool NETWORKVISIBILITY_Plugin::ensureFolderExist(std::string path) {
  fs::path pp(path);
  if (fs::exists(pp)) {
    if (fs::is_directory(pp)) {
      return true;
    } else {
      return false;
    }
  } else {
    try {
      return fs::create_directory(pp);
    } catch (std::exception &e) {
      std::cout << e.what() << std::endl;
      return false;
    }
  }
}

bool NETWORKVISIBILITY_Plugin::processBlock(const nlohmann::json &blockJson) {
  // Verify block
  if (blockJson["Worker"].get<std::string>() != WORKER_NAME) {
    spdlog::warn("Worker[{0}] does not match {1}.\n", blockJson["Worker"].get<std::string>(), WORKER_NAME);
    return false;
  }
  initModule();
  
  try {
    py::dict block = json2pydict::convert(blockJson);
    (*bridge).attr("call_plugin")(block);
  } catch (py::error_already_set &e) {
    spdlog::error(e.what());
  }

  return true;
}
//----------------------------------------------------------------

nlohmann::json NETWORKVISIBILITY_Plugin::getDefaultParameters() {
  nlohmann::json blockJson, paramJson;

  return blockJson;
}

//================================================================