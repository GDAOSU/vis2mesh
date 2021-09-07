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
//  CLASS O3DHPRVISIBILITY_plugin - IMPLEMENTATION
//
//================================================================

//== INCLUDES ====================================================

#include "O3DHPRVISIBILITY_plugin.h"
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

O3DHPRVISIBILITY_Plugin::O3DHPRVISIBILITY_Plugin() : bridge(new py::module_) {
  py::initialize_interpreter();
  auto sys = py::module_::import("sys");
  py::print("Python Configuration:");
  py::print(sys.attr("executable"));
  py::print(sys.attr("version"));

  *bridge = py::module_::import("hpr_predict");
}
O3DHPRVISIBILITY_Plugin::~O3DHPRVISIBILITY_Plugin() {
//  py::finalize_interpreter();
}
std::string O3DHPRVISIBILITY_Plugin::getWorkerName() { return WORKER_NAME; }
bool O3DHPRVISIBILITY_Plugin::operator()(const nlohmann::json &blockJson) {
  return processBlock(blockJson);
}
bool O3DHPRVISIBILITY_Plugin::ensureFolderExist(std::string path) {
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

bool O3DHPRVISIBILITY_Plugin::processBlock(const nlohmann::json &blockJson) {
  // Verify block
  if (blockJson["Worker"].get<std::string>() != WORKER_NAME) {
    spdlog::warn("Worker[{0}] does not match {1}.\n", blockJson["Worker"].get<std::string>(), WORKER_NAME);
    return false;
  }
  py::dict block = json2pydict::convert(blockJson);
  (*bridge).attr("call_plugin")(block);
  return true;
}
//----------------------------------------------------------------

nlohmann::json O3DHPRVISIBILITY_Plugin::getDefaultParameters() {
  nlohmann::json blockJson, paramJson;

  return blockJson;
}
//================================================================