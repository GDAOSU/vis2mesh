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
//  CLASS O3DHPRVISIBILITY_plugin
//
//    This class is 
//
//================================================================

#ifndef VVMESH_APPS_VISIBILITY_NETWORK_O3DHPRVISIBILITY_PLUGIN_H_
#define VVMESH_APPS_VISIBILITY_NETWORK_O3DHPRVISIBILITY_PLUGIN_H_

//== INCLUDES ====================================================

#include <memory>
#include <nlohmann/json.hpp>
#include "Plugin/plugininterface.h"

//== CLASS DEFINITION ============================================

namespace pybind11 { class module_; }

class O3DHPRVISIBILITY_Plugin : public PluginInterface {
 public:
  O3DHPRVISIBILITY_Plugin();
  ~O3DHPRVISIBILITY_Plugin();
  virtual std::string getWorkerName();
  virtual bool operator()(const nlohmann::json &blockJson);
  virtual nlohmann::json getDefaultParameters();
 private:
  pybind11::module_* bridge;
  const char *WORKER_NAME = "O3DHPRVISIBILITY";

  bool ensureFolderExist(std::string path);
  bool processBlock(const nlohmann::json &blockJson);
};

//================================================================


#endif //VVMESH_APPS_VISIBILITY_NETWORK_O3DHPRVISIBILITY_PLUGIN_H_
