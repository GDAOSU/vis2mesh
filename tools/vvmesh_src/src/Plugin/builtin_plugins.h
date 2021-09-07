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
//  CLASS builtin_plugins
//
//    This class is 
//
//================================================================

#ifndef VVMESH_SRC_PLUGIN_BUILTIN_PLUGINS_H_
#define VVMESH_SRC_PLUGIN_BUILTIN_PLUGINS_H_

//== INCLUDES ====================================================

#include "plugininterface.h"
#include <iostream>

//== CLASS DEFINITION ============================================
class DLL_API DefEnvPlugin : public PluginInterface {
 public:
  virtual std::string getWorkerName() { return WORKER_NAME; }
  virtual bool operator()(const nlohmann::json &blockJson);
  virtual nlohmann::json getDefaultParameters();
 private:
  const char *WORKER_NAME = "DEFENV";
};

class DLL_API DefContextPlugin : public PluginInterface {
 public:
  virtual std::string getWorkerName() { return WORKER_NAME; }
  virtual bool operator()(const nlohmann::json &blockJson);
  virtual nlohmann::json getDefaultParameters();
 private:
  const char *WORKER_NAME = "DEFCONTEXT";
};

class DLL_API PrintPlugin : public PluginInterface {
 public:
  virtual std::string getWorkerName() { return WORKER_NAME; }
  virtual bool operator()(const nlohmann::json &blockJson);
  virtual nlohmann::json getDefaultParameters();
 private:
  const char *WORKER_NAME = "PRINT";
};

/**
 * Read list file on disk using GLOB. Save them to context variable.
 */
class DLL_API GlobPlugin : public PluginInterface {
 public:
  virtual std::string getWorkerName() { return WORKER_NAME; }
  virtual bool operator()(const nlohmann::json &blockJson);
  virtual nlohmann::json getDefaultParameters();
 private:
  const char *WORKER_NAME = "GLOB";
};

class DLL_API RegexReplacePlugin : public PluginInterface {
 public:
  virtual std::string getWorkerName() { return WORKER_NAME; }
  virtual bool operator()(const nlohmann::json &blockJson);
  virtual nlohmann::json getDefaultParameters();
 private:
  bool processTask(const nlohmann::json &taskJson);
  const char *WORKER_NAME = "REGEX_REPLACE";
};
//================================================================



#endif //VVMESH_SRC_PLUGIN_BUILTIN_PLUGINS_H_
