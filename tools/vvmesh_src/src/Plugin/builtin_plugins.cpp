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
//  CLASS builtin_plugins - IMPLEMENTATION
//
//================================================================

//== INCLUDES ====================================================

#include "builtin_plugins.h"
#include <spdlog/spdlog.h>
#include "filesystem/ghc/filesystem.hpp"
#include "filesystem/glob/glob.hpp"
//== CONSTANTS ===================================================



//== IMPLEMENTATION ==============================================
bool DefEnvPlugin::operator()(const nlohmann::json &blockJson) {
  // Verify block
  if (blockJson["Worker"].get<std::string>() != WORKER_NAME) {
    spdlog::warn("Worker[{0}] does not match {1}.\n", blockJson["Worker"].get<std::string>(), WORKER_NAME);
    return false;
  }
  nlohmann::json paramJson = blockJson["Param"];

  for (auto it = paramJson.begin(); it != paramJson.end(); ++it) {
    if (it->is_primitive()) {
      spdlog::debug("Get Env {0}: {1}", it.key(), it->get<std::string>());
      env()->insert(std::make_pair(it.key(), it->get<std::string>()));
    }
  }
  return true;
}
nlohmann::json DefEnvPlugin::getDefaultParameters() {
  nlohmann::json blockJson, paramJson;
  blockJson["Worker"] = WORKER_NAME;
//  blockJson["Worker"].setComment(std::string("// This plugin defines environmental variables shared by all Plugins. "
//                                 "Values should be basic types. Environmental variables could be used for inline substitution."
//                                 " e.g. /home/[USER]/qwer.txt"), before);
  paramJson["USER"] = "My";
  blockJson["Param"] = paramJson;
  return blockJson;
}
//----------------------------------------------------------------

bool DefContextPlugin::operator()(const nlohmann::json &blockJson) {
  // Verify block
  if (blockJson["Worker"].get<std::string>() != WORKER_NAME) {
    spdlog::warn("Worker[{0}] does not match {1}.\n", blockJson["Worker"].get<std::string>(), WORKER_NAME);
    return false;
  }
  nlohmann::json paramJson = blockJson["Param"];

  for (auto it = paramJson.begin(); it != paramJson.end(); ++it) {
    (*context())[it.key()] = *it;
  }
  return true;
}
nlohmann::json DefContextPlugin::getDefaultParameters() {
  nlohmann::json blockJson, paramJson, listJson;
//  auto before = Json::CommentPlacement::commentBefore;
  blockJson["Worker"] = WORKER_NAME;
//  blockJson["Worker"].setComment(std::string("// This plugin defines context variables shared by all Plugins. "
//                                 "Values should be Json object. Context variables could be used for evaluation."
//                                 " e.g. <KEY1>."), before);
  listJson.push_back(0);
  listJson.push_back(3);
  listJson.push_back(7);
  paramJson["KEY1"] = listJson;
  blockJson["Param"] = paramJson;
  return blockJson;
}
//----------------------------------------------------------------
bool PrintPlugin::operator()(const nlohmann::json &blockJson) {
  std::cout << "/// PRINT STATUS ///\n";
  std::cout << "/// Env ///\n";
  for (auto it = env()->begin(); it != env()->end(); ++it) {
    std::cout << it->first << ": " << it->second << "\n";
  }
  std::cout << "/// Context ///\n";
  for (auto it = context()->begin(); it != context()->end(); ++it) {
    std::cout << it->first << ": " << it->second << "\n";
  }
  std::cout << "/// Block ///\n";
  std::cout << blockJson << std::endl;
  std::cout << "/// END ///\n";
  return true;
}
nlohmann::json PrintPlugin::getDefaultParameters() {
  nlohmann::json blockJson;
//  auto before = Json::CommentPlacement::commentBefore;
  blockJson["Worker"] = WORKER_NAME;
//  blockJson["Worker"].setComment(std::string("// Print environmental vars and context vals, and self."), before);
  blockJson["Print"] = "me";
  return blockJson;
}
//----------------------------------------------------------------

bool GlobPlugin::operator()(const nlohmann::json &blockJson) {
  // Verify block
  if (blockJson["Worker"].get<std::string>() != WORKER_NAME) {
    spdlog::warn("Worker[{0}] does not match {1}.\n", blockJson["Worker"].get<std::string>(), WORKER_NAME);
    return false;
  }
  nlohmann::json paramJson = blockJson["Param"];

  for (auto it = paramJson.begin(); it != paramJson.end(); ++it) {
    std::string key = it.key();
    std::string query = it->get<std::string>();

    nlohmann::json list;
    for (auto &p: glob::glob(query)) {
      list.push_back(p.string());
    }
    (*context())[key] = list;
  }
  return true;
}

nlohmann::json GlobPlugin::getDefaultParameters() {
  nlohmann::json blockJson, paramJson;
//  auto before = Json::CommentPlacement::commentBefore;
  blockJson["Worker"] = WORKER_NAME;
//  blockJson["Worker"]
//      .setComment(std::string("// This plugin search files on disk with glob and store them as context vars."), before);
  paramJson["home_files"] = "/home/*";
  blockJson["Param"] = paramJson;
  return blockJson;
}

//----------------------------------------------------------------
#include <regex>

bool RegexReplacePlugin::processTask(const nlohmann::json &taskJson) {
  nlohmann::json _in = taskJson["in"].get<nlohmann::json>();
  if (_in.is_null()) return false;
  std::string _outKey = taskJson["out"].get<std::string>(); // key of context
  const std::regex trimPattern("[<>]");
  _outKey = std::regex_replace(_outKey, trimPattern, "");

  std::regex pattern(taskJson["pattern"].get<std::string>());
  std::string replace = taskJson["replace"].get<std::string>();

  if (_in.is_array()) {
    for (int i = 0; i < _in.size(); ++i) {
      _in[i] = std::regex_replace(_in[i].get<std::string>(), pattern, replace);
    }
  } else {
    _in = std::regex_replace(_in.get<std::string>(), pattern, replace);
  }
  (*context())[_outKey] = _in;
  return true;
}
bool RegexReplacePlugin::operator()(const nlohmann::json &blockJson) {
  // Verify block
  if (blockJson["Worker"].get<std::string>() != WORKER_NAME) {
    spdlog::warn("Worker[{0}] does not match {1}.\n", blockJson["Worker"].get<std::string>(), WORKER_NAME);
    return false;
  }
  nlohmann::json paramJson = blockJson["Param"];

  if (paramJson.is_array()) {
    for (int i = 0; i < paramJson.size(); ++i)
      if (!processTask(paramJson[i]))
        return false;
  } else {
    if (!processTask(paramJson))
      return false;
  }

  return true;
}

nlohmann::json RegexReplacePlugin::getDefaultParameters() {
  nlohmann::json blockJson, paramJson, taskJson;
//  auto before = Json::CommentPlacement::commentBefore;
  blockJson["Worker"] = WORKER_NAME;
//  blockJson["Worker"]
//      .setComment(std::string("// This plugin search files on disk with glob and store them as context vars."), before);
  taskJson["in"] = "<VAR1>";
  taskJson["out"] = "<MOD_VAR1>";
  taskJson["pattern"] = "abc(*)fg";
  taskJson["replace"] = "d";
  paramJson.push_back(taskJson);
  blockJson["Param"] = paramJson;
  return blockJson;
}
//================================================================