#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../builtin_plugins.h"
#include "../plugininterface.h"
#include "../pluginmanager.h"
#include "../preprocess_json.h"

int value1 = 0;
class TestPlugin : public PluginInterface {
 public:
  virtual std::string getWorkerName() { return "TESTPLUGIN"; }
  virtual bool operator()(const nlohmann::json &blockJson) {
    value1 = blockJson.value<int>("value", 100);
    return true;
  }
};

PluginManager pm;

TEST_CASE ("call_nonregistered_plugin") {
  nlohmann::json param;
  param["Worker"] = "TESTPLUGIN";
  param["value"] = 1000;
  CHECK(pm.callPlugin(param) == false);
}

TEST_CASE ("register_no_exist") {
  CHECK(pm.registerPlugin(std::make_unique<TestPlugin>()) == true);
  CHECK(pm.registerPlugin(std::make_unique<DefEnvPlugin>()) == true);
  CHECK(pm.registerPlugin(std::make_unique<DefContextPlugin>()) == true);
  CHECK(pm.registerPlugin(std::make_unique<PrintPlugin>()) == true);
  CHECK(pm.registerPlugin(std::make_unique<GlobPlugin>()) == true);
  CHECK(pm.registerPlugin(std::make_unique<RegexReplacePlugin>()) == true);
}

TEST_CASE ("register_exist_plugin") {
  CHECK(pm.registerPlugin(std::make_unique<TestPlugin>()) == false);
}

TEST_CASE ("call_registered_plugin") {
  nlohmann::json param;
  param["Worker"] = "TESTPLUGIN";
  param["value"] = 1000;
  value1 = 0;
  CHECK(pm.callPlugin(param) == true);
  CHECK(value1 == 1000);
}

TEST_CASE ("test_preprocess_json") {
  nlohmann::json param;
  param["hi"] = 100;
  param["MY"] = "100";
  param["E"] = "SSEAA";
  param["CHID1"] = param;
  param["CHID2"] = param;

  std::unordered_map<std::string, std::string> keywordlist;
  keywordlist.insert(std::make_pair("SSE", "EEX"));
  keywordlist.insert(std::make_pair("10", "20"));
  nlohmann::json replaced = preprocess_str_json(param, keywordlist, true);
  CHECK(replaced["hi"].get<int>() == 100);
  CHECK(replaced["MY"].get<std::string>() == "200");
  CHECK(replaced["E"] == nlohmann::json("EEXAA"));
  CHECK(replaced["CHID1"]["E"].get<std::string>() == "EEXAA");
  CHECK(replaced["CHID2"]["CHID1"]["E"].get<std::string>() == "EEXAA");
}

TEST_CASE ("test_env_plugin")
{
  nlohmann::json block;
  block["Worker"] = "DEFENV";
  nlohmann::json param;
  param["HI"] = "114";
  param["HIHI"] = "1251";
  block["Param"] = param;
  CHECK(pm.callPlugin(block));
  REQUIRE_MESSAGE(pm.environmental_vars["HI"] == "114", pm.environmental_vars["HI"]);
  nlohmann::json printBlock;
  printBlock["Worker"] = "PRINT";
  nlohmann::json printParam;
  printParam["OOO 114"] = "OOO 114";
  printParam["HI HI HIHI"] = "HI HI HIHI";
  printParam["[HI] HI [HIHI]=>114 HI 1251"] = "[HI] HI [HIHI]";
  printParam["WELL"] = "WELL";
  printParam["[HIHI HI]"] = "[HIHI HI]";
  printBlock["Param"] = printParam;
  CHECK(pm.callPlugin(printBlock));
}

TEST_CASE ("test_context_plugin")
{
  nlohmann::json block;
  block["Worker"] = "DEFCONTEXT";
  nlohmann::json param;
  param["HI"] = "114";
  param["HIHI"] = "1251";
  nlohmann::json p1, p2;
  nlohmann::json arr;
  p1["wegew"] = 12512;
  p1["weg"] = 15.f;
  p1["xx1"] = 44;
  p2["path"] = "/home/esxs";
  p1["p2"] = p2;
  param["p1"] = p1;
  int x = 1;
  for (int i = 0; i < 10; ++i) {
    x += i;
    arr.push_back(x);
  }
  param["arr"] = arr;
  param["TEST"] = "<arr>";
  block["Param"] = param;
  CHECK(pm.callPlugin(block));
  nlohmann::json printBlock;
  printBlock["Worker"] = "PRINT";
  printBlock["Param"] = param;
  CHECK(pm.callPlugin(printBlock));
}

#include "filesystem/glob/glob.hpp"
TEST_CASE ("glob_plugin")
{
  nlohmann::json block;
  block["Worker"] = "GLOB";
  nlohmann::json param;
  param["input_cams"] =
      "/home/sxs/GDA/ONR/ONR_DEMO_2021AU/Conflation/Dataset/Garage/DRONE_RGBMESH_11_building/DRONE_RGBPOINT_11_building.ply_WORK/cam999_GT.POINT_DELAY/render/cam*.json";
  param["input_depths"] =
      "/home/sxs/GDA/ONR/ONR_DEMO_2021AU/Conflation/Dataset/Garage/DRONE_RGBMESH_11_building/DRONE_RGBPOINT_11_building.ply_WORK/cam999_GT.POINT_DELAY/render/pt*.flt";
  block["Param"] = param;
  CHECK(pm.callPlugin(block));
  nlohmann::json printBlock;
  printBlock["Worker"] = "PRINT";
  CHECK(pm.callPlugin(printBlock));
}

TEST_CASE ("regex_plugin")
{
  nlohmann::json block;
  block["Worker"] = "REGEX_REPLACE";
  nlohmann::json param1, param2, task;

  task["in"] = "<input_cams>";
  task["out"] = "<mesh>";
  task["pattern"] = "cam(\\d+).json";
  task["replace"] = "mesh$1.png";
  param1.push_back(task);
  block["Param"] = param1;
  CHECK(pm.callPlugin(block));
  nlohmann::json printBlock;
  printBlock["Worker"] = "PRINT";
//  CHECK(pm.callPlugin(printBlock));
}