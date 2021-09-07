#ifndef PREPROCESS_JSON_H
#define PREPROCESS_JSON_H

#include "dllmacro.h"
#include <string>
#include <nlohmann/json.hpp>
#include <unordered_map>
DLL_API nlohmann::json preprocess_value_json(
    const nlohmann::json _json,
    std::unordered_map<std::string, nlohmann::json> kvpair);

DLL_API nlohmann::json preprocess_str_json(
    const nlohmann::json _json, std::unordered_map<std::string, std::string> kvpair, bool regex);
#endif  // PREPROCESS_JSON_H
