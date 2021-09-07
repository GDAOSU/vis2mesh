#include "preprocess_json.h"

#include <regex>
#include <queue>
#include <unordered_map>

std::string regex_escape(std::string input) {
  const std::regex specialChars{R"([-[\]{}()*+?.,\^$|#\s])"};
  return std::regex_replace(input, specialChars, R"(\$&)");
}

nlohmann::json preprocess_value_json(
    const nlohmann::json _json,
    std::unordered_map<std::string, nlohmann::json> kvpair) {
  const std::regex tag_regex("^<(.*)>$");
  std::smatch matches;

  nlohmann::json json = _json;
  std::queue<nlohmann::json *> nodelist;
  nodelist.push(&json);
  while (!nodelist.empty()) {
    nlohmann::json *pnode = nodelist.front();
    nodelist.pop();
    for (nlohmann::json::iterator it = pnode->begin(); it != pnode->end(); ++it) {
      if (it->is_string()) {
        std::string strval = it->get<std::string>();
        std::string tag = "";
        if (std::regex_search(strval, matches, tag_regex)) {
          tag = (matches.end() - 1)->str();
        }
        auto pos = kvpair.find(tag);
        if (pos != kvpair.end())
          (*it) = pos->second;
      } else if (it->is_object()) {
        nodelist.push(&(*it));
      } else if (it->is_array()) {
        nodelist.push(&(*it));
      }
    }
  }

  return json;
}

nlohmann::json preprocess_str_json(
    const nlohmann::json _json,
    std::unordered_map<std::string, std::string> kvpair,
    bool regex) {
  // Compile to regex
  std::vector<std::pair<std::regex, std::string>> regexps;
  for (auto &p: kvpair) {
    std::string sanitized = regex ? p.first : regex_escape(p.first);
    regexps.emplace_back(std::make_pair(std::regex(sanitized), p.second));
  }

  nlohmann::json json = _json;
  std::queue<nlohmann::json *> nodelist;
  nodelist.push(&json);
  while (!nodelist.empty()) {
    nlohmann::json *pnode = nodelist.front();
    nodelist.pop();
    for (nlohmann::json::iterator it = pnode->begin(); it != pnode->end(); ++it) {
      if (it->is_string()) {
        std::string strval = it->get<std::string>();
        for (auto &rp: regexps) {
          strval = std::regex_replace(strval, rp.first, rp.second);
        }
        (*it) = strval;
      } else if (it->is_object()) {
        nodelist.push(&(*it));
      } else if (it->is_array()) {
        nodelist.push(&(*it));
      }
    }
  }

  return json;
}
