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

#include "json2pydict.h"
#include <spdlog/spdlog.h>

namespace json2pydict {
pybind11::object convertBool(const nlohmann::json &_json) { return pybind11::bool_(_json.get<bool>()); }
pybind11::object convertNumeric(const nlohmann::json &_json) { return pybind11::float_(_json.get<double>()); }
pybind11::object convertIntegral(const nlohmann::json &_json) { return pybind11::int_(_json.get<long>()); }
pybind11::object convertString(const nlohmann::json &_json) { return pybind11::str(_json.get<std::string>()); }

pybind11::object convert(const nlohmann::json &_json) {
  if (_json.is_array()) return convertArray(_json);
  else if (_json.is_object()) return convertObject(_json);
  else if (_json.is_boolean()) return convertBool(_json);
  else if (_json.is_number_float()) return convertNumeric(_json);
  else if (_json.is_number_integer()) return convertIntegral(_json);
  else if (_json.is_string()) return convertString(_json);
  else {
    spdlog::error("Unknown type: {0}", _json.type());
    return pybind11::str("Error");
  }
}

pybind11::object convertObject(const nlohmann::json &_json) {
  pybind11::dict dict;
  for (auto it = _json.begin(); it != _json.end(); ++it) {
    dict[it.key().c_str()] = convert(it.value());
  }
  return dict;
}

pybind11::object convertArray(const nlohmann::json &_json) {
  pybind11::list list;
  for (auto it = _json.begin(), end = _json.end(); it != end; ++it) {
    list.append(convert(*it));
  }
  return list;
}
}