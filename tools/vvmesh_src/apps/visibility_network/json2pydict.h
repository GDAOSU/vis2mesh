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

#ifndef VVMESH_APPS_VISIBILITY_NETWORK_JSON2PYDICT_H_
#define VVMESH_APPS_VISIBILITY_NETWORK_JSON2PYDICT_H_

#include <pybind11/embed.h>
#include <nlohmann/json.hpp>

namespace json2pydict {
pybind11::object convertArray(const nlohmann::json &_json);
pybind11::object convertObject(const nlohmann::json &_json);
pybind11::object convertBool(const nlohmann::json &_json);
pybind11::object convertNumeric(const nlohmann::json &_json);
pybind11::object convertIntegral(const nlohmann::json &_json);
pybind11::object convertString(const nlohmann::json &_json);
pybind11::object convert(const nlohmann::json &_json);
}

#endif //VVMESH_APPS_VISIBILITY_NETWORK_JSON2PYDICT_H_
