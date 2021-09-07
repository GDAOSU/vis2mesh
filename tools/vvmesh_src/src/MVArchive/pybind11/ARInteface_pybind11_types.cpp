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

#include "MVArchive/ARInterface.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

namespace py = pybind11;

namespace pybind11 {
namespace detail {
template<typename Tp, int m, int n>
struct type_caster<cv::Matx<Tp, m, n>> {
private:
using _MatTMN = cv::Matx<Tp, m, n>;

public:
PYBIND11_TYPE_CASTER(_MatTMN, _("Matx<T, m, n>"));

bool load(py::handle src, bool convert) {
  if (!convert && !py::array_t<Tp>::check_(src)) {
    return false;
  }
  py::array_t<Tp> buf = py::array_t<Tp>::ensure(src);
  if (!buf || buf.ndim() != 2 || buf.shape(0) != m || buf.shape(1) != n) {
    return false;
  }
  for (auto i0 = 0; i0 < m; ++i0)
    for (auto i1 = 0; i1 < n; ++i1) {
      value(i0, i1) = buf.mutable_at(i0, i1);
    }
  return true;
}
static py::handle cast(const _MatTMN &src, py::return_value_policy policy,
                       py::handle parent) {
  py::array_t<Tp> a({m, n});
  for (auto i0 = 0; i0 < m; ++i0)
    for (auto i1 = 0; i1 < n; ++i1) {
      a.mutable_at(i0, i1) = src(i0, i1);
    }
  return a.release();
}
};

template<typename Tp>
struct type_caster<cv::Point_<Tp>> {
public:
PYBIND11_TYPE_CASTER(cv::Point_<Tp>, _("Point_<T>"));

bool load(py::handle src, bool convert) {
  if (!convert && !py::array_t<Tp>::check_(src)) {
    return false;
  }
  py::array_t<Tp> buf = py::array_t<Tp>::ensure(src);
  if (!buf || buf.ndim() != 1 || buf.shape(0) != 2) {
    return false;
  }
  value.x = buf.mutable_at(0);
  value.y = buf.mutable_at(1);
  return true;
}
static py::handle cast(const cv::Point_<Tp> &src,
                       py::return_value_policy policy, py::handle parent) {
  py::array_t<Tp> a(2);
  a.mutable_at(0) = src.x;
  a.mutable_at(1) = src.y;
  return a.release();
}
};

template<typename Tp>
struct type_caster<cv::Point3_<Tp>> {
public:
PYBIND11_TYPE_CASTER(cv::Point3_<Tp>, _("Point3_<T>"));

bool load(py::handle src, bool convert) {
  if (!convert && !py::array_t<Tp>::check_(src)) {
    return false;
  }
  py::array_t<Tp> buf = py::array_t<Tp>::ensure(src);
  if (!buf || buf.ndim() != 1 || buf.shape(0) != 3) {
    return false;
  }

  value.x = buf.mutable_at(0);
  value.y = buf.mutable_at(1);
  value.z = buf.mutable_at(2);
  return true;
}
static py::handle cast(const cv::Point3_<Tp> &src,
                       py::return_value_policy policy, py::handle parent) {
  py::array_t<Tp> a(3);
  a.mutable_at(0) = src.x;
  a.mutable_at(1) = src.y;
  a.mutable_at(2) = src.z;
  return a.release();
}
};
}  // namespace detail
}  // namespace pybind11

PYBIND11_MAKE_OPAQUE(std::vector<bool>);
PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint16_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint32_t>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<uint32_t>>);
PYBIND11_MAKE_OPAQUE(std::vector<MVSA::Interface::Image>);
PYBIND11_MAKE_OPAQUE(std::vector<MVSA::Interface::Platform>);
PYBIND11_MAKE_OPAQUE(std::vector<MVSA::Interface::Platform::Camera>);
PYBIND11_MAKE_OPAQUE(std::vector<MVSA::Interface::Platform::Pose>);
PYBIND11_MAKE_OPAQUE(std::vector<MVSA::Interface::Vertex>);
PYBIND11_MAKE_OPAQUE(std::vector<MVSA::Interface::Vertex::View>);
PYBIND11_MAKE_OPAQUE(std::vector<MVSA::Interface::Line>);
PYBIND11_MAKE_OPAQUE(std::vector<MVSA::Interface::Line::View>);
PYBIND11_MAKE_OPAQUE(std::vector<MVSA::Interface::Normal>);
PYBIND11_MAKE_OPAQUE(std::vector<MVSA::Interface::Color>);
PYBIND11_MAKE_OPAQUE(std::vector<MVSA::Interface::Mesh::Vertex>);
PYBIND11_MAKE_OPAQUE(std::vector<MVSA::Interface::Mesh::Face>);
PYBIND11_MAKE_OPAQUE(std::vector<MVSA::Interface::Mesh::Normal>);
PYBIND11_MAKE_OPAQUE(std::vector<MVSA::Interface::Mesh::TexCoord>);
PYBIND11_MAKE_OPAQUE(std::vector<MVSA::Interface::Mesh::Texture>);
