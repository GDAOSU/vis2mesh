// Copyright 2020 Shaun Song <sxsong1207@qq.com>
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

#include "../FBORender.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace FBORender;

//
//namespace pybind11{
//namespace detail{
//
//}
//}


PYBIND11_MODULE(fborender, m) {
  m.doc() = "FBO Renderer and ZB based occlusion detection";


//
//  py::class_<Interface::Platform::Camera>(m, "Platform.Camera")
//      .def_readwrite("name", &Interface::Platform::Camera::name)
//      .def_readwrite("width", &Interface::Platform::Camera::width)
//      .def_readwrite("height", &Interface::Platform::Camera::height)
//      .def_readwrite("K", &Interface::Platform::Camera::K)
//      .def_readwrite("R", &Interface::Platform::Camera::R)
//      .def_readwrite("C", &Interface::Platform::Camera::C)
//      .def(py::init())
//      .def("HasResolution", &Interface::Platform::Camera::HasResolution)
//      .def("IsNormalized", &Interface::Platform::Camera::IsNormalized)
//      .def("GetNormalizationScale", &Interface::Platform::Camera::GetNormalizationScale);
//
//  py::class_<Interface::Platform::Pose>(m, "Platform.Pose")
//      .def_readwrite("R", &Interface::Platform::Pose::R)
//      .def_readwrite("C", &Interface::Platform::Pose::C);
//
//  py::class_<Interface::Platform>(m, "Platform")
//      .def_readwrite("name", &Interface::Platform::name)
//      .def_readwrite("cameras", &Interface::Platform::cameras)
//      .def_readwrite("poses", &Interface::Platform::poses)
//      .def("GetK", &Interface::Platform::GetK)
//      .def("GetPose", &Interface::Platform::GetPose);
//
//  py::class_<Interface::Image>(m, "Image").def(py::init())
//      .def_readwrite("name", &Interface::Image::name)
//      .def_readwrite("width", &Interface::Image::width)
//      .def_readwrite("height", &Interface::Image::height)
//      .def_readwrite("platformID", &Interface::Image::platformID)
//      .def_readwrite("cameraID", &Interface::Image::cameraID)
//      .def_readwrite("poseID", &Interface::Image::poseID)
//      .def_readwrite("ID", &Interface::Image::ID)
//      .def("IsValid", &Interface::Image::IsValid);
//
//  py::class_<Interface::Vertex>(m, "Vertex")
//      .def_readwrite("X", &Interface::Vertex::X)
//      .def_readwrite("views", &Interface::Vertex::views);
//
//  py::class_<Interface::Vertex::View>(m, "Vertex.View")
//      .def_readwrite("imageID", &Interface::Vertex::View::imageID)
//      .def_readwrite("confidence", &Interface::Vertex::View::confidence);
//
//  py::class_<Interface::Line>(m, "Line")
//      .def_readwrite("pt1", &Interface::Line::pt1)
//      .def_readwrite("pt2", &Interface::Line::pt2)
//      .def_readwrite("views", &Interface::Line::views);
//
//  py::class_<Interface::Line::View>(m, "Line.View")
//      .def_readwrite("imageID", &Interface::Line::View::imageID)
//      .def_readwrite("confidence", &Interface::Line::View::confidence);
//
//  py::class_<Interface::Normal>(m, "Normal")
//      .def_readwrite("n", &Interface::Normal::n);
//
//  py::class_<Interface::Color>(m, "Color")
//      .def_readwrite("c", &Interface::Color::c);
//
//  py::class_<Interface::Mesh::Vertex>(m, "Mesh.Vertex")
//      .def_readwrite("X", &Interface::Mesh::Vertex::X);
//  py::class_<Interface::Mesh::Normal>(m, "Mesh.Normal")
//      .def_readwrite("n", &Interface::Mesh::Normal::n);
//  py::class_<Interface::Mesh::Face>(m, "Mesh.Face")
//      .def_readwrite("f", &Interface::Mesh::Face::f);
//  py::class_<Interface::Mesh::TexCoord>(m, "Mesh.TexCoord")
//      .def_readwrite("tc", &Interface::Mesh::TexCoord::tc);
//  py::class_<Interface::Mesh::Texture>(m, "Mesh.Texture")
//      .def_readwrite("path", &Interface::Mesh::Texture::path)
//      .def_readwrite("width", &Interface::Mesh::Texture::width)
//      .def_readwrite("height", &Interface::Mesh::Texture::height)
//      .def_readwrite("data", &Interface::Mesh::Texture::data);
//
//  py::class_<Interface::Mesh>(m, "Mesh")
//      .def_readwrite("vertices", &Interface::Mesh::vertices)
//      .def_readwrite("faces", &Interface::Mesh::faces)
//      .def_readwrite("vertexNormals", &Interface::Mesh::vertexNormals)
//      .def_readwrite("vertexVertices", &Interface::Mesh::vertexVertices)
//      .def_readwrite("vertexFaces", &Interface::Mesh::vertexFaces)
//      .def_readwrite("vertexBoundary", &Interface::Mesh::vertexBoundary)
//      .def_readwrite("faceNormals", &Interface::Mesh::faceNormals)
//      .def_readwrite("faceTexcoords", &Interface::Mesh::faceTexcoords)
//      .def_readwrite("faceMapIdxs", &Interface::Mesh::faceMapIdxs)
//      .def_readwrite("textureDiffuses", &Interface::Mesh::textureDiffuses);
//
//  py::class_<Interface>(m, "Interface")
//      .def_readwrite("platforms", &Interface::platforms)
//      .def_readwrite("images", &Interface::images)
//      .def_readwrite("vertices", &Interface::vertices)
//      .def_readwrite("verticesNormal", &Interface::verticesNormal)
//      .def_readwrite("verticesColor", &Interface::verticesColor)
//      .def_readwrite("lines", &Interface::lines)
//      .def_readwrite("linesNormal", &Interface::linesNormal)
//      .def_readwrite("linesColor", &Interface::linesColor)
//      .def_readwrite("transform", &Interface::transform)
//      .def_readwrite("mesh", &Interface::mesh)
//      .def("GetK", &Interface::GetK)
//      .def("GetPose", &Interface::GetPose);
//
//  py::class_<Scene, Interface>(m, "Scene")
//      .def(py::init())
//      .def(py::init<const std::string &>())
//      .def("load", &Scene::load)
//      .def("save", &Scene::save, py::arg("filename"), py::arg("compress") = true)
//      .def("info", &Scene::info);
}