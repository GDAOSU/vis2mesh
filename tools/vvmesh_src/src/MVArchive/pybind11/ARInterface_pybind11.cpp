// Copyright 2020 Shaun Song <sxsong1207@qq.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "MVArchive/ARInterface.h"
#include "MVArchive/ARInterface_impl.hpp"
#include "ARInterface_pybind11_scene.h"

#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>


namespace py = pybind11;
using namespace MVSA;

PYBIND11_MODULE(mvaformat, m) {
  m.doc() = "MultiView Archive Data format";

  py::bind_vector<std::vector<bool>>(m, "BoolArr");
  py::bind_vector<std::vector<uint8_t>>(m, "Uint8Arr");
  py::bind_vector<std::vector<uint16_t>>(m, "Uint16Arr");
  py::bind_vector<std::vector<uint32_t>>(m, "Uint32Arr");
  py::bind_vector<std::vector<std::vector<uint32_t>>>(m, "Uint32ArrArr");
  py::bind_vector<std::vector<Interface::Image>>(m, "ImageArr");
  py::bind_vector<std::vector<Interface::Platform>>(m, "PlatformArr");
  py::bind_vector<std::vector<Interface::Platform::Camera>>(m, "CameraArr");
  py::bind_vector<std::vector<Interface::Platform::Pose>>(m, "PoseArr");
  py::bind_vector<std::vector<Interface::Vertex>>(m, "VertexArr");
  py::bind_vector<std::vector<Interface::Vertex::View>>(m, "VerViewArr");
  py::bind_vector<std::vector<Interface::Line>>(m, "LineArr");
  py::bind_vector<std::vector<Interface::Line::View>>(m, "LineViewArr");
  py::bind_vector<std::vector<Interface::Normal>>(m, "NormalArr");
  py::bind_vector<std::vector<Interface::Color>>(m, "ColorArr");
  py::bind_vector<std::vector<Interface::Mesh::Vertex>>(m, "MeshVertexArr");
  py::bind_vector<std::vector<Interface::Mesh::Face>>(m, "MeshFaceArr");
  py::bind_vector<std::vector<Interface::Mesh::Normal>>(m, "MeshNormalArr");
  py::bind_vector<std::vector<Interface::Mesh::TexCoord>>(m, "MeshTexCoordArr");
  py::bind_vector<std::vector<Interface::Mesh::Texture>>(m, "MeshTextureArr");

  py::class_<Interface::Platform::Camera>(m, "Platform.Camera")
      .def(py::init())
      .def_readwrite("name", &Interface::Platform::Camera::name)
      .def_readwrite("width", &Interface::Platform::Camera::width)
      .def_readwrite("height", &Interface::Platform::Camera::height)
      .def_readwrite("K", &Interface::Platform::Camera::K)
      .def_readwrite("R", &Interface::Platform::Camera::R)
      .def_readwrite("C", &Interface::Platform::Camera::C)
      .def(py::init())
      .def("HasResolution", &Interface::Platform::Camera::HasResolution)
      .def("IsNormalized", &Interface::Platform::Camera::IsNormalized)
      .def_static("GetNormalizationScale",
                  &Interface::Platform::Camera::GetNormalizationScale);

  py::class_<Interface::Platform::Pose>(m, "Platform.Pose")
      .def(py::init())
      .def_readwrite("R", &Interface::Platform::Pose::R)
      .def_readwrite("C", &Interface::Platform::Pose::C);

  py::class_<Interface::Platform>(m, "Platform")
      .def(py::init())
      .def_readwrite("name", &Interface::Platform::name)
      .def_readwrite("cameras", &Interface::Platform::cameras)
      .def_readwrite("poses", &Interface::Platform::poses)
      .def("GetK", &Interface::Platform::GetK)
      .def("GetPose", &Interface::Platform::GetPose);

  py::class_<Interface::Image>(m, "Image")
      .def(py::init())
      .def_readwrite("name", &Interface::Image::name)
      .def_readwrite("width", &Interface::Image::width)
      .def_readwrite("height", &Interface::Image::height)
      .def_readwrite("platformID", &Interface::Image::platformID)
      .def_readwrite("cameraID", &Interface::Image::cameraID)
      .def_readwrite("poseID", &Interface::Image::poseID)
      .def_readwrite("ID", &Interface::Image::ID)
      .def("IsValid", &Interface::Image::IsValid)
      .def("__repr__", [](const Interface::Image &a) {
        return "<Image #" + std::to_string(a.ID) + " (" +
            std::to_string(a.width) + "x" + std::to_string(a.height) +
            ")>: " + a.name;
      });

  py::class_<Interface::Vertex>(m, "Vertex")
      .def(py::init())
      .def_readwrite("X", &Interface::Vertex::X)
      .def_readwrite("views", &Interface::Vertex::views);

  py::class_<Interface::Vertex::View>(m, "Vertex.View")
      .def(py::init())
      .def_readwrite("imageID", &Interface::Vertex::View::imageID)
      .def_readwrite("confidence", &Interface::Vertex::View::confidence);

  py::class_<Interface::Line>(m, "Line")
      .def(py::init())
      .def_readwrite("pt1", &Interface::Line::pt1)
      .def_readwrite("pt2", &Interface::Line::pt2)
      .def_readwrite("views", &Interface::Line::views);

  py::class_<Interface::Line::View>(m, "Line.View")
      .def(py::init())
      .def_readwrite("imageID", &Interface::Line::View::imageID)
      .def_readwrite("confidence", &Interface::Line::View::confidence);

  py::class_<Interface::Normal>(m, "Normal")
      .def(py::init())
      .def_readwrite("n", &Interface::Normal::n);

  py::class_<Interface::Color>(m, "Color")
      .def(py::init())
      .def_readwrite("c", &Interface::Color::c);

  py::class_<Interface::Mesh::Vertex>(m, "Mesh.Vertex")
      .def(py::init())
      .def_readwrite("X", &Interface::Mesh::Vertex::X);
  py::class_<Interface::Mesh::Normal>(m, "Mesh.Normal")
      .def(py::init())
      .def_readwrite("n", &Interface::Mesh::Normal::n);
  py::class_<Interface::Mesh::Face>(m, "Mesh.Face")
      .def(py::init())
      .def_readwrite("f", &Interface::Mesh::Face::f);
  py::class_<Interface::Mesh::TexCoord>(m, "Mesh.TexCoord")
      .def(py::init())
      .def_readwrite("tc", &Interface::Mesh::TexCoord::tc);
  py::class_<Interface::Mesh::Texture>(m, "Mesh.Texture")
      .def(py::init())
      .def_readwrite("path", &Interface::Mesh::Texture::path)
      .def_readwrite("width", &Interface::Mesh::Texture::width)
      .def_readwrite("height", &Interface::Mesh::Texture::height)
      .def_readwrite("data", &Interface::Mesh::Texture::data);

  py::class_<Interface::Mesh>(m, "Mesh")
      .def(py::init())
      .def_readwrite("vertices", &Interface::Mesh::vertices)
      .def_readwrite("faces", &Interface::Mesh::faces)
      .def_readwrite("vertexNormals", &Interface::Mesh::vertexNormals)
      .def_readwrite("vertexVertices", &Interface::Mesh::vertexVertices)
      .def_readwrite("vertexFaces", &Interface::Mesh::vertexFaces)
      .def_readwrite("vertexBoundary", &Interface::Mesh::vertexBoundary)
      .def_readwrite("faceNormals", &Interface::Mesh::faceNormals)
      .def_readwrite("faceTexcoords", &Interface::Mesh::faceTexcoords)
      .def_readwrite("faceMapIdxs", &Interface::Mesh::faceMapIdxs)
      .def_readwrite("textureDiffuses", &Interface::Mesh::textureDiffuses);

  py::class_<Interface>(m, "Interface")
      .def(py::init())
      .def_readwrite("format", &Interface::format)
      .def_readwrite("filePath", &Interface::filePath)
      .def_readwrite("platforms", &Interface::platforms)
      .def_readwrite("images", &Interface::images)
      .def_readwrite("vertices", &Interface::vertices)
      .def_readwrite("verticesNormal", &Interface::verticesNormal)
      .def_readwrite("verticesColor", &Interface::verticesColor)
      .def_readwrite("lines", &Interface::lines)
      .def_readwrite("linesNormal", &Interface::linesNormal)
      .def_readwrite("linesColor", &Interface::linesColor)
      .def_readwrite("transform", &Interface::transform)
      .def_readwrite("mesh", &Interface::mesh)
      .def("GetK", &Interface::GetK)
      .def("GetPose", &Interface::GetPose);

  py::enum_<ArchiveFormat>(m, "ArchiveFormat")
      .value("STDIO", ArchiveFormat::STDIO)
#if _USE_GZSTREAM
      .value("GZSTREAM", ArchiveFormat::GZSTREAM)
#endif // _USE_GZSTREAM
#if _USE_ZSTDSTREAM
      .value("ZSTDSTREAM", ArchiveFormat::ZSTDSTREAM)
#endif // _USE_ZSTDSTREAM
#ifdef _USE_COMPRESSED_STREAMS
    .value("BROTLI", ArchiveFormat::BROTLI)
    .value("LZ4", ArchiveFormat::LZ4)
    .value("LZMA", ArchiveFormat::LZMA)
    .value("ZLIB", ArchiveFormat::ZLIB)
    .value("ZSTD", ArchiveFormat::ZSTD)
#endif // _USE_COMPRESSED_STREAMS
      ;

  py::class_<py::Scene, Interface>(m, "Scene")
      .def(py::init())
      .def(py::init<const std::string &>())
      .def("load", &py::Scene::load)
      .def("save", &py::Scene::save, py::arg("filename"),
           py::arg("format") = ArchiveFormat::STDIO)
#ifdef _USE_GZSTREAM
      .def("save_gzstream", &py::Scene::save, py::arg("filename"),
           py::arg("format") = ArchiveFormat::GZSTREAM)
#endif // _USE_GZSTREAM
#ifdef _USE_ZSTDSTREAM
      .def("save_zstdstream", &py::Scene::save, py::arg("filename"),
           py::arg("format") = ArchiveFormat::ZSTDSTREAM)
#endif // _USE_ZSTDSTREAM
      .def("info", &py::Scene::info)
      .def("diagnose", &py::Scene::diagnose)
      .def("clean_unused_platforms_poses_cameras", &py::Scene::clean_unused_platforms_poses_cameras)
      .def("clean_unused_images", &py::Scene::clean_unused_images)
      .def("garbage_collect", &py::Scene::garbage_collect)
      .def("inflate_image_confidence", &py::Scene::inflate_image_confidence, py::arg("scale"))
      .def("append_images", &py::Scene::append_images, py::arg("other"), py::arg("platform_offset"))
      .def("append_vertices_lines", &py::Scene::append_vertices_lines, py::arg("other"), py::arg("image_offset"))
      .def("append_mesh", &py::Scene::append_mesh, py::arg("other"))
      .def("append", &py::Scene::append, py::arg("other"));
}