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
//  CLASS ARInterface_pybind11_scene
//
//    This class is 
//
//================================================================

#ifndef VVMESH_SRC_ARCHIVE_PYBIND11_ARINTERFACE_PYBIND11_SCENE_H_
#define VVMESH_SRC_ARCHIVE_PYBIND11_ARINTERFACE_PYBIND11_SCENE_H_

//== INCLUDES ====================================================
#include "MVArchive/ARInterface.h"

//== CLASS DEFINITION ============================================
namespace pybind11 {
class Scene : public MVSA::Interface {
 public:
  Scene();
  Scene(const std::string &filename);
  bool load(const std::string &filename);
  bool save(const std::string &filename, int format = MVSA::ArchiveFormat::STDIO);
  void info();
  bool diagnose() const;
  void clean_unused_images();
  void clean_unused_platforms_poses_cameras();
  void garbage_collect();
  Scene &inflate_image_confidence(float scale);
  Scene &append_images(const Scene &other, size_t platform_offset);
  Scene &append_vertices_lines(const Scene &other, size_t image_offset);
  Scene &append_mesh(const Scene &other);
  Scene &append(const Scene &other);
};
}
//================================================================



#endif //VVMESH_SRC_ARCHIVE_PYBIND11_ARINTERFACE_PYBIND11_SCENE_H_
