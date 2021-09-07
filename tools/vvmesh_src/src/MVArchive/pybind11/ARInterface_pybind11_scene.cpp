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
//  CLASS ARInterface_pybind11_scene - IMPLEMENTATION
//
//================================================================

//== INCLUDES ====================================================

#include "ARInterface_pybind11_scene.h"

#include "filesystem/ghc/filesystem.hpp"
namespace fs = ghc::filesystem;

#ifdef _USE_OPENMP
#include <omp.h>
#endif

//== CONSTANTS ===================================================

#define RED "\x1B[31m"
#define GRN "\x1B[32m"
#define YEL "\x1B[33m"
#define BLU "\x1B[34m"
#define MAG "\x1B[35m"
#define CYN "\x1B[36m"
#define WHT "\x1B[37m"
#define RESET "\x1B[0m"


//== IMPLEMENTATION ==============================================
namespace pybind11 {
static const char *FormatName(int format) {
  switch (format) {
    case MVSA::ArchiveFormat::STDIO:return "STDIO";
#ifdef _USE_GZSTREAM
    case MVSA::ArchiveFormat::GZSTREAM:return "GZSTREAM";
#endif // _USE_GZSTREAM
#ifdef _USE_ZSTDSTREAM
    case MVSA::ArchiveFormat::ZSTDSTREAM:return "ZSTDSTREAM";
#endif // _USE_ZSTDSTREAM
#ifdef _USE_COMPRESSED_STREAMS
      case MVSA::ArchiveFormat::BROTLI:
          return "BROTLI";
        case MVSA::ArchiveFormat::LZ4:
          return "LZ4";
        case MVSA::ArchiveFormat::LZMA:
          return "LZMA";
        case MVSA::ArchiveFormat::ZLIB:
          return "ZLIB";
        case MVSA::ArchiveFormat::ZSTD:
          return "ZSTD";
#endif // _USE_COMPRESSED_STREAMS
  }
  return "Unknown";
}

Scene::Scene() {}

Scene::Scene(const std::string &filename) { load(filename); }

bool Scene::load(const std::string &filename) {
  if (!MVSA::MVArchive::SerializeLoad(*this, filename)) return false;
  return true;
}

bool Scene::save(const std::string &filename, int format) {
  if (!MVSA::MVArchive::SerializeSave(*this, filename, format)) return false;
  return true;
}

void Scene::info() {
  printf("File Format: %s\n", FormatName(this->format));
  printf("File Path: %s\n", this->filePath.c_str());
  printf("#img: %zu #platform: %zu\n", this->images.size(),
         this->platforms.size());
  printf("#pts: %zu #color: %zu #normal: %zu\n", this->vertices.size(),
         this->verticesColor.size(), this->verticesNormal.size());
  printf("#lines: %zu #color: %zu #normal: %zu\n", this->lines.size(),
         this->linesColor.size(), this->linesNormal.size());
  printf("Mesh #v: %zu #face: %zu #ftex: %zu #texmaps: %zu\n",
         this->mesh.vertices.size(), this->mesh.faces.size(),
         this->mesh.faceTexcoords.size(), this->mesh.textureDiffuses.size());
  printf(
      "Transform:\n%.3f\t%.3f\t%.3f\t%.3f\n%.3f\t%.3f\t%.3f\t%.3f\n%.3f\t%."
      "3f\t%.3f\t%.3f\n%.3f\t%.3f\t%.3f\t%.3f\n",
      this->transform(0, 0), this->transform(0, 1), this->transform(0, 2),
      this->transform(0, 3), this->transform(1, 0), this->transform(1, 1),
      this->transform(1, 2), this->transform(1, 3), this->transform(2, 0),
      this->transform(2, 1), this->transform(2, 2), this->transform(2, 3),
      this->transform(3, 0), this->transform(3, 1), this->transform(3, 2),
      this->transform(3, 3));
}
//----------------------------------------------------------------
bool Scene::diagnose() const {
  bool good = true;
  size_t num_v = this->vertices.size();
  size_t num_vc = this->verticesColor.size();
  size_t num_vn = this->verticesNormal.size();
  size_t num_img = this->images.size();
  size_t num_plat = this->platforms.size();

  // check imgs
  for (size_t i = 0; i < num_img; ++i) {
    const Image &img = this->images[i];
    if (img.ID == NO_ID) {
      printf(RED
             "Error: Image#%lu ID invalid.\n", i);
      good = false;
      continue;
    }
    if (img.platformID >= num_plat) {
      printf(RED
             "Error: Image#%lu platformID invalid: %u\n", i,
             img.platformID);
      good = false;
      continue;
    }
    const Platform &plat = this->platforms[img.platformID];
    if (img.cameraID >= plat.cameras.size()) {
      printf(RED
             "Error: Image#%lu cameraID invalid: %u\n", i, img.cameraID);
      good = false;
      continue;
    }
    if (img.poseID >= plat.poses.size()) {
      printf(RED
             "Error: Image#%lu poseID invalid: %u\n", i, img.poseID);
      good = false;
      continue;
    }
    
    bool fileExist = fs::exists(img.name);
    if (!fileExist) {
      if (img.width > 0 && img.height > 0) {
        printf(YEL
               "Warn: Image#%lu file not exists: %s\n", i,
               img.name.c_str());
      } else {
        printf(RED
               "Error: Image#%lu file not exists: %s\n", i,
               img.name.c_str());
        good = false;
        continue;
      }
    }
  }
  // check vertices
  for (size_t i = 0; i < num_v; ++i) {
    const Vertex &v = this->vertices[i];
    for (const auto &vw: v.views) {
      if (vw.imageID >= num_img) {
        printf(RED
               "Error: Vertex#%lu related to invalid image: %u\n", i,
               vw.imageID);
        good = false;
        continue;
      }
    }
  }
  if (num_v != num_vc)
    printf(YEL
           "Warn: #V != #VColor (%lu, %lu)\n", num_v, num_vc);
  if (num_v != num_vn)
    printf(YEL
           "Warn: #V != #VNorm (%lu, %lu)\n", num_v, num_vn);
  if (good)
    printf(GRN
           "Pass\n");
  else
    printf(RED
           "Error\n");
  return good;
}

void Scene::clean_unused_images() {
  std::vector<uint32_t> image_reidx(images.size(), NO_ID);
  // scan image usage
#pragma omp parallel for shared(image_reidx)
  for (auto &v: vertices)
    for (auto &vi: v.views)
      image_reidx[vi.imageID] = 1;

  // scan from start to end, assign new idx for used images (==1)
  // since newIdx <= oldIdx consistently, we can move it directly.
  int _imgcnt = 0;
  for (size_t oldidx = 0; oldidx < image_reidx.size(); ++oldidx) {
    if (image_reidx[oldidx] == 1) {
      image_reidx[oldidx] = _imgcnt++;
      images[image_reidx[oldidx]] = images[oldidx];
    }
  }
  images.resize(_imgcnt);
  // Remap idx on vertices
#pragma omp parallel for
  for (auto &v: vertices)
    for (auto &vi: v.views)
      vi.imageID = image_reidx[vi.imageID];
}

void Scene::clean_unused_platforms_poses_cameras() {
  size_t num_plat = platforms.size();
  std::vector<uint32_t> platform_reidx(num_plat, NO_ID);
  std::vector<std::vector<uint32_t>> camera_reidx(num_plat);
  std::vector<std::vector<uint32_t>> pose_reidx(num_plat);
  // Allocate memory
  for (size_t i = 0; i < num_plat; ++i) {
    camera_reidx[i].resize(platforms[i].cameras.size(), NO_ID);
    pose_reidx[i].resize(platforms[i].poses.size(), NO_ID);
  }
  // scan image usage
#pragma omp parallel for shared(platform_reidx, camera_reidx, pose_reidx)
  for (auto &img: images) {
    platform_reidx[img.platformID] = 1;
    camera_reidx[img.platformID][img.cameraID] = 1;
    pose_reidx[img.platformID][img.poseID] = 1;
  }

  // scan from start to end, assign new idx for used images (==1)
  // since newIdx <= oldIdx consistently, we can move it directly.
  for (size_t plid = 0; plid < num_plat; ++plid) {
    int _camcnt = 0, _posecnt = 0;

    auto &cameras = platforms[plid].cameras;
    auto &poses = platforms[plid].poses;

    for (size_t oldidx = 0; oldidx < camera_reidx[plid].size(); ++oldidx) {
      if (camera_reidx[plid][oldidx] == 1) {
        camera_reidx[plid][oldidx] = _camcnt++;
        cameras[camera_reidx[plid][oldidx]] = cameras[oldidx];
      }
    }
    cameras.resize(_camcnt);

    for (size_t oldidx = 0; oldidx < pose_reidx[plid].size(); ++oldidx) {
      if (pose_reidx[plid][oldidx] == 1) {
        pose_reidx[plid][oldidx] = _posecnt++;
        poses[pose_reidx[plid][oldidx]] = poses[oldidx];
      }
    }
    poses.resize(_posecnt);
  }

  int _platcnt = 0;
  for (size_t oldidx = 0; oldidx < num_plat; ++oldidx) {
    if (platform_reidx[oldidx] == 1) {
      platform_reidx[oldidx] = _platcnt++;
      platforms[platform_reidx[oldidx]] = platforms[oldidx];
    }
  }
  platforms.resize(_platcnt);

  // Remap idx on vertices
#pragma omp parallel for
  for (auto &img: images) {
    img.cameraID = camera_reidx[img.platformID][img.cameraID];
    img.poseID = pose_reidx[img.platformID][img.poseID];
    img.platformID = platform_reidx[img.platformID];
  }
}

void Scene::garbage_collect() {
  clean_unused_images();
  clean_unused_platforms_poses_cameras();
}

Scene &Scene::inflate_image_confidence(float scale) {
#pragma omp parallel for
  for (auto &v: vertices)
    for (auto &vi: v.views)
      vi.confidence *= scale;
  return *this;
}

Scene &Scene::append_images(const Scene &other, size_t platform_offset) {
  for (auto &p: other.platforms)
    platforms.push_back(p);

  for (auto img: other.images) {
    img.platformID += platform_offset;
    images.push_back(img);
  }
  return *this;
}

Scene &Scene::append_vertices_lines(const Scene &other, size_t image_offset) {
  //TODO: HANDLE Transform
  for (auto v: other.vertices) {
    for (auto &vi: v.views)
      vi.imageID += image_offset;
    vertices.push_back(v);
  }
  for (auto &vc: other.verticesColor)
    verticesColor.push_back(vc);
  for (auto &vn: other.verticesNormal)
    verticesNormal.push_back(vn);
  /////
  for (auto l: other.lines) {
    for (auto &vi: l.views)
      vi.imageID += image_offset;
    lines.push_back(l);
  }
  for (auto &lc: other.linesColor)
    linesColor.push_back(lc);
  for (auto &ln: other.linesNormal)
    linesNormal.push_back(ln);
  return *this;
}

Scene &Scene::append_mesh(const Scene &other) {
  //TODO: HANDLE Transform
  //TODO: Solve non-manifold
  size_t v_offset = mesh.vertices.size();
  size_t f_offset = mesh.faces.size();
  size_t texid_offset = mesh.textureDiffuses.size();

  for (auto &v: other.mesh.vertices)
    mesh.vertices.push_back(v);
  for (auto &vn: other.mesh.vertexNormals)
    mesh.vertexNormals.push_back(vn);

  for (auto vv: other.mesh.vertexVertices) {
    for (auto &vidx: vv)
      vidx += v_offset;
    mesh.vertexVertices.push_back(vv);
  }

  for (auto vf: other.mesh.vertexFaces) {
    for (auto &fidx: vf)
      fidx += f_offset;
    mesh.vertexFaces.push_back(vf);
  }

  for (auto &vb: other.mesh.vertexBoundary)
    mesh.vertexBoundary.push_back(vb);

  for (auto f: other.mesh.faces) {
    f.f.x += v_offset;
    f.f.y += v_offset;
    f.f.z += v_offset;
    mesh.faces.push_back(f);
  }
  for (auto &fn: other.mesh.faceNormals)
    mesh.faceNormals.push_back(fn);
  for (auto &tc: other.mesh.faceTexcoords)
    mesh.faceTexcoords.push_back(tc);
  for (auto &midx: other.mesh.faceMapIdxs)
    mesh.faceMapIdxs.push_back(midx + texid_offset);

  for (auto &map: other.mesh.textureDiffuses)
    mesh.textureDiffuses.push_back(map);
  return *this;
}

Scene &Scene::append(const Scene &other) {

  int platform_offset = this->platforms.size();
  int image_offset = this->images.size();
  append_images(other, platform_offset);
  append_vertices_lines(other, image_offset);
  append_mesh(other);
  return *this;
}
}
//================================================================