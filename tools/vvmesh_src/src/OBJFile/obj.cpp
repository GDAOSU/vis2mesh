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

#include "obj.h"
#include <fstream>
#include <iostream>
#include <iomanip>

//#include <filesystem/path.h>
#include "filesystem/ghc/filesystem.hpp"

namespace fs = ghc::filesystem;

#define OBJ_INDEX_OFFSET 1
#ifndef NO_ID
#define NO_ID UINT32_MAX
#endif

ObjModel::MaterialLib::Material::Material(const cv::Mat &_diffuse_map) :
    diffuse_map(_diffuse_map) {

}
bool ObjModel::MaterialLib::Material::LoadDiffuseMap() {

  if (diffuse_map.empty())
    diffuse_map = cv::imread(diffuse_name);
  return true;
}

ObjModel::MaterialLib::MaterialLib() {

}

bool ObjModel::MaterialLib::Save(const std::string &prefix, bool texLossless) const {
  std::ofstream out((prefix + ".mtl").c_str());
  if (!out.good())
    return false;


  const std::string pathName = fs::path(prefix).parent_path().string();
  const std::string name = fs::path(prefix).filename();

  for (int i = 0; i < (int) materials.size(); ++i) {
    Material mat = materials[i];
    // save material description
    out << "newmtl " << mat.name << "\n"
        << "Ka 1.000000 1.000000 1.000000" << "\n"
        << "Kd 0.900000 0.900000 0.900000" << "\n"
        << "Ks 0.000000 0.000000 0.000000" << "\n"
        << "Tr 1.000000" << "\n"
        << "illum 1" << "\n"
        << "Ns 1.000000" << "\n";
    // save material maps
    if (mat.diffuse_map.empty())
      continue;
    std::string diff_basename;
    std::string diff_path;
    if (mat.diffuse_name.empty()) {
      diff_basename = name + "_" + mat.name + "_map_Kd." + (texLossless ? "png" : "jpg");
      diff_path = (fs::path(pathName) / diff_basename).string();
      mat.diffuse_name = diff_path;
    } else {
      diff_basename = fs::path(mat.diffuse_name).filename();
      diff_path = (fs::path(pathName) / diff_basename).string();
    }

    out << "map_Kd " << diff_basename << "\n";

    fs::create_directories(pathName);
    bool bRet = cv::imwrite(diff_path, mat.diffuse_map);
    if (!bRet)
      return false;
  }
  return true;
}

bool ObjModel::MaterialLib::Load(const std::string &fileName) {
  const size_t numMaterials(materials.size());
  std::ifstream in(fileName.c_str());
  std::string keyword;
  while (in.good() && in >> keyword) {
    if (keyword == "newmtl") {
      in >> keyword;
      materials.push_back(Material(keyword));
    } else if (keyword == "map_Kd") {
      assert(numMaterials < materials.size());
      std::string &diffuse_name = materials.back().diffuse_name;
      in >> diffuse_name;
      diffuse_name = (fs::path(fileName).parent_path() / diffuse_name).string();
    }
  }
  return numMaterials < materials.size();
}

ObjModel::Group &ObjModel::AddGroup(const std::string &material_name) {
  groups.push_back(Group());
  Group &group = groups.back();
  group.material_name = material_name;
  if (!GetMaterial(material_name))
    material_lib.materials.push_back(MaterialLib::Material(material_name));
  return group;
}

ObjModel::MaterialLib::Material *ObjModel::GetMaterial(const std::string &name) {
  MaterialLib::Materials::iterator it(std::find_if(material_lib.materials.begin(),
                                                   material_lib.materials.end(),
                                                   [&name](const MaterialLib::Material &mat) {
                                                     return mat.name == name;
                                                   }));
  if (it == material_lib.materials.end())
    return NULL;
  return &(*it);
}

bool ObjModel::Load(const std::string &fileName) {
  assert(vertices.empty() && groups.empty() && material_lib.materials.empty());
  std::ifstream fin(fileName.c_str());
  std::string line, keyword;
  std::istringstream in;
  while (fin.good()) {
    std::getline(fin, line);
    if (line.empty() || line[0u] == '#')
      continue;
    in.str(line);
    in >> keyword;
    if (keyword == "v") {
      Vertex v;
      in >> v[0] >> v[1] >> v[2];
      if (in.fail())
        return false;
      vertices.push_back(v);
      Color c;
      in >> c[0] >> c[1] >> c[2];
      if (!in.fail())
        vertex_colors.push_back(c);

    } else if (keyword == "vt") {
      TexCoord vt;
      in >> vt[0] >> vt[1];
      if (in.fail())
        return false;
      texcoords.push_back(vt);
    } else if (keyword == "vn") {
      Normal vn;
      in >> vn[0] >> vn[1] >> vn[2];
      if (in.fail())
        return false;
      normals.push_back(vn);
    } else if (keyword == "f") {
      Face f;
      memset(&f, 0xFF, sizeof(Face));
      for (size_t k = 0; k < 3; ++k) {
        in >> keyword;
        switch (std::sscanf(keyword.c_str(), "%u/%u/%u", f.vertices + k, f.texcoords + k, f.normals + k)) {
          case 1:f.vertices[k] -= OBJ_INDEX_OFFSET;
            break;
          case 2:f.vertices[k] -= OBJ_INDEX_OFFSET;
            if (f.texcoords[k] != NO_ID)
              f.texcoords[k] -= OBJ_INDEX_OFFSET;
            if (f.normals[k] != NO_ID)
              f.normals[k] -= OBJ_INDEX_OFFSET;
            break;
          case 3:f.vertices[k] -= OBJ_INDEX_OFFSET;
            f.texcoords[k] -= OBJ_INDEX_OFFSET;
            f.normals[k] -= OBJ_INDEX_OFFSET;
            break;
          default:return false;
        }
      }
      if (in.fail())
        return false;
      if (groups.empty())
        AddGroup("default");
      groups.back().faces.push_back(f);
    } else if (keyword == "mtllib") {
      in >> keyword;
      std::string mtlpath;

      if (fs::path(keyword).is_absolute()) {
        mtlpath = keyword;
      } else {
        mtlpath = (fs::path(fileName).parent_path() / keyword).string();
      }
      if (!material_lib.Load(mtlpath))
        return false;
    } else if (keyword == "usemtl") {
      Group group;
      in >> group.material_name;
      if (in.fail())
        return false;
      groups.push_back(group);
    }
    in.clear();
  }
  return !vertices.empty();
}

bool ObjModel::Save(const std::string &fileName, unsigned precision, bool texLossless) const {
  if (vertices.empty())
    return false;

  std::string namext = fs::path(fileName).filename();
  std::string name = fs::path(fileName).replace_extension("").filename(); // Base name

  std::string prefix = (fs::path(fileName).parent_path() / name).string();

  if (!material_lib.Save(prefix, texLossless))
    return false;

  std::ofstream out((prefix + ".obj").c_str());
  if (!out.good())
    return false;

  out << "mtllib " << name << ".mtl" << "\n";

  out << std::fixed << std::setprecision(precision);
  bool hasColor = vertices.size() == vertex_colors.size();
  for (size_t i = 0; i < vertices.size(); ++i) {
    out << "v "
        << vertices[i][0] << " "
        << vertices[i][1] << " "
        << vertices[i][2];
    if (hasColor) {
      out << " "
          << vertex_colors[i][0] << " "
          << vertex_colors[i][1] << " "
          << vertex_colors[i][2];
    }
    out << "\n";
  }

  for (size_t i = 0; i < texcoords.size(); ++i) {
    out << "vt "
        << texcoords[i][0] << " "
        << texcoords[i][1] << "\n";
  }

  for (size_t i = 0; i < normals.size(); ++i) {
    out << "vn "
        << normals[i][0] << " "
        << normals[i][1] << " "
        << normals[i][2] << "\n";
  }

  for (size_t i = 0; i < groups.size(); ++i) {
    out << "usemtl " << groups[i].material_name << "\n";
    for (size_t j = 0; j < groups[i].faces.size(); ++j) {
      const Face &face = groups[i].faces[j];
      out << "f";
      for (size_t k = 0; k < 3; ++k) {
        out << " " << face.vertices[k] + OBJ_INDEX_OFFSET;
        if (!texcoords.empty()) {
          out << "/" << face.texcoords[k] + OBJ_INDEX_OFFSET;
          if (!normals.empty())
            out << "/" << face.normals[k] + OBJ_INDEX_OFFSET;
        } else if (!normals.empty())
          out << "//" << face.normals[k] + OBJ_INDEX_OFFSET;
      }
      out << "\n";
    }
  }
  return true;
}