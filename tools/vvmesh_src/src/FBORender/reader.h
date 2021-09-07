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
#ifndef GLOFFSCREENRENDER_READER_H
#define GLOFFSCREENRENDER_READER_H

#include <fstream>
#include <vector>
namespace FBORender {
union PointRecType {
  struct {
    float x;
    float y;
    float z;
    float r;
    float g;
    float b;
    float radius;
  };
  float d[7];
};
typedef std::vector<PointRecType> PointCloudType;

inline PointCloudType readTXTPointCloud(std::string path) {
  std::ifstream ifs(path);
  PointCloudType pointcloud;
  while (!ifs.eof()) {
    float x, y, z, r, g, b, rad;
    ifs >> x >> y >> z >> r >> g >> b >> rad;
    pointcloud.emplace_back(PointRecType{x, y, z, r, g, b, rad});
  }
  ifs.close();
  return pointcloud;
}

union PtType {
  struct {
    float x;
    float y;
    float z;
  };
  float d[3];
};

union CellType {
  struct {
    uint32_t i0;
    uint32_t i1;
    uint32_t i2;
  };
  uint32_t d[3];
};

struct TriMeshType {
  std::vector<PtType> points;
  std::vector<CellType> cells;
};

inline TriMeshType readOBJMesh(std::string path) {
  std::ifstream ifs(path);
  TriMeshType mesh;

  std::string line;
  while (std::getline(ifs, line)) {
    std::istringstream iss(line);
    std::string cmd;

    iss >> cmd;

    if (cmd == "v") {
      float x, y, z;
      iss >> x >> y >> z;
      mesh.points.emplace_back(PtType{x, y, z});
    } else if (cmd == "f") {
      uint32_t i0, i1, i2;
      iss >> i0 >> i1 >> i2;
      mesh.cells.emplace_back(CellType{i0 - 1, i1 - 1, i2 - 1});
    }
  }
  ifs.close();
  return mesh;
}
}  // namespace FBORender
#endif  // GLOFFSCREENRENDER_READER_H
