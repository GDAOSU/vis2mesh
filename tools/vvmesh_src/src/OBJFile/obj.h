#ifndef OBJ_H
#define OBJ_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include <stdint.h>

typedef struct Point3f{
  Point3f()
  {
    std::fill_n(data,3,0.f);
  }
  Point3f(float _x, float _y, float _z)
  {
    data[0]=_x;
    data[1]=_y;
    data[2]=_z;
  }
  float data[3];
  float &operator[](int i) { return data[i]; }
  const float &operator[](int i) const { return data[i]; }
  float &x() { return data[0]; }
  const float &x() const { return data[0]; }
  float &y() { return data[1]; }
  const float &y() const { return data[1]; }
  float &z() { return data[2]; }
  const float &z() const { return data[2]; }
} Point3f;

typedef struct Point2f{
  Point2f()
  {
    std::fill_n(data,2,0.f);
  }
  Point2f(float _x, float _y){
    data[0]=_x;
    data[1]=_y;
  }
  float data[2];
  float &operator[](int i) { return data[i]; }
  const float &operator[](int i) const { return data[i]; }
  float &x() { return data[0]; }
  const float &x() const { return data[0]; }
  float &y() { return data[1]; }
  const float &y() const { return data[1]; }
} Point2f;

class ObjModel {
 public:
  struct MaterialLib {
    struct Material {
      std::string name;
      std::string diffuse_name;
      cv::Mat diffuse_map;

      Material() {}
      Material(const std::string &_name) : name(_name) {}
      Material(const cv::Mat &_diffuse_map);

      bool LoadDiffuseMap();
    };
    typedef std::vector<Material> Materials;

    Materials materials;
    MaterialLib();
    // Saves the material lib to a .mtl file and all textures of its materials with the given prefix name
    bool Save(const std::string &prefix, bool texLossless = false) const;
    // Loads the material lib from a .mtl file and all textures of its materials with the given file name
    bool Load(const std::string &fileName);
  };

  typedef Point3f Vertex;
  typedef Point3f Color;
  typedef Point2f TexCoord;
  typedef Point3f Normal;

  typedef uint32_t Index;

  struct Face {
    Index vertices[3];
    Index texcoords[3];
    Index normals[3];
  };

  struct Group {
    std::string material_name;
    std::vector<Face> faces;
  };

  typedef std::vector<Vertex> Vertices;
  typedef std::vector<Color> Colors;
  typedef std::vector<TexCoord> TexCoords;
  typedef std::vector<Normal> Normals;
  typedef std::vector<Group> Groups;

 protected:
  Vertices vertices;
  Colors vertex_colors;
  TexCoords texcoords;
  Normals normals;
  Groups groups;
  MaterialLib material_lib;
 public:
  ObjModel() {}

  // Saves the obj model to an .obj file, its material lib and the materials with the given file name
  bool Save(const std::string &fileName, unsigned precision = 6, bool texLossless = false) const;
  // Loads the obj model from an .obj file, its material lib and the materials with the given file name
  bool Load(const std::string &fileName);

  // Creates a new group with the given material name
  Group &AddGroup(const std::string &material_name);
  // Retrieves a material from the library based on its name
  MaterialLib::Material *GetMaterial(const std::string &name);

  MaterialLib &get_material_lib() { return material_lib; }
  Vertices &get_vertices() { return vertices; }
  Colors &get_vertex_colors() {return vertex_colors;}
  TexCoords &get_texcoords() { return texcoords; }
  Normals &get_normals() { return normals; }
  Groups &get_groups() { return groups; }
};

#endif // OBJ_H