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

#ifndef GLOFFSCREENRENDER_SRC_RENDEROBJECTS_H
#define GLOFFSCREENRENDER_SRC_RENDEROBJECTS_H
#include "dllmacro.h"
#include <glm/glm.hpp>
#include <map>
#include <string>
#include <vector>
#include <GL/glew.h>

namespace FBORender {
GLuint load_shader(const char *code, GLint gl_shader_type);

class DLL_API Shader {
 public:
  Shader();
  ~Shader();
  inline GLuint program() { return mProgram; }
  bool compile(const char *vs_src, const char *gs_src, const char *fs_src);

  bool compile(const char *cs_src);

  void use();
  void unuse();

 protected:
  GLuint mProgram;
};

class DLL_API PointCloudRenderObject {
 public:
  enum { POINT = 0, DOT, BOX, DIAMOND, NUM_PATTERNS };
  PointCloudRenderObject();
  ~PointCloudRenderObject();

  void setPattern(int pattern) { mRenderPattern = pattern; }
  int pattern() { return mRenderPattern; }
  /**
   * @brief build VAO from given points
   * @param data column:[x y z r g b radius]
   * @param num_points number of points
   */
  void buildVAO(const float *data, const size_t num_points);

  void draw(const glm::mat4 &mvMat, const glm::mat4 &projMat,
            const glm::mat4 &mvpMat);

 protected:
  std::vector<GLuint> mVAOs;
  GLuint mDataVBO, mInstIDVBO;
  GLuint mNumPoints;
  int mRenderPattern;
  std::vector<Shader> mShaders;
  std::vector<std::map<std::string, GLint>> mShaderVectorDicts;
};

class DLL_API TriMeshRenderObject {
 public:
  TriMeshRenderObject();
  ~TriMeshRenderObject();

  /**
   * @brief build VAO from given points
   * @param data column:[x y z r g b radius]
   * @param num_points number of points
   */
  void buildVAO(const float *data, const size_t num_points,
                const uint32_t *indices, const size_t num_triangles);
  void draw(const glm::mat4 &mvMat, const glm::mat4 &projMat,
            const glm::mat4 &mvpMat);

 protected:
  GLuint mVAO, mDataVBO, mEBO, mInstIDVBO;
  GLuint mNumVertex, mNumTriangles;
  Shader mShader;
  std::map<std::string, GLint> mShaderVectorDict;
};


class DLL_API RGBTriMeshRenderObject {
 public:
 public:
  RGBTriMeshRenderObject();
  ~RGBTriMeshRenderObject();

  /**
   * @brief build VAO from given points
   * @param data column:[x y z r g b radius]
   * @param num_points number of points
   */
  void buildVAO(const float *data, const size_t num_points,
                const uint32_t *indices, const size_t num_triangles);
  void draw(const glm::mat4 &mvMat, const glm::mat4 &projMat,
            const glm::mat4 &mvpMat);

 protected:
  GLuint mVAO, mDataVBO, mEBO, mInstIDVBO;
  GLuint mNumVertex, mNumTriangles;
  Shader mShader;
  std::map<std::string, GLint> mShaderVectorDict;
};

class DLL_API TexturedTriMeshRenderObject {
 public:
  class DLL_API Patch {
   public:
    enum Format {
      Format_Grayscale = 1,
      Format_GrayscaleAlpha = 2,
      Format_RGB = 3,
      Format_RGBA = 4,
      Format_BGR = 5,
    };
    Patch(std::map<std::string, GLint> &shaderVectorDict);
    ~Patch();
    void setTextureID(GLuint);
    void setTexture(const void *data, int width, int height, Format format);
    void buildVAO(const float *geo_coordinates, const float *tex_coordinates,
                  const size_t num_points, const uint32_t *geo_indices,
                  const uint32_t *tex_indices, const size_t num_triangles,
                  const uint32_t id_offset = 0);
    void draw();
    void clear();

   private:
    GLenum ToTextureFormat(Format format);

   protected:
    GLuint mTextureID;
    int mCreated;

    GLuint mVAO, mDataVBO, mEBO, mInstIDVBO;
    GLuint mNumVertex, mNumTriangles;
    std::map<std::string, GLint> mShaderVectorDict;
  };
  TexturedTriMeshRenderObject();
  ~TexturedTriMeshRenderObject();
  void draw(const glm::mat4 &mvMat, const glm::mat4 &projMat,
            const glm::mat4 &mvpMat);
  Patch *CreatePatch();

 protected:
  Shader mShader;
  std::map<std::string, GLint> mShaderVectorDict;
  std::vector<Patch *> mPatches;
};

class DLL_API WWZBFilterObject {
 public:
  WWZBFilterObject();
  ~WWZBFilterObject();

  void filter(const GLuint inColor, const GLuint inDepth, const GLuint inID,
              const GLuint outColor, const GLuint outDepth, const GLuint outID,
              const float invalid_depth, const uint32_t invalid_index,
              const float focal, const uint32_t kernelSize,
              const float angleTolerance);

 protected:
  Shader mShader;
  std::map<std::string, GLint> mShaderVectorDict;
};
}  // namespace FBORender
#endif  // GLOFFSCREENRENDER_SRC_RENDEROBJECTS_H
