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

#include "renderobjects.h"
#include "util.h"

#include <stdexcept>

namespace FBORender {
GLuint load_shader(const char *code, GLint gl_shader_type) {
  GLuint shader = glCreateShader(gl_shader_type);
  glShaderSource(shader, 1, (const char **) &code, NULL);
  glCompileShader(shader);

  GLint status;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
  if (status != GL_TRUE) {
    GLint log_length;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
    std::vector<char> v(log_length);
    glGetShaderInfoLog(shader, log_length, NULL, v.data());
    fprintf(stderr, "==[SHADER_COMP]==\n%s\n", v.data());
    glDeleteShader(shader);
    return -1;
  }
  return shader;
}

Shader::Shader() { mProgram = glCreateProgram(); }

Shader::~Shader() { glDeleteProgram(mProgram); }
bool Shader::compile(const char *vs_src, const char *gs_src,
                     const char *fs_src) {
  GLuint vs = 0, gs = 0, fs = 0;
  if (vs_src != NULL) {
    vs = load_shader(vs_src, GL_VERTEX_SHADER);
    if (!vs) return false;
    glAttachShader(mProgram, vs);
  }
  if (gs_src != NULL) {
    gs = load_shader(gs_src, GL_GEOMETRY_SHADER);
    if (!gs) {
      if (vs) glDeleteShader(vs);
      return false;
    }
    glAttachShader(mProgram, gs);
  }
  if (fs_src != NULL) {
    fs = load_shader(fs_src, GL_FRAGMENT_SHADER);
    if (!fs) {
      if (vs) glDeleteShader(vs);
      if (gs) glDeleteShader(gs);
      return false;
    }
    glAttachShader(mProgram, fs);
  }

  glLinkProgram(mProgram);
  GLint status;
  glGetProgramiv(mProgram, GL_LINK_STATUS, &status);
  if (vs) glDeleteShader(vs);
  if (gs) glDeleteShader(gs);
  if (fs) glDeleteShader(fs);

  if (status != GL_TRUE) {
    GLint log_length;
    glGetProgramiv(mProgram, GL_INFO_LOG_LENGTH, &log_length);
    std::vector<char> v(log_length);
    glGetProgramInfoLog(mProgram, log_length, NULL, v.data());
    fprintf(stderr, "==[PROGRAM_LINK]==\n%s\n", v.data());
    return false;
  }
  return true;
}

bool Shader::compile(const char *cs_src) {
  GLuint cs = 0;
  if (cs_src != NULL) {
    cs = load_shader(cs_src, GL_COMPUTE_SHADER);
    if (!cs) return false;
    glAttachShader(mProgram, cs);
  }

  glLinkProgram(mProgram);
  GLint status;
  glGetProgramiv(mProgram, GL_LINK_STATUS, &status);
  glDeleteShader(cs);

  if (status != GL_TRUE) {
    GLint log_length;
    glGetProgramiv(mProgram, GL_INFO_LOG_LENGTH, &log_length);
    std::vector<char> v(log_length);
    glGetProgramInfoLog(mProgram, log_length, NULL, v.data());
    fprintf(stderr, "==[PROGRAM_LINK]==\n%s\n", v.data());
    return false;
  }
  return true;
}

void Shader::use() { glUseProgram(mProgram); }
void Shader::unuse() { glUseProgram(0); }

PointCloudRenderObject::PointCloudRenderObject() {
  mVAOs.clear();
  mDataVBO = 0;
  mInstIDVBO = 0;

  std::string pvs =
#include "generated/pointcloud.vs"
    ;
  std::string pgs_point =
#include "generated/pointcloud_point.gs"
    ;
  std::string pgs_dot =
#include "generated/pointcloud_dot.gs"
    ;
  std::string pgs_box =
#include "generated/pointcloud_box.gs"
    ;
  std::string pgs_diamond =
#include "generated/pointcloud_diamond.gs"
    ;
  std::string pfs =
#include "generated/pointcloud.fs"
    ;

  mShaders.resize(NUM_PATTERNS);
  mShaders[POINT].compile(pvs.c_str(), pgs_point.c_str(), pfs.c_str());
  mShaders[DOT].compile(pvs.c_str(), pgs_dot.c_str(), pfs.c_str());
  mShaders[BOX].compile(pvs.c_str(), pgs_box.c_str(), pfs.c_str());
  mShaders[DIAMOND].compile(pvs.c_str(), pgs_diamond.c_str(), pfs.c_str());

  mShaderVectorDicts.resize(NUM_PATTERNS);
  for (int type = 0; type < NUM_PATTERNS; ++type) {
    mShaderVectorDicts[type]["vertexMC"] =
        glGetAttribLocation(mShaders[type].program(), "vertexMC");
    mShaderVectorDicts[type]["vertexColorVSInput"] =
        glGetAttribLocation(mShaders[type].program(), "vertexColorVSInput");
    mShaderVectorDicts[type]["radius"] =
        glGetAttribLocation(mShaders[type].program(), "radius");
    mShaderVectorDicts[type]["primitive_id"] =
        glGetAttribLocation(mShaders[type].program(), "primitive_id");
    mShaderVectorDicts[type]["MCVCMatrix"] =
        glGetUniformLocation(mShaders[type].program(), "MCVCMatrix");
    mShaderVectorDicts[type]["VCDCMatrix"] =
        glGetUniformLocation(mShaders[type].program(), "VCDCMatrix");
    mShaderVectorDicts[type]["MCDCMatrix"] =
        glGetUniformLocation(mShaders[type].program(), "MCDCMatrix");
  }
}

PointCloudRenderObject::~PointCloudRenderObject() {
  if (mDataVBO) glDeleteBuffers(1, &mDataVBO);
  if (mInstIDVBO) glDeleteBuffers(1, &mInstIDVBO);
  if (!mVAOs.empty()) glDeleteVertexArrays(mVAOs.size(), mVAOs.data());
}

/**
 * @brief build VAO from given points
 * @param data column:[x y z r g b radius]
 * @param num_points number of points
 */
void PointCloudRenderObject::buildVAO(const float *data,
                                      const size_t num_points) {
  mNumPoints = num_points;
  // Release previous
  if (mDataVBO) glDeleteBuffers(1, &mDataVBO);
  if (mInstIDVBO) glDeleteBuffers(1, &mInstIDVBO);
  if (!mVAOs.empty()) {
    glDeleteVertexArrays(mVAOs.size(), mVAOs.data());
    mVAOs.clear();
  }

  // Create VBO and upload
  glGenBuffers(1, &mDataVBO);
  glGenBuffers(1, &mInstIDVBO);
  glBindBuffer(GL_ARRAY_BUFFER, mDataVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 7 * num_points, data,
               GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, mInstIDVBO);
  std::vector<uint32_t> InstID;
  InstID.resize(num_points);
  for (int i = 0; i < num_points; ++i) InstID[i] = i;
  glBufferData(GL_ARRAY_BUFFER, sizeof(uint32_t) * num_points, InstID.data(),
               GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Create VAO
  mVAOs.resize(NUM_PATTERNS);
  glGenVertexArrays(NUM_PATTERNS, mVAOs.data());
  for (int type = 0; type < NUM_PATTERNS; ++type) {
    glBindVertexArray(mVAOs[type]);
    bool useColor = mShaderVectorDicts[type]["vertexColorVSInput"] != -1;
    bool useRadius = mShaderVectorDicts[type]["radius"] != -1;
    bool usePointID = mShaderVectorDicts[type]["primitive_id"] != -1;

    glEnableVertexAttribArray(mShaderVectorDicts[type]["vertexMC"]);
    if (useColor)
      glEnableVertexAttribArray(mShaderVectorDicts[type]["vertexColorVSInput"]);
    if (useRadius)
      glEnableVertexAttribArray(mShaderVectorDicts[type]["radius"]);
    if (usePointID)
      glEnableVertexAttribArray(mShaderVectorDicts[type]["primitive_id"]);

    glBindBuffer(GL_ARRAY_BUFFER, mDataVBO);
    glVertexAttribPointer(mShaderVectorDicts[type]["vertexMC"], 3, GL_FLOAT,
                          GL_FALSE, sizeof(float) * 7,
                          (GLvoid *) (sizeof(float) * 0));
    if (useColor)
      glVertexAttribPointer(mShaderVectorDicts[type]["vertexColorVSInput"], 3,
                            GL_FLOAT, GL_FALSE, sizeof(float) * 7,
                            (GLvoid *) (sizeof(float) * 3));
    if (useRadius)
      glVertexAttribPointer(mShaderVectorDicts[type]["radius"], 1, GL_FLOAT,
                            GL_FALSE, sizeof(float) * 7,
                            (GLvoid *) (sizeof(float) * 6));
    glBindBuffer(GL_ARRAY_BUFFER, mInstIDVBO);
    if (usePointID)
      glVertexAttribIPointer(mShaderVectorDicts[type]["primitive_id"], 1,
                             GL_UNSIGNED_INT, sizeof(uint32_t), NULL);
    glBindVertexArray(0);
  }
}

void PointCloudRenderObject::draw(const glm::mat4 &mvMat,
                                  const glm::mat4 &projMat,
                                  const glm::mat4 &mvpMat) {
  mShaders[pattern()].use();
  glUniformMatrix4fv(mShaderVectorDicts[pattern()]["MCVCMatrix"], 1, GL_FALSE,
                     glm::value_ptr(mvMat));
  glUniformMatrix4fv(mShaderVectorDicts[pattern()]["VCDCMatrix"], 1, GL_FALSE,
                     glm::value_ptr(projMat));
  glUniformMatrix4fv(mShaderVectorDicts[pattern()]["MCDCMatrix"], 1, GL_FALSE,
                     glm::value_ptr(mvpMat));

  glBindVertexArray(mVAOs[pattern()]);
  glDrawArrays(GL_POINTS, 0, mNumPoints);
  glBindVertexArray(0);
  mShaders[pattern()].unuse();
}

TriMeshRenderObject::TriMeshRenderObject() {
  mVAO = 0;
  mDataVBO = 0;
  mInstIDVBO = 0;
  mEBO = 0;

  std::string pvs =
#include "generated/trimesh.vs"
  ;
  std::string pfs =
#include "generated/trimesh.fs"
  ;

  mShader.compile(pvs.c_str(), nullptr, pfs.c_str());

  mShaderVectorDict["vertexMC"] =
      glGetAttribLocation(mShader.program(), "vertexMC");
  mShaderVectorDict["normalMC"] =
      glGetAttribLocation(mShader.program(), "normalMC");
  mShaderVectorDict["primitive_id"] =
      glGetAttribLocation(mShader.program(), "primitive_id");

  mShaderVectorDict["MCVCMatrix"] =
      glGetUniformLocation(mShader.program(), "MCVCMatrix");
  mShaderVectorDict["VCDCMatrix"] =
      glGetUniformLocation(mShader.program(), "VCDCMatrix");
  mShaderVectorDict["MCDCMatrix"] =
      glGetUniformLocation(mShader.program(), "MCDCMatrix");
}
TriMeshRenderObject::~TriMeshRenderObject() {
  if (mDataVBO) glDeleteBuffers(1, &mDataVBO);
  if (mEBO) glDeleteBuffers(1, &mEBO);
  if (mInstIDVBO) glDeleteBuffers(1, &mInstIDVBO);
  if (mVAO) glDeleteVertexArrays(1, &mVAO);
}

/**
 * @brief build VAO from given points
 * @param data column:[x y z r g b radius]
 * @param num_points number of points
 */
void TriMeshRenderObject::buildVAO(const float *data, const size_t num_points,
                                   const uint32_t *indices,
                                   const size_t num_triangles) {
  mNumTriangles = num_triangles;
  mNumVertex = 3 * mNumTriangles;

  std::vector<float> dataVec;
  std::vector<uint32_t> elemVec;
  std::vector<uint32_t> InstID;
  dataVec.resize(6 * mNumVertex);
  elemVec.resize(mNumVertex);
  InstID.resize(mNumVertex);
  for (int i = 0; i < mNumTriangles; ++i) {
    const uint32_t &i0 = indices[3 * i];
    const uint32_t &i1 = indices[3 * i + 1];
    const uint32_t &i2 = indices[3 * i + 2];

    glm::vec3 p0 = glm::make_vec3(data + 3 * i0);
    glm::vec3 p1 = glm::make_vec3(data + 3 * i1);
    glm::vec3 p2 = glm::make_vec3(data + 3 * i2);

    glm::vec3 v1 = p2 - p0;
    glm::vec3 v2 = p2 - p1;
    glm::vec3 n = glm::normalize(glm::cross(v1, v2));

    memcpy(dataVec.data() + 18 * i, data + 3 * i0, sizeof(float) * 3);
    memcpy(dataVec.data() + 18 * i + 6, data + 3 * i1, sizeof(float) * 3);
    memcpy(dataVec.data() + 18 * i + 12, data + 3 * i2, sizeof(float) * 3);

    memcpy(dataVec.data() + 18 * i + 3, glm::value_ptr(n), sizeof(float) * 3);
    memcpy(dataVec.data() + 18 * i + 9, glm::value_ptr(n), sizeof(float) * 3);
    memcpy(dataVec.data() + 18 * i + 15, glm::value_ptr(n), sizeof(float) * 3);

    elemVec[3 * i] = 3 * i;
    elemVec[3 * i + 1] = 3 * i + 1;
    elemVec[3 * i + 2] = 3 * i + 2;

    InstID[3 * i] = i;
    InstID[3 * i + 1] = i;
    InstID[3 * i + 2] = i;
  }

  // Release previous
  if (mDataVBO) glDeleteBuffers(1, &mDataVBO);
  if (mEBO) glDeleteBuffers(1, &mEBO);
  if (mInstIDVBO) glDeleteBuffers(1, &mInstIDVBO);
  if (mVAO) glDeleteVertexArrays(1, &mVAO);

  // Create VBO and upload
  glGenBuffers(1, &mDataVBO);
  glGenBuffers(1, &mEBO);
  glGenBuffers(1, &mInstIDVBO);

  glBindBuffer(GL_ARRAY_BUFFER, mDataVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * mNumVertex, dataVec.data(),
               GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mEBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * mNumVertex,
               elemVec.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, mInstIDVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(uint32_t) * mNumVertex, InstID.data(),
               GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  // Create VAO
  glGenVertexArrays(1, &mVAO);
  glBindVertexArray(mVAO);
  bool useNormal = mShaderVectorDict["normalMC"] != -1;
  bool useFaceID = mShaderVectorDict["primitive_id"] != -1;

  glEnableVertexAttribArray(mShaderVectorDict["vertexMC"]);
  if (useNormal) glEnableVertexAttribArray(mShaderVectorDict["normalMC"]);
  if (useFaceID) glEnableVertexAttribArray(mShaderVectorDict["primitive_id"]);

  glBindBuffer(GL_ARRAY_BUFFER, mDataVBO);
  glVertexAttribPointer(mShaderVectorDict["vertexMC"], 3, GL_FLOAT, GL_FALSE,
                        sizeof(float) * 6, (GLvoid *) (sizeof(float) * 0));
  if (useNormal)
    glVertexAttribPointer(mShaderVectorDict["normalMC"], 3, GL_FLOAT, GL_FALSE,
                          sizeof(float) * 6, (GLvoid *) (sizeof(float) * 3));
  glBindBuffer(GL_ARRAY_BUFFER, mInstIDVBO);
  if (useFaceID)
    glVertexAttribIPointer(mShaderVectorDict["primitive_id"], 1,
                           GL_UNSIGNED_INT, sizeof(uint32_t), NULL);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mEBO);
  glBindVertexArray(0);
}
void TriMeshRenderObject::draw(const glm::mat4 &mvMat, const glm::mat4 &projMat,
                               const glm::mat4 &mvpMat) {
  mShader.use();
  glUniformMatrix4fv(mShaderVectorDict["MCVCMatrix"], 1, GL_FALSE,
                     glm::value_ptr(mvMat));
  glUniformMatrix4fv(mShaderVectorDict["VCDCMatrix"], 1, GL_FALSE,
                     glm::value_ptr(projMat));
  glUniformMatrix4fv(mShaderVectorDict["MCDCMatrix"], 1, GL_FALSE,
                     glm::value_ptr(mvpMat));

  glBindVertexArray(mVAO);
  glDrawElements(GL_TRIANGLES, mNumVertex, GL_UNSIGNED_INT, nullptr);
  glBindVertexArray(0);
  mShader.unuse();
}


RGBTriMeshRenderObject::RGBTriMeshRenderObject() {
  mVAO = 0;
  mDataVBO = 0;
  mInstIDVBO = 0;
  mEBO = 0;

  std::string pvs =
#include "generated/rgbtrimesh.vs"
  ;
  std::string pfs =
#include "generated/rgbtrimesh.fs"
  ;

  mShader.compile(pvs.c_str(), nullptr, pfs.c_str());

  mShaderVectorDict["vertexMC"] =
      glGetAttribLocation(mShader.program(), "vertexMC");
  mShaderVectorDict["vertexColor"] =
      glGetAttribLocation(mShader.program(), "vertexColor");
  mShaderVectorDict["primitive_id"] =
      glGetAttribLocation(mShader.program(), "primitive_id");

  mShaderVectorDict["MCVCMatrix"] =
      glGetUniformLocation(mShader.program(), "MCVCMatrix");
  mShaderVectorDict["VCDCMatrix"] =
      glGetUniformLocation(mShader.program(), "VCDCMatrix");
  mShaderVectorDict["MCDCMatrix"] =
      glGetUniformLocation(mShader.program(), "MCDCMatrix");
}
RGBTriMeshRenderObject::~RGBTriMeshRenderObject() {
  if (mDataVBO) glDeleteBuffers(1, &mDataVBO);
  if (mEBO) glDeleteBuffers(1, &mEBO);
  if (mInstIDVBO) glDeleteBuffers(1, &mInstIDVBO);
  if (mVAO) glDeleteVertexArrays(1, &mVAO);
}

/**
 * @brief build VAO from given points
 * @param data column:[x y z r g b radius]
 * @param num_points number of points
 */
void RGBTriMeshRenderObject::buildVAO(const float *data, const size_t num_points,
                                      const uint32_t *indices,
                                      const size_t num_triangles) {
  mNumTriangles = num_triangles;
  mNumVertex = 3 * mNumTriangles;

  std::vector<float> dataVec;
  std::vector<uint32_t> elemVec;
  std::vector<uint32_t> InstID;
  dataVec.resize(6 * mNumVertex);
  elemVec.resize(mNumVertex);
  InstID.resize(mNumVertex);
  for (int i = 0; i < mNumTriangles; ++i) {
    const uint32_t &i0 = indices[3 * i];
    const uint32_t &i1 = indices[3 * i + 1];
    const uint32_t &i2 = indices[3 * i + 2];

    memcpy(dataVec.data() + 18 * i, data + 6 * i0, sizeof(float) * 6);
    memcpy(dataVec.data() + 18 * i + 6, data + 6 * i1, sizeof(float) * 6);
    memcpy(dataVec.data() + 18 * i + 12, data + 6 * i2, sizeof(float) * 6);

    elemVec[3 * i] = 3 * i;
    elemVec[3 * i + 1] = 3 * i + 1;
    elemVec[3 * i + 2] = 3 * i + 2;

    InstID[3 * i] = i;
    InstID[3 * i + 1] = i;
    InstID[3 * i + 2] = i;
  }

  // Release previous
  if (mDataVBO) glDeleteBuffers(1, &mDataVBO);
  if (mEBO) glDeleteBuffers(1, &mEBO);
  if (mInstIDVBO) glDeleteBuffers(1, &mInstIDVBO);
  if (mVAO) glDeleteVertexArrays(1, &mVAO);

  // Create VBO and upload
  glGenBuffers(1, &mDataVBO);
  glGenBuffers(1, &mEBO);
  glGenBuffers(1, &mInstIDVBO);

  glBindBuffer(GL_ARRAY_BUFFER, mDataVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * mNumVertex, dataVec.data(),
               GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mEBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * mNumVertex,
               elemVec.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, mInstIDVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(uint32_t) * mNumVertex, InstID.data(),
               GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  // Create VAO
  glGenVertexArrays(1, &mVAO);
  glBindVertexArray(mVAO);
  bool useColor = mShaderVectorDict["vertexColor"] != -1;
  bool useFaceID = mShaderVectorDict["primitive_id"] != -1;

  glEnableVertexAttribArray(mShaderVectorDict["vertexMC"]);
  if (useColor) glEnableVertexAttribArray(mShaderVectorDict["vertexColor"]);
  if (useFaceID) glEnableVertexAttribArray(mShaderVectorDict["primitive_id"]);

  glBindBuffer(GL_ARRAY_BUFFER, mDataVBO);
  glVertexAttribPointer(mShaderVectorDict["vertexMC"], 3, GL_FLOAT, GL_FALSE,
                        sizeof(float) * 6, (GLvoid *) (sizeof(float) * 0));
  if (useColor)
    glVertexAttribPointer(mShaderVectorDict["vertexColor"], 3, GL_FLOAT, GL_FALSE,
                          sizeof(float) * 6, (GLvoid *) (sizeof(float) * 3));
  glBindBuffer(GL_ARRAY_BUFFER, mInstIDVBO);
  if (useFaceID)
    glVertexAttribIPointer(mShaderVectorDict["primitive_id"], 1,
                           GL_UNSIGNED_INT, sizeof(uint32_t), NULL);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mEBO);
  glBindVertexArray(0);
}
void RGBTriMeshRenderObject::draw(const glm::mat4 &mvMat, const glm::mat4 &projMat,
                                  const glm::mat4 &mvpMat) {
  mShader.use();
  glUniformMatrix4fv(mShaderVectorDict["MCVCMatrix"], 1, GL_FALSE,
                     glm::value_ptr(mvMat));
  glUniformMatrix4fv(mShaderVectorDict["VCDCMatrix"], 1, GL_FALSE,
                     glm::value_ptr(projMat));
  glUniformMatrix4fv(mShaderVectorDict["MCDCMatrix"], 1, GL_FALSE,
                     glm::value_ptr(mvpMat));

  glBindVertexArray(mVAO);
  glDrawElements(GL_TRIANGLES, mNumVertex, GL_UNSIGNED_INT, nullptr);
  glBindVertexArray(0);
  mShader.unuse();
}

TexturedTriMeshRenderObject::Patch::Patch(
    std::map<std::string, GLint> &shaderVectorDict)
    : mTextureID(0),
      mCreated(0),
      mDataVBO(0),
      mEBO(0),
      mInstIDVBO(0),
      mVAO(0),
      mShaderVectorDict(shaderVectorDict) {}

TexturedTriMeshRenderObject::Patch::~Patch() { clear(); }

void TexturedTriMeshRenderObject::Patch::clear() {
  if (mCreated && mTextureID) glDeleteTextures(1, &mTextureID);
  mCreated = 0;
  mTextureID = 0;
  if (mDataVBO) glDeleteBuffers(1, &mDataVBO);
  if (mEBO) glDeleteBuffers(1, &mEBO);
  if (mInstIDVBO) glDeleteBuffers(1, &mInstIDVBO);
  if (mVAO) glDeleteVertexArrays(1, &mVAO);
  mDataVBO = 0;
  mVAO = 0;
  mEBO = 0;
  mInstIDVBO = 0;
}

GLenum TexturedTriMeshRenderObject::Patch::ToTextureFormat(Format format) {
  switch (format) {
    case Format_Grayscale:return GL_LUMINANCE;
    case Format_GrayscaleAlpha:return GL_LUMINANCE_ALPHA;
    case Format_RGB:return GL_RGB;
    case Format_RGBA:return GL_RGBA;
    case Format_BGR:return GL_BGR;
    default:throw std::runtime_error("Unrecognised Bitmap::Format");
  }
}

void TexturedTriMeshRenderObject::Patch::setTextureID(GLuint id) {
  clear();
  mTextureID = id;
  mCreated = 0;
}

void TexturedTriMeshRenderObject::Patch::setTexture(const void *data, int width,
                                                    int height, Format format) {
  clear();
  GLenum glformat = ToTextureFormat(format);
  glGenTextures(1, &mTextureID);
  glBindTexture(GL_TEXTURE_2D, mTextureID);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, (GLsizei) width, (GLsizei) height, 0,
               glformat, GL_UNSIGNED_BYTE, data);
  glBindTexture(GL_TEXTURE_2D, 0);

  GLenum error = glGetError();

  if (error != GL_NO_ERROR) {
    char msg[50];
    sprintf(msg, "OpenGL Error %d", error);
    throw std::runtime_error(msg);
  }
  mCreated = 1;
}

void TexturedTriMeshRenderObject::Patch::buildVAO(const float *geo_coordinates,
                                                  const float *tex_coordinates,
                                                  const size_t num_points,
                                                  const uint32_t *geo_indices,
                                                  const uint32_t *tex_indices,
                                                  const size_t num_triangles,
                                                  const uint32_t id_offset) {
  mNumTriangles = num_triangles;
  mNumVertex = 3 * mNumTriangles;

  std::vector<float> dataVec;
  std::vector<uint32_t> elemVec;
  std::vector<uint32_t> InstID;
  dataVec.resize(6 * mNumVertex);
  elemVec.resize(mNumVertex);
  InstID.resize(mNumVertex);

  for (int i = 0; i < mNumTriangles; ++i) {
    memcpy(dataVec.data() + 15 * i, geo_coordinates + 3 * geo_indices[3 * i],
           sizeof(float) * 3);
    memcpy(dataVec.data() + 15 * i + 5,
           geo_coordinates + 3 * geo_indices[3 * i + 1], sizeof(float) * 3);
    memcpy(dataVec.data() + 15 * i + 10,
           geo_coordinates + 3 * geo_indices[3 * i + 2], sizeof(float) * 3);

    memcpy(dataVec.data() + 15 * i + 3,
           tex_coordinates + 2 * tex_indices[3 * i], sizeof(float) * 2);
    memcpy(dataVec.data() + 15 * i + 8,
           tex_coordinates + 2 * tex_indices[3 * i + 1], sizeof(float) * 2);
    memcpy(dataVec.data() + 15 * i + 13,
           tex_coordinates + 2 * tex_indices[3 * i + 2], sizeof(float) * 2);

    elemVec[3 * i] = 3 * i;
    elemVec[3 * i + 1] = 3 * i + 1;
    elemVec[3 * i + 2] = 3 * i + 2;

    InstID[3 * i] = i + id_offset;
    InstID[3 * i + 1] = i + id_offset;
    InstID[3 * i + 2] = i + id_offset;
  }
  // Release previous
  if (mDataVBO) glDeleteBuffers(1, &mDataVBO);
  if (mEBO) glDeleteBuffers(1, &mEBO);
  if (mInstIDVBO) glDeleteBuffers(1, &mInstIDVBO);
  if (mVAO) glDeleteVertexArrays(1, &mVAO);

  // Create VBO and upload
  glGenBuffers(1, &mDataVBO);
  glGenBuffers(1, &mEBO);
  glGenBuffers(1, &mInstIDVBO);
  glBindBuffer(GL_ARRAY_BUFFER, mDataVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 5 * mNumVertex, dataVec.data(),
               GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mEBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * mNumVertex,
               elemVec.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, mInstIDVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(uint32_t) * mNumVertex, InstID.data(),
               GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Create VAO
  glGenVertexArrays(1, &mVAO);
  glBindVertexArray(mVAO);
  bool useTC = mShaderVectorDict["vertexTC"] != -1;
  bool useFaceID = mShaderVectorDict["primitive_id"] != -1;

  glEnableVertexAttribArray(mShaderVectorDict["vertexMC"]);
  if (useTC) glEnableVertexAttribArray(mShaderVectorDict["vertexTC"]);
  if (useFaceID) glEnableVertexAttribArray(mShaderVectorDict["primitive_id"]);

  glBindBuffer(GL_ARRAY_BUFFER, mDataVBO);
  glVertexAttribPointer(mShaderVectorDict["vertexMC"], 3, GL_FLOAT, GL_FALSE,
                        sizeof(float) * 5, (GLvoid *) (sizeof(float) * 0));
  if (useTC)
    glVertexAttribPointer(mShaderVectorDict["vertexTC"], 2, GL_FLOAT, GL_FALSE,
                          sizeof(float) * 5, (GLvoid *) (sizeof(float) * 3));
  glBindBuffer(GL_ARRAY_BUFFER, mInstIDVBO);
  if (useFaceID)
    glVertexAttribIPointer(mShaderVectorDict["primitive_id"], 1,
                           GL_UNSIGNED_INT, sizeof(uint32_t), NULL);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mEBO);
  glBindVertexArray(0);

  GLenum error = glGetError();

  if (error != GL_NO_ERROR) {
    char msg[50];
    sprintf(msg, "OpenGL Error %d", error);
    throw std::runtime_error(msg);
  }
}

void TexturedTriMeshRenderObject::Patch::draw() {
  glActiveTexture(GL_TEXTURE0);
  glUniform1i(mShaderVectorDict["texImage"], 0);
  glBindTexture(GL_TEXTURE_2D, mTextureID);

  glBindVertexArray(mVAO);
  glDrawElements(GL_TRIANGLES, mNumVertex, GL_UNSIGNED_INT, nullptr);
  glBindVertexArray(0);

  glBindTexture(GL_TEXTURE_2D, 0);
}

TexturedTriMeshRenderObject::TexturedTriMeshRenderObject() {
  std::string pvs =
#include "generated/texturedtrimesh.vs"
  ;
  std::string pfs =
#include "generated/texturedtrimesh.fs"
  ;

  mShader.compile(pvs.c_str(), nullptr, pfs.c_str());

  mShaderVectorDict["vertexMC"] =
      glGetAttribLocation(mShader.program(), "vertexMC");
  mShaderVectorDict["vertexTC"] =
      glGetAttribLocation(mShader.program(), "vertexTC");
  mShaderVectorDict["primitive_id"] =
      glGetAttribLocation(mShader.program(), "primitive_id");
  mShaderVectorDict["texImage"] =
      glGetUniformLocation(mShader.program(), "texImage");

  mShaderVectorDict["MCVCMatrix"] =
      glGetUniformLocation(mShader.program(), "MCVCMatrix");
  mShaderVectorDict["VCDCMatrix"] =
      glGetUniformLocation(mShader.program(), "VCDCMatrix");
  mShaderVectorDict["MCDCMatrix"] =
      glGetUniformLocation(mShader.program(), "MCDCMatrix");
}

TexturedTriMeshRenderObject::~TexturedTriMeshRenderObject() {
  for (auto p : mPatches) delete p;
  mPatches.clear();
}

TexturedTriMeshRenderObject::Patch *TexturedTriMeshRenderObject::CreatePatch() {
  mPatches.emplace_back(new Patch(mShaderVectorDict));
  return mPatches.back();
}

void TexturedTriMeshRenderObject::draw(const glm::mat4 &mvMat,
                                       const glm::mat4 &projMat,
                                       const glm::mat4 &mvpMat) {
  mShader.use();
  glUniformMatrix4fv(mShaderVectorDict["MCVCMatrix"], 1, GL_FALSE,
                     glm::value_ptr(mvMat));
  glUniformMatrix4fv(mShaderVectorDict["VCDCMatrix"], 1, GL_FALSE,
                     glm::value_ptr(projMat));
  glUniformMatrix4fv(mShaderVectorDict["MCDCMatrix"], 1, GL_FALSE,
                     glm::value_ptr(mvpMat));

  for (auto it = mPatches.begin(); it != mPatches.end(); ++it) {
    (*it)->draw();
  }

  mShader.unuse();
}

WWZBFilterObject::WWZBFilterObject() {

  std::string pcs =
#include "generated/wwzb.cs"
  ;
  mShader.compile(pcs.c_str());

  mShaderVectorDict["inputColor"] =
      glGetUniformLocation(mShader.program(), "inputColor");
  mShaderVectorDict["inputDepth"] =
      glGetUniformLocation(mShader.program(), "inputDepth");
  mShaderVectorDict["inputID"] =
      glGetUniformLocation(mShader.program(), "inputID");

  mShaderVectorDict["outputColor"] =
      glGetUniformLocation(mShader.program(), "outputColor");
  mShaderVectorDict["outputDepth"] =
      glGetUniformLocation(mShader.program(), "outputDepth");
  mShaderVectorDict["outputID"] =
      glGetUniformLocation(mShader.program(), "outputID");

  mShaderVectorDict["kernel_size"] =
      glGetUniformLocation(mShader.program(), "kernel_size");
  mShaderVectorDict["focal"] = glGetUniformLocation(mShader.program(), "focal");
  mShaderVectorDict["angle_thres"] =
      glGetUniformLocation(mShader.program(), "angle_thres");

  mShaderVectorDict["invalid_index"] =
      glGetUniformLocation(mShader.program(), "invalid_index");
  mShaderVectorDict["invalid_depth"] =
      glGetUniformLocation(mShader.program(), "invalid_depth");
}

WWZBFilterObject::~WWZBFilterObject() {}

void WWZBFilterObject::filter(const GLuint inColor, const GLuint inDepth,
                              const GLuint inID, const GLuint outColor,
                              const GLuint outDepth, const GLuint outID,
                              const float invalid_depth,
                              const uint32_t invalid_index, const float focal,
                              const uint32_t kernelSize,
                              const float angleTolerance) {
  glBindTexture(GL_TEXTURE_2D, inColor);
  GLint width = 0;
  GLint height = 0;
  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
  glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
  glBindTexture(GL_TEXTURE_2D, 0);

  mShader.use();
  glBindImageTexture(mShaderVectorDict["inputColor"], inColor, 0, GL_FALSE, 0,
                     GL_READ_ONLY, GL_RGBA8);
  glBindImageTexture(mShaderVectorDict["inputDepth"], inDepth, 0, GL_FALSE, 0,
                     GL_READ_WRITE, GL_R32F);
  glBindImageTexture(mShaderVectorDict["inputID"], inID, 0, GL_FALSE, 0,
                     GL_READ_WRITE, GL_R32UI);

  glBindImageTexture(mShaderVectorDict["outputColor"], outColor, 0, GL_FALSE, 0,
                     GL_READ_WRITE, GL_RGBA8);
  glBindImageTexture(mShaderVectorDict["outputDepth"], outDepth, 0, GL_FALSE, 0,
                     GL_READ_WRITE, GL_R32F);
  glBindImageTexture(mShaderVectorDict["outputID"], outID, 0, GL_FALSE, 0,
                     GL_READ_WRITE, GL_R32UI);

  glUniform1i(mShaderVectorDict["kernel_size"], kernelSize);
  glUniform1f(mShaderVectorDict["focal"], focal);
  glUniform1f(mShaderVectorDict["angle_thres"], angleTolerance);

  glUniform1f(mShaderVectorDict["invalid_depth"], invalid_depth);
  glUniform1ui(mShaderVectorDict["invalid_index"], invalid_index);

  glDispatchCompute(width, height, 1);
  glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
  glFinish();
  mShader.unuse();
}

}  // namespace FBORender
