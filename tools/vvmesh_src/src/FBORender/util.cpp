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

#include "util.h"

#include <cstring>
#include <fstream>
#include <iostream>

#include <EGL/egl.h>

namespace FBORender {
using namespace std;

#ifdef __linux__
#include <libgen.h>
#include <unistd.h>
std::string _GetModuleDirPath() {
  char result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  const char *path;
  if (count != -1) {
    path = dirname(result);
  }
  return path;
}
#endif

std::string load_text(std::string path) {
  std::ifstream ifs(path);
  if (!ifs.good()) {
    std::cerr << "Error:" << path << " not found." << std::endl;
    throw std::runtime_error("File not found");
    return "";
  }
  ifs.seekg(0, std::ios::end);
  std::string str;
  str.reserve(ifs.tellg());
  ifs.seekg(0, std::ios::beg);

  str.assign((std::istreambuf_iterator<char>(ifs)),
             std::istreambuf_iterator<char>());
  return str;
}

class EGLContextHolder {
 public:
  EGLContextHolder() { isValid = false; }
  ~EGLContextHolder() {
    if (isValid) {
      eglTerminate(eglDpy);
    }
  }

  bool requestContext() {
    if (isValid) return true;
    // 1. Initialize EGL
    eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);

    EGLint major, minor;
    eglInitialize(eglDpy, &major, &minor);
    // 2. Select an appropriate configuration
    EGLint numConfigs;
    EGLConfig eglCfg;
    static const EGLint configAttribs[] = {EGL_SURFACE_TYPE,
                                           EGL_PBUFFER_BIT,
                                           EGL_BLUE_SIZE,
                                           8,
                                           EGL_GREEN_SIZE,
                                           8,
                                           EGL_RED_SIZE,
                                           8,
                                           EGL_DEPTH_SIZE,
                                           8,
                                           EGL_RENDERABLE_TYPE,
                                           EGL_OPENGL_BIT,
                                           EGL_NONE};
    // static const int pbufferWidth = 9;
    // static const int pbufferHeight = 9;
    // static const EGLint pbufferAttribs[] = {
    //     EGL_WIDTH, pbufferWidth, EGL_HEIGHT, pbufferHeight, EGL_NONE,
    // };

    eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);
    // 3. Create a surface
    // eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, pbufferAttribs);
    // 4. Bind the API
    eglBindAPI(EGL_OPENGL_API);
    // 5. Create a context and make it current
    eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, NULL);
    isValid = true;
    return true;
  }
  bool makeCurrent() {
    if (!isValid) requestContext();
    if (!isValid) return false;
    if (eglMakeCurrent(eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, eglCtx) ==
        EGL_FALSE ||
        eglGetError() != EGL_SUCCESS) {
      fprintf(stderr, "Error during eglMakeCurrent.\n");
      return false;
    }
    return true;
  }

  EGLDisplay eglDpy;
  // EGLSurface eglSurf;
  EGLContext eglCtx;
  int isValid;
};

static EGLContextHolder eglContextHolder;
int initGLContext() {
  if (!eglContextHolder.requestContext()) return false;
  if (!eglContextHolder.makeCurrent()) return false;
  glewInit();
  return true;
}

int GLMakeContextCurrent() {
  if (!eglContextHolder.makeCurrent()) {
    return false;
  }
  return true;
}

void flipY(GLsizei width, GLsizei height, GLenum format, GLenum type,
           void *data) {
  size_t elemPerPixel = 0;
  switch (format) {
    case GL_STENCIL_INDEX:
    case GL_DEPTH_COMPONENT:
    case GL_DEPTH_STENCIL:
    case GL_RED:
    case GL_RED_INTEGER:
    case GL_GREEN:
    case GL_GREEN_INTEGER:
    case GL_BLUE:
    case GL_BLUE_INTEGER: {
      elemPerPixel = 1;
      break;
    }
    case GL_RGB:
    case GL_BGR: {
      elemPerPixel = 3;
      break;
    }
    case GL_RGBA:
    case GL_BGRA: {
      elemPerPixel = 4;
      break;
    }
  }

  size_t bytePerPixel = 0;
  switch (type) {
    case GL_UNSIGNED_BYTE:
    case GL_BYTE:
    case GL_UNSIGNED_BYTE_3_3_2:
    case GL_UNSIGNED_BYTE_2_3_3_REV: {
      bytePerPixel = 1;
      break;
    }
    case GL_UNSIGNED_SHORT:
    case GL_UNSIGNED_SHORT_5_6_5:
    case GL_UNSIGNED_SHORT_5_6_5_REV:
    case GL_UNSIGNED_SHORT_4_4_4_4:
    case GL_UNSIGNED_SHORT_4_4_4_4_REV:
    case GL_UNSIGNED_SHORT_1_5_5_5_REV:
    case GL_SHORT:
    case GL_HALF_FLOAT: {
      bytePerPixel = 2;
      break;
    }
    case GL_UNSIGNED_INT:
    case GL_UNSIGNED_INT_8_8_8_8:
    case GL_UNSIGNED_INT_8_8_8_8_REV:
    case GL_UNSIGNED_INT_10_10_10_2:
    case GL_UNSIGNED_INT_2_10_10_10_REV:
    case GL_UNSIGNED_INT_24_8:
    case GL_UNSIGNED_INT_10F_11F_11F_REV:
    case GL_UNSIGNED_INT_5_9_9_9_REV:
    case GL_INT:
    case GL_FLOAT:
    case GL_FLOAT_32_UNSIGNED_INT_24_8_REV: {
      bytePerPixel = 4;
      break;
    }
  }
  size_t bytePerScan = elemPerPixel * bytePerPixel * width;

  int numSwap = height / 2;
  char *swapTmp = new char[bytePerScan];
  for (int i = 0; i < numSwap; ++i) {
    char *line0 = (char *) data + i * bytePerScan;
    char *line1 = (char *) data + (height - 1 - i) * bytePerScan;
    memcpy(swapTmp, line0, bytePerScan);
    memcpy(line0, line1, bytePerScan);
    memcpy(line1, swapTmp, bytePerScan);
  }
  delete swapTmp;
}

void glReadPixelsFlipY(GLint x, GLint y, GLsizei width, GLsizei height,
                       GLenum format, GLenum type, void *data) {
  glReadPixels(x, y, width, height, format, type, data);
  flipY(width, height, format, type, data);
}

void cvMatrix2glMatrix(const Eigen::Matrix<double, 3, 3> &R_comp,
                       const Eigen::Matrix<double, 3, 1> &C_comp,
                       const Eigen::Matrix<double, 3, 3> &K_comp,
                       const int imageWidth, const int imageHeight,
                       glm::mat4 &mvMat, glm::mat4 &projMat, double _near,
                       double _far) {
  Eigen::Matrix<double, 4, 4> glK, glProjFromPixelSpace;

  glK << K_comp(0, 0), K_comp(0, 1), K_comp(0, 2), 0, K_comp(1, 0),
      K_comp(1, 1), K_comp(1, 2), 0, K_comp(2, 0), K_comp(2, 1), K_comp(2, 2),
      0, 0, 0, 0, 1;

  double l = 0;
  double r = imageWidth;
  double t = 0;
  double b = -imageHeight;
  double n = _near;
  double f = _far;

  glProjFromPixelSpace << 2. / (r - l), 0, -(r + l) / (r - l), 0, 0,
      -2. / (t - b), -(t + b) / (t - b), 0, 0, 0, -(f + n) / (n - f),
      -2. * f * n / (f - n), 0, 0, 1., 0;

  Eigen::Matrix4d glProj = glProjFromPixelSpace * glK;

  auto tmp = -R_comp * C_comp;
  mvMat[0][0] = R_comp(0, 0);
  mvMat[0][1] = R_comp(0, 1);
  mvMat[0][2] = R_comp(0, 2);
  mvMat[1][0] = R_comp(1, 0);
  mvMat[1][1] = R_comp(1, 1);
  mvMat[1][2] = R_comp(1, 2);
  mvMat[2][0] = R_comp(2, 0);
  mvMat[2][1] = R_comp(2, 1);
  mvMat[2][2] = R_comp(2, 2);
  mvMat[3][0] = 0;
  mvMat[3][1] = 0;
  mvMat[3][2] = 0;

  mvMat[0][3] = tmp(0, 0);
  mvMat[1][3] = tmp(1, 0);
  mvMat[2][3] = tmp(2, 0);
  mvMat[3][3] = 1;

  mvMat = glm::transpose(mvMat);

  projMat[0][0] = glProj(0, 0);
  projMat[0][1] = glProj(0, 1);
  projMat[0][2] = glProj(0, 2);
  projMat[0][3] = glProj(0, 3);
  projMat[1][0] = glProj(1, 0);
  projMat[1][1] = glProj(1, 1);
  projMat[1][2] = glProj(1, 2);
  projMat[1][3] = glProj(1, 3);
  projMat[2][0] = glProj(2, 0);
  projMat[2][1] = glProj(2, 1);
  projMat[2][2] = glProj(2, 2);
  projMat[2][3] = glProj(2, 3);
  projMat[3][0] = glProj(3, 0);
  projMat[3][1] = glProj(3, 1);
  projMat[3][2] = glProj(3, 2);
  projMat[3][3] = glProj(3, 3);

  projMat = glm::transpose(projMat);

  return;
}
}  // namespace FBORender
