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
#ifndef GLOFFSCREENRENDER_UTIL_H
#define GLOFFSCREENRENDER_UTIL_H
#include "dllmacro.h"
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <Eigen/Core>
#include <string>
namespace FBORender {
DLL_API std::string _GetModuleDirPath();

DLL_API std::string load_text(std::string path);

DLL_API int initGLContext();

DLL_API int GLMakeContextCurrent();

DLL_API void flipY(GLsizei width, GLsizei height, GLenum format, GLenum type,
           void *data);

DLL_API void glReadPixelsFlipY(GLint x, GLint y, GLsizei width, GLsizei height,
                       GLenum format, GLenum type, void *data);

DLL_API void cvMatrix2glMatrix(const Eigen::Matrix<double, 3, 3> &R_comp,
                       const Eigen::Matrix<double, 3, 1> &C_comp,
                       const Eigen::Matrix<double, 3, 3> &K_comp,
                       const int imageWidth, const int imageHeight,
                       glm::mat4 &mvMat, glm::mat4 &projMat, double near = 1,
                       double far = 500);
}  // namespace FBORender

#endif  // GLOFFSCREENRENDER_UTIL_H
