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

#ifndef GLOFFSCREENRENDER_SRC_MULTIDRAWFBO_H
#define GLOFFSCREENRENDER_SRC_MULTIDRAWFBO_H
#include "dllmacro.h"
#include "util.h"
namespace FBORender {
class DLL_API MultidrawFBO {
 public:
  enum { CHRGB = 0, CHZ = 1, CHID = 2, CHDEPTH = 3, NUM_RBOS = 4 };

  MultidrawFBO();
  ~MultidrawFBO();
  inline GLuint fbo() { return mFBO; }
  inline int width() { return mWidth; }
  inline int height() { return mHeight; }

  bool resize(int width, int height);

  GLuint getTexID(int target, bool flt);

  bool getTexImage(int target, bool flt, void* data);
  bool setTexImage(int target, bool flt, const void* data);

 protected:
  GLuint createRGB8UTex(int width, int height);
  GLuint createR32FTex(int width, int height);
  GLuint createR32UITex(int width, int height);

  int mWidth, mHeight;
  GLuint mFBO;
  GLuint mRBOs[NUM_RBOS];
  GLuint mTexs[NUM_RBOS];
  GLuint mFltTexs[NUM_RBOS];
};
}  // namespace FBORender
#endif  // GLOFFSCREENRENDER_SRC_MULTIDRAWFBO_H
