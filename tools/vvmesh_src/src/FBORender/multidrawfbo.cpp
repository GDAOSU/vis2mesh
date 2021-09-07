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

#include "multidrawfbo.h"
#include <cstdio>
#include <vector>
namespace FBORender {
MultidrawFBO::MultidrawFBO() {
  mFBO = 0;
  for (int i = 0; i < NUM_RBOS; ++i) {
    mRBOs[i] = UINT32_MAX;
    mTexs[i] = UINT32_MAX;
  }

  mWidth = -1;
  mHeight = -1;
}
MultidrawFBO::~MultidrawFBO() { resize(-1, -1); }

GLuint MultidrawFBO::createRGB8UTex(int width, int height) {
  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);
  return tex;
}

GLuint MultidrawFBO::createR32FTex(int width, int height) {
  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, width, height);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);
  return tex;
}

GLuint MultidrawFBO::createR32UITex(int width, int height) {
  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, width, height);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);
  return tex;
}

bool MultidrawFBO::resize(int width, int height) {
  if (width == mWidth && height == mHeight) {
    return true;
  }
  if (mWidth != -1 && mHeight != -1) {
    for (int i = 0; i < NUM_RBOS; ++i) {
      if (mRBOs[i] != UINT32_MAX) glDeleteRenderbuffers(1, &mRBOs[i]);
      if (mTexs[i] != UINT32_MAX) glDeleteTextures(1, &mTexs[i]);
      if (mFltTexs[i] != UINT32_MAX) glDeleteTextures(1, &mFltTexs[i]);
    }
    glDeleteFramebuffers(1, &mFBO);
    mWidth = -1;
    mHeight = -1;
    mFBO = 0;
    for (int i = 0; i < NUM_RBOS; ++i) mRBOs[i] = 0;
  }

  if (width == -1 || height == -1) {
    return true;
  }

  glGenFramebuffers(1, &mFBO);
  glGenRenderbuffers(NUM_RBOS, mRBOs);

  glBindFramebuffer(GL_FRAMEBUFFER, mFBO);

  mTexs[CHRGB] = createRGB8UTex(width, height);
  mFltTexs[CHRGB] = createRGB8UTex(width, height);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         mTexs[CHRGB], 0);

  mTexs[CHZ] = createR32FTex(width, height);
  mFltTexs[CHZ] = createR32FTex(width, height);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D,
                         mTexs[CHZ], 0);

  mTexs[CHID] = createR32UITex(width, height);
  mFltTexs[CHID] = createR32UITex(width, height);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D,
                         mTexs[CHID], 0);

  glBindRenderbuffer(GL_RENDERBUFFER, mRBOs[CHDEPTH]);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
  glBindRenderbuffer(GL_RENDERBUFFER, 0);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                            GL_RENDERBUFFER, mRBOs[CHDEPTH]);

  GLint status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    fprintf(stderr, "==[FRAMEBUFFER ERROR][S:%d]== %d x %d\n", status, width,
            height);
    glDeleteFramebuffers(1, &mFBO);
    for (int i = 0; i < NUM_RBOS; ++i) {
      if (mRBOs[i] != UINT32_MAX) glDeleteRenderbuffers(1, &mRBOs[i]);
      if (mTexs[i] != UINT32_MAX) glDeleteTextures(1, &mTexs[i]);
      if (mFltTexs[i] != UINT32_MAX) glDeleteTextures(1, &mFltTexs[i]);
    }

    return false;
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glViewport(0, 0, width, height);
  mWidth = width;
  mHeight = height;
  return true;
}

GLuint MultidrawFBO::getTexID(int target, bool flt) {
  int tex = 0;
  if (flt) {
    tex = mFltTexs[target];
  } else {
    tex = mTexs[target];
  }
  return tex;
}
bool MultidrawFBO::getTexImage(int target, bool flt, void *data) {
  int tex = 0;
  if (flt) {
    tex = mFltTexs[target];
  } else {
    tex = mTexs[target];
  }

  glBindTexture(GL_TEXTURE_2D, tex);
  switch (target) {
    case CHRGB: {
      glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE, data);
      flipY(mWidth, mHeight, GL_BGR, GL_UNSIGNED_BYTE, data);
      break;
    }
    case CHZ: {
      glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, data);
      flipY(mWidth, mHeight, GL_RED, GL_FLOAT, data);
      break;
    }
    case CHID: {
      glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, data);
      flipY(mWidth, mHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, data);
      break;
    }
  }
  glBindTexture(GL_TEXTURE_2D, 0);
  return true;
}

bool MultidrawFBO::setTexImage(int target, bool flt, const void *data) {
  int tex = 0;
  if (flt) {
    tex = mFltTexs[target];
  } else {
    tex = mTexs[target];
  }

  glBindTexture(GL_TEXTURE_2D, tex);
  std::vector<uint8_t> _buffer;

  switch (target) {
    case CHRGB: {
      _buffer.resize(3 * mWidth * mHeight);
      memcpy(_buffer.data(), data, _buffer.size());
      flipY(mWidth, mHeight, GL_BGR, GL_UNSIGNED_BYTE, _buffer.data());
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_BGR,
                      GL_UNSIGNED_BYTE, _buffer.data());
      break;
    }
    case CHZ: {
      _buffer.resize(sizeof(float) * mWidth * mHeight);
      memcpy(_buffer.data(), data, _buffer.size());
      flipY(mWidth, mHeight, GL_RED, GL_FLOAT, _buffer.data());
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RED, GL_FLOAT,
                      _buffer.data());
      break;
    }
    case CHID: {
      _buffer.resize(sizeof(uint32_t) * mWidth * mHeight);
      memcpy(_buffer.data(), data, _buffer.size());
      flipY(mWidth, mHeight, GL_RED_INTEGER, GL_UNSIGNED_INT, _buffer.data());
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RED_INTEGER,
                      GL_UNSIGNED_INT, _buffer.data());
      break;
    }
  }
  glBindTexture(GL_TEXTURE_2D, 0);

  return true;
}

}  // namespace FBORender
