#ifndef ENV_SETUP_H
#define ENV_SETUP_H
#include "FBORender/FBORender.h"
#include <memory>

static std::shared_ptr<FBORender::MultidrawFBO> fbo;

inline bool setup() {
  if (!fbo) {
    fbo.reset(new FBORender::MultidrawFBO);
  }
  FBORender::initGLContext();
  FBORender::GLMakeContextCurrent();
  return true;
}
#endif  // ENV_SETUP_H
