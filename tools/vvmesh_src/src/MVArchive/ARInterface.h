// Copyright 2019 Shaun Song <sxsong1207@qq.com>
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
//
// This code is deep modified from: https://github.com/cdcseacave/openMVS/blob/master/libs/MVS/Interface.h

#ifndef _ARINTERFACE_MVS_H_
#define _ARINTERFACE_MVS_H_

// I N C L U D E S /////////////////////////////////////////////////
#include "dllmacro.h"
#include <vector>
#include <fstream>
#include <memory>
#include <cstring>
// D E F I N E S ///////////////////////////////////////////////////

#define MVSI_PROJECT_ID "MVSI"  // identifies the project stream
#define MVSI_PROJECT_VER \
  ((uint32_t)3)  // identifies the version of a project stream

// set a default namespace name if none given
#ifndef _INTERFACE_NAMESPACE
#define _INTERFACE_NAMESPACE MVSA
#endif

//// uncomment to enable custom OpenCV data types
//// (should be uncommented if OpenCV is not available)
#if !defined(_USE_OPENCV) && !defined(_USE_CUSTOM_CV)
#define _USE_CUSTOM_CV
#else
#include <opencv2/core.hpp>
#endif // !defined(_USE_OPENCV) && !defined(_USE_CUSTOM_CV)

#ifndef NO_ID
#define NO_ID (std::numeric_limits<uint32_t>::max)()
#endif

// S T R U C T S ///////////////////////////////////////////////////

#ifdef _USE_CUSTOM_CV
#include <Eigen/Core>
namespace cv {

// simple cv::Point3_
template<typename Type>
class Point3_ {
 public:
  typedef Type value_type;

  inline Point3_() {}
  inline Point3_(Type _x, Type _y, Type _z) : x(_x), y(_y), z(_z) {}
#ifdef _USE_EIGEN
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF_VECTORIZABLE_FIXED_SIZE(Type, 3)
  typedef Eigen::Matrix<Type, 3, 1> EVec;
  typedef Eigen::Map<EVec> EVecMap;
  template <typename Derived>
  inline Point3_(const Eigen::EigenBase<Derived>& rhs) {
    operator EVecMap() = rhs;
  }
  template <typename Derived>
  inline Point3_& operator=(const Eigen::EigenBase<Derived>& rhs) {
    operator EVecMap() = rhs;
    return *this;
  }
  inline operator const EVecMap() const { return EVecMap((Type*)this); }
  inline operator EVecMap() { return EVecMap((Type*)this); }
#endif

  Type operator()(int r) const { return (&x)[r]; }
  Type &operator()(int r) { return (&x)[r]; }
  Point3_ operator+(const Point3_ &X) const {
    return Point3_(x + X.x, y + X.y, z + X.z);
  }
  Point3_ operator-(const Point3_ &X) const {
    return Point3_(x - X.x, y - X.y, z - X.z);
  }

 public:
  Type x, y, z;
};

// simple cv::Point_
template<typename Type>
class Point_ {
 public:
  typedef Type value_type;

  inline Point_() {}
  inline Point_(Type _x, Type _y) : x(_x), y(_y) {}
#ifdef _USE_EIGEN
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF_VECTORIZABLE_FIXED_SIZE(Type, 2)
  typedef Eigen::Matrix<Type, 2, 1> EVec;
  typedef Eigen::Map<EVec> EVecMap;
  template <typename Derived>
  inline Point_(const Eigen::EigenBase<Derived>& rhs) {
    operator EVecMap() = rhs;
  }
  template <typename Derived>
  inline Point_& operator=(const Eigen::EigenBase<Derived>& rhs) {
    operator EVecMap() = rhs;
    return *this;
  }
  inline operator const EVecMap() const { return EVecMap((Type*)this); }
  inline operator EVecMap() { return EVecMap((Type*)this); }
#endif

  Type operator()(int r) const { return (&x)[r]; }
  Type &operator()(int r) { return (&x)[r]; }
  Point_ operator+(const Point_ &X) const { return Point_(x + X.x, y + X.y); }
  Point_ operator-(const Point_ &X) const { return Point_(x - X.x, y - X.y); }

 public:
  Type x, y;
};

// simple cv::Matx
template<typename Type, int m, int n>
class Matx {
 public:
  typedef Type value_type;
  enum { rows = m, cols = n, channels = rows * cols };

  inline Matx() {}
#ifdef _USE_EIGEN
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF_VECTORIZABLE_FIXED_SIZE(Type, m* n)
  typedef Eigen::Matrix<Type, m, n, (n > 1 ? Eigen::RowMajor : Eigen::Default)>
      EMat;
  typedef Eigen::Map<const EMat> CEMatMap;
  typedef Eigen::Map<EMat> EMatMap;
  template <typename Derived>
  inline Matx(const Eigen::EigenBase<Derived>& rhs) {
    operator EMatMap() = rhs;
  }
  template <typename Derived>
  inline Matx& operator=(const Eigen::EigenBase<Derived>& rhs) {
    operator EMatMap() = rhs;
    return *this;
  }
  inline operator CEMatMap() const { return CEMatMap((const Type*)val); }
  inline operator EMatMap() { return EMatMap((Type*)val); }
#endif

  Type operator()(int r, int c) const { return val[r * n + c]; }
  Type &operator()(int r, int c) { return val[r * n + c]; }
  Point3_<Type> operator*(const Point3_<Type> &X) const {
    Point3_<Type> R;
    for (int r = 0; r < m; r++) {
      R(r) = Type(0);
      for (int c = 0; c < n; c++) R(r) += operator()(r, c) * X(c);
    }
    return R;
  }
  template<int k>
  Matx<Type, m, k> operator*(const Matx<Type, n, k> &M) const {
    Matx<Type, m, k> R;
    for (int r = 0; r < m; r++) {
      for (int l = 0; l < k; l++) {
        R(r, l) = Type(0);
        for (int c = 0; c < n; c++) R(r, l) += operator()(r, c) * M(c, l);
      }
    }
    return R;
  }
  Matx<Type, n, m> t() const {
    Matx<Type, n, m> M;
    for (int r = 0; r < m; r++)
      for (int c = 0; c < n; c++) M(c, r) = operator()(r, c);
    return M;
  }

  static Matx eye() {
    Matx M;
    memset(M.val, 0, sizeof(Type) * m * n);
    const int shortdim(m < n ? m : n);
    for (int i = 0; i < shortdim; i++) M(i, i) = 1;
    return M;
  }

 public:
  Type val[m * n];
};

}  // namespace cv
#endif
/*----------------------------------------------------------------*/

namespace _INTERFACE_NAMESPACE {

// custom serialization
namespace MVArchive {

// Basic serialization types
struct DLL_API ArchiveSave {
  std::shared_ptr<std::ostream> stream;
  uint32_t version;
  ArchiveSave(std::shared_ptr<std::ostream> _stream, uint32_t _version)
      : stream(_stream), version(_version) {}
  template<typename _Tp>
  ArchiveSave &operator&(const _Tp &obj);
};
struct DLL_API ArchiveLoad {
  std::shared_ptr<std::istream> stream;
  uint32_t version;
  ArchiveLoad(std::shared_ptr<std::istream> _stream, uint32_t _version)
      : stream(_stream), version(_version) {}
  template<typename _Tp>
  ArchiveLoad &operator&(_Tp &obj);
};

// Main exporter & importer
template<typename _Tp>
bool SerializeSave(const _Tp &obj, const std::string &fileName,
                   int format = 0 /*ArchiveFormat::STDIO*/,
                   uint32_t version = MVSI_PROJECT_VER);
template<typename _Tp>
bool SerializeLoad(_Tp &obj, const std::string &fileName,
                   int *pFormat = nullptr, uint32_t *pVersion = nullptr);

}  // namespace MVArchive
/*----------------------------------------------------------------*/


enum ArchiveFormat {
  STDIO = 0
#ifdef _USE_GZSTREAM
  , GZSTREAM = 1
#endif // _USE_GZSTREAM
#ifdef _USE_ZSTDSTREAM
  , ZSTDSTREAM = 2
#endif // _USE_ZSTDSTREAM
#ifdef _USE_COMPRESSED_STREAMS
  , BROTLI=11, LZ4=12, LZMA=13, ZLIB=14, ZSTD=15
#endif // _USE_COMPRESSED_STREAMS
};

// interface used to export/import MVS input data;
//  - MAX(width,height) is used for normalization
//  - row-major order is used for storing the matrices
struct DLL_API Interface {
  int format;
  std::string filePath;
  typedef cv::Point3_<float> Pos3f;
  typedef cv::Point3_<double> Pos3d;
  typedef cv::Matx<double, 3, 3> Mat33d;
  typedef cv::Matx<double, 4, 4> Mat44d;
  typedef cv::Point3_<uint8_t> Col3;  // x=B, y=G, z=R
  /*----------------------------------------------------------------*/

  // structure describing a mobile platform with cameras attached to it
  struct Platform {
    // structure describing a camera mounted on a platform
    struct Camera {
      std::string name;        // camera's name
      uint32_t width, height;  // image resolution in pixels for all images sharing this camera (optional)
      Mat33d K;  // camera's intrinsics matrix (normalized if image resolution not specified)
      Mat33d R;  // camera's rotation matrix relative to the platform
      Pos3d C;   // camera's translation vector relative to the platform

      Camera() : width(0), height(0) {}
      bool HasResolution() const { return width > 0 && height > 0; }
      bool IsNormalized() const { return !HasResolution(); }

      static float GetNormalizationScale(uint32_t width,
                                         uint32_t height) {
        return float(std::max(width, height));
      }

      template<class Archive>
      void serialize(Archive &ar, const unsigned int version) {
        ar & name;
        if (version > 0) {
          ar & width;
          ar & height;
        }
        ar & K;
        ar & R;
        ar & C;
      }
    };
    typedef std::vector<Camera> CameraArr;

    // structure describing a pose along the trajectory of a platform
    struct Pose {
      Mat33d R;  // platform's rotation matrix
      Pos3d C;  // platform's translation vector in the global coordinate system

      template<class Archive>
      void serialize(Archive &ar, const unsigned int /*version*/) {
        ar & R;
        ar & C;
      }
    };
    typedef std::vector<Pose> PoseArr;

    std::string name;   // platform's name
    CameraArr cameras;  // cameras mounted on the platform
    PoseArr poses;      // trajectory of the platform

    const Mat33d &GetK(uint32_t cameraID) const { return cameras[cameraID].K; }

    Pose GetPose(uint32_t cameraID, uint32_t poseID) const {
      const Camera &camera = cameras[cameraID];
      const Pose &pose = poses[poseID];
      // add the relative camera pose to the platform
      return Pose{camera.R * pose.R, pose.R.t() * camera.C + pose.C};
    }

    template<class Archive>
    void serialize(Archive &ar, const unsigned int /*version*/) {
      ar & name;
      ar & cameras;
      ar & poses;
    }
  };
  typedef std::vector<Platform> PlatformArr;
  /*----------------------------------------------------------------*/

  // structure describing an image
  struct Image {
    std::string name;        // image file name
    uint32_t width, height;  // image resolution in pixels for all images
    // sharing this camera (optional)
    uint32_t platformID;     // ID of the associated platform
    uint32_t
        cameraID;     // ID of the associated camera on the associated platform
    uint32_t poseID;  // ID of the pose of the associated platform
    uint32_t ID;      // ID of this image in the global space (optional)

    Image() : poseID(NO_ID), ID(NO_ID) {}

    bool IsValid() const { return poseID != NO_ID; }

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & name;
      ar & width;
      ar & height;
      ar & platformID;
      ar & cameraID;
      ar & poseID;
      if (version > 2) {
        ar & ID;
      }
    }
  };
  typedef std::vector<Image> ImageArr;
  /*----------------------------------------------------------------*/

  // structure describing a 3D point
  struct Vertex {
    // structure describing one view for a given 3D feature
    struct View {
      uint32_t imageID;  // image ID corresponding to this view
      float confidence;  // view's confidence (0 - not available)

      template<class Archive>
      void serialize(Archive &ar, const unsigned int /*version*/) {
        ar & imageID;
        ar & confidence;
      }
    };
    typedef std::vector<View> ViewArr;

    Pos3f X;        // 3D point position
    ViewArr views;  // list of all available views for this 3D feature

    template<class Archive>
    void serialize(Archive &ar, const unsigned int /*version*/) {
      ar & X;
      ar & views;
    }
  };
  typedef std::vector<Vertex> VertexArr;
  /*----------------------------------------------------------------*/

  // structure describing a 3D line
  struct Line {
    // structure describing one view for a given 3D feature
    struct View {
      uint32_t imageID;  // image ID corresponding to this view
      float confidence;  // view's confidence (0 - not available)

      template<class Archive>
      void serialize(Archive &ar, const unsigned int /*version*/) {
        ar & imageID;
        ar & confidence;
      }
    };
    typedef std::vector<View> ViewArr;

    Pos3f pt1;      // 3D line segment end-point
    Pos3f pt2;      // 3D line segment end-point
    ViewArr views;  // list of all available views for this 3D feature

    template<class Archive>
    void serialize(Archive &ar, const unsigned int /*version*/) {
      ar & pt1;
      ar & pt2;
      ar & views;
    }
  };
  typedef std::vector<Line> LineArr;
  /*----------------------------------------------------------------*/

  // structure describing a 3D point's normal (optional)
  struct Normal {
    Pos3f n;  // 3D feature normal

    template<class Archive>
    void serialize(Archive &ar, const unsigned int /*version*/) {
      ar & n;
    }
  };
  typedef std::vector<Normal> NormalArr;
  /*----------------------------------------------------------------*/

  // structure describing a 3D point's color (optional)
  struct Color {
    Col3 c;  // 3D feature color

    template<class Archive>
    void serialize(Archive &ar, const unsigned int /*version*/) {
      ar & c;
    }
  };
  typedef std::vector<Color> ColorArr;
  /*----------------------------------------------------------------*/

  struct Mesh {
    typedef uint32_t FIndex;
    typedef uint32_t VIndex;
    typedef uint16_t MapIndex;
    struct Vertex {
      Pos3f X;

      template<class Archive>
      void serialize(Archive &ar, const unsigned int /*version*/) {
        ar & X;
      }
    };
    struct Normal {
      Pos3f n;

      template<class Archive>
      void serialize(Archive &ar, const unsigned int /*version*/) {
        ar & n;
      }
    };
    struct Face {
      cv::Point3_<VIndex> f;

      template<class Archive>
      void serialize(Archive &ar, const unsigned int /*version*/) {
        ar & f;
      }
    };
    struct TexCoord {
      cv::Point_<float> tc;

      template<class Archive>
      void serialize(Archive &ar, const unsigned int /*version*/) {
        ar & tc;
      }
    };

    struct Texture {
      std::string path;
      uint32_t width;
      uint32_t height;
      std::vector<uint8_t> data;

      template<class Archive>
      void serialize(Archive &ar, const unsigned int /*version*/) {
        ar & path;
        ar & width;
        ar & height;
        ar & data;
      }
    };

    typedef std::vector<Vertex> VertexArr;
    typedef std::vector<Face> FaceArr;
    typedef std::vector<Normal> NormalArr;
    typedef std::vector<VIndex> VertexIdxArr;
    typedef std::vector<FIndex> FaceIdxArr;
    typedef std::vector<VertexIdxArr> VertexVertexArr;
    typedef std::vector<FaceIdxArr> VertexFaceArr;
    typedef std::vector<bool> BoolArr;
    typedef std::vector<uint8_t> BoundaryArr;
    typedef std::vector<TexCoord> TexCoordArr;
    typedef std::vector<MapIndex> FaceMapIndexArr;
    typedef std::vector<Texture> TextureMapArr;

    VertexArr vertices;
    FaceArr faces;

    NormalArr vertexNormals;  // for each vertex, the normal to the surface in that point (optional)
    VertexVertexArr vertexVertices;  // for each vertex, the list of adjacent vertices (optional)
    VertexFaceArr vertexFaces;  // for each vertex, the list of faces containing it (optional)
    BoundaryArr vertexBoundary;  // for each vertex, stores if it is at the boundary or not (optional)

    NormalArr faceNormals;  // for each face, the normal to it (optional)
    TexCoordArr faceTexcoords;  // for each face, the texture-coordinates corresponding to the contained vertices (optional)
    FaceMapIndexArr faceMapIdxs;  // for each face, the id of texture diffuses in number (optional)

    TextureMapArr
        textureDiffuses;  // texture containing the diffuse color (optional)

    template<class Archive>
    void serialize(Archive &ar, const unsigned int /*version*/) {
      ar & vertices;
      ar & faces;
      ar & vertexNormals;
      ar & vertexVertices;
      ar & vertexFaces;
      ar & vertexBoundary;
      ar & faceNormals;
      ar & faceTexcoords;
      ar & faceMapIdxs;
      ar & textureDiffuses;
    }
  };

  PlatformArr platforms;  // array of platforms
  ImageArr images;        // array of images
  VertexArr vertices;     // array of reconstructed 3D points
  NormalArr verticesNormal;  // array of reconstructed 3D points' normal (optional)
  ColorArr verticesColor;  // array of reconstructed 3D points' color (optional)
  LineArr lines;           // array of reconstructed 3D lines
  NormalArr linesNormal;   // array of reconstructed 3D lines' normal (optional)
  ColorArr linesColor;     // array of reconstructed 3D lines' color (optional)
  Mat44d transform;  // transformation used to convert from absolute to relative coordinate system (optional)
  Mesh mesh;         // mesh structure
  Interface() : transform(Mat44d::eye()) {}

  const Mat33d &GetK(uint32_t imageID) const {
    const Image &image = images[imageID];
    return platforms[image.platformID].GetK(image.cameraID);
  }

  Platform::Pose GetPose(uint32_t imageID) const {
    const Image &image = images[imageID];
    return platforms[image.platformID].GetPose(image.cameraID, image.poseID);
  }

  template<class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar & platforms;
    ar & images;
    ar & vertices;
    ar & verticesNormal;
    ar & verticesColor;
    if (version > 0) {
      ar & lines;
      ar & linesNormal;
      ar & linesColor;
      if (version > 1) {
        ar & transform;
      }
    }
    ar & mesh;
  }
};
/*----------------------------------------------------------------*/

}  // namespace _INTERFACE_NAMESPACE

#endif  // _INTERFACE_MVS_H_
