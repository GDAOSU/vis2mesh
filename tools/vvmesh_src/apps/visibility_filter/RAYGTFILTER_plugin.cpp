#include "RAYGTFILTER_plugin.h"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <spdlog/spdlog.h>

#include "FBORender/FBORender.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Dense>

#include <pcl/io/auto_io.h>

#ifdef _USE_OPENMP
#include <omp.h>
#endif

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
typedef CGAL::Simple_cartesian<double> K;
typedef K::FT FT;
typedef K::Ray_3 Ray;
typedef K::Line_3 Line;
typedef K::Direction_3 Direction;
typedef K::Segment_3 Segment;
typedef K::Point_3 Point;
typedef K::Triangle_3 Triangle;
typedef std::list<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;
typedef boost::optional<Tree::Intersection_and_primitive_id<Segment>::Type> Segment_intersection;

struct RAYGTFILTER_Plugin_Cache {
  std::string cached_cloudpath;
  std::string cached_meshpath;
  pcl::PointCloud<pcl::PointXYZ> cached_cloud;
  Tree cached_querytree;
  std::list<Triangle> cached_triangles;
};

RAYGTFILTER_Plugin::RAYGTFILTER_Plugin() : cache(new RAYGTFILTER_Plugin_Cache) {
}
RAYGTFILTER_Plugin::~RAYGTFILTER_Plugin() {}

std::string RAYGTFILTER_Plugin::getWorkerName() { return WORKER_NAME; }
bool RAYGTFILTER_Plugin::operator()(const nlohmann::json &blockJson) {
  return processBlock(blockJson);
}

bool RAYGTFILTER_Plugin::exists_test(const std::string &name) {
  std::ifstream f(name.c_str());
  return f.good();
}

namespace MyOperator {
static omp_lock_t lock;
class RAYGTFILTERObject {
 public:
  void filter(const unsigned char *inColor, const float *inDepth,
              const uint32_t *inID, unsigned char *outColor, float *outDepth,
              uint32_t *outID, const uint32_t width, const uint32_t height,
              const float invalid_depth, const uint32_t invalid_index,
              const Eigen::Vector3f &camCenter,
              const pcl::PointCloud<pcl::PointXYZ> &cloud,
              const Tree &querytree,
              const double EPSILON = std::numeric_limits<float>::epsilon()) {
    std::fill_n(outColor, 3 * width * height, 0);
    std::fill_n(outDepth, width * height, invalid_depth);
    std::fill_n(outID, width * height, invalid_index);
    std::unordered_map<uint32_t, bool> cached_result;

    Point a(camCenter[0], camCenter[1], camCenter[2]);
#pragma omp parallel for collapse(2)
    for (int ri = 0; ri < height; ++ri)
      for (int ci = 0; ci < width; ++ci) {
        int idx = ri * width + ci;
        uint32_t ptid = inID[idx];
        if (ptid == invalid_index) continue;
        omp_set_lock(&lock);
        auto cache_it = cached_result.find(ptid);
        omp_unset_lock(&lock);
        bool isFG, hasInlier;
        if (cache_it == cached_result.end()) {
          // not found in cache
          Point q(cloud[ptid].x, cloud[ptid].y, cloud[ptid].z);
          auto ext_dir = Direction(q - a).vector();
          Point b = q + ext_dir / std::sqrt(ext_dir.squared_length()) * EPSILON;
          Segment segment_query(b, a);
          std::list<Segment_intersection> intersections;
          querytree.all_intersections(segment_query, std::back_inserter(intersections));
          std::vector<std::pair<float, CGAL::Oriented_side>> _intInfo;
          _intInfo.reserve(intersections.size());
          isFG = false;
          for (auto it = intersections.begin(), end = intersections.end(); it != end; ++it) {
            Segment_intersection &intersection = *it;
            if (intersection) {
              const Point *p = boost::get<Point>(&(intersection->first)); // gets intersection object
              if (p) {
                Triangle tri_hitted = *intersection->second;
                _intInfo.emplace_back(std::make_pair(std::sqrt((*p - b).squared_length()) - EPSILON,
                                                     tri_hitted.supporting_plane().oriented_side(a)));
              }
            }
          }
          std::sort(_intInfo.begin(), _intInfo.end(), [](std::pair<float, CGAL::Oriented_side> &p1,
                                                         std::pair<float, CGAL::Oriented_side> &p2) {
            return p1.first < p2.first;
          });
          /**  \ face right,  / face left
           *  b(      q      ) --------> a
           *  -EPSILON      EPSILON
           */
          for (int l = 0; l < _intInfo.size(); ++l) {
            if (_intInfo[l].first > EPSILON) {
              /** Stop examine when out of EPSILON
               * b(    q    ) p-------> a
               */
              isFG = false;
              break;
            } else {
              /** Because loop from left to right, the face closer to source a will cover the previous state.
               * b(  p/  q    ) -------> a , the q is not FG, because it is back-face
               * b(  p\  q    ) -------> a , the q is FG
               * b(   q  p\  ) -------> a , the q is FG
               * b(   q  p/  ) -------> a , the q is not FG, because it is back-face
               */
              isFG = _intInfo[l].second == CGAL::Oriented_side::ON_POSITIVE_SIDE;
            }
          }
          omp_set_lock(&lock);
          cached_result[ptid] = isFG;
          omp_unset_lock(&lock);
        } else {
          isFG = cache_it->second;
        }
        if (isFG) {
          std::copy_n(inColor + 3 * idx, 3, outColor + 3 * idx);
          outDepth[idx] = inDepth[idx];
          outID[idx] = inID[idx];
        }
      }
    omp_destroy_lock(&lock);
  }
};
}  // namespace MyOperator

bool RAYGTFILTER_Plugin::processBlock(const nlohmann::json &blockJson) {
  // Verify block
  if (blockJson["Worker"].get<std::string>() != WORKER_NAME) {
    spdlog::warn("Worker[{0}] does not match {1}.\n", blockJson["Worker"].get<std::string>(), WORKER_NAME);
    return false;
  }
  nlohmann::json paramJson = blockJson["Param"];

  // Processing Unit
  std::string input_rgb = paramJson["input_rgb"].get<std::string>();
  std::string input_depth = paramJson["input_depth"].get<std::string>();
  std::string input_inst = paramJson["input_instance"].get<std::string>();
  std::string input_cam = paramJson["input_cam"].get<std::string>();

  std::string input_cloud = paramJson["input_cloud"].get<std::string>();
  std::string input_trimesh = paramJson["input_mesh"].get<std::string>();

  bool enable_meshcull = false;
  if (!exists_test(input_rgb)) {
    fprintf(stderr, "[%s] File %s not exists.\n", WORKER_NAME, input_rgb.c_str());
    return false;
  }
  if (!exists_test(input_depth)) {
    fprintf(stderr, "[%s] File %s not exists.\n", WORKER_NAME, input_depth.c_str());
    return false;
  }
  if (!exists_test(input_inst)) {
    fprintf(stderr, "[%s] File %s not exists.\n", WORKER_NAME, input_inst.c_str());
    return false;
  }
  if (!exists_test(input_cam)) {
    fprintf(stderr, "[%s] File %s not exists.\n", WORKER_NAME, input_cam.c_str());
    return false;
  }
  if (!exists_test(input_cloud)) {
    fprintf(stderr, "[%s] File %s not exists.\n", WORKER_NAME, input_cloud.c_str());
    return false;
  }
  if (!exists_test(input_trimesh)) {
    fprintf(stderr, "[%s] File %s not exists.\n", WORKER_NAME, input_trimesh.c_str());
    return false;
  }

  bool bgmask = paramJson.value<std::string>("mask_type", "bg") == "bg";
  double tolerance = paramJson.value<double>("tolerance", std::numeric_limits<float>::epsilon());
  std::string output_rgb = paramJson.value<std::string>("output_rgb", "");
  std::string output_depth = paramJson.value<std::string>("output_depth", "");
  std::string output_inst = paramJson.value<std::string>("output_instance", "");
  std::string output_mask = paramJson.value<std::string>("output_mask", "");
  bool isOutRGB = !output_rgb.empty();
  bool isOutDepth = !output_depth.empty();
  bool isOutInstance = !output_inst.empty();
  bool isOutputMask = !output_mask.empty();
  //////////////////////
  Eigen::Matrix3f K, R;
  Eigen::Vector3f C;
  {
    nlohmann::json camJson;
    std::ifstream _ifs(input_cam, std::ios::binary);
    _ifs >> camJson;
    _ifs.close();
    for (int i = 0; i < 9; ++i)
      K(i / 3, i % 3) = camJson["K"][i / 3][i % 3].get<float>();
    for (int i = 0; i < 9; ++i)
      R(i / 3, i % 3) = camJson["R"][i / 3][i % 3].get<float>();
    for (int i = 0; i < 3; ++i)
      C(i) = camJson["C"][i].get<float>();
  }

  cv::Mat cvrgb = cv::imread(input_rgb);
  int width = cvrgb.cols;
  int height = cvrgb.rows;
  std::vector<float> rawdepth(width * height), meshdepth(width * height),
      meshculldepth(enable_meshcull ? width * height : 0);
  std::vector<uint32_t> rawinst(width * height);
  std::ifstream ifs;
  int fileshape[2];

  ifs.open(input_depth, std::ifstream::binary);
  ifs.read((char *) fileshape, sizeof(int) * 2);
  assert(fileshape[0] == width);
  assert(fileshape[1] == height);
  ifs.read((char *) rawdepth.data(), sizeof(float) * rawdepth.size());
  ifs.close();
  ifs.open(input_inst, std::ifstream::binary);
  ifs.read((char *) fileshape, sizeof(int) * 2);
  assert(fileshape[0] == width);
  assert(fileshape[1] == height);
  ifs.read((char *) rawinst.data(), sizeof(uint32_t) * rawinst.size());
  ifs.close();
  // Read points or retrieve from cache
  if (input_cloud != cache->cached_cloudpath) {
    cache->cached_cloud.clear();
    pcl::io::load(input_cloud, cache->cached_cloud);
    cache->cached_cloudpath = input_cloud;
    PCL_INFO("Load cloud (+v %d)\n", cache->cached_cloud.size());
  }
  // Read mesh or retrieve from cache
  if (input_trimesh != cache->cached_meshpath) {
    cache->cached_triangles.clear();
    cache->cached_querytree.clear();
    pcl::PolygonMesh _mesh;
    pcl::io::load(input_trimesh, _mesh);
    PCL_INFO("Load mesh (+v %d) (+f %d)\n", _mesh.cloud.width * _mesh.cloud.height,
             _mesh.polygons.size());
    cache->cached_meshpath = input_trimesh;

    pcl::PointCloud<pcl::PointXYZ> _v;
    pcl::fromPCLPointCloud2(_mesh.cloud, _v);
    for (auto &p: _mesh.polygons) {
      auto &i0 = p.vertices[0];
      auto &i1 = p.vertices[1];
      auto &i2 = p.vertices[2];
      cache->cached_triangles.push_back(Triangle(Point(_v[i0].x, _v[i0].y, _v[i0].z),
                                                 Point(_v[i1].x, _v[i1].y, _v[i1].z),
                                                 Point(_v[i2].x, _v[i2].y, _v[i2].z)));
    }
    cache->cached_querytree.rebuild(cache->cached_triangles.begin(), cache->cached_triangles.end());
  }

  std::vector<uint8_t> rgbFil;
  std::vector<float> depFil;
  std::vector<uint32_t> instFil;
  std::vector<uint8_t> maskFil;

  rgbFil.resize(3 * width * height);
  depFil.resize(width * height);
  instFil.resize(width * height);

  MyOperator::RAYGTFILTERObject filter;
  filter.filter(cvrgb.data, rawdepth.data(), rawinst.data(),
                rgbFil.data(), depFil.data(), instFil.data(), width,
                height, -1.f, UINT32_MAX,
                C,
                cache->cached_cloud,
                cache->cached_querytree,
                tolerance);

  if (isOutputMask) {
    maskFil.resize(width * height);
    if (bgmask) {
      for (int i = 0; i < width * height; ++i) {
        maskFil[i] = (rawinst[i] == instFil[i]) ? 0 : 255;
      }
    } else {
      for (int i = 0; i < width * height; ++i) {
        if (rawinst[i] == UINT32_MAX) {
          continue;
        }
        maskFil[i] = (rawinst[i] == instFil[i]) ? 255 : 0;
      }
    }
  }
//  }

  ///////// OUTPUT SECTION
  if (isOutRGB) {
    cv::Mat cvout(height, width, CV_8UC3, rgbFil.data());
    cv::imwrite(output_rgb, cvout);
  }

  int imgsize[2] = {width, height};
  std::ofstream ofs;
  if (isOutDepth) {
    ofs.open(output_depth, std::ios::binary);
    ofs.write((char *) imgsize, sizeof(int) * 2);
    ofs.write((char *) depFil.data(), sizeof(float) * depFil.size());
    ofs.close();
  }

  if (isOutInstance) {
    ofs.open(output_inst, std::ifstream::binary);
    ofs.write((char *) imgsize, sizeof(int) * 2);
    ofs.write((char *) instFil.data(), sizeof(uint32_t) * instFil.size());
    ofs.close();
  }

  if (isOutputMask) {
    cv::Mat cvmask(height, width, CV_8UC1, maskFil.data());
    cv::imwrite(output_mask, cvmask);
  }

  return true;
}
