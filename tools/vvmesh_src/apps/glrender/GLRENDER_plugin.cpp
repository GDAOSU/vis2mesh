#include "GLRENDER_plugin.h"
#include <Eigen/Core>

#include <pcl/io/auto_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "filesystem/ghc/filesystem.hpp"
namespace fs = ghc::filesystem;
#include <spdlog/spdlog.h>

#include <Eigen/Core>

#include "env_setup.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <OBJFile/obj.h>

#include <exception>
struct CamPose {
  Eigen::Matrix3d K, R;
  Eigen::Vector3d C;
  int width;
  int height;

  friend std::ostream &operator<<(std::ostream &os, const CamPose &cp) {
    os << "------------\n"
       << "K: " << cp.K << "\n"
       << "R: " << cp.R << "\n"
       << "C: " << cp.C.transpose() << "\n"
       << "w " << cp.width << " h " << cp.height << std::endl;
    return os;
  }

  nlohmann::json toJson() {
    nlohmann::json json;

    json["width"] = width;
    json["height"] = height;

    nlohmann::json mat3x3Json;
    nlohmann::json vec3Json;
    vec3Json.push_back(0.);
    vec3Json.push_back(0.);
    vec3Json.push_back(0.);

    mat3x3Json.push_back(vec3Json);
    mat3x3Json.push_back(vec3Json);
    mat3x3Json.push_back(vec3Json);

    nlohmann::json KJson = mat3x3Json;
    KJson[0][0] = K(0, 0);
    KJson[0][1] = K(0, 1);
    KJson[0][2] = K(0, 2);
    KJson[1][0] = K(1, 0);
    KJson[1][1] = K(1, 1);
    KJson[1][2] = K(1, 2);
    KJson[2][0] = K(2, 0);
    KJson[2][1] = K(2, 1);
    KJson[2][2] = K(2, 2);

    nlohmann::json RJson = mat3x3Json;
    RJson[0][0] = R(0, 0);
    RJson[0][1] = R(0, 1);
    RJson[0][2] = R(0, 2);
    RJson[1][0] = R(1, 0);
    RJson[1][1] = R(1, 1);
    RJson[1][2] = R(1, 2);
    RJson[2][0] = R(2, 0);
    RJson[2][1] = R(2, 1);
    RJson[2][2] = R(2, 2);

    nlohmann::json CJson = vec3Json;
    CJson[0] = C[0];
    CJson[1] = C[1];
    CJson[2] = C[2];

    json["K"] = KJson;
    json["R"] = RJson;
    json["C"] = CJson;
    return json;
  }
};

GLRENDER_Plugin::GLRENDER_Plugin(
    std::shared_ptr<FBORender::MultidrawFBO> pFBO) {
  fbo = pFBO;
}
std::string GLRENDER_Plugin::getWorkerName() { return WORKER_NAME; }
bool GLRENDER_Plugin::operator()(const nlohmann::json &blockJson) {
  return processBlock(blockJson);
}
bool GLRENDER_Plugin::ensureFolderExist(std::string path) {
  fs::path pp(path);
  if (fs::exists(pp)) {
    if (fs::is_directory(pp)) {
      return true;
    } else {
      return false;
    }
  } else {
    try {
      return fs::create_directory(pp);
    } catch (std::exception &e) {
      std::cout << e.what() << std::endl;
      return false;
    }
  }
}

bool GLRENDER_Plugin::processBlock(const nlohmann::json &blockJson) {
  // Verify block
  if (blockJson["Worker"].get<std::string>() != WORKER_NAME) {
    spdlog::warn("Worker[{0}] does not match {1}.\n", blockJson["Worker"].get<std::string>(), WORKER_NAME);
    return false;
  }
  nlohmann::json paramJson = blockJson["Param"];
  // Processing Unit
  std::string input_cloud = paramJson["input_cloud"].get<std::string>();
  std::string input_trimesh = paramJson["input_mesh"].get<std::string>();
  std::string input_rgbtrimesh = paramJson["input_rgbmesh"].get<std::string>();
  std::string input_textrimesh = paramJson["input_texmesh"].get<std::string>();
  std::string input_cam_list = paramJson["input_cam"].get<std::string>();
  std::string output_data_folder = paramJson["output_folder"].get<std::string>();
  bool renderPts = !input_cloud.empty();
  bool renderMesh = !input_trimesh.empty();
  bool renderRgbMesh = !input_rgbtrimesh.empty();
  bool renderTexMesh = !input_textrimesh.empty();
  bool renderCull =
      paramJson["render_cull"].get<bool>() && (renderMesh || renderTexMesh);
  bool cleanOutDir = paramJson.value<bool>("output_clean", false);
  float znear = paramJson.value<float>("znear", 0.1f);
  float zfar = paramJson.value<float>("zfar", 1000.f);
  std::string shader = paramJson["shader"].get<std::string>();
  int radius_k = paramJson["radius_k"].get<int>();

  int ptsShader = FBORender::PointCloudRenderObject::POINT;
  if (shader == "DOT" || shader == "ELLIPSE") {
    ptsShader = FBORender::PointCloudRenderObject::DOT;
  } else if (shader == "BOX" || shader == "RECTANGLE" || shader == "RECT") {
    ptsShader = FBORender::PointCloudRenderObject::BOX;
  } else if (shader == "DIAMOND") {
    ptsShader = FBORender::PointCloudRenderObject::DIAMOND;
  }
  if (!ensureFolderExist(output_data_folder)) {
    std::cerr << "Cannot create folder for output" << std::endl;
  }

  fs::path output_data_Path(output_data_folder);

  nlohmann::json camInfoJson;
  {
    std::ifstream ifs(input_cam_list, std::iostream::binary);
    ifs >> camInfoJson;
    ifs.close();
  }
  printf("Load Cam Info: \n");

  std::vector<CamPose> camPoseList;
  for (auto it = camInfoJson["imgs"].begin(), end = camInfoJson["imgs"].end();
       it != end; ++it) {
    CamPose cam;
    cam.width = (*it)["width"].get<int>();
    cam.height = (*it)["height"].get<int>();
    nlohmann::json matJson = (*it)["K"];
    cam.K(0, 0) = matJson[0][0].get<double>();
    cam.K(0, 1) = matJson[0][1].get<double>();
    cam.K(0, 2) = matJson[0][2].get<double>();
    cam.K(1, 0) = matJson[1][0].get<double>();
    cam.K(1, 1) = matJson[1][1].get<double>();
    cam.K(1, 2) = matJson[1][2].get<double>();
    cam.K(2, 0) = matJson[2][0].get<double>();
    cam.K(2, 1) = matJson[2][1].get<double>();
    cam.K(2, 2) = matJson[2][2].get<double>();
    matJson = (*it)["R"];
    cam.R(0, 0) = matJson[0][0].get<double>();
    cam.R(0, 1) = matJson[0][1].get<double>();
    cam.R(0, 2) = matJson[0][2].get<double>();
    cam.R(1, 0) = matJson[1][0].get<double>();
    cam.R(1, 1) = matJson[1][1].get<double>();
    cam.R(1, 2) = matJson[1][2].get<double>();
    cam.R(2, 0) = matJson[2][0].get<double>();
    cam.R(2, 1) = matJson[2][1].get<double>();
    cam.R(2, 2) = matJson[2][2].get<double>();
    matJson = (*it)["C"];
    cam.C[0] = matJson[0].get<double>();
    cam.C[1] = matJson[1].get<double>();
    cam.C[2] = matJson[2].get<double>();

    camPoseList.push_back(cam);
  }
  std::cout << "Peek First Cam:\n" << camPoseList.front();

  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  pcl::PolygonMesh mesh;
  if (renderPts) pcl::io::load(input_cloud, cloud);
  if (renderMesh) pcl::io::load(input_trimesh, mesh);

  PCL_INFO("Load cloud (+v %d)\n", cloud.size());
  PCL_INFO("Load mesh (+v %d) (+f %d)\n", mesh.cloud.width * mesh.cloud.height,
           mesh.polygons.size());

  /// Estimate Radius
  std::vector<float> cloudRadius;
  {
    cloudRadius.resize(cloud.size(), 1);
    if (ptsShader != FBORender::PointCloudRenderObject::POINT &&
        cloud.size() > radius_k + 1) {
      pcl::KdTreeFLANN<pcl::PointXYZRGB> tree;
      tree.setInputCloud(cloud.makeShared());
      for (uint32_t i = 0, num_pts = cloud.size(); i < num_pts; ++i) {
        std::vector<int> knnind;
        std::vector<float> knnsqdist;
        tree.nearestKSearch(i, radius_k + 1, knnind, knnsqdist);
        float sum_radius = 0;
        for (int j = 1; j <= radius_k; ++j) {
          sum_radius += std::sqrt(knnsqdist[j]);
        }
        sum_radius /= float(radius_k);
        cloudRadius[i] = sum_radius;
      }
    }
  }

  // Render
  const glm::vec4 bgcolor = {0.f, 0.f, 0.f, 1.f};
  constexpr float INVALID_DEPTH = -1;
  constexpr uint32_t INVALID_INDEX = UINT32_MAX;
  const float bgZ = INVALID_DEPTH;
  const uint32_t bgInstance = INVALID_INDEX;
  const float bgDepth = 1.f;

  FBORender::PointCloudType fboCloud;
  FBORender::TriMeshType fboMesh;

  FBORender::PointCloudRenderObject pcro;
  FBORender::TriMeshRenderObject tmro;
  FBORender::RGBTriMeshRenderObject rgbtmro;
  FBORender::TexturedTriMeshRenderObject textmro;

  if (renderPts) {
    // Convert to FBORender::Cloud
    fboCloud.reserve(cloud.size());
    for (uint32_t i = 0; i < cloud.size(); ++i) {
      auto &pt = cloud[i];
      fboCloud.emplace_back(FBORender::PointRecType{
          pt.x, pt.y, pt.z, float(pt.r) / 255.f, float(pt.g) / 255.f,
          float(pt.b) / 255.f, cloudRadius[i]});
    }

    pcro.setPattern(ptsShader);
    pcro.buildVAO(reinterpret_cast<float *>(fboCloud.data()), fboCloud.size());
    fboCloud.clear();
  }
  if (renderMesh) {
    // Convert to FBORender::Mesh
    pcl::PointCloud<pcl::PointXYZ> meshPts;
    pcl::fromPCLPointCloud2(mesh.cloud, meshPts);
    fboMesh.points.reserve(meshPts.size());
    fboMesh.cells.reserve(mesh.polygons.size());
    for (auto &pt: meshPts) {
      fboMesh.points.emplace_back(FBORender::PtType{pt.x, pt.y, pt.z});
    }
    for (auto &fc: mesh.polygons) {
      assert(fc.vertices.size() == 3);
      fboMesh.cells.emplace_back(
          FBORender::CellType{fc.vertices[0], fc.vertices[1], fc.vertices[2]});
    }

    tmro.buildVAO(reinterpret_cast<float *>(fboMesh.points.data()),
                  fboMesh.points.size(),
                  reinterpret_cast<uint32_t *>(fboMesh.cells.data()),
                  fboMesh.cells.size());
    fboMesh.points.clear();
    fboMesh.cells.clear();
  }

  if (renderRgbMesh) {
    std::string _file_ext = fs::path(input_rgbtrimesh).extension().string();

    if (_file_ext == ".obj") {
      ObjModel obj;
      obj.Load(input_rgbtrimesh);
      std::vector<float> geo(6 * obj.get_vertices().size());
      assert(obj.get_vertices().size() == obj.get_vertex_colors().size());

      for (size_t i = 0; i < obj.get_vertices().size(); ++i) {
        for (int j = 0; j < 3; ++j) geo[6 * i + j] = obj.get_vertices()[i][j];
        for (int j = 3; j < 6; ++j) geo[6 * i + j] = obj.get_vertex_colors()[i][j];
      }
      assert(obj.get_groups().size() == 1);

      auto &objgroup = obj.get_groups()[0];
      size_t n_faces = objgroup.faces.size();

      std::vector<uint32_t> geoidx(3 * n_faces);
      // compile to vector
      for (size_t i = 0; i < n_faces; ++i)
        for (int j = 0; j < 3; ++j) {
          geoidx[3 * i + j] = objgroup.faces[i].vertices[j];
        }
      rgbtmro.buildVAO(geo.data(), obj.get_vertices().size(),
                       geoidx.data(), n_faces);
      PCL_INFO("Load rgb mesh obj (+v %d) (+f %d)\n", obj.get_vertices().size(),
               n_faces);
    } else if (_file_ext == ".ply") {
      pcl::io::load(input_rgbtrimesh, mesh);
      pcl::PointCloud<pcl::PointXYZRGB> meshPts;
      pcl::fromPCLPointCloud2(mesh.cloud, meshPts);

      std::vector<float> geo(6 * meshPts.size());
      for (size_t i = 0; i < meshPts.size(); ++i) {
        geo[6 * i + 0] = meshPts[i].x;
        geo[6 * i + 1] = meshPts[i].y;
        geo[6 * i + 2] = meshPts[i].z;
        geo[6 * i + 3] = meshPts[i].r / 255.f;
        geo[6 * i + 4] = meshPts[i].g / 255.f;
        geo[6 * i + 5] = meshPts[i].b / 255.f;
      }
      size_t n_faces = mesh.polygons.size();
      std::vector<uint32_t> geoidx(3 * n_faces);
      // compile to vector
      for (size_t i = 0; i < n_faces; ++i)
        for (int j = 0; j < 3; ++j) {
          geoidx[3 * i + j] = mesh.polygons[i].vertices[j];
        }
      rgbtmro.buildVAO(geo.data(), meshPts.size(),
                       geoidx.data(), n_faces);
      PCL_INFO("Load rgb mesh ply (+v %d) (+f %d)\n", meshPts.size(),
               n_faces);
    } else {
      throw std::runtime_error(_file_ext + " not recognized.");
    }
  }

  if (renderTexMesh) {
    ObjModel obj;
    obj.Load(input_textrimesh);
    std::vector<float> geo(3 * obj.get_vertices().size()),
        uv(2 * obj.get_texcoords().size());
    for (size_t i = 0; i < obj.get_vertices().size(); ++i)
      for (int j = 0; j < 3; ++j) geo[3 * i + j] = obj.get_vertices()[i][j];
    for (size_t i = 0; i < obj.get_texcoords().size(); ++i)
      for (int j = 0; j < 2; ++j) uv[2 * i + j] = obj.get_texcoords()[i][j];

    uint32_t id_offset = 0;
    for (size_t gi = 0; gi < obj.get_groups().size(); ++gi) {
      auto &objgroup = obj.get_groups()[gi];
      if (objgroup.faces.empty()) continue;
      auto objmat = obj.GetMaterial(objgroup.material_name);
      auto patch = textmro.CreatePatch();
      GLuint texid;
      glGenTextures(1, &texid);
      if (!objmat->LoadDiffuseMap()) {
        std::cerr << "Warning: " << objmat->diffuse_name << " not found."
                  << std::endl;
        throw new std::runtime_error("Diffuse map not found");
        return false;
      }
      cv::Mat flipped;
      cv::flip(objmat->diffuse_map, flipped, 0);
      patch->setTexture(flipped.data, flipped.cols, flipped.rows,
                        patch->Format_BGR);
      size_t n_faces = objgroup.faces.size();

      std::vector<uint32_t> geoidx(3 * n_faces), uvidx(3 * n_faces);
      // compile to vector
      for (size_t i = 0; i < n_faces; ++i)
        for (int j = 0; j < 3; ++j) {
          uvidx[3 * i + j] = objgroup.faces[i].texcoords[j];
          geoidx[3 * i + j] = objgroup.faces[i].vertices[j];
        }
      patch->buildVAO(geo.data(), uv.data(), obj.get_vertices().size(),
                      geoidx.data(), uvidx.data(), n_faces, id_offset);
      id_offset += n_faces;
    }
    PCL_INFO("Load textured mesh (+v %d) (+f %d)\n", obj.get_vertices().size(),
             id_offset);
  }
  printf("Render Object done.\n");

  glDepthFunc(GL_LESS);
  glEnable(GL_DEPTH_TEST);
  GLenum drawBuffers[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1,
                           GL_COLOR_ATTACHMENT2};
  printf("FBO done.\n");

  glm::mat4 mvMat, projMat, mvpMat;
  std::vector<uint8_t> rgbPts, rgbMesh, rgbClMesh;
  std::vector<float> depthPts, depthMesh, depthClMesh;
  std::vector<uint32_t> instPts, instMesh, instClMesh;
  int cnt = 0;
  for (CamPose &cam: camPoseList) {
    fbo->resize(cam.width, cam.height);
    if (renderPts) {
      rgbPts.resize(cam.width * cam.height * 3);
      depthPts.resize(cam.width * cam.height);
      instPts.resize(cam.width * cam.height);
    }
    if (renderMesh || renderRgbMesh || renderTexMesh) {
      rgbMesh.resize(cam.width * cam.height * 3);
      depthMesh.resize(cam.width * cam.height);
      instMesh.resize(cam.width * cam.height);
    }
    if (renderCull) {
      rgbClMesh.resize(cam.width * cam.height * 3);
      depthClMesh.resize(cam.width * cam.height);
      instClMesh.resize(cam.width * cam.height);
    }
    FBORender::cvMatrix2glMatrix(cam.R, cam.C, cam.K, cam.width, cam.height,
                                 mvMat, projMat, znear, zfar);
    mvpMat = projMat * mvMat;
    if (renderMesh) {
      // 1st pass: render doubleface mesh
      glDisable(GL_CULL_FACE);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo->fbo());
      glDrawBuffers(3, drawBuffers);
      glClearBufferfv(GL_COLOR, 0, glm::value_ptr(bgcolor));
      glClearBufferfv(GL_COLOR, 1, &bgZ);
      glClearBufferuiv(GL_COLOR, 2, &bgInstance);
      glClearBufferfv(GL_DEPTH, 0, &bgDepth);
      tmro.draw(mvMat, projMat, mvpMat);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
      fbo->getTexImage(FBORender::MultidrawFBO::CHRGB, false, rgbMesh.data());
      fbo->getTexImage(FBORender::MultidrawFBO::CHZ, false, depthMesh.data());
      fbo->getTexImage(FBORender::MultidrawFBO::CHID, false, instMesh.data());
      if (renderCull) {
        // 2nd pass: render mesh with face culling
        glEnable(GL_CULL_FACE);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo->fbo());
        glDrawBuffers(3, drawBuffers);
        glClearBufferfv(GL_COLOR, 0, glm::value_ptr(bgcolor));
        glClearBufferfv(GL_COLOR, 1, &bgZ);
        glClearBufferuiv(GL_COLOR, 2, &bgInstance);
        glClearBufferfv(GL_DEPTH, 0, &bgDepth);
        tmro.draw(mvMat, projMat, mvpMat);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        fbo->getTexImage(FBORender::MultidrawFBO::CHRGB, false, rgbMesh.data());
        fbo->getTexImage(FBORender::MultidrawFBO::CHZ, false,
                         depthClMesh.data());
        fbo->getTexImage(FBORender::MultidrawFBO::CHID, false,
                         instClMesh.data());
      }
    }

    if (renderRgbMesh) {
      // 1st pass: render doubleface mesh
      glDisable(GL_CULL_FACE);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo->fbo());
      glDrawBuffers(3, drawBuffers);
      glClearBufferfv(GL_COLOR, 0, glm::value_ptr(bgcolor));
      glClearBufferfv(GL_COLOR, 1, &bgZ);
      glClearBufferuiv(GL_COLOR, 2, &bgInstance);
      glClearBufferfv(GL_DEPTH, 0, &bgDepth);
      rgbtmro.draw(mvMat, projMat, mvpMat);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
      fbo->getTexImage(FBORender::MultidrawFBO::CHRGB, false, rgbMesh.data());
      fbo->getTexImage(FBORender::MultidrawFBO::CHZ, false, depthMesh.data());
      fbo->getTexImage(FBORender::MultidrawFBO::CHID, false, instMesh.data());
      if (renderCull) {
        // 2nd pass: render mesh with face culling
        glEnable(GL_CULL_FACE);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo->fbo());
        glDrawBuffers(3, drawBuffers);
        glClearBufferfv(GL_COLOR, 0, glm::value_ptr(bgcolor));
        glClearBufferfv(GL_COLOR, 1, &bgZ);
        glClearBufferuiv(GL_COLOR, 2, &bgInstance);
        glClearBufferfv(GL_DEPTH, 0, &bgDepth);
        rgbtmro.draw(mvMat, projMat, mvpMat);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        fbo->getTexImage(FBORender::MultidrawFBO::CHRGB, false, rgbMesh.data());
        fbo->getTexImage(FBORender::MultidrawFBO::CHZ, false,
                         depthClMesh.data());
        fbo->getTexImage(FBORender::MultidrawFBO::CHID, false,
                         instClMesh.data());
      }
    }

    if (renderTexMesh) {
      // 1st pass: render doubleface mesh
      glDisable(GL_CULL_FACE);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo->fbo());
      glDrawBuffers(3, drawBuffers);
      glClearBufferfv(GL_COLOR, 0, glm::value_ptr(bgcolor));
      glClearBufferfv(GL_COLOR, 1, &bgZ);
      glClearBufferuiv(GL_COLOR, 2, &bgInstance);
      glClearBufferfv(GL_DEPTH, 0, &bgDepth);
      textmro.draw(mvMat, projMat, mvpMat);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
      fbo->getTexImage(FBORender::MultidrawFBO::CHRGB, false, rgbMesh.data());
      fbo->getTexImage(FBORender::MultidrawFBO::CHZ, false, depthMesh.data());
      fbo->getTexImage(FBORender::MultidrawFBO::CHID, false, instMesh.data());
      if (renderCull) {
        // 2nd pass: render mesh with face culling
        glEnable(GL_CULL_FACE);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo->fbo());
        glDrawBuffers(3, drawBuffers);
        glClearBufferfv(GL_COLOR, 0, glm::value_ptr(bgcolor));
        glClearBufferfv(GL_COLOR, 1, &bgZ);
        glClearBufferuiv(GL_COLOR, 2, &bgInstance);
        glClearBufferfv(GL_DEPTH, 0, &bgDepth);
        textmro.draw(mvMat, projMat, mvpMat);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        fbo->getTexImage(FBORender::MultidrawFBO::CHRGB, false,
                         rgbClMesh.data());
        fbo->getTexImage(FBORender::MultidrawFBO::CHZ, false,
                         depthClMesh.data());
        fbo->getTexImage(FBORender::MultidrawFBO::CHID, false,
                         instClMesh.data());
      }
    }

    if (renderPts) {
      // 3rd pass: render points
      glDisable(GL_CULL_FACE);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo->fbo());
      glDrawBuffers(3, drawBuffers);
      glClearBufferfv(GL_COLOR, 0, glm::value_ptr(bgcolor));
      glClearBufferfv(GL_COLOR, 1, &bgZ);
      glClearBufferuiv(GL_COLOR, 2, &bgInstance);
      glClearBufferfv(GL_DEPTH, 0, &bgDepth);
      pcro.draw(mvMat, projMat, mvpMat);
      glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
      fbo->getTexImage(FBORender::MultidrawFBO::CHRGB, false, rgbPts.data());
      fbo->getTexImage(FBORender::MultidrawFBO::CHZ, false, depthPts.data());
      fbo->getTexImage(FBORender::MultidrawFBO::CHID, false, instPts.data());
    }
    // Write to Image
    std::string output_rgbpt_path =
        (output_data_Path / ("pt" + std::to_string(cnt) + ".png")).string();
    std::string output_deppt_path =
        (output_data_Path / ("pt" + std::to_string(cnt) + ".flt")).string();
    std::string output_instpt_path =
        (output_data_Path / ("pt" + std::to_string(cnt) + ".uint")).string();
    std::string output_rgbms_path =
        (output_data_Path / ("mesh" + std::to_string(cnt) + ".png")).string();
    std::string output_depms_path =
        (output_data_Path / ("mesh" + std::to_string(cnt) + ".flt")).string();
    std::string output_instms_path =
        (output_data_Path / ("mesh" + std::to_string(cnt) + ".uint")).string();
    std::string output_rgbmscl_path =
        (output_data_Path / ("meshcull" + std::to_string(cnt) + ".png"))
            .string();
    std::string output_depmscl_path =
        (output_data_Path / ("meshcull" + std::to_string(cnt) + ".flt"))
            .string();
    std::string output_instmscl_path =
        (output_data_Path / ("meshcull" + std::to_string(cnt) + ".uint"))
            .string();
    std::string output_cam_path =
        (output_data_Path / ("cam" + std::to_string(cnt) + ".json")).string();

    std::cout << "RGB " << output_rgbpt_path << std::endl;

    int imgsize[2] = {cam.height, cam.width};
    std::ofstream ofs;
    if (renderPts) {
      cv::Mat cvrgb(cam.height, cam.width, CV_8UC3, rgbPts.data());
      cv::imwrite(output_rgbpt_path, cvrgb);

      ofs.open(output_deppt_path, std::ios::binary);
      ofs.write((char *) imgsize, sizeof(int) * 2);
      ofs.write((char *) depthPts.data(), sizeof(float) * depthPts.size());
      ofs.close();
      ofs.open(output_instpt_path, std::ios::binary);
      ofs.write((char *) imgsize, sizeof(int) * 2);
      ofs.write((char *) instPts.data(), sizeof(uint32_t) * instPts.size());
      ofs.close();
    }

    if (renderMesh || renderRgbMesh || renderTexMesh) {
      cv::Mat cvrgb(cam.height, cam.width, CV_8UC3, rgbMesh.data());
      cv::imwrite(output_rgbms_path, cvrgb);

      ofs.open(output_depms_path, std::ios::binary);
      ofs.write((char *) imgsize, sizeof(int) * 2);
      ofs.write((char *) depthMesh.data(), sizeof(float) * depthMesh.size());
      ofs.close();
      ofs.open(output_instms_path, std::ios::binary);
      ofs.write((char *) imgsize, sizeof(int) * 2);
      ofs.write((char *) instMesh.data(), sizeof(uint32_t) * instMesh.size());
      ofs.close();
    }

    if (renderCull) {
      cv::Mat cvrgb(cam.height, cam.width, CV_8UC3, rgbClMesh.data());
      cv::imwrite(output_rgbmscl_path, cvrgb);

      ofs.open(output_depmscl_path, std::ios::binary);
      ofs.write((char *) imgsize, sizeof(int) * 2);
      ofs.write((char *) depthClMesh.data(), sizeof(float) * depthClMesh.size());
      ofs.close();
      ofs.open(output_instmscl_path, std::ios::binary);
      ofs.write((char *) imgsize, sizeof(int) * 2);
      ofs.write((char *) instClMesh.data(), sizeof(uint32_t) * instClMesh.size());
      ofs.close();
    }
    constexpr bool saveCam = true;
    if (saveCam) {
      nlohmann::json camJson = cam.toJson();

      ofs.open(output_cam_path);
      ofs << camJson;
      ofs.close();
    }
    cnt++;
  }
  return true;
}

nlohmann::json GLRENDER_Plugin::getDefaultParameters() {
//  auto sameline = Json::CommentPlacement::commentAfterOnSameLine;
//  auto before = Json::CommentPlacement::commentBefore;
  nlohmann::json blockJson, paramJson;
  blockJson["Worker"] = WORKER_NAME;
//  blockJson["Worker"]
//      .setComment(std::string("// This plugin renders point cloud, mesh (rgb or textured) with given camera parameters."),
//                  before);

  paramJson["input_cloud"] = "(required) <file path to .ply file>";
  paramJson["input_mesh"] = "(optional) <any mesh file, .obj or .ply>";
  paramJson["input_rgbmesh"] = "(optional) <vertex rgb mesh .ply>";
  paramJson["input_texmesh"] = "(optional) <mtl texutred mesh .obj>";
  paramJson["input_cam"] = "<input cam.json>";
  paramJson["output_folder"] = "<a folder for output>";
  paramJson["render_cull"] = false;
//  paramJson["render_cull"].setComment(std::string("// Whether render culled mesh."), sameline);
  paramJson["output_clean"] = false;
//  paramJson["output_clean"].setComment(std::string("// Whether remove all files in output folder."), sameline);
  paramJson["znear"] = 0.1f;
  paramJson["zfar"] = 1000.f;
  paramJson["shader"] = "DOT";
//  paramJson["shader"].setComment(std::string("// 4 modes: POINT, DOT (or ELLIPSE), BOX (or RECT), DIAMOND."), sameline);
  paramJson["radius_k"] = 1;
//  paramJson["radius_k"].setComment(std::string("// radius scales for 3 modes other than POINT."), sameline);

  blockJson["Param"] = paramJson;
  return blockJson;
}