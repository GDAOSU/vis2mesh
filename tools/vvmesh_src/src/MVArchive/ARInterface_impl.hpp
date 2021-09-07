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
#ifndef OPENFLIPPER_ARINTERFACE_IMPL_HPP
#define OPENFLIPPER_ARINTERFACE_IMPL_HPP

#include "ARInterface.h"
#include <memory>

#ifdef _USE_COMPRESSED_STREAMS
#include <compressed_streams/brotli_stream.h>
#include <compressed_streams/zstd_stream.h>
#include <compressed_streams/zlib_stream.h>
#include <compressed_streams/lz4_stream.h>
#include <compressed_streams/lzma_stream.h>
namespace cs = compressed_streams;
#endif // _USE_COMPRESSED_STREAMS

#ifdef _USE_ZSTDSTREAM
#include "zstdstream/zstdstream.h"
#endif // _USE_ZSTDSTREAM

#ifdef _USE_GZSTREAM
#include "gzstream/gzstream.h"
#endif  // _USE_GZSTREAM

namespace _INTERFACE_NAMESPACE {
// custom serialization
namespace MVArchive {
template<typename _Tp>
bool Save(ArchiveSave &a, const _Tp &obj) {
  const_cast<_Tp &>(obj).serialize(a, a.version);
  return true;
}
template<typename _Tp>
bool Load(ArchiveLoad &a, _Tp &obj) {
  obj.serialize(a, a.version);
  return true;
}

inline std::shared_ptr<std::ostream> create_ostream(const std::string &fileName,
                                                    std::ofstream &ofs,
                                                    const int &format) {
  switch (format) {
    case ArchiveFormat::STDIO: return std::make_shared<std::ofstream>(fileName, std::ios::out | std::ios::binary);
#ifdef _USE_GZSTREAM
    case ArchiveFormat::GZSTREAM:return std::make_shared<ogzstream>(fileName.c_str(), std::ios::out | std::ios::binary);
#endif
#ifdef _USE_ZSTDSTREAM
    case ArchiveFormat::ZSTDSTREAM:return std::make_shared<zstd::ofstream>(fileName, std::ios::out | std::ios::binary);
#endif
#ifdef _USE_COMPRESSED_STREAMS
      case ArchiveFormat::BROTLI: return std::make_shared<cs::BrotliOStream>(ofs);
      case ArchiveFormat::LZ4: return std::make_shared<cs::Lz4OStream>(ofs);
      case ArchiveFormat::LZMA: return std::make_shared<cs::LzmaOStream>(ofs);
      case ArchiveFormat::ZSTD: return std::make_shared<cs::ZstdOStream>(ofs);
      case ArchiveFormat::ZLIB: return std::make_shared<cs::ZlibOStream>(ofs);
#endif
    default:return nullptr;
  }
}

inline std::shared_ptr<std::istream> create_istream(const std::string &fileName, int *pFormat) {
  char szHeader[4];
  std::shared_ptr<std::istream> ifs;

  {
    // stdio
    *pFormat = ArchiveFormat::STDIO;
    ifs = std::make_shared<std::ifstream>(fileName, std::ios_base::in | std::ios_base::binary);
    if (!ifs->good())
      return nullptr;
    ifs->read(szHeader, 4);
    if (strncmp(szHeader, MVSI_PROJECT_ID, 4) == 0)
      return ifs;
  }
#ifdef _USE_GZSTREAM
  {
    // gzstream
    *pFormat = ArchiveFormat::GZSTREAM;
    ifs = std::make_shared<igzstream>(fileName.c_str(), std::ios::in | std::ios::binary);
    if (!ifs->good())
      return nullptr;
    ifs->read(szHeader, 4);
    if (strncmp(szHeader, MVSI_PROJECT_ID, 4) == 0)
      return ifs;
  }
#endif
#ifdef _USE_ZSTDSTREAM
  {
    //zstdstream
    *pFormat = ArchiveFormat::ZSTDSTREAM;
    ifs = std::make_shared<zstd::ifstream>(fileName.c_str(), std::ios::in | std::ios::binary);
    if (!ifs->good())
      return nullptr;
    ifs->read(szHeader, 4);
    if (strncmp(szHeader, MVSI_PROJECT_ID, 4) == 0)
      return ifs;
  }
#endif
#ifdef _USE_COMPRESSED_STREAMS
  {
    *pFormat = ArchiveFormat::BROTLI;
    auto baseifs = std::make_shared<std::ifstream>(fileName, std::ios::in | std::ios::binary);
    ifs = std::make_shared<cs::BrotliIStream>(*baseifs);
    if (!ifs->good())
      return nullptr;
    ifs->read(szHeader, 4);
    if (strncmp(szHeader, MVSI_PROJECT_ID, 4) == 0)
      return ifs;
  }
  {
    *pFormat = ArchiveFormat::LZ4;
    auto baseifs = std::make_shared<std::ifstream>(fileName, std::ios::in | std::ios::binary);
    ifs = std::make_shared<cs::Lz4IStream>(*baseifs);
    if (!ifs->good())
      return nullptr;
    ifs->read(szHeader, 4);
    if (strncmp(szHeader, MVSI_PROJECT_ID, 4) == 0)
      return ifs;
  }
  {
    *pFormat = ArchiveFormat::LZMA;
    auto baseifs = std::make_shared<std::ifstream>(fileName, std::ios::in | std::ios::binary);
    ifs = std::make_shared<cs::LzmaIStream>(*baseifs);
    if (!ifs->good())
      return nullptr;
    ifs->read(szHeader, 4);
    if (strncmp(szHeader, MVSI_PROJECT_ID, 4) == 0)
      return ifs;
  }
  {
    *pFormat = ArchiveFormat::ZLIB;
    auto baseifs = std::make_shared<std::ifstream>(fileName, std::ios::in | std::ios::binary);
    ifs = std::make_shared<cs::ZlibIStream>(*baseifs);
    if (!ifs->good())
      return nullptr;
    ifs->read(szHeader, 4);
    if (strncmp(szHeader, MVSI_PROJECT_ID, 4) == 0)
      return ifs;
  }
  {
    *pFormat = ArchiveFormat::ZSTD;
    auto baseifs = std::make_shared<std::ifstream>(fileName, std::ios::in | std::ios::binary);
    ifs = std::make_shared<cs::ZstdIStream>(*baseifs);
    if (!ifs->good())
      return nullptr;
    ifs->read(szHeader, 4);
    if (strncmp(szHeader, MVSI_PROJECT_ID, 4) == 0)
      return ifs;
  }
#endif
  *pFormat = -1;
  return nullptr;
}

// Main exporter & importer
template<typename _Tp>
bool SerializeSave(const _Tp &obj, const std::string &fileName, int format, uint32_t version) {
  // TODO:
  // ofs object has to be in this scope, otherwise, nothing can be written.
  std::ofstream ofs = std::ofstream(fileName, std::ios_base::out | std::ios_base::binary);
  auto stream = create_ostream(fileName, ofs, format);
  if (stream == nullptr)
    return false;
  if (!stream->good())
    return false;
  // write header
  if (version > 0) {
    // save project ID
    stream->write(MVSI_PROJECT_ID, 4);
    // save project version
    stream->write((const char *) &version, sizeof(uint32_t));
    // reserve some bytes
    const uint32_t reserved(0);
    stream->write((const char *) &reserved, sizeof(uint32_t));
  }

  // serialize out the current state
  MVArchive::ArchiveSave serializer(stream, version);
  try { serializer & obj; }
  catch (const std::bad_alloc &e) {
    return false;
  }
  return true;
}

template<typename _Tp>
bool SerializeLoad(_Tp &obj, const std::string &fileName, int *pFormat, uint32_t *pVersion) {
  // open the input stream
  int format;
  auto stream = create_istream(fileName, &format);
  if (stream == nullptr)
    return false;
  if (!stream->good())
    return false;
  // read header
  uint32_t version(0);
  // load project version
  stream->read((char *) &version, sizeof(uint32_t));
  if (!stream || version > MVSI_PROJECT_VER)
    return false;
  // skip reserved bytes
  uint32_t reserved;
  stream->read((char *) &reserved, sizeof(uint32_t));

  // serialize in the current state
  MVArchive::ArchiveLoad serializer(stream, version);

  try { serializer & obj; }
  catch (const std::bad_alloc &e) {
    return false;
  }
  obj.format = format;
  obj.filePath = fileName;
  if (pFormat)
    *pFormat = format;
  if (pVersion)
    *pVersion = version;
  return true;
}

#define ARCHIVE_DEFINE_TYPE(TYPE)                   \
  template <>                                       \
  bool Save<TYPE>(ArchiveSave & a, const TYPE& v) { \
    a.stream->write((const char*)&v, sizeof(TYPE));  \
    return true;                                    \
  }                                                 \
  template <>                                       \
  bool Load<TYPE>(ArchiveLoad & a, TYPE & v) {      \
    a.stream->read((char*)&v, sizeof(TYPE));         \
    return true;                                    \
  }

ARCHIVE_DEFINE_TYPE(bool)
ARCHIVE_DEFINE_TYPE(int)
ARCHIVE_DEFINE_TYPE(uint8_t)
ARCHIVE_DEFINE_TYPE(uint16_t)
ARCHIVE_DEFINE_TYPE(uint32_t)
ARCHIVE_DEFINE_TYPE(uint64_t)
ARCHIVE_DEFINE_TYPE(float)
ARCHIVE_DEFINE_TYPE(double)

template<typename _Tp>
bool Save(ArchiveSave &a, const _Tp &obj);

template<typename _Tp>
bool Load(ArchiveLoad &a, _Tp &obj);

template<typename _Tp>
ArchiveSave &ArchiveSave::operator&(const _Tp &obj) {
  Save(*this, obj);
  return *this;
}
template<typename _Tp>
ArchiveLoad &ArchiveLoad::operator&(_Tp &obj) {
  Load(*this, obj);
  return *this;
}

// Serialization support for cv::Matx
template<typename _Tp, int m, int n>
bool Save(ArchiveSave &a, const cv::Matx<_Tp, m, n> &_m);
template<typename _Tp, int m, int n>
bool Load(ArchiveLoad &a, cv::Matx<_Tp, m, n> &_m);

// Serialization support for cv::Point3_
template<typename _Tp>
bool Save(ArchiveSave &a, const cv::Point3_<_Tp> &pt);
template<typename _Tp>
bool Load(ArchiveLoad &a, cv::Point3_<_Tp> &pt);

// Serialization support for cv::Point_
template<typename _Tp>
bool Save(ArchiveSave &a, const cv::Point_<_Tp> &pt);
template<typename _Tp>
bool Load(ArchiveLoad &a, cv::Point_<_Tp> &pt);

// Serialization support for std::string
template<>
bool Save<std::string>(ArchiveSave &a, const std::string &s);
template<>
bool Load<std::string>(ArchiveLoad &a, std::string &s);

// Serialization support for std::vector
template<typename _Tp>
bool Save(ArchiveSave &a, const std::vector<_Tp> &v);
template<typename _Tp>
bool Load(ArchiveLoad &a, std::vector<_Tp> &v);

// Serialization support for cv::Matx
template<typename _Tp, int m, int n>
bool Save(ArchiveSave &a, const cv::Matx<_Tp, m, n> &_m) {
  a.stream->write((const char *) _m.val, sizeof(_Tp) * m * n);
  return true;
}
template<typename _Tp, int m, int n>
bool Load(ArchiveLoad &a, cv::Matx<_Tp, m, n> &_m) {
  a.stream->read((char *) _m.val, sizeof(_Tp) * m * n);
  return true;
}

// Serialization support for cv::Point3_
template<typename _Tp>
bool Save(ArchiveSave &a, const cv::Point3_<_Tp> &pt) {
  a.stream->write((const char *) &pt.x, sizeof(_Tp) * 3);
  return true;
}
template<typename _Tp>
bool Load(ArchiveLoad &a, cv::Point3_<_Tp> &pt) {
  a.stream->read((char *) &pt.x, sizeof(_Tp) * 3);
  return true;
}

// Serialization support for cv::Point_
template<typename _Tp>
bool Save(ArchiveSave &a, const cv::Point_<_Tp> &pt) {
  a.stream->write((const char *) &pt.x, sizeof(_Tp) * 2);
  return true;
}
template<typename _Tp>
bool Load(ArchiveLoad &a, cv::Point_<_Tp> &pt) {
  a.stream->read((char *) &pt.x, sizeof(_Tp) * 2);
  return true;
}

// Serialization support for std::string
template<>
bool Save<std::string>(ArchiveSave &a, const std::string &s) {
  const uint64_t size(s.size());
  Save(a, size);
  if (size > 0)
    a.stream->write(&s[0], sizeof(char) * size);
  return true;
}
template<>
bool Load<std::string>(ArchiveLoad &a, std::string &s) {
  uint64_t size;
  Load(a, size);
  if (size > 0) {
    s.resize(size);
    a.stream->read(&s[0], sizeof(char) * size);
  }
  return true;
}

// Serialization support for std::vector
template<typename _Tp>
bool Save(ArchiveSave &a, const std::vector<_Tp> &v) {
  const uint64_t size(v.size());
  Save(a, size);
  for (uint64_t i = 0; i < size; ++i)
    Save(a, v[i]);
  return true;
}
template<typename _Tp>
bool Load(ArchiveLoad &a, std::vector<_Tp> &v) {
  uint64_t size;
  Load(a, size);
  if (size > 0) {
    v.resize(size);
    for (uint64_t i = 0; i < size; ++i)
      Load(a, v[i]);
  }
  return true;
}

}// namespace MVArchive
}// namespace _INTERFACE_NAMESPACE

#endif //OPENFLIPPER_ARINTERFACE_IMPL_HPP
