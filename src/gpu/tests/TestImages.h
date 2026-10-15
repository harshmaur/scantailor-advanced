// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#ifndef SCANTAILOR_GPU_TESTS_TESTIMAGES_H_
#define SCANTAILOR_GPU_TESTS_TESTIMAGES_H_

#include <QByteArray>
#include <QImage>
#include <boost/test/detail/global_typedef.hpp>
#include <boost/test/tools/assertion_result.hpp>
#include <cstdint>

#include "Gpu.h"

namespace gpu {
namespace tests {
/**
 * Random pixels. Format_Indexed8 gives a grayscale image.
 */
QImage randomImage(QImage::Format format, int width, int height, uint32_t seed);

/**
 * A light page with dark text-like strokes arranged in lines.
 * Format_Indexed8 gives a grayscale image.
 */
QImage textLikeImage(QImage::Format format, int width, int height, uint32_t seed);

struct ImageDiff {
  qint64 mismatches = 0;
  int maxChannelDiff = 0;
};

/**
 * Compares the visible pixels of two images of the same size and format.
 */
ImageDiff compareImages(const QImage& lhs, const QImage& rhs);

struct GpuAvailable {
  boost::test_tools::assertion_result operator()(boost::unit_test::test_unit_id) const {
    boost::test_tools::assertion_result result(isAvailable());
    if (!result) {
      result.message() << "no GPU device with compute support, skipping the GPU tests";
    }
    return result;
  }
};

class EnvVar {
 public:
  EnvVar(const char* name, const QByteArray& value) : m_name(name), m_old(qgetenv(name)) { qputenv(name, value); }

  ~EnvVar() {
    if (m_old.isNull()) {
      qunsetenv(m_name);
    } else {
      qputenv(m_name, m_old);
    }
  }

 private:
  const char* m_name;
  QByteArray m_old;
};
}  // namespace tests
}  // namespace gpu
#endif  // SCANTAILOR_GPU_TESTS_TESTIMAGES_H_
