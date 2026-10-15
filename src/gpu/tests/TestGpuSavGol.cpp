// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#include <SavGolFilter.h>

#include <QImage>
#include <boost/test/unit_test.hpp>

#include "Gpu.h"
#include "TestImages.h"

namespace gpu {
namespace tests {
namespace {
struct GpuAvailable {
  boost::test_tools::assertion_result operator()(boost::unit_test::test_unit_id) const {
    boost::test_tools::assertion_result result(isAvailable());
    if (!result) {
      result.message() << "no GPU device with compute support, skipping the GPU tests";
    }
    return result;
  }
};

struct SavGolSettingsFixture {
  SavGolSettingsFixture() {
    setEnabled(true);
    testing::setMinSavGolPixels(0);
  }

  ~SavGolSettingsFixture() {
    setEnabled(true);
    testing::setMinSavGolPixels(-1);
    testing::setMaxBufferBytes(0);
  }
};

struct Window {
  int size;
  int horDegree;
  int vertDegree;
};

// The windows and degrees smoothToGrayscale() uses, then a few others.
const Window appWindows[] = {{5, 3, 3}, {7, 4, 4}, {11, 4, 4}, {11, 2, 2}};
const Window otherWindows[] = {{1, 0, 0}, {3, 1, 1}, {9, 0, 0}, {13, 5, 3}, {21, 4, 4}, {32, 2, 6}};

QImage whitePage(const int width, const int height) {
  QImage image = textLikeImage(QImage::Format_Indexed8, width, height, 1);
  image.fill(255);
  for (int y = 3; y + 6 < height; y += 13) {
    for (int x = (y % 7); x + 4 < width; x += 9) {
      for (int dy = 0; dy < 5; ++dy) {
        for (int dx = 0; dx < 3; ++dx) {
          image.scanLine(y + dy)[x + dx] = 0;
        }
      }
    }
  }
  return image;
}

void checkGpuMatchesCpu(const QImage& src, const QSize& window, const int horDegree, const int vertDegree) {
  setEnabled(false);
  const QImage cpu = imageproc::savGolFilter(src, window, horDegree, vertDegree);
  setEnabled(true);
  const qint64 dispatches = testing::dispatchCount();
  const QImage gpu = imageproc::savGolFilter(src, window, horDegree, vertDegree);

  BOOST_TEST_CONTEXT("src " << src.width() << "x" << src.height() << " window " << window.width() << "x"
                            << window.height() << " degrees " << horDegree << " " << vertDegree) {
    const bool fits = (window.width() <= src.width()) && (window.height() <= src.height());
    BOOST_CHECK_EQUAL(testing::dispatchCount() > dispatches, fits);
    BOOST_REQUIRE_EQUAL(cpu.format(), gpu.format());
    BOOST_REQUIRE(cpu.size() == gpu.size());
    BOOST_CHECK_EQUAL(compareImages(cpu, gpu).mismatches, 0);
  }
}

void checkGpuMatchesCpu(const QImage& src, const Window& window) {
  checkGpuMatchesCpu(src, QSize(window.size, window.size), window.horDegree, window.vertDegree);
}
}  // namespace

BOOST_AUTO_TEST_SUITE(GpuSavGolEnvironmentTestSuite)

BOOST_FIXTURE_TEST_CASE(test_savgol_env_off_uses_cpu, SavGolSettingsFixture) {
  const QByteArray old = qgetenv("SCANTAILOR_GPU");
  qputenv("SCANTAILOR_GPU", "0");
  const QImage src = textLikeImage(QImage::Format_Indexed8, 101, 67, 1);
  const qint64 dispatches = testing::dispatchCount();
  const QImage result = imageproc::savGolFilter(src, QSize(11, 11), 4, 4);
  BOOST_CHECK_EQUAL(testing::dispatchCount(), dispatches);
  if (old.isNull()) {
    qunsetenv("SCANTAILOR_GPU");
  } else {
    qputenv("SCANTAILOR_GPU", old);
  }

  setEnabled(false);
  BOOST_CHECK_EQUAL(compareImages(imageproc::savGolFilter(src, QSize(11, 11), 4, 4), result).mismatches, 0);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(GpuSavGolTestSuite, *boost::unit_test::precondition(GpuAvailable()))

BOOST_FIXTURE_TEST_CASE(test_savgol_min_size_uses_cpu, SavGolSettingsFixture) {
  testing::setMinSavGolPixels(10000);
  const QImage src = randomImage(QImage::Format_Indexed8, 99, 99, 2);
  const qint64 dispatches = testing::dispatchCount();
  imageproc::savGolFilter(src, QSize(5, 5), 3, 3);
  BOOST_CHECK_EQUAL(testing::dispatchCount(), dispatches);
}

BOOST_FIXTURE_TEST_CASE(test_savgol_app_windows, SavGolSettingsFixture) {
  uint32_t seed = 10;
  for (const Window& window : appWindows) {
    checkGpuMatchesCpu(randomImage(QImage::Format_Indexed8, 211, 157, ++seed), window);
    checkGpuMatchesCpu(textLikeImage(QImage::Format_Indexed8, 301, 233, ++seed), window);
    checkGpuMatchesCpu(whitePage(173, 129), window);
  }
}

BOOST_FIXTURE_TEST_CASE(test_savgol_other_windows, SavGolSettingsFixture) {
  uint32_t seed = 20;
  for (const Window& window : otherWindows) {
    checkGpuMatchesCpu(randomImage(QImage::Format_Indexed8, 97, 83, ++seed), window);
    checkGpuMatchesCpu(textLikeImage(QImage::Format_Indexed8, 131, 89, ++seed), window);
  }
  const QImage src = textLikeImage(QImage::Format_Indexed8, 77, 61, 30);
  checkGpuMatchesCpu(src, QSize(7, 3), 2, 1);
  checkGpuMatchesCpu(src, QSize(3, 9), 0, 4);
  checkGpuMatchesCpu(src, QSize(15, 5), 6, 2);
}

BOOST_FIXTURE_TEST_CASE(test_savgol_odd_and_small_sizes, SavGolSettingsFixture) {
  const QSize sizes[] = {{1, 1}, {4, 50}, {50, 4}, {10, 10}, {11, 11}, {12, 11}, {11, 12}, {13, 7}, {65, 17}, {67, 33}};
  for (const QSize& size : sizes) {
    const QImage src
        = randomImage(QImage::Format_Indexed8, size.width(), size.height(), size.width() * 31 + size.height());
    for (const Window& window : appWindows) {
      checkGpuMatchesCpu(src, window);
    }
  }
}

BOOST_FIXTURE_TEST_CASE(test_savgol_color_source, SavGolSettingsFixture) {
  checkGpuMatchesCpu(textLikeImage(QImage::Format_RGB32, 123, 97, 40), QSize(11, 11), 4, 4);
}

BOOST_FIXTURE_TEST_CASE(test_savgol_banding, SavGolSettingsFixture) {
  const QImage text = textLikeImage(QImage::Format_Indexed8, 301, 211, 50);
  const QImage white = whitePage(157, 203);
  for (const int rowsPerBuffer : {11, 12, 17, 40}) {
    for (const QImage& src : {text, white}) {
      testing::setMaxBufferBytes(qint64(rowsPerBuffer) * src.bytesPerLine());
      const qint64 dispatches = testing::dispatchCount();
      checkGpuMatchesCpu(src, QSize(11, 11), 4, 4);
      BOOST_CHECK_GT(testing::dispatchCount() - dispatches, 2);
    }
  }
}

BOOST_FIXTURE_TEST_CASE(test_savgol_band_too_small_falls_back, SavGolSettingsFixture) {
  const QImage src = randomImage(QImage::Format_Indexed8, 64, 64, 60);
  testing::setMaxBufferBytes(10 * 64);
  setEnabled(false);
  const QImage cpu = imageproc::savGolFilter(src, QSize(11, 11), 4, 4);
  setEnabled(true);
  const qint64 dispatches = testing::dispatchCount();
  const QImage result = imageproc::savGolFilter(src, QSize(11, 11), 4, 4);
  BOOST_CHECK_EQUAL(testing::dispatchCount(), dispatches);
  BOOST_CHECK_EQUAL(compareImages(cpu, result).mismatches, 0);
}

BOOST_FIXTURE_TEST_CASE(test_savgol_large_window_uses_cpu, SavGolSettingsFixture) {
  const QImage src = textLikeImage(QImage::Format_Indexed8, 90, 80, 70);
  setEnabled(false);
  const QImage cpu = imageproc::savGolFilter(src, QSize(33, 33), 2, 2);
  setEnabled(true);
  const qint64 dispatches = testing::dispatchCount();
  const QImage result = imageproc::savGolFilter(src, QSize(33, 33), 2, 2);
  BOOST_CHECK_EQUAL(testing::dispatchCount(), dispatches);
  BOOST_CHECK_EQUAL(compareImages(cpu, result).mismatches, 0);
}

BOOST_AUTO_TEST_SUITE_END()
}  // namespace tests
}  // namespace gpu
