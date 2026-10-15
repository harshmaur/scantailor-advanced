// Copyright (C) 2019  Joseph Artsimovich <joseph.artsimovich@gmail.com>, 4lex4 <4lex49@zoho.com>
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#include <GrayImage.h>
#include <Scale.h>

#include <QImage>
#include <QSize>
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "Utils.h"

namespace imageproc {
namespace tests {
using namespace utils;

BOOST_AUTO_TEST_SUITE(ScaleTestSuite)

BOOST_AUTO_TEST_CASE(test_null_image) {
  const GrayImage nullImg;
  BOOST_CHECK(scaleToGray(nullImg, QSize(1, 1)).isNull());
}

// Exact area-averaging (box filter) downscale in floating point.
// Deliberately independent of QImage::scaled(), whose smooth scaling path
// depends on the Qt version and the image format.
static GrayImage boxScaleDown(const GrayImage& src, const QSize& dstSize) {
  const int sw = src.width();
  const int sh = src.height();
  const int dw = dstSize.width();
  const int dh = dstSize.height();
  const double xRatio = static_cast<double>(sw) / dw;
  const double yRatio = static_cast<double>(sh) / dh;

  GrayImage dst(dstSize);
  for (int dy = 0; dy < dh; ++dy) {
    const double top = dy * yRatio;
    const double bottom = (dy + 1) * yRatio;
    for (int dx = 0; dx < dw; ++dx) {
      const double left = dx * xRatio;
      const double right = (dx + 1) * xRatio;
      double sum = 0.0;
      for (int sy = static_cast<int>(top); sy < sh && sy < bottom; ++sy) {
        const double wy = std::min<double>(sy + 1, bottom) - std::max<double>(sy, top);
        for (int sx = static_cast<int>(left); sx < sw && sx < right; ++sx) {
          const double wx = std::min<double>(sx + 1, right) - std::max<double>(sx, left);
          sum += wx * wy * src.data()[sy * src.stride() + sx];
        }
      }
      dst.data()[dy * dst.stride() + dx] = static_cast<uint8_t>(std::lround(sum / (xRatio * yRatio)));
    }
  }
  return dst;
}

static bool fuzzyCompare(const GrayImage& img1, const GrayImage& img2) {
  BOOST_REQUIRE(img1.size() == img2.size());

  const int width = img1.width();
  const int height = img1.height();
  const uint8_t* line1 = img1.data();
  const uint8_t* line2 = img2.data();
  const int line1Stride = img1.stride();
  const int line2Stride = img2.stride();

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (std::abs(int(line1[x]) - int(line2[x])) > 1) {
        return false;
      }
    }
    line1 += line1Stride;
    line2 += line2Stride;
  }
  return true;
}

static bool checkScale(const GrayImage& img, const QSize& newSize) {
  return fuzzyCompare(scaleToGray(img, newSize), boxScaleDown(img, newSize));
}

BOOST_AUTO_TEST_CASE(test_random_image) {
  GrayImage img(QSize(100, 100));
  uint8_t* line = img.data();
  for (int y = 0; y < img.height(); ++y) {
    for (int x = 0; x < img.width(); ++x) {
      line[x] = static_cast<uint8_t>(rand() % 256);
    }
    line += img.stride();
  }

  // Only downscaling is checked: it is plain area averaging.
  // QImage::scaled() is not used as the reference because Qt 6 applies area
  // averaging to Indexed8 images only for downscaling by more than 2x (or for
  // very large images) and uses bilinear sampling otherwise.
  // scaleToGray() snaps source coordinates to 1/32 pixel, so the sizes are
  // chosen to give source/destination ratios that are multiples of 1/32.

  BOOST_CHECK(checkScale(img, QSize(50, 50)));
  BOOST_CHECK(checkScale(img, QSize(80, 80)));
  BOOST_CHECK(checkScale(img, QSize(25, 25)));
  BOOST_CHECK(checkScale(img, QSize(80, 50)));
}

BOOST_AUTO_TEST_SUITE_END()
}  // namespace tests
}  // namespace imageproc