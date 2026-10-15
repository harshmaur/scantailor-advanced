// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#include <CylindricalSurfaceDewarper.h>
#include <RasterDewarper.h>

#include <QColor>
#include <QImage>
#include <QRectF>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <random>
#include <vector>

#include "Internal.h"
#include "TestImages.h"

namespace gpu {
namespace tests {
using dewarping::CylindricalSurfaceDewarper;
using dewarping::RasterDewarper;

namespace {
struct DewarpSettingsFixture {
  DewarpSettingsFixture() {
    setEnabled(true);
    testing::setMinPixels(0);
  }

  ~DewarpSettingsFixture() {
    setEnabled(true);
    testing::setMinPixels(-1);
    testing::setMaxBufferBytes(0);
  }
};

struct Model {
  std::vector<QPointF> top;
  std::vector<QPointF> bottom;
  double depth;
};

std::vector<QPointF> curve(const QSizeF& size, const double y, const double sag, const double tilt) {
  std::vector<QPointF> points;
  const int count = 9;
  for (int i = 0; i < count; ++i) {
    const double t = double(i) / (count - 1);
    points.emplace_back((0.05 + 0.9 * t) * size.width(),
                        (y + sag * 4 * t * (1 - t) + tilt * (t - 0.5)) * size.height());
  }
  return points;
}

// A page bent towards the spine, as produced by the output stage.
Model pageModel(const QSizeF& size, const double sag = 0.03, const double tilt = 0.01, const double depth = 2.0) {
  return {curve(size, 0.08, sag, tilt), curve(size, 0.92, -sag * 0.6, tilt * 0.5), depth};
}

std::vector<Model> sampleModels(const QSizeF& size) {
  return {pageModel(size),
          pageModel(size, 0.0, 0.0, 1.0),
          pageModel(size, -0.05, 0.02, 1.5),
          pageModel(size, 0.08, -0.03, 3.0),
          {{{0.1 * size.width(), 0.1 * size.height()}, {0.9 * size.width(), 0.15 * size.height()}},
           {{0.12 * size.width(), 0.85 * size.height()}, {0.88 * size.width(), 0.95 * size.height()}},
           2.0}};
}

QImage dewarp(const QImage& src, const QSize& dstSize, const Model& model, const QRectF& domain, const QColor& bg) {
  const CylindricalSurfaceDewarper dewarper(model.top, model.bottom, model.depth);
  return RasterDewarper::dewarp(src, dstSize, dewarper, domain, bg);
}

void checkGpuMatchesCpu(const QImage& src,
                        const QSize& dstSize,
                        const Model& model,
                        const QRectF& domain,
                        const QColor& bg,
                        const int minDispatches = 1) {
  setEnabled(false);
  const QImage cpu = dewarp(src, dstSize, model, domain, bg);
  setEnabled(true);
  const qint64 dispatches = testing::dispatchCount();
  const QImage gpu = dewarp(src, dstSize, model, domain, bg);

  BOOST_TEST_CONTEXT("src " << src.width() << "x" << src.height() << " format " << src.format() << " dst "
                            << dstSize.width() << "x" << dstSize.height() << " depth " << model.depth << " bg "
                            << bg.name(QColor::HexArgb).toStdString()) {
    BOOST_CHECK_GE(testing::dispatchCount() - dispatches, minDispatches);
    BOOST_REQUIRE_EQUAL(cpu.format(), gpu.format());
    BOOST_REQUIRE(cpu.size() == gpu.size());
    BOOST_CHECK_EQUAL(compareImages(cpu, gpu).mismatches, 0);
  }
}

const std::vector<QImage::Format> allFormats = {QImage::Format_Indexed8, QImage::Format_RGB32, QImage::Format_ARGB32};

const std::vector<QColor> backgrounds = {Qt::white, QColor(10, 200, 30), QColor(10, 20, 30, 128)};
}  // namespace

BOOST_AUTO_TEST_SUITE(GpuDewarpEnvironmentTestSuite)

BOOST_FIXTURE_TEST_CASE(test_dewarp_env_off_uses_cpu, DewarpSettingsFixture) {
  const EnvVar env("SCANTAILOR_GPU", "0");
  const QImage src = textLikeImage(QImage::Format_RGB32, 120, 90, 1);
  const Model model = pageModel(src.size());
  const qint64 dispatches = testing::dispatchCount();
  const QImage result = dewarp(src, src.size(), model, QRectF(5, 5, 110, 80), Qt::white);
  BOOST_CHECK_EQUAL(testing::dispatchCount(), dispatches);
  setEnabled(false);
  const QImage cpu = dewarp(src, src.size(), model, QRectF(5, 5, 110, 80), Qt::white);
  BOOST_CHECK_EQUAL(compareImages(cpu, result).mismatches, 0);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(GpuDewarpTestSuite, *boost::unit_test::precondition(GpuAvailable()))

BOOST_FIXTURE_TEST_CASE(test_dewarp_min_size_uses_cpu, DewarpSettingsFixture) {
  testing::setMinPixels(10000);
  const QImage src = randomImage(QImage::Format_RGB32, 60, 60, 3);
  const qint64 dispatches = testing::dispatchCount();
  dewarp(src, src.size(), pageModel(src.size()), QRectF(0, 0, 60, 60), Qt::white);
  BOOST_CHECK_EQUAL(testing::dispatchCount(), dispatches);
}

BOOST_FIXTURE_TEST_CASE(test_dewarp_odd_sizes, DewarpSettingsFixture) {
  const QSize sizes[] = {{1, 1}, {7, 13}, {13, 7}, {33, 17}, {101, 67}};
  for (const QSize& size : sizes) {
    for (const QImage::Format format : allFormats) {
      const QImage src = randomImage(format, size.width(), size.height(), size.width() * 31 + size.height());
      const Model model = pageModel(QSizeF(size));
      checkGpuMatchesCpu(src, size, model, QRectF(QPointF(0, 0), QSizeF(size)), Qt::white);
      checkGpuMatchesCpu(src, size + QSize(3, 2), model, QRectF(1, 1, size.width(), size.height()), Qt::white);
    }
  }
}

BOOST_FIXTURE_TEST_CASE(test_dewarp_formats_models_and_backgrounds, DewarpSettingsFixture) {
  uint32_t seed = 100;
  for (const QImage::Format format : allFormats) {
    const QImage random = randomImage(format, 97, 71, ++seed);
    const QImage text = textLikeImage(format, 283, 211, ++seed);
    for (const QImage& src : {random, text}) {
      for (const Model& model : sampleModels(src.size())) {
        for (const QColor& bg : backgrounds) {
          const QSize dstSize = src.size() + QSize(20, 10);
          checkGpuMatchesCpu(src, dstSize, model, QRectF(10, 5, src.width(), src.height()), bg);
          checkGpuMatchesCpu(src, src.size() / 2, model, QRectF(QPointF(0, 0), QSizeF(src.size() / 2)), bg);
        }
      }
    }
  }
}

BOOST_FIXTURE_TEST_CASE(test_dewarp_converted_sources, DewarpSettingsFixture) {
  const QImage gray = textLikeImage(QImage::Format_Indexed8, 151, 113, 200);
  const QImage rgb = textLikeImage(QImage::Format_RGB32, 151, 113, 201);
  for (const QImage& src : {gray.convertToFormat(QImage::Format_Mono), rgb.convertToFormat(QImage::Format_RGB888),
                            rgb.convertToFormat(QImage::Format_ARGB32_Premultiplied)}) {
    checkGpuMatchesCpu(src, src.size(), pageModel(src.size()), QRectF(QPointF(0, 0), QSizeF(src.size())), Qt::white);
  }
}

BOOST_FIXTURE_TEST_CASE(test_dewarp_random_models, DewarpSettingsFixture) {
  std::mt19937 rng(500);
  std::uniform_real_distribution<double> sag(-0.1, 0.1);
  std::uniform_real_distribution<double> tilt(-0.05, 0.05);
  std::uniform_real_distribution<double> depth(1.0, 3.0);
  std::uniform_real_distribution<double> scale(0.5, 2.0);
  std::uniform_real_distribution<double> offset(-20.0, 20.0);
  for (int i = 0; i < 30; ++i) {
    const QImage::Format format = allFormats[i % allFormats.size()];
    const QImage src
        = (i % 2) ? randomImage(format, 57 + i, 49 + 2 * i, i) : textLikeImage(format, 150 + 3 * i, 120 + i, i);
    const Model model = pageModel(src.size(), sag(rng), tilt(rng), depth(rng));
    const QSizeF domainSize = QSizeF(src.size()) * scale(rng);
    const QRectF domain(QPointF(offset(rng), offset(rng)), domainSize);
    const QSize dstSize = domainSize.toSize() + QSize(15, 11);
    checkGpuMatchesCpu(src, dstSize, model, domain, backgrounds[i % backgrounds.size()]);
  }
}

BOOST_FIXTURE_TEST_CASE(test_dewarp_extreme_models, DewarpSettingsFixture) {
  const QImage src = textLikeImage(QImage::Format_RGB32, 200, 160, 300);
  const Model model = pageModel(src.size(), 0.1, 0.05, 3.0);
  checkGpuMatchesCpu(src, QSize(180, 150), model, QRectF(85, 70, 2, 2), Qt::white);
  checkGpuMatchesCpu(src, QSize(180, 150), model, QRectF(-5000, -4000, 10000, 9000), Qt::white);

  const qint64 recomputed = testing::recomputedCount();
  checkGpuMatchesCpu(src, QSize(3, 2), model, QRectF(0, 0, 3, 2), QColor(10, 200, 30));
  BOOST_CHECK_GT(testing::recomputedCount(), recomputed);
}

BOOST_FIXTURE_TEST_CASE(test_dewarp_banding, DewarpSettingsFixture) {
  uint32_t seed = 600;
  for (const QImage::Format format : allFormats) {
    const QImage src = textLikeImage(format, 301, 211, ++seed);
    const QSize dstSize(311, 221);
    const qint64 gridRowBytes = qint64(dstSize.width() + 1) * 8;
    for (const int rows : {20, 45}) {
      testing::setMaxBufferBytes(std::max<qint64>(rows * gridRowBytes, qint64(rows) * src.bytesPerLine()));
      for (const Model& model : sampleModels(src.size())) {
        checkGpuMatchesCpu(src, dstSize, model, QRectF(5, 5, src.width(), src.height()), Qt::white, 2);
      }
    }
  }
}

BOOST_FIXTURE_TEST_CASE(test_dewarp_band_too_small_falls_back, DewarpSettingsFixture) {
  const QImage src = randomImage(QImage::Format_RGB32, 64, 64, 700);
  testing::setMaxBufferBytes(64);
  checkGpuMatchesCpu(src, src.size(), pageModel(src.size()), QRectF(0, 0, 64, 64), Qt::white, 0);
}

BOOST_FIXTURE_TEST_CASE(test_dewarp_realistic_models_need_no_cpu_columns, DewarpSettingsFixture) {
  const QImage src = textLikeImage(QImage::Format_RGB32, 411, 533, 800);
  for (const Model& model : sampleModels(src.size())) {
    const qint64 recomputed = testing::recomputedCount();
    checkGpuMatchesCpu(src, src.size(), model, QRectF(QPointF(0, 0), QSizeF(src.size())), Qt::white);
    BOOST_CHECK_EQUAL(testing::recomputedCount(), recomputed);
  }
}

BOOST_AUTO_TEST_SUITE_END()
}  // namespace tests
}  // namespace gpu
