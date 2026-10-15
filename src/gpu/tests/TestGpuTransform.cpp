// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#include <GrayImage.h>
#include <Transform.h>

#include <QImage>
#include <QTransform>
#include <boost/test/unit_test.hpp>
#include <random>
#include <vector>

#include "Gpu.h"
#include "TestImages.h"

namespace gpu {
namespace tests {
using imageproc::OutsidePixels;

namespace {
struct GpuAvailable {
  boost::test_tools::assertion_result operator()(boost::unit_test::test_unit_id) const {
    boost::test_tools::assertion_result result(isAvailable());
    if (!result) {
      result.message() << "no GPU device with compute support, skipping the GPU tests";
    } else {
      BOOST_TEST_MESSAGE("GPU device: " << deviceName().toStdString());
    }
    return result;
  }
};

struct GpuSettingsFixture {
  GpuSettingsFixture() {
    setEnabled(true);
    testing::setMinTransformPixels(0);
  }

  ~GpuSettingsFixture() {
    setEnabled(true);
    testing::setMinTransformPixels(-1);
    testing::setMaxBufferBytes(0);
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

enum class Mode { Transform, TransformToGray };

QImage run(const Mode mode,
           const QImage& src,
           const QTransform& xform,
           const QRect& dstRect,
           const OutsidePixels& outside,
           const QSizeF& minMappingArea) {
  if (mode == Mode::Transform) {
    return imageproc::transform(src, xform, dstRect, outside, minMappingArea);
  }
  return imageproc::transformToGray(src, xform, dstRect, outside, minMappingArea).toQImage();
}

void checkGpuMatchesCpu(const Mode mode,
                        const QImage& src,
                        const QTransform& xform,
                        const QRect& dstRect,
                        const OutsidePixels& outside,
                        const QSizeF& minMappingArea = QSizeF(0.9, 0.9),
                        const int minDispatches = 1) {
  setEnabled(false);
  const QImage cpu = run(mode, src, xform, dstRect, outside, minMappingArea);
  setEnabled(true);
  const qint64 dispatches = testing::dispatchCount();
  const QImage gpu = run(mode, src, xform, dstRect, outside, minMappingArea);

  BOOST_TEST_CONTEXT("src " << src.width() << "x" << src.height() << " format " << src.format() << " dst "
                            << dstRect.width() << "x" << dstRect.height() << " flags " << outside.flags())
  BOOST_TEST_CONTEXT("xform " << xform.m11() << " " << xform.m12() << " " << xform.m21() << " " << xform.m22() << " "
                              << xform.dx() << " " << xform.dy()) {
    BOOST_CHECK_GE(testing::dispatchCount() - dispatches, minDispatches);
    BOOST_REQUIRE_EQUAL(cpu.format(), gpu.format());
    BOOST_REQUIRE(cpu.size() == gpu.size());
    const ImageDiff diff = compareImages(cpu, gpu);
    if (cpu.format() == QImage::Format_ARGB32) {
      BOOST_CHECK_LE(diff.maxChannelDiff, 1);
    } else {
      BOOST_CHECK_EQUAL(diff.mismatches, 0);
    }
  }
}

QRect mappedRect(const QImage& src, const QTransform& xform, const int margin = 0) {
  return xform.mapRect(QRectF(src.rect())).toAlignedRect().adjusted(-margin, -margin, margin, margin);
}

std::vector<OutsidePixels> allOutsideModes() {
  return {OutsidePixels::assumeColor(Qt::white),
          OutsidePixels::assumeColor(QColor(10, 200, 30)),
          OutsidePixels::assumeColor(QColor(10, 20, 30, 128)),
          OutsidePixels::assumeWeakColor(Qt::black),
          OutsidePixels::assumeWeakColor(QColor(250, 20, 30)),
          OutsidePixels::assumeWeakNearest()};
}

std::vector<QTransform> sampleTransforms() {
  return {QTransform(),
          QTransform().rotate(1.7),
          QTransform().rotate(-3.3).translate(0.25, -0.75),
          QTransform().rotate(45),
          QTransform().rotate(90),
          QTransform().rotate(180),
          QTransform().rotate(270),
          QTransform().scale(2.5, 2.5),
          QTransform().scale(0.3, 0.3),
          QTransform().scale(1.0 / 3, 1.0 / 3),
          QTransform().scale(1.5, 0.7).rotate(-12),
          QTransform().rotate(90).scale(1.5, 1.5).rotate(0.4)};
}

const std::vector<QImage::Format> allFormats = {QImage::Format_Indexed8, QImage::Format_RGB32, QImage::Format_ARGB32};
}  // namespace

BOOST_AUTO_TEST_SUITE(GpuEnvironmentTestSuite)

BOOST_FIXTURE_TEST_CASE(test_env_off_uses_cpu, GpuSettingsFixture) {
  const EnvVar env("SCANTAILOR_GPU", "0");
  BOOST_CHECK(!isEnabled());
  BOOST_CHECK(!isAvailable());

  const QImage src = textLikeImage(QImage::Format_RGB32, 101, 67, 1);
  const QTransform xform = QTransform().rotate(1.7);
  const QRect dstRect = mappedRect(src, xform);
  const OutsidePixels outside = OutsidePixels::assumeColor(Qt::white);
  const qint64 dispatches = testing::dispatchCount();
  const QImage result = imageproc::transform(src, xform, dstRect, outside);
  BOOST_CHECK_EQUAL(testing::dispatchCount(), dispatches);

  setEnabled(false);
  const QImage cpu = imageproc::transform(src, xform, dstRect, outside);
  BOOST_CHECK_EQUAL(compareImages(cpu, result).mismatches, 0);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(GpuTransformTestSuite, *boost::unit_test::precondition(GpuAvailable()))

BOOST_FIXTURE_TEST_CASE(test_env_on_overrides_setting, GpuSettingsFixture) {
  const EnvVar env("SCANTAILOR_GPU", "1");
  setEnabled(false);
  BOOST_CHECK(isEnabled());
  const QImage src = randomImage(QImage::Format_RGB32, 33, 17, 2);
  const qint64 dispatches = testing::dispatchCount();
  imageproc::transform(src, QTransform().rotate(5), src.rect(), OutsidePixels::assumeColor(Qt::white));
  BOOST_CHECK_GT(testing::dispatchCount(), dispatches);
}

BOOST_FIXTURE_TEST_CASE(test_min_size_uses_cpu, GpuSettingsFixture) {
  testing::setMinTransformPixels(1000);
  const QImage src = randomImage(QImage::Format_RGB32, 20, 20, 3);
  const qint64 dispatches = testing::dispatchCount();
  imageproc::transform(src, QTransform().rotate(5), src.rect(), OutsidePixels::assumeColor(Qt::white));
  BOOST_CHECK_EQUAL(testing::dispatchCount(), dispatches);
}

BOOST_FIXTURE_TEST_CASE(test_odd_sizes, GpuSettingsFixture) {
  const QSize sizes[] = {{1, 1}, {7, 13}, {13, 7}, {33, 17}, {101, 67}};
  for (const QSize& size : sizes) {
    for (const QImage::Format format : allFormats) {
      const QImage src = randomImage(format, size.width(), size.height(), size.width() * 31 + size.height());
      for (const QTransform& xform : {QTransform(), QTransform().rotate(1.7), QTransform().scale(3, 3)}) {
        checkGpuMatchesCpu(Mode::Transform, src, xform, mappedRect(src, xform, 2),
                           OutsidePixels::assumeColor(Qt::white));
      }
    }
  }
}

BOOST_FIXTURE_TEST_CASE(test_all_formats_transforms_and_outside_modes, GpuSettingsFixture) {
  uint32_t seed = 100;
  for (const QImage::Format format : allFormats) {
    const QImage random = randomImage(format, 61, 47, ++seed);
    const QImage text = textLikeImage(format, 203, 151, ++seed);
    for (const QTransform& xform : sampleTransforms()) {
      for (const OutsidePixels& outside : allOutsideModes()) {
        checkGpuMatchesCpu(Mode::Transform, random, xform, mappedRect(random, xform, 3), outside);
        checkGpuMatchesCpu(Mode::Transform, text, xform, mappedRect(text, xform, 3), outside);
      }
    }
  }
}

BOOST_FIXTURE_TEST_CASE(test_transform_to_gray, GpuSettingsFixture) {
  uint32_t seed = 200;
  for (const QImage::Format format : allFormats) {
    const QImage src = textLikeImage(format, 157, 113, ++seed);
    for (const QTransform& xform : sampleTransforms()) {
      for (const OutsidePixels& outside : allOutsideModes()) {
        checkGpuMatchesCpu(Mode::TransformToGray, src, xform, mappedRect(src, xform, 3), outside);
      }
    }
  }
}

BOOST_FIXTURE_TEST_CASE(test_mono_source, GpuSettingsFixture) {
  const QImage src = textLikeImage(QImage::Format_Indexed8, 99, 71, 300).convertToFormat(QImage::Format_Mono);
  for (const QTransform& xform : sampleTransforms()) {
    checkGpuMatchesCpu(Mode::Transform, src, xform, mappedRect(src, xform, 1), OutsidePixels::assumeColor(Qt::white));
  }
}

BOOST_FIXTURE_TEST_CASE(test_partial_dst_rects_and_mapping_area, GpuSettingsFixture) {
  const QImage src = textLikeImage(QImage::Format_RGB32, 180, 120, 400);
  const QTransform xform = QTransform().rotate(2.1);
  const QRect rects[] = {QRect(10, 5, 50, 40), QRect(-30, -20, 60, 50), QRect(150, 100, 80, 70)};
  for (const QRect& rect : rects) {
    for (const OutsidePixels& outside : allOutsideModes()) {
      checkGpuMatchesCpu(Mode::Transform, src, xform, rect, outside);
      checkGpuMatchesCpu(Mode::Transform, src, xform, rect, outside, QSizeF(2.5, 1.5));
      checkGpuMatchesCpu(Mode::Transform, src, xform, rect, outside, QSizeF(0.1, 0.1));
    }
  }
}

BOOST_FIXTURE_TEST_CASE(test_random_transforms, GpuSettingsFixture) {
  std::mt19937 rng(500);
  std::uniform_real_distribution<double> angle(-180.0, 180.0);
  std::uniform_real_distribution<double> scale(0.2, 3.0);
  std::uniform_real_distribution<double> shift(-5.0, 5.0);
  for (int i = 0; i < 40; ++i) {
    const QImage::Format format = allFormats[i % allFormats.size()];
    const QImage src = (i % 2) ? randomImage(format, 37 + i, 29 + 2 * i, i) : textLikeImage(format, 90 + i, 70, i);
    const QTransform xform
        = QTransform().translate(shift(rng), shift(rng)).rotate(angle(rng)).scale(scale(rng), scale(rng));
    const std::vector<OutsidePixels> modes = allOutsideModes();
    checkGpuMatchesCpu(Mode::Transform, src, xform, mappedRect(src, xform, 2), modes[i % modes.size()]);
  }
}

BOOST_FIXTURE_TEST_CASE(test_banding, GpuSettingsFixture) {
  uint32_t seed = 600;
  for (const QImage::Format format : allFormats) {
    const QImage src = textLikeImage(format, 301, 211, ++seed);
    for (const int rowsPerBuffer : {30, 45}) {
      testing::setMaxBufferBytes(qint64(rowsPerBuffer) * src.bytesPerLine());
      for (const QTransform& xform : {QTransform().rotate(1.7), QTransform().rotate(-4), QTransform().scale(0.5, 0.5),
                                      QTransform().scale(2, 2).rotate(1)}) {
        for (const OutsidePixels& outside :
             {OutsidePixels::assumeColor(Qt::white), OutsidePixels::assumeWeakNearest()}) {
          checkGpuMatchesCpu(Mode::Transform, src, xform, mappedRect(src, xform, 2), outside, QSizeF(0.9, 0.9), 2);
        }
      }
    }
    testing::setMaxBufferBytes(src.sizeInBytes());
    const QTransform xform = QTransform().rotate(90).scale(1.3, 1.3);
    checkGpuMatchesCpu(Mode::Transform, src, xform, mappedRect(src, xform, 2), OutsidePixels::assumeWeakNearest(),
                       QSizeF(0.9, 0.9), 2);
  }
}

BOOST_FIXTURE_TEST_CASE(test_band_too_small_falls_back, GpuSettingsFixture) {
  const QImage src = randomImage(QImage::Format_RGB32, 64, 64, 700);
  testing::setMaxBufferBytes(64);
  const qint64 dispatches = testing::dispatchCount();
  setEnabled(false);
  const QImage cpu
      = imageproc::transform(src, QTransform().rotate(3), src.rect(), OutsidePixels::assumeColor(Qt::white));
  setEnabled(true);
  const QImage result
      = imageproc::transform(src, QTransform().rotate(3), src.rect(), OutsidePixels::assumeColor(Qt::white));
  BOOST_CHECK_EQUAL(testing::dispatchCount(), dispatches);
  BOOST_CHECK_EQUAL(compareImages(cpu, result).mismatches, 0);
}

BOOST_FIXTURE_TEST_CASE(test_exact_transforms_need_no_cpu_rows, GpuSettingsFixture) {
  const QImage src = textLikeImage(QImage::Format_RGB32, 211, 157, 800);
  for (const QTransform& xform : {QTransform(), QTransform().rotate(90), QTransform().rotate(180),
                                  QTransform().scale(2, 2), QTransform().scale(0.5, 0.5).translate(3, 7)}) {
    const qint64 recomputed = testing::recomputedRows();
    checkGpuMatchesCpu(Mode::Transform, src, xform, mappedRect(src, xform, 2), OutsidePixels::assumeColor(Qt::white));
    BOOST_CHECK_EQUAL(testing::recomputedRows(), recomputed);
  }
}

BOOST_AUTO_TEST_SUITE_END()
}  // namespace tests
}  // namespace gpu
