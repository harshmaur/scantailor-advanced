// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#ifndef SCANTAILOR_GPU_TESTS_TESTIMAGES_H_
#define SCANTAILOR_GPU_TESTS_TESTIMAGES_H_

#include <QImage>
#include <cstdint>

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
}  // namespace tests
}  // namespace gpu
#endif  // SCANTAILOR_GPU_TESTS_TESTIMAGES_H_
