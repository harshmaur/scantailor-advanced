// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#ifndef SCANTAILOR_GPU_TRANSFORMKERNEL_H_
#define SCANTAILOR_GPU_TRANSFORMKERNEL_H_

#include <cstdint>
#include <vector>

namespace gpu {
enum class TransformFormat { Gray8, Rgb32, Argb32 };

constexpr int OUTSIDE_COLOR = 1 << 0;
constexpr int OUTSIDE_NEAREST = 1 << 1;
constexpr int OUTSIDE_WEAK = 1 << 2;

/**
 * Mirrors imageproc::transform(): a destination pixel (dx, dy) maps to the source point
 * (fx + (dx + 0.5) * m11, fy + (dx + 0.5) * m12) in 1/32 pixel units, where
 * fx = (dy + 0.5) * m21 + tx and fy = (dy + 0.5) * m22 + ty.
 */
struct TransformParams {
  TransformFormat format;
  const uint8_t* src;
  int srcWidth;
  int srcHeight;
  int srcBytesPerLine;
  uint8_t* dst;
  int dstWidth;
  int dstHeight;
  int dstBytesPerLine;
  double m11, m12, m21, m22, tx, ty;
  int src32UnitWidth;
  int src32UnitHeight;
  uint32_t outsideColor;
  int outsideFlags;
};

/**
 * \brief Runs the transform on the GPU.
 *
 * \return false if the GPU was not used or failed; the contents of dst are then undefined.
 *         On success, the rows listed in \p rowsToRecompute could not be computed
 *         exactly on the GPU and must be recomputed by the caller.
 */
bool transform(const TransformParams& params, std::vector<int>& rowsToRecompute);
}  // namespace gpu
#endif  // SCANTAILOR_GPU_TRANSFORMKERNEL_H_
