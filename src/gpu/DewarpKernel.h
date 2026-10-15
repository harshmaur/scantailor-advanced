// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#ifndef SCANTAILOR_GPU_DEWARPKERNEL_H_
#define SCANTAILOR_GPU_DEWARPKERNEL_H_

#include <cstdint>
#include <functional>
#include <vector>

#include "TransformKernel.h"

namespace gpu {
/**
 * Mirrors the area mapping of RasterDewarper: destination pixel (x, y) covers the quadrilateral
 * formed by the grid points (x, y), (x + 1, y), (x, y + 1) and (x + 1, y + 1) in source coordinates.
 */
struct DewarpParams {
  TransformFormat format;
  const uint8_t* src;
  int srcWidth;
  int srcHeight;
  int srcBytesPerLine;
  uint8_t* dst;
  int dstWidth;
  int dstHeight;
  int dstBytesPerLine;
  uint32_t bgColor;
  /** Writes the dstWidth + 1 grid points of grid row \p row as (x, y) pairs. Called on the calling thread. */
  std::function<void(int row, float* out)> gridRow;
};

/**
 * \brief Runs the area mapping on the GPU.
 *
 * \return false if the GPU was not used or failed; the contents of dst are then undefined.
 *         On success, the columns listed in \p columnsToRecompute could not be computed
 *         exactly on the GPU and must be recomputed by the caller.
 */
bool dewarp(const DewarpParams& params, std::vector<int>& columnsToRecompute);
}  // namespace gpu
#endif  // SCANTAILOR_GPU_DEWARPKERNEL_H_
