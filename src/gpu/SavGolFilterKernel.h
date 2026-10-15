// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#ifndef SCANTAILOR_GPU_SAVGOLFILTERKERNEL_H_
#define SCANTAILOR_GPU_SAVGOLFILTERKERNEL_H_

#include <cstdint>

namespace gpu {
constexpr int MAX_SAVGOL_KERNEL_SIZE = 32;

/**
 * Mirrors the separable passes of imageproc::savGolFilter() on an 8 bit grayscale image:
 * temp(x, y) = sum over j of src(x - kernelWidth / 2 + j, y) * horKernel[j], then
 * dst(x, y) = int(sum over j of temp(x, y - kernelHeight / 2 + j) * vertKernel[j]) clamped to [0, 255],
 * both summed in float from j = 0 up, with every multiplication and addition rounded separately.
 */
struct SavGolParams {
  const uint8_t* src;
  int srcBytesPerLine;
  uint8_t* dst;
  int dstBytesPerLine;
  int width;
  int height;
  int kernelWidth;
  int kernelHeight;
  const float* horKernel;
  const float* vertKernel;
};

/**
 * \brief Computes the pixels whose kernel window lies inside the image on the GPU.
 *
 * \return false if the GPU was not used or failed. On success, rows [kernelHeight / 2, height - kernelHeight + 1
 *         + kernelHeight / 2) of dst are written; their columns outside the window range are zero.
 */
bool savGolFilter(const SavGolParams& params);
}  // namespace gpu
#endif  // SCANTAILOR_GPU_SAVGOLFILTERKERNEL_H_
