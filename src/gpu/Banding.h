// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#ifndef SCANTAILOR_GPU_BANDING_H_
#define SCANTAILOR_GPU_BANDING_H_

#include <QtGlobal>
#include <functional>
#include <utility>
#include <vector>

namespace gpu {
/**
 * Output rows [dstBegin, dstEnd) that read source rows [srcBegin, srcEnd).
 */
struct Band {
  int dstBegin;
  int dstEnd;
  int srcBegin;
  int srcEnd;
};

/**
 * \brief Splits the output into horizontal bands so that neither the output rows
 *        nor the source rows of a band exceed the per-buffer size limit.
 *
 * \param srcRows Returns the half-open range of source rows needed by output rows [begin, end).
 * \return The bands in top to bottom order, or an empty vector if a single output row does not fit.
 */
std::vector<Band> planBands(int dstRows,
                            qint64 dstBytesPerRow,
                            qint64 srcBytesPerRow,
                            int maxRowsPerBand,
                            const std::function<std::pair<int, int>(int, int)>& srcRows);
}  // namespace gpu
#endif  // SCANTAILOR_GPU_BANDING_H_
