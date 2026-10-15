// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#ifndef SCANTAILOR_GPU_INTERNAL_H_
#define SCANTAILOR_GPU_INTERNAL_H_

#include <QtGlobal>

namespace gpu {
bool isForcedOff();

qint64 maxBufferBytes();

qint64 minPixels();

void countDispatch();

void countRecomputed(qint64 count);

namespace testing {
/**
 * \brief Overrides the maximum size of a single GPU buffer. 0 restores the default.
 */
void setMaxBufferBytes(qint64 bytes);

/**
 * \brief Overrides the minimum number of pixels for which the GPU is used. A negative value restores the default.
 */
void setMinPixels(qint64 pixels);

qint64 dispatchCount();

/**
 * \brief The number of output rows or columns that GPU kernels left to the CPU.
 */
qint64 recomputedCount();
}  // namespace testing
}  // namespace gpu
#endif  // SCANTAILOR_GPU_INTERNAL_H_
