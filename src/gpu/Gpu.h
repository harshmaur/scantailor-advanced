// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#ifndef SCANTAILOR_GPU_GPU_H_
#define SCANTAILOR_GPU_GPU_H_

#include <QString>
#include <QtGlobal>

namespace gpu {
/**
 * \brief Enables or disables GPU processing for the whole process.
 *
 * The SCANTAILOR_GPU environment variable overrides this:
 * "0" forces GPU processing off, "1" forces it on.
 * Software renderers are only used if SCANTAILOR_GPU_SOFTWARE is "1".
 */
void setEnabled(bool enabled);

bool isEnabled();

/**
 * \brief Returns true if a GPU device with compute support could be created.
 *
 * The device is created on the first call, which may take a while.
 */
bool isAvailable();

/**
 * \brief The name of the GPU device and its graphics API, or an empty string if not available.
 */
QString deviceName();

namespace testing {
/**
 * \brief Overrides the maximum size of a single GPU buffer. 0 restores the default.
 */
void setMaxBufferBytes(qint64 bytes);

/**
 * \brief Overrides the minimum number of output pixels for which transform() uses the GPU.
 *        A negative value restores the default.
 */
void setMinTransformPixels(qint64 pixels);

/**
 * \brief The number of compute dispatches that completed successfully.
 */
qint64 dispatchCount();

/**
 * \brief The number of output rows that GPU kernels left to the CPU.
 */
qint64 recomputedRows();
}  // namespace testing
}  // namespace gpu
#endif  // SCANTAILOR_GPU_GPU_H_
