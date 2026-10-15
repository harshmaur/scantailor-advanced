// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#ifndef SCANTAILOR_GPU_GPU_H_
#define SCANTAILOR_GPU_GPU_H_

#include <QString>

/**
 * GPU kernels must produce exactly the CPU result: the output is binarized afterwards,
 * so a one level difference can flip pixels and make the output depend on the machine.
 */
namespace gpu {
enum class State { Unknown, Available, Unavailable };

/**
 * \brief Enables or disables GPU processing for the whole process.
 *
 * SCANTAILOR_GPU=0 turns GPU processing off regardless of this setting.
 * Software renderers are only used if SCANTAILOR_GPU_SOFTWARE is "1".
 * SCANTAILOR_GPU_BACKEND ("metal", "d3d12", "d3d11" or "vulkan") restricts the graphics API.
 */
void setEnabled(bool enabled);

bool isEnabled();

/**
 * \brief Starts creating the GPU device in the background.
 *
 * Call once a QGuiApplication exists. The device is released when the application quits.
 */
void start();

/**
 * \brief Whether a GPU device with compute support exists. Does not block.
 */
State state();

/**
 * \brief Like state(), but waits until the device has been created.
 *
 * Returns false without creating the device if no QGuiApplication exists yet.
 */
bool isAvailable();

/**
 * \brief The name of the GPU device and its graphics API, or an empty string if not available.
 */
QString deviceName();
}  // namespace gpu
#endif  // SCANTAILOR_GPU_GPU_H_
