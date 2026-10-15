// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#include "Gpu.h"

#include <atomic>

#include "Internal.h"

namespace gpu {
namespace {
constexpr qint64 DEFAULT_MAX_BUFFER_BYTES = qint64(64) << 20;
constexpr qint64 DEFAULT_MIN_TRANSFORM_PIXELS = 1 << 20;
constexpr qint64 DEFAULT_MIN_DEWARP_PIXELS = 1 << 20;
constexpr qint64 DEFAULT_MIN_SAVGOL_PIXELS = 1 << 20;

std::atomic<bool> enabledSetting(true);
std::atomic<qint64> maxBufferBytesOverride(0);
std::atomic<qint64> minTransformPixelsOverride(-1);
std::atomic<qint64> minDewarpPixelsOverride(-1);
std::atomic<qint64> minSavGolPixelsOverride(-1);
std::atomic<qint64> dispatches(0);
std::atomic<qint64> recomputed(0);
std::atomic<qint64> recomputedCols(0);
}  // namespace

void setEnabled(const bool enabled) {
  enabledSetting = enabled;
}

bool isEnabled() {
  const QByteArray env = qgetenv("SCANTAILOR_GPU");
  if (env == "0") {
    return false;
  }
  if (env == "1") {
    return true;
  }
  return enabledSetting;
}

bool isForcedOff() {
  return qgetenv("SCANTAILOR_GPU") == "0";
}

qint64 maxBufferBytes() {
  const qint64 bytes = maxBufferBytesOverride;
  return bytes > 0 ? bytes : DEFAULT_MAX_BUFFER_BYTES;
}

qint64 minTransformPixels() {
  const qint64 pixels = minTransformPixelsOverride;
  return pixels >= 0 ? pixels : DEFAULT_MIN_TRANSFORM_PIXELS;
}

qint64 minDewarpPixels() {
  const qint64 pixels = minDewarpPixelsOverride;
  return pixels >= 0 ? pixels : DEFAULT_MIN_DEWARP_PIXELS;
}

qint64 minSavGolPixels() {
  const qint64 pixels = minSavGolPixelsOverride;
  return pixels >= 0 ? pixels : DEFAULT_MIN_SAVGOL_PIXELS;
}

void countDispatch() {
  ++dispatches;
}

void countRecomputedRows(const qint64 rows) {
  recomputed += rows;
}

void countRecomputedColumns(const qint64 columns) {
  recomputedCols += columns;
}

namespace testing {
void setMaxBufferBytes(const qint64 bytes) {
  maxBufferBytesOverride = bytes;
}

void setMinTransformPixels(const qint64 pixels) {
  minTransformPixelsOverride = pixels;
}

void setMinDewarpPixels(const qint64 pixels) {
  minDewarpPixelsOverride = pixels;
}

void setMinSavGolPixels(const qint64 pixels) {
  minSavGolPixelsOverride = pixels;
}

qint64 dispatchCount() {
  return dispatches;
}

qint64 recomputedRows() {
  return recomputed;
}

qint64 recomputedColumns() {
  return recomputedCols;
}
}  // namespace testing
}  // namespace gpu
