// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#include "Gpu.h"

#include <atomic>

#include "Internal.h"

namespace gpu {
namespace {
constexpr qint64 DEFAULT_MAX_BUFFER_BYTES = qint64(64) << 20;
constexpr qint64 DEFAULT_MIN_PIXELS = 1 << 20;

std::atomic<bool> enabledSetting(true);
std::atomic<qint64> maxBufferBytesOverride(0);
std::atomic<qint64> minPixelsOverride(-1);
std::atomic<qint64> dispatches(0);
std::atomic<qint64> recomputed(0);
}  // namespace

void setEnabled(const bool enabled) {
  enabledSetting = enabled;
}

bool isEnabled() {
  return !isForcedOff() && enabledSetting;
}

bool isForcedOff() {
  return qgetenv("SCANTAILOR_GPU") == "0";
}

qint64 maxBufferBytes() {
  const qint64 bytes = maxBufferBytesOverride;
  return bytes > 0 ? bytes : DEFAULT_MAX_BUFFER_BYTES;
}

qint64 minPixels() {
  const qint64 pixels = minPixelsOverride;
  return pixels >= 0 ? pixels : DEFAULT_MIN_PIXELS;
}

void countDispatch() {
  ++dispatches;
}

void countRecomputed(const qint64 count) {
  recomputed += count;
}

namespace testing {
void setMaxBufferBytes(const qint64 bytes) {
  maxBufferBytesOverride = bytes;
}

void setMinPixels(const qint64 pixels) {
  minPixelsOverride = pixels;
}

qint64 dispatchCount() {
  return dispatches;
}

qint64 recomputedCount() {
  return recomputed;
}
}  // namespace testing
}  // namespace gpu
