// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#include "DewarpKernel.h"
#include "Gpu.h"
#include "TransformKernel.h"

namespace gpu {
void start() {}

State state() {
  return State::Unavailable;
}

bool isAvailable() {
  return false;
}

QString deviceName() {
  return QString();
}

bool transform(const TransformParams&, std::vector<int>&) {
  return false;
}

bool dewarp(const DewarpParams&, std::vector<int>&) {
  return false;
}
}  // namespace gpu
