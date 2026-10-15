// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#include "Gpu.h"

namespace gpu {
bool isAvailable() {
  return false;
}

QString deviceName() {
  return QString();
}
}  // namespace gpu
