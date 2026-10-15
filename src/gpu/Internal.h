// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#ifndef SCANTAILOR_GPU_INTERNAL_H_
#define SCANTAILOR_GPU_INTERNAL_H_

#include <QtGlobal>

namespace gpu {
bool isForcedOff();

qint64 maxBufferBytes();

qint64 minTransformPixels();

qint64 minSavGolPixels();

void countDispatch();

void countRecomputedRows(qint64 rows);
}  // namespace gpu
#endif  // SCANTAILOR_GPU_INTERNAL_H_
