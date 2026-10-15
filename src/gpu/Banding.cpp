// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#include "Banding.h"

#include <algorithm>

#include "Internal.h"

namespace gpu {
std::vector<Band> planBands(const int dstRows,
                            const qint64 dstBytesPerRow,
                            const qint64 srcBytesPerRow,
                            const int maxRowsPerBand,
                            const std::function<std::pair<int, int>(int, int)>& srcRows) {
  const qint64 cap = maxBufferBytes();
  int rows = static_cast<int>(std::min<qint64>({dstRows, maxRowsPerBand, cap / dstBytesPerRow}));

  std::vector<Band> bands;
  for (int begin = 0; begin < dstRows;) {
    rows = std::min(rows, dstRows - begin);
    std::pair<int, int> src;
    while (true) {
      if (rows <= 0) {
        return {};
      }
      src = srcRows(begin, begin + rows);
      if ((src.second - src.first) * srcBytesPerRow <= cap) {
        break;
      }
      rows /= 2;
    }
    bands.push_back({begin, begin + rows, src.first, src.second});
    begin += rows;
  }
  return bands;
}
}  // namespace gpu
