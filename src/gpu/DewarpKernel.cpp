// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#include "DewarpKernel.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "Banding.h"
#include "Device.h"
#include "Gpu.h"
#include "Internal.h"

namespace gpu {
namespace {
constexpr int GROUP_SIZE = 16;
constexpr int MAX_GROUPS = 65535;

struct Params {
  qint32 srcWidth;
  qint32 srcHeight;
  qint32 srcStride;
  qint32 srcRowBegin;
  qint32 srcRowCount;
  qint32 dstWidth;
  qint32 dstStride;
  qint32 bandRows;
  qint32 gridStride;
  quint32 bgColor;
  float maxRight;
  float maxBottom;
};

const char* shaderFor(const TransformFormat format) {
  switch (format) {
    case TransformFormat::Gray8:
      return ":/gpu/dewarp_gray.comp.qsb";
    case TransformFormat::Rgb32:
      return ":/gpu/dewarp_rgb.comp.qsb";
    case TransformFormat::Argb32:
      return ":/gpu/dewarp_argb.comp.qsb";
  }
  return nullptr;
}

bool dewarpBand(const DewarpParams& p,
                const Band& band,
                const std::vector<float>& grid,
                std::vector<char>& unsureColumns) {
  const int dw = p.dstWidth;
  const int bandRows = band.dstEnd - band.dstBegin;
  const int srcRows = band.srcEnd - band.srcBegin;
  const int itemsPerRow = (p.format == TransformFormat::Gray8) ? (dw + 3) / 4 : dw;
  const int groupsX = (itemsPerRow + GROUP_SIZE - 1) / GROUP_SIZE;
  const int groupsY = (bandRows + GROUP_SIZE - 1) / GROUP_SIZE;
  const quint32 gridBytes = quint32(grid.size() * sizeof(float));
  const quint32 srcBytes = quint32(qint64(srcRows) * p.srcBytesPerLine);
  const quint32 dstBytes = quint32(qint64(bandRows) * p.dstBytesPerLine);
  const quint32 statusBytes = quint32(dw * sizeof(quint32));

  return Device::instance().run([&](ComputeJob& job) {
    const int maxGroups = job.rhi()->resourceLimit(QRhi::MaxThreadGroupsPerDimension);
    if ((groupsX > maxGroups) || (groupsY > maxGroups)) {
      return false;
    }
    const std::vector<Binding> bindings = {{Access::Uniform, job.buffer(Access::Uniform, sizeof(Params))},
                                           {Access::Read, job.buffer(Access::Read, srcBytes)},
                                           {Access::Write, job.buffer(Access::Write, dstBytes)},
                                           {Access::Read, job.buffer(Access::Read, gridBytes)},
                                           {Access::ReadWrite, job.buffer(Access::ReadWrite, statusBytes)}};
    QRhiShaderResourceBindings* srb = job.bindings(bindings);
    QRhiComputePipeline* pipeline = job.pipeline(QLatin1String(shaderFor(p.format)), bindings);
    if (!srb || !pipeline) {
      return false;
    }

    const Params params{p.srcWidth,
                        p.srcHeight,
                        p.srcBytesPerLine / 4,
                        band.srcBegin,
                        srcRows,
                        dw,
                        p.dstBytesPerLine / 4,
                        bandRows,
                        dw + 1,
                        p.bgColor,
                        32.0f * (float(p.srcWidth) + 10000.f),
                        32.0f * (float(p.srcHeight) + 10000.f)};
    const std::vector<quint32> zeros(dw, 0);
    QRhiResourceUpdateBatch* before = job.updates();
    before->updateDynamicBuffer(bindings[0].buffer, 0, sizeof(params), &params);
    before->uploadStaticBuffer(bindings[1].buffer, 0, srcBytes, p.src + qint64(band.srcBegin) * p.srcBytesPerLine);
    before->uploadStaticBuffer(bindings[3].buffer, 0, gridBytes, grid.data());
    before->uploadStaticBuffer(bindings[4].buffer, 0, statusBytes, zeros.data());

    QRhiReadbackResult dstResult;
    QRhiReadbackResult statusResult;
    QRhiResourceUpdateBatch* after = job.updates();
    after->readBackBuffer(bindings[2].buffer, 0, dstBytes, &dstResult);
    after->readBackBuffer(bindings[4].buffer, 0, statusBytes, &statusResult);

    if (!job.dispatch(pipeline, srb, groupsX, groupsY, before, after)) {
      return false;
    }
    if ((quint32(dstResult.data.size()) != dstBytes) || (quint32(statusResult.data.size()) != statusBytes)) {
      return false;
    }
    std::memcpy(p.dst + qint64(band.dstBegin) * p.dstBytesPerLine, dstResult.data.constData(), dstBytes);
    const auto* status = reinterpret_cast<const quint32*>(statusResult.data.constData());
    for (int x = 0; x < dw; ++x) {
      if (status[x] != 0) {
        unsureColumns[x] = 1;
      }
    }
    return true;
  });
}
}  // namespace

bool dewarp(const DewarpParams& p, std::vector<int>& columnsToRecompute) {
  columnsToRecompute.clear();
  const int dw = p.dstWidth;
  const int dh = p.dstHeight;
  const int sh = p.srcHeight;
  const int bytesPerPixel = (p.format == TransformFormat::Gray8) ? 1 : 4;
  if ((qint64(dw) * dh < minDewarpPixels()) || (dw <= 0) || (dh <= 0) || !isEnabled() || (p.srcBytesPerLine % 4 != 0)
      || (p.dstBytesPerLine % 4 != 0) || (p.srcBytesPerLine < p.srcWidth * bytesPerPixel)
      || (p.dstBytesPerLine < dw * bytesPerPixel) || !Device::instance().isAvailable()) {
    return false;
  }

  const int gridStride = dw + 1;
  const qint64 gridBytesPerRow = qint64(gridStride) * 2 * sizeof(float);
  const int maxRowsPerBand
      = static_cast<int>(std::min<qint64>(qint64(MAX_GROUPS) * GROUP_SIZE, maxBufferBytes() / gridBytesPerRow - 1));
  if (maxRowsPerBand < 1) {
    return false;
  }

  std::vector<float> rowPoints(size_t(gridStride) * 2);
  auto srcRows = [&](const int begin, const int end) {
    float minY = INFINITY;
    float maxY = -INFINITY;
    for (const int row : {begin, end}) {
      p.gridRow(row, rowPoints.data());
      for (int x = 0; x < gridStride; ++x) {
        const float y = rowPoints[size_t(x) * 2 + 1];
        if (y < minY) {
          minY = y;
        }
        if (y > maxY) {
          maxY = y;
        }
      }
    }
    if (minY > maxY) {
      minY = maxY = 0;
    }
    const int top = static_cast<int>(std::floor(qBound(-2.0, double(minY), sh + 2.0))) - 1;
    const int bottom = static_cast<int>(std::floor(qBound(-2.0, double(maxY), sh + 2.0))) + 2;
    const int clampedTop = qBound(0, top, sh - 1);
    return std::make_pair(clampedTop, qBound(clampedTop + 1, bottom, sh));
  };
  const std::vector<Band> bands
      = planBands(dh, std::max<qint64>(gridBytesPerRow, p.dstBytesPerLine), p.srcBytesPerLine, maxRowsPerBand, srcRows);
  if (bands.empty()) {
    return false;
  }

  std::vector<char> unsureColumns(dw, 0);
  std::vector<float> grid;
  for (const Band& band : bands) {
    const int gridRows = band.dstEnd - band.dstBegin + 1;
    grid.resize(size_t(gridRows) * gridStride * 2);
    for (int row = 0; row < gridRows; ++row) {
      p.gridRow(band.dstBegin + row, &grid[size_t(row) * gridStride * 2]);
    }
    if (!dewarpBand(p, band, grid, unsureColumns)) {
      return false;
    }
  }

  for (int x = 0; x < dw; ++x) {
    if (unsureColumns[x]) {
      columnsToRecompute.push_back(x);
    }
  }
  countRecomputedColumns(static_cast<qint64>(columnsToRecompute.size()));
  return true;
}
}  // namespace gpu
