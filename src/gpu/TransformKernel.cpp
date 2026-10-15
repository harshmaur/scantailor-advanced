// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#include "TransformKernel.h"

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
constexpr int COORD_WORDS = 5;
constexpr double MAX_COORD = 1 << 24;
constexpr double MAX_EXACT_COORD = 1 << 20;

struct Params {
  qint32 srcWidth;
  qint32 srcHeight;
  qint32 srcStride;
  qint32 srcRowBegin;
  qint32 srcRowCount;
  qint32 dstWidth;
  qint32 dstStride;
  qint32 bandRows;
  qint32 unitWidth;
  qint32 unitHeight;
  qint32 outsideFlags;
  quint32 outsideColor;
};

struct Fixed {
  qint64 value;
  bool exact;
};

Fixed toFixed(const double value) {
  const double scaled = std::ldexp(value, 32);
  const auto rounded = static_cast<qint64>(std::nearbyint(scaled));
  return {rounded, (static_cast<double>(rounded) == scaled) && (std::abs(value) < MAX_EXACT_COORD)};
}

// base + factor * m, where the CPU may or may not fuse the multiplication and the addition.
// The result is exact only if every way of evaluating it gives the same double.
bool encodeAffine(const double factor, const double m, const double base, quint32* out) {
  const double product = factor * m;
  const double sum = product + base;
  if (!(std::abs(product) < MAX_COORD) || !(std::abs(sum) < MAX_COORD) || !(std::abs(base) < MAX_COORD)) {
    return false;
  }
  const Fixed fixedProduct = toFixed(product);
  const Fixed fixedBase = toFixed(base);
  const Fixed fixedSum = toFixed(sum);
  const bool exact = (std::fma(factor, m, -product) == 0.0) && fixedProduct.exact && fixedBase.exact && fixedSum.exact
                     && (fixedProduct.value + fixedBase.value == fixedSum.value);
  out[0] = static_cast<quint32>(static_cast<quint64>(fixedSum.value) >> 32);
  out[1] = static_cast<quint32>(fixedSum.value);
  out[2] = exact ? 1 : 0;
  return true;
}

bool encodeCoords(const double factor,
                  const double mx,
                  const double baseX,
                  const double my,
                  const double baseY,
                  quint32* out) {
  quint32 x[3];
  quint32 y[3];
  if (!encodeAffine(factor, mx, baseX, x) || !encodeAffine(factor, my, baseY, y)) {
    return false;
  }
  out[0] = x[0];
  out[1] = x[1];
  out[2] = y[0];
  out[3] = y[1];
  out[4] = x[2] | (y[2] << 1);
  return true;
}

const char* shaderFor(const TransformFormat format) {
  switch (format) {
    case TransformFormat::Gray8:
      return ":/gpu/transform_gray.comp.qsb";
    case TransformFormat::Rgb32:
      return ":/gpu/transform_rgb.comp.qsb";
    case TransformFormat::Argb32:
      return ":/gpu/transform_argb.comp.qsb";
  }
  return nullptr;
}
}  // namespace

bool transform(const TransformParams& p, std::vector<int>& rowsToRecompute) {
  rowsToRecompute.clear();
  const int dw = p.dstWidth;
  const int dh = p.dstHeight;
  const int bytesPerPixel = (p.format == TransformFormat::Gray8) ? 1 : 4;
  if ((qint64(dw) * dh < minTransformPixels()) || !isEnabled() || (p.srcBytesPerLine % 4 != 0)
      || (p.dstBytesPerLine % 4 != 0) || (p.srcBytesPerLine < p.srcWidth * bytesPerPixel)
      || (p.dstBytesPerLine < dw * bytesPerPixel) || !Device::instance().isAvailable()) {
    return false;
  }

  std::vector<quint32> cols(size_t(dw) * COORD_WORDS);
  for (int dx = 0; dx < dw; ++dx) {
    if (!encodeCoords(dx + 0.5, p.m11, 0.0, p.m12, 0.0, &cols[size_t(dx) * COORD_WORDS])) {
      return false;
    }
  }
  std::vector<quint32> rows(size_t(dh) * COORD_WORDS);
  for (int dy = 0; dy < dh; ++dy) {
    if (!encodeCoords(dy + 0.5, p.m21, p.tx, p.m22, p.ty, &rows[size_t(dy) * COORD_WORDS])) {
      return false;
    }
  }

  const int halfUnit = p.src32UnitHeight >> 1;
  auto srcRows = [&p, dw, halfUnit](const int begin, const int end) {
    double minY = MAX_COORD;
    double maxY = -MAX_COORD;
    for (const int dy : {begin, end - 1}) {
      for (const int dx : {0, dw - 1}) {
        const double y = (dy + 0.5) * p.m22 + p.ty + (dx + 0.5) * p.m12;
        minY = std::min(minY, y);
        maxY = std::max(maxY, y);
      }
    }
    const int top = static_cast<int>(std::floor((minY - halfUnit - 2) / 32)) - 1;
    const int bottom = static_cast<int>(std::floor((maxY - halfUnit + p.src32UnitHeight + 2) / 32)) + 1;
    return std::make_pair(qBound(0, top, p.srcHeight - 1), qBound(0, bottom, p.srcHeight - 1) + 1);
  };

  const int itemsPerRow = (p.format == TransformFormat::Gray8) ? (dw + 3) / 4 : dw;
  const int groupsX = (itemsPerRow + GROUP_SIZE - 1) / GROUP_SIZE;

  const bool ok = Device::instance().run([&](ComputeJob& job) {
    const int maxGroups = std::min(65535, job.rhi()->resourceLimit(QRhi::MaxThreadGroupsPerDimension));
    if (groupsX > maxGroups) {
      return false;
    }
    const std::vector<Band> bands
        = planBands(dh, p.dstBytesPerLine, p.srcBytesPerLine, maxGroups * GROUP_SIZE, srcRows);
    if (bands.empty()) {
      return false;
    }
    int maxBandRows = 0;
    int maxSrcRows = 0;
    for (const Band& band : bands) {
      maxBandRows = std::max(maxBandRows, band.dstEnd - band.dstBegin);
      maxSrcRows = std::max(maxSrcRows, band.srcEnd - band.srcBegin);
    }

    const std::vector<Binding> bindings
        = {{Access::Uniform, job.buffer(Access::Uniform, sizeof(Params))},
           {Access::Read, job.buffer(Access::Read, quint32(maxSrcRows * p.srcBytesPerLine))},
           {Access::Write, job.buffer(Access::Write, quint32(maxBandRows * p.dstBytesPerLine))},
           {Access::Read, job.buffer(Access::Read, quint32(cols.size() * sizeof(quint32)))},
           {Access::Read, job.buffer(Access::Read, quint32(maxBandRows * COORD_WORDS * sizeof(quint32)))},
           {Access::ReadWrite, job.buffer(Access::ReadWrite, quint32((maxBandRows + 1) * sizeof(quint32)))}};
    QRhiShaderResourceBindings* srb = job.bindings(bindings);
    QRhiComputePipeline* pipeline = job.pipeline(QLatin1String(shaderFor(p.format)), bindings);
    if (!srb || !pipeline) {
      return false;
    }
    QRhiBuffer* const paramsBuffer = bindings[0].buffer;
    QRhiBuffer* const srcBuffer = bindings[1].buffer;
    QRhiBuffer* const dstBuffer = bindings[2].buffer;
    QRhiBuffer* const colsBuffer = bindings[3].buffer;
    QRhiBuffer* const rowsBuffer = bindings[4].buffer;
    QRhiBuffer* const statusBuffer = bindings[5].buffer;

    const std::vector<quint32> zeros(maxBandRows + 1, 0);
    bool colsUploaded = false;
    for (const Band& band : bands) {
      const int bandRows = band.dstEnd - band.dstBegin;
      const Params params{p.srcWidth,
                          p.srcHeight,
                          p.srcBytesPerLine / 4,
                          band.srcBegin,
                          band.srcEnd - band.srcBegin,
                          dw,
                          p.dstBytesPerLine / 4,
                          bandRows,
                          p.src32UnitWidth,
                          p.src32UnitHeight,
                          p.outsideFlags,
                          p.outsideColor};

      QRhiResourceUpdateBatch* before = job.updates();
      before->updateDynamicBuffer(paramsBuffer, 0, sizeof(params), &params);
      before->uploadStaticBuffer(srcBuffer, 0, quint32((band.srcEnd - band.srcBegin) * p.srcBytesPerLine),
                                 p.src + qint64(band.srcBegin) * p.srcBytesPerLine);
      if (!colsUploaded) {
        before->uploadStaticBuffer(colsBuffer, 0, quint32(cols.size() * sizeof(quint32)), cols.data());
        colsUploaded = true;
      }
      before->uploadStaticBuffer(rowsBuffer, 0, quint32(bandRows * COORD_WORDS * sizeof(quint32)),
                                 &rows[size_t(band.dstBegin) * COORD_WORDS]);
      before->uploadStaticBuffer(statusBuffer, 0, quint32((bandRows + 1) * sizeof(quint32)), zeros.data());

      const quint32 dstBytes = quint32(bandRows * p.dstBytesPerLine);
      const quint32 statusBytes = quint32((bandRows + 1) * sizeof(quint32));
      QRhiReadbackResult dstResult;
      QRhiReadbackResult statusResult;
      QRhiResourceUpdateBatch* after = job.updates();
      after->readBackBuffer(dstBuffer, 0, dstBytes, &dstResult);
      after->readBackBuffer(statusBuffer, 0, statusBytes, &statusResult);

      const int groupsY = (bandRows + GROUP_SIZE - 1) / GROUP_SIZE;
      if (!job.dispatch(pipeline, srb, groupsX, groupsY, before, after)) {
        return false;
      }
      if ((quint32(dstResult.data.size()) != dstBytes) || (quint32(statusResult.data.size()) != statusBytes)) {
        return false;
      }
      const auto* status = reinterpret_cast<const quint32*>(statusResult.data.constData());
      if (status[0] != 0) {
        return false;
      }
      std::memcpy(p.dst + qint64(band.dstBegin) * p.dstBytesPerLine, dstResult.data.constData(), dstBytes);
      for (int row = 0; row < bandRows; ++row) {
        if (status[row + 1] != 0) {
          rowsToRecompute.push_back(band.dstBegin + row);
        }
      }
    }
    return true;
  });
  if (!ok) {
    rowsToRecompute.clear();
    return false;
  }
  countRecomputedRows(static_cast<qint64>(rowsToRecompute.size()));
  return true;
}
}  // namespace gpu
