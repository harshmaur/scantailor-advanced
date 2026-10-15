// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#include "SavGolFilterKernel.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "Banding.h"
#include "Device.h"

namespace gpu {
namespace {
constexpr int GROUP_SIZE = 16;

struct Params {
  qint32 srcStride;
  qint32 srcRowCount;
  qint32 dstStride;
  qint32 bandRows;
  qint32 width;
  qint32 kernelWidth;
  qint32 kernelHeight;
  qint32 left;
};
}  // namespace

bool savGolFilter(const SavGolParams& p) {
  const int kw = p.kernelWidth;
  const int kh = p.kernelHeight;
  if ((kw < 1) || (kh < 1) || (kw > MAX_SAVGOL_KERNEL_SIZE) || (kh > MAX_SAVGOL_KERNEL_SIZE) || (kw > p.width)
      || (kh > p.height)
      || !shouldUseGpu(qint64(p.width) * p.height, 1, p.width, p.srcBytesPerLine, p.width, p.dstBytesPerLine)) {
    return false;
  }

  const int top = kh / 2;
  const int rows = p.height - kh + 1;
  const int groupsX = ((p.width + 3) / 4 + GROUP_SIZE - 1) / GROUP_SIZE;
  std::vector<float> coeffs(p.horKernel, p.horKernel + kw);
  coeffs.insert(coeffs.end(), p.vertKernel, p.vertKernel + kh);

  const int maxGroups = Device::instance().maxGroups();
  return Device::instance().run([&](ComputeJob& job) {
    if (groupsX > maxGroups) {
      return false;
    }
    const std::vector<Band> bands
        = planBands(rows, p.dstBytesPerLine, p.srcBytesPerLine, maxGroups * GROUP_SIZE,
                    [kh](const int begin, const int end) { return std::make_pair(begin, end + kh - 1); });
    if (bands.empty()) {
      return false;
    }
    int maxBandRows = 0;
    for (const Band& band : bands) {
      maxBandRows = std::max(maxBandRows, band.dstEnd - band.dstBegin);
    }

    const std::vector<Binding> bindings
        = {{Access::Uniform, job.buffer(Access::Uniform, sizeof(Params))},
           {Access::Read, job.buffer(Access::Read, quint32((maxBandRows + kh - 1) * p.srcBytesPerLine))},
           {Access::Write, job.buffer(Access::Write, quint32(maxBandRows * p.dstBytesPerLine))},
           {Access::Read, job.buffer(Access::Read, quint32(coeffs.size() * sizeof(float)))}};
    QRhiShaderResourceBindings* srb = job.bindings(bindings);
    QRhiComputePipeline* pipeline = job.pipeline(QStringLiteral(":/gpu/savgol.comp.qsb"), bindings);
    if (!srb || !pipeline) {
      return false;
    }
    QRhiBuffer* const paramsBuffer = bindings[0].buffer;
    QRhiBuffer* const srcBuffer = bindings[1].buffer;
    QRhiBuffer* const dstBuffer = bindings[2].buffer;
    QRhiBuffer* const coeffsBuffer = bindings[3].buffer;

    bool coeffsUploaded = false;
    for (const Band& band : bands) {
      const int bandRows = band.dstEnd - band.dstBegin;
      const int srcRows = band.srcEnd - band.srcBegin;
      const Params params{p.srcBytesPerLine / 4, srcRows, p.dstBytesPerLine / 4, bandRows, p.width, kw, kh, kw / 2};

      QRhiResourceUpdateBatch* before = job.updates();
      before->updateDynamicBuffer(paramsBuffer, 0, sizeof(params), &params);
      before->uploadStaticBuffer(srcBuffer, 0, quint32(srcRows * p.srcBytesPerLine),
                                 p.src + qint64(band.srcBegin) * p.srcBytesPerLine);
      if (!coeffsUploaded) {
        before->uploadStaticBuffer(coeffsBuffer, 0, quint32(coeffs.size() * sizeof(float)), coeffs.data());
        coeffsUploaded = true;
      }

      const quint32 dstBytes = quint32(bandRows * p.dstBytesPerLine);
      QRhiReadbackResult dstResult;
      QRhiResourceUpdateBatch* after = job.updates();
      after->readBackBuffer(dstBuffer, 0, dstBytes, &dstResult);

      const int groupsY = (bandRows + GROUP_SIZE - 1) / GROUP_SIZE;
      if (!job.dispatch(pipeline, srb, groupsX, groupsY, before, after)) {
        return false;
      }
      if (quint32(dstResult.data.size()) != dstBytes) {
        return false;
      }
      const char* const result = dstResult.data.constData();
      for (int row = 0; row < bandRows; ++row) {
        std::memcpy(p.dst + qint64(top + band.dstBegin + row) * p.dstBytesPerLine,
                    result + qint64(row) * p.dstBytesPerLine, size_t(p.width));
      }
    }
    return true;
  });
}
}  // namespace gpu
