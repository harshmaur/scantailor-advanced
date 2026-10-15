// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#include <GrayImage.h>
#include <Transform.h>

#include <QElapsedTimer>
#include <QGuiApplication>
#include <QTransform>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "Gpu.h"
#include "TestImages.h"

using imageproc::OutsidePixels;

namespace {
struct Timing {
  double medianMs;
  QImage result;
};

Timing timeTransform(const QImage& src, const QTransform& xform, const bool useGpu, const int runs) {
  gpu::setEnabled(useGpu);
  const QRect dstRect = xform.mapRect(QRectF(src.rect())).toAlignedRect();
  const OutsidePixels outside = OutsidePixels::assumeColor(Qt::white);
  std::vector<double> times;
  QImage result;
  for (int i = 0; i < runs; ++i) {
    QElapsedTimer timer;
    timer.start();
    result = imageproc::transform(src, xform, dstRect, outside);
    times.push_back(timer.nsecsElapsed() / 1e6);
  }
  std::sort(times.begin(), times.end());
  return {times[times.size() / 2], result};
}

void benchPage(const char* name, const QImage& src, const QTransform& xform, const int runs) {
  const Timing cpu = timeTransform(src, xform, false, runs);
  const qint64 recomputedBefore = gpu::testing::recomputedRows();
  const qint64 dispatchesBefore = gpu::testing::dispatchCount();
  const Timing gpu = timeTransform(src, xform, true, runs);
  const qint64 recomputed = (gpu::testing::recomputedRows() - recomputedBefore) / runs;
  const qint64 dispatches = (gpu::testing::dispatchCount() - dispatchesBefore) / runs;
  const gpu::tests::ImageDiff diff = gpu::tests::compareImages(cpu.result, gpu.result);
  std::printf(
      "%-6s %5dx%-5d cpu %9.1f ms  gpu %9.1f ms  dispatches %lld  cpu rows %lld  mismatches %lld  max diff %d\n", name,
      src.width(), src.height(), cpu.medianMs, gpu.medianMs, static_cast<long long>(dispatches),
      static_cast<long long>(recomputed), static_cast<long long>(diff.mismatches), diff.maxChannelDiff);
}
}  // namespace

int main(int argc, char** argv) {
  QGuiApplication app(argc, argv);
  const int runs = (argc > 1) ? std::max(1, std::atoi(argv[1])) : 3;

  QElapsedTimer timer;
  timer.start();
  const bool available = gpu::isAvailable();
  std::printf("GPU: %s (probe %lld ms)\n", available ? qPrintable(gpu::deviceName()) : "not available",
              static_cast<long long>(timer.elapsed()));
  if (!available) {
    return 1;
  }

  const QTransform rotation = QTransform().rotate(1.7);
  gpu::testing::setMinTransformPixels(0);
  {
    const QImage warmup = gpu::tests::textLikeImage(QImage::Format_RGB32, 64, 64, 1);
    timer.restart();
    timeTransform(warmup, rotation, true, 1);
    std::printf("first GPU job (pipeline creation): %lld ms\n", static_cast<long long>(timer.elapsed()));
  }

  std::printf("600 DPI page, 1.7 degree rotation, median of %d runs:\n", runs);
  benchPage("rgb32", gpu::tests::textLikeImage(QImage::Format_RGB32, 5100, 6600, 7), rotation, runs);
  benchPage("gray", gpu::tests::textLikeImage(QImage::Format_Indexed8, 5100, 6600, 8), rotation, runs);

  std::printf("smaller images:\n");
  for (const int side : {256, 512, 1024, 2048}) {
    benchPage("rgb32", gpu::tests::textLikeImage(QImage::Format_RGB32, side, side, side), rotation, runs);
    benchPage("gray", gpu::tests::textLikeImage(QImage::Format_Indexed8, side, side, side), rotation, runs);
  }
  return 0;
}
