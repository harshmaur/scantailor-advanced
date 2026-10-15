// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#define BOOST_TEST_MODULE gpu_tests

#include <QGuiApplication>
#include <boost/test/unit_test.hpp>
#include <cstdio>
#include <memory>

#include "Gpu.h"

namespace {
struct GuiApplicationFixture {
  GuiApplicationFixture() {
#ifdef Q_OS_LINUX
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM") && qEnvironmentVariableIsEmpty("DISPLAY")
        && qEnvironmentVariableIsEmpty("WAYLAND_DISPLAY")) {
      qputenv("QT_QPA_PLATFORM", "offscreen");
    }
#endif
    app = std::make_unique<QGuiApplication>(argc, argv);
  }

  void setup() {
    const bool available = gpu::isAvailable();
    if (available) {
      std::printf("GPU device: %s\n", qPrintable(gpu::deviceName()));
    } else {
      std::fprintf(stderr, "No GPU device with compute support: the GPU comparison tests are skipped.\n");
    }
    BOOST_REQUIRE_MESSAGE(available || (qgetenv("SCANTAILOR_GPU_REQUIRED") != "1"),
                          "SCANTAILOR_GPU_REQUIRED=1 but no GPU device with compute support");
  }

  int argc = 1;
  char name[10] = "gpu_tests";
  char* argv[2] = {name, nullptr};
  std::unique_ptr<QGuiApplication> app;
};
}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(GuiApplicationFixture);
