// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#include "TestImages.h"

#include <QVector>
#include <algorithm>
#include <cstdlib>
#include <random>

namespace gpu {
namespace tests {
namespace {
QImage createImage(const QImage::Format format, const int width, const int height) {
  QImage image(width, height, format);
  if (format == QImage::Format_Indexed8) {
    QVector<QRgb> palette(256);
    for (int i = 0; i < 256; ++i) {
      palette[i] = qRgb(i, i, i);
    }
    image.setColorTable(palette);
  }
  return image;
}

void setPixel(QImage& image, const int x, const int y, const QRgb color) {
  if (image.format() == QImage::Format_Indexed8) {
    image.scanLine(y)[x] = static_cast<uchar>(qGray(color));
  } else {
    reinterpret_cast<QRgb*>(image.scanLine(y))[x] = color;
  }
}
}  // namespace

QImage randomImage(const QImage::Format format, const int width, const int height, const uint32_t seed) {
  QImage image = createImage(format, width, height);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> byte(0, 255);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int alpha = 255;
      if (format == QImage::Format_ARGB32) {
        const int choice = byte(rng);
        alpha = (choice < 32) ? 0 : (choice < 96) ? 255 : byte(rng);
      }
      setPixel(image, x, y, qRgba(byte(rng), byte(rng), byte(rng), alpha));
    }
  }
  return image;
}

QImage textLikeImage(const QImage::Format format, const int width, const int height, const uint32_t seed) {
  QImage image = createImage(format, width, height);
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> noise(0, 12);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int level = 240 + noise(rng);
      setPixel(image, x, y, qRgb(level, level - 4, level - 10));
    }
  }

  const int lineSpacing = std::max(4, height / 60);
  const int glyphHeight = std::max(2, lineSpacing / 2);
  const int stroke = std::max(1, glyphHeight / 8);
  std::uniform_int_distribution<int> glyphWidth(std::max(1, glyphHeight / 2), std::max(1, glyphHeight));
  std::uniform_int_distribution<int> gap(1, std::max(1, glyphHeight / 3));
  std::uniform_int_distribution<int> ink(10, 70);
  std::uniform_int_distribution<int> shape(0, 3);
  for (int top = lineSpacing / 2; top + glyphHeight < height; top += lineSpacing) {
    for (int left = width / 12; left < width - width / 12;) {
      const int w = glyphWidth(rng);
      const int s = shape(rng);
      const int level = ink(rng);
      const QRgb color = qRgb(level, level, level + 20);
      for (int y = top; y < std::min(height, top + glyphHeight); ++y) {
        for (int x = left; x < std::min(width, left + w); ++x) {
          const bool leftStem = x < left + stroke;
          const bool rightStem = x >= left + w - stroke;
          const bool topBar = y < top + stroke;
          const bool middleBar = std::abs(y - (top + glyphHeight / 2)) < (stroke + 1) / 2;
          const bool bottomBar = y >= top + glyphHeight - stroke;
          const bool on = (s == 0)   ? (leftStem || rightStem || topBar || bottomBar)
                          : (s == 1) ? (leftStem || middleBar)
                          : (s == 2) ? (leftStem || topBar || middleBar)
                                     : (rightStem || bottomBar);
          if (on) {
            setPixel(image, x, y, color);
          }
        }
      }
      left += w + gap(rng);
    }
  }
  return image;
}

ImageDiff compareImages(const QImage& lhs, const QImage& rhs) {
  ImageDiff diff;
  const int bytesPerPixel = lhs.depth() / 8;
  for (int y = 0; y < lhs.height(); ++y) {
    const uchar* lhsLine = lhs.constScanLine(y);
    const uchar* rhsLine = rhs.constScanLine(y);
    for (int x = 0; x < lhs.width(); ++x) {
      bool mismatch = false;
      for (int i = 0; i < bytesPerPixel; ++i) {
        const int channelDiff = std::abs(lhsLine[x * bytesPerPixel + i] - rhsLine[x * bytesPerPixel + i]);
        diff.maxChannelDiff = std::max(diff.maxChannelDiff, channelDiff);
        mismatch = mismatch || (channelDiff != 0);
      }
      if (mismatch) {
        ++diff.mismatches;
      }
    }
  }
  return diff;
}
}  // namespace tests
}  // namespace gpu
