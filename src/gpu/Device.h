// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#ifndef SCANTAILOR_GPU_DEVICE_H_
#define SCANTAILOR_GPU_DEVICE_H_

#include <rhi/qrhi.h>

#include <QString>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "Gpu.h"
#include "TransformKernel.h"

#if QT_CONFIG(vulkan) && __has_include(<vulkan/vulkan.h>)
#define SCANTAILOR_GPU_VULKAN
class QVulkanInstance;
#endif

namespace gpu {
enum class Access { Uniform, Read, Write, ReadWrite };

struct Binding {
  Access access;
  QRhiBuffer* buffer;
};

class Device;

/**
 * \brief Whether a kernel should run on the GPU: the image is large enough, GPU processing is enabled,
 *        the strides are multiples of 4 bytes that fit the rows, and a device is available.
 */
bool shouldUseGpu(qint64 pixels,
                  int bytesPerPixel,
                  int srcWidth,
                  int srcBytesPerLine,
                  int dstWidth,
                  int dstBytesPerLine);

/** The .qsb resource path of the variant of \p kernel for \p format. */
QString shaderPath(const char* kernel, TransformFormat format);

/**
 * \brief The resources of one GPU job. Lives on the device thread for the duration of the job.
 *
 * Buffers and bindings created here are released when the job ends.
 * Binding i of a bindings() call is bound to binding point i of the shader.
 */
class ComputeJob {
 public:
  QRhi* rhi() const;

  /** Returns nullptr on failure. Uniform buffers are dynamic, storage buffers are static. */
  QRhiBuffer* buffer(Access access, quint32 size);

  QRhiShaderResourceBindings* bindings(const std::vector<Binding>& bindings);

  /** \p shader is a .qsb resource path. The pipeline is cached for the lifetime of the device. */
  QRhiComputePipeline* pipeline(const QString& shader, const std::vector<Binding>& layout);

  QRhiResourceUpdateBatch* updates();

  /**
   * \brief Records \p before, the dispatch and \p after in one frame and waits for it to finish.
   *
   * Read-backs requested in \p after are complete when this returns true.
   */
  bool dispatch(QRhiComputePipeline* pipeline,
                QRhiShaderResourceBindings* bindings,
                int groupsX,
                int groupsY,
                QRhiResourceUpdateBatch* before,
                QRhiResourceUpdateBatch* after);

 private:
  friend class Device;

  explicit ComputeJob(Device& device);

  Device& m_device;
  std::vector<std::unique_ptr<QRhiBuffer>> m_buffers;
  std::vector<std::unique_ptr<QRhiShaderResourceBindings>> m_bindings;
};

/**
 * \brief The process-wide GPU device. QRhi is owned and used by one dedicated thread.
 *
 * Callers on any thread submit jobs with run() and block until the job finishes.
 * Jobs run one at a time.
 */
class Device {
 public:
  static Device& instance();

  void start();

  void shutdown();

  State state();

  bool isAvailable();

  QString name();

  /** The maximum number of thread groups per dispatch dimension. Valid while the device is available. */
  int maxGroups();

  /** Returns false if the device is not available or the job failed. */
  bool run(const std::function<bool(ComputeJob&)>& job);

 private:
  friend class ComputeJob;
  class Thread;
  struct Task;

  struct Pipeline {
    std::unique_ptr<QRhiShaderResourceBindings> layout;
    std::unique_ptr<QRhiComputePipeline> pipeline;
  };

  Device();

  bool startThread();

  void threadLoop();

  bool initialize();

  void release();

  void markUnavailable(const QString& reason);

  QRhiShaderResourceBindings* createBindings(const std::vector<Binding>& bindings);

  std::mutex m_mutex;
  std::condition_variable m_cond;
  std::deque<Task*> m_tasks;
  std::unique_ptr<Thread> m_thread;
  bool m_stopping = false;
  State m_state = State::Unknown;
  QString m_name;
  int m_maxGroups = 0;

  // Accessed only on the device thread.
#ifdef SCANTAILOR_GPU_VULKAN
  std::unique_ptr<QVulkanInstance> m_vulkanInstance;
#endif
  std::unique_ptr<QRhi> m_rhi;
  std::unique_ptr<QRhiBuffer> m_dummyUniform;
  std::unique_ptr<QRhiBuffer> m_dummyStorage;
  std::map<QString, Pipeline> m_pipelines;
  bool m_deviceLost = false;
};
}  // namespace gpu
#endif  // SCANTAILOR_GPU_DEVICE_H_
