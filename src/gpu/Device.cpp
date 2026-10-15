// Copyright (C) 2026  ScanTailor Advanced contributors
// Use of this source code is governed by the GNU GPLv3 license that can be found in the LICENSE file.

#include "Device.h"

#include <rhi/qrhi_platform.h>
#include <rhi/qshader.h>

#include <QDebug>
#include <QFile>
#include <QGuiApplication>
#include <QStringList>
#include <QThread>
#include <QVarLengthArray>
#include <algorithm>
#include <exception>

#ifdef SCANTAILOR_GPU_VULKAN
#include <QVulkanInstance>
#endif

#include "Gpu.h"
#include "Internal.h"

#ifdef Q_OS_DARWIN
extern "C" void* objc_autoreleasePoolPush();
extern "C" void objc_autoreleasePoolPop(void* pool);
#endif

namespace gpu {
class Device::Thread : public QThread {
 public:
  explicit Thread(Device& device) : m_device(device) {}

 protected:
  void run() override { m_device.threadLoop(); }

 private:
  Device& m_device;
};

struct Device::Task {
  std::function<void()> fn;
  bool done = false;
};

Device::Device() = default;

Device& Device::instance() {
  static Device* device = new Device();
  return *device;
}

void Device::start() {
  std::lock_guard<std::mutex> lock(m_mutex);
  if (!isForcedOff()) {
    startThread();
  }
}

State Device::state() {
  if (isForcedOff()) {
    return State::Unavailable;
  }
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_state;
}

bool Device::isAvailable() {
  const State current = state();
  if (current != State::Unknown) {
    return current == State::Available;
  }
  run([](ComputeJob&) { return true; });
  return state() == State::Available;
}

QString Device::name() {
  if (!isAvailable()) {
    return QString();
  }
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_name;
}

int Device::maxGroups() {
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_maxGroups;
}

bool Device::run(const std::function<bool(ComputeJob&)>& job) {
  bool result = false;
  std::function<void()> fn = [this, &job, &result] {
    if (!m_rhi) {
      return;
    }
    {
      ComputeJob computeJob(*this);
      try {
        result = job(computeJob);
      } catch (const std::exception& e) {
        qWarning() << "GPU job failed:" << e.what();
        result = false;
      }
    }
    if (m_deviceLost) {
      release();
      markUnavailable(QStringLiteral("the GPU device was lost"));
      result = false;
    }
  };

  std::unique_lock<std::mutex> lock(m_mutex);
  if ((m_state == State::Unavailable) || m_stopping || !startThread()) {
    return false;
  }
  Task task{fn};
  m_tasks.push_back(&task);
  m_cond.notify_all();
  m_cond.wait(lock, [&task] { return task.done; });
  return result;
}

bool Device::startThread() {
  if (m_thread) {
    return true;
  }
  auto* app = qobject_cast<QGuiApplication*>(QCoreApplication::instance());
  if (!app || m_stopping) {
    return false;
  }
  m_thread = std::make_unique<Thread>(*this);
  m_thread->start();
  QObject::connect(app, &QCoreApplication::aboutToQuit, [] { Device::instance().shutdown(); });
  qAddPostRoutine([] { Device::instance().shutdown(); });
  return true;
}

void Device::shutdown() {
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_thread || m_stopping) {
      return;
    }
    m_stopping = true;
  }
  m_cond.notify_all();
  m_thread->wait();
  std::lock_guard<std::mutex> lock(m_mutex);
  m_state = State::Unavailable;
}

void Device::threadLoop() {
  initialize();
  std::unique_lock<std::mutex> lock(m_mutex);
  while (true) {
    m_cond.wait(lock, [this] { return m_stopping || !m_tasks.empty(); });
    if (m_tasks.empty()) {
      break;
    }
    Task* task = m_tasks.front();
    m_tasks.pop_front();
    lock.unlock();
#ifdef Q_OS_DARWIN
    void* const pool = objc_autoreleasePoolPush();
    task->fn();
    objc_autoreleasePoolPop(pool);
#else
    task->fn();
#endif
    lock.lock();
    task->done = true;
    m_cond.notify_all();
  }
  lock.unlock();
  release();
}

bool Device::initialize() {
  const bool allowSoftware = qgetenv("SCANTAILOR_GPU_SOFTWARE") == "1";
  const QByteArray backend = qgetenv("SCANTAILOR_GPU_BACKEND").toLower();
  auto wanted = [&backend](const char* name) { return backend.isEmpty() || (backend == name); };
  QRhi::Flags flags;
  if (allowSoftware) {
    flags |= QRhi::PreferSoftwareRenderer;
  }

  QStringList errors;
  auto tryCreate = [&](const QRhi::Implementation impl, QRhiInitParams* params, const char* apiName) {
    std::unique_ptr<QRhi> rhi(QRhi::create(impl, params, flags));
    if (!rhi) {
      errors.push_back(QStringLiteral("%1: failed to create a device").arg(QLatin1String(apiName)));
      return false;
    }
    if (!rhi->isFeatureSupported(QRhi::Compute)) {
      errors.push_back(QStringLiteral("%1: compute shaders are not supported").arg(QLatin1String(apiName)));
      return false;
    }
    if ((rhi->driverInfo().deviceType == QRhiDriverInfo::CpuDevice) && !allowSoftware) {
      errors.push_back(QStringLiteral("%1: %2 is a software renderer (set SCANTAILOR_GPU_SOFTWARE=1 to use it)")
                           .arg(QLatin1String(apiName), QString::fromUtf8(rhi->driverInfo().deviceName)));
      return false;
    }
    m_rhi = std::move(rhi);
    return true;
  };

#if defined(Q_OS_MACOS) || defined(Q_OS_IOS)
  QRhiMetalInitParams metalParams;
  if (wanted("metal")) {
    tryCreate(QRhi::Metal, &metalParams, "Metal");
  }
#elif defined(Q_OS_WIN)
  QRhiD3D12InitParams d3d12Params;
  QRhiD3D11InitParams d3d11Params;
  if (wanted("d3d12")) {
    tryCreate(QRhi::D3D12, &d3d12Params, "Direct3D 12");
  }
  if (!m_rhi && wanted("d3d11")) {
    tryCreate(QRhi::D3D11, &d3d11Params, "Direct3D 11");
  }
#elif defined(SCANTAILOR_GPU_VULKAN)
  if (wanted("vulkan")) {
    m_vulkanInstance = std::make_unique<QVulkanInstance>();
    m_vulkanInstance->setExtensions(QRhiVulkanInitParams::preferredInstanceExtensions());
    if (m_vulkanInstance->create()) {
      QRhiVulkanInitParams vulkanParams;
      vulkanParams.inst = m_vulkanInstance.get();
      tryCreate(QRhi::Vulkan, &vulkanParams, "Vulkan");
    } else {
      errors.push_back(
          QStringLiteral("Vulkan: failed to create an instance (error %1)").arg(m_vulkanInstance->errorCode()));
    }
  }
#else
  errors.push_back(QStringLiteral("Qt was built without a supported graphics API"));
#endif

  if (!m_rhi && errors.isEmpty()) {
    errors.push_back(QStringLiteral("SCANTAILOR_GPU_BACKEND=%1 is not available on this platform")
                         .arg(QString::fromLatin1(backend)));
  }
  if (m_rhi) {
    m_dummyUniform.reset(m_rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, 256));
    m_dummyStorage.reset(m_rhi->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, 256));
    if (!m_dummyUniform->create() || !m_dummyStorage->create()) {
      errors.push_back(QStringLiteral("failed to create buffers"));
      release();
    }
  }
  if (!m_rhi) {
    release();
    markUnavailable(errors.join(QStringLiteral("; ")));
    return false;
  }

  std::lock_guard<std::mutex> lock(m_mutex);
  m_state = State::Available;
  m_maxGroups = std::min(65535, m_rhi->resourceLimit(QRhi::MaxThreadGroupsPerDimension));
  m_name = QStringLiteral("%1 (%2)").arg(QString::fromUtf8(m_rhi->driverInfo().deviceName),
                                         QLatin1String(m_rhi->backendName()));
  return true;
}

void Device::release() {
  m_pipelines.clear();
  m_dummyStorage.reset();
  m_dummyUniform.reset();
  m_rhi.reset();
#ifdef SCANTAILOR_GPU_VULKAN
  m_vulkanInstance.reset();
#endif
}

void Device::markUnavailable(const QString& reason) {
  std::lock_guard<std::mutex> lock(m_mutex);
  if (m_state != State::Unavailable) {
    m_state = State::Unavailable;
    qWarning().noquote() << "GPU processing is not available:" << reason;
  }
}

QRhiShaderResourceBindings* Device::createBindings(const std::vector<Binding>& bindings) {
  QVarLengthArray<QRhiShaderResourceBinding, 8> list;
  const auto stage = QRhiShaderResourceBinding::ComputeStage;
  for (size_t i = 0; i < bindings.size(); ++i) {
    const int point = static_cast<int>(i);
    QRhiBuffer* const buffer = bindings[i].buffer;
    switch (bindings[i].access) {
      case Access::Uniform:
        list.push_back(QRhiShaderResourceBinding::uniformBuffer(point, stage, buffer));
        break;
      case Access::Read:
        list.push_back(QRhiShaderResourceBinding::bufferLoad(point, stage, buffer));
        break;
      case Access::Write:
        list.push_back(QRhiShaderResourceBinding::bufferStore(point, stage, buffer));
        break;
      case Access::ReadWrite:
        list.push_back(QRhiShaderResourceBinding::bufferLoadStore(point, stage, buffer));
        break;
    }
  }
  std::unique_ptr<QRhiShaderResourceBindings> srb(m_rhi->newShaderResourceBindings());
  srb->setBindings(list.cbegin(), list.cend());
  if (!srb->create()) {
    return nullptr;
  }
  return srb.release();
}

ComputeJob::ComputeJob(Device& device) : m_device(device) {}

QRhi* ComputeJob::rhi() const {
  return m_device.m_rhi.get();
}

QRhiBuffer* ComputeJob::buffer(const Access access, const quint32 size) {
  std::unique_ptr<QRhiBuffer> buffer((access == Access::Uniform)
                                         ? rhi()->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, size)
                                         : rhi()->newBuffer(QRhiBuffer::Static, QRhiBuffer::StorageBuffer, size));
  if (!buffer->create()) {
    return nullptr;
  }
  m_buffers.push_back(std::move(buffer));
  return m_buffers.back().get();
}

QRhiShaderResourceBindings* ComputeJob::bindings(const std::vector<Binding>& bindings) {
  for (const Binding& binding : bindings) {
    if (!binding.buffer) {
      return nullptr;
    }
  }
  QRhiShaderResourceBindings* srb = m_device.createBindings(bindings);
  if (srb) {
    m_bindings.emplace_back(srb);
  }
  return srb;
}

QRhiComputePipeline* ComputeJob::pipeline(const QString& shader, const std::vector<Binding>& layout) {
  auto it = m_device.m_pipelines.find(shader);
  if (it != m_device.m_pipelines.end()) {
    return it->second.pipeline.get();
  }

  std::vector<Binding> dummyBindings(layout);
  for (Binding& binding : dummyBindings) {
    binding.buffer
        = (binding.access == Access::Uniform) ? m_device.m_dummyUniform.get() : m_device.m_dummyStorage.get();
  }
  Device::Pipeline pipeline;
  pipeline.layout.reset(m_device.createBindings(dummyBindings));
  if (!pipeline.layout) {
    return nullptr;
  }

  QFile file(shader);
  if (!file.open(QIODevice::ReadOnly)) {
    qWarning() << "GPU: cannot open shader" << shader;
    return nullptr;
  }
  const QShader computeShader = QShader::fromSerialized(file.readAll());
  if (!computeShader.isValid()) {
    qWarning() << "GPU: invalid shader" << shader;
    return nullptr;
  }

  pipeline.pipeline.reset(rhi()->newComputePipeline());
  pipeline.pipeline->setShaderStage(QRhiShaderStage(QRhiShaderStage::Compute, computeShader));
  pipeline.pipeline->setShaderResourceBindings(pipeline.layout.get());
  if (!pipeline.pipeline->create()) {
    qWarning() << "GPU: failed to create the pipeline for" << shader;
    return nullptr;
  }
  QRhiComputePipeline* const result = pipeline.pipeline.get();
  m_device.m_pipelines.emplace(shader, std::move(pipeline));
  return result;
}

QRhiResourceUpdateBatch* ComputeJob::updates() {
  return rhi()->nextResourceUpdateBatch();
}

bool ComputeJob::dispatch(QRhiComputePipeline* const pipeline,
                          QRhiShaderResourceBindings* const bindings,
                          const int groupsX,
                          const int groupsY,
                          QRhiResourceUpdateBatch* const before,
                          QRhiResourceUpdateBatch* const after) {
  QRhiCommandBuffer* cb = nullptr;
  const QRhi::FrameOpResult beginResult = (pipeline && bindings) ? rhi()->beginOffscreenFrame(&cb) : QRhi::FrameOpError;
  if (beginResult != QRhi::FrameOpSuccess) {
    if (before) {
      before->release();
    }
    if (after) {
      after->release();
    }
    m_device.m_deviceLost = m_device.m_deviceLost || (beginResult == QRhi::FrameOpDeviceLost);
    return false;
  }

  cb->beginComputePass(before);
  cb->setComputePipeline(pipeline);
  cb->setShaderResources(bindings);
  cb->dispatch(groupsX, groupsY, 1);
  cb->endComputePass(after);

  const QRhi::FrameOpResult endResult = rhi()->endOffscreenFrame();
  if (endResult != QRhi::FrameOpSuccess) {
    m_device.m_deviceLost = m_device.m_deviceLost || (endResult == QRhi::FrameOpDeviceLost);
    return false;
  }
  countDispatch();
  return true;
}

bool shouldUseGpu(const qint64 pixels,
                  const int bytesPerPixel,
                  const int srcWidth,
                  const int srcBytesPerLine,
                  const int dstWidth,
                  const int dstBytesPerLine) {
  return (pixels >= minPixels()) && isEnabled() && (srcBytesPerLine % 4 == 0) && (dstBytesPerLine % 4 == 0)
         && (srcBytesPerLine >= srcWidth * bytesPerPixel) && (dstBytesPerLine >= dstWidth * bytesPerPixel)
         && Device::instance().isAvailable();
}

QString shaderPath(const char* kernel, const TransformFormat format) {
  const char* const suffix = (format == TransformFormat::Gray8)   ? "gray"
                             : (format == TransformFormat::Rgb32) ? "rgb"
                                                                  : "argb";
  return QStringLiteral(":/gpu/%1_%2.comp.qsb").arg(QLatin1String(kernel), QLatin1String(suffix));
}

void start() {
  Device::instance().start();
}

State state() {
  return Device::instance().state();
}

bool isAvailable() {
  return Device::instance().isAvailable();
}

QString deviceName() {
  return Device::instance().name();
}
}  // namespace gpu
