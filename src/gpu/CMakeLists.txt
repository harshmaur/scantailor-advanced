set(sources
    Gpu.cpp Gpu.h
    Internal.h
    SavGolFilterKernel.h
    TransformKernel.h
    DewarpKernel.h)

if (GPU_SUPPORT)
  list(APPEND sources
      Device.cpp Device.h
      Banding.cpp Banding.h
      SavGolFilterKernel.cpp
      TransformKernel.cpp
      DewarpKernel.cpp)
else()
  list(APPEND sources Stub.cpp)
endif()

add_library(gpu STATIC ${sources})
target_link_libraries(gpu PUBLIC ${Qt_Core_lib})
target_include_directories(gpu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

if (GPU_SUPPORT)
  target_link_libraries(gpu PRIVATE Qt6::Gui Qt6::GuiPrivate)
  foreach (format gray rgb argb)
    string(TOUPPER "${format}" format_define)
    qt_add_shaders(
        gpu "gpu_transform_${format}"
        PREFIX "/gpu"
        GLSL "430"
        HLSL 50
        MSL 12
        QUIET
        DEFINES "FORMAT_${format_define}=1"
        FILES shaders/transform.comp
        OUTPUTS "transform_${format}.comp.qsb")
    qt_add_shaders(
        gpu "gpu_dewarp_${format}"
        PREFIX "/gpu"
        GLSL "430"
        HLSL 50
        MSL 12
        QUIET
        DEFINES "FORMAT_${format_define}=1"
        FILES shaders/dewarp.comp
        OUTPUTS "dewarp_${format}.comp.qsb")
  endforeach()
  qt_add_shaders(
      gpu "gpu_savgol"
      PREFIX "/gpu"
      GLSL "430"
      HLSL 50
      MSL 12
      QUIET
      FILES shaders/savgol.comp
      OUTPUTS "savgol.comp.qsb")
endif()

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()
