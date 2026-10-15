set(sources
    Gpu.cpp Gpu.h
    Internal.h)

if (GPU_SUPPORT)
  list(APPEND sources
      Device.cpp Device.h
      Banding.cpp Banding.h)
else()
  list(APPEND sources Stub.cpp)
endif()

add_library(gpu STATIC ${sources})
target_link_libraries(gpu PUBLIC ${Qt_Core_lib})
target_include_directories(gpu PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

if (GPU_SUPPORT)
  target_link_libraries(gpu PRIVATE Qt6::Gui Qt6::GuiPrivate)
endif()
