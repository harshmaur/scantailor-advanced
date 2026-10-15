add_library(gpu_test_utils STATIC TestImages.cpp TestImages.h)
target_link_libraries(gpu_test_utils PUBLIC imageproc ${Qt_Gui_lib})

add_executable(gpu_tests main.cpp TestGpuSavGol.cpp TestGpuTransform.cpp TestGpuDewarp.cpp)
target_link_libraries(
    gpu_tests
    PRIVATE gpu_test_utils gpu imageproc dewarping ${Qt_Gui_lib} Boost::unit_test_framework ${EXTRA_LIBS})
add_test(NAME gpu_tests COMMAND gpu_tests --log_level=message)

add_executable(gpu_bench GpuBench.cpp)
target_link_libraries(gpu_bench PRIVATE gpu_test_utils gpu imageproc dewarping ${Qt_Gui_lib} ${EXTRA_LIBS})
