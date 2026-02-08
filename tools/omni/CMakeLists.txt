# omni

find_package(Threads REQUIRED)

# CoreML / ANE support option
option(ENABLE_COREML "Enable CoreML/ANE support for vision encoder" OFF)

add_library(omni
            omni.cpp
            omni.h
            audition.cpp
            audition.h
            vision.cpp
            vision.h
            omni-impl.h
            token2wav/token2wav-impl.cpp
            token2wav/token2wav-impl.h
            token2wav/token2wav.cpp
            token2wav/token2wav.h
            )

# Add CoreML related files when enabled
if(ENABLE_COREML)
    target_sources(omni PRIVATE
        coreml/omni_coreml.h
        coreml/omni_coreml.mm
        coreml/coreml_minicpmo45_vit_all_f16.h
        coreml/coreml_minicpmo45_vit_all_f16.m
    )
    # Define compile-time macro for code guards
    target_compile_definitions(omni PRIVATE ENABLE_COREML)

    # Enable ARC for Objective-C files
    set_source_files_properties(coreml/omni_coreml.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    set_source_files_properties(coreml/coreml_minicpmo45_vit_all_f16.m PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

target_link_libraries     (omni PUBLIC ggml llama common)
target_link_libraries     (omni PRIVATE Threads::Threads)
target_include_directories(omni PUBLIC  .)
target_include_directories(omni PRIVATE ../..)
target_include_directories(omni PRIVATE ../../vendor)
target_compile_features   (omni PRIVATE cxx_std_17)

# Link CoreML and Accelerate frameworks when CoreML is enabled
if(ENABLE_COREML)
    target_link_libraries(omni PRIVATE
        "-framework Foundation"
        "-framework CoreML"
        "-framework Accelerate"
        "-ObjC"
    )
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties     (omni PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(omni PRIVATE LLAMA_BUILD)
    target_compile_definitions(omni PUBLIC  LLAMA_SHARED)
endif()

set(OMNI_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/omni.h
    )

set_target_properties(omni
    PROPERTIES
    PUBLIC_HEADER "${MTMD_PUBLIC_HEADERS}")

install(TARGETS omni LIBRARY PUBLIC_HEADER)

if (NOT MSVC)
    # for stb_image.h and miniaudio.h
    target_compile_options(omni PRIVATE -Wno-cast-qual)
endif()

if (TARGET BUILD_INFO)
    add_dependencies(omni        BUILD_INFO)
endif()

set(TARGET llama-omni-cli)
add_executable         (${TARGET} omni-cli.cpp)
set_target_properties  (${TARGET} PROPERTIES OUTPUT_NAME llama-omni-cli)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_link_libraries  (${TARGET} PRIVATE common omni Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# # Token2Wav 测试程序 - 完整 C++ 语音合成
# add_executable(token2wav-example token2wav/token2wav-example.cpp)
# target_link_libraries(token2wav-example PRIVATE common omni Threads::Threads)
# target_compile_features(token2wav-example PRIVATE cxx_std_17)

# ==================== pybind11 Python binding ====================
# 构建 omni_engine.cpython-*.so，供 Python 直接 import
# 用法: cmake .. -DBUILD_PYBIND=ON -DPython3_EXECUTABLE=/path/to/python
option(BUILD_PYBIND "Build pybind11 Python module (omni_engine)" OFF)
if(BUILD_PYBIND)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    # 使用 pip 安装的 pybind11 的 CMake 配置
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_package(pybind11 REQUIRED HINTS ${pybind11_DIR})

    pybind11_add_module(omni_engine py_omni.cpp)
    target_link_libraries(omni_engine PRIVATE omni common ggml llama Threads::Threads)
    target_include_directories(omni_engine PRIVATE . ../.. ../../vendor)
    target_compile_features(omni_engine PRIVATE cxx_std_17)

    # CoreML frameworks (macOS)
    if(ENABLE_COREML)
        target_link_libraries(omni_engine PRIVATE
            "-framework Foundation"
            "-framework CoreML"
            "-framework Accelerate"
        )
    endif()

    message(STATUS "pybind11: omni_engine module will be built")
    message(STATUS "  Python: ${Python3_EXECUTABLE}")
    message(STATUS "  pybind11: ${pybind11_DIR}")
endif()

