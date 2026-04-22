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
            token2wav/token2wav-profile.h
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

# Phase 0.3: 给 token2wav-impl.cpp 里所有 suspicious op 的返回值贴上
# "OP@file:line" 的名字，配合 OMNI_T2W_PRINT_GRAPH=1 做源码级归因。
# Debug / profile 用，release 保持关闭。
option(OMNI_T2W_TRACK_OPS "tag token2wav ops with source location (debug only)" OFF)
if (OMNI_T2W_TRACK_OPS)
    set_source_files_properties(token2wav/token2wav-impl.cpp
        PROPERTIES COMPILE_DEFINITIONS OMNI_T2W_TRACK_OPS=1)
endif()

# nsys-friendly NVTX 标注（仅在能找到 CUDA 头文件时启用，header-only，无需 link 库）
if (GGML_CUDA OR GGML_USE_CUDA)
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND)
        target_include_directories(omni PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    elseif (EXISTS "/usr/local/cuda/include/nvtx3/nvToolsExt.h")
        target_include_directories(omni PRIVATE /usr/local/cuda/include)
    endif()
endif()

# Link CoreML and Accelerate frameworks when CoreML is enabled
if(ENABLE_COREML)
    target_link_libraries(omni PRIVATE
        "-framework Foundation"
        "-framework CoreML"
        "-framework Accelerate"
        "-ObjC"
    )
endif()

# Windows-specific definitions
if (WIN32)
    target_compile_definitions(omni PRIVATE NOMINMAX _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties     (omni PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(omni PRIVATE LLAMA_BUILD)
    target_compile_definitions(omni PUBLIC  LLAMA_SHARED)
    if (WIN32)
        set_target_properties(omni PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
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

# 单工模式 - 纯音频测试（批量 prefill + 单次 decode）
set(TARGET_SINGLE_TEST_AUDIO llama-omni-single-test-audio)
add_executable         (${TARGET_SINGLE_TEST_AUDIO} test/single_test_audio.cpp)
set_target_properties  (${TARGET_SINGLE_TEST_AUDIO} PROPERTIES OUTPUT_NAME llama-omni-single-test-audio)
target_link_libraries  (${TARGET_SINGLE_TEST_AUDIO} PRIVATE common omni Threads::Threads)
target_compile_features(${TARGET_SINGLE_TEST_AUDIO} PRIVATE cxx_std_17)

# 单工模式 - Omni (audio+vision) 测试
set(TARGET_SINGLE_TEST_OMNI llama-omni-single-test-omni)
add_executable         (${TARGET_SINGLE_TEST_OMNI} test/single_test_omni.cpp)
set_target_properties  (${TARGET_SINGLE_TEST_OMNI} PROPERTIES OUTPUT_NAME llama-omni-single-test-omni)
target_link_libraries  (${TARGET_SINGLE_TEST_OMNI} PRIVATE common omni Threads::Threads)
target_compile_features(${TARGET_SINGLE_TEST_OMNI} PRIVATE cxx_std_17)

# 双工模式测试
set(TARGET_DUPLEX_TEST llama-omni-test-duplex)
add_executable         (${TARGET_DUPLEX_TEST} test/test-duplex.cpp)
set_target_properties  (${TARGET_DUPLEX_TEST} PROPERTIES OUTPUT_NAME llama-omni-test-duplex)
target_link_libraries  (${TARGET_DUPLEX_TEST} PRIVATE common omni Threads::Threads)
target_compile_features(${TARGET_DUPLEX_TEST} PRIVATE cxx_std_17)

# Token2Wav 测试程序 - 纯 C++ 单线程 baseline（用于 profile 对照，不占 LLM/TTS GPU）
add_executable(token2wav-example token2wav/token2wav-example.cpp)
target_link_libraries(token2wav-example PRIVATE common omni Threads::Threads)
target_compile_features(token2wav-example PRIVATE cxx_std_17)
