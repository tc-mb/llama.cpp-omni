# omni

find_package(Threads REQUIRED)

# CoreML / ANE support option：在 macOS 上默认开，其它平台默认关
if(APPLE)
    option(ENABLE_COREML "Enable CoreML/ANE support for vision encoder" ON)
else()
    option(ENABLE_COREML "Enable CoreML/ANE support for vision encoder" OFF)
endif()

add_library(omni
            omni.cpp
            omni.h
            tts-condition-graph.cpp
            tts-condition-graph.h
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
        coreml/coreml_t2w_dit.h
        coreml/coreml_t2w_dit.mm
    )
    # Define compile-time macro for code guards
    target_compile_definitions(omni PRIVATE ENABLE_COREML)

    # Enable ARC for Objective-C files
    set_source_files_properties(coreml/omni_coreml.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    set_source_files_properties(coreml/coreml_minicpmo45_vit_all_f16.m PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    set_source_files_properties(coreml/coreml_t2w_dit.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

target_link_libraries     (omni PUBLIC ggml llama llama-common)
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
    PUBLIC_HEADER "${OMNI_PUBLIC_HEADERS}")

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
target_link_libraries  (${TARGET} PRIVATE llama-common omni Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# 单工模式 - 纯音频测试（批量 prefill + 单次 decode）
set(TARGET_SINGLE_TEST_AUDIO llama-omni-single-test-audio)
add_executable         (${TARGET_SINGLE_TEST_AUDIO} test/single_test_audio.cpp)
set_target_properties  (${TARGET_SINGLE_TEST_AUDIO} PROPERTIES OUTPUT_NAME llama-omni-single-test-audio)
target_link_libraries  (${TARGET_SINGLE_TEST_AUDIO} PRIVATE llama-common omni Threads::Threads)
target_compile_features(${TARGET_SINGLE_TEST_AUDIO} PRIVATE cxx_std_17)

# 单工模式 - Omni (audio+vision) 测试
set(TARGET_SINGLE_TEST_OMNI llama-omni-single-test-omni)
add_executable         (${TARGET_SINGLE_TEST_OMNI} test/single_test_omni.cpp)
set_target_properties  (${TARGET_SINGLE_TEST_OMNI} PROPERTIES OUTPUT_NAME llama-omni-single-test-omni)
target_link_libraries  (${TARGET_SINGLE_TEST_OMNI} PRIVATE llama-common omni Threads::Threads)
target_compile_features(${TARGET_SINGLE_TEST_OMNI} PRIVATE cxx_std_17)

# VoxCPM2 MiniCPMLM wrapper (shared with omni and test)
add_library(voxcpm2_llm STATIC
    voxcpm2/voxcpm2_llm.cpp
    voxcpm2/voxcpm2_llm.h
)
target_link_libraries     (voxcpm2_llm PUBLIC llama llama-common)
target_include_directories(voxcpm2_llm PUBLIC  . voxcpm2)
target_include_directories(voxcpm2_llm PRIVATE ../..)
target_compile_features   (voxcpm2_llm PRIVATE cxx_std_17)

# VoxCPM2 FSQ module
add_library(voxcpm2_fsq STATIC
    voxcpm2/voxcpm2_fsq.cpp
    voxcpm2/voxcpm2_fsq.h
)
target_link_libraries     (voxcpm2_fsq PUBLIC ggml llama-common)
target_include_directories(voxcpm2_fsq PUBLIC  . voxcpm2)
target_include_directories(voxcpm2_fsq PRIVATE ../..)
target_compile_features   (voxcpm2_fsq PRIVATE cxx_std_17)

# VoxCPM2 acoustic transformer helpers (LocEnc / LocDiT)
add_library(voxcpm2_acoustic STATIC
    voxcpm2/voxcpm2_transformer.cpp
    voxcpm2/voxcpm2_transformer.h
    voxcpm2/voxcpm2_localenc.cpp
    voxcpm2/voxcpm2_localenc.h
    voxcpm2/voxcpm2_locdit.cpp
    voxcpm2/voxcpm2_locdit.h
    voxcpm2/voxcpm2_cfm.cpp
    voxcpm2/voxcpm2_cfm.h
    voxcpm2/voxcpm2_components.cpp
    voxcpm2/voxcpm2_components.h
    voxcpm2/voxcpm2_audiovae.cpp
    voxcpm2/voxcpm2_audiovae.h
)
target_link_libraries     (voxcpm2_acoustic PUBLIC ggml llama-common)
target_include_directories(voxcpm2_acoustic PUBLIC  . voxcpm2)
target_include_directories(voxcpm2_acoustic PRIVATE ../..)
target_compile_features   (voxcpm2_acoustic PRIVATE cxx_std_17)

# VoxCPM2 integrated runtime
add_library(voxcpm2_runtime STATIC
    voxcpm2/voxcpm2_runtime.cpp
    voxcpm2/voxcpm2_runtime.h
)
target_link_libraries     (voxcpm2_runtime PUBLIC voxcpm2_llm voxcpm2_fsq voxcpm2_acoustic llama ggml llama-common)
target_include_directories(voxcpm2_runtime PUBLIC  . voxcpm2)
target_include_directories(voxcpm2_runtime PRIVATE ../..)
target_compile_features   (voxcpm2_runtime PRIVATE cxx_std_17)

# 双工模式测试
set(TARGET_DUPLEX_TEST llama-omni-test-duplex)
add_executable         (${TARGET_DUPLEX_TEST} test/test-duplex.cpp)
set_target_properties  (${TARGET_DUPLEX_TEST} PROPERTIES OUTPUT_NAME llama-omni-test-duplex)
target_link_libraries  (${TARGET_DUPLEX_TEST} PRIVATE llama-common omni Threads::Threads)
target_compile_features(${TARGET_DUPLEX_TEST} PRIVATE cxx_std_17)

# 双工可行性 profiling（低侵入：仅复用 omni API + audio_output_cb 钩子）
set(TARGET_DUPLEX_PERF llama-omni-perf-duplex)
add_executable         (${TARGET_DUPLEX_PERF} perf/perf-duplex.cpp)
set_target_properties  (${TARGET_DUPLEX_PERF} PROPERTIES OUTPUT_NAME llama-omni-perf-duplex)
target_link_libraries  (${TARGET_DUPLEX_PERF} PRIVATE llama-common omni Threads::Threads)
target_compile_features(${TARGET_DUPLEX_PERF} PRIVATE cxx_std_17)

# Token2Wav 测试程序 - 纯 C++ 单线程 baseline（用于 profile 对照，不占 LLM/TTS GPU）
add_executable(token2wav-example token2wav/token2wav-example.cpp)
target_link_libraries(token2wav-example PRIVATE llama-common omni Threads::Threads)
target_compile_features(token2wav-example PRIVATE cxx_std_17)

# VoxCPM2 CLI demo
set(TARGET_VOXCPM2_CLI voxcpm2-cli)
add_executable         (${TARGET_VOXCPM2_CLI} voxcpm2/voxcpm2_cli.cpp)
set_target_properties  (${TARGET_VOXCPM2_CLI} PROPERTIES OUTPUT_NAME voxcpm2-cli)
target_link_libraries  (${TARGET_VOXCPM2_CLI} PRIVATE voxcpm2_runtime llama-common ggml Threads::Threads)
target_include_directories(${TARGET_VOXCPM2_CLI} PRIVATE voxcpm2)
target_compile_features(${TARGET_VOXCPM2_CLI} PRIVATE cxx_std_17)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET_VOXCPM2_CLI} RUNTIME)
endif()

# Token2Mel-DiT CoreML 单元测试（仅 macOS + CoreML enable 时有效）
if(ENABLE_COREML)
    set(TARGET_T2W_DIT_ANE_TEST llama-omni-test-t2w-dit-ane)
    add_executable         (${TARGET_T2W_DIT_ANE_TEST} test/test_t2w_dit_ane.cpp)
    set_target_properties  (${TARGET_T2W_DIT_ANE_TEST} PROPERTIES OUTPUT_NAME llama-omni-test-t2w-dit-ane)
    target_link_libraries  (${TARGET_T2W_DIT_ANE_TEST} PRIVATE llama-common omni Threads::Threads)
    target_compile_features(${TARGET_T2W_DIT_ANE_TEST} PRIVATE cxx_std_17)

    # CoreML 端到端 test：用真实 prompt + 真实 mu 跑完整 5-step ODE × N chunks
    set(TARGET_T2W_ANE_E2E llama-omni-test-t2w-ane-end2end)
    add_executable         (${TARGET_T2W_ANE_E2E} test/test_t2w_ane_end2end.cpp)
    set_target_properties  (${TARGET_T2W_ANE_E2E} PROPERTIES OUTPUT_NAME llama-omni-test-t2w-ane-end2end)
    target_link_libraries  (${TARGET_T2W_ANE_E2E} PRIVATE llama-common omni Threads::Threads)
    target_compile_features(${TARGET_T2W_ANE_E2E} PRIVATE cxx_std_17)
endif()
