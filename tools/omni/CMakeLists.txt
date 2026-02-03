# omni

find_package(Threads REQUIRED)

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

target_link_libraries     (omni PUBLIC ggml llama common)
target_link_libraries     (omni PRIVATE Threads::Threads)
target_include_directories(omni PUBLIC  .)
target_include_directories(omni PRIVATE ../..)
target_include_directories(omni PRIVATE ../../vendor)
target_compile_features   (omni PRIVATE cxx_std_17)

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

# # TTS 推理测试程序
# add_executable(test_tts_inference test/test_tts_inference.cpp)
# target_link_libraries(test_tts_inference PRIVATE common omni Threads::Threads)
# target_compile_features(test_tts_inference PRIVATE cxx_std_17)

# # Test TTS hidden states
# add_executable(test_tts_hidden_states test/test_tts_hidden_states.cpp)
# target_link_libraries(test_tts_hidden_states PRIVATE common omni Threads::Threads)
# target_compile_features(test_tts_hidden_states PRIVATE cxx_std_17)

# # 纯文本输入测试 - 完整 C++ TTS 流程
# add_executable(omni-text-test test/omni-cli-text-test.cpp)
# target_link_libraries(omni-text-test PRIVATE common omni Threads::Threads)
# target_compile_features(omni-text-test PRIVATE cxx_std_17)

# # Token2Wav 测试程序 - 完整 C++ 语音合成
# add_executable(token2wav-example token2wav/token2wav-example.cpp)
# target_link_libraries(token2wav-example PRIVATE common omni Threads::Threads)
# target_compile_features(token2wav-example PRIVATE cxx_std_17)

