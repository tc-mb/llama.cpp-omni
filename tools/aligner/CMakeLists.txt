add_library(qwen3_aligner STATIC
            qwen3-aligner.cpp
            qwen3-aligner.h
            )

target_link_libraries     (qwen3_aligner PUBLIC  ggml llama)
target_link_libraries     (qwen3_aligner PRIVATE mtmd Threads::Threads)
target_include_directories(qwen3_aligner PUBLIC  .)
target_include_directories(qwen3_aligner PRIVATE ../..)
target_include_directories(qwen3_aligner PRIVATE ../../vendor)
target_compile_features   (qwen3_aligner PRIVATE cxx_std_17)

if (BUILD_SHARED_LIBS)
    set_target_properties (qwen3_aligner PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

set(TARGET llama-aligner-cli)
add_executable          (${TARGET} aligner-cli.cpp)
set_target_properties   (${TARGET} PROPERTIES OUTPUT_NAME llama-aligner-cli)
install                 (TARGETS ${TARGET} RUNTIME)
target_link_libraries   (${TARGET} PRIVATE qwen3_aligner Threads::Threads)
target_compile_features (${TARGET} PRIVATE cxx_std_17)
