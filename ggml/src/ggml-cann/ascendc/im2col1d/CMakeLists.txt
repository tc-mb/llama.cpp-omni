set(ASCEND_CANN_PACKAGE_PATH "${CANN_INSTALL_DIR}")
set(ASCEND_KERNEL_LAUNCH_ONLY ON)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" GGML_CANN_ASCENDC_HOST_ARCH)
if (GGML_CANN_ASCENDC_HOST_ARCH STREQUAL "x86_64" OR GGML_CANN_ASCENDC_HOST_ARCH STREQUAL "amd64")
    set(GGML_CANN_ASCENDC_PLATFORM_DIR "x86_64-linux")
else()
    set(GGML_CANN_ASCENDC_PLATFORM_DIR "aarch64-linux")
endif()

string(TOLOWER "${SOC_TYPE}" GGML_CANN_ASCENDC_SOC_TYPE)
if (GGML_CANN_ASCENDC_SOC_TYPE STREQUAL "ascend910" OR
    GGML_CANN_ASCENDC_SOC_TYPE STREQUAL "ascend910c")
    # The benchmark config uses the generic Ascend910 family name.  AscendC
    # requires the concrete 910C compiler target used by this platform.
    set(SOC_VERSION "ascend910_9382")
else()
    set(SOC_VERSION "${GGML_CANN_ASCENDC_SOC_TYPE}")
endif()

include("${CANN_INSTALL_DIR}/${GGML_CANN_ASCENDC_PLATFORM_DIR}/tikcpp/ascendc_kernel_cmake/ascendc.cmake")

ascendc_library(ggml_cann_im2col1d_kernel STATIC
    im2col1d_f32.cpp)
set_target_properties(ggml_cann_im2col1d_kernel PROPERTIES
    OUTPUT_NAME ggml-cann-im2col1d-kernel)

ascendc_library(ggml_cann_set_rows_f32_f16_kernel STATIC
    ../set_rows_f32_f16/set_rows_f32_f16.cpp)
set_target_properties(ggml_cann_set_rows_f32_f16_kernel PROPERTIES
    OUTPUT_NAME ggml-cann-set-rows-f32-f16-kernel)

ascendc_library(ggml_cann_kv_pair_update_kernel STATIC
    ../kv_pair_update/kv_pair_update.cpp)
set_target_properties(ggml_cann_kv_pair_update_kernel PROPERTIES
    OUTPUT_NAME ggml-cann-kv-pair-update-kernel)

ascendc_library(ggml_cann_modulate_fusion_kernel STATIC
    ../modulate_fusion/modulate_fusion.cpp)
set_target_properties(ggml_cann_modulate_fusion_kernel PROPERTIES
    OUTPUT_NAME ggml-cann-modulate-fusion-kernel)

ascendc_library(ggml_cann_attn_time_pack_kernel STATIC
    ../attn_time_pack/attn_time_pack.cpp)
set_target_properties(ggml_cann_attn_time_pack_kernel PROPERTIES
    OUTPUT_NAME ggml-cann-attn-time-pack-kernel)
