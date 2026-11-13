#pragma once

#include "llama.h"

#include "llama-impl.h"
#include "llama-arch.h"
#include "llama-hparams.h"
#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <cstddef>
#include <cstring>
#include <map>
#include <stdexcept>
#include <unordered_map>

using llama_buf_map = std::unordered_map<uint32_t, ggml_backend_buffer_t>;

// lists of buffer types used for each layer
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

enum llama_fver {
    GGUF_FILE_VERSION_V1 = 1,
    GGUF_FILE_VERSION_V2 = 2,
    GGUF_FILE_VERSION_V3 = 3,
};

const char * llama_file_version_name(llama_fver version);

struct llama_model_loader {
    // 一个专家区载体张量：一层交错后的专家字节放在一个 I8 张量里。它的内容不是任何
    // 架构绑定的权重，含义由模型文件自带的布局声明赋予。take_expert_region() 把它从
    // weights_map 里取出后记在这里：文件下标 + 绝对偏移 + 字节数。
    struct llama_moe_region {
        uint16_t idx    = 0;   // 源文件下标
        size_t   offset = 0;   // 张量数据在文件内的绝对偏移
        size_t   nbytes = 0;
    };

    // Holds information on a model weight
    struct llama_tensor_weight {
        uint16_t  idx; // source file index
        size_t   offs; // tensor data offset in the original file

        ggml_tensor * tensor;

        llama_tensor_weight(const llama_file * file, uint16_t idx, const struct gguf_context * gguf_ctx, ggml_tensor * tensor) : idx(idx), tensor(tensor) {
            const int tensor_idx = gguf_find_tensor(gguf_ctx,  ggml_get_name(tensor));
            if (tensor_idx < 0) {
                throw std::runtime_error(format("tensor '%s' not found in the model", ggml_get_name(tensor)));
            }

            offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
            if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size()) {
                throw std::runtime_error(format("tensor '%s' data is not within the file bounds, model is corrupted or incomplete", ggml_get_name(tensor)));
            }
        }
    };

    // custom comparator to sort weights more nicely by layer
    struct weight_name_comparer {
        bool operator()(const std::string & a, const std::string & b) const {
            int a_layer = -1;
            int b_layer = -1;
            sscanf(a.c_str(), "blk.%d.", &a_layer);
            sscanf(b.c_str(), "blk.%d.", &b_layer);
            if (a_layer != b_layer) {
                return a_layer < b_layer;
            }
            return a < b;
        }
    };

    static const int TENSOR_NOT_REQUIRED    = 1 << 0;
    static const int TENSOR_DUPLICATED      = 1 << 1;
    static const int TENSOR_SKIP            = 1 << 2;
    static const int TENSOR_SKIP_IF_VIRTUAL = 1 << 3;

    int n_kv      = 0;
    int n_tensors = 0;
    int n_created = 0;

    /// 其中「不来自 gguf、由 create_tensor_alloc 合成」的张量数。
    /// gguf 剥离专家后，这些张量在文件里没有对应项，校验张量总数时要排除。
    int n_created_synthesized = 0;

    /// 最近一次 `create_tensor` 用的 ggml context。见 create_tensor 里的注释。
    struct ggml_context * last_create_ctx = nullptr;
    /// expert 张量所在的 ggml context。在 `virtualize_expert_tensor` 里从
    /// `last_create_ctx` 捕获（两者紧邻调用，所以是精确的）。decode 窗口的
    /// view 张量在这里新建。
    struct ggml_context * expert_ctx = nullptr;

    uint64_t n_elements = 0;
    size_t   n_bytes    = 0;

    bool use_mmap = false;
    bool use_direct_io = false;
    bool check_tensors;
    bool no_alloc;
    bool partial_load = false;

    llama_files files;
    llama_ftype ftype;
    llama_fver  fver;

    llama_mmaps mappings;

    std::map<std::string, llama_tensor_weight, weight_name_comparer> weights_map;

    /// 被 take_expert_region() 取走的专家区载体张量。
    std::map<std::string, llama_moe_region> moe_regions;
    /// 专家区在文件内的最小偏移；没有时保持 (size_t) -1。
    size_t moe_region_begin = (size_t) -1;

    std::unordered_map<std::string, llama_model_kv_override> kv_overrides;
    const llama_model_tensor_buft_override * tensor_buft_overrides;

    gguf_context_ptr metadata_ptr;
    struct gguf_context * metadata; // either metadata_ptr.get() or externally set
    llama_model_set_tensor_data_t set_tensor_data;
    void * set_tensor_data_ud;
    std::vector<ggml_context_ptr> contexts;

    std::string arch_name;
    LLM_KV      llm_kv    = LLM_KV(LLM_ARCH_UNKNOWN);

    size_t size_done = 0;
    size_t size_data = 0;
    std::vector<std::pair<size_t, size_t>> mmaps_used;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // track tensors that had to be moved for debugging:
    size_t n_tensors_moved = 0;
    std::string first_tensor_moved_name;
    std::string first_tensor_moved_type_name;
    ggml_backend_buffer_type_t first_moved_from_buft = nullptr;
    ggml_backend_buffer_type_t first_moved_to_buft = nullptr;

    llama_model_loader(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & fname,
        std::vector<std::string> & splits, // optional, only need if the split does not follow naming scheme
        FILE * file,
        bool use_mmap,
        bool use_direct_io,
        bool check_tensors,
        bool no_alloc,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p,
        bool partial_load = false);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(const std::string & key, T & result, bool required = true);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(enum llm_kv kid, T & result, bool required = true);

    template<typename T>
    bool get_arr(const std::string & key, std::vector<T> & result, bool required = true);

    template<typename T, size_t N_MAX>
    bool get_arr(const std::string & key, std::array<T, N_MAX> & result, bool required = true);

    template<typename T>
    bool get_arr(enum llm_kv kid, T & result, bool required = true);

    template<typename T>
    bool get_key(const std::string & key, T & result, bool required = true);

    template<typename T>
    bool get_key(enum llm_kv kid, T & result, bool required = true);

    template<typename T, size_t N_MAX>
    bool get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, bool required = true);

    template<typename T>
    bool get_key_or_arr(enum llm_kv kid, T & result, uint32_t n, bool required = true);

    bool get_key_or_arr(enum llm_kv kid, uint32_t & result, bool required = true);

    std::string get_arch_name() const;

    enum llm_arch get_arch() const;

    const llama_tensor_weight * get_weight(const char * name) const;

    const llama_tensor_weight & require_weight(const char * name) const;

    struct ggml_tensor * get_tensor_meta(const char * name) const;

    struct ggml_tensor * require_tensor_meta(const std::string & name) const;

    const struct ggml_tensor * check_tensor_dims(const std::string & name, const std::vector<int64_t> & ne, bool required) const;

    struct ggml_tensor * create_tensor(
        const llama_hparams & hparams, const buft_list_t * buft_list_cpu, const buft_list_t * buft_list_input, const buft_list_t * buft_list_output,
        const buft_list_t * buft_list_layer, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags);

    /// 在**指定的 buft** 上新建一个空张量（不读 gguf、不进 weights_map）。
    ///
    /// 给「内容不由模型文件提供」的张量用：MoE 专家剥离出 gguf 之后，gguf 里根本没有
    /// 这些张量，只能用 experts manifest 里的形状/类型合成一个占位张量，内容后续由
    /// preadv 填。不进 weights_map 是关键 —— 否则 load_all_data 会去文件里找它们。
    struct ggml_tensor * create_tensor_alloc(
        const llama_hparams & hparams,
        ggml_backend_buffer_type_t buft,
        const std::string & name,
        const std::initializer_list<int64_t> & ne,
        enum ggml_type type);

    /// 取（必要时新建）某个 buffer type 对应的 ggml context。
    /// 顺手把 metadata 容量按模型规模预留好，见实现里的说明。
    struct ggml_context * ctx_for_buft(const llama_hparams & hparams, ggml_backend_buffer_type_t buft);

    void virtualize_expert_tensor(
        struct ggml_tensor * tensor,
        int64_t n_expert,
        int64_t n_slots,
        enum ggml_type experts_type,
        size_t bytes_per_expert);

    // Make dst a view of src so later backend alloc only pays for src.
    // dst may be a smaller / different quant of the same expert geometry.
    void alias_expert_slot_tensor(struct ggml_tensor * dst, struct ggml_tensor * src);

    /// 把 `dst` 变成 `base` 上一个**窗口**的 view：起始 slot 为 `slot_offset`、
    /// 长度 `n_slots` 个 expert。
    ///
    /// 与 `alias_expert_slot_tensor` 的区别：那个要求 dst/src 的 ne[2] 相同（整块别名），
    /// 这个允许 dst 只是 base 的一段子窗口 —— 用于「40 层各自在自己的偏移上开窗口」。
    /// dst 由本函数**新建**（在 expert 张量所在的那个 ggml context 里），所以调用方
    /// 拿到的是一块新 metadata，不需要预先有张量对象。
    struct ggml_tensor * window_expert_slot_tensor(
        const char * name,
        struct ggml_tensor * base,
        int64_t slot_offset,
        int64_t n_slots,
        size_t bytes_per_expert);

    struct ggml_tensor * create_tensor_as_view(struct ggml_context * ctx, struct ggml_tensor * base, const std::string & name, const std::initializer_list<int64_t> & ne, size_t offset, bool required = true);

    /// 取出一个专家区载体张量，并从 weights_map 里摘掉。
    ///
    /// 摘掉是三条性质的来源：load_all_data 不会试图加载它（它没有 ggml 张量）；size_data
    /// 不含它（进度与「加载完成」不被十几 GiB 拖偏）；get_mapping_range 也不会把它算进
    /// 映射区间（Metal 的 mmap buffer 因此只覆盖常驻前缀，而不是整个文件）。
    ///
    /// 名字不在文件里时返回 nullptr —— manifest 布局走的就是这条路。
    const llama_moe_region * take_expert_region(const char * name);

    /// 专家区的最小文件偏移；没有专家区时为 (size_t) -1。
    /// init_mappings() 用它把加载期预读限制在常驻前缀内。
    size_t expert_region_offset() const;

    void done_getting_tensors(bool partial = false) const;

    void init_mappings(bool prefetch = true, llama_mlocks * mlock_mmaps = nullptr,
                       size_t prefetch_limit = (size_t) -1);

    void get_mapping_range(size_t * first, size_t * last, void ** addr, int idx, ggml_context * ctx) const;

    // for backwards compatibility, does not support ggml-backend
    void load_data_for(struct ggml_tensor * cur) const;

    // Returns false if cancelled by progress_callback
    bool load_all_data(
            struct ggml_context * ctx,
            llama_buf_map & bufs,
            llama_mlocks * lmlocks,
            llama_progress_callback progress_callback,
            void * progress_callback_user_data);

    std::string ftype_name() const;

    void print_info() const;
};
