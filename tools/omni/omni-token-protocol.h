#pragma once

#include "llama.h"

struct omni_context;

enum class OmniTokenType {
    NORMAL,
    SPEAK,
    LISTEN,
    CHUNK_EOS,
    CHUNK_TTS_EOS,
    TURN_EOS,
    TTS_EOS,
    EOS,
};

bool omni_init_token_protocol(struct omni_context * ctx_omni);
OmniTokenType omni_get_token_type(const struct omni_context * ctx_omni, llama_token token);
bool omni_is_end_token(const struct omni_context * ctx_omni, llama_token token);
bool omni_is_chunk_end_token(const struct omni_context * ctx_omni, llama_token token);
const char * omni_get_token_type_name(OmniTokenType type);
