#include "omni-token-protocol.h"

#include "omni.h"

#include <cstring>

bool omni_init_token_protocol(struct omni_context * ctx_omni) {
    if (ctx_omni == nullptr || ctx_omni->model == nullptr) {
        return false;
    }

    const struct llama_vocab * vocab = llama_model_get_vocab(ctx_omni->model);
    if (vocab == nullptr) {
        return false;
    }

    auto find_token = [&](const char * token_str) -> llama_token {
        llama_token tokens[4];
        const int n_tokens = llama_tokenize(vocab, token_str, std::strlen(token_str), tokens, 4, false, true);
        if (n_tokens == 1) {
            return tokens[0];
        }

        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (int i = 0; i < n_vocab; ++i) {
            char buf[128];
            const int len = llama_token_to_piece(vocab, i, buf, sizeof(buf), 0, true);
            if (len > 0 && len < (int) sizeof(buf)) {
                buf[len] = '\0';
                if (std::strcmp(buf, token_str) == 0) {
                    return i;
                }
            }
        }

        return -1;
    };

    ctx_omni->special_token_speak = find_token("<|speak|>");
    ctx_omni->special_token_listen = find_token("<|listen|>");
    ctx_omni->special_token_chunk_eos = find_token("<|chunk_eos|>");
    ctx_omni->special_token_chunk_tts_eos = find_token("<|chunk_tts_eos|>");
    ctx_omni->special_token_turn_eos = find_token("<|turn_eos|>");
    ctx_omni->special_token_tts_eos = find_token("<|tts_eos|>");
    ctx_omni->special_token_eos = llama_vocab_eos(vocab);

    const llama_token tts_bos = find_token("<|tts_bos|>");
    if (tts_bos >= 0) {
        ctx_omni->tts_bos_token_id = tts_bos;
    }

    ctx_omni->special_token_unit_end = find_token("</unit>");
    ctx_omni->special_token_tts_pad = find_token("<|tts_pad|>");
    return true;
}

OmniTokenType omni_get_token_type(const struct omni_context * ctx_omni, llama_token token) {
    if (ctx_omni == nullptr) {
        return OmniTokenType::NORMAL;
    }

    if (token == ctx_omni->special_token_speak) {
        return OmniTokenType::SPEAK;
    }
    if (token == ctx_omni->special_token_listen) {
        return OmniTokenType::LISTEN;
    }
    if (token == ctx_omni->special_token_chunk_eos) {
        return OmniTokenType::CHUNK_EOS;
    }
    if (token == ctx_omni->special_token_chunk_tts_eos) {
        return OmniTokenType::CHUNK_TTS_EOS;
    }
    if (token == ctx_omni->special_token_turn_eos) {
        return OmniTokenType::TURN_EOS;
    }
    if (token == ctx_omni->special_token_tts_eos) {
        return OmniTokenType::TTS_EOS;
    }
    if (token == ctx_omni->special_token_eos) {
        return OmniTokenType::EOS;
    }

    return OmniTokenType::NORMAL;
}

bool omni_is_end_token(const struct omni_context * ctx_omni, llama_token token) {
    const OmniTokenType type = omni_get_token_type(ctx_omni, token);

    if (ctx_omni != nullptr && ctx_omni->duplex_mode) {
        return type == OmniTokenType::LISTEN ||
               type == OmniTokenType::CHUNK_EOS ||
               type == OmniTokenType::CHUNK_TTS_EOS;
    }

    return type == OmniTokenType::TTS_EOS ||
           type == OmniTokenType::EOS;
}

bool omni_is_chunk_end_token(const struct omni_context * ctx_omni, llama_token token) {
    const OmniTokenType type = omni_get_token_type(ctx_omni, token);
    return type == OmniTokenType::CHUNK_EOS ||
           type == OmniTokenType::CHUNK_TTS_EOS;
}

const char * omni_get_token_type_name(OmniTokenType type) {
    switch (type) {
        case OmniTokenType::NORMAL:        return "NORMAL";
        case OmniTokenType::SPEAK:         return "SPEAK";
        case OmniTokenType::LISTEN:        return "LISTEN";
        case OmniTokenType::CHUNK_EOS:     return "CHUNK_EOS";
        case OmniTokenType::CHUNK_TTS_EOS: return "CHUNK_TTS_EOS";
        case OmniTokenType::TURN_EOS:      return "TURN_EOS";
        case OmniTokenType::TTS_EOS:       return "TTS_EOS";
        case OmniTokenType::EOS:           return "EOS";
        default:                           return "UNKNOWN";
    }
}
