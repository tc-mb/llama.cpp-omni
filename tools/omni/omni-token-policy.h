#pragma once

#include <cstdint>

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
struct OmniTokenPolicy {
    int32_t speak        = -1;
    int32_t listen       = -1;
    int32_t chunk_eos    = -1;
    int32_t chunk_tts_eos = -1;
    int32_t turn_eos     = -1;
    int32_t tts_eos      = -1;
    int32_t im_end       = -1;
    int32_t slash_s      = -1;
    int32_t eos          = -1;

    OmniTokenType classify(int32_t token) const {
        if (token == speak) {
            return OmniTokenType::SPEAK;
        }
        if (token == listen) {
            return OmniTokenType::LISTEN;
        }
        if (token == chunk_eos) {
            return OmniTokenType::CHUNK_EOS;
        }
        if (token == chunk_tts_eos) {
            return OmniTokenType::CHUNK_TTS_EOS;
        }
        if (token == turn_eos) {
            return OmniTokenType::TURN_EOS;
        }
        if (token == tts_eos) {
            return OmniTokenType::TTS_EOS;
        }
        if (token == im_end || token == slash_s || token == eos) {
            return OmniTokenType::EOS;
        }
        return OmniTokenType::NORMAL;
    }

    bool is_simplex_terminator(int32_t token) const {
        return token >= 0 &&
               (token == tts_eos || token == im_end || token == slash_s || token == eos);
    }

    bool is_tts_condition_token(int32_t token) const {
        if (classify(token) != OmniTokenType::NORMAL) {
            return false;
        }

        switch (token) {
            case 151667: // <think>
            case 151668: // </think>
            case 151704: // <|tts_eos|>
            case 151706: // <|speak|>
            case 151705: // <|listen|>
            case 151718: // <|chunk_eos|>
            case 151721: // <|chunk_tts_eos|>
            case 151717: // <|turn_eos|>
            case 271:    // reserved newline token filtered by the reference pipeline
                return false;
            default:
                break;
        }

        return token >= 0 && token < 150000;
    }
};
