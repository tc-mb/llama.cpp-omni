#pragma once

#include <string>

inline bool utf8_cont(unsigned char c) {
    return (c & 0xC0) == 0x80;
}

// Stream-safe UTF-8: emit only complete code points, hold an incomplete trailing
// byte sequence in `pending` until the next fragment, and replace invalid bytes
// with U+FFFD. Prevents broken multi-byte chars when text is chunked over WS
// or SSE: a single token piece can end in the middle of a character.
inline std::string sanitize_utf8_stream(std::string & pending,
                                        const std::string & fragment,
                                        bool flush = false) {
    static const std::string replacement = "\xEF\xBF\xBD";
    std::string input = pending + fragment;
    pending.clear();

    std::string out;
    size_t i = 0;
    while (i < input.size()) {
        const unsigned char c = static_cast<unsigned char>(input[i]);
        if (c < 0x80) {
            out.push_back(static_cast<char>(c));
            i++;
            continue;
        }

        int need = 0;
        if (c >= 0xC2 && c <= 0xDF) {
            need = 1;
        } else if (c >= 0xE0 && c <= 0xEF) {
            need = 2;
        } else if (c >= 0xF0 && c <= 0xF4) {
            need = 3;
        } else {
            out += replacement;
            i++;
            continue;
        }

        if (i + need >= input.size()) {
            pending = input.substr(i);
            break;
        }

        bool ok = true;
        for (int j = 1; j <= need; ++j) {
            ok = ok && utf8_cont(static_cast<unsigned char>(input[i + j]));
        }
        if (ok && c == 0xE0) {
            ok = static_cast<unsigned char>(input[i + 1]) >= 0xA0;
        } else if (ok && c == 0xED) {
            ok = static_cast<unsigned char>(input[i + 1]) < 0xA0;
        } else if (ok && c == 0xF0) {
            ok = static_cast<unsigned char>(input[i + 1]) >= 0x90;
        } else if (ok && c == 0xF4) {
            ok = static_cast<unsigned char>(input[i + 1]) < 0x90;
        }

        if (!ok) {
            out += replacement;
            i++;
            continue;
        }

        out.append(input, i, need + 1);
        i += need + 1;
    }

    if (flush && !pending.empty()) {
        out += replacement;
        pending.clear();
    }
    return out;
}
