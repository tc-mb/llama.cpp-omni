// Smoke tool for Qwen3-ForcedAligner: feed it audio and a list of units, get per-unit timestamps.
//
//   llama-aligner-cli --aligner-lm LM.gguf --aligner-audio AUDIO.gguf \
//                     -f speech.wav --units "a,b,c" [--ngl 99] [--ctx 4096] [--json]
//
// --units is comma separated. Segmentation is the caller's business; nothing is tokenized here.

#include "qwen3-aligner.h"

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

void print_usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s --aligner-lm LM.gguf --aligner-audio AUDIO.gguf -f AUDIO.wav --units \"a,b,c\"\n"
            "  --units-file F  read units from a UTF-8 file, one per line (Windows consoles\n"
            "                  rewrite non-ASCII arguments through the active code page)\n"
            "  --ngl N         layers to offload to the GPU (default: -1, all)\n"
            "  --ctx N         context size (default: 4096)\n"
            "  --threads N     CPU threads (default: 4)\n"
            "  --json          print JSON\n",
            argv0);
}

std::vector<std::string> split_units(const std::string & s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        const size_t comma = s.find(',', start);
        const size_t end   = comma == std::string::npos ? s.size() : comma;
        if (end > start) out.emplace_back(s.substr(start, end - start));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

/** One unit per line, UTF-8; blank lines skipped, trailing \r stripped */
bool read_units_file(const std::string & path, std::vector<std::string> & units) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    std::string line;
    int c;
    auto flush = [&]() {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (!line.empty()) units.push_back(line);
        line.clear();
    };
    while ((c = fgetc(f)) != EOF) {
        if (c == '\n') flush();
        else line.push_back((char) c);
    }
    flush();
    fclose(f);
    return true;
}

bool read_file(const std::string & path, std::vector<unsigned char> & buf) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); return false; }
    buf.resize((size_t) n);
    const bool ok = fread(buf.data(), 1, buf.size(), f) == buf.size();
    fclose(f);
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    qwen3_aligner_params params;
    std::string audio_file;
    std::string units_arg;
    std::string units_file;
    bool as_json = false;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s needs a value\n", what);
                exit(1);
            }
            return argv[++i];
        };
        if      (a == "--aligner-lm")    params.lm_path      = next("--aligner-lm");
        else if (a == "--aligner-audio") params.audio_path   = next("--aligner-audio");
        else if (a == "-f" || a == "--file") audio_file      = next("-f");
        else if (a == "--units")         units_arg           = next("--units");
        else if (a == "--units-file")    units_file          = next("--units-file");
        else if (a == "--ngl")           params.n_gpu_layers = std::stoi(next("--ngl"));
        else if (a == "--ctx")           params.n_ctx        = std::stoi(next("--ctx"));
        else if (a == "--threads")       params.n_threads    = std::stoi(next("--threads"));
        else if (a == "--json")          as_json             = true;
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else {
            fprintf(stderr, "unknown argument: %s\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (params.lm_path.empty() || params.audio_path.empty() || audio_file.empty() ||
        (units_arg.empty() && units_file.empty())) {
        print_usage(argv[0]);
        return 1;
    }

    std::vector<std::string> units;
    if (!units_file.empty()) {
        if (!read_units_file(units_file, units)) {
            fprintf(stderr, "cannot read the units file: %s\n", units_file.c_str());
            return 1;
        }
    } else {
        units = split_units(units_arg);
    }
    if (units.empty()) {
        fprintf(stderr, "the unit list is empty\n");
        return 1;
    }

    llama_backend_init();

    std::string err;
    qwen3_aligner * aligner = qwen3_aligner_init(params, err);
    if (!aligner) {
        fprintf(stderr, "failed to load the aligner: %s\n", err.c_str());
        llama_backend_free();
        return 1;
    }

    std::vector<unsigned char> raw;
    if (!read_file(audio_file, raw)) {
        fprintf(stderr, "cannot read the audio file: %s\n", audio_file.c_str());
        qwen3_aligner_free(aligner);
        llama_backend_free();
        return 1;
    }

    std::vector<float> pcm;
    if (!qwen3_aligner_decode_audio(aligner, raw.data(), raw.size(), pcm, err)) {
        fprintf(stderr, "failed to decode the audio: %s\n", err.c_str());
        qwen3_aligner_free(aligner);
        llama_backend_free();
        return 1;
    }

    std::vector<qwen3_aligner_span> spans;
    if (!qwen3_aligner_align(aligner, pcm.data(), pcm.size(), units, spans, err)) {
        fprintf(stderr, "alignment failed: %s\n", err.c_str());
        qwen3_aligner_free(aligner);
        llama_backend_free();
        return 1;
    }

    const double duration = (double) pcm.size() / qwen3_aligner_sample_rate(aligner);
    if (as_json) {
        printf("{\"duration\":%.3f,\"units\":[", duration);
        for (size_t i = 0; i < spans.size(); i++) {
            printf("%s{\"text\":\"%s\",\"start\":%.3f,\"end\":%.3f}",
                   i ? "," : "", spans[i].text.c_str(), spans[i].start, spans[i].end);
        }
        printf("]}\n");
    } else {
        printf("%.3fs, %zu units\n", duration, spans.size());
        for (const auto & s : spans) {
            printf("  %7.3f  %7.3f  %s\n", s.start, s.end, s.text.c_str());
        }
    }

    qwen3_aligner_free(aligner);
    llama_backend_free();
    return 0;
}
