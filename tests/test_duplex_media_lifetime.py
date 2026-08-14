#!/usr/bin/env python3
import pathlib
import subprocess
import tempfile
import textwrap


ROOT = pathlib.Path(__file__).resolve().parents[1]
OMNI_CPP = ROOT / "tools" / "omni" / "omni.cpp"


def test_duplex_owned_media_survives_source_unlink_and_cleans_up():
    source = OMNI_CPP.read_text(encoding="utf-8")
    begin = source.index("// DUPLEX_OWNED_MEDIA_FILE_BEGIN")
    end = source.index("// DUPLEX_OWNED_MEDIA_FILE_END")
    implementation = source[begin:end]

    harness = textwrap.dedent(
        f"""
        #include <cerrno>
        #include <fcntl.h>
        #include <fstream>
        #include <memory>
        #include <string>
        #include <unistd.h>

        {implementation}

        struct Request {{
            DuplexOwnedMediaFile audio;
            DuplexOwnedMediaFile image;
        }};

        static std::string make_source(const std::string & contents) {{
            char name[] = "/tmp/omni-duplex-media-test-XXXXXX";
            const int fd = mkstemp(name);
            if (fd < 0 || write(fd, contents.data(), contents.size()) !=
                    static_cast<ssize_t>(contents.size()) || close(fd) != 0) {{
                return {{}};
            }}
            return name;
        }}

        static bool readable_as(const std::string & path,
                                const std::string & expected) {{
            std::ifstream input(path, std::ios::binary);
            return std::string(std::istreambuf_iterator<char>(input), {{}}) == expected;
        }}

        int main() {{
            const std::string audio_contents = "audio-payload";
            const std::string image_contents = "image-payload";
            const std::string audio_source = make_source(audio_contents);
            const std::string image_source = make_source(image_contents);
            if (audio_source.empty() || image_source.empty()) return 1;

            std::string owned_audio;
            std::string owned_image;
            {{
                auto request = std::make_unique<Request>();
                if (!request->audio.acquire(audio_source) ||
                    !request->image.acquire(image_source)) return 2;
                owned_audio = request->audio.path;
                owned_image = request->image.path;

                if (unlink(audio_source.c_str()) != 0 ||
                    unlink(image_source.c_str()) != 0) return 3;
                if (!readable_as(owned_audio, audio_contents) ||
                    !readable_as(owned_image, image_contents)) return 4;
            }}

            if (access(owned_audio.c_str(), F_OK) == 0 ||
                access(owned_image.c_str(), F_OK) == 0) return 5;
            return 0;
        }}
        """
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        test_cpp = tmp_path / "test.cpp"
        executable = tmp_path / "test"
        test_cpp.write_text(harness, encoding="utf-8")
        subprocess.run(
            ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
             str(test_cpp), "-o", str(executable)],
            check=True,
        )
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    test_duplex_owned_media_survives_source_unlink_and_cleans_up()
