"""Compile both actual T2W consumer blocks with no model/backend dependencies.

The blocks are extracted, not reimplemented. Keep extraction fail-closed so a
refactor cannot silently stop exercising production queue-consumer code.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class TurnBoundaryTests(unittest.TestCase):
    def test_python_and_cpp_consumers_preserve_turns(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "omni.cpp").read_text(encoding="utf-8")
        header = (root / "omni.h").read_text(encoding="utf-8")
        start = header.index("struct T2WOut {")
        packet = header[start:header.index("\n};", start) + 3]
        for name in ("python", "cpp"):
            with self.subTest(consumer=name), tempfile.TemporaryDirectory() as directory:
                start = source.index(f"void t2w_thread_func_{name}(")
                # The local vector is the start of the production collection block.
                start = source.index("        std::vector<llama_token> new_tokens;", start)
                end = source.index("        lock.unlock();", start)
                block = source[start:end]
                self.assertEqual(block.count("while (!queue.empty())"), 1)
                prefix = r'''
#include <chrono>
#include <cstdint>
#include <queue>
#include <vector>
#include <cstdlib>
using llama_token = int32_t;
'''
                harness = r'''
struct Result { std::vector<llama_token> tokens; bool final; bool chunk_end; int round; };
Result consume(std::queue<T2WOut *> & queue) {
    auto dequeue_time = std::chrono::steady_clock::now();
BODY
    return {new_tokens, is_final, is_chunk_end, received_round_idx};
}
void require(bool ok) { if (!ok) std::abort(); }
void push(std::queue<T2WOut *> & q, std::vector<llama_token> tokens,
          bool final, int round, bool chunk_end = false) {
    auto * item = new T2WOut();
    item->audio_tokens = tokens; item->is_final = final;
    item->round_idx = round; item->is_chunk_end = chunk_end; q.push(item);
}
int main() {
    // Two complete turns can already be queued while the vocoder is busy.
    std::queue<T2WOut *> q;
    push(q, {11, 12}, false, 0);
    push(q, {13}, true, 0);
    push(q, {21, 22}, true, 1);
    auto first = consume(q);
    require(first.tokens == std::vector<llama_token>({11, 12, 13}));
    require(first.final && first.round == 0 && q.size() == 1);
    auto second = consume(q);
    require(second.tokens == std::vector<llama_token>({21, 22}));
    require(second.final && second.round == 1 && q.empty());
    // Empty final markers must flush the old buffer, not absorb a new turn.
    push(q, {}, true, 2);
    push(q, {31}, true, 3);
    first = consume(q);
    require(first.tokens.empty() && first.final && first.round == 2 && q.size() == 1);
    second = consume(q);
    require(second.tokens == std::vector<llama_token>({31}) && second.round == 3);
    // Non-final chunks still coalesce; chunk_end is not a semantic turn end.
    push(q, {41}, false, 4, true);
    push(q, {42}, false, 4);
    first = consume(q);
    require(first.tokens == std::vector<llama_token>({41, 42}));
    require(!first.final && first.chunk_end && q.empty());
    first = consume(q);
    require(first.tokens.empty() && !first.final && first.round == -1);
}
'''.replace("BODY", block)
                cpp = Path(directory) / "consumer.cpp"
                binary = Path(directory) / "consumer"
                cpp.write_text(prefix + packet + harness, encoding="utf-8")
                subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17",
                                "-fsanitize=address,undefined", "-g", str(cpp), "-o", str(binary)],
                               check=True, capture_output=True, text=True)
                subprocess.run([str(binary)], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
