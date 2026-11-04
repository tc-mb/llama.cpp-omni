"""Compile actual WS dequeue blocks without requiring model weights or a server."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class WebSocketQueueTests(unittest.TestCase):
    def test_both_consumers_drain_completed_producers(self):
        source = (Path(__file__).resolve().parents[1] / "ws_handler.cpp").read_text(encoding="utf-8")
        needle = "                    octx->text_cv.wait_for(lk, std::chrono::milliseconds(200), [&]{"
        blocks = []
        start = 0
        while True:
            at = source.find(needle, start)
            if at < 0:
                break
            begin = source.rfind("                {", 0, at)
            end = source.index("\n\n                if (!frag.empty())", at)
            blocks.append(source[begin:end])
            start = end
        self.assertEqual(len(blocks), 2, "both production consumers must be exercised")
        prefix = r'''
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
struct Context {
    std::mutex text_mtx;
    std::condition_variable text_cv;
    std::deque<std::string> text_queue;
    bool text_done_flag = true;
};
std::vector<std::string> consume(Context * octx) {
    std::vector<std::string> seen;
    while (true) {
        std::string frag;
'''
        suffix = r'''
        if (!frag.empty()) seen.push_back(frag);
    }
    return seen;
}
void require(bool ok) { if (!ok) std::abort(); }
int main() {
    const std::vector<std::vector<std::string>> cases = {
        {}, {"hello"}, {"__IS_LISTEN__"}, {"a", "b", "c"},
        {"text", "__END_OF_TURN__"}, {"\xe4\xbd", "\xa0"}
    };
    for (const auto & expected : cases) {
        Context c;
        c.text_queue.assign(expected.begin(), expected.end());
        require(consume(&c) == expected);
        require(c.text_queue.empty());
    }
    Context c;
    c.text_done_flag = false;
    std::thread producer([&] {
        std::lock_guard<std::mutex> lock(c.text_mtx);
        c.text_queue.push_back("late");
        c.text_done_flag = true;
        c.text_cv.notify_one();
    });
    auto seen = consume(&c);
    producer.join();
    require(seen == std::vector<std::string>{"late"});
}
'''
        for index, block in enumerate(blocks):
            with self.subTest(consumer=index), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path / "queue.cpp").write_text(prefix + block + suffix, encoding="utf-8")
                subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-pthread", "-g",
                                "-fsanitize=address,undefined", str(path / "queue.cpp"),
                                "-o", str(path / "queue")], check=True)
                subprocess.run([str(path / "queue")], check=True, timeout=10)


if __name__ == "__main__":
    unittest.main()
