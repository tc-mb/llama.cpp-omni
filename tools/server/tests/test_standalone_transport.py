#!/usr/bin/env python3
"""Model-free transport regressions for the standalone servers.

Run directly (no pytest dependencies):
    LLAMA_OMNI_SERVER_BIN=build/bin/llama-omni-server \\
        python3 tools/server/tests/test_standalone_transport.py -v

Optional LLAMA_TTS_SERVER_BIN checks TLS configuration rejection before loading
models. LLAMA_OMNI_SERVER_NO_TLS_BIN and LLAMA_TTS_SERVER_NO_TLS_BIN exercise
separate builds configured with LLAMA_OPENSSL=OFF (without bundled TLS backends).
The primary Omni binary must support TLS. OpenSSL's command-line tool generates
short-lived test credentials; no model files or external network are needed.
"""

import base64
import contextlib
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import socket
import ssl
import struct
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[3]
TIMEOUT = 10


def server_env():
    # Do not inherit model downloads, certificates, or other user CLI defaults.
    return {key: value for key, value in os.environ.items()
            if not key.startswith("LLAMA_ARG_")}


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def read_exact(sock, size):
    result = b""
    while len(result) < size:
        chunk = sock.recv(size - len(result))
        if not chunk:
            raise EOFError("WebSocket closed before the expected frame")
        result += chunk
    return result


class StandaloneTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if __name__ != "__main__" and "LLAMA_OMNI_SERVER_BIN" not in os.environ:
            raise unittest.SkipTest("Set LLAMA_OMNI_SERVER_BIN to opt into standalone transport tests")
        cls.omni = str(Path(os.environ.get(
            "LLAMA_OMNI_SERVER_BIN", ROOT / "build/bin/llama-omni-server")).resolve())
        if not Path(cls.omni).is_file():
            raise RuntimeError("Set LLAMA_OMNI_SERVER_BIN to a TLS-enabled Omni binary")
        if shutil.which("openssl") is None:
            raise RuntimeError("openssl is required to generate test certificates")
        cls.temp = tempfile.TemporaryDirectory(prefix="llama-transport-")
        cls.addClassCleanup(cls.temp.cleanup)
        folder = Path(cls.temp.name)
        cls.cert, cls.key = folder / "cert.pem", folder / "key.pem"
        cls.other_key = folder / "other-key.pem"
        cls.malformed = folder / "malformed.pem"
        cls.missing = folder / "missing.pem"
        cls.malformed.write_text("not a PEM certificate or key\n")
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-days", "1", "-subj", "/CN=localhost",
            "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
            "-keyout", str(cls.key), "-out", str(cls.cert),
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=TIMEOUT)
        subprocess.run([
            "openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt",
            "rsa_keygen_bits:2048", "-out", str(cls.other_key),
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=TIMEOUT)
        cls.tls_context = ssl.create_default_context(cafile=str(cls.cert))

    def tls_args(self, cert=None, key=None):
        return ["--ssl-cert-file", str(cert or self.cert),
                "--ssl-key-file", str(key or self.key)]

    def health(self, port, tls=False, path="/health", host="127.0.0.1"):
        if tls:
            connection = http.client.HTTPSConnection(
                host, port, timeout=1, context=self.tls_context)
        else:
            connection = http.client.HTTPConnection(host, port, timeout=1)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            body = response.read()
            self.assertEqual(response.status, 200, body)
            self.assertEqual(json.loads(body), {"engine": "comni", "status": "ok"})
            self.assertEqual(response.getheader("X-Engine"), "comni")
        finally:
            connection.close()

    @contextlib.contextmanager
    def running(self, tls=False, binary=None, extra_env=None, host="127.0.0.1"):
        port = free_port()
        args = [binary or self.omni, "--port", str(port)]
        if host is not None:
            args += ["--host", host]
        if tls and extra_env is None:
            args += self.tls_args()
        env = server_env()
        env.update(extra_env or {})
        with tempfile.TemporaryFile(mode="w+b") as log:
            process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                deadline = time.monotonic() + TIMEOUT
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        log.seek(0)
                        self.fail("Server exited during startup: " + log.read().decode(errors="replace"))
                    try:
                        self.health(port, tls)
                        break
                    except (OSError, http.client.HTTPException):
                        time.sleep(0.05)
                else:
                    log.seek(0)
                    self.fail("Server never became healthy: " + log.read().decode(errors="replace"))
                yield port
                self.assertIsNone(process.poll(), "Server exited unexpectedly")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)

    def rejected(self, args, binary=None, port=None, expected=r"(?i)TLS|SSL"):
        selected_port = free_port() if port is None else port
        process = subprocess.Popen(
            [binary or self.omni, "--host", "127.0.0.1", "--port", str(selected_port)] + args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=server_env())
        try:
            output, _ = process.communicate(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate(timeout=3)
            self.fail("Invalid startup configuration did not exit: " + output.decode(errors="replace"))
        output = output.decode(errors="replace")
        self.assertGreater(process.returncode, 0, output)
        self.assertRegex(output, expected)
        self.assertNotIn("Loading VoxCPM2 runtime", output)
        self.assertNotIn("No VoxCPM2 models specified", output)
        if port is None:
            with self.assertRaises(OSError):
                socket.create_connection(("127.0.0.1", selected_port), timeout=0.5).close()

    def invalid_tls_cases(self, binary):
        cases = {
            "certificate only": ["--ssl-cert-file", str(self.cert)],
            "key only": ["--ssl-key-file", str(self.key)],
            "missing certificate": self.tls_args(cert=self.missing),
            "missing key": self.tls_args(key=self.missing),
            "malformed certificate": self.tls_args(cert=self.malformed),
            "malformed key": self.tls_args(key=self.malformed),
            "mismatched key": self.tls_args(key=self.other_key),
        }
        for name, args in cases.items():
            with self.subTest(case=name):
                self.rejected(args, binary=binary)

    def test_http_health(self):
        with self.running() as port:
            self.health(port, path="/v1/health")

    def test_https_health_and_plain_http_rejection(self):
        with self.running(tls=True) as port:
            self.health(port, tls=True, path="/v1/health")
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            try:
                try:
                    connection.request("GET", "/health")
                    response = connection.getresponse()
                except (OSError, http.client.HTTPException):
                    pass
                else:
                    self.assertNotEqual(response.status, 200)
            finally:
                connection.close()

    def test_https_environment_configuration(self):
        with self.running(tls=True, extra_env={
            "LLAMA_ARG_SSL_CERT_FILE": str(self.cert),
            "LLAMA_ARG_SSL_KEY_FILE": str(self.key),
        }) as port:
            self.health(port, tls=True)

    def test_invalid_tls_configuration(self):
        self.invalid_tls_cases(self.omni)

    def test_tts_invalid_tls_before_model_loading(self):
        binary = os.environ.get("LLAMA_TTS_SERVER_BIN")
        if not binary:
            self.skipTest("LLAMA_TTS_SERVER_BIN not supplied")
        self.invalid_tls_cases(binary)

    def test_occupied_port(self):
        with socket.socket() as blocker:
            blocker.bind(("127.0.0.1", 0))
            blocker.listen()
            for args in ([], self.tls_args()):
                with self.subTest(tls=bool(args)):
                    self.rejected(args, port=blocker.getsockname()[1], expected=r"(?i)bind|listen")

    def require_second_loopback(self):
        # Verify the test platform supports a second loopback address first.
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.2", 0))
            except OSError:
                self.skipTest("A second loopback address is unavailable")

    def test_host_defaults_to_all_interfaces(self):
        self.require_second_loopback()
        with self.running(host=None) as port:
            self.health(port, host="127.0.0.2")

    def test_host_is_honored(self):
        self.require_second_loopback()
        with self.running() as port:
            with self.assertRaises(OSError):
                socket.create_connection(("127.0.0.2", port), timeout=0.5).close()

    def test_host_environment_overrides_default(self):
        self.require_second_loopback()
        with self.running(host=None, extra_env={"LLAMA_ARG_HOST": "127.0.0.1"}) as port:
            with self.assertRaises(OSError):
                socket.create_connection(("127.0.0.2", port), timeout=0.5).close()

    def test_host_cli_overrides_environment(self):
        self.require_second_loopback()
        with self.running(extra_env={"LLAMA_ARG_HOST": "0.0.0.0"}) as port:
            with self.assertRaises(OSError):
                socket.create_connection(("127.0.0.2", port), timeout=0.5).close()
        with self.running(host="0.0.0.0", extra_env={"LLAMA_ARG_HOST": "127.0.0.1"}) as port:
            self.health(port, host="127.0.0.2")

    def websocket(self, port, tls):
        with contextlib.ExitStack() as stack:
            sock = stack.enter_context(socket.create_connection(("127.0.0.1", port), timeout=2))
            if tls:
                sock = stack.enter_context(self.tls_context.wrap_socket(sock, server_hostname="localhost"))
            key = base64.b64encode(os.urandom(16)).decode()
            request = (
                f"GET /backend HTTP/1.1\r\nHost: localhost:{port}\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
            sock.sendall(request.encode())
            header = b""
            while not header.endswith(b"\r\n\r\n"):
                self.assertLess(len(header), 16384)
                header += read_exact(sock, 1)
            lines = header.decode().split("\r\n")
            self.assertEqual(lines[0].split()[1], "101", header)
            # Header names are case-insensitive; the base64 digest is not.
            headers = dict((name.lower(), value.strip())
                           for name, value in (line.split(":", 1) for line in lines[1:] if ":" in line))
            self.assertEqual(headers["upgrade"].lower(), "websocket")
            self.assertIn("upgrade", headers["connection"].lower())
            expected = base64.b64encode(hashlib.sha1(
                (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
            self.assertEqual(headers["sec-websocket-accept"], expected)
            # Invalid JSON is rejected before session allocation or model loading.
            payload, mask = b"not-json", os.urandom(4)
            masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
            sock.sendall(bytes((0x81, 0x80 | len(payload))) + mask + masked)
            first, second = read_exact(sock, 2)
            self.assertEqual(first & 0x0F, 8, "Expected WebSocket close frame")
            self.assertFalse(second & 0x80, "Server frames must not be masked")
            size = second & 0x7F
            self.assertLessEqual(size, 125)
            close_payload = read_exact(sock, size)
            if size >= 2:
                self.assertEqual(struct.unpack("!H", close_payload[:2])[0], 1000)
            mask = os.urandom(4)
            masked = bytes(value ^ mask[index % 4] for index, value in enumerate(close_payload))
            sock.sendall(bytes((0x88, 0x80 | size)) + mask + masked)

    def test_websocket_http(self):
        with self.running() as port:
            self.websocket(port, tls=False)
            self.health(port)

    def test_websocket_https(self):
        with self.running(tls=True) as port:
            self.websocket(port, tls=True)
            self.health(port, tls=True)

    def test_no_tls_build_http(self):
        binary = os.environ.get("LLAMA_OMNI_SERVER_NO_TLS_BIN")
        if not binary:
            self.skipTest("LLAMA_OMNI_SERVER_NO_TLS_BIN not supplied")
        with self.running(binary=binary) as port:
            self.health(port)

    def test_no_tls_build_rejects_tls_configuration(self):
        binaries = [os.environ[name] for name in (
            "LLAMA_OMNI_SERVER_NO_TLS_BIN", "LLAMA_TTS_SERVER_NO_TLS_BIN") if name in os.environ]
        if not binaries:
            self.skipTest("No TLS-disabled binaries supplied")
        for binary in binaries:
            for args in (self.tls_args(), ["--ssl-cert-file", str(self.cert)],
                         ["--ssl-key-file", str(self.key)]):
                with self.subTest(binary=binary, args=args):
                    self.rejected(args, binary=binary)


if __name__ == "__main__":
    unittest.main()
