if (NOT DEFINED OMNI_SERVER)
    message(FATAL_ERROR "OMNI_SERVER must point to llama-omni-server")
endif()

find_program(CURL_EXECUTABLE curl)
if (NOT CURL_EXECUTABLE)
    message(FATAL_ERROR "curl is required for the HTTP startup regression test")
endif()

set(PORT 19876)
set(LOG_FILE "${CMAKE_CURRENT_BINARY_DIR}/test-omni-server-http-startup.log")
string(CONCAT SERVER_SCRIPT
    "server='${OMNI_SERVER}'; "
    "log='${LOG_FILE}'; "
    "port='${PORT}'; "
    "\"$server\" --port \"$port\" >\"$log\" 2>&1 & "
    "pid=$!; "
    "status=1; "
    "i=0; "
    "while [ \"$i\" -lt 30 ]; do "
    "  if ! kill -0 \"$pid\" 2>/dev/null; then status=2; break; fi; "
    "  if ${CURL_EXECUTABLE} --fail --silent --show-error "
    "      \"http://127.0.0.1:$port/health\" >/dev/null 2>&1; then "
    "    status=0; break; "
    "  fi; "
    "  i=$((i + 1)); "
    "  sleep 0.2; "
    "done; "
    "kill \"$pid\" 2>/dev/null || true; "
    "wait \"$pid\" 2>/dev/null || true; "
    "exit \"$status\""
)

execute_process(
    COMMAND /bin/sh -c "${SERVER_SCRIPT}"
    RESULT_VARIABLE result
)

if (NOT result EQUAL 0)
    file(READ "${LOG_FILE}" server_log)
    message(FATAL_ERROR
        "llama-omni-server did not serve plain HTTP without TLS credentials "
        "(result=${result})\n${server_log}")
endif()
