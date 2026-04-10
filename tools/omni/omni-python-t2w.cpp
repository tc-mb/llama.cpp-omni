#include "omni-python-t2w.h"

#include "omni-impl.h"
#include "omni.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <process.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <windows.h>
#else
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

void print_with_timestamp(const char * format, ...);

static bool omni_python_t2w_has_status(const std::string & response, const char * status) {
    const std::string plain = std::string("\"status\":\"") + status + "\"";
    const std::string spaced = std::string("\"status\": \"") + status + "\"";
    return response.find(plain) != std::string::npos ||
           response.find(spaced) != std::string::npos;
}

bool omni_start_python_t2w_service(struct omni_context * ctx_omni) {
    if (ctx_omni->python_t2w_initialized) {
        return true;
    }

    const std::string script_path = ctx_omni->python_t2w_script_dir + "/token2wav_service.py";
    FILE * check = fopen(script_path.c_str(), "r");
    if (check == nullptr) {
        LOG_ERR("Python T2W: 脚本不存在: %s\n", script_path.c_str());
        return false;
    }
    fclose(check);

    print_with_timestamp("Python T2W: 启动服务进程 %s\n", script_path.c_str());

#ifdef _WIN32
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;

    HANDLE hStdinRead;
    HANDLE hStdinWrite;
    HANDLE hStdoutRead;
    HANDLE hStdoutWrite;

    if (!CreatePipe(&hStdinRead, &hStdinWrite, &sa, 0)) {
        LOG_ERR("Python T2W: CreatePipe (stdin) 失败\n");
        return false;
    }
    SetHandleInformation(hStdinWrite, HANDLE_FLAG_INHERIT, 0);

    if (!CreatePipe(&hStdoutRead, &hStdoutWrite, &sa, 0)) {
        LOG_ERR("Python T2W: CreatePipe (stdout) 失败\n");
        CloseHandle(hStdinRead);
        CloseHandle(hStdinWrite);
        return false;
    }
    SetHandleInformation(hStdoutRead, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.hStdInput = hStdinRead;
    si.hStdOutput = hStdoutWrite;
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    si.dwFlags |= STARTF_USESTDHANDLES;
    ZeroMemory(&pi, sizeof(pi));

    if (!ctx_omni->python_t2w_gpu_id.empty()) {
        _putenv_s("CUDA_VISIBLE_DEVICES", ctx_omni->python_t2w_gpu_id.c_str());
    }

    const std::string win_cmd = "python \"" + script_path + "\"";
    char cmd_buf[2048];
    strncpy(cmd_buf, win_cmd.c_str(), sizeof(cmd_buf) - 1);
    cmd_buf[sizeof(cmd_buf) - 1] = '\0';

    if (!CreateProcessA(NULL, cmd_buf, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        LOG_ERR("Python T2W: CreateProcess 失败, error=%lu\n", GetLastError());
        CloseHandle(hStdinRead);
        CloseHandle(hStdinWrite);
        CloseHandle(hStdoutRead);
        CloseHandle(hStdoutWrite);
        return false;
    }

    CloseHandle(hStdinRead);
    CloseHandle(hStdoutWrite);
    CloseHandle(pi.hThread);

    ctx_omni->python_t2w_pid = (int) (intptr_t) pi.hProcess;

    const int stdin_fd = _open_osfhandle((intptr_t) hStdinWrite, 0);
    const int stdout_fd = _open_osfhandle((intptr_t) hStdoutRead, 0);
    if (stdin_fd < 0 || stdout_fd < 0) {
        LOG_ERR("Python T2W: _open_osfhandle 失败\n");
        TerminateProcess(pi.hProcess, 1);
        CloseHandle(pi.hProcess);
        return false;
    }

    ctx_omni->python_t2w_stdin = _fdopen(stdin_fd, "w");
    ctx_omni->python_t2w_stdout = _fdopen(stdout_fd, "r");
#else
    int stdin_pipe[2];
    int stdout_pipe[2];

    if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0) {
        LOG_ERR("Python T2W: 创建管道失败\n");
        return false;
    }

    const pid_t pid = fork();
    if (pid < 0) {
        LOG_ERR("Python T2W: fork 失败\n");
        return false;
    }

    if (pid == 0) {
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);

        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);

        close(stdin_pipe[0]);
        close(stdout_pipe[1]);

        if (!ctx_omni->python_t2w_gpu_id.empty()) {
            setenv("CUDA_VISIBLE_DEVICES", ctx_omni->python_t2w_gpu_id.c_str(), 1);
        }

        execlp("/Users/tianchi/software/miniconda3/bin/python", "python", script_path.c_str(), (char *) NULL);
        _exit(1);
    }

    close(stdin_pipe[0]);
    close(stdout_pipe[1]);

    ctx_omni->python_t2w_pid = pid;
    ctx_omni->python_t2w_stdin = fdopen(stdin_pipe[1], "w");
    ctx_omni->python_t2w_stdout = fdopen(stdout_pipe[0], "r");
#endif

    if (ctx_omni->python_t2w_stdin == nullptr || ctx_omni->python_t2w_stdout == nullptr) {
        LOG_ERR("Python T2W: fdopen 失败\n");
        omni_stop_python_t2w_service(ctx_omni);
        return false;
    }

    setvbuf(ctx_omni->python_t2w_stdin, NULL, _IOLBF, 0);
    setvbuf(ctx_omni->python_t2w_stdout, NULL, _IOLBF, 0);

    char buffer[4096];
    if (fgets(buffer, sizeof(buffer), ctx_omni->python_t2w_stdout) != nullptr) {
        print_with_timestamp("Python T2W: 服务响应: %s", buffer);
        if (omni_python_t2w_has_status(buffer, "ready")) {
            ctx_omni->python_t2w_initialized = true;
            print_with_timestamp("Python T2W: 服务就绪\n");
            return true;
        }
    }

    LOG_ERR("Python T2W: 服务未能正常启动\n");
    omni_stop_python_t2w_service(ctx_omni);
    return false;
}

void omni_stop_python_t2w_service(struct omni_context * ctx_omni) {
    if (ctx_omni->python_t2w_stdin != nullptr) {
        fprintf(ctx_omni->python_t2w_stdin, "{\"cmd\":\"quit\"}\n");
        fflush(ctx_omni->python_t2w_stdin);
        fclose(ctx_omni->python_t2w_stdin);
        ctx_omni->python_t2w_stdin = nullptr;
    }

    if (ctx_omni->python_t2w_stdout != nullptr) {
        fclose(ctx_omni->python_t2w_stdout);
        ctx_omni->python_t2w_stdout = nullptr;
    }

    if (ctx_omni->python_t2w_pid > 0) {
#ifdef _WIN32
        HANDLE hProcess = (HANDLE) (intptr_t) ctx_omni->python_t2w_pid;
        if (WaitForSingleObject(hProcess, 500) == WAIT_TIMEOUT) {
            TerminateProcess(hProcess, 1);
            WaitForSingleObject(hProcess, 1000);
        }
        CloseHandle(hProcess);
#else
        int status;
        waitpid(ctx_omni->python_t2w_pid, &status, WNOHANG);
        if (kill(ctx_omni->python_t2w_pid, 0) == 0) {
            kill(ctx_omni->python_t2w_pid, SIGTERM);
            usleep(100000);
            if (kill(ctx_omni->python_t2w_pid, 0) == 0) {
                kill(ctx_omni->python_t2w_pid, SIGKILL);
            }
        }
        waitpid(ctx_omni->python_t2w_pid, &status, 0);
#endif
        ctx_omni->python_t2w_pid = -1;
    }

    ctx_omni->python_t2w_initialized = false;
    print_with_timestamp("Python T2W: 服务已停止\n");
}

bool omni_send_python_t2w_command(struct omni_context * ctx_omni, const std::string & cmd_json, std::string & response) {
    if (!ctx_omni->python_t2w_initialized ||
        ctx_omni->python_t2w_stdin == nullptr ||
        ctx_omni->python_t2w_stdout == nullptr) {
        return false;
    }

    fprintf(ctx_omni->python_t2w_stdin, "%s\n", cmd_json.c_str());
    fflush(ctx_omni->python_t2w_stdin);

    char buffer[8192];
    if (fgets(buffer, sizeof(buffer), ctx_omni->python_t2w_stdout) == nullptr) {
        return false;
    }

    response = buffer;
    while (!response.empty() && (response.back() == '\n' || response.back() == '\r')) {
        response.pop_back();
    }
    return true;
}

bool omni_init_python_t2w_model(struct omni_context * ctx_omni, const std::string & device) {
    if (!ctx_omni->python_t2w_initialized && !omni_start_python_t2w_service(ctx_omni)) {
        return false;
    }

    std::string python_device = "cuda:0";
    if (device.find("cpu") != std::string::npos) {
        python_device = "cpu";
    }

    char cmd[1024];
    snprintf(
        cmd,
        sizeof(cmd),
        "{\"cmd\":\"init\",\"model_dir\":\"%s\",\"device\":\"%s\",\"float16\":true,\"n_timesteps\":5}",
        ctx_omni->python_t2w_model_dir.c_str(),
        python_device.c_str());

    std::string response;
    if (!omni_send_python_t2w_command(ctx_omni, cmd, response)) {
        LOG_ERR("Python T2W: init 命令发送失败\n");
        return false;
    }

    print_with_timestamp("Python T2W init 响应: %s\n", response.c_str());
    return omni_python_t2w_has_status(response, "ok");
}

bool omni_set_python_t2w_ref_audio(struct omni_context * ctx_omni, const std::string & ref_audio_path) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "{\"cmd\":\"set_ref_audio\",\"ref_audio_path\":\"%s\"}", ref_audio_path.c_str());

    std::string response;
    if (!omni_send_python_t2w_command(ctx_omni, cmd, response)) {
        LOG_ERR("Python T2W: set_ref_audio 命令发送失败\n");
        return false;
    }

    print_with_timestamp("Python T2W set_ref_audio 响应: %s\n", response.c_str());
    return omni_python_t2w_has_status(response, "ok");
}

bool omni_process_python_t2w_tokens(
    struct omni_context * ctx_omni,
    const std::vector<int32_t> & tokens,
    bool last_chunk,
    const std::string & output_path,
    double & inference_time_ms,
    double & audio_duration) {
    std::string tokens_json = "[";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            tokens_json += ",";
        }
        tokens_json += std::to_string(tokens[i]);
    }
    tokens_json += "]";

    char cmd[8192];
    snprintf(
        cmd,
        sizeof(cmd),
        "{\"cmd\":\"process\",\"tokens\":%s,\"last_chunk\":%s,\"output_path\":\"%s\"}",
        tokens_json.c_str(),
        last_chunk ? "true" : "false",
        output_path.c_str());

    std::string response;
    if (!omni_send_python_t2w_command(ctx_omni, cmd, response)) {
        LOG_ERR("Python T2W: process 命令发送失败\n");
        return false;
    }

    size_t pos = response.find("\"inference_time_ms\":");
    if (pos != std::string::npos) {
        inference_time_ms = atof(response.c_str() + pos + 20);
    }

    pos = response.find("\"audio_duration\":");
    if (pos != std::string::npos) {
        audio_duration = atof(response.c_str() + pos + 17);
    }

    return omni_python_t2w_has_status(response, "ok");
}

bool omni_reset_python_t2w_cache(struct omni_context * ctx_omni) {
    std::string response;
    if (!omni_send_python_t2w_command(ctx_omni, "{\"cmd\":\"reset\"}", response)) {
        LOG_ERR("Python T2W: reset 命令发送失败\n");
        return false;
    }

    return omni_python_t2w_has_status(response, "ok");
}
