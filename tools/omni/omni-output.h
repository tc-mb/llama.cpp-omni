#pragma once

#include <string>

bool cross_platform_mkdir_p(const std::string & path);
bool omni_ensure_directory(const std::string & dir_path);
void omni_archive_output_dir(const std::string & base_output_dir, const std::string & archive_root = "./old_output");
void omni_merge_wav_files(const std::string & output_dir, int num_chunks);
