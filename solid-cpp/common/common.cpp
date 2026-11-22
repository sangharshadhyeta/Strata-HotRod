#include "ggml.h"
#include "gguf.h"
#include "ggml-backend-moe-cache.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "unicode.h"

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <mutex>
#include <deque>
#include <regex>
#include <sys/wait.h>

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <locale>
#include <windows.h>
#include <string.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(__linux__)
#include <sys/types.h>
#include <pwd.h>
#endif

#if defined(_AIX)
#include <sys/systemcfg.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

common_time_meas::common_time_meas(int64_t & t_acc, bool disable) : t_start_us(disable ? -1 : ggml_time_us()), t_acc(t_acc) {}

common_time_meas::~common_time_meas() {
    if (t_start_us >= 0) {
        t_acc += ggml_time_us() - t_start_us;
    }
}

//
// CPU utils
//

int32_t common_cpu_get_num_physical_cores() {
#if defined(_AIX)
    int32_t logical_cpus = _system_configuration.ncpus;
    int32_t smt_threads = _system_configuration.smt_threads;
    if (smt_threads > 0) {
        return static_cast<int32_t>(logical_cpus / smt_threads);
    }
    if (logical_cpus > 0) {
        return static_cast<int32_t>(logical_cpus);
    }
#elif defined(__linux__)
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu=0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu/cpu"
            + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break; // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (!siblings.empty()) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t num_physical_cores;
    size_t len = sizeof(num_physical_cores);
    int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
#elif defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    unsigned int n_threads_win = std::thread::hardware_concurrency();
    unsigned int default_threads = n_threads_win > 0 ? (n_threads_win <= 4 ? n_threads_win : n_threads_win / 2) : 4;

    DWORD buffer_size = 0;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buffer_size)) {
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            return default_threads;
        }
    }

    std::vector<char> buffer(buffer_size);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &buffer_size)) {
        return default_threads;
    }

    int32_t num_physical_cores = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
    while (buffer_size > 0) {
        if (info->Relationship == RelationProcessorCore) {
            num_physical_cores += info->Processor.GroupCount;
        }
        buffer_size -= info->Size;
        info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(reinterpret_cast<char*>(info) + info->Size);
    }

    return num_physical_cores > 0 ? num_physical_cores : default_threads;
#endif
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>

static void cpuid(unsigned leaf, unsigned subleaf,
                  unsigned *eax, unsigned *ebx, unsigned *ecx, unsigned *edx) {
    __asm__("movq\t%%rbx,%%rsi\n\t"
            "cpuid\n\t"
            "xchgq\t%%rbx,%%rsi"
            : "=a"(*eax), "=S"(*ebx), "=c"(*ecx), "=d"(*edx)
            : "0"(leaf), "2"(subleaf));
}

static int pin_cpu(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
}

static bool is_hybrid_cpu(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return !!(edx & (1u << 15));
}

static bool is_running_on_efficiency_core(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(0x1a, 0, &eax, &ebx, &ecx, &edx);
    int intel_atom = 0x20;
    int core_type = (eax & 0xff000000u) >> 24;
    return core_type == intel_atom;
}

static int cpu_count_math_cpus(int n_cpu) {
    int result = 0;
    for (int cpu = 0; cpu < n_cpu; ++cpu) {
        if (pin_cpu(cpu)) {
            return -1;
        }
        if (is_running_on_efficiency_core()) {
            continue; // efficiency cores harm lockstep threading
        }
        ++cpu; // hyperthreading isn't useful for linear algebra
        ++result;
    }
    return result;
}

#endif // __x86_64__ && __linux__

/**
 * Returns number of CPUs on system that are useful for math.
 */
int32_t common_cpu_get_num_math() {
#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
    int n_cpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_cpu < 1) {
        return common_cpu_get_num_physical_cores();
    }
    if (is_hybrid_cpu()) {
        cpu_set_t affinity;
        if (!pthread_getaffinity_np(pthread_self(), sizeof(affinity), &affinity)) {
            int result = cpu_count_math_cpus(n_cpu);
            pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
            if (result > 0) {
                return result;
            }
        }
    }
#elif defined(__powerpc64__) || defined(__powerpc__)
    int32_t smt_factor = 1;
    int phy_cpus = common_cpu_get_num_physical_cores();
    int logical_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (phy_cpus > 0 && logical_cpus > phy_cpus) {
        smt_factor = logical_cpus / phy_cpus;
    }
    return phy_cpus * std::min(smt_factor, 2);
#endif
    return common_cpu_get_num_physical_cores();
}

// Helper for setting process priority

#if defined(_WIN32)

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    DWORD p = NORMAL_PRIORITY_CLASS;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p = BELOW_NORMAL_PRIORITY_CLASS; break;
        case GGML_SCHED_PRIO_NORMAL:   p = NORMAL_PRIORITY_CLASS;       break;
        case GGML_SCHED_PRIO_MEDIUM:   p = ABOVE_NORMAL_PRIORITY_CLASS; break;
        case GGML_SCHED_PRIO_HIGH:     p = HIGH_PRIORITY_CLASS;         break;
        case GGML_SCHED_PRIO_REALTIME: p = REALTIME_PRIORITY_CLASS;     break;
    }

    if (!SetPriorityClass(GetCurrentProcess(), p)) {
        COM_WRN("failed to set process priority class %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#else // MacOS and POSIX
#include <sys/types.h>
#include <sys/resource.h>

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    int p = 0;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p =  5;  break;
        case GGML_SCHED_PRIO_NORMAL:   p =  0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   p = -5;  break;
        case GGML_SCHED_PRIO_HIGH:     p = -10; break;
        case GGML_SCHED_PRIO_REALTIME: p = -20; break;
    }

    if (setpriority(PRIO_PROCESS, 0, p) != 0) {
        COM_WRN("failed to set process priority %d : %s (%d)\n", prio, strerror(errno), errno);
        return false;
    }
    return true;
}

#endif

//
// CLI argument parsing
//


void postprocess_cpu_params(common_cpu_params & cpuparams, const common_cpu_params * role_model) {
    int32_t n_set = 0;

    if (cpuparams.n_threads < 0) {
        // Assuming everything about cpuparams is invalid
        if (role_model != nullptr) {
            cpuparams = *role_model;
        } else {
            cpuparams.n_threads = common_cpu_get_num_math();
        }
    }

    for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (cpuparams.cpumask[i]) {
            n_set++;
        }
    }

    if (n_set && n_set < cpuparams.n_threads) {
        // Not enough set bits, may experience performance issues.
        COM_WRN("Not enough set bits in CPU mask (%d) to satisfy requested thread count: %d\n", n_set, cpuparams.n_threads);
    }
}

bool parse_cpu_range(const std::string & range, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    size_t dash_loc = range.find('-');
    if (dash_loc == std::string::npos) {
        COM_ERR("%s", "Format of CPU range is invalid! Expected [<start>]-[<end>].\n");
        return false;
    }

    size_t start_i;
    size_t end_i;

    if (dash_loc == 0) {
        start_i = 0;
    } else {
        start_i = std::stoull(range.substr(0, dash_loc));
        if (start_i >= GGML_MAX_N_THREADS) {
            COM_ERR("%s", "Start index out of bounds!\n");
            return false;
        }
    }

    if (dash_loc == range.length() - 1) {
        end_i = GGML_MAX_N_THREADS - 1;
    } else {
        end_i = std::stoull(range.substr(dash_loc + 1));
        if (end_i >= GGML_MAX_N_THREADS) {
            COM_ERR("%s", "End index out of bounds!\n");
            return false;
        }
    }

    for (size_t i = start_i; i <= end_i; i++) {
        boolmask[i] = true;
    }

    return true;
}

bool parse_cpu_mask(const std::string & mask, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    // Discard potential 0x prefix
    size_t start_i = 0;
    if (mask.length() >= 2 && mask.substr(0, 2) == "0x") {
        start_i = 2;
    }

    size_t num_digits = mask.length() - start_i;
    num_digits = std::min<size_t>(num_digits, 128);

    size_t end_i = num_digits + start_i;

    for (size_t i = start_i, n = (num_digits*4 - 1); i < end_i; i++, n-=4) {
        char c = mask.at(i);
        int8_t id = c;

        if ((c >= '0' && c <= '9')) {
            id -= '0';
        } else if (c >= 'a' && c <= 'f') {
            id -= 'a' - 10;
        } else if (c >= 'A' && c <= 'F') {
            id -= 'A' - 10;
        } else {
            COM_ERR("Invalid hex character '%c' at position %d\n", c, int32_t(i));
            return false;
        }

        boolmask[  n  ] = boolmask[  n  ] || ((id & 8) != 0);
        boolmask[n - 1] = boolmask[n - 1] || ((id & 4) != 0);
        boolmask[n - 2] = boolmask[n - 2] || ((id & 2) != 0);
        boolmask[n - 3] = boolmask[n - 3] || ((id & 1) != 0);
    }

    return true;
}

void common_init() {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    common_log_set_prefix(common_log_main(), true);
    common_log_set_timestamps(common_log_main(), true);

    llama_log_set(common_log_default_callback, NULL);
}

void common_params_print_info(const common_params & params, bool print_devices) {
#ifdef NDEBUG
    const char * build_type = "";
#else
    const char * build_type = " (debug)";
#endif
    COM_TRC("%s: build %d (%s) with %s for %s%s\n", __func__, llama_build_number(), llama_commit(), llama_compiler(), llama_build_target(), build_type);

    COM_INF("%s: verbosity = %d (adjust with the `-lv N` CLI arg)\n", __func__, common_log_get_verbosity_thold());

    // device enumeration creates a primary context on CUDA backends, skip it when the caller does not own any device
    if (print_devices) {
        COM_TRC("%s", "device_info:\n");
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * dev = ggml_backend_dev_get(i);
            size_t free, total;
            ggml_backend_dev_memory(dev, &free, &total);
            COM_TRC("  - %-8s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total / 1024 / 1024, free / 1024 / 1024);
        }
    }
    COM_TRC("%s\n", common_params_get_system_info(params).c_str());
}

std::string common_params_get_system_info(const common_params & params) {
    std::ostringstream os;

    os << "system_info: n_threads = " << params.cpuparams.n_threads;
    if (params.cpuparams_batch.n_threads != -1) {
        os << " (n_threads_batch = " << params.cpuparams_batch.n_threads << ")";
    }
#if defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    DWORD logicalProcessorCount = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    os << " / " << logicalProcessorCount << " | " << llama_print_system_info();
#else
    os << " / " << std::thread::hardware_concurrency() << " | " << llama_print_system_info();
#endif

    return os.str();
}

//
// String utils
//

std::string string_format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

std::string string_strip(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();
    while (start < end && std::isspace(str[start])) {
        start++;
    }
    while (end > start && std::isspace(str[end - 1])) {
        end--;
    }
    return str.substr(start, end - start);
}

std::string string_lcs(std::string_view a, std::string_view b) {
    if (a.empty() || b.empty()) return {};

    std::vector<std::vector<size_t>> dp(a.size() + 1, std::vector<size_t>(b.size() + 1, 0));
    size_t best_len = 0;
    size_t best_end_a = 0;

    for (size_t i = 1; i <= a.size(); ++i) {
        for (size_t j = 1; j <= b.size(); ++j) {
            if (a[i - 1] == b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                if (dp[i][j] > best_len) {
                    best_len = dp[i][j];
                    best_end_a = i;
                }
            }
        }
    }
    return std::string(a.substr(best_end_a - best_len, best_len));
}

std::string string_get_sortable_timestamp() {
    using clock = std::chrono::system_clock;

    const clock::time_point current_time = clock::now();
    const time_t as_time_t = clock::to_time_t(current_time);
    char timestamp_no_ns[100];
    std::strftime(timestamp_no_ns, 100, "%Y_%m_%d-%H_%M_%S", std::localtime(&as_time_t));

    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        current_time.time_since_epoch() % 1000000000).count();
    char timestamp_ns[11];
    snprintf(timestamp_ns, 11, "%09" PRId64, ns);

    return std::string(timestamp_no_ns) + "." + std::string(timestamp_ns);
}

void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

std::string regex_escape(const std::string & s) {
    static const std::regex special_chars("[.^$|()*+?\\[\\]{}\\\\]");
    return std::regex_replace(s, special_chars, "\\$&");
}

std::string string_join(const std::vector<std::string> & values, const std::string & separator) {
    std::ostringstream result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            result << separator;
        }
        result << values[i];
    }
    return result.str();
}

std::vector<std::string> string_split(const std::string & str, const std::string & delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    parts.push_back(str.substr(start));

    return parts;
}

std::string string_repeat(const std::string & str, size_t n) {
    if (n == 0) {
        return "";
    }

    std::string result;
    result.reserve(str.length() * n);

    for (size_t i = 0; i < n; ++i) {
        result += str;
    }

    return result;
}

std::string string_from(bool value) {
    return value ? "true" : "false";
}

std::string string_from(const std::vector<int> & values) {
    std::stringstream buf;

    buf << "[ ";
    bool first = true;
    for (auto e : values) {
        if (first) {
            first = false;
        } else {
            buf << ", ";
        }
        buf << std::to_string(e);
    }
    buf << " ]";

    return buf.str();
}

std::string string_from(const struct llama_context * ctx, const std::vector<llama_token> & tokens) {
    std::stringstream buf;

    buf << "[ ";

    bool first = true;
    for (const auto & token : tokens) {
        if (!first) {
            buf << ", ";
        } else {
            first = false;
        }

        auto detokenized = common_token_to_piece(ctx, token);

        buf << "'" << detokenized << "'"
            << ":" << std::to_string(token);
    }

    buf << " ]";

    return buf.str();
}

std::string string_from(const struct llama_context * ctx, const struct llama_batch & batch) {
    std::stringstream buf;

    buf << "[ ";

    bool first = true;
    for (int i = 0; i < batch.n_tokens; ++i) {
        if (!first) {
            buf << ", ";
        } else {
            first = false;
        }

        auto detokenized = common_token_to_piece(ctx, batch.token[i]);

        buf << "\n"          << std::to_string(i)
            << ", token '"   << detokenized << "'"
            << ", pos "      << std::to_string(batch.pos[i])
            << ", n_seq_id " << std::to_string(batch.n_seq_id[i])
            << ", seq_id "   << std::to_string(batch.seq_id[i][0])
            << ", logits "   << std::to_string(batch.logits[i]);
    }

    buf << " ]";

    return buf.str();
}

void string_process_escapes(std::string & input) {
    std::size_t input_len = input.length();
    std::size_t output_idx = 0;

    for (std::size_t input_idx = 0; input_idx < input_len; ++input_idx) {
        if (input[input_idx] == '\\' && input_idx + 1 < input_len) {
            switch (input[++input_idx]) {
                case 'n':  input[output_idx++] = '\n'; break;
                case 'r':  input[output_idx++] = '\r'; break;
                case 't':  input[output_idx++] = '\t'; break;
                case '\'': input[output_idx++] = '\''; break;
                case '\"': input[output_idx++] = '\"'; break;
                case '\\': input[output_idx++] = '\\'; break;
                case 'x':
                    // Handle \x12, etc
                    if (input_idx + 2 < input_len) {
                        const char x[3] = { input[input_idx + 1], input[input_idx + 2], 0 };
                        char *err_p = nullptr;
                        const long val = std::strtol(x, &err_p, 16);
                        if (err_p == x + 2) {
                            input_idx += 2;
                            input[output_idx++] = char(val);
                            break;
                        }
                    }
                    // fall through
                default:   input[output_idx++] = '\\';
                           input[output_idx++] = input[input_idx]; break;
            }
        } else {
            input[output_idx++] = input[input_idx];
        }
    }

    input.resize(output_idx);
}

bool string_parse_kv_override(const char * data, std::vector<llama_model_kv_override> & overrides) {
    const char * sep = strchr(data, '=');
    if (sep == nullptr || sep - data >= 128) {
        COM_ERR("%s: malformed KV override '%s'\n", __func__, data);
        return false;
    }
    llama_model_kv_override kvo;
    std::strncpy(kvo.key, data, sep - data);
    kvo.key[sep - data] = 0;
    sep++;
    if (strncmp(sep, "int:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        kvo.val_i64 = std::atol(sep);
    } else if (strncmp(sep, "float:", 6) == 0) {
        sep += 6;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
        kvo.val_f64 = std::atof(sep);
    } else if (strncmp(sep, "bool:", 5) == 0) {
        sep += 5;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
        if (std::strcmp(sep, "true") == 0) {
            kvo.val_bool = true;
        } else if (std::strcmp(sep, "false") == 0) {
            kvo.val_bool = false;
        } else {
            COM_ERR("%s: invalid boolean value for KV override '%s'\n", __func__, data);
            return false;
        }
    } else if (strncmp(sep, "str:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        if (strlen(sep) > 127) {
            COM_ERR("%s: malformed KV override '%s', value cannot exceed 127 chars\n", __func__, data);
            return false;
        }
        strncpy(kvo.val_str, sep, 127);
        kvo.val_str[127] = '\0';
    } else {
        COM_ERR("%s: invalid type for KV override '%s'\n", __func__, data);
        return false;
    }
    overrides.emplace_back(std::move(kvo));
    return true;
}

static inline bool glob_class_match(const char c, const char * pattern, const char * class_end) {
    const char * class_start = pattern;
    bool negated = false;

    if (*class_start == '!') {
        negated = true;
        class_start++;
    }

    // If first character after negation is ']' or '-', treat it as literal
    if (*class_start == ']' || *class_start == '-') {
        if (class_start < class_end && *class_start == c) {
            return !negated;
        }
        class_start++;
    }

    bool matched = false;

    while (class_start < class_end) {
        if (class_start + 2 < class_end && class_start[1] == '-' && class_start[2] != ']') {
            char start_char = *class_start;
            char end_char = class_start[2];
            if (c >= start_char && c <= end_char) {
                matched = true;
                break;
            }
            class_start += 3;
        } else {
            if (*class_start == c) {
                matched = true;
                break;
            }
            class_start++;
        }
    }

    return negated ? !matched : matched;
}

// simple glob: * matches non-/ chars, ** matches anything including /, [] matches character class
static inline bool glob_match(const char * pattern, const char * str) {
    if (*pattern == '\0') {
        return *str == '\0';
    }
    if (pattern[0] == '*' && pattern[1] == '*') {
        const char * p = pattern + 2;
        if (glob_match(p, str)) return true;
        if (*str != '\0') return glob_match(pattern, str + 1);
        return false;
    }
    if (*pattern == '*') {
        const char * p = pattern + 1;
        for (; *str != '\0' && *str != '/'; str++) {
            if (glob_match(p, str)) return true;
        }
        return glob_match(p, str);
    }
    if (*pattern == '?' && *str != '\0' && *str != '/') {
        return glob_match(pattern + 1, str + 1);
    }
    if (*pattern == '[') {
        const char * class_end = pattern + 1;
        // If first character after '[' is ']' or '-', treat it as literal
        if (*class_end == ']' || *class_end == '-') {
            class_end++;
        }
        while (*class_end != '\0' && *class_end != ']') {
            class_end++;
        }
        if (*class_end == ']') {
            if (*str == '\0') return false;
            bool matched = glob_class_match(*str, pattern + 1, class_end);
            return matched && glob_match(class_end + 1, str + 1);
        } else {
            if (*str == '[') {
                return glob_match(pattern + 1, str + 1);
            }
            return false;
        }
    }
    if (*pattern == *str) {
        return glob_match(pattern + 1, str + 1);
    }
    return false;
}

bool glob_match(const std::string & pattern, const std::string & str) {
    return glob_match(pattern.c_str(), str.c_str());
}

//
// Filesystem utils
//

// Validate if a filename is safe to use
// To validate a full path, split the path by the OS-specific path separator, and validate each part with this function
bool fs_validate_filename(const std::string & filename, bool allow_subdirs) {
    if (!filename.length()) {
        // Empty filename invalid
        return false;
    }
    if (filename.length() > 255) {
        // Limit at common largest possible filename on Linux filesystems
        // to avoid unnecessary further validation
        // (On systems with smaller limits it will be caught by the OS)
        return false;
    }

    size_t offset = 0;
    while (offset < filename.size()) {
        utf8_parse_result result = common_parse_utf8_codepoint(filename, offset);

        if (result.status != utf8_parse_result::SUCCESS) {
            return false;
        }
        uint32_t c = result.codepoint;

        if ((result.bytes_consumed == 2 && c < 0x80) ||
            (result.bytes_consumed == 3 && c < 0x800) ||
            (result.bytes_consumed == 4 && c < 0x10000)) {
            return false;
        }

        // Check for forbidden codepoints:
        // - Control characters
        // - Unicode equivalents of illegal characters
        // - UTF-16 surrogate pairs
        // - UTF-8 replacement character
        // - Byte order mark (BOM)
        // - Illegal characters: / \ : * ? " < > |
        if (c <= 0x1F // Control characters (C0)
            || c == 0x7F // Control characters (DEL)
            || (c >= 0x80 && c <= 0x9F) // Control characters (C1)
            || c == 0xFF0E // Fullwidth Full Stop (period equivalent)
            || c == 0x2215 // Division Slash (forward slash equivalent)
            || c == 0x2216 // Set Minus (backslash equivalent)
            || (c >= 0xD800 && c <= 0xDFFF) // UTF-16 surrogate pairs
            || c > 0x10FFFF // Max Unicode limit
            || c == 0xFFFD // Replacement Character (UTF-8)
            || c == 0xFEFF // Byte Order Mark (BOM)
            || c == ':' || c == '*' // Illegal characters
            || c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
            return false;
        }
        if (!allow_subdirs && (c == '/' || c == '\\')) {
            // Subdirectories not allowed, reject path separators
            return false;
        }
        offset += result.bytes_consumed;
    }

    // Reject any leading or trailing ' ', or any trailing '.', these are stripped on Windows and will cause a different filename
    // Unicode and other whitespace is not affected, only 0x20 space
    if (filename.front() == ' ' || filename.back() == ' ' || filename.back() == '.') {
        return false;
    }

    // Reject any ".." (currently stricter than necessary, it should be fine to just check for == ".." instead)
    if (filename.find("..") != std::string::npos) {
        return false;
    }

    // Reject "."
    if (filename == ".") {
        return false;
    }

    return true;
}

#include <iostream>


#ifdef _WIN32
static std::wstring utf8_to_wstring(const std::string & str) {
    if (str.empty()) {
        return std::wstring();
    }

    int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);

    if (size <= 0) {
        return std::wstring();
    }

    std::wstring wstr(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], size);

    return wstr;
}
#endif

// returns true if successful, false otherwise
bool fs_create_directory_with_parents(const std::string & path) {
#ifdef _WIN32
    std::wstring wpath = utf8_to_wstring(path);

    // if the path already exists, check whether it's a directory
    const DWORD attributes = GetFileAttributesW(wpath.c_str());
    if ((attributes != INVALID_FILE_ATTRIBUTES) && (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
        return true;
    }

    size_t pos_slash = 0;

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('\\', pos_slash)) != std::string::npos) {
        const std::wstring subpath = wpath.substr(0, pos_slash);

        pos_slash += 1;

        // skip the drive letter, in some systems it can return an access denied error
        if (subpath.length() == 2 && subpath[1] == ':') {
            continue;
        }

        const bool success = CreateDirectoryW(subpath.c_str(), NULL);

        if (!success) {
            const DWORD error = GetLastError();

            // if the path already exists, ensure that it's a directory
            if (error == ERROR_ALREADY_EXISTS) {
                const DWORD attributes = GetFileAttributesW(subpath.c_str());
                if (attributes == INVALID_FILE_ATTRIBUTES || !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    return false;
                }
            } else {
                return false;
            }
        }
    }

    return true;
#else
    // if the path already exists, check whether it's a directory
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        return S_ISDIR(info.st_mode);
    }

    size_t pos_slash = 1; // skip leading slashes for directory creation

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('/', pos_slash)) != std::string::npos) {
        const std::string subpath = path.substr(0, pos_slash);
        struct stat info;

        // if the path already exists, ensure that it's a directory
        if (stat(subpath.c_str(), &info) == 0) {
            if (!S_ISDIR(info.st_mode)) {
                return false;
            }
        } else {
            // create parent directories
            const int ret = mkdir(subpath.c_str(), 0755);
            if (ret != 0) {
                return false;
            }
        }

        pos_slash += 1;
    }

    return true;
#endif // _WIN32
}

bool fs_is_directory(const std::string & path) {
    std::filesystem::path dir(path);
    return std::filesystem::exists(dir) && std::filesystem::is_directory(dir);
}

std::string common_get_env(const std::string & name) {
    const char * value = std::getenv(name.c_str());
    return value == nullptr ? "" : value;
}

void common_set_env(const std::string & name, const std::string & value) {
#if defined(_WIN32)
    _putenv_s(name.c_str(), value.c_str());
#else
    if (value.empty()) {
        unsetenv(name.c_str());
    } else {
        setenv(name.c_str(), value.c_str(), 1);
    }
#endif
}

std::string fs_get_cache_directory() {
    std::string cache_directory = "";
    auto ensure_trailing_slash = [](std::string p) {
        // Make sure to add trailing slash
        if (p.empty() || p.back() != DIRECTORY_SEPARATOR) {
            p += DIRECTORY_SEPARATOR;
        }
        return p;
    };
    cache_directory = common_get_env("LLAMA_CACHE");
    if (cache_directory.empty()) {
#if defined(__linux__) || defined(__FreeBSD__) || defined(_AIX) || \
        defined(__OpenBSD__) || defined(__NetBSD__)
        const std::string xdg_cache_home = common_get_env("XDG_CACHE_HOME");
        const std::string home           = common_get_env("HOME");
        if (!xdg_cache_home.empty()) {
            cache_directory = xdg_cache_home;
        } else if (!home.empty()) {
            cache_directory = home + "/.cache/";
        } else {
#if defined(__linux__)
            /* no $HOME is defined, fallback to getpwuid */
            struct passwd *pw = getpwuid(getuid());
            if ((!pw) || (!pw->pw_dir)) {
                throw std::runtime_error("Failed to find $HOME directory");
            }

            cache_directory = std::string(pw->pw_dir) + std::string("/.cache/");
#else /* defined(__linux__) */
            throw std::runtime_error("Failed to find $HOME directory");
#endif /* defined(__linux__) */
        }
#elif defined(__APPLE__)
        cache_directory = common_get_env("HOME");
        if (cache_directory.empty()) {
            throw std::runtime_error("Failed to find $HOME directory");
        }
        cache_directory += "/Library/Caches/";
#elif defined(_WIN32)
        cache_directory = common_get_env("LOCALAPPDATA");
        if (cache_directory.empty()) {
            throw std::runtime_error("Failed to find %LOCALAPPDATA% directory");
        }
#elif defined(__EMSCRIPTEN__)
        GGML_ABORT("not implemented on this platform");
#else
#  error Unknown architecture
#endif
        cache_directory = ensure_trailing_slash(cache_directory);
        cache_directory += "llama.cpp";
    }
    return ensure_trailing_slash(cache_directory);
}

std::string fs_get_config_directory() {
    std::string config_directory = "";
    auto ensure_trailing_slash = [](std::string p) {
        if (p.empty() || p.back() != DIRECTORY_SEPARATOR) {
            p += DIRECTORY_SEPARATOR;
        }
        return p;
    };
#if defined(__linux__) || defined(__FreeBSD__) || defined(_AIX) || \
        defined(__OpenBSD__) || defined(__NetBSD__) || defined(__APPLE__)
    const std::string xdg_config_home = common_get_env("XDG_CONFIG_HOME");
    const std::string home            = common_get_env("HOME");
    if (!xdg_config_home.empty()) {
        config_directory = xdg_config_home;
    } else if (!home.empty()) {
        config_directory = home + "/.config/";
    } else {
#if defined(__linux__)
        /* no $HOME is defined, fallback to getpwuid */
        struct passwd *pw = getpwuid(getuid());
        if ((!pw) || (!pw->pw_dir)) {
            throw std::runtime_error("Failed to find $HOME directory");
        }

        config_directory = std::string(pw->pw_dir) + std::string("/.config/");
#else
        throw std::runtime_error("Failed to find $HOME directory");
#endif
    }
#elif defined(_WIN32)
    config_directory = common_get_env("APPDATA");
    if (config_directory.empty()) {
        throw std::runtime_error("Failed to find %APPDATA% directory");
    }
#elif defined(__EMSCRIPTEN__)
    // caller decides what to do when there is no config directory
    throw std::runtime_error("not implemented on this platform");
#else
#  error Unknown architecture
#endif
    config_directory = ensure_trailing_slash(config_directory);
    config_directory += "llama.cpp";
    return ensure_trailing_slash(config_directory);
}

std::string fs_get_cache_file(const std::string & filename) {
    GGML_ASSERT(filename.find(DIRECTORY_SEPARATOR) == std::string::npos);
    std::string cache_directory = fs_get_cache_directory();
    const bool success = fs_create_directory_with_parents(cache_directory);
    if (!success) {
        throw std::runtime_error("failed to create cache directory: " + cache_directory);
    }
    return cache_directory + filename;
}

std::vector<common_file_info> fs_list(const std::string & path, bool include_directories) {
    std::vector<common_file_info> files;
    if (path.empty()) return files;

    std::filesystem::path dir(path);
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        return files;
    }

    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        try {
            // Only include regular files (skip directories)
            const auto & p = entry.path();
            if (std::filesystem::is_regular_file(p)) {
                common_file_info info;
                info.path   = p.string();
                info.name   = p.filename().string();
                info.is_dir = false;
                try {
                    info.size = static_cast<size_t>(std::filesystem::file_size(p));
                } catch (const std::filesystem::filesystem_error &) {
                    info.size = 0;
                }
                files.push_back(std::move(info));
            } else if (include_directories && std::filesystem::is_directory(p)) {
                common_file_info info;
                info.path   = p.string();
                info.name   = p.filename().string();
                info.size   = 0; // Directories have no size
                info.is_dir = true;
                files.push_back(std::move(info));
            }
        } catch (const std::filesystem::filesystem_error &) {
            // skip entries we cannot inspect
            continue;
        }
    }

    return files;
}

std::ifstream fs_open_ifstream(const std::string & fname, std::ios_base::openmode mode) {
#ifdef _WIN32
    int wlen = MultiByteToWideChar(CP_UTF8, 0, fname.c_str(), -1, NULL, 0);
    if (!wlen) { return std::ifstream(); }
    std::vector<wchar_t> wfname(wlen);
    (void)MultiByteToWideChar(CP_UTF8, 0, fname.c_str(), -1, wfname.data(), wlen);
    return std::ifstream(wfname.data(), mode);
#else
    return std::ifstream(fname, mode);
#endif
}

//
// TTY utils
//

bool tty_can_use_colors() {
    // Check NO_COLOR environment variable (https://no-color.org/)
    if (const char * no_color = std::getenv("NO_COLOR")) {
        if (no_color[0] != '\0') {
            return false;
        }
    }

    // Check TERM environment variable
    if (const char * term = std::getenv("TERM")) {
        if (std::strcmp(term, "dumb") == 0) {
            return false;
        }
    }

    // Check if stdout and stderr are connected to a terminal
    // We check both because log messages can go to either
    bool stdout_is_tty = isatty(fileno(stdout));
    bool stderr_is_tty = isatty(fileno(stderr));

    return stdout_is_tty || stderr_is_tty;
}

//
// Model utils
//

// TODO: move to common/sampling
static void common_init_sampler_from_model(
    const llama_model * model,
    common_params_sampling & sparams) {

    const uint64_t config = sparams.user_sampling_config;

    auto get_int32 = [&](const char * key, int32_t & dst, uint64_t user_config) {
        if (config & user_config) {
            return;
        }

        char buf[64] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char * end = nullptr;
            int32_t v = strtol(buf, &end, 10);
            if (end && end != buf) {
                dst = v;
            }
        }
    };

    auto get_float = [&](const char * key, float & dst, uint64_t user_config) {
        if (config & user_config) {
            return;
        }

        char buf[128] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char * end = nullptr;
            float v = strtof(buf, &end);
            if (end && end != buf) {
                dst = v;
            }
        }
    };

    // Sampling sequence
    if (!(config & common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS)) {
        char buf[512] = {0};
        if (llama_model_meta_val_str(model, llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE), buf, sizeof(buf)) > 0) {
            const std::vector<std::string> sampler_names = string_split<std::string>(std::string(buf), ';');
            if (!sampler_names.empty()) {
                sparams.samplers = common_sampler_types_from_names(sampler_names);
            }
        }
    }

    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_K),           sparams.top_k,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_P),           sparams.top_p,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_P);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIN_P),           sparams.min_p,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIN_P);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY), sparams.xtc_probability, common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD),   sparams.xtc_threshold,   common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TEMP),            sparams.temp,            common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TEMP);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N),  sparams.penalty_last_n,  common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT),  sparams.penalty_repeat,  common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT),        sparams.mirostat,        common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU),    sparams.mirostat_tau,    common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA),    sparams.mirostat_eta,    common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA);
}

struct common_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top

    llama_model_ptr   model;
    llama_context_ptr context;

    std::vector<llama_adapter_lora_ptr> lora;

    std::vector<common_sampler_ptr> samplers;
    std::vector<llama_sampler_seq_config> samplers_seq_config;
};

// True when the real (non-probe) load will keep CPU-resident model weights
// on a plain lazy mmap: LOAD_MODE_AUTO resolves to mmap whenever the device
// supports it (the common case), and LOAD_MODE_MMAP asks for it explicitly.
// MLOCK/MMAP_MLOCK force full residency (that's the point of mlock), and
// NONE/DIRECT_IO never mmap at all, so those still need the real byte count.
static bool common_mmap_is_lazy(const llama_model_params & mparams) {
    return mparams.load_mode == LLAMA_LOAD_MODE_AUTO || mparams.load_mode == LLAMA_LOAD_MODE_MMAP;
}

// margin: bytes that must remain free per device on top of what's needed.
// Defaults to 0 (bare fit) for callers that only ask "would this load at all".
// A caller whose answer will be re-judged by common_fit_params() must pass the
// same margin that check will demand - otherwise placement approves a config
// that fit then rejects, and the two silently disagree.
//
// host_model_lazy_mmap: whether the host (CPU) entry's `model` bytes are a
// lazily-paged zero-copy mapping rather than a real allocation - see
// common_mmap_is_lazy(). When true, those bytes are excluded from what must
// fit in available RAM: the kernel reclaims clean, unmodified file pages
// under pressure instead of committing them, so CPU-offloaded ("cold") MoE
// experts don't need to be simultaneously resident the way a real copy would
// - only the host's KV cache and compute-buffer bytes (context/compute) are
// genuine allocations that still have to fit.
static bool common_device_memory_data_fits(
        const common_device_memory_data_vec & data, int64_t margin = 0, bool host_model_lazy_mmap = false) {
    for (size_t i = 0; i < data.size(); i++) {
        const auto & d = data[i];
        if (d.total <= 0) {
            continue; // not a real device (host aggregate, or a device with unknown budget)
        }
        // by construction (common_get_device_memory_data_impl), the host
        // aggregate is always the last entry
        const bool is_host = (i + 1 == data.size());
        const size_t model_bytes = (is_host && host_model_lazy_mmap) ? 0 : d.model;
        const int64_t needed = (int64_t) model_bytes + (int64_t) d.context + (int64_t) d.compute;
        if (needed + margin > d.free) {
            return false;
        }
    }
    return true;
}

// The buffer type -ncmoe places its overridden experts in. Plain
// ggml_backend_cpu_buffer_type() is host memory the CPU backend can compute
// on, but it's invisible to op_offload: op_offload's backend-assignment path
// (ggml_backend_sched_backend_id_from_cur) only considers diverting a node to
// GPU for buffers a GPU device's own get_host_buffer_type() produced (the
// same buffer type llama-model.cpp's make_cpu_buft_list already uses for
// tensors that overflow VRAM naturally, for exactly this reason - see its
// own comment: "useful when processing of large batches is offloaded to a
// GPU device, since it reduces the time spent on data transfers"). Plain CPU
// buft tensors stay CPU-computed forever, at any batch size - verified
// empirically: -ncmoe layers never engaged op_offload even at n_tokens on
// the order of thousands. Resolving the same host buffer type here (falling
// back to plain CPU buft if no device offers one, e.g. CPU-only builds)
// makes explicit -ncmoe placements op_offload-eligible during prefill the
// same way naturally-overflowing tensors already are, without touching
// ggml_moe_cache at all - its CPU-dispatch gate checks buffer *usage*
// (WEIGHTS), not buffer *type*, so decode-time behavior is unaffected.
static ggml_backend_buffer_type_t common_moe_cpu_override_buft() {
    static ggml_backend_buffer_type_t buft = [] () -> ggml_backend_buffer_type_t {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                continue;
            }
            ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
            if (host_buft) {
                if (getenv("MOE_CACHE_DEBUG_GATE")) {
                    fprintf(stderr, "[ncmoe-buft-dbg] resolved HOST buft=%p (name=%s) from dev=%s\n",
                            (void*)host_buft, ggml_backend_buft_name(host_buft), ggml_backend_dev_name(dev));
                }
                return host_buft;
            }
        }
        if (getenv("MOE_CACHE_DEBUG_GATE")) {
            fprintf(stderr, "[ncmoe-buft-dbg] no GPU host buft found, falling back to plain CPU buft\n");
        }
        return ggml_backend_cpu_buffer_type();
    }();
    return buft;
}

// Shared by Layer 1 (use the safe floor as-is) and Layer 2/--moe-calibrate
// (use the safe floor as the starting point for empirical throughput
// candidates above it). Keeps its own C-string storage alive for the
// lifetime of the process, same as the override lists it hands back.
static std::vector<llama_model_tensor_buft_override> common_moe_build_cpu_overrides(uint32_t n) {
    static std::list<std::string> pattern_storage;
    std::vector<llama_model_tensor_buft_override> ov;
    ggml_backend_buffer_type_t buft = common_moe_cpu_override_buft();
    for (uint32_t i = 0; i < n; i++) {
        pattern_storage.push_back(llm_ffn_exps_block_regex((int) i));
        ov.push_back({pattern_storage.back().c_str(), buft});
    }
    ov.push_back({nullptr, nullptr});
    return ov;
}

static bool common_moe_fits_with_n(
        const char * path_model, const llama_model_params & mparams_base,
        const llama_context_params & cparams, uint32_t n,
        std::vector<ggml_backend_dev_t> & devs, uint32_t & hp_ngl, uint32_t & hp_n_ctx_train, uint32_t & hp_n_expert,
        int64_t margin = 0) {
    std::vector<llama_model_tensor_buft_override> trial = common_moe_build_cpu_overrides(n);
    llama_model_params mparams_trial = mparams_base;
    mparams_trial.tensor_buft_overrides = trial.data();
    try {
        common_device_memory_data_vec trial_data = common_get_device_memory_data(
                path_model, &mparams_trial, &cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, GGML_LOG_LEVEL_ERROR);
        return common_device_memory_data_fits(trial_data, margin, common_mmap_is_lazy(mparams_base));
    } catch (const std::exception &) {
        return false;
    }
}

// Measure the draft / MTP context's device memory and fold it into
// params.fit_params_target, so every fit or calibration probe that runs after
// this point accounts for VRAM the draft is going to take.
//
// Two consumers need this and used to disagree. The server measured the draft
// in load_model and added it to the target; --moe-calibrate runs earlier, in
// server.cpp, and never did - so calibration benchmarked placements with the
// draft's VRAM still notionally free (2105 MiB of it on Qwen3.8-Flash-Next),
// picked a placement that only fits without the draft, and the serving-side fit
// guard then overrode it. That override is recorded in this file's own comments
// as an unexplained conservatism; this is where it came from.
//
// Measures only - it deliberately does not apply the result, because the two
// consumers apply it differently and one process does both. --moe-calibrate
// does not exit when it finishes; it goes on to serve in the same process, so
// a helper that folded the bytes into fit_params_target would have them added
// once here and a second time by the server's own load_model. Calibration also
// wants them added ONCE to a margin that is already 3x the fit target, not
// tripled along with it.
//
// Fills out_per_device with bytes indexed against the caller's device list and
// returns the total. 0 / empty if there is no draft or it could not be
// measured; failure is not fatal, an unmeasured draft is the status quo.
size_t common_measure_draft_memory(const common_params & params, std::vector<size_t> & out_per_device) {
    out_per_device.clear();
    if (!params.speculative.has_dft()) {
        return 0;
    }

    common_params params_dft = common_base_params_to_speculative(params);

    auto mparams_dft = common_model_params_to_llama(params_dft);
    auto cparams_dft = common_context_params_to_llama(params_dft);

    const bool spec_mtp = std::find(params.speculative.types.begin(),
                                    params.speculative.types.end(),
                                    COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();
    if (spec_mtp) {
        cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }
    cparams_dft.n_rs_seq = 0;

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0, hp_nct = 0, hp_nex = 0;

    // A draft head that borrows embeddings / LM head from its target cannot
    // build a context without one - see common_make_probe_context.
    common_probe_context probe_tgt;
    common_device_memory_data_vec dmd;
    try {
        try {
            dmd = common_get_device_memory_data(params_dft.model.path.c_str(), &mparams_dft, &cparams_dft,
                                                devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);
        } catch (const std::exception &) {
            // these take a non-const ref; this helper must not mutate the caller's params
            common_params params_tgt = params;
            auto mparams_tgt = common_model_params_to_llama(params_tgt);
            auto cparams_tgt = common_context_params_to_llama(params_tgt);
            common_make_probe_context(params_tgt.model.path.c_str(), &mparams_tgt, &cparams_tgt,
                                      probe_tgt, GGML_LOG_LEVEL_ERROR);
            if (probe_tgt.ctx == nullptr) {
                throw;
            }
            cparams_dft.ctx_other = probe_tgt.ctx;
            dmd = common_get_device_memory_data(params_dft.model.path.c_str(), &mparams_dft, &cparams_dft,
                                                devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);
        }
    } catch (const std::exception & e) {
        LOG_WRN("%s: could not measure the draft's device memory (%s) - probes after this point will "
                "treat its VRAM as free\n", __func__, e.what());
        return 0;
    }

    std::vector<ggml_backend_dev_t> tgt_devices = params.devices;
    if (tgt_devices.empty()) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            tgt_devices.push_back(ggml_backend_dev_get(i));
        }
    }
    out_per_device.assign(tgt_devices.size(), 0);

    size_t total = 0;
    for (size_t j = 0; j < devs.size(); ++j) {
        const size_t bytes = dmd[j].model + dmd[j].context + dmd[j].compute;
        for (size_t i = 0; i < tgt_devices.size(); i++) {
            if (tgt_devices[i] == devs[j]) {
                out_per_device[i] += bytes;
                total += bytes;
                break;
            }
        }
    }
    return total;
}

// An MTP head is recognised by what is in its header: gemma-4's has its own architecture ("gemma4-assistant"),
// and a head that shares its target's architecture (Qwen's "mtp-*-shared") declares <arch>.nextn_shared_target_tensors.
// A full model that merely contains a trailing MTP block has neither.
static bool common_draft_file_is_mtp_head(const std::string & path) {
    struct gguf_init_params ip = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
    struct gguf_context * gctx = gguf_init_from_file(path.c_str(), ip);
    if (!gctx) {
        return false;
    }
    bool mtp = false;
    const int64_t k_arch = gguf_find_key(gctx, "general.architecture");
    if (k_arch >= 0 && gguf_get_kv_type(gctx, k_arch) == GGUF_TYPE_STRING) {
        const std::string arch = gguf_get_val_str(gctx, k_arch);
        const std::string suffix = "-assistant";
        if (arch.size() > suffix.size() && arch.compare(arch.size() - suffix.size(), suffix.size(), suffix) == 0) {
            mtp = true;
        } else {
            const int64_t k_shared = gguf_find_key(gctx, (arch + ".nextn_shared_target_tensors").c_str());
            mtp = k_shared >= 0 && gguf_get_kv_type(gctx, k_shared) == GGUF_TYPE_BOOL && gguf_get_val_bool(gctx, k_shared);
        }
    }
    gguf_free(gctx);
    return mtp;
}

void common_speculative_default_type_from_draft(common_params & params) {
    if (!params.speculative.has_dft() || params.speculative.draft.mparams.path.empty()) {
        return;
    }
    // only when no type was asked for: the default is the single entry NONE
    if (params.speculative.types != std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_NONE}) {
        return;
    }
    if (common_draft_file_is_mtp_head(params.speculative.draft.mparams.path)) {
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        LOG_INF("%s: the draft file is an MTP head - using --spec-type draft-mtp (pass --spec-type to choose another)\n", __func__);
    }
}

// The draft's VRAM on the first device, for the serving-time placement probes. Calibration folds the draft into
// its margin (see the --moe-calibrate probe); the serving paths did not, so a launch that reached them with no
// calibration entry placed the target into the whole device and then could not load the draft (cudaMalloc out
// of memory for the 446 MiB MTP head, server exit - found 1 Oct 2026 by dropping -ncmoe from the gemma launch).
// 0 when there is no draft or it cannot be measured, which is the previous behaviour.
static int64_t common_draft_reserve_device0(const common_params & params) {
    std::vector<size_t> per_device;
    common_measure_draft_memory(params, per_device);
    return per_device.empty() ? 0 : (int64_t) per_device[0];
}

struct common_moe_fit_probe_result {
    bool     is_moe        = false;
    bool     already_fits  = false;
    bool     found_safe_n  = false;
    uint32_t safe_n        = 0;
    uint32_t n_layer       = 0;
};

// Binary-searches the minimal MoE CPU-offload layer count N such that the
// model fits in available device memory - fit is monotonic in N (more
// offload never needs more memory), so this is O(log n_layer) no-alloc
// probes, not O(n_layer). A live, per-launch computation: the answer
// depends on -c/--parallel/free VRAM at that moment, not a cached constant.
static common_moe_fit_probe_result common_moe_find_safe_layers(
        const char * path_model, const llama_model_params & mparams_base, const llama_context_params & cparams,
        int64_t margin = 0) {
    common_moe_fit_probe_result result;

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0, hp_n_ctx_train = 0, hp_n_expert = 0;
    common_device_memory_data_vec data;
    try {
        data = common_get_device_memory_data(path_model, &mparams_base, &cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert,
                GGML_LOG_LEVEL_ERROR);
    } catch (const std::exception & e) {
        // Deliberately logged, not swallowed: an empty result here reads
        // identically to "not is_moe" to every caller, which previously
        // misreported a real probe crash (invalid n_seq_max, OOM, etc.) as
        // "model has no MoE experts" - found by hitting exactly that with
        // --moe-calibrate + --model-draft together.
        LOG_ERR("%s: no-alloc probe failed: %s\n", __func__, e.what());
        return result;
    }
    result.n_layer = hp_ngl;
    result.is_moe  = hp_n_expert > 0;

    if (common_device_memory_data_fits(data, margin, common_mmap_is_lazy(mparams_base))) {
        result.already_fits = true;
        return result;
    }
    if (!result.is_moe) {
        return result;
    }
    if (!common_moe_fits_with_n(path_model, mparams_base, cparams, hp_ngl, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, margin)) {
        return result; // nothing fits even with everything offloaded
    }

    uint32_t lo = 1, hi = hp_ngl;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (common_moe_fits_with_n(path_model, mparams_base, cparams, mid, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, margin)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    result.found_safe_n = true;
    result.safe_n        = lo;
    return result;
}

// Bumped when calibration gains a check that changes which configurations it
// is willing to record. An entry written before a gate existed was never
// tested by it, so the fields that gate protects must not be applied from it.
// 1 = measured with the degeneracy guard, the short-prompt quality pass, the
// output-fidelity check against a substitution-free reference, and the
// repeat-identical-request reproducibility check.
// 2 = the answer check reads the answer channel only (empty after </think> is
// wrong), samples as served rather than greedily, asks 8 questions, and takes
// its bar and tolerance from the substitution-off reference measured twice -
// never from the first candidate. Version 1 approved rank 1 / -1.0 sigma on
// Qwen3.8-Flash-Next, which served 1-2 correct answers in 10; every version-1
// substitution setting is suspect for the same reason, so none is applied.
static constexpr int COMMON_MOE_CALIBRATION_GATES_VERSION = 2;
// The version of the shipped defaults an entry was written under (1 Oct 2026: depth 4, p_min 0.5,
// probabilistic acceptance, neuron subsetting off). An entry from before it recorded the speculative
// settings from whatever the launch carried - a flag left on a command line became a "calibrated"
// value that outranked every later default (gemma-4's depth 8) - so its speculative fields are read
// as unmeasured. Placement, threads, cache size and the substitution floor are unaffected.
static constexpr int COMMON_MOE_DEFAULTS_VERSION = 2;

struct common_moe_calibration_entry {
    int         n_cpu_moe       = 0;
    int         n_threads       = -1;
    int         n_threads_batch = -1;
    int         spec_n_max      = -1; // -1 = no MTP calibration recorded
    int         defaults_version = 0;  // COMMON_MOE_DEFAULTS_VERSION when written; 0 = before it existed
    int         concurrency     = 1;  // > 1: tok_per_sec is aggregate throughput at this many concurrent requests, not solo
    double      tok_per_sec     = 0.0;
    int         moe_cache_mb    = -1; // -1 = not calibrated, use --moe-cache auto
    int         substitute_min_rank = -1; // -1 = not calibrated, use the runtime default gate
    int         admit_after         = -1; // -1 = not calibrated, use the runtime default (2)
    int         readmit_after       = -1; // -1 = not calibrated, use the runtime default (8)
    int         spec_prob_accept    = -1; // -1 = not calibrated; 0 = exact-match only, 1 = probabilistic
    // The drafter cascade, as a --spec-type list. Empty = not calibrated, leave
    // the user's own --spec-type alone. The speculative framework already runs
    // several drafters in priority order and falls through to the next only when
    // the previous one proposes nothing (see common_speculative_init), so an
    // n-gram drafter in front of MTP is free on the steps it hits - it proposes
    // from the suffix tree with no forward pass at all - and costs nothing but a
    // lookup on the steps it misses, where MTP still runs. Whether that trade
    // actually pays depends on how repetitive the traffic is, which is exactly
    // the kind of thing this fork measures rather than assumes.
    std::string spec_types;                   // "" = not calibrated
    // Draft confidence gate (--spec-draft-p-min): stop drafting once the draft's
    // own top probability falls below this. -1 = not calibrated. Its default of
    // 0 can never stop anything (see common_warn_p_min_disabled), so every step
    // drafts to n_max whatever its confidence - harmless at a shallow depth, a
    // measured multi-x slowdown at a deep one. Calibrated together with depth,
    // since letting a deeper draft pay is the whole point of the gate.
    double      spec_p_min = -1.0;
    // Draft expert placement: -1 = not calibrated, 0 = on the GPU (the draft's
    // default), 1 = on the CPU. The GPU placement costs the target's expert cache
    // the draft's expert bytes (~1.7 GB on Qwen3.8-Flash-Next); whether that trade
    // pays is measured, not assumed.
    int         spec_draft_cpu_moe = -1;
    // GGML_OP_OFFLOAD_MIN_BATCH: MoE batches at or above this go to the GPU (routed
    // experts copied over PCIe from the host mapping), below it they stay on the
    // CPU where the expert cache serves them. -1 = not calibrated (backend default
    // 32, an upstream number never measured on a model bigger than RAM).
    int         op_offload_min_batch = -1;
    // Physical micro-batch (-ub) for prompt processing. -1 = not calibrated (512).
    // A larger one copies each offloaded expert once for more tokens - published:
    // -ub 2048 took a dual-3060 prefill on this model from 36 to 303 tok/s.
    int         n_ubatch = -1;
    // GGML_SCHED_PREFETCH_EXPERTS (already in the scheduler, off by default):
    // overlaps offloaded expert uploads with compute. Measured here before on
    // Gemma-4 at +13.2% pp2048 for 820 MiB of VRAM (see fit.cpp) - VRAM the
    // expert cache would otherwise use, so measured per model rather than
    // switched on. -1 = not calibrated, 0 = off won, 1 = on won.
    int         sched_prefetch_experts = -1;
    // Share of each expert-cache pool reserved for lookahead predictions (the
    // prediction ring, GGML_CUDA_MOE_CACHE_RING_PCT). -1 = not calibrated, 0 =
    // measured and off. Chosen on decode throughput and then answer-checked.
    int         moe_cache_ring_pct = -1;
    // Features that were on, or fixed at a number, with nothing measured behind
    // them. -1 = not calibrated; 0 = measured off/default; >0 = the measured value.
    // neuron_reduce_k already existed as a stored field but was only ever read back
    // from the environment - it is searched now, and 0 means "measured off".
    int         group_admit      = -1;
    int         coverage_evict   = -1;
    int         host_expert_mb   = -1;
    // Whether the MTP draft is served exact experts. Only measured when the draft's
    // experts are CPU-offloaded; on the GPU nothing of the draft reaches the cache.
    int         draft_exact      = -1;
    // Ranking and timing constants that shipped as guesses. Stored as the winning
    // value, or -1 when the sweep did not run. Empty string = nothing measured.
    std::string tuned_constants;
    // What share of the expert-cache budget the MTP draft's own pools may take.
    // -1 = not calibrated (the weight-based split, which gives the draft ~2%
    // purely because it has 3 tensors against the target's 144).
    int         draft_share_pct = -1;
    // How stand-ins are chosen. 0 = by heat (the long-standing default), 1 = by
    // atlas similarity to the missed expert with a fallback to heat, 2 = atlas
    // only, declining the substitution when nothing resident is near enough.
    // -1 = not calibrated.
    int         substitute_atlas = -1;
    // Atlas prewarm on a restored prompt-cache prefix (top_k per tensor).
    // -1 = not calibrated, 0 = measured and not worth it.
    int         atlas_prewarm_k  = -1;
    // The live-trained prerouter (moe-cache.cu: predictor_weights, an online
    // logistic regression on the hidden state that feeds moe_cache_atlas_admit
    // and persists across restarts). It has shipped off and unmeasured behind
    // GGML_CUDA_MOE_CACHE_TRAIN_PREDICTOR since it was written - the same
    // opt-in trap the substitution floor was in. -1 = not calibrated.
    int         train_predictor  = -1;
    // What the trained prerouter is allowed to influence beyond admission.
    // -1 = not calibrated; 0 = no influence.
    double      predictor_evict_w = -1.0;
    double      predictor_sub_w   = -1.0;
    // What the predictor is trained to answer and what it gets to see.
    // -1 = not calibrated.
    // What each resumable stage actually measured, so a resumed run can show
    // the number rather than just the word "resumed". Without these a resumed
    // row renders "RESUMED" where every other row renders "SELECTED - N tok/s",
    // which tells the reader a stage was skipped but not what it decided or
    // what that decision was worth.
    double      tps_offload_min_batch = -1.0;
    double      tps_ubatch            = -1.0;
    double      tps_spec_n_max        = -1.0;
    // KV cache precision. The one consumer nobody had ever bid against: at 64k
    // it holds ~1.4 GiB more than at 4k on this model, which is exactly the
    // margin every other stage has been failing by. Empty = not calibrated.
    std::string kv_type;
    // The fused stand-in picker's signal weights. -1 = not calibrated.
    // The verify pass's own substitution floor. -1 = use the ordinary floor.
    int         sub_min_rank_verify = -2;   // -2 = not calibrated, -1 = "same as generation"
    double      sub_w_atlas       = -1.0;
    double      sub_w_coact       = -1.0;
    int         reduced_share_pct    = -1;  // the arbiter's split: reduced pool's share of the budget
    int         admit_exact_weight   = -1;  // extra admission weight for un-substitutable demand
    int         predictor_admit      = -1;  // may it warm experts (needs a slot)
    int         predictor_next_layer = -1;  // predict the NEXT layer's picks
    int         lookahead_depth  = -1;
    // Stand-in quality bar, in standard deviations below the mean
    // co-activation of the residents examined. This is the continuous form of
    // the wait-or-substitute balance: high = wait for the real expert, low =
    // run with whatever is resident. NaN/unset = not calibrated. Searched
    // separately from the rank floor because they gate different things -
    // rank is about how much the ROUTER wanted this expert, sigma is about
    // how good the available replacement actually is.
    double      substitute_quality_sigma = std::numeric_limits<double>::quiet_NaN();
    // Per-device fit margin (-fitt/--fit-target, MiB). -1 = not calibrated.
    //
    // This is not a minor knob: common_maybe_raise_moe_for_ctx reserves
    // 3 x fit_target of VRAM before deciding placement, and that reservation
    // is what silently RAISES a requested -ncmoe until the context "fits".
    // Measured on this fork (Ornith-1.5-35B-Q4_K_M, RTX 3060 12 GB, -c 4096,
    // requesting -ncmoe 8):
    //
    //     fitt=1024 (default) -> forced to ncmoe 27, 47.32 tok/s,  8951 MiB used
    //     fitt=640            -> forced to ncmoe 24, 50.37 tok/s, 10307 MiB used
    //     fitt=448            -> forced to ncmoe 23, 51.61 tok/s, 10741 MiB used
    //
    // i.e. the default margin left ~3 GiB of a 12 GiB card unused and pushed
    // 4 extra layers of experts onto the CPU, which is where that throughput
    // went. The margin genuinely cannot be dropped to zero (it covers real
    // weight loading and lazy CUDA graph capture that the no-alloc fit probe
    // cannot see - removing it reproduced a hard 4k context collapse), so the
    // right value is neither "3 GiB always" nor "as small as possible": it is
    // hardware- and model-specific, which makes it exactly the kind of thing
    // this calibration exists to measure instead of guess.
    int         fit_target_mb   = -1;
    // Heat-aware intra-expert neuron subsetting (GGML_CUDA_MOE_CACHE_NEURON_REDUCE
    // and friends). -1 = not calibrated / leave to the environment.
    //
    // Recorded here because this mechanism changes the VRAM arithmetic every
    // other value in this struct was measured under: a converted expert costs
    // K rows instead of the tensor's full width (measured 50% for K=256 on
    // Ornith's 576 KiB gate/up experts), so the same hit rate needs less cache
    // VRAM than it used to. A calibration taken with reduction off is not
    // valid for a run with it on, and vice versa - hence storing it rather
    // than letting the two drift silently apart.
    int         neuron_reduce_k         = -1; // > 0 also means "reduction was enabled"
    int         neuron_reduce_budget_mb = -1;
    // How many layers common_moe_calibrate found best kept resident on GPU
    // (the -ngl a real launch should use), -1 = not calibrated / use whatever
    // -ngl was requested. Distinct from n_cpu_moe: this trades GPU-resident
    // dense/attention compute for VRAM the expert cache converts into hit
    // rate, the same "extra CPU offload pays for itself in cache hits"
    // effect n_cpu_moe's own search already knows about (see the ncmoe_hi
    // extension comment above) - just on the layer-residency axis instead of
    // the expert-placement axis. Only searched/applied when the model didn't
    // already fit on GPU as the model's own full layer count (see
    // common_moe_calibrate): a model with room to spare has nothing to trade.
    int         n_gpu_layers            = -1;
    std::string calibrated_at;
    // False when this entry was found by the -ngl-relaxed fallback below, i.e.
    // it was measured at a different -ngl than this launch is using. The
    // fields that depend on layer residency (placement, VRAM sizing) are then
    // not applicable and must be skipped; the ones that don't are still real
    // measurements of this machine and this model. See
    // common_moe_calibration_lookup.
    bool        ngl_exact               = true;
    // 0 for any entry written before gate versioning existed.
    int         gates_version           = 0;
};

// Key identifies "the same launch, calibrated before": GPU signature (name +
// total VRAM per device - not free VRAM, which fluctuates and would make
// the cache miss on unrelated background load), model file identity (path +
// size, cheap proxy for content without hashing multi-GB files), and the
// context shape that determines KV-cache footprint. Any change here (new
// GPU, different model, different -c/--parallel/-ngl) is a real cache miss,
// not staleness - a stale entry is caught separately by the fits-check in
// the lookup path, not by this key.
static std::string common_moe_calibration_key(const char * path_model, const common_params & params) {
    std::string gpu_sig;
    const size_t ndev = ggml_backend_dev_count();
    for (size_t i = 0; i < ndev; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        size_t dfree = 0, dtotal = 0;
        ggml_backend_dev_memory(dev, &dfree, &dtotal);
        // Round to the nearest GiB. Total VRAM was chosen over free precisely
        // because it should not move - but the value the backend reports does
        // drift by a few MiB run to run (observed 11900 and 11909 MiB for the
        // same card, against nvidia-smi's 12288), and at MiB precision that
        // silently rewrites the key and orphans every entry cached under the
        // old one. Qwen re-calibrated on every launch for exactly this reason,
        // whatever -c and --parallel it was given. A GiB bucket is still
        // specific enough to separate real hardware while being immune to the
        // jitter.
        const size_t dtotal_gib = (dtotal + (512ull << 20)) >> 30;
        gpu_sig += string_format("%s:%zuG;", ggml_backend_dev_name(dev), dtotal_gib);
    }
    long long model_size = 0;
    struct stat st;
    if (stat(path_model, &st) == 0) {
        model_size = (long long) st.st_size;
    }
    // Resolve the context the same way every other consumer here does. Left
    // raw, a bare launch (n_ctx == 0, "the context the model was trained
    // with") built a key of c0 that could never match an entry cached at a
    // resolved context, so it re-calibrated from scratch every start and then
    // could not find what it had just written.
    const uint32_t key_ctx = params.n_ctx > 0 ? params.n_ctx : 65536;
    return string_format("%s|%s|%lld|c%u|p%u|ngl%d",
            gpu_sig.c_str(), path_model, model_size, key_ctx, params.n_parallel, params.n_gpu_layers);
}

// The same key with the -ngl component removed. Used to find an entry
// measured on this GPU, this model and this context shape but at a different
// layer residency - see common_moe_calibration_lookup for why that is worth
// finding rather than treating as a plain miss.
static std::string common_moe_calibration_key_prefix(const char * path_model, const common_params & params) {
    // Everything up to and including the model size - i.e. GPU + model
    // identity, without the context/parallel/-ngl shape fields that drift.
    const std::string key = common_moe_calibration_key(path_model, params);
    const size_t cut = key.find("|c");
    return cut == std::string::npos ? key : key.substr(0, cut);
}

static void common_moe_apply_quality_knobs(const common_moe_calibration_entry & cal, const char * path_model) {
    // An entry recorded before the output gates existed says nothing about
    // output. The substitution floor is the clearest case: a gemma-4 entry
    // measured at 12:46 recorded rank 2 purely because it was fastest, and
    // when the fidelity check was added later it scored that same rank 2 at
    // 0.33 - the output diverged from the substitution-free reference after a
    // third of its tokens while still reading as fluent English, which is
    // exactly what the old degeneracy-only guard could not see. Applying it
    // anyway is how a stale measurement quietly degrades every later launch.
    //
    // Deliberately narrow: this skips only the fields the new gates protect.
    // Placement, thread count, cache size and fit margin are throughput
    // measurements that the gates never had an opinion about, so they stay
    // valid and keep being applied from the same entry.
    if (cal.gates_version < COMMON_MOE_CALIBRATION_GATES_VERSION) {
        if (cal.substitute_min_rank >= 0 || !std::isnan(cal.substitute_quality_sigma)) {
            LOG_WRN("%s: ignoring the cached substitution settings - that entry (%s) predates the output "
                    "fidelity and reproducibility checks, so they were never applied to it. Re-run "
                    "--moe-calibrate to measure them; everything else from the entry still applies\n",
                    __func__, cal.calibrated_at.empty() ? "undated" : cal.calibrated_at.c_str());
        }
        return;
    }
    if (cal.substitute_min_rank >= 0 && !getenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK")) {
#if defined(_WIN32)
        _putenv_s("GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK",
                std::to_string(cal.substitute_min_rank).c_str());
#else
        setenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK",
                std::to_string(cal.substitute_min_rank).c_str(), 1);
#endif
        LOG_WRN("%s: using calibrated substitution floor of rank %d\n", __func__, cal.substitute_min_rank);
    }
    // admit_after: how many times an expert must be demanded before it earns a
    // slot. Not a property gate (see the rejected cost-gate/cost-bias attempts
    // elsewhere in this file) - demand->count is still what is compared against
    // this number, calibration only measures what the number should be instead
    // of it being a guess from 2026-08-13 that nothing ever revisited.
    if (cal.admit_after >= 0 && !getenv("GGML_CUDA_MOE_CACHE_ADMIT_AFTER")) {
#if defined(_WIN32)
        _putenv_s("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", std::to_string(cal.admit_after).c_str());
#else
        setenv("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", std::to_string(cal.admit_after).c_str(), 1);
#endif
        LOG_WRN("%s: using calibrated admission threshold of %d\n", __func__, cal.admit_after);
    }
    // readmit_after: the SAME demand->count gate, but for a pool that is
    // already full and must evict something to admit this one - see
    // moe_cache_plan_impl's own admit_after computation (max of the two
    // once free_slots is empty, which is almost always). A much higher bar
    // than admit_after is deliberate: an eviction is not free the way
    // filling a free slot is, so a one-off demand should not be allowed to
    // displace an already-proven resident. Measured directly on this
    // model: loosening it toward admit_after's own value thrashes the pool
    // (hit rate 22% -> 1.3%, THROTTLE=1), while raising it well above the
    // 2026-08-13 constant of 8 instead WON (THROTTLE=32 measured 26-30%
    // hit rate against the default's 22%) - the opposite direction from
    // admit_after, and large enough to clear this machine's run-to-run
    // noise, unlike admit_after's own ladder.
    if (cal.readmit_after >= 0 && !getenv("GGML_CUDA_MOE_CACHE_THROTTLE")) {
#if defined(_WIN32)
        _putenv_s("GGML_CUDA_MOE_CACHE_THROTTLE", std::to_string(cal.readmit_after).c_str());
#else
        setenv("GGML_CUDA_MOE_CACHE_THROTTLE", std::to_string(cal.readmit_after).c_str(), 1);
#endif
        LOG_WRN("%s: using calibrated readmission threshold of %d\n", __func__, cal.readmit_after);
    }
    if (!std::isnan(cal.substitute_quality_sigma) &&
        !getenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_QUALITY_SIGMA")) {
#if defined(_WIN32)
        _putenv_s("GGML_CUDA_MOE_CACHE_SUBSTITUTE_QUALITY_SIGMA",
                std::to_string(cal.substitute_quality_sigma).c_str());
#else
        setenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_QUALITY_SIGMA",
                std::to_string(cal.substitute_quality_sigma).c_str(), 1);
#endif
        LOG_WRN("%s: using calibrated stand-in quality bar of %.1f sigma\n",
                __func__, cal.substitute_quality_sigma);
    }
    auto set_env_int = [](const char * name, long long v) {
#if defined(_WIN32)
        _putenv_s(name, std::to_string(v).c_str());
#else
        setenv(name, std::to_string(v).c_str(), 1);
#endif
    };
    auto set_env_double = [](const char * name, double v) {
        const std::string s = string_format("%.6g", v);
#if defined(_WIN32)
        _putenv_s(name, s.c_str());
#else
        setenv(name, s.c_str(), 1);
#endif
    };
    // Neuron subsetting. 0 is a measured answer ("off won"), and the feature
    // defaults to ON, so it has to be switched off explicitly - the old form only
    // ever turned it on, which meant a measured-off result could not be expressed.
    // The budget is applied only when one was recorded; it is not searched yet.
    if (cal.neuron_reduce_k >= 0 && !getenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE")) {
        if (cal.neuron_reduce_k == 0) {
            set_env_int("GGML_CUDA_MOE_CACHE_NEURON_REDUCE", 0);
            LOG_WRN("%s: neuron subsetting measured slower than off - disabling it\n", __func__);
        } else {
            set_env_int("GGML_CUDA_MOE_CACHE_NEURON_REDUCE", 1);
            set_env_int("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_K", cal.neuron_reduce_k);
            if (cal.neuron_reduce_budget_mb > 0) {
                set_env_int("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_BUDGET_MB", cal.neuron_reduce_budget_mb);
            }
            LOG_WRN("%s: using calibrated neuron subsetting K=%d\n", __func__, cal.neuron_reduce_k);
        }
    }
    // The off-by-default cache features, each recorded only if it beat the
    // incumbent by a real margin. 0 means measured and not worth it, so nothing
    // is set and the runtime default (off) stands.
    if (cal.group_admit == 1 && !getenv("GGML_CUDA_MOE_CACHE_GROUP_ADMIT")) {
        set_env_int("GGML_CUDA_MOE_CACHE_GROUP_ADMIT", 1);
        LOG_WRN("%s: using calibrated group admission (on)\n", __func__);
    }
    if (cal.coverage_evict == 1 && !getenv("GGML_CUDA_MOE_CACHE_COVERAGE_EVICT")) {
        set_env_int("GGML_CUDA_MOE_CACHE_COVERAGE_EVICT", 1);
        LOG_WRN("%s: using calibrated coverage eviction (on)\n", __func__);
    }
    // The tuning constants, stored as the env assignments that beat their defaults.
    // Each is skipped when the operator set it; a value that never won was never
    // recorded, so the shipped default stands on its own.
    if (!cal.tuned_constants.empty()) {
        for (const auto & kv : string_split<std::string>(cal.tuned_constants, ' ')) {
            const auto eq = kv.find('=');
            if (eq == std::string::npos || eq == 0) {
                continue;
            }
            const std::string name = kv.substr(0, eq);
            if (getenv(name.c_str())) {
                continue;
            }
#if defined(_WIN32)
            _putenv_s(name.c_str(), kv.substr(eq + 1).c_str());
#else
            setenv(name.c_str(), kv.substr(eq + 1).c_str(), 1);
#endif
            LOG_WRN("%s: using calibrated %s\n", __func__, kv.c_str());
        }
    }
    if (cal.train_predictor >= 0 && !getenv("GGML_CUDA_MOE_CACHE_TRAIN_PREDICTOR")) {
        set_env_int("GGML_CUDA_MOE_CACHE_TRAIN_PREDICTOR", cal.train_predictor);
        // Point serving at the weights calibration trained, and let it keep
        // learning from there - NOT frozen. Freezing is a calibration-only
        // device, to keep candidate comparisons fair; a served model has no
        // comparison to protect and every reason to keep adapting to the
        // traffic it actually gets. Calibration's corpus is a starting point,
        // not the target distribution.
        if (cal.train_predictor > 0 && !getenv("GGML_CUDA_MOE_CACHE_TRAIN_STATE_FILE")) {
            const std::string cache_dir = fs_get_cache_directory();
            const std::string base = path_model ? std::string(path_model) : std::string("model");
            const size_t slash = base.find_last_of("/\\");
            const std::string state = cache_dir + "predictor-" +
                (slash == std::string::npos ? base : base.substr(slash + 1)) + ".bin";
#if defined(_WIN32)
            _putenv_s("GGML_CUDA_MOE_CACHE_TRAIN_STATE_FILE", state.c_str());
#else
            setenv("GGML_CUDA_MOE_CACHE_TRAIN_STATE_FILE", state.c_str(), 1);
#endif
            LOG_WRN("%s: prerouter state: %s (loaded, and kept learning)\n", __func__, state.c_str());
            // The input shape travels with the weights here too - serving that
            // built a different vector than the weights were fitted against
            // would fail the dimension check and silently run untrained.
            if (cal.predictor_next_layer >= 0) {
                set_env_int("GGML_CUDA_MOE_CACHE_TRAIN_NEXT_LAYER", cal.predictor_next_layer);
            }
        }
        LOG_WRN("%s: live-trained prerouter %s (calibrated)\n", __func__,
                cal.train_predictor ? "on" : "off");
    }

    if (cal.predictor_next_layer >= 0 && !getenv("GGML_CUDA_MOE_CACHE_TRAIN_NEXT_LAYER")) {
        set_env_int("GGML_CUDA_MOE_CACHE_TRAIN_NEXT_LAYER", cal.predictor_next_layer);
        LOG_WRN("%s: prerouter target: %s (calibrated)\n", __func__,
                cal.predictor_next_layer ? "the next layer's picks" : "this layer's picks");
    }
    if (cal.reduced_share_pct >= 0 && !getenv("GGML_CUDA_MOE_CACHE_REDUCED_SHARE_PCT")) {
        set_env_int("GGML_CUDA_MOE_CACHE_REDUCED_SHARE_PCT", cal.reduced_share_pct);
        LOG_WRN("%s: budget split - reduced pool gets %d%% (calibrated)\n", __func__, cal.reduced_share_pct);
    }
    if (cal.admit_exact_weight >= 0 && !getenv("GGML_CUDA_MOE_CACHE_ADMIT_EXACT_WEIGHT")) {
        set_env_int("GGML_CUDA_MOE_CACHE_ADMIT_EXACT_WEIGHT", cal.admit_exact_weight);
        LOG_WRN("%s: admission weights un-substitutable demand %dx (calibrated)\n",
                __func__, cal.admit_exact_weight);
    }
    if (cal.predictor_admit >= 0 && !getenv("GGML_CUDA_MOE_CACHE_PREDICTOR_ADMIT")) {
        set_env_int("GGML_CUDA_MOE_CACHE_PREDICTOR_ADMIT", cal.predictor_admit);
        LOG_WRN("%s: prerouter warming %s (calibrated)\n", __func__, cal.predictor_admit ? "on" : "off");
    }
    if (cal.predictor_evict_w >= 0.0 && !getenv("GGML_CUDA_MOE_CACHE_PREDICTOR_EVICT_WEIGHT")) {
        set_env_double("GGML_CUDA_MOE_CACHE_PREDICTOR_EVICT_WEIGHT", cal.predictor_evict_w);
        LOG_WRN("%s: prerouter eviction protection at weight %.2f (calibrated)\n", __func__, cal.predictor_evict_w);
    }
    if (cal.predictor_sub_w >= 0.0 && !getenv("GGML_CUDA_MOE_CACHE_PREDICTOR_SUB_WEIGHT")) {
        set_env_double("GGML_CUDA_MOE_CACHE_PREDICTOR_SUB_WEIGHT", cal.predictor_sub_w);
        LOG_WRN("%s: prerouter substitution tiebreak at weight %.2f (calibrated)\n", __func__, cal.predictor_sub_w);
    }

    if (cal.atlas_prewarm_k > 0 && !getenv("LLAMA_PROMPT_CACHE_MOE_PREWARM")) {
        set_env_int("LLAMA_PROMPT_CACHE_MOE_PREWARM", 1);
        set_env_int("LLAMA_PROMPT_CACHE_MOE_PREWARM_K", cal.atlas_prewarm_k);
        LOG_WRN("%s: using calibrated atlas prewarm, top_k %d per tensor\n", __func__, cal.atlas_prewarm_k);
    }
    // Mode 3 is not an atlas mode - it is the per-token-rank picker, which
    // needs the atlas picker off and the rank picker on. Writing the mode
    // number straight into SUBSTITUTE_ATLAS would set it to 3, which the cache
    // reads as "atlas on, some unknown variant", and the rank picker - the one
    // the stage actually chose - would never run.
    if (cal.sub_min_rank_verify > -2 && !getenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK_VERIFY")) {
        set_env_int("GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK_VERIFY", cal.sub_min_rank_verify);
        LOG_WRN("%s: verify passes substitute from rank %d%s (calibrated)\n", __func__,
                cal.sub_min_rank_verify,
                cal.sub_min_rank_verify < 0 ? " (same floor as generation)" : "");
    }
    if (cal.sub_w_atlas >= 0.0 && !getenv("GGML_CUDA_MOE_CACHE_SUB_W_ATLAS")) {
        set_env_double("GGML_CUDA_MOE_CACHE_SUB_W_ATLAS", cal.sub_w_atlas);
        set_env_double("GGML_CUDA_MOE_CACHE_SUB_W_COACT", cal.sub_w_coact);
        LOG_WRN("%s: stand-in fusion weights atlas %.2f / co-activation %.2f (calibrated)\n",
                __func__, cal.sub_w_atlas, cal.sub_w_coact);
    }
    if (cal.substitute_atlas > 0 && !getenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_ATLAS")) {
        if (cal.substitute_atlas == 4) {
            set_env_int("GGML_CUDA_MOE_CACHE_SUBSTITUTE_ATLAS", 4);
            LOG_WRN("%s: choosing stand-ins by the fused picker - this token's router ranking first, "
                    "then atlas resemblance and co-activation together (calibrated)\n", __func__);
        } else if (cal.substitute_atlas == 3) {
            set_env_int("GGML_CUDA_MOE_CACHE_SUBSTITUTE_ATLAS", 0);
            set_env_int("GGML_CUDA_MOE_CACHE_SUBSTITUTE_STRICT_RANK", 1);
            LOG_WRN("%s: choosing stand-ins by this token's router rank (calibrated)\n", __func__);
        } else {
            set_env_int("GGML_CUDA_MOE_CACHE_SUBSTITUTE_ATLAS", cal.substitute_atlas);
            LOG_WRN("%s: choosing stand-ins by atlas similarity (mode %d)\n", __func__, cal.substitute_atlas);
        }
    }
    if (cal.draft_share_pct > 0 && !getenv("GGML_CUDA_MOE_CACHE_DRAFT_SHARE_PCT")) {
        set_env_int("GGML_CUDA_MOE_CACHE_DRAFT_SHARE_PCT", cal.draft_share_pct);
        LOG_WRN("%s: using calibrated draft cache share of %d%%\n", __func__, cal.draft_share_pct);
    }
    if (cal.draft_exact == 0 && !getenv("GGML_CUDA_MOE_CACHE_DRAFT_EXACT")) {
        set_env_int("GGML_CUDA_MOE_CACHE_DRAFT_EXACT", 0);
        LOG_WRN("%s: stand-ins measured faster than exact experts in the draft - allowing them\n", __func__);
    }
    if (cal.host_expert_mb > 0 && !getenv("GGML_CUDA_MOE_CACHE_HOST_MB")) {
        set_env_int("GGML_CUDA_MOE_CACHE_HOST_MB", cal.host_expert_mb);
        LOG_WRN("%s: using calibrated host hot-expert buffer of %d MiB\n", __func__, cal.host_expert_mb);
    }
}

static std::string common_moe_calibration_cache_path() {
    return fs_get_cache_directory() + "moe-calibration.json";
}

static bool common_moe_calibration_lookup(
        const char * path_model, const common_params & params, common_moe_calibration_entry & out) {
    // A calibration candidate never reads the cache. See the long note at the
    // candidate launcher (common_moe_bench_candidate_server): the parent passes
    // every knob it is measuring explicitly, and where it deliberately passes
    // nothing - because it measured that no value fits - the child must keep its
    // own default rather than adopt a previous run's answer. Guarded here rather
    // than at one caller because the cached entry also carries placement, fit
    // margin and the quality knobs, each applied from its own site.
    if (getenv("LLAMA_MOE_CALIBRATION_CHILD")) {
        return false;
    }
    std::ifstream f(common_moe_calibration_cache_path());
    static const bool trace_lookup = getenv("LLAMA_MOE_CALIB_TRACE") != nullptr;
    if (trace_lookup) {
        // fprintf, not LOG_WRN: the logging macros are gated on a verbosity
        // threshold that is not necessarily set yet this early in startup, and
        // a diagnostic that can be silently swallowed is worse than none - it
        // reads as "this code did not run" when it did.
        fprintf(stderr, "[calib-trace] path='%s' good=%d\n",
                common_moe_calibration_cache_path().c_str(), (int) f.good());
        fflush(stderr);
    }
    if (!f.good()) {
        return false;
    }
    try {
        nlohmann::json j;
        f >> j;
        const std::string key = common_moe_calibration_key(path_model, params);
        bool exact = j.contains(key);
        if (!exact) {
            // Print the key on a miss. Three separate theories about why a
            // cached entry was not being found (context, parallelism, GPU
            // signature) were each wrong in turn, and every one of them would
            // have been settled immediately by seeing the string itself.
            if (trace_lookup) {
                fprintf(stderr, "[calib-trace] MISS key='%s' (cache has %zu entries)\n",
                        key.c_str(), (size_t) j.size());
                fflush(stderr);
            }
        }
        std::string use_key = key;
        if (!exact) {
            // Relaxed retry: same GPU, model and context shape, different
            // -ngl. Pinning -ngl used to discard the entire entry, because
            // -ngl is part of the key - so `-ngl 20` on a box calibrated at
            // the default threw away the measured thread count, substitution
            // floor and neuron-reduce settings along with the placement, and
            // silently fell back to runtime defaults for all of them. Only
            // some of those values depend on layer residency. Find the entry
            // anyway and let the apply path take the parts that still hold
            // (see ngl_exact). Prefer the fastest such entry when several
            // -ngl values have been calibrated - they were all measured on
            // this same machine and model.
            // Match on GPU + model + size only, ignoring the context,
            // parallelism and -ngl fields. Those three have each drifted in
            // practice - the GPU signature moved 11900 vs 11909 MiB for one
            // card, calibration rewrites n_ctx and n_parallel as it adapts, and
            // -ngl is pinned by the user - and any one of them changing orphans
            // the entry, so the run re-calibrates and then cannot find what it
            // just wrote. An entry for this model on this machine is worth more
            // than a perfect key match: the apply path already refuses anything
            // that does not transfer (see ngl_exact), so a relaxed hit costs
            // nothing beyond the values that legitimately carry over.
            const std::string prefix = common_moe_calibration_key_prefix(path_model, params);
            const size_t model_at = prefix.find('|');
            const std::string model_part = model_at == std::string::npos
                    ? prefix : prefix.substr(model_at);
            double best = -1.0;
            for (const auto & item : j.items()) {
                const size_t item_at = item.key().find('|');
                const std::string item_model = item_at == std::string::npos
                        ? item.key() : item.key().substr(item_at);
                const bool same_prefix = item.key().compare(0, prefix.size(), prefix) == 0;
                // model_part starts at the first '|' and covers path + size, so
                // this pins the model file identity while letting the GPU
                // signature and the trailing shape fields vary.
                const bool same_model = item_model.compare(0, model_part.size(), model_part) == 0;
                if (!same_prefix && !same_model) {
                    continue;
                }
                const double tps = item.value().value("tok_per_sec", 0.0);
                if (tps > best) {
                    best    = tps;
                    use_key = item.key();
                }
            }
            if (best < 0.0) {
                return false;
            }
        }
        const auto & e = j.at(use_key);
        out.ngl_exact = exact;
        out.n_cpu_moe       = e.value("n_cpu_moe", 0);
        out.n_threads       = e.value("n_threads", -1);
        out.n_threads_batch = e.value("n_threads_batch", -1);
        out.spec_n_max      = e.value("spec_n_max", -1);
        out.concurrency     = e.value("concurrency", 1);
        out.tok_per_sec     = e.value("tok_per_sec", 0.0);
        out.moe_cache_mb    = e.value("moe_cache_mb", -1);
        out.substitute_min_rank = e.value("substitute_min_rank", -1);
        out.admit_after         = e.value("admit_after", -1);
        out.readmit_after       = e.value("readmit_after", -1);
        out.spec_prob_accept    = e.value("spec_prob_accept", -1);
        out.spec_types          = e.value("spec_types", std::string());
        out.spec_p_min          = e.value("spec_p_min", -1.0);
        out.defaults_version    = e.value("defaults_version", 0);
        if (out.defaults_version < COMMON_MOE_DEFAULTS_VERSION) {
            out.spec_n_max = -1;          // launch values, not measurements: let the defaults apply
            out.spec_p_min = -1.0;
            out.spec_prob_accept = -1;
        }
        out.spec_draft_cpu_moe  = e.value("spec_draft_cpu_moe", -1);
        out.op_offload_min_batch = e.value("op_offload_min_batch", -1);
        out.n_ubatch             = e.value("n_ubatch", -1);
        out.sched_prefetch_experts = e.value("sched_prefetch_experts", -1);
        out.moe_cache_ring_pct     = e.value("moe_cache_ring_pct", -1);
        out.group_admit            = e.value("group_admit", -1);
        out.coverage_evict         = e.value("coverage_evict", -1);
        out.host_expert_mb         = e.value("host_expert_mb", -1);
        out.draft_exact            = e.value("draft_exact", -1);
        out.tuned_constants        = e.value("tuned_constants", std::string());
        out.draft_share_pct        = e.value("draft_share_pct", -1);
        out.substitute_atlas       = e.value("substitute_atlas", -1);
        out.atlas_prewarm_k        = e.value("atlas_prewarm_k", -1);
        out.train_predictor        = e.value("train_predictor", -1);
        out.predictor_evict_w      = e.value("predictor_evict_w", -1.0);
        out.predictor_sub_w        = e.value("predictor_sub_w", -1.0);
        out.tps_offload_min_batch  = e.value("tps_offload_min_batch", -1.0);
        out.tps_ubatch             = e.value("tps_ubatch", -1.0);
        out.tps_spec_n_max         = e.value("tps_spec_n_max", -1.0);
        out.kv_type                = e.value("kv_type", std::string());
        out.sub_min_rank_verify    = e.value("sub_min_rank_verify", -2);
        out.sub_w_atlas            = e.value("sub_w_atlas", -1.0);
        out.sub_w_coact            = e.value("sub_w_coact", -1.0);
        out.reduced_share_pct      = e.value("reduced_share_pct", -1);
        out.admit_exact_weight     = e.value("admit_exact_weight", -1);
        out.predictor_admit        = e.value("predictor_admit", -1);
        out.predictor_next_layer   = e.value("predictor_next_layer", -1);
        out.lookahead_depth        = e.value("lookahead_depth", -1);
        // Saved as NaN when no stand-in bar was measured, and nlohmann writes NaN
        // as null. value() only falls back to its default when the key is absent -
        // on a present null it throws, the catch below turns that into a miss, and
        // every such entry could never be read back: each launch recalibrated.
        {
            const auto it = e.find("substitute_quality_sigma");
            out.substitute_quality_sigma = it != e.end() && it->is_number()
                    ? it->get<double>() : std::numeric_limits<double>::quiet_NaN();
        }
        // Absent in entries written before these were calibrated - the
        // value() defaults keep such an entry loadable rather than making
        // it a hard cache miss, since the fields it DOES carry are still
        // valid measurements.
        out.fit_target_mb           = e.value("fit_target_mb", -1);
        out.neuron_reduce_k         = e.value("neuron_reduce_k", -1);
        out.neuron_reduce_budget_mb = e.value("neuron_reduce_budget_mb", -1);
        out.n_gpu_layers            = e.value("n_gpu_layers", -1);
        out.calibrated_at   = e.value("calibrated_at", std::string());
        out.gates_version   = e.value("gates_version", 0);
        return true;
    } catch (const std::exception & err) {
        // A read that throws used to vanish here as a plain miss, which looked
        // exactly like "no entry" and sent the launch into recalibrating.
        fprintf(stderr, "[calib] cached calibration entry could not be read, treating it as a miss: %s\n", err.what());
        fflush(stderr);
        return false;
    }
}

static void common_moe_calibration_save(
        const char * path_model, const common_params & params, const common_moe_calibration_entry & entry) {
    const std::string path = common_moe_calibration_cache_path();
    nlohmann::json j = nlohmann::json::object();
    {
        std::ifstream f(path);
        if (f.good()) {
            try {
                f >> j;
            } catch (const std::exception &) {
                j = nlohmann::json::object();
            }
        }
    }
    const std::string key = common_moe_calibration_key(path_model, params);
    j[key] = {
        {"n_cpu_moe",       entry.n_cpu_moe},
        {"n_threads",       entry.n_threads},
        {"n_threads_batch", entry.n_threads_batch},
        {"spec_n_max",      entry.spec_n_max},
        {"concurrency",     entry.concurrency},
        {"tok_per_sec",     entry.tok_per_sec},
        {"moe_cache_mb",    entry.moe_cache_mb},
        {"substitute_min_rank", entry.substitute_min_rank},
        {"admit_after", entry.admit_after},
        {"readmit_after", entry.readmit_after},
        {"spec_prob_accept", entry.spec_prob_accept},
        {"spec_types", entry.spec_types},
        {"spec_p_min", entry.spec_p_min},
        {"spec_draft_cpu_moe", entry.spec_draft_cpu_moe},
        {"op_offload_min_batch", entry.op_offload_min_batch},
        {"n_ubatch", entry.n_ubatch},
        {"sched_prefetch_experts", entry.sched_prefetch_experts},
        {"moe_cache_ring_pct", entry.moe_cache_ring_pct},
        {"group_admit", entry.group_admit},
        {"coverage_evict", entry.coverage_evict},
        {"host_expert_mb", entry.host_expert_mb},
        {"draft_exact", entry.draft_exact},
        {"tuned_constants", entry.tuned_constants},
        {"draft_share_pct", entry.draft_share_pct},
        {"substitute_atlas", entry.substitute_atlas},
        {"atlas_prewarm_k", entry.atlas_prewarm_k},
        {"train_predictor", entry.train_predictor},
        {"predictor_evict_w", entry.predictor_evict_w},
        {"predictor_sub_w", entry.predictor_sub_w},
        {"tps_offload_min_batch", entry.tps_offload_min_batch},
        {"tps_ubatch", entry.tps_ubatch},
        {"tps_spec_n_max", entry.tps_spec_n_max},
        {"kv_type", entry.kv_type},
        {"sub_min_rank_verify", entry.sub_min_rank_verify},
        {"sub_w_atlas", entry.sub_w_atlas},
        {"sub_w_coact", entry.sub_w_coact},
        {"reduced_share_pct", entry.reduced_share_pct},
        {"admit_exact_weight", entry.admit_exact_weight},
        {"predictor_admit", entry.predictor_admit},
        {"predictor_next_layer", entry.predictor_next_layer},
        {"lookahead_depth", entry.lookahead_depth},
        {"substitute_quality_sigma", entry.substitute_quality_sigma},
        {"gates_version", COMMON_MOE_CALIBRATION_GATES_VERSION},
        {"defaults_version", COMMON_MOE_DEFAULTS_VERSION},
        {"fit_target_mb",           entry.fit_target_mb},
        {"neuron_reduce_k",         entry.neuron_reduce_k},
        {"neuron_reduce_budget_mb", entry.neuron_reduce_budget_mb},
        {"n_gpu_layers",    entry.n_gpu_layers},
        {"calibrated_at",   entry.calibrated_at},
    };
    fs_create_directory_with_parents(fs_get_cache_directory());
    std::ofstream out(path);
    out << j.dump(2);
}

bool common_moe_should_auto_calibrate(common_params & params) {
    // Same rule as common_maybe_autoplace_moe_cpu(): an explicit -ncmoe (or
    // any other explicit tensor-buffer-type override) is a deliberate user
    // choice and auto-calibration must never second-guess it.
    for (const auto & o : params.tensor_buft_overrides) {
        if (o.pattern != nullptr) {
            return false;
        }
    }

    const char * path_model = params.model.path.c_str();
    common_moe_calibration_entry cached;
    if (common_moe_calibration_lookup(path_model, params, cached)) {
        return false; // already have a cached answer for this combination
    }

    // Cheap no-alloc probe (GGUF header + device memory, no tensor data
    // loaded) - only trigger the (multi-minute) automatic calibration run
    // for models that actually need MoE CPU-offload placement decided.
    // Dense models, and MoE models that already fit as configured, have
    // nothing for --moe-calibrate to usefully tune.
    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);
    common_moe_fit_probe_result probe = common_moe_find_safe_layers(path_model, mparams, cparams);
    if (!probe.is_moe || probe.already_fits) {
        return false;
    }
    return true;
}

// Golden-section search over integers in [lo, hi] for the argmax of a
// unimodal (single-peak) function - the shape our own -ncmoe and
// spec-draft-n-max sweeps actually showed empirically (a real interior
// peak, not monotonic), not a guess. O(log(hi-lo)) evaluations instead of
// O(hi-lo) for a full grid - matters because each evaluation here is a
// real model load, not a cheap probe. Memoizes every point actually
// measured (golden-section revisits nearby points) and returns it via
// `trace` for logging/debugging.
template <typename F>
static int common_golden_section_search_max(
        int lo, int hi, F && measure, std::map<int, double> & trace) {
    if (lo >= hi) {
        if (trace.find(lo) == trace.end()) {
            trace[lo] = measure(lo);
        }
        return lo;
    }
    auto measured = [&](int x) -> double {
        auto it = trace.find(x);
        if (it != trace.end()) {
            return it->second;
        }
        const double v = measure(x);
        trace[x] = v;
        return v;
    };

    const double gr = 0.6180339887498949; // 1/phi
    int x1 = lo + (int) std::lround((1.0 - gr) * (hi - lo));
    int x2 = lo + (int) std::lround(gr * (hi - lo));
    x1 = std::max(lo, std::min(hi, x1));
    x2 = std::max(lo, std::min(hi, x2));
    double f1 = measured(x1);
    double f2 = measured(x2);

    while (hi - lo > 2) {
        if (f1 < f2) {
            lo = x1;
            x1 = x2; f1 = f2;
            x2 = std::max(lo, std::min(hi, lo + (int) std::lround(gr * (hi - lo))));
            if (x2 == x1) {
                break;
            }
            f2 = measured(x2);
        } else {
            hi = x2;
            x2 = x1; f2 = f1;
            x1 = std::max(lo, std::min(hi, lo + (int) std::lround((1.0 - gr) * (hi - lo))));
            if (x1 == x2) {
                break;
            }
            f1 = measured(x1);
        }
    }
    // small remaining range - just check every point directly, cheap now
    int best = lo;
    double best_val = measured(lo);
    for (int x = lo + 1; x <= hi; x++) {
        const double v = measured(x);
        if (v > best_val) {
            best_val = v;
            best = x;
        }
    }
    return best;
}

// Spawns a real llama-server subprocess with the given placement/n_max/
// thread count, waits for it to become healthy, sends real chat-completion
// requests (averaged over a couple of representative prompts), and returns
// measured decode tok/s (already net-of-rejection when MTP is active - see
// the MTP section of docs/moe-cache-colibri-notes.md: predicted_per_second
// is tokens_predicted/elapsed, and tokens_predicted only counts tokens that
// survived speculative verification, so rejected draft attempts are
// already excluded from the numerator while their wasted compute is
// captured in the denominator).
//
// This is the *only* benchmark path in this file - an earlier in-process
// version (direct llama_decode() calls, bypassing the HTTP server) existed
// for the non-MTP -ncmoe search, on the theory that avoiding a real
// draft/verify reimplementation for MTP was the main correctness risk.
// That was true but incomplete: the in-process version also fed the model
// a raw, un-chat-templated prompt, and this model family (instruction/
// reasoning-tuned) produces degenerate, highly-repetitive output on
// unformatted prompts *independent of MTP, moe-cache, or placement* -
// confirmed directly by testing (see docs). Degenerate repetitive text is
// trivially predictable, which would have silently inflated every
// measurement taken with it: MTP acceptance rate (repeating the same token
// is an easy guess for the draft), and MoE-cache hit rate (repetitive text
// routes to a narrow, unrealistic set of experts). Spawning the real
// server and using /v1/chat/completions (which applies the GGUF's own
// chat template server-side, exactly like a real client would) avoids
// both the draft/verify-reimplementation risk and the prompt-formatting
// risk in one path, at the cost of a real subprocess per candidate.
// Wraps a string in single quotes for safe use as one shell argument,
// escaping any embedded single quotes via the standard '\'' trick (close
// the quote, emit an escaped literal quote, reopen). Without this, a
// prompt containing an apostrophe (e.g. "Newton's second law" - a real
// prompt in this file's own probe pool, found the hard way when it broke
// the concurrency-benchmark path with a shell syntax error) corrupts the
// surrounding --data-binary '...' argument and the request silently
// becomes a shell parse error instead of an HTTP call.
static std::string common_shell_quote(const std::string & s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

// Fetches one /v1/chat/completions response and returns (predicted_n, predicted_ms),
// or (-1, 0) on any failure - used by both the solo and concurrent benchmark paths below.
// Degeneracy score for one candidate's generated text, in [0,1] where higher
// is worse. Calibration maximises tok/s, and without this it would happily
// pick a configuration that is fast BECAUSE it is broken: forcing
// substitution on for every router pick measured 11.95 tok/s (vs ~0.7
// otherwise) while emitting word salad ("a question to how a steps
// explaining on the how quicksort, e shouldicym is"), and an earlier
// variant produced verbatim repetition loops. Both were only caught by
// reading the output by hand - exactly the kind of thing a throughput-only
// objective cannot see.
//
// Two cheap checks, no extra model passes and no logprobs needed, each
// aimed at one of the two failure modes actually observed:
//   - repeated 4-grams (a loop repeats long spans verbatim)
//   - single-word dominance (both failure modes degenerate into hammering a
//     few function words - "to how a ... to how a")
// Deliberately not a language-quality judgement, just a degeneracy floor:
// it has to reject output that is obviously broken, not rank prose.
static double common_moe_degeneracy_score(const std::string & text) {
    std::vector<std::string> words;
    {
        std::istringstream iss(text);
        std::string w;
        while (iss >> w) {
            words.push_back(w);
        }
    }
    if (words.size() < 16) {
        return 0.0; // too short to judge - don't reject on noise
    }
    std::unordered_map<std::string, int> word_counts;
    for (const auto & w : words) {
        word_counts[w]++;
    }
    int max_word = 0;
    for (const auto & [w, c] : word_counts) {
        max_word = std::max(max_word, c);
    }
    const double max_word_share = (double) max_word / (double) words.size();

    std::unordered_map<std::string, int> gram_counts;
    size_t n_grams = 0;
    for (size_t i = 0; i + 4 <= words.size(); i++) {
        gram_counts[words[i] + " " + words[i+1] + " " + words[i+2] + " " + words[i+3]]++;
        n_grams++;
    }
    const double distinct_gram_ratio = n_grams > 0 ? (double) gram_counts.size() / (double) n_grams : 1.0;

    // Worst of the two, each normalised so ~1.0 means clearly degenerate.
    const double rep_score  = 1.0 - distinct_gram_ratio;      // 0 = all 4-grams unique
    const double dom_score  = max_word_share;                  // 0 = perfectly varied vocabulary
    return std::max(rep_score, dom_score);
}

// At or above this score a candidate is rejected outright. 0.45 sits well
// clear of healthy prose (normal answers score low on both components - long
// verbatim 4-gram repeats are rare, and no single word takes ~half the
// output) while catching both observed failures, which are not marginal:
// they repeat spans verbatim or hammer one word for a large fraction of the
// text. Override with GGML_MOE_DEGENERACY_REJECT if a model legitimately
// trips it (a format-constrained generator emitting highly repetitive
// structure, say).
static double common_moe_degeneracy_reject_threshold() {
    static const double v = [] {
        const char * e = getenv("GGML_MOE_DEGENERACY_REJECT");
        const double x = e ? atof(e) : 0.45;
        return x > 0.0 ? x : 0.45;
    }();
    return v;
}

// Agreement between a candidate's greedy output and a substitution-free
// reference's, as the fraction of the reference that the candidate reproduces
// before the two part ways.
//
// This exists because the degeneracy guard above cannot see the failure mode
// substitution actually causes. That guard is a lexical health check: it
// catches verbatim 4-gram repeats and single-word hammering. Serving expert
// A's weights where the router asked for B does not produce either - it
// produces fluent, varied, grammatical text that says the wrong thing, which
// scores near zero on both components and passes the gate untouched. A
// throughput ladder gated only on degeneracy will therefore happily pick a
// rung that hallucinates, which is exactly what was observed on gemma-4 after
// calibration settled on rank 2.
//
// Greedy decoding makes divergence sticky: once two runs pick different
// tokens the contexts differ and they rarely reconverge. So the length of the
// shared prefix is the natural measure of "at what point did substitution
// change the model's mind", and it needs no notion of what a good answer is -
// only that this configuration says what the unsubstituted one would have
// said. Compared word-wise rather than byte-wise so that whitespace and
// tokenizer boundaries don't register as disagreement.
static double common_moe_output_fidelity(const std::string & reference, const std::string & candidate) {
    auto split = [](const std::string & s) {
        std::vector<std::string> out;
        std::istringstream iss(s);
        std::string w;
        while (iss >> w) {
            out.push_back(w);
        }
        return out;
    };
    const std::vector<std::string> a = split(reference);
    const std::vector<std::string> b = split(candidate);
    if (a.empty()) {
        return 1.0; // no reference to disagree with - don't reject on noise
    }
    size_t common = 0;
    while (common < a.size() && common < b.size() && a[common] == b[common]) {
        common++;
    }
    return (double) common / (double) a.size();
}

struct common_moe_bench_result {
    double predicted_n  = -1.0;
    double predicted_ms = 0.0;
    double degeneracy   = 0.0;
    // What the model actually emitted, for the fidelity comparison in the
    // substitution ladder. See common_moe_output_fidelity().
    std::string text;
    // Speculative decoding, when a draft is attached: how many tokens were
    // drafted and how many the target accepted. The speed multiplier is made
    // of this, so a candidate that wins on tok/s without it cannot say why.
    // -1 when the server reported no drafting.
    double draft_n          = -1.0;
    double draft_n_accepted = -1.0;
    // The prompt pass, for the offload-threshold stage - which batch path a
    // prompt chunk takes is what that threshold decides, and decode never
    // exercises it.
    double prompt_n  = -1.0;
    double prompt_ms = 0.0;
    // The answer channel alone: `content`, i.e. what follows </think>. `text`
    // joins content and reasoning for the degeneracy and fidelity checks, which
    // is right for them - but a correctness check that reads it counts an answer
    // that only ever appeared inside the reasoning, with nothing after </think>,
    // as correct. That is exactly the failure it exists to catch.
    std::string answer;
};

// The draft acceptance of the most recent candidate, for the stages that measure
// MTP levers to log beside their tok/s. Written by the candidate spawner, read
// by the stage that called it; calibration runs one candidate at a time.
static double g_moe_last_draft_n          = -1.0;
static double g_moe_last_draft_n_accepted = -1.0;

static std::string common_moe_last_acceptance_str() {
    if (g_moe_last_draft_n <= 0.0 || g_moe_last_draft_n_accepted < 0.0) {
        return std::string();
    }
    return string_format(", draft acceptance %.2f (%.0f of %.0f)",
            g_moe_last_draft_n_accepted / g_moe_last_draft_n, g_moe_last_draft_n_accepted, g_moe_last_draft_n);
}

static common_moe_bench_result common_moe_bench_one_request_full(int port, const char * prompt, int n_predict, int seed = 1234, bool greedy = false, bool cache_prompt = true);

static std::pair<double, double> common_moe_bench_one_request(int port, const char * prompt, int n_predict) {
    const auto r = common_moe_bench_one_request_full(port, prompt, n_predict);
    return {r.predicted_n, r.predicted_ms};
}

static common_moe_bench_result common_moe_bench_one_request_full(int port, const char * prompt, int n_predict, int seed, bool greedy, bool cache_prompt) {
    nlohmann::json req = {
        {"messages", nlohmann::json::array({
            {{"role", "user"}, {"content", prompt}}
        })},
        {"max_tokens", n_predict},
        // Seeded, but sampling params deliberately left alone: whatever the
        // model ships as its recommended sampling (unsloth's GGUF metadata for
        // these quants, or the user's own flags) is the condition it will
        // actually be served under, so that is the condition the ladder has to
        // measure. Pinning the seed is enough to make two runs of the SAME
        // config comparable, which is all the fidelity check needs - and it
        // avoids tuning the substitution floor against a greedy regime nobody
        // runs. Residual run-to-run wobble is measured, not assumed: see the
        // reference self-fidelity noise floor in the substitution ladder.
        {"seed", seed},
        // The model card's thinking-mode sampling in full. min_p in particular was
        // being left at llama.cpp's 0.05 rather than the documented 0.0 - an extra
        // tail cut on top of top_p, under which every probe this session was
        // measured. The server-side launch flags set the same values; sending them
        // per request too means a probe is unaffected by what the candidate was
        // launched with.
        {"min_p", 0.0},
        {"presence_penalty", 0.0},
        {"repeat_penalty", 1.0},
    };
    if (!cache_prompt) {
        req["cache_prompt"] = false; // time a real prefill, not a prefix-cache hit
    }
    if (greedy) {
        // Only for the determinism check: pinning the sampler to argmax takes
        // it out of the picture entirely, so any variation left is the forward
        // pass. Deliberately not used for the throughput or fidelity probes,
        // which have to measure the sampling the model actually ships with.
        req["temperature"] = 0.0;
        req["top_k"] = 1;
    }
    const std::string req_body = req.dump();
    // Built as a string, not into a fixed buffer. It was char[4096], so any
    // request body longer than ~4 KB was silently cut mid-JSON and the server
    // answered a malformed request with no timings: the prompt micro-batch stage's
    // ~8000-character probe failed every candidate in seconds, and with no winner
    // the expert-prefetch candidate after it never ran.
    const std::string req_cmd = string_format(
        "curl -s http://127.0.0.1:%d/v1/chat/completions -H 'Content-Type: application/json' --data-binary %s",
        port, common_shell_quote(req_body).c_str());
    FILE * rp = popen(req_cmd.c_str(), "r");
    if (!rp) {
        return {};
    }
    std::string body;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), rp)) > 0) {
        body.append(buf, n);
    }
    pclose(rp);
    try {
        auto j = nlohmann::json::parse(body);
        if (j.contains("timings") && j["timings"].contains("predicted_n")) {
            common_moe_bench_result r;
            r.predicted_n  = j["timings"]["predicted_n"].get<double>();
            r.predicted_ms = j["timings"]["predicted_ms"].get<double>();
            if (j["timings"].contains("prompt_n") && j["timings"].contains("prompt_ms")) {
                r.prompt_n  = j["timings"]["prompt_n"].get<double>();
                r.prompt_ms = j["timings"]["prompt_ms"].get<double>();
            }
            if (j["timings"].contains("draft_n") && j["timings"].contains("draft_n_accepted")) {
                r.draft_n          = j["timings"]["draft_n"].get<double>();
                r.draft_n_accepted = j["timings"]["draft_n_accepted"].get<double>();
            }
            // Score whatever the model actually emitted. Reasoning models put
            // most of a short probe's tokens in reasoning_content rather than
            // content, and a degenerate run is degenerate in either channel -
            // so both are scored and the worse one wins.
            if (j.contains("choices") && !j["choices"].empty() && j["choices"][0].contains("message")) {
                const auto & msg = j["choices"][0]["message"];
                double worst = 0.0;
                for (const char * field : {"content", "reasoning_content"}) {
                    if (msg.contains(field) && msg[field].is_string()) {
                        const std::string s = msg[field].get<std::string>();
                        worst = std::max(worst, common_moe_degeneracy_score(s));
                        if (!s.empty()) {
                            if (!r.text.empty()) {
                                r.text += ' ';
                            }
                            r.text += s;
                        }
                        if (strcmp(field, "content") == 0) {
                            r.answer = s;
                        }
                    }
                }
                r.degeneracy = worst;
            }
            return r;
        }
    } catch (const std::exception &) {
        // falls through to the failure return below
    }
    return {};
}

// Wall-clock deadline for the whole calibration run, set by
// common_moe_calibrate() and checked here rather than at each search's own
// call site. Every candidate in every stage funnels through this function,
// which is what makes it the correct choke point: a first attempt put the
// check in one stage's helper lambda and several stages (the -ngl search
// among them) call this directly instead, so a 600s budget sailed 26 minutes
// past its deadline with 89 more projected. Zero means "no deadline set".
static std::atomic<long long> g_moe_calibrate_deadline_ms{0};
// How many of the reproducibility probes agreed for the first configuration this
// calibration measured - the noise floor later candidates are judged against.
// -1 until something has been measured.
static std::atomic<int> g_moe_repro_floor{-1};

// Inputs for deriving the time budget from this model's measured cost instead
// of a flat constant. See common_moe_record_candidate_cost.
static std::atomic<long long> g_moe_calibrate_start_ms{0};
static std::atomic<int>       g_moe_planned_candidates{0};
// How many verifiable-answer probes the first configuration measured got right.
// The bar later candidates are held to - measured, never assumed, because a
// heavily quantized model may legitimately fail one.
static std::atomic<int>       g_moe_verifiable_ref{-1};
static std::atomic<int>       g_moe_verifiable_ref2{-1};         // second reference run, for the tolerance
static std::atomic<bool>      g_moe_verifying_reference{false};  // the next verifications set the bar
// Where every draft candidate places the draft's experts from here on: false =
// on the GPU (the draft's own default), true = on the CPU (--spec-draft-cpu-moe).
// Calibration-wide rather than a per-call argument so that once the placement
// stage picks one, every later stage measures with it instead of silently
// falling back to the default.
static std::atomic<bool>      g_moe_calib_draft_cpu_moe{false};
// GGML_OP_OFFLOAD_MIN_BATCH every candidate launches with from here on (-1 = the
// backend default). Same calibration-wide pattern as the draft placement.
static std::atomic<int>       g_moe_calib_offload_min_batch{-1};
// When set, a candidate reports prompt-processing throughput (prompt tok/s over a
// short and a long uncached prompt) instead of decode throughput.
static std::atomic<bool>      g_moe_calib_measure_prefill{false};
// -ub every candidate launches with from here on (-1 = the default, 512), and
// whether the prefill probe uses its extra-long prompt (the micro-batch stage:
// a knob that only matters past one micro-batch cannot show on a short prompt).
static std::atomic<int>       g_moe_calib_ubatch{-1};
// The KV precision every candidate after the KV stage runs with. A string
// because that is what the flag takes, guarded because candidates are launched
// from the stage loop while this is being set.
static std::mutex  g_moe_calib_kv_type_mu;
static std::string g_moe_calib_kv_type;
static void common_moe_calib_set_kv_type(const std::string & t) {
    std::lock_guard<std::mutex> lock(g_moe_calib_kv_type_mu);
    g_moe_calib_kv_type = t;
}
static std::string common_moe_calib_get_kv_type() {
    std::lock_guard<std::mutex> lock(g_moe_calib_kv_type_mu);
    return g_moe_calib_kv_type;
}
static std::atomic<bool>      g_moe_calib_prefill_xlong{false};
// GGML_SCHED_PREFETCH_EXPERTS every candidate launches with from here on.
static std::atomic<bool>      g_moe_calib_prefetch{false};
// The draft acceptance mode every candidate runs with once it has been chosen.
//
// --spec-prob-accept was measured ~2000 lines after the depth envelope, so the
// envelope picked n_max under exact-match acceptance and a +10-18% lever was
// switched on afterwards - measured 12.63 vs 11.43 tok/s on Qwen3.8-Flash-Next
// and 77.18 vs 65.36 on gemma-4, the largest single lever in that run. Which
// depth is best depends on how tokens are accepted, so deciding acceptance
// after depth searches the wrong curve, and at 64k it produced "the draft
// loses to no draft" from a configuration the draft would never be served in.
static std::atomic<int>       g_moe_calib_prob_accept{-1};
// GGML_CUDA_MOE_CACHE_RING_PCT every candidate launches with from here on (-1 = unset).
static std::atomic<int>       g_moe_calib_ring_pct{-1};
// Arbitrary extra environment every candidate launches with from here on, for the
// feature stages below: a stage sets one knob, measures, then keeps its winner or
// clears it. Calibration runs one candidate at a time.
// The candidate kept alive for live sweeps: its port, and the pid to kill when
// the allocation configuration changes and it has to be replaced.
static int   g_moe_live_port = -1;
static pid_t g_moe_live_pid  = -1;

static void common_moe_live_stop() {
    if (g_moe_live_pid > 0) {
        kill(g_moe_live_pid, SIGKILL);
        int status = 0;
        waitpid(g_moe_live_pid, &status, 0);
    }
    g_moe_live_pid  = -1;
    g_moe_live_port = -1;
}

static std::mutex             g_moe_calib_env_mu;
static std::string            g_moe_calib_extra_env;
static void common_moe_calib_set_env(const std::string & kv) {
    std::lock_guard<std::mutex> lock(g_moe_calib_env_mu);
    g_moe_calib_extra_env = kv.empty() ? std::string() : kv + " ";
}
// Env that every candidate gets, on top of whatever the current stage sets.
// Stages overwrite the per-stage slot freely, so anything that must hold for
// the WHOLE run (the frozen predictor state, below) cannot live there.
static std::string g_moe_calib_base_env;
static void common_moe_calib_set_base_env(const std::string & kv) {
    std::lock_guard<std::mutex> lock(g_moe_calib_env_mu);
    g_moe_calib_base_env = kv.empty() ? std::string() : kv + " ";
}
static std::string common_moe_calib_get_env() {
    std::lock_guard<std::mutex> lock(g_moe_calib_env_mu);
    // Base first, so a stage that deliberately sets the same knob (the
    // prerouter stage toggling TRAIN_PREDICTOR) still wins - later
    // assignments override earlier ones in the command line this builds.
    return g_moe_calib_base_env + g_moe_calib_extra_env;
}

// A candidate can end badly in two different ways and they must not be
// conflated. -1.0 means the run did not happen (server failed to come up, port
// clash, budget spent) - worth one retry. COMMON_MOE_TPS_REJECTED means it ran
// fine and the output was judged unusable - retrying that is guaranteed to
// reach the same verdict at twice the cost. Both stay < 0 so every "tps > 0"
// ranking test is unaffected.
//
// This conflation is how the quality gates did their real damage: four stages
// retry on tps < 0, so every wrongly-rejected candidate was measured twice,
// which is what exhausted the time budget and left the later stages - thread
// count, cache size, fit margin - skipped in milliseconds.
static constexpr double COMMON_MOE_TPS_REJECTED = -2.0;
// The candidate's process died before it could serve - a model-load OOM, a bad
// placement, a config this machine cannot hold. Deterministic: the same command
// line will fail the same way, so retrying it buys nothing and costs a full
// launch. Distinct from -1.0 ("did not happen", worth one retry) and from
// REJECTED ("ran, output unusable"). Kept < 0 like the others so every
// "tps > 0" ranking test is unaffected.
static constexpr double COMMON_MOE_TPS_LAUNCH_FAILED = -3.0;

// Sampling the calibration probes run under, and the values to quote when
// anyone asks "how should this model be run".
//
// These are Unsloth's recommended settings for the quantized Qwen3.8-Flash-Next
// and Gemma-4 GGUFs this fork targets, not a generic default and not greedy.
// They live here as named constants because they were re-derived from memory
// three separate times in one day, each time slightly differently, and every
// throughput number is only comparable against others measured the same way.
//
// Greedy (temp 0, top-k 1) is deliberately NOT this: it is the right choice for
// a reproducible A/B, where removing sampling variance is the point, but it is
// not the configuration anything is actually served under, and a quality
// judgement made under it can be wrong in both directions - short factual
// prompts behave differently, and speculative acceptance is far higher at
// temp 0 than at temp 1, so a draft measured greedily overstates itself.
#define COMMON_MOE_PROBE_TEMP   "1.0"
#define COMMON_MOE_PROBE_TOP_P  "0.95"
// top_k 20 per Qwen's own model card (thinking mode: temp 1.0, top_p 0.95,
// top_k 20, min_p 0). This was 64, Unsloth's figure; the model card is the
// primary source for the model it describes.
#define COMMON_MOE_PROBE_TOP_K  "20"
// min_p and the penalties, which the model card also specifies and which were
// being left at llama.cpp's own defaults - min_p 0.05 against the documented 0.0.
// An extra tail cut on top of top_p 0.95 is exactly the kind of thing that can
// leave a reasoning model with nothing acceptable to emit after it finishes
// thinking, and it meant every candidate was measured under sampling the model's
// authors did not describe. Thinking-mode values; the instruct-mode row differs
// (temp 0.7, top_p 0.80, presence_penalty 1.5) and is not what these probes use.
#define COMMON_MOE_PROBE_MIN_P  "0.0"
#define COMMON_MOE_PROBE_PRESENCE "0.0"
#define COMMON_MOE_PROBE_REPEAT "1.0"
static std::atomic<long long> g_moe_candidate_ms_sum{0};
static std::atomic<int>       g_moe_candidate_count{0};
static std::atomic<bool>      g_moe_budget_is_derived{false};

// A fixed budget cannot be right for both models this fork runs: gemma-4's full
// search fits in 600s (10m19s measured), while Qwen3.8-Flash-Next spends ~60s
// per candidate, so the same 600s bought ten of the ~40 candidates the stages
// wanted. Everything after the substitution ladder - thread count, expert-cache
// size, fit margin - was then skipped in milliseconds and recorded as "failed",
// which is indistinguishable in the cache from "measured and no good". That is
// how a run ends up serving a configuration nothing measured.
//
// So price the run instead of guessing it: two real candidates say what one
// costs on this model and this hardware, and the status page's own planned-
// candidate estimate says how many the stages intend to run. Derive once, early,
// and only ever extend - never cut a budget the user asked for. An explicit
// GGML_MOE_CALIBRATE_BUDGET_S disables this entirely; the ceiling exists because
// this can block a plain launch, and is itself overridable.
static long long common_moe_steady_now_ms();
// Defined here rather than with the rest of the per-stage budget machinery
// below, because the budget derivation above it has to extend an in-flight
// stage's deadline when the global budget grows. See common_moe_stage_begin.
static std::atomic<long long> g_moe_stage_deadline_ms{0};

// The budget prices itself, per stage, from what candidates actually cost.
//
// It used to be derived ONCE, from the first two candidates, times a hand-kept
// planned-candidate count, then split among stages by their share of that
// count. On Qwen3.8-Flash-Next at 64k all three inputs were wrong at once: the
// first two candidates were cheap placement probes (23s), the substitution
// ladder's cost 2-3 minutes each, and the planned count (118) still included
// every stage the short calibration had since switched off. The ladder - the
// stage that matters most - got a share sized for 23s candidates and never ran.
//
// Now: every candidate updates the price (the larger of the all-time mean and
// the mean of the last three, so cheap early probes cannot set the price for
// expensive late stages), and the stage in flight gets 2x its OWN planned
// candidates at the current price - re-priced after each candidate, only ever
// extended. The global deadline is a runaway guard (the cap), not a plan.
// GGML_MOE_CALIBRATE_BUDGET_S still pins a fixed budget and disables all this.
static std::mutex             g_moe_cost_mu;
static std::deque<long long>  g_moe_recent_candidate_ms;
static std::atomic<long long> g_moe_stage_start_ms{0};
static std::atomic<int>       g_moe_stage_planned{0};

static double common_moe_candidate_price_s() {
    const int n = g_moe_candidate_count.load(std::memory_order_relaxed);
    if (n <= 0) {
        return 0.0;
    }
    const double mean_all = (double) g_moe_candidate_ms_sum.load(std::memory_order_relaxed) / n / 1000.0;
    double mean_recent = 0.0;
    {
        std::lock_guard<std::mutex> lock(g_moe_cost_mu);
        if (!g_moe_recent_candidate_ms.empty()) {
            long long sum = 0;
            for (long long ms : g_moe_recent_candidate_ms) {
                sum += ms;
            }
            mean_recent = (double) sum / g_moe_recent_candidate_ms.size() / 1000.0;
        }
    }
    return std::max(mean_all, mean_recent);
}

static long long common_moe_stage_want_deadline_ms() {
    const long long start = g_moe_stage_start_ms.load(std::memory_order_relaxed);
    const int planned     = g_moe_stage_planned.load(std::memory_order_relaxed);
    const double price_s  = common_moe_candidate_price_s();
    if (start == 0 || planned <= 0 || price_s <= 0.0) {
        return 0;
    }
    const long long want = start + std::max(60000LL, (long long) (2.0 * planned * price_s * 1000.0));
    const long long global = g_moe_calibrate_deadline_ms.load(std::memory_order_relaxed);
    return global > 0 ? std::min(want, global) : want;
}

static void common_moe_record_candidate_cost(long long ms) {
    g_moe_candidate_ms_sum.fetch_add(ms, std::memory_order_relaxed);
    g_moe_candidate_count.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(g_moe_cost_mu);
        g_moe_recent_candidate_ms.push_back(ms);
        while (g_moe_recent_candidate_ms.size() > 3) {
            g_moe_recent_candidate_ms.pop_front();
        }
    }
    if (!g_moe_budget_is_derived.load(std::memory_order_relaxed)) {
        return; // pinned by GGML_MOE_CALIBRATE_BUDGET_S
    }
    // Re-price the stage in flight. Only ever extend it.
    const long long cur  = g_moe_stage_deadline_ms.load(std::memory_order_relaxed);
    const long long want = common_moe_stage_want_deadline_ms();
    if (cur != 0 && want > cur) {
        g_moe_stage_deadline_ms.store(want, std::memory_order_relaxed);
        LOG_INF("%s: stage re-priced - candidates here cost %.0fs, deadline extended by %llds\n",
                __func__, common_moe_candidate_price_s(), (want - cur) / 1000);
    }
}

static std::atomic<bool>      g_moe_calibrate_budget_warned{false};

static long long common_moe_steady_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Per-stage share of what is left.
//
// The budget was one global deadline, so whichever stage happened to run while it
// expired consumed everything and every stage after it was skipped in
// milliseconds - and a skipped stage records "failed", which a cached entry
// cannot tell apart from "measured and no good". That is how a run ends up
// serving settings nothing measured.
//
// A stage now gets the remaining time in proportion to the candidates it plans,
// against the candidates the whole run still plans, with slack so a stage may
// overrun its exact share without eating its successors'. Stages already treat
// budget_spent() as "stop and keep the best so far", so hitting a stage deadline
// degrades to a shorter search rather than to nothing - and with the entry saved
// after each stage, what it did measure survives.
static void common_moe_stage_begin(const char * name, int planned) {
    g_moe_stage_deadline_ms.store(0, std::memory_order_relaxed);
    g_moe_stage_start_ms.store(common_moe_steady_now_ms(), std::memory_order_relaxed);
    g_moe_stage_planned.store(planned, std::memory_order_relaxed);
    if (planned <= 0) {
        return;
    }
    if (!g_moe_budget_is_derived.load(std::memory_order_relaxed)) {
        // Pinned budget: the old proportional share of whatever is left.
        const long long global = g_moe_calibrate_deadline_ms.load(std::memory_order_relaxed);
        const long long now = common_moe_steady_now_ms();
        const long long left = global - now;
        if (global == 0 || left <= 0) {
            return;
        }
        const int planned_total = g_moe_planned_candidates.load(std::memory_order_relaxed);
        const int done          = g_moe_candidate_count.load(std::memory_order_relaxed);
        const int remaining     = std::max(planned, planned_total - done);
        const double share = 1.6 * (double) planned / (double) remaining;
        const long long budget = (long long) ((double) left * std::min(1.0, share));
        g_moe_stage_deadline_ms.store(now + std::max(60000LL, budget), std::memory_order_relaxed);
        return;
    }
    // No candidate priced yet: no stage limit, only the runaway guard.
    const long long want = common_moe_stage_want_deadline_ms();
    g_moe_stage_deadline_ms.store(want, std::memory_order_relaxed);
    if (want != 0) {
        LOG_INF("%s: stage '%s' - %d candidate(s) at %.0fs each on this model, allowed %llds\n",
                __func__, name, planned, common_moe_candidate_price_s(),
                (want - common_moe_steady_now_ms()) / 1000);
    }
}

static void common_moe_stage_end() {
    g_moe_stage_deadline_ms.store(0, std::memory_order_relaxed);
}

static bool common_moe_calibrate_budget_spent() {
    const long long stage = g_moe_stage_deadline_ms.load(std::memory_order_relaxed);
    if (stage != 0 && common_moe_steady_now_ms() >= stage) {
        static std::atomic<bool> stage_warned{false};
        if (!stage_warned.exchange(true)) {
            LOG_WRN("%s: a stage reached its own share of the budget and stopped early - later stages "
                    "keep their share instead of being skipped entirely\n", __func__);
        }
        return true;
    }
    const long long deadline = g_moe_calibrate_deadline_ms.load(std::memory_order_relaxed);
    if (deadline == 0 || common_moe_steady_now_ms() < deadline) {
        return false;
    }
    if (!g_moe_calibrate_budget_warned.exchange(true)) {
        LOG_WRN("%s: calibration time budget spent - skipping remaining candidates and keeping the best "
                "configuration measured so far (raise GGML_MOE_CALIBRATE_BUDGET_S for a longer search)\n",
                __func__);
        common_moe_calibration_status_set("time budget spent - finalizing best measured configuration");
    }
    return true;
}

static double common_moe_bench_candidate_server(
        const std::string & self_exe, const std::string & path_model, const std::string & mtp_path,
        uint32_t n_cpu_moe, int n_max, int n_threads, int port, uint32_t n_ctx, int n_predict,
        int n_concurrency = 1, int moe_cache_mb = -1, int fit_target_mb = -1, int n_gpu_layers = 99,
        int substitute_min_rank = -1, std::string * out_sample = nullptr, int probe_seed = 1234,
        double substitute_quality_sigma = std::numeric_limits<double>::quiet_NaN(),
        bool verify_answers = false,
        bool with_reasoning = false,
        const std::string & extra_env = std::string(),
        int spec_prob_accept = -1,
        const std::string & spec_types = std::string(),
        double spec_p_min = -1.0,
        // Live sweeps. live_port > 0 measures on an already-running candidate
        // instead of launching one, after applying live_tunables through
        // POST /moe-tuning; keep_alive leaves this candidate running so later
        // calls can reuse it. About 80% of a candidate is loading the model, and
        // policy knobs do not need a reload - see moe_cache_set_tunable.
        int live_port = -1,
        const std::string & live_tunables = std::string(),
        bool keep_alive = false) {
    if (common_moe_calibrate_budget_spent()) {
        return -1.0; // reported as a failed candidate; every search here keeps its best measured point
    }
    // Every candidate in every stage funnels through here (the same property
    // that makes it the right place for the deadline check above), so it is also
    // where a candidate's real cost on this model can be learned.
    struct candidate_timer {
        long long t0 = common_moe_steady_now_ms();
        ~candidate_timer() {
            common_moe_record_candidate_cost(common_moe_steady_now_ms() - t0);
        }
    } timer;
    std::string mtp_args;
    if (!mtp_path.empty()) {
        char buf[2048];
        snprintf(buf, sizeof(buf), "--model-draft '%s' --spec-type %s --spec-draft-n-max %d ",
                mtp_path.c_str(), spec_types.empty() ? "draft-mtp" : spec_types.c_str(), n_max);
        mtp_args = buf;
        // Only ever appended for a candidate that is explicitly measuring this
        // lever; -1 leaves the server on its own default (off, exact-match).
        if (spec_prob_accept == 1) {
            mtp_args += "--spec-prob-accept ";
        } else if (spec_prob_accept == 0) {
            mtp_args += "--no-spec-prob-accept ";
        }
        if (spec_p_min >= 0.0) {
            mtp_args += string_format("--spec-draft-p-min %.2f ", spec_p_min);
        }
        if (g_moe_calib_draft_cpu_moe.load()) {
            mtp_args += "--spec-draft-cpu-moe ";
        }
    }
    std::string threads_args;
    if (n_threads > 0) {
        char buf[128];
        snprintf(buf, sizeof(buf), "-t %d -tb %d ", n_threads, n_threads);
        threads_args = buf;
    }
    if (g_moe_calib_ubatch.load() > 0) {
        threads_args += string_format("-ub %d ", g_moe_calib_ubatch.load());
    }
    {
        const std::string kvt = common_moe_calib_get_kv_type();
        if (!kvt.empty()) {
            threads_args += string_format("-ctk %s -ctv %s ", kvt.c_str(), kvt.c_str());
        }
    }
    // At concurrency > 1, --parallel must match so the candidate server can
    // actually hold n_concurrency slots, and -c needs enough headroom for
    // all of them at once (n_predict + a real prompt, per slot) - reusing
    // the caller's n_ctx here would starve most slots down to a handful of
    // tokens each and force truncation/failure well before this is a real
    // concurrent-throughput measurement.
    // Always pass the slot count, 1 included. llama-server's own default is
    // "auto" (-1), which it resolves to 4 slots with a unified KV cache - so a
    // candidate launched at concurrency 1 with no --parallel ran with 4 slots
    // (clamped to 2 by the moe-cache's concurrency limit: "n_slots = 2,
    // kv_unified = 'true'" in every candidate log), while the configuration it
    // was calibrating serves with exactly 1. The extra slot's KV cache came out
    // of the VRAM the expert cache is sized from, and the batch hints followed
    // the wrong slot count, so every stage measured a deployment nobody runs.
    std::string parallel_args = string_format("--parallel %d ", std::max(1, n_concurrency));
    uint32_t ctx_for_launch = n_ctx;
    if (n_concurrency > 1) {
        ctx_for_launch = std::max<uint32_t>(n_ctx, (uint32_t) n_concurrency * 384);
    }
    char cmd[4096];
    // --temp/--top-p/--top-k match this model family's documented MTP
    // sampling recommendation (Unsloth's llama.cpp MTP guide) - found to
    // matter for output correctness, not just MTP: benchmarking with
    // mismatched sampling produced degenerate, highly-repetitive
    // generations that (being trivially predictable) artificially inflate
    // both MoE-cache hit rate and MTP acceptance rate, biasing every
    // throughput number measured this way. Real, representative numbers
    // require realistic, non-degenerate generation.
    // --no-token-freq-log: this candidate's probe traffic is a handful of
    // fixed prompts repeated across many candidates, not representative of
    // real deployment usage - would badly skew a histogram meant to guide
    // FR-Spec vocab trimming toward this benchmark's own prompt pool
    // instead of real traffic.
    char cache_arg[32];
    // 0 is a real candidate - the cache switched off - not "unset": on
    // Qwen3.8-Flash-Next the cache is the only source of run-to-run variation
    // (five greedy repeats byte-identical with it off), so whether it earns its
    // place has to be measured, not assumed.
    if (moe_cache_mb == 0) {
        snprintf(cache_arg, sizeof(cache_arg), "off");
    } else {
        snprintf(cache_arg, sizeof(cache_arg), moe_cache_mb > 0 ? "%d" : "auto", moe_cache_mb);
    }
    // -fitt controls the VRAM margin common_maybe_raise_moe_for_ctx reserves
    // before deciding placement, and that margin is what silently raises
    // -ncmoe until the requested context fits. Passing it through means a
    // candidate is benchmarked at the placement it actually asked for
    // instead of whatever the default 3 x 1024 MiB margin forced it to -
    // without this, sweeping -ncmoe below the margin's floor measures the
    // same effective configuration several times over (confirmed: -ncmoe
    // 22/16/10 all silently became 27 and returned near-identical tok/s).
    std::string fit_args;
    // A candidate must serve at the context it was asked for, or not at all.
    //
    // Candidates ran with fit ENABLED, so each one's own fit logic was free to
    // trade the requested context down until the rest of its configuration
    // fitted - and it did, silently. Measured directly: a calibration launched
    // with -c 65536 produced candidate after candidate reporting
    // "n_ctx_slot = 4096" and "context size of 4096 ... falling back to the
    // minimum context size of 4096". Every value in that run was chosen at 4k.
    //
    // That is not a small discrepancy, because VRAM is the binding constraint
    // and context is most of what consumes it. At 4k there is room to spare, so
    // the sweeps picked moe-cache 2048 and -ub 2048 - and both of them OOM at
    // 65536 with the draft attached (676 MiB short). The run optimised a
    // configuration this machine cannot serve, and nothing in it could tell,
    // because a candidate that quietly shrank its context still returned a
    // perfectly good tok/s number for a configuration nobody asked about.
    //
    // With fit off, the parent's explicit -ngl / -ncmoe / --moe-cache / -c are
    // what the candidate runs. One that cannot hold them fails to launch, which
    // is the honest answer: that combination does not fit. The launch-failure
    // sentinel already keeps such a candidate from being retried or mistaken
    // for a quality rejection.
    // The one exception is the fit-margin stage itself, which passes a
    // fit_target_mb because the margin is the quantity it is measuring - and
    // -fitt means nothing with fit disabled. That stage keeps fit on and is
    // explicitly measuring what fit does with the context; every other stage
    // wants the context it asked for.
    if (fit_target_mb > 0) {
        char buf[64];
        snprintf(buf, sizeof(buf), "-fitt %d ", fit_target_mb);
        fit_args = buf;
    } else {
        fit_args = "-fit off ";
    }
    // Substitution aggressiveness is an env var, not a flag, so it is passed
    // as a shell assignment on the candidate's own command line - the
    // subprocess must see the value being benchmarked, not whatever this
    // calibrating process happens to have inherited.
    std::string subst_env;
    if (!std::isnan(substitute_quality_sigma)) {
        char qbuf[96];
        snprintf(qbuf, sizeof(qbuf), "GGML_CUDA_MOE_CACHE_SUBSTITUTE_QUALITY_SIGMA=%g ",
                substitute_quality_sigma);
        subst_env += qbuf;
    }
    if (substitute_min_rank >= 0) {
        char buf[96];
        snprintf(buf, sizeof(buf), "GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK=%d ", substitute_min_rank);
        subst_env = buf;
    }
    // Ranking probes run with reasoning off; the confirm step turns it back on.
    // On a reasoning model a short probe is spent entirely inside the "We need
    // answer user: ..." preamble - the throughput is still real tok/s, but the
    // text scored for degeneracy is preamble rather than output, and a
    // correctness probe never reaches an answer at all (measured: the reference
    // itself scored 1 of 4 at 24 tokens). With thinking disabled a short probe
    // contains a real answer, so many candidates stay affordable AND checkable.
    // The winner is then confirmed at full length WITH reasoning, because that
    // is the configuration actually served - so the cheap regime only ever ranks,
    // and never has the last word.
    const char * reasoning_args = with_reasoning ? "" : "--reasoning off ";
    // Keep the candidate's own output. It used to go to /dev/null, which meant a
    // candidate that failed to come up left no evidence whatsoever - one such
    // failure (spec-draft-n-max=1 on qwen4exp) skipped four MTP stages and could
    // not be diagnosed afterwards at all. One file per port, overwritten by the
    // next candidate on that port, so this costs one small file, not a pile.
    // Candidates are launched without --moe-calibrate (they must not calibrate
    // recursively), so the runtime tuning endpoint is enabled by its own variable -
    // without it every live POST was silently rejected and the knob never changed.
    // A calibration candidate must not read the cached calibration entry.
    //
    // Every knob a candidate runs with is chosen by the parent and passed
    // explicitly; the cache holds a PREVIOUS run's answers for this same model.
    // Where the parent deliberately passes nothing - because it measured that
    // no value fits - the child saw its own default, matched the "user did not
    // ask for this" test in common_moe_apply_prefill_knobs, and substituted the
    // stale cached value instead.
    //
    // Measured on Qwen3.8-Flash-Next at 64k: the parent found that -ub 2048 does
    // not fit once the draft is attached, stepped to 512, found that did not fit
    // either, and left the micro-batch unset. Every candidate after that point
    // then launched with the cached -ub 2048, OOMed allocating the MTP context's
    // compute buffers (676 MiB on a full card), and was recorded as "does not fit
    // at the requested context" - 30 consecutive candidates, the whole
    // substitution ladder, the quality bar, admission, the ring, the cascade and
    // the final validation, all of them measuring a configuration the parent had
    // already rejected.
    //
    // This is the same bug shape this file keeps hitting - a value decided in one
    // place and inherited somewhere it was never meant to apply - reaching the
    // candidates through the cache rather than through a stage.
    std::string env_prefix = std::string("LLAMA_MOE_CALIBRATION_CHILD=1 LLAMA_MOE_TUNING_ENDPOINT=1 ") + extra_env;
    if (g_moe_calib_offload_min_batch.load() > 0) {
        env_prefix = string_format("GGML_OP_OFFLOAD_MIN_BATCH=%d ", g_moe_calib_offload_min_batch.load()) + env_prefix;
    }
    if (g_moe_calib_prefetch.load()) {
        env_prefix = std::string("GGML_SCHED_PREFETCH_EXPERTS=1 ") + env_prefix;
    }
    if (g_moe_calib_ring_pct.load() > 0) {
        env_prefix = string_format("GGML_CUDA_MOE_CACHE_RING_PCT=%d ", g_moe_calib_ring_pct.load()) + env_prefix;
    }
    env_prefix = common_moe_calib_get_env() + env_prefix;
    char log_path[256];
    snprintf(log_path, sizeof(log_path), "%s/llama-moe-calib-candidate-%d.log",
             getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp", port);
    snprintf(cmd, sizeof(cmd),
        "%s%s'%s' -m '%s' -ngl %d -ncmoe %u --moe-cache %s -c %u %s%s%s%s%s"
        "--temp " COMMON_MOE_PROBE_TEMP " --top-p " COMMON_MOE_PROBE_TOP_P
        " --top-k " COMMON_MOE_PROBE_TOP_K " --min-p " COMMON_MOE_PROBE_MIN_P
        " --presence-penalty " COMMON_MOE_PROBE_PRESENCE
        " --repeat-penalty " COMMON_MOE_PROBE_REPEAT " --no-token-freq-log "
        "--port %d --no-webui > '%s' 2>&1 & echo $!",
        env_prefix.c_str(), subst_env.c_str(), self_exe.c_str(), path_model.c_str(), n_gpu_layers, n_cpu_moe, cache_arg, ctx_for_launch,
        mtp_args.c_str(), threads_args.c_str(), parallel_args.c_str(), fit_args.c_str(), reasoning_args, port, log_path);
    pid_t pid = -1;
    const bool reuse = live_port > 0;
    if (reuse) {
        // Measure on a candidate that is already loaded. Only policy knobs can
        // change this way; everything the launch line decides (placement, cache
        // size, ubatch, draft) must already match, which is the caller's job.
        port = live_port;
        if (!live_tunables.empty()) {
            const std::string tune_cmd = string_format(
                    "curl -s -o /dev/null -m 10 -X POST http://127.0.0.1:%d/moe-tuning "
                    "-H 'Content-Type: application/json' --data-binary %s",
                    port, common_shell_quote(live_tunables).c_str());
            if (system(tune_cmd.c_str()) != 0) {
                LOG_WRN("%s: could not apply live tunables on port %d\n", __func__, port);
                return -1.0;
            }
        }
    } else {
        FILE * pf = popen(cmd, "r");
        if (!pf) {
            return -1.0;
        }
        char pidbuf[32] = {0};
        const bool got_pid = fgets(pidbuf, sizeof(pidbuf), pf) != nullptr;
        pclose(pf);
        if (!got_pid) {
            return -1.0;
        }
        pid = (pid_t) atol(pidbuf);
        if (pid <= 0) {
            return -1.0;
        }
    }

    // A reused candidate is never ours to kill. A kept-alive one is spared only at
    // the single success return at the end - not here: this lambda also runs on
    // every failure path, and sparing it there leaked the process, which then held
    // the GPU so every later launch died with "unable to allocate CUDA0 buffer"
    // while calibration waited on candidates that could never load.
    auto cleanup = [&]() {
        if (reuse || pid <= 0) {
            return;
        }
        kill(pid, SIGKILL);
        int status = 0;
        waitpid(pid, &status, 0);
    };

    char health_cmd[256];
    snprintf(health_cmd, sizeof(health_cmd),
        "curl -s -o /dev/null -w '%%{http_code}' http://127.0.0.1:%d/health 2>/dev/null", port);
    // Wait on the child's liveness, not on a fixed number of seconds. The old
    // form gave every candidate the same 60s (30 polls x 2s) to become healthy,
    // which silently encoded an assumption about load time: a candidate that also
    // loads a draft model loads two models and can cross it, and a candidate on a
    // model far larger than RAM crosses it while doing nothing wrong. Measured on
    // qwen4exp, that is what failed spec-draft-n-max=1 - and one such failure used
    // to skip the entire MTP half of calibration.
    //
    // A dead child is diagnosed immediately (no reason to keep polling a process
    // that has exited), and a live one is given room, so the ceiling stops being a
    // judgement about how long loading ought to take and becomes a backstop
    // against a genuine hang.
    bool ready = !reuse ? false : true;   // a reused candidate is already serving
    bool child_exited = false;
    for (int i = 0; !ready && i < 300 && !child_exited; i++) {
        FILE * hp = popen(health_cmd, "r");
        if (hp) {
            char code[8] = {0};
            const bool got = fgets(code, sizeof(code), hp) != nullptr;
            pclose(hp);
            if (got && strncmp(code, "200", 3) == 0) {
                ready = true;
                break;
            }
        }
        // Still loading, or gone? Ask two ways, because one of them lies.
        //
        // waitpid(WNOHANG) only reports the exit if THIS process is the one
        // that reaps it. When something else has already reaped the child it
        // returns -1/ECHILD, which the == pid test reads as "still loading" -
        // so a candidate that died in five seconds sat out the whole 600s
        // backstop, twice over with the retry. Observed directly: a candidate
        // OOMed on model load at 12:44:37 and was not declared dead until
        // 12:54:24, and the message said "never became healthy" rather than
        // "exited before serving", which is what pinned it on this test.
        //
        // kill(pid, 0) answers the question that actually matters - is there
        // still a process there - and does not care who reaps it.
        // kill(pid, 0) is the ONLY valid liveness test here, because the
        // candidate is not our child: `cmd` backgrounds llama-server through a
        // shell and echoes its pid, so the shell is reaped immediately and the
        // server is a grandchild. waitpid on it therefore returns -1/ECHILD on
        // every single poll - which is why the original `== pid` test never
        // fired and a candidate that died in five seconds sat out the whole
        // 600s backstop.
        //
        // Treating that ECHILD as "exited" is the trap, and it is worse than
        // the bug it looks like it fixes: ECHILD is the NORMAL state for this
        // pid, so every candidate gets declared dead on its first poll while
        // it is still loading, and the process is then leaked holding the
        // whole GPU. Measured directly: one leaked candidate held 11.6 GiB of
        // a 12 GiB card and every launch after it died on cudaMalloc.
        //
        // waitpid is kept only for the case where the pid really is a child
        // (it costs nothing and is correct there); ESRCH from kill is what
        // actually decides.
        int status = 0;
        if (waitpid(pid, &status, WNOHANG) == pid) {
            child_exited = true;
            break;
        }
        if (kill(pid, 0) != 0 && errno == ESRCH) {
            child_exited = true;
            break;
        }
        struct timespec ts{2, 0};
        nanosleep(&ts, nullptr);
    }
    if (!ready) {
        LOG_WRN("%s: candidate on port %d %s - its log is at %s\n", __func__, port,
                child_exited ? "exited before serving" : "never became healthy", log_path);
        if (!child_exited) {
            cleanup();
        }
        // A process that exited before serving will exit again on the same
        // command line - say so, so the caller does not spend a second launch
        // proving it.
        //
        // With candidates now pinned to the requested context, the commonest
        // reason for this is the honest one: that combination does not fit on
        // this card at this context. Name it, because "failed" reads as a bug
        // and this is a measurement - the configuration was tried and could
        // not be held.
        if (child_exited) {
            std::ifstream lf(log_path);
            if (lf) {
                std::string line;
                while (std::getline(lf, line)) {
                    if (line.find("cudaMalloc failed: out of memory") != std::string::npos ||
                        line.find("unable to allocate") != std::string::npos) {
                        LOG_WRN("%s: this configuration does not fit at the requested context - "
                                "not a failure to retry, an answer\n", __func__);
                        break;
                    }
                }
            }
        }
        return child_exited ? COMMON_MOE_TPS_LAUNCH_FAILED : -1.0;
    }

    // /v1/chat/completions, not /completion: the raw completion endpoint
    // bypasses the model's chat template entirely, and this model family
    // (instruction/reasoning-tuned) produces degenerate output on
    // unformatted raw prompts independent of MTP/moe-cache/placement -
    // confirmed directly by testing (see docs). The chat endpoint applies
    // the GGUF's own template server-side, matching how any real client
    // would actually talk to this server.
    static const char * const probe_prompts[] = {
        "Explain how photosynthesis works in three sentences, then describe the role of chlorophyll.",
        "Write a Python function that implements binary search, then explain its time complexity.",
        "Summarize the plot of a mystery novel in two sentences, then suggest a twist ending.",
        "Describe the water cycle briefly, then explain how it relates to weather patterns.",
        "Explain Newton's second law of motion, then give a real-world example.",
        "Describe how a car engine works in simple terms, then list its main components.",
        "What are the main causes of inflation, and how do central banks respond?",
        "Explain how vaccines work, then describe herd immunity.",
    };
    static constexpr int n_probe_prompts = sizeof(probe_prompts) / sizeof(probe_prompts[0]);

    // Short, low-context prompts, scored for quality but never for speed.
    //
    // Every prompt above is a substantial instruction, and a model with a
    // paragraph of task to hold onto is the EASY case for degeneracy: there is
    // plenty of context anchoring the next token. The failure that actually
    // reached a user on gemma-4 was a bare "hi" answered with the same
    // sentence four times over plus leaked "Input:/Output:" scaffolding -
    // classic repetition the guard above catches easily, on a kind of prompt
    // the ladder had never once put in front of it. Cheap to add: the prompts
    // are tiny, and a healthy reply to them is short, so the only runs that
    // cost real time here are the broken ones worth catching.
    static const char * const quality_prompts[] = {
        "hi",
        "thanks!",
        "What's 2 + 2?",
    };
    static constexpr int n_quality_prompts = sizeof(quality_prompts) / sizeof(quality_prompts[0]);

    // Coherence is not correctness. Degeneracy above catches broken generation -
    // repetition, word salad - and the fidelity bar catches drift from the
    // substitution-free reference, but the fidelity bar only gates when the two
    // reference runs gave it evidence to measure a tolerance from, and on
    // Qwen3.8-Flash-Next it came back 0.00. In that gap a configuration can emit
    // perfectly fluent, perfectly self-consistent, *wrong* text and pass every
    // check. Substitution serves a different expert than the router asked for, so
    // that is exactly the failure it would produce if pushed too far.
    //
    // These have answers that are checkable regardless of wording, so a wrong
    // answer is wrong no matter how well phrased. Accept every reasonable form -
    // the probe is testing the model's routing, not its formatting.
    struct verifiable_probe {
        const char * prompt;
        const char * accept[4];
    };
    static const verifiable_probe verifiable_probes[] = {
        { "What is 2 + 2? Reply with just the number.",                    { "4", "four", nullptr, nullptr } },
        { "What is the capital of France? Reply with just the city name.", { "paris", nullptr, nullptr, nullptr } },
        { "Complete with one word: the opposite of hot is ___.",           { "cold", "chilly", nullptr, nullptr } },
        { "How many days are in a week? Reply with just the number.",      { "7", "seven", nullptr, nullptr } },
        // Plain questions with no "reply with just" instruction, as a user would
        // ask them. These are the ones that exposed rank 1 / -1.0 sigma on
        // Qwen3.8-Flash-Next: served, it answered 2 of 10 (empty after </think>,
        // or fragments like "boiling point"), while four instructed one-word
        // probes scored greedily had passed it 4 of 4.
        { "What is 17 times 23?",                                          { "391", nullptr, nullptr, nullptr } },
        { "Name the largest planet.",                                      { "jupiter", nullptr, nullptr, nullptr } },
        { "What is the boiling point of water in Celsius?",                { "100", nullptr, nullptr, nullptr } },
        { "Who wrote Hamlet?",                                             { "shakespeare", nullptr, nullptr, nullptr } },
    };
    static constexpr int n_verifiable = sizeof(verifiable_probes) / sizeof(verifiable_probes[0]);

    double result_tps = -1.0;
    if (n_concurrency <= 1) {
        // Solo path: average per-request predicted_per_second across
        // n_solo_probes sequential probes. Deliberately reduced from 2 to 1
        // (see n_samples_per_candidate's comment below - the same
        // speed-vs-noise trade, made together) on a slow-decoding model
        // where every probe is expensive; this drops the averaging that
        // partially absorbed run-to-run variance within a single candidate.
        // With a draft attached, time code and reasoning rather than prose. Draft
        // acceptance is content-dependent - published for this model: code 90%,
        // reasoning 73%, prose 38% (little or no speedup) - so a prose-only probe
        // measures speculative decoding at its worst and ranks depth, placement
        // and acceptance for a workload nobody runs. Two probes, not one, for the
        // same reason.
        static const char * const mtp_probe_prompts[] = {
            "Write a Python function that implements binary search, then explain its time complexity.",
            "A train leaves at 3pm at 60 km/h; a second leaves the same station at 4pm at 90 km/h on the "
            "same track. When does the second catch the first? Work it out step by step.",
            // Prose, where a draft is accepted ~38% of the time against 73-90% for code and
            // reasoning: a depth picked on those two alone ran 8 deep and cost a mixed
            // workload up to a third of its speed (gemma-4, 1 Oct 2026: depth 4 measured
            // +32% on prose). Added to the mix, not instead of it.
            "Explain in a few paragraphs how central banks respond to inflation, and why markets "
            "react to what they announce.",
        };
        const int n_solo_probes = mtp_path.empty() ? 1 : 3;
        double sum_tps = 0.0;   // (kept for the single-probe path)
        double sum_tok = 0.0, sum_ms = 0.0;   // the mix is scored by total tokens over total time
        int n_ok = 0;
        double worst_degeneracy = 0.0;
        // One untimed request first. A candidate server has just started, so its
        // expert cache is empty (the loader's --warmup is an empty run and routes
        // nothing), and a probe timed straight away measures cold misses rather
        // than the configuration. Measured: the table read ~4 tok/s for settings
        // that served ~9 warm. A cold probe does not just read low, it ranks
        // wrong - it penalises exactly the settings that rely on a warm cache.
        (void) common_moe_bench_one_request_full(port, probe_prompts[n_solo_probes], n_predict, probe_seed);
        if (g_moe_calib_measure_prefill.load()) {
            // Two uncached prompts: one short enough that its chunk stays under
            // any threshold being tried, one long enough to cross most of them,
            // so the score reflects both batch paths rather than one size.
            static const std::string prefill_long = [] {
                std::string p = "Read the following notes and then summarise them in one sentence.\n";
                for (int i = 0; i < 6; i++) {
                    p += "The town council met on Tuesday to review the budget for road repairs, the new "
                         "library wing, and the summer festival. Members agreed to delay the festival "
                         "decision until the next session, approved the library plan, and asked the "
                         "engineers for a revised estimate on the roads before any contract is signed. ";
                }
                return p;
            }();
            static const std::string prefill_xlong = [] {
                std::string p = "Read the following notes carefully and list every decision they record.\n";
                for (int i = 0; i < 24; i++) {
                    p += "The town council met on Tuesday to review the budget for road repairs, the new "
                         "library wing, and the summer festival. Members agreed to delay the festival "
                         "decision until the next session, approved the library plan, and asked the "
                         "engineers for a revised estimate on the roads before any contract is signed. ";
                }
                return p;
            }();
            const bool xlong = g_moe_calib_prefill_xlong.load();
            std::vector<const char *> prefill_prompts;
            if (xlong) {
                prefill_prompts = { prefill_xlong.c_str() };
            } else {
                prefill_prompts = { probe_prompts[0], prefill_long.c_str() };
            }
            double tok = 0.0, ms = 0.0;
            for (const char * pp : prefill_prompts) {
                const auto r = common_moe_bench_one_request_full(port, pp, 4, probe_seed, false, /* cache_prompt */ false);
                if (r.prompt_n > 0 && r.prompt_ms > 0) {
                    tok += r.prompt_n;
                    ms  += r.prompt_ms;
                }
            }
            cleanup();
            return ms > 0 ? tok / (ms / 1000.0) : -1.0;
        }
        g_moe_last_draft_n          = -1.0;
        g_moe_last_draft_n_accepted = -1.0;
        for (int i = 0; i < n_solo_probes; i++) {
            const char * timed_prompt = mtp_path.empty() ? probe_prompts[i] : mtp_probe_prompts[i];
            const auto r = common_moe_bench_one_request_full(port, timed_prompt, n_predict, probe_seed);
            if (r.predicted_n > 0 && r.predicted_ms > 0) {
                sum_tps += r.predicted_n / (r.predicted_ms / 1000.0);
                sum_tok += r.predicted_n;
                sum_ms  += r.predicted_ms;
                worst_degeneracy = std::max(worst_degeneracy, r.degeneracy);
                n_ok++;
                if (r.draft_n > 0) {
                    g_moe_last_draft_n          = std::max(0.0, g_moe_last_draft_n) + r.draft_n;
                    g_moe_last_draft_n_accepted = std::max(0.0, g_moe_last_draft_n_accepted) + r.draft_n_accepted;
                }
            }
        }
        // Total tokens over total time, not the mean of the rates: a mean of rates lets the
        // fast probe (code) hide the slow one (prose), which is the one that sets how long a
        // real mixed workload takes. Identical for a single probe.
        result_tps = n_ok > 0 ? (sum_ms > 0 ? sum_tok / (sum_ms / 1000.0) : sum_tps / n_ok) : -1.0;
        // Quality-only pass over the short prompts. Kept out of the timing
        // average deliberately: these generate few tokens when healthy, so
        // folding them into tok/s would measure prompt length, not decode
        // rate. The fidelity sample is taken from here rather than from the
        // timing probe because this is the harder case - with almost no
        // context to anchor on, a configuration that has damaged the model's
        // routing shows it here first.
        if (result_tps > 0) {
            for (int i = 0; i < n_quality_prompts; i++) {
                const auto q = common_moe_bench_one_request_full(port, quality_prompts[i], n_predict, probe_seed);
                if (q.predicted_n <= 0) {
                    continue;
                }
                worst_degeneracy = std::max(worst_degeneracy, q.degeneracy);
                if (out_sample && i == 0) {
                    *out_sample = q.text;
                }
            }
        }
        // Fluent-but-wrong check. Scored against what the first configuration of
        // this run actually managed, so a quant that cannot do one of these is
        // not punished for it - only a candidate that answers fewer than the
        // reference did is rejected.
        // Only where it can be afforded and can actually work: once, on the
        // configuration about to be committed. Two reasons it is not per-candidate.
        // Cost - 4 probes at a length a reasoning model needs is ~170s against a
        // ~60s candidate, times 32 candidates. And validity - the first version ran
        // these at 24 tokens, which on Qwen3.8-Flash-Next is spent inside the
        // reasoning preamble ("We need answer user: ..."), so the reference itself
        // scored 1 of 4 and every candidate was rejected against that noise. A
        // correctness probe has to let the model finish thinking before it is
        // scored, or it measures verbosity.
        if (result_tps > 0 && verify_answers) {
            int correct = 0;
            for (int i = 0; i < n_verifiable; i++) {
                // Served sampling, not greedy: the gate has to see the regime the
                // answers will actually be produced under. Greedy is where this
                // passed rank 1 while temp 1.0 served broken answers - the argmax
                // path can survive a stand-in that sampling does not.
                const auto v = common_moe_bench_one_request_full(
                        port, verifiable_probes[i].prompt, 320, probe_seed, /* greedy */ false);
                if (v.predicted_n <= 0 || v.answer.empty()) {
                    continue; // no answer after </think> is a wrong answer, not a skipped probe
                }
                std::string low = v.answer;
                std::transform(low.begin(), low.end(), low.begin(),
                        [](unsigned char c) { return (char) std::tolower(c); });
                for (const char * want : verifiable_probes[i].accept) {
                    if (want && low.find(want) != std::string::npos) {
                        correct++;
                        break;
                    }
                }
            }
            const int ref = g_moe_verifiable_ref.load();
            // The bar comes from the substitution-off reference, measured twice,
            // never from a candidate. It used to be set by whichever candidate was
            // verified first - on Qwen3.8-Flash-Next that was rank 1, which then
            // passed against its own score.
            const int ref2 = g_moe_verifiable_ref2.load();
            const int bar  = ref2 >= 0 ? std::min(ref, ref2) : ref;
            // The reference's own run-to-run difference is the tolerance: the
            // model's natural variation at the served sampling, measured rather
            // than chosen. Two identical references give 0 - the strictest bar,
            // because that is the model saying it answers these reliably.
            // Tolerance, with a floor.
            //
            // |ref - ref2| is the model's observed run-to-run spread, and when
            // the two reference runs happen to agree exactly that difference is
            // 0 - which was then used as "any deviation disqualifies". That is a
            // degenerate inference: two identical draws do not show the variance
            // is zero, they show it has not been observed in two draws. With
            // eight binary probes, a two-answer difference in a single run is
            // well inside what noise produces.
            //
            // It cost a real result. Measured on Qwen3.8-Flash-Next at 64k,
            // substitution rank 4 ran at 9.91 tok/s against rank 6's 6.25 - 59%
            // faster - and was rejected for answering 6 of 8 where the reference
            // answered 8 of 8, on one sample, against a tolerance of zero
            // derived from two runs that both happened to score perfectly.
            //
            // So the floor is one probe: a candidate has to be worse than the
            // reference by more than a single answer before a single run is
            // allowed to disqualify it. Anything that really is broken fails by
            // far more than one - the failures this gate was built for scored 2
            // of 10 - and the rejection is re-measured below in any case.
            const int tol  = std::max(ref2 >= 0 ? std::abs(ref - ref2) : 0, 1);
            if (g_moe_verifying_reference.load()) {
                if (ref < 0) {
                    g_moe_verifiable_ref.store(correct);
                } else {
                    g_moe_verifiable_ref2.store(correct);
                }
                LOG_INF("%s: verifiable answers with substitution off (reference): %d of %d\n",
                        __func__, correct, n_verifiable);
            } else if (ref < 0) {
                // No substitution-off reference was measured (the budget ran out before
                // it), so there is nothing to judge this candidate against. Letting it
                // set its own bar means it then passes by definition - a broken
                // candidate grading itself. Reject instead: the stage records nothing,
                // the setting stays uncalibrated, and the runtime default applies.
                LOG_WRN("%s: no substitution-off reference for the answer check - rejecting this candidate "
                        "rather than letting it set its own bar (%d of %d correct)\n",
                        __func__, correct, n_verifiable);
                common_moe_calibration_status_note("output check", "verifiable answers",
                        "rejected - no reference measured to judge against", false);
                result_tps = COMMON_MOE_TPS_REJECTED;
            } else if (correct < bar - tol) {
                // A rejection is a measurement too, and this one is asymmetric
                // with how the rest of the run treats evidence: every WIN here
                // is re-measured before it is believed - the cache ladder, the
                // depth envelope, the acceptance mode all confirm - while a
                // rejection got a single sample and was final. On eight binary
                // probes that is the weaker of the two decisions being given
                // the stronger power.
                LOG_WRN("%s: candidate rejected - answered %d of %d verifiable probes against the "
                        "substitution-off reference's %d (tolerance %d, its own run-to-run spread), at %.2f "
                        "tok/s. Fluent output that is wrong is not a faster configuration, it is a broken one\n",
                        __func__, correct, n_verifiable, bar, tol, result_tps);
                common_moe_calibration_status_note("output check", "verifiable answers",
                        string_format("rejected - %d of %d correct, reference %d +/- %d (was %.2f tok/s)",
                                correct, n_verifiable, bar, tol, result_tps), false);
                result_tps = COMMON_MOE_TPS_REJECTED;
            }
        }

        // Reject rather than rank: a candidate that generates degenerate text
        // is not a slower-but-valid point on the throughput curve, it is not a
        // usable configuration at all, so it must not be able to win on speed.
        // Reported as a failed candidate (-1), the same as one whose server
        // never came up.
        if (result_tps > 0 && worst_degeneracy >= common_moe_degeneracy_reject_threshold()) {
            LOG_WRN("%s: candidate rejected - output is degenerate (score %.2f >= %.2f) at %.2f tok/s; "
                    "throughput bought with broken generation is not a valid result\n",
                    __func__, worst_degeneracy, common_moe_degeneracy_reject_threshold(), result_tps);
            common_moe_calibration_status_note("output check", "degeneracy",
                    string_format("rejected - score %.2f (was %.2f tok/s)", worst_degeneracy, result_tps), false);
            result_tps = COMMON_MOE_TPS_REJECTED;
        }
        // Determinism gate: same request, same seed, repeated. A configuration
        // that answers one question two different ways is not a slower-but-
        // valid point on the throughput curve - the forward pass is returning
        // different numbers each time, and no amount of tok/s redeems that.
        //
        // This exists because calibration certified a gemma-4 configuration at
        // 48 tok/s and shipped it as the default, when repeating one identical
        // greedy request against it gave 9 distinct answers out of 12. Nothing
        // already here could see that: the degeneracy guard scores each reply
        // on its own and every one of those nine was lexically healthy, and the
        // fidelity check compares a candidate against a reference but never
        // compares a candidate against itself. Cheap to close - the replies are
        // short and a broken config usually diverges within a few tokens.
        // Majority agreement, not unanimity. Requiring all N identical was
        // tried first and is wrong: measured on this hardware, even a pure-CPU
        // run (-ngl 0) disagrees with itself occasionally at this length, so a
        // unanimity gate rejects every configuration including the good ones
        // and calibration finds nothing. Majority is the natural
        // parameter-free line - it asks whether there IS a single answer this
        // configuration mostly gives, which cleanly separates the failure that
        // prompted this (9 distinct answers in 12 requests) from healthy noise
        // (11 of 12 agreeing), without a tolerance anybody had to pick.
        if (result_tps > 0 && !common_moe_calibrate_budget_spent()) {
            constexpr int n_det_probes = 5;
            std::vector<std::string> answers;
            for (int i = 0; i < n_det_probes; i++) {
                const auto r = common_moe_bench_one_request_full(port, quality_prompts[0], 12, probe_seed, /* greedy */ true);
                if (r.predicted_n > 0) {
                    answers.push_back(r.text);
                }
            }
            int modal = 0;
            for (const auto & a : answers) {
                modal = std::max(modal, (int) std::count(answers.begin(), answers.end(), a));
            }
            // What this check is for: a forward pass reading memory it does not
            // own, which no throughput redeems. What it must NOT punish: the
            // nondeterminism this stack has by design. The expert cache admits
            // experts between requests, and an expert computed on the GPU and on
            // the CPU goes through different quantized dot products, so identical
            // greedy requests legitimately diverge by a word as the cache warms.
            //
            // Measured on Qwen3.8-Flash-Next: cache off gives 5 of 5 byte-identical
            // answers, cache on gives 1 of 5 - all five fluent and saying the same
            // thing. So with the cache on, this metric reports cache-warming state,
            // not correctness. Two earlier versions of this check both got it wrong:
            // demanding exact agreement (modal*2 <= n) rejected the substitution-free
            // baseline itself, and comparing against the first candidate's own count
            // rejected on a 2-vs-3 difference, which at n=5 is pure sampling noise.
            // Both rejected substitute-min-rank=2 - measured at 24.27 tok/s in the
            // ladder and 13.19 mean in a controlled A/B, the best configuration this
            // model has - and calibration then cached a slower one.
            //
            // Reject only the signature corruption actually leaves: no two of the
            // five runs agreeing at all. Benign cache warming converges, so at least
            // two runs match; a pass reading unowned memory produces a different
            // answer every time. Garbage that stays self-consistent is still caught
            // by the degeneracy check above, and drift that stays fluent by the
            // fidelity bar, which is measured against this model's own resampling
            // distance rather than assumed.
            const int repro_seen = g_moe_repro_floor.load();
            if (repro_seen < 0 && !answers.empty()) {
                g_moe_repro_floor.store(modal);
                LOG_INF("%s: greedy agreement for this run's first configuration: %d of %zu repeats\n",
                        __func__, modal, answers.size());
            }
            // Reported, never a veto. Three versions of this check rejected on
            // text identity and all three were wrong, because identity is not a
            // correctness property of an adaptive cache: coherent answers that
            // differ by a word are a correct forward pass, not a failure.
            //
            // The check also never did the job it was added for. The one real
            // corruption this fork has hit - the LFRU device-to-device path on
            // qwen4exp (a921f14f6) - produced *deterministic* garbage, "independent
            // of sampling settings", so an agreement test could not have caught it;
            // the degeneracy check did. Meanwhile it repeatedly threw out
            // substitute-min-rank=2, the best configuration this model has
            // (24.27 tok/s in the ladder, 13.19 mean in a controlled A/B), and
            // calibration cached a slower one in its place.
            //
            // So correctness is judged by what it is actually made of: degeneracy
            // above rejects broken generation, and the fidelity bar rejects drift
            // from the substitution-free reference using this model's own
            // resampling distance as the tolerance. Agreement stays as a logged
            // diagnostic, because a sudden collapse in it is still worth seeing.
            if (!answers.empty()) {
                LOG_INF("%s: greedy agreement %d of %zu (diagnostic - coherence is judged by the degeneracy "
                        "and fidelity checks, not by text identity)\n", __func__, modal, answers.size());
            }
        }
    } else {
        // Concurrent path: fire n_concurrency real requests at once (cycling
        // through the prompt pool so it's not the same prompt N times, which
        // would bias moe-cache/MTP toward an unrealistically easy repeated
        // pattern - see the endpoint-bias lesson elsewhere in this file),
        // measure wall-clock for all of them to complete, and report
        // aggregate tok/s = total generated tokens / wall time. This is the
        // actual quantity a concurrent deployment cares about, not the
        // average of what each individual request would have gotten alone.
        std::vector<double> predicted_n_per_req(n_concurrency, -1.0);
        std::vector<std::thread> threads;
        threads.reserve(n_concurrency);
        const auto t_start = std::chrono::steady_clock::now();
        for (int i = 0; i < n_concurrency; i++) {
            const char * prompt = probe_prompts[i % n_probe_prompts];
            threads.emplace_back([&predicted_n_per_req, i, port, prompt, n_predict]() {
                const auto [predicted_n, predicted_ms] = common_moe_bench_one_request(port, prompt, n_predict);
                predicted_n_per_req[i] = predicted_n;
                (void) predicted_ms; // wall-clock comes from the outer timer, not per-request timing
            });
        }
        for (auto & t : threads) {
            t.join();
        }
        const auto t_end = std::chrono::steady_clock::now();
        const double wall_s = std::chrono::duration<double>(t_end - t_start).count();

        double total_tokens = 0.0;
        int n_ok = 0;
        for (double predicted_n : predicted_n_per_req) {
            if (predicted_n > 0) {
                total_tokens += predicted_n;
                n_ok++;
            }
        }
        // Require every concurrent request to have succeeded - a partial
        // failure under real concurrency is itself a signal this candidate
        // can't actually sustain the target concurrency, not just noise to
        // average past.
        result_tps = (n_ok == n_concurrency && wall_s > 0) ? total_tokens / wall_s : -1.0;
    }

    if (keep_alive && !reuse && pid > 0 && result_tps > 0) {
        // Held open for the sweep that asked for it, and only on a real
        // measurement: a candidate that failed has nothing to reuse and would just
        // hold the GPU against every launch that follows.
        common_moe_live_stop();          // replace any previous live candidate
        g_moe_live_pid  = pid;
        g_moe_live_port = port;
        return result_tps;
    }
    cleanup();
    return result_tps;
}

static std::string common_self_exe_path() {
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) {
        return "";
    }
    buf[n] = '\0';
    return std::string(buf);
}

// --moe-calibrate: empirically finds the throughput-optimal -ncmoe and
// thread count for this exact GPU+model+context combination via a handful
// of real short benchmarks, caches the result, then the caller exits
// (matches the existing --fit-moe-cache preview-before-you-commit pattern
// rather than mixing calibration into the same run as real serving).
// -ncmoe is searched via golden-section search (interior-peak-aware, not a
// fixed grid). When a draft/MTP model is configured, spec-draft-n-max is
// also searched the same way, within an envelope found first by doubling
// until cache health collapses (see the n_max=8 finding in the docs - a
// real, measured collapse, not a guess at where the envelope ends).
//
// Concurrency-aware: if params.n_parallel > 1, every candidate is
// benchmarked with that many real concurrent requests (aggregate
// throughput), not solo decode speed - the optimal placement/threads/n_max
// can differ meaningfully at real concurrency (confirmed this session: the
// MMVQ and bulk-offload GPU-kernel-dispatch cliffs both only appear above
// certain concurrent batch sizes, invisible to a solo benchmark entirely).
// cparams.n_seq_max already equals params.n_parallel via
// common_context_params_to_llama, so the safe-floor probe below already
// accounts for the larger KV-cache footprint N concurrent slots need - no
// separate concurrency-specific probe required. Deliberately reuses
// --parallel itself as the concurrency target (not a separate flag): the
// calibration cache key already includes n_parallel, so calibrating and
// deploying with the same --parallel value is what makes the cache
// lookup find this entry later - a separate flag that could drift out of
// sync with --parallel would be a real footgun here.
static std::mutex                    g_moe_calib_status_mutex;
static std::string                   g_moe_calib_status_stage = "starting";
static std::chrono::steady_clock::time_point g_moe_calib_status_start = std::chrono::steady_clock::now();
static int                           g_moe_calib_status_done  = 0;
static int                           g_moe_calib_status_total = 0; // 0 = no estimate yet
static long long                     g_moe_calib_status_eta_s = -1; // frozen at each candidate_done(), not live

void common_moe_calibration_status_start() {
    std::lock_guard<std::mutex> lock(g_moe_calib_status_mutex);
    g_moe_calib_status_stage = "starting";
    g_moe_calib_status_start = std::chrono::steady_clock::now();
    g_moe_calib_status_done  = 0;
    g_moe_calib_status_total = 0;
    g_moe_calib_status_eta_s = -1;
}

void common_moe_calibration_status_set(const std::string & stage) {
    std::lock_guard<std::mutex> lock(g_moe_calib_status_mutex);
    g_moe_calib_status_stage = stage;
}

void common_moe_calibration_status_set_total(int total_candidates) {
    g_moe_planned_candidates.store(total_candidates, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(g_moe_calib_status_mutex);
    g_moe_calib_status_total = total_candidates;
}

static std::vector<common_moe_calibration_decision> g_moe_calib_decisions;

void common_moe_calibration_status_note(
        const std::string & lever, const std::string & value,
        const std::string & result, bool accepted, bool chosen) {
    // Stamp the regime the number was measured in.
    //
    // The stages do not all measure the same thing - some report prompt
    // throughput and some decode, some run with the draft attached and some
    // without, the ladder uses a short probe with reasoning off and the confirm
    // a long one with reasoning on and the answers checked. All of it then
    // lands in one table, where a reader (including me) compares rows that are
    // not comparable: this stage reported 10.17 tok/s for q8_0 KV measured with
    // no draft while the next stage reported 7.5 for the same setting with one,
    // and both are true about different machines.
    //
    // The flags that distinguish them already exist for other purposes, so the
    // label costs nothing and makes the table honest about what it is showing.
    std::string tagged = result;
    if (result.find("tok/s") != std::string::npos) {
        const bool prefill = g_moe_calib_measure_prefill.load();
        const bool drafted = g_moe_calib_prob_accept.load() >= 0 || !g_moe_calib_kv_type.empty();
        (void) drafted;
        if (prefill && result.find("prompt tok/s") == std::string::npos) {
            tagged += " [prompt]";
        } else if (!prefill) {
            tagged += " [decode]";
        }
    }
    std::lock_guard<std::mutex> lock(g_moe_calib_status_mutex);
    // Bounded: a long run on a slow model can produce a lot of these, and the
    // page only ever renders them. Oldest first out, newest kept.
    if (g_moe_calib_decisions.size() >= 64) {
        g_moe_calib_decisions.erase(g_moe_calib_decisions.begin());
    }
    if (chosen) {
        // Only one winner per lever: a later commit for the same lever
        // supersedes an earlier one (the confirmation step can step the
        // substitution floor back after the ladder already picked it).
        for (auto & d : g_moe_calib_decisions) {
            if (d.lever == lever) {
                d.chosen = false;
            }
        }
    }
    g_moe_calib_decisions.push_back({lever, value, tagged, accepted, chosen});
}

void common_moe_calibration_status_candidate_done() {
    std::lock_guard<std::mutex> lock(g_moe_calib_status_mutex);
    g_moe_calib_status_done++;
    // Recompute the ETA only here, at a real completion instant - not live
    // on every status poll, which would inflate the estimate every second
    // spent waiting on the current (still in-flight) candidate purely
    // because the numerator (elapsed) keeps growing while the denominator
    // (done) sits still until it actually finishes. Confirmed happening in
    // practice: watched a live run's ETA climb from 248m to 371m across a
    // single still-in-progress candidate before this fix.
    if (g_moe_calib_status_total > 0 && g_moe_calib_status_done < g_moe_calib_status_total) {
        const long long elapsed_s = (long long) std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - g_moe_calib_status_start).count();
        g_moe_calib_status_eta_s = (long long) (((double) elapsed_s / g_moe_calib_status_done) *
                (g_moe_calib_status_total - g_moe_calib_status_done));
        // Never promise longer than the time budget allows. The extrapolation
        // above assumes every remaining candidate gets measured, but the
        // budget cuts the run the moment it expires - so on a slow model the
        // naive number is not just pessimistic, it is describing a search
        // that will not be permitted to happen: a real run showed "~70m57s
        // left" on a 600s budget that would stop it inside 10 minutes.
        const long long deadline_ms = g_moe_calibrate_deadline_ms.load(std::memory_order_relaxed);
        if (deadline_ms != 0) {
            const long long remaining_budget_s = (deadline_ms - common_moe_steady_now_ms()) / 1000;
            g_moe_calib_status_eta_s = std::min(g_moe_calib_status_eta_s, std::max(0LL, remaining_budget_s));
        }
    } else {
        g_moe_calib_status_eta_s = -1;
    }
}

common_moe_calibration_status common_moe_calibration_status_get_struct() {
    common_moe_calibration_status s;
    std::chrono::steady_clock::time_point start;
    {
        std::lock_guard<std::mutex> lock(g_moe_calib_status_mutex);
        s.stage   = g_moe_calib_status_stage;
        start     = g_moe_calib_status_start;
        s.done    = g_moe_calib_status_done;
        s.total   = g_moe_calib_status_total;
        s.eta_s   = g_moe_calib_status_eta_s; // frozen at the last candidate_done() - see its comment
        s.decisions = g_moe_calib_decisions;
    }
    s.elapsed_s = (long long) std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start).count();
    return s;
}

std::string common_moe_calibration_status_get() {
    const common_moe_calibration_status s = common_moe_calibration_status_get_struct();

    std::string progress_note;
    if (s.total > 0) {
        // done can exceed total - the estimate is a single upfront guess
        // (see common_moe_calibrate), not a hard ceiling; a stage that
        // genuinely needed more candidates than guessed just shows 100%+
        // rather than a nonsensical negative "time left".
        const int pct = (int) std::lround(100.0 * std::min(s.done, s.total) / s.total);
        std::string eta;
        if (s.eta_s >= 0) {
            eta = string_format(", ~%lldm%02llds left", s.eta_s / 60, s.eta_s % 60);
        }
        progress_note = string_format(" [candidate %d/~%d, %d%%%s]", s.done, s.total, pct, eta.c_str());
    }

    char buf[420];
    snprintf(buf, sizeof(buf),
            "Calibrating MoE placement for maximum throughput: %s%s (elapsed %lldm%02llds). "
            "This runs once per hardware+model+context combination and is cached - a normal "
            "launch afterward (without --moe-calibrate) starts instantly using the cached result.",
            s.stage.c_str(), progress_note.c_str(), s.elapsed_s / 60, s.elapsed_s % 60);
    return buf;
}

// Rough evaluation-count estimate for common_golden_section_search_max over
// [lo, hi] - used only to size the status page's progress bar/ETA, not the
// search itself (which is exact and memoized regardless of this guess being
// off). Mirrors the real loop's shrink rate without running it.
static int common_golden_section_eval_estimate(int lo, int hi) {
    if (lo >= hi) {
        return 1;
    }
    const double gr = 1.618033988749895;
    int evals = 2; // initial x1, x2
    double width = hi - lo;
    while (width > 2.0) {
        width /= gr;
        evals++;
    }
    return evals + 3; // close-out linear scan of the final <=3-point range
}

static void common_enforce_moe_cache_parallel_limit(common_params & params, llama_context_params & cparams);

void common_moe_calibrate(common_params & params) {
    common_moe_calibration_status_start();
    const char * path_model = params.model.path.c_str();
    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);
    // Clamp concurrency BEFORE measuring anything. Serving applies this limit
    // on its own (the moe-cache corrupts output above it), so calibrating
    // above it measures a configuration that can never run - and worse, every
    // candidate carries the memory cost of slots that will be thrown away.
    // Observed on Qwen3.8-Flash-Next: calibration benchmarked at --parallel 4,
    // reported 0.17 aggregate tok/s, and then every -ngl candidate after the
    // first failed outright, while serving clamped the very same run to 2. The
    // cached entry described a concurrency the server refuses to use.
    common_enforce_moe_cache_parallel_limit(params, cparams);
    const int concurrency = std::max(1, (int) params.n_parallel);

    // Free device memory, sampled now - before any candidate server exists.
    // The expert-cache size ladder below is derived from this. It cannot be
    // read at that stage instead: by then a child server is holding the model
    // and the card looks nearly full, which would collapse the ladder to its
    // smallest rung for reasons that have nothing to do with this model.
    size_t calib_free_vram_bytes = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        size_t dfree = 0, dtotal = 0;
        ggml_backend_dev_memory(dev, &dfree, &dtotal);
        calib_free_vram_bytes = std::max(calib_free_vram_bytes, std::min(dfree, dtotal));
    }


    LOG_INF("%s: probing safe MoE CPU-offload floor for this GPU+model+context combination ...\n", __func__);
    common_moe_calibration_status_set("probing safe MoE CPU-offload floor");
    // Probe with the SAME margin common_maybe_raise_moe_for_ctx will demand at
    // serving time (3x the per-device fit target - see its own comment for why
    // a bare fit is not enough there). Probing with the bare-fit default here
    // instead put the search floor below anything serving would accept, so
    // calibration spent its entire budget measuring placements that were then
    // discarded: a real run measured ncmoe 46/45/44, picked 44 as the fastest,
    // and the fit guard immediately overrode it with "calibrated placement of
    // 44 CPU layer(s) is below the safe minimum for this context - keeping the
    // fit search's more conservative answer", serving 47. Every candidate in
    // that budget was unusable by construction.
    // Before deriving the margin: the draft's VRAM is real and this probe is
    // otherwise blind to it (calibration runs in server.cpp well before the
    // server's own load_model measures the draft). Folding it into the target
    // here makes the margin below cover the draft too, so the placements this
    // run benchmarks are placements that still fit once the draft is loaded.
    std::vector<size_t> draft_per_device;
    const size_t draft_bytes = common_measure_draft_memory(params, draft_per_device);
    if (draft_bytes > 0) {
        LOG_INF("%s: reserving %.2f MiB for the draft before probing\n",
                __func__, draft_bytes / (1024.0 * 1024.0));
    }

    // 3x the fit target is the serving margin (see below), and the draft is an
    // actual allocation on top of it - added once, not tripled with the margin.
    const int64_t fit_margin = 3 * (int64_t) params.fit_params_target[0] +
                               (int64_t) (draft_per_device.empty() ? 0 : draft_per_device[0]);
    common_moe_fit_probe_result probe = common_moe_find_safe_layers(path_model, mparams, cparams, fit_margin);
    if (!probe.is_moe) {
        LOG_WRN("%s: model has no MoE experts - nothing for --moe-calibrate to do\n", __func__);
        return;
    }

    uint32_t safe_n = probe.already_fits ? 0 : probe.safe_n;
    if (!probe.already_fits && !probe.found_safe_n) {
        // Halve the context until it fits, rather than giving up. Bailing here
        // meant a bare launch on a model whose default context is larger than
        // this card can hold skipped calibration entirely and served
        // unconfigured - observed on Qwen3.8-Flash-Next, where the whole
        // "point it at a model" path silently produced nothing because the
        // model's own default context does not fit a 12GB card even with
        // every expert on the CPU. The serving path already resolves this the
        // same way (it trades context for placement and says so); there is no
        // reason calibration should be the one component that refuses.
        //
        // Halving rather than a fine search on purpose: each step costs a real
        // probe, this only has to find a context that FITS so the actual
        // search can start, and the fit machinery downstream still gets the
        // final say on placement at whatever context this lands on.
        // Resolve both the way the rest of this file does. A bare launch
        // leaves n_ctx at 0 meaning "the model's own trained context", and
        // reading it raw made `requested / 2` zero, so the loops below never
        // executed even once - which is why this still reported "does not fit
        // even at 512" while an explicit -c 2048 --parallel 1 fitted
        // immediately. Same for n_seq_max.
        const uint32_t requested     = cparams.n_ctx     > 0 ? cparams.n_ctx     : 65536;
        const uint32_t requested_par = cparams.n_seq_max > 0 ? cparams.n_seq_max : 1;
        bool found = false;
        // Concurrency first, and context only after. On a hybrid model the
        // recurrent-state cache is sized by n_seq_max, not by n_ctx, so on
        // Qwen3.8-Flash-Next the allocation that actually failed was the rs
        // cache and halving the context could not have helped however far it
        // went - the first version of this fallback shrank the context to 512
        // and still gave up, which is what showed the variable was wrong.
        for (uint32_t try_par = requested_par / 2; try_par >= 1 && !found; try_par /= 2) {
            cparams.n_seq_max = try_par;
            probe = common_moe_find_safe_layers(path_model, mparams, cparams, fit_margin);
            if (probe.already_fits || probe.found_safe_n) {
                LOG_WRN("%s: %u concurrent slots do not fit this device even with every expert on the CPU "
                        "- calibrating at %u instead, which does. Pass --parallel explicitly to pin a "
                        "different number\n", __func__, requested_par, try_par);
                params.n_parallel = (int32_t) try_par;
                safe_n = probe.already_fits ? 0 : probe.safe_n;
                found = true;
            }
            if (try_par == 1) {
                break;
            }
        }
        // An explicit -c is a pin, not a hint. The halving below exists for the
        // n_ctx == 0 ("the model's own trained context") case, where nobody
        // asked for that size and silently calibrating smaller is better than
        // refusing outright. When the context WAS asked for, halving it
        // produces an entry for a context the user will never serve at, while
        // the log says "Pass -c explicitly to pin a different one" - advice
        // this loop then ignored, because it overrode the explicit -c too.
        //
        // The probe is also the wrong thing to defer to here. It demands the
        // full serving margin (3x the per-device fit target, so 3 GiB at the
        // default) on top of the model and KV, while the runtime treats the
        // expert cache as elastic and shrinks it to whatever is left - which
        // is why a context this probe rejects can and does serve. Measured on
        // Qwen3.8-Flash-Next: the probe refused 65536, and 65536 then served
        // at n_ctx_slot = 65536 with every expert on the CPU.
        //
        // So honour the pin and calibrate at the deepest offload, which is the
        // only placement such a context could use anyway. Say plainly that the
        // margin was not met, so a genuine shortage is still visible.
        if (!found && cparams.n_ctx > 0 && probe.n_layer > 0) {
            LOG_WRN("%s: the requested %u-token context does not leave the full serving margin, but -c "
                    "pinned it - calibrating at %u with every expert on the CPU rather than silently "
                    "calibrating for a context you did not ask for. Pass -c 0 to let the probe choose "
                    "a size that meets the margin\n", __func__, requested, requested);
            cparams.n_ctx = requested;
            params.n_ctx  = (int32_t) requested;
            safe_n        = probe.n_layer;
            found         = true;
        }
        for (uint32_t try_ctx = requested / 2; try_ctx >= 512 && !found; try_ctx /= 2) {
            cparams.n_ctx = try_ctx;
            probe = common_moe_find_safe_layers(path_model, mparams, cparams, fit_margin);
            if (probe.already_fits || probe.found_safe_n) {
                LOG_WRN("%s: the requested %u-token context does not fit this device even with every expert "
                        "on the CPU - calibrating at %u instead, which does. Pass -c explicitly to pin a "
                        "different one\n", __func__, requested, try_ctx);
                params.n_ctx = try_ctx;
                safe_n = probe.already_fits ? 0 : probe.safe_n;
                found = true;
                break;
            }
        }
        if (!found) {
            LOG_ERR("%s: config does not fit in available device memory even with all MoE experts on CPU, "
                    "one slot and a 512-token context; add VRAM before calibrating\n", __func__);
            return;
        }
    }

    const int n_threads_default = params.cpuparams.n_threads > 0 ? params.cpuparams.n_threads : common_cpu_get_num_math();

    // The entry is built as the run goes, not assembled at the end, and saved
    // after every stage that decides something. A run that is interrupted - by a
    // crash, a stopped process, or a spent budget - then keeps everything it had
    // already measured instead of losing all of it. Measured cost of this: one
    // small JSON write per stage against candidates that take tens of seconds.
    //
    // Safe to save half-finished by construction: every field defaults to -1,
    // "not calibrated", and the apply side skips those, so a partial entry means
    // "use what was measured, runtime defaults for the rest" rather than anything
    // half-configured.
    common_moe_calibration_entry entry;
    entry.concurrency = concurrency;

    // Resume, rather than start over.
    //
    // Every stage checkpoints what it measured, so an interrupted run already
    // keeps its results - but nothing READ them back, so the next run
    // re-measured everything the last one had already paid for. On this model
    // a single interrupted run had spent twenty minutes on the speculative
    // envelope alone before it was stopped.
    //
    // Seeding `entry` from the cached one turns every stage's own "already
    // measured?" test into the resume check, with no per-stage bookkeeping:
    // a field that is still -1 was never measured, and its stage runs.
    //
    // Guarded on gates_version, because an entry measured under different
    // quality gates is not a measurement of the same thing - that is the same
    // rule the apply side already enforces. GGML_MOE_CALIBRATE_FRESH=1 forces
    // a full re-measure.
    // "RESUMED - 64 tokens at 21.1 prompt tok/s (2026-09-12 01:34)" reads as a
    // measurement; "RESUMED" alone reads as a gap in the table.
    auto resumed_result = [](const std::string & what, double tps, const std::string & when) {
        GGML_UNUSED(what);
        // The result column carries the measurement; the value column already
        // carries the setting. Repeating the setting here fills the row
        // without adding anything - which is what an entry written before the
        // per-stage tps_* fields existed produced, since there was no
        // throughput to show and the value was echoed instead.
        //
        // When the number is missing, say so. An entry from an older run
        // genuinely does not know what its stages measured, and "no throughput
        // recorded" is information; the value repeated back is not.
        std::string out = "RESUMED";
        if (tps > 0.0) {
            out += string_format(" - %.2f tok/s", tps);
        } else {
            out += " - earlier run recorded no throughput for this stage";
        }
        if (!when.empty()) {
            out += " (measured " + when + ")";
        }
        return out;
    };
    bool resumed_any = false;
    {
        const char * fresh = getenv("GGML_MOE_CALIBRATE_FRESH");
        if (!(fresh && atoi(fresh) != 0)) {
            common_moe_calibration_entry prev;
            if (common_moe_calibration_lookup(path_model, params, prev) &&
                prev.gates_version == COMMON_MOE_CALIBRATION_GATES_VERSION &&
                prev.concurrency == concurrency) {
                const int keep_concurrency = entry.concurrency;
                entry = prev;
                entry.concurrency   = keep_concurrency;
                entry.gates_version = COMMON_MOE_CALIBRATION_GATES_VERSION;
                resumed_any = true;
                LOG_WRN("%s: resuming from the cached entry measured %s - stages it already decided are "
                        "skipped. Set GGML_MOE_CALIBRATE_FRESH=1 to re-measure everything\n",
                        __func__, prev.calibrated_at.c_str());
            }
        }
    }
    (void) resumed_any;
    // Seed the fields a partial entry would otherwise report as a measurement.
    // n_cpu_moe defaults to 0, which reads as "no layers on the CPU" rather than
    // "not calibrated", so a server loading a checkpoint mid-run applied a
    // placement nothing had chosen. -1 is the value the apply side skips.
    if (!resumed_any) {
        entry.n_cpu_moe = -1;
    }
    entry.gates_version = COMMON_MOE_CALIBRATION_GATES_VERSION;
    auto checkpoint = [&](const char * stage) {
        time_t ck_now = time(nullptr);
        char ck_buf[32];
        strftime(ck_buf, sizeof(ck_buf), "%Y-%m-%d %H:%M:%S", localtime(&ck_now));
        entry.calibrated_at = ck_buf;
        common_moe_calibration_save(path_model, params, entry);
        LOG_DBG("%s: checkpointed after %s\n", __func__, stage);
    };
    // 32, not 64: per-candidate cost is dominated by this probe on a
    // CPU-offloaded model (64 tokens at ~0.6 tok/s is ~107s, against ~25s to
    // spawn and load), and halving it roughly doubles how many levers fit in
    // the time budget. Still well above the degeneracy guard's 16-word floor,
    // so candidates that generate broken text are still caught.
    const int n_predict = 32;

    // Hard wall-clock budget for the whole run. Per-candidate cost varies by
    // orders of magnitude across models (a candidate is a full server spawn
    // plus a real generation probe: seconds on a small model that fits in
    // VRAM, ~90s+ on one decoding at ~1 tok/s), so a fixed candidate count
    // cannot bound runtime - the same search that finishes in a minute on
    // gemma-4 projected to 8-9 HOURS on qwen4exp, which is not a calibration
    // anyone waits through before their first chat.
    //
    // A deadline degrades gracefully where a smaller fixed count would not:
    // fast models still run the full search, slow ones get a truncated but
    // valid one, since stages run in priority order (placement first, then
    // the refinements) and each candidate is skipped once the budget is
    // spent. Skipped candidates report as failures, which every search here
    // already handles by keeping the best point actually measured.
    const bool budget_pinned_by_user = getenv("GGML_MOE_CALIBRATE_BUDGET_S") != nullptr;
    // Pinned by the user, or else the runaway guard: stages price themselves
    // (see common_moe_record_candidate_cost), so the global number only has to
    // stop a run that has gone wrong.
    const double calibration_budget_s = [] {
        if (const char * e = getenv("GGML_MOE_CALIBRATE_BUDGET_S")) {
            const double v = atof(e);
            if (v > 0.0) {
                return v;
            }
        }
        const char * m = getenv("GGML_MOE_CALIBRATE_BUDGET_MAX_S");
        const double cap = m ? atof(m) : 7200.0;
        return cap > 0.0 ? cap : 7200.0;
    }();
    g_moe_calibrate_budget_warned.store(false);
    const long long calibrate_start_ms = common_moe_steady_now_ms();
    g_moe_calibrate_start_ms.store(calibrate_start_ms);
    g_moe_calibrate_deadline_ms.store(
            calibrate_start_ms + (long long) (calibration_budget_s * 1000.0));
    g_moe_repro_floor.store(-1);
    g_moe_verifiable_ref.store(-1);
    g_moe_verifiable_ref2.store(-1);
    g_moe_verifying_reference.store(false);
    g_moe_calib_draft_cpu_moe.store(false);
    g_moe_calib_offload_min_batch.store(-1);
    g_moe_calib_measure_prefill.store(false);
    g_moe_calib_ubatch.store(-1);
    g_moe_calib_prefill_xlong.store(false);
    g_moe_calib_prefetch.store(false);
    g_moe_calib_prob_accept.store(-1);
    g_moe_calib_ring_pct.store(-1);
    common_moe_calib_set_env(std::string());

    // Train the prerouter from the first candidate of the run, not in a stage
    // of its own near the end.
    //
    // Learning is free of measurement bias in a way that acting is not: an SGD
    // step on a worker thread changes nothing about a candidate's throughput,
    // so every candidate pays the same negligible cost and none of them is
    // advantaged by running later. Acting - warming, eviction protection,
    // substitution - is what would bias a comparison, and that stays off
    // (GGML_CUDA_MOE_CACHE_PREDICTOR_ADMIT and the two influence weights all
    // default to 0) until the stages that measure it turn it on.
    //
    // The dedicated-stage version threw away every routing decision made
    // before it ran, which on a long run is most of them.
    {
        const std::string cache_dir = fs_get_cache_directory();
        const std::string base = path_model ? std::string(path_model) : std::string("model");
        const size_t slash = base.find_last_of("/\\");
        const std::string state = cache_dir + "predictor-" +
            (slash == std::string::npos ? base : base.substr(slash + 1)) + ".bin";
        // Same path serving reads, so what this run learns is what serving
        // starts from - and it accumulates across runs rather than being
        // rebuilt from nothing each time.
        common_moe_calib_set_base_env(string_format(
                "GGML_CUDA_MOE_CACHE_TRAIN_PREDICTOR=1 GGML_CUDA_MOE_CACHE_TRAIN_NEXT_LAYER=1 "
                "GGML_CUDA_MOE_CACHE_TRAIN_STATE_FILE=%s", state.c_str()));
        LOG_INF("%s: the prerouter trains on every candidate of this run, into %s - what it learns is "
                "kept, and it does not act until the stages that measure it say so\n", __func__, state.c_str());
    }

    g_moe_candidate_ms_sum.store(0);
    g_moe_candidate_count.store(0);
    {
        std::lock_guard<std::mutex> lock(g_moe_cost_mu);
        g_moe_recent_candidate_ms.clear();
    }
    g_moe_planned_candidates.store(0);
    g_moe_budget_is_derived.store(!budget_pinned_by_user);
    LOG_INF("%s: time budget for this run: %.0fs (%s)\n", __func__, calibration_budget_s,
            budget_pinned_by_user ? "GGML_MOE_CALIBRATE_BUDGET_S"
                                  : "runaway guard only - each stage is priced from its own measured candidates");

    const std::string self_exe = common_self_exe_path();
    if (self_exe.empty()) {
        LOG_ERR("%s: could not resolve /proc/self/exe - --moe-calibrate needs this to spawn benchmark "
                "subprocesses (real llama-server instances, so benchmarks go through the same chat-template "
                "and sampling path a real client would use, not a raw/unformatted prompt)\n", __func__);
        return;
    }
    // A fresh port per subprocess, not one reused across the whole run:
    // SIGKILLing a candidate's server doesn't guarantee the OS releases its
    // listening socket before the next candidate tries to bind the same
    // port (TIME_WAIT) - reusing one port caused a real intermittent
    // "failed" candidate when this was tested.
    int port_counter = 18900 + (int) (getpid() % 500);
    auto next_port = [&]() { return port_counter++; };
    const uint32_t ctx = cparams.n_ctx > 0 ? cparams.n_ctx : 65536;

    // Two independent samples per candidate, averaged - a single subprocess
    // run is one noisy sample, and this isn't a theoretical concern: a
    // documented n_max=3 run-to-run swing of 65% (42.06 vs 69.21 tok/s)
    // elsewhere in this history, and a real thread-tuning decision that
    // shipped on a single n_threads=12 sample of 48.71 tok/s when the true
    // 3-repeat mean was 61.82 - an outlier large enough to flip a closer
    // decision, confirmed by re-measuring after the fact. Each sample gets
    // its own one-retry-on-failure (a launch failure is a different problem
    // than ordinary timing noise); a sample that still fails after its
    // retry is dropped from the average rather than failing the whole
    // candidate, so one bad launch doesn't cost two good measurements.
    //
    // Reduced from 2 to 1 on a deliberate speed-for-noise trade (chosen by
    // the user, not a default): with this model decoding at ~1 tok/s, each
    // extra sample costs several real minutes, and calibration's own search
    // ranges grew (the -ngl search above did not exist when the 2-sample
    // default and the swings cited above were measured). Single-sample
    // candidates reopen exactly the outlier risk those swings describe -
    // still guarded by the retry-on-failure below (a launch failure, not
    // ordinary timing noise) and by golden-section's own "always trust the
    // best of what was truly measured" check, but with no averaging to
    // absorb a plain bad-luck sample within one candidate.
    constexpr int n_samples_per_candidate = 1;
    // The substitution floor decided so far, carried into every later stage's
    // candidates. Stages are supposed to build on each other's decisions, not
    // each measure a different machine: without this the ladder settles on a
    // rank, and then the -ngl, thread, cache-size and fit-margin stages all
    // benchmark with substitution back at the runtime default. Seen plainly on
    // gemma-4 - the ladder peaked at 42.70 tok/s and every stage after it
    // measured ~36, not because anything degraded but because they had quietly
    // dropped the setting the ladder had just established, and then tuned
    // cache size against a configuration that is not the one being served.
    int active_min_rank = -1;
    // The best configuration so far, measured with the SAME short probe the
    // later stages use for their candidates. Distinct from best_tps, which after
    // the substitution ladder holds a full-length confirmed number (128 tokens,
    // reasoning on). Comparing a 32-token no-reasoning candidate against that is
    // apples to oranges and the incumbent always wins: measured on
    // Qwen3.8-Flash-Next, all five stand-in sigma rungs (7.90-8.68) and every
    // -ngl rung (6.08-7.99) lost to a confirmed 11.53 and none could ever have
    // won, whatever their merit.
    double cheap_incumbent_tps = 0.0;
    // Carried into the later stages the same way active_min_rank is, so
    // everything measured after this point is measured at the balance that
    // will actually be served.
    double active_quality_sigma = std::numeric_limits<double>::quiet_NaN();
    // Same reasoning for the GPU-resident layer count: it is decided by its
    // own stage and then every later stage passed a hardcoded 99 (full
    // residency), so thread count, cache size and fit margin were all tuned
    // against a layer placement the server will not be using. 99 remains the
    // pre-decision default, matching the parameter's own default.
    int active_ngl = 99;
    auto bench_one_sample = [&](uint32_t n_cpu_moe, int n_max, const std::string & mtp_path, int n_threads) -> double {
        const int pa = g_moe_calib_prob_accept.load();
        double tps = common_moe_bench_candidate_server(
                self_exe, path_model, mtp_path, n_cpu_moe, n_max, n_threads, next_port(), ctx, n_predict, concurrency,
                -1, -1, active_ngl, active_min_rank, nullptr, 1234,
                std::numeric_limits<double>::quiet_NaN(), false, false, std::string(), pa);
        // Retry only an infrastructure failure - a quality rejection is deterministic.
        if (tps < 0 && tps != COMMON_MOE_TPS_REJECTED && tps != COMMON_MOE_TPS_LAUNCH_FAILED) {
            LOG_WRN("%s:   candidate sample failed, retrying ...\n", __func__);
            tps = common_moe_bench_candidate_server(
                    self_exe, path_model, mtp_path, n_cpu_moe, n_max, n_threads, next_port(), ctx, n_predict, concurrency,
                    -1, -1, active_ngl, active_min_rank, nullptr, 1234,
                    std::numeric_limits<double>::quiet_NaN(), false, false, std::string(), pa);
        }
        return tps;
    };
    auto bench_with_retry = [&](uint32_t n_cpu_moe, int n_max, const std::string & mtp_path, int n_threads) -> double {
        // The time budget is enforced inside common_moe_bench_candidate_server
        // itself - the one function every stage's candidates funnel through,
        // including the several that bypass this lambda entirely.
        double sum = 0.0;
        int n_ok = 0;
        for (int i = 0; i < n_samples_per_candidate; i++) {
            const double tps = bench_one_sample(n_cpu_moe, n_max, mtp_path, n_threads);
            if (tps > 0) {
                sum += tps;
                n_ok++;
            }
        }
        return n_ok > 0 ? sum / n_ok : -1.0;
    };

    // Search range: from the safe floor up to a bound past which more
    // conservative placement essentially never helps (our own sweep never
    // found the peak more than ~15% of n_layer above the floor) - wide
    // enough to contain the peak, narrow enough that golden-section search
    // stays cheap.
    uint32_t ncmoe_hi = std::min<uint32_t>(probe.n_layer, safe_n + std::max<uint32_t>(4, probe.n_layer / 8));

    const uint32_t ncmoe_span = ncmoe_hi > safe_n ? ncmoe_hi - safe_n : 0;
    static const uint32_t ncmoe_search_min_span = [] {
        const char * e = getenv("GGML_MOE_CALIBRATE_NCMOE_MIN_SPAN");
        const long v = e ? atol(e) : 3;
        return (uint32_t) (v >= 0 ? v : 3);
    }();
    // Single upfront, deliberately generous estimate of the total candidate
    // count for the whole calibration run (ncmoe search, the -ngl search and
    // its possible ncmoe re-search, spec-draft-n-max if MTP is configured,
    // plus the fixed-size thread/cache/fit-margin stages later) - purely for
    // the status page's progress bar and ETA. Not recomputed per branch
    // (whether -ngl actually wins and triggers a re-search, whether a
    // boundary extends, whether a fit-margin candidate fails early): erring
    // generous means "time left" only ever counts down, never climbs back up
    // as stages run, at the cost of finishing a bit before it reaches 100%.
    {
        // Placement contributes one baseline measurement when the fit probe
        // fixed it (the common case on a tightly-constrained model), not a
        // whole search - counting the search anyway made the bar claim ~34
        // candidates for a run that only ever intended to measure a handful.
        const bool ncmoe_will_search = ncmoe_span >= ncmoe_search_min_span;
        int est = ncmoe_will_search
                ? common_golden_section_eval_estimate((int) safe_n, (int) ncmoe_hi)
                : 1;
        est += 7; // substitution-floor ladder (6 rungs, 1 load) + long-probe confirmation
        est += 5; // offload threshold: 32, 64, 128, 256, 400 - one load, live thereafter
        est += 3; // prompt micro-batch: 512, 2048, then expert prefetch on at the winner
        est += 4; // prediction ring: 0, 5, 10 percent (one load) + the winner answer-checked
        est += 4; // stand-in selection: heat / atlas+fallback / atlas-only + answer check
        est += 3; // atlas prewarm: off, top_k 4, top_k 16
        est += 5; // neuron subsetting: off/128/256/512, then the winner answer-checked
        est += 4; // off-by-default cache features: incumbent + group admit, coverage evict, host buffer
        // The constant sweep shares ONE server: 1 launch + 15 live measurements +
        // 5 settle calls. Still 21 units of the budget's time accounting, but only
        // one model load, which is ~80% of what a launched candidate costs.
        // The tuning-constant sweep only runs when explicitly asked for - its
        // effects are inside this machine's noise floor - so it must not be in
        // the plan by default. Counting it anyway shrinks every stage's share
        // of the budget by about a fifth, and a stage that hits its share early
        // stops and reports its remaining candidates as failures, which is the
        // conflation this file fixed elsewhere today.
        {
            const char * e = getenv("GGML_MOE_CALIBRATE_TUNE_CONSTANTS");
            if (e && atoi(e) != 0) {
                est += 21; // 1 launch, 5 knobs x 3 values, 5 settles
            }
        }
        est += 3;  // KV precision: f16, q8_0, and the draft-attached re-check
        est += 3;  // rank-weighted admission: 0, 2x, 4x
        est += 3;  // verify substitution: generation floor, rank 2, every rank
        est += 3;  // budget partition: the arbiter's feasible splits
        est += 2;  // prerouter input shapes: this layer, next layer
        est += 2;  // trained prerouter: off, on
        est += 3;  // prerouter influence: inert, the role, the falsifier
        est += 2;  // final validation: the combination, with and without the prerouter
        est += 2; // answer bar: substitution-off reference, twice
        const uint32_t ngl_hi_est = (uint32_t) probe.n_layer + 1;
        const uint32_t ngl_lo_est = probe.n_layer > 3 ? probe.n_layer - probe.n_layer / 3 : 0;
        est += common_golden_section_eval_estimate((int) ngl_lo_est, (int) ngl_hi_est); // -ngl search
        if (ncmoe_will_search) {
            est += common_golden_section_eval_estimate((int) safe_n, (int) ncmoe_hi);   // possible ncmoe re-search at new -ngl
        }
        if (params.speculative.has_dft()) {
            est += 6;                                                // spec-draft-n-max envelope doubling
            est += 5;                                                 // no-draft baseline, placement, stand-ins, 2 cache shares
            est += 2;                                                 // depth winner + runner-up measured again
            est += 2;                                                 // depth re-check after substitution (top two)
            est += common_golden_section_eval_estimate(1, 32);        // spec-draft-n-max golden-section
            est += 2;                                                 // spec-prob-accept: off, on
            est += 2;                                                 // drafter cascade: mtp alone, ngram-suffix in front
            est += 4;                                                 // confidence gate: p_min 0.5/0.8 x depth d/2d
        }
        est += 2; // thread-count candidates
        est += 3; // admission-delay candidates (1, 2, 4)
        // Expert-cache size knee. Counted from the ladder that will actually run
        // (off, then 512 MiB doubling up to measured free VRAM), not the fixed
        // list of 8 this used to assume - four of those rungs were above what a
        // 12 GiB card can ever grant, so the estimate overstated the work, and
        // since the time budget is derived from this number it overstated the
        // budget too.
        {
            int cache_rungs = 1; // "off" is always measured
            for (size_t mb = 512; mb <= (calib_free_vram_bytes >> 20); mb *= 2) {
                cache_rungs++;
            }
            est += cache_rungs;
        }
        est += 4; // fit-margin search (upper bound; stops early on first failure)
        common_moe_calibration_status_set_total(est);
    }

    // Placement is a feasibility question first and a throughput question a
    // distant second, so a narrow feasible range is decided by the (free,
    // no-alloc) fit probe rather than benchmarked. Measured on qwen4exp: once
    // the probe used the same margin serving demands, the whole range was
    // [47, 48] and the two options differed by 3% (0.61 vs 0.59 tok/s) - yet
    // benchmarking them consumed 7 of a 9-minute budget, starving the
    // substitution search, whose own range spans 20x. Spending the budget on
    // the levers that actually move is the entire point of having one.
    //
    // The threshold is on candidate COUNT, not on any measured spread: the
    // spread is only knowable by paying for the very benchmarks this is
    // deciding whether to skip. Above it the search runs as before, since a
    // genuinely wide range can hide a real optimum.
    // No complete expert layer resident in VRAM. Every layer's experts go to
    // the CPU, and only the MTP head stays.
    //
    // This is a design decision rather than a measurement, and the arithmetic
    // says why. A resident layer costs 1.15 GiB and buys all 512 of that
    // layer's experts - hot and cold alike - while the same 1.15 GiB spent as
    // cache holds about 509 experts chosen by demand, across all 48 layers.
    // With 10 of 512 firing per token, the cache's selection is worth far more
    // than the layer's completeness: measured hit rate is 29% at 14%
    // residency, so routing is roughly twice as concentrated as uniform, and
    // the cache is what exploits that.
    //
    // So the placement search is not a search. The experts stream, the hottest
    // are held, and the VRAM that would have pinned one layer's worth of
    // mostly-cold weights goes to the MTP head and the cache instead.
    // GGML_MOE_CALIBRATE_SEARCH_NCMOE=1 restores the search for a card where
    // a resident layer might actually pay.
    const bool search_ncmoe = [] {
        const char * e = getenv("GGML_MOE_CALIBRATE_SEARCH_NCMOE");
        return e && atoi(e) != 0;
    }();
    if (!search_ncmoe) {
        LOG_WRN("%s: no expert layer resident - all %u layers' experts on the CPU, only the MTP head and "
                "the hottest experts in VRAM. A resident layer costs 1.15 GiB for all 512 of its experts; "
                "the same bytes as cache hold ~509 chosen by demand across every layer "
                "(GGML_MOE_CALIBRATE_SEARCH_NCMOE=1 to search placements instead)\n",
                __func__, probe.n_layer);
        common_moe_calibration_status_note("expert placement",
                string_format("ncmoe %u - none resident", probe.n_layer),
                "DESIGN - the cache holds experts chosen by demand, not a layer's worth of mostly-cold ones",
                true, true);
    } else if (ncmoe_span < ncmoe_search_min_span) {
        LOG_INF("%s: MoE CPU-offload depth fixed at %u by the fit probe - only %u placement(s) fit this "
                "context with the serving margin, not enough spread to be worth benchmarking; spending the "
                "budget on the levers that move instead\n", __func__, safe_n, ncmoe_span + 1);
    }

    if (search_ncmoe && ncmoe_span >= ncmoe_search_min_span) {
        LOG_INF("%s: golden-section search for -ncmoe in [%u, %u] at n_threads=%d%s (real llama-server subprocess, "
                "chat-templated prompts, per candidate) ...\n", __func__, safe_n, ncmoe_hi, n_threads_default,
                concurrency > 1 ? string_format(", concurrency=%d (aggregate throughput)", concurrency).c_str() : "");
        common_moe_calibration_status_set(string_format("searching MoE CPU-offload depth (-ncmoe) in [%u, %u]", safe_n, ncmoe_hi));
    } else {
        common_moe_calibration_status_set(string_format("measuring baseline at the probe-fixed placement (-ncmoe %u)", safe_n));
    }

    std::map<int, double> ncmoe_trace;
    auto measure_ncmoe = [&](int n) -> double {
        // Memoized: the boundary-extension search below re-probes points the
        // first search already measured, and a sample costs a full server spawn.
        auto it = ncmoe_trace.find(n);
        if (it != ncmoe_trace.end()) {
            return it->second;
        }
        const double tps = bench_with_retry((uint32_t) n, 0, "", n_threads_default);
        LOG_INF("%s:   ncmoe=%d -> %s\n", __func__, n, tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
        common_moe_calibration_status_candidate_done();
        return tps; // failed candidates measure as -1, golden-section still works (just avoids them)
    };
    uint32_t best_n;
    double best_tps;
    if (!search_ncmoe) {
        // Every layer's experts on the CPU - the design rule above. Measured
        // once so later stages have a real baseline to compare against.
        best_n   = probe.n_layer;
        best_tps = measure_ncmoe((int) best_n);
    } else if (ncmoe_span < ncmoe_search_min_span) {
        // Probe-decided (see the comment above): take the fit probe's own
        // floor and measure it ONCE, so later stages still have a real
        // baseline tok/s to compare against without paying for a search.
        best_n   = safe_n;
        best_tps = measure_ncmoe((int) safe_n);
    } else {
        best_n   = (uint32_t) common_golden_section_search_max((int) safe_n, (int) ncmoe_hi, measure_ncmoe, ncmoe_trace);
        best_tps = ncmoe_trace.at((int) best_n);
    }
    // Golden-section search assumes a unimodal objective and only ever
    // bisects toward whichever side of its *current* probe pair scores
    // higher - it never revisits a point once the search has narrowed
    // away from it, even if that point is sitting right there in the
    // trace map already measured. Confirmed to matter in practice for
    // the spec-draft-n-max search below (see docs/moe-cache-colibri-notes.md,
    // "SECOND BUG FOUND": n_max=2 scored higher than the declared winner
    // n_max=9, silently ignored because the search had already bisected
    // past it). This costs zero extra subprocess spawns - ncmoe_trace
    // already holds every point actually measured - so always trust the
    // best of what was truly measured over what golden-section converged
    // to, not just for n_max but here too as a general safety net.
    for (const auto & kv : ncmoe_trace) {
        if (kv.second > best_tps) {
            best_tps = kv.second;
            best_n = (uint32_t) kv.first;
        }
    }
    if (best_tps < 0) {
        LOG_ERR("%s: every placement candidate failed to benchmark; not writing a cache entry\n", __func__);
        return;
    }

    // The initial hi bound is a heuristic ("the peak is never more than ~15%
    // of n_layer above the safe floor"), and a heuristic that is wrong shows
    // itself in exactly one way: the winner sits *on* the boundary, meaning
    // throughput was still climbing when the search ran out of room. Measured
    // on Nemotron 3.5 Lightning, where [37,43] returned 43 with the trace
    // rising monotonically to it - because on this model extra CPU offload
    // frees VRAM that the expert cache converts straight back into hit rate.
    // Extend to the hard ceiling and keep going rather than shipping a bound
    // artifact as an optimum.
    if (best_n == ncmoe_hi && ncmoe_hi < (uint32_t) probe.n_layer) {
        const uint32_t prev_hi = ncmoe_hi;
        ncmoe_hi = (uint32_t) probe.n_layer;   // the hard ceiling: one extension is all there is
        LOG_INF("%s: peak landed on the search boundary (%u) - throughput was still rising, "
                "extending the range to [%u, %u]\n", __func__, prev_hi, prev_hi, ncmoe_hi);
        common_golden_section_search_max((int) prev_hi, (int) ncmoe_hi, measure_ncmoe, ncmoe_trace);
        for (const auto & kv : ncmoe_trace) {
            if (kv.second > best_tps) {
                best_tps = kv.second;
                best_n   = (uint32_t) kv.first;
            }
        }
    }

    // Offload threshold. It decides, per MoE op, between two batch paths: at or
    // above it the op goes to the GPU with its routed experts copied over PCIe from
    // the host mapping; below it the op stays on the CPU, where the expert cache
    // serves resident experts from VRAM. The backend default of 32 is an upstream
    // number chosen for models that fit in memory. Measured on prompt processing,
    // the only thing that exercises it - decode is one token and never crosses it.
    // Candidates stop at 400: at the cache's 4096-row limit and ~10 experts per
    // token, a larger chunk cannot be served by the cache anyway.
    // The two prefill stages - offload threshold and micro-batch - between them
    // cost seven of the run's most expensive candidates (each processes an
    // extra-long prompt) to decide two knobs that then govern every DECODE
    // stage after them. That is the exact inheritance bug this file has hit
    // repeatedly, paid for at the highest price per candidate in the run. Both
    // now take the value the server was launched with, or a measured default.
    const bool sweep_prefill = [] {
        const char * e = getenv("GGML_MOE_CALIBRATE_SWEEP_PREFILL");
        return e && atoi(e) != 0;
    }();

    int best_offload_min_batch = -1;
    if (entry.op_offload_min_batch > 0) {
        best_offload_min_batch = entry.op_offload_min_batch;
        // Carry it to the candidates, not just to the entry. A resumed value
        // that is recorded but never applied means the run measures one
        // configuration and saves another.
        g_moe_calib_offload_min_batch.store(best_offload_min_batch);
        LOG_WRN("%s: resuming - offload threshold %d already measured, skipping that stage\n",
                __func__, best_offload_min_batch);
        // Show it. A resumed value is still this run's answer, and a decisions
        // table with a hole where a lever should be reads as "never measured"
        // rather than "measured earlier".
        common_moe_calibration_status_note("offload threshold",
                string_format("%d tokens", best_offload_min_batch),
                resumed_result(string_format("%d tokens", best_offload_min_batch),
                               entry.tps_offload_min_batch, entry.calibrated_at), true, true);
    } else if (!sweep_prefill) {
        // A prefill knob, decided on prefill numbers, then inherited by every
        // decode stage after it - the shape of bug this file has hit most. 256
        // has won or tied on every model measured here, and the five candidates
        // it costs are five of the most expensive in the run (they all process
        // an extra-long prompt). GGML_MOE_CALIBRATE_SWEEP_PREFILL=1 measures it.
        best_offload_min_batch = 256;
        g_moe_calib_offload_min_batch.store(best_offload_min_batch);
        entry.op_offload_min_batch = best_offload_min_batch;
        LOG_WRN("%s: offload threshold at 256 by default - a prefill knob whose five candidates are the "
                "run's most expensive, and which has not moved across models here "
                "(GGML_MOE_CALIBRATE_SWEEP_PREFILL=1 to measure it)\n", __func__);
        common_moe_calibration_status_note("offload threshold", "256 tokens",
                "DEFAULT - unchanged across every model measured here", true, true);
        checkpoint("offload threshold");
    } else if (!common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: measuring the offload threshold (GGML_OP_OFFLOAD_MIN_BATCH) on prompt processing ...\n", __func__);
        common_moe_calibration_status_set("measuring the MoE offload threshold");
        common_moe_stage_begin("offload threshold", 5);
        g_moe_calib_measure_prefill.store(true);
        double best_pp = -1.0;
        // One server for all five. The threshold is read per op-assignment through
        // the tunables registry (see ggml_backend_cuda_device_offload_op), so a
        // POST changes the very next graph rather than needing a reload.
        bool offload_live = false;
        for (const int threshold : { 32, 64, 128, 256, 400 }) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            g_moe_calib_offload_min_batch.store(threshold);
            const std::string body = string_format("{\"GGML_OP_OFFLOAD_MIN_BATCH\": \"%d\"}", threshold);
            double pp = -1.0;
            if (offload_live && g_moe_live_port > 0) {
                pp = common_moe_bench_candidate_server(
                        self_exe, path_model, "", best_n, 0, n_threads_default, g_moe_live_port, ctx,
                        n_predict, concurrency, -1, -1, active_ngl, active_min_rank,
                        nullptr, 1234, std::numeric_limits<double>::quiet_NaN(), false, false,
                        std::string(), -1, std::string(), -1.0,
                        g_moe_live_port, body, /* keep_alive */ true);
            } else {
                pp = common_moe_bench_candidate_server(
                        self_exe, path_model, "", best_n, 0, n_threads_default, next_port(), ctx,
                        n_predict, concurrency, -1, -1, active_ngl, active_min_rank,
                        nullptr, 1234, std::numeric_limits<double>::quiet_NaN(), false, false,
                        std::string(), -1, std::string(), -1.0,
                        -1, std::string(), /* keep_alive */ true);
                offload_live = g_moe_live_port > 0;
            }
            common_moe_calibration_status_candidate_done();
            LOG_INF("%s:   offload threshold %d -> %s\n", __func__, threshold,
                    pp > 0 ? string_format("%.1f prompt tok/s", pp).c_str() : "failed");
            common_moe_calibration_status_note("offload threshold", string_format("%d", threshold),
                    pp > 0 ? string_format("%.1f prompt tok/s", pp) : std::string("failed"), pp > 0);
            if (pp > best_pp) {
                best_pp                = pp;
                best_offload_min_batch = threshold;
            }
        }
        common_moe_live_stop();
        common_moe_stage_end();
        g_moe_calib_measure_prefill.store(false);
        g_moe_calib_offload_min_batch.store(best_offload_min_batch);
        entry.op_offload_min_batch = best_offload_min_batch;
        entry.tps_offload_min_batch = best_pp;   // what the winner actually measured
        checkpoint("offload threshold");
        if (best_offload_min_batch > 0) {
            LOG_INF("%s: offload threshold: %d at %.1f prompt tok/s\n", __func__, best_offload_min_batch, best_pp);
            common_moe_calibration_status_note("offload threshold", string_format("%d", best_offload_min_batch),
                    string_format("SELECTED - %.1f prompt tok/s", best_pp), true, true);
        }
    }

    // Micro-batch size, at the chosen offload threshold. Measured on a prompt of
    // ~1500 tokens - several micro-batches - because below one micro-batch the
    // knob has nothing to act on.
    int best_ubatch   = -1;
    int best_prefetch = -1;
    if (entry.n_ubatch > 0) {
        best_ubatch   = entry.n_ubatch;
        best_prefetch = entry.sched_prefetch_experts;
        g_moe_calib_ubatch.store(best_ubatch);
        if (best_prefetch >= 0) {
            g_moe_calib_prefetch.store(best_prefetch != 0);
        }
        LOG_WRN("%s: resuming - prompt micro-batch -ub %d and expert prefetch already measured, "
                "skipping that stage\n", __func__, best_ubatch);
        common_moe_calibration_status_note("prompt micro-batch",
                string_format("-ub %d", best_ubatch),
                resumed_result(string_format("-ub %d", best_ubatch),
                               entry.tps_ubatch, entry.calibrated_at), true, true);
        if (best_prefetch >= 0) {
            common_moe_calibration_status_note("expert prefetch",
                    best_prefetch ? "on" : "off",
                    resumed_result(best_prefetch ? "on" : "off", -1.0, entry.calibrated_at), true, true);
        }
    } else if (!sweep_prefill) {
        // -ub is a warm-up lever, not a throughput lever: it sets how much of
        // the prompt crosses the PCIe bus at once, and the stage that measured
        // it reported prompt tok/s - a number none of the decode stages that
        // then inherit it are about. Take what the server was launched with.
        best_ubatch = (int) params.n_ubatch;
        g_moe_calib_ubatch.store(best_ubatch);
        entry.n_ubatch = best_ubatch;
        // Prefetch was never actually decided by its own candidate either - the
        // stage measured it, logged the verdict, then stored true regardless,
        // because the decode stages after it rely on that path existing. A
        // candidate whose result cannot change the outcome is not a measurement.
        best_prefetch = 1;
        g_moe_calib_prefetch.store(true);
        entry.sched_prefetch_experts = 1;
        LOG_WRN("%s: prompt micro-batch left at the launch value (-ub %d) and expert prefetch on - both "
                "were decided on prefill numbers that the decode stages after them are not about "
                "(GGML_MOE_CALIBRATE_SWEEP_PREFILL=1 to measure them)\n", __func__, best_ubatch);
        common_moe_calibration_status_note("prompt micro-batch", string_format("-ub %d", best_ubatch),
                "LAUNCH VALUE - a warm-up lever, measured on prefill, inherited by decode stages", true, true);
        common_moe_calibration_status_note("expert prefetch", "on",
                "DEFAULT - the decode stages rely on the path; its candidate could never change that",
                true, true);
        checkpoint("prompt micro-batch");
    } else if (!common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: measuring the prompt micro-batch size (-ub) on a long prompt ...\n", __func__);
        common_moe_calibration_status_set("measuring the prompt micro-batch size");
        g_moe_calib_measure_prefill.store(true);
        g_moe_calib_prefill_xlong.store(true);
        double best_pp = -1.0;
        static const int ubatch_candidates[] = { 512, 2048 };
        for (const int ub : ubatch_candidates) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            g_moe_calib_ubatch.store(ub);
            const double pp = common_moe_bench_candidate_server(
                    self_exe, path_model, "", best_n, 0, n_threads_default, next_port(), ctx, n_predict,
                    concurrency, -1, -1, active_ngl, active_min_rank);
            common_moe_calibration_status_candidate_done();
            LOG_INF("%s:   -ub %d -> %s\n", __func__, ub,
                    pp > 0 ? string_format("%.1f prompt tok/s", pp).c_str() : "failed");
            common_moe_calibration_status_note("prompt micro-batch", string_format("-ub %d", ub),
                    pp > 0 ? string_format("%.1f prompt tok/s", pp) : std::string("failed"), pp > 0);
            if (pp > best_pp) {
                best_pp     = pp;
                best_ubatch = ub;
            }
        }
        g_moe_calib_ubatch.store(best_ubatch);
        entry.n_ubatch = best_ubatch;
        entry.tps_ubatch = best_pp;
        checkpoint("prompt micro-batch");
        if (best_ubatch > 0) {
            LOG_INF("%s: prompt micro-batch: -ub %d at %.1f prompt tok/s\n", __func__, best_ubatch, best_pp);
            common_moe_calibration_status_note("prompt micro-batch", string_format("-ub %d", best_ubatch),
                    string_format("SELECTED - %.1f prompt tok/s", best_pp), true, true);
        }
        // Expert prefetch at the chosen micro-batch - an existing scheduler
        // feature, off by default, that only acts on offloaded ops (so after the
        // threshold, and with a prompt long enough to cross it).
        if (best_ubatch > 0 && !common_moe_calibrate_budget_spent()) {
            g_moe_calib_prefetch.store(true);
            const double pp = common_moe_bench_candidate_server(
                    self_exe, path_model, "", best_n, 0, n_threads_default, next_port(), ctx, n_predict,
                    concurrency, -1, -1, active_ngl, active_min_rank);
            common_moe_calibration_status_candidate_done();
            LOG_INF("%s:   expert prefetch on -> %s (off: %.1f prompt tok/s)\n", __func__,
                    pp > 0 ? string_format("%.1f prompt tok/s", pp).c_str() : "failed", best_pp);
            // Provisional, and left ON for the stages that follow.
            //
            // This measures prompt throughput, which is the only thing
            // scheduler prefetch acts on - but the decision it writes applies
            // to every candidate for the rest of the run, including the
            // decode-measured stages that depend on the prefetch path
            // existing. Measured here: prefetch scored 42.3 prompt tok/s
            // against 48.5 off, so it was disabled - and the substitution
            // ladder, the ring, the prerouter and the partition were then all
            // measured with a mechanism they rely on switched off, on the
            // strength of a prefill number none of them are about.
            //
            // So the verdict is recorded and the mechanism is left enabled.
            // Prefetch costs prefetch_n_slots (3) max-sized expert tensors of
            // device memory, which is single-digit MiB on this model - cheap
            // enough that leaving it on through the search is the safer error
            // than turning it off, and the final validation re-measures the
            // saved combination either way.
            best_prefetch = pp > best_pp ? 1 : 0;
            entry.sched_prefetch_experts = best_prefetch;
            checkpoint("expert prefetch");
            if (best_prefetch == 0) {
                LOG_WRN("%s: expert prefetch measured slower on prompt processing, but it is left ON for "
                        "the rest of the search - the decode stages after this one rely on that path, and "
                        "a prefill number is not a verdict about them\n", __func__);
            }
            g_moe_calib_prefetch.store(true);
            common_moe_calibration_status_note("expert prefetch", best_prefetch ? "on" : "off",
                    string_format("%s - %.1f prompt tok/s (kept on through the search either way)",
                                  best_prefetch ? "SELECTED" : "measured slower on prefill",
                                  best_prefetch ? pp : best_pp), true, true);
        }
        g_moe_calib_measure_prefill.store(false);
        g_moe_calib_prefill_xlong.store(false);
    }

    // The draft first. With a draft model attached, speculative decoding is the
    // regime this model is served in, and every lever after this - substitution
    // above all - has to be measured inside it. The draft itself is served exact
    // (a draft scheduler's scope never substitutes, see
    // ggml_backend_sched_set_moe_cache_scope), so whatever the target's stand-ins
    // cost, the measurement below sees it with MTP doing its share of the work
    // rather than assuming what that share is. Ordering the depth search after
    // substitution meant the substitution floor was chosen for a no-draft
    // configuration the server would never run.
    // Speculative decoding is a design constraint here, not a knob.
    //
    // The architecture of this fork is: no expert layer resident in VRAM, the
    // MTP head resident, and the hottest experts held in the cache and served
    // on demand. The draft is part of that design - it is what generates,
    // while the target verifies and the cache decides what is worth keeping -
    // so the calibration's job is to make it work as well as possible, not to
    // ask whether to have it.
    //
    // That is a change from what this file did earlier today, which was free to
    // record speculative decoding off whenever a measurement preferred the
    // alternative. It did so twice, and both times on a comparison taken in a
    // configuration where the draft could not work: the arms ran at f16 KV,
    // where 64k leaves 1405 MiB against the draft's 1819, so the "draft" arm
    // was really a dead cache (2.64 tok/s against a measured no-cache rate of
    // 2.75). A design decision is not the right thing to infer from that.
    //
    // GGML_MOE_CALIBRATE_ALLOW_NO_DRAFT=1 restores the old behaviour for
    // anyone who wants the comparison made rather than assumed.
    // Short by default. The exhaustive sweep is opt-in.
    //
    // Stages that prove an individual mechanism - is the prerouter worth it,
    // does rank-weighted admission help, which stand-in picker wins - all share
    // a bias: they start from the feature being OFF and need evidence to enable
    // it. On a machine with a 12-20% run-to-run spread that disables real
    // benefits by coin flip, and the effect compounds with every stage added.
    //
    // Measured on gemma-4 at the same context: the September 4 entry, from a
    // calibration with a fraction of today's stages, ran 68.75 tok/s. Today's,
    // with all of them, produced 58.94 - 14% slower, on the same model and the
    // same card, because more stages meant more chances to switch something off
    // for a reason that was really noise. A hand-assembled configuration with
    // everything enabled then beat both.
    //
    // So those stages are off by default and the mechanisms simply stay on. What
    // remains is what genuinely differs between models and hardware and has
    // effects far larger than the noise: where the experts live, how much cache
    // there is, how aggressively stand-ins may be used, and how deep the draft
    // goes. GGML_MOE_CALIBRATE_FULL=1 runs everything.
    const bool full_sweep = [] {
        const char * e = getenv("GGML_MOE_CALIBRATE_FULL");
        return e && atoi(e) != 0;
    }();
    if (!full_sweep) {
        LOG_WRN("%s: short calibration - measuring placement, cache size, substitution and draft depth, "
                "and leaving every other mechanism on rather than spending a candidate proving each one "
                "(GGML_MOE_CALIBRATE_FULL=1 for the exhaustive sweep)\n", __func__);
    }

    const bool allow_no_draft = [] {
        const char * e = getenv("GGML_MOE_CALIBRATE_ALLOW_NO_DRAFT");
        return e && atoi(e) != 0;
    }();
    int    best_n_max     = -1; // -1 not calibrated; 0 measured: no draft beat every depth
    int    depth_runner_up = -1; // the second-best depth, re-measured once substitution is chosen
    double best_n_max_tps = -1.0;
    double no_draft_tps   = -1.0;


    // KV precision - the consumer nobody had ever bid against.
    //
    // Every other stage fights over the VRAM left AFTER the KV cache, and the
    // KV cache was never a variable: calibration passed type_k through and
    // never varied it. On this model that is the binding difference between
    // the context we measure at and the one we serve at. With 2 KV heads and
    // 256-wide K and V, an attention layer holds 2048 bytes per token, so the
    // step from 4k to 64k costs roughly 1.4 GiB - and 1.4 GiB is the margin
    // every failure today has been on the wrong side of: the micro-batch OOM
    // was 676 MiB short, the expert cache had to halve, and the draft stopped
    // paying for the VRAM it displaced.
    //
    // Halving KV precision gives most of that back. Whether it is worth taking
    // is a quality question, not a throughput one, so it is measured through
    // the same answer gate that polices the substitution ladder rather than
    // ranked on tok/s alone. Placed before the draft-depth search because that
    // is where the shortage first bites - a depth that cannot fit at f16 may
    // fit at q8_0, and every stage after inherits the larger budget.
    std::string best_kv_type;
    if (!common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: measuring KV cache precision (it has never been varied, and at this context it "
                "holds the margin everything else is short of) ...\n", __func__);
        common_moe_calibration_status_set("measuring KV cache precision");
        // q8_0 and f16 only. q4_0 is not a candidate.
        //
        // KV precision is a quality knob with a throughput side effect, and
        // the two do not deserve equal weight here: this is a reasoning model,
        // the KV cache holds the whole chain of thought, and 4-bit K and V
        // degrade it in a way the answer gate is not guaranteed to catch - a
        // reply can stay fluent and verifiable while the reasoning behind it
        // drifts. q8_0 is the floor, and q4_0 is excluded by policy rather
        // than measured and rejected, because a win it produced would be a win
        // we should not take.
        //
        // f16 stays in only as the reference point, so the log records what
        // q8_0 actually costs and what it buys. Measured at 64k on this model:
        // f16 leaves 1405 MiB for everything else and cannot fit the draft at
        // all; q8_0 leaves 2393 MiB and fits it with room to spare. That is
        // the difference between serving 64k with speculative decoding and
        // choosing between them.
        // q8_0 for the target, f16 for the draft. The asymmetry is the point.
        //
        // common_base_params_to_speculative gives the draft its own
        // cache_type_k/v, so -ctk/-ctv here quantize the target only - which is
        // what we want, for opposite reasons on each side.
        //
        // The target is where the bytes are: 48 layers against the draft's
        // single NextN block, so q8_0 on the target frees ~1 GiB at 64k while
        // q8_0 on the draft would free almost nothing. And the target's
        // degradation is the kind the answer gate can see - it checks what the
        // model answers.
        //
        // The draft's cannot be seen that way. Acceptance is the draft's
        // entire value, and it is measured in AGREEMENT with the target, not
        // in output anyone reads: at 0.97 the draft pays for itself, at 0.43 it
        // costs more than it returns (both measured today at 64k). A quantized
        // KV degrades precisely that agreement, since the head predicts from
        // state it then has to agree with - and it would buy a rounding error
        // of VRAM for the risk.
        //
        // So: quantize the large consumer whose output is checked, and leave
        // the small one whose worth is measured in how often it is right.
        // q8_0 by default, not measured.
        //
        // At any context worth calibrating, this is the single largest VRAM
        // lever and it has won every time it has been looked at: measured at
        // 64k it frees 988 MiB, which is the difference between the 1819 MiB
        // MTP head fitting and not fitting at all, and it measured faster in
        // both directions it was tried (10.17 against 9.41, 9.17 against 8.36).
        // f16 at 64k leaves 1405 MiB against a draft that needs 1819 - it
        // simply does not fit the design.
        //
        // So it stops being a candidate and becomes what the design assumes.
        // The comparison remains available for a model where the trade might
        // differ; it is not worth two candidates in every run to re-derive an
        // answer this consistent. GGML_MOE_CALIBRATE_KV_SWEEP=1 measures it.
        double best_kv_tps = -1.0;
        const bool sweep_kv = [] {
            const char * e = getenv("GGML_MOE_CALIBRATE_KV_SWEEP");
            return e && atoi(e) != 0;
        }();
        if (!sweep_kv) {
            best_kv_type = "q8_0";
            common_moe_calib_set_kv_type(best_kv_type);
            entry.kv_type = best_kv_type;
            LOG_WRN("%s: KV cache at q8_0 by default - it frees the VRAM the draft needs and has measured "
                    "faster every time it was compared (set GGML_MOE_CALIBRATE_KV_SWEEP=1 to measure it "
                    "against f16)\n", __func__);
            common_moe_calibration_status_note("KV precision", "q8_0",
                    "DEFAULT - frees ~988 MiB at 64k, which is what makes the draft fit", true, true);
            checkpoint("KV precision");
        } else {
        common_moe_stage_begin("KV precision", 2);
        // At the CONSERVATIVE micro-batch, not whatever the prompt stage chose.
        //
        // KV precision is a question about the budget: how much is left for
        // everything else. Measuring it at a micro-batch large enough to starve
        // the expert cache answers a different question - the stage reported
        // 2.56 tok/s that way, which is the no-cache rate, not a KV verdict.
        // The conservative micro-batch leaves the most room, so what this
        // measures is the KV choice rather than the consequences of an -ub the
        // next stage may overturn anyway.
        const int kv_saved_ub = g_moe_calib_ubatch.load();
        g_moe_calib_ubatch.store(512);
        for (const char * kvt : { "f16", "q8_0" }) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            common_moe_calib_set_kv_type(std::string(kvt) == "f16" ? std::string() : std::string(kvt));
            // With the draft attached, if one is configured.
            //
            // This stage first measured with no draft, which made its numbers
            // incomparable with every stage after it - and the KV choice is
            // precisely a decision about who gets the freed VRAM, which is not
            // answerable without the consumer that wants it. Measured at 64k:
            // this stage reported 10.17 tok/s for q8_0 while the very next
            // stage, with the draft loaded, reported 7.5 for the same KV
            // setting. Two true numbers about different machines.
            //
            // The rule the rest of this file already follows: a stage's number
            // is only worth something against another number from the same
            // regime, and the regime that matters is the one being served.
            const double tps = common_moe_bench_candidate_server(
                    self_exe, path_model,
                    // has_dft() only means a draft was named on the command
                    // line. best_n_max == 0 means a stage has since MEASURED
                    // that no draft beats every depth, and a stage that keeps
                    // attaching one after that is measuring a configuration
                    // this run has already rejected - which is how f16 KV came
                    // back as "failed" immediately after the micro-batch guard
                    // recorded speculative decoding off.
                    (params.speculative.has_dft() && best_n_max != 0)
                        ? params.speculative.draft.mparams.path : std::string(),
                    best_n,
                    (params.speculative.has_dft() && best_n_max != 0)
                        ? std::max(1, params.speculative.draft.n_max) : 0,
                    n_threads_default, next_port(), ctx, n_predict, concurrency, -1, -1,
                    active_ngl, active_min_rank, nullptr, 1234, active_quality_sigma);
            common_moe_calibration_status_candidate_done();
            LOG_INF("%s:   KV %s -> %s\n", __func__, kvt,
                    tps > 0 ? string_format("%.2f tok/s", tps).c_str()
                            : (tps == COMMON_MOE_TPS_REJECTED ? "rejected on answer quality" : "failed"));
            common_moe_calibration_status_note("KV precision", kvt,
                    tps > 0 ? string_format("%.2f tok/s", tps)
                            : (tps == COMMON_MOE_TPS_REJECTED ? std::string("rejected on answer quality")
                                                              : std::string("failed")), tps > 0);
            if (tps > best_kv_tps) {
                best_kv_tps  = tps;
                best_kv_type = std::string(kvt) == "f16" ? std::string() : std::string(kvt);
            }
        }
        common_moe_calib_set_kv_type(best_kv_type);
        g_moe_calib_ubatch.store(kv_saved_ub);   // the -ub decision is the next stage's
        common_moe_stage_end();
        }   // sweep_kv
        entry.kv_type = best_kv_type;
        checkpoint("KV precision");
        if (best_kv_tps > 0) {
            LOG_INF("%s: KV precision: %s at %.2f tok/s\n", __func__,
                    best_kv_type.empty() ? "f16" : best_kv_type.c_str(), best_kv_tps);
            common_moe_calibration_status_note("KV precision",
                    best_kv_type.empty() ? std::string("f16") : best_kv_type,
                    string_format("SELECTED - %.2f tok/s", best_kv_tps), true, true);
        }
    }

    // After the KV stage, not before it. Both orders are wrong for opposite
    // reasons, and this is the one that is wrong less.
    //
    // Before: the KV stage inherited an -ub chosen with no draft in the
    // picture, attached the draft anyway, and measured a starved cache -
    // 2.56 tok/s, which is the no-cache rate rather than a KV verdict.
    //
    // After, without care: the guard decides whether -ub and the draft can
    // coexist using f16 KV, which at 64k leaves 1405 MiB against the draft's
    // 1819 - so the draft is judged in the one configuration where it cannot
    // work, and the stage that frees the gigabyte it needs runs afterwards.
    // Measured: the draft arm came back at 2.64 tok/s, again the no-cache
    // rate, and speculative decoding was recorded off on that basis.
    //
    // The circularity is only apparent. KV is a budget question and the
    // micro-batch is a prefill one, so KV can be settled at the CONSERVATIVE
    // micro-batch - the one that leaves the most room - and the -ub/draft
    // conflict then resolved against a budget that already includes whatever
    // KV freed. Which is this order, with the KV stage passing -ub 512 rather
    // than whatever the prompt stage chose.

    // This has to run before the KV stage, not after it. That stage now
    // measures with the draft attached - its no-draft numbers were not
    // comparable with anything downstream - so it inherits whatever -ub the
    // prompt stage chose, and -ub was chosen with no draft in the picture.
    // Measured at 64k: -ub 2048 plus the draft plus q8_0 KV leaves the expert
    // cache nothing, and the KV stage reported 2.56 tok/s - which is the
    // no-cache rate (2.75 measured separately), not a KV verdict. The guard
    // has to resolve the -ub/draft conflict first so everything after it
    // measures a configuration that will actually be served.

    // The micro-batch was chosen before any draft existed - check it survives one.
    //
    // -ub is measured on prompt processing, and that stage runs with no draft
    // attached. The draft's compute buffers scale with the micro-batch, so a
    // value that is comfortable without one can be impossible with it. Measured
    // on Qwen3.8-Flash-Next at 64k: -ub 2048 won its stage at 47.5 prompt
    // tok/s, and then every candidate from here on died with "failed to create
    // MTP context", 676 MiB short - the draft depth search, substitution, the
    // quality bar, the cache ladder, the prerouter. Nothing was wrong with any
    // of them.
    //
    // This has to run BEFORE the depth search rather than after it: the first
    // version keyed off mtp_configured, which is derived from best_n_max, which
    // the depth search produces - so it only ran once the stage it was meant to
    // protect had already failed.
    //
    // Earlier runs never saw any of this, because fit silently shrank the
    // context until the combination fitted - the same mechanism that made a run
    // asking for 64k measure 4k throughout.
    if (params.speculative.has_dft() && !params.speculative.draft.mparams.path.empty() &&
        g_moe_calib_ubatch.load() > 0 && !common_moe_calibrate_budget_spent()) {
        const std::string dft_path = params.speculative.draft.mparams.path;
        const int probe_n_max = std::max(1, params.speculative.draft.n_max);
        const int chosen_ub = g_moe_calib_ubatch.load();
        auto try_with_draft = [&]() {
            const double r = common_moe_bench_candidate_server(
                    self_exe, path_model, dft_path, best_n, probe_n_max,
                    n_threads_default, next_port(), ctx, n_predict, concurrency, -1, -1,
                    active_ngl, active_min_rank);
            common_moe_calibration_status_candidate_done();
            return r;
        };
        double check = try_with_draft();
        for (const int smaller : { 2048, 512 }) {
            if (check != COMMON_MOE_TPS_LAUNCH_FAILED || smaller >= g_moe_calib_ubatch.load()) {
                continue;
            }
            LOG_WRN("%s: -ub %d does not fit once the draft is attached at this context - stepping down "
                    "to %d\n", __func__, g_moe_calib_ubatch.load(), smaller);
            common_moe_calibration_status_note("prompt micro-batch",
                    string_format("-ub %d", g_moe_calib_ubatch.load()),
                    "does not fit with the draft attached", false, false);
            g_moe_calib_ubatch.store(smaller);
            entry.n_ubatch = smaller;
            check = try_with_draft();
        }
        if (check == COMMON_MOE_TPS_LAUNCH_FAILED) {
            LOG_WRN("%s: no micro-batch on the ladder fits with the draft attached - leaving it unset so "
                    "the server's own default applies\n", __func__);
            g_moe_calib_ubatch.store(-1);
            entry.n_ubatch = -1;
        } else if (g_moe_calib_ubatch.load() != chosen_ub) {
            // The micro-batch and the draft cannot both have the VRAM, and
            // stepping -ub down is only ONE of the two ways to resolve that.
            // The other is to keep the micro-batch and drop the draft, and
            // nothing here was measuring it - the guard simply assumed the
            // draft was the thing worth keeping.
            //
            // On this hardware that assumption is wrong in both directions
            // that matter. -ub 2048 is worth about 40% of prompt throughput
            // (47.5 against 34.6 prompt tok/s), and the draft has measured
            // NEGATIVE at this context in every run today - 7.56 against 8.39
            // tok/s for no draft at all, with the draft accepting 0.97 of what
            // it proposed. So the guard was giving up a large, reproducible
            // prefill win to protect a feature that costs throughput.
            //
            // So measure it. One extra candidate: the full micro-batch with no
            // draft, against the reduced micro-batch with one. If the first
            // wins, speculative decoding is recorded off here and -ub keeps its
            // value - which is a real verdict about this machine, not a
            // preference about which feature is more interesting.
            const int reduced_ub = g_moe_calib_ubatch.load();
            const double with_draft = check;
            double no_draft = -1.0;
            if (!common_moe_calibrate_budget_spent()) {
                g_moe_calib_ubatch.store(chosen_ub);
                no_draft = common_moe_bench_candidate_server(
                        self_exe, path_model, std::string(), best_n, 0,
                        n_threads_default, next_port(), ctx, n_predict, concurrency, -1, -1,
                        active_ngl, active_min_rank);
                common_moe_calibration_status_candidate_done();
                LOG_INF("%s:   -ub %d with no draft -> %s (against -ub %d with the draft at %.2f)\n",
                        __func__, chosen_ub,
                        no_draft > 0 ? string_format("%.2f tok/s", no_draft).c_str() : "failed",
                        reduced_ub, with_draft);
            }
            if (!allow_no_draft && no_draft > 0 && with_draft > 0 && no_draft > with_draft) {
                LOG_INF("%s: -ub %d without a draft measured faster (%.2f vs %.2f tok/s), but the draft is "
                        "part of this design rather than a candidate - keeping -ub %d so it fits. The "
                        "micro-batch buys prefill once per request; the draft buys a token multiplier on "
                        "every step, and what to do with the VRAM it leaves is the cache's question\n",
                        __func__, chosen_ub, no_draft, with_draft, reduced_ub);
                g_moe_calib_ubatch.store(reduced_ub);
                entry.n_ubatch = reduced_ub;
                common_moe_calibration_status_note("prompt micro-batch",
                        string_format("-ub %d", reduced_ub),
                        string_format("REVISED - the draft is a design constraint; -ub %d without it "
                                      "measured %.2f but is not an option", chosen_ub, no_draft), true, true);
            } else if (no_draft > 0 && with_draft > 0 && no_draft > with_draft) {
                LOG_WRN("%s: the full micro-batch without a draft beats the reduced one with it (%.2f vs "
                        "%.2f tok/s) - keeping -ub %d and recording speculative decoding off. The draft "
                        "cannot have the VRAM the micro-batch needs and return more than it costs\n",
                        __func__, no_draft, with_draft, chosen_ub);
                g_moe_calib_ubatch.store(chosen_ub);
                entry.n_ubatch   = chosen_ub;
                entry.spec_n_max = 0;
                best_n_max       = 0;
                common_moe_calibration_status_note("prompt micro-batch",
                        string_format("-ub %d", chosen_ub),
                        string_format("KEPT - %.2f tok/s with no draft beats %.2f with one",
                                      no_draft, with_draft), true, true);
                common_moe_calibration_status_note("speculative decoding", "off",
                        string_format("SELECTED - it cannot hold -ub %d, and dropping -ub to %d costs more "
                                      "than the draft returns", chosen_ub, reduced_ub), true, true);
            } else {
                g_moe_calib_ubatch.store(reduced_ub);
                entry.n_ubatch = reduced_ub;
                LOG_INF("%s: prompt micro-batch revised to -ub %d to fit alongside the draft%s\n",
                        __func__, reduced_ub,
                        no_draft > 0 ? string_format(" (the full micro-batch without a draft measured %.2f, "
                                                     "which does not beat %.2f)", no_draft, with_draft).c_str()
                                     : "");
                common_moe_calibration_status_note("prompt micro-batch",
                        string_format("-ub %d", reduced_ub),
                        "REVISED - the larger value could not hold the draft, and the draft earns its place",
                        true, true);
            }
        }
        checkpoint("micro-batch revalidated with the draft");
    }

    const bool search_depth = [] {
        const char * e = getenv("GGML_MOE_CALIBRATE_SEARCH_DEPTH");
        return e && atoi(e) != 0;
    }();
    if (params.speculative.has_dft() && entry.spec_n_max >= 0) {
        best_n_max = entry.spec_n_max;
        LOG_WRN("%s: resuming - speculative depth %d already measured, skipping the envelope search "
                "and the golden-section refinement\n", __func__, best_n_max);
        common_moe_calibration_status_note("speculative depth",
                best_n_max > 0 ? string_format("n-max %d", best_n_max) : std::string("off"),
                resumed_result(best_n_max > 0 ? string_format("n-max %d", best_n_max) : std::string("off"),
                               entry.tps_spec_n_max, entry.calibrated_at), true, true);
    } else if (params.speculative.has_dft() && !search_depth) {
        // The single most expensive stage in the run: a no-draft baseline, an
        // acceptance A/B, an envelope that doubles 1..32, a golden-section
        // refinement inside it and a re-check after substitution - every one a
        // full server spawn, because depth and acceptance are launch arguments
        // and cannot be POSTed to a running server the way every other knob
        // here can. That is most of the run's wall clock for two values the
        // launch command already carries.
        // The later stages run at the launch depth, but it is NOT written into the entry: it is
        // not a measurement, and a cached launch flag would outrank every future default.
        best_n_max = std::max(1, params.speculative.draft.n_max);
        g_moe_calib_prob_accept.store(params.speculative.draft.prob_accept ? 1 : 0);
        LOG_WRN("%s: speculative depth left at the launch value (n-max %d, %s acceptance) - depth and "
                "acceptance are launch arguments, so every candidate is a full server spawn "
                "(GGML_MOE_CALIBRATE_SEARCH_DEPTH=1 to search them)\n", __func__, best_n_max,
                params.speculative.draft.prob_accept ? "probabilistic" : "exact-match");
        common_moe_calibration_status_note("speculative depth", string_format("n-max %d", best_n_max),
                "LAUNCH VALUE - the only knob here that needs a server restart per candidate", true, true);
        common_moe_calibration_status_note("draft acceptance",
                params.speculative.draft.prob_accept ? "probabilistic" : "exact-match",
                "LAUNCH VALUE", true, true);
        checkpoint("speculative depth");
    } else if (params.speculative.has_dft()) {
        {
            // Find the envelope: double n_max until throughput drops below
            // half the n_max=1 baseline (the real n_max=8 collapse we
            // measured went from ~80% cache health to ~14-21% - a >2x
            // throughput cliff, not a gentle decline, so "less than half"
            // is a safe, real signal for "past the edge" rather than
            // ordinary run-to-run noise).
            // Acceptance mode first, then depth.
            //
            // --spec-prob-accept changes how many drafted tokens survive
            // verification, so it changes which depth is worth paying for -
            // a mode that accepts more makes a wider draft pay off and a
            // narrower one leave value behind. Measured after the depth
            // search, as it used to be, the envelope searched the curve for
            // the wrong acceptance mode and a +10-18% lever was switched on
            // afterwards: 12.63 vs 11.43 tok/s on this model, 77.18 vs 65.36
            // on gemma-4, where it was the largest single lever in the run.
            // At 64k that ordering produced "the draft loses to no draft"
            // from a configuration the draft would never be served in.
            //
            // Decided here at one fixed reference depth, then held for the
            // whole envelope. One extra candidate, and every depth after it is
            // measured in the regime it will actually run in. The later stage
            // still re-checks it at the chosen depth, because the interaction
            // runs both ways.
            if (!common_moe_calibrate_budget_spent()) {
                // Depth 1 first, then the configured depth.
                //
                // Depth 1 is the cleanest place to isolate the acceptance mode -
                // one drafted token, so nothing about the width confounds it -
                // and the cheapest. But the mode matters MORE at width, because
                // there are more tokens for tolerance to save, so testing only
                // at depth 1 could call a difference invisible that is real at
                // 3 or 4. Both, and the aggregate decides.
                const int cfg_depth = std::max(1, params.speculative.draft.n_max);
                const int ref_depth = 1;
                double pa_tps[2] = { -1.0, -1.0 };
                for (int pa = 0; pa <= 1; pa++) {
                    if (common_moe_calibrate_budget_spent()) {
                        break;
                    }
                    g_moe_calib_prob_accept.store(pa);
                    pa_tps[pa] = bench_with_retry(best_n, ref_depth,
                                                  params.speculative.draft.mparams.path, n_threads_default);
                    LOG_INF("%s:   draft acceptance %s at depth %d -> %s%s\n", __func__,
                            pa ? "probabilistic" : "exact-match", ref_depth,
                            pa_tps[pa] > 0 ? string_format("%.2f tok/s", pa_tps[pa]).c_str() : "failed",
                            pa_tps[pa] > 0 ? common_moe_last_acceptance_str().c_str() : "");
                    common_moe_calibration_status_note("draft acceptance",
                            pa ? "probabilistic" : "exact-match",
                            pa_tps[pa] > 0 ? string_format("%.2f tok/s (before the depth search)", pa_tps[pa])
                                           : std::string("failed"), pa_tps[pa] > 0);
                }
                // The same comparison at the configured width, where tolerance
                // has more tokens to act on. Skipped when the configured depth
                // IS 1, since that is the measurement just taken.
                double pa_wide[2] = { -1.0, -1.0 };
                if (cfg_depth > ref_depth) {
                    for (int pa = 0; pa <= 1; pa++) {
                        if (common_moe_calibrate_budget_spent()) {
                            break;
                        }
                        g_moe_calib_prob_accept.store(pa);
                        pa_wide[pa] = bench_with_retry(best_n, cfg_depth,
                                                       params.speculative.draft.mparams.path, n_threads_default);
                        common_moe_calibration_status_candidate_done();
                        LOG_INF("%s:   draft acceptance %s at depth %d -> %s%s\n", __func__,
                                pa ? "probabilistic" : "exact-match", cfg_depth,
                                pa_wide[pa] > 0 ? string_format("%.2f tok/s", pa_wide[pa]).c_str() : "failed",
                                pa_wide[pa] > 0 ? common_moe_last_acceptance_str().c_str() : "");
                        common_moe_calibration_status_note("draft acceptance",
                                string_format("%s at depth %d", pa ? "probabilistic" : "exact-match", cfg_depth),
                                pa_wide[pa] > 0 ? string_format("%.2f tok/s", pa_wide[pa]) : std::string("failed"),
                                pa_wide[pa] > 0);
                    }
                    // Aggregate the two widths: a mode has to be better across
                    // them, not just at whichever one it happened to win.
                    for (int pa = 0; pa <= 1; pa++) {
                        if (pa_tps[pa] > 0 && pa_wide[pa] > 0) {
                            pa_tps[pa] = (pa_tps[pa] + pa_wide[pa]) / 2.0;
                        } else if (pa_wide[pa] > 0) {
                            pa_tps[pa] = pa_wide[pa];
                        }
                    }
                }

                // Confirm before believing, and break a tie on acceptance.
                //
                // This knob shapes the whole depth search below it, so a
                // verdict reached on a noise-sized margin is leverage applied
                // to a coin flip. Measured at depth 3: exact-match 8.15 tok/s
                // at 0.78 acceptance, probabilistic 7.88 at 0.94 - a 3.4% gap
                // on a machine that has shown 10-20% between identical
                // configurations all day.
                //
                // And 3.4% cannot be a real cost, because probabilistic
                // acceptance does not do more work. The whole difference on
                // the drafting side is one push_back of a probability the
                // sampler has already computed (see result_probs in
                // speculative.cpp); on the verify side it accepts on that
                // probability instead of on exact match. More tolerant
                // acceptance means more tokens survive each verify round and
                // therefore FEWER rounds - it should be faster, not slower.
                //
                // So: re-measure when the gap is inside the noise, and if it
                // stays inside it, take the mode that accepts more. Acceptance
                // is the mechanism's actual output and it is measured over
                // hundreds of tokens; throughput at this margin is not
                // measuring anything.
                int pa_pick = (pa_tps[1] > 0 && pa_tps[1] > pa_tps[0]) ? 1 : (pa_tps[0] > 0 ? 0 : -1);
                if (pa_tps[0] > 0 && pa_tps[1] > 0) {
                    const double hi  = std::max(pa_tps[0], pa_tps[1]);
                    const double lo  = std::min(pa_tps[0], pa_tps[1]);
                    const double gap = (hi - lo) / hi;
                    if (gap < 0.10 && !common_moe_calibrate_budget_spent()) {
                        LOG_INF("%s:   the two acceptance modes are %.1f%% apart, which is inside this "
                                "machine's run-to-run spread - re-measuring both\n", __func__, 100.0 * gap);
                        double acc[2] = { -1.0, -1.0 };
                        double tps2[2] = { -1.0, -1.0 };
                        for (int pa = 0; pa <= 1; pa++) {
                            if (common_moe_calibrate_budget_spent()) {
                                break;
                            }
                            g_moe_calib_prob_accept.store(pa);
                            tps2[pa] = bench_with_retry(best_n, ref_depth,
                                                        params.speculative.draft.mparams.path, n_threads_default);
                            acc[pa] = g_moe_last_draft_n > 0.0
                                ? g_moe_last_draft_n_accepted / g_moe_last_draft_n : -1.0;
                            common_moe_calibration_status_candidate_done();
                            LOG_INF("%s:   %s measured again -> %s (first %.2f)%s\n", __func__,
                                    pa ? "probabilistic" : "exact-match",
                                    tps2[pa] > 0 ? string_format("%.2f tok/s", tps2[pa]).c_str() : "failed",
                                    pa_tps[pa], common_moe_last_acceptance_str().c_str());
                        }
                        if (tps2[0] > 0 && tps2[1] > 0) {
                            const double m0 = (pa_tps[0] + tps2[0]) / 2.0;
                            const double m1 = (pa_tps[1] + tps2[1]) / 2.0;
                            const double mhi = std::max(m0, m1);
                            if (mhi > 0.0 && (mhi - std::min(m0, m1)) / mhi < 0.10) {
                                // Still a tie on throughput: decide on acceptance.
                                pa_pick = (acc[1] >= 0.0 && acc[0] >= 0.0 && acc[1] > acc[0]) ? 1 : pa_pick;
                                LOG_INF("%s: the two modes are within noise on throughput (%.2f vs %.2f "
                                        "mean) - taking %s, which accepted %.2f against %.2f\n",
                                        __func__, m0, m1, pa_pick ? "probabilistic" : "exact-match",
                                        pa_pick ? acc[1] : acc[0], pa_pick ? acc[0] : acc[1]);
                            } else {
                                pa_pick = m1 > m0 ? 1 : 0;
                                LOG_INF("%s: %s wins on the mean of two samples (%.2f vs %.2f)\n", __func__,
                                        pa_pick ? "probabilistic" : "exact-match",
                                        std::max(m0, m1), std::min(m0, m1));
                            }
                        }
                    }
                }
                g_moe_calib_prob_accept.store(pa_pick);
                if (pa_pick >= 0) {
                    entry.spec_prob_accept = pa_pick;
                    LOG_INF("%s: draft acceptance: %s - the depth search below runs in that mode\n",
                            __func__, pa_pick ? "probabilistic" : "exact-match");
                    checkpoint("draft acceptance");
                }
            }

            LOG_INF("%s: finding spec-draft-n-max envelope (doubling until collapse) via real llama-server subprocesses ...\n", __func__);
            common_moe_calibration_status_set("searching speculative-decoding depth (spec-draft-n-max)");
            common_moe_stage_begin("draft depth", 14);
            // No draft at all, measured exactly like every depth below. Without it
            // the search can only rank depths against each other, so it would ship
            // a depth even on a machine where speculative decoding loses outright.
            no_draft_tps = bench_with_retry(best_n, 0, std::string(), n_threads_default);
            LOG_INF("%s:   no draft (speculative decoding off) -> %s\n", __func__,
                    no_draft_tps > 0 ? string_format("%.2f tok/s", no_draft_tps).c_str() : "failed");
            common_moe_calibration_status_note("speculative decoding", "off",
                    no_draft_tps > 0 ? string_format("%.2f tok/s", no_draft_tps) : std::string("failed"), no_draft_tps > 0);
            common_moe_calibration_status_candidate_done();
            std::map<int, double> nmax_trace;
            double baseline = -1.0;
            int last_good = 1;
            // The baseline is the first depth that actually benchmarks, not
            // specifically n=1. Measured on qwen4exp: n=1 failed to come up
            // twice (a draft depth of 1 is the oddest configuration in the
            // sweep, and a candidate that fails to load is not evidence about
            // depth), baseline stayed <= 0, and that single probe silently
            // skipped the golden-section search, --spec-prob-accept and the
            // drafter cascade with it - four MTP knobs behind one fragile
            // candidate. A failure now costs only its own rung.
            int n_failures = 0;
            // Stop once two rungs in a row come in below the best so far - past the
            // peak, measured, rather than "below half of depth 1", which on
            // Qwen3.8-Flash-Next let the envelope run to 32 (2.66 tok/s is still
            // above half of 4.59) after the peak at 8 and spend three candidates
            // on depths that were never going to win.
            double best_so_far = -1.0;
            int    n_below     = 0;
            for (int n = 1; n <= 32; n *= 2) {
                const double tps = bench_with_retry(best_n, n, params.speculative.draft.mparams.path, n_threads_default);
                nmax_trace[n] = tps;
                LOG_INF("%s:   spec-draft-n-max=%d -> %s%s\n", __func__, n, tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed",
                        tps > 0 ? common_moe_last_acceptance_str().c_str() : "");
                common_moe_calibration_status_note("speculative depth", string_format("n-max %d", n),
                        tps > 0 ? string_format("%.2f tok/s%s", tps, common_moe_last_acceptance_str().c_str())
                                : std::string("failed"), tps > 0);
                common_moe_calibration_status_candidate_done();
                if (tps > 0 && baseline <= 0.0) {
                    baseline = tps; // first depth that stood up anchors the collapse test
                }
                if (tps < 0) {
                    // A candidate the clock skipped is not a candidate that
                    // failed. When a stage reaches its share of the budget every
                    // remaining launch returns instantly, and reading those as
                    // failures is how an incomplete search gets recorded as a
                    // finished one: measured here, n-max 4 and 8 were "failed"
                    // 1ms apart, tripped the two-in-a-row rule below, and the
                    // search settled on n-max 1 at 9.20 tok/s - while the run
                    // before had measured n-max 4 at 10.87. Stop cleanly and
                    // say why, so the depths that never ran are not mistaken
                    // for depths that were measured and rejected.
                    if (common_moe_calibrate_budget_spent()) {
                        LOG_WRN("%s: the draft-depth stage ran out of its share of the budget before "
                                "n-max %d - keeping the best of the depths that did run, and NOT "
                                "recording the rest as failures\n", __func__, n);
                        break;
                    }
                    // Two consecutive failures means something is wrong with the
                    // draft itself rather than with this depth - stop paying for it.
                    if (++n_failures >= 2) {
                        LOG_WRN("%s: two spec-draft-n-max candidates in a row failed - stopping the envelope search\n", __func__);
                        break;
                    }
                    continue;
                }
                n_failures = 0;
                last_good = n; // the golden search may still land between this rung and the peak

                // Stop doubling once the draft stops being believed, not only
                // once the process dies. Acceptance is the direct measure of
                // whether a wider draft is producing anything verification
                // will keep, and it collapses well before the depth becomes
                // unloadable: measured on Qwen3.8-Flash-Next at 64k, 0.91 /
                // 0.83 / 0.87 at n-max 1/2/4 and then 0.40 at 8 - after which
                // 16 and 32 were tried anyway and both died on a model-load
                // OOM. Each of those cost a full launch plus the health
                // backstop, for rungs the acceptance number had already ruled
                // out. Half the tokens rejected means the next rung, which
                // doubles the width again, cannot plausibly recover.
                const double acc = g_moe_last_draft_n > 0.0
                    ? g_moe_last_draft_n_accepted / g_moe_last_draft_n : -1.0;
                if (acc >= 0.0 && acc < 0.5) {
                    LOG_INF("%s: draft acceptance fell to %.2f at n-max %d - stopping the envelope search "
                            "rather than doubling into widths the draft is no longer believed at\n",
                            __func__, acc, n);
                    break;
                }

                if (tps > best_so_far) {
                    best_so_far = tps;
                    n_below     = 0;
                } else if (++n_below >= 2) {
                    break;
                }
            }
            const int nmax_hi = std::max(1, last_good);

            if (baseline > 0) {
                LOG_INF("%s: golden-section search for spec-draft-n-max in [1, %d] at ncmoe=%u ...\n",
                        __func__, nmax_hi, best_n);
                auto measure_nmax = [&](int n) -> double {
                    auto it = nmax_trace.find(n);
                    if (it != nmax_trace.end()) {
                        return it->second;
                    }
                    const double tps = bench_with_retry(best_n, n, params.speculative.draft.mparams.path, n_threads_default);
                    LOG_INF("%s:   spec-draft-n-max=%d -> %s%s\n", __func__, n, tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed",
                        tps > 0 ? common_moe_last_acceptance_str().c_str() : "");
                common_moe_calibration_status_note("speculative depth", string_format("n-max %d", n),
                        tps > 0 ? string_format("%.2f tok/s%s", tps, common_moe_last_acceptance_str().c_str())
                                : std::string("failed"), tps > 0);
                    common_moe_calibration_status_candidate_done();
                    return tps;
                };
                best_n_max = common_golden_section_search_max(1, nmax_hi, measure_nmax, nmax_trace);
                // Validate against the full trace, not just what
                // golden-section converged to - see the comment on the
                // identical check after the ncmoe search above. This is
                // the fix for a confirmed real bug: the envelope-doubling
                // phase above already measures n=1,2,4,8,... before
                // golden-section ever runs, but golden-section's own
                // bisection can (and, measured once, did: n_max=2 scored
                // 56.06 tok/s vs the bisection's own pick of n_max=9 at
                // 54.55) narrow away from those low values without ever
                // reconsidering them.
                for (const auto & kv : nmax_trace) {
                    if (kv.second > nmax_trace.at(best_n_max)) {
                        best_n_max = kv.first;
                    }
                }
                // Every depth above is one short probe, and a single lucky sample can
                // win outright - on Qwen3.8-Flash-Next depth 8 read 6.59 tok/s between
                // neighbours at 4.40-5.10. Measure the top two again and decide on the
                // mean of both samples each.
                {
                    int runner_up = -1;
                    for (const auto & kv : nmax_trace) {
                        if (kv.first != best_n_max && kv.second > 0 &&
                            (runner_up < 0 || kv.second > nmax_trace.at(runner_up))) {
                            runner_up = kv.first;
                        }
                    }
                    for (const int depth : { best_n_max, runner_up }) {
                        if (depth <= 0 || common_moe_calibrate_budget_spent()) {
                            continue;
                        }
                        const double again = bench_with_retry(best_n, depth, params.speculative.draft.mparams.path, n_threads_default);
                        common_moe_calibration_status_candidate_done();
                        if (again > 0) {
                            LOG_INF("%s:   spec-draft-n-max=%d measured again -> %.2f tok/s (first %.2f, mean %.2f)%s\n",
                                    __func__, depth, again, nmax_trace.at(depth), (again + nmax_trace.at(depth)) / 2.0,
                                    common_moe_last_acceptance_str().c_str());
                            nmax_trace[depth] = (again + nmax_trace.at(depth)) / 2.0;
                        }
                    }
                    const int first_winner = best_n_max;
                    if (runner_up > 0 && nmax_trace.at(runner_up) > nmax_trace.at(best_n_max)) {
                        best_n_max = runner_up;
                    }
                    depth_runner_up = best_n_max == first_winner ? runner_up : first_winner;
                }
                LOG_INF("%s: spec-draft-n-max=%d wins (%.2f tok/s)\n", __func__, best_n_max, nmax_trace.at(best_n_max));
                entry.spec_n_max = best_n_max;
                entry.tps_spec_n_max = nmax_trace.at(best_n_max);
                checkpoint("draft depth");
                common_moe_stage_end();
                // The whole speculative-decoding stage wrote no row, so a model
                // with a draft head showed nothing for it in the decisions table
                // even though it is one of the largest levers such a model has.
                common_moe_calibration_status_note("speculative depth",
                        string_format("n-max %d", best_n_max),
                        string_format("SELECTED - %.2f tok/s (mean of two samples where confirmed)", nmax_trace.at(best_n_max)),
                        true, /* chosen */ true);
                // The n_max search's own winning number (MTP active) is the
                // real answer for this deployment, not the earlier ncmoe-only
                // number (MTP off) - carry it forward so the final report and
                // cache entry don't undersell what was actually found.
                if (nmax_trace.at(best_n_max) > best_tps) {
                    best_tps = nmax_trace.at(best_n_max);
                }
                best_n_max_tps = nmax_trace.at(best_n_max);
            } else {
                LOG_WRN("%s: spec-draft-n-max=1 itself failed to benchmark - skipping n_max calibration\n", __func__);
            }
        }
    }

    // Proven, not promised: the best depth has to beat no draft at all - but
    // it has to be given its own measured configuration first.
    //
    // This verdict used to land here, and zeroing best_n_max here is circular:
    // the three stages below (draft placement, exactness, cache share) are all
    // gated on best_n_max > 0, so a draft was judged in the worst
    // configuration it will ever have - its experts competing with the
    // target's in an untuned cache, no share of its own, no placement - and
    // the stages that would fix exactly that were skipped BECAUSE it lost.
    // Measured on Qwen3.8-Flash-Next at 64k: the draft was accepting 0.87-0.88
    // of what it proposed, which is a draft doing its job well, and still lost
    // on throughput - a verification-cost problem, which is what placement and
    // share address.
    //
    // So the comparison is deferred: remember the number to beat, configure
    // the draft, then decide against the configured draft's real throughput.
    const double no_draft_tps_to_beat = no_draft_tps;
    if (best_n_max > 0 && no_draft_tps > 0.0 && best_n_max_tps <= no_draft_tps) {
        LOG_INF("%s: the best draft depth (%d, %.2f tok/s) does not yet beat no draft at all (%.2f tok/s) - "
                "configuring the draft first, then deciding\n",
                __func__, best_n_max, best_n_max_tps, no_draft_tps);
    }

    // Draft placement, at the chosen depth. The draft defaults to fully GPU-resident
    // (its own n_gpu_layers, and the target's -ncmoe override is not applied to it),
    // which is almost certainly right for a head that runs on every step - but its
    // expert bytes come straight out of the target's expert cache, so it is measured
    // against the other placement rather than assumed.
    int best_draft_cpu_moe = -1;
    int best_draft_exact   = -1; // -1 not calibrated, 1 exact only, 0 stand-ins allowed
    int best_share_pct     = -1; // draft's share of the expert-cache budget
    // Hoisted: the deferred no-draft comparison below needs the draft's best
    // CONFIGURED throughput, which is produced inside the share stage.
    double best_share_tps  = -1.0;
    // The design answers this one outright: the MTP head is the thing that IS
    // resident in VRAM. It runs on every decode step, it is small (440 MiB on
    // gemma-4, 1819 on qwen4exp), and it is the single weight the whole
    // no-resident-layer design chose to keep on the card. Offloading its experts
    // to the CPU to buy expert-cache bytes is that decision run backwards, and
    // the two follow-on stages - draft-exact and draft cache share - only exist
    // once it has been offloaded, so pinning it here removes four candidates.
    // GGML_MOE_CALIBRATE_SEARCH_DRAFT=1 measures the placement instead.
    const bool search_draft_place = [] {
        const char * e = getenv("GGML_MOE_CALIBRATE_SEARCH_DRAFT");
        return e && atoi(e) != 0;
    }();
    if (best_n_max > 0 && !search_draft_place) {
        best_draft_cpu_moe = 0;
        entry.spec_draft_cpu_moe = 0;
        g_moe_calib_draft_cpu_moe.store(false);
        LOG_WRN("%s: draft experts on the GPU - the MTP head is what the design keeps resident, and "
                "offloading it is that choice run backwards (GGML_MOE_CALIBRATE_SEARCH_DRAFT=1 to "
                "measure the placement)\n", __func__);
        common_moe_calibration_status_note("draft placement", "experts on GPU",
                "DESIGN - the MTP head is the weight the card holds; it runs on every decode step",
                true, true);
        checkpoint("draft placement");
    } else if (best_n_max > 0 && best_n_max_tps > 0.0 && !common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: measuring draft expert placement (GPU vs CPU) at spec-draft-n-max=%d ...\n", __func__, best_n_max);
        common_moe_calibration_status_set("measuring draft expert placement");
        g_moe_calib_draft_cpu_moe.store(true);
        const double cpu_tps = bench_with_retry(best_n, best_n_max, params.speculative.draft.mparams.path, n_threads_default);
        common_moe_calibration_status_candidate_done();
        LOG_INF("%s:   draft experts on CPU -> %s%s (on GPU: %.2f tok/s)\n", __func__,
                cpu_tps > 0 ? string_format("%.2f tok/s", cpu_tps).c_str() : "failed",
                cpu_tps > 0 ? common_moe_last_acceptance_str().c_str() : "", best_n_max_tps);
        best_draft_cpu_moe = cpu_tps > best_n_max_tps ? 1 : 0;
        entry.spec_draft_cpu_moe = best_draft_cpu_moe;
        checkpoint("draft placement");
        g_moe_calib_draft_cpu_moe.store(best_draft_cpu_moe == 1);
        common_moe_calibration_status_note("draft placement", best_draft_cpu_moe ? "experts on CPU" : "on GPU",
                string_format("SELECTED - %.2f tok/s", best_draft_cpu_moe ? cpu_tps : best_n_max_tps), true, true);

        // Only meaningful once the draft's experts are CPU-offloaded: on the GPU
        // none of the draft reaches the expert cache, so there is nothing to stand
        // in for. Offloaded, the trade reverses - a draft miss costs a host/NVMe
        // read, against a slightly worse guess whose only cost is a lost
        // acceptance - so it is measured rather than assumed either way.
        if (best_draft_cpu_moe == 1 && cpu_tps > 0 && !common_moe_calibrate_budget_spent()) {
            common_moe_calib_set_env("GGML_CUDA_MOE_CACHE_DRAFT_EXACT=0");
            const double sub_tps = bench_with_retry(best_n, best_n_max, params.speculative.draft.mparams.path, n_threads_default);
            common_moe_calibration_status_candidate_done();
            LOG_INF("%s:   stand-ins allowed in the draft -> %s (exact: %.2f tok/s)\n", __func__,
                    sub_tps > 0 ? string_format("%.2f tok/s", sub_tps).c_str() : "failed", cpu_tps);
            best_draft_exact = sub_tps > cpu_tps ? 0 : 1;
            common_moe_calib_set_env(best_draft_exact ? std::string() : std::string("GGML_CUDA_MOE_CACHE_DRAFT_EXACT=0"));
            common_moe_calibration_status_note("draft experts",
                    best_draft_exact ? "exact only" : "stand-ins allowed",
                    string_format("SELECTED - %.2f tok/s", best_draft_exact ? cpu_tps : sub_tps), true, true);

            // How much of the expert cache the draft may hold. Left alone the split
            // is by tensor count, which hands the draft ~2% for no measured reason
            // while it runs on every decode step. Measured only when its experts are
            // CPU-offloaded, because that is the only case where it holds any.
            best_share_tps = std::max(cpu_tps, sub_tps);
            const std::string exact_env = best_draft_exact ? std::string() : std::string(" GGML_CUDA_MOE_CACHE_DRAFT_EXACT=0");
            for (const int pct : { 10, 25 }) {
                if (common_moe_calibrate_budget_spent()) {
                    break;
                }
                common_moe_calib_set_env(string_format("GGML_CUDA_MOE_CACHE_DRAFT_SHARE_PCT=%d", pct) + exact_env);
                const double tps = bench_with_retry(best_n, best_n_max, params.speculative.draft.mparams.path, n_threads_default);
                common_moe_calibration_status_candidate_done();
                LOG_INF("%s:   draft cache share %d%% -> %s\n", __func__, pct,
                        tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
                common_moe_calibration_status_note("draft cache share", string_format("%d%%", pct),
                        tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
                if (tps > best_share_tps) {
                    best_share_tps = tps;
                    best_share_pct = pct;
                }
            }
            common_moe_calib_set_env(exact_env.empty() ? std::string() : exact_env.substr(1));
            entry.draft_exact     = best_draft_exact;
            entry.draft_share_pct = best_share_pct;
            checkpoint("draft cache share");
            if (best_share_pct > 0) {
                LOG_INF("%s: draft cache share: %d%% at %.2f tok/s\n", __func__, best_share_pct, best_share_tps);
                common_moe_calibration_status_note("draft cache share", string_format("%d%%", best_share_pct),
                        string_format("SELECTED - %.2f tok/s", best_share_tps), true, true);
            }
        }
    }

    // Now the deferred verdict, against the draft's best CONFIGURED throughput
    // rather than its worst. best_share_tps/cpu_tps only exist when those
    // stages ran, so fall back to the depth search's own number.
    if (best_n_max > 0 && no_draft_tps_to_beat > 0.0) {
        double configured_tps = best_n_max_tps;
        if (best_share_pct > 0 && best_share_tps > configured_tps) {
            configured_tps = best_share_tps;
        }
        if (configured_tps <= no_draft_tps_to_beat) {
            LOG_WRN("%s: the draft at its best measured configuration (%.2f tok/s) still does not beat no draft "
                    "at all (%.2f tok/s) on this machine - recording speculative decoding as measured slower; "
                    "later stages run without it\n", __func__, configured_tps, no_draft_tps_to_beat);
            common_moe_calibration_status_note("speculative decoding", "off",
                    string_format("SELECTED - %.2f tok/s beats the configured draft's %.2f",
                                  no_draft_tps_to_beat, configured_tps),
                    true, true);
            best_n_max = 0;
        } else {
            LOG_INF("%s: the draft beats no draft once configured (%.2f vs %.2f tok/s) - keeping speculative "
                    "decoding\n", __func__, configured_tps, no_draft_tps_to_beat);
            common_moe_calibration_status_note("speculative decoding", "on",
                    string_format("SELECTED - configured draft %.2f tok/s beats no draft's %.2f",
                                  configured_tps, no_draft_tps_to_beat),
                    true, true);
        }
    }

    // The draft every later candidate runs with - the substitution ladder, its
    // answer bar, the stand-in quality bar and the -ngl search. The depth the
    // search found, or the configured depth when it found nothing.
    const bool        sub_draft    = params.speculative.has_dft() && !params.speculative.draft.mparams.path.empty() && best_n_max != 0;
    const std::string sub_mtp_path = sub_draft ? params.speculative.draft.mparams.path : std::string();
    const int         sub_n_max    = !sub_draft ? 0 : (best_n_max > 0 ? best_n_max : std::max(1, params.speculative.draft.n_max));

    // Trade GPU-resident dense/attention-layer compute for VRAM the expert
    // cache converts into hit rate. The ncmoe search above only ever moved
    // MoE EXPERT weights off GPU - every layer's attention/dense compute
    // (the majority of the graph, on a model this sparsely routed) stayed
    // GPU-resident regardless, at whatever VRAM the fit search left for it.
    // If the CPU/NVMe path for cold experts is the actual bottleneck, giving
    // up some GPU-resident layers to grow the expert cache's own (auto-sized,
    // free-VRAM-minus-reserve) budget can pay for itself, the same way extra
    // -ncmoe offload sometimes does above - just on the layer-residency axis
    // instead of the expert-placement axis. Reaching this point already means
    // the model did not fit fully on GPU (probe.already_fits was false), so
    // there is always something to trade here.
    // Substitution aggressiveness, measured BEFORE the -ngl / thread / cache
    // stages rather than after them. Serving a cache miss with a resident
    // stand-in is the largest throughput lever measured on this model
    // (0.6 -> ~12 tok/s at its most aggressive) AND the one that destroys
    // output when overused, so it is also the one most worth spending a
    // constrained budget on. Ordering it last meant a slow model never
    // reached it: a real clean-slate run spent its whole budget on the
    // placement and -ngl searches and recorded substitute_min_rank=-1,
    // having measured the biggest lever not at all while fully exploring
    // -ngl, which moved almost nothing on the same model.
    //
    // Benchmarked at the winning placement with the other knobs still at
    // their defaults, which is sound because this is a router-quality
    // boundary (which picks may be served approximately) rather than a
    // memory-sizing decision - it does not depend on the thread count or
    // cache budget chosen later.
    int best_min_rank = -1;
    double best_min_rank_tps = 0.0;
    if (!common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: measuring substitution aggressiveness at ncmoe=%u (degenerate output is rejected, "
                "not ranked) ...\n", __func__, best_n);
        common_moe_calibration_status_set("measuring substitution aggressiveness");
        common_moe_stage_begin("substitution ladder", 9);
        // Coarse ladder rather than a golden-section search: the response is
        // a quality cliff, not a smooth curve, and each point costs a full
        // server spawn on a budget that is already tight for slow models.
        // Descend to 0, not to a floor picked in advance. An earlier version
        // stopped at 2 because live testing had shown rank 0 producing word
        // salad - but that made the floor an assumption baked into the code
        // rather than something this run measured, on a machine and model it
        // may not hold for. The degeneracy guard rejects whatever actually
        // breaks, so the ladder can safely ask the question instead.
        // Establish what this model says when nothing is substituted, and how
        // much its own sampling moves the answer around. A rank floor above
        // top_k can never fire, so this is a genuinely substitution-free run.
        //
        // Two reference runs, same config, different seeds: the first is the
        // reference text, the second measures how far apart two legitimate
        // answers from this model already sit. That distance becomes the
        // tolerance - a candidate has to stay at least as close to the
        // reference as the model's own resampling does. Measuring the bar
        // instead of picking one keeps it honest across models and hardware:
        // a model that answers near-identically run to run gets a strict bar
        // automatically, a chattier one gets a loose one, and neither number
        // is written down here.
        constexpr int subst_off_rank = 1000000;
        std::string ref_text;
        std::string ref_alt_text;
        double fidelity_bar = -1.0;
        const double ref_tps = common_moe_bench_candidate_server(
                self_exe, path_model, sub_mtp_path, best_n, sub_n_max, n_threads_default, next_port(), ctx, n_predict,
                concurrency, -1, -1, active_ngl, subst_off_rank, &ref_text, 1234);
        common_moe_calibration_status_candidate_done();
        if (ref_tps > 0 && !ref_text.empty() && !common_moe_calibrate_budget_spent()) {
            common_moe_bench_candidate_server(
                    self_exe, path_model, sub_mtp_path, best_n, sub_n_max, n_threads_default, next_port(), ctx, n_predict,
                    concurrency, -1, -1, active_ngl, subst_off_rank, &ref_alt_text, 5678);
            common_moe_calibration_status_candidate_done();
            if (!ref_alt_text.empty()) {
                fidelity_bar = common_moe_output_fidelity(ref_text, ref_alt_text);
                // A bar of 1.0 is not a strict measurement, it is the absence
                // of one. It means the two reference runs came back identical,
                // so this model showed no answer-to-answer variation to
                // calibrate a tolerance from - and a bar of exactly 1.0 then
                // demands that a candidate reproduce the reference token for
                // token, which no substitution can do however good its
                // stand-ins are. Observed live: the bar measured 1.00 and the
                // ladder rejected every rung at fidelity 0.62, including ones
                // that may well have been fine. With no evidence about natural
                // variation, the honest move is to not gate on it and let the
                // degeneracy and reproducibility checks carry the decision,
                // rather than to invent a tolerance here.
                if (fidelity_bar >= 1.0) {
                    LOG_INF("%s:   the two reference runs were identical, so there is no measured "
                            "answer-to-answer variation to set a fidelity tolerance from - judging the "
                            "ladder on degeneracy and reproducibility alone\n", __func__);
                    fidelity_bar = -1.0;
                }
            }
        }
        if (ref_tps > 0) {
            LOG_INF("%s:   substitution off -> %.2f tok/s (reference)%s\n", __func__, ref_tps,
                    fidelity_bar >= 0.0
                        ? string_format(", fidelity bar %.2f from the model's own resampling", fidelity_bar).c_str()
                        : ", no fidelity bar - candidates judged on degeneracy alone");
            best_min_rank_tps = ref_tps;
            best_min_rank     = subst_off_rank;
        }

        // Whether any rung was actually measured. The reference run alone takes
        // minutes on a model that does not fit, and on Qwen3.8-Flash-Next it used
        // the rest of the budget: the ladder never ran, and the "substitution
        // off" placeholder below was cached as if it had been measured, so every
        // later launch applied it and the runtime default never got a say.
        bool ladder_ran = false;
        std::vector<std::pair<int, double>> ladder_results; // (rank, cheap-probe tok/s)
        // One server for the whole ladder. The floor is a live policy knob, so each
        // rung is a POST; and because the process persists, every rung is measured
        // against a cache warmed the same way, which is what makes the rungs
        // comparable to each other rather than to their own cold starts.
        bool ladder_live = false;
        for (const int rank : {10, 6, 4, 2, 1, 0}) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            ladder_ran = true;
            std::string cand_text;
            const std::string body =
                    string_format("{\"GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK\": \"%d\"}", rank);
            double tps = -1.0;
            if (ladder_live && g_moe_live_port > 0) {
                tps = common_moe_bench_candidate_server(
                        self_exe, path_model, sub_mtp_path, best_n, sub_n_max, n_threads_default,
                        g_moe_live_port, ctx, n_predict, concurrency, -1, -1, active_ngl, rank,
                        &cand_text, 1234, std::numeric_limits<double>::quiet_NaN(), false, false,
                        std::string(), -1, std::string(), -1.0,
                        g_moe_live_port, body, /* keep_alive */ true);
            } else {
                tps = common_moe_bench_candidate_server(
                        self_exe, path_model, sub_mtp_path, best_n, sub_n_max, n_threads_default,
                        next_port(), ctx, n_predict, concurrency, -1, -1, active_ngl, rank,
                        &cand_text, 1234, std::numeric_limits<double>::quiet_NaN(), false, false,
                        std::string(), -1, std::string(), -1.0,
                        -1, std::string(), /* keep_alive */ true);
                ladder_live = g_moe_live_port > 0;
            }
            common_moe_calibration_status_candidate_done();
            // Reported, not a veto - the same mistake the reproducibility check
            // made, found the same way. Fidelity is text similarity to the
            // substitution-free reference, and substitution deliberately serves a
            // different expert than the router asked for, so the wording diverges
            // by design. Measured on Qwen3.8-Flash-Next: this rejected rank 6
            // (fidelity 0.30), rank 4 (0.00) and rank 2 (0.00), leaving rank 10 at
            // 6.18 tok/s - while rank 2 measures 13.19 tok/s in a controlled A/B
            // and produces plainly correct prose. A low similarity score means
            // "said differently", not "said wrongly", and no amount of tuning the
            // threshold fixes a metric that cannot tell those apart.
            //
            // Correctness is judged by things that do not depend on phrasing: the
            // degeneracy guard rejects broken generation, and the verifiable-answer
            // probes in common_moe_bench_candidate_server reject a candidate that
            // answers fewer checkable questions than this run's own reference did.
            // That is the fluent-but-wrong failure, caught by being wrong rather
            // than by being different.
            double fidelity = -1.0;
            if (tps > 0 && fidelity_bar >= 0.0 && !cand_text.empty()) {
                fidelity = common_moe_output_fidelity(ref_text, cand_text);
            }
            LOG_INF("%s:   substitute-min-rank=%d -> %s%s\n", __func__, rank,
                    tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed or rejected as degenerate",
                    fidelity >= 0.0 ? string_format(" (fidelity %.2f)", fidelity).c_str() : "");
            common_moe_calibration_status_note("substitution floor",
                    string_format("rank %d", rank),
                    tps > 0 ? string_format("%.2f tok/s%s", tps,
                            fidelity >= 0.0 ? string_format(", fidelity %.2f", fidelity).c_str() : "")
                            : std::string("failed or degenerate"),
                    tps > 0);
            if (tps > 0) {
                ladder_results.emplace_back(rank, tps);
            }
            if (tps > best_min_rank_tps) {
                best_min_rank_tps = tps;
                best_min_rank     = rank;
            }
        }

        // Shortlist, then decide in the regime we actually serve.
        //
        // The ladder ranks on short probes with reasoning disabled, which is
        // ~30% cheaper per candidate and lets more levers fit the budget - but
        // it is not the workload being served, and it does not reliably preserve
        // the ordering. Measured on Qwen3.8-Flash-Next: with thinking, rank 2
        // (7.69) beat rank 6 (7.21); without it the two tied at exactly 8.56 and
        // the tie went to rank 6 purely because `tps > best` keeps whichever was
        // measured first, and the ladder runs 10, 6, 4, 2 - so every tie breaks
        // toward the conservative end. That cost 11.48 -> 9.95 tok/s confirmed.
        //
        // So the cheap regime only narrows the field. Everything within 15% of
        // its best is re-measured at full length WITH reasoning, and the winner
        // is chosen on those numbers. Capped at three, because the confirm is the
        // most expensive single measurement in the run.
        //
        // 15%, not the 5% this first shipped with: the cheap probe's error against
        // the confirmed number is far larger than 5% - measured at +34% for rank 2
        // (8.60 -> 11.53) against +6% for rank 1 (8.72 -> 9.23) in the same run - so
        // a 5% window discards rungs the cheap number has no power to rule out.
        std::vector<std::pair<int, double>> shortlist;
        if (!ladder_results.empty()) {
            double best_cheap = 0.0;
            for (const auto & r : ladder_results) {
                best_cheap = std::max(best_cheap, r.second);
            }
            for (const auto & r : ladder_results) {
                if (r.second >= best_cheap * 0.85) {
                    shortlist.push_back(r);
                }
            }
            std::sort(shortlist.begin(), shortlist.end(),
                    [](const std::pair<int, double> & a, const std::pair<int, double> & b) {
                        // Best cheap number first; on a tie prefer the more
                        // aggressive rung, so a tie costs nothing if both confirm.
                        if (a.second != b.second) {
                            return a.second > b.second;
                        }
                        return a.first < b.first;
                    });
            if (shortlist.size() > 3) {
                shortlist.resize(3);
            }
        }
        // Confirm the winner over a LONGER generation before committing it.
        // The ladder's own probes are short (n_predict, kept small so more
        // levers fit the budget), and short-probe coherence does not prove
        // long-generation coherence: every degenerate output observed by hand
        // on this model appeared at 150-250 tokens, well past where these
        // probes stop looking. Rather than leave that as a known blind spot,
        // re-measure the winner at 4x the probe length and, if the guard
        // rejects it there, fall back one rung toward the safe end and
        // confirm that instead.
        if (!ladder_ran && best_min_rank == subst_off_rank) {
            LOG_WRN("%s:   substitution ladder did not run before the time budget was spent - recording it as "
                    "not calibrated rather than off, so the runtime default applies until a run measures it\n",
                    __func__);
            best_min_rank     = -1;
            best_min_rank_tps = 0.0;
        }
        common_moe_stage_end();   // the confirm below is part of the decision, not of the search
        common_moe_live_stop();   // the confirm step below runs at a different length, with reasoning on
        const int confirm_predict = n_predict * 4;

        // Confirm each shortlisted rung at full length, with reasoning on - the
        // configuration that will actually be served - and take the best of those.
        // Short probes that never reach 150-250 tokens cannot see the degenerate
        // output this model produces there, which is the other reason the confirm
        // exists: it is both the tie-break and the long-generation check.
        // The answer bar, measured before any candidate is judged against it:
        // substitution off, the confirm's own length, reasoning on, served
        // sampling - the same conditions every confirm runs under. Twice, with
        // different seeds, so the tolerance is the model's own variation.
        if (!shortlist.empty() && !common_moe_calibrate_budget_spent()) {
            g_moe_verifying_reference.store(true);
            for (const int seed : {1234, 5678}) {
                if (common_moe_calibrate_budget_spent()) {
                    break;
                }
                common_moe_calibration_status_set("measuring the answer bar with substitution off");
                common_moe_bench_candidate_server(
                        self_exe, path_model, sub_mtp_path, best_n, sub_n_max, n_threads_default, next_port(), ctx,
                        confirm_predict, concurrency, -1, -1, active_ngl, subst_off_rank,
                        nullptr, seed, std::numeric_limits<double>::quiet_NaN(),
                        /* verify_answers */ true, /* with_reasoning */ true);
                common_moe_calibration_status_candidate_done();
            }
            g_moe_verifying_reference.store(false);
            const int r1 = g_moe_verifiable_ref.load();
            const int r2 = g_moe_verifiable_ref2.load();
            if (r1 >= 0) {
                LOG_INF("%s:   answer bar from substitution off: %d%s\n", __func__,
                        r2 >= 0 ? std::min(r1, r2) : r1,
                        r2 >= 0 ? string_format(" (runs %d and %d, tolerance %d)", r1, r2, std::abs(r1 - r2)).c_str()
                                : " (one run, no tolerance)");
            }
        }

        int    confirmed_rank = -1;
        double confirmed_tps  = 0.0;
        for (const auto & cand : shortlist) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            common_moe_calibration_status_set(string_format(
                    "confirming substitution floor rank %d over %d tokens", cand.first, confirm_predict));
            double tps = common_moe_bench_candidate_server(
                    self_exe, path_model, sub_mtp_path, best_n, sub_n_max, n_threads_default, next_port(), ctx,
                    confirm_predict, concurrency, -1, -1, active_ngl, cand.first,
                    nullptr, 1234, std::numeric_limits<double>::quiet_NaN(),
                    /* verify_answers */ true, /* with_reasoning */ true);
            common_moe_calibration_status_candidate_done();
            // Re-measure a rejection before accepting it, exactly as a win is
            // re-measured before it is believed. The answer check runs eight
            // binary probes on one generation; a candidate can lose two of them
            // to sampling and a 59% throughput difference should not turn on
            // that. Measured here: rank 4 at 9.91 tok/s was discarded for 6 of
            // 8 against the reference's 8, and rank 6 at 6.25 was taken
            // instead.
            //
            // Only for a quality rejection. An infrastructure failure or a
            // configuration that cannot launch is deterministic and retrying it
            // buys nothing - the sentinels already separate those three cases.
            if (tps == COMMON_MOE_TPS_REJECTED && !common_moe_calibrate_budget_spent()) {
                LOG_INF("%s:   rank %d failed the answer check - re-measuring before discarding it, since "
                        "eight probes on one generation is a thin basis for dropping a rung\n",
                        __func__, cand.first);
                tps = common_moe_bench_candidate_server(
                        self_exe, path_model, sub_mtp_path, best_n, sub_n_max, n_threads_default,
                        next_port(), ctx, confirm_predict, concurrency, -1, -1, active_ngl, cand.first,
                        nullptr, 5678 /* a different seed, or it is the same draw */,
                        std::numeric_limits<double>::quiet_NaN(),
                        /* verify_answers */ true, /* with_reasoning */ true);
                common_moe_calibration_status_candidate_done();
                if (tps > 0) {
                    LOG_INF("%s:   rank %d held on the second run -> %.2f tok/s (the first run's rejection "
                            "did not reproduce)\n", __func__, cand.first, tps);
                    common_moe_calibration_status_note("substitution floor",
                            string_format("rank %d", cand.first),
                            string_format("held on re-measure - %.2f tok/s", tps), true);
                }
            }
            if (tps <= 0) {
                LOG_WRN("%s:   rank %d did NOT hold over %d tokens (twice)\n", __func__, cand.first, confirm_predict);
                common_moe_calibration_status_note("substitution floor",
                        string_format("rank %d", cand.first),
                        string_format("did not hold over %d tokens", confirm_predict), false);
                continue;
            }
            LOG_INF("%s:   rank %d confirmed over %d tokens -> %.2f tok/s (cheap probe said %.2f)\n",
                    __func__, cand.first, confirm_predict, tps, cand.second);
            common_moe_calibration_status_note("substitution floor",
                    string_format("rank %d", cand.first),
                    string_format("confirmed over %d tokens - %.2f tok/s", confirm_predict, tps), true);
            if (tps > confirmed_tps) {
                confirmed_tps  = tps;
                confirmed_rank = cand.first;
            }
        }

        if (confirmed_rank >= 0) {
            best_min_rank     = confirmed_rank;
            best_min_rank_tps = confirmed_tps;
            // Keep the winner's CHEAP-regime number too. Every later stage
            // measures its candidates with the short no-reasoning probe, so it
            // needs an incumbent measured the same way to compare against - see
            // cheap_incumbent_tps below.
            for (const auto & r : ladder_results) {
                if (r.first == confirmed_rank) {
                    cheap_incumbent_tps = r.second;
                    break;
                }
            }
        } else {
            // Nothing on the shortlist survived its long generation. Step toward
            // the safe end from the cheap ladder's best and confirm that instead,
            // rather than committing a rung no long run ever validated.
            while (best_min_rank >= 0 && !common_moe_calibrate_budget_spent()) {
                const int safer = best_min_rank < 2 ? 2 : (best_min_rank < 4 ? 4 : (best_min_rank < 6 ? 6 : 10));
                if (safer == best_min_rank || safer > 10) {
                    best_min_rank = -1; // leave the runtime default in place
                    break;
                }
                best_min_rank = safer;
                common_moe_calibration_status_set(string_format(
                        "confirming substitution floor rank %d over %d tokens", best_min_rank, confirm_predict));
                const double tps = common_moe_bench_candidate_server(
                        self_exe, path_model, sub_mtp_path, best_n, sub_n_max, n_threads_default, next_port(), ctx,
                        confirm_predict, concurrency, -1, -1, active_ngl, best_min_rank,
                        nullptr, 1234, std::numeric_limits<double>::quiet_NaN(),
                        /* verify_answers */ true, /* with_reasoning */ true);
                common_moe_calibration_status_candidate_done();
                if (tps > 0) {
                    best_min_rank_tps = tps;
                    LOG_INF("%s:   rank %d confirmed over %d tokens -> %.2f tok/s\n",
                            __func__, best_min_rank, confirm_predict, tps);
                    common_moe_calibration_status_note("substitution floor",
                            string_format("rank %d", best_min_rank),
                            string_format("confirmed over %d tokens - %.2f tok/s", confirm_predict, tps), true);
                    break;
                }
                LOG_WRN("%s:   rank %d did NOT hold over %d tokens - stepping back further\n",
                        __func__, best_min_rank, confirm_predict);
            }
        }
        if (best_min_rank >= 0) {
            LOG_INF("%s: substitution floor: rank %d at %.2f tok/s - carried into the remaining stages\n",
                    __func__, best_min_rank, best_min_rank_tps);
            common_moe_calibration_status_note("substitution floor",
                    string_format("rank %d", best_min_rank),
                    string_format("SELECTED - %.2f tok/s, carried into the remaining stages", best_min_rank_tps),
                    true, /* chosen */ true);
            // Everything measured from here on is measured at this floor, so
            // the later knobs are tuned against the configuration that will
            // actually be served.
            active_min_rank = best_min_rank;
            best_tps = std::max(best_tps, best_min_rank_tps);
        }

        // Stand-in quality bar, measured at the floor just chosen. The rank
        // floor answers "how much did the router want this expert"; this
        // answers "is the replacement on offer actually any good", and it is
        // the one that decides when inference waits for a real fetch and when
        // it runs on what is already resident. Descending order because the
        // strict end is the safe end: a high sigma demands a stand-in far
        // better than typical and mostly declines, which costs throughput and
        // nothing else, while the permissive end is where output dies - on
        // this hardware an ungated run reaches 51 tok/s emitting "* * * *"
        // and nothing more. Every rung goes through the same degeneracy,
        // fidelity and reproducibility gates as the ladder above, so the
        // permissive end is rejected on evidence rather than avoided by a
        // number written here.
        double best_sigma = std::numeric_limits<double>::quiet_NaN();
        double best_sigma_tps = cheap_incumbent_tps > 0.0 ? cheap_incumbent_tps : best_min_rank_tps;
        if (!common_moe_calibrate_budget_spent()) {
            LOG_INF("%s: measuring stand-in quality bar at rank %d (this is the wait-or-substitute "
                    "balance) ...\n", __func__, active_min_rank);
            common_moe_calibration_status_set("measuring stand-in quality bar");
            // On the server the substitution ladder left open. The bar is read
            // per substitution decision through the tunables registry (see
            // moe_cache_substitute_quality_sigma), so a POST changes the very
            // next token - five model loads become five requests. Falls back to
            // a spawn per candidate only if no live server survived the ladder.
            bool sigma_live = g_moe_live_port > 0;
            for (const double sigma : {2.0, 1.0, 0.0, -1.0, -2.0}) {
                if (common_moe_calibrate_budget_spent()) {
                    break;
                }
                std::string cand_text;
                double tps = -1.0;
                if (sigma_live && g_moe_live_port > 0) {
                    const std::string body = string_format(
                            "{\"GGML_CUDA_MOE_CACHE_SUBSTITUTE_QUALITY_SIGMA\": \"%.1f\"}", sigma);
                    tps = common_moe_bench_candidate_server(
                            self_exe, path_model, sub_mtp_path, best_n, sub_n_max, n_threads_default,
                            g_moe_live_port, ctx, n_predict, concurrency, -1, -1, active_ngl,
                            active_min_rank, &cand_text, 1234, sigma, false, false,
                            std::string(), -1, std::string(), -1.0,
                            g_moe_live_port, body, /* keep_alive */ true);
                } else {
                    tps = common_moe_bench_candidate_server(
                            self_exe, path_model, sub_mtp_path, best_n, sub_n_max, n_threads_default,
                            next_port(), ctx, n_predict, concurrency, -1, -1, active_ngl,
                            active_min_rank, &cand_text, 1234, sigma, false, false,
                            std::string(), -1, std::string(), -1.0,
                            -1, std::string(), /* keep_alive */ true);
                    sigma_live = g_moe_live_port > 0;
                }
                common_moe_calibration_status_candidate_done();
                // Reported, not a veto - the same fix the substitution ladder needed,
                // for the same reason. Fidelity is similarity to the substitution-free
                // reference, and a stand-in is a different expert by construction, so
                // divergence is the mechanism working rather than failing. Proved on
                // Qwen3.8-Flash-Next: substitute-min-rank=2 scores fidelity 0.00 and
                // answers 4 of 4 verifiable probes correctly. A metric that rates a
                // demonstrably correct answer at zero cannot be a rejection criterion.
                // Degeneracy still rejects broken generation per candidate, and the
                // confirm step verifies the winner's answers.
                double fidelity = -1.0;
                if (tps > 0 && fidelity_bar >= 0.0 && !cand_text.empty()) {
                    fidelity = common_moe_output_fidelity(ref_text, cand_text);
                }
                LOG_INF("%s:   quality-sigma=%.1f -> %s%s\n", __func__, sigma,
                        tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed or rejected",
                        fidelity >= 0.0 ? string_format(" (fidelity %.2f)", fidelity).c_str() : "");
                common_moe_calibration_status_note("stand-in quality bar",
                        string_format("%+.1f sigma", sigma),
                        tps > 0 ? string_format("%.2f tok/s%s", tps,
                                fidelity >= 0.0 ? string_format(", fidelity %.2f", fidelity).c_str() : "")
                                : std::string("failed or rejected"),
                        tps > 0);
                if (tps > best_sigma_tps) {
                    best_sigma_tps = tps;
                    best_sigma     = sigma;
                }
            }
            // Confirm it in the regime it will be served in, exactly as the
            // substitution floor is. The ladder above measures with the short
            // no-reasoning probe, and a stand-in bar can break a failure mode that
            // regime cannot see: measured on Qwen3.8-Flash-Next, -2 sigma passed
            // every cheap probe and every degeneracy check, then served an empty
            // content field with 200 tokens of unterminated reasoning, and a bare
            // "</think>" for the next prompt. Speed was fine. Output was destroyed.
            // Nothing downstream can catch this, because the calibration that
            // chose it never ran with reasoning on.
            // The confirm below runs at a different length with reasoning on -
            // a different regime, so it gets its own server, exactly as the
            // substitution ladder's confirm does.
            common_moe_live_stop();
            if (!std::isnan(best_sigma) && !common_moe_calibrate_budget_spent()) {
                const int sigma_confirm_predict = n_predict * 4;
                common_moe_calibration_status_set(string_format(
                        "confirming stand-in quality bar %+.1f sigma over %d tokens",
                        best_sigma, sigma_confirm_predict));
                const double ctps = common_moe_bench_candidate_server(
                        self_exe, path_model, sub_mtp_path, best_n, sub_n_max, n_threads_default, next_port(), ctx,
                        sigma_confirm_predict, concurrency, -1, -1, active_ngl, active_min_rank,
                        nullptr, 1234, best_sigma, /* verify_answers */ true, /* with_reasoning */ true);
                common_moe_calibration_status_candidate_done();
                if (ctps > 0) {
                    best_sigma_tps = ctps;
                    LOG_INF("%s:   %+.1f sigma confirmed over %d tokens -> %.2f tok/s\n",
                            __func__, best_sigma, sigma_confirm_predict, ctps);
                } else {
                    LOG_WRN("%s:   %+.1f sigma did NOT hold with reasoning on - leaving the stand-in bar "
                            "uncalibrated rather than shipping it\n", __func__, best_sigma);
                    common_moe_calibration_status_note("stand-in quality bar",
                            string_format("%+.1f sigma", best_sigma),
                            std::string("did not hold with reasoning on - not committed"), false);
                    best_sigma = std::numeric_limits<double>::quiet_NaN();
                }
            }
            if (!std::isnan(best_sigma)) {
                LOG_INF("%s: stand-in quality bar: %.1f sigma at %.2f tok/s\n",
                        __func__, best_sigma, best_sigma_tps);
                common_moe_calibration_status_note("stand-in quality bar",
                        string_format("%+.1f sigma", best_sigma),
                        string_format("SELECTED - %.2f tok/s", best_sigma_tps), true, /* chosen */ true);
                best_tps = std::max(best_tps, best_sigma_tps);
            }
        }
        active_quality_sigma = best_sigma;
    }

    // Depth again, now that substitution is chosen. The depth search ran at the
    // runtime's default substitution, and the two interact both ways: a deeper
    // draft widens each verify pass's expert union, which changes how often a
    // stand-in is needed; a floor changes what each verify pass costs. Top
    // candidates only - the winner and the runner-up, both measured fresh under
    // the chosen floor and stand-in bar so neither carries a number from the
    // other regime.
    if (best_n_max > 0 && depth_runner_up > 0 && !common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: re-checking speculative depth %d vs %d at the chosen substitution settings ...\n",
                __func__, best_n_max, depth_runner_up);
        common_moe_calibration_status_set("re-checking speculative depth after substitution");
        double recheck_tps   = -1.0;
        int    recheck_depth = -1;
        for (const int depth : { best_n_max, depth_runner_up }) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            const double tps = common_moe_bench_candidate_server(
                    self_exe, path_model, sub_mtp_path, best_n, depth, n_threads_default, next_port(), ctx,
                    n_predict, concurrency, -1, -1, active_ngl, active_min_rank, nullptr, 1234,
                    active_quality_sigma);
            common_moe_calibration_status_candidate_done();
            LOG_INF("%s:   spec-draft-n-max=%d after substitution -> %s%s\n", __func__, depth,
                    tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed",
                    tps > 0 ? common_moe_last_acceptance_str().c_str() : "");
            common_moe_calibration_status_note("speculative depth (after substitution)",
                    string_format("n-max %d", depth),
                    tps > 0 ? string_format("%.2f tok/s%s", tps, common_moe_last_acceptance_str().c_str())
                            : std::string("failed"), tps > 0);
            if (tps > recheck_tps) {
                recheck_tps   = tps;
                recheck_depth = depth;
            }
        }
        if (recheck_depth > 0) {
            if (recheck_depth != best_n_max) {
                LOG_INF("%s: speculative depth moves %d -> %d under the chosen substitution settings\n",
                        __func__, best_n_max, recheck_depth);
                best_n_max = recheck_depth;
            }
            common_moe_calibration_status_note("speculative depth (after substitution)",
                    string_format("n-max %d", best_n_max),
                    string_format("SELECTED - %.2f tok/s", recheck_tps), true, true);
        }
    }

    uint32_t best_ngl = (uint32_t) probe.n_layer + 1; // "all" - mirrors llama_model::n_gpu_layers()'s own +1
    // -ngl only ever had something to trade because a layer's experts were
    // resident. With every layer's experts on the CPU (the design above), the
    // only things -ngl still governs are attention, the norms and the output
    // head - a few hundred MiB that every token passes through twice. Moving
    // those to the CPU to buy expert-cache VRAM is the trade run backwards:
    // it makes the hot path slower to enlarge a cache that is already sized by
    // whatever the card has left. Measured on gemma-4 under this design the
    // cache grew 4197 -> 11277 MiB on its own, so there was no VRAM for a
    // lowered -ngl to free that the cache had not already taken.
    //
    // The search also costs more than its own candidates: when it "wins" it
    // re-runs the whole -ncmoe search underneath the new residency.
    // GGML_MOE_CALIBRATE_SEARCH_NGL=1 restores it for a card where attention
    // is big enough relative to VRAM for the trade to go the other way.
    const bool search_ngl = [] {
        const char * e = getenv("GGML_MOE_CALIBRATE_SEARCH_NGL");
        return e && atoi(e) != 0;
    }();
    if (!search_ngl) {
        LOG_WRN("%s: -ngl left at full residency - with every layer's experts on the CPU, -ngl governs "
                "only attention and the norms, and the expert cache already grows into whatever VRAM is "
                "left (GGML_MOE_CALIBRATE_SEARCH_NGL=1 to search layer residency instead)\n", __func__);
        common_moe_calibration_status_note("GPU residency", "full (default -ngl)",
                "DESIGN - experts stream from the CPU; the GPU holds attention, the MTP head and the cache",
                true, true);
    } else {
        const uint32_t ngl_hi = best_ngl;
        const uint32_t ngl_lo = probe.n_layer > 3 ? probe.n_layer - probe.n_layer / 3 : 0;
        LOG_INF("%s: golden-section search for -ngl in [%u, %u] at ncmoe=%u (trading GPU-resident layers "
                "for expert-cache VRAM) ...\n", __func__, ngl_lo, ngl_hi, best_n);
        common_moe_calibration_status_set(string_format("searching GPU-resident layer count (-ngl) in [%u, %u]", ngl_lo, ngl_hi));

        std::map<int, double> ngl_trace;
        // Seeded in the SAME regime the candidates below are measured in. This
        // used to take best_tps, which after the substitution ladder holds a
        // full-length confirmed number: the search then compared 32-token
        // no-reasoning candidates (5.55-8.09 measured) against a confirmed 12.02
        // seed, so best_ngl always landed on ngl_hi and "full GPU residency still
        // wins" was true by construction rather than by measurement.
        ngl_trace[(int) ngl_hi] = cheap_incumbent_tps > 0.0 ? cheap_incumbent_tps : best_tps;
        // bench_with_retry doesn't take an ngl override, so this wraps
        // common_moe_bench_candidate_server directly instead.
        auto measure_ngl = [&](int n) -> double {
            double sum = 0.0; int n_ok = 0;
            for (int i = 0; i < n_samples_per_candidate; i++) {
                double tps = common_moe_bench_candidate_server(
                        self_exe, path_model, sub_mtp_path, best_n, sub_n_max, n_threads_default, next_port(), ctx, n_predict,
                        concurrency, -1, -1, n, active_min_rank);
                // Retry only an infrastructure failure - a quality rejection is deterministic.
                if (tps < 0 && tps != COMMON_MOE_TPS_REJECTED && tps != COMMON_MOE_TPS_LAUNCH_FAILED) {
                    tps = common_moe_bench_candidate_server(
                            self_exe, path_model, sub_mtp_path, best_n, sub_n_max, n_threads_default, next_port(), ctx, n_predict,
                            concurrency, -1, -1, n, active_min_rank);
                }
                if (tps > 0) { sum += tps; n_ok++; }
            }
            const double result = n_ok > 0 ? sum / n_ok : -1.0;
            LOG_INF("%s:   ngl=%d -> %s\n", __func__, n, result > 0 ? string_format("%.2f tok/s", result).c_str() : "failed");
            // This stage advanced the progress bar but never wrote a row, so the
            // decisions table silently omitted every GPU-residency candidate - the
            // one stage whose candidates are the most expensive to re-run by hand.
            common_moe_calibration_status_note("GPU-resident layers",
                    string_format("-ngl %d", n),
                    result > 0 ? string_format("%.2f tok/s", result) : std::string("failed"),
                    result > 0);
            common_moe_calibration_status_candidate_done();
            return result;
        };

        best_ngl = (uint32_t) common_golden_section_search_max((int) ngl_lo, (int) ngl_hi, measure_ngl, ngl_trace);
        double best_ngl_tps = ngl_trace.at((int) best_ngl);
        for (const auto & kv : ngl_trace) {
            if (kv.second > best_ngl_tps) {
                best_ngl_tps = kv.second;
                best_ngl     = (uint32_t) kv.first;
            }
        }
        if (best_ngl == ngl_lo && ngl_lo > 0) {
            LOG_INF("%s: -ngl peak landed on the search boundary (%u) - throughput was still rising as more "
                    "layers moved to CPU, extending the range to [0, %u]\n", __func__, ngl_lo, ngl_lo);
            common_golden_section_search_max(0, (int) ngl_lo, measure_ngl, ngl_trace);
            for (const auto & kv : ngl_trace) {
                if (kv.second > best_ngl_tps) {
                    best_ngl_tps = kv.second;
                    best_ngl     = (uint32_t) kv.first;
                }
            }
        }

        const double ngl_incumbent = cheap_incumbent_tps > 0.0 ? cheap_incumbent_tps : best_tps;
        if (best_ngl_tps > ngl_incumbent && best_ngl < ngl_hi) {
            LOG_INF("%s: -ngl=%u wins over full GPU residency (%.2f vs %.2f tok/s) - re-searching ncmoe at "
                    "this layer residency, since the safe floor and available VRAM both just changed\n",
                    __func__, best_ngl, best_ngl_tps, best_tps);
            common_moe_calibration_status_set(string_format("re-searching -ncmoe at -ngl=%u", best_ngl));
            best_tps = best_ngl_tps;
            common_moe_calibration_status_note("GPU residency", string_format("ngl=%u", best_ngl),
                    string_format("SELECTED - %.2f tok/s, beats full residency's %.2f",
                                  best_ngl_tps, ngl_incumbent), true, true);

            llama_model_params mparams_ngl = mparams;
            mparams_ngl.n_gpu_layers = (int) best_ngl;
            // Same serving-time margin as the initial probe above - a re-search
            // floor computed on a bare fit would reintroduce exactly the
            // mismatch that made the first search's winner unusable.
            common_moe_fit_probe_result probe_ngl =
                    common_moe_find_safe_layers(path_model, mparams_ngl, cparams, fit_margin);
            const uint32_t safe_n_ngl = probe_ngl.already_fits ? 0 :
                    (probe_ngl.found_safe_n ? probe_ngl.safe_n : safe_n);
            const uint32_t ncmoe_hi_ngl = std::min<uint32_t>(probe.n_layer, safe_n_ngl + std::max<uint32_t>(4, probe.n_layer / 8));

            if (safe_n_ngl < ncmoe_hi_ngl) {
                std::map<int, double> ncmoe_trace2;
                auto measure_ncmoe2 = [&](int n) -> double {
                    double tps = common_moe_bench_candidate_server(
                            self_exe, path_model, sub_mtp_path, (uint32_t) n, sub_n_max, n_threads_default, next_port(), ctx, n_predict,
                            concurrency, -1, -1, (int) best_ngl, active_min_rank);
                    // Retry only an infrastructure failure - a quality rejection is deterministic.
                    if (tps < 0 && tps != COMMON_MOE_TPS_REJECTED && tps != COMMON_MOE_TPS_LAUNCH_FAILED) {
                        tps = common_moe_bench_candidate_server(
                                self_exe, path_model, sub_mtp_path, (uint32_t) n, sub_n_max, n_threads_default, next_port(), ctx, n_predict,
                                concurrency, -1, -1, (int) best_ngl, active_min_rank);
                    }
                    LOG_INF("%s:   ncmoe=%d (ngl=%u) -> %s\n", __func__, n, best_ngl,
                            tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
                    common_moe_calibration_status_candidate_done();
                    return tps;
                };
                uint32_t best_n2 = (uint32_t) common_golden_section_search_max(
                        (int) safe_n_ngl, (int) ncmoe_hi_ngl, measure_ncmoe2, ncmoe_trace2);
                double best_tps2 = ncmoe_trace2.at((int) best_n2);
                for (const auto & kv : ncmoe_trace2) {
                    if (kv.second > best_tps2) {
                        best_tps2 = kv.second;
                        best_n2   = (uint32_t) kv.first;
                    }
                }
                if (best_tps2 > best_tps) {
                    LOG_INF("%s: ncmoe=%u wins at ngl=%u (%.2f tok/s, was %.2f)\n",
                            __func__, best_n2, best_ngl, best_tps2, best_tps);
                    best_n   = best_n2;
                    best_tps = best_tps2;
                }
            }
        } else {
            // Report the best ALTERNATIVE, not best_ngl_tps. When full residency
            // is itself the winning candidate, best_ngl_tps is the incumbent's
            // own score, so this printed the same number twice - "14.25 vs
            // 14.25 tok/s" - which reads as a tie nobody can act on rather than
            // as "nothing beat leaving it alone".
            double best_alt_tps = -1.0;
            int    best_alt_ngl = -1;
            for (const auto & kv : ngl_trace) {
                if ((uint32_t) kv.first != ngl_hi && kv.second > best_alt_tps) {
                    best_alt_tps = kv.second;
                    best_alt_ngl = kv.first;
                }
            }
            if (best_alt_ngl >= 0) {
                LOG_INF("%s: full GPU residency wins (%.2f tok/s) - the best alternative was ngl=%d at "
                        "%.2f tok/s, so -ngl is left at default\n",
                        __func__, ngl_incumbent, best_alt_ngl, best_alt_tps);
                common_moe_calibration_status_note("GPU residency", "full (default -ngl)",
                        string_format("SELECTED - %.2f tok/s, best alternative ngl=%d at %.2f",
                                      ngl_incumbent, best_alt_ngl, best_alt_tps), true, true);
            } else {
                LOG_INF("%s: full GPU residency wins (%.2f tok/s) - no alternative -ngl measured usable\n",
                        __func__, ngl_incumbent);
                common_moe_calibration_status_note("GPU residency", "full (default -ngl)",
                        string_format("SELECTED - %.2f tok/s, no usable alternative", ngl_incumbent),
                        true, true);
            }
            best_ngl = ngl_hi;
        }
        // Carry it forward, so the thread / cache-size / fit-margin stages
        // below measure at the layer residency actually chosen.
        if (best_ngl <= (uint32_t) probe.n_layer) {
            active_ngl = (int) best_ngl;
        }
    }

    // Now uses the same correct subprocess/chat-template path as ncmoe and
    // n_max above, so - unlike before this fix - it's safe to also tune
    // threads under MTP when MTP was calibrated: same real generation
    // conditions, numbers are directly comparable to best_tps above.
    // Every later stage - threads, cache size, fit margin, admission, acceptance -
    // must be measured in the regime the server will actually serve in. If a draft
    // model is configured, that regime includes the draft, whether or not its depth
    // search produced a number. The old form dropped the draft entirely when the
    // depth search failed, so a run launched with -md tuned all four remaining knobs
    // against a no-draft configuration it would never run: measured on qwen4exp,
    // where one failed n_max=1 probe silently turned the whole run into a
    // calibration of the wrong thing. Fall back to the depth the run is configured
    // with instead; spec_n_max simply stays uncalibrated.
    const bool  mtp_configured   = params.speculative.has_dft() && !params.speculative.draft.mparams.path.empty() && best_n_max != 0;
    const std::string mtp_path_for_threads = mtp_configured ? params.speculative.draft.mparams.path : std::string();
    const int n_max_for_threads = best_n_max > 0
            ? best_n_max
            : (mtp_configured ? std::max(1, params.speculative.draft.n_max) : 0);


    const int n_threads_physical = common_cpu_get_num_physical_cores();
    const int n_threads_logical  = (int) std::thread::hardware_concurrency();
    std::vector<int> thread_candidates = { n_threads_default };
    // An explicit -t is an instruction, not an opening bid. When the launch
    // value already equals the physical core count - the answer this sweep has
    // reached on every run here - the remaining candidate is SMT, which oversubscribes
    // the cores the expert streaming already saturates.
    // GGML_MOE_CALIBRATE_SWEEP_THREADS=1 sweeps it anyway.
    const bool sweep_threads = [] {
        const char * e = getenv("GGML_MOE_CALIBRATE_SWEEP_THREADS");
        return e && atoi(e) != 0;
    }();
    if (!sweep_threads && n_threads_default == n_threads_physical) {
        LOG_WRN("%s: thread count left at %d - the launch value already equals the physical core count, "
                "so the only remaining candidate is SMT over cores the expert streaming already saturates "
                "(GGML_MOE_CALIBRATE_SWEEP_THREADS=1 to sweep it)\n", __func__, n_threads_default);
        common_moe_calibration_status_note("thread count", string_format("%d threads", n_threads_default),
                "LAUNCH VALUE - equals the physical core count", true, true);
    } else {
        if (n_threads_physical > 0 && n_threads_physical != n_threads_default) {
            thread_candidates.push_back(n_threads_physical);
        }
        if (n_threads_logical > 0 && n_threads_logical != n_threads_default && n_threads_logical != n_threads_physical) {
            thread_candidates.push_back(n_threads_logical);
        }
    }

    int    best_threads = n_threads_default;
    double best_threads_tps = cheap_incumbent_tps > 0.0 ? cheap_incumbent_tps : best_tps;
    if (thread_candidates.size() > 1) {
        common_moe_calibration_status_set(string_format("benchmarking %zu thread-count candidate(s)", thread_candidates.size()));
        LOG_INF("%s: benchmarking %zu thread-count candidate(s) at ncmoe=%u%s ...\n", __func__, thread_candidates.size(), best_n,
                best_n_max > 0 ? string_format(", spec-draft-n-max=%d", best_n_max).c_str() : "");
        // Record the incumbent explicitly. It is measured - the ladder above ran
        // at it - but skipping it silently left the table showing only the
        // candidate that lost, reading as though the winning thread count had
        // never been tried at all.
        common_moe_calibration_status_note("thread count",
                string_format("%d threads", n_threads_default),
                string_format("%.2f tok/s (incumbent - every stage above ran at this)", best_threads_tps),
                true);
        for (int nt : thread_candidates) {
            if (nt == n_threads_default) {
                continue; // already measured above, as the incumbent just recorded
            }
            const double tps = bench_with_retry(best_n, n_max_for_threads, mtp_path_for_threads, nt);
            LOG_INF("%s:   n_threads=%d -> %s\n", __func__, nt, tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
            common_moe_calibration_status_note("thread count", string_format("%d threads", nt),
                    tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
            common_moe_calibration_status_candidate_done();
            if (tps > best_threads_tps) {
                best_threads_tps = tps;
                best_threads     = nt;
            }
        }
    }

    // Measure the remainder, rather than assume the card is empty.
    //
    // calib_free_vram_bytes above is the free VRAM with NOTHING loaded, and the
    // expert-cache ladder was derived from it - which is why an 8192 MiB rung
    // was ever a candidate on a 12 GiB card. It cannot coexist with the model,
    // a 64k KV cache, the compute buffers and the draft; measured by hand, the
    // real remainder at 64k/f16 is 1405 MiB, and 2393 MiB at q8_0 KV. So the
    // ladder was spending most of its candidates on rungs that could not exist,
    // and the ones that "won" won by a margin inside the noise of rungs that
    // were never allocated in the first place (see the slab-shortfall
    // accounting in moe-cache.cu - a pool asked for a size and silently took a
    // fraction of it).
    //
    // The fix is to measure it: hold one candidate open with the cache off,
    // read the device memory the parent can see while the child has the model
    // loaded, and treat what is left as the budget everything else divides.
    // That is a real number for the configuration being calibrated rather than
    // a property of an idle card.
    auto measure_remainder = [&](const char * why) -> size_t {
        if (common_moe_calibrate_budget_spent()) {
            return 0;
        }
        LOG_INF("%s: measuring the VRAM remainder %s (one candidate, expert cache off) ...\n", __func__, why);
        const double r = common_moe_bench_candidate_server(
                self_exe, path_model,
                (params.speculative.has_dft() && best_n_max != 0)
                    ? params.speculative.draft.mparams.path : std::string(),
                probe.n_layer,
                (params.speculative.has_dft() && best_n_max != 0)
                    ? std::max(1, params.speculative.draft.n_max) : 0,
                n_threads_default, next_port(), ctx, n_predict, concurrency,
                0 /* expert cache off - this measures the floor, not the cache */, -1,
                99, -1, nullptr, 1234,
                std::numeric_limits<double>::quiet_NaN(), false, false, std::string(), -1,
                std::string(), -1.0, -1, std::string(), true /* keep alive */);
        size_t left = 0;
        if (r > 0 || g_moe_live_port > 0) {
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    continue;
                }
                size_t dfree = 0, dtotal = 0;
                ggml_backend_dev_memory(dev, &dfree, &dtotal);
                left = std::max(left, std::min(dfree, dtotal));
            }
        }
        common_moe_live_stop();
        if (left > 0) {
            LOG_WRN("%s: the expert cache and everything that divides it have %zu MiB to work with, not the "
                    "%zu MiB an idle card reports\n", __func__, left >> 20, calib_free_vram_bytes >> 20);
            common_moe_calibration_status_note("VRAM remainder", string_format("%zu MiB", left >> 20),
                    string_format("measured with the model loaded (idle card reports %zu MiB)",
                                  calib_free_vram_bytes >> 20), true, true);
        } else {
            LOG_WRN("%s: could not measure the remainder - falling back to the idle-card figure, which "
                    "overstates it\n", __func__);
        }
        return left;
    };


    // Expert-cache size: same question as -ncmoe, same answer - measure this
    // model's own knee rather than assume one. Our own sweep found the
    // relationship is not "more is better": Nemotron flattened at 4 GiB (of a
    // 7-8 GiB default) with no loss, while a too-small request starved the
    // fit search's VRAM margin and collapsed concurrent slots entirely. Start
    // small and grow only while it still pays; stop and take the smallest
    // point once growth stops paying, mirroring the n_max envelope-doubling
    // approach below rather than a full sweep of every candidate.
    LOG_INF("%s: finding expert-cache size knee (growing while it still helps) at ncmoe=%u ...\n",
            __func__, best_n);
    common_moe_calibration_status_set("searching expert-cache VRAM budget size");
    // Derived from this GPU, not a fixed list. The runtime clamps the cache to
    // what is actually free after the model and KV cache land
    // (available = free + already-allocated - reserve), so any rung above that
    // ceiling is silently the same configuration as the one below it: on a
    // 12 GiB card the old ladder's 6144/8192/12288/16384 rungs were four extra
    // server spawns all measuring the same thing, on a stage that already ran
    // out of time before reaching them. Measured on this box: a 12 GiB card
    // serving Qwen3.8-Flash-Next settles at a 3966 MiB budget, so the ceiling
    // is what free VRAM allows, never the card's nominal size.
    //
    // 0 = cache off, measured first. The pick below takes the smallest
    // candidate within 3% of the best, so off wins a tie - and off is also
    // the only configuration whose greedy output is exactly reproducible.
    //
    // ... which was true while a resident expert layer competed with the cache
    // for the same VRAM. Under the fixed design nothing else is bidding: the
    // experts all stream from the CPU, so "auto" means the cache takes the
    // whole remainder after the model, the KV cache and the MTP head. Every
    // rung this ladder can nominate is therefore SMALLER than what auto would
    // have claimed, and the ladder's own tie-break prefers the smallest rung
    // inside the noise band - so it systematically hands VRAM back for nothing.
    // Measured on gemma-4: auto grew the cache to 11277 MiB and served 96%+ of
    // expert requests from VRAM at 52% residency; the calibrated 2048 MiB
    // entry, chosen by this ladder, did not come close.
    // GGML_MOE_CALIBRATE_CACHE_SWEEP=1 restores the ladder.
    const bool sweep_cache = [] {
        const char * e = getenv("GGML_MOE_CALIBRATE_CACHE_SWEEP");
        return e && atoi(e) != 0;
    }();
    std::vector<int> cache_candidates_mb;
    if (sweep_cache) {
        cache_candidates_mb.push_back(0);
    }
    size_t remainder_bytes = 0;
    if (sweep_cache) {
        // The ladder is capped by the MEASURED remainder, not by what an idle
        // card reports.
        //
        // calib_free_vram_bytes is sampled before anything is loaded, so on a
        // 12 GiB card it offered rungs up to 8192 MiB - a size that cannot
        // coexist with the model, a 64k KV cache, the compute buffers and the
        // draft. Measured by hand at 64k: 1405 MiB is actually left at f16 KV,
        // 2393 MiB at q8_0. So most of the ladder was spending candidates on
        // rungs that could not exist, and the winner was decided among rungs
        // that were never fully allocated - the pool allocator halves its slot
        // count until cudaMalloc succeeds, so an 8192 MiB rung silently became
        // whatever fit (see the slab-shortfall accounting in moe-cache.cu).
        remainder_bytes = measure_remainder("the expert cache has to fit inside");
        const int cap_mb = (int) ((remainder_bytes > 0 ? remainder_bytes : calib_free_vram_bytes) >> 20);
        for (int mb = 512; mb <= cap_mb; mb *= 2) {
            cache_candidates_mb.push_back(mb);
        }
        LOG_INF("%s: expert-cache size candidates derived from %d MiB %s: %zu rung(s) up to %d MiB\n",
                __func__, cap_mb,
                remainder_bytes > 0 ? "measured free with the model loaded" : "free on an idle card (unmeasured)",
                cache_candidates_mb.size() - 1,
                cache_candidates_mb.size() > 1 ? cache_candidates_mb.back() : 0);
    }
    // Test every candidate rather than stopping at the first non-improving
    // step: the curve is not guaranteed monotonic below the knee - our own
    // sweep found a placement cliff at small cache sizes (too little VRAM
    // margin starves the fit search of concurrent slots, e.g. 4 -> 1, a
    // collapse unrelated to cache-hit-rate that a local stop rule would
    // mistake for "smaller is fine"). Picking the smallest point within 3%
    // of the *global* max, after seeing the whole curve, is robust to that
    // dip in a way a running best-so-far comparison is not.
    // One rung, measured the same way the sweep measures it - used by the sweep
    // below and again by the tie-break that confirms a near-tie.
    auto bench_cache_mb = [&](int mb) {
        return common_moe_bench_candidate_server(
                self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                best_threads, next_port(), ctx, n_predict, concurrency, mb, -1,
                active_ngl, active_min_rank);
    };
    std::vector<std::pair<int, double>> cache_results;
    for (size_t i = 0; i < cache_candidates_mb.size(); i++) {
        const int mb = cache_candidates_mb[i];
        double tps = common_moe_bench_candidate_server(
                self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                best_threads, next_port(), ctx, n_predict, concurrency, mb, -1, active_ngl, active_min_rank);
        // Retry only an infrastructure failure - a quality rejection is deterministic.
        if (tps < 0 && tps != COMMON_MOE_TPS_REJECTED && tps != COMMON_MOE_TPS_LAUNCH_FAILED) {
            tps = common_moe_bench_candidate_server(
                    self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                    best_threads, next_port(), ctx, n_predict, concurrency, mb, -1, active_ngl, active_min_rank);
        }
        LOG_INF("%s:   moe-cache=%dMiB -> %s\n", __func__, mb,
                tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
        common_moe_calibration_status_note("expert-cache size",
                mb == 0 ? std::string("off") : string_format("%d MiB", mb),
                tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
        common_moe_calibration_status_candidate_done();
        if (tps > 0) {
            cache_results.emplace_back(mb, tps);
        }
    }
    int    best_cache_mb  = -1; // -1 stays "auto" if the sweep found nothing usable
    double best_cache_tps = -1.0;
    for (const auto & r : cache_results) {
        best_cache_tps = std::max(best_cache_tps, r.second);
    }
    if (best_cache_tps > 0) {
        // The band has to be wider than the noise, or the noise chooses.
        //
        // This was 3%, which is far tighter than this hardware's measured
        // run-to-run spread: fitt=448 read 12.45 then 10.35 tok/s in one run,
        // and ncmoe=21 read 68.39 and 61.47 an hour apart. Measured on
        // Qwen3.8-Flash-Next, 2048 MiB scored 11.14 against 8192 MiB's 11.52
        // and missed the 3% band by 0.03 tok/s - 0.26% - so a cache four times
        // the size won on a difference an order of magnitude below the noise
        // floor, taking VRAM from everything else for nothing.
        //
        // 5%: wide enough to catch a rung the old 3% excluded on noise (2048
        // MiB at 11.14 against 11.52 is 3.3% back, inside 5% and outside 3%),
        // narrow enough that it is not simply preferring the smallest cache.
        // Widening alone would be the wrong fix either way - it would hand the
        // decision to a single cheap probe of a much smaller cache - so the
        // band only nominates, and the nomination has to survive a re-measure
        // against the best rung, the same confirm-before-believing the depth
        // search already does with its "measured again" pass. Override with
        // GGML_MOE_CALIBRATE_CACHE_BAND_PCT.
        static const double band = [] {
            const char * e = getenv("GGML_MOE_CALIBRATE_CACHE_BAND_PCT");
            const double v = e ? atof(e) : 5.0;
            return v > 0.0 && v < 50.0 ? v : 5.0;
        }();
        int    nominee_mb  = -1;
        double nominee_tps = -1.0;
        for (const auto & r : cache_results) {
            if (r.second >= best_cache_tps * (1.0 - band / 100.0)) {
                nominee_mb  = r.first;
                nominee_tps = r.second;
                break;
            }
        }
        int    best_mb_raw = -1;
        for (const auto & r : cache_results) {
            if (r.second >= best_cache_tps) {
                best_mb_raw = r.first;
                break;
            }
        }
        best_cache_mb = nominee_mb;
        if (nominee_mb > 0 && best_mb_raw > 0 && nominee_mb != best_mb_raw &&
            !common_moe_calibrate_budget_spent()) {
            LOG_INF("%s: expert-cache %d MiB (%.2f tok/s) is within %.0f%% of %d MiB (%.2f tok/s) and costs "
                    "%d MiB less - re-measuring both before taking the smaller one\n",
                    __func__, nominee_mb, nominee_tps, band, best_mb_raw, best_cache_tps,
                    best_mb_raw - nominee_mb);
            const double nom2  = bench_cache_mb(nominee_mb);
            const double best2 = common_moe_calibrate_budget_spent() ? -1.0 : bench_cache_mb(best_mb_raw);
            if (nom2 > 0 && best2 > 0) {
                const double nom_mean  = (nominee_tps + nom2) / 2.0;
                const double best_mean = (best_cache_tps + best2) / 2.0;
                LOG_INF("%s:   %d MiB mean %.2f tok/s, %d MiB mean %.2f tok/s\n",
                        __func__, nominee_mb, nom_mean, best_mb_raw, best_mean);
                // Still only has to be close, not better - the whole point is
                // that the smaller cache is worth real VRAM elsewhere.
                best_cache_mb  = nom_mean >= best_mean * (1.0 - band / 100.0) ? nominee_mb : best_mb_raw;
                best_cache_tps = best_cache_mb == nominee_mb ? nom_mean : best_mean;
            }
        }
    }
    // Say what was chosen. This stage measured every rung and then reported
    // none of them as the answer - the only stage that did not - so the log and
    // the decisions table both showed a ladder with no winner at the bottom,
    // and the chosen size could only be inferred by re-applying the knee rule
    // by hand.
    if (best_cache_mb > 0) {
        double chosen_tps = -1.0;
        for (const auto & r : cache_results) {
            if (r.first == best_cache_mb) {
                chosen_tps = r.second;
                break;
            }
        }
        LOG_INF("%s: expert-cache size: %d MiB at %.2f tok/s (smallest within 3%% of the best rung's "
                "%.2f tok/s - a bigger cache that is not measurably faster is VRAM taken from everything "
                "else)\n", __func__, best_cache_mb, chosen_tps, best_cache_tps);
        common_moe_calibration_status_note("expert-cache size",
                string_format("%d MiB", best_cache_mb),
                string_format("SELECTED - %.2f tok/s", chosen_tps), true, true);
    } else if (sweep_cache) {
        LOG_WRN("%s: expert-cache size: no rung measured usable - leaving it on auto\n", __func__);
        common_moe_calibration_status_note("expert-cache size", "auto",
                "no rung measured usable", false, true);
    } else {
        LOG_WRN("%s: expert cache on auto - nothing else bids for that VRAM once every layer's experts "
                "are on the CPU, so auto takes the whole remainder and any fixed rung would only be "
                "smaller (GGML_MOE_CALIBRATE_CACHE_SWEEP=1 to sweep sizes)\n", __func__);
        common_moe_calibration_status_note("expert-cache size", "auto",
                "DESIGN - the cache takes whatever VRAM is left; a fixed rung can only be smaller",
                true, true);
    }

    // Fit margin (-fitt). Searched LAST and deliberately re-searching
    // -ncmoe underneath it, because the two are not independent: the margin
    // is a floor on how little offload is allowed, so lowering it does not
    // just free VRAM, it unlocks placements the -ncmoe search above was
    // never permitted to evaluate. Measured on Ornith-1.5-35B-Q4_K_M /
    // RTX 3060 12 GB / -c 4096, requesting -ncmoe 8:
    //
    //     fitt=1024 (default) -> forced to ncmoe 27, 47.32 tok/s
    //     fitt=640            -> forced to ncmoe 24, 50.37 tok/s
    //     fitt=448            -> forced to ncmoe 23, 51.61 tok/s
    //     fitt=320            -> forced to ncmoe 22, 52.85 tok/s
    //
    // +11.7% end to end, entirely from VRAM the default margin was holding
    // in reserve and CPU work that reservation forced. Descending (largest
    // margin first, i.e. safest first) and stopping at the first candidate
    // that fails to launch or produces no improvement: below some point the
    // margin stops covering the post-probe allocations it exists for (real
    // weight loading, lazy CUDA graph capture) and the server either fails
    // outright or collapses its context - a documented, reproduced failure,
    // so this walks toward the edge and stops rather than bisecting across
    // it. Each candidate asks for the minimum offload (safe_n) and lets the
    // fit logic raise it to whatever that margin actually permits, which is
    // the quantity being measured.
    int    best_fit_mb  = -1;
    // ... and that is exactly why it stops being searched under the fixed
    // design. Every number above is a PLACEMENT result: the margin paid because
    // tightening it let more expert layers stay on the GPU. With -ncmoe pinned
    // to every layer there is no placement left to unlock - the margin's only
    // remaining effect is to shrink the headroom the expert cache grows into,
    // while moving the server closer to the launch failures this walk exists to
    // stay clear of. GGML_MOE_CALIBRATE_SEARCH_FIT=1 restores the search.
    const bool search_fit = [] {
        const char * e = getenv("GGML_MOE_CALIBRATE_SEARCH_FIT");
        return e && atoi(e) != 0;
    }();
    // Baseline MUST be the best throughput already measured at the DEFAULT
    // margin under the SAME cache size and thread count these candidates
    // run with - i.e. the cache-knee winner, not best_threads_tps. Getting
    // this wrong is not a subtle accounting slip: best_threads_tps was
    // measured during the thread sweep at a different (auto) cache size, so
    // comparing margin candidates against it compares across two different
    // configurations at once. Measured consequence on gemma-4-26B-A4B,
    // concurrency 4: the cache knee found 71.86 tok/s at 4096 MiB with the
    // default 1024 MiB margin, every tightened margin scored ~50, and this
    // search still declared fitt=448 the winner because it was only being
    // compared against the thread stage's 43.57. That entry would have
    // shipped a ~29% REGRESSION as a calibrated optimum. Seeding with the
    // cache-stage best means -1 ("keep the default") correctly survives
    // whenever tightening does not actually help, which for this model it
    // does not.
    double best_fit_tps = best_cache_tps > 0 ? best_cache_tps : best_threads_tps;
    std::map<int, double> fit_trace;   // margin -> tok/s, for the re-measure of the top two
    {
        const int default_fit_mb = (int) (params.fit_params_target[0] / (1024 * 1024));
        static const int fit_candidates_mb[] = {640, 448, 320, 256};
        if (!search_fit) {
            LOG_WRN("%s: fit margin left at the default %d MiB - with no expert layer resident there is "
                    "no placement for a tighter margin to unlock, only headroom taken from the expert "
                    "cache (GGML_MOE_CALIBRATE_SEARCH_FIT=1 to search it)\n", __func__, default_fit_mb);
            common_moe_calibration_status_note("fit margin",
                    string_format("%d MiB (default)", default_fit_mb),
                    "DESIGN - the margin only bought placements, and placement is fixed", true, true);
        } else {
        common_moe_calibration_status_set("searching fit margin (-fitt)");
        LOG_INF("%s: searching fit margin (-fitt) below the default of %d MiB at ncmoe>=%u ...\n",
                __func__, default_fit_mb, safe_n);
        for (size_t i = 0; i < sizeof(fit_candidates_mb) / sizeof(fit_candidates_mb[0]); i++) {
            const int mb = fit_candidates_mb[i];
            if (mb >= default_fit_mb) {
                continue; // only ever tighten below the default, never loosen past it
            }
            const double tps = common_moe_bench_candidate_server(
                    self_exe, path_model, mtp_path_for_threads, safe_n, n_max_for_threads,
                    best_threads, next_port(), ctx, n_predict, concurrency, best_cache_mb, mb, active_ngl, active_min_rank);
            LOG_INF("%s:   fitt=%dMiB -> %s\n", __func__, mb,
                    tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
            common_moe_calibration_status_note("fit margin", string_format("%d MiB", mb),
                    tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
            common_moe_calibration_status_candidate_done();
            if (tps <= 0) {
                LOG_INF("%s:   fitt=%dMiB did not come up - stopping here, this is the edge the "
                        "margin exists to stay clear of\n", __func__, mb);
                break;
            }
            if (tps > 0) {
                fit_trace[mb] = tps;
            }
            if (tps > best_fit_tps) {
                best_fit_tps = tps;
                best_fit_mb  = mb;
            }
        }
        // The winner was tracked and saved but never announced, so the log showed
        // four numbers and left the choice to be inferred - and the decisions table
        // had no chosen row for this stage at all.
        // Each margin above is one short probe, and the readings are not monotonic -
        // measured on gemma-4: 640 MiB 76.55, 448 MiB 69.33, 320 MiB 80.86, 256 MiB
        // 80.70. A 0.2% gap between the top two cannot be called on one sample
        // each, and the 448 dip says the noise is larger than that. Measure the top
        // two again and decide on the mean of both samples, as the depth search does.
        if (best_fit_mb > 0 && !fit_trace.empty()) {
            int runner_up = -1;
            for (const auto & kv : fit_trace) {
                if (kv.first != best_fit_mb && kv.second > 0 &&
                    (runner_up < 0 || kv.second > fit_trace.at(runner_up))) {
                    runner_up = kv.first;
                }
            }
            for (const int mb : { best_fit_mb, runner_up }) {
                if (mb <= 0 || common_moe_calibrate_budget_spent()) {
                    continue;
                }
                const double again = common_moe_bench_candidate_server(
                        self_exe, path_model, sub_mtp_path, safe_n, sub_n_max, n_threads_default,
                        next_port(), ctx, n_predict, concurrency, best_cache_mb, mb,
                        active_ngl, active_min_rank, nullptr, 1234, active_quality_sigma);
                common_moe_calibration_status_candidate_done();
                if (again > 0) {
                    LOG_INF("%s:   fitt=%dMiB measured again -> %.2f tok/s (first %.2f, mean %.2f)\n",
                            __func__, mb, again, fit_trace.at(mb), (again + fit_trace.at(mb)) / 2.0);
                    fit_trace[mb] = (again + fit_trace.at(mb)) / 2.0;
                }
            }
            if (runner_up > 0 && fit_trace.at(runner_up) > fit_trace.at(best_fit_mb)) {
                best_fit_mb = runner_up;
            }
            best_fit_tps = fit_trace.at(best_fit_mb);
        }
        }
        entry.fit_target_mb = best_fit_mb;
        checkpoint("fit margin");
        if (best_fit_mb > 0) {
            LOG_INF("%s: fit margin: %d MiB at %.2f tok/s\n", __func__, best_fit_mb, best_fit_tps);
            common_moe_calibration_status_note("fit margin", string_format("%d MiB", best_fit_mb),
                    string_format("SELECTED - %.2f tok/s (mean of two samples where confirmed)", best_fit_tps),
                    true, /* chosen */ true);
        }
    }

    time_t now = time(nullptr);
    char timebuf[32];
    strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", localtime(&now));

    // Admission threshold: how many times an expert must be demanded before it
    // earns a slot. admit_after=2 / readmit_after=8 (the fallback when the pool
    // is full) have been constants since 2026-08-13 and were never revisited.
    // Ladder is small on purpose - each rung is a full candidate spawn, and this
    // stage runs after everything else has already spent most of the budget.
    // Compared against the same-regime incumbent (cheap_incumbent_tps), not
    // best_tps, for the reason documented on that variable's declaration: a
    // 32-token candidate can never beat a 128-token confirmed number.
    int    best_admit_after     = -1;
    double best_admit_after_tps = cheap_incumbent_tps > 0.0 ? cheap_incumbent_tps : best_tps;
    // Three candidates that have never separated by more than the noise.
    //
    // Measured on Qwen3.8-Flash-Next at 64k: 1 -> 2.42, 2 -> 2.46, 4 -> 2.44
    // tok/s. A 1.6% spread on a machine whose run-to-run spread is 12-20%, for
    // three of the most expensive candidates in that run at 3.3 minutes each -
    // ten minutes sampling noise and then fixing it for every stage after.
    // Gemma-4 read 66.55 / 69.24 / 49.14, which looks decisive until you note
    // the incumbent was 70.62 and nothing was selected either way.
    //
    // The shipped constant stays. GGML_MOE_CALIBRATE_SEARCH_ADMIT=1 searches it.
    const bool search_admit = [] {
        const char * e = getenv("GGML_MOE_CALIBRATE_SEARCH_ADMIT");
        return e && atoi(e) != 0;
    }();
    if (!search_admit) {
        LOG_WRN("%s: admission threshold left at its shipped constant - the ladder has never separated by "
                "more than this machine's run-to-run spread (1.6%% across the three rungs on qwen4exp at "
                "64k) (GGML_MOE_CALIBRATE_SEARCH_ADMIT=1 to search it)\n", __func__);
        common_moe_calibration_status_note("admission threshold", "default",
                "DEFAULT - three rungs that have never separated by more than the noise", true, true);
    } else if (!common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: searching admission threshold (admit_after) ...\n", __func__);
        common_moe_calibration_status_set("searching admission threshold");
        for (const int candidate : {1, 2, 4}) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            char env_buf[64];
            snprintf(env_buf, sizeof(env_buf), "GGML_CUDA_MOE_CACHE_ADMIT_AFTER=%d ", candidate);
            const double tps = common_moe_bench_candidate_server(
                    self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                    best_threads, next_port(), ctx, n_predict, concurrency, best_cache_mb, -1,
                    active_ngl, active_min_rank, nullptr, 1234,
                    std::numeric_limits<double>::quiet_NaN(), false, false, env_buf);
            LOG_INF("%s:   admit_after=%d -> %s\n", __func__, candidate,
                    tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
            common_moe_calibration_status_note("admission threshold",
                    string_format("admit_after=%d", candidate),
                    tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
            common_moe_calibration_status_candidate_done();
            if (tps > best_admit_after_tps) {
                best_admit_after_tps = tps;
                best_admit_after     = candidate;
            }
        }
        if (best_admit_after >= 0) {
            LOG_INF("%s: admission threshold: admit_after=%d at %.2f tok/s\n",
                    __func__, best_admit_after, best_admit_after_tps);
            common_moe_calibration_status_note("admission threshold",
                    string_format("admit_after=%d", best_admit_after),
                    string_format("SELECTED - %.2f tok/s", best_admit_after_tps), true, true);
        }
    }

    // Readmission threshold: the SAME demand->count gate, but for the case
    // (almost always, once the pool fills) where admission must evict a
    // resident rather than take a free slot - see moe_cache_plan_impl's own
    // admit_after computation, which takes the max of this and admit_after
    // once free_slots is empty. A direct test of the opposite extreme
    // (THROTTLE=1, i.e. this gate removed entirely) showed why the shipped
    // constant of 8 cannot be lowered by guessing: hit rate collapsed from
    // 22% to 1.3% and throughput fell below every other configuration
    // measured, because an eviction is not free the way filling a free
    // slot is, and a one-off demand should not be allowed to keep
    // displacing an already-proven resident. But the SAME direct test
    // found the other direction wins: THROTTLE=32 measured 26-30% hit rate
    // against the default's 22%, large enough to clear this machine's
    // 12-20% run-to-run spread - unlike admit_after's own ladder, which
    // never has. So unlike admit_after, this stage is worth running by
    // default; GGML_MOE_CALIBRATE_SEARCH_READMIT=0 skips it.
    int    best_readmit_after     = -1;
    double best_readmit_after_tps = cheap_incumbent_tps > 0.0 ? cheap_incumbent_tps : best_tps;
    const bool search_readmit = [] {
        const char * e = getenv("GGML_MOE_CALIBRATE_SEARCH_READMIT");
        return !e || atoi(e) != 0;
    }();
    if (!search_readmit) {
        LOG_WRN("%s: readmission threshold left at its shipped constant "
                "(GGML_MOE_CALIBRATE_SEARCH_READMIT=0 was set)\n", __func__);
        common_moe_calibration_status_note("readmission threshold", "default",
                "DEFAULT - search explicitly disabled", true, true);
    } else if (!common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: searching readmission threshold (readmit_after) ...\n", __func__);
        common_moe_calibration_status_set("searching readmission threshold");
        for (const int candidate : {8, 16, 32, 64}) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            char env_buf[64];
            snprintf(env_buf, sizeof(env_buf), "GGML_CUDA_MOE_CACHE_THROTTLE=%d ", candidate);
            const double tps = common_moe_bench_candidate_server(
                    self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                    best_threads, next_port(), ctx, n_predict, concurrency, best_cache_mb, -1,
                    active_ngl, active_min_rank, nullptr, 1234,
                    std::numeric_limits<double>::quiet_NaN(), false, false, env_buf);
            LOG_INF("%s:   readmit_after=%d -> %s\n", __func__, candidate,
                    tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
            common_moe_calibration_status_note("readmission threshold",
                    string_format("readmit_after=%d", candidate),
                    tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
            common_moe_calibration_status_candidate_done();
            if (tps > best_readmit_after_tps) {
                best_readmit_after_tps = tps;
                best_readmit_after     = candidate;
            }
        }
        if (best_readmit_after >= 0) {
            LOG_INF("%s: readmission threshold: readmit_after=%d at %.2f tok/s\n",
                    __func__, best_readmit_after, best_readmit_after_tps);
            common_moe_calibration_status_note("readmission threshold",
                    string_format("readmit_after=%d", best_readmit_after),
                    string_format("SELECTED - %.2f tok/s", best_readmit_after_tps), true, true);
        }
    }

    // One measurement on the candidate held open for live sweeps, applying the
    // given policy knobs first, and one that launches and holds a candidate open.
    // Declared here because the ring is the first sweep that can use them; the
    // speculative arguments are left at their defaults on purpose, since the
    // stages that choose them run later and every live sweep below is a policy
    // knob whose value is independent of them.
    auto measure_live = [&](const std::string & body, int predict) -> double {
        if (g_moe_live_port <= 0) {
            return -1.0;
        }
        return common_moe_bench_candidate_server(
                self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                best_threads, g_moe_live_port, ctx, predict, concurrency, best_cache_mb, -1,
                active_ngl, active_min_rank, nullptr, 1234, active_quality_sigma,
                false, false, std::string(), -1, std::string(), -1.0,
                g_moe_live_port, body, /* keep_alive */ true);
    };
    auto open_live = [&](int predict) -> double {
        return common_moe_bench_candidate_server(
                self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                best_threads, next_port(), ctx, predict, concurrency, best_cache_mb, -1,
                active_ngl, active_min_rank, nullptr, 1234, active_quality_sigma,
                false, false, std::string(), -1, std::string(), -1.0,
                -1, std::string(), /* keep_alive */ true);
    };

    // How stand-ins are CHOSEN, once the floor has decided how often. Every
    // existing picker ranks candidates by how wanted they are in general - heat,
    // or co-activation - which is not the question substitution poses. Atlas
    // similarity ranks by resemblance to the missing expert instead, and its
    // strict form declines when nothing resident is near enough, paying exact
    // compute rather than serving something unrelated.
    //
    // Measured only when substitution is actually in use: with the floor at "never
    // substitute" there is nothing to choose. Judged on throughput AND the answer
    // bar, because the whole claim is that a better-chosen stand-in costs less
    // correctness - a claim that has to be shown, not asserted.
    int best_sub_atlas = -1;
    if (best_min_rank >= 0 && best_min_rank < 10 && full_sweep && !common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: measuring how stand-ins are chosen (heat vs atlas similarity) ...\n", __func__);
        common_moe_calibration_status_set("measuring stand-in selection");
        common_moe_stage_begin("stand-in selection", 3);
        double best_sub_tps = -1.0;
        // Four methods, not three. Mode 3 is the per-token-rank picker, which
        // has measured evidence it is the best of them and had never been
        // measured here: on the router's own score for the stand-in chosen it
        // scored 0.03856 against pairwise co-activation's 0.02130 (+81%) and
        // the full-probs oracle's 0.03790. It was reachable only through
        // GGML_CUDA_MOE_CACHE_SUBSTITUTE_STRICT_RANK, which nothing set - the
        // same way the prerouter sat behind TRAIN_PREDICTOR=0 for its whole
        // life. It declines more often than the co-activation scan, and a
        // declined stand-in costs time while a wrong one costs quality, so
        // whether the trade pays is exactly a question for measurement.
        // Mode 4 is the fused picker - all three signals at once rather than
        // the best of them. The other four stay as measurable points so the
        // fusion has to earn its place against each individual signal rather
        // than being assumed better for being more elaborate.
        for (const int mode : { 0, 1, 2, 3, 4 }) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            const char * label = mode == 0 ? "by heat"
                               : mode == 1 ? "atlas, heat fallback"
                               : mode == 2 ? "atlas only"
                               : mode == 3 ? "by this token's router rank"
                                           : "fused: rank, then atlas + co-activation";
            // Mode 3 turns the atlas picker off and the rank picker on; the
            // others leave the rank picker off, so exactly one method is
            // measured per candidate.
            const std::string body = mode == 3
                ? std::string("{\"GGML_CUDA_MOE_CACHE_SUBSTITUTE_ATLAS\": \"0\", "
                              "\"GGML_CUDA_MOE_CACHE_SUBSTITUTE_STRICT_RANK\": \"1\"}")
                : string_format("{\"GGML_CUDA_MOE_CACHE_SUBSTITUTE_ATLAS\": \"%d\", "
                                "\"GGML_CUDA_MOE_CACHE_SUBSTITUTE_STRICT_RANK\": \"0\"}", mode);
            // mode 4 carries the fusion weights; the picker reads them live.
            double tps = (mode == 0) ? open_live(n_predict) : measure_live(body, n_predict);
            if (mode == 0 && g_moe_live_port > 0) {
                // the launch itself ran at the default; re-measure through the knob
                tps = measure_live(body, n_predict);
            }
            common_moe_calibration_status_candidate_done();
            LOG_INF("%s:   stand-ins %s -> %s\n", __func__, label,
                    tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
            common_moe_calibration_status_note("stand-in selection", label,
                    tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
            if (tps > best_sub_tps) {
                best_sub_tps   = tps;
                best_sub_atlas = mode;
            }
        }
        common_moe_live_stop();
        common_moe_stage_end();
        if (best_sub_atlas > 0 && !common_moe_calibrate_budget_spent()) {
            common_moe_calib_set_env(string_format("GGML_CUDA_MOE_CACHE_SUBSTITUTE_ATLAS=%d", best_sub_atlas));
            const double checked = common_moe_bench_candidate_server(
                    self_exe, path_model, sub_mtp_path, best_n, sub_n_max, n_threads_default,
                    next_port(), ctx, n_predict * 4, concurrency, best_cache_mb, -1,
                    active_ngl, active_min_rank, nullptr, 1234, active_quality_sigma,
                    /* verify_answers */ true, /* with_reasoning */ true);
            common_moe_calibration_status_candidate_done();
            if (checked <= 0) {
                LOG_WRN("%s:   atlas stand-ins did not pass the answer check - keeping heat selection\n", __func__);
                common_moe_calibration_status_note("stand-in selection", "atlas",
                        "rejected by the answer check", false);
                best_sub_atlas = 0;
            }
            common_moe_calib_set_env(std::string());
        }
        // If the fusion won, measure the weights it won with rather than
        // shipping the 1.0/1.0 they were written with. Two knobs added with
        // guessed defaults and no stage measuring them is the same debt this
        // file has spent the day paying off elsewhere - the prerouter at
        // TRAIN_PREDICTOR=0, the substitution floor behind a fit guard,
        // pick_rank behind an env var nothing set.
        //
        // The ratio is what matters, not the magnitude: scores are only ever
        // compared against other candidates', so (1,1) and (2,2) rank
        // identically. One axis swept with the other pinned covers everything
        // that changes the ordering, and a weight at 0 reduces the fusion to
        // the single-signal picker the modes above already measured - which
        // makes those points the cross-check.
        if (best_sub_atlas == 4 && !common_moe_calibrate_budget_spent()) {
            double best_w_tps   = -1.0;
            double best_w_atlas = 1.0, best_w_coact = 1.0;
            for (const auto & w : { std::make_pair(1.0, 1.0), std::make_pair(1.0, 0.25),
                                    std::make_pair(1.0, 4.0), std::make_pair(0.25, 1.0),
                                    std::make_pair(4.0, 1.0) }) {
                if (common_moe_calibrate_budget_spent()) {
                    break;
                }
                common_moe_calib_set_env(string_format(
                        "GGML_CUDA_MOE_CACHE_SUBSTITUTE_ATLAS=4 "
                        "GGML_CUDA_MOE_CACHE_SUB_W_ATLAS=%.2f GGML_CUDA_MOE_CACHE_SUB_W_COACT=%.2f",
                        w.first, w.second));
                const double tps = common_moe_bench_candidate_server(
                        self_exe, path_model, sub_mtp_path, best_n, sub_n_max, best_threads,
                        next_port(), ctx, n_predict, concurrency, best_cache_mb, -1,
                        active_ngl, active_min_rank, nullptr, 1234, active_quality_sigma);
                common_moe_calibration_status_candidate_done();
                LOG_INF("%s:   fusion weights atlas %.2f / coact %.2f -> %s\n", __func__,
                        w.first, w.second,
                        tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
                common_moe_calibration_status_note("stand-in fusion weights",
                        string_format("atlas %.2f / coact %.2f", w.first, w.second),
                        tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
                if (tps > best_w_tps) {
                    best_w_tps   = tps;
                    best_w_atlas = w.first;
                    best_w_coact = w.second;
                }
            }
            common_moe_calib_set_env(std::string());
            if (best_w_tps > 0) {
                entry.sub_w_atlas = best_w_atlas;
                entry.sub_w_coact = best_w_coact;
                LOG_INF("%s: stand-in fusion weights: atlas %.2f / coact %.2f at %.2f tok/s\n",
                        __func__, best_w_atlas, best_w_coact, best_w_tps);
                common_moe_calibration_status_note("stand-in fusion weights",
                        string_format("atlas %.2f / coact %.2f", best_w_atlas, best_w_coact),
                        string_format("SELECTED - %.2f tok/s", best_w_tps), true, true);
            }
        }

        entry.substitute_atlas = best_sub_atlas;
        checkpoint("stand-in selection");
        if (best_sub_atlas >= 0) {
            LOG_INF("%s: stand-in selection: %s\n", __func__,
                    best_sub_atlas == 0 ? "by heat" : (best_sub_atlas == 1 ? "atlas with heat fallback" : "atlas only"));
        }
    }

    // Atlas prewarm on a restored prefix. The atlas is the only thing that can say
    // anything BEFORE a request has routed anything: the lookahead predicts from
    // the current hidden state and has nothing to work with at turn zero, and the
    // ring can only hold what has already been predicted. On a restored
    // prompt-cache prefix the saved topic hint names the neighbourhood the next
    // tokens will want, so the cache can be warmed toward it instead of
    // rediscovering it a miss at a time.
    //
    // Wired end to end since the prompt-cache work and shipped OFF, never measured
    // - the same state neuron subsetting was in. Measured where it can matter: a
    // second turn on a prefix the first turn cached, which is what the probe loop
    // does naturally (it repeats its prompts).
    int best_prewarm_k = -1;
    if (full_sweep && !common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: measuring atlas prewarm on a restored prefix (it ships off, unmeasured) ...\n", __func__);
        common_moe_calibration_status_set("measuring atlas prewarm");
        common_moe_stage_begin("atlas prewarm", 3);
        double best_prewarm_tps = -1.0;
        // LLAMA_PROMPT_CACHE_MOE_PREWARM is latched at first use (it gates a
        // startup-time decision path), so each value needs its own candidate.
        for (const int k : { 0, 4, 16 }) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            const std::string env = k == 0
                ? std::string("LLAMA_PROMPT_CACHE_MOE_PREWARM=0")
                : string_format("LLAMA_PROMPT_CACHE_MOE_PREWARM=1 LLAMA_PROMPT_CACHE_MOE_PREWARM_K=%d", k);
            common_moe_calib_set_env(env);
            const double tps = common_moe_bench_candidate_server(
                    self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                    best_threads, next_port(), ctx, n_predict, concurrency, best_cache_mb, -1,
                    active_ngl, active_min_rank, nullptr, 1234, active_quality_sigma);
            common_moe_calibration_status_candidate_done();
            LOG_INF("%s:   atlas prewarm %s -> %s\n", __func__,
                    k == 0 ? "off" : string_format("top_k %d", k).c_str(),
                    tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
            common_moe_calibration_status_note("atlas prewarm",
                    k == 0 ? std::string("off") : string_format("top_k %d", k),
                    tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
            if (tps > best_prewarm_tps) {
                best_prewarm_tps = tps;
                best_prewarm_k   = k;
            }
        }
        common_moe_calib_set_env(std::string());
        common_moe_stage_end();
        entry.atlas_prewarm_k = best_prewarm_k;
        checkpoint("atlas prewarm");
        if (best_prewarm_k >= 0) {
            LOG_INF("%s: atlas prewarm: %s at %.2f tok/s\n", __func__,
                    best_prewarm_k == 0 ? "off" : string_format("top_k %d", best_prewarm_k).c_str(),
                    best_prewarm_tps);
            common_moe_calibration_status_note("atlas prewarm",
                    best_prewarm_k == 0 ? std::string("off") : string_format("top_k %d", best_prewarm_k),
                    string_format("SELECTED - %.2f tok/s", best_prewarm_tps), true, true);
        }
    }

    // Train the prerouter on calibration's own traffic, once, before anything
    // measures it.
    //
    // The traffic is already the right shape: the probe prompts span
    // photosynthesis, binary search, mystery fiction, the water cycle,
    // Newton's laws, car engines, inflation, vaccines and a step-by-step rate
    // problem - eight or nine distinct topics, which is what a router-
    // prediction corpus needs. It needs varied hidden states, not correct
    // answers, so the existing prompts serve without being written for this.
    //
    // Deliberately ONE pass, with every later candidate frozen against what it
    // produced. A predictor that kept learning across candidates would make
    // the whole run order-biased: the same candidate measured late would beat
    // itself measured early, on accumulated training alone, and that bias
    // would land on stages that have nothing to do with the predictor. Every
    // stage before this one runs with the predictor off entirely, for the same
    // reason - uniform, not absent-then-present.
    std::string predictor_state_file;
    if (full_sweep && !common_moe_calibrate_budget_spent()) {
        const std::string cache_dir = fs_get_cache_directory();
        const std::string base = path_model ? std::string(path_model) : std::string("model");
        const size_t slash = base.find_last_of("/\\");
        predictor_state_file = cache_dir + "predictor-" +
            (slash == std::string::npos ? base : base.substr(slash + 1)) + ".bin";

        LOG_INF("%s: training the prerouter on this run's own traffic (one pass, then frozen) ...\n", __func__);
        common_moe_calibration_status_set("training the prerouter");
        // What the predictor is asked to answer, and what it is allowed to
        // see, are not one question - so each shape trains its own weights and
        // is judged on them. Each candidate writes its own state file; the
        // winner's is what every later stage freezes against.
        //
        //   this layer    the original target. Asks which experts THIS layer
        //                 wants, of an input the real router is about to
        //                 answer exactly - so it can only ever rank experts
        //                 outside the current selection.
        //   next layer    the actual prerouter question, and the only one
        //                 prefetch can spend.
        //   + prev picks  a hashed block of what the previous layer chose.
        //                 The hidden state (59.3% at depth 1) and the expert-id
        //                 successor table (33.8% raw) were only ever measured
        //                 SEPARATELY; this is the first time one model sees
        //                 both.
        //   - ngram       drop the architecture block the model supplies (the
        //                 PLE n-gram identity on qwen4exp), to measure what it
        //                 is worth rather than assume it.
        struct train_shape { const char * label; int next_layer; int aux_cap; };
        // Two shapes, not three. The n-gram question is settled: removing the
        // architecture's PLE block cost 70.00 -> 49.88 tok/s on gemma-4 and
        // 12.37 -> 9.51 on qwen3.8-flash-next - 29% and 23%, well outside this
        // machine's spread and in the same direction both times. Re-measuring a
        // settled 25% every run spends a candidate re-deriving a known answer.
        // Same-layer against next-layer stays, because that is a question about
        // what the model is asked to predict rather than about noise.
        const train_shape shapes[] = {
            { "this layer's picks",        0, 32 },
            { "next layer's picks",        1, 32 },
        };
        common_moe_stage_begin("prerouter training", (int)(sizeof(shapes)/sizeof(shapes[0])));
        double train_tps = -1.0;
        int best_next_layer = -1, best_aux_cap = 32;
        std::string best_state_file;
        for (size_t si = 0; si < sizeof(shapes)/sizeof(shapes[0]); si++) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            const auto & sh = shapes[si];
            const std::string state_i = predictor_state_file + "." + std::to_string(si);
            // The ring, and permission to use it, on every prerouter candidate.
            // Without them the predictor cannot act at all: its admissions go
            // through the suggestion-only path, and that path's own comment
            // records that pool.free_slots empties permanently within the
            // first few tokens of any real run. Measured with the ring at its
            // default of 0, every candidate here would score the cost of
            // training with none of the benefit - and the stage would report
            // "the predictor does not help" about a predictor that was never
            // allowed to place anything.
            common_moe_calib_set_env(string_format(
                    "%sGGML_CUDA_MOE_CACHE_TRAIN_PREDICTOR=1 GGML_CUDA_MOE_CACHE_TRAIN_STATE_FILE=%s "
                    "GGML_CUDA_MOE_CACHE_TRAIN_NEXT_LAYER=%d GGML_CUDA_MOE_CACHE_AUX_FEATURE_CAP=%d",
                    "GGML_CUDA_MOE_CACHE_RING_PCT=10 GGML_CUDA_MOE_CACHE_ATLAS_ADMIT_RING=1 ",
                    state_i.c_str(), sh.next_layer, sh.aux_cap));
            const double tps = common_moe_bench_candidate_server(
                    self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                    best_threads, next_port(), ctx, n_predict, concurrency, best_cache_mb, -1,
                    active_ngl, active_min_rank, nullptr, 1234, active_quality_sigma);
            common_moe_calibration_status_candidate_done();
            LOG_INF("%s:   prerouter input %s -> %s\n", __func__, sh.label,
                    tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
            common_moe_calibration_status_note("prerouter input", sh.label,
                    tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
            if (tps > train_tps) {
                train_tps         = tps;
                best_next_layer   = sh.next_layer;
                best_aux_cap      = sh.aux_cap;
                best_state_file   = state_i;
            }
        }
        entry.predictor_next_layer   = best_next_layer;
        checkpoint("prerouter input");
        if (!best_state_file.empty()) {
            predictor_state_file = best_state_file;
        }
        common_moe_calib_set_env(std::string());
        common_moe_stage_end();
        LOG_INF("%s: prerouter training pass %s (state: %s)\n", __func__,
                train_tps > 0 ? "completed" : "did not complete - later stages measure an untrained predictor",
                predictor_state_file.c_str());
        common_moe_calibration_status_note("prerouter training", "one pass over the probe set",
                train_tps > 0 ? string_format("trained at %.2f tok/s", train_tps)
                              : std::string("did not complete"), train_tps > 0);
        // From here on every candidate loads those weights and none of them
        // writes back - see moe_cache_train_frozen.
        if (train_tps > 0) {
            // The shape travels with the weights. A frozen candidate that built
            // a different input vector than the weights were fitted against
            // would score against the wrong layout - the dimension check would
            // reject it and silently serve an untrained predictor, which reads
            // as "the predictor does not help" rather than as a bug.
            common_moe_calib_set_base_env(string_format(
                    "GGML_CUDA_MOE_CACHE_RING_PCT=10 GGML_CUDA_MOE_CACHE_ATLAS_ADMIT_RING=1 "
                    "GGML_CUDA_MOE_CACHE_TRAIN_STATE_FILE=%s GGML_CUDA_MOE_CACHE_TRAIN_FREEZE=1 "
                    "GGML_CUDA_MOE_CACHE_TRAIN_NEXT_LAYER=%d GGML_CUDA_MOE_CACHE_AUX_FEATURE_CAP=%d",
                    predictor_state_file.c_str(), best_next_layer, best_aux_cap));
        }
    }

    // The budget partition, searched as a vector rather than as scalars.
    //
    // Every consumer here bids for one number, and each was deciding its own
    // share at a different moment by a different rule - the reduced pool at a
    // flat 1/8, the draft at a percentage, the primary taking whatever was
    // left. Measured consequence on Qwen3.8-Flash-Next at 64k: an 8192 MiB
    // cache ran at 9.90 tok/s against 2048 MiB's 11.40, four times smaller and
    // 15% faster, because the extra bytes came out of consumers that needed
    // them more. Optimising each share while holding the others fixed is
    // coordinate descent on a shared constraint - it finds a local optimum and
    // the stage order decides which one.
    //
    // So the shares are measured together. The primary is never a declared
    // share: it is the remainder, with a floor no combination may cross,
    // because a cache whose primary pool cannot exist is worse than no cache.
    int best_reduced_share = -1;
    if (full_sweep && !common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: measuring the budget partition (reduced pool's share of the expert cache) ...\n", __func__);
        common_moe_calibration_status_set("measuring the budget partition");
        // The arbiter enumerates; this stage only measures. Which splits are
        // legal follows from the primary pool's floor, which the arbiter owns -
        // a list written here would be re-deciding that constraint from outside
        // and would drift from it the moment the floor changed.
        int part_reduced[16];
        int part_draft[16];
        int n_parts = 0;
        if (ggml_moe_cache.partition_candidates) {
            n_parts = ggml_moe_cache.partition_candidates(part_reduced, part_draft, 16);
        }
        if (n_parts <= 0) {
            // No cache registered (CPU-only build, or the cache is off): there
            // is no partition to search.
            LOG_INF("%s: no expert cache registered - nothing to partition\n", __func__);
        }
        common_moe_stage_begin("budget partition", std::max(1, n_parts));
        double best_part_tps = -1.0;
        int    best_draft_share_from_part = -1;
        for (int pi = 0; pi < n_parts; pi++) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            const int pct   = part_reduced[pi];
            const int dpct  = part_draft[pi];
            // A draft share is only meaningful when a draft is attached.
            if (dpct > 0 && !mtp_configured) {
                continue;
            }
            std::string part_env = string_format(
                    "GGML_CUDA_MOE_CACHE_REDUCED_SHARE_PCT=%d GGML_CUDA_MOE_CACHE_NEURON_REDUCE=%d",
                    pct, pct > 0 ? 1 : 0);
            if (dpct >= 0) {
                part_env += string_format(" GGML_CUDA_MOE_CACHE_DRAFT_SHARE_PCT=%d", dpct);
            }
            common_moe_calib_set_env(part_env);
            const double tps = common_moe_bench_candidate_server(
                    self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                    best_threads, next_port(), ctx, n_predict, concurrency, best_cache_mb, -1,
                    active_ngl, active_min_rank, nullptr, 1234, active_quality_sigma);
            common_moe_calibration_status_candidate_done();
            const std::string label = (pct == 0 && dpct < 0)
                ? std::string("all to the primary pool")
                : string_format("reduced %d%%%s", pct,
                                dpct >= 0 ? string_format(", draft %d%%", dpct).c_str() : "");
            LOG_INF("%s:   %s -> %s\n", __func__, label.c_str(),
                    tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
            common_moe_calibration_status_note("budget partition", label,
                    tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
            if (tps > best_part_tps) {
                best_part_tps              = tps;
                best_reduced_share         = pct;
                best_draft_share_from_part = dpct;
            }
        }
        if (best_draft_share_from_part >= 0) {
            best_share_pct = best_draft_share_from_part;
            entry.draft_share_pct = best_share_pct;
        }
        common_moe_calib_set_env(std::string());
        common_moe_stage_end();
        entry.reduced_share_pct = best_reduced_share;
        checkpoint("budget partition");
        if (best_reduced_share >= 0) {
            LOG_INF("%s: budget partition: reduced pool %d%% at %.2f tok/s\n",
                    __func__, best_reduced_share, best_part_tps);
            common_moe_calibration_status_note("budget partition",
                    best_reduced_share == 0 ? std::string("all to the primary pool")
                                            : string_format("reduced %d%%", best_reduced_share),
                    string_format("SELECTED - %.2f tok/s", best_part_tps), true, true);
        }
    }

    // Rank-weighted admission: how much an un-substitutable miss counts.
    //
    // With top-10 routing and a substitution floor of 4, ranks 0-3 are 40% of
    // activations and can never be served by a stand-in - a miss on one costs a
    // CPU matmul, every time. Ranks 4-9 miss for free. Admission could not tell
    // the two apart: it counted demand and nothing else, so scarce VRAM was
    // handed out without regard for whether a miss on that expert was
    // expensive or free.
    //
    // This weighs the earned demand signal rather than replacing it - the
    // distinction this file already draws when it rejected letting page-cache
    // residency skip the admission bar. Rank is not a property a candidate
    // happens to have; it is the router's own ordering, the same signal
    // substitution is gated on.
    int best_admit_exact_w = -1;
    if (full_sweep && !common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: measuring rank-weighted admission (un-substitutable demand counts extra) ...\n", __func__);
        common_moe_calibration_status_set("measuring rank-weighted admission");
        common_moe_stage_begin("rank-weighted admission", 3);
        double best_aw_tps = -1.0;
        for (const int w : { 0, 2, 4 }) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            common_moe_calib_set_env(string_format("GGML_CUDA_MOE_CACHE_ADMIT_EXACT_WEIGHT=%d", w));
            const double tps = common_moe_bench_candidate_server(
                    self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                    best_threads, next_port(), ctx, n_predict, concurrency, best_cache_mb, -1,
                    active_ngl, active_min_rank, nullptr, 1234, active_quality_sigma);
            common_moe_calibration_status_candidate_done();
            LOG_INF("%s:   un-substitutable demand weight %d -> %s\n", __func__, w,
                    tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
            common_moe_calibration_status_note("rank-weighted admission",
                    w == 0 ? std::string("off (rank-blind)") : string_format("%dx", w),
                    tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
            if (tps > best_aw_tps) {
                best_aw_tps        = tps;
                best_admit_exact_w = w;
            }
        }
        common_moe_calib_set_env(std::string());
        common_moe_stage_end();
        entry.admit_exact_weight = best_admit_exact_w;
        checkpoint("rank-weighted admission");
        if (best_admit_exact_w >= 0) {
            LOG_INF("%s: rank-weighted admission: %s at %.2f tok/s\n", __func__,
                    best_admit_exact_w == 0 ? "off" : string_format("%dx", best_admit_exact_w).c_str(),
                    best_aw_tps);
            common_moe_calibration_status_note("rank-weighted admission",
                    best_admit_exact_w == 0 ? std::string("off") : string_format("%dx", best_admit_exact_w),
                    string_format("SELECTED - %.2f tok/s", best_aw_tps), true, true);
        }
    }

    // The verify pass's own substitution floor.
    //
    // Only meaningful with a draft attached, because only a verify pass
    // presents more than one token to a decode-time dispatch. It is measured
    // separately from the generation floor because the two are paid for in
    // different currencies: a stand-in during generation lands in a token the
    // user reads, while a stand-in during verification can only move the
    // accept/reject boundary - every token out of an accepted position is the
    // draft's own, exact. So this floor can go lower than the one above it,
    // and the answer gate still guards the result.
    //
    // The reason to want it lower is where the cost is: a verify batch demands
    // the union of N+1 tokens' routing, which is the widest expert demand
    // anything in this system makes, and on a model whose experts do not fit
    // that width is where the draft's throughput goes. Measured at 64k with
    // 0.97 acceptance, the draft still lost to no draft at all - not for
    // predicting badly, but for what its verify pass costs.
    int best_verify_rank = -2;
    if (mtp_configured && full_sweep && !common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: measuring how hard a verify pass may lean on stand-ins ...\n", __func__);
        common_moe_calibration_status_set("measuring the verify substitution floor");
        common_moe_stage_begin("verify substitution", 3);
        double best_vr_tps = -1.0;
        // -1 = same as generation (today's behaviour and the baseline); then
        // progressively more tolerant, down to substituting every rank.
        for (const int vr : { -1, 2, 0 }) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            common_moe_calib_set_env(vr < 0 ? std::string()
                    : string_format("GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK_VERIFY=%d", vr));
            const double tps = common_moe_bench_candidate_server(
                    self_exe, path_model, sub_mtp_path, best_n, sub_n_max, best_threads,
                    next_port(), ctx, n_predict, concurrency, best_cache_mb, -1,
                    active_ngl, active_min_rank, nullptr, 1234, active_quality_sigma,
                    /* verify_answers */ true, /* with_reasoning */ true);
            common_moe_calibration_status_candidate_done();
            const std::string label = vr < 0 ? std::string("same floor as generation")
                                             : string_format("from rank %d", vr);
            LOG_INF("%s:   verify substitution %s -> %s%s\n", __func__, label.c_str(),
                    tps > 0 ? string_format("%.2f tok/s", tps).c_str()
                            : (tps == COMMON_MOE_TPS_REJECTED ? "rejected on answer quality" : "failed"),
                    tps > 0 ? common_moe_last_acceptance_str().c_str() : "");
            common_moe_calibration_status_note("verify substitution", label,
                    tps > 0 ? string_format("%.2f tok/s%s", tps, common_moe_last_acceptance_str().c_str())
                            : (tps == COMMON_MOE_TPS_REJECTED ? std::string("rejected on answer quality")
                                                              : std::string("failed")), tps > 0);
            if (tps > best_vr_tps) {
                best_vr_tps      = tps;
                best_verify_rank = vr;
            }
        }
        common_moe_calib_set_env(std::string());
        common_moe_stage_end();
        entry.sub_min_rank_verify = best_verify_rank;
        checkpoint("verify substitution");
        if (best_vr_tps > 0) {
            LOG_INF("%s: verify substitution: %s at %.2f tok/s\n", __func__,
                    best_verify_rank < 0 ? "same floor as generation"
                                         : string_format("from rank %d", best_verify_rank).c_str(),
                    best_vr_tps);
            common_moe_calibration_status_note("verify substitution",
                    best_verify_rank < 0 ? std::string("same floor as generation")
                                         : string_format("from rank %d", best_verify_rank),
                    string_format("SELECTED - %.2f tok/s", best_vr_tps), true, true);
        }
    }

    // The live-trained prerouter. moe-cache already carries a full online
    // predictor - a logistic regression over the hidden state, trained by SGD
    // on every real routing decision (positives: the experts the router chose;
    // negatives: sampled non-choices), scored before each update so its
    // accuracy is not self-confirming, feeding the same free-slot-only
    // admission the atlas uses, and persisted across restarts. None of that
    // has ever been measured: it sits behind GGML_CUDA_MOE_CACHE_TRAIN_PREDICTOR
    // and defaults to 0, so every run so far has served with it off.
    //
    // It is measured here rather than shipped on for the reason prefetch
    // always is in this file: a predictor that is wrong is not neutral. The
    // ring at 5% collapsed the hit rate from 28.8% to 6.4% by spending slots
    // on guesses, and this admits on the same kind of evidence. So: measure
    // off against on, and let the number decide.
    //
    // Learning rate is swept with it because the two are not separable - the
    // predictor trains during the same short candidate run it is judged on, so
    // a rate too low never leaves its prior inside the window and a rate too
    // high tracks noise. Both look like "the predictor does not help".
    int best_train_predictor = -1;
    if (full_sweep && !common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: measuring the live-trained prerouter (it ships off, unmeasured) ...\n", __func__);
        common_moe_calibration_status_set("measuring the trained prerouter");
        common_moe_stage_begin("trained prerouter", 2);
        double best_pred_tps = -1.0;
        struct pred_cand { const char * label; const char * env; int on; };
        // Every arm carries the same ring, including "off" - the ring changes
        // throughput on its own (it is the router-lookahead's landing space
        // too), so an A/B where only the "on" arms have one measures the ring,
        // not the predictor.
        // Two arms. The learning rate is dropped: 11.07 against 10.84 tok/s
        // between lr 0.01 and 0.05 is 2%, a quarter of the spread this machine
        // shows between identical candidates, so that candidate resolves
        // nothing. On/off is worth measuring - that is a mechanism acting or
        // not, rather than a parameter tweak.
        const pred_cand cands[] = {
            { "off",  "GGML_CUDA_MOE_CACHE_RING_PCT=10 GGML_CUDA_MOE_CACHE_ATLAS_ADMIT_RING=1 GGML_CUDA_MOE_CACHE_TRAIN_PREDICTOR=0", 0 },
            { "on",   "GGML_CUDA_MOE_CACHE_RING_PCT=10 GGML_CUDA_MOE_CACHE_ATLAS_ADMIT_RING=1 GGML_CUDA_MOE_CACHE_TRAIN_PREDICTOR=1", 1 },
        };
        for (const auto & c : cands) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            common_moe_calib_set_env(c.env);
            const double tps = common_moe_bench_candidate_server(
                    self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                    best_threads, next_port(), ctx, n_predict, concurrency, best_cache_mb, -1,
                    active_ngl, active_min_rank, nullptr, 1234, active_quality_sigma);
            common_moe_calibration_status_candidate_done();
            LOG_INF("%s:   prerouter %s -> %s\n", __func__, c.label,
                    tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
            common_moe_calibration_status_note("trained prerouter", c.label,
                    tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
            if (tps > best_pred_tps) {
                best_pred_tps       = tps;
                best_train_predictor = c.on;
            }
        }
        common_moe_calib_set_env(std::string());
        common_moe_stage_end();
        entry.train_predictor = best_train_predictor;
        checkpoint("trained prerouter");
        if (best_train_predictor >= 0) {
            LOG_INF("%s: trained prerouter: %s at %.2f tok/s\n", __func__,
                    best_train_predictor ? "on" : "off", best_pred_tps);
            common_moe_calibration_status_note("trained prerouter",
                    best_train_predictor ? std::string("on") : std::string("off"),
                    string_format("SELECTED - %.2f tok/s", best_pred_tps), true, true);
        }
    }

    // What the trained prerouter is allowed to influence beyond admission.
    // The learned score answers "is this expert about to be wanted", and that
    // is useful to more than one consumer: eviction should not discard what is
    // about to be wanted, and substitution - when it has no better way to
    // separate two equally-plausible stand-ins - may as well prefer one that
    // is wanted anyway. Both ship at weight 0 and are measured here.
    //
    // Only worth spending budget on when the predictor itself won: with it
    // off there is no score to weight, and every candidate here would measure
    // the same thing twice.
    int    best_pred_admit   = -1;
    double best_pred_evict_w = -1.0;
    double best_pred_sub_w   = -1.0;
    if (best_train_predictor > 0 && full_sweep && !common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: measuring what the prerouter may influence beyond admission ...\n", __func__);
        common_moe_calibration_status_set("measuring prerouter influence");
        common_moe_stage_begin("prerouter influence", 3);
        double best_infl_tps = -1.0;
        struct infl_cand { const char * label; int admit; double evict_w; double sub_w; };
        // The roles are assigned now, not searched: lookahead owns admission
        // because it is punctual, the prerouter owns retention because a late
        // prediction still tells you what not to discard. Two models agreed
        // and the timing argument explains why (see spec_src in moe-cache.cu).
        //
        // So this measures the role's STRENGTH, plus two arms that can still
        // falsify it: "inert" as the floor, and "admission too" in case a
        // model exists where the prerouter's queue is never contended and it
        // can be punctual after all.
        const infl_cand cands[] = {
            // Three arms. The strength sweep is gone - 1.0 against 2.0 on a
            // capped additive bias is a second-order difference on a machine
            // that cannot resolve first-order ones. What remains: the inert
            // floor (is it doing anything), the role plus the one consumer the
            // two models disagreed about, and the falsifier.
            { "inert (floor)",                  0, 0.0, 0.0 },
            { "eviction + substitution",        0, 1.0, 1.0 },
            { "eviction + admission (falsify)", 1, 1.0, 0.0 },
        };
        for (const auto & c : cands) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            // Admission is switched on only for the arm that measures it; the
            // other two arms measure consumers that need no slot at all, and
            // giving them a warming path would fold two effects into one
            // number.
            common_moe_calib_set_env(string_format(
                    "GGML_CUDA_MOE_CACHE_RING_PCT=10 GGML_CUDA_MOE_CACHE_ATLAS_ADMIT_RING=1 "
                    "GGML_CUDA_MOE_CACHE_PREDICTOR_ADMIT=%d "
                    "GGML_CUDA_MOE_CACHE_PREDICTOR_EVICT_WEIGHT=%.3f "
                    "GGML_CUDA_MOE_CACHE_PREDICTOR_SUB_WEIGHT=%.3f",
                    c.admit, c.evict_w, c.sub_w));
            const double tps = common_moe_bench_candidate_server(
                    self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                    best_threads, next_port(), ctx, n_predict, concurrency, best_cache_mb, -1,
                    active_ngl, active_min_rank, nullptr, 1234, active_quality_sigma);
            common_moe_calibration_status_candidate_done();
            LOG_INF("%s:   prerouter %s -> %s\n", __func__, c.label,
                    tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
            common_moe_calibration_status_note("prerouter influence", c.label,
                    tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
            if (tps > best_infl_tps) {
                best_infl_tps     = tps;
                best_pred_admit   = c.admit;
                best_pred_evict_w = c.evict_w;
                best_pred_sub_w   = c.sub_w;
            }
        }
        common_moe_calib_set_env(std::string());
        common_moe_stage_end();
        entry.predictor_admit   = best_pred_admit;
        entry.predictor_evict_w = best_pred_evict_w;
        entry.predictor_sub_w   = best_pred_sub_w;
        checkpoint("prerouter influence");
        if (best_pred_evict_w >= 0.0) {
            LOG_INF("%s: prerouter influence: evict %.2f sub %.2f at %.2f tok/s\n", __func__,
                    best_pred_evict_w, best_pred_sub_w, best_infl_tps);
            common_moe_calibration_status_note("prerouter influence",
                    string_format("evict %.2f sub %.2f", best_pred_evict_w, best_pred_sub_w),
                    string_format("SELECTED - %.2f tok/s", best_infl_tps), true, true);
        }
    }

    // Prediction ring: how much of each pool the router lookahead may use. At full
    // occupancy a prediction otherwise has nowhere to go; too large a ring takes
    // slots from experts that were really demanded. Measured on decode, as served
    // (draft, depth, floor and cache size already chosen). A ring slot holds the
    // exact expert, so it changes no arithmetic - but the winner is still re-run
    // through the answer check against the substitution-off bar before it is kept.
    int best_ring_pct = -1;
    // The ring has measured best at 0% - off - on every model run here, and the
    // candidates that lost did not lose to the ring. On qwen4exp at 64k the three
    // rungs read 2.48 / 1.49 / 1.69 tok/s while draft acceptance was IDENTICAL
    // across all three (0.69, 52 of 75): the same generation, three different
    // throughputs, which is the definition of a noise reading. Gemma-4 read
    // 70.65 / 54.26 / 67.17 and again chose 0%.
    //
    // Off is also what the runtime already does, so the stage's usual outcome is
    // to spend three candidates confirming the default.
    // GGML_MOE_CALIBRATE_SEARCH_RING=1 measures it.
    const bool search_ring = [] {
        const char * e = getenv("GGML_MOE_CALIBRATE_SEARCH_RING");
        return e && atoi(e) != 0;
    }();
    if (!search_ring) {
        LOG_WRN("%s: prediction ring left off - 0%% has won on every model measured here, and on qwen4exp "
                "at 64k all three rungs produced identical draft acceptance with different throughput, "
                "which is a noise reading (GGML_MOE_CALIBRATE_SEARCH_RING=1 to measure it)\n", __func__);
        common_moe_calibration_status_note("prediction ring", "0% (off)",
                "DEFAULT - 0% has won on every model; the rungs differ only by noise", true, true);
    } else if (!common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: measuring the prediction ring (GGML_CUDA_MOE_CACHE_RING_PCT) ...\n", __func__);
        common_moe_calibration_status_set("measuring the prediction ring");
        double best_ring_tps = -1.0;
        // One server, three ring sizes: the ring percentage is a live policy knob,
        // so each value is a POST rather than a reload. The ring only fills as the
        // pool runs, which a persistent process gives it - a cold start would
        // measure an empty ring every time.
        bool ring_live = false;
        for (const int pct : { 0, 5, 10 }) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            g_moe_calib_ring_pct.store(pct);
            const std::string body = string_format("{\"GGML_CUDA_MOE_CACHE_RING_PCT\": \"%d\"}", pct);
            double tps = ring_live ? measure_live(body, n_predict) : -1.0;
            if (!ring_live) {
                tps = open_live(n_predict);
                ring_live = g_moe_live_port > 0;
                if (ring_live && pct != 0) {
                    tps = measure_live(body, n_predict);   // the launch ran at the default
                }
            }
            common_moe_calibration_status_candidate_done();
            LOG_INF("%s:   ring %d%% -> %s%s\n", __func__, pct,
                    tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed",
                    tps > 0 ? common_moe_last_acceptance_str().c_str() : "");
            common_moe_calibration_status_note("prediction ring", string_format("%d%%", pct),
                    tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
            if (tps > best_ring_tps) {
                best_ring_tps = tps;
                best_ring_pct = pct;
            }
        }
        // Release the live candidate BEFORE the confirm below launches its own: it
        // runs at a different length with reasoning on, so it cannot reuse this
        // one, and a 12 GiB card has no room for both - the new candidate failed
        // with "unable to allocate CUDA0 buffer" and calibration sat in the health
        // wait. The ladder already did this; this stage did not.
        common_moe_live_stop();
        if (best_ring_pct > 0 && !common_moe_calibrate_budget_spent()) {
            g_moe_calib_ring_pct.store(best_ring_pct);
            const double checked = common_moe_bench_candidate_server(
                    self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                    best_threads, next_port(), ctx, n_predict * 4, concurrency, best_cache_mb, -1,
                    active_ngl, active_min_rank, nullptr, 1234, active_quality_sigma,
                    /* verify_answers */ true, /* with_reasoning */ true);
            common_moe_calibration_status_candidate_done();
            if (checked <= 0) {
                LOG_WRN("%s:   ring %d%% did not pass the answer check - keeping the ring off\n",
                        __func__, best_ring_pct);
                common_moe_calibration_status_note("prediction ring", string_format("%d%%", best_ring_pct),
                        "rejected by the answer check", false);
                best_ring_pct = 0;
            }
        }
        common_moe_live_stop();
        entry.moe_cache_ring_pct = best_ring_pct;
        checkpoint("prediction ring");
        g_moe_calib_ring_pct.store(best_ring_pct > 0 ? best_ring_pct : -1);
        if (best_ring_pct >= 0) {
            LOG_INF("%s: prediction ring: %d%%\n", __func__, best_ring_pct);
            common_moe_calibration_status_note("prediction ring", string_format("%d%%", best_ring_pct),
                    string_format("SELECTED - %.2f tok/s", best_ring_tps), true, true);
        }
    }

    // Probabilistic draft acceptance. Only meaningful with a draft model, and
    // only worth measuring at the sampling actually served: at greedy the target
    // almost always reproduces the drafted token anyway, so exact-match already
    // captures nearly everything and this looks like a no-op. The probes run at
    // COMMON_MOE_PROBE_TEMP, which is where the two paths diverge.
    //
    // The flag's own help says it "only ever raises or matches the accept rate
    // versus exact-match, never lowers it". That is a claim about acceptance,
    // not about throughput - a higher accept rate still has to pay for the
    // probability bookkeeping - so measure the thing we care about rather than
    // trusting the monotonicity argument.
    int    best_prob_accept     = -1;
    double best_prob_accept_tps = cheap_incumbent_tps > 0.0 ? cheap_incumbent_tps : best_tps;
    if (!mtp_path_for_threads.empty() && n_max_for_threads > 0 && !common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: measuring probabilistic draft acceptance (--spec-prob-accept) ...\n", __func__);
        common_moe_calibration_status_set("measuring probabilistic draft acceptance");
        for (const int candidate : {0, 1}) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            const double tps = common_moe_bench_candidate_server(
                    self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                    best_threads, next_port(), ctx, n_predict, concurrency, best_cache_mb, -1,
                    active_ngl, active_min_rank, nullptr, 1234,
                    std::numeric_limits<double>::quiet_NaN(), false, false, std::string(), candidate);
            LOG_INF("%s:   spec-prob-accept=%s -> %s%s\n", __func__, candidate ? "on" : "off",
                    tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed",
                    tps > 0 ? common_moe_last_acceptance_str().c_str() : "");
            common_moe_calibration_status_note("draft acceptance",
                    candidate ? std::string("probabilistic") : std::string("exact-match"),
                    tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
            common_moe_calibration_status_candidate_done();
            if (tps > best_prob_accept_tps) {
                best_prob_accept_tps = tps;
                best_prob_accept     = candidate;
            }
        }
        if (best_prob_accept >= 0) {
            LOG_INF("%s: draft acceptance: %s at %.2f tok/s\n", __func__,
                    best_prob_accept ? "probabilistic" : "exact-match", best_prob_accept_tps);
            common_moe_calibration_status_note("draft acceptance",
                    best_prob_accept ? std::string("probabilistic") : std::string("exact-match"),
                    string_format("SELECTED - %.2f tok/s", best_prob_accept_tps), true, true);
        }
    }

    // The drafter cascade. MTP alone pays one draft forward pass every decode
    // step; an n-gram drafter in front of it proposes from the live suffix tree
    // with no forward pass at all, and MTP only runs on the steps the n-gram
    // drafter declines (common_speculative_init orders the impls and stops at
    // the first one that produces a draft). On this hardware the scarce resource
    // is expert bandwidth, not compute, so a step that skips the draft model
    // entirely is worth more than the accept-rate arithmetic alone suggests -
    // but only on traffic repetitive enough for the suffix tree to hit, so
    // measure it rather than defaulting it on.
    std::string best_spec_types;
    double best_spec_types_tps = cheap_incumbent_tps > 0.0 ? cheap_incumbent_tps : best_tps;
    if (!mtp_path_for_threads.empty() && n_max_for_threads > 0 && !common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: measuring the drafter cascade (--spec-type) ...\n", __func__);
        common_moe_calibration_status_set("measuring the drafter cascade");
        for (const char * candidate : {"draft-mtp", "ngram-suffix,draft-mtp"}) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            const double tps = common_moe_bench_candidate_server(
                    self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                    best_threads, next_port(), ctx, n_predict, concurrency, best_cache_mb, -1,
                    active_ngl, active_min_rank, nullptr, 1234,
                    std::numeric_limits<double>::quiet_NaN(), false, false, std::string(),
                    best_prob_accept, candidate);
            LOG_INF("%s:   spec-type=%s -> %s%s\n", __func__, candidate,
                    tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed",
                    tps > 0 ? common_moe_last_acceptance_str().c_str() : "");
            common_moe_calibration_status_note("drafter cascade", std::string(candidate),
                    tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
            common_moe_calibration_status_candidate_done();
            if (tps > best_spec_types_tps) {
                best_spec_types_tps = tps;
                best_spec_types     = candidate;
            }
        }
        if (!best_spec_types.empty()) {
            LOG_INF("%s: drafter cascade: %s at %.2f tok/s\n", __func__,
                    best_spec_types.c_str(), best_spec_types_tps);
            common_moe_calibration_status_note("drafter cascade", best_spec_types,
                    string_format("SELECTED - %.2f tok/s", best_spec_types_tps), true, true);
        }
    }

    // Draft confidence gate, searched jointly with depth. The depth search above
    // ran with p_min at its default of 0, which never stops a draft early, so it
    // could only find the depth that pays when every step drafts in full. The gate
    // changes that trade: a deeper draft that stops once the draft is unsure keeps
    // the cheap, confident tokens and skips the ones verification would reject -
    // and on this hardware each rejected token also widened the verify pass's
    // expert set for nothing. So each rung is tried at the chosen depth and at
    // twice it; the incumbent is the chosen depth with no gate.
    double best_p_min     = -1.0;
    int    best_p_min_nmax = n_max_for_threads;
    if (!mtp_path_for_threads.empty() && n_max_for_threads > 0 && !common_moe_calibrate_budget_spent()) {
        double best_p_min_tps = std::max(best_spec_types_tps, best_prob_accept_tps);
        LOG_INF("%s: measuring the draft confidence gate (--spec-draft-p-min) with depth ...\n", __func__);
        common_moe_calibration_status_set("measuring the draft confidence gate");
        const int depths[] = { n_max_for_threads, std::min(32, n_max_for_threads * 2) };
        for (const int depth : depths) {
            for (const double p_min : {0.5, 0.8}) {
                if (common_moe_calibrate_budget_spent()) {
                    break;
                }
                const double tps = common_moe_bench_candidate_server(
                        self_exe, path_model, mtp_path_for_threads, best_n, depth,
                        best_threads, next_port(), ctx, n_predict, concurrency, best_cache_mb, -1,
                        active_ngl, active_min_rank, nullptr, 1234,
                        std::numeric_limits<double>::quiet_NaN(), false, false, std::string(),
                        best_prob_accept, best_spec_types, p_min);
                LOG_INF("%s:   spec-draft-p-min=%.2f at spec-draft-n-max=%d -> %s%s\n", __func__, p_min, depth,
                        tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed",
                        tps > 0 ? common_moe_last_acceptance_str().c_str() : "");
                common_moe_calibration_status_note("draft confidence gate",
                        string_format("p_min %.2f, depth %d", p_min, depth),
                        tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
                common_moe_calibration_status_candidate_done();
                if (tps > best_p_min_tps) {
                    best_p_min_tps  = tps;
                    best_p_min      = p_min;
                    best_p_min_nmax = depth;
                }
            }
            if (depths[1] == depths[0]) {
                break; // depth already at the cap - the second pass would repeat the first
            }
        }
        if (best_p_min >= 0.0) {
            LOG_INF("%s: draft confidence gate: p_min %.2f at depth %d, %.2f tok/s\n", __func__,
                    best_p_min, best_p_min_nmax, best_p_min_tps);
            common_moe_calibration_status_note("draft confidence gate",
                    string_format("p_min %.2f, depth %d", best_p_min, best_p_min_nmax),
                    string_format("SELECTED - %.2f tok/s", best_p_min_tps), true, true);
            // a deeper draft won only because of the gate, so the two are saved together
            if (best_n_max > 0 || best_p_min_nmax != n_max_for_threads) {
                best_n_max = best_p_min_nmax;
            }
        }
    }

    // Features that were shipping on a number nobody measured. Each is one knob at
    // a time against the configuration chosen above, so a winner is a real win in
    // the regime actually served rather than in isolation. Throughput decides;
    // the degeneracy guard still rejects broken output, and none of these changes
    // which expert is computed - they change which are kept, when, and at what
    // width - except neuron reduction, which is confirmed at length below.
    auto measure_feature = [&](const char * label, const char * value, const std::string & env) -> double {
        common_moe_calib_set_env(env);
        const double tps = common_moe_bench_candidate_server(
                self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                best_threads, next_port(), ctx, n_predict, concurrency, best_cache_mb, -1,
                active_ngl, active_min_rank, nullptr, 1234, active_quality_sigma);
        common_moe_calibration_status_candidate_done();
        LOG_INF("%s:   %s=%s -> %s%s\n", __func__, label, value,
                tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed",
                tps > 0 ? common_moe_last_acceptance_str().c_str() : "");
        common_moe_calibration_status_note(label, value,
                tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
        return tps;
    };

    // Heat-aware neuron subsetting: keep only the K highest-mass neurons of an
    // expert so more experts fit the same VRAM. It has been ON by default with
    // K = 256 and a 256 MiB budget, neither ever measured - and unlike the others
    // here it changes the arithmetic, so its winner is confirmed at full length
    // with the answer check before it is kept.
    int best_neuron_k = -1;
    if (full_sweep && !common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: measuring heat-aware neuron subsetting (it has been on at K=256, unmeasured) ...\n", __func__);
        common_moe_calibration_status_set("measuring neuron subsetting");
        common_moe_stage_begin("neuron subsetting", 5);
        double best_neuron_tps = -1.0;
        for (const int k : { 0, 128, 256, 512 }) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            const std::string env = k == 0
                ? std::string("GGML_CUDA_MOE_CACHE_NEURON_REDUCE=0")
                : string_format("GGML_CUDA_MOE_CACHE_NEURON_REDUCE=1 GGML_CUDA_MOE_CACHE_NEURON_REDUCE_K=%d", k);
            const double tps = measure_feature("neuron subsetting",
                    k == 0 ? "off" : string_format("K=%d", k).c_str(), env);
            if (tps > best_neuron_tps) {
                best_neuron_tps = tps;
                best_neuron_k   = k;
            }
        }
        if (best_neuron_k > 0 && !common_moe_calibrate_budget_spent()) {
            common_moe_calib_set_env(string_format(
                    "GGML_CUDA_MOE_CACHE_NEURON_REDUCE=1 GGML_CUDA_MOE_CACHE_NEURON_REDUCE_K=%d", best_neuron_k));
            const double checked = common_moe_bench_candidate_server(
                    self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                    best_threads, next_port(), ctx, n_predict * 4, concurrency, best_cache_mb, -1,
                    active_ngl, active_min_rank, nullptr, 1234, active_quality_sigma,
                    /* verify_answers */ true, /* with_reasoning */ true);
            common_moe_calibration_status_candidate_done();
            if (checked <= 0) {
                LOG_WRN("%s:   neuron subsetting K=%d did not pass the answer check - recording it off\n",
                        __func__, best_neuron_k);
                common_moe_calibration_status_note("neuron subsetting", string_format("K=%d", best_neuron_k),
                        "rejected by the answer check", false);
                best_neuron_k = 0;
            }
        }
        if (best_neuron_k >= 0) {
            common_moe_calib_set_env(best_neuron_k > 0
                    ? string_format("GGML_CUDA_MOE_CACHE_NEURON_REDUCE=1 GGML_CUDA_MOE_CACHE_NEURON_REDUCE_K=%d", best_neuron_k)
                    : std::string("GGML_CUDA_MOE_CACHE_NEURON_REDUCE=0"));
            LOG_INF("%s: neuron subsetting: %s\n", __func__,
                    best_neuron_k > 0 ? string_format("K=%d", best_neuron_k).c_str() : "off");
            common_moe_calibration_status_note("neuron subsetting",
                    best_neuron_k > 0 ? string_format("K=%d", best_neuron_k) : std::string("off"),
                    string_format("SELECTED - %.2f tok/s", best_neuron_tps), true, true);
        }
    }
    common_moe_stage_end();
    const std::string neuron_env = common_moe_calib_get_env();

    // The remaining knobs, each on/off against the incumbent. All three have been
    // off since they were written, with no measurement either way: group admission
    // (admit an expert's whole co-activation group at once), coverage eviction
    // (evict by how well the rest of the pool already covers a slot's routing
    // neighbourhood), and the host hot-expert buffer (a RAM tier below VRAM).
    int best_group_admit = -1, best_coverage_evict = -1, best_host_mb = -1;
    if (!common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: measuring the cache features that ship off (group admit, coverage evict, host buffer) ...\n", __func__);
        common_moe_calibration_status_set("measuring the off-by-default cache features");
        const double incumbent = measure_feature("cache features", "none (incumbent)", neuron_env);
        struct feature { const char * label; const char * value; const char * env; int * out; };
        const feature features[] = {
            { "group admit",    "on",      "GGML_CUDA_MOE_CACHE_GROUP_ADMIT=1",    &best_group_admit    },
            { "coverage evict", "on",      "GGML_CUDA_MOE_CACHE_COVERAGE_EVICT=1", &best_coverage_evict },
            { "host buffer",    "512 MiB", "GGML_CUDA_MOE_CACHE_HOST_MB=512",      &best_host_mb        },
        };
        // Group admit and coverage eviction are live policy knobs now, so those two
        // are POSTed to the incumbent's own server rather than relaunching. The host
        // buffer is not - it allocates - so it still needs its own launch.
        for (const auto & f : features) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            const bool live_ok = g_moe_live_port > 0 && strcmp(f.label, "host buffer") != 0;
            double tps = -1.0;
            if (live_ok) {
                const std::string key = strcmp(f.label, "group admit") == 0
                        ? "GGML_CUDA_MOE_CACHE_GROUP_ADMIT" : "GGML_CUDA_MOE_CACHE_COVERAGE_EVICT";
                tps = measure_live(string_format("{\"%s\": \"1\"}", key.c_str()), n_predict);
                LOG_INF("%s:   %s=%s -> %s\n", __func__, f.label, f.value,
                        tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
                common_moe_calibration_status_note(f.label, f.value,
                        tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
                // put it back before the next feature is judged
                (void) measure_live(string_format("{\"%s\": \"0\"}", key.c_str()), 1);
            } else {
                tps = measure_feature(f.label, f.value, neuron_env + " " + f.env);
            }
            // Only adopted on a real gain over the incumbent - a tie keeps the
            // simpler configuration, and every one of these costs memory or work.
            const bool win = tps > 0 && incumbent > 0 && tps > incumbent * 1.02;
            *f.out = win ? (strcmp(f.label, "host buffer") == 0 ? 512 : 1) : 0;
            if (win) {
                common_moe_calibration_status_note(f.label, f.value,
                        string_format("SELECTED - %.2f tok/s vs %.2f", tps, incumbent), true, true);
            }
        }
        common_moe_calib_set_env(neuron_env);
    }

    // The ranking and timing constants. Every one of these decides which experts
    // survive, or how fast the cache reacts, and every one shipped as a number
    // somebody picked. Swept one at a time, each against the incumbent, keeping a
    // winner only on a real margin so a tie leaves the existing default in place.
    // Ordered by how much of the system reads them, because a spent budget stops
    // the sweep wherever it has got to.
    std::string tuned_constants;
    if (!common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: sweeping the ranking and timing constants that shipped as guesses ...\n", __func__);
        // 21 candidates, one sample each, at effect sizes this machine cannot
        // resolve. Off unless asked for.
        //
        // These five are internal policy constants - the NVMe cost tier, the
        // cold-after threshold, the protected-segment cap, the eviction window,
        // the cold-sweep interval. Each was swept across three values with a
        // single measurement per value, and the spreads that came back are
        // inside the run-to-run spread of an unchanged configuration on this
        // hardware: 12-20% has been measured repeatedly between identical
        // candidates (exact-match 7.52 then 6.71, fitt=448 12.45 then 10.35,
        // ncmoe=21 68.39 then 61.47). A stage that cannot separate its own
        // effect from noise is not measuring its knob; it is sampling the
        // noise, and then FIXING a value that every later stage inherits.
        //
        // That is not hypothetical. It is how the substitution floor settled on
        // rank 6 over a rank 4 that ran 59% faster, and how -ub 2048 was locked
        // in before the draft that cannot coexist with it existed.
        //
        // So they keep their defaults, which are the values this fork has
        // actually run on, and the budget goes to the stages whose effects are
        // larger than the noise: what fits (deterministic), and what the answer
        // gate judges (where quality decides rather than throughput).
        // GGML_MOE_CALIBRATE_TUNE_CONSTANTS=1 restores the sweep.
        const bool sweep_constants = [] {
            const char * e = getenv("GGML_MOE_CALIBRATE_TUNE_CONSTANTS");
            return e && atoi(e) != 0;
        }();
        if (!sweep_constants) {
            LOG_INF("%s: skipping the tuning-constant sweep - 21 candidates at effect sizes inside this "
                    "machine's measured run-to-run spread, which samples noise and then fixes it for every "
                    "later stage (set GGML_MOE_CALIBRATE_TUNE_CONSTANTS=1 to sweep them)\n", __func__);
            common_moe_calibration_status_note("tuning constants", "left at their defaults",
                    "not measured - effects are inside this machine's noise floor", true, true);
        } else {
        common_moe_calibration_status_set("sweeping the tuning constants");
        common_moe_stage_begin("tuning constants", 21);
        struct knob { const char * label; const char * env; const char * values[3]; };
        const knob knobs[] = {
            // read by eviction, admission and substitution - the widest blast radius
            { "NVMe cost tier",     "GGML_CUDA_MOE_CACHE_COST_TIER_NVME",       { "2.0", "5.8", "12.0" } },
            // how long an expert may sit untouched before it counts as cold
            { "cold after",         "GGML_CUDA_MOE_CACHE_COLD_AFTER_S",         { "30",  "120", "300"  } },
            // how much of the pool may be protected from eviction at once
            { "protected cap",      "GGML_CUDA_MOE_CACHE_PROTECTED_CAP_PCT",    { "25",  "50",  "75"   } },
            // how many slots the victim search examines
            { "evict window",       "GGML_CUDA_MOE_CACHE_EVICT_WINDOW",         { "16",  "32",  "64"   } },
            // how often the cold sweep runs
            { "cold sweep",         "GGML_CUDA_MOE_CACHE_COLD_SWEEP_S",         { "1",   "3",   "10"   } },
        };
        // One server for the whole sweep. Every knob below is a live policy knob,
        // so a value change is a POST rather than an 88 GB reload - the difference
        // between 15 launches and 1. The first call launches and keeps it alive;
        // the rest reuse it. Each measurement still runs the same probe, and the
        // candidate is warmed by the one before it, which is the regime served.
        std::string carried = neuron_env;
        common_moe_calib_set_env(carried);
        const double live_incumbent = open_live(n_predict);
        common_moe_calibration_status_candidate_done();
        if (live_incumbent <= 0 || g_moe_live_port <= 0) {
            LOG_WRN("%s: could not hold a candidate open for the constant sweep - skipping it\n", __func__);
            common_moe_live_stop();
        } else {
        for (const auto & k : knobs) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            double best_tps = -1.0;
            const char * best_value = nullptr;
            for (const char * v : k.values) {
                if (common_moe_calibrate_budget_spent()) {
                    break;
                }
                const std::string body = string_format("{\"%s\": \"%s\"}", k.env, v);
                const double tps = measure_live(body, n_predict);
                common_moe_calibration_status_candidate_done();
                LOG_INF("%s:   %s=%s -> %s\n", __func__, k.label, v,
                        tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
                common_moe_calibration_status_note(k.label, v,
                        tps > 0 ? string_format("%.2f tok/s", tps) : std::string("failed"), tps > 0);
                if (tps > best_tps) {
                    best_tps   = tps;
                    best_value = v;
                }
            }
            // Leave the knob on its winner for the knobs that follow, so each is
            // measured against the configuration the earlier ones chose.
            if (best_value) {
                // leave the knob on its winner for the knobs that follow
                (void) measure_live(string_format("{\"%s\": \"%s\"}", k.env, best_value), 1);
            }
            // The middle entry of each row is the shipped default; only carry a
            // winner forward when it actually beat it, so the sweep cannot drift
            // the configuration on noise.
            if (best_value && strcmp(best_value, k.values[1]) != 0) {
                carried += " " + std::string(k.env) + "=" + best_value;
                if (!tuned_constants.empty()) {
                    tuned_constants += " ";
                }
                tuned_constants += std::string(k.env) + "=" + best_value;
                LOG_INF("%s:   %s: %s beats the default %s (%.2f tok/s)\n",
                        __func__, k.label, best_value, k.values[1], best_tps);
                common_moe_calibration_status_note(k.label, best_value,
                        string_format("SELECTED - %.2f tok/s", best_tps), true, true);
            } else if (best_value) {
                LOG_INF("%s:   %s: the default %s stands\n", __func__, k.label, k.values[1]);
            }
        }
        }
        common_moe_live_stop();
        common_moe_stage_end();
        common_moe_calib_set_env(carried);
        }   // sweep_constants
    }

    entry.n_cpu_moe       = (int) best_n;
    entry.gates_version   = COMMON_MOE_CALIBRATION_GATES_VERSION;
    entry.n_threads       = best_threads;
    entry.n_threads_batch = best_threads;
    entry.spec_n_max      = best_n_max;
    entry.concurrency     = concurrency;
    // best_fit_tps is already seeded from the cache-knee best, so it is the
    // running maximum across every stage that measured a full candidate -
    // reporting anything lower here would understate what this entry's own
    // settings actually achieved (the pre-existing code reported only
    // best_threads_tps, which ignored the cache sweep entirely and on
    // gemma-4 understated the result by 71.86 -> 43.57).
    // Include the substitution ladder's own best: substitute_min_rank is part
    // of the configuration this entry caches, so a figure that ignores it
    // understates what the cached config actually delivers. Measured on
    // qwen4exp: the ladder found 1.64 tok/s at rank 2 while the entry
    // reported 0.65, the placement/-ngl number - the same understatement the
    // cache-sweep comment above already describes for a different stage.
    // The confirmed full-length number when there is one, not a max across
    // regimes. best_threads_tps and best_fit_tps are short no-reasoning probes;
    // best_min_rank_tps is a 128-token confirm with reasoning on. Taking the max
    // of the three let a 32-token fit-margin probe of 15.10 become the cached
    // headline while the configuration actually served ~13.
    entry.tok_per_sec     = best_min_rank_tps > 0.0
            ? best_min_rank_tps
            : std::max({best_threads_tps, best_fit_tps});
    entry.moe_cache_mb    = best_cache_mb;
    entry.substitute_min_rank = best_min_rank;
    entry.substitute_quality_sigma = active_quality_sigma;
    entry.fit_target_mb   = best_fit_mb;
    entry.admit_after     = best_admit_after;
    entry.readmit_after   = best_readmit_after;
    entry.spec_prob_accept = best_prob_accept;
    entry.spec_types      = best_spec_types;
    entry.spec_p_min      = best_p_min;
    entry.spec_draft_cpu_moe = best_draft_cpu_moe;
    entry.op_offload_min_batch = best_offload_min_batch;
    // NOT best_ubatch. The draft-fit check above is the only thing that knows
    // whether the chosen micro-batch survives once the draft is attached, and
    // when it steps -ub down (or unsets it) it updates g_moe_calib_ubatch and
    // entry.n_ubatch - not this stale local. Overwriting from best_ubatch here
    // put the REJECTED value in the saved entry.
    //
    // Measured on Qwen3.8-Flash-Next at 64k: the run found -ub 2048 does not fit
    // with the draft, stepped to 512, confirmed 512 serves, logged "keeping -ub
    // 512 so it fits" - and then wrote n_ubatch 2048. Launching normally on that
    // entry re-applies 2048 and OOMs allocating the MTP context, so calibration
    // shipped a configuration it had itself measured as not fitting.
    const int final_ubatch = g_moe_calib_ubatch.load();
    entry.n_ubatch             = final_ubatch != 0 ? final_ubatch : best_ubatch;
    entry.sched_prefetch_experts = best_prefetch;
    entry.moe_cache_ring_pct     = best_ring_pct;
    entry.neuron_reduce_k        = best_neuron_k;
    entry.group_admit            = best_group_admit;
    entry.coverage_evict         = best_coverage_evict;
    entry.host_expert_mb         = best_host_mb;
    entry.draft_exact            = best_draft_exact;
    entry.tuned_constants        = tuned_constants;
    entry.draft_share_pct        = best_share_pct;
    entry.substitute_atlas       = best_sub_atlas;
    entry.atlas_prewarm_k        = best_prewarm_k;
    entry.train_predictor        = best_train_predictor;
    entry.kv_type                = best_kv_type;
    entry.sub_min_rank_verify    = best_verify_rank;
    entry.reduced_share_pct      = best_reduced_share;
    entry.admit_exact_weight     = best_admit_exact_w;
    entry.predictor_admit        = best_pred_admit;
    entry.predictor_evict_w      = best_pred_evict_w;
    entry.predictor_sub_w        = best_pred_sub_w;
    // -1 ("not calibrated / use default") when full GPU residency won its own
    // search above - only recorded as an explicit override when giving up
    // some GPU-resident layers actually measured faster.
    entry.n_gpu_layers    = best_ngl <= (uint32_t) probe.n_layer ? (int) best_ngl : -1;
    // Record the reduction settings this calibration was actually measured
    // under, read from the same environment the cache itself reads (rather
    // than from params, which has no field for them) - see the field
    // comments on common_moe_calibration_entry for why an entry measured
    // with reduction on is not valid for a run with it off.
    {
        const char * en = getenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE");
        if (en && atoi(en) != 0) {
            const char * k  = getenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_K");
            const char * bm = getenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_BUDGET_MB");
            entry.neuron_reduce_k         = k  ? atoi(k)  : 256;
            entry.neuron_reduce_budget_mb = bm ? atoi(bm) : 256;
        }
    }
    entry.calibrated_at   = timebuf;

    // Validate the COMBINATION, not just the parts.
    //
    // Every stage above picks a per-stage winner while the other knobs sit at
    // whatever the previous stage left them on. Nothing has ever run the full
    // set together, so the entry that gets saved is a configuration this
    // machine has never actually executed - and that is not a theoretical
    // worry: the full calibrated config crashed on its first served request
    // (CUDA error in ggml_cuda_mul_mat_cublas_impl) while every stage that
    // chose its values had passed.
    //
    // So: one final candidate with the entry exactly as it will be saved. It
    // is a check, not a search - it changes nothing, it only records whether
    // the combination runs and what it measures. A failure here is recorded
    // and reported rather than silently saved, because a config that cannot
    // serve is worse than one that is merely slower.
    if (!common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: validating the winning combination (nothing above ran the full set together) ...\n", __func__);
        common_moe_calibration_status_set("validating the combination");
        common_moe_stage_begin("final validation", 3);
        // Apply the entry the way a real launch would, so this measures what
        // serving will actually run rather than a hand-built approximation.
        common_moe_apply_quality_knobs(entry, path_model);
        const double final_tps = common_moe_bench_candidate_server(
                self_exe, path_model, sub_mtp_path, best_n, sub_n_max,
                best_threads, next_port(), ctx, n_predict, concurrency, best_cache_mb, -1,
                active_ngl, active_min_rank, nullptr, 1234, active_quality_sigma);
        common_moe_calibration_status_candidate_done();

        // The same combination without the prerouter.
        //
        // The prerouter's own stage compares on against off with every other
        // knob at whatever the preceding stage left it on - which is the only
        // comparison available at that point in the run, and is not the
        // comparison that matters. What matters is whether the configuration
        // actually being saved is faster with the predictor than without it,
        // and nothing measured that: the on/off stage ran before substitution,
        // the ring, the tuning constants and neuron subsetting were decided.
        //
        // Measured on gemma-4, the stage reported +21% (50.16 -> 60.93) while
        // the full combination validated at 58.94 - three numbers that cannot
        // be compared to each other. This is the one that can.
        double final_tps_no_pred = -1.0;
        if (entry.train_predictor > 0) {
            common_moe_calib_set_base_env(std::string());
            common_moe_calib_set_env("GGML_CUDA_MOE_CACHE_TRAIN_PREDICTOR=0");
            final_tps_no_pred = common_moe_bench_candidate_server(
                    self_exe, path_model, sub_mtp_path, best_n, sub_n_max,
                    best_threads, next_port(), ctx, n_predict, concurrency, best_cache_mb, -1,
                    active_ngl, active_min_rank, nullptr, 1234, active_quality_sigma);
            common_moe_calib_set_env(std::string());
            common_moe_calibration_status_candidate_done();
            LOG_INF("%s: the winning combination WITHOUT the prerouter: %s\n", __func__,
                    final_tps_no_pred > 0 ? string_format("%.2f tok/s", final_tps_no_pred).c_str() : "failed");
            common_moe_calibration_status_note("final validation", "same combination, prerouter off",
                    final_tps_no_pred > 0 ? string_format("%.2f tok/s", final_tps_no_pred) : std::string("failed"),
                    final_tps_no_pred > 0, false);
            // Proven, not promised: if the predictor does not earn its place in
            // the configuration being saved, it is not saved as on.
            if (final_tps > 0 && final_tps_no_pred > final_tps) {
                LOG_WRN("%s: the prerouter's own stage preferred it, but the full combination is faster "
                        "without it (%.2f vs %.2f tok/s) - recording it off. A stage-level win measured "
                        "against different neighbours is not a win here\n",
                        __func__, final_tps_no_pred, final_tps);
                entry.train_predictor   = 0;
                entry.predictor_evict_w = 0.0;
                entry.predictor_sub_w   = 0.0;
                entry.tok_per_sec       = final_tps_no_pred;
                common_moe_calibration_status_note("trained prerouter", "off",
                        string_format("OVERRULED by final validation - %.2f vs %.2f tok/s",
                                      final_tps_no_pred, final_tps), true, true);
            } else if (final_tps > 0 && final_tps_no_pred > 0) {
                LOG_INF("%s: the prerouter holds up on the full combination: %.2f with, %.2f without\n",
                        __func__, final_tps, final_tps_no_pred);
                common_moe_calibration_status_note("trained prerouter", "on",
                        string_format("CONFIRMED on the full combination - %.2f vs %.2f tok/s",
                                      final_tps, final_tps_no_pred), true, true);
            }
        }
        // Everything on, as the floor this run may not ship below.
        //
        // Each stage above starts from a feature being OFF and requires
        // evidence to turn it on. On a machine with a 12-20% run-to-run spread
        // that is a systematic bias toward off: a feature genuinely worth 8%
        // fails to prove itself about half the time, and is disabled by the
        // coin flip rather than by its effect. Run enough stages that way and
        // the saved entry is a configuration with most of its mechanisms
        // switched off, each for a defensible-looking reason.
        //
        // Measured directly: a hand-assembled configuration with every feature
        // enabled outran what this calibration produced. That is the bias, not
        // a coincidence - so the all-on configuration is measured here as a
        // reference, and if the calibrated combination cannot beat it, the
        // calibrated entry is not what gets saved. Proving each feature
        // individually is how the run learns; it is not permitted to be how
        // the run loses.
        double all_on_tps = -1.0;
        if (!common_moe_calibrate_budget_spent()) {
            common_moe_calib_set_env(
                    "GGML_SCHED_PREFETCH_EXPERTS=1 "
                    "GGML_CUDA_MOE_CACHE_RING_PCT=10 GGML_CUDA_MOE_CACHE_ATLAS_ADMIT_RING=1 "
                    "GGML_CUDA_MOE_CACHE_ADMIT_AFTER=1 "
                    "GGML_CUDA_MOE_CACHE_SUBSTITUTE_ATLAS=4 "
                    "GGML_CUDA_MOE_CACHE_TRAIN_PREDICTOR=1 GGML_CUDA_MOE_CACHE_TRAIN_NEXT_LAYER=1 "
                    "GGML_CUDA_MOE_CACHE_PREDICTOR_EVICT_WEIGHT=1.0 "
                    "GGML_CUDA_MOE_CACHE_PREDICTOR_SUB_WEIGHT=1.0 "
                    "GGML_CUDA_MOE_CACHE_GROUP_ADMIT=1 GGML_CUDA_MOE_CACHE_COVERAGE_EVICT=1 "
                    "GGML_CUDA_MOE_CACHE_ATLAS_WARM=1 GGML_CUDA_MOE_CACHE_ATLAS_ALIGN_PROTECT=1 "
                    "LLAMA_PROMPT_CACHE_MOE_PREWARM=1 LLAMA_PROMPT_CACHE_MOE_PREWARM_K=4");
            all_on_tps = common_moe_bench_candidate_server(
                    self_exe, path_model, sub_mtp_path, best_n, sub_n_max,
                    best_threads, next_port(), ctx, n_predict, concurrency, best_cache_mb, -1,
                    active_ngl, active_min_rank, nullptr, 1234, active_quality_sigma);
            common_moe_calib_set_env(std::string());
            common_moe_calibration_status_candidate_done();
            LOG_INF("%s: every feature on -> %s\n", __func__,
                    all_on_tps > 0 ? string_format("%.2f tok/s", all_on_tps).c_str() : "failed");
            common_moe_calibration_status_note("final validation", "every feature on",
                    all_on_tps > 0 ? string_format("%.2f tok/s", all_on_tps) : std::string("failed"),
                    all_on_tps > 0, false);
        }
        if (all_on_tps > 0 && final_tps > 0 && all_on_tps > final_tps) {
            LOG_WRN("%s: every feature on (%.2f tok/s) beats the combination this run assembled (%.2f) - "
                    "saving the all-on configuration instead. Each stage starts from a feature being off "
                    "and needs evidence to enable it, which on a machine with this much run-to-run spread "
                    "disables things by coin flip; the run is allowed to learn that way but not to ship "
                    "worse than the configuration it started from\n", __func__, all_on_tps, final_tps);
            entry.sched_prefetch_experts = 1;
            entry.moe_cache_ring_pct     = 10;
            entry.admit_after            = 1;
            entry.substitute_atlas       = 4;
            entry.train_predictor        = 1;
            entry.predictor_next_layer   = 1;
            entry.predictor_evict_w      = 1.0;
            entry.predictor_sub_w        = 1.0;
            entry.group_admit            = 1;
            entry.coverage_evict         = 1;
            entry.atlas_prewarm_k        = 4;
            entry.tok_per_sec            = all_on_tps;
            common_moe_calibration_status_note("final validation", "every feature on",
                    string_format("SELECTED - %.2f tok/s, ahead of the assembled combination's %.2f",
                                  all_on_tps, final_tps), true, true);
        }
        common_moe_stage_end();
        if (final_tps > 0) {
            LOG_INF("%s: the winning combination runs: %.2f tok/s\n", __func__, final_tps);
            common_moe_calibration_status_note("final validation", "all calibrated values together",
                    string_format("%.2f tok/s", final_tps), true, true);
            entry.tok_per_sec = final_tps;
        } else {
            LOG_WRN("%s: the winning combination did NOT run cleanly, though every stage that chose its "
                    "values did. Saving it anyway (so the measurements are not lost) but this entry has "
                    "not been shown to serve - re-run calibration or pin the failing knob by hand\n",
                    __func__);
            common_moe_calibration_status_note("final validation", "all calibrated values together",
                    "FAILED - the parts passed, the combination did not", false, true);
        }
    }

    common_moe_calibration_save(path_model, params, entry);

    const char * tps_label = concurrency > 1 ? "aggregate tok/s" : "tok/s";
    const std::string ngl_note = entry.n_gpu_layers >= 0 ?
            string_format(", ngl=%d (some layers moved to CPU to grow the expert cache)", entry.n_gpu_layers) : "";
    if (best_n_max > 0) {
        LOG_INF("%s: calibration complete - ncmoe=%d, n_threads=%d, spec-draft-n-max=%d%s%s, measured %.2f %s. "
                "Cached to %s - launch normally (without --moe-calibrate) to use it.\n",
                __func__, entry.n_cpu_moe, entry.n_threads, entry.spec_n_max,
                concurrency > 1 ? string_format(", concurrency=%d", concurrency).c_str() : "", ngl_note.c_str(),
                entry.tok_per_sec, tps_label, common_moe_calibration_cache_path().c_str());
    } else {
        LOG_INF("%s: calibration complete - ncmoe=%d, n_threads=%d%s%s, measured %.2f %s. "
                "Cached to %s - launch normally (without --moe-calibrate) to use it.\n",
                __func__, entry.n_cpu_moe, entry.n_threads,
                concurrency > 1 ? string_format(", concurrency=%d", concurrency).c_str() : "", ngl_note.c_str(),
                entry.tok_per_sec, tps_label, common_moe_calibration_cache_path().c_str());
    }
}

// Runs as a fallback *after* the general --fit system has already had its
// chance (whether --fit is on or off): --fit's only levers are -c and -ngl,
// so as soon as a user pins both explicitly (a completely ordinary thing to
// do - e.g. -ngl 99 -c 16384) it has nothing left to try and gives up,
// while common_init_result() discards that failure status and attempts the
// real load anyway. This fills the specific gap --fit doesn't cover: partial
// MoE-expert CPU offload while leaving -ngl/-c exactly as the user set them.
// Only steps in when nothing has already decided placement (-ncmoe, -ot,
// -cmoe, or --fit itself already writing real overrides all leave this
// alone) and the config as given would actually fail to fit - this predicts
// the failure via the same no-alloc probe --fit-moe-cache uses, instead of
// catching the crash after a real allocation fails (CUDA allocator state
// after a real failed cudaMalloc isn't something to rely on being cleanly
// retriable). No-op (and no added startup cost beyond one no-alloc probe)
// whenever the config already fits.
//
// Checks the --moe-calibrate cache first (single verification probe, not
// the full binary search) - if a prior calibration run still fits under
// current conditions, its throughput-optimal answer is used directly
// instead of just the memory-safe floor. Falls back to the live binary
// search if there's no cache entry, or if conditions changed enough that
// the cached placement no longer fits.
// MUL_MAT_ID (MoE expert routing) silently drops off the fast MMVQ kernel
// onto a slower general-batch path above a per-(GPU-arch, quant-type)
// batch-size threshold - confirmed empirically this session as a real,
// severe throughput cliff (>2x slower per token, RTX 3060 + Gemma-4-26B-A4B:
// ~12.6 tok/s/sequence at concurrency=8, ~6.0 at concurrency=9, the exact
// boundary of MMVQ_MAX_BATCH_SIZE=8 in ggml/src/ggml-cuda/mmvq.cu), not a
// gentle diminishing return - and for NVIDIA cc>=Ada Lovelace (which
// includes H200/Hopper) the threshold is a flat 8 regardless of quant type.
// This is a GPU-kernel-dispatch limit, not a placement/VRAM question, so
// unlike the rest of Layer 1 it can't be fixed by choosing a different
// -ncmoe - the only real lever is the operator's own --parallel choice, so
// this only ever warns, never silently changes anything.
//
// The real threshold depends on the model's actual expert-tensor quant
// type, which isn't cheaply available before a full model load. Instead of
// guessing one type, this queries every quant type MoE GGUFs commonly ship
// with and reports the minimum across them - a true conservative lower
// bound (the real answer for whatever type this model actually uses can
// only be >= this), so the warning can under-fire slightly early but can
// never miss a real risk by assuming a too-generous type.
//
// Checks every GPU device, not just the first: on a tensor-split multi-GPU
// setup, layers (and therefore MUL_MAT_ID calls) are spread across devices,
// so the full concurrent batch hits whichever device is computing the
// current layer - the cliff can happen on any of them, and the safe ceiling
// for the whole deployment is the minimum across every device, not just
// device 0's. Each device gets its own ggml_backend_dev_t passed straight
// through to the backend's own proc-address function, which recovers that
// backend's real internal device index from it - this file never assumes
// global device index i lines up with the backend's own numbering.
static int common_moe_min_mmvq_max_batch(void) {
    static const ggml_type candidate_types[] = {
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
        GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_XS, GGML_TYPE_MXFP4,
    };

    int min_batch = INT_MAX;
    bool found_any = false;

    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (!reg) {
            continue;
        }
        auto get_max_batch = (int (*)(int, ggml_backend_dev_t)) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_get_mmid_mmvq_max_batch");
        if (!get_max_batch) {
            continue; // this device's backend doesn't expose this (not CUDA) - nothing known for it
        }
        found_any = true;
        for (ggml_type t : candidate_types) {
            min_batch = std::min(min_batch, get_max_batch((int) t, dev));
        }
    }

    return found_any ? min_batch : -1;
}

// MTP/speculative verification batches (n_max+1) candidate positions per
// sequence into a single MUL_MAT_ID call, not just 1. Shared by both the
// MMVQ-cliff warning below and the moe-cache max_batch fix right after it -
// moe-cache's own hint (llama-context.cpp's set_max_batch_hint) does NOT
// apply this multiplier, it passes n_seq_max alone (checked directly in
// the source, not assumed) - a real, separate gap this session found while
// investigating the MMVQ cliff, fixed by common_moe_apply_mtp_aware_max_batch_hint
// below rather than touching llama_context's public API surface.
static int common_moe_verify_width(const common_params & params) {
    return params.speculative.has_dft() ? (params.speculative.draft.n_max + 1) : 1;
}

// Hard safety clamp on concurrent slots while the MoE expert cache is active.
//
// This is a CORRECTNESS guard, not a throughput knob, which is why it clamps
// instead of warning: above the limit the cache produces silently corrupted
// output - the model emits garbage vocabulary tokens (`<unused49>` on
// gemma-4) instead of text - and the session never recovers, so every later
// request on that server is affected too, including sequential ones.
//
// Measured directly, gemma-4-26B-A4B / RTX 3060 / -c 4096, 8 concurrent
// requests x 3 rounds per configuration, counting responses containing
// garbage tokens:
//
//     --parallel 2, cache on        ->  0/24   clean
//     --parallel 3, cache on        -> 21/24   corrupt, and permanent
//     --parallel 4, cache on        -> 20/24   corrupt, and permanent
//     --parallel 8, cache on        -> 24/24   corrupt, and permanent
//     --parallel 4, --moe-cache off ->  0/24   clean
//
// The last row is what makes this the cache's own bug rather than a general
// concurrency problem in the server, and it reproduced with neuron
// subsetting, neuron heat and the atlas all disabled - so it is pre-existing
// rather than anything the heat-aware work introduced. Root cause is not yet
// found; until it is, refusing to run in the configuration that corrupts is
// strictly better than producing wrong output at speed. See docs/plan.md.
//
// GGML_CUDA_MOE_CACHE_MAX_PARALLEL overrides the limit for anyone
// deliberately investigating the bug (set it higher to reproduce), and a
// calibrated entry can raise it for a machine where a higher value has
// actually been verified clean - but the DEFAULT is the measured-safe value,
// because the failure mode is silent.
static void common_enforce_moe_cache_parallel_limit(common_params & params, llama_context_params & cparams) {
    const char * mode = getenv("GGML_CUDA_MOE_CACHE_MODE");
    const char * en   = getenv("GGML_CUDA_MOE_CACHE");
    const bool cache_off = (mode && std::string(mode) == "off") || (en && std::string(en) == "0");
    if (cache_off || params.n_parallel <= 0) {
        return;
    }
    int limit = 2; // measured safe ceiling - see the table above
    if (const char * ov = getenv("GGML_CUDA_MOE_CACHE_MAX_PARALLEL")) {
        const int v = atoi(ov);
        if (v > 0) {
            limit = v;
        }
    }
    if (params.n_parallel <= limit) {
        return;
    }
    LOG_WRN("%s: --parallel %d exceeds the MoE expert cache's measured-safe concurrency limit of %d - "
            "clamping to %d. Above this limit the cache corrupts output silently and permanently "
            "(garbage vocabulary tokens, and the session never recovers). This is a correctness "
            "guard, not a tuning choice. Run with --moe-cache off to use higher concurrency, or set "
            "GGML_CUDA_MOE_CACHE_MAX_PARALLEL to override if you have verified a higher value clean "
            "on this machine.\n",
            __func__, params.n_parallel, limit, limit);
    params.n_parallel = limit;
    // cparams.n_seq_max must move with it, not just the reported slot count.
    // The per-slot context has already been divided by the ORIGINAL slot
    // count by this point, so clamping params alone leaves the server with
    // (say) 2 slots each holding 4096/8 = 512 tokens instead of 4096/2 =
    // 2048 - silently costing three quarters of each slot's context as a
    // side effect of a safety guard, which would be its own bug.
    if (cparams.n_seq_max > (uint32_t) limit) {
        cparams.n_seq_max = (uint32_t) limit;
    }
}

static void common_warn_concurrency_cliff(const common_params & params) {
    // Ignoring the MTP multiplier here would under-warn: even --parallel 1
    // can hit the real cliff once MTP's own verify-width alone exceeds the
    // threshold, so it has to be applied before deciding whether there's
    // anything to check at all, not gated behind an n_parallel>1 fast path.
    const int verify_width    = common_moe_verify_width(params);
    const int effective_batch = params.n_parallel * verify_width;
    if (effective_batch <= 1) {
        return;
    }
    const int max_batch = common_moe_min_mmvq_max_batch();
    if (max_batch > 0 && effective_batch > max_batch) {
        LOG_WRN("%s: --parallel %d%s = effective concurrent batch %d exceeds this GPU's MoE "
                "routing fast-path batch limit (<=%d for at least one common quant type) - "
                "concurrent decode above that limit hits a real GPU-kernel-dispatch throughput "
                "cliff (measured >2x slower per token above the boundary, not a gentle decline). "
                "This is not a moe-cache/placement issue and -ncmoe can't fix it; consider "
                "keeping the effective batch (--parallel%s) at or below %d, or accept the "
                "throughput drop above it.\n",
                __func__, params.n_parallel,
                verify_width > 1 ? string_format(" x MTP verify-width %d", verify_width).c_str() : "",
                effective_batch, max_batch,
                verify_width > 1 ? " x MTP verify-width" : "",
                max_batch);
    }
}

// The shared p_min confidence gate (common_params_speculative_draft::p_min -
// read by draft-simple, EAGLE3, DFlash/DSpark, and MTP; the ngram-family
// types don't use it, they're pattern matchers with no per-token
// confidence) defaults to 0.0f. That default can never actually gate
// anything: every real sampled probability satisfies p >= 0.0, so the
// "only collect very high-confidence draft tokens" early-stop in each of
// those drafters' draft() loops (e.g. `if (cur_p->data[0].p < params.p_min)`)
// is dead code at the default. At n_max's own default of 3 this is bounded
// (at most a couple of low-confidence forward passes wasted, already
// measured this session as immaterial for MTP specifically) - which is
// presumably why it went unnoticed here. But raising n_max on any
// p_min-honoring drafter without also raising p_min removes the only
// mechanism that stops a drafter once its confidence has collapsed, and it
// then burns compute drafting every remaining slot up to n_max regardless,
// tokens verification is going to reject anyway. Reported upstream with
// measured costs at exactly this combination: 2.6x slower at n_max=16,
// 3.7x at n_max=48, 8.6x at n_max=64, for the same output
// (ggml-org/llama.cpp#25908, #26100). Purely diagnostic, mirrors
// common_warn_concurrency_cliff just below - never changes behavior or
// silently picks a value on the operator's behalf.
static void common_warn_p_min_disabled(const common_params & params) {
    if (!params.speculative.has_dft()) {
        return; // no p_min-honoring drafter configured
    }
    if (params.speculative.draft.p_min > 0.0f) {
        return; // gate is live - an explicit choice already made here
    }
    if (params.speculative.draft.n_max <= 3) {
        return; // at or below the validated default width; cost is bounded
    }
    LOG_WRN("%s: --spec-draft-n-max %d with --spec-draft-p-min left at its default (0.0) - "
            "the confidence early-stop can never fire at 0.0 (every real probability is >= 0.0), "
            "so the draft model will draft all %d tokens every round even once its own confidence "
            "has collapsed, burning compute on tokens verification is going to reject anyway. "
            "Measured cost elsewhere at this kind of width: 2.6x-8.6x slower wall time for the same "
            "output (see ggml-org/llama.cpp#25908). Consider setting --spec-draft-p-min explicitly "
            "(e.g. 0.5-0.75) now that n_max is above its default of 3.\n",
            __func__, params.speculative.draft.n_max, params.speculative.draft.n_max);
}

bool common_moe_cache_get_expert_map(std::vector<uint8_t> & out_bytes, int & out_rows, int & out_cols) {
    out_rows = 0;
    out_cols = 0;
    if (!ggml_moe_cache.get_expert_map) {
        return false; // no CUDA backend registered, or built without it
    }
    // First call with a zero-capacity probe to learn the real shape (the
    // provider reports rows/cols even on a "buffer too small" 0 return),
    // then size the real buffer and fetch for real - avoids guessing a
    // grid size up front for a model we haven't measured yet.
    int rows = 0, cols = 0;
    ggml_moe_cache.get_expert_map(nullptr, 0, &rows, &cols);
    if (rows <= 0 || cols <= 0) {
        return false; // nothing cached yet
    }
    out_bytes.assign((size_t) rows * (size_t) cols, 0);
    if (!ggml_moe_cache.get_expert_map(out_bytes.data(), (int) out_bytes.size(), &rows, &cols)) {
        out_bytes.clear();
        return false;
    }
    out_rows = rows;
    out_cols = cols;
    return true;
}

bool common_moe_cache_get_substitute_map(std::vector<uint8_t> & out_bits, int & out_rows, int & out_cols) {
    out_rows = 0;
    out_cols = 0;
    if (!ggml_moe_cache.get_substitute_map) {
        return false;
    }
    int rows = 0, cols = 0;
    ggml_moe_cache.get_substitute_map(nullptr, 0, &rows, &cols);
    if (rows <= 0 || cols <= 0) {
        return false;
    }
    const size_t need = ((size_t) rows * (size_t) cols + 7) / 8;
    out_bits.assign(need, 0);
    if (!ggml_moe_cache.get_substitute_map(out_bits.data(), (int) out_bits.size(), &rows, &cols)) {
        out_bits.clear();
        return false;
    }
    out_rows = rows;
    out_cols = cols;
    return true;
}

// Which (layer,expert) cells belong to the speculative draft rather than the
// target. A live snapshot, unlike the substitute map, which is read-and-cleared.
bool common_moe_cache_get_draft_map(std::vector<uint8_t> & out_bits, int & out_rows, int & out_cols) {
    out_rows = 0;
    out_cols = 0;
    if (!ggml_moe_cache.get_draft_map) {
        return false;
    }
    int rows = 0, cols = 0;
    ggml_moe_cache.get_draft_map(nullptr, 0, &rows, &cols);
    if (rows <= 0 || cols <= 0) {
        return false;
    }
    const size_t need = ((size_t) rows * (size_t) cols + 7) / 8;
    out_bits.assign(need, 0);
    if (!ggml_moe_cache.get_draft_map(out_bits.data(), (int) out_bits.size(), &rows, &cols)) {
        out_bits.clear();
        return false;
    }
    out_rows = rows;
    out_cols = cols;
    return true;
}

bool common_moe_cache_get_neuron_concentration_map(std::vector<float> & out_values, int & out_rows, int & out_cols) {
    out_rows = 0;
    out_cols = 0;
    if (!ggml_moe_cache.get_neuron_concentration_map) {
        return false;
    }
    int rows = 0, cols = 0;
    ggml_moe_cache.get_neuron_concentration_map(nullptr, 0, &rows, &cols);
    if (rows <= 0 || cols <= 0) {
        return false;
    }
    const size_t need = (size_t) rows * (size_t) cols;
    out_values.assign(need, -1.0f);
    if (!ggml_moe_cache.get_neuron_concentration_map(out_values.data(), (int) out_values.size(), &rows, &cols)) {
        out_values.clear();
        return false;
    }
    out_rows = rows;
    out_cols = cols;
    return true;
}

bool common_moe_cache_get_summary(common_moe_cache_summary & out) {
    out = common_moe_cache_summary{};
    if (!ggml_moe_cache.get_summary) {
        return false; // no CUDA backend registered, or built without it
    }
    ggml_moe_cache_summary raw{};
    ggml_moe_cache.get_summary(&raw);
    out.hits             = raw.hits;
    out.misses           = raw.misses;
    out.evictions        = raw.evictions;
    out.fill_failures    = raw.fill_failures;
    out.admission_skips  = raw.admission_skips;
    out.prefetches       = raw.prefetches;
    out.slots_used       = raw.slots_used;
    out.slots_total      = raw.slots_total;
    out.protected_slots  = raw.protected_slots;
    out.avg_heat         = raw.avg_heat;
    out.allocated_bytes  = raw.allocated_bytes;
    out.budget_bytes     = raw.budget_bytes;
    out.substitutions       = raw.substitutions;
    out.substitute_declined = raw.substitute_declined;
    out.rank_hits.assign(raw.rank_hits, raw.rank_hits + GGML_MOE_CACHE_MAX_RANK);
    out.rank_misses.assign(raw.rank_misses, raw.rank_misses + GGML_MOE_CACHE_MAX_RANK);
    out.req_dir_x        = raw.req_dir_x;
    out.req_dir_y        = raw.req_dir_y;
    out.req_dir_valid    = raw.req_dir_valid != 0;
    return raw.slots_total > 0;
}

// Registers one MoE tensor's measured topic-affinity cells (from
// --expert-atlas-file) with moe-cache's step-0 request-direction tracking.
// host_base identifies the tensor the same way moe_cache_key already does;
// the caller (server_context::init, which has both the loaded model and the
// parsed atlas file) is responsible for resolving that pointer and calling
// this once per MoE tensor found for each atlas-covered layer.
bool common_moe_cache_set_atlas(
        const void * host_base, const std::vector<int32_t> & expert,
        const std::vector<float> & x, const std::vector<float> & y, const std::vector<float> & spec,
        const std::vector<float> & dims, int n_dims) {
    if (!ggml_moe_cache.set_atlas || !host_base || expert.empty()) {
        return false;
    }
    if (expert.size() != x.size() || expert.size() != y.size() || expert.size() != spec.size()) {
        return false;
    }
    // Only pass the embedding through if it is exactly the expected shape -
    // a short or ragged array would be read past its end per cell. A mismatch
    // drops to the 2D path rather than failing the registration, since the
    // 2D pair is still valid data.
    const bool dims_ok = n_dims > 0 && dims.size() == expert.size() * (size_t) n_dims;
    ggml_moe_cache.set_atlas(host_base, expert.data(), x.data(), y.data(), spec.data(),
            dims_ok ? dims.data() : nullptr, dims_ok ? n_dims : 0, (int) expert.size());
    return true;
}

std::vector<common_moe_cache_co_activation_entry> common_moe_cache_get_co_activation(
        bool cross_layer, int max_entries) {
    std::vector<common_moe_cache_co_activation_entry> out;
    if (!ggml_moe_cache.get_co_activation || max_entries <= 0) {
        return out;
    }
    std::vector<ggml_moe_cache_co_activation_entry> raw(max_entries);
    const int n = ggml_moe_cache.get_co_activation(cross_layer ? 1 : 0, raw.data(), max_entries);
    out.reserve((size_t) std::max(n, 0));
    for (int i = 0; i < n; i++) {
        out.push_back({raw[i].tensor_from, raw[i].expert_from,
                        raw[i].tensor_to,   raw[i].expert_to, raw[i].count});
    }
    return out;
}

// moe-cache's own admission gate (GGML_CUDA_MOE_CACHE_MAX_BATCH) defaults
// from a live hint set by llama_context (real n_seq_max, floored at 8,
// ceilinged at 64 - see moe-cache.cu's MOE_CACHE_MAX_BATCH_CEILING) - but
// that hint is n_seq_max alone, with no MTP verify-width multiplier applied
// (checked directly at the call site, llama-context.cpp's reserve()).
// Under MTP, the real per-step batch moe-cache actually sees is
// n_seq_max * verify_width, so without this fix the cache gate stays sized
// for the no-MTP case and silently stops helping (falls back to
// uncached CPU compute, not a crash) for any MTP-active deployment whose
// effective batch exceeds whatever n_seq_max alone would have set.
//
// Fixed without touching llama_context's public API: this file already
// knows params.speculative.draft.n_max (the context layer doesn't), so it
// sets GGML_CUDA_MOE_CACHE_MAX_BATCH explicitly when MTP is active,
// reusing the existing "explicit env var always wins over the hint"
// precedence in moe_cache_read_config() rather than adding a new one.
// Never overrides an operator-set value - only steps in for the specific
// case the live hint under-serves.
static void common_moe_apply_mtp_aware_max_batch_hint(const common_params & params) {
    if (getenv("GGML_CUDA_MOE_CACHE_MAX_BATCH")) {
        return; // explicit operator choice - never override it
    }
    const int verify_width = common_moe_verify_width(params);
    if (verify_width <= 1) {
        return; // no MTP - the plain n_seq_max hint already covers this correctly
    }
    // MOE_CACHE_MAX_BATCH_CEILING in moe-cache.cu - duplicated here since
    // that constant is internal to the CUDA backend, not exposed publicly;
    // both need to move together if the real ceiling ever changes.
    // Only ever RAISE the cache's batch limit to cover the verify pass, never lower
    // it. This used to set min(64, n_parallel * verify_width) as an explicit
    // override - 4 for one slot at depth 3 - and an explicit override bypasses the
    // cache's own floor (the op-offload threshold, 32), so turning MTP on HALVED
    // the batch the cache would accept and sent every prompt chunk of 5+ tokens
    // around it. The ceiling is the real row capacity, not the stale 64.
    const int floor     = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? std::max(8, atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH"))) : 32;
    const int needed    = params.n_parallel * verify_width;
    if (needed <= floor) {
        return; // the cache's own default already covers the verify pass
    }
    const int effective = std::min(GGML_MOE_CACHE_MAX_BATCH_ROWS, needed);
    setenv("GGML_CUDA_MOE_CACHE_MAX_BATCH", std::to_string(effective).c_str(), 1);
    LOG_INF("%s: MTP active (spec-draft-n-max=%d) - set GGML_CUDA_MOE_CACHE_MAX_BATCH=%d "
            "(n_parallel=%d x verify-width=%d) so moe-cache's own admission gate matches the "
            "real effective batch size, not just n_seq_max alone\n",
            __func__, params.speculative.draft.n_max, effective, params.n_parallel, verify_width);
}

static bool common_maybe_autoplace_moe_cpu(
        const char * path_model, common_params & params,
        llama_model_params & mparams, const llama_context_params & cparams) {
    for (const auto & o : params.tensor_buft_overrides) {
        if (o.pattern != nullptr) {
            return false;
        }
    }

    common_moe_calibration_entry cached;
    if (common_moe_calibration_lookup(path_model, params, cached)) {
        std::vector<ggml_backend_dev_t> devs;
        uint32_t hp_ngl = 0, hp_n_ctx_train = 0, hp_n_expert = 0;
        // Placement and VRAM sizing are only meaningful at the -ngl they were
        // measured under - how many layers sit on the GPU is precisely what
        // decides how much room is left for experts and cache. The rest of the
        // entry is not layer-residency dependent: thread count is a property
        // of this CPU, the substitution floor is a router-quality boundary
        // (see the ladder's own comment), and neuron-reduce is a magnitude
        // threshold on expert weights. Those stay valid, so a relaxed match
        // applies them rather than dropping the whole entry on the floor and
        // silently reverting every knob to its default.
        // n_cpu_moe < 0 means the entry was written before placement was decided -
        // a checkpoint from a run still in progress. Casting that to uint32_t would
        // ask whether 4294967295 CPU layers fit, so it is rejected explicitly.
        const bool use_placement = cached.ngl_exact && cached.n_cpu_moe >= 0 &&
            common_moe_fits_with_n(path_model, mparams, cparams, (uint32_t) cached.n_cpu_moe,
                                    devs, hp_ngl, hp_n_ctx_train, hp_n_expert);
        if (use_placement || !cached.ngl_exact) {
            if (use_placement) {
                params.tensor_buft_overrides = common_moe_build_cpu_overrides((uint32_t) cached.n_cpu_moe);
                mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
            }
            if (cached.n_threads > 0) {
                params.cpuparams.n_threads = cached.n_threads;
            }
            if (cached.n_threads_batch > 0) {
                params.cpuparams_batch.n_threads = cached.n_threads_batch;
            }
            // Only apply the calibrated n_max if the user is already running
            // with a draft model configured and didn't pin n_max themselves
            // - this never turns MTP on for someone who didn't ask for it,
            // it only tunes n_max for someone who already did.
            if (cached.spec_n_max > 0 && params.speculative.has_dft() &&
                params.speculative.draft.n_max == common_params_speculative_draft::N_MAX_DEFAULT) {
                params.speculative.draft.n_max = cached.spec_n_max;
            }
            // 0 is a measured answer: no draft beat every depth on this machine.
            // Same guard - only when the user left the depth at its default.
            if (cached.spec_n_max == 0 && params.speculative.has_dft() &&
                params.speculative.draft.n_max == common_params_speculative_draft::N_MAX_DEFAULT) {
                LOG_WRN("%s: calibration measured speculative decoding slower than none on this machine - "
                        "serving without it (pass --spec-draft-n-max to override; the draft model is still loaded)\n",
                        __func__);
                params.speculative.types = { COMMON_SPECULATIVE_TYPE_NONE };
            }
            // (the offload threshold is applied once, with the other prompt-processing
            //  settings, in common_moe_apply_prefill_knobs - before the fit pass)
            if (cached.spec_draft_cpu_moe == 1 && params.speculative.has_dft() &&
                params.speculative.draft.tensor_buft_overrides.size() <= 1) {
                // The list must END with the {nullptr, nullptr} sentinel, and arg
                // parsing has already appended one (common/arg.cpp) by the time this
                // runs - so pushing here left a real entry after the sentinel and
                // none at the end. common_model_params_to_llama asserts on exactly
                // that, and the server aborted on startup: "Tensor buffer overrides
                // not terminated with empty pattern". Insert before the sentinel,
                // or add one if this list has none yet.
                auto & ov = params.speculative.draft.tensor_buft_overrides;
                if (!ov.empty() && ov.back().pattern == nullptr) {
                    ov.insert(ov.end() - 1, llm_ffn_exps_cpu_override());
                } else {
                    ov.push_back(llm_ffn_exps_cpu_override());
                    ov.push_back({nullptr, nullptr});
                }
            }
            // Same rule again: only for someone who already has a draft
            // configured, and only when they left acceptance at its default.
            // 0 is a real measured answer ("exact-match won"), so the guard is
            // >= 0, not > 0 - unlike n_max above, where 0 is not a valid depth.
            if (cached.spec_prob_accept >= 0 && params.speculative.has_dft() &&
                !params.speculative.draft.prob_accept) {
                params.speculative.draft.prob_accept = cached.spec_prob_accept != 0;
            }
            // Same rule once more: only for a run that already has a draft model,
            // and only when the user left --spec-type at its default, which parses
            // as the single NONE entry that the launcher then replaces with
            // draft-mtp. An explicit --spec-type is the user's decision.
            // And the confidence gate: only for a run with a draft model that left
            // --spec-draft-p-min at its default of 0.
            if (cached.spec_p_min >= 0.0 && params.speculative.has_dft() &&
                params.speculative.draft.p_min == common_params_speculative_draft::P_MIN_DEFAULT) {
                params.speculative.draft.p_min = (float) cached.spec_p_min;
            }
            if (!cached.spec_types.empty() && params.speculative.has_dft() &&
                params.speculative.types.size() == 1 &&
                params.speculative.types[0] == COMMON_SPECULATIVE_TYPE_NONE) {
                params.speculative.types =
                        common_speculative_types_from_names(string_split<std::string>(cached.spec_types, ','));
            }
            // Same rule as spec_n_max above: only refine --moe-cache's own
            // "auto" intent, never override a size the user explicitly
            // requested. GGML_CUDA_MOE_CACHE_MODE=auto is the marker for
            // that - "off" or an explicit numeric budget means the user
            // already decided and calibration should not second-guess it.
            const char * cache_mode_env = getenv("GGML_CUDA_MOE_CACHE_MODE");
            const bool cache_mode_is_auto = !cache_mode_env || std::string(cache_mode_env) == "auto";
            // A calibrated 0 means calibration measured the cache off as the winner:
            // apply it the way --moe-cache off does, not as "unset" (which left the
            // cache on under auto and silently overrode the measurement).
            if (cached.moe_cache_mb == 0 && cache_mode_is_auto && use_placement) {
#if defined(_WIN32)
                _putenv_s("GGML_CUDA_MOE_CACHE", "0");
                _putenv_s("GGML_CUDA_MOE_CACHE_MODE", "off");
                _putenv_s("GGML_CUDA_MOE_CACHE_BUDGET_MB", "");
#else
                setenv("GGML_CUDA_MOE_CACHE", "0", 1);
                setenv("GGML_CUDA_MOE_CACHE_MODE", "off", 1);
                unsetenv("GGML_CUDA_MOE_CACHE_BUDGET_MB");
#endif
                LOG_WRN("%s: using calibrated expert cache: off (measured as the winner)\n", __func__);
            }
            if (cached.moe_cache_mb > 0 && cache_mode_is_auto && use_placement) {
#if defined(_WIN32)
                _putenv_s("GGML_CUDA_MOE_CACHE", "1");
                _putenv_s("GGML_CUDA_MOE_CACHE_MODE", "on");
                _putenv_s("GGML_CUDA_MOE_CACHE_BUDGET_MB", std::to_string(cached.moe_cache_mb).c_str());
#else
                setenv("GGML_CUDA_MOE_CACHE", "1", 1);
                setenv("GGML_CUDA_MOE_CACHE_MODE", "on", 1);
                setenv("GGML_CUDA_MOE_CACHE_BUDGET_MB", std::to_string(cached.moe_cache_mb).c_str(), 1);
#endif
                params.moe_cache_force = true;
            }
            // Same rule again: only apply calibrated neuron-reduce settings
            // when the user hasn't already set the env var themselves - an
            // explicit GGML_CUDA_MOE_CACHE_NEURON_REDUCE (on or off) is a
            // deliberate choice calibration should not second-guess. This
            // was a real, confirmed gap until now: calibration has recorded
            // these fields since neuron_reduce_k/_budget_mb were added to
            // common_moe_calibration_entry, but nothing ever read them back
            // on a normal launch - an entry with real measured values (e.g.
            // gemma-4's k=256, budget_mb=256) sat unused every time.
            // Same rule again: a calibrated substitution floor is applied only
            // when the user hasn't pinned one themselves. Measured per model
            // because the safe point genuinely differs - it is the boundary
            // between "fast" and "generates word salad", and the degeneracy
            // guard in calibration is what makes recording it trustworthy.
            const bool subst_rank_is_default = !getenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK");
            const bool applied_subst_rank = cached.substitute_min_rank >= 0 && subst_rank_is_default;
            if (applied_subst_rank) {
#if defined(_WIN32)
                _putenv_s("GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK",
                        std::to_string(cached.substitute_min_rank).c_str());
#else
                setenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK",
                        std::to_string(cached.substitute_min_rank).c_str(), 1);
#endif
            }
            if (cached.admit_after >= 0 && !getenv("GGML_CUDA_MOE_CACHE_ADMIT_AFTER")) {
#if defined(_WIN32)
                _putenv_s("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", std::to_string(cached.admit_after).c_str());
#else
                setenv("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", std::to_string(cached.admit_after).c_str(), 1);
#endif
            }
            if (cached.readmit_after >= 0 && !getenv("GGML_CUDA_MOE_CACHE_THROTTLE")) {
#if defined(_WIN32)
                _putenv_s("GGML_CUDA_MOE_CACHE_THROTTLE", std::to_string(cached.readmit_after).c_str());
#else
                setenv("GGML_CUDA_MOE_CACHE_THROTTLE", std::to_string(cached.readmit_after).c_str(), 1);
#endif
            }
            const bool neuron_reduce_is_default = !getenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE");
            const bool applied_neuron_reduce =
                    cached.neuron_reduce_k > 0 && cached.neuron_reduce_budget_mb > 0 && neuron_reduce_is_default;
            if (applied_neuron_reduce) {
#if defined(_WIN32)
                _putenv_s("GGML_CUDA_MOE_CACHE_NEURON_REDUCE", "1");
                _putenv_s("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_K", std::to_string(cached.neuron_reduce_k).c_str());
                _putenv_s("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_BUDGET_MB",
                        std::to_string(cached.neuron_reduce_budget_mb).c_str());
#else
                setenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE", "1", 1);
                setenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_K", std::to_string(cached.neuron_reduce_k).c_str(), 1);
                setenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_BUDGET_MB",
                        std::to_string(cached.neuron_reduce_budget_mb).c_str(), 1);
#endif
            }
            // Same rule again: only apply the calibrated -ngl when the user
            // left it at the default (-1, "auto"/all) - an explicit -ngl is a
            // deliberate choice calibration should never silently override.
            // Safe to apply after the fits-check above without re-verifying:
            // fewer GPU-resident layers only ever *reduces* VRAM demand, so a
            // placement that already fit at the (higher) default -ngl still
            // fits at the calibrated (lower) one.
            const bool applied_ngl = use_placement && cached.n_gpu_layers >= 0 && params.n_gpu_layers == -1;
            if (applied_ngl) {
                params.n_gpu_layers  = cached.n_gpu_layers;
                mparams.n_gpu_layers = cached.n_gpu_layers;
            }
            if (!use_placement) {
                // Relaxed match: the machine-level knobs above are applied,
                // but placement was measured at a different -ngl and does not
                // transfer. Fall through to the live probe to decide it -
                // returning here would leave the model unplaced.
                LOG_WRN("%s: -ngl was pinned to %d, but this machine/model was calibrated at a different "
                        "-ngl; applied the values that do not depend on it (n_threads=%d%s%s%s) and "
                        "placing the experts live - run --moe-calibrate with this -ngl for a measured placement\n",
                        __func__, params.n_gpu_layers, cached.n_threads,
                        cached.spec_n_max > 0 ? string_format(", spec-draft-n-max=%d", cached.spec_n_max).c_str() : "",
                        applied_neuron_reduce ? string_format(", neuron-reduce=k%d/%dMiB", cached.neuron_reduce_k, cached.neuron_reduce_budget_mb).c_str() : "",
                        applied_subst_rank ? string_format(", substitute-min-rank=%d", cached.substitute_min_rank).c_str() : "");
            } else {
            LOG_WRN("%s: using calibrated MoE placement from cache (ncmoe=%d, n_threads=%d%s%s%s%s%s%s, measured %.2f %s on %s) "
                    "- run --moe-calibrate again if hardware/model/context changed\n",
                    __func__, cached.n_cpu_moe, cached.n_threads,
                    cached.spec_n_max > 0 ? string_format(", spec-draft-n-max=%d", cached.spec_n_max).c_str() : "",
                    cached.moe_cache_mb > 0 && cache_mode_is_auto ? string_format(", moe-cache=%dMiB", cached.moe_cache_mb).c_str() : "",
                    applied_ngl ? string_format(", ngl=%d", cached.n_gpu_layers).c_str() : "",
                    applied_neuron_reduce ? string_format(", neuron-reduce=k%d/%dMiB", cached.neuron_reduce_k, cached.neuron_reduce_budget_mb).c_str() : "",
                    applied_subst_rank ? string_format(", substitute-min-rank=%d", cached.substitute_min_rank).c_str() : "",
                    cached.concurrency > 1 ? string_format(", concurrency=%d", cached.concurrency).c_str() : "",
                    cached.tok_per_sec, cached.concurrency > 1 ? "aggregate tok/s" : "tok/s", cached.calibrated_at.c_str());
            return true;
            }
        } else {
            LOG_WRN("%s: cached calibration no longer fits current conditions - recalibrating placement live\n", __func__);
        }
    }

    common_moe_fit_probe_result probe = common_moe_find_safe_layers(path_model, mparams, cparams, common_draft_reserve_device0(params));
    if (probe.already_fits || !probe.is_moe) {
        return false;
    }
    if (!probe.found_safe_n) {
        LOG_WRN("%s: could not find any MoE CPU-offload placement that fits in available device memory; "
                "proceeding with the original configuration, load will likely fail\n", __func__);
        return false;
    }

    LOG_WRN("%s: model does not fit in available device memory as configured; "
            "auto-placed the MoE experts of the first %u layers on CPU to fit "
            "(pass -ncmoe explicitly to override, or run --moe-calibrate once for a throughput-optimal placement)\n",
            __func__, probe.safe_n);
    params.tensor_buft_overrides = common_moe_build_cpu_overrides(probe.safe_n);
    mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
    return true;
}

// Raise MoE CPU offload until the *requested* context fits, before --fit ever
// considers shrinking that context.
//
// Order of operations is the whole point here. common_fit_params() runs first
// and, finding the model doesn't fit, shrinks n_ctx until it does - at whatever
// MoE placement it was handed. common_maybe_autoplace_moe_cpu() then runs and
// asks only "does the model fit?", which is now trivially true (at the reduced
// context), so it does nothing. Each step locally succeeds while together they
// silently trade away context the user explicitly asked for, to preserve a
// placement nobody deliberately chose. Measured on a 12GB card with a 26B MoE:
// `-ncmoe 15 -c 65536` collapsed to 4096 ctx / 1 slot, while `-ncmoe 30` (the
// same model, more experts in CPU RAM) served the full 65536 at 4 slots and
// ~90% of the throughput - the context was never the thing that had to give.
//
// So placement is chosen here, against the requested context, and only if no
// placement can host it does --fit fall back to reducing context as before
// (a genuine hardware limit rather than a self-inflicted one).
//
// -ncmoe acts as a floor rather than an off-switch: an explicit value is never
// lowered, but it is raised when the requested context demands it. This
// deliberately differs from common_maybe_autoplace_moe_cpu()'s "bail if any
// override is set" rule - that early-out means passing -ncmoe silently disables
// the very logic meant to make this decision (the same trap as -ngl disabling
// --fit entirely), which is exactly how the collapse above went unnoticed.
// Prompt-processing settings from a calibration entry. Applied next to the
// quality knobs in common_maybe_raise_moe_for_ctx, which runs BEFORE the fit
// pass - fit sizes compute buffers from n_ubatch and reserves expert-prefetch
// VRAM only when that env var is set, so applying either later is too late.
// Every setting is skipped when the user chose it explicitly.
static void common_moe_apply_prefill_knobs(const common_moe_calibration_entry & cal, common_params & params) {
    auto set_env = [](const char * name, int value) {
#if defined(_WIN32)
        _putenv_s(name, std::to_string(value).c_str());
#else
        setenv(name, std::to_string(value).c_str(), 1);
#endif
    };
    if (cal.op_offload_min_batch > 0 && !getenv("GGML_OP_OFFLOAD_MIN_BATCH")) {
        set_env("GGML_OP_OFFLOAD_MIN_BATCH", cal.op_offload_min_batch);
        LOG_INF("%s: using calibrated MoE offload threshold of %d tokens\n", __func__, cal.op_offload_min_batch);
    }
    // The prediction ring is read when the moe-cache session is created, which is
    // after this - so setting it here reaches it.
    if (cal.moe_cache_ring_pct > 0 && !getenv("GGML_CUDA_MOE_CACHE_RING_PCT")) {
        set_env("GGML_CUDA_MOE_CACHE_RING_PCT", cal.moe_cache_ring_pct);
        LOG_INF("%s: using calibrated prediction ring of %d%% per pool\n", __func__, cal.moe_cache_ring_pct);
    }
    if (cal.sched_prefetch_experts == 1 && !getenv("GGML_SCHED_PREFETCH_EXPERTS")) {
        set_env("GGML_SCHED_PREFETCH_EXPERTS", 1);
        LOG_INF("%s: using calibrated expert prefetch (on)\n", __func__);
    }
    if (cal.n_ubatch > 0 && params.n_ubatch == 512 /* default, see common_params */) {
        params.n_ubatch = cal.n_ubatch;
        params.n_batch  = std::max(params.n_batch, params.n_ubatch);
        LOG_INF("%s: using calibrated prompt micro-batch -ub %d\n", __func__, cal.n_ubatch);
    }
}

// The calibrated quality knobs, applied from exactly one place and on every
// launch. None of them depends on placement, context size, or whether --fit
// ran: the substitution floor is a router-quality boundary and neuron-reduce
// is a magnitude threshold on expert weights.
//
// This has now escaped twice. First they lived inside the placement branch, so
// pinning -ngl dropped them; then they moved up into
// common_maybe_raise_moe_for_ctx, which only runs when --fit is on, so
// `-fit off` (or an explicit -ncmoe placement, which is how one reaches a
// large context on a card this size) dropped them just as silently. The
// failure is invisible from the outside and identical every time: without the
// floor, moe_cache_substitute_min_rank falls back to its pace-driven default
// of 10, no rank_bucket in a top-k router ever reaches it, and the gate never
// runs - substitutions 0, declined 0, with a cache that is otherwise filling
// and hitting normally.
//
// So it is deliberately NOT guarded by anything. Add new calibrated settings
// that do not depend on placement here, not in a caller.
// The trained pre-router, on by default.
//
// It was measured to win its role - eviction protection, with router lookahead
// keeping admission (gemma-4: 71.41 tok/s against 56.59 for admission alone; the
// docs' prerouter section) - and still never served, because only the full
// calibration sweep ever turned it on, and the default (short) calibration
// skips that stage: a gemma entry reads train_predictor = -1, "never decided".
// An unmeasured entry is not an "off" verdict, so with nothing calibrated and
// nothing set by the operator the proven configuration applies: learn the NEXT
// layer's picks (59.2 -> 70.0 against the current layer's), weights in the
// per-model state file, loaded at every start and kept learning. A calibrated
// 0 (measured off) or an explicit env setting is left alone, and so is
// --moe-cache off.
static void common_moe_default_prerouter(const char * path_model) {
    const char * cache_env = getenv("GGML_CUDA_MOE_CACHE");
    if (cache_env && strcmp(cache_env, "0") == 0) {
        return;
    }
    if (getenv("LLAMA_MOE_CALIBRATION_CHILD")) {
        return; // a calibration candidate: its stages set this knob themselves, from a neutral baseline
    }
    if (getenv("GGML_CUDA_MOE_CACHE_TRAIN_PREDICTOR")) {
        return; // the operator's choice, or a calibrated one, applied above
    }
    auto set_env = [](const char * name, const std::string & v) {
#if defined(_WIN32)
        _putenv_s(name, v.c_str());
#else
        setenv(name, v.c_str(), 1);
#endif
    };
    set_env("GGML_CUDA_MOE_CACHE_TRAIN_PREDICTOR", "1");
    if (!getenv("GGML_CUDA_MOE_CACHE_TRAIN_NEXT_LAYER")) {
        set_env("GGML_CUDA_MOE_CACHE_TRAIN_NEXT_LAYER", "1");
    }
    if (!getenv("GGML_CUDA_MOE_CACHE_TRAIN_STATE_FILE")) {
        const std::string base = path_model ? std::string(path_model) : std::string("model");
        const size_t slash = base.find_last_of("/\\");
        const std::string state = fs_get_cache_directory() + "predictor-" +
            (slash == std::string::npos ? base : base.substr(slash + 1)) + ".bin";
        set_env("GGML_CUDA_MOE_CACHE_TRAIN_STATE_FILE", state);
        LOG_INF("%s: pre-router on by default (eviction protection, next-layer target); weights: %s\n",
                __func__, state.c_str());
    }
}

static void common_moe_apply_calibrated_quality(const char * path_model, common_params & params) {
    common_moe_calibration_entry cal_q;
    if (common_moe_calibration_lookup(path_model, params, cal_q)) {
        common_moe_apply_quality_knobs(cal_q, path_model);
        common_moe_apply_prefill_knobs(cal_q, params);
    }
    common_moe_default_prerouter(path_model);
}

static bool common_maybe_raise_moe_for_ctx(
        const char * path_model, common_params & params,
        llama_model_params & mparams, const llama_context_params & cparams) {
    // Only meaningful for an explicit request. n_ctx == 0 ("auto") means the
    // user expressed no preference, so there is no context to protect and the
    // existing post-fit autoplace path already handles it.
    //
    // The quality knobs above are deliberately applied BEFORE this returns.
    // They do not depend on the context at all - the substitution floor is a
    // router-quality boundary and neuron-reduce is a magnitude threshold - and
    // leaving them behind this guard meant a launch with no -c at all silently
    // discarded every one of them. That is exactly the launch this project
    // tells people to use: measured on a bare `llama-server -m model.gguf`,
    // substitutions were 0 and declined 0, i.e. the gate never even ran, while
    // the same binary with `-c 4096` applied the floor and used it. The
    // fewer flags you passed, the more of the calibration was thrown away.
    if (params.n_ctx == 0) {
        return false;
    }

    // Any non-MoE-block override pattern means the user hand-placed tensors
    // with -ot; rebuilding the override list from a layer count would discard
    // that. Only the uniform "first N layers" shape -ncmoe produces is safe to
    // extend, so bail on anything else.
    uint32_t current_n = 0;
    for (const auto & o : params.tensor_buft_overrides) {
        if (o.pattern == nullptr) {
            continue;
        }
        if (o.pattern != llm_ffn_exps_block_regex((int) current_n)) {
            return false;
        }
        current_n++;
    }

    // Must match what common_fit_params() will demand of this same config, or
    // placement approves a layout fit then rejects - see fit.cpp's step0_margin,
    // which requires 3x the configured per-device target to cover costs its
    // no-alloc probe can't see (real weight loading, lazy CUDA graph capture).
    // Judging placement by a bare fit here is what let a 65536-token request
    // get "fixed" by a single extra offloaded layer and still collapse to 4096.
    // A calibrated fit margin, if one was measured for this exact
    // GPU+model+context, replaces the default before the margin is computed
    // - it has to be applied HERE rather than alongside the placement
    // decision below, because this margin is precisely what determines how
    // far -ncmoe gets raised. Applying it afterwards would compute the
    // placement against the untuned default and then record a margin that
    // never influenced anything. Only ever tightens: a calibrated value
    // above the default is ignored, on the same "calibration is a floor,
    // never a licence to undershoot safety" principle the placement lookup
    // below already follows.
    {
        common_moe_calibration_entry cal_fit;
        if (common_moe_calibration_lookup(path_model, params, cal_fit) && cal_fit.fit_target_mb > 0 &&
            cal_fit.ngl_exact) {
            const size_t want = (size_t) cal_fit.fit_target_mb * 1024 * 1024;
            if (want < params.fit_params_target[0]) {
                LOG_WRN("%s: using calibrated fit margin of %d MiB per device (default %zu MiB) - "
                        "measured %.2f tok/s on %s\n", __func__, cal_fit.fit_target_mb,
                        params.fit_params_target[0] / (1024 * 1024), cal_fit.tok_per_sec,
                        cal_fit.calibrated_at.c_str());
                std::fill(params.fit_params_target.begin(), params.fit_params_target.end(), want);
            }
        }
    }
    // The draft is a real allocation on top of the margin: added once, not tripled with it (as in calibration).
    const int64_t margin = 3 * (int64_t) params.fit_params_target[0] + common_draft_reserve_device0(params);

    // Prefer a calibrated placement when one exists. The search below answers
    // "what is the least offload that makes this context fit?", which is a
    // sufficiency question, not a throughput one - and the two have different
    // answers. Every layer left resident holds weights competing with the expert
    // cache for the same VRAM, and cached experts serve the offloaded layers, so
    // trading resident weights for cache capacity keeps paying well past the
    // point where the context merely fits. Measured on Nemotron 3.5 (53 layers):
    // the fit search stopped at 46 for 1239 cache slots and a 57.1% hit rate,
    // while offloading all 53 gave 2307 slots and 68.3%.
    //
    // --moe-calibrate already measures the throughput optimum empirically. This
    // only consults what it recorded; it does not guess past "fits" on its own,
    // because the right number is hardware- and model-specific and the honest
    // way to find it is to measure it.
    {
        common_moe_calibration_entry cal;
        if (common_moe_calibration_lookup(path_model, params, cal) && cal.n_cpu_moe > 0 && cal.ngl_exact) {
            std::vector<ggml_backend_dev_t> cdevs;
            uint32_t c_ngl = 0, c_nct = 0, c_nex = 0;
            // Keep the full fit margin here. Dropping it to 0 -- on the
            // reasoning that a calibrated entry is a measurement rather than a
            // prediction, so it only needs to still fit -- was tried and
            // reproduced the 4k collapse outright: ncmoe=40 passed a zero-margin
            // probe, then real weight loading plus lazy CUDA graph capture
            // overran VRAM and the downstream search clamped the server to
            // n_ctx=4096 with a single slot. The margin covers allocation that
            // happens *after* the probe, so a measurement taken elsewhere does
            // not excuse skipping it.
            //
            // That makes calibration a floor, never a licence to undershoot: it
            // can only move placement to be *more* CPU-offloaded than the safe
            // minimum (which is where it pays -- freed VRAM turns into expert
            // cache hit rate). When it recommends less, the fit search's more
            // conservative answer wins.
            if (common_moe_fits_with_n(path_model, mparams, cparams, (uint32_t) cal.n_cpu_moe,
                                       cdevs, c_ngl, c_nct, c_nex, margin)) {
                LOG_WRN("%s: using calibrated placement of %d CPU layer(s) (measured %.2f tok/s on %s) "
                        "rather than the %s that merely fits\n",
                        __func__, cal.n_cpu_moe, cal.tok_per_sec, cal.calibrated_at.c_str(),
                        "minimum");
                // Same expert-cache-size consultation as common_maybe_autoplace_moe_cpu -
                // this is the placement path that actually runs when the user asks
                // for a context and lets -ncmoe auto-raise to fit it (--moe-cache
                // auto, no explicit -ncmoe), which is most real launches. Only
                // refines --moe-cache's own "auto" intent, same guard as there.
                {
                    const char * cache_mode_env = getenv("GGML_CUDA_MOE_CACHE_MODE");
                    const bool cache_mode_is_auto = !cache_mode_env || std::string(cache_mode_env) == "auto";
                    if (cal.moe_cache_mb == 0 && cache_mode_is_auto) {
#if defined(_WIN32)
                        _putenv_s("GGML_CUDA_MOE_CACHE", "0");
                        _putenv_s("GGML_CUDA_MOE_CACHE_MODE", "off");
                        _putenv_s("GGML_CUDA_MOE_CACHE_BUDGET_MB", "");
#else
                        setenv("GGML_CUDA_MOE_CACHE", "0", 1);
                        setenv("GGML_CUDA_MOE_CACHE_MODE", "off", 1);
                        unsetenv("GGML_CUDA_MOE_CACHE_BUDGET_MB");
#endif
                        LOG_WRN("%s: using calibrated expert cache: off (measured as the winner)\n", __func__);
                    }
                    if (cal.moe_cache_mb > 0 && cache_mode_is_auto) {
#if defined(_WIN32)
                        _putenv_s("GGML_CUDA_MOE_CACHE", "1");
                        _putenv_s("GGML_CUDA_MOE_CACHE_MODE", "on");
                        _putenv_s("GGML_CUDA_MOE_CACHE_BUDGET_MB", std::to_string(cal.moe_cache_mb).c_str());
#else
                        setenv("GGML_CUDA_MOE_CACHE", "1", 1);
                        setenv("GGML_CUDA_MOE_CACHE_MODE", "on", 1);
                        setenv("GGML_CUDA_MOE_CACHE_BUDGET_MB", std::to_string(cal.moe_cache_mb).c_str(), 1);
#endif
                        params.moe_cache_force = true;
                        LOG_WRN("%s: using calibrated expert-cache size of %d MiB\n", __func__, cal.moe_cache_mb);
                    }
                }
                // Substitution floor and neuron-reduce, for the same reason the
                // cache size is applied here: this is the path most real
                // launches take (context auto-raises -ncmoe, no explicit flag),
                // and it returns before common_maybe_autoplace_moe_cpu ever
                // runs - so settings applied only there were measured, cached,
                // and then silently ignored at serving. Caught live: a run
                // measured substitute_min_rank=2 at 1.64 tok/s, cached it, and
                // then served 0.66 tok/s with substitutions=0 because this
                // branch never applied it. Same "only fill in what the user
                // left at default" rule as everywhere else.
                params.tensor_buft_overrides  = common_moe_build_cpu_overrides((uint32_t) cal.n_cpu_moe);
                mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
                // c_ngl was filled by the fit probe above with the model's real
                // layer count - report against that, not the override count.
                if (c_ngl > 0) {
                    params.placed_n_layer = (int32_t) c_ngl;
                }
                const int32_t n_layer_cal = c_ngl > 0 ? (int32_t) c_ngl : cal.n_cpu_moe;
                params.placed_n_cpu_moe_req   = std::min((int32_t) current_n, n_layer_cal);
                params.placed_n_cpu_moe_final = std::min(cal.n_cpu_moe, n_layer_cal);
                params.placed_n_ctx_req       = params.n_ctx;
                return true;
            }
            LOG_WRN("%s: calibrated placement of %d CPU layer(s) is below the safe minimum for this "
                    "context - keeping the fit search's more conservative answer\n",
                    __func__, cal.n_cpu_moe);
        }
    }

    const common_moe_fit_probe_result probe =
        common_moe_find_safe_layers(path_model, mparams, cparams, margin);

    // Record what was asked for regardless of the outcome below, so the Brain
    // view can distinguish "23 layers because you asked for it" from "23 layers
    // because 65536 context demanded it".
    // Clamp what gets reported to the layers that actually exist. `-ncmoe 99`
    // on a 53-layer model is a legitimate "offload everything" idiom - the extra
    // patterns simply match nothing - but reporting 99 CPU layers to /props (and
    // from there to the model-info dialog and the Brain placement view) states
    // something untrue about the model.
    const int32_t n_layer_real = probe.n_layer > 0 ? (int32_t) probe.n_layer : (int32_t) current_n;
    params.placed_n_cpu_moe_req   = std::min((int32_t) current_n, n_layer_real);
    params.placed_n_cpu_moe_final = std::min((int32_t) current_n, n_layer_real);
    params.placed_n_ctx_req       = params.n_ctx;
    if (probe.n_layer > 0) {
        params.placed_n_layer = (int32_t) probe.n_layer;
    }

    if (!probe.is_moe || probe.already_fits) {
        return false; // dense model, or the requested context already fits as configured
    }
    if (!probe.found_safe_n || probe.safe_n <= current_n) {
        // Nothing fits even with every expert offloaded, or more offload than
        // already configured wouldn't help - a real hardware ceiling. Leave it
        // to --fit to reduce context, which is the correct last resort.
        return false;
    }

    LOG_WRN("%s: requested context of %d does not fit with the MoE experts of %u layer(s) on CPU - "
            "raised to %u layer(s) so the requested context fits, instead of reducing the context to fit "
            "the placement (pass -fit off to disable this)\n",
            __func__, params.n_ctx, current_n, probe.safe_n);

    params.tensor_buft_overrides  = common_moe_build_cpu_overrides(probe.safe_n);
    mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
    params.placed_n_cpu_moe_final = std::min((int32_t) probe.safe_n, n_layer_real);
    return true;
}

common_init_result::common_init_result(common_params & params, bool model_only) :
    pimpl(new impl{}) {
    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    // Unconditional, and before the --fit branch below: these do not depend on
    // fit, placement or context, and every previous home for them turned out to
    // be behind a guard that some ordinary launch failed to satisfy.
    common_moe_apply_calibrated_quality(params.model.path.c_str(), params);

    if (params.fit_params) {
        // must run before common_fit_params() - see the function's own comment
        common_maybe_raise_moe_for_ctx(params.model.path.c_str(), params, mparams, cparams);
        // cparams was built above, before the calibration entry could set the
        // prompt micro-batch - carry it over before fit sizes compute buffers from it.
        cparams.n_ubatch = params.n_ubatch;
        cparams.n_batch  = params.n_batch;

        COM_TRC("%s", "fitting params to device memory ...\n");
        COM_TRC("%s", "(for bugs during this step try to reproduce them with -fit off, or provide --verbose logs if the bug only occurs with -fit on)\n");
        common_fit_params(params.model.path.c_str(), &mparams, &cparams,
            params.tensor_split,
            params.tensor_buft_overrides.data(),
            params.fit_params_target.data(),
            params.fit_params_min_ctx,
            params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
    }

    common_maybe_autoplace_moe_cpu(params.model.path.c_str(), params, mparams, cparams);

    // mparams.n_gpu_layers is otherwise a local copy that gets discarded once the real model/context
    // are built below - write the resolved value back so callers (e.g. the server's /props endpoint)
    // can report what --fit actually decided, not just the -1/"auto" the user passed in. Mirrors the
    // same write-back already done for tensor_buft_overrides above.
    params.n_gpu_layers = mparams.n_gpu_layers;

    // cparams.n_seq_max can also be reduced by --fit (see the context-vs-concurrency priority logic
    // in fit.cpp: the requested context size is kept fixed and concurrent slots give way to it, not
    // the other way around). Writing it back to params.n_parallel is not just for reporting purposes
    // this time - the server creates one slot object per params.n_parallel (server-context.cpp) after
    // this constructor returns, independent of the llama_context that was actually built. Without this
    // write-back the server would create more slots than the context's real n_seq_max supports, a real
    // mismatch under concurrent requests, not just a stale number in a dialog.
    if (cparams.n_seq_max != (uint32_t) params.n_parallel && params.n_parallel > 0) {
        LOG_WRN("%s: --fit reduced concurrent slots from %d to %u to keep the requested context size\n",
                __func__, params.n_parallel, cparams.n_seq_max);
    }
    params.n_parallel = (int32_t) cparams.n_seq_max;

    // Both run after autoplace, not before: the calibration-cache lookup
    // inside it can override params.speculative.draft.n_max to the real
    // calibrated value, and both of these need to see that final value, not
    // the pre-lookup default - otherwise an MTP setup with a cached n_max
    // would be checked/sized against the wrong (understated) effective
    // batch size.
    common_warn_concurrency_cliff(params);
    common_enforce_moe_cache_parallel_limit(params, cparams);
    common_moe_apply_mtp_aware_max_batch_hint(params);
    common_warn_p_min_disabled(params);

    // ...and so does the output reservation. A caller that pre-sizes
    // n_outputs_max (the server does, in load_model) computes it from the
    // speculative width it can see at that point, which is still the default
    // - the calibrated n_max only lands in the autoplace call above. Every
    // draft token needs an output slot, so a cached n_max larger than the
    // default leaves the reservation exactly (n_max_cal - n_max_default)
    // short and the first decode trips
    // GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max) in llama-context.
    // Re-derive against the final width here, where it is known. Grow only:
    // a caller that deliberately reserved more keeps its own number.
    if (params.n_outputs_max > 0) {
        const auto lim = common_speculative_get_output_limits(
                params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
        const int32_t total   = std::max(params.n_outputs_max,         std::max(1, lim.total));
        const int32_t per_seq = std::max(params.n_outputs_max_per_seq, std::max(1, lim.per_seq));
        if (total != params.n_outputs_max || per_seq != params.n_outputs_max_per_seq) {
            LOG_WRN("%s: raised the output reservation to %d (%d per sequence) to cover the "
                    "calibrated speculative width of %d\n",
                    __func__, total, per_seq, common_speculative_n_max(&params.speculative));
        }
        params.n_outputs_max         = total;
        params.n_outputs_max_per_seq = per_seq;
        cparams.n_outputs_max         = (uint32_t) total;
        cparams.n_outputs_max_per_seq = (uint32_t) per_seq;
    }

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == NULL) {
        return;
    }

    pimpl->model.reset(model);

    if (model_only) {
        return;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // load and optionally apply lora adapters
    for (auto & la : params.lora_adapters) {
        llama_adapter_lora_ptr lora;
        lora.reset(llama_adapter_lora_init(model, la.path.c_str()));
        if (lora == nullptr) {
            COM_ERR("failed to load lora adapter '%s'\n", la.path.c_str());
            return;
        }

        char buf[1024];
        la.ptr = lora.get();
        llama_adapter_meta_val_str(la.ptr, "adapter.lora.task_name", buf, sizeof(buf));
        la.task_name = buf;
        llama_adapter_meta_val_str(la.ptr, "adapter.lora.prompt_prefix", buf, sizeof(buf));
        la.prompt_prefix = buf;
        pimpl->lora.emplace_back(std::move(lora)); // copy to list of loaded adapters
    }

    // updates params.sampling
    // TODO: fix naming
    common_init_sampler_from_model(model, params.sampling);

    if (params.sampling.ignore_eos && llama_vocab_eos(vocab) == LLAMA_TOKEN_NULL) {
        COM_WRN("%s", "vocab does not have an EOS token, ignoring --ignore-eos\n");
        params.sampling.ignore_eos = false;
    }

    // initialize once
    for (llama_token i = 0; i < llama_vocab_n_tokens(vocab); i++) {
        if (llama_vocab_is_eog(vocab, i)) {
            COM_TRC("added %s logit bias = %f\n", common_token_to_piece(vocab, i).c_str(), -INFINITY);
            params.sampling.logit_bias_eog.push_back({i, -INFINITY});
        }
    }

    if (params.sampling.ignore_eos) {
        // add EOG biases to the active set of logit biases
        params.sampling.logit_bias.insert(
                params.sampling.logit_bias.end(),
                params.sampling.logit_bias_eog.begin(), params.sampling.logit_bias_eog.end());
    }

    // init the backend samplers as part of the context creation
    pimpl->samplers.resize(cparams.n_seq_max);
    pimpl->samplers_seq_config.resize(cparams.n_seq_max);

    for (int i = 0; i < (int) cparams.n_seq_max; ++i) {
        pimpl->samplers[i].reset(common_sampler_init(model, params.sampling));
        pimpl->samplers_seq_config[i] = { i, common_sampler_get(pimpl->samplers[i].get()) };
    }

    if (params.sampling.backend_sampling) {
        cparams.samplers   = pimpl->samplers_seq_config.data();
        cparams.n_samplers = pimpl->samplers_seq_config.size();
    }

    llama_context * lctx = llama_init_from_model(model, cparams);
    if (lctx == NULL) {
        COM_ERR("failed to create context with model '%s'\n", params.model.path.c_str());
        return;
    }

    pimpl->context.reset(lctx);
}

llama_model * common_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_init_result::context() {
    return pimpl->context.get();
}

common_sampler * common_init_result::sampler(llama_seq_id seq_id) {
    if (seq_id < 0 || seq_id >= (int) pimpl->samplers.size()) {
        return nullptr;
    }
    return pimpl->samplers[seq_id].get();
}

void common_init_result::reset_samplers() {
    for (int i = 0; i < (int) pimpl->samplers.size(); ++i) {
        llama_sampler_reset(common_sampler_get(pimpl->samplers[i].get()));
    }
}

std::vector<llama_adapter_lora_ptr> & common_init_result::lora() {
    return pimpl->lora;
}

common_init_result_ptr common_init_from_params(common_params & params, bool model_only) {
    common_init_result_ptr res(new common_init_result(params, model_only));

    llama_model * model = res->model();
    if (model == NULL) {
        COM_ERR("failed to load model '%s'\n", params.model.path.c_str());
        return res;
    }

    if (model_only) {
        return res;
    }

    llama_context * lctx = res->context();
    if (lctx == NULL) {
        COM_ERR("failed to create context with model '%s'\n", params.model.path.c_str());
        return res;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (params.ctx_shift && !llama_memory_can_shift(llama_get_memory(lctx))) {
        COM_WRN("%s", "KV cache shifting is not supported for this context, disabling KV cache shifting\n");
        params.ctx_shift = false;
    }

    if (!params.control_vectors.empty()) {
        if (params.control_vector_layer_start <= 0) params.control_vector_layer_start = 1;
        if (params.control_vector_layer_end   <= 0) params.control_vector_layer_end   = llama_model_n_layer(model);

        const auto cvec = common_control_vector_load(params.control_vectors);
        if (cvec.n_embd == -1) {
            return res;
        }

        int err = llama_set_adapter_cvec(
                lctx,
                cvec.data.data(),
                cvec.data.size(),
                cvec.n_embd,
                params.control_vector_layer_start,
                params.control_vector_layer_end);
        if (err) {
            return res;
        }
    }

    if (llama_pooling_type(lctx) == LLAMA_POOLING_TYPE_RANK) {
        bool ok = true;

        if (llama_vocab_bos(vocab) == LLAMA_TOKEN_NULL) {
            COM_WRN("%s", "vocab does not have a  BOS token, reranking will not work\n");
            ok = false;
        }

        bool has_eos = llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL;
        bool has_sep = llama_vocab_sep(vocab) != LLAMA_TOKEN_NULL;
        bool has_rerank_prompt = llama_model_chat_template(model, "rerank") != NULL;

        if (!has_eos && !has_sep && !has_rerank_prompt) {
            COM_WRN("%s", "vocab does not have an EOS token, SEP token, or rerank prompt. Reranking will not work\n");
            ok = false;
        } else if (!has_eos) {
            COM_WRN("%s", "vocab does not have an EOS token, using SEP token as fallback\n");
        }

        if (!ok) {
            return res;
        }
    }

    if (!params.lora_init_without_apply) {
        common_set_adapter_lora(lctx, params.lora_adapters);
    }

    if (params.warmup) {
        COM_TRC("%s", "warming up the model with an empty run - please wait ... (--no-warmup to disable)\n");

        std::vector<llama_token> tmp;
        llama_token bos = llama_vocab_bos(vocab);
        llama_token eos = llama_vocab_eos(vocab);

        // some models (e.g. T5) don't have a BOS token
        if (bos != LLAMA_TOKEN_NULL) {
            tmp.push_back(bos);
        }
        if (eos != LLAMA_TOKEN_NULL) {
            tmp.push_back(eos);
        }
        if (tmp.empty()) {
            tmp.push_back(0);
        }

        if (llama_model_has_encoder(model)) {
            llama_encode(lctx, llama_batch_get_one(tmp.data(), tmp.size()));
            llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
            if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
                decoder_start_token_id = bos;
            }
            tmp.clear();
            tmp.push_back(decoder_start_token_id);
        }
        if (llama_model_has_decoder(model)) {
            llama_decode(lctx, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) params.n_batch)));
        }
        llama_memory_clear(llama_get_memory(lctx), true);
        llama_synchronize(lctx);
        llama_perf_context_reset(lctx);

        // reset samplers to reset RNG state after warmup to the seeded state
        res->reset_samplers();
    }

    return res;
}

common_init_result::~common_init_result() = default;

std::string common_get_model_endpoint() {
    std::string endpoint = common_get_env("MODEL_ENDPOINT");
    if (endpoint.empty()) {
        // the HF_ENDPOINT variable is respected for backward compatibility
        endpoint = common_get_env("HF_ENDPOINT");
    }
    if (endpoint.empty()) {
        return "https://huggingface.co/";
    }
    if (endpoint.back() != '/') {
        endpoint += '/';
    }
    return endpoint;
}

char * common_get_model_or_exit(int argc, char * argv[]) {
    if (argc > 1) {
        return argv[1];
    }

    char * path = getenv("LLAMACPP_TEST_MODELFILE");
    if (!path || strlen(path) == 0) {
        fprintf(stderr, "\033[33mWARNING: No model file provided. Skipping this test. Set LLAMACPP_TEST_MODELFILE=<gguf_model_path> to silence this warning and run this test.\n\033[0m");
        exit(EXIT_SUCCESS);
    }

    return path;
}

common_context_seq_rm_type common_context_can_seq_rm(llama_context * ctx) {
    auto * mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        return COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    }

    common_context_seq_rm_type res = COMMON_CONTEXT_SEQ_RM_TYPE_PART;

    llama_memory_clear(mem, true);

    // eval 2 tokens to check if the context is compatible
    std::vector<llama_token> tmp;
    tmp.push_back(0);
    tmp.push_back(0);

    int ret = llama_decode(ctx, llama_batch_get_one(tmp.data(), tmp.size()));
    if (ret != 0) {
        COM_ERR("llama_decode() failed: %d\n", ret);
        res = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
        goto done;
    }

    if (llama_n_rs_seq(ctx) > 0) {
        COM_TRC("%s", "the context supports bounded partial sequence removal\n");
        res = COMMON_CONTEXT_SEQ_RM_TYPE_RS;
        goto done;
    }

    // try to remove the last tokens
    if (!llama_memory_seq_rm(mem, 0, 1, -1)) {
        COM_TRC("%s", "the context does not support partial sequence removal\n");
        res = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        goto done;
    }

done:
    llama_memory_clear(mem, true);
    llama_synchronize(ctx);

    return res;
}

static void common_context_seq_rm(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    auto * mem = llama_get_memory(ctx);
    if (!llama_memory_seq_rm(mem, seq_id, p0, p1)) {
        GGML_ABORT("%s", string_format("failed to remove sequence %d with p0=%d, p1=%d\n", seq_id, p0, p1).c_str());
    }
}

static void common_context_seq_cp(llama_context * ctx, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    auto * mem = llama_get_memory(ctx);
    llama_memory_seq_cp(mem, seq_id_src, seq_id_dst, p0, p1);
}

static void common_context_seq_add(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    auto * mem = llama_get_memory(ctx);
    llama_memory_seq_add(mem, seq_id, p0, p1, delta);
}

void common_memory::init(llama_context * ctx_tgt, llama_context * ctx_dft) {
    this->ctx_tgt = ctx_tgt;
    this->ctx_dft = ctx_dft;
}

void common_memory::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    common_context_seq_rm(ctx_tgt, seq_id, p0, p1);
    if (ctx_dft) {
        common_context_seq_rm(ctx_dft, seq_id, p0, p1);
    }
}

void common_memory::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) const {
    common_context_seq_cp(ctx_tgt, seq_id_src, seq_id_dst, p0, p1);
    if (ctx_dft) {
        common_context_seq_cp(ctx_dft, seq_id_src, seq_id_dst, p0, p1);
    }
}

void common_memory::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) const {
    common_context_seq_add(ctx_tgt, seq_id, p0, p1, delta);
    if (ctx_dft) {
        common_context_seq_add(ctx_dft, seq_id, p0, p1, delta);
    }
}

void common_set_adapter_lora(struct llama_context * ctx, std::vector<common_adapter_lora_info> & lora) {
    std::vector<llama_adapter_lora *> loras;
    std::vector<float> scales;

    for (auto & la: lora) {
        loras.push_back(la.ptr);
        scales.push_back(la.scale);
    }

    llama_set_adapters_lora(ctx, loras.data(), loras.size(), scales.data());
}

struct llama_model_params common_model_params_to_llama(common_params & params) {
    auto mparams = llama_model_default_params();

    if (!params.devices.empty()) {
        mparams.devices = params.devices.data();
    }

    mparams.n_gpu_layers    = params.n_gpu_layers;
    mparams.main_gpu        = params.main_gpu;
    mparams.split_mode      = params.split_mode;
    mparams.load_mode       = params.load_mode;
    mparams.lazy_mode = params.lazy_mode;
    mparams.tensor_split    = params.tensor_split;
    mparams.check_tensors   = params.check_tensors;
    mparams.use_extra_bufts = !params.no_extra_bufts;
    mparams.no_host         = params.no_host;

    if (params.kv_overrides.empty()) {
        mparams.kv_overrides = NULL;
    } else {
        GGML_ASSERT(params.kv_overrides.back().key[0] == 0 && "KV overrides not terminated with empty key");
        mparams.kv_overrides = params.kv_overrides.data();
    }

    if (params.tensor_buft_overrides.empty()) {
        mparams.tensor_buft_overrides = NULL;
    } else {
        GGML_ASSERT(params.tensor_buft_overrides.back().pattern == nullptr && "Tensor buffer overrides not terminated with empty pattern");
        mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
    }

    mparams.progress_callback           = params.load_progress_callback;
    mparams.progress_callback_user_data = params.load_progress_callback_user_data;
    mparams.no_alloc                    = params.no_alloc;
    mparams.load_mtp                    = std::find(params.speculative.types.begin(), params.speculative.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

    return mparams;
}

struct llama_context_params common_context_params_to_llama(const common_params & params) {
    auto cparams = llama_context_default_params();

    cparams.n_ctx             = params.n_ctx;
    cparams.n_seq_max         = params.n_parallel;
    cparams.n_rs_seq          = params.speculative.need_n_rs_seq();
    cparams.n_outputs_max     = std::max(params.n_outputs_max, 0);
    cparams.n_outputs_max_per_seq = std::max(params.n_outputs_max_per_seq, 0);
    cparams.n_batch           = params.n_batch;
    cparams.n_ubatch          = params.n_ubatch;
    cparams.n_threads         = params.cpuparams.n_threads;
    cparams.n_threads_batch   = params.cpuparams_batch.n_threads == -1 ?
                                params.cpuparams.n_threads : params.cpuparams_batch.n_threads;
    cparams.embeddings        = params.embedding;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.rope_freq_base    = params.rope_freq_base;
    cparams.rope_freq_scale   = params.rope_freq_scale;
    cparams.yarn_ext_factor   = params.yarn_ext_factor;
    cparams.yarn_attn_factor  = params.yarn_attn_factor;
    cparams.yarn_beta_fast    = params.yarn_beta_fast;
    cparams.yarn_beta_slow    = params.yarn_beta_slow;
    cparams.yarn_orig_ctx     = params.yarn_orig_ctx;
    cparams.pooling_type      = params.pooling_type;
    cparams.attention_type    = params.attention_type;
    cparams.flash_attn_type   = params.flash_attn_type;
    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;
    cparams.offload_kqv       = !params.no_kv_offload;
    cparams.kv_stream         = params.kv_stream;
    cparams.kv_resident       = params.kv_resident;
    cparams.no_perf           = params.no_perf;
    cparams.op_offload        = !params.no_op_offload;
    cparams.swa_full          = params.swa_full;
    cparams.kv_unified        = params.kv_unified;

    cparams.type_k = params.cache_type_k;
    cparams.type_v = params.cache_type_v;
    cparams.type_r = params.cache_type_r;
    cparams.type_s = params.cache_type_s;

    return cparams;
}

struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const common_cpu_params & params) {
    struct ggml_threadpool_params tpp;

    ggml_threadpool_params_init(&tpp, params.n_threads); // setup the defaults

    if (params.mask_valid) {
        std::memcpy(&tpp.cpumask, &params.cpumask, GGML_MAX_N_THREADS);
    }

    tpp.prio       = params.priority;
    tpp.poll       = params.poll;
    tpp.strict_cpu = params.strict_cpu;

    return tpp;
}

//
// Batch utils
//

void common_batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

void common_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
}

//
// Vocab utils
//

std::vector<llama_token> common_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_tokenize(vocab, text, add_special, parse_special);
}

std::vector<llama_token> common_tokenize(
    const struct llama_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
    }
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string common_token_to_piece(const struct llama_context * ctx, llama_token token, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_token_to_piece(vocab, token, special);
}

std::string common_token_to_piece(const struct llama_vocab * vocab, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

std::string common_detokenize(const struct llama_context * ctx, const std::vector<llama_token> & tokens, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_detokenize(vocab, tokens, special);
}

std::string common_detokenize(const struct llama_vocab * vocab, const std::vector<llama_token> & tokens, bool special) {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
        GGML_ASSERT(n_chars <= (int32_t)text.size());  // whitespace trimming is performed after per-token detokenization
    }

    text.resize(n_chars);

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

//
// Embedding utils
//

void common_embd_normalize(const float * inp, float * out, int n, int embd_norm) {
    double sum = 0.0;

    switch (embd_norm) {
        case -1: // no normalisation
            sum = 1.0;
            break;
        case 0: // max absolute
            for (int i = 0; i < n; i++) {
                if (sum < std::abs(inp[i])) {
                    sum = std::abs(inp[i]);
                }
            }
            sum /= 32760.0; // make an int16 range
            break;
        case 2: // euclidean
            for (int i = 0; i < n; i++) {
                sum += inp[i] * inp[i];
            }
            sum = std::sqrt(sum);
            break;
        default: // p-norm (euclidean is p-norm p=2)
            for (int i = 0; i < n; i++) {
                sum += std::pow(std::abs(inp[i]), embd_norm);
            }
            sum = std::pow(sum, 1.0 / embd_norm);
            break;
    }

    const float norm = sum > 0.0 ? 1.0 / sum : 0.0f;

    for (int i = 0; i < n; i++) {
        out[i] = inp[i] * norm;
    }
}

float common_embd_similarity_cos(const float * embd1, const float * embd2, int n){
    double sum  = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i = 0; i < n; i++) {
        sum  += embd1[i] * embd2[i];
        sum1 += embd1[i] * embd1[i];
        sum2 += embd2[i] * embd2[i];
    }

    // Handle the case where one or both vectors are zero vectors
    if (sum1 == 0.0 || sum2 == 0.0) {
        if (sum1 == 0.0 && sum2 == 0.0) {
            return 1.0f; // two zero vectors are similar
        }
        return 0.0f;
    }

    return sum / (sqrt(sum1) * sqrt(sum2));
}

//
// Control vector utils
//

static common_control_vector_data common_control_vector_load_one(const common_control_vector_load_info & load_info) {
    common_control_vector_data result = { -1, {} };

    ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ctx,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(load_info.fname.c_str(), meta_gguf_params);
    if (!ctx_gguf) {
        COM_ERR("failed to load control vector file from %s\n", load_info.fname.c_str());
        return result;
    }

    int32_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    if (n_tensors == 0) {
        COM_WRN("no direction tensors found in %s\n", load_info.fname.c_str());
    }

    for (int i = 0; i < n_tensors; i++) {
        std::string name = gguf_get_tensor_name(ctx_gguf, i);

        int layer_idx = -1;

        // split on '.'
        size_t dotpos = name.find('.');
        if (dotpos != std::string::npos && name.substr(0, dotpos) == "direction") {
            try {
                layer_idx = std::stoi(name.substr(dotpos + 1));
            } catch (...) {
                layer_idx = -1;
            }
        }
        if (layer_idx < 0) {
            COM_ERR("invalid/unparsable direction tensor layer index in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        } else if (layer_idx == 0) {
            COM_ERR("invalid (zero) direction tensor layer index in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        struct ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());
        if (tensor->type != GGML_TYPE_F32) {
            COM_ERR("invalid (non-F32) direction tensor type in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }
        if (ggml_n_dims(tensor) != 1) {
            COM_ERR("invalid (non-1D) direction tensor shape in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result.n_embd = ggml_nelements(tensor);
        } else if (ggml_nelements(tensor) != result.n_embd) {
            COM_ERR("direction tensor in %s does not match previous dimensions\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        // extend if necessary - do not store data for layer 0 (it's not used)
        result.data.resize(std::max(result.data.size(), static_cast<size_t>(result.n_embd * layer_idx)), 0.0f);

        const float * src = (const float *) tensor->data;
        float * dst = result.data.data() + result.n_embd * (layer_idx - 1);  // layer 1 at [0]
        for (int j = 0; j < result.n_embd; j++) {
            dst[j] += src[j] * load_info.strength;  // allows multiple directions for same layer in same file
        }

    }

    if (result.n_embd == -1) {
        COM_WRN("skipping %s due to invalid direction tensors\n", load_info.fname.c_str());
        result.data.clear();
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    return result;
}

common_control_vector_data common_control_vector_load(const std::vector<common_control_vector_load_info> & load_infos) {
    common_control_vector_data result = { -1, {} };

    for (const auto & info : load_infos) {
        auto cur = common_control_vector_load_one(info);

        if (cur.n_embd == -1) {
            result.n_embd = -1;
            break;
        }
        if (result.n_embd != -1 && result.n_embd != cur.n_embd) {
            COM_ERR("control vectors in %s does not match previous dimensions\n", info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result = std::move(cur);
        } else {
            result.data.resize(std::max(result.data.size(), cur.data.size()), 0.0f);  // extend if necessary
            for (size_t i = 0; i < cur.data.size(); i++) {
                result.data[i] += cur.data[i];
            }
        }
    }

    if (result.n_embd == -1) {
        COM_ERR("%s", "no valid control vector files passed\n");
        result.data.clear();
    }

    return result;
}

ggml_opt_dataset_t common_opt_dataset_init(struct llama_context * ctx, const std::vector<llama_token> & tokens, int64_t stride) {
    const int64_t ne_datapoint = llama_n_ctx(ctx);
    const int64_t ndata        = (tokens.size() - ne_datapoint - 1) / stride;
    ggml_opt_dataset_t result = ggml_opt_dataset_init(
        GGML_TYPE_I32, GGML_TYPE_I32, ne_datapoint, ne_datapoint, ndata, /*ndata_shard =*/ 1);

    llama_token * data   = (llama_token *) ggml_opt_dataset_data(result)->data;
    llama_token * labels = (llama_token *) ggml_opt_dataset_labels(result)->data;

    for (int64_t idata = 0; idata < ndata; ++idata) {
        memcpy(data   + idata*ne_datapoint, tokens.data() + idata*stride + 0, ne_datapoint*sizeof(llama_token));
        memcpy(labels + idata*ne_datapoint, tokens.data() + idata*stride + 1, ne_datapoint*sizeof(llama_token));
    }

    return result;
}

ggml_opt_optimizer_params common_opt_lr_pars(void * userdata) {
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(nullptr);
    const lr_opt &            d      = *(lr_opt *) userdata;
    result.adamw.alpha = result.sgd.alpha = d.get_lr(d.epoch);
    result.sgd.wd = result.adamw.wd = d.wd;
    return result;
}

// TODO make all command line args case-insensitive
static inline bool eq_case_insensitive(char const* a, char const* b) {
    return !
#if defined(_MSC_VER)
        _stricmp
#else
        strcasecmp
#endif // defined(_MSC_VER)
        (a, b);
}

enum ggml_opt_optimizer_type common_opt_get_optimizer(const char * n) {
    if (eq_case_insensitive("adamw", n)) {
        return GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    }
    if (eq_case_insensitive("sgd", n)) {
        return GGML_OPT_OPTIMIZER_TYPE_SGD;
    }
    return GGML_OPT_OPTIMIZER_TYPE_COUNT;
}

// TODO simplify to use just log and exp
static float const k_log_2 = std::log(2.f);

void lr_opt::init() {
    if (lr_min > 0 && lr_min < lr0) {
        float nhalf = std::log(lr0 / lr_min) / k_log_2;
        float e     = epochs;
        if (decay_epochs > 0 && decay_epochs < e) {
            e = decay_epochs;
        } else {
            decay_epochs = e;
        }
        scale_epoch = nhalf / e;
    }
}

float lr_opt::get_lr(float epoch) const {
    float r = lr_min <= 0 ? lr0 :
        epoch >= decay_epochs ? lr_min :
        lr0 * std::pow(0.5f, epoch * scale_epoch);
    LOG_INF("epoch %.2g lr=%.2g\n", epoch, r);
    return r;
}

bool common_replay_last_token(struct llama_context * ctx, llama_token last_token, int32_t pos) {
    llama_batch batch = llama_batch_get_one(&last_token, 1);
    batch.pos = &pos;
    if (llama_decode(ctx, batch)) {
        LOG_ERR("%s: failed to replay last token\n", __func__);
        return false;
    }
    return true;
}

bool common_prompt_batch_decode(
              struct llama_context * ctx,
    const std::vector<llama_token> & all_tokens,
                               int   n_new,
                               int & n_past,
                               int   n_batch,
                  std::string_view   state_path,
                              bool   save_state) {
    if (n_new == 0) {
        return true;
    }
    const int offset = all_tokens.size() - n_new;

    if (save_state && n_new > 1) {
        const int n_tokens_before_last = n_new - 1;

        GGML_ASSERT(n_new <= n_batch);

        // Decode all but the last token so we can save the memory state before decoding the last token.
        // This is done so we can restore the session state later and replay the last token.
        // Memory implementations in recurrent/hybrid models don't support removing tokens from their
        // memory, so we can't just remove the last token from the memory and replay the last token which
        // is the reason for this logic.
        if (llama_decode(ctx, llama_batch_get_one(const_cast<llama_token*>(all_tokens.data() + offset), n_tokens_before_last))) {
            COM_ERR("%s", "failed to eval\n");
            return false;
        }
        n_past += n_tokens_before_last;

        llama_state_save_file(ctx, state_path.data(), all_tokens.data(), all_tokens.size());
        COM_INF("saved session before last token to %s, n_new = %zu\n", state_path.data(), all_tokens.size());

        llama_token last_token = all_tokens.back();
        llama_batch batch = llama_batch_get_one(&last_token, 1);
        int32_t pos = n_past;
        batch.pos = &pos;

        if (llama_decode(ctx, batch)) {
            COM_ERR("%s", "failed to eval last token\n");
            return false;
        }
        n_past++;
    } else {
        if (llama_decode(ctx, llama_batch_get_one(const_cast<llama_token*>(all_tokens.data() + offset), n_new))) {
            COM_ERR("%s", "failed to eval\n");
            return false;
        }
        n_past += n_new;
    }

    return true;
}

size_t common_prompt_checkpoint::size() const {
    return data_tgt.size() + data_dft.size() + data_spec.size();
}

bool common_prompt_checkpoint::empty() const {
    return data_tgt.empty();
}

void common_prompt_checkpoint::clear() {
    n_tokens = 0;

    pos_min = 0;
    pos_max = 0;

    data_tgt.clear();
    data_dft.clear();
    data_spec.clear();
}

void common_prompt_checkpoint::update_pos(
        int64_t n_tokens,
        llama_pos pos_min,
        llama_pos pos_max) {
    this->n_tokens = n_tokens;
    this->pos_min  = pos_min;
    this->pos_max  = pos_max;
}

void common_prompt_checkpoint::update_tgt(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) {
    if (ctx == nullptr) {
        return;
    }

    const size_t ckpt_size = llama_state_seq_get_size_ext(ctx, seq_id, flags);

    data_tgt.resize(ckpt_size);

    const size_t n = llama_state_seq_get_data_ext(ctx, data_tgt.data(), ckpt_size, seq_id, flags);
    if (n != ckpt_size) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", ckpt_size, n);
    }
}

void common_prompt_checkpoint::update_dft(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) {
    if (ctx == nullptr) {
        return;
    }

    const size_t ckpt_size = llama_state_seq_get_size_ext(ctx, seq_id, flags);

    data_dft.resize(ckpt_size);

    const size_t n = llama_state_seq_get_data_ext(ctx, data_dft.data(), ckpt_size, seq_id, flags);
    if (n != ckpt_size) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", ckpt_size, n);
    }
}

void common_prompt_checkpoint::load_tgt(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    if (ctx == nullptr) {
        return;
    }

    if (data_tgt.empty()) {
        return;
    }

    const size_t n = llama_state_seq_set_data_ext(ctx, data_tgt.data(), data_tgt.size(), seq_id, flags);
    if (n != data_tgt.size()) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", data_tgt.size(), n);
    }
}

void common_prompt_checkpoint::load_dft(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    if (ctx == nullptr) {
        return;
    }

    if (data_dft.empty()) {
        return;
    }

    const size_t n = llama_state_seq_set_data_ext(ctx, data_dft.data(), data_dft.size(), seq_id, flags);
    if (n != data_dft.size()) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", data_dft.size(), n);
    }
}

void common_prompt_checkpoint::clear_tgt() {
    data_tgt.clear();
}

void common_prompt_checkpoint::clear_dft() {
    data_dft.clear();
    data_spec.clear();
}
