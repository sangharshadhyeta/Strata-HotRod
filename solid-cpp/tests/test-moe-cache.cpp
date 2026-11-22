#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-moe-cache.h"

#ifdef MOE_CACHE_TEST_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int64_t n_in      = 256;
constexpr int64_t n_out     = 128;
constexpr int64_t n_expert  = 64;
constexpr int64_t n_used    = 2;
constexpr int64_t n_tokens  = 1;
constexpr int     max_steps = 160;

struct log_capture {
    std::mutex mutex;
    std::condition_variable cv;
    std::string text;

    void clear() {
        std::lock_guard<std::mutex> lock(mutex);
        text.clear();
    }

    std::string get() {
        std::lock_guard<std::mutex> lock(mutex);
        return text;
    }

    bool wait_for(
            const char * pattern,
            const std::atomic<bool> & stop,
            std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, timeout, [&] {
            return text.find(pattern) != std::string::npos || stop.load();
        }) && text.find(pattern) != std::string::npos;
    }
};

static void log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    log_capture & capture = *static_cast<log_capture *>(user_data);
    {
        std::lock_guard<std::mutex> lock(capture.mutex);
        capture.text += text;
    }
    capture.cv.notify_all();
}

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

static bool has_positive_field(const std::string & text, const char * field) {
    size_t position = 0;
    while ((position = text.find(field, position)) != std::string::npos) {
        position += strlen(field);
        char * end = nullptr;
        const long long value = strtoll(text.c_str() + position, &end, 10);
        if (end != text.c_str() + position && value > 0) {
            return true;
        }
    }
    return false;
}

static long long max_field_value(const std::string & text, const char * field) {
    long long result = -1;
    size_t position = 0;
    while ((position = text.find(field, position)) != std::string::npos) {
        position += strlen(field);
        char * end = nullptr;
        const long long value = strtoll(text.c_str() + position, &end, 10);
        if (end != text.c_str() + position) {
            result = std::max(result, value);
        }
    }
    return result;
}

static size_t count_field_at_least(
        const std::string & text, const char * field, long long minimum) {
    size_t result = 0;
    size_t position = 0;
    while ((position = text.find(field, position)) != std::string::npos) {
        position += strlen(field);
        char * end = nullptr;
        const long long value = strtoll(text.c_str() + position, &end, 10);
        if (end != text.c_str() + position && value >= minimum) {
            result++;
        }
    }
    return result;
}

static size_t count_occurrences(const std::string & text, const char * pattern) {
    size_t result = 0;
    size_t position = 0;
    while ((position = text.find(pattern, position)) != std::string::npos) {
        result++;
        position += strlen(pattern);
    }
    return result;
}

static ggml_backend_dev_t find_cuda_device() {
    ggml_backend_load_all();
    for (size_t index = 0; index < ggml_backend_dev_count(); index++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU &&
            strcmp(ggml_backend_reg_name(reg), "CUDA") == 0) {
            return device;
        }
    }
    return nullptr;
}

static ggml_backend_dev_t find_other_cuda_device(
        ggml_backend_dev_t excluded) {
    for (size_t index = 0; index < ggml_backend_dev_count(); index++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        if (device != excluded &&
            ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU &&
            strcmp(ggml_backend_reg_name(reg), "CUDA") == 0) {
            return device;
        }
    }
    return nullptr;
}

static ggml_backend_t init_cpu_backend() {
    for (size_t index = 0; index < ggml_backend_dev_count(); index++) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (ggml_backend_dev_type(device) !=
                GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        ggml_backend_t backend =
            ggml_backend_dev_init(device, nullptr);
        if (!backend) {
            continue;
        }
        ggml_backend_reg_t reg =
            ggml_backend_dev_backend_reg(device);
        auto set_n_threads =
            (ggml_backend_set_n_threads_t)
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_set_n_threads");
        if (set_n_threads) {
            set_n_threads(backend, 4);
        }
        return backend;
    }
    return nullptr;
}

static bool compare_output(
        const std::vector<float> & reference,
        const std::vector<float> & actual,
        double max_nmse) {
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (size_t index = 0; index < reference.size(); index++) {
        if (!std::isfinite(actual[index])) {
            return false;
        }
        const double difference = (double) actual[index] - reference[index];
        squared_error += difference * difference;
        squared_reference += (double) reference[index] * reference[index];
    }
    return squared_error / std::max(squared_reference, 1e-12) <= max_nmse;
}

static void configure_cache(const char * fail_stage) {
    set_env("GGML_CUDA_MOE_CACHE", "1");
    set_env("GGML_CUDA_MOE_CACHE_MODE", "on");
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "4");
    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", "0");
    set_env("GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB", "1");
    set_env("GGML_CUDA_MOE_CACHE_MAX_BATCH", "1");
    set_env("GGML_CUDA_MOE_CACHE_INSERTS", "4");
    set_env("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", "1");
    set_env("GGML_CUDA_MOE_CACHE_THROTTLE", "1");
    set_env("GGML_CUDA_MOE_CACHE_QUEUE", "16");
    set_env("GGML_CUDA_MOE_CACHE_STATS", "1");
    set_env("GGML_CUDA_MOE_CACHE_NDEV", "1");
    set_env("GGML_CUDA_MOE_CACHE_MIN_CC", "0");
    set_env("GGML_CUDA_MOE_CACHE_FAIL", fail_stage);
}

struct test_graph {
    ggml_context * ctx = nullptr;
    ggml_tensor * out = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
};

static test_graph make_graph(
        ggml_backend_t cpu,
        ggml_tensor * weights,
        ggml_tensor * activations,
        ggml_tensor * ids) {
    ggml_init_params params = {
        8 * ggml_tensor_overhead() + ggml_graph_overhead(),
        nullptr,
        true,
    };
    test_graph result;
    result.ctx = ggml_init(params);
    if (!result.ctx) {
        return result;
    }
    result.out = ggml_mul_mat_id(result.ctx, weights, activations, ids);
    ggml_set_name(result.out, "moe_cache_test_out");
    result.graph = ggml_new_graph(result.ctx);
    ggml_build_forward_expand(result.graph, result.out);
    result.buffer = ggml_backend_alloc_ctx_tensors(result.ctx, cpu);
    return result;
}

static void free_graph(test_graph & graph) {
    if (graph.buffer) {
        ggml_backend_buffer_free(graph.buffer);
    }
    if (graph.ctx) {
        ggml_free(graph.ctx);
    }
    graph = {};
}

static bool run_scenario(
        const char * name,
        const char * fail_stage,
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        test_graph & graph,
        const std::vector<float> & reference,
        log_capture & capture) {
    configure_cache(fail_stage);
    capture.clear();

    ggml_backend_t backends[] = { cuda, cpu };
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
            backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (!scheduler) {
        fprintf(stderr, "%s: failed to create scheduler\n", name);
        return false;
    }
    ggml_backend_sched_set_tensor_backend(scheduler, graph.out, cpu);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph.graph) ||
        ggml_backend_sched_get_tensor_backend(scheduler, graph.out) != cpu) {
        fprintf(stderr, "%s: MUL_MAT_ID was not assigned to CPU\n", name);
        ggml_backend_sched_free(scheduler);
        return false;
    }

    bool output_ok = true;
    std::vector<float> actual(reference.size());
    for (int step = 0; step < max_steps; step++) {
        const enum ggml_status status =
            ggml_backend_sched_graph_compute(scheduler, graph.graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: graph compute failed at step %d: %s\n",
                    name, step, ggml_status_to_string(status));
            output_ok = false;
            break;
        }
        ggml_backend_tensor_get(
                graph.out, actual.data(), 0, actual.size() * sizeof(float));
        if (!compare_output(reference, actual, 5e-4)) {
            fprintf(stderr, "%s: output mismatch at step %d\n", name, step);
            output_ok = false;
            break;
        }
        if (step >= 64) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    ggml_backend_sched_free(scheduler);
    const std::string log = capture.get();
    bool stage_ok = false;
    if (!fail_stage) {
        stage_ok = has_positive_field(log, "hits=");
    } else if (strcmp(fail_stage, "dispatch") == 0) {
        stage_ok = has_positive_field(log, "dispatch-fail=");
    } else if (strcmp(fail_stage, "collect") == 0) {
        stage_ok = has_positive_field(log, "collect-fail=");
    } else if (strcmp(fail_stage, "insert") == 0) {
        stage_ok = has_positive_field(log, "fill-fail=");
    } else if (strcmp(fail_stage, "slab") == 0) {
        stage_ok = log.find("allocation failed") != std::string::npos;
    }
    if (!stage_ok) {
        fprintf(stderr, "%s: cache stage was not observed\n%s", name, log.c_str());
    }
    printf("%s: %s\n", name, output_ok && stage_ok ? "OK" : "FAIL");
    return output_ok && stage_ok;
}

static bool run_invalidation_scenario(
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        test_graph & graph,
        ggml_tensor * weights,
        const std::vector<uint8_t> & replacement_row,
        const std::vector<float> & old_reference,
        log_capture & capture) {
    configure_cache(nullptr);
    capture.clear();

    ggml_backend_t backends[] = { cuda, cpu };
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
            backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (!scheduler) {
        fprintf(stderr, "cache-invalidate: failed to create scheduler\n");
        return false;
    }
    ggml_backend_sched_set_tensor_backend(scheduler, graph.out, cpu);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph.graph) ||
        ggml_backend_sched_get_tensor_backend(scheduler, graph.out) != cpu) {
        fprintf(stderr, "cache-invalidate: MUL_MAT_ID was not assigned to CPU\n");
        ggml_backend_sched_free(scheduler);
        return false;
    }

    std::vector<float> actual(old_reference.size());
    long long hits_before = -1;
    bool output_ok = true;
    for (int step = 0; step < max_steps; step++) {
        const enum ggml_status status =
            ggml_backend_sched_graph_compute(scheduler, graph.graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "cache-invalidate: warmup failed at step %d: %s\n",
                    step, ggml_status_to_string(status));
            output_ok = false;
            break;
        }
        ggml_backend_tensor_get(
                graph.out, actual.data(), 0, actual.size() * sizeof(float));
        if (!compare_output(old_reference, actual, 5e-4)) {
            fprintf(stderr, "cache-invalidate: warmup mismatch at step %d\n", step);
            output_ok = false;
            break;
        }
        hits_before = max_field_value(capture.get(), "hits=");
        if (hits_before > 0) {
            break;
        }
        if (step >= 64) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    if (output_ok && hits_before <= 0) {
        fprintf(stderr, "cache-invalidate: no cache hit before mutation\n");
        output_ok = false;
    }

    std::vector<float> new_reference(old_reference.size());
    if (output_ok) {
        ggml_backend_tensor_set(
                weights, replacement_row.data(), 0, replacement_row.size());
        if (ggml_backend_graph_compute(cpu, graph.graph) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "cache-invalidate: CPU reference compute failed\n");
            output_ok = false;
        } else {
            ggml_backend_tensor_get(
                    graph.out, new_reference.data(), 0,
                    new_reference.size() * sizeof(float));
            float max_change = 0.0f;
            for (size_t index = 0; index < new_reference.size(); index++) {
                max_change = std::max(
                        max_change, std::abs(new_reference[index] - old_reference[index]));
            }
            if (max_change < 0.01f) {
                fprintf(stderr, "cache-invalidate: mutation did not change the reference\n");
                output_ok = false;
            }
        }
    }

    capture.clear();
    bool repopulated = false;
    if (output_ok) {
        for (int step = 0; step < max_steps; step++) {
            const enum ggml_status status =
                ggml_backend_sched_graph_compute(scheduler, graph.graph);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "cache-invalidate: compute failed at step %d: %s\n",
                        step, ggml_status_to_string(status));
                output_ok = false;
                break;
            }
            ggml_backend_tensor_get(
                    graph.out, actual.data(), 0, actual.size() * sizeof(float));
            if (!compare_output(new_reference, actual, 5e-4)) {
                fprintf(stderr, "cache-invalidate: stale output at step %d\n", step);
                output_ok = false;
                break;
            }
            if (max_field_value(capture.get(), "hits=") > hits_before) {
                repopulated = true;
                break;
            }
            if (step >= 64) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    ggml_backend_sched_free(scheduler);
    if (output_ok && !repopulated) {
        fprintf(stderr, "cache-invalidate: mutated expert was not repopulated\n%s",
                capture.get().c_str());
        output_ok = false;
    }
    printf("cache-invalidate: %s\n", output_ok ? "OK" : "FAIL");
    return output_ok;
}

constexpr int64_t stress_n_out  = 65;
constexpr int64_t stress_n_used = 64;

struct stress_fixture {
    ggml_context * ctx = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * activations = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    test_graph graph;
    std::vector<float> reference;
};

static void free_stress_fixture(stress_fixture & fixture) {
    free_graph(fixture.graph);
    if (fixture.buffer) {
        ggml_backend_buffer_free(fixture.buffer);
    }
    if (fixture.ctx) {
        ggml_free(fixture.ctx);
    }
    fixture = {};
}

static bool init_stress_fixture(stress_fixture & fixture, ggml_backend_t cpu) {
    const ggml_init_params params = {
        8 * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    fixture.ctx = ggml_init(params);
    if (!fixture.ctx) {
        return false;
    }

    fixture.weights = ggml_new_tensor_3d(
            fixture.ctx, GGML_TYPE_Q4_0, n_in, stress_n_out, n_expert);
    fixture.ids = ggml_new_tensor_2d(
            fixture.ctx, GGML_TYPE_I32, stress_n_used, n_tokens);
    fixture.activations = ggml_new_tensor_3d(
            fixture.ctx, GGML_TYPE_F32, n_in, stress_n_used, n_tokens);
    ggml_set_name(fixture.weights, "blk.1.ffn_up_exps.weight");
    ggml_set_name(fixture.ids, "moe_cache_stress_ids");
    ggml_set_name(fixture.activations, "moe_cache_stress_activations");

    fixture.buffer = ggml_backend_alloc_ctx_tensors(fixture.ctx, cpu);
    if (!fixture.buffer) {
        free_stress_fixture(fixture);
        return false;
    }
    ggml_backend_buffer_set_usage(
            fixture.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<float> weights_f32(ggml_nelements(fixture.weights));
    for (size_t index = 0; index < weights_f32.size(); index++) {
        weights_f32[index] =
            0.13f * std::sin((float) (index % 983) * 0.019f) -
            0.04f * std::cos((float) (index % 419) * 0.029f);
    }
    std::vector<uint8_t> weights_q4(ggml_nbytes(fixture.weights));
    const size_t quantized = ggml_quantize_chunk(
            GGML_TYPE_Q4_0, weights_f32.data(), weights_q4.data(),
            0, stress_n_out * n_expert, n_in, nullptr);
    if (quantized != weights_q4.size()) {
        fprintf(stderr, "stress: unexpected quantized size\n");
        free_stress_fixture(fixture);
        return false;
    }
    ggml_backend_tensor_set(
            fixture.weights, weights_q4.data(), 0, weights_q4.size());

    std::vector<int32_t> ids_data(stress_n_used);
    for (int32_t index = 0; index < stress_n_used; index++) {
        ids_data[index] = index;
    }
    ggml_backend_tensor_set(
            fixture.ids, ids_data.data(), 0,
            ids_data.size() * sizeof(ids_data[0]));

    std::vector<float> activation_data(
            ggml_nelements(fixture.activations));
    for (size_t index = 0; index < activation_data.size(); index++) {
        activation_data[index] =
            0.35f * std::sin((float) index * 0.067f) +
            0.17f * std::cos((float) index * 0.103f);
    }
    ggml_backend_tensor_set(
            fixture.activations, activation_data.data(), 0,
            activation_data.size() * sizeof(float));

    fixture.graph = make_graph(
            cpu, fixture.weights, fixture.activations, fixture.ids);
    if (!fixture.graph.ctx || !fixture.graph.buffer) {
        fprintf(stderr, "stress: failed to create graph\n");
        free_stress_fixture(fixture);
        return false;
    }

    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, fixture.graph.graph) !=
            GGML_STATUS_SUCCESS) {
        fprintf(stderr, "stress: CPU reference compute failed\n");
        free_stress_fixture(fixture);
        return false;
    }
    fixture.reference.resize(ggml_nelements(fixture.graph.out));
    ggml_backend_tensor_get(
            fixture.graph.out, fixture.reference.data(), 0,
            fixture.reference.size() * sizeof(float));
    return true;
}

static ggml_backend_sched_t make_scheduler(
        const char * name,
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        test_graph & graph) {
    ggml_backend_t backends[] = { cuda, cpu };
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
            backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (!scheduler) {
        fprintf(stderr, "%s: failed to create scheduler\n", name);
        return nullptr;
    }
    ggml_backend_sched_set_tensor_backend(scheduler, graph.out, cpu);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph.graph) ||
        ggml_backend_sched_get_tensor_backend(scheduler, graph.out) != cpu) {
        fprintf(stderr, "%s: MUL_MAT_ID was not assigned to CPU\n", name);
        ggml_backend_sched_free(scheduler);
        return nullptr;
    }
    return scheduler;
}

static bool compute_matches(
        const char * name,
        ggml_backend_sched_t scheduler,
        test_graph & graph,
        const std::vector<float> & reference,
        int step) {
    const enum ggml_status status =
        ggml_backend_sched_graph_compute(scheduler, graph.graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph compute failed at step %d: %s\n",
                name, step, ggml_status_to_string(status));
        return false;
    }
    std::vector<float> actual(reference.size());
    ggml_backend_tensor_get(
            graph.out, actual.data(), 0, actual.size() * sizeof(float));
    if (!compare_output(reference, actual, 5e-4)) {
        fprintf(stderr, "%s: output mismatch at step %d\n", name, step);
        return false;
    }
    return true;
}

static bool run_precensus_invalidation(
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        test_graph & graph,
        ggml_tensor * weights,
        const uint8_t * unchanged_expert,
        size_t expert_size,
        const std::vector<float> & reference,
        log_capture & capture) {
    configure_cache(nullptr);
    capture.clear();

    ggml_backend_sched_t scheduler = make_scheduler(
            "cache-precensus-invalidate", cuda, cpu, graph);
    if (!scheduler) {
        return false;
    }

    bool output_ok = compute_matches(
            "cache-precensus-invalidate", scheduler, graph, reference, 0);
    if (output_ok) {
        ggml_backend_tensor_set(
                weights, unchanged_expert,
                (n_expert - 1) * expert_size, expert_size);
    }
    for (int step = 1; step < 80 && output_ok; step++) {
        output_ok = compute_matches(
                "cache-precensus-invalidate", scheduler, graph,
                reference, step);
        if (step >= 64) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    ggml_backend_sched_free(scheduler);
    const std::string log = capture.get();
    const bool census_ok =
        log.find("slots=64 ") != std::string::npos;
    if (!census_ok) {
        fprintf(stderr,
                "cache-precensus-invalidate: tensor census was not stable\n%s",
                log.c_str());
    }
    printf("cache-precensus-invalidate: %s\n",
            output_ok && census_ok ? "OK" : "FAIL");
    return output_ok && census_ok;
}

static bool run_concurrent_sessions(
        ggml_backend_dev_t cuda_device,
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        stress_fixture & fixture,
        log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_STATS", "0");
    capture.clear();

    ggml_backend_t cuda_second =
        ggml_backend_dev_init(cuda_device, nullptr);
    ggml_backend_t cpu_second = init_cpu_backend();
    if (!cuda_second || !cpu_second) {
        fprintf(stderr, "cache-concurrent: failed to create second backends\n");
        if (cuda_second) {
            ggml_backend_free(cuda_second);
        }
        if (cpu_second) {
            ggml_backend_free(cpu_second);
        }
        return false;
    }
    test_graph second_graph = make_graph(
            cpu_second, fixture.weights, fixture.activations, fixture.ids);
    if (!second_graph.ctx || !second_graph.buffer) {
        fprintf(stderr, "cache-concurrent: failed to create second graph\n");
        free_graph(second_graph);
        ggml_backend_free(cuda_second);
        ggml_backend_free(cpu_second);
        return false;
    }

    ggml_backend_sched_t first = make_scheduler(
            "cache-concurrent-1", cuda, cpu, fixture.graph);
    ggml_backend_sched_t second = make_scheduler(
            "cache-concurrent-2", cuda_second, cpu_second, second_graph);
    if (!first || !second) {
        if (first) {
            ggml_backend_sched_free(first);
        }
        if (second) {
            ggml_backend_sched_free(second);
        }
        free_graph(second_graph);
        ggml_backend_free(cuda_second);
        ggml_backend_free(cpu_second);
        return false;
    }

    constexpr int concurrent_steps = 112;
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::atomic<bool> output_ok{true};
    auto run = [&](const char * name, ggml_backend_sched_t scheduler,
                   test_graph & graph) {
        ready.fetch_add(1);
        while (!start.load()) {
            std::this_thread::yield();
        }
        for (int step = 0; step < concurrent_steps && output_ok.load(); step++) {
            if (!compute_matches(
                    name, scheduler, graph, fixture.reference, step)) {
                output_ok.store(false);
                break;
            }
            if (step >= 64) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    };

    std::thread first_thread(
            run, "cache-concurrent-1", first, std::ref(fixture.graph));
    std::thread second_thread(
            run, "cache-concurrent-2", second, std::ref(second_graph));
    while (ready.load() != 2) {
        std::this_thread::yield();
    }
    start.store(true);
    first_thread.join();
    second_thread.join();

    ggml_backend_sched_free(first);
    ggml_backend_sched_free(second);
    const std::string log = capture.get();
    const bool cache_ok =
        count_occurrences(log, " pool[") >= 2 &&
        count_field_at_least(log, "hits=", stress_n_used) >= 2 &&
        count_field_at_least(log, "used=", stress_n_used) >= 2;
    if (!cache_ok) {
        fprintf(stderr, "cache-concurrent: full cache use was not observed\n%s",
                log.c_str());
    }

    free_graph(second_graph);
    ggml_backend_free(cuda_second);
    ggml_backend_free(cpu_second);
    printf("cache-concurrent: %s\n",
            output_ok.load() && cache_ok ? "OK" : "FAIL");
    return output_ok.load() && cache_ok;
}

static bool run_repeated_lifecycle(
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        stress_fixture & fixture,
        log_capture & capture) {
    configure_cache(nullptr);
    capture.clear();

    constexpr int cycles = 8;
    constexpr int census_steps = 65;
    bool output_ok = true;
    for (int cycle = 0; cycle < cycles && output_ok; cycle++) {
        ggml_backend_sched_t scheduler = make_scheduler(
                "cache-lifecycle", cuda, cpu, fixture.graph);
        if (!scheduler) {
            output_ok = false;
            break;
        }
        for (int step = 0; step < census_steps; step++) {
            const enum ggml_status status =
                ggml_backend_sched_graph_compute(scheduler, fixture.graph.graph);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr,
                        "cache-lifecycle: compute failed in cycle %d step %d: %s\n",
                        cycle, step, ggml_status_to_string(status));
                output_ok = false;
                break;
            }
        }
        if (output_ok) {
            std::vector<float> actual(fixture.reference.size());
            ggml_backend_tensor_get(
                    fixture.graph.out, actual.data(), 0,
                    actual.size() * sizeof(float));
            output_ok = compare_output(
                    fixture.reference, actual, 5e-4);
            if (!output_ok) {
                fprintf(stderr,
                        "cache-lifecycle: output mismatch in cycle %d\n", cycle);
            }
        }
        ggml_backend_sched_free(scheduler);
    }

    const std::string log = capture.get();
    const bool cache_ok =
        count_occurrences(log, " pool[") >= cycles &&
        has_positive_field(log, "enqueued=");
    if (!cache_ok) {
        fprintf(stderr, "cache-lifecycle: fill startup was not observed\n%s",
                log.c_str());
    }
    printf("cache-lifecycle: %s\n",
            output_ok && cache_ok ? "OK" : "FAIL");
    return output_ok && cache_ok;
}

static bool run_fill_invalidation(
        ggml_backend_t cuda,
        ggml_backend_t cpu,
        stress_fixture & fixture,
        log_capture & capture) {
    configure_cache(nullptr);
    capture.clear();

    ggml_backend_sched_t scheduler = make_scheduler(
            "cache-fill-invalidate", cuda, cpu, fixture.graph);
    if (!scheduler) {
        return false;
    }

    std::vector<float> replacement_f32(n_in * stress_n_out);
    for (size_t index = 0; index < replacement_f32.size(); index++) {
        replacement_f32[index] =
            1.1f + 0.2f * std::sin((float) index * 0.043f);
    }
    std::vector<uint8_t> replacement_q4(
            ggml_row_size(GGML_TYPE_Q4_0, n_in) * stress_n_out);
    const size_t replacement_size = ggml_quantize_chunk(
            GGML_TYPE_Q4_0, replacement_f32.data(), replacement_q4.data(),
            0, stress_n_out, n_in, nullptr);
    if (replacement_size != replacement_q4.size()) {
        fprintf(stderr,
                "cache-fill-invalidate: unexpected replacement size\n");
        ggml_backend_sched_free(scheduler);
        return false;
    }

    std::atomic<bool> stop{false};
    std::atomic<bool> mutation_started{false};
    std::atomic<bool> mutation_done{false};
    std::thread mutator([&] {
        if (capture.wait_for(
                " pool[", stop, std::chrono::seconds(5))) {
            mutation_started.store(true);
            ggml_backend_tensor_set(
                    fixture.weights, replacement_q4.data(), 0,
                    replacement_q4.size());
            mutation_done.store(true);
        }
    });

    bool output_ok = true;
    for (int step = 0; step < max_steps && !mutation_done.load(); step++) {
        const enum ggml_status status =
            ggml_backend_sched_graph_compute(scheduler, fixture.graph.graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr,
                    "cache-fill-invalidate: warmup failed at step %d: %s\n",
                    step, ggml_status_to_string(status));
            output_ok = false;
            break;
        }
        if (mutation_started.load()) {
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!mutation_done.load() &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            break;
        }
    }
    stop.store(true);
    capture.cv.notify_all();
    mutator.join();
    if (!mutation_started.load() || !mutation_done.load()) {
        fprintf(stderr,
                "cache-fill-invalidate: concurrent mutation did not complete\n");
        output_ok = false;
    }

    std::vector<float> new_reference(fixture.reference.size());
    if (output_ok &&
        ggml_backend_graph_compute(cpu, fixture.graph.graph) !=
            GGML_STATUS_SUCCESS) {
        fprintf(stderr,
                "cache-fill-invalidate: CPU reference compute failed\n");
        output_ok = false;
    }
    if (output_ok) {
        ggml_backend_tensor_get(
                fixture.graph.out, new_reference.data(), 0,
                new_reference.size() * sizeof(float));
        float max_change = 0.0f;
        for (size_t index = 0; index < new_reference.size(); index++) {
            max_change = std::max(
                    max_change,
                    std::abs(new_reference[index] - fixture.reference[index]));
        }
        if (max_change < 0.01f) {
            fprintf(stderr,
                    "cache-fill-invalidate: mutation did not change output\n");
            output_ok = false;
        }
    }

    capture.clear();
    bool repopulated = false;
    if (output_ok) {
        for (int step = 0; step < max_steps; step++) {
            if (!compute_matches(
                    "cache-fill-invalidate", scheduler, fixture.graph,
                    new_reference, step)) {
                output_ok = false;
                break;
            }
            if (has_positive_field(capture.get(), "hits=")) {
                repopulated = true;
                break;
            }
            if (step >= 64) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
    ggml_backend_sched_free(scheduler);
    if (output_ok && !repopulated) {
        fprintf(stderr,
                "cache-fill-invalidate: cache was not repopulated\n%s",
                capture.get().c_str());
        output_ok = false;
    }
    printf("cache-fill-invalidate: %s\n", output_ok ? "OK" : "FAIL");
    return output_ok;
}

static void * create_direct_session(
        ggml_backend_t cuda, ggml_backend_t cpu) {
    if (!ggml_moe_cache.session_create) {
        return nullptr;
    }
    void * backends[] = { cuda, cpu };
    return ggml_moe_cache.session_create(backends, 2);
}

static bool direct_begin_ready(
        const char * name, const void * base, size_t expert_size,
        int64_t direct_n_in, int64_t direct_n_out,
        int direct_type, int64_t direct_n_expert) {
    void * node = ggml_moe_cache.begin(
            name, base, expert_size, direct_n_in, direct_n_out,
            direct_type, direct_n_expert, 1);
    if (!node) {
        return false;
    }
    ggml_moe_cache.end(node);
    return true;
}

static bool wait_for_direct_pool(
        const char * name, const void * base, size_t expert_size,
        int64_t direct_n_in, int64_t direct_n_out,
        int direct_type, int64_t direct_n_expert) {
    for (int step = 0; step < 80; step++) {
        if (direct_begin_ready(
                name, base, expert_size, direct_n_in, direct_n_out,
                direct_type, direct_n_expert)) {
            return true;
        }
    }
    return false;
}

static int direct_plan_one(
        const char * name, const void * base, size_t expert_size,
        int64_t direct_n_in, int64_t direct_n_out,
        int direct_type, int64_t direct_n_expert, int32_t expert) {
    void * node = ggml_moe_cache.begin(
            name, base, expert_size, direct_n_in, direct_n_out,
            direct_type, direct_n_expert, 1);
    if (!node) {
        return -1;
    }
    int32_t slot = -1;
    const int hits = ggml_moe_cache.plan(node, &expert, 1, &slot);
    ggml_moe_cache.end(node);
    return hits;
}

static bool run_scope_isolation(
        ggml_backend_t cuda, ggml_backend_t cpu, ggml_tensor * weights) {
    if (!ggml_moe_cache.session_enter || !ggml_moe_cache.session_leave ||
        !ggml_moe_cache.begin || !ggml_moe_cache.end ||
        !ggml_moe_cache.invalidate || !ggml_moe_cache.session_destroy) {
        fprintf(stderr, "cache-scope: incomplete cache API\n");
        return false;
    }

    configure_cache(nullptr);
    void * outer = create_direct_session(cuda, cpu);
    if (!outer) {
        fprintf(stderr, "cache-scope: failed to create outer session\n");
        return false;
    }

    const size_t expert_size = ggml_nbytes(weights) / weights->ne[2];
    ggml_moe_cache.session_enter(outer);
    const bool warmed = wait_for_direct_pool(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    ggml_moe_cache.session_leave(outer);

    set_env("GGML_CUDA_MOE_CACHE_RESERVE_MB", "1048576");
    void * dormant = create_direct_session(cuda, cpu);
    if (!dormant) {
        fprintf(stderr, "cache-scope: failed to create dormant session\n");
        ggml_moe_cache.session_destroy(outer);
        return false;
    }
    ggml_moe_cache.session_enter(dormant);
    const bool dormant_begin = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    ggml_moe_cache.session_leave(dormant);

    ggml_moe_cache.session_enter(outer);
    const bool outer_before = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);

    ggml_moe_cache.session_enter(nullptr);
    const bool null_leaked = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    ggml_moe_cache.session_leave(nullptr);
    const bool outer_after_null = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);

    ggml_moe_cache.session_enter(dormant);
    const bool dormant_leaked = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    ggml_moe_cache.session_leave(dormant);
    const bool outer_after_dormant = direct_begin_ready(
            weights->name, weights->data, expert_size,
            weights->ne[0], weights->ne[1], weights->type, weights->ne[2]);
    ggml_moe_cache.session_leave(outer);

    ggml_moe_cache.session_destroy(dormant);
    ggml_moe_cache.session_destroy(outer);
    const bool ok = warmed && !dormant_begin && outer_before &&
        !null_leaked && outer_after_null &&
        !dormant_leaked && outer_after_dormant;
    printf("cache-scope: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_shape_liveness(
        ggml_backend_t cuda, ggml_backend_t cpu) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "16");
    void * session = create_direct_session(cuda, cpu);
    if (!session) {
        fprintf(stderr, "cache-shape-liveness: failed to create session\n");
        return false;
    }

    constexpr int64_t shape_a_in = 256;
    constexpr int64_t shape_b_in = 512;
    constexpr int64_t shape_out = 128;
    constexpr int64_t shape_experts = 64;
    const size_t shape_a_expert =
        ggml_row_size(GGML_TYPE_Q4_0, shape_a_in) * shape_out;
    const size_t shape_b_expert =
        ggml_row_size(GGML_TYPE_Q4_0, shape_b_in) * shape_out;
    std::vector<uint8_t> shape_a(shape_a_expert * shape_experts);
    std::vector<uint8_t> shape_b(shape_b_expert * shape_experts);

    ggml_moe_cache.session_enter(session);
    (void)direct_begin_ready(
            "blk.2.ffn_up_exps.weight", shape_a.data(), shape_a_expert,
            shape_a_in, shape_out, GGML_TYPE_Q4_0, shape_experts);
    ggml_moe_cache.invalidate(shape_a.data(), shape_a.size());
    const bool shape_b_ready = wait_for_direct_pool(
            "blk.3.ffn_up_exps.weight", shape_b.data(), shape_b_expert,
            shape_b_in, shape_out, GGML_TYPE_Q4_0, shape_experts);
    const bool shape_a_ready = wait_for_direct_pool(
            "blk.2.ffn_up_exps.weight", shape_a.data(), shape_a_expert,
            shape_a_in, shape_out, GGML_TYPE_Q4_0, shape_experts);
    ggml_moe_cache.session_leave(session);
    ggml_moe_cache.session_destroy(session);

    const bool ok = shape_b_ready && shape_a_ready;
    printf("cache-shape-liveness: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_route_override(
        ggml_backend_dev_t first_device,
        ggml_backend_t cuda, ggml_backend_t cpu) {
    ggml_backend_dev_t second_device =
        find_other_cuda_device(first_device);
    if (!second_device) {
        printf("cache-route-override: SKIP (one CUDA device)\n");
        return true;
    }
    ggml_backend_t second_cuda =
        ggml_backend_dev_init(second_device, nullptr);
    if (!second_cuda) {
        fprintf(stderr,
                "cache-route-override: failed to initialize second device\n");
        return false;
    }

    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "8");
    set_env("GGML_CUDA_MOE_CACHE_NDEV", "2");
    void * backends[] = { cuda, second_cuda, cpu };
    void * session = ggml_moe_cache.session_create(backends, 3);
    if (!session) {
        fprintf(stderr,
                "cache-route-override: failed to create session\n");
        ggml_backend_free(second_cuda);
        return false;
    }

    constexpr int64_t shape_a_in = 1024;
    constexpr int64_t shape_a_out = 160;
    constexpr int64_t shape_b_in = 512;
    constexpr int64_t shape_b_out = 384;
    constexpr int64_t shape_experts = 64;
    const size_t shape_a_expert =
        ggml_row_size(GGML_TYPE_Q4_0, shape_a_in) * shape_a_out;
    const size_t shape_b_expert =
        ggml_row_size(GGML_TYPE_Q4_0, shape_b_in) * shape_b_out;
    std::vector<uint8_t> shape_a(shape_a_expert * shape_experts);
    std::vector<uint8_t> shape_b(shape_b_expert * shape_experts);

    ggml_moe_cache.session_enter(session);
    const bool shape_a_ready = wait_for_direct_pool(
            "blk.4.ffn_up_exps.weight", shape_a.data(), shape_a_expert,
            shape_a_in, shape_a_out, GGML_TYPE_Q4_0, shape_experts);
    const bool shape_b_ready = wait_for_direct_pool(
            "blk.4.ffn_down_exps.weight", shape_b.data(), shape_b_expert,
            shape_b_in, shape_b_out, GGML_TYPE_Q4_0, shape_experts);
    ggml_moe_cache.session_leave(session);
    ggml_moe_cache.session_destroy(session);
    ggml_backend_free(second_cuda);

    const bool ok = shape_a_ready && shape_b_ready;
    printf("cache-route-override: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

static bool run_admission_policy_once(
        ggml_backend_t cuda, ggml_backend_t cpu,
        int new_expert_misses, log_capture & capture) {
    configure_cache(nullptr);
    set_env("GGML_CUDA_MOE_CACHE_BUDGET_MB", "8");
    set_env("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", "2");
    set_env("GGML_CUDA_MOE_CACHE_THROTTLE", "8");
    set_env("GGML_CUDA_MOE_CACHE_STATS", "0");
    // Scratch reservation scales with the configured max_batch (so a session
    // that will only ever see small batches doesn't pay for headroom it can
    // never use). This test wants a pool one slot short of n_expert (65) to
    // exercise eviction at capacity, so pin max_batch to the value the slot
    // math below assumes - configure_cache() sets it to 1, which would leave
    // a much bigger pool (66+, since less scratch is withheld from budget)
    // and never force the eviction this test is checking for.
    set_env("GGML_CUDA_MOE_CACHE_MAX_BATCH", "64");
    capture.clear();

    void * session = create_direct_session(cuda, cpu);
    if (!session) {
        return false;
    }

    constexpr int64_t policy_n_in = 1024;
    constexpr int64_t policy_n_out = 206;
    constexpr int64_t policy_n_expert = 65;
    const size_t expert_size =
        ggml_row_size(GGML_TYPE_Q4_0, policy_n_in) * policy_n_out;
    std::vector<uint8_t> weights(expert_size * policy_n_expert);
    const char * name = "blk.5.ffn_up_exps.weight";

    ggml_moe_cache.session_enter(session);
    bool ok = wait_for_direct_pool(
            name, weights.data(), expert_size,
            policy_n_in, policy_n_out,
            GGML_TYPE_Q4_0, policy_n_expert);
    for (int32_t expert = 0; expert < 64 && ok; expert++) {
        if (direct_plan_one(
                name, weights.data(), expert_size,
                policy_n_in, policy_n_out,
                GGML_TYPE_Q4_0, policy_n_expert, expert) != 0) {
            ok = false;
            break;
        }
        if (expert == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (direct_plan_one(
                name, weights.data(), expert_size,
                policy_n_in, policy_n_out,
                GGML_TYPE_Q4_0, policy_n_expert, expert) != 0) {
            ok = false;
            break;
        }

        bool hit = false;
        for (int attempt = 0; attempt < 100; attempt++) {
            if (direct_plan_one(
                    name, weights.data(), expert_size,
                    policy_n_in, policy_n_out,
                    GGML_TYPE_Q4_0, policy_n_expert, expert) == 1) {
                hit = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ok &= hit;
    }

    for (int miss = 0; miss < new_expert_misses && ok; miss++) {
        ok &= direct_plan_one(
                name, weights.data(), expert_size,
                policy_n_in, policy_n_out,
                GGML_TYPE_Q4_0, policy_n_expert, 64) == 0;
    }
    if (new_expert_misses == 8 && ok) {
        bool hit = false;
        for (int attempt = 0; attempt < 100; attempt++) {
            if (direct_plan_one(
                    name, weights.data(), expert_size,
                    policy_n_in, policy_n_out,
                    GGML_TYPE_Q4_0, policy_n_expert, 64) == 1) {
                hit = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ok &= hit;
    }

    ggml_moe_cache.session_leave(session);
    ggml_moe_cache.session_destroy(session);
    const std::string log = capture.get();
    const long long expected_enqueued =
        new_expert_misses == 8 ? 65 : 64;
    const long long expected_evictions =
        new_expert_misses == 8 ? 1 : 0;
    return ok &&
        max_field_value(log, "slots=") == 64 &&
        max_field_value(log, "enqueued=") == expected_enqueued &&
        max_field_value(log, "evictions=") == expected_evictions;
}

static bool run_admission_policy(
        ggml_backend_t cuda, ggml_backend_t cpu,
        log_capture & capture) {
    const bool seven = run_admission_policy_once(
            cuda, cpu, 7, capture);
    const bool eight = run_admission_policy_once(
            cuda, cpu, 8, capture);
    const bool ok = seven && eight;
    printf("cache-admission-policy: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

#ifdef MOE_CACHE_TEST_HAS_CUDA
// Exercises the full-layer prefill double buffer (prefill_prefetch/_wait)
// directly: correctness of the copy, ping-pong slot alternation across
// distinct tensors of the same shape, and that a wait() call correctly
// rejects a slot that holds a different tensor than the one asked for.
static bool run_prefill_prefetch(ggml_backend_t cuda, ggml_backend_t cpu) {
    if (!ggml_moe_cache.prefill_register_successor || !ggml_moe_cache.prefill_advance ||
        !ggml_moe_cache.prefill_wait || !ggml_moe_cache.prefill_release) {
        printf("prefill-prefetch: SKIP (not available)\n");
        return true;
    }
    // The register/advance split is gated behind this env var (default off -
    // see ggml_cuda_mul_mat_id's comment in ggml-cuda.cu for the full
    // rationale). This test drives it directly to validate the buffer
    // mechanism's correctness (data integrity, slot reuse, the
    // fork/consumed-event ordering); it uses a single stream as both the
    // "origin" (what a capture would use) and the "consumer", which doesn't
    // exercise actual CUDA-graph capture itself - that needs a real graph
    // capture/replay test, not written here.
    set_env("GGML_CUDA_MOE_PREFILL_BUFFER", "1");
    configure_cache(nullptr);
    void * session = create_direct_session(cuda, cpu);
    if (!session) {
        printf("prefill-prefetch: FAIL (no session)\n");
        return false;
    }
    ggml_moe_cache.session_enter(session);

    const size_t tensor_bytes = 4 * 1024 * 1024;
    std::vector<uint8_t> host_a(tensor_bytes);
    std::vector<uint8_t> host_b(tensor_bytes);
    for (size_t i = 0; i < tensor_bytes; i++) {
        host_a[i] = (uint8_t)(i & 0xFF);
        host_b[i] = (uint8_t)(~i & 0xFF);
    }
    // Dummy "this layer" keys - register_successor only ever uses these as
    // map keys (never dereferences them), so any distinct stable addresses
    // work; real callers pass an actual weight tensor's ->data.
    int key1 = 1, key2 = 2;

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    bool ok = true;

    ggml_moe_cache.prefill_register_successor(&key1, host_a.data(), tensor_bytes, 0);
    ggml_moe_cache.prefill_advance(&key1, stream);
    int slot_a = -1;
    const void * dev_a = ggml_moe_cache.prefill_wait(host_a.data(), tensor_bytes, stream, &slot_a);
    ok = ok && dev_a != nullptr && slot_a >= 0;

    std::vector<uint8_t> readback(tensor_bytes);
    if (ok) {
        cudaMemcpyAsync(readback.data(), dev_a, tensor_bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        ok = ok && memcmp(readback.data(), host_a.data(), tensor_bytes) == 0;
    }

    // A second tensor of the same shape, registered under a different key,
    // must ping-pong into the other slot, not clobber the first (still
    // referenced) one.
    if (ok) {
        ggml_moe_cache.prefill_register_successor(&key2, host_b.data(), tensor_bytes, 0);
        ggml_moe_cache.prefill_advance(&key2, stream);
    }

    // The first slot's data must still be exactly what host_a wrote - a real
    // regression here would mean the ping-pong allocator reused a live slot.
    const void * dev_a_again = ok
        ? ggml_moe_cache.prefill_wait(host_a.data(), tensor_bytes, stream, nullptr) : nullptr;
    ok = ok && dev_a_again == dev_a;

    // A host pointer that was never prefetched for this shape must fail -
    // no false positive from an unrelated live slot.
    if (ok) {
        std::vector<uint8_t> host_never(tensor_bytes);
        const void * never = ggml_moe_cache.prefill_wait(host_never.data(), tensor_bytes, stream, nullptr);
        ok = ok && never == nullptr;
    }

    int slot_b = -1;
    const void * dev_b = ok
        ? ggml_moe_cache.prefill_wait(host_b.data(), tensor_bytes, stream, &slot_b) : nullptr;
    ok = ok && dev_b != nullptr && dev_b != dev_a && slot_b >= 0 && slot_b != slot_a;
    if (ok) {
        cudaMemcpyAsync(readback.data(), dev_b, tensor_bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        ok = ok && memcmp(readback.data(), host_b.data(), tensor_bytes) == 0;
    }

    // Release both slots (as a real caller must, right after issuing its last
    // read of each), then advance both keys again with new successors - this
    // exercises the actual race the consumed-event exists to prevent: key1's
    // new successor (host_c below) reuses the physical buffer slot_a pointed
    // at, and must not be allowed to start copying until the release for
    // host_a was recorded. If the wait-on-consumed in moe_cache_prefill_advance
    // were missing or broken, this would still very likely pass on a fast/idle
    // GPU (the race window is small) - it's a regression guard, not a proof of
    // absence - but a wrong wait direction or a skipped record would show up
    // as corrupted readback contents here often enough to catch in practice.
    if (ok) {
        ggml_moe_cache.prefill_release(host_a.data(), tensor_bytes, slot_a, stream);
        ggml_moe_cache.prefill_release(host_b.data(), tensor_bytes, slot_b, stream);
    }

    std::vector<uint8_t> host_c(tensor_bytes);
    std::vector<uint8_t> host_d(tensor_bytes);
    for (size_t i = 0; i < tensor_bytes; i++) {
        host_c[i] = (uint8_t)((i * 3 + 7) & 0xFF);
        host_d[i] = (uint8_t)((i * 5 + 11) & 0xFF);
    }

    if (ok) {
        ggml_moe_cache.prefill_register_successor(&key1, host_c.data(), tensor_bytes, 0);
        ggml_moe_cache.prefill_advance(&key1, stream);
    }
    int slot_c = -1;
    const void * dev_c = ok
        ? ggml_moe_cache.prefill_wait(host_c.data(), tensor_bytes, stream, &slot_c) : nullptr;
    ok = ok && dev_c == dev_a && slot_c == slot_a; // same physical buffer, reused
    if (ok) {
        cudaMemcpyAsync(readback.data(), dev_c, tensor_bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        ok = ok && memcmp(readback.data(), host_c.data(), tensor_bytes) == 0;
    }
    if (ok) {
        ggml_moe_cache.prefill_release(host_c.data(), tensor_bytes, slot_c, stream);
    }

    if (ok) {
        ggml_moe_cache.prefill_register_successor(&key2, host_d.data(), tensor_bytes, 0);
        ggml_moe_cache.prefill_advance(&key2, stream);
    }
    int slot_d = -1;
    const void * dev_d = ok
        ? ggml_moe_cache.prefill_wait(host_d.data(), tensor_bytes, stream, &slot_d) : nullptr;
    ok = ok && dev_d == dev_b && slot_d == slot_b;
    if (ok) {
        cudaMemcpyAsync(readback.data(), dev_d, tensor_bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        ok = ok && memcmp(readback.data(), host_d.data(), tensor_bytes) == 0;
    }
    if (ok) {
        ggml_moe_cache.prefill_release(host_d.data(), tensor_bytes, slot_d, stream);
    }

    cudaStreamDestroy(stream);
    ggml_moe_cache.session_leave(session);
    ggml_moe_cache.session_destroy(session);
    printf("prefill-prefetch: %s\n", ok ? "OK" : "FAIL");
    return ok;
}
#endif // MOE_CACHE_TEST_HAS_CUDA

} // namespace

int main() {
    log_capture capture;
    ggml_log_set(log_callback, &capture);

    ggml_backend_dev_t cuda_device = find_cuda_device();
    if (!cuda_device) {
        printf("SKIP: CUDA backend unavailable\n");
        return 0;
    }
    ggml_backend_reg_t cuda_reg =
        ggml_backend_dev_backend_reg(cuda_device);

    ggml_backend_t cuda = ggml_backend_dev_init(cuda_device, nullptr);
    ggml_backend_t cpu = init_cpu_backend();
    if (!cuda || !cpu) {
        fprintf(stderr, "failed to initialize CUDA and CPU backends\n");
        if (cuda) {
            ggml_backend_free(cuda);
        }
        if (cpu) {
            ggml_backend_free(cpu);
        }
        return 1;
    }
    ggml_init_params static_params = {
        8 * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    ggml_context * static_ctx = ggml_init(static_params);
    if (!static_ctx) {
        fprintf(stderr, "failed to create tensor context\n");
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return 1;
    }

    ggml_tensor * weights = ggml_new_tensor_3d(
            static_ctx, GGML_TYPE_Q4_0, n_in, n_out, n_expert);
    ggml_tensor * ids = ggml_new_tensor_2d(
            static_ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * activations = ggml_new_tensor_3d(
            static_ctx, GGML_TYPE_F32, n_in, 1, n_tokens);
    ggml_set_name(weights, "blk.0.ffn_up_exps.weight");
    ggml_set_name(ids, "moe_cache_test_ids");
    ggml_set_name(activations, "moe_cache_test_activations");

    ggml_backend_buffer_t static_buffer =
        ggml_backend_alloc_ctx_tensors(static_ctx, cpu);
    if (!static_buffer) {
        fprintf(stderr, "failed to allocate CPU tensors\n");
        ggml_free(static_ctx);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return 1;
    }
    ggml_backend_buffer_set_usage(
            static_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<float> weights_f32(ggml_nelements(weights));
    for (size_t index = 0; index < weights_f32.size(); index++) {
        weights_f32[index] =
            0.15f * std::sin((float) (index % 997) * 0.017f) +
            0.05f * std::cos((float) (index % 431) * 0.031f);
    }
    std::vector<uint8_t> weights_q4(ggml_nbytes(weights));
    const size_t quantized = ggml_quantize_chunk(
            GGML_TYPE_Q4_0, weights_f32.data(), weights_q4.data(),
            0, n_out * n_expert, n_in, nullptr);
    if (quantized != weights_q4.size()) {
        fprintf(stderr, "unexpected quantized size: %zu != %zu\n",
                quantized, weights_q4.size());
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return 1;
    }
    ggml_backend_tensor_set(
            weights, weights_q4.data(), 0, weights_q4.size());

    const int32_t ids_data[n_used] = { 0, 1 };
    ggml_backend_tensor_set(ids, ids_data, 0, sizeof(ids_data));
    std::vector<float> activation_data(ggml_nelements(activations));
    for (size_t index = 0; index < activation_data.size(); index++) {
        activation_data[index] =
            0.4f * std::sin((float) index * 0.07f) -
            0.2f * std::cos((float) index * 0.11f);
    }
    ggml_backend_tensor_set(
            activations, activation_data.data(), 0,
            activation_data.size() * sizeof(float));

    test_graph graph = make_graph(cpu, weights, activations, ids);
    if (!graph.ctx || !graph.buffer) {
        fprintf(stderr, "failed to create test graph\n");
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return 1;
    }

    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, graph.graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "CPU reference compute failed\n");
        free_graph(graph);
        ggml_backend_buffer_free(static_buffer);
        ggml_free(static_ctx);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        return 1;
    }
    std::vector<float> reference(ggml_nelements(graph.out));
    ggml_backend_tensor_get(
            graph.out, reference.data(), 0, reference.size() * sizeof(float));

    bool ok = true;
    ok &= run_scenario("cache-hit", nullptr, cuda, cpu, graph, reference, capture);
    ok &= run_scenario("dispatch-fallback", "dispatch", cuda, cpu, graph, reference, capture);
    ok &= run_scenario("collect-fallback", "collect", cuda, cpu, graph, reference, capture);
    ok &= run_scenario("insert-fallback", "insert", cuda, cpu, graph, reference, capture);
    ok &= run_scenario("slab-fallback", "slab", cuda, cpu, graph, reference, capture);
    const size_t expert_size = ggml_nbytes(weights) / n_expert;
    ok &= run_precensus_invalidation(
            cuda, cpu, graph, weights,
            weights_q4.data() + (n_expert - 1) * expert_size,
            expert_size, reference, capture);

    const int32_t repeated_ids[n_used] = { 0, 0 };
    ggml_backend_tensor_set(ids, repeated_ids, 0, sizeof(repeated_ids));
    set_env("GGML_CUDA_MOE_CACHE", "0");
    if (ggml_backend_graph_compute(cpu, graph.graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cache-invalidate: initial CPU reference compute failed\n");
        ok = false;
    } else {
        std::vector<float> old_reference(ggml_nelements(graph.out));
        ggml_backend_tensor_get(
                graph.out, old_reference.data(), 0,
                old_reference.size() * sizeof(float));

        std::vector<float> replacement_f32(n_in);
        for (size_t index = 0; index < replacement_f32.size(); index++) {
            replacement_f32[index] =
                1.25f + 0.3f * std::sin((float) index * 0.041f);
        }
        std::vector<uint8_t> replacement_q4(
                ggml_row_size(GGML_TYPE_Q4_0, n_in));
        const size_t replacement_size = ggml_quantize_chunk(
                GGML_TYPE_Q4_0, replacement_f32.data(), replacement_q4.data(),
                0, 1, n_in, nullptr);
        if (replacement_size != replacement_q4.size()) {
            fprintf(stderr, "cache-invalidate: unexpected replacement size\n");
            ok = false;
        } else {
            ok &= run_invalidation_scenario(
                    cuda, cpu, graph, weights, replacement_q4,
                    old_reference, capture);
        }
    }

    stress_fixture stress;
    if (!init_stress_fixture(stress, cpu)) {
        fprintf(stderr, "failed to initialize cache stress fixture\n");
        ok = false;
    } else {
        ok &= run_concurrent_sessions(
                cuda_device, cuda, cpu, stress, capture);
        ok &= run_repeated_lifecycle(cuda, cpu, stress, capture);
        ok &= run_fill_invalidation(cuda, cpu, stress, capture);
    }
    free_stress_fixture(stress);
    ok &= run_scope_isolation(cuda, cpu, weights);
    ok &= run_shape_liveness(cuda, cpu);
    ok &= run_route_override(cuda_device, cuda, cpu);
    ok &= run_admission_policy(cuda, cpu, capture);
#ifdef MOE_CACHE_TEST_HAS_CUDA
    ok &= run_prefill_prefetch(cuda, cpu);
#endif

    free_graph(graph);
    ggml_backend_free(cuda);
#ifdef GGML_BACKEND_DL
    ggml_backend_unload(cuda_reg);
    ggml_backend_buffer_free(static_buffer);
    static_buffer = nullptr;
    printf("cache-backend-unload: OK\n");

    ggml_backend_dev_t reloaded_device = find_cuda_device();
    ggml_backend_t reloaded_cuda = reloaded_device
        ? ggml_backend_dev_init(reloaded_device, nullptr) : nullptr;
    configure_cache(nullptr);
    void * reloaded_session = reloaded_cuda
        ? create_direct_session(reloaded_cuda, cpu) : nullptr;
    const bool reload_ok = reloaded_session != nullptr;
    if (reloaded_session) {
        ggml_moe_cache.session_destroy(reloaded_session);
    }
    if (reloaded_cuda) {
        ggml_backend_reg_t reloaded_reg =
            ggml_backend_dev_backend_reg(reloaded_device);
        ggml_backend_free(reloaded_cuda);
        ggml_backend_unload(reloaded_reg);
    }
    printf("cache-backend-reload: %s\n", reload_ok ? "OK" : "FAIL");
    ok &= reload_ok;
#else
    (void) cuda_reg;
    printf("cache-backend-unload: SKIP (static backend)\n");
#endif
    if (static_buffer) {
        ggml_backend_buffer_free(static_buffer);
    }
    ggml_free(static_ctx);
    ggml_quantize_free();
    ggml_backend_free(cpu);
    ggml_log_set(nullptr, nullptr);
    return ok ? 0 : 1;
}
