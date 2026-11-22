// llama-frspec-vocab-trim: reads the real-traffic token-frequency
// histogram a running llama-server accumulates (server-token-freq.h,
// --token-freq-log, off by default), ranks tokens by how often they were
// actually generated, and writes a small sidecar mapping file (the "d2t"
// convention - trimmed-vocab position -> original token id) that a
// model's MTP path can load at server startup to trim its draft-vocab
// projection to just the tokens that matter for this deployment's real
// traffic (FR-Spec-style, see docs/moe-cache-colibri-notes.md).
//
// The histogram is keyed by the server's *target* model path (-m), not
// the draft's (-md): it records confirmed output tokens, which are
// always in the target's vocab space regardless of which draft produced
// the (possibly rejected) candidates for them.
//
// Deliberately standalone and re-runnable, not baked into the server
// itself: regenerate any time (a cron job, or just by hand after enough
// real traffic has accumulated) as the deployment's actual workload
// becomes clearer, without ever touching the original draft checkpoint.
// The original file is never read or written by this tool at all - only
// the histogram (input) and the sidecar mapping (output).

#include "common.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

// Mirrors server_token_freq_logger::path_for_model (tools/server/server-token-freq.cpp)
// exactly - duplicated rather than linked against the server-context
// library, since that pulls in the whole HTTP server stack for a five-line
// hash. Both sides must stay in agreement; if this ever drifts, "no
// histogram found" is a loud, obvious failure, not silent corruption.
static std::string histogram_path_for_model(const std::string & path_model) {
    const size_t h = std::hash<std::string>{}(path_model);
    char buf[32];
    snprintf(buf, sizeof(buf), "%016zx", h);
    return fs_get_cache_directory() + "token-freq-" + buf + ".json";
}

static void print_usage() {
    fprintf(stderr,
        "usage: llama-frspec-vocab-trim --model <path> --top-n <N> -o <output.d2t>\n"
        "       llama-frspec-vocab-trim --histogram <path> --top-n <N> -o <output.d2t>\n"
        "\n"
        "Reads the real-traffic token-frequency histogram a llama-server instance\n"
        "accumulated (--token-freq-log, off by default), ranks tokens by real\n"
        "generation frequency, and writes the top N as a sidecar vocab-trim mapping\n"
        "file for FR-Spec-style MTP draft-vocab trimming.\n"
        "\n"
        "  --model <path>        the server's *target* model path (-m), NOT the draft's\n"
        "                        (-md) - the histogram records confirmed output tokens,\n"
        "                        which live in the target's vocab space and are keyed\n"
        "                        by the target's path on the server side; used to locate\n"
        "                        the histogram file via the same hash convention the\n"
        "                        server uses\n"
        "  --histogram <path>    read this histogram file directly instead of deriving\n"
        "                        the path from --model\n"
        "  --top-n <N>           number of most-frequent tokens to keep (required)\n"
        "  -o, --output <path>   where to write the sidecar mapping (required)\n");
}

int main(int argc, char ** argv) {
    std::string path_model;
    std::string path_histogram;
    std::string path_output;
    int64_t top_n = -1;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", arg.c_str());
                exit(1);
            }
            return argv[++i];
        };
        if (arg == "--model" || arg == "--model-draft") {
            path_model = next();
        } else if (arg == "--histogram") {
            path_histogram = next();
        } else if (arg == "--top-n") {
            top_n = std::stoll(next());
        } else if (arg == "-o" || arg == "--output") {
            path_output = next();
        } else if (arg == "-h" || arg == "--help") {
            print_usage();
            return 0;
        } else {
            fprintf(stderr, "error: unknown argument %s\n", arg.c_str());
            print_usage();
            return 1;
        }
    }

    if (path_model.empty() && path_histogram.empty()) {
        fprintf(stderr, "error: need --model or --histogram\n");
        print_usage();
        return 1;
    }
    if (top_n <= 0) {
        fprintf(stderr, "error: --top-n must be a positive integer\n");
        return 1;
    }
    if (path_output.empty()) {
        fprintf(stderr, "error: --output is required\n");
        return 1;
    }

    const std::string histogram_path = !path_histogram.empty()
        ? path_histogram
        : histogram_path_for_model(path_model);

    std::ifstream f(histogram_path);
    if (!f.good()) {
        fprintf(stderr,
            "error: no histogram found at %s\n"
            "Run a real llama-server instance against this model first with --token-freq-log "
            "and let it serve some real traffic - this tool has nothing "
            "to rank without at least some generated tokens to learn from.\n",
            histogram_path.c_str());
        return 1;
    }

    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception & e) {
        fprintf(stderr, "error: failed to parse histogram %s: %s\n", histogram_path.c_str(), e.what());
        return 1;
    }

    std::vector<std::pair<int64_t, int64_t>> counts; // (token_id, count)
    counts.reserve(j.size());
    for (auto it = j.begin(); it != j.end(); ++it) {
        counts.emplace_back(std::stoll(it.key()), it.value().get<int64_t>());
    }

    if (counts.empty()) {
        fprintf(stderr, "error: histogram %s is empty - no real traffic recorded yet\n", histogram_path.c_str());
        return 1;
    }

    // Descending by count; ties broken by token id for a stable, reproducible
    // ordering across re-runs on an unchanged histogram.
    std::sort(counts.begin(), counts.end(), [](const auto & a, const auto & b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });

    const int64_t n_keep = std::min<int64_t>(top_n, (int64_t) counts.size());
    if (n_keep < top_n) {
        fprintf(stderr,
            "warning: histogram only has %zu distinct tokens, fewer than the requested "
            "top-n=%lld - writing all %lld of them. More real traffic will widen this "
            "over time; re-run once more has accumulated.\n",
            counts.size(), (long long) top_n, (long long) n_keep);
    }

    std::ofstream out(path_output);
    if (!out.good()) {
        fprintf(stderr, "error: could not open %s for writing\n", path_output.c_str());
        return 1;
    }
    for (int64_t i = 0; i < n_keep; i++) {
        out << counts[(size_t) i].first << "\n";
    }

    const int64_t total_observations = [&]() {
        int64_t sum = 0;
        for (const auto & c : counts) sum += c.second;
        return sum;
    }();
    const int64_t kept_observations = [&]() {
        int64_t sum = 0;
        for (int64_t i = 0; i < n_keep; i++) sum += counts[(size_t) i].second;
        return sum;
    }();
    const double coverage = total_observations > 0 ? 100.0 * (double) kept_observations / (double) total_observations : 0.0;

    printf("wrote %lld tokens to %s (from %zu distinct observed tokens, %lld total observations)\n"
           "top %lld tokens cover %.1f%% of all real traffic seen so far - this is the "
           "expected draft-hit-rate ceiling for the trimmed vocab\n",
           (long long) n_keep, path_output.c_str(), counts.size(), (long long) total_observations,
           (long long) n_keep, coverage);

    return 0;
}
