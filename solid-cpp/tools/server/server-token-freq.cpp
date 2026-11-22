#include "server-token-freq.h"

#include "common.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <functional>

std::string server_token_freq_logger::path_for_model(const std::string & path_model) {
    // Keyed by a hash of the model path (not the raw path itself, which can
    // contain characters unsafe for a filename) so different models never
    // share a histogram file.
    const size_t h = std::hash<std::string>{}(path_model);
    char buf[32];
    snprintf(buf, sizeof(buf), "%016zx", h);
    return fs_get_cache_directory() + "token-freq-" + buf + ".json";
}

server_token_freq_logger::server_token_freq_logger(const std::string & path_model, int32_t n_vocab) {
    if (n_vocab <= 0) {
        return; // enabled() stays false - nothing sensible to count into
    }
    path = path_for_model(path_model);
    counts.assign((size_t) n_vocab, 0);

    std::ifstream f(path);
    if (f.good()) {
        try {
            nlohmann::json j;
            f >> j;
            for (auto it = j.begin(); it != j.end(); ++it) {
                const int64_t tok = std::stoll(it.key());
                if (tok >= 0 && tok < n_vocab) {
                    counts[(size_t) tok] = it.value().get<int64_t>();
                }
            }
            LOG_INF("%s: loaded existing token-frequency histogram (%s) - counts accumulate "
                    "across restarts, reflecting real traffic over this deployment's whole life\n",
                    __func__, path.c_str());
        } catch (const std::exception & e) {
            LOG_WRN("%s: failed to load existing token-frequency histogram (%s): %s - starting fresh\n",
                    __func__, path.c_str(), e.what());
        }
    }
}

void server_token_freq_logger::record(int32_t token_id) {
    if (!enabled() || token_id < 0 || (size_t) token_id >= counts.size()) {
        return;
    }
    counts[(size_t) token_id]++;
    if (++since_flush >= flush_every) {
        flush();
    }
}

void server_token_freq_logger::flush() {
    if (!enabled()) {
        return;
    }
    since_flush = 0;
    nlohmann::json j = nlohmann::json::object();
    for (size_t i = 0; i < counts.size(); i++) {
        if (counts[i] > 0) {
            j[std::to_string(i)] = counts[i];
        }
    }
    fs_create_directory_with_parents(fs_get_cache_directory());
    std::ofstream out(path);
    out << j.dump();
}
