#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Accumulates a persistent, per-token-id generation-frequency histogram
// driven by real serving traffic (not a synthetic corpus), used to derive
// a frequency-ranked vocabulary subset for FR-Spec-style MTP draft-vocab
// trimming (see docs/moe-cache-colibri-notes.md, "FR-Spec draft-vocab
// trimming"). Loads any existing on-disk histogram for this model at
// construction and keeps accumulating into it - the counts reflect real
// traffic over the deployment's whole life, not a single session, and
// survive restarts. Record is a plain array increment (no hashing, no
// lock - called from the single-threaded decode loop only), so the
// per-token cost is negligible next to actual generation.
struct server_token_freq_logger {
    server_token_freq_logger(const std::string & path_model, int32_t n_vocab);

    void record(int32_t token_id);

    // Persists now; also called periodically from record() so a crash
    // between flushes loses at most a small window of counts, not
    // everything since the last restart.
    void flush();

    static std::string path_for_model(const std::string & path_model);

    bool enabled() const { return !path.empty(); }

private:
    std::string path;
    std::vector<int64_t> counts;
    int64_t since_flush = 0;
    static constexpr int64_t flush_every = 4096;
};
