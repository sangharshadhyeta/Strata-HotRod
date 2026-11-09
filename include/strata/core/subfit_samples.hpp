// include/strata/core/subfit_samples.hpp - SPEC-substitute-fit stage 1a: the sample file the engine writes (STRATA_SUBFIT_LOG=1)
// and tools/subfit_fit reads.  Nothing here touches CUDA; the device side is in kernels/subfit_capture.hpp.
//
// File layout (little endian):
//   header: magic u32 ("STSB"), n_embd u32, n_layers u32, version u32 (1)
//   then records, each starting with a tag u8:
//     tag 1 (stand-in sample, cut window + exact successor): layer u16, e u16, s u16, rank u8, x[N] bf16, y_s[N] bf16, y_e[N] bf16
//     tag 2 (probe sample, a resident exact pick at ranks 0-9 of row 0): layer u16, e u16, rank u8, w f32, x[N] bf16, y_e[N] bf16
//       x    = the expert's input (the layer's normed input row), y_e = the expert's raw output before the router weight,
//       w    = the pick's router weight (the combine multiplies y_e by it), e = the true expert, s = none (the tool chooses)
// The file is capped at kSubfitCapBytes: once a record would pass the cap nothing more is written (logged once).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::core::subfit {

constexpr uint32_t kMagic = 0x42535453u;   // bytes "STSB" in file order
constexpr uint32_t kVersion = 1;
constexpr uint64_t kCapBytes = 300ull * 1024ull * 1024ull;
constexpr int kProbeEveryDefault = 64;

/// STRATA_SUBFIT_LOG set to a non-zero integer (the logger's only switch).
bool log_on_env(const char* v);
/// STRATA_SUBFIT_PROBE_EVERY: decode windows per probe window (default 64, at least 1).
int probe_every_env(const char* v);

uint16_t f32_to_bf16(float f);
float bf16_to_f32(uint16_t b);

struct Header {
    uint32_t magic = kMagic, n_embd = 0, n_layers = 0, version = kVersion;
};

/// One record read back (tag 1 fills s and ys; tag 2 fills w).
struct Record {
    uint8_t tag = 0;
    uint16_t layer = 0, e = 0, s = 0;
    uint8_t rank = 0;
    float w = 0.0f;
    std::vector<float> x, ys, ye;
};

/// Appends records to one sample file.  open() writes the header to a new file, or appends to a file whose header matches.
class SampleWriter {
public:
    SampleWriter() = default;
    ~SampleWriter();
    SampleWriter(const SampleWriter&) = delete;
    SampleWriter& operator=(const SampleWriter&) = delete;

    bool open(const std::string& path, uint32_t n_embd, uint32_t n_layers, std::string& err);
    void close();
    bool is_open() const { return f_ != nullptr; }

    /// Each returns false when nothing was written (the cap is reached, or the file is not open).
    bool write_stand_in(uint16_t layer, uint16_t e, uint16_t s, uint8_t rank, const float* x, const float* ys, const float* ye);
    bool write_probe(uint16_t layer, uint16_t e, uint8_t rank, float w, const float* x, const float* ye);

    /// The file's size limit (default kCapBytes); tests lower it.
    void set_cap(uint64_t bytes) { cap_ = bytes; }
    uint64_t bytes() const { return bytes_; }
    uint64_t samples_total() const { return total_; }
    /// The samples written since the last call (the request-end line), then reset.
    uint64_t take_request_count();
    bool capped() const { return capped_; }

private:
    bool put_record(const std::vector<uint8_t>& rec);
    FILE* f_ = nullptr;
    uint32_t n_ = 0, layers_ = 0;
    uint64_t bytes_ = 0, total_ = 0, req_ = 0, cap_ = kCapBytes;
    bool capped_ = false, cap_logged_ = false;
};

/// Reads a sample file front to back.
class SampleReader {
public:
    SampleReader() = default;
    ~SampleReader();
    SampleReader(const SampleReader&) = delete;
    SampleReader& operator=(const SampleReader&) = delete;

    bool open(const std::string& path, std::string& err);
    const Header& header() const { return h_; }
    /// false at the end of the file (or a truncated record: err is set then).
    bool next(Record& r, std::string& err);

private:
    FILE* f_ = nullptr;
    Header h_;
};

}  // namespace strata::core::subfit
