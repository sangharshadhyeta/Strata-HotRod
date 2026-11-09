// src/core/subfit_samples.cpp - the sample file of SPEC-substitute-fit stage 1a (see the header for the layout).
#include "strata/core/subfit_samples.hpp"

#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

namespace strata::core::subfit {

bool log_on_env(const char* v) { return v != nullptr && std::atoi(v) != 0; }

int probe_every_env(const char* v) {
    if (v == nullptr) return kProbeEveryDefault;
    const int n = std::atoi(v);
    return n >= 1 ? n : kProbeEveryDefault;
}

uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40u);   // NaN stays NaN
    u += 0x7fffu + ((u >> 16) & 1u);                                                // round to nearest even
    return (uint16_t) (u >> 16);
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

namespace {
void put_u16(std::vector<uint8_t>& o, uint16_t v) { o.push_back((uint8_t) (v & 0xff)); o.push_back((uint8_t) (v >> 8)); }
void put_u32(std::vector<uint8_t>& o, uint32_t v) { for (int i = 0; i < 4; ++i) o.push_back((uint8_t) (v >> (8 * i))); }
void put_f32(std::vector<uint8_t>& o, float f) { uint32_t u; std::memcpy(&u, &f, 4); put_u32(o, u); }
void put_bf16s(std::vector<uint8_t>& o, const float* v, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) put_u16(o, f32_to_bf16(v[i]));
}
uint16_t get_u16(const uint8_t* p) { return (uint16_t) (p[0] | (p[1] << 8)); }
uint32_t get_u32(const uint8_t* p) { return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24); }
float get_f32(const uint8_t* p) { const uint32_t u = get_u32(p); float f; std::memcpy(&f, &u, 4); return f; }
}  // namespace

SampleWriter::~SampleWriter() { close(); }

void SampleWriter::close() {
    if (f_ != nullptr) std::fclose(f_);
    f_ = nullptr;
}

bool SampleWriter::open(const std::string& path, uint32_t n_embd, uint32_t n_layers, std::string& err) {
    close();
    n_ = n_embd;
    layers_ = n_layers;
    struct stat st{};
    const bool exists = ::stat(path.c_str(), &st) == 0 && st.st_size > 0;
    if (exists) {
        FILE* r = std::fopen(path.c_str(), "rb");
        uint8_t hb[16];
        if (r == nullptr || std::fread(hb, 1, sizeof hb, r) != sizeof hb) {
            if (r) std::fclose(r);
            err = "subfit: cannot read the header of " + path;
            return false;
        }
        std::fclose(r);
        if (get_u32(hb) != kMagic || get_u32(hb + 4) != n_embd || get_u32(hb + 8) != n_layers || get_u32(hb + 12) != kVersion) {
            err = "subfit: " + path + " holds samples of another model or format; not appended to";
            return false;
        }
        f_ = std::fopen(path.c_str(), "ab");
        bytes_ = (uint64_t) st.st_size;
    } else {
        f_ = std::fopen(path.c_str(), "wb");
        if (f_ != nullptr) {
            std::vector<uint8_t> hb;
            put_u32(hb, kMagic);
            put_u32(hb, n_embd);
            put_u32(hb, n_layers);
            put_u32(hb, kVersion);
            if (std::fwrite(hb.data(), 1, hb.size(), f_) != hb.size()) {
                close();
                err = "subfit: cannot write the header of " + path;
                return false;
            }
            std::fflush(f_);
        }
        bytes_ = 16;
    }
    if (f_ == nullptr) {
        err = "subfit: cannot open " + path;
        return false;
    }
    return true;
}

bool SampleWriter::put_record(const std::vector<uint8_t>& rec) {
    if (f_ == nullptr) return false;
    if (bytes_ + rec.size() > cap_) {
        capped_ = true;
        if (!cap_logged_) {
            cap_logged_ = true;
            std::fprintf(stderr, "strata subfit: the sample file reached %llu MB; no more samples are written\n",
                         (unsigned long long) (cap_ >> 20));
        }
        return false;
    }
    if (std::fwrite(rec.data(), 1, rec.size(), f_) != rec.size()) return false;
    std::fflush(f_);
    bytes_ += rec.size();
    ++total_;
    ++req_;
    return true;
}

bool SampleWriter::write_stand_in(uint16_t layer, uint16_t e, uint16_t s, uint8_t rank, const float* x, const float* ys, const float* ye) {
    std::vector<uint8_t> rec;
    rec.reserve(8 + 6 * (size_t) n_);
    rec.push_back(1);
    put_u16(rec, layer);
    put_u16(rec, e);
    put_u16(rec, s);
    rec.push_back(rank);
    put_bf16s(rec, x, n_);
    put_bf16s(rec, ys, n_);
    put_bf16s(rec, ye, n_);
    return put_record(rec);
}

bool SampleWriter::write_probe(uint16_t layer, uint16_t e, uint8_t rank, float w, const float* x, const float* ye) {
    std::vector<uint8_t> rec;
    rec.reserve(10 + 4 * (size_t) n_);
    rec.push_back(2);
    put_u16(rec, layer);
    put_u16(rec, e);
    rec.push_back(rank);
    put_f32(rec, w);
    put_bf16s(rec, x, n_);
    put_bf16s(rec, ye, n_);
    return put_record(rec);
}

uint64_t SampleWriter::take_request_count() {
    const uint64_t r = req_;
    req_ = 0;
    return r;
}

SampleReader::~SampleReader() {
    if (f_ != nullptr) std::fclose(f_);
}

bool SampleReader::open(const std::string& path, std::string& err) {
    if (f_ != nullptr) std::fclose(f_);
    f_ = std::fopen(path.c_str(), "rb");
    uint8_t hb[16];
    if (f_ == nullptr || std::fread(hb, 1, sizeof hb, f_) != sizeof hb) {
        err = "subfit: cannot read " + path;
        return false;
    }
    h_.magic = get_u32(hb);
    h_.n_embd = get_u32(hb + 4);
    h_.n_layers = get_u32(hb + 8);
    h_.version = get_u32(hb + 12);
    if (h_.magic != kMagic || h_.version != kVersion) {
        err = "subfit: " + path + " is not a version 1 sample file";
        return false;
    }
    return true;
}

bool SampleReader::next(Record& r, std::string& err) {
    if (f_ == nullptr) return false;
    uint8_t tag;
    if (std::fread(&tag, 1, 1, f_) != 1) return false;   // end of file
    const uint32_t n = h_.n_embd;
    auto read_bytes = [&](std::vector<uint8_t>& buf, size_t k) {
        buf.resize(k);
        return std::fread(buf.data(), 1, k, f_) == k;
    };
    auto read_bf16 = [&](const std::vector<uint8_t>& buf, std::vector<float>& out) {
        out.resize(n);
        for (uint32_t i = 0; i < n; ++i) out[i] = bf16_to_f32(get_u16(buf.data() + 2 * (size_t) i));
    };
    std::vector<uint8_t> buf;
    r = Record{};
    r.tag = tag;
    if (tag == 1) {
        if (!read_bytes(buf, 7)) { err = "subfit: truncated record"; return false; }
        r.layer = get_u16(buf.data());
        r.e = get_u16(buf.data() + 2);
        r.s = get_u16(buf.data() + 4);
        r.rank = buf[6];
        if (!read_bytes(buf, 6 * (size_t) n)) { err = "subfit: truncated record"; return false; }
        read_bf16(buf, r.x);   // x, then ys, then ye in the same buffer
        r.ys.resize(n);
        r.ye.resize(n);
        for (uint32_t i = 0; i < n; ++i) r.ys[i] = bf16_to_f32(get_u16(buf.data() + 2 * ((size_t) n + i)));
        for (uint32_t i = 0; i < n; ++i) r.ye[i] = bf16_to_f32(get_u16(buf.data() + 2 * ((size_t) 2 * n + i)));
        return true;
    }
    if (tag == 2) {
        if (!read_bytes(buf, 9)) { err = "subfit: truncated record"; return false; }
        r.layer = get_u16(buf.data());
        r.e = get_u16(buf.data() + 2);
        r.rank = buf[4];
        r.w = get_f32(buf.data() + 5);
        if (!read_bytes(buf, 4 * (size_t) n)) { err = "subfit: truncated record"; return false; }
        read_bf16(buf, r.x);
        r.ye.resize(n);
        for (uint32_t i = 0; i < n; ++i) r.ye[i] = bf16_to_f32(get_u16(buf.data() + 2 * ((size_t) n + i)));
        return true;
    }
    err = "subfit: unknown record tag " + std::to_string((int) tag);
    return false;
}

}  // namespace strata::core::subfit
