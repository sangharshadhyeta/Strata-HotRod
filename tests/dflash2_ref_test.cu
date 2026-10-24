// tests/dflash2_ref_test.cu - the native DFlash 2 drafter against the reference dumps of stage A (22 Oct 2026).
//
//   dflash2_ref_test <drafter.gguf> <dump dir> <qwen-head dir> [--strict]
//
// The dump dir holds round<k>.npz from tools/drafter/dflash_ref_dump.py (z-lab's model on the CPU, bf16 weights).  The
// test feeds each round's context features as the dump has them, proposes the block with the dump's anchor, and compares
// per decoder layer (max abs diff, cosine), the context's hidden_norm(fc(...)), the final hidden rows, the selector's
// top-16 candidate sets and the draft tokens.  The embedding and the head come from the qwen-head safetensors on the host
// (test only; the engine passes its own).  The GGUF is Q4_K_M and the reference is bf16, so the numbers differ by the
// quantization: the verdict is printed, and --strict makes a draft-token mismatch a failure.  Exit 1 on errors.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../src/drafter/dflash2.hpp"

using strata::drafter::Dflash2;
using strata::drafter::Dflash2Trace;

namespace {

std::vector<char> read_file(const std::string& path, bool& ok) {
    std::ifstream f(path, std::ios::binary);
    ok = (bool) f;
    if (!ok) return {};
    f.seekg(0, std::ios::end);
    std::vector<char> buf((size_t) f.tellg());
    f.seekg(0);
    f.read(buf.data(), (std::streamsize) buf.size());
    return buf;
}

// ---- npz (zip of .npy, stored entries) ------------------------------------------------------------------------------

struct NpyView {
    std::string descr;             // "<f4", "<i8" or "<u2"
    std::vector<int64_t> shape;    // empty for a scalar
    const char* data = nullptr;
    size_t count = 1;
};

uint16_t u16(const char* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}
uint32_t u32(const char* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

struct Npz {
    std::vector<char> buf;
    std::map<std::string, std::pair<size_t, size_t>> entries;   // name -> (offset, size) in buf

    bool open(const std::string& path, std::string& err) {
        bool ok = false;
        buf = read_file(path, ok);
        if (!ok) {
            err = "cannot read " + path;
            return false;
        }
        size_t at = 0;
        while (at + 30 <= buf.size() && u32(buf.data() + at) == 0x04034b50u) {
            const char* h = buf.data() + at;
            const uint16_t method = u16(h + 8);
            uint64_t comp = u32(h + 18);
            const uint16_t nlen = u16(h + 26), xlen = u16(h + 28);
            if (method != 0) {
                err = path + ": compressed npz entries are not supported";
                return false;
            }
            std::string name(h + 30, nlen);
            if (comp == 0xFFFFFFFFu) {
                // zip64 (numpy's savez): the local header holds 0xFFFFFFFF and the real sizes sit in
                // extra field 0x0001 as uncompressed (8 bytes) then compressed (8 bytes).
                const char* x = h + 30 + nlen;
                size_t q = 0;
                while (q + 4 <= xlen) {
                    const uint16_t id = u16(x + q), sz = u16(x + q + 2);
                    if (id == 0x0001 && sz >= 16) {
                        uint64_t c8 = 0;
                        std::memcpy(&c8, x + q + 12, 8);
                        comp = c8;
                        break;
                    }
                    q += 4 + sz;
                }
            }
            const size_t data = at + 30 + nlen + xlen;
            if (name.size() > 4 && name.compare(name.size() - 4, 4, ".npy") == 0) name.resize(name.size() - 4);
            entries[name] = {data, (size_t) comp};
            at = data + (size_t) comp;
        }
        return true;
    }

    bool get(const std::string& name, NpyView& v, std::string& err) const {
        auto it = entries.find(name);
        if (it == entries.end()) {
            err = "no entry " + name;
            return false;
        }
        const char* p = buf.data() + it->second.first;
        if (std::memcmp(p, "\x93NUMPY", 6) != 0) {
            err = name + ": not an npy entry";
            return false;
        }
        const uint16_t hl = u16(p + 8);
        const std::string hdr(p + 10, hl);
        auto field = [&](const std::string& key) -> size_t {
            const size_t k = hdr.find("'" + key + "': ");
            return k == std::string::npos ? std::string::npos : k + key.size() + 4;
        };
        size_t d = field("descr");
        if (d == std::string::npos) {
            err = name + ": no descr";
            return false;
        }
        v.descr = hdr.substr(d + 1, hdr.find('\'', d + 1) - d - 1);
        size_t s = field("shape");
        if (s == std::string::npos) {
            err = name + ": no shape";
            return false;
        }
        const size_t close = hdr.find(')', s);
        std::string dims = hdr.substr(s + 1, close - s - 1);
        v.shape.clear();
        v.count = 1;
        size_t i = 0;
        while (i < dims.size()) {
            while (i < dims.size() && (dims[i] == ' ' || dims[i] == ',')) i++;
            if (i >= dims.size()) break;
            size_t j = i;
            while (j < dims.size() && dims[j] >= '0' && dims[j] <= '9') j++;
            const int64_t n = std::stoll(dims.substr(i, j - i));
            v.shape.push_back(n);
            v.count *= (size_t) n;
            i = j;
        }
        v.data = p + 10 + hl;
        return true;
    }
};

std::vector<float> as_f32(const NpyView& v) {
    std::vector<float> out(v.count);
    std::memcpy(out.data(), v.data, v.count * 4);
    return out;
}
std::vector<int64_t> as_i64(const NpyView& v) {
    std::vector<int64_t> out(v.count);
    std::memcpy(out.data(), v.data, v.count * 8);
    return out;
}

// ---- safetensors (the head and the embedding, bf16) -----------------------------------------------------------------

struct StRef {
    std::string file;
    uint64_t begin = 0;   // absolute byte offset of the tensor's data
    std::vector<uint64_t> shape;
};

bool st_find(const std::string& file, const std::string& name, StRef& out) {
    std::ifstream f(file, std::ios::binary);
    if (!f) return false;
    uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 8);
    std::string hdr(n, '\0');
    f.read(hdr.data(), (std::streamsize) n);
    const size_t at = hdr.find("\"" + name + "\"");
    if (at == std::string::npos) return false;
    const size_t off = hdr.find("\"data_offsets\":[", at);
    const size_t shp = hdr.find("\"shape\":[", at);
    if (off == std::string::npos || shp == std::string::npos) return false;
    const uint64_t a = std::stoull(hdr.substr(off + 16));
    out.file = file;
    out.begin = 8 + n + a;
    out.shape.clear();
    size_t i = shp + 9;
    while (i < hdr.size() && hdr[i] != ']') {
        if (hdr[i] >= '0' && hdr[i] <= '9') {
            size_t j = i;
            while (j < hdr.size() && hdr[j] >= '0' && hdr[j] <= '9') j++;
            out.shape.push_back(std::stoull(hdr.substr(i, j - i)));
            i = j;
        } else {
            i++;
        }
    }
    return true;
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ---- comparison ------------------------------------------------------------------------------------------------------

struct Diff {
    double max_abs = 0.0;
    double ref_max = 0.0;
    double cos = 0.0;
};

Diff diff(const std::vector<float>& ours, const std::vector<float>& ref) {
    Diff d;
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < ref.size() && i < ours.size(); i++) {
        const double a = ours[i], b = ref[i];
        d.max_abs = std::max(d.max_abs, std::fabs(a - b));
        d.ref_max = std::max(d.ref_max, std::fabs(b));
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    d.cos = (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;
    return d;
}

void print_diff(const char* what, const Diff& d) {
    std::printf("  %-14s max|diff| %10.4g   cos %.6f   (max|ref| %.4g)\n", what, d.max_abs, d.cos, d.ref_max);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <drafter.gguf> <dump dir> <qwen-head dir> [--strict]\n", argv[0]);
        return 1;
    }
    const std::string gguf = argv[1];
    const std::string dumps = argv[2];
    const std::string head_dir = argv[3];
    const bool strict = argc > 4 && std::string(argv[4]) == "--strict";
    std::string err;

    // The target's embedding and head (host, test only).
    StRef emb, head;
    for (const auto& e : std::filesystem::directory_iterator(head_dir)) {
        const std::string f = e.path().string();
        if (f.size() < 12 || f.compare(f.size() - 12, 12, ".safetensors") != 0) continue;
        if (emb.file.empty()) st_find(f, "model.language_model.embed_tokens.weight", emb);
        if (head.file.empty()) st_find(f, "lm_head.weight", head);
    }
    if (emb.file.empty() || head.file.empty() || emb.shape.size() != 2 || head.shape.size() != 2) {
        std::fprintf(stderr, "embed_tokens / lm_head not found in %s\n", head_dir.c_str());
        return 1;
    }
    const int H = (int) emb.shape[1];
    const int V = (int) emb.shape[0];
    if (H != Dflash2::kHidden || V != Dflash2::kVocab) {
        std::fprintf(stderr, "unexpected embedding shape %d x %d\n", V, H);
        return 1;
    }
    std::vector<uint16_t> head_w((size_t) V * H);
    {
        std::ifstream f(head.file, std::ios::binary);
        f.seekg((std::streamoff) head.begin);
        f.read(reinterpret_cast<char*>(head_w.data()), (std::streamsize) (head_w.size() * 2));
        if (!f) {
            std::fprintf(stderr, "cannot read lm_head\n");
            return 1;
        }
    }
    std::ifstream emb_f(emb.file, std::ios::binary);

    // The host embedding and head as the engine's callbacks.
    strata::drafter::EmbedFn embed = [&](const int32_t* ids_dev, int n, float* out_dev, void* stream,
                                         std::string& e) -> bool {
        std::vector<int32_t> ids(n);
        std::vector<float> rows((size_t) n * H);
        if (cudaMemcpy(ids.data(), ids_dev, sizeof(int32_t) * n, cudaMemcpyDeviceToHost) != cudaSuccess) {
            e = "embed: copy ids";
            return false;
        }
        std::vector<uint16_t> row(H);
        for (int i = 0; i < n; i++) {
            emb_f.seekg((std::streamoff) (emb.begin + (uint64_t) ids[i] * H * 2));
            emb_f.read(reinterpret_cast<char*>(row.data()), (std::streamsize) (H * 2));
            for (int k = 0; k < H; k++) rows[(size_t) i * H + k] = bf16_to_f32(row[k]);
        }
        if (cudaMemcpy(out_dev, rows.data(), sizeof(float) * rows.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
            e = "embed: copy rows";
            return false;
        }
        (void) stream;
        return true;
    };
    strata::drafter::HeadFn head_fn = [&](const float* hid_dev, int n, float* logits_dev, void* stream,
                                          std::string& e) -> bool {
        std::vector<float> hid((size_t) n * H);
        std::vector<float> logits((size_t) n * V);
        if (cudaMemcpy(hid.data(), hid_dev, sizeof(float) * hid.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
            e = "head: copy hidden";
            return false;
        }
        for (int v = 0; v < V; v++) {
            const uint16_t* w = head_w.data() + (size_t) v * H;
            for (int r = 0; r < n; r++) {
                const float* x = hid.data() + (size_t) r * H;
                float acc = 0.f;
                for (int k = 0; k < H; k++) acc += x[k] * bf16_to_f32(w[k]);
                logits[(size_t) r * V + v] = acc;
            }
        }
        if (cudaMemcpy(logits_dev, logits.data(), sizeof(float) * logits.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
            e = "head: copy logits";
            return false;
        }
        (void) stream;
        return true;
    };

    Dflash2 drafter;
    if (!drafter.load(gguf, err)) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    drafter.reset();
    // The engine's calls take an explicit stream; the test uses its own.
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::fprintf(stderr, "cudaStreamCreate\n");
        return 1;
    }

    int total_draft = 0, matched_draft = 0;
    double worst_cos = 1.0;
    for (int k = 0;; k++) {
        const std::string path = dumps + "/round" + std::to_string(k) + ".npz";
        std::ifstream probe(path);
        if (!probe) break;
        Npz z;
        if (!z.open(path, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        NpyView v;
        int64_t start = 0, anchor = 0, ctx_lo = 0, ctx_hi = 0, cache_len = 0;
        auto scalar = [&](const char* name, int64_t& out) -> bool {
            if (!z.get(name, v, err)) return false;
            out = as_i64(v)[0];
            return true;
        };
        if (!scalar("start", start) || !scalar("anchor", anchor) || !scalar("ctx_lo", ctx_lo) ||
            !scalar("ctx_hi", ctx_hi) || !scalar("cache_len", cache_len)) {
            std::fprintf(stderr, "%s: %s\n", path.c_str(), err.c_str());
            return 1;
        }
        std::printf("round %d: start %lld anchor %lld ctx %lld..%lld (cache %lld)\n", k, (long long) start,
                    (long long) anchor, (long long) ctx_lo, (long long) ctx_hi, (long long) cache_len);

        // The context features (bf16 bits) to the device, then the feed.
        if (!z.get("ctx_bits", v, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        const size_t nctx = (size_t) (ctx_hi - ctx_lo);
        uint16_t* feats_dev = nullptr;
        if (cudaMalloc(&feats_dev, nctx * Dflash2::kCtxFeat * 2) != cudaSuccess) {
            std::fprintf(stderr, "cudaMalloc feats\n");
            return 1;
        }
        cudaMemcpy(feats_dev, v.data, nctx * Dflash2::kCtxFeat * 2, cudaMemcpyHostToDevice);
        Dflash2Trace ft;
        if (!drafter.feed((const uint16_t*) feats_dev, (int) ctx_lo, (int) ctx_hi, stream, err, &ft)) {
            std::fprintf(stderr, "feed: %s\n", err.c_str());
            return 1;
        }
        cudaFree(feats_dev);
        if (z.get("th_norm", v, err)) print_diff("th_norm", diff(ft.th_norm, as_f32(v)));

        // The block.
        std::vector<int32_t> draft;
        Dflash2Trace tr;
        if (!drafter.propose((int) start, (int32_t) anchor, Dflash2::kBlock, embed, head_fn, draft, stream, err, &tr)) {
            std::fprintf(stderr, "propose: %s\n", err.c_str());
            return 1;
        }
        for (int l = 0; l < Dflash2::kLayers; l++) {
            if (!z.get("layer" + std::to_string(l), v, err)) {
                std::fprintf(stderr, "%s\n", err.c_str());
                return 1;
            }
            const Diff d = diff(tr.layer[l], as_f32(v));
            worst_cos = std::min(worst_cos, d.cos);
            print_diff(("layer " + std::to_string(l)).c_str(), d);
        }
        if (z.get("final", v, err)) print_diff("final", diff(tr.final, as_f32(v)));

        // Candidate sets (the order of the top-16 is not the reference's; the sets are compared).
        if (z.get("cand", v, err)) {
            const std::vector<int64_t> cand = as_i64(v);
            const int rows = (int) (cand.size() / Dflash2::kTopK);
            int same = 0;
            for (int r = 0; r < rows; r++) {
                std::set<int64_t> a, b;
                for (int c = 0; c < Dflash2::kTopK; c++) {
                    a.insert(tr.cand[(size_t) r * Dflash2::kTopK + c]);
                    b.insert(cand[(size_t) r * Dflash2::kTopK + c]);
                }
                same += a == b;
            }
            std::printf("  top-%d sets equal on %d of %d rows\n", Dflash2::kTopK, same, rows);
        }

        // The draft tokens.
        if (!z.get("draft", v, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
        }
        const std::vector<int64_t> ref = as_i64(v);
        std::printf("  draft   ours:");
        for (int32_t t : draft) std::printf(" %d", t);
        std::printf("\n  draft    ref:");
        for (int64_t t : ref) std::printf(" %lld", (long long) t);
        std::printf("\n");
        for (size_t i = 0; i < ref.size() && i < draft.size(); i++) {
            total_draft++;
            matched_draft += draft[i] == ref[i];
        }
    }
    std::printf("draft tokens equal to the reference: %d of %d; worst layer cosine %.6f\n", matched_draft, total_draft,
                worst_cos);
    const bool pass = total_draft > 0 && matched_draft == total_draft;
    std::printf("verdict: %s\n", pass ? "PASS (every draft token equal)" : "DIFFERS (see the numbers above)");
    if (strict && !pass) return 1;
    return 0;
}
