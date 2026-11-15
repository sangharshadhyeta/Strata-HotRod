// tests/core/expert_layout_latent_test.cpp - Edge0's latent A-only packs (native_experts.txt v6), without a model or a GPU.
//
// A latent pack's experts are not n_embd wide: the A stage runs at the latent width (1792 for the 145B, 1280 for the 120B) with
// 640 inside, and the header says so ("expert_embd 1792, expert_ff 640").  This checks, on the CPU-only layout reader:
//   1. NVFP4 at 1792/640 makes exactly the blob size Edge0's own pack records (1,935,376 B), so the layout code needs no new
//      format arithmetic;
//   2. a v6 pack parses: widths, per-layer formats, offsets, totals; the layout says it is latent;
//   3. every refusal: v6 without its widths, widths in another version's header, v5 (Hadamard-folded experts: no kernels
//      here), v7, a blob that is not the formats' size, a zero width;
//   4. a v6 pack at the model's own widths is not latent, and the layout is rebuilt on every load (an ordinary pack after a
//      latent one is not latent).
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using strata::kernels::cpu::ExpertLayout;
using strata::kernels::cpu::NativeFmt;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-96s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-expert-layout-latent-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void write_text(const fs::path& p, const std::string& s) { std::ofstream(p, std::ios::binary) << s; }

bool load(const fs::path& dir, int64_t n_layers, int64_t n_expert, std::string& err) {
    err.clear();
    return strata::kernels::cpu::expert_layout_load(dir.string(), n_layers, n_expert, err);
}

// the five-column form (no GGUF offsets: NVFP4 and latent packs carry none) with the header Edge0's packer writes
std::string manifest(int version, const std::string& widths, int64_t blob, int n_layers = 2, int n_expert = 512) {
    std::string s = "# strata native experts v" + std::to_string(version) + ": latent A-only" + widths + "\n";
    for (int l = 0; l < n_layers; ++l)
        s += std::to_string(l) + " 40 40 " + std::to_string((int64_t) l * n_expert * blob) + " " + std::to_string(blob) + "\n";
    return s;
}
}  // namespace

int main(int argc, char** argv) {
    std::printf("expert_layout_latent_test\n");
    NativeFmt f;
    std::string err;
    // `expert_layout_latent_test PACK_DIR`: a real Edge0 folder (native_experts.txt and experts.bin beside it), read-only: the
    // header parses as latent, and the layers' totals are exactly the size of experts.bin
    if (argc == 2) {
        const fs::path dir = argv[1];
        const bool ok = load(dir, 48, 512, err);
        const ExpertLayout& R = strata::kernels::cpu::expert_layout();
        std::error_code ec;
        const auto bin = fs::file_size(dir / "experts.bin", ec);
        std::printf("  %s: %s", dir.string().c_str(), ok ? "loaded" : ("REFUSED: " + err).c_str());
        if (ok)
            std::printf(" v%d, latent %d, experts %lld wide / %lld inside, %lld layers x %lld experts of %llu B, total %llu B, experts.bin %llu B\n",
                        R.version, (int) R.latent, (long long) R.expert_embd, (long long) R.expert_ff, (long long) R.n_layers,
                        (long long) R.n_expert, (unsigned long long) R.blob_bytes(0), (unsigned long long) R.total, (unsigned long long) bin);
        check(ok && R.latent && !ec && R.total == (uint64_t) bin, "the real pack's layer table adds up to experts.bin exactly");
        std::printf(g_fail ? "expert_layout_latent_test: %d FAILED\n" : "expert_layout_latent_test: real pack ok\n", g_fail);
        return g_fail ? 1 : 0;
    }
    if (!strata::kernels::cpu::native_fmt(40, 40, 1792, 640, f, err)) {
        check(false, "NVFP4 at the 145B's latent geometry 1792/640: " + err);
        return 1;
    }
    const int64_t blob = (int64_t) f.bytes;
    check(blob == 1935376, "NVFP4 at 1792/640 makes Edge0's blob_bytes 1935376 (ours: " + std::to_string(blob) + ")");

    const std::string widths = " (expert_embd 1792, expert_ff 640)";
    TempDir d;
    write_text(d.path / "native_experts.txt", manifest(6, widths, blob));
    bool ok = load(d.path, 2, 512, err);
    const ExpertLayout& L = strata::kernels::cpu::expert_layout();
    check(ok && L.version == 6 && L.latent && L.expert_embd == 1792 && L.expert_ff == 640 && L.fmt.size() == 2 &&
              L.fmt[0].n_embd == 1792 && L.fmt[0].n_ff == 640 && (int64_t) L.bytes[1] == blob &&
              (int64_t) L.total == 2 * 512 * blob,
          "v6: widths read, the layout is latent, layers and totals as the pack says" + (ok ? "" : ": " + err));

    write_text(d.path / "native_experts.txt", manifest(6, "", blob));
    ok = load(d.path, 2, 512, err);
    check(!ok && err.find("expert_embd") != std::string::npos, "v6 without its widths is refused (\"" + err.substr(0, 48) + "...\")");
    write_text(d.path / "native_experts.txt", manifest(4, widths, blob));
    ok = load(d.path, 2, 512, err);
    check(!ok && err.find("only v6") != std::string::npos, "widths in a v4 header are refused");
    write_text(d.path / "native_experts.txt", manifest(5, widths, blob));
    ok = load(d.path, 2, 512, err);
    check(!ok && err.find("v5") != std::string::npos, "v5 (Hadamard-folded experts) is refused by name");
    write_text(d.path / "native_experts.txt", manifest(7, widths, blob));
    ok = load(d.path, 2, 512, err);
    check(!ok && err.find("v7") != std::string::npos, "v7 is refused");
    write_text(d.path / "native_experts.txt", manifest(6, widths, blob - 1));
    ok = load(d.path, 2, 512, err);
    check(!ok && err.find("blob is") != std::string::npos, "a blob that is not the formats' size is refused");
    write_text(d.path / "native_experts.txt", manifest(6, " (expert_embd 0, expert_ff 640)", blob));
    check(!load(d.path, 2, 512, err), "a zero width is refused");

    // a v6 pack whose widths are the model's own (2560/640) is not latent
    NativeFmt own;
    if (strata::kernels::cpu::native_fmt(40, 40, 2560, 640, own, err)) {
        write_text(d.path / "native_experts.txt", manifest(6, " (expert_embd 2560, expert_ff 640)", (int64_t) own.bytes));
        ok = load(d.path, 2, 512, err);
        check(ok && !L.latent && L.expert_embd == 2560, "v6 at the model's own widths is not latent" + (ok ? "" : ": " + err));
    } else {
        check(false, "NVFP4 at 2560/640: " + err);
    }

    // the layout is rebuilt on every load: an ordinary pack after a latent one is not latent
    write_text(d.path / "native_experts.txt", manifest(6, widths, blob));
    load(d.path, 2, 512, err);
    NativeFmt q;
    if (strata::kernels::cpu::native_fmt(12, 7, 2560, 640, q, err)) {
        TempDir d2;
        std::string s = "# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] (n_expert 2)\n";
        s += "0 12 7 0 " + std::to_string((int64_t) q.bytes) + " 100 200 300\n";
        s += "1 12 7 " + std::to_string((int64_t) q.bytes * 2) + " " + std::to_string((int64_t) q.bytes) + " 400 500 600\n";
        write_text(d2.path / "native_experts.txt", s);
        ok = load(d2.path, 2, 512, err);
        check(ok && L.latent == false && L.expert_embd == 0 && L.version == 3, "an ordinary pack after a latent one is not latent" + (ok ? "" : ": " + err));
    } else {
        check(false, "Q4_K/Q5_1 at 2560/640: " + err);
    }

    std::printf(g_fail ? "expert_layout_latent_test: %d FAILED\n" : "expert_layout_latent_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
