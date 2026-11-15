// src/core/latent_tab.cpp - the session-level latent MoE tables (branch hotrod-edge0).  The shape (one upload before the cache is sized, the
// device pair shared through latent_share.hpp) follows Edge0's LatentTab in edge0-strata (MIT, a fork of Strata); the sizing is ours: every
// number is read off the pack (see the header).
#include "strata/core/latent_tab.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace strata::core {
namespace {

// bytes of one group's D and U for one layer at a block size (Q8_0: 34 B / 32 values, Q4_0: 18 B / 32 values)
uint64_t group_layer_bytes(int64_t n_embd, int64_t latent, int block_bytes) {
    return (uint64_t) ((latent * (n_embd / 32) + n_embd * (latent / 32)) * (int64_t) block_bytes);
}

bool env_int(const char* name, int& v) {
    const char* e = std::getenv(name);
    if (e == nullptr || e[0] == 0) return false;
    v = std::atoi(e);
    return true;
}

}  // namespace

bool latent_dense_identify(uint64_t file_bytes, int64_t n_layers, int64_t n_embd, int64_t latent, int& fmt, int& groups) {
    if (n_layers <= 0 || n_embd % 32 != 0 || latent % 32 != 0 || file_bytes == 0) return false;
    int found = 0, f = 0, g = 0;
    for (const int cand : {8, 4}) {
        const uint64_t per = (uint64_t) n_layers * group_layer_bytes(n_embd, latent, cand == 4 ? 18 : 34);
        if (per == 0 || file_bytes % per != 0) continue;
        const uint64_t n = file_bytes / per;
        if (n < 1 || n > 64) continue;
        ++found;
        f = cand;
        g = (int) n;
    }
    if (found != 1) return false;
    fmt = f;
    groups = g;
    return true;
}

bool latent_dense_choose(const std::string& pack_dir, int64_t n_layers, int64_t n_embd, int64_t latent, uint64_t free_vram, int share_pct,
                         LatentDenseChoice& out, std::string& err) {
    namespace fs = std::filesystem;
    if (share_pct <= 0 && !env_int("STRATA_LATENT_DU_SHARE_PCT", share_pct)) share_pct = 50;
    share_pct = std::clamp(share_pct, 1, 100);
    std::vector<LatentDenseChoice> cands;
    std::string notes;
    auto consider = [&](const std::string& path) {
        std::error_code ec;
        if (!fs::is_regular_file(path, ec)) return;
        LatentDenseChoice c;
        c.path = path;
        c.bytes = (uint64_t) fs::file_size(path, ec);
        if (ec || !latent_dense_identify(c.bytes, n_layers, n_embd, latent, c.fmt, c.groups)) {
            notes += " " + path + " is not a D/U file for " + std::to_string((long long) n_layers) + " layers, hidden " +
                     std::to_string((long long) n_embd) + ", latent " + std::to_string((long long) latent) + ";";
            return;
        }
        cands.push_back(c);
    };
    const char* forced = std::getenv("STRATA_LATENT_DENSE");
    if (forced != nullptr && forced[0] != 0) {
        consider(forced);
    } else {
        consider(pack_dir + "/latent_dense_q8.bin");
        consider(pack_dir + "/latent_dense_q4.bin");
    }
    if (cands.empty()) {
        err = "no latent D/U file: looked for " + std::string(forced != nullptr && forced[0] != 0 ? forced : pack_dir + "/latent_dense_q8.bin or latent_dense_q4.bin") +
              ";" + notes;
        return false;
    }
    int want_fmt = 0;
    if (env_int("STRATA_LATENT_DENSE_FMT", want_fmt) && want_fmt != 4 && want_fmt != 8) {
        err = "STRATA_LATENT_DENSE_FMT must be 4 or 8";
        return false;
    }
    const LatentDenseChoice* pick = nullptr;
    if (want_fmt != 0) {
        for (const auto& c : cands)
            if (c.fmt == want_fmt) pick = &c;
        if (pick == nullptr) {
            err = "STRATA_LATENT_DENSE_FMT=" + std::to_string(want_fmt) + " but no such D/U file was found" + (notes.empty() ? "" : ";" + notes);
            return false;
        }
    } else if (cands.size() == 1) {
        pick = &cands[0];
    } else {   // both: the better format while it leaves the rest of the card its share
        const LatentDenseChoice* q8 = nullptr;
        const LatentDenseChoice* q4 = nullptr;
        for (const auto& c : cands) (c.fmt == 8 ? q8 : q4) = &c;
        pick = (q8 != nullptr && q4 != nullptr) ? ((q8->bytes <= free_vram / 100 * (uint64_t) share_pct) ? q8 : q4) : cands.data();
    }
    if (pick->bytes > free_vram) {
        err = "the latent D/U matrices (" + pick->path + ", Q" + (pick->fmt == 8 ? "8" : "4") + "_0) need " +
              std::to_string((double) pick->bytes / 1073741824.0) + " GiB of VRAM and " + std::to_string((double) free_vram / 1073741824.0) +
              " GiB are free";
        return false;
    }
    out = *pick;
    return true;
}

LatentTab::~LatentTab() {
    if (latent_shared_d() == d_) latent_shared_d() = nullptr;
    if (latent_shared_u() == u_) latent_shared_u() = nullptr;
    if (latent_shared_info().labels_dev == labels_dev_) latent_shared_info() = LatentShared{};
    if (d_) cudaFree(d_);
    if (u_) cudaFree(u_);
    if (labels_dev_) cudaFree(labels_dev_);
}

bool LatentTab::load(const std::string& pack_dir, int64_t n_layers, int64_t n_embd, int64_t latent, int64_t n_expert, uint64_t free_vram,
                     std::string& err) {
    if (d_ != nullptr) {   // already loaded: republish
        latent_shared_d() = d_;
        latent_shared_u() = u_;
        return true;
    }
    LatentDenseChoice ch;
    if (!latent_dense_choose(pack_dir, n_layers, n_embd, latent, free_vram, 0, ch, err)) return false;
    // labels.bin: one int32 group label per (layer, expert), beside the pack (STRATA_LATENT_LABELS names another file)
    const char* lf = std::getenv("STRATA_LATENT_LABELS");
    const std::string lp = (lf != nullptr && lf[0] != 0) ? std::string(lf) : pack_dir + "/labels.bin";
    {
        std::error_code ec;
        const auto sz = std::filesystem::file_size(lp, ec);
        const uint64_t want = (uint64_t) n_layers * (uint64_t) n_expert * 4;
        if (ec) { err = "latent labels: " + lp + " is missing"; return false; }
        if ((uint64_t) sz != want) {
            err = "latent labels: " + lp + " is " + std::to_string((unsigned long long) sz) + " B, " + std::to_string((long long) n_layers) + " layers x " +
                  std::to_string((long long) n_expert) + " experts x 4 B is " + std::to_string((unsigned long long) want);
            return false;
        }
        labels_.resize((size_t) n_layers * (size_t) n_expert);
        std::ifstream in(lp, std::ios::binary);
        in.read((char*) labels_.data(), (std::streamsize) (labels_.size() * 4));
        if (!in) { err = "latent labels: " + lp + " could not be read"; return false; }
        for (const int32_t v : labels_)
            if (v < 0 || v >= ch.groups) {
                err = "latent labels: group " + std::to_string((long long) v) + " in " + lp + ", but the D/U file has " + std::to_string(ch.groups) + " groups";
                return false;
            }
    }
    if (!dense_.open(ch.path, n_layers, ch.groups, n_embd, latent, err, ch.fmt)) {
        err = "latent D/U: " + err;
        return false;
    }
    const uint64_t db = (uint64_t) n_layers * dense_.d_bytes(), ub = (uint64_t) n_layers * dense_.u_bytes();
    if (cudaMalloc((void**) &d_, db) != cudaSuccess || cudaMalloc((void**) &u_, ub) != cudaSuccess) {
        const cudaError_t ce = cudaGetLastError();
        if (d_) { cudaFree(d_); d_ = nullptr; }
        if (u_) { cudaFree(u_); u_ = nullptr; }
        err = std::string("the latent D/U device buffers (") + std::to_string((double) (db + ub) / 1073741824.0) + " GiB) could not be had: " + cudaGetErrorString(ce);
        return false;
    }
    for (int64_t l = 0; l < n_layers; ++l)
        if (cudaMemcpy(d_ + (size_t) l * dense_.d_bytes(), dense_.d(l), dense_.d_bytes(), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(u_ + (size_t) l * dense_.u_bytes(), dense_.u(l), dense_.u_bytes(), cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "the latent D/U upload failed at layer " + std::to_string((long long) l);
            cudaFree(d_); cudaFree(u_);
            d_ = u_ = nullptr;
            return false;
        }
    groups_ = ch.groups;
    fmt_ = ch.fmt;
    device_bytes_ = db + ub;
    // the labels on the device too (the decode path's plan rewrite and U kernel read them there)
    const size_t lb = labels_.size() * sizeof(int32_t);
    if (cudaMalloc((void**) &labels_dev_, lb) != cudaSuccess || cudaMemcpy(labels_dev_, labels_.data(), lb, cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "the latent labels could not be uploaded";
        if (labels_dev_) { cudaFree(labels_dev_); labels_dev_ = nullptr; }
        cudaFree(d_); cudaFree(u_);
        d_ = u_ = nullptr;
        return false;
    }
    device_bytes_ += lb;
    {
        LatentShared& info = latent_shared_info();
        info.groups = groups_;
        info.fmt = fmt_;
        info.d_layer_bytes = dense_.d_bytes();
        info.u_layer_bytes = dense_.u_bytes();
        info.labels_dev = labels_dev_;
        info.labels_host = labels_.data();
    }
    latent_shared_d() = d_;
    latent_shared_u() = u_;
    return true;
}

}  // namespace strata::core
