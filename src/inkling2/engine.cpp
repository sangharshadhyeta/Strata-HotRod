// src/inkling2/engine.cpp - the Inkling 2 GPU engine (include/strata/inkling2/engine.hpp), F2 of docs/INKLING-PORT.md.
#include "strata/inkling2/engine.hpp"

#include "strata/inkling2/ops_cpu.hpp"   // host-side loading helpers only (decode of small tensors, conv folding): verified against the reference
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/inkling2_kernels.hpp"
#include "strata/kernels/inkling2_router.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strata {
namespace inkling2 {
namespace {

namespace K = strata::kernels;
namespace ik = strata::kernels::ik;
namespace C = strata::inkling2::cpu;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "inkling2 engine: %s: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

constexpr int kMaxBlk = 12;   // block states kept (the prelude's 1 + the core's 2 + the rest's 1, plus the stream and a spare)

}  // namespace

struct Engine::Impl {
    const GgufFile& g;
    const Config c;
    const Layout& lay;
    int cap;
    std::vector<void*> allocs;
    uint64_t dense_b = 0, expert_b = 0;

    struct LW {   // one stored layer's device weights
        void *q = nullptr, *k = nullptr, *v = nullptr, *gate = nullptr, *out = nullptr, *rel = nullptr;                     // Q8_0 rows
        float *attn_norm = nullptr, *qn = nullptr, *kn = nullptr, *vn = nullptr;
        float *qclip = nullptr, *rclip = nullptr, *relbias = nullptr, *sink = nullptr, *kconv = nullptr, *vconv = nullptr, *postconv = nullptr, *ffnpost = nullptr;
        float *kproj = nullptr, *vproj = nullptr, *res_q = nullptr, *ffn_res_q = nullptr, *ffn_norm = nullptr, *router = nullptr, *router_bias = nullptr;
        float *ld[4] = {}, *lf[4] = {}, *lg[4] = {};
        std::vector<float> gs, ds;                         // per-expert scales (NVFP4)
        const uint8_t *gu_dev = nullptr, *dn_dev = nullptr;  // the experts, device-addressable (registered mapping)
        uint64_t gu_plane = 0, dn_plane = 0, gate_bytes = 0;
        int etype = 0;
    };
    std::vector<LW> lw;
    void *d_embd = nullptr, *d_out = nullptr;
    float *d_embd_norm = nullptr, *d_out_norm = nullptr, *d_out_res = nullptr, *d_loop_rq = nullptr, *d_loop_kq = nullptr;

    // per-visit state
    std::vector<float*> Kc, Vc;                                // [cap][KV][D]
    std::vector<float*> ks, vs, ps, fs;                        // conv states [C][taps-1]
    float* attn_scratch = nullptr;

    // scratch
    float *h = nullptr, *pre_h = nullptr, *blk = nullptr, *ms = nullptr, *ins = nullptr, *xs = nullptr, *msgs = nullptr, *tmp = nullptr;
    float *q = nullptr, *k = nullptr, *v = nullptr, *r = nullptr, *gate = nullptr, *sinkv = nullptr, *kc = nullptr, *vc = nullptr, *dynk = nullptr, *dynv = nullptr;
    float *att = nullptr, *proj = nullptr, *a = nullptr, *rr = nullptr, *m_in = nullptr, *x = nullptr, *rl = nullptr, *gb = nullptr, *ub = nullptr, *hb = nullptr, *db = nullptr;
    float *moe = nullptr, *mm = nullptr, *ov = nullptr, *cat = nullptr, *g64 = nullptr, *fgt = nullptr, *gnn = nullptr, *logits = nullptr, *hd = nullptr;
    int32_t* d_idx = nullptr;
    float* d_w = nullptr;
    void *xq = nullptr, *hq = nullptr, *aq = nullptr;
    std::vector<std::vector<int>> picks;
    int pos_cur = -1;

    Impl(const GgufFile& gg, const Config& cc, const Layout& ll, int max_ctx) : g(gg), c(cc), lay(ll), cap(max_ctx) { init(); }
    ~Impl() { for (void* p : allocs) cudaFree(p); for (void* p : pinned) cudaFreeHost(p); }

    void* dmalloc(size_t n) { void* p = nullptr; CK(cudaMalloc(&p, n)); allocs.push_back(p); return p; }
    float* fmalloc(size_t n) { return (float*) dmalloc(n * 4); }
    float* fzero(size_t n) { float* p = fmalloc(n); CK(cudaMemset(p, 0, n * 4)); return p; }
    float* up(const C::Mat& m) { float* p = fmalloc(m.size()); CK(cudaMemcpy(p, m.data(), m.size() * 4, cudaMemcpyHostToDevice)); dense_b += m.size() * 4; return p; }
    void* up_raw(const TensorInfo& t) {
        const uint64_t n = tensor_payload_bytes(t);
        void* p = dmalloc(n);
        CK(cudaMemcpy(p, g.tensor_data(t), n, cudaMemcpyHostToDevice));
        dense_b += n;
        return p;
    }
    void* q8(const TensorInfo& t, const char* what) {
        if (t.type != kQ8_0) throw std::runtime_error(std::string(what) + ": expected Q8_0, got " + t.type_name());
        return up_raw(t);
    }

    /// An expert tensor read into pinned, mapped host memory (zero-copy over PCIe for the kernels): the same idea as the RAM tier's arena (F3 replaces this by
    /// the ExpertStore).  Read with pread from the GGUF in chunks.
    std::vector<void*> pinned;
    const uint8_t* load_pinned(const TensorInfo& t) {
        const uint64_t n = tensor_payload_bytes(t), off0 = g.data_start() + t.offset;
        void* host = nullptr;
        CK(cudaHostAlloc(&host, n, cudaHostAllocMapped | cudaHostAllocPortable));
        pinned.push_back(host);
        const int fd = ::open(g.path().c_str(), O_RDONLY);
        if (fd < 0) { std::fprintf(stderr, "inkling2 engine: cannot open %s\n", g.path().c_str()); std::exit(1); }
        for (uint64_t done = 0; done < n;) {
            const ssize_t r = ::pread(fd, (char*) host + done, (size_t) std::min<uint64_t>(n - done, 64ull << 20), (off_t) (off0 + done));
            if (r <= 0) { std::fprintf(stderr, "inkling2 engine: short read of %s at %llu\n", t.name.c_str(), (unsigned long long) done); std::exit(1); }
            done += (uint64_t) r;
        }
        ::close(fd);
        void* dp = nullptr;
        CK(cudaHostGetDevicePointer(&dp, host, 0));
        expert_b += n;
        return (const uint8_t*) dp;
    }

    void init() {
        const int64_t E = c.n_embd, H = c.n_head, KV = c.n_head_kv, D = c.head_dim, R = c.rel_states, taps = c.conv_taps, FF = c.n_ff_exp, NE = c.n_expert;
        lw.assign((size_t) c.n_stored, LW{});
        const float dsc = (float) (1.0 / std::sqrt((double) (2 * taps + 1)));
        d_embd = q8(*lay.token_embd, "token_embd");
        d_out = q8(*lay.output, "output");
        d_embd_norm = up(C::vec(g, *lay.token_embd_norm));
        d_out_norm = up(C::vec(g, *lay.output_norm));
        d_out_res = up(C::vec(g, *lay.output_res_query));
        d_loop_rq = up(C::vec(g, *lay.loop_res_query));
        d_loop_kq = up(C::vec(g, *lay.loop_kv_res_query));
        for (int64_t l = 0; l < c.n_stored; ++l) {
            const LayerTensors& L = lay.layers[(size_t) l];
            LW& w = lw[(size_t) l];
            w.q = q8(*L.attn_q, "attn_q"); w.k = q8(*L.attn_k, "attn_k"); w.v = q8(*L.attn_v, "attn_v");
            w.gate = q8(*L.attn_gate, "attn_gate"); w.out = q8(*L.attn_output, "attn_output"); w.rel = q8(*L.attn_rel, "attn_rel");
            if (L.full) { w.qn = up(C::vec(g, *L.attn_q_inp_norm)); w.kn = up(C::vec(g, *L.attn_k_inp_norm)); w.vn = up(C::vec(g, *L.attn_v_inp_norm)); }
            else w.attn_norm = up(C::vec(g, *L.attn_norm));
            w.qclip = up(C::vec(g, *L.attn_q_clip)); w.rclip = up(C::vec(g, *L.attn_rel_clip)); w.relbias = up(C::vec(g, *L.attn_rel_bias));
            w.sink = up(C::vec(g, *L.attn_sink_proj));
            w.kconv = up(C::fold_conv(C::vec(g, *L.attn_k_conv), KV * D, taps, L.full));
            w.vconv = up(C::fold_conv(C::vec(g, *L.attn_v_conv), KV * D, taps, L.full));
            w.postconv = up(C::fold_conv(C::vec(g, *L.attn_post_conv), E, taps, false));
            w.ffnpost = up(C::fold_conv(C::vec(g, *L.ffn_post_conv), E, taps, false));
            if (L.full) {
                C::Mat kp = C::vec(g, *L.attn_k_conv_proj), vp = C::vec(g, *L.attn_v_conv_proj);
                for (float& z : kp) z *= dsc;
                for (float& z : vp) z *= dsc;
                w.kproj = up(kp); w.vproj = up(vp);
            }
            if (l > 0) w.res_q = up(C::vec(g, *L.attn_res_query));
            w.ffn_res_q = up(C::vec(g, *L.ffn_res_query)); w.ffn_norm = up(C::vec(g, *L.ffn_norm));
            w.router = up(C::vec(g, *L.router)); w.router_bias = up(C::vec(g, *L.router_bias));
            if (L.core) for (int j = 0; j < 4; ++j) { w.ld[j] = up(C::vec(g, *L.loop_down[j])); w.lf[j] = up(C::vec(g, *L.loop_forget[j])); w.lg[j] = up(C::vec(g, *L.loop_gain[j])); }
            w.etype = L.expert_nvfp4 ? 40 : 8;
            if (L.expert_nvfp4) { w.gs = C::vec(g, *L.gate_up_scale); w.ds = C::vec(g, *L.down_scale); }
            const uint64_t rb_gu = C::row_bytes(L.gate_up_exps->type, E), rb_dn = C::row_bytes(L.down_exps->type, FF);
            w.gu_plane = (uint64_t) 2 * FF * rb_gu;
            w.dn_plane = (uint64_t) E * rb_dn;
            w.gate_bytes = (uint64_t) FF * rb_gu;
            w.gu_dev = load_pinned(*L.gate_up_exps);
            w.dn_dev = load_pinned(*L.down_exps);
        }
        (void) NE;
        // per-visit state
        for (int64_t v = 0; v < c.n_visits; ++v) {
            Kc.push_back(fzero((size_t) cap * KV * D)); Vc.push_back(fzero((size_t) cap * KV * D));
            ks.push_back(fzero((size_t) KV * D * (taps - 1))); vs.push_back(fzero((size_t) KV * D * (taps - 1)));
            ps.push_back(fzero((size_t) E * (taps - 1))); fs.push_back(fzero((size_t) E * (taps - 1)));
        }
        attn_scratch = fmalloc((size_t) H * cap);
        h = fmalloc(E); pre_h = fmalloc(E); blk = fmalloc((size_t) kMaxBlk * E); ms = fmalloc((size_t) kMaxBlk * E); ins = fmalloc(3 * E); xs = fmalloc(3 * E); msgs = fmalloc(4 * E);
        tmp = fmalloc(2 * E);
        q = fmalloc(H * D); k = fmalloc(KV * D); v = fmalloc(KV * D); r = fmalloc(H * R); gate = fmalloc(H * D); sinkv = fmalloc(H);
        kc = fmalloc(KV * D); vc = fmalloc(KV * D); dynk = fmalloc(KV * taps); dynv = fmalloc(KV * taps);
        att = fmalloc(H * D); proj = fmalloc(E); a = fmalloc(E); rr = fmalloc(E); m_in = fmalloc(E); x = fmalloc(E); rl = fmalloc(c.n_expert);
        gb = fmalloc(FF); ub = fmalloc(FF); hb = fmalloc(FF); db = fmalloc(E); moe = fmalloc(E); mm = fmalloc(E); ov = fmalloc(E);
        cat = fmalloc(2 * E); g64 = fmalloc(c.gate_rank); fgt = fmalloc(E); gnn = fmalloc(E); logits = fmalloc((size_t) c.n_vocab); hd = fmalloc(E);
        d_idx = (int32_t*) dmalloc(16 * 4); d_w = fmalloc(16);
        xq = dmalloc((size_t) (E / 32) * 36 * 2); hq = dmalloc((size_t) (FF / 32) * 36 * 2); aq = dmalloc((size_t) (H * D / 32) * 36 * 2);
        picks.assign((size_t) c.n_visits, {});
    }

    void reset() {
        const int64_t E = c.n_embd, KV = c.n_head_kv, D = c.head_dim, taps = c.conv_taps;
        for (int64_t v = 0; v < c.n_visits; ++v) {
            CK(cudaMemset(ks[(size_t) v], 0, (size_t) KV * D * (taps - 1) * 4)); CK(cudaMemset(vs[(size_t) v], 0, (size_t) KV * D * (taps - 1) * 4));
            CK(cudaMemset(ps[(size_t) v], 0, (size_t) E * (taps - 1) * 4)); CK(cudaMemset(fs[(size_t) v], 0, (size_t) E * (taps - 1) * 4));
        }
        pos_cur = -1;
    }

    // ---- helpers over the reused kernels
    void mmvq_f(int type, const void* W, const float* xin, int n_in, int n_out, float* y, void* qbuf) {   // y[n_out] = W . xin (activation through q8_1)
        K::quantize_q8_1_rows(xin, 1, n_in, qbuf, nullptr);
        K::iq_mmvq(type, W, qbuf, y, n_in, n_out, 1, nullptr);
    }
    void rms_w(float* xx, const float* w) { K::rms_norm_weighted(xx, w, 1, c.n_embd, c.rms_eps, nullptr); }
    void copy(float* dst, const float* src, size_t n) { CK(cudaMemcpyAsync(dst, src, n * 4, cudaMemcpyDeviceToDevice, nullptr)); }
    void cap_seam(const SeamSink& sink, int v, const char* name, const float* dev, int n) { if (sink) sink(v, name, dev, n); }

    /// The loop control of a core layer's input (module 0 attn, 1 k, 2 v, 3 mlp): xx <- (1 - tanh(f)) xx + tanh(gn) msg
    void control(const LW& w, int module, float* xx, const float* msg) {
        const int64_t E = c.n_embd;
        ik::control_cat(xx, msg, cat, (int) E, nullptr);
        ik::matmul_f32(w.ld[module], cat, g64, 1, (int) (2 * E), (int) c.gate_rank, nullptr);
        ik::matmul_f32(w.lf[module], g64, fgt, 1, (int) c.gate_rank, (int) E, nullptr);
        ik::matmul_f32(w.lg[module], g64, gnn, 1, (int) c.gate_rank, (int) E, nullptr);
        ik::control_mix(xx, msg, fgt, gnn, tmp, (int) E, nullptr);
        copy(xx, tmp, (size_t) E);
    }

    /// One layer visit of stored layer l (visit v), nb = the block states held; returns the new nb; the new stream is in h.
    int layer(int64_t l, int vis, int nb, bool msgs_on, const SeamSink& sink, int pos) {
        const int64_t E = c.n_embd, H = c.n_head, KV = c.n_head_kv, D = c.head_dim, R = c.rel_states, taps = c.conv_taps, FF = c.n_ff_exp;
        const LayerTensors& L = lay.layers[(size_t) l];
        const LW& w = lw[(size_t) l];
        const bool boundary = l % c.attn_res_block == 0;
        // the three inputs
        if (l == 0) { for (int j = 0; j < 3; ++j) copy(ins + j * E, h, (size_t) E); }
        else {
            copy(blk + (size_t) nb * E, h, (size_t) E);   // the stream joins the block states
            const int nq = L.full ? 3 : 1;
            ik::attn_res(blk, nb + 1, w.res_q, nq, ins, (int) E, nullptr);   // writes [nq][E] into ins
            if (nq == 1) { copy(ins + E, ins, (size_t) E); copy(ins + 2 * E, ins, (size_t) E); }
        }
        if (msgs_on) {
            for (int j = 0; j < 3; ++j) control(w, j, ins + j * E, msgs + j * E);
            cap_seam(sink, vis, "loop_q_in", ins, (int) E);
        }
        // the norms
        if (!L.full) { for (int j = 0; j < 3; ++j) { copy(xs + j * E, ins + (msgs_on ? j : 0) * E, (size_t) E); rms_w(xs + j * E, w.attn_norm); } }
        else { const float* nw[3] = {w.qn, w.kn, w.vn}; for (int j = 0; j < 3; ++j) { copy(xs + j * E, ins + j * E, (size_t) E); rms_w(xs + j * E, nw[j]); } }
        // projections (q, r, gate, sink from the first input; k from the second; v from the third)
        K::quantize_q8_1_rows(xs, 1, E, xq, nullptr);
        K::iq_mmvq(8, w.q, xq, q, (int) E, (int) (H * D), 1, nullptr);
        K::iq_mmvq(8, w.rel, xq, r, (int) E, (int) (H * R), 1, nullptr);
        K::iq_mmvq(8, w.gate, xq, gate, (int) E, (int) (H * D), 1, nullptr);
        ik::matmul_f32(w.sink, xs, sinkv, 1, (int) E, (int) H, nullptr);
        mmvq_f(8, w.k, xs + E, (int) E, (int) (KV * D), k, (char*) xq + (E / 32) * 36);
        mmvq_f(8, w.v, xs + 2 * E, (int) E, (int) (KV * D), v, (char*) xq + (E / 32) * 36);
        // the short convs on k and v
        if (!L.full) {
            K::gdn_conv_step(ks[(size_t) vis], k, w.kconv, kc, KV * D, taps, nullptr);
            K::gdn_conv_step(vs[(size_t) vis], v, w.vconv, vc, KV * D, taps, nullptr);
        } else {
            ik::matmul_f32(w.kproj, xs + E, dynk, 1, (int) E, (int) (KV * taps), nullptr);
            ik::matmul_f32(w.vproj, xs + 2 * E, dynv, 1, (int) E, (int) (KV * taps), nullptr);
            ik::conv_dynamic(ks[(size_t) vis], k, w.kconv, dynk, kc, (int) (KV * D), (int) taps, (int) KV, nullptr);
            ik::conv_dynamic(vs[(size_t) vis], v, w.vconv, dynv, vc, (int) (KV * D), (int) taps, (int) KV, nullptr);
        }
        cap_seam(sink, vis, "k_conv", kc, (int) (KV * D)); cap_seam(sink, vis, "v_conv", vc, (int) (KV * D));
        ik::qkv_prep(q, kc, vc, r, w.qclip, w.rclip, 1, (int) H, (int) KV, (int) D, (int) R, c.qk_clip, c.rel_clip, c.rms_eps, L.full, c.log_scale_alpha, (float) c.log_scale_floor, pos, nullptr);
        cap_seam(sink, vis, "q", q, (int) (H * D)); cap_seam(sink, vis, "k", kc, (int) (KV * D)); cap_seam(sink, vis, "v", vc, (int) (KV * D)); cap_seam(sink, vis, "r", r, (int) (H * R));
        ik::kv_append(Kc[(size_t) vis], Vc[(size_t) vis], kc, vc, pos, (int) KV, (int) D, nullptr);
        ik::attention(q, r, gate, sinkv, Kc[(size_t) vis], Vc[(size_t) vis], w.relbias, att, attn_scratch, pos, L.full ? 0 : (int) c.window, L.full ? (int) c.rel_extent : (int) c.window, cap, (int) H, (int) KV, (int) D, (int) R, nullptr);
        cap_seam(sink, vis, "attn_gated", att, (int) (H * D));
        mmvq_f(8, w.out, att, (int) (H * D), (int) E, proj, aq);
        K::gdn_conv_step(ps[(size_t) vis], proj, w.postconv, a, E, taps, nullptr);
        cap_seam(sink, vis, "attn_out", a, (int) E);
        // the stream: a block boundary restarts it at a
        if (boundary) { copy(blk + (size_t) nb * E, h, (size_t) E); copy(rr, a, (size_t) E); }
        else ik::add(h, a, rr, (int) E, nullptr);
        const int nbl = nb + (boundary ? 1 : 0);   // the blocks this layer's FFN residual sees (the boundary's input counts)
        copy(blk + (size_t) nbl * E, rr, (size_t) E);
        ik::attn_res(blk, nbl + 1, w.ffn_res_q, 1, m_in, (int) E, nullptr);
        if (msgs_on) { control(w, 3, m_in, msgs + 3 * E); cap_seam(sink, vis, "loop_ffn_in", m_in, (int) E); }
        copy(x, m_in, (size_t) E);
        rms_w(x, w.ffn_norm);
        // the router and the experts
        ik::matmul_f32(w.router, x, rl, 1, (int) E, (int) c.n_expert, nullptr);
        K::inkling2_route(rl, w.router_bias, 1, (int) c.n_expert, (int) c.n_expert_used, c.expert_weight_scale, d_idx, d_w, nullptr);
        int32_t idx[16];
        float wt[16];
        CK(cudaMemcpy(idx, d_idx, (size_t) c.n_expert_used * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(wt, d_w, (size_t) c.n_expert_used * 4, cudaMemcpyDeviceToHost));
        picks[(size_t) vis].assign(idx, idx + c.n_expert_used);
        CK(cudaMemset(moe, 0, (size_t) E * 4));
        K::quantize_q8_1_rows(x, 1, E, xq, nullptr);
        for (int64_t j = 0; j < c.n_expert_used; ++j) {
            const int e = idx[j];
            const uint8_t* plane = w.gu_dev + (uint64_t) e * w.gu_plane;
            K::iq_mmvq(w.etype, plane, xq, gb, (int) E, (int) FF, 1, nullptr);
            K::iq_mmvq(w.etype, plane + w.gate_bytes, xq, ub, (int) E, (int) FF, 1, nullptr);
            const float sg = w.gs.empty() ? 1.0f : w.gs[(size_t) e], sd = w.ds.empty() ? 1.0f : w.ds[(size_t) e];
            ik::situ(gb, ub, hb, (int) FF, sg, c.situ_beta, c.situ_linear_beta, nullptr);
            K::quantize_q8_1_rows(hb, 1, FF, hq, nullptr);
            K::iq_mmvq(w.etype, w.dn_dev + (uint64_t) e * w.dn_plane, hq, db, (int) FF, (int) E, 1, nullptr);
            ik::axpy(moe, db, wt[j] * sd, (int) E, nullptr);
        }
        K::gdn_conv_step(fs[(size_t) vis], moe, w.ffnpost, mm, E, taps, nullptr);
        cap_seam(sink, vis, "ffn_out", mm, (int) E);
        ik::add(rr, mm, h, (int) E, nullptr);
        cap_seam(sink, vis, "out", h, (int) E);
        (void) L;
        return nbl;
    }

    void step(int64_t token, int pos, std::vector<float>* out, const SeamSink& sink) {
        const int64_t E = c.n_embd;
        if (pos >= cap) { std::fprintf(stderr, "inkling2 engine: position %d beyond the %d the caches hold\n", pos, cap); std::exit(1); }
        const uint64_t rb = C::row_bytes(kQ8_0, E);
        ik::dequant_q8_0_row((const uint8_t*) d_embd + (uint64_t) token * rb, h, (int) E, nullptr);
        rms_w(h, d_embd_norm);
        int nb = 0, pre_nb = 0, nms = 0;
        const int64_t core0 = c.core_start, clen = c.core_len();
        for (int vv = 0; vv < c.n_visits; ++vv) {
            const int64_t l = c.stored_of_visit(vv);
            bool msgs_on = false;
            if (vv == core0) { copy(pre_h, h, (size_t) E); pre_nb = nb; }
            if (vv >= core0 && vv < core0 + c.num_loops * clen) {
                const int64_t p = (vv - core0) / clen, i = l - core0;
                if (i == 0 && p > 0) { copy(h, pre_h, (size_t) E); nb = pre_nb; }   // a new pass restarts from the prelude
                if (p > 0) {   // this layer's four messages from the pass before (bank + its final stream)
                    ik::attn_res(ms, nms, d_loop_rq + (size_t) 2 * i * E, 2, tmp, (int) E, nullptr);          // [q's message, the MoE's message]
                    copy(msgs + 0 * E, tmp, (size_t) E); copy(msgs + 3 * E, tmp + E, (size_t) E);
                    ik::attn_res(ms, nms, d_loop_kq + (size_t) 2 * i * E, 2, tmp, (int) E, nullptr);          // [k's message, v's message]
                    copy(msgs + 1 * E, tmp, (size_t) E); copy(msgs + 2 * E, tmp + E, (size_t) E);
                    msgs_on = true;
                }
            }
            nb = layer(l, vv, nb, msgs_on, sink, pos);
            const bool core_visit = vv >= core0 && vv < core0 + c.num_loops * clen;
            if (core_visit && l == c.core_end - 1) {   // a pass ends: its blocks after the first and its final stream become the next pass's bank
                nms = 0;
                for (int b = pre_nb + 1; b < nb; ++b) copy(ms + (size_t) nms++ * E, blk + (size_t) b * E, (size_t) E);
                copy(ms + (size_t) nms++ * E, h, (size_t) E);
            }
            // a block boundary keeps its input in the block list (layer() wrote it at blk[nb_before]); the stream carried on in h
        }
        // the head: the output attention residual, the norm, the logits through the Q8_0 head, the softcap
        copy(blk + (size_t) nb * E, h, (size_t) E);
        ik::attn_res(blk, nb + 1, d_out_res, 1, hd, (int) E, nullptr);
        rms_w(hd, d_out_norm);
        mmvq_f(8, d_out, hd, (int) E, (int) c.n_vocab, logits, xq);
        ik::softcap(logits, (int) c.n_vocab_real, c.final_softcap, nullptr);
        if (out) { out->resize((size_t) c.n_vocab_real); CK(cudaMemcpy(out->data(), logits, (size_t) c.n_vocab_real * 4, cudaMemcpyDeviceToHost)); }
        CK(cudaDeviceSynchronize());
        pos_cur = pos;
    }
};

Engine::Engine(const GgufFile& g, const Config& c, const Layout& lay, int max_ctx) : p_(new Impl(g, c, lay, max_ctx)) {}
Engine::~Engine() = default;
void Engine::step(int64_t token, int pos, std::vector<float>* logits, const SeamSink& sink) { p_->step(token, pos, logits, sink); }
void Engine::reset() { p_->reset(); }
const std::vector<std::vector<int>>& Engine::last_picks() const { return p_->picks; }
uint64_t Engine::expert_bytes() const { return p_->expert_b; }
uint64_t Engine::dense_bytes() const { return p_->dense_b; }

}  // namespace inkling2
}  // namespace strata
