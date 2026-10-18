# GLM-5.3-Flash (`glm5-next`) - what the forward pass is

Source of truth: llama.cpp PR 27773, commit `de25343` (`src/models/glm5-next.cpp`, `deepseek4.cpp` for mHC, `delta-net-base.cpp`
for the delta rule), read on 18 Oct 2026.  Every shape below was checked against the real GSQ-RCO 3.0-bit file by
`strata-glm5-check` (1383 tensors, nothing unnamed).  The reference build lives in `/mnt/nvme/ref/llama-glm`.

## Shape of the model

45 trunk layers (+ 1 optional MTP block, `blk.45`).  Hidden N = 4096, vocab 154880, 64 heads.
Layer kind from `attention.head_count_kv[l]`: 0 = KDA (34 layers), 1 = nope-MLA with indexer (11 layers: 3, 7, ..., 43).
Layers 0-2 have a dense SwiGLU FFN (12288); layers 3-44 a MoE: 288 experts, 8 per token, 1 shared expert, expert width 2048.

## Residual stream: mHC (4 streams)

`x` is `[N, 4]` per token.  Input embedding is repeated into the 4 streams.  Before the attention and again before the FFN of every
trunk layer:

    flat   = rms_norm(x.flatten(4N), eps)                       # no weight
    mixes  = hc_fn[4N, 24] . flat                               # 24 = (2 + 4) * 4
    pre    = sigmoid(mixes[0:4]  * scale[0] + base[0:4])  + eps
    post   = 2 * sigmoid(mixes[4:8] * scale[1] + base[4:8])
    comb   = sinkhorn( mixes[8:24] * scale[2] + base[8:24] ).reshape(4, 4)
    cur    = sum_h pre[h] * x[h]                                # fed to the norm + sublayer
    ...sublayer...
    x'[dst] = post[dst] * out + sum_src comb[dst, src] * x[src]

`sinkhorn`: row softmax, `+eps`, then column-normalise (`/(colsum+eps)`), then `iters-1` more (row-normalise, column-normalise); iters = 20, eps = 1e-6.
After the last layer: `h = mean over streams`, `rms_norm(h) * output_norm`, `logits = output . h`.  The MTP block has no hc tensors.

## KDA layer (linear attention, 34 layers)

    q, k, v = silu(causal_conv1d(W{q,k,v} x))        # kernel 4, depthwise per channel, 8192 channels each; state: last 3 inputs
    q, k    = l2_norm(q), l2_norm(k)  per head (eps 1e-6);  q *= 1/sqrt(128)
    g       = f_b(f_a(x)) + dt_bias                   # f_a [N,128], f_b [128,8192]
    g       = lower_bound * sigmoid(-g * A)           # A = ssm_a [64] = -exp(A_log); lower_bound = -5 (kda.gate_lower_bound); per key channel
    beta    = sigmoid(ssm_beta x)                     # [64]
    per head, state S [128 key, 128 value]:
        S   = S * exp(g)[:, None]                     # per-key-channel decay
        d   = (v - S^T k) * beta
        S   = S + k (x) d
        o   = S^T q
    o = rms_norm(o, ssm_norm) * sigmoid(g_b(g_a(x)))  # per head
    out = attn_output . o                             # [8192 -> 4096]

State per layer: 64 x 128 x 128 f32 = 4 MB (x34 = 142 MB) + conv state 3 x 8192 x 3.

## MLA layer (11 layers, nope: no rotary at all, `rope.dimension_count = 0`)

    qr = rms_norm(W_q_a x, q_a_norm)                  # 1536
    q  = W_q_b qr -> [64 heads, 256]
    c  = rms_norm(W_kv_a_mqa x, kv_a_norm)            # 512: THE cache (one latent head, no separate K/V, no rope key)
    q' = per head: W_k_b[h] (256 -> 512) applied to q (absorbed)   # k_b [256, 512, 64]
    attend over the SELECTED cells only:  p = softmax(q' . c_sel / sqrt(256));  o = p . c_sel   # 512 per head
    o  = W_v_b[h] (512 -> 256) o_h                    # v_b [512, 256, 64]
    out = attn_output . o                             # [16384 -> 4096]

Cache per token per MLA layer: 512 values (+ the indexer row below).  11 layers: tiny KV, long context is cheap.

## Indexer (DSA, with k-pool compression; every MLA layer owns one, `attention.indexer.types` all true)

    iq = W_idx_q_b qr -> [32 heads, 128]
    ik = layer_norm(W_idx_k x, k_norm weight + bias)  # [128]
    ig = W_kpool_gate x                               # [128] gate per channel
    cache row per token: [ik | ig]
    every completed group of kpool = 4 consecutive tokens -> one pooled key:
        pooled = sum_over_4( softmax_over_4(ig + ape) * ik )  per channel      # ape [128, 4]
    w     = (W_idx_proj x) / sqrt(128 * 32)           # [32]
    score = sum_h w[h] * relu(iq[h] . pooled)         # one score per pool
    choose the top (2048 / 4 = 512) pools by score; their 4 member cells each = 2048 cells,
    plus the incomplete tail group (kpool-1 cells) when `kpool_select_tail`
    the attention attends over exactly those cells (and the causal mask)

For a context shorter than 2048 cells, every cell is selected (the indexer is a no-op), which makes short-context parity checks independent of it.

## MoE (layers 3-44)

    logits = ffn_gate_inp . x                         # [288]
    p      = sigmoid(logits)                          # expert_gating_func = 2
    select top-8 of  p + exp_probs_b                  # bias used for SELECTION only
    w      = p[selected]; w = w / sum(w); w *= 2.5    # expert_weights_norm, expert_weights_scale
    out    = sum_e w_e * down_e( clamp-SwiGLU(gate_e x, up_e x) )  + shared_expert(x)     # swiglu clamp 10.0
    expert tensors: gate/up [4096, 2048, 288], down [2048, 4096, 288]; shared expert: same shapes with 1 expert

The clamp: `swiglu_clamp_exp` (10.0) - see `build_moe_ffn` / `build_ffn` for where it is applied before porting.

## MTP block (blk.45, from Unsloth; not in the GSQ-RCO file)

An MLA + MoE layer (with an indexer, no hc tensors) plus `nextn.eh_proj [2N, N]`, `nextn.enorm`, `nextn.hnorm`, and an optional
`nextn.shared_head_norm`.  llama.cpp loads it but throws "NextN graph not implemented yet", so there is no reference for its
graph.  The expected recipe (DeepSeek-style) is `eh_proj(concat(enorm(embed(next token)), hnorm(h_nextn)))`, then the block, then
the shared output head; `index_share_mtp = true` says the block reuses the main model's indexer selection.  To be confirmed against the
original model code before the head is trusted.
