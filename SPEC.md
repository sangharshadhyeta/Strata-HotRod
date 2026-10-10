# Engine spec: Qwen3.8-Flash-Next (arch `qwen4exp`), GSQ-RCO generic GGUF

Release: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (the generic, unpruned one; chosen over the Coder release).
Target variant: **IQ3_S. LOCKED by the user on 2026-10-08. No further variant comparison.** The Q2_0 header was read earlier and is kept only as a size reference.

**DECISION (user, 2026-10-10): this is no longer a from-scratch engine.** We fork Strata (branch `solid-wins` in
`/home/Projects/Strata`, C++/CUDA engine with a Python server) and port solid.cpp's measured wins into it one concept at
a time, each step built and tested against the previous build before the next (the hot-rod rule). This supersedes the
earlier lines "written from scratch", "Host language: Python" and "GPU kernel route and allowed libraries: [OPEN]"
(Strata's existing kernels and dequantisers are the route; vLLM and solid.cpp stay read-only references). Sections 1-11
remain the model reference (the math, the tensors, the formats); sections 12-13 are the residency policy that is to be
implemented inside Strata's engine; section 14 is the port order.

**"Decided" in this file records a choice by the user, not a result.** Every decision stays UNTESTED until its row in the
test ledger (section 15) holds a measurement on the target; a decision whose test fails is reopened, whoever made it.
The expected effects written in the text (for example the NVMe traffic of 13.55) are hypotheses with their test named.

Theme, unchanged: keep off the GPU and out of RAM everything that is not being used at that instant, and bring each
piece in just in time for its computation. The unit of change is Strata's round (at most 96 swaps per 4 verify windows,
13.48), not a per-token or per-request event as in solid.cpp.

Test result (T1, 2026-10-10): the stock engine with `STRATA_IO_PREFETCH=1 STRATA_IO_PF_STAGE=1` (no code change) reached an
all-runs median of 27.2 and 27.5 tokens/s against 17.0 and 17.1 for the default, with the drive read per token down from
about 150 to 80 MB (section 15, T1). That is above the win bar of the next paragraph on the all-runs median; the code prompts
alone are still under it.

Rule (user, 2026-10-10): the transfer rate between the hardware is a resource as important as RAM and VRAM. Three links
are budgeted like capacities: NVMe to RAM, RAM to VRAM over PCIe, and NVMe straight to VRAM (design rates in 13.53:
NVMe 3.55 GB/s, pinned PCIe 24.5 GB/s, RAM 36.6 GB/s, VRAM 254 GB/s). Every step reports bytes moved per link per
generated token and the link's utilisation; a swap, a promotion or a prefetch is priced against the link it uses (from
NVMe it costs the slow link, from RAM the PCIe link); a change that raises the hit rate but pushes more bytes over a
saturated link is not a win. The baseline already shows the disk link, not compute, setting the speed.

Win bar (user, 2026-10-10): more than 25 tokens/s decode on the target (IQ3_S, this machine).
Baselines measured on the target on 2026-10-10 with the stock Strata engine (greedy, thinking off, MTP on, 3 warm
requests per kind after a warm-up): IQ3_S at 32K context: code 16.9, reasoning 16.9, prose 29.7, all-runs median 17.2
tok/s, NVMe read 1.87 GB/s (about 150 MB per generated token). IQ3_S with the English draft vocabulary: 16.6 (no gain).
Unsloth UD-IQ4_XS at 8K context: 4.9 tok/s, NVMe read 1.48 GB/s (about 370 MB per token). Each file measured once.

Single-model inference engine. This spec is derived from reading the reference code (solid.cpp, vLLM, Strata) and the
GGUF headers. The reference repos are read-only. Nothing here is generic: every choice is for this one model.

Target: Rocky Linux, RTX 3060 12 GB (Ampere, sm_86), 32 GB RAM, 512 GB NVMe.

Status tags used below:
- [SRC] read directly in reference source, two or more sources agree
- [SRC1] read in one source only, needs a second check
- [INFER] (no longer used; section 2 is now fully verified from the header)
- [OPEN] not known yet, must be found by experiment

All numeric values marked [INFER] or [OPEN] are decided by the GGUF header (step 1 of the plan).
No number in this file may be hard-coded into the engine until it is read from the file.

---------------------------------------------------------------------------------------------

## 0. Configuration (what we are building for)

### 0.1 Target machine

| Item | Value |
|---|---|
| OS | Rocky Linux |
| GPU | NVIDIA RTX 3060, 12 GB VRAM (Ampere, compute capability 8.6) |
| RAM | 32 GB |
| Disk | 512 GB NVMe |
| Dev machine (not a target) | Windows laptop, Ryzen 7 7735U, 15.3 GB RAM, integrated Radeon, C: 127.6 GB free, F: 79.4 GB free |

### 0.2 Model release

| Item | Value |
|---|---|
| Repo | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (generic, unpruned) |
| Architecture tag | `qwen4exp` |
| Routed experts | 512 per layer, 10 active per token |
| Trunk layers | 48 (12 full attention, 36 GDN) |
| Tensors per file set | 1,224 |
| Files per variant | 2 shards: weights (shard 1), n-gram table (shard 2, identical in every variant) |
| Base model (BF16) | 354 GB |

### 0.3 Variants (all sizes from the model card and Strata fingerprints)

| Variant | Avg bpw | Shard 1 (weights) | Shard 2 (n-gram) | Total | Expert types inside | Role for us |
|---|---|---|---|---|---|---|
| Q2_0 | 2.40 | 37.6 GB | 28.8 GB | 66.4 GB | Q2_0 only | Header read (data/gguf_header_q2_0) |
| IQ2_XS | 2.50 | 39.2 GB | 28.8 GB | 68.0 GB | IQ1_M, IQ2_S, IQ2_XXS, Q2_0 | |
| IQ3_XXS | 3.00 | 47.0 GB | 28.8 GB | 75.8 GB | IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, Q2_0 | |
| IQ3_S | 3.50 | 54.8 GB | 28.8 GB | 83.6 GB | IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q2_0 | **LOCKED target. Header read (data/gguf_header_iq3_s)** |

Published quality (vendor-reported, 3 tasks; BF16 base = 354 GB): IQ3_S task average 93.26 vs BF16 93.12 (AIME25 100.00, GPQA-D 92.93, LCB v6 86.86). IQ3_XXS 92.57, IQ2_XS 89.16, Q2_0 89.07.
Caveats: self-reported by the quantiser; only three tasks; no independent or long-context test found. Unsloth publishes no quality numbers for its quants (UD-Q4_K_XL 111 GB up to Q8_0 192 GB), so it cannot be ranked against IQ3_S from public data.
Swift 1.5 (ukisai) is a quantisation of a different, fine-tuned model (Swift Flash Next), not this model, and reports KL divergence only. Not comparable.
IQ3_S needs these dequantisers for routed experts: IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_0 (per Strata fingerprint). Its own header must be read before section 6 is final.

The n-gram table (shard 2) is IQ4_NL for every variant, so IQ4_NL is needed from the start.

### 0.4 Sizes from the headers (placement is NOT decided)

| Piece | Q2_0 build | IQ3_S build (target) |
|---|---|---|
| Routed experts (144 tensors) | 33,973,862,400 B (all Q2_0) | 50,292,326,400 B (mixed, see section 6) |
| Everything else in shard 1 | 3,636,231,680 B | 4,511,552,000 B |
| Shard 1 total (last tensor offset) | 37,610,094,080 B | 54,803,878,400 B |
| Shard 2: n-gram table (IQ4_NL) | about 28.8 GB | about 28.8 GB |
| Header size | 11,024,650 B | 11,024,355 B |

Rule from the user (2026-10-08): do not assume what goes in VRAM. Where each tensor group lives
(GPU, pinned RAM, mmap from NVMe) is decided by measurement on the target machine, not by guess.
The machine offers 12 GB VRAM, 32 GB RAM and a 512 GB NVMe. Nothing here says what fits where.

Header comparison (Q2_0 vs IQ3_S, both parsed by my reader): tensor names, shapes and order are identical in the two files.
All model hyperparameters and the whole tokenizer are identical. Only the per-tensor quantisation types differ,
plus a few general.* keys (file_type 41 vs 26, description, tags, license/base_model keys present only in Q2_0).
So section 2 holds for every variant; section 6 is per variant.

### 0.5 Model portions and placement (IQ3_S build; sizes from its header)

The last column is our own decision record. Only decided items are filled in. "Undecided" means it will be settled by measurement on the target.

| # | Portion | What it is | Tensors | Size | Touched per token | Placement |
|---|---|---|---|---|---|---|
| 1 | Token embedding | Token id to a 2560-vector | 1 | 0.34 GB | 1 row | Undecided |
| 2 | Output head | 2560-vector to 248,320 logits | 1 | 0.52 GB | all | Undecided |
| 3 | Hyper-connections | Mix the 4 residual streams (2 per layer + 1 final) | 387 | 1.28 GB | all | Undecided |
| 4 | Routed experts | 512 per layer x 48 layers (gate, up, down) | 144 | 50.29 GB | 10 of 512 per layer | Tiered by the usage score (decided 2026-10-08): top slice is the fast tier, next slice is not all kept in RAM either, the rest only on NVMe. Cut points: undecided, to be measured |
| 5 | Shared expert | One always-on expert per layer, plus its gate | 192 | 0.15 GB | all | Undecided |
| 6 | Router | Picks the 10 experts per layer | 48 | 0.13 GB | all | Undecided |
| 7 | Indexer | Picks key blocks to attend to (12 attention layers) | 48 | 0.04 GB | 12 layers | Undecided |
| 8 | Attention | Q, K, V, output, norms (12 layers) | 72 | 0.44 GB | 12 layers | Undecided |
| 9 | GDN | Linear attention: projections, conv, decay, norm (36 layers) | 324 | 1.56 GB | 36 layers | Undecided |
| 10 | PLE layer | The extra block on layer 1 | 6 | 0.07 GB | layer 1 only | Undecided |
| 11 | N-gram table | 16 heads of about 20M rows x 160 values (shard 2, IQ4_NL) | 1 | about 28.8 GB | 16 rows | **NVMe (decided)** |
| | Runtime state | KV + indexer cache (12 layers), GDN state (36 layers), PLE history | | small | | Undecided |
| | **Total** | | **1,224** | **about 83.6 GB** | | |

Routed experts and the n-gram table are 79.1 GB of the 83.6 GB (95%). Everything else is 4.5 GB and is used on every token.

---------------------------------------------------------------------------------------------

## 1. Model overview

A 48-layer decoder. Every layer is:

    HC-mix -> token mixer -> HC-combine -> HC-mix -> MoE FFN -> HC-combine

- Token mixer is Gated DeltaNet (GDN, linear attention) on 3 of every 4 layers.
  Layer i (0-based) is full attention when (i+1) % 4 == 0, otherwise GDN. [SRC: solid.cpp load_arch_hparams, `full_attention_interval` default 4; vLLM `layer_types`]
- Full-attention layers use QSA: dense GQA attention restricted to the top-k blocks chosen by a
  small "indexer". Dense is used until the context passes the budget. [SRC]
- There is no layer norm. Hyper-connections (HC) with 4 parallel residual streams replace
  every norm. The last HC mixer also replaces the output norm. [SRC]
- FFN is a sparse MoE (softmax router, top-k routed experts) plus one shared expert with a sigmoid gate. [SRC]
- One layer also has PLE: an n-gram hash embedding injected into the residual. Its table is the
  huge 51B-parameter one, held on NVMe/RAM. [SRC]
- An MTP (multi-token-prediction) draft head exists in the base checkpoint. It is separate
  and optional; both converters drop it. Out of scope for v1. [SRC]
- Vision tower (Qwen3-VL ViT, `mmproj` file, 908 MB) is separate. Out of scope for v1.

The generic GGUF has 512 routed experts per layer, 10 active per token [SRC: model card]. (The Coder release has the same 1,224 tensors with 256 experts, so pruning only changes the expert dimension.)

## 2. Hyperparameters (read from the Q2_0 shard-1 header, 2026-10-08) [VERIFIED]

Source of truth: `data/gguf_header_q2_0/header.json` (parsed by my own GGUF reader, header = 11,024,650 bytes, GGUF v3, 73 metadata keys, 1,223 tensors in shard 1; shard 2 holds the 1,224th).

| Symbol | Meaning | Value | GGUF key |
|---|---|---|---|
| L | trunk layers | 48 | `qwen4exp.block_count` |
| ctx | max context | 262,144 | `context_length` |
| D | hidden size | 2560 | `embedding_length` |
| vocab | tokens | 248,320 | `tokenizer.ggml.tokens` length |
| HC | hyper-connection streams | 4 | `hyper_connection.count` |
| R_hc | HC low rank | 320 | `hyper_connection.low_rank` |
| n_head | attention heads | 24 | `attention.head_count` |
| n_kv_head | KV heads | 2 | `attention.head_count_kv` |
| d_head | head dim (key = value) | 256 | `attention.key_length`, `value_length` |
| n_rot | rotary dims | 64 | `rope.dimension_count` |
| rope sections | IMRoPE sections | [11, 11, 10, 0] | `rope.dimension_sections` |
| rope base | | 10,000,000 | `rope.freq_base` |
| rms eps | | 1e-6 (9.99999997e-07) | `attention.layer_norm_rms_epsilon` |
| full-attention interval | | 4 (layers 3,7,...,47) | `full_attention_interval` |
| GDN conv kernel | | 4 | `ssm.conv_kernel` |
| GDN d_k = d_v | | 128 | `ssm.state_size` |
| GDN H_k (key heads) | | 16 | `ssm.group_count` |
| GDN H_v (value heads) | | 48 | `ssm.time_step_rank` |
| GDN inner size | = H_v*d_v | 6144 | `ssm.inner_size` |
| n_expert | routed experts | 512 | `expert_count` |
| n_expert_used | top-k | 10 | `expert_used_count` |
| n_ff_exp | routed expert width | 640 | `expert_feed_forward_length` |
| n_ff_shexp | shared expert width | 640 | `expert_shared_feed_forward_length` |
| indexer heads n_h | | 4 | `attention.indexer.head_count` |
| indexer dim d_idx | | 128 | `attention.indexer.key_length` |
| indexer top-k budget | | 2048 | `attention.indexer.top_k` |
| compress ratio r | on attention layers | 4 (0 on GDN layers) | `attention.compress_ratios` |
| PLE layer | 0-based layer index | 1 | `ple.layers` = [1] |
| PLE n-gram N | | 3 | `ple.ngram_size` |
| PLE heads per n-gram P | | 8 (so 16 heads total) | `ple.heads_per_ngram` |
| PLE conv kernel K_p | | 4 (dilation = N = 3) | `ple.conv_kernel` |
| d_ple | row width | 160 (16 heads x 160 = 2560 = D) | `embedding_length_per_layer_input` |
| PLE multipliers | uint64 x 3 | 23703573157769, 20109073645365, 8052911324071 | `ple.layer_multipliers` |
| PLE head vocab sizes | uint64 x 16 | about 20,000,000 each (primes: 20000003, 20000023, ...) | `ple.head_vocab_sizes` |
| PLE head offsets | uint64 x 16 | 0, 20000003, 40000026, ... (cumulative) | `ple.head_offsets` |
| PLE table rows | 16 x ~20M | about 320,001,000 rows x 160 = 51.2B params | shard 2 |
| PLE EOS id | for n-gram cuts | 248044 | `ple.eos_token_id` |
| PLE image token id | | 248056 | `ple.image_token_id` |
| tokenizer | | `gpt2` BPE, pre-tokenizer `qwen35` | `tokenizer.ggml.model`, `.pre` |
| merges | | 247,587 | `tokenizer.ggml.merges` |
| tokenizer EOS | | 248046 | `tokenizer.ggml.eos_token_id` |
| BOS / PAD | | 248044 (add_bos = false) | |
| sampling defaults | | temp 1.0, top_k 20, top_p 0.95 | `general.sampling.*` |
| chat template | Jinja, about 10k chars | in `tokenizer.chat_template` | |
| file type / quant version | | 41 / 2 | `general.file_type` |

Derived and checked against tensor shapes: attn_q width 12288 = 24 x 256 x 2; attn_k/v 512 = 2 x 256;
attn_output input 6144 = 24 x 256; GDN qkv 10240 = 2 x (16 x 128) + 48 x 128; HC norm 10240 = 4 x 2560.

Things the header flags that I did not know before:
- PLE EOS id (248044) is the BOS/PAD id, while the tokenizer EOS is 248046. The reference converter takes the last entry of `eos_token_id`. Whether 248044 is right for the hash cut is a question for the reference run. [OPEN]
- The GGUF keeps the tensors under different names than solid.cpp's graph code (`output_hc_*`, not `hc_head_*`). Always use the names in the file.

## 3. Tensor inventory (GGUF names are `blk.N.*`, ggml dim order is [in, out])

Global:
- `token_embd.weight` [D, vocab]
- `output.weight` [D, vocab] (not tied)
- `output_hc_norm` [HC*D], `output_hc_down` [HC*D, R_hc], `output_hc_up` [R_hc, HC*D]   (final mixer, no inject)
- `per_layer_token_embd.weight` [d_ple, ple_rows]   (the huge n-gram table, lazily read; shard 2 of the file)

Every layer:
- `hc_attn_norm` [HC*D], `hc_attn_down` [HC*D, R_hc], `hc_attn_up` [R_hc, HC*D], `hc_attn_inject` [HC*D, HC]
- `hc_ffn_*` same four tensors
- `ffn_gate_inp` [D, n_expert]; `ffn_gate_exps`/`ffn_up_exps` [D, n_ff_exp, n_expert] (maybe fused as gate_up); `ffn_down_exps` [n_ff_exp, D, n_expert]
- shared expert: `ffn_gate_shexp`, `ffn_up_shexp` [D, n_ff_shexp]; `ffn_down_shexp` [n_ff_shexp, D]; `ffn_gate_inp_shexp` [D] (one scalar logit per token)

Full-attention layers (i+1)%4==0:
- `attn_q` [D, 2*d_head*n_head] (query and output-gate interleaved per head), `attn_k`, `attn_v`, `attn_output` [d_head*n_head, D]
- `attn_q_norm`, `attn_k_norm` [d_head]
- indexer: `indexer.q_proj` [D, n_h*d_idx], `indexer.k_proj` [D, d_idx], `indexer.q_norm`, `indexer.k_norm` [d_idx]
  (HF checkpoint has one fused `index_qk_proj`; converter splits first n_h*d_idx rows to q, rest to k) [SRC]

GDN layers:
- `attn_qkv` [D, 2*key_dim + value_dim] where key_dim = d_k*H_k, value_dim = d_v*H_v
- `attn_gate` [D, value_dim] (this is z)
- `ssm_conv1d` [ssm_d_conv, 2*key_dim + value_dim]
- `ssm_dt.bias` [H_v], `ssm_a` [H_v] (stored as -exp(A_log)), `ssm_beta` [D, H_v], `ssm_alpha` [D, H_v]
- `ssm_norm` [d_v], `ssm_out` [value_dim, D]

PLE layer only (exactly one; index is 1, 0-based, from `ple.layers`):
- `ple_key` [D, HC*D], `ple_value` [D, D]
- `ple_norm_key`, `ple_norm_query`, `ple_norm_conv` [HC*D]
- `ple_conv1d` [K_p, HC*D]

Note: gamma folding. The HF checkpoint uses Gemma-style norms (`out = xhat * (1 + w)`). The
reference converter stores `1 + w` for `ple.norm_*` and indexer q/k norms, and the "inherited
rule" folds HC norms and attn q/k norms too. [SRC] Check in the GGUF whether each norm tensor
already holds `1 + w`; if it does, multiply by it plainly. [OPEN: per-tensor, verify numerically]

## 4. Forward pass

Notation: x is the residual state `[HC, D]` per token. All norms use f32 math.

### 4.1 Embedding and start state [SRC]
    e = token_embd[token]            # [D]
    x = repeat(e, HC)                # HC identical copies

### 4.2 Per-layer loop [SRC]
    for il in 0..L-1:
        if il is the PLE layer:  x = PLE(x, il)
        (m, inj) = HC_mix(x, attn weights)
        a = token_mixer(m)               # GDN or QSA attention, [D]
        x = HC_combine(x, a, inj)
        (m, inj) = HC_mix(x, ffn weights)
        f = MoE(m)                       # [D]
        x = HC_combine(x, f, inj)
    (h, _) = HC_mix(x, output_hc_*, no inject)    # [D], this is the final "norm"
    logits = output.weight @ h

### 4.3 HC_mix [SRC, solid.cpp build_hc_mix == vLLM GatedResidual.mix, hc_per_branch_norm=True]
    xn_s  = x_s / sqrt(mean(x_s^2) + eps)           for each stream s separately (length D)
    xn    = flatten(xn_s) * gamma                    # gamma has HC*D entries
    lo    = silu( down @ xn / HC )                   # down: [HC*D -> R_hc]
    gate  = sigmoid( up @ lo )                       # up:   [R_hc -> HC*D]
    mixed = mean over s of ( gate_s * xn_s )         # [D]
    inject = inject_w @ xn                           # [HC]  (skipped for the final head mixer)
Returns `mixed` and `xn`/`inject` for the matching combine.

### 4.4 HC_combine [SRC]
    w_s = 2 * sigmoid( inject_s / HC )               # inject_s from this layer's matching mix
    x_s = x_s + block_out * w_s                      # block_out [D] broadcast over streams
Uses the pre-combine `x` (the residual that went into the mix), not the normalised copy.

### 4.5 Full attention with QSA [SRC for structure, OPEN for details]
    qfull = Wq m                       # [n_head, 2*d_head]: first d_head = q, second = output gate
    q = rmsnorm(q, attn_q_norm); k = rmsnorm(Wk m, attn_k_norm); v = Wv m
    q, k = IMRoPE(q, k, pos)           # 4-section interleaved multimodal rope, text uses same position on all axes
    attn = softmax(q k^T * (1/sqrt(d_head))) v, over the selected keys (GQA)
    out  = Wo( attn * sigmoid(gate) )
Selection (QSA):
- Indexer keys: `k_raw = Wk_idx m` (one head, d_idx), stored RAW in an indexer cache.
- Keys are grouped into blocks of r consecutive tokens. Block key = mean of the raw keys in the
  block, then RMSNorm (`indexer.k_norm`), then RoPE at the block position. [SRC1: solid.cpp pools before norm; vLLM kernel order not yet read, VERIFY]
- Query: `q_idx = Wq_idx m` as n_h heads, RMSNorm (`indexer.q_norm`), RoPE.
- Score per block = sum over indexer heads of relu( q_idx_h . blockkey ). [SRC1]
- Every token inherits its block's score. Mask invisible cells. Take top `width = min(n_kv, budget + r - 1)`
  tokens, which is whole blocks plus the incomplete tail block. [SRC]
- Attention then runs only over those cells.
- Below the budget (n_kv <= budget) selection keeps everything, so attention is dense. [SRC]
Open: exact visibility/causality rules for the tail block and for a query inside a block (the mask bias). [OPEN]

### 4.6 Gated DeltaNet layer [SRC; this is the Qwen3.5/3-Next GDN with one change]
    qkv = Wqkv m ; z = Wgate m ; beta = sigmoid(Wbeta m) ; alpha = Walpha m   # per value head
    qkv = silu( causal_depthwise_conv1d(qkv, ssm_conv1d) )   # conv state: last (ssm_d_conv-1) inputs
    split into q [H_k, d_k], k [H_k, d_k], v [H_v, d_v]
    q = l2norm(q); k = l2norm(k)                    # eps = rms eps
    repeat q,k from H_k to H_v heads (H_v is a multiple of H_k)
    g = ssm_a * softplus(alpha + dt_bias)           # ssm_a already = -exp(A_log); g <= 0
    q = q / sqrt(d_k)
    per value head h, state S_h in R^{d_k x d_v}, per token t:
        S = exp(g) * S
        d = (v - S^T k) * beta
        S = S + k d^T
        o = S^T q
    o = rmsnorm(o, ssm_norm) * sigmoid(z)           # the only difference from Qwen3.5: sigmoid, not silu
    out = ssm_out o
Prefill may use the chunked (block-parallel) form; decode uses the recurrence above. Both must
give the same result to within float tolerance. State to keep per GDN layer: S [H_v, d_k, d_v] f32 and the conv tail.

### 4.7 MoE FFN [SRC for shape, OPEN for router details]
    logits = Wgate_inp m                            # [n_expert]
    p = softmax(logits); idx = topk(p, n_expert_used)
    w = p[idx], renormalised to sum 1 ("expert_weights_norm") then * expert_weights_scale   [SRC1, verify both flags in GGUF]
    routed = sum_j w_j * Down_e( silu(Gate_e m) * Up_e m )
    shared = Down_s( silu(Gate_s m) * Up_s m ) * sigmoid( w_gate_inp . m )
    out = routed + shared
Note: no load-balancing bias in the router. [SRC comment, solid.cpp]

### 4.8 PLE layer (the n-gram embedding) [SRC]
Token ids for hashing:
    for each position t, ctx[0] = token[t]; for s = 1..N-1: ctx[s] = token[t-s]
    an EOS (or a position before the sequence start) cuts the window: that and every older
    entry become EOS. The token's own EOS does not cut its own context.
Row lookup for each head:
    for n = 2..N: mixed_n = XOR_{j<n} ( ctx[j] * mult[j] )
        for g in 0..P-1:  head h = (n-2)*P + g
            row_h = (mixed_n mod vocab_size[h]) + offset[h]
    emb = concat_h table[row_h]                      # [(N-1)*P*d_ple] per token
Then, with E = emb and x the HC residual state:
    key   = ple_key E   -> reshape [HC, D] -> per-stream RMSNorm * ple_norm_key
    query = x           -> per-stream RMSNorm * ple_norm_query
    s     = sum_D(key * query) / sqrt(D)    per stream   -> [HC]
    gate  = sigmoid( sign(s) * sqrt(max(|s|, 1e-6)) )
    value = ple_value E ([D]); gated = value broadcast over streams * gate
    c     = per-stream RMSNorm(gated) * ple_norm_conv
    conv  = silu( causal depthwise conv over time, kernel K_p, dilation N, weights ple_conv1d )   # tap k reads (K_p-1-k)*N tokens back
    x     = x + gated + conv
The conv needs a history of (K_p-1)*N tokens of `c` per sequence (a per-layer state).
CRITICAL OPEN POINT on the hash: solid.cpp computes in unsigned 64-bit and uses unsigned `%`.
vLLM computes in signed int64 with `torch.remainder` (non-negative result) and comments that
the products "may overflow int64". If a product exceeds 2^63 the two disagree. The constants
are about 45 bits and token ids are up to ~18 bits, so this can happen. [OPEN: decide by
testing against reference-generated row ids. Do not pick one from memory.]

## 5. Persistent state per sequence

- Full-attention layers: K and V cache (n_kv_head x d_head), plus the raw indexer key cache (d_idx per token).
  Only 12 layers (every 4th) have these, so the KV cache is small. [SRC]
- GDN layers (36): S [H_v, d_k, d_v] f32 and conv tail [ssm_d_conv-1, 2*key_dim+value_dim].
- PLE layer: conv history [(K_p-1)*N, HC*D] and the last N-1 token ids (for the n-gram window).
- Rollback (needed only for speculative decoding): snapshots of recurrent state per accepted-token count. Out of scope for v1.

## 6. Quantised weight formats [VERIFIED from the Q2_0 header]

Tensor count per type in shard 1 (1,223 tensors): BF16 483, F32 292, type42 (= Q2_0) 202, Q3_K 91,
IQ4_XS 56, Q4_K 38, Q5_K 16, Q4_0 16, Q6_K 12, Q5_0 7, IQ4_NL 7, Q8_0 2, F16 1.
Shard 2 adds IQ4_NL for the n-gram table.

| Type | Seen in | Block layout |
|---|---|---|
| Q2_0 (id 42) | all 144 routed expert tensors, many shared-expert / attn_q / attn_gate / attn_qkv tensors, ple_key | 2.25 bits per weight. Block size and scale layout: [OPEN, read ggml source in refs/solid.cpp] |
| Q3_K | token_embd, attn_k, some shared experts and attn tensors | standard ggml |
| Q4_K | attn_gate, attn_output, ssm_out, some others | standard ggml |
| Q5_K | output.weight, attn_v, ssm_out, others | standard ggml |
| Q6_K | attn_output, ssm_out, shared experts | standard ggml |
| IQ4_XS | attn_qkv, attn_gate, shared experts | standard ggml |
| IQ4_NL | some shared-expert down tensors; the whole n-gram table (shard 2) | standard ggml |
| Q4_0, Q5_0, Q8_0 | shared-expert down tensors | standard ggml |
| BF16 | HC down/up/inject, router, shared-expert gate, indexer q/k projections, ple_value, ssm alpha/beta, output_hc_down/up | 16-bit |
| F16 | ple_conv1d | 16-bit |
| F32 | all norms, ssm_conv1d, ssm_a, ssm_dt.bias | 32-bit |

So the dequantisers needed for the Q2_0 build are: Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL, Q4_0, Q5_0, Q8_0,
plus BF16/F16/F32 loads. The other variants add more i-quants (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, per Strata fingerprints).
Reference code for block layouts: `refs/solid.cpp/ggml/src` (ggml-common.h, ggml-quants.c).

### 6.1 IQ3_S build (target), shard 1 [VERIFIED from its header]

Tensor count per type (1,223 tensors): BF16 484, F32 292, Q6_K 129, IQ4_NL 86, Q4_K 47, IQ4_XS 45, IQ2_S 40, Q5_K 35,
IQ3_XXS 34, IQ3_S 20, type42 (Q2_0) 9, F16 1, Q8_0 1.

Routed expert tensors (144) by type: IQ4_NL 39 (18.40 GB), IQ3_XXS 34 (10.92 GB), IQ2_S 40 (10.75 GB),
IQ3_S 20 (7.21 GB), Q2_0 9 (2.12 GB), IQ4_XS 2 (0.89 GB). These match the Strata fingerprint exactly.

Dequantisers needed in total for IQ3_S: IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q2_0 (experts); Q4_K, Q5_K, Q6_K, Q8_0 (dense);
BF16, F16, F32. IQ3_S differs from Q2_0 in having no Q3_K, Q4_0, Q5_0 tensors, but adding Q6_K at scale, IQ2_S, IQ3_XXS, IQ3_S and Q8_0.
Union over the two builds is already 16 formats.

Which tensor uses which type varies by layer (RCO picks per tensor). The full per-tensor type list is in `data/gguf_header_q2_0/header.json`.

## 7. Memory placement

Not decided. See section 0.4. The engine must make placement a configuration of each tensor group
(device memory, pinned host memory, memory-mapped file) so it can be changed by experiment without
rewriting kernels. Experiments to run on the target, in order:
1. Measure usable VRAM and RAM with the CUDA context and OS loaded.
2. Measure NVMe sequential and random-read throughput for the access sizes the model needs
   (one expert tensor slice, one n-gram row of 160 values).
3. Try placements for each tensor group and record tokens per second.

## 8. Tokenizer and chat format

- Model `gpt2` (byte-level BPE), pre-tokenizer name `qwen35`, 248,320 tokens, 247,587 merges, token types array. All in the header. [VERIFIED]
- The last tokens are `[PAD248316]` ... `[PAD248319]` (padding up to the table size). [VERIFIED]
- BOS = PAD = 248044, EOS = 248046, `add_bos_token` = false. [VERIFIED]
- Chat template: Jinja, about 10k characters, stored in `tokenizer.chat_template`; it handles images, videos, tools. [VERIFIED present, not yet read in full]
- The pre-tokenizer regex for `qwen35` is not in the file as text; it is implied by the name. [OPEN: find the exact regex in refs/solid.cpp (llama-vocab.cpp) or Qwen docs]
- Special tokens such as `<|vision_start|>` and `<|im_start|>` appear in the token list. [to extract]

## 9. Build plan (each step has a pass/fail check)

1. GGUF reader: parse header, metadata, tensor table, shard 2. Dump every value in sections 2, 3, 6. Check: tensor count is 1224 per file set [SRC fingerprint].
2. Dequantisers for each type in the file. Check: against a reference dump of a few blocks.
3. Tokenizer. Check: round-trips on text, matches reference token ids.
4. Single-layer forward on CPU for each layer kind (HC, GDN, QSA attention, MoE, PLE), in f32. Check each against reference outputs.
5. Full 48-layer CPU forward for one token. Check: logits against a reference run.
6. State (KV, GDN, conv), prefill and decode loop, sampling.
7. CUDA kernels on the 3060: dense GEMV/GEMM, GDN, attention, MoE; expert paging from RAM.
8. Speed work, then (optionally) MTP drafting, vision.

Reference outputs: we need ground truth logits and per-layer activations from a trusted run.
Options: run solid.cpp/Strata/vLLM on a machine that fits the model (e.g. rented), or the
target PC itself once built. We do not need to run them here.

## 10. Known disagreements and risks between the reference implementations

- PLE hash arithmetic (signed vs unsigned), section 4.8.
- Indexer block key: pool-then-norm (solid.cpp) vs the order inside vLLM's fused kernel. Check `refs/vllm/vllm/models/qwen4_exp/nvidia/ops/qsa_prepare.py`.
- solid.cpp MTP head attends densely (marked "v1 simplification" there). Not relevant to the trunk.
- Strata's own commit log says stage-buffer pinning corrupted IQ3_S decode after long prompts. A reminder to test long prompts early.

## 11. Source map (where each fact came from)

- Architecture graph: `refs/solid.cpp/src/models/qwen4exp.cpp`
- HF checkpoint conversion, norm folding, int64 hash constants: `refs/solid.cpp/conversion/qwen4exp.py`
- GDN recurrence: `refs/solid.cpp/src/models/delta-net-base.cpp`
- HC reference math: `refs/vllm/vllm/models/qwen4_exp/common/hyperconnection.py`
- PLE hash/gate/conv doc: `refs/vllm/vllm/models/qwen4_exp/common/ops/ple.py`
- Decoder layer wiring: `refs/vllm/vllm/models/qwen4_exp/nvidia/model.py`
- QSA indexer: `refs/vllm/vllm/models/qwen4_exp/nvidia/indexer_qsa.py` (first 330 lines read)
- Not yet read: qsa.py and ops/qsa*.py (attention details), mtp.py, Strata kernels, tokenizer code, Q2_0.

## 12. Reference: how other engines decide which experts to keep close to the GPU (for study only)

Read from the cloned repos on 2026-10-08. We write our own; none of this is adopted.
Claims about solid.cpp come from its docs and code comments; nothing here was run.

### 12.1 The Strata profile file (`refs/Strata/data/expert-profile.bin`)

- Format: magic `STRP`, version 1, then n_layers, n_expert, slots, n_ranked (uint32 each), then n_ranked pairs (layer uint16, expert uint16) ranked most-routed first, then a layer x expert table giving each pair's rank. I checked that the table matches the ranked list exactly.
- The file holds an order, not counts. Routing skew cannot be measured from it directly.
- 24,576 pairs for the full model (48 x 512); 12,288 for the Coder profile (48 x 256).
- The top ranks are spread over all layers. Top 4,105 pairs (16.7% of all): 53 to 109 per layer, median 86. Layers 41, 15, 36, 31 and 40 contribute the most to the top 2,000.
- It is built by `tools/make_profile.py` from routing traces recorded with `--dump-routing`.
- Skew shows up in measured hit rates. Strata reports a held-out hit rate of 0.645 with 4,105 slots (16.7% of experts), about 3.9 times a uniform pick. Its own source warns that the earlier 0.657 figure was in-sample and that one coverage model claimed 94.9% where 69.2% was measured at 5,805 held.
- solid.cpp logged hit rates of 0.26 to 0.33 on Qwen with 1,350 slots (5.5% of experts), about 5 to 6 times uniform. Different rigs and workloads, so the two are not directly comparable.

### 12.2 Strata vs the user's solid.cpp

| Aspect | Strata | solid.cpp |
|---|---|---|
| Where all experts live | Pinned in RAM (its docs list 48 to 64 GB per variant) | Memory-mapped from the model file |
| What picks the GPU set | A shipped ranked profile, then adaptive swaps that follow the conversation | Demand-driven: admit after the second miss, LRU/heat eviction; no shipped profile |
| Prediction | Router lookahead one layer ahead, warming pages and optionally filling spare slots | Lookahead node in the graph, a live-trained pre-router (used for eviction and substitution, not admission), an atlas of similar experts |
| On a miss | The CPU computes the same expert in place while the GPU does the cached ones | CPU by default; newest commits fetch the expert to the GPU on the miss |
| Exactness | Every expert computed exactly | Approximate options exist: substitution (on by default), router bias toward cached experts, gate-mass skip |
| Prompt reading | Streams next-layer experts to the GPU during attention | Cache is decode-only by default |
| Own measurements | Held-out hit rate 0.645 at 4,105 slots | Hit rate 0.26 to 0.33 at 1,350 slots; experts reread about 14 times from NVMe at about 28% of drive speed |

Notes from solid.cpp's own ledger (docs/ROADMAP.md): substitution lowered greedy agreement with the exact reference to 21.0% against a same-build noise floor of 67.1%; the Qwen speed numbers were taken under disk contention and are marked "rerun before trusting".

## 13. Expert residency design: decision table (we go step by step)

Scope: which routed experts the GPU holds, and how that set changes. Placement of everything else is a separate decision (section 0.5).
Facts for the Strata and solid.cpp columns were read from the cloned repos (docs, comments, code) on 2026-10-08. Nothing was run.
The Decision column holds only what the user has decided. "Open" means not discussed yet.

| # | Step | Strata | solid.cpp | Decision |
|---|---|---|---|---|
| 1 | Starting state | Cache filled at start from a shipped ranked profile (all 24,576 pairs, ranked by routing frequency on a reference workload) | Starts empty; opt-in history file can pre-admit the hottest, bounded to half of each pool | **As Strata: start from a shipped ranked profile** (decided 2026-10-08) |
| 2 | Trigger | Periodic: one adaptive round every `adapt_every` verify windows (default 4); a round is skipped if the last round's copies have not landed | Event-driven: a miss can trigger an admission at once (admitted after the 2nd miss) | **Decided (user, 2026-10-08): trigger like solid.cpp (event-driven: a miss can start a change); swapping like Strata (a candidate is paired with the least-used resident expert of the same layer and exchanged for it only when clearly more used; budgets per layer never change).** Count rules, margin, rate limits are still open (13.2, 13.7) |
| 3 | What is counted | Open, see 13.2 | | **Open: next to discuss** |
| 4 | Victim choice | Per layer: least-used resident expert, swapped only if the candidate's count exceeds it by 1.5 | Pool-wide: probation drains first, protected capped at 50%, heat tiebreak among 8 candidates, 25% hysteresis, 8 fresh misses before a newcomer may evict | Open |
| 5 | Slots per layer | Fixed by the profile; swaps stay inside a layer. Its first-come global form measured 2.97% hit rate (cache filled by ~26 layers of the first token); its analysis scored 8 slots/layer at 21.4% and 64/layer at 70.4% | Pools per tensor shape and device; layers compete inside a pool | **Decided: per-layer budgets, no layer competes with another** (user, 2026-10-08). REVISED the same day (13.32): the starting budgets come from the phase 1 weights, and per-layer expert counts then move live in phase 2 within the floor of 10, a cap and a slow decay. **Decided: per-layer budgets come from the profile, counted in bytes not slots** (user, 2026-10-08), because expert size differs by layer in the GSQ-RCO files (see 13.5) |
| 6 | How much moves per round | At most `adapt_swaps` (default 96) per round, ranked by gain | At most 8 fills per node; queue up to 128 jobs / 512 MiB; one low-priority copy stream per device | Open |
| 7 | Copy timing | Batched copies between windows while the drafter works; the next window waits for them | Background fills that may land after the layer that needed them | Open |
| 8 | A miss while a move is pending | The CPU computes the evicted expert meanwhile | CPU by default; newest commits fetch to the GPU on the miss; optional substitution answers the miss with a resident stand-in | **Decided (user, 2026-10-08): every miss is covered by a resident stand-in. No exact miss path, so no uncovered misses.** Quality must be shown with a KL-divergence measurement against an exact run, not greedy agreement. Open: what chooses the stand-in (13.2c) |
| 9 | Forward-looking signals | None in the swap round (router lookahead exists separately and warms pages) | Lookahead matmul, trained pre-router, atlas | Open |
| 10 | Persistence | Optional: write the learned state back as a profile; next start uses it | Optional merged history file, refused on model mismatch | Open |
| 11 | Exactness | Every expert computed exactly; output not bit-identical run to run | Exact for hits and misses; approximations (substitution, router bias, gate-mass skip) are optional | Open |

### 13.1 Trigger: history (superseded by the decision in row 2)

Periodic was first chosen, then reopened, then event-driven was chosen. The notes below are kept for the record; the refinements apply only if the trigger is ever revisited. Proposed refinements to discuss:
- Define the period in tokens, not verify windows. Windows exist because of speculative decoding, which we do not have at first.
- A round only submits copies; the swap becomes visible when the copy has landed. Compute never waits on a copy, and a late copy is simply used a round later.
- Skip a round while the previous round's copies are in flight (as Strata does).
- Period and per-round cap are parameters, measured on the target. Strata's own calibration found a slower-CPU machine gained +7.9% by swapping more.
- Evidence is not conclusive either way. Event-driven reacts faster; periodic batches copies and gives one place to apply hysteresis. A period of one token approaches event-driven, so periodic with a tunable period can cover both ends.

### 13.2 Counts: Strata vs solid.cpp (read from code 2026-10-08; decision column is for us to fill)

| Question | Strata | solid.cpp | Decision |
|---|---|---|---|
| What counts are kept? | **One** float per (layer, expert): `usage`, 24,576 floats | **Three** things: (a) per non-resident expert a demand record: `count` (misses), `unsubstituted` (misses no stand-in covered), `exact_required` (misses at a top router rank); (b) per resident slot a `heat`; (c) optional cross-run history of selection counts | |
| What increments it? | +1 for every expert the router picks, for every row of a verify window (hits and misses alike) | (a) +1 per observed miss; (b) +4 per hit (cap 2^20); (c) +1 per selection, merged across runs | |
| Are non-resident experts counted? | Yes, same counter as residents | Yes, but in a separate record (a), a different unit from heat | |
| Is the count reset on a move? | No. An evicted expert keeps its count and can swap back in | Yes. The demand record is erased when the fill lands; a slot's heat is zeroed when its content is replaced | |
| Decay | All counts x0.7 after each adaptive round (clock = rounds, every 4 windows) | Heat is halved every 512 fills (clock = fills, not time or tokens); demand records do not decay, they reset | |
| Threshold to become a candidate | Decayed count at least 2.0 | `count` at least 2 while free slots exist; at least `readmit_after` = 8 once the pool is full (so the throttle is built into the count) | |
| Margin to beat the victim | Candidate must exceed the victim's count by more than 1.5, same units | No direct candidate-vs-victim comparison: victim = lowest heat among the 8 coldest-recency slots, with a heat hysteresis | |
| Compared per layer or globally? | Per layer | Per pool (all layers sharing a tensor shape) | |
| Which routing is counted? | The router's real picks | The router's picks; admission can be rank-weighted (`exact_required`, weight 0 by default) and counts misses covered by a stand-in separately | |
| Where does the count start? | Zero; the profile only sets the starting residents | Zero; the history file, if used, pre-admits up to half of each pool | |
| Known problem recorded in its own code | Counts include drafted rows that may be rejected (not checked) | Decay fires only on fills, so within one busy request heat balloons before decay runs; a hard 50% cap on the protected segment was added to bound it | |

Note for the discussion: Strata's swap compares a candidate and a victim in the same unit (usage). solid.cpp compares different units (a miss tally against slot heat), which is why it needs thresholds, hysteresis and a throttle. A swap rule "like Strata" needs one common unit.

### 13.2b Counts: current lean and candidate features (2026-10-08)

Lean (not decided): the user leans to Strata's counts (one decayed usage counter per layer-expert, compared in one unit).

Under substitution the count must follow what the router wanted (the missed expert E), not what was computed (the stand-in S). Otherwise S looks hot, E looks cold, and E is never swapped in.

Candidate features from solid.cpp (all optional or default-off there, no reported win in its notes; hypotheses to test):
1. Covered vs uncovered misses: tally separately the misses that a stand-in answered and those it did not (`unsubstituted`). A swap gains more for an uncovered miss.
2. Rank-aware demand: extra weight for misses at the router's top ranks (`exact_required`, weight 0 by default there).
3. A higher bar once the pool is full (threshold 2 rises to 8). Strata's margin of 1.5 plays the same role, so likely not needed as well.
Not proposed: heat with protected/probation segments (built to bound heat that balloons between fills; a single decayed count does not have that problem), cross-run history (Strata can write its learned state back as a profile).

New open sub-questions under row 8 (miss handling):
- Which misses may substitution cover? solid.cpp's policy: only ranks at or below a minimum rank (with top-10 routing and minimum rank 4, ranks 0 to 3 are about 40% of activations and are computed exactly). This is a policy setting there.
- What answers a miss that substitution may not cover: a CPU path as in Strata, or a synchronous fetch to the GPU as in solid.cpp's newest commits?

### 13.2c Counts decided, and what substitution-for-every-miss leaves open (2026-10-08)

Decided:
- Counts: Strata's single decayed usage counter per layer-expert, compared in one unit, with one change: the count follows the router's pick (the missed expert E), not the stand-in that was computed.
- None of solid.cpp's other count machinery is used (heat with protected/probation segments, per-expert miss records, cross-run history, rank-weighted admission as a separate mechanism).
- Every miss is answered by a resident stand-in of the same layer. "Covered vs uncovered misses" no longer exists as a feature.

Open:
- Stand-in choice: **SUPERSEDED on 2026-10-08 (see 13.12): the atlas is dropped; the router's own ranking picks the stand-in.** Was: by similarity, the atlas decides. (solid.cpp builds its atlas from topic-labelled probe runs and co-activation data; rank-aware demand there only weights admission.) Open: how the atlas is built for this model, how the closest resident expert is scored, and whether the router rank breaks ties.
- Can one resident expert stand in for several misses in the same token?
- A similarity map for this model has to be built; solid.cpp's atlas tool was run on Gemma only.
- Quality bar: the KL divergence we accept against an exact run. Reference points: Strata's held-out hit rate was 0.645 at 4,105 slots, so about a third of picks would be stand-ins at that setting (different rig and slot sizes than ours). solid.cpp's own notes say the top-ranked picks carry most of the gate weight, which is why its code limits stand-ins to lower ranks (a policy setting there).

### 13.2d What Strata decides per pick, and what is left for us (read from src/core/expert_source.cpp)

For each distinct expert the router picks in a layer, Strata decides who computes it: resident in VRAM goes to the GPU from its slot; on a second GPU goes there; already landed in a spare swap slot goes to the GPU; otherwise it is a miss, where `(misses * pcie_frac) / 256` of the misses (the last ones in routing order, at most 64) are computed by the GPU reading them over PCIe and the rest by the CPU from RAM. The computed expert is always the one the router asked for. Strata has no stand-in, no similarity data and no atlas.

For us the miss branch disappears: every pick is computed on the GPU, either the router's expert (resident) or its stand-in (not resident). No CPU path, no PCIe split. What carries over is the (layer, expert) to slot table and grouping tokens by expert.

### 13.3 Evidence on substitution from solid.cpp's own notes (docs/ROADMAP.md, docs/LEARNINGS.md)

| Model | Speed vs exact | Greedy agreement with exact | Same-build noise floor |
|---|---|---|---|
| Gemma-4 | +1.6% | 21.0% | 67.1% |
| Qwen3.8-Flash-Next | -1.0% | 54.6% | 41.5% |
| Ornith-1.5-35B (retired) | +8.11% (the figure behind its default-on setting) | not in these notes | not in these notes |

The Qwen runs were taken under disk contention and are marked "rerun before trusting". Greedy agreement is a weak metric because identical settings only agree 42 to 67% of the time. Proposed measure: KL divergence of next-token distributions against an exact run on fixed text, plus a task score. Substitution also needs a similarity map (the atlas) built for this model.

### 13.4 Check for the claim that cross-layer competition hurt solid.cpp

Not proven. solid.cpp already prints per-layer hit rates in its SUMMARY output (docs/LEARNINGS.md, L4). A layer sitting near zero while others are high would confirm starvation. Strata's 2.97% was the first-come admission form, not an eviction-based pool, so it is a warning, not a measurement of solid.cpp.

### 13.5 Expert size is not uniform in the GSQ-RCO files (from the two headers)

One expert = its gate, up and down slices (each is 1/512 of a stacked tensor). The type is chosen per tensor, so it is the same for all 512 experts of a layer but differs between layers.

| Build | Bytes per expert | Notes |
|---|---|---|
| Q2_0 | 1,382,400 in every layer (1.32 MiB) | All 48 layers identical; gate, up and down are all Q2_0 |
| IQ3_S | 1,510,400 to 2,662,400, mean 2,046,400 (max/min 1.76x) | Seven distinct sizes across layers; in all 48 layers gate, up and down use different types (e.g. layer 0 IQ3_XXS + IQ3_XXS + IQ4_NL = 2,176,000; layer 1 IQ2_S + IQ2_S + Q2_0 = 1,510,400) |

Consequences:
- A budget must be counted in bytes. A slot count means different things in different layers.
- A layer's slots all have one size, so a swap inside a layer always swaps equal sizes. That is why the per-layer design needs no size matching during adaptation.
- A byte budget per layer comes from the profile: take pairs in rank order and add up their sizes. **Decided (user, 2026-10-08): fill by raw routing frequency, not frequency per byte.**
- Strata supports unequal slot sizes (`open_sized`, "each slot keeps its first size"); I have not checked whether its profile fill weighs size.
- The profile order depends on routing, which depends slightly on the quantisation, so each variant needs its own profile built from traces of that variant.

### 13.6 Check of the idea "more quantised layers are less important" (IQ3_S header x Strata profile)

Tested on 2026-10-08 using the per-layer expert sizes from the IQ3_S header and the ranking in Strata's shipped profile (one profile, one workload):
- Correlation between a layer's expert size and its share of the top-ranked pairs: 0.22 (top 2,000), 0.13 (top 4,105), -0.27 (top 8,000). Weak, and the sign changes.
- Mean rank position per layer vs expert size: Pearson 0.22, Spearman 0.21 (a negative value would mean bigger experts rank hotter).
- The 8 layers with the biggest experts have mean rank 12,360; the 8 with the smallest 11,891; all layers 12,288.
So this profile shows no clear link between a layer's bit-width and how often its experts are routed. The ranking-by-frequency decision does not depend on that link. GSQ-RCO picks bit-widths by quantisation sensitivity under a size budget, which is a different thing from routing frequency.

### 13.7 What the trigger + swapping decision leaves open

Strata's swap runs as a periodic round: it looks at all layers together, ranks swaps by gain, and keeps the best `adapt_swaps` (default 96). With an event-driven trigger the same swap happens when a miss arrives:
1. A miss on expert E in layer L is routed; E's count goes up.
2. If E qualifies (Strata: count at least 2), it is paired with the least-used resident expert V of layer L.
3. The swap happens only if E's count exceeds V's by the margin (Strata: 1.5). E replaces V in V's slot (same layer, same size), and V is computed elsewhere until it is swapped back in.

Open points, decided in the counts discussion:
- Rate limit: **decided as Strata (user, 2026-10-08)**: `adapt_swaps` = 96 swaps at most per round, a round every `adapt_every` = 4 verify windows, and no new round of copies while the previous round's copies are in flight. Both are tunable (Strata's docs show 160 swaps per window helping a slow-CPU machine by +7.9%). Units to fix later: what one "window" is in our engine (one token without speculative decoding; Strata's window checks about 2.4 to 3.2 tokens with its draft model). Strata's global ranking by gain across layers has no per-event equivalent; the cap above replaces it.
- Whether the counts decay on a clock, per swap or per token (Strata: x0.7 per round).
- Throttle against churn (solid.cpp: 8 fresh misses before a newcomer may evict).
- What happens to the missed expert while its copy is pending (row 8, leaning substitution).

### 13.8 The atlas vs the shipped profile (read from solid.cpp and Strata, 2026-10-08)

| | Strata shipped profile | solid.cpp atlas |
|---|---|---|
| Question it answers | How often is each (layer, expert) routed? (popularity) | Which experts are alike, i.e. used for the same things? (similarity) |
| Contents | 24,576 ranked pairs plus a rank table; order only | Per observed cell: position x, y, a specialization score, and a multi-dimensional embedding (`dims`; one note mentions 28 dimensions) |
| Built from | Routing traces: how many times each pair was picked | Topic-labelled probe prompts (27 prompts, 9 categories, a replication rule of at least 2 probes), and co-activation graphs (experts in the same routing decision; layer L top pick to layer L+1 top pick), factored spectrally; the number of axes is chosen from the data by an eigengap rule; accumulates over runs |
| Used for | Which experts start resident; fixes per-layer budgets | Choosing which resident expert stands in for a missed one (also warming and eviction in solid.cpp) |
| Needs the live resident set? | No | Yes: it picks among whatever is resident at that moment |
| Own warning in the code | Strata's source: ranking is workload-dependent (in-sample 0.657 vs held-out 0.645) | solid.cpp comment: correlation between 2D atlas distance and distance in the 28-dim embedding is -0.003 (noise), so decisions should consult the embedding, not the 2D picture |

The two are different inputs. The profile decides what is resident; the atlas decides which resident expert covers a miss.

Interaction to discuss: residents are chosen by popularity alone (decided), while stand-in quality depends on how close a resident is to the missing expert. A resident set picked purely by frequency may leave regions of the similarity space with no near resident. Not a decision, a point to measure.

Option to discuss: besides topic and co-activation proxies, similarity could be measured directly by how different two experts' outputs are on the same inputs. Not decided.

### 13.9 Atlas construction: decided and debatable (2026-10-08)

Decided (user): build the atlas from topic probes and co-activation, as solid.cpp does.
Debatable, not decided (user): matching experts by how different their outputs are. Whatever method builds the atlas, it is judged the same way: the KL divergence of the final output against an exact run, so the debate can be settled by measurement.
For: measures the quantity we care about. Against: needs real activations from a working model; an expert's output difference is only a proxy because the gate weight scales its contribution.
Sequencing: the atlas and the profile both need routing traces from a running model (our engine or a reference run). Neither can be built before that.

### 13.10 Atlas source narrowed, and the router-ranked stand-in (2026-10-08)

SUPERSEDED (see 13.12): was "co-activation only, topic probes dropped". No atlas is built at all now.
Stated rule (user): if the router ranks the experts and the top-ranked one is missing, the best-ranked resident becomes the substitute. Details to confirm:
1. Weight of the stand-in: its own router score renormalised with the other picks, or the weight of the expert it replaces.
2. Co-activation as a tie-break among the router's top residents, or replaced entirely by the router's ranking.
3. If the best-ranked resident is already one of the token's other picks, take the next one.
How it differs from an atlas: uses this token's own router scores (all 512 per layer), needs no stored map; it is the router's top-k restricted to resident experts. Closest solid.cpp relative: router bias toward resident experts (LLAMA_MOE_CACHE_BIAS). Its notes: Qwen hit rate 0.26 to 0.33 at bias 0.5; Gemma greedy agreement 31% against a 67% noise floor; Qwen runs flagged as contaminated, so weak evidence. Risk: when no resident scores well for a token, the best-ranked resident can still be a poor match. Judged, like any method, by KL divergence of the final output against an exact run.

### 13.11 Stand-in weight decided; the design as it stands (2026-10-08)

Decided (user): a stand-in takes its own router score, renormalised together with the other picks of that token. (Reconfirmed 2026-10-08; a misreading in 13.12 was corrected, see 13.13.)
Confirmed (user): the replacement (swap) system stays alongside the stand-ins. Swaps change slowly which experts are resident; stand-ins cover a miss for the current token.

Per token, per layer, as decided so far:
1. The router scores all 512 experts and takes its top 10 (the unmasked picks).
2. Each unmasked pick gets +1 in the decayed usage counter (Strata's counter). Counts follow the router's own picks, never the stand-ins. Counting stand-in-adjusted picks would make it impossible to learn that a non-resident expert is wanted.
3. A pick that is not resident is replaced by the best-ranked resident expert of the same layer; the weights of all picks are renormalised, the stand-in with its own router score.
4. A miss can start a swap (event-driven). The candidate (decayed count at least 2.0) is paired with the least-used resident of the same layer and replaces it only if its count exceeds the victim's by more than 1.5.
5. Rate limit: at most 96 swaps per 4 windows, none while the previous round's copies are in flight. Copies run in the background.
Start state and budgets: shipped ranked profile, per-layer byte budgets filled by raw routing frequency; the IQ3_S build is locked.

Closed later: co-activation dropped (13.12); duplicates decided (13.13).

### 13.12 No atlas; the stand-in does not take a score (2026-10-08)

Decided (user):
1. No atlas is needed. Neither topic probes nor co-activation data are built. The stand-in is chosen by the router's own ranking: the best-ranked resident expert of the same layer, by this token's router scores.
2. [CORRECTED, see 13.13] "Score" meant the +1 in the usage counter, not the gate weight. The earlier reading (the required expert's gate weight goes to the stand-in, no renormalisation) was a misunderstanding and is withdrawn.
3. The usage counter (Strata's) still counts the originally required expert.

Consequences:
- Nothing to build for similarity. The sequencing note in 13.9 now concerns only the profile, which still needs routing traces from a running model.
- Withdrawn: the consequences that depended on the gate-weight reading.

### 13.13 Correction (2026-10-08)

The user's point in 13.12 was about the usage counter: the +1 for a pick goes to the expert the router required, never to the stand-in, because the stand-in was not the router's pick. Normalisation stays.

Standing rules:
- Gate weights: the stand-in takes its own router score, renormalised together with the other picks of that token (decision of 13.11 stands).
- Usage counter: +1 only for the originally required (unmasked) pick. The stand-in's use is never counted.
- No atlas, no co-activation, no topic probes (13.12 item 1 stands). The stand-in is the best-ranked resident expert of the same layer by the token's own router scores.
- Duplicates: **decided (user, 2026-10-08)**: the stand-in is the best-ranked resident that is not already one of the token's picks. If the best-ranked resident is already picked, take the next one down the ranking. (Reason: with the stand-in's own score used, an already-picked resident would count twice.)

### 13.14 Waiting experts are tiered by the same score (2026-10-08)

Decided (user): not all experts outside the GPU tier are kept in RAM. They follow the same scoring. Ranking by the one usage counter: top slice in the fast tier (per-layer byte budgets, decided), next slice in RAM, the remainder only on NVMe, which holds the master copy of every expert. Cut points come from each tier's capacity and are measured on the target (not decided).

Consequences: every miss is covered by a resident stand-in, so RAM and NVMe are only sources for swaps. A slow tier delays when a swap lands; it never stalls a token.

Prior art read: solid.cpp notes "One policy, three tiers" (2026-08-15, rank by score, top slice to VRAM, next to RAM, rest on disk) and "The RAM buffer, reframed" (2026-08-16). Its measurements: advisory demotion through the OS moved a few percent against a 21x gap; an owned host buffer ran about 30% slower under a 5 GiB cgroup cap, because it took memory from the page cache and did promotions inline on the decode path under a lock; its stated fix is to promote on a background worker. Those tests used an artificial cap, not the real memory split. Strata's low-RAM mode keeps in RAM exactly what the GPU does not hold, and reads the rest from the file when it does not fit.

Open: (a) is the RAM tier a buffer we allocate and manage, or the OS page cache over a memory-mapped file (only the first can enforce the score); (b) does RAM get fixed per-layer byte budgets like the fast tier; (c) promotions into RAM as background prefetch driven by the ranking.

### 13.15 RAM tier answers (2026-10-08)

1. Owned versus page cache: decided by the user "based on the learnings of solid.cpp". Read as: the RAM tier is a buffer we allocate and manage ourselves, so the score can be enforced (the OS page cache can only be advised). Promotion runs on a background worker, never inline on the token path. If experts are read from NVMe with direct reads and not memory-mapped, the page cache does not compete with the buffer (to verify, not assumed).
2. Budget: decided by the user, follow solid.cpp: one global RAM pool, not per-layer. Reason (assistant, agreed): per-layer budgets matter in the fast tier only because each layer needs residents to stand in for misses; no computation depends on RAM, so a layer with little RAM only means slower swap-ins. Open: is the RAM copy exclusive of the fast tier (Strata and solid.cpp both are) or does it also duplicate what is resident there.
3. Promotions: CONFIRMED by the user 2026-10-08. Was proposed (assistant): a background worker promotes NVMe to RAM when an expert's score passes the weakest RAM resident by a margin, with a rate limit; the RAM slice is then "next in line" for the fast tier, so an event-driven swap copies from RAM. NVMe reads need many in flight (solid.cpp measured a single page-faulting thread at about 28% of drive speed).

### 13.16 The RAM tier is exclusive: swaps are exchanges (2026-10-08)

Decided (user): the RAM tier does not duplicate what is resident in the fast tier. A swap can be between RAM and the fast tier: the expert evicted from the fast tier is kept in RAM, in the place of the expert that replaced it. Matches Strata's exchange behaviour (copy the evicted expert back into the RAM place of the incoming one).

Consequences (assistant, not yet decided):
- A swap stays inside one layer, and all experts of a layer have one size, so the exchange swaps equal-sized places.
- The global RAM pool holds experts of several sizes. IQ3_S has seven distinct sizes (1,510,400 to 2,662,400 bytes), so the pool is organised as seven size classes.
- If the incoming expert comes from NVMe, the evicted one takes a free RAM place of its size class if its score is above the RAM cut (or displaces the weakest resident of that class); otherwise it is dropped. NVMe is a read-only master copy, so nothing is ever written back to it.
- Copying the evicted expert to host memory (about 2 MB over PCIe) counts against the swap rate limit.

### 13.17 Do layers differ in how many experts a token needs, and in swap demand? (2026-10-08)

Experts per token: the same in every layer. The header has one `expert_used_count` = 10 for all 48 layers (no per-layer array). Only the cost of a swap differs by layer, since expert size differs (1.51 to 2.66 MB in IQ3_S).

Swap demand per layer is not measured anywhere in our files. An indirect hint from Strata's shipped ranking: every layer receives the same routing mass (10 picks per token), so a layer with more pairs in the global top-N has its picks concentrated on fewer experts, and a layer with few has them spread out.
- Top 2,000 pairs: per layer min 7 (layer 0), median 43, max 67 (layer 41); 9.6x spread. By band: layers 0-11 hold 353, 12-23 hold 495, 24-35 hold 524, 36-47 hold 628.
- Top 4,105 pairs: min 53 (layer 0), median 86, max 109 (layer 37); 2.1x spread.
- Least concentrated: layers 0, 22, 1, 6, 2, 7, 28, 30. Most concentrated: layers 41, 15, 36, 31, 40, 39, 37, 42.
Reading: early layers, and layer 0 most of all, spread their picks widely, so they are the likeliest to miss and to need swaps. Limits: one profile from one workload, and the ranking shows concentration, not swap counts. solid.cpp's own test of a shallow-layer bonus (more cache for early layers) was a tie at 2x and -2.4% at 4x on Gemma, and was removed.
Suggestion (not decided): count the swap rate limit in bytes, not swaps, since a swap in a layer with big experts moves more data than one with small experts.

### 13.18 Open: should flat layers get more slots? (2026-10-08)

User's proposal: layers that need more swaps should get more slots. Facts: every layer needs exactly 10 experts per token (13.17); swap demand per layer is probably higher in early, flat layers (layer 0 least concentrated), measured only indirectly.
Conflict with the decided rule: filling per-layer budgets by raw routing frequency gives flat layers fewer slots (layer 0: 53 of 512 in the top 4,105 vs layer 37: 109), because each of their experts is individually less popular. That rule maximises total hits, not per-layer hit rate and not output quality under substitution.
Unknown: how much a stand-in damages the output in an early layer vs a late layer. solid.cpp's shallow-layer bonus test (tie at 2x, -2.4% at 4x on Gemma) measured speed with exact compute, so it does not transfer.
Cost note: layer 0 experts are 2,176,000 bytes each (IQ3_XXS + IQ3_XXS + IQ4_NL), so 20 more slots there is about 43.5 MB.
Options: A keep frequency-derived budgets; B shift slots toward flat layers to even out hit rates; C start with A, then set per-layer budgets from a per-layer KL measurement of stand-in damage. Not decided.

### 13.19 Per-layer budgets: option C, driven by router requests (2026-10-08)

Decided (user): option C, but the divergence is measured on the router's requests (how much of what the router asked for is not resident), not on output quality. A reference run is not possible, so there is no KL divergence against an exact run in this plan.
Consequence (noted once): the metric measures how much of the output relies on stand-ins, not how good the stand-ins are. Nothing in the plan measures actual output damage. (Possible later check without a third-party engine: our own engine run with every expert available as the reference. Not decided.)
How the divergence could be computed from what we already keep: per layer, unmet picks = share of counted router picks that fall on non-resident experts (from the usage counters); unmet gate mass = the router's weight on missing picks as a share of all its weight (needs one extra tally per layer). Suggested: unmet gate mass.
Open: (1) which measure; (2) objective: minimise total unmet demand (equals option A when every pick counts 1) or cap the worst layer (shifts slots toward flat layers); (3) timing: set once from a profile run, or rebalanced live from the usage counters (this would revise the earlier "fixed per layer from the start").

### 13.20 The stand-in rule equals masked top-k routing (2026-10-08)

Checked numerically: 20,000 random trials (random router scores over 512 experts, random resident sets of 20 to 200, top-10 picks). The decided rule (each missing pick is replaced by the best-ranked resident that is not already picked, weights renormalised) produced exactly the same set as taking the 10 best-scoring resident experts. Zero differences.
Consequence: the stand-in mechanism can be implemented as one mask on the router scores before the top-10 pick, restricting routing to resident experts. The usage counter still needs the unmasked top-10, so the router's top-10 is computed twice per layer (masked for computing, unmasked for counting).
Gate-weight vs pick-count measure (13.19): the user leans to the gate-weight measure. Whether it differs from the pick count depends on how concentrated this model's gate weights are across the 10 picks, which is unmeasured (solid.cpp measured it on Gemma only: ranks 4 to 7 hold about 32% of the mass). If the weights are flat across the picks the two measures nearly coincide.
User's question "should we deliberately push it?": readings offered: (A) bias the router toward resident experts, already done in the hardest form by the mask; (B) swap faster toward the router's top-requested experts; (C) change how the router spreads gate weight, not possible without changing the model. Awaiting the user's reading.

### 13.21 Minimum slots per layer (2026-10-08)

Hard floor: at least 10 resident experts per layer, because the router takes 10 per token and the stand-in rule (masked top-k, 13.20) needs 10 distinct residents. In IQ3_S bytes: 15.10 MB (lightest layer) to 26.62 MB (heaviest) per layer; 982.3 MB (0.98 GB) for all 48 layers, 1.95% of the 50.29 GB routed experts. Swaps are exchanges, so the count never dips below 10 during a swap. This is the minimum for the mechanism to run; a layer held at exactly 10 routes every token through the same 10 experts, so the practical floor for quality is higher and is to be measured.
Open: the user's "pushing" idea (router requests a light-weight expert while the heavy-weight expert's rank is kept) is unclear. Readings offered: R1 weight the counts by gate weight; R2 alter the router's own weights; R3 accept stand-ins for light misses but protect heavy-weight experts from eviction. Awaiting the user's example.

### 13.22 No bias on the router; the request alone says what to keep (2026-10-08)

Decided (user): nothing is done to the router beyond the decided mask. The router's requests alone tell which experts are better to keep. No soft bias toward resident experts, no weighting of the router's scores, no override of the heavy-weight expert's rank. The usage counter stays +1 per request (13.2c). The "pushing" question in 13.21 is closed.
The stand-in mask (13.20) is the decided stand-in rule, not a bias, and stays.
Still open from 13.19: the measure of unmet demand for the per-layer budgets (picks vs gate mass), the objective (minimise total vs cap the worst layer), and the timing (once vs live).

### 13.23 Per-layer budget procedure, as ordered by the user (2026-10-08)

1. Floor: every layer first gets its minimum of 10 resident experts (about 0.98 GB for all 48 layers in IQ3_S).
2. Divergence: the remaining bytes are distributed by each layer's unmet demand, measured from the usage counters (the share of the router's requests not resident).
3. Weight: the unmet demand is then refined by gate weight, so a miss on a heavily weighted pick counts for more than a miss on a light one.
Open inside step 2: where the next expert goes: (a) to the layer with the highest unmet demand (worst-first, evens out hit rates, favours flat layers such as layer 0), or (b) to the highest-demand expert in any layer (minimises total unmet demand, equals frequency fill). Timing (once from a profile run vs live) is still open.

### 13.24 Step 2 of the budget procedure: balance worst-layer-first with highest demand (2026-10-08)

User: worst layer first, but balanced with the highest demand; one worst layer must not fill the fast tier, or the other layers' stand-ins become useless.
Suggested rule (assistant, awaiting the user's decision): after the 10-expert floor, add the next expert to the layer that maximises gain = d x u, where d = router demand of that layer's next-best non-resident expert and u = that layer's current unmet share (refined by gate weight at step 3). This equals minimising the sum of squared unmet shares across layers; raising u to a higher power moves toward worst-first, power 1 is plain frequency fill. Floor 10 per layer, cap of about 2x the equal share per layer.
Synthetic check (made-up Zipf-style demand curves with exponent 0 to 1.6, 48 layers, 4,105 slots, floor 10; NOT model data): total/frequency fill gave slots min/median/max 27/94/142 with worst-layer unmet share 0.80 and mean 0.40; d x u gave 12/73/207, worst 0.58, mean 0.41; higher power 10/67/221, worst 0.55; near worst-first 10/58/234, worst 0.52, mean 0.43. Worst-first gave the flattest layer 234 slots (4x the median), which is the hogging the user described. A truly flat layer can never be covered well: each extra slot buys very little.
Real curves are unknown: Strata's profile stores only an order. Values (power, cap) to be tuned on the target from the unmet-gate-mass distribution per layer.

### 13.25 User's objective: keep misses homogeneous across layers (2026-10-08)

User's proposal: keep the misses homogeneous across the layers. Assistant assessment: a sound goal; it is the minimax rule (minimise the largest per-layer unmet share) with the 10-expert floor and a cap. Perfect homogeneity cannot be reached: layers with concentrated demand sit at the floor with some unmet share that cannot be raised by taking slots away.
Synthetic check (made-up Zipf-style curves, exponent 0 to 1.6, 48 layers, 4,105 slots, equal share 86; NOT model data):
- frequency fill: slots 27/94/142, unmet share lowest/highest 0.08/0.80, spread 0.72, mean 0.40.
- equalise misses, no cap: 10/51/239, 0.16/0.51, spread 0.35, mean 0.44.
- equalise, cap 2x equal share (171): 10/73/171, 0.16/0.64, spread 0.48, mean 0.43.
- equalise, cap 1.5x (128): 10/128/128, 0.16/0.73, spread 0.57, mean 0.42.
Caveat: equal miss rates are not equal damage; per-layer sensitivity to a stand-in is unmeasurable without a reference run. Equal miss rates are a fairness stand-in only.
Status: the user's lean, not yet final. Open: the cap value (1.5x, 2x the equal share, or other).

### 13.26 Are all layers equally important? (2026-10-08)

Not measured for this model; a reference run is not possible, so there is no direct measure. Closest signal: how the quantiser (GSQ-RCO) spent bits, chosen by sensitivity to quantisation error under a size budget. Routed-expert bits per weight (IQ3_S): layers 0-11 mean 3.08, 12-23 3.22, 24-35 3.47, 36-47 3.55; max 4.33 at layer 47 (last), 3.79 at layers 17, 25, 27, 28, 29, 35, 40; min 2.46 at layers 1, 8, 10, 13, 36. Attention layers average 3.48 bpw vs 3.28 for GDN layers. Dense weights follow suit: layer 47's attention tensors are all Q6_K, layer 23's include Q5_K.
Reading: later layers, and layer 47 most of all, were judged more sensitive to precision loss. This does not show that a stand-in does more damage there; the link is unproven. Earlier check: bit-width has no clear link with routing frequency (13.6).
Open: weight all layers equally (default) or weight the unmet share by a per-layer prior taken from the RCO bit-widths.

### 13.27 Standards for ranking layer importance (literature search, 2026-10-08)

Methods found (none run by us):
- Block Influence (ShortGPT): 1 minus cosine similarity between a layer's input and output hidden states, averaged over calibration text. Needs a forward pass of our own engine, no reference run.
- Angular distance (Gromov et al.): same similarity as an angle, usually over contiguous blocks; same layer order for single layers.
- Perplexity impact of removing a layer (Shortened LLaMA): needs our own loss on text; it is an output-quality measure.
- FinerCut: attention and feed-forward parts scored separately.
- MoE router statistics: router mass, REAP (routing probability x activation norm), learned router bias, router change under fine-tuning. Need router requests only.
- Quantisation sensitivity: what GSQ-RCO used; already in the file.
Caveats: Block Influence order varies with calibration data; hidden-state change is a proxy; rankings tuned on perplexity may not transfer to reasoning; sources disagree on which router signal to trust (activation frequency was less consistent than learned bias in one paper); a depth analysis of one large MoE found diffuse early routing and collapsing late-layer confidence, whereas Strata's profile for our model shows later layers more concentrated, so the pattern is model-specific.
Ranking available now (expert bits per weight, IQ3_S, higher = judged more sensitive to precision loss): tier 4.33 bpw: [47]; 3.79: [17, 25, 27, 28, 29, 35, 40, 44, 45]; 3.54: [0, 7, 14, 16, 18, 22, 31, 33, 37, 38, 41, 42, 43, 46]; 3.21: [2, 3, 4, 5, 6, 9, 11, 15, 19, 23, 24, 26, 32, 34, 39]; 3.04: [21]; 2.79: [12, 20, 30]; 2.46: [1, 8, 10, 13, 36]. The link between precision sensitivity and damage from a stand-in is unproven.
Proposal (assistant, awaiting the user): use the tiers as starting layer weights now; recheck with Block Influence once the engine runs on real text.

### 13.28 Layer importance: router statistics and quantisation sensitivity (2026-10-08)

User: router statistics and quantisation sensitivity, leaning to the second.
Roles (assistant): router statistics give the per-layer unmet demand (unmet gate-mass share, steps 2 and 3 of 13.23); quantisation sensitivity (the RCO bit-widths of 13.27) gives a per-layer weight on it. Score of a layer = weight x unmet share; the allocation (13.25) then works harder to keep misses low in layers the quantiser judged more sensitive.
Proposed weight: proportional to the layer's expert bits per weight divided by the mean (3.33 bpw): from about 0.74 (2.46 bpw layers 1, 8, 10, 13, 36) to about 1.30 (4.33 bpw, layer 47), a spread of 1.76x. Could be made steeper or flatter. The ranking has only seven tiers, so it is coarse.
Status: see 13.29 (decided phases). Block Influence from our own engine remains the later recheck (13.27).

### 13.29 Phases and weight scale decided (2026-10-08)

Decided (user): start with quantisation sensitivity, then move toward router statistics. The weight is proportional to the layer's expert bits per weight divided by the mean.
Phase 1 (no unmet-demand data yet): per-layer budgets from the weights alone (plus the 10-expert floor and a cap); within a layer, experts are taken in the profile's rank order. Phase 2 (once router statistics exist): the weighted unmet-share allocation of 13.25 and 13.28.
Open: what the weight scales in phase 1. Expert size is proportional to the layer's bits per weight, so if the weight scales BYTES per layer, every layer ends up with the same number of experts (the weight cancels). If it scales the expert COUNT per layer, sensitive layers hold more experts: with an illustrative equal share of 85 experts per layer, layer 1 (weight 0.74) gets about 63 experts (95 MB) and layer 47 (weight 1.30) about 111 experts (294 MB), a 3.1x spread in bytes.

### 13.30 Phase 1 scale decided (2026-10-08)

Decided (user): "as of now", in phase 1 the weight scales the expert COUNT per layer (not the bytes). Per-layer expert count is proportional to the layer's weight (expert bits per weight divided by the mean), with the 10-expert floor and the cap; experts within a layer are taken in the profile's rank order. Bytes per layer follow from count times that layer's expert size. "As of now" means phase 2 (router statistics) may change it.

### 13.31 Why "as of now" (assistant's reading of the user's reason, 2026-10-08)

Reading: a static weight from quantisation sensitivity cannot follow the topic; the topic in use may put little router weight on the layers the quantiser judged sensitive, so slots could sit where the current topic does not need them. Hence router statistics in phase 2.
Tension with an earlier decision: swaps change which experts a layer holds, but not how many (row 5 of 13: fixed per layer from the start). If demand shifts between layers with the topic, fixed counts cannot follow it. Open (this is question 3 of 13.19): in phase 2, do per-layer expert counts move live with the router statistics (within the floor of 10, a cap, and a slow decay), or stay fixed after a profile run with only the choice of experts adapting?

### 13.32 The live pruned model, and the one decision proposed for change (2026-10-08)

User's framing: the Coder variant is a static pruned version (256 of 512 experts per layer); our system runs a pruned version per request, live.
DECIDED (user, 2026-10-08: "yes live rebalancing"): row 5 "fixed per layer from the start" becomes "start from the phase 1 weights, then per-layer expert counts may drift live in phase 2", within the floor of 10, the cap, and a slow decay.
Kept: ranked-profile start, event-driven trigger with Strata-style swaps and rate limit, every miss covered by a resident stand-in, counts follow the router's own requests.
Cautions: the Coder keeps half the experts in every layer; our resident share is far smaller, so its 91.3% SWE-bench retention says nothing about ours. The Coder has the same expert count in every layer: one data point for equal counts, no evidence that quantisation weighting helps or hurts. Strata ships a Coder profile (data/expert-profile-coder.bin), so topic-specific starting profiles are possible; an idea, not a decision.

### 13.33 The pruned version is based on topic (2026-10-08)

User: the live pruned version is based on topic. Consequences: the usage counter learns what the topic asks for; its decay (Strata: x0.7 per round) is the topic memory, so the decay constant matters; the swap rate limit sets how fast the resident set turns over on a topic change.
Open: behaviour at a request/conversation boundary: (a) carry over the resident set and counts and let decay fade them; (b) reset to the shipped profile; (c) start from a saved topic-specific profile (as the Coder profile does; Strata can also save the learned state as a profile, even one per project). Not decided.

### 13.34 Live rebalancing decided (2026-10-08)

Decided (user): live rebalancing. Per-layer expert counts start from the phase 1 weights (13.30) and then move live in phase 2 as the router statistics change, within the floor of 10 experts per layer, the cap (value open), and a slow decay so one odd token cannot swing the budgets.
Open for this: what triggers a budget move (how often, how many experts per move, and whether it shares the swap rate limit of 96 swaps per 4 windows), and the cap value.

### 13.35 Request boundary decided: carry over (2026-10-08)

Decided (user): at a request/conversation boundary, carry over the resident set, the per-layer budgets and the usage counts from the previous topic. The decay handles the old topic: it fades the old counts as the new topic's requests arrive, and the swaps and live rebalancing follow. There is no reset and no special start at a boundary.
Consequence: the decay constant and the swap rate limit together set how fast the resident set turns over after a topic change; these are the two values to tune on the target. Both are open (Strata's defaults: x0.7 per round, 96 swaps per 4 windows).
Open from 13.34: how often a budget move happens, how many experts per move, whether it shares the swap rate limit, and the cap value.

### 13.36 Victim choice decided: Strata's rule plus a grace period (2026-10-09)

Decided (user): option B. The victim of a swap is the least-used resident expert of the same layer (Strata's rule, one decayed usage counter), and a newly swapped-in expert cannot be chosen as a victim for a grace period. The idea is borrowed from LFRU's probation segment without a second counter (no heat, no protected segment; the earlier decision to leave out solid.cpp's other count machinery stands). The user's expectation: the grace period also lowers how often swaps happen.
Why it should (assistant): a newcomer that just won a swap cannot be swapped out again right away, so back-and-forth swaps between two close experts stop.
Open: the length of the grace period (in rounds, tokens or swaps), and whether it also applies to experts promoted into RAM. To be tuned on the target.

Corrections to the comparison with solid.cpp's page (2026-10-09), pending the user's decision to record it as its own section:
- Of 21 decisions, 4 conflict (2 repeat documented losses: substituting every miss, an RAM buffer we own; 2 go against listed wins: LFRU eviction, a RAM tier fed from counts). The co-activation item is withdrawn: the page's fix replaced heat-ranked stand-ins with co-activation-ranked ones, and ours is neither.
- The page's +41% (buffer fed by prediction vs by historical heat) compares two variants of a feature that still measured below having no buffer, so it is "less wrong", not a win over nothing. solid.cpp's "historical heat" for the RAM buffer is a per-expert selection counter (+4 per router pick, capped, epoch-aged); our count is the same family (backward-looking), decayed by x0.7 per round.
- Our substitution differs from solid.cpp's: at least 10 residents per layer guaranteed; the stand-in is the router's own best-ranked resident for the token; no exact fallback. Whether that avoids the page's rank-0 degeneracy is untested; keep a substitution floor and an off switch for the RAM buffer as testable options.

### 13.37 Rate limit values confirmed (2026-10-09)

Confirmed (user): "96 per 4 seems good": at most 96 swaps per round, a round every 4 windows, none while the previous round's copies are in flight (13.7 values kept). To be tuned on the target if needed.
Still open: what one "window" is in our engine (one token without speculative decoding), so the limit converts to 96 swaps per 4 tokens; and whether the 96 counts swaps or bytes (layers differ in expert size, 1.51 to 2.66 MB).

### 13.38 Speculative decoding is a requirement (2026-10-09)

Decided (user): speculative tokens (the model's own multi-token-prediction draft block) are a must.
Facts checked: our IQ3_S GGUF has no draft block: block_count is 48, the highest block index is 47, and there are no nextn/mtp/eh_proj/enorm/hnorm tensors or keys. Both reference converters drop it. Strata fetches the draft layer separately (about 6 GB, +1 GB with images; tools/mtp_pack.py builds it from the BF16 tensors); Unsloth also publishes MTP files separately. The draft block is structurally another trunk block (full attention plus its own 512-expert MoE), plus a fusion projection and its own hyper-connection head.
Consequences:
- Units: a "window" is a verify window (the target checks several tokens in one pass). The rate limit "96 swaps per 4 windows" (13.37) therefore keeps Strata's meaning. Strata's windows check about 2.4 to 3.2 tokens per pass; solid.cpp's draft depth 4 won over 8 (a wider window widens the union of experts needed).
- Recurrent state rollback is in scope: when drafts are rejected, the 36 GDN layers' state and conv tails, and the PLE history, must be restorable per accepted-token count (solid.cpp keeps per-slot rollback copies). This replaces the earlier "out of scope for v1" in section 5.
- The draft block needs its own expert residency (a 49th layer with 512 experts, counts and a budget), or to be kept fully resident.
- Counting: a window routes several tokens per layer. Open: do counts include the picks of draft tokens that get rejected (Strata counts every row in the window)?
- The draft and the target both use the stand-in rule; acceptance is measured against the target under stand-ins.
Open: draft depth (Strata drafts up to 3; solid.cpp found 4 best on Gemma and 8 worse), the draft file and its quantisation, and whether the 96 limit counts swaps or bytes.

### 13.39 Counting with speculative windows: assistant's opinion (2026-10-09)

Question: do the usage counts include the picks of draft tokens that are rejected?
Rows of a verify window: inputs x0 (last verified token), d1..dk. If a drafts are accepted, rows 0..a have inputs that became part of the sequence; rows a+1..k have rejected inputs and describe a sequence that never happened.
Opinion (assistant, awaiting the user): count only rows 0..a, applied after acceptance is known (store per-row picks in the window, apply the +1s afterwards). Reasons: the router's request is what tells what to keep; rejected rows are requests for a token the model did not produce; on prose the draft is accepted only about 40% of the time (solid.cpp's page: 0.38 to 0.68 on prose, 0.73 to 0.90 on code), so many rows would be noise; the drafter's habits would bias the counters. Cost: negligible.
Counter-argument: Strata counts every row (held-out hit rate 0.645); a rejected draft is usually a near miss and probably correlated with real demand; not measured either way here.
Make it a switch and compare hit rate and unmet demand on the target. Proposed default: kept rows only.

### 13.40 Counting decided: kept rows only (2026-10-09)

Decided (user): the usage counter counts only the rows of a verify window whose input tokens were kept (rows 0..a after a accepted drafts). The per-row picks are stored during the window and the +1s are applied after acceptance is known. A switch to count every row (Strata's behaviour) stays available for comparison on the target; it is not the default.

### 13.41 Draft depth: evidence and proposal (2026-10-09)

Evidence read:
- solid.cpp (Gemma-4, paired, three rounds; docs/index.html): depth 4 (was 8) +22%; depth 6 +12%; depth 8 loses 11% against the shipped depth 4; adaptive depth lost 4.2%; Qwen's own calibration also chose 4; confidence gate p_min 0.5 worth +13%. Its depth 4 means 4 drafts, a window of 5 rows (it keeps the verify batch inside a 5-wide MoE kernel path). At 64K context with the draft attached, depth 4 did not load; 2 or less fit.
- Strata (RTX 3060, IQ3_XXS, Qwen3.8-Flash-Next): `--spec 4` = a verify window of 4 tokens (1 verified + 3 drafts); about 43 tokens/s; 62 to 68% of drafts accepted; 2.4 to 3.2 tokens per window. Drafts are extended only while the draft layer is sure enough (`--spec-min-p`).
Caveats for us: solid.cpp's reason (a wider window widens the union of experts the cache must serve) matters less when every miss is covered by a stand-in, but the window width still sets the cost of the grouped expert kernels (to measure); prose accepts drafts only 38 to 68% of the time; VRAM can cap the depth.
Proposal (assistant, awaiting the user): start with a window of 4 rows (3 drafts) and a confidence gate of 0.5; test 5 rows (4 drafts) on the target.

### 13.42 Draft depth decided (2026-10-09)

Decided (user): a verify window of 4 rows (1 verified token + up to 3 drafts) and a draft confidence gate of 0.5. No test of 5 rows. The window is extended only while the draft layer is at least 0.5 sure. The rate limit of 13.37 (96 swaps per round, a round every 4 windows) counts in these windows.
Still open around speculation: the draft file and its quantisation, residency of the draft block's 512 experts, and whether the 96 limit counts swaps or bytes.

### 13.43 Draft block: file, quantisation and residency (2026-10-09)

User's lean: draft file Q8 ("I guess"), fully resident.
Strata (include/strata/core/mtp.hpp, tools/mtp_pack.py, tools/mtp_rt.py): packs its own draft file from the 31 `mtp.*` tensors of the original BF16 checkpoint (about 6 GB BF16; the GGUF converters drop the head). Large projections (attention, indexer, shared expert, input fusion) run as Q8_0; hyper-connection and router weights BF16; norms F32 (dense set about 0.18 GB). The 512 routed experts: Q2_0 in the shipped runtime (708 MB; one expert is 1,382,400 bytes, the same as the main Q2_0 build), with Q4_0 (1.42 GB) and Q8_0 (2.67 GB) as comparison arms in the pack tool. All 512 experts live in VRAM and run through the grouped kernels; the layer takes about 0.9 GB. Its probe on the BF16 original accepts 0.89, 0.86, 0.85 of greedy drafts at steps 1 to 3. No recorded acceptance comparison across expert formats was found; its plan picks the format by measured acceptance.
solid.cpp: draft about 1.8 GiB for Qwen, a Q8_0 MTP file for Gemma, draft cache kept f16.
Draft quality affects speed only; the verify window decides every emitted token.
Resident cost: Strata's mix about 0.9 GB; all-Q8 about 2.9 GB.
Decided: none yet. Open: expert quantisation (Q8_0 as the user guessed, or Q2_0 as Strata), and the source of the draft file (our own pack from the BF16 tensors, or a published MTP GGUF).

### 13.44 Draft block decided: Strata's approach (2026-10-09)

Decided (user): the draft block follows Strata's approach. Routed experts in Q2_0 (708 MB for 512 experts); large projections Q8_0; hyper-connection and router weights BF16; norms F32. The whole block is fully resident (about 0.9 GB with its head, per Strata). Draft quality affects speed only; the verify window decides every token.
Still open: the source of the draft file (our own pack from the original BF16 `mtp.*` tensors, as Strata does, or a published MTP GGUF), and whether the 96 limit counts swaps or bytes.

### 13.45 Where the draft file comes from (2026-10-09)

User: "the draft file comes from the same place the rest of the model is."
Checked: the ISTA repo (ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) has no draft file: its root holds the folders IQ2_XS, IQ3_S, IQ3_XXS, Q2_0, assets, tensor-allocation, README.md and the 908 MB vision projector; subfolder contents not listed. Strata's tools/mtp_fetch.py says "The GSQ-RCO GGUF ships no MTP head". The BF16 base checkpoint Qwen/Qwen3.8-Flash-Next (360 GB, 131 shards) has 31 `mtp.*` tensors spread over 28 shards; Strata reads each shard's safetensors header with HTTP range requests, then fetches only those tensors (about 6 GB BF16), pinned to a revision, each checked against SHA256. Its comment (#327): a proxy that ignored the Range header returned the wrong bytes and the drafter then silently accepted nothing, so a range read needs a 206 reply with the right Content-Range and every tensor needs a hash check.
Reading to confirm with the user: the draft file comes from the original base checkpoint (the same source the quantised model was made from), packed by us: experts Q2_0, projections Q8_0, hyper-connection/router BF16, norms F32 (13.44). Needs a packer of our own (Strata's is round-to-nearest with a per-block scale chosen to minimise squared error, not GSQ).
Alternative, not verified: Unsloth publishes MTP GGUFs matched to its own main file; compatibility with ISTA's main file is untested.

### 13.46 Draft file source decided (2026-10-09)

Decided (user): the draft block is built from the original base checkpoint (Qwen/Qwen3.8-Flash-Next, the 31 `mtp.*` BF16 tensors, about 6 GB, fetched by range requests), packed by our own packer to the layout of 13.44, with a hash check on every tensor. Rule from Strata's experience (#327): a range read must return a 206 reply whose Content-Range matches the request, and every tensor is checked against a SHA256 of a pinned revision, otherwise a bad proxy can make the drafter silently accept nothing.
Work items this adds: a range-request fetcher with revision pinning and hash verification, and a Q2_0 / Q8_0 packer. The hashes for our pinned revision must be recorded when the tensors are first fetched (Strata's own hash list belongs to its pinned revision and is a cross-check only if we use the same revision).
Still open: whether the 96 limit counts swaps or bytes.

### 13.47 Rate limit: swaps or bytes (2026-10-09)

Facts (IQ3_S expert sizes 1,510,400 to 2,662,400 bytes, mean 2,046,400): 96 swaps per round copy 145.0 MB (all lightest layers) to 255.6 MB (all heaviest), mean 196.5 MB. A byte budget of 196.5 MB per round allows about 130 swaps in the lightest layers and about 74 in the heaviest. With the exclusive RAM tier a swap sourced from RAM is an exchange (the evicted expert is copied back to RAM), about double the traffic; a swap sourced from NVMe adds a read of the newcomer. Rough scale (assistant, from Strata's reported 43 tokens/s at about 2.9 tokens per window on an RTX 3060, so a round every 4 windows is about 0.27 s): about 0.7 GB/s upstream at the full limit; an estimate, not ours.
Trade-off: swaps protect stability (churn) and vary the copy load by 1.76x; bytes protect the link but make heavy layers (the late, more quantisation-sensitive ones) adapt more slowly per round.
Earlier suggestion (assistant): a byte budget. WITHDRAWN on 2026-10-09, see below.

PCIe facts (2026-10-09). Theoretical per direction: Gen3 about 1 GB/s per lane (x16 about 15.75 GB/s after encoding), Gen4 about 2 (x16 about 32), Gen5 about 4 (x16 about 63); x8 halves. The RTX 3060 is a PCIe 4.0 x16 card (one forum post says some boards wire it x8); the target PC's slot generation and width are unknown and must be measured; nvidia-smi shows Gen1 at idle and the link only trains up under load.
Measured on an RTX 3060 (solid.cpp's notes, docs/moe-cache-colibri-notes.md, PCIe 4.0 x16): RAM to VRAM pinned 24.5 GB/s (pageable 12 to 17); NVMe (WD Blue SN5000) 3.55 GB/s, 3.43 at queue depth 1; RAM 36.6 GB/s; VRAM 254 GB/s effective in the MoE op.
At the full limit of 96 swaps per round the upstream traffic is about 0.7 GB/s (about 1.4 GB/s with the copy-back of the exchange), roughly 3 to 6% of the pinned PCIe figure; a 196 MB round takes about 8 ms over PCIe against a window of about 65 ms. An NVMe-sourced round is the tighter case: about 0.7 GB/s is about 20% of the drive's measured speed, about 55 ms per round.
Conclusion (assistant): PCIe is not the binding limit; the "protects the link" argument for bytes does not hold, and a byte budget would slow the late, quantisation-sensitive layers. Recommendation now: keep Strata's 96 swaps per round (as the user leaned). Watch NVMe-sourced rounds on the target.

### 13.48 Rate limit final: swaps (2026-10-09)

Decided (user): the rate limit counts swaps, not bytes: at most 96 swaps per round, a round every 4 verify windows (a window is 4 rows, 13.42), and no new round while the previous round's copies are in flight. This closes the swaps-or-bytes question. The NVMe-sourced rounds are to be watched on the target.

### 13.49 Grace period length decided (2026-10-09)

Decided (user): the grace period is one round, i.e. 4 verify windows. A newly swapped-in expert cannot be chosen as a victim until the next round; with a round every 4 windows (13.48) this means it is protected for exactly one round. (The earlier record in 13.36 listed the length as open because the "96 per 4" remark was read as the rate limit.)
Still open: whether the grace period also applies to experts promoted into RAM.

### 13.50 Grace period also applies to RAM promotions (2026-10-09)

Decided (user): the grace period (one round, 4 verify windows, 13.49) also applies to experts promoted into the RAM tier. A newly promoted expert cannot be displaced from RAM until the next round. This closes the grace-period question.

### 13.51 The RAM stage runs one round ahead of the VRAM stage (2026-10-09)

Stated by the user and confirmed by the assistant as following from the design: the RAM promotion round runs one cycle (round) earlier than the VRAM swap round. A candidate for VRAM is then already in RAM when the swap round selects it (PCIe copy about 8 ms for a full round, against about 55 ms from NVMe). The RAM stage sees counts one round older, which is harmless because counts decay only x0.7 per round. Staging load: up to about 96 experts per round, about 196 MB of NVMe reads. The grace period fits: a promoted expert is protected for one round, and the next VRAM swap takes it as an exchange, not a displacement.
Consequence: a strict pipeline NVMe -> RAM -> VRAM with no direct NVMe -> VRAM swap. A newly hot expert needs two rounds (about 0.5 s at Strata's rate) to reach VRAM instead of one.
Open: strict pipeline (no direct NVMe to VRAM) or allow a direct NVMe read when a candidate is not yet staged.

### 13.52 Not a strict pipeline: RAM preferred, direct NVMe allowed (2026-10-09)

Decided (user): not the strict pipeline. A VRAM swap takes its candidate from RAM when it is staged and reads directly from NVMe when it is not (solid.cpp has this direct path); RAM is preferred. Supersedes the open question of 13.51.
Facts: on solid.cpp's RTX 3060 box RAM to VRAM pinned measured 24.5 GB/s and the NVMe 3.55 GB/s; one 7 MB expert step took 289 us from RAM and 1,994 us from NVMe. solid.cpp's "direct" path is a pread() of the whole expert into a pinned host staging buffer followed by the host-to-device copy, not a GPU-direct transfer.
Time check for the n-1 staging round: a round of 4 windows is about 270 ms at Strata's rate (about 43 tokens/s, 2.9 tokens per window). Staging 96 experts is 196 MB (256 MB at the largest sizes): 55 to 72 ms at 3.55 GB/s, about a quarter of a round. On a SATA drive (0.5 GB/s) it would take 390 to 510 ms, longer than a round, so rounds would be skipped. The target's NVMe model and PCIe generation are unknown.
Read shape: an expert's gate, up and down slices sit in three separate tensors, so 96 experts is about 288 reads of roughly 0.4 to 1.0 MB each; throughput at that size is unmeasured. Option (not decided): a one-time repack with each expert's three slices contiguous (a sidecar pack), turning it into 96 reads.
Open: the target's NVMe drive model and PCIe generation (to be measured).

### 13.53 Design values for the target, taken from solid.cpp's measurements (2026-10-09)

Decided (user): no measurement of the target's drive or link; take solid.cpp's measured values (its box is also an RTX 3060 12 GB). Design values: VRAM 254 GB/s effective in the MoE op; RAM to VRAM pinned over PCIe 4.0 x16 24.5 GB/s (pageable 12 to 17); RAM 36.6 GB/s; NVMe 3.55 GB/s (3.43 at queue depth 1). These are the user's chosen design values, not measurements on the target.
Result of 13.52 under these values: staging 96 experts (196 to 256 MB) takes 55 to 72 ms against a round of about 270 ms, so the n-1 round has about 4x margin.
Risk noted once: if the target's drive or slot is slower (a SATA drive at 0.5 GB/s needs 390 to 510 ms for the same staging; a Gen3 or x8 slot halves the PCIe figure), the staging check is the first assumption to fail. This closes the open item of 13.52.

### 13.54 Budget moves: how often and how many (proposal, 2026-10-09)

No precedent: Strata fixes per-layer budgets and solid.cpp pools them, so these numbers are our own design, untested.
Mechanism (assistant): a budget move takes bytes from a donor layer (lowest weighted unmet share) and gives them to a recipient layer (highest), respecting the floor of 10 experts, the cap and the grace period. It runs through the normal swap machinery (the donor evicts its lowest-count residents, the recipient admits its best candidates) and so spends some of that round's 96 swap slots. Experts differ in size between layers (1.51 to 2.66 MB), so a small free-byte reserve absorbs the difference.
Options (round about 270 ms; time to shift 10% of the fast tier, illustrated with 4,000 resident experts, the real tier size is undecided): slow = a step every 32 rounds (8.6 s), up to 4 swap slots, about 14 min; balanced = every 16 rounds (4.3 s), up to 8 slots, about 3.6 min; fast = every 8 rounds (2.2 s), up to 16 slots, about 54 s. A step uses at most 8 of that round's 96 slots (about 8%) and none in the other rounds.
Proposal (assistant, awaiting the user): balanced. Budgets follow the topic (minutes), swaps follow tokens.

Request length check for 13.54 (2026-10-09; Strata's reported 43 tokens/s on an RTX 3060 with IQ3_XXS is the only comparable speed, our engine's speed is unknown; prompt reading over 1,000 tokens/s on long texts): 200 tokens about 5 s; 500 tokens about 12 s; 2,000 tokens (reasoning with a cap) about 47 s; 8,000 tokens about 3 min. Balanced option (a step every 4.3 s, up to 8 experts, 0.2% of an illustrative 4,000-expert tier per step) moves about 0.2%, 0.6%, 2.2% and 8.7% of the tier within those requests; fast would be about 4x that (under 1% for a short request, about 35% for a long one). Budgets therefore change over several requests, which fits carry-over (13.35).

### 13.55 Link rates measured on the target (2026-10-10; the user allowed measuring instead of borrowing 13.53)

Machine quiet (no service, no download). Programs: `nvme_bench.c`, `pcie_bench.cu` (kept in the job's `linkbench` folder).
Each figure is one run; the NVMe reads are O_DIRECT unless marked mmap.

| Link | Measured | Note |
|---|---|---|
| NVMe, the drive's link | PCIe 3.0 x4 (8 GT/s), the drive supports 4.0 x4 | Behind chipset root port `00:1c.4` (Gigabyte H610M K DDR4); the port's own maximum is 8 GT/s. Not a software setting. Stays at 8 GT/s under load. `read_ahead_kb` of the device is 4096 |
| NVMe, sequential 1 MiB, one reader | 2.87 GB/s | |
| NVMe, random, 4 or more reads in flight | **3.57 GB/s** (the ceiling) | 2 MiB: 3.53 (4 in flight), 3.57 (16, 32); 896 KiB: 3.44, 3.56, 3.57; 512 KiB: 3.25, 3.56; 128 KiB: 2.20, 2.69 (20,500 reads/s) |
| NVMe, random, ONE read in flight | 2.78 GB/s at 2 MiB, 1.90 at 896 KiB, 1.38 at 512 KiB, 0.81 at 128 KiB | Latency 755 / 482 / 380 / 163 microseconds a read |
| NVMe via mmap, one 896 KiB slice per read, kernel default read-around | 0.69 GB/s useful (1 reader), 1.07 GB/s useful (8 readers) | The device read 2.57 and 3.55 GB/s to deliver that: **3.7x and 3.3x read amplification**; the same drive gives 3.44 GB/s useful with 4 explicit reads in flight |
| NVMe via mmap with MADV_RANDOM | 0.15 GB/s (1), 0.25 GB/s (8) | No amplification (0.99x) but each 4 KiB page is its own synchronous read: not a fix by itself |
| RAM copy (memcpy) | 17.0 GB/s (1 thread), 16.3 (6), 15.5 (12) | Bytes copied per second; the memory traffic is twice that |
| RAM to VRAM, pinned, 0.5 to 256 MiB | **23.4 to 24.6 GB/s** (21.0 at 512 MiB) | The GPU link trains to PCIe 4.0 x16 under load (gen 1 at idle). 2 MiB: 24.2 GB/s |
| RAM to VRAM, pageable | 17.3 to 17.6 GB/s up to 8 MiB, 11.5 to 12.1 above | Pinned is needed for the engine's swaps |
| VRAM to RAM, pinned | 25.1 to 26.3 GB/s | |
| One swap round, 96 x 2 MiB pinned, one stream | **8.3 ms** (24.2 GB/s) | A single synchronous 2 MiB copy: 89 microseconds |
| Chain NVMe (O_DIRECT, random) to pinned RAM to VRAM, 896 KiB | 1.71 GB/s (1 in flight), 3.42 (4), 3.56 (8); 3.51 at 2 MiB, 4 in flight | NVMe-limited: the PCIe step adds nothing once 4 reads are in flight |

What it says:
- The design values of 13.53 are confirmed on this machine (NVMe 3.55 against 3.57; pinned PCIe 24.5 against 24.6).
- The NVMe is the narrow link by a factor of about 7, and the engine must keep at least 4 reads in flight to use it.
- Staging 96 experts (196 to 256 MB) takes 55 to 72 ms at 3.57 GB/s with 4 or more reads in flight, but 100 to 130 ms or more with one in flight (13.52's check holds only with the queue filled).
- **The bound is on the stock engine's bytes per token, not on the design (user, 2026-10-10).** At about 150 MB read per generated token (the stock IQ3_S baseline) a fully used NVMe delivers at most 3.57 GB/s / 0.150 GB = about 24 tokens/s, and the baseline used 1.87 GB/s (52% of the ceiling). That limit holds only while bytes per token stay where they are; the port exists to lower them, in three ways. (1) Less read amplification: the mmap measurements show 3.3 to 3.7 times the bytes used. (2) A higher hit rate from the swap rounds and per-layer budgets. (3) Every miss covered by a stand-in (13.11), so a miss no longer reads from the NVMe at all and the NVMe carries only the swap traffic, which the rate limit caps at 96 swaps per 4 windows: at most 96 x 2.05 MB = 196.5 MB per round, about 11.6 tokens per round at 2.9 tokens per window, so at most about 17 MB per token (an upper bound, from the Strata figures in 13.47 and not measured here), where the NVMe would allow about 210 tokens/s. The touched experts per token are 48 layers x 10 x 2.05 MB = about 982 MB when nothing is resident, so the baseline's 150 MB already means most bytes are served from VRAM or the page cache. Whether the 150 MB is mostly amplified misses is a hypothesis to test with the step-0 counters (bytes read against bytes used). The 150 MB figure was counted over the whole run including warm-up page-ins, so it is an estimate. Once the NVMe stops binding, the limits move to the CPU computing the experts that are not on the GPU, to the GPU, and to the quality cost of stand-ins.
- Hardware: a board with a CPU-attached PCIe 4.0 x4 M.2 slot would double the drive's ceiling on paper; no free CPU-attached port exists on this board (only `00:01.0`, the GPU's, is CPU-attached). Premature before step 1 (the read path) shows how much of the present link is used.

## 14. Port order into the Strata fork (proposal, 2026-10-10; user said "go ahead" the same day; order not yet fixed)

Base: the Strata engine on branch `solid-wins`. Each step is one concept, built and judged before the next starts.
Every step is judged by the same three checks as solid.cpp's hot-rod port: (1) the operator or unit tests it touches,
(2) greedy-output agreement with the previous build, read against a previous-build-against-itself control (greedy
output is not bit-reproducible here, so a bare "identical" count means nothing), (3) paired speed with its own
identical-config control. The machine is quiet for every run (no other service, no download, no build). A change counts as
a win only if it clears the control's spread; solid.cpp's own runs on this card varied by 10 to 20% between identical
runs. Decode is timed with `qwen_tps.py` (chat endpoint, code / reasoning / prose, thinking off); the NVMe read volume
per generated token is recorded with every run, because the baseline says the disk sets the speed.

| # | Step | Where it comes from | Why now | Test beyond the three checks |
|---|---|---|---|---|
| 0 | Counters only: bytes moved per link per generated token (NVMe to RAM, RAM to VRAM, NVMe to VRAM) and link utilisation, per-layer hit / miss, swaps per round, nothing changes | Strata's `/metrics`, solid.cpp's PART_STATS | Without it no later step can be explained | Speed unchanged within the control |
| 1 | Read path for experts that do not fit RAM: no page-locking of mmap'd experts, no kernel read-around for one expert slice (random-access advice or direct reads) | solid.cpp page: two bugs of this kind cost Qwen roughly 4x (2.81 to 12.85 tok/s) | Cheapest step and the baseline reads about 150 MB per token | Bytes read per token before / after |
| 1b | What is kept where (mostly VRAM): placement is measured per machine, taken from solid.cpp's calibration instead of Strata's formulas. The requested context is fixed first and placement gives way to it (solid.cpp's inversion: 4,096 to 65,536 tokens, 4 slots, once the order was fixed); candidates are pinned to the requested context and one that cannot hold it fails (its early runs measured at 4k what they served at 64k and picked a cache that ran out of memory); the search is ordered by what each knob does: knobs that change the fixed cost first (context, KV precision, micro-batch, placement, draft width), then knobs that divide what is left (expert-cache size, the draft's share), then knobs that spend no bytes (admission, stand-ins, eviction); cheap probes only shortlist, anything within 15% of the best is re-measured at full answer length in the configuration actually served; a degeneracy check (repeated 4-grams, single-word dominance) vetoes a candidate, a fidelity score does not; a reserve floor of VRAM is kept for what is allocated after the cache sizes itself (graphs, prefill buffers; 1,500 MiB on solid.cpp's card). Built on Strata's own `tools/calibrate.py`; results cached per model, GPU and context, and carrying the version of the checks | solid.cpp page: "Nothing to tune", "The calibration was optimising a machine that does not exist", "A correction", the ordering table | The user's instruction of 2026-10-10: positioning is taken from solid and tested | Re-run on the target: the chosen placement against Strata's formula, paired, with an identical-config control; bytes per link per token |
| 2 | Stand-in for every miss as a mask on the router before the top-k (13.20), with an off switch; counts still follow the unmasked picks | 13.11, 13.13, 13.20 | The largest single lever solid.cpp measured on a model that does not fit (12.85 against 8.3 tok/s with it off) | Degeneracy check (repeated 4-grams, single-word dominance), long-answer check at reasoning length, hit rate, speed |
| 3 | Counts and swaps by the spec: one decayed counter per layer-expert (x0.7 per round), unmasked picks, kept rows only (13.40), margin 1.5, grace of one round (13.49), 96 swaps per 4 windows, none while copies are in flight | 13.2c, 13.36, 13.37, 13.40, 13.48 | Gives step 2 its feedback | Hit rate and unmet demand per layer over a long run |
| 4 | Per-layer budgets: phase 1 by bits per weight with floor 10 and cap (13.30); phase 2 live rebalancing, "balanced" option (13.34, 13.54) | 13.21 to 13.35 | Builds on the counters of step 3 | Per-layer unmet share spread, speed |
| 5 | RAM tier as an owned, exclusive buffer fed by counts, background promotion one round ahead, grace on promotions (13.14 to 13.16, 13.50 to 13.52) | the user's decision, against solid.cpp's two negative results for a host-resident hot buffer (-56% dedicated buffer, -12% mlock pinning) | Last, because the evidence says it can lose; ships with an off switch | Compare with the page cache alone, same run |
| 6 | Optional: a retention score from a pre-router (solid.cpp: it won at eviction protection, lost at admission) | solid.cpp page | Only if step 3 shows evictions of soon-wanted experts | Hit rate, speed |

Not ported, by decision or by solid.cpp's own measurement: the atlas (13.12), heat with protected / probation segments
(13.2c), cross-run history, substitution by rank floor, neuron subsetting (a loss on gemma-4 and Qwen), router bias
(not a default there; 13.22 declines it), lazy KV commit (opt-in there).
Already in Strata: draft window of 4 rows with a 0.5 confidence gate, int8 KV, router lookahead, shipped ranked profile.
Open for this port: how the router is masked inside Strata's graph (read src/core before step 2), and which quality
check stands in for a KL divergence against an exact run, which is not possible here (13.19).

## 15. Test ledger (2026-10-10): every decision, its hypothesis, its test, and where it stands

Common rules for every row. The machine is quiet. Arms are interleaved with an identical-config control, and the control's
spread is the bar a change must clear (solid.cpp measured 10 to 20% between identical runs on this card). The model's
file cache is emptied before each start. Quality is judged by the degeneracy check (repeated 4-grams, single-word
dominance) and a long-answer check at reasoning length; agreement with an earlier build is read against a build-against-itself
control. Each run records tokens/s per prompt kind, bytes per token per link (13.55), and the hit rate per layer.
Status values: UNTESTED, RUNNING, PASSED (the number), FAILED (the number, decision reopened).

| # | Decision (where it comes from) | Hypothesis | Evidence so far | Test | Passes when | Status |
|---|---|---|---|---|---|---|
| T1 | Staged explicit reads for experts that miss the page cache, not page faults (Strata docs; my link measurements) | Bytes read from the drive per token fall and tokens/s rise | Link test: mmap read-around 3.3 to 3.7x amplification; Strata docs: +25.7% on a 3060 with 32 GB (IQ3_XXS) | A1 B1 A2 B2 with `STRATA_IO_PREFETCH=1 STRATA_IO_PF_STAGE=1` against the default | B beats both A runs by more than their spread | **PASSED (2026-10-10).** Median of the 9 timed runs per arm: A1 17.0, A2 17.1 (control, 0.1 apart), B1 27.2, B2 27.5 tok/s = +60%. Per kind (median of 3, B1 / B2 against A1 / A2): code 16.4 / 22.1 against 14.7 / 16.0; reasoning 27.2 / 31.6 against 17.3 / 17.2; prose 30.0 / 31.6 against 26.9 / 29.1. Drive reads per generated token 79.7 / 81.0 MB against 151.6 / 149.6 MB (-47%); major faults 18,312 / 20,099 against 292,769 / 283,229. Single runs within an arm vary widely (code B1 14.8 to 27.9); code is still under 25 in both B medians. Not checked here: that the answers are token-identical to the default (Strata's docs say the greedy hashes match). Only two runs per arm |
| T2 | Ranked-profile start (Strata's shipped profile, 13 row 1) | The shipped profile's hit rate holds on our prompts and on IQ3_S | Strata's 0.645 held-out hit rate was on its own workload and variant | Hit rate per layer on our prompt kinds, empty start against shipped profile | Profile beats an empty start in the first requests | UNTESTED |
| T3 | Adaptive rounds of 96 swaps per 4 windows (user, 13.48; Strata default) | 96 per 4 is better than none, than fewer, than more | Strata docs: a slow-CPU box gained +7.9% at 160 per window | `--adapt-swaps` 0, 48, 96, 192 with `--adapt-every` 4 | A setting clears the control; the bytes the swaps move per token stay under the link budget | UNTESTED |
| T4 | A stand-in for every miss, as a mask on the router (13.20) | Speed rises with no loss of answer quality, and NVMe reads per token fall to the swap traffic | solid.cpp: 12.85 vs 8.3 vs 3.7 tok/s (rank 2 / off / cache off) on one run; but paired tests on gemma-4 and Qwen found no speed gain and more drift; rank 0 was degenerate | Mask on / off, with the degeneracy and long-answer checks, then the floor of residents per layer (10 and higher) | Faster than off by more than the spread, and no degeneracy at reasoning length | UNTESTED |
| T5 | Counts follow the router's unmasked picks, kept rows only (13.40) | Better hit rate and less churn than counting every row | Reasoned only; Strata counts every row | Switch: kept rows against every row | Hit rate or unmet demand better by more than the spread | UNTESTED |
| T6 | Swap margin 1.5 and a grace period of one round (13.36, 13.49) | Fewer back-and-forth swaps with no loss of hit rate | Reasoned only | Swaps per round and the share of swapped-in experts evicted within two rounds, grace on / off | The churn falls and the hit rate does not | UNTESTED |
| T7 | Per-layer expert counts from bits per weight, floor 10, cap (13.30) | Better than equal counts and than the profile's frequency fill | Weak: bit-width has no clear link with routing frequency (13.6) | Three fills, same total bytes | The spread of unmet share or the speed beats the others | UNTESTED |
| T8 | Live rebalancing of budgets, "balanced" option (13.34, 13.54) | Follows a topic change | None | A request sequence with a topic switch | Hit rate recovers faster than with fixed budgets | UNTESTED |
| T9 | RAM tier as an owned, exclusive buffer fed by counts, promotion one round ahead (13.14 to 13.16, 13.51) | Better than the OS page cache alone | solid.cpp measured two host-resident hot buffers as losses (-56%, -12%); Strata's staged mode is +25.7% at 32 GB and -32.6% at 16 GB | With and without the buffer, same run | Beats the page cache alone | UNTESTED |
| T10 | Direct NVMe to VRAM allowed besides the RAM route (13.52) | Few swaps need it, and it does not stall a round | Link test: chain reaches 3.5 GB/s with 4 reads in flight | Count the share of swaps sourced from NVMe and the rounds skipped | Rounds are not skipped | UNTESTED |
| T11 | Draft window of 4 rows, gate 0.5 (13.42) | Best on IQ3_S here | solid.cpp: depth 4 +22%, gate +13% on gemma-4; Strata's default is the same | `--spec` 3, 4, 5 and `--spec-min-p` 0.5 against 0.7 | The default is not beaten by more than the spread | UNTESTED |
| T12 | Placement measured, not a formula (step 1b) | The search finds a better split of VRAM than Strata's formula | solid.cpp: placement and margin results were opposite on two models | Strata's formula against the search, paired | The search wins at the context actually served | UNTESTED |
| T13 | KV precision int8 (Strata default) | Right at 32K on 12 GB | solid.cpp: f16 was 3.5 to 4% faster when VRAM was comfortable, q8_0 was needed for the draft to load | int8 against f16 at 32K, with the draft loaded | Winner by more than the spread | UNTESTED |
| T14 | The link budget premise: with stand-ins, NVMe traffic is only swaps, at most about 17 MB per token (13.55) | The counters show it | None on this box | Bytes read from the drive per token once T4 and T3 are on | At or under 17 MB per token in steady state | UNTESTED |
| T16 | Tune the staged reads: more I/O threads, deeper prefetch (follow-up to T1) | More reads in flight or a deeper look-ahead lifts the code prompts | Link test: 4 or more reads in flight saturate the drive | Arms B3 (reference), C16 (`STRATA_IO_PF_THREADS=16`), D3 (`STRATA_IO_PREFETCH_DEPTH=3`), B4 (reference), medians of 9 runs | A tuned arm beats both reference arms | **FAILED (2026-10-10).** B3 26.1, B4 29.1 (with T1's B1 27.2 and B2 27.5 the reference spread is about 3 tok/s); C16 25.3 (inside the spread, 39,114 major faults against about 18,000); D3 21.4 (worse, 92.2 MB per token against 80). Defaults kept: 8 threads, depth 2 |
| T17 | The RAM budget chosen dynamically from the memory available (the user's direction, 2026-10-10; Strata's `--resident-experts`: budget = MemAvailable minus 4 GiB headroom, owned pinned copy, the rest read unbuffered; solid.cpp derives its budgets from `MemAvailable` and the cgroup limit the same way) | Bounded by construction, so no thrash, and no slower than the page cache | solid.cpp: the kernel's recency-only page cache collapsed generation 21x under a memory shortage; its pinning was measured worse | One watched arm (R1), `--resident-experts` added to the IQ3_S config, 29 GB backstop cap, one request per prompt kind, against the capped staged arm (25 GB) | At least as fast as 11.4 tok/s with at least the 1.9 GB of free memory the staged arm left | **FAILED (2026-10-10).** 9.8 tok/s (code 9.8, reasoning 9.6, prose 12.7), 384 MB read per token (staged: 295); the scope took 28.7 GB (24.8 GB owned buffer), the least free memory was 225 MB and the least available 2.6 GB, swap rose to 786 MB and 760 MB was still in use after the run. The 4 GiB headroom is spent by the rest of the system and the staging; the buffer is filled by position, not by use. The budget must follow the counts and be re-checked while running (step 5, T9), not only at start. A single run per arm |
| T15 | English draft vocabulary (`--draft-vocab en`) | Faster on English and code | Strata docs: 1 to 2% | Measured once: 16.6 against 17.2 (no gain, within the spread) | | FAILED (one run each; not repeated) |

The order of the tests follows the port order of section 14; T1 is first because it needs no code.

### 15.1 Notes from reading Strata's code for step 2 (the stand-in mask), 2026-10-10

Read only; nothing is implemented yet.
- The router is two kernels in `moe_route` (`src/core/layer.cpp`): a BF16 projection writes `b.logits` (512 floats), then a fused top-10 (`native_router_top10`, or `router_top10` for other geometries) does softmax over all experts, a stable descending argsort (ties by index), the gather, and the renormalisation to `b.ids` / `b.weights`. The ids and weights are then published to the host by the doorbell, and the host pool computes the misses.
- Because the weights are renormalised over the chosen ten, the softmax denominator cancels. Setting the logits of non-resident experts to minus infinity before the top-10 therefore gives exactly "the ten best resident experts, renormalised" (the rule of 13.20), with no change to the top-10 kernel.
- The residency table is already on the GPU: `thits.d_res` for the token graph and `vh.d_res` for verify windows, indexed `layer * n_expert + expert` (slot, or not-resident), refreshed by `res_put` after every swap round. The hit path is already "decided on the device" inside the captured graph, so a small mask kernel between the projection and the top-10 reads it at run time and stays valid across graph replays.
- The counts must follow the router's unmasked picks (13.13): run a second top-10 on the unmasked logits and count on the device (an atomic increment per (layer, expert)), copied to the host usage array once per round, instead of publishing a second id list per layer. The existing host-side decay (x0.7) and swap selection (`adapt` in `generate.cpp`) stay.
- With every miss covered there is no expert for the CPU pool to compute; the CPU path and the PCIe share of 13.2d disappear for routed experts, and the per-layer floor of 10 residents (13.21) must hold at all times, including during a swap round (swaps are exchanges).
- Risks to test (T4, T14): quality at reasoning length, the drafter's acceptance under stand-ins (the draft layer has its own 512 experts, resident), and the windows of 4 rows (verify windows route several tokens through the same mask).

### 15.2 The memory incident of 10 Oct 2026 and the capped testbed

What happened: two runs of the interleaved A/B (the fork's new default, then the old page-fault path) froze this 31 GB machine and it was hard-reset twice (about 06:09 and 06:20). Evidence: the last system-log lines before the first reset are the power key ("Powering off") after the terminal host had already failed; XFS journal recovery on every filesystem at boot; no kernel OOM-kill lines and no hardware, storage or machine-check errors in `/var/log/messages`; the failed run's own log showed threads stuck on the drive, a 14.5 s GPU wait, and about 3.7 GB swapped out and 1.8 GB swapped in within one arm. So: a thrash and hang, not an orderly OOM kill. The cause is inferred, not observed.

Trace of a capped run (`tools/linkbench/memtrace.py`, one row a second, the server in a `systemd-run` scope with a hard cap and no swap):
- The engine's own anonymous memory peaks at 1.9 to 2.6 GB. Nothing is locked or pinned. The scope's size is file cache of the memory-mapped model (20 to 22 GB of the 22.5 to 24.6 GB). The engine's resident set (13 to 19 GB) is mostly mapped model pages.
- So an uncapped run lets the file cache grow into all of RAM (28 GB in the earlier runs) and squeezes everything else, with swap traffic on the SATA drive (swappiness is 10, swap is 16 GB on the slow SATA SSD). The cap confines that: swap stayed at 0, nothing was swapped in or out anywhere, and the rest of the machine kept 1.9 to 3.9 GB truly free.
- The price is speed. The model's working set wants about 28 GB of cache; at a cap of 24 GB the drive is read for about 314 MB per token, at 25 GB about 295 MB (uncapped, 80 MB).

Capped testbed, measured (one request per prompt kind, so only direction is reliable):

| Arm (cap 25 GB, no swap) | tok/s code / reasoning / prose | median | drive MB per token | major faults | least free memory |
|---|---|---:|---:|---:|---:|
| Staged reads (the fork's default) | 10.7 / 11.4 / 13.7 | 11.4 | 295 | 43,896 | 1.9 GB |
| Page-fault path (`STRATA_IO_PREFETCH=0`) | 8.6 / 8.4 / 11.6 | 8.6 | 454 | 220,942 | 2.2 GB |
| Staged reads, cap 24 GB | 10.5 / 10.2 / 12.8 | 10.5 | 314 | 48,293 | 3.9 GB |

Result: under the same safe cap staging is about 33% faster and reads 35% less from the drive than the page-fault path, in all three prompt kinds. This reproduces T1's direction (T1 itself: 17.0 / 17.1 to 27.2 / 27.5 uncapped) at a lower speed. From here every measurement is taken inside the cap (`AB_CAP_GB`, default 24; 25 for the arms above) and compared with these numbers, not with the uncapped ones: a change has to show up where the file cache is short, which is what the stand-in and tiering work targets.

## 16. Defaults of the fork, for reference (2026-10-10)

What a plain start does today, and which test chose it (section 15).

| Setting | Default | Switch | Why / ledger row |
|---|---|---|---|
| Staged expert reads (explicit pread into staging buffers instead of page faults) | **ON** (stock Strata: off) | `STRATA_IO_PREFETCH=0`, `STRATA_IO_PF_STAGE=0` | T1 passed: +60% uncapped (27.2 against 17 tok/s); capped at 25 GB +33% |
| I/O threads for staged reads | 8 | `STRATA_IO_PF_THREADS` | T16 failed: 16 no better |
| Prefetch depth | 2 | `STRATA_IO_PREFETCH_DEPTH` | T16 failed: 3 worse |
| Stand-in mask (router restricted to VRAM-resident experts) | **OFF** | `STRATA_STANDIN=1` | built (09d4bed), untested (T4, T14) |
| Swap round | every 4 verify windows, 96 swaps | `--adapt-every`, `--adapt-swaps` | Strata's own refresh unit |
| `--resident-experts` (RAM budget = MemAvailable minus 4 GiB, `STRATA_RESIDENT_HEADROOM_GIB`) | not used | `--resident-experts` | T17 failed: 9.8 tok/s, 225 MB free at worst, swap grew to 786 MB |
| Model config `strata-iq3_s.json` | `--mmap-experts --spec 4 --spec-min-p 0.5 --mtp /mnt/nvme/strata-data/mtp/rt --kv int8 --max-context 32768 --expert-cache auto --prefill auto`, draft vocabulary `en` | the json | T15 (en vocabulary) failed on a single run |
| Test containment | systemd scope, `MemoryMax` 24 GB (`AB_CAP_GB`), `MemoryHigh` 1 GB under, no swap; guard stops an arm at under 1.5 GB available or 1.5 GB swap growth | `tools/linkbench/strata_ab.py` | 15.2: two hard resets from uncapped runs |

Known state (corrected the same day, T18): the staged default reaches about 27.7 tok/s inside a hard 29 GB `MemoryMax` with no `MemoryHigh`. The 11.4 tok/s of the 24-25 GB scope was mostly the testbed's `MemoryHigh` throttle (230,000 throttle events), not the memory limit. The capped numbers of 15.2 and T17 are therefore not comparable with the uncapped ones. Test scopes use `AB_NO_HIGH=1` from now on. Open: swap still takes about 1 GB of idle processes (swappiness 10) while the model's file cache grows; the memory governor, if wanted, is about that, not about speed.

| T19 | Stand-in mask (step 2; rows T4 and T14): the router picks only experts resident in VRAM (`STRATA_STANDIN=1`) | Fewer bytes read per token, so faster, with no loss of output quality | solid.cpp's substitution: no speed gain on Qwen in its own paired tests | Arms M0 off, M1 on, M0b off, M1b on; `MemoryMax` 29 GB only; 3 requests per kind; answers saved and scored for repeated 4-word runs | On clearly faster than the spread of the two off arms, with answers not degenerate | **VOID (2026-10-10): the mask was in the wrong place.** The engine's own decode hit rate with the mask on was still 44 to 60% (62,078 hits in 141,120 lookups), which a working mask makes 100%. The decode window (2 to 8 tokens, 512 experts, k=10) routes through `native_router_top10_multi` in `verify.cpp`, not through `moe_route_window`, where the mask had been put; only single-token steps were masked. Fixed in `verify.cpp`. **Rerun C1 (1 request per kind, hard 29 GB cap): the mask now works, and the answers are broken.** Decode hit rate 100.0% (132,960 of 132,960 lookups), 88.2 / 86.2 / 101.5 tok/s, 85 MB read per token, but reasoning repeats a wrong step (54% repeated 4-grams: "Head-start advantage ... 90 km" over and over), prose collapses into "a word of a word of ..." (96%), code answers with a description instead of code. FAILED as it stands: the speed is real and is worth nothing. Likely cause, to be tested next: the resident set is a static profile (2,721 of 20,480 experts, about 13%) and the swap round counts only the experts the router picks, which the mask has already confined to the resident ones, so the set can never follow demand (step 3, counting what the unmasked router wanted, is the missing half). The mask must stay off by default until counts drive the swaps and the degeneracy check passes. Strata already has an opt-in `STRATA_ROUTE_RESIDENT` (margin-based preference for resident experts, `native_route_resident`), to be compared in the rerun. The numbers of the void run, for the record: Off: 26.8 and 20.2 tok/s (the control moved by 6.6 on its own, the noise floor). On: 19.8 and 19.0. Drive MB per token: off 91 and 121, on 128 and 135, so it did not read less. Answers: no repeated 4-word runs (0.00 to 0.01) in any arm; text differs from the control, so the mask is active. Stays off by default. The first arm ran fastest in every earlier test too, so arm order may matter; a longer interleave (6 or more arms) would be needed to separate a small effect from it |
| T20 | Strata's own residency-biased routing (`STRATA_ROUTE_RESIDENT=<margin>`, ranks 6-9 by default: a non-resident weak pick becomes the best resident expert when it scores within the margin) | Fewer misses, so faster, with sound answers | Strata's own experiment, never measured here | Arms B0 (off), RR1, RR2, RR4 (margins 1, 2, 4 logits), 2 requests per kind, `MemoryMax` 29 GB only; answers saved and scored | An arm clearly above the control's warm requests with no degenerate answers | **PROMISING (2026-10-10), one run.** Warm (second) requests, tok/s: B0 27.6 / 14.4 / 29.5 (code / reasoning / prose); RR1 36.0 / 36.2 / 36.7; RR2 36.3 / 38.4 / 35.7; RR4 36.7 / 35.5 / 36.4. Drive MB per token 125 (B0), 106 (RR1, RR2), 104 (RR4). No repeated 4-word runs (0.00 to 0.02), answers open as the control's do, so no collapse; the reasoning answers are cut at 256 tokens before their result, so final answers are not compared yet. Margins 1 to 4 are alike, so the gain does not come from the margin size. No OOM or kill events in any arm (least available 26.0 to 26.8 GB). To do: more requests per arm, longer answers, a correctness check against the control, then the counting step |
| T21 | The counting step (SPEC 13.20 step 3, integrated into Strata's `route_resident`: the router's own top-10 counted on the device before any swap, drained into the usage table before each 96-in-4 swap round; `STRATA_ROUTE_RESIDENT_COUNT=1`, preset by `STRATA_STANDIN=1`) | With counts of what the router wanted, the resident set follows demand, so the full stand-in (every rank, unlimited margin) keeps sound answers | T19: the mask alone collapsed the answers | Arms K0 (off), K1 (margin 2), K2 (margin 2 + counting), K3 (`STRATA_STANDIN=1`), 3 requests per kind; then T1: K3 with `STRATA_TRACE_ADAPT=1` | K3 answers as sound as K0's | **FAILED for the full stand-in; counting itself no gain (2026-10-10).** Medians: K0 24.8, K1 30.0, K2 26.1, K3 33.3 tok/s; drive MB per token 100 / 95 / 114 / 74. K3: hit rate 100%, but code answers describe the task instead of writing code, prose answers of 2 to 17 words or the leaked text `<|im_start|>assistant`; its high speeds (44 to 56) come from those short answers. The trace (T1): the 96-in-4 rounds run and are always full: 96 swaps in each of 26 rounds, 2,496 in all (the resident set is 2,721), so the router wants more than 96 per round can bring in, and swapping still reads from the drive (T1 159 MB per token) while its answers stay broken (code described, prose empty). K2: counting with Strata's margin gave no gain over K1 (26.1 against 30.0, drive reads 114 against 95). Reading: replacing the router's top picks is the damage, not the age of the resident set; Strata's own variant (ranks 6-9 only, margin) keeps quality and gains about 20% (K1) to 30% (T20). Kept: counting code (off by default, `STRATA_ROUTE_RESIDENT_COUNT=1`), the serve swap trace (`STRATA_TRACE_ADAPT=1`). Next: tune ranks and margin of the gentle variant on more requests, the kernel's cost, a correctness check on longer answers |
| T22 | Where does the stand-in start to degrade the answers, and why are the experts not swapped in time? (user, 2026-10-10) | Quality breaks when the router's strongest picks are replaced, not the weak ones; the swap budget (96 per 4 windows) is far below what the router wants | The sweep over which ranks may be replaced (unlimited margin) | Arms S0 (off) and ranks 8-9, 6-9, 4-9, 2-9, 0-9 (S1..S5); `STRATA_TRACE_ADAPT=1`; 2 requests per kind; hard 29 GB cap; counting off; answers saved; the engine's own counters | Quality holds up to the rank where the answers first leave the control's own spread | **MEASURED (2026-10-10), one run.** Control against itself first differs at word 72 / 113 / 21 (code / reasoning / prose). Warm tok/s, median of the arm: S0 22.9, S1 28.3, S2 25.8, S3 37.1, S4 42.2, S5 64.5 (best single requests: S5 80.0 code, 61.7 reasoning, 64.4 prose). Ranks 8-9 and 6-9: answers inside the control's spread, working code. Ranks 4-9: correct code and steps, first differing word 17 / 13 / 2, 37 tok/s. Ranks 2-9: code still works but drops the type hints and fence, one answer opens "Here is the step-by solution", 42 tok/s; first sign of damage. Ranks 0-9: collapse from the first word (0 / 0 / 1), 6 of 6 prose and code answers under 40 words (a description of the task, not code); the speed is not usable. Router's own picks against the resident set (2,721 experts, 13% of the 20,480): the top-1 pick is not resident in 31 to 33% of decode rows, a non-resident pick sits in the top 5 in 74 to 78% of rows (87% in S4), and the router weight on non-resident picks is 42 to 50%; by rank the non-resident share rises from 31-33% (rank 0) to 58-63% (rank 9). Swap supply against demand per round (candidates = non-resident experts wanted at least twice, from host counts that already exclude the replaced picks, so a floor): S1 1,151 candidates for 96 swaps (8%), S2 513 (19%), S3 318 (30%), S4 about 57 to 140; S5 0 (nothing counted, the set froze). Reading: the damage starts when ranks 0-1 are replaced, and the router puts about 42% of its weight on experts not in VRAM, so a stand-in for every miss replaces nearly half the computation; swaps cover only 8 to 30% of the wanted experts per round |
| T23 | Does something in Strata's current build slow the full stand-in compared with the earlier 85-101 tok/s run? (user, 2026-10-10) | The integrated router kernel is not the slowdown | The earlier mask-in-router run (C1) at 85-101 tok/s with a frozen resident set | P0, P1: `STRATA_STANDIN=1`, `STRATA_ROUTE_RESIDENT_COUNT=0` (nothing for the swap rounds to bring in, as in C1), 2 requests per kind, hard 29 GB cap | The current build reaches 80 or more on the same conditions | **REPRODUCED for code, not for prose (2026-10-10).** Code 84.7 / 80.5 (P0), 84.4 / 80.2 (P1); reasoning 48.9 / 49.0, 57.3 / 60.0; prose 61.1 / 61.0, 64.6 / 64.5. The answers are the broken kind (code a 27-word description, reasoning cut after step 1, prose one sentence), so the speed is not usable. The earlier 101 on prose came from a 255-word looping answer the draft heads predicted almost perfectly; now the answers end after about 16 words. Drive reads 254 to 313 MB per token are the prompt reads divided over very short answers. Conclusion: the router kernel does not hold the speed down; what limits the sound arms is the miss rate and the NVMe cost of a miss |
| T24 | What does the router substitute, and with what weight and score gap? (user, 2026-10-10; step 1 of the stand-in work) | The substitute is the router's own next-best resident expert, and damage starts when replaced picks carry a large weight | Detail counters in `route_resident_k` (decode rows only, `STRATA_TRACE_ADAPT=1`) | D0 (tiny margin, the router's natural picks), D2 / D3 / D4 (unlimited margin, ranks 6-9 / 4-9 / 2-9); 2 requests per kind, hard 29 GB cap | Numbers that explain the sweep | **MEASURED (2026-10-10), one run.** Router weight at rank 0..9 (% of a row): 21.6, 14.7, 11.7, 9.9, 8.7, 7.8, 7.1, 6.6, 6.1, 5.8 (the top pick holds 22%, the top 5 hold 67%, ranks 6-9 hold 26%). Non-resident share of the picks by rank: 34% (rank 0) to 58% (rank 9). Replaced picks, ranks 6-9: weight under 5% for 19% and 5-10% for 81%; the substitute is the router's own rank 10-19 for 64%, 20-39 for 22%; its score is within 1 logit of the replaced pick for 91%. Ranks 4-9: weight 5-10% for 84%; score gap within 1 logit for 84%; substitute in the router's top 40 for 78%. Ranks 2-9: 15% of the replaced picks carried 10-20% of the weight, the gap exceeded 1 logit for 28%, and the substitute fell outside the router's top 40 for 33%; this is where the sweep's first damage appeared. CORRECTED after reading the answers (the user's rule, 15: quality is read, not scored): the code answers at ranks 2-9 are correct (a merge of overlapping intervals with working examples checked by hand, only shorter and one without the code fence); the damage seen there is word-level in the prose ("glemed" for "gleamed", and earlier "step-by solution"); ranks 4-9 and 6-9 read clean in what was read (2 code, 2 prose, 1 partial reasoning); every reasoning answer was cut at 256 tokens before its result, so reasoning correctness is not yet judged. The scores quoted in T21 to T24 (repeat-4gram, first differing word) are context only. Warm tok/s medians: D0 19.1 (cold requests included), D2 33.2, D3 37.4, D4 34.9; answers of D2 and D3 are as long and as clean as the control's. The stand-in is therefore the next-best resident expert by the router's score and is close in score while the replaced pick's weight stays under about 10%; replacing picks that carry 10% or more is where the quality goes |
| T25 | Are the answers at ranks 6-9, 4-9 and 2-9 as good as the control's when READ to the end (user: quality is read, not scored; 900-token answers)? | Replacing the weaker ranks keeps the answers correct and speeds them up | T22 and T24 (sweep and detail), read only in part | L0 (off), L2 / L3 / L4 (ranks 6-9 / 4-9 / 2-9), 3 samples per kind, 900 tokens, hard 29 GB cap, no throttle | Every reasoning sample reaches the right result; code correct; prose fluent | **PASSED for what was read (2026-10-10), one run per arm, 3 samples per kind.** Reasoning (trains: meeting 11:36, 156 km from A): 3 of 3 correct in all four arms (one L4 sample was cut by the token limit at the last multiplication after 11:36 and 2.6 h). Code (merge of overlapping intervals): read in L4, 3 of 3 correct, examples checked by hand (earlier runs D0 / D3 / D4 read the same way: correct). Prose (a lighthouse keeper finds a bottle): read in L4, 3 of 3 fluent, no typos or loops; earlier arms had isolated word glitches ("glemed", "step-by solution") at ranks 2-9 in 2 of about 8 prose and reasoning openings read, so the glitch rate at 2-9 is low but not zero. Not read in this run: the code and prose of L0, L2, L3. Warm tok/s (2nd and 3rd request): control code 16-20, reasoning 23-24, prose 29-31; ranks 6-9: 35 / 35-40 / 34-42; ranks 4-9: 39-45 / 44-50 / 35-43; ranks 2-9: 45-46 / 48-49 / 38-45. No OOM or kill events. The all-ranks setting stays broken (T21, T22): repetition of a wrong step, "a word of a word", a description instead of code, empty prose, a leaked `<|im_start|>assistant` |
| T26 | Is rank 1-9 (everything but the router's top pick replaced) still sound, read to the end? (user, 2026-10-10) | The floor can go below 2-9 | T25: 2-9 read correct | M1: ranks 1-9, `STRATA_ROUTE_RESIDENT=1e30`, 3 samples per kind, 900 tokens, hard 29 GB cap; every answer read | The same 9 of 9 sound answers as the control and ranks 2-9 | **FAILED for code and reasoning, passed for prose (2026-10-10), 9 samples, one run.** Code: 1 of 3 correct; one answer stopped after `def merge_intervals(intervals):` (2 words), one has a crash bug (`merged = []` with `if not merged or ...` takes the merge branch on the first interval and reads `merged[-1]` of an empty list), one correct. Reasoning (trains, 11:36 and 156 km from A): 2 of 3 correct; one reached the wrong meeting time 11:52; typos "step-step solution" and "Analy the Head Start". Prose: 3 of 3 fluent, no typos or loops. Warm tok/s (2nd and 3rd request): code 42.0 / 53.1, reasoning 53.2 / 61.2, prose 38.5 / 44.4. No OOM or kill events (least available 26.4 GB). So the sound limit for code and reasoning is between ranks 2-9 (9 of 9 sound) and 1-9 (6 of 9); prose is the most tolerant task, as the per-task design (SPEC 17) assumed. One run of 3 samples per kind: it shows damage exists at 1-9, not its exact rate |
| T27 | What do ranks 0-9 (every pick replaceable) write over full-length answers, and at what speed? (user, 2026-10-10: "run once to get 85-101 tps") | The 80-100 tok/s is real speed but not of useful answers | T21, T22, T23 (the same setting on short answers) | M0: ranks 0-9, `STRATA_ROUTE_RESIDENT=1e30`, 3 samples per kind, 900 tokens, hard 29 GB cap; every answer read | Answers as good as the control's | **FAILED (2026-10-10), 9 samples, one run.** Warm tok/s: code 84.8 / 80.6 / 80.5, reasoning 77.1 / 78.8 / 78.9, prose 61.1 / 61.0 / 60.8. Answers read, identical across the three samples of each kind: code is a 27-word restatement of the task ("The user is asking for a Python function that merges overlapping intervals ...") with no code; reasoning is 6 words ("Let me work this out step-by") and ends; prose is one 16-word sentence ("The user's request is a bit of a message in a bottle on a stormy night."), no story. The speeds are over 6 to 27 tokens and say nothing about speed on a real answer. The failure is a premature end after a restating opening, deterministic, not a repetition loop. No OOM or kill events (least available 26.1 GB) |
| T18 | Was the capped testbed itself the slowdown? (user's suspicion) | `MemoryHigh` throttling, not the memory limit, caused 11.4 | Throttle events in the trace (230,000+) | N1: staged default, `MemoryMax` 29 GB only, no `MemoryHigh`, 3 requests per kind, live feed | N1 clearly above 11.4 | **CONFIRMED (2026-10-10).** Median 27.7 tok/s (code 17.0/27.0/28.6, reasoning 20.3/27.9/31.2, prose 21.4/27.7/37.2), 86 MB read per token, 0 throttle events, least available memory 25.8 GB, swap 1.08 GB (idle processes), no guard trip. One run |

## 17. Decided: a stand-in floor per task, switched inside one answer (user, 2026-10-10; built, measured, REMOVED the same day)

**Status 2026-10-10: removed.** The per-part floors, the thinking/fence tracker, the request keys and the server field were built and benchmarked (18.1: the higher code floor gave no accuracy gain and ran slower), then taken out once `2-9` became the install's default (`strata-iq3_s.json` env: `STRATA_ROUTE_RESIDENT=1e30`, `STRATA_ROUTE_RESIDENT_RANKS=2-9`; verified from a plain start: ranks 2-9, median 42 tok/s on the speed prompts). What stays: the floor in host-mapped memory (`route_cfg_set_floor`), for a controller that follows the drive's load (19.8). The text below is the record of the design; recover the code from the git history before this commit if the question returns. The arm runner `strata_ab.py` serves a config copy without the stand-in keys, so an arm's own environment (or the control's absence of one) decides.

The decision is about what we build, not about what the engine already supports.

- **Idea.** How many of the router's picks may be replaced by resident stand-ins is a speed-for-quality lever (T22, T24), and the model's tasks tolerate different amounts: the thinking inside `<think>` is the most forgiving, the prose written for the user less, code the least. One answer contains all three, so the floor changes with the part being written: three floors (the `lo` of `STRATA_ROUTE_RESIDENT_RANKS`) in one table, thinking <= prose <= code in rank (a lower floor replaces more).
- **What is built.** A segment tracker in the engine: it reads the committed tokens and knows whether the answer is inside the thinking, in prose or in a code block; the request tells it the opening state and the token ids it needs (`<think>`, `</think>`, the code fence). The three floors live in a small device table that the router kernel reads each window, so a switch needs no graph recapture. The server sends the ids and the floors with the request; the defaults come from the per-task numbers below, not from hand-setting.
- **What decides the floors.** Per-task accuracy at each floor, read as well as scored: reasoning on GSM8K (written step by step, and with the model's own thinking on), code on HumanEval (run against its tests), prose by reading samples. A task gets the lowest floor whose answers are still as good as the control's within the noise of an identical-copy control. If the tasks do not differ, one floor stays and this section is withdrawn.
- **A request-level dial.** The floor is exposed like the temperature: a stand-in level in the request (0 = off, then the floors above), mapped to the rank floors of the three tasks. Unlike the temperature it changes the speed, because it changes which experts compute (it is closer to a quantisation level), and it is not repeatable: what is substituted depends on what is resident, so the same prompt can come out differently by history even when greedy.
- **Order.** 1. per-task numbers (`tools/linkbench/bench.py`, question sets from `fetch_bench.py`); 2. the device floor table with three floors fixed from the environment (today's behaviour, per segment); 3. the segment tracker and its token ids from the server; 4. the same benchmark with the floors switching.
- **Implementation state (2026-10-10), committed, ENGINE NOT YET BUILT** (the benchmark sweep holds the machine; compiling would disturb its speed figures). **SUPERSEDED LATER THE SAME DAY by the fence rule of 17.3, item 4: no `ans=`, no `"strata_standin_task"`, four ids (think open and close, the fence token, the fence token with a leading space), `seg=0|1`, and the floor in force `eff` set from the thinking and fence states; read the paragraph below for the history and 17.3 for what is built.** Step 2: `RouteCfg` in host-mapped memory (`native_router.hpp`), floors per task, read by `route_resident_k` at each launch; `STRATA_ROUTE_FLOORS=t,p,c` sets the environment's three floors, default all three equal to `STRATA_ROUTE_RESIDENT_RANKS`'s `lo` (behaviour-neutral). Step 3, engine: request keys `seg=` (starting segment), `ans=` (what the answer is once thinking ends: 1 prose, 2 code), `segids=` (think open, think close) and `floors=` (this request's three floors) on the `GEN` line, a tracker called at both token-commit sites (`</think>` moves to the request's answer kind, `<think>` to thinking), per-request floors restored to the environment's when absent. **Design rule (the user, 2026-10-10: no regex and no pattern matching on tokens for the decision, it is a rabbit hole): the tracker compares a committed token id for equality with the model's own two control tokens and looks at nothing else; whether the answer is prose or code is declared by the request (`"strata_standin_task": "code"`), never guessed from the text or the tokens. A first version toggled "code" on a "```" token; that guess (fence merged into "```python", other fence styles, inline backticks, a fence inside thinking) was removed before it was ever built.** Step 3, server: a request that carries `"strata_standin_floors": [thinking, prose, code]` (and optionally `"strata_standin_task": "code"`) gets the keys added (`Service.with_segments`, the two ids from the tokenizer once, `segment_ids`); a request without the field is unchanged. **Tested so far:** the server half, in isolation with a stand-in tokenizer (ids resolved; field absent = unchanged; bad field ignored; prose or code, thinking or not, gives `seg=1 ans=1`, `seg=0 ans=1`, `seg=2 ans=2`, `seg=0 ans=2`). **Not yet tested:** the engine half (it does not compile-check until built), the real token ids of `<think>` and `</think>` in this tokenizer, and the neutral-behaviour check (all floors equal gives today's answers). A request-level dial by name (a level mapped to the three floors) is not built: the floors come from the benchmark (SPEC 18).

### 17.1 Open question: the floor together with the sampling temperature (user, 2026-10-10; UNTESTED, a hypothesis)

Every measurement in this document so far is greedy: the engine's default is greedy and an absent temperature means greedy (`serve/server.py`, sampling keys); the benchmark sends temperature 0. Nothing here says what the floor does at a higher temperature.

- **Why the two should interact.** A stand-in perturbs the model's scores, because a different expert computes part of a layer. Greedy decoding only changes the text where two tokens score close, which is the divergence seen, and the answers stayed sound down to ranks 2-9. A higher temperature samples from the whole distribution including the low-probability tail; if the stand-ins thicken or shift that tail, a sampled low-probability token (a misspelling, a wrong digit) becomes more likely. The isolated word glitches seen at the lower floors ("glemed", "step-by solution") are the kind of token that would show first. Prediction: a lower floor with a higher temperature degrades more than either alone.
- **A possible benefit, equally untested.** A modest temperature can break repetition loops, the degeneracy the user named as the real failure; the floor might allow a lower temperature or the reverse.
- **Test, after the benchmark sweep (SPEC 15, T-rows to follow):** the same GSM8K and HumanEval at two or three floors (the control, 2-9, 1.5-9) and temperatures 0.3, 0.7 and 1.0, with more than one sample per question at the higher temperatures (sampling adds its own noise), and with a sample of the answers read, as the rule is.

### 17.2 A middle path: the floor follows the model's own confidence (user, 2026-10-10; UNTESTED, a hypothesis)

The user's idea: the system writes the answer (code, sums) and could know how sure it is, so instead of one floor per task (17) the floor follows that confidence token by token: aggressive stand-ins where the model is sure, exact experts where it is not.

- **What exists.** The engine already computes a per-position confidence for each draft token (`pr[...]`, `mtp.chain_prob()`, used by `--spec-min-p` to decide how long each verify window is) and the verify pass has the full output distribution of every position. The draft confidence is known BEFORE the window runs, so a floor for the window could be chosen from it at no extra cost: through the same device config as 17 (a "confident" and an "unsure" floor in place of thinking / prose / code), with no text and no token pattern looked at.
- **What is not known, and decides whether it works.** (1) Whether confidence predicts where stand-ins do damage. The damage seen in E2 is wrong sums in a fluent answer; digits of a sum are usually written with HIGH confidence, so a confident token is not necessarily a safe one, and a draft's confidence says how sure the draft head is, not how much the perturbed experts have moved the answer. (2) Which confidence: the draft head's, the verify pass's top probability, the router's own score gap and entropy at the layer (the cheapest and the closest to the thing being replaced).
- **Test (cheap, before building anything): does confidence predict damage?** Log, for each committed token, the top-1 probability the verify pass saw (a trace switch in the engine, like `STRATA_TRACE_ADAPT`), run the control and a stand-in arm on the same GSM8K questions, and compare the confidence the CONTROL had at the tokens where the stand-in arm's answer first departs or goes wrong against its confidence elsewhere. If the damage sits at low-confidence tokens, a confidence floor can protect it; if it sits at high-confidence tokens (digits of sums), it cannot, and the router's own signals (score gap, weight, SPEC 19.8) are the better lever.
- **Order:** after the per-task floors (17) are built and checked; the trace switch is a small change and goes in the same build.

### 17.3 What signal says "now code, now reasoning, now prose"? (user, 2026-10-10; the first two are built, the third is a hypothesis)

`--spec-min-p` (the draft head's probability that its guess is right) is NOT such a signal: it is high wherever the text is predictable, in code boilerplate and prose clichés alike, so it says how predictable the text is, not what kind. The candidates, in order of robustness:
1. **The model's own control tokens `<think>` and `</think>`, by id equality** (17, built): reasoning against the answer.
2. **The request declares the answer's kind** (`"strata_standin_task"`, 17, built): code against prose for the whole answer; coarse, code inside a prose answer is not separated.
3. **WITHDRAWN (user, 2026-10-10: "we were going to drop the atlas"; SPEC 13.12, 2026-10-08: no atlas is needed, the router's own ranking picks the stand-in).** A table of per-expert topic affinity is an atlas under another name and contradicts that decision; it is not to be built. What stays of the idea is only the question it asked, whether the router's own state can say what is being written, which would have to be answered without any such table. Text kept for the record: "Which experts the router picks (UNTESTED here)." Solid's atlas measured the underlying fact: every expert has a topic affinity from labelled probes in nine categories, replicated before an expert counts as a specialist, and the map predicts which experts fire together (link-prediction AUC 0.7116 against 0.5437 for expert frequency alone and 0.5761 for an embedding built from the firing data itself); solid also found that this map did not by itself make the running system faster, which is a different use from a classifier. Our use: a table of how much each expert leans to code, reasoning or prose, applied to the experts the router picked over the last few windows; no text, no token pattern.
   (Withdrawn with item 3: the test below, which counted per-expert use per task, would have built that table.) ~~Test: run prompt sets whose task is known in advance, count per-expert use per task, check whether a readout separates the three.~~
4. **The code fence "```" is NOT a signal for code (user, 2026-10-10: "are you sure ``` is the right signal for code?").** It appears in prose that quotes code, may be one token with the language name ("```python"), is absent from code that is not fenced (tool-call arguments, JSON, indented code, inline snippets) and can appear inside thinking; every variant would need its own rule. The first tracker toggled on it; removed before it was built. The coarse way that remains is signals 1 and 2: the model's two thinking tokens by id equality, and the request declaring the answer's kind. A mixed answer (prose with a code block) gets the kind the request declared; a finer per-token signal has no candidate that is not a pattern or an atlas, so there is none.
   **MEASURED afterwards (2026-10-10, the user asked to check the saved answers, then proposed allowing the fence "also within the thinking").** With this model's tokenizer a fence is one token: `'```'` (id 71093), `' ```'` with a leading space (52451) the only other token holding three backticks; the language name is a separate token ("```python" = `'```'`, `'python'`); four backticks are two `'``'`; `~~~` is not a fence. In the saved answers: HumanEval control 30 / 30 and B2 30 / 30 fenced, each with one balanced block, no unfenced code, an even number of fence tokens everywhere (180 fence tokens in all, every one id 71093, none 52451); no fence in any of 188 GSM8K answers (no false positive in maths); **but E2, the arm where damage began, has 2 of 30 code answers with no fence, and every unfenced code answer among 73 earlier code-prompt answers comes from a stand-in arm (ranks 2-9 three times, ranks 1-9 once; none from the control).** So the fence goes missing as the output degrades, the case the floor most needs to be careful in. **[CORRECTED later the same day, 18.1: reading the unfenced answers showed that most are sound code (P2: 5 of 6 unfenced pass, E2: 2 of 2), and `HumanEval/40` is unfenced with the same 230-token answer in E1 and P2; omitting the fence is partly the model's style on some problems, so "the fence goes missing as the output degrades" is too strong. What is true: a missed fence is common enough that the fence cannot be the only protection, which the escalate-only design already assumes.]** **Verdict: usable only if it can make things safer and never riskier.** (a) The request declares the baseline, so a code request already holds the code floor and does not depend on the fence. (b) A fence token (71093 or 52451, id equality, nothing else read) may only RAISE the floor to the code floor for code inside a prose answer or inside thinking, and a second fence returns it; a spurious fence costs speed only, a missed fence costs nothing beyond the baseline. (c) The fence state resets at `</think>` and at each request start, so one missed or extra fence cannot leave the segment wrong for long.
   **DECIDED AND BUILT (user, 2026-10-10: "yes, build the fence rule that way, but (a) is not cheap and I don't want to get into that"; "the answer prose may have code in it too, same rule as the thinking block").** Points (b) and (c) are built; (a), the request declaring the baseline, is NOT: no `strata_standin_task`, no `ans=`. The rule: the tracker compares each committed token id for equality with four ids the server looks up once (`<think>`, `</think>`, the fence token, the fence token with a leading space) and reads nothing else; `<think>` and `</think>` set the thinking state (and `</think>` clears the fence state); a fence token toggles the fence state, in the thinking and in the answer alike; the floor in force is the floor of the part (thinking or answer) and, inside a fence, never less careful than the code floor (`eff = max(part floor, code floor)`, a higher number replaces less). A request is `seg=0` when it starts inside the thinking, `seg=1` in the answer. **Consequence of dropping (a), to be kept in mind when the floors are chosen:** with no declared baseline, code that the model writes WITHOUT a fence gets the answer's floor, and unfenced code is what appeared in the stand-in arms (2 of 30 E2 answers). So the answer floor has to be one at which unfenced code is still sound (the benchmark says: ranks 2-9 held 30 / 30 on HumanEval; the edge arms show where it stops), and the code floor only adds care inside a fence. A spurious fence (in a maths or prose answer: none in 188) costs speed only. **The flag (user, 2026-10-10: "a flag to choose per type of request, or keep the lowest default floor for all types"):** `STRATA_STANDIN_PARTS=1` turns the per-part floors on; off (the default) there is ONE floor, the environment's `STRATA_ROUTE_RESIDENT_RANKS` `lo`, for every request and every part of an answer, and a request's floors, `STRATA_ROUTE_FLOORS` and the tracker are ignored. **Built and tested (2026-10-10, engine compiled, no errors):** (1) the neutral check, flag off, ranks 2-9: the new binary gives correct answers at the speeds of the earlier ranks 2-9 runs (reasoning: 156 km reached in all 3 samples, two of them verified both ways; code: 3 of 3 correct; warm tok/s code 33 / 32 / 45, reasoning 43 / 53 / 58, prose 32 / 39 / 42, against 34 to 46, 40 to 49 and 30 to 45 before; no OOM or kill events); (2) the tracker against the real tokens (`seg_probe.py`, `STRATA_TRACE_SEG=1`, flag on, floors 2,2,4): the thinking request changes state at token 248069 (`</think>`), the code request toggles the fence at token 71093 on and off around its block, the maths request produces no state change at all. **Not yet tested:** the per-part floors on the benchmark (floors 2,2,4 against 2,2,2).

## 18. Benchmark sweep: accuracy against the stand-in floor (started 2026-10-10; filled in as each arm finishes)

GSM8K (the first 50 of a fixed 100-question sample, ground-truth number) and HumanEval (the first 30 of a fixed 40-problem sample, run against the benchmark's own tests), greedy, no limit on an answer's length except the 32,768-token context, hard 29 GB memory cap with no throttle, one run per arm. Rows are rebuilt by `tools/linkbench/bench_table.py`. The scores are ground-truth correctness; an answer sample of each arm is read as well (SPEC 15, quality is read) and noted below the table. With 50 and 30 questions, one question is 2 and 3.3 points: a difference of a few points between arms is noise, which the second control copy (BC2) shows.

<!-- BENCH-TABLE-START -->
| Arm | Replaced | GSM8K correct | HumanEval pass | Decode tok/s (median) | First token (s, median) | Whole-request tok/s | Longest answer (tokens) | State |
|---|---|---:|---:|---:|---:|---:|---:|---|
| BC | control (nothing replaced) | 48 / 50 | 30 / 30 | 11.6 | 8.2 | 9.7 | 4445 | done |
| B2 | ranks 2-9 | 47 / 50 | 30 / 30 | 29.9 | 7.9 | 15.3 | 542 | done |
| E1 | 2-9, rank 1 only by a substitute within 1 logit (the edge, margin 1) | 48 / 50 | 29 / 30 | 36.3 | 7.5 | 17.4 | 603 | done |
| E2 | 2-9, rank 1 only within 2 logits (edge margin 2) | 44 / 50 | 28 / 30 | 36.1 | 7.7 | 18.0 | 1408 | done |
| E4 | 2-9, rank 1 only within 4 logits (edge margin 4) | | | | | | | pending |
| B1 | ranks 1-9 (rank 1 at any distance) | | | | | | | pending |
| BC2 | control, second copy (the noise floor) | | | | | | | pending |
| B0 | ranks 0-9 | | | | | | | pending |
| B6 | ranks 6-9 | | | | | | | pending |
| B4 | ranks 4-9 | | | | | | | pending |
| B9 | rank 9 only | | | | | | | pending |
| B8 | ranks 8-9 | | | | | | | pending |
| B7 | ranks 7-9 | | | | | | | pending |
| B5 | ranks 5-9 | | | | | | | pending |
| B3 | ranks 3-9 | | | | | | | pending |
<!-- BENCH-TABLE-END -->

Reading notes (answers read per arm):
- **BC (control), read:** the two GSM8K misses are different kinds. Question 18 (tomato vines, reference 21, answered 14) is a real error: the model decided only Steve's six a day counts and left out his girlfriend's three. Question 31 (air conditioner, reference 81, answered 135) is the model being right and the reference wrong: saving 5 hours of a 900 W unit is 900 x 5 = 4.5 kWh a day, 135 kWh in 30 days; the reference subtracts the energy of 5 hours from the energy of 8 hours and gets 2,700 Wh a day, which is the energy still used, not saved (a known label problem). The score is kept as the benchmark gives it (48 / 50); read, the control has 49 of 50 correct. The longest answer, 4,445 tokens (GSM8K question 38, a cheerleader-pyramid puzzle), is not a loop: the model deliberated at length and finished with a worked, correct answer (21 feet); a 700-token cap would have scored it wrong. All 30 HumanEval functions pass their tests.
- **B2 (ranks 2-9), read:** GSM8K 47 / 50, HumanEval 30 / 30. The three misses: questions 18 and 31 are the control's own two (the same wrong answer 14 for the vines, and again 135 against the faulty reference 81); question 38 (the cheerleader pyramid) is new: B2 answered in 467 tokens that the pyramid is 60 + 64 + 64 = 188 inches, 15.67 feet (three layers counted, the reference is 21 feet), where the control reached 21 only after 4,445 tokens of deliberation. The answer is fluent and well formed: an ordinary reasoning slip on a puzzle where the control itself needed a very long think, not degeneration, and one question cannot show a quality loss. Read, the control has 49 of 50 correct and B2 48 of 50. Time to the first token is the same in both arms (8.2 s control, 7.9 s B2, about 107 prompt tokens, prompt reading about 13 tok/s) and decode speed is 11.6 against 29.9 tok/s (2.6x); the whole-request column shows 1.7x because the first token takes 23% of the control's request and 46% of B2's.
- **E2 (ranks 2-9 plus rank 1 replaced only by a substitute within 2 logits), read:** GSM8K 44 / 50, HumanEval 28 / 30, decode 36.1 tok/s (3.1x the control, 1.2x B2), first token 7.7 s. Of the six GSM8K misses, question 31 is the benchmark's label error again; the other five are new against the control and B2: four are plain arithmetic slips inside otherwise sound, fluent solutions (question 2: 3 x 16.50 written as 48.50, it is 49.50, total 242 for 243; question 12: 200 + 400 + 100 written as 600, answer 1,200 for 1,400; question 21: 900 + 90 + 180 written as 1,090, answer 355 for 315; question 44: 65 - 40 written as 15, answer 60 for 100) and one is a misreading (question 14: half of Jan's age taken now instead of two years ago, 24 for 23). HumanEval/32 (find_zero, a root finder) and HumanEval/92 (any_int) fail their tests, where the control and B2 passed all 30. Reading: no loops, no broken words, every answer well formed, but simple sums come out wrong four times in 50, which is digit-level damage and not the model's usual kind of error (the control's two real misses are a misread puzzle and the benchmark's label). Suggestive that margin 2 on rank 1 is past the edge, not conclusive at 50 and 30 questions: BC2, the second control, shows the run-to-run noise, and E1 (margin 1) tests a tighter margin.
- **E1 (ranks 2-9 plus rank 1 replaced only by a substitute within 1 logit), read:** GSM8K 48 / 50, HumanEval 29 / 30, decode 36.3 tok/s (3.1x the control, 1.2x B2), first token 7.5 s. The two GSM8K misses are the control's own (question 18, the misread vines puzzle, same wrong answer 14; question 31, the benchmark's label error, 135 against the faulty 81): read, 49 of 50, the control's score, and none of E2's arithmetic slips. The one HumanEval failure (HumanEval/92) is an EMPTY answer: 231 tokens generated and no visible text at all, the failure kind of the empty and truncated answers at ranks 0-9; it did not occur in the control, B2 or E2, and once in 30 cannot be told from noise without the second control. Reading: margin 1 on rank 1 keeps the answers sound in the 50 maths questions where margin 2 broke sums, at the same decode speed as E2, with one empty code answer. Not conclusive: the sweep was stopped here by the user's decision to move to SPEC 17 first; E4 (margin 4), ranks 1-9, the second control (BC2), ranks 0-9 and the milder floors are not run yet and are queued for later (they need no new code; the arms are listed in `restart_chain.py`). **CORRECTION (2026-10-10, the user asked "are these really empty?"):** that failure is "no visible `content`", not proven empty: the benchmark script then saved only `content`, and the server files reasoning text under `reasoning_content` and tool calls under `tool_calls`; the answer may hold text. Runs after commit d0651db keep the whole reply (18.1).

### 18.1 SPEC 17 runs: HumanEval side by side (user, 2026-10-10; rebuilt by `tools/linkbench/s17_table.py`)

The same 30 HumanEval problems, greedy, no limit on an answer's length, hard 29 GB cap, one run per arm. "Unfenced" counts answers with no code fence at all. The old-binary arms (BC, B2, E1, E2) ran before the SPEC 17 build; the others on the SPEC 17 binary. The purpose: P1 (floors 2,2,4) fell to 26 of 30 where B2 had 30 of 30, and the question is whether that comes from the code floor, the new binary, the request shape, or noise.

<!-- S17-TABLE-START -->
| Arm | Setting | Binary | HumanEval pass | Unfenced answers | No visible content | Whole-request answer tok/s | State |
|---|---|---|---:|---:|---|---:|---|
| BC | control (nothing replaced) | old binary | 30 / 30 | 0 | none | 7.6 | done |
| B2 | ranks 2-9 | old binary | 30 / 30 | 0 | none | 11.2 | done |
| E1 | 2-9 + rank 1 within 1 logit | old binary | 29 / 30 | 2 | 92 | 11.8 | done |
| E2 | 2-9 + rank 1 within 2 logits | old binary | 28 / 30 | 2 | none | 14.8 | done |
| R1 | ranks 2-9, flag off, no floors in the request (B2's setup) | SPEC 17 binary | 29 / 30 | 3 | none | 15.5 | done |
| P2 | floors 2,2,2, parts flag on | SPEC 17 binary | 28 / 30 | 6 | none | 10.9 | done |
| P1 | floors 2,2,4, parts flag on | SPEC 17 binary | 26 / 30 | 2 | 112 | 9.0 | done |
| P1b | floors 2,2,4 again, whole reply kept | SPEC 17 binary | 27 / 30 | 5 | 48, 100 | 9.4 | done |

- E1 HumanEval/92: 231 tokens, no `content`; the raw reply was not kept (run before d0651db): UNVERIFIED
- P1 HumanEval/112: 56 tokens, no `content`; the raw reply was not kept (run before d0651db): UNVERIFIED
- P1b HumanEval/48: 101 tokens, finish `tool_calls`; reasoning_content 0 words, tool_calls 1, content 0 words; reasoning head: ''
- P1b HumanEval/100: 367 tokens, finish `stop`; reasoning_content 0 words, tool_calls 0, content 0 words; reasoning head: ''
<!-- S17-TABLE-END -->

Reading notes:
- **What the unfenced answers are (read, the user asked "the answers are unfenced, but what is the quality?").** Most are plain, sound code written without a fence. P2: 6 unfenced, 5 pass their tests (`/40`, `/44`, `/52`, `/80`, `/88`); the one failure, `/48`, is only `    return text == text[::-1]` (9 tokens): the body without its `def` line, an incomplete answer. E2: 2 unfenced (`/68`, `/84`), both pass. E1: 2 unfenced, `/40` passes and `/92` has no visible content (unverified, see the correction above). P1: 2 unfenced, both fail: `/104` starts with a single stray backtick before `def` (a syntax error in otherwise reasonable code) and `/112` has no visible content (unverified). `HumanEval/40` is unfenced in BOTH E1 and P2 with the same 230-token answer: for that problem it is how the model answers, not damage from the stand-in. So an unfenced answer is not by itself a sign of damage; the damaged ones are the incomplete (no `def`), the stray backtick, and the no-content ones.
- **TERMS (the user, 2026-10-10: "damage is not the wrong answer but the artefacts like repetition loops", then "instead of calling damages in the wrong answers we call degradation, not degeneration").** **Degeneration** is an artefact in the output's form: a repetition loop, repeated characters or tokens, a leaked control token, a stray character, a broken word. **Degradation** is a loss of accuracy: more wrong answers than the control gives (the control makes some too: question 18, the label error 31), reported as accuracy, separately from degeneration. The word "damage" is not used any more. Earlier rows of this document that say "damaged", "damage" or "the edge where it degenerates" for wrong answers (E2's arithmetic slips; ranks 1-9's wrong meeting time and crashing code; T22, T24, T26) mean DEGRADATION; where they speak of loops, "a word of a word", leaked tokens, stray characters or broken words (ranks 0-9, C1, K3, the stray backtick, "step-step") they mean DEGENERATION. `tools/linkbench/artefacts.py` scans every saved answer for loops (a phrase repeating back to back), character runs, leaked control tokens, a stray opening backtick; its flags only choose what to READ.
- **Artefact scan of the saved answers (read after flagging).** No repetition loop in any benchmark arm (BC 0 of 80 answers flagged; B2 1; E1 0; E2 0; P1 2; P2 1; R1 0 so far). Loops exist only in the old mask-in-router run C1 (ranks 0-9, a 3-word phrase repeated 83 times in a prose answer, "a word of a word of ..."); the full stand-in run K3 leaked `<|im_start|>` (a 2-word prose answer). The benchmark flags, all read: B2 `HumanEval/48` wrote a complete `<think> ... </think>` block, with its reasoning, inside the visible answer although thinking was off for the request, then the code (passes); P1 `/36` and P2 `/60` begin with a stray `</think>` before sound code (pass); P1 `/104` begins with a single stray backtick before `def` (a syntax error). So leaked thinking tags and a stray character appear at about 1 to 3 answers in 30 to 80 in the 2-9 arms (B2, P1, P2) and in none of the control's 80 answers; E1 and E2 (2-9 plus the rank-1 edge) have none flagged. Under this definition E2, called damaged above for its wrong sums, shows no artefact: the edge of artefacts lies further out than E2's margin 2, and E4, 1-9 and 0-9 on the benchmark are the arms that can show it (1-9 earlier showed broken words: "step-step", "Analy", read in the speed-prompt arms, which the scan cannot flag). The no-content answers (E1 `/92`, P1 `/112`) may be a think block filed under `reasoning_content`: UNVERIFIED until a raw reply is read.
- **Verdict on the SPEC 17 runs (R1 and P1b in; 2026-10-10).** (1) **The binary is not the cause:** R1 (B2's exact setup on the SPEC 17 binary, flag off, no floors in the request) scored 29 / 30; the 2-9 family scores 28 to 30 over five runs (B2 30, R1 29, P2 28, E1 29, E2 28), so B2's 30 / 30 was partly luck and the noise of this 30-problem test is about 2 answers. (2) **The higher code floor did not help:** floors 2,2,4 scored 26 / 30 (P1) and 27 / 30 (P1b), the two lowest, and ran slower (whole-request 9.0 and 9.4 tok/s against 10.9 to 15.5 for 2-9); failures 7 of 60 answers against 6 of 150 for the 2-9 family (Fisher p about 0.07: suggestive, not proof). A more careful floor inside fences should not lower accuracy and there is no mechanism known for it; until a run shows a gain from it, the code floor earns nothing. (3) **Degradation** (wrong answers): `HumanEval/32` (a root finder, numeric) fails in E2, P2, P1 and P1b and never in the control, B2, R1 or E1; the other failures are single problems (36 in R1; 92 in E1 and E2; 48 in P2 and P1b; 100, 104, 112, 116 once each). **Degeneration** (artefacts): no loop in any benchmark arm; leaked `<think>` or `</think>` tags in B2 (1), R1 (1), P2 (1), P1 (1), P1b (2), a stray opening backtick in P1; none in the control, E1 or E2. (4) **The no-content answers, read from the saved raw reply:** P1b `/48`: finish `tool_calls`, content null, ONE hallucinated tool call (a function named `code`, empty arguments) although no tools were offered: a degeneration artefact, not an empty answer; P1b `/100`: 367 tokens, finish `stop`, content null, no reasoning, no tool call: nothing visible in any field, where the tokens went is UNEXPLAINED and needs the raw token stream (a trace of the committed token ids, to be added); E1 `/92` and P1 `/112` ran before the raw reply was kept: UNVERIFIED. (5) **What the floor is, by these data:** one floor, `2-9`, with the parts flag OFF (the default); per-part floors stay built and off. 2-9 gives about twice the control's whole-request speed (R1 15.5 against 7.6 tok/s) and 2.6x its decode speed, accuracy within the noise of the control, no loop, and rare stray control tokens (about 1 answer in 30 to 80). 2-9 plus the rank-1 edge within 1 logit (E1) was as accurate as the control and had no degeneration flagged, with one unexplained no-content answer; it is the candidate for the next step down, to be repeated with the whole reply kept. The thinking floor, the second control (noise of a repeat of the control itself), margin 4, ranks 1-9 and 0-9 on the benchmark and the milder floors remain untested.

## 19. Queued for later (user, 2026-10-10): the swap budget, and what the links allow

Nothing here has been run; the figures marked "worked out" are arithmetic from measured link rates, not measurements.

- **What can be set.** `--adapt-swaps N` (default 96), `--adapt-every N` windows (default 4) and `--adapt-decay` (default 0.7) are flags, so they go in the `args` of the config JSON and need no rebuild. Two constants are written into the swap rule in `src/program/generate.cpp` and are not flags: a candidate needs a count of at least 2.0, and must beat the victim by at least 1.5. They would need a code edit and stay as they are for now.
- **Why the budget is worth testing.** The trace of T21 / T22 showed every round using its full 96 swaps with more than a thousand candidates waiting, so the budget is binding, not spare.
- **What the links allow (measured: pinned RAM to VRAM 24.2 GB/s, one round of 96 x 2 MiB in 8.3 ms; NVMe 3.57 GB/s; section 13.55).** Worked out for a round of about 270 to 390 ms: from pinned RAM over PCIe about 3,000 to 4,500 experts of 2 MiB per round, so PCIe is not the limit and the 2,721 GPU cache slots are; from the NVMe about 500 to 700 per round if nothing else used the drive. In the control arms the misses already read about 100 MB per token at 20 to 28 tok/s, about 2 to 3.5 GB/s, nearly the whole drive; that leaves about 0.5 to 1 GB/s for swaps, about 100 to 250 experts per round, so 96 is already near a sensible budget for drive-sourced swaps, and raising it blindly can lose speed.
- **The tests, after the floor-edge sweep (SPEC 18), at the chosen floor (ranks 2-9 unless the sweep says otherwise).**
  1. `--adapt-swaps` 48, 96, 192 and `--adapt-every` 2, 4, 8: nine short arms; tok/s, the engine's drive bytes per token, and a sample of the answers read.
  2. The RAM-tier mode again, WITHOUT the memory throttle (`--resident-experts`, AB_NO_HIGH=1): T17 failed under the throttled scope, which was a testbed artefact (T18). In that mode swaps come over PCIe from pinned RAM (22.5 GiB of the roughly 83 GB model), where a much larger swap budget is affordable; try it with `--adapt-swaps` 192 and 384. **PRIORITY (user, 2026-10-10): run this first after the floor-edge sweep.** It is the user's structure: the hot experts held in RAM by us, not by the kernel's page-cache LRU; the cold experts read unbuffered (`O_DIRECT`, small staging buffers only) so they never enter the page cache and never displace the hot ones. The default staged read is buffered (the page cache grows to 26-27 GB in every trace), so cold experts there compete with hot ones under the kernel's recency-only eviction (solid measured a 21x collapse from that under memory shortage). A direct drive-to-VRAM DMA is not available here (item 7), but this part of the argument does not need it. A second cost to look at: the RAM bounce is about 7 GB/s of memory traffic (3.5 in, 3.5 out) against the measured 36.6 GB/s, shared with the CPU pool; unmeasured. Arms: control path (default staged, buffered), `--resident-experts` (hot owned, cold unbuffered), with the memory guard on and without `MemoryHigh`; read the answers as usual.
  3. Only if 1 or 2 shows a gain: the two constants.
  4. **Misses over PCIe instead of on the CPU** (`--pcie-frac` 0.55 default, 0.8, 1.0; the CPU pool computes the rest). Solid's record: its CPU fallback for an unsubstituted miss cost 48 CPU-seconds in a 9.7-second request and stalled requests under memory pressure, so it made "a miss never reaches the CPU" the default (`FORCE_GPU`); the first version collapsed the hit rate 22% to 0.9% by evicting residents and was fixed with a small reserved region ("within noise" of off). Solid gives no speed figure for the flag. For us (worked out, not measured): at floor 2-9 about 29 rank 0-1 misses per token remain; at 0.6 ms of drive time per 2 MiB expert that is about 17 ms per token at best, a limit near 58 tok/s, worse without fetch-ahead.
  5. **A small reserved region for fresh misses** (solid's idea). Strata's cache changes only at a swap round, so the collapse solid saw (a miss evicting a resident) cannot occur here; the analogue is the delay until the next round. T22 hints at reuse inside a round (about 1,150 candidates carried about 3,400 wanted picks, three each), which an immediate-admission region could catch. The no-code test of the same idea first: `--adapt-every 1 --adapt-swaps 24` (the same swaps per window as 96 per 4, four times finer in time). Only if that gains, build the region.
  6. **Solid's own wins that are real and not in Strata** (solid's table): expert prefetch for prefill, overlapped weight uploads, +36% on prompt speed (ncmoe 20, pp2048). Our prompts are slow (15 tok/s on short prompts). The other names raised: solid's "D2D" is a device-to-device copy of resident experts into the prefill buffer; its own record calls the mechanism working but the buffer costs 5% and it was superseded, so it is not a clear win. No "S2D" exists in solid's code or docs; if a storage-to-device read is meant, solid measured explicit `pread()` against mmap as within noise (750.4 against 729.3 ms), while Strata's staged read gained +60% (T1), so Strata is not behind there.
  7. **Direct NVMe to VRAM (GPUDirect Storage), checked on this machine (2026-10-10, `gdscheck -p`, MEASURED).** The driver (`nvidia_fs` 2.30) and the cuFile tools are installed, but: the NVMe is in compat mode; the GPU line reads "GPU 0 NVIDIA GeForce RTX 3060 **NO GDS SUPPORT** - Reason: GDR=unsupported" (a GeForce card has no GPUDirect RDMA), with a 256 MiB BAR1 window; `use_pci_p2pdma` is false. The topology also separates the two devices: the GPU sits on the CPU's root port (`00:01.0`), the NVMe behind the chipset (`00:1c.4`), the same chipset port that holds the drive at PCIe 3.0 x4 (13.55). So a true DMA from the drive into VRAM is not available here. cuFile's compat mode only reads through bounce buffers in host RAM, which is what Strata's staged read already does (NVMe to pinned staging buffers to VRAM, T1), so there is nothing to gain from it. "No CPU on the path" does not mean no RAM: the cold experts pass through pinned RAM either way; the RAM tier (hot experts held in pinned RAM) and the staged path are the realistic form of the idea, and item 2 above (RAM-tier mode without the throttle) is its test. On a machine with a data-centre GPU and the drive on the same root complex this item would be worth redoing. **What stands in for GDS here (the user's remark: "GDS was not supported so we built something similar"):** solid.cpp's explicit `pread()` of a whole expert into a pinned host staging buffer, then the copy to the GPU (made the default, `EXPLICIT_READ`; solid's plan notes record no cuFile/GDS code at all), and Strata's staged read (T1, +60%). The measured chain NVMe to pinned RAM to VRAM reaches 3.56 GB/s with 8 reads in flight against the drive's own ceiling of 3.57 GB/s (13.55), so the substitute already runs at the limit of the link; the limit is the drive's PCIe 3.0 x4 chipset port, not the route, and a true direct path could not be faster on this machine.
  8. **Is there spare drive bandwidth once stand-ins are used? (user, 2026-10-10: "send directly to VRAM instead of via RAM"; MEASURED from the engine's per-request logs, benchmark requests of 100+ tokens, whole-request time, prompt included, so the decode-only rate is not isolated).** Drive read rate, median / 90th percentile / maximum: control 2.31 / 2.57 / 2.84 GB/s (76 requests, 258 MB read per generated token); ranks 2-9 3.00 / 3.20 / 3.34 GB/s (55 requests so far, 170 MB per token), against the drive's measured ceiling of 3.57 GB/s. Stand-ins cut the bytes read per token by a third, but the engine runs about twice as fast, so the drive is busier: about 65% of its ceiling for the control, about 84% (90th percentile 90%) at 2-9, leaving roughly 0.3 to 0.5 GB/s. RAM to VRAM over PCIe (24 GB/s) carries only about 3 GB/s, so the RAM bounce is not what limits; any direct path would share the same drive link and could save a copy and some latency, not bandwidth. **What the data supports instead: a floor that follows the drive's load.** Replace fewer picks (fetch the exact expert) while the measured drive rate is below its ceiling, and more while it is saturated: solid's wait-or-substitute balance (its "stand-in quality bar"), driven by a measured quantity and not a fixed setting. To be built after the per-task floors (SPEC 17); the controller's input is the drive read rate over the last few windows (the engine already counts the OS bytes read), its output the floor within the per-task limits.
  9. **Prefill (user, 2026-10-10: "there was prefill for Strata too, is it not used?").** It is used: the config has `--prefill auto`; the log says "prompt chunk auto: 8192 tokens, a 96-slot ring", the prompt path borrows 1,781 GPU cache slots (3.41 GiB), and the CPU share is on by default for chunks below 1,024 tokens ("the idle CPU takes some of the experts the GPU would stream"). Measured: a prompt of about 105 tokens is read in 5.6 to 8 s, 13 to 18 tok/s (engine log, benchmark requests). Worked out, not measured: 105 tokens x 10 picks is about 1,050 picks per layer over 512 experts, so about 87% of each layer's experts (about 446) are touched under uniform routing, which means most of the model's expert weights must be resident or streamed even for a short prompt; at the drive's 3.5 GB/s that alone would be far longer than 8 s, so the VRAM cache, the page cache and the CPU share already carry much of it. The time to the first token is therefore set by how many of those experts are not already in memory, and it did not change with the stand-in floor (7.9 s at 2-9 against 8.2 s control), since stand-ins act on decode only. To test, after the sweep: prefill speed against prompt length (100, 400, 1,600 tokens: longer prompts amortise the same expert reads, so tok/s should rise), with `STRATA_PREFILL_CPU_SHARE=0` against the default, and solid's overlapped-upload prefetch (+36% in its table) if the ring turns out not to overlap reads with compute.
  **What Strata's prefill can take from solid's (read from both code bases, 2026-10-10).** Solid's prefill wins: (a) expert prefetch with overlapped weight uploads, +36.0% (1403 to 1908 tok/s, pp2048, ncmoe 20, Gemma-4, from thecodacus' branch); (b) the native full-tensor prefetch double buffer (`GGML_SCHED_PREFETCH_EXPERTS`), +12.7% (952 to 1073 tok/s); (c) a disk prompt cache, a restart's re-prefill of a 3,623-token prompt 5,775 ms to 271 ms (21x). Solid's losses, not to take: the moe-cache prefill double buffer (`MOE_PREFILL_BUFFER`) -5% refuted; the device-to-device hit/miss split works but its buffer costs 5% (superseded); a smaller micro-batch helps decode but costs prefill (default stays); reserving prefetch slots up front costs 13 to 17% of the cache. **Strata already has the equivalent of (a) and (b) for long prompts:** from 1,024-token chunks its prompt path streams whole expert blobs through a ring (sized to the rig; its comments record about 1,000 tok/s on a 120K prompt on the 4-GPU rig, and `--prefill auto` picks the largest chunk that keeps the ring full). **The gap is the short prompt:** below 1,024 tokens (our benchmark prompts are about 105) Strata takes the CPU share and staging instead of the ring, and reads at 13 to 18 tok/s; solid has no result for that case either. So what is worth trying is not a port but a measurement: where the 5.6 to 8 s of a short prompt goes (expert reads against compute, per layer), the CPU share on and off, and a ring forced on for short prompts (`STRATA_PREFILL_RING`, and the chunk limit that turns it on) before deciding whether a layer-ahead prefetch is worth building. (c) is partly present: the engine has `SAVE` / `RESTORE` and reuses a prompt prefix inside a session ("29 reused + 7 read"); a disk cache that survives a restart is not verified here.
  **Prefill measured (2026-10-10; `tools/linkbench/prefill_probe.py`, one token asked for, a different slice of text per request, 3 per length, memory-capped, 2-9 default; prompt lengths as the server counted them).** Seconds to the first token (prompt tok/s):

  | Prompt tokens | PF0 default (CPU share on, staged) | PF1 CPU share off | PF2 CPU share off, ring forced for short prompts |
  |---|---|---|---|
  | 131 to 258 | 8.0 to 10.9 s (13 to 30) | 9.4 to 11.0 s (13 to 27) | 27.4 to 27.7 s (5 to 9) |
  | 431 to 565 | 13.4 to 15.1 s (29 to 37) | 15.1 to 17.8 s (26 to 32) | 27.8 to 28.1 s (15 to 20) |
  | 1,881 to 2,029 | 28.9 to 30.0 s (65 to 68) | 27.3 to 28.7 s (66 to 72) | 26.9 to 29.0 s (65 to 76) |

  Reading: (1) **The CPU share helps short prompts, as its comment says:** about 10 to 15% at 430 to 570 tokens (13.4 to 15.1 s against 15.1 to 17.8 s); no difference from 1,024 tokens up (it does not apply there); at 130 to 260 tokens the difference is inside the run-to-run spread. Keep it on. (2) **Forcing the ring for short prompts is much worse:** a flat 27 to 28 s up to 570 tokens, the time the ring needs to stream the model's experts once. At 1,900 to 2,000 tokens it is level with the staged path (65 to 76 against 65 to 68 tok/s), so the crossover is near there (one length tested; nothing between 570 and 1,880). (3) **The short-prompt time is a fixed floor plus a per-token part:** about 8 s at 131 tokens, 8.7 s at 258, 13.4 s at 466, 29 to 30 s at 1,900 to 2,000: roughly 7 to 8 s fixed plus 11 to 12 ms per token. The fixed floor is what a chat request waits for first and it is NOT explained yet; PF2 shows it is not "the ring is too slow". (4) Not done: where the fixed floor goes, per phase (`STRATA_PREFILL_TIMING=1`), and whether a layer-ahead prefetch of the experts a prompt will touch would shorten it. The lengths I asked for (100, 400, 1,600) came out at 131 to 258, 431 to 565, 1,881 to 2,029 (my characters-per-token estimate was off). The arms ran back to back on one install, so the spread between repeats (8.0 to 10.9 s at the shortest) is the noise to read the differences against.
  **Prefill port 1: solid's disk prompt cache (c), built on Strata's manual save/restore (user, 2026-10-10: "start with the Strata manual one, then the solid one improves it with requests"; `serve/prompt_cache.py`, config `prompt_cache_path`, on in `strata-iq3_s.json`).** MEASURED first, with the manual API (`/slots/0?action=save|restore`): a conversation's second turn in the same session read 16 of 2,083 tokens (0.71 s, against 22.6 s cold); after a server restart and a manual restore (0.6 s, 268 MB) its third turn read 16 of 2,103 (1.05 s). A prompt whose text CHANGES in the middle reused nothing, even in the same session (0 reused of 2,061): this model's recurrent state cannot be rewound to an arbitrary token, so a saved state helps only a prompt that continues it. BUILT: the same two calls, made by the server: after a request that READ at least 256 prompt tokens it saves what the engine holds (one file per prompt, the tokens in a small sidecar, written whole, the model/engine/arguments fingerprint in it, an eighth of the free disk as the limit, oldest dropped first, an older entry that a newer one continues is deleted); before a request it restores the longest entry whose tokens all start the new prompt, when that brings at least 256 tokens the engine does not hold. The engine checks the file again itself. RESULT (same conversation, no manual step): turn 1 cold 23.7 s and saved by itself in 0.33 s (255 MiB); after a restart turn 3 took 1.75 s (restored 2,066 tokens in 0.19 s), the same one-word reply as with the manual restore; an unrelated long prompt restored nothing and read normally (23.5 s), then saved too. Findings along the way: (1) the engine holds the prompt, every generated token including the stop token, and ONE more token the server is not told (counts 2,067 against 2,063+3, 1,551 against 1,546+4), so the entry records the tokens the server knows and a count more than two off drops the entry; (2) a save holds the request's end by 0.3 s (it runs before the stream closes; moving it after the last event is possible but not done); (3) a prompt that starts the same and then differs (a shared system prompt with different questions) gains nothing from the cache until the engine keeps a checkpoint at the end of the shared start; the engine keeps one checkpoint, "1 checkpoints" in its log; not tested. NOT yet tested: eviction under the size limit, a corrupt or foreign entry, two requests back to back with different long prompts, the cache on a loaded request path with tools or images (images are excluded).
  **Prefill port 2: many reads in flight for the prompt path (solid's (a), overlapped uploads, and (b), the double buffer; user: "port the features of a and b that are missing in Strata").** What Strata already had, read from its code and then MEASURED with `STRATA_PREFILL_TIMING=1`: from 1,024-token chunks every expert streams in a fixed order through a pinned ring ahead of the copy engine (this is (a)/(b)'s idea, built, and the reason the ring is not worth forcing on short prompts, PF2). What was missing is on the SOURCE side. The timing of a 1,845-token prompt: GPU timeline 25.6 s, 83.3% of it charged to "dequant" (the GPU waiting for the expert it needs, the timer charges a wait to the phase before it), "wait copy" 2.6%; the engine read 50 GB from the drive in 27.5 s = 1.8 GB/s against a 3.57 GB/s ceiling, through 35,756 major page faults and 1.1 GB of `pread`. The copy threads were the defaults, `max(2, min(4, cores/4))` threads and a 16-deep ring, chosen for a RAM source; the code's own comment records that a file read through page faults needs many reads in flight (32 threads and a 128-deep ring: 71 s to 29 s on another machine) but selects that profile only for the GGUF read in place. This pack's `experts.bin` is read the same way (a quarter or more of its blobs outside the RAM copy). MEASURED, same prompts, memory-capped, 2-9 default, 3 repeats (seconds to the first token): at about 2,000 tokens default 26.4 to 29.9 s, 16 threads + ring 64 21.4 to 21.8 s, 32 threads + ring 128 18.1 to 20.5 s (batch 8: 18.9 to 20.0 s; ring 256 + batch 8: 18.6 to 19.7 s); at about 3,800 to 4,600 tokens default 25.0 to 29.6 s, 32 + 128 21.3 to 21.9 s, with batch 8 20.9 to 21.7 s, ring 256 + batch 8 19.2 to 21.2 s; prompts under about 540 tokens do not use the ring and did not change in any arm. BUILT: the stager takes that profile (32 threads, 128-deep ring, batch 8) when more than a quarter of the blobs are outside the RAM copy (`src/prefill/prefill.cpp`; `STRATA_STAGER_SSD=0` or `STRATA_STAGER_THREADS` override). RESULT on the built binary, new default against the same binary with `STRATA_STAGER_SSD=0`: about 2,000 tokens 20.0 to 21.0 s against 29.4 to 30.2 s (-31%, 83 to 111 against 56 to 76 prompt tok/s); about 4,000 to 4,700 tokens 21.0 to 23.2 s against 30.5 to 31.4 s (-28%, 191 to 204 against 131 to 155 tok/s); 130 to 540 tokens unchanged (8.7 to 15.4 s both). The same conversation test as above gave the same replies word for word (`Gated`, `DeltaNet`, `Substitution`, `MoE`) with the cold turn 1 at 17.1 s (was 23.7 s). Reading: a long prompt now costs about one pass over the experts (about 20 s from 2,000 to 4,700 tokens), at about 2.5 GB/s of the drive's 3.57; what is left below that is not explained (the stager's copy from the mapping is still page-fault driven; reading with `pread` into the pinned buffers, or the `io prefetch` threads the decode path already has, is the next candidate and is NOT tried). **ONE PATH (user: "it should have been a single pipeline, not two"; solid has a routed path and a full-tensor path too, switched on batch size: `GGML_OP_OFFLOAD_MIN_BATCH` 32, full-tensor prefetch only when picks >= 2 x n_expert).** Strata's two schedules (the fixed-order ring from 1,024 tokens, routed-only below) were compared by forcing the routed-only schedule for every size (`STRATA_PREFILL_STREAM_MIN=1000000`, the reader at 32 threads in both). SPEED (3 repeats, seconds to the first token, ring against routed-only): about 1,700 to 2,200 tokens 19.5 to 20.3 against 17.3 to 18.6 (CPU share off: 17.8 to 18.8); about 4,100 to 4,700 21.9 to 22.5 against 21.5 to 21.6 (21.0 to 21.5); 7,420 tokens 23.2 against 24.2; 8,667 tokens 38.3 against 38.8 (2 repeats). So routed-only is 8 to 12% faster up to about 2,500 tokens, level at 4,000, and within 2 to 4% slower at 7,400 to 8,700; the ring buys nothing here. OUTPUT (six prompts of 1,674 to 2,465 tokens, greedy, 40 tokens, no prompt cache): 2 of 6 replies identical word for word, 4 differ in wording (different kernels: the engine's own comment says a smaller chunk takes other kernels); all 12 replies READ as correct, sound summaries of the same passages (for example the decayed-counter decision: "replacing solid.cpp's complex heat and demand metrics" against "rejecting solid.cpp's complex heat and demand mechanisms in favor of a unified unit"); no artefact. Not measured: accuracy on a benchmark through each path. DONE: the install config sets `STRATA_PREFILL_STREAM_MIN=1000000` (`strata-iq3_s.json` env), so every prompt on this install takes the routed-only schedule with the many-reads reader; the engine default (ring from 1,024 tokens) is left for the rigs it was built and measured on, and for prompts beyond what was tried here (the install's context is 32,768). From a plain start: 1,691 to 1,960 tokens in 16.8 to 16.9 s, 128 to 258 tokens 8.9 to 10.1 s. Solid's other lesson, calibrating these thresholds per machine, is NOT adopted; this install has the one value that was measured.
  **Router lookahead for the prompt path (user: "try the router lookahead for short prompts"; Strata's own decode-side mechanism, `RouterLookahead`, not solid's).** What it is: after each MoE layer's routing, the next layers' routers (two ahead, `STRATA_IO_PREFETCH_DEPTH`) are applied on the host to a sample of this layer's router input, and the experts they predict are warmed by the file tier's preads (8 I/O threads) while this layer's own experts are read. It only warms pages: which experts are computed, and how, is untouched. What had to change: the lookahead took at most 8 tokens a prediction (a decode window) and could not see a prompt; it now takes a settable number (`set_max_tokens`, dots run 8 tokens at a time), and `Prefill::set_lookahead` copies an evenly spaced sample of the layer's router input (bf16, `m.mixed_bf`) to the host behind the routing ids (`STRATA_PREFILL_LOOKAHEAD_TOK`, 0 = off). The first build changed nothing: the server's prompt path is a second hookup (`generate.cpp`, the serve branch) that I had not patched (the log line "the prompt path feeds the router lookahead" was missing, which is how I found it). MEASURED, seconds to the first token, 3 repeats, ranks 2-9, routed-only path, same binary: sample of 32 tokens: 7.3 to 8.6 s at 150 to 258 tokens, 10.8 to 12.0 s at 426 to 500, 16.2 to 17.8 s at about 1,600 to 2,100; sample 128: 6.9 to 7.6, 10.1 to 10.7, 15.1 to 17.3; sample 512: 3.8 to 5.8, 7.5 to 10.5, 13.9 to 16.3; lookahead off: 8.4 to 10.8, 13.2 to 15.0, 17.7 to 19.2. Depth 3 or 4 with 128 tokens gave no more than depth 2. Reading: about 45 to 55% less at 150 to 260 tokens, 30 to 45% at 430 to 500, 15 to 25% at 1,600 to 2,100, against the same binary with it off. NOT proven: that it leaves the answers as they were. Six long prompts, lookahead off against on, 3 identical and 3 differing in wording; the CPU share (which chooses experts by measured timing) is the suspect and was not isolated, since the confirming run with the share off was stopped. By construction it only warms pages. The install config sets `STRATA_PREFILL_LOOKAHEAD_TOK=512`; the engine default stays off. The very short end repeats improve (5.8, 4.5, 3.8 s): part of that is the page cache warming across requests, so a cold first request is nearer the first figure.
  **How good the lookahead's prediction is, and for which layers (MEASURED, `STRATA_PREFILL_LOOKAHEAD_STATS=1`, 7 prompts of 150 to 2,150 tokens, sample 512, depth 2).** For each layer the prompt routed, of the experts that were not on the GPU, the share the lookahead's set held: predicted from ONE layer back (+1) 79.5 to 86.6% (46 or 47 of 48 layers had a prediction), from TWO layers back (+2) 75.4 to 82.7% (45 or 46 of 48), either 85.9 to 91.4%. So it runs for every layer but the first (nothing precedes layer 0 to predict it) and, for +2, the first two; a busy thread skipped at most one layer per prompt. Each layer's prediction comes from the layer BEFORE it (and the one before that), never from its own input. Where its output goes today: (1) `warm()`, the file tier's preads into the page cache, always; (2) the Foresight swap space (`fs->predict`, copies the predicted experts into VRAM ahead of the layer), ONLY when `STRATA_FS_SLOTS` > 0 and for +1 only; it is OFF in this install and untested here. It does NOT feed the swap round (the decayed usage count, 96 swaps per 4 windows) and does not touch the stand-in.
  NOT done for the SHORT prompt (under 1,024 tokens, 8 to 15 s): it does not use the ring and its reads are the same page faults; its fixed floor is about the bytes of experts it touches (23 GB in 9.3 s at 122 tokens, 2.5 GB/s), so a lookahead read for the staged path is the candidate there. (Done since: the lookahead now feeds the prompt path, above.)

## 20. solid.cpp's page, read in full and checked point by point (user, 2026-10-10; the live page is byte-identical to `docs/index.html`, 167,625 bytes; read as text, not rendered: no browser here)

What the page says about the part you linked (the resident set per layer, the lookahead, the decode swap), and what Strata has or measured:
1. **Lookahead and prerouter roles.** The lookahead (a graph node, exactly one layer of lead) is punctual and certain, so it ADMITS; the prerouter is late and lossy, so it RETAINS (eviction protection 71.41 against 56.59 tok/s as an admitter, gemma probe regime). Admission needs somewhere to land: the prediction ring, where a prediction may displace only an older prediction, never a resident; the evicting variant measured -6.55 points and was reverted. The ring itself measured a tie (+2% +-1) on gemma at 64k; warming features "can only fill free slots", and a full pool leaves them nowhere. Strata: the lookahead exists and warms pages; its Foresight swap space (VRAM slots per layer on a separate copy stream) is the analogue of the ring; both are OFF/idle in decode here (item 9). No prerouter (no trained predictor) in Strata.
2. **The lookahead's accuracy.** The page: stale hidden state caps the next-layer prediction at 59.3%; Qwen's n-gram (PLE) embedding is known exactly in advance and was the most valuable input (70.00 to 49.88 tok/s when removed). Strata (MEASURED today, prompt path, 7 prompts): non-resident routed experts found in the +1 prediction 79.5 to 86.6%, +2 75.4 to 82.7%, either 85.9 to 91.4%. Untested: adding the next layer's PLE term to the lookahead's input (the page's own best lever).
3. **The lookahead measured as a loss in decode on gemma-4**: depth 2 -6% +-2 (5 of 24 requests faster), hit rate lower every round; "lookahead boundary at 8" (the shipped 8-token limit) beat the real verify width on hit rate every round; the earlier lookahead prefetch row is +1.1 points of hit rate (88.3 to 89.4%). Not measured on Qwen cleanly ("being measured now"). Strata's decode already takes 8 rows (`RouterLookahead` default); the prompt path takes up to 512 (measured above, helps).
4. **+41% "buffer fed by prediction"** is against a heat-fed buffer that was itself below no buffer (5.26 against 8.97 tok/s): a signal-quality result, "less wrong" (also 13.36).
5. **+65% frozen cache.** A regression from solid's own live budget re-check (`be9c73134`, a 2 s re-check of free VRAM) fixed by `e77dfcf7d` (budget never below what is allocated: the cache went inert, 37.2 to 61.3 tok/s). Strata has no `moe_cache` code (it vendors only llama.cpp's quant layouts) and sizes its cache once; symptom test run today: decode 42.2 tok/s before and 43.0 after two ~2,400-token prompts, hit counters still rising.
6. **Router bias against substitution (Cache-Conditional Experts).** The page: substitution never moves the hit rate; a bias (1+beta) on resident experts' selection scores does (gemma 0.85 to 0.92, Qwen 0.26 to 0.33), exact computation of what runs; bias plus substitution never better than bias alone. Strata decided NO bias (13.22). Recorded for the user's review, not acted on.
7. **Substitution findings that match Strata's own:** rank 0 degenerates (page: "a question to how a steps explaining..." ; Strata: loops), rank 2 is the working floor on Qwen (page 11.53 against rank 1 9.23 tok/s; Strata 2-9), a stand-in quality bar that passed short probes failed with reasoning on (empty answer, a bare `</think>`: Strata saw leaked tags and empty or tool-call replies at 2-9 with THINKING OFF; thinking-on is untested here), and lexical degeneracy is the only gate that ever caught a real bug (Strata: artefact scan plus reading).
8. **Misses.** The page's 2 Oct finding: an unsubstituted miss went to the CPU while its fetch was already queued (48 CPU-seconds in 9.7 s wall); the fix lets the row wait for the fetch (`SYNC_FETCH`, off, 10 tok/s still limited elsewhere). Strata (MEASURED today, `STRATA_DECODE_TIMING=1`, speed prompts, 2-9): per layer-window about 0.8 to 3.0 experts go to the CPU (about 2.3 typical), 21 to 37 are VRAM hits, PCIe 0.00; the per-layer host work is 10 to 64 ms of a 42 to 103 ms window (47 of 84 ms in the longest run), so the CPU miss path is the largest single cost left in decode.
9. **Foresight in decode, MEASURED today:** with `STRATA_FS_SLOTS=2` the engine reports, after every request, "0 experts served from the swap space, 0 copies (0 from the router look-ahead)", and decode speed is identical (45.0 tok/s). It is wired (`expert_pool_dispatch_multi` calls the lookahead; `set_foresight` is called) yet never copies: the reason is NOT found yet.
10. **Per-layer resident budgets (13.17 to 13.35).** Floor 10 per layer, budgets from router-request unmet demand, misses kept homogeneous across layers (cap about 2x equal share), live rebalancing within floor, cap and decay, carry-over at request boundaries. The page's pools are per tensor shape across layers (layers compete); its own shallow-layer bonus tied at 2x and lost at 4x.
11. **Already in Strata, checked:** prompt cache on disk (built today), prefetch for prompts (ring plus today's reader and lookahead), expert cache with a ranked start profile, MTP drafting (`--mtp`, a native pack REQUIRES `--spec`), draft depth 4 and gate 0.5 (13.42), int8 KV, an n-gram or suffix drafter beside the MTP head, whole-blob preads on 8 threads (`LIVE_PREFETCH`'s analogue), `POSIX_FADV_RANDOM` on the expert files (the page's read-around bug), draft-vocab trimming (`--draft-vocab en`), neuron concentration measured (about 81% median width for 90% mass, so no sparse path), no atlas.
9a. **WHY FORESIGHT NEVER COPIED IN DECODE (found, 2026-10-10).** `ForesightSwap::request_locked` returns at once unless `src->pinned(layer, expert)` (the async copy needs the pinned arena), and the PCIe miss path in `expert_pool_dispatch_multi` takes the same test (`d.src->pinned(...) ? blob : nullptr`). `FileExpertSource::pinned()` is true only when the pinned RAM copy of the experts outside the GPU cache (the "complement", built by `--resident-experts` / `--resident-budget-gib N`) exists. This install runs `--mmap-experts` with no RAM tier ("RAM 0 blobs" in every request), so NO expert is pinned: every GPU miss goes to the CPU, PCIe stays 0.00, and Foresight and the lookahead's slot feed are dead code. With the RAM tier on they work. MEASURED (decode speed prompts, 3 repeats, 2-9, memory-capped, `STRATA_DECODE_TIMING=1`): RAM tier 12 GiB (RB1) median 40.5 tok/s, per layer-window CPU 0.67 to 1.45 experts and PCIe 0.15 to 0.31 (default install: CPU 0.84 to 3.0, PCIe 0.00, median 44.8 to 45.0); with Foresight 2 slots per layer (RB2) 40.6 tok/s, 28,618 copies (27,444 of them fed by the router lookahead), 4,054 experts served from the swap space, 4,032 not landed in time, 273,724 refused for no free slot, 20,773 over the copy budget; RAM tier 20 GiB (RB3) 39.2, with Foresight (RB4) 39.8. Reading: (1) the lookahead DOES feed the swap space when something is pinned, as designed; (2) the swap space served about 3% of the non-resident picks (copies are wasted 86%: slots are reusable only `depth` = 4 windows after a use, 2 per layer); (3) the RAM tier makes the host cheaper (per-layer host work 9 to 22 ms against 17 to 64) yet decode is no faster than the default, because the GPU's own wait (about 26 to 28 ms a window) is the floor and the pinned copy takes page cache from the rest; (4) so Foresight is not worth switching on as it stands; more slots cost VRAM (about 456 MiB are free after load; a slot is about 2.2 MB, 48 layers).
9b. **Results of the spec items run on 2026-10-10 (all decode speed prompts, 3 repeats, 2-9, memory-capped, one arm per start; the default install measured 44.8 to 45.0 tok/s, the spread between identical runs is about 3).** (a) **MTP (the page's +28.1%):** without `--mtp` (`--spec 4` stays: a native pack refuses to start without `--spec`) median 33.3 against 44.8: +34% with MTP (reasoning 35 to 59, code 33 to 45, prose 38 to 44). (b) **Swap budget (T3, 13.37):** `--adapt-swaps 48` 45.3, 192 43.7; `--adapt-every 2` 42.3, 8 47.3; every window with 24 swaps (the spec's no-code test of the reserved-region idea, 19.5) 40.3. None clears the spread above the default 96 swaps per 4 windows; only the finest rounds lose clearly. (c) **`--pcie-frac` with the RAM tier on (the PCIe path needs pinned experts, 9a):** 0.8 gave 41.5 and 1.0 gave 41.7 against 40.5 at the default 0.55: no difference. (d) **Thinking ON (the page's failure case: a stand-in bar that passed short probes, then served an empty answer and a bare `</think>` once reasoning was on):** eight short prompts, control against 2-9, greedy, one run each, whole replies READ: all 16 replies finished normally with a non-empty, sound answer ("hi", "thanks!", "2 + 2", a string-reversal function, the sky in two sentences, a train catch-up problem answered 7pm by both, three primes above 50, Romeo and Juliet in three sentences); no empty content, no stray `</think>`, no tool call. The page's failure did NOT reproduce in this sample; the sample is small (eight prompts, brief reasoning of 90 to 560 characters, one repeat) and heavy reasoning is untested. (e) **A 256-slot expert cache** (not a true cache-off; stand-in quality not read): 20.3 tok/s against 44.8 (the page's cache gain on its own model is +31.3%; here the full cache is +120% against a small one). (f) **Swap grace period (T6, 13.36) and kept-rows counting (T5, 13.40):** BUILT behind `STRATA_SWAP_GRACE=N` (rounds, 0 = off) and `STRATA_COUNT_KEPT=1`, default off, not measured yet (see the results that follow if they were run). (g) **PLE term for the lookahead input:** not applicable: Strata applies PLE as one block at layer 1 (`verify.cpp:850`, `prefill.cpp:2042`), not a per-layer slice.
9c. **More results (2026-10-10, same method; the rebuilt control measured 44.6).** (a) **The speed cost of a higher stand-in floor** (does a floor that follows the drive's load, 19.8, have room?): 2-9 44.8, 3-9 39.6, 4-9 29.6, 6-9 25.5, 9-9 21.2. Raising the floor costs speed at once, because the exact fetch goes down the CPU path (9a); a floor that rose when the drive is idle would LOSE speed, so 19.8's premise does not hold in this install (it could in a RAM-tier install where a miss can be fetched over PCIe: untested). (b) **Prompt near the window:** the install's context is 32,768 tokens; a document of about 45,000 tokens was refused at once with HTTP 400 (clean refusal, as the page says is right); a 23,656-token prompt completed in 60 s (394 tok/s) and the server answered a short request afterwards. (A 30,000-token prompt: see the entry that follows, if it ran.) (c) **Second control (HumanEval, 30 problems, no stand-in, no length cap): 30 of 30**, as the first control (BC 30 of 30). The 2-9 runs scored 30 (B2) and 29 (R1); with other floors or edges 28 to 30. So the control has not failed a problem in 60; the 2-9 family failed 6 of 150 (not significant, but not zero): 2-9 may cost about one problem in 30, and the earlier "within noise" reading is a screen, not a proof. (d) **Foresight with a larger budget and a shorter reuse (RB6: 2 slots, depth 2, 64 copies a window):** median 40.6; 54,565 copies, 4,585 served (8%), 243,716 refused for no free slot; no gain. The four-slot arm (RB5) was lost to the runner's parser (fixed) and is queued again. (e) **Swap grace period (T6) and kept-rows counting (T5), rebuilt binary, same prompts:** control 44.6, hit rate 94.33%; grace 1 round 45.7 (94.19%); grace 3 rounds 40.2 (94.27%); kept rows only 42.9 (94.38%). Neither clears the spread and the hit rate does not move (0.2 points either way); both stay off. (f) **Per-layer misses (13.19, 13.25), the new `LAYERS` trace, shipped profile, 2,721 slots:** the unmet share of the router's requests per layer: min 0.001 (layer 35), median 0.013 to 0.020, max 0.161 to 0.164 at layer 0 (22 resident experts); layers 1 (35 resident), 2 (41), 22 (44), 38 and 6 hold 7 to 11%. Resident experts per layer: min 22, median 57, max 73. So the misses are far from homogeneous and sit in the flat early layers (13.17's finding), which the frequency-ranked profile gives the fewest slots. Swaps per round in steady state: 34.7 on average (317 rounds), well under the cap of 96. (g) **`tools/layer_budget_profile.py`** builds a profile that moves budget toward the layers that miss most (floor 10, cap 2x the equal share); its arms are queued.
9d. **Per-layer budgets tested (13.19 to 13.25, 20.10; T7 in part; 2026-10-10).** `tools/layer_budget_profile.py` took the measured per-layer unmet shares and resident counts (the shipped profile, 2,721 slots) and re-ranked the profile so layers that miss most hold more (weights = resident x (unmet / mean unmet)^beta, floor 10, cap 2x the equal share, ordering key (j + 0.5) / weight so the budgets hold for any cache size near N). Arms (decode speed prompts, 3 repeats, 2-9, each with `STRATA_TRACE_ADAPT`; the control re-measured in the same session): control (shipped profile) 43.3 tok/s, hit rate 94.08%, worst layer (layer 0, 22 resident) unmet 0.152 to 0.162, then layer 1 0.119 to 0.136, layer 22 0.093, layer 2 0.072 to 0.081; beta 0.25 (layer 0: 35 resident) 43.6 tok/s, hit rate 94.51% (+0.43 points), layer 0 unmet 0.095 to 0.115, resident per layer 35 to 72, swaps per round 37.1; beta 0.5 (layer 0: 54 resident, a few layers 76 to 87) 42.2 tok/s, hit rate 94.52% (+0.44), layer 0 unmet 0.082 to 0.084, next worst 0.035 to 0.046, swaps per round 38.4. Reading: moving slots toward the flat early layers halves the worst layer's unmet share (16% to 8 to 11%), raises the overall hit rate a little instead of lowering it (the frequency fill was not already optimal for hits), and costs no speed (within the spread of 3); this is 13.25's homogeneous-misses aim, reached with a profile alone, no engine change. Output READ (code, reasoning, prose, control against beta 0.5): all three coherent and correct (the train problem 11:36, 156 km from A, verification arithmetic checks in both); artefact scan clean. HumanEval accuracy of beta 0.25 is queued (the two controls scored 30 and 30). The profile is generated from a trace of THIS install's speed prompts (code, reasoning, prose), so it is topic-specific; 13.35 (carry over at request boundaries) and live rebalancing (T8) are not tested.
9e. **Foresight with four slots per layer cannot run on this card**: graph instantiation fails (`verify: instantiate: out of memory`) and requests return no tokens; two slots per layer is the limit here.
9f. **The rest of the ledger's untested rows, run 2026-10-10 (decode speed prompts, 3 repeats, 2-9; control 44.6, spread about 3).** T11 (draft window and gate): `--spec 3` 43.9, `--spec 5` 43.9 (the default 4 is not beaten); `--spec-min-p 0.3` 38.5 and 0.7 40.1 (both clearly below 0.5): default stands. T13 (KV precision at 32K with the draft loaded; the engine takes `fp16`, `int8`, `q4_0` or `k8v4`; a first arm with the wrong name `f16` never started): `fp16` 41.4, `k8v4` 45.9, `int8` 44.6: default stands (`q4_0` left out, the page excludes it by policy for a reasoning model). BP25 (HumanEval, 30 problems, the beta 0.25 per-layer-budget profile, 2-9): **28 of 30** (the failures: `/32`, the usual numeric root-finder, answered wrongly; `/24`, correct code that begins with a leaked `<tool_call>` token line, a syntax error: a degeneration artefact); two answers flagged by the artefact scan (`/24` `<tool_call>`, `/64` `<think></think>`), the same kind and rate as the other 2-9 arms; the controls: 30 and 30 flagged none. So the profile changes neither accuracy nor artefacts measurably; it halves the worst layer's unmet share and adds 0.4 points of hit rate. It is kept as an OPTION (`data/expert-profile-b25.bin`, built from this install's own trace), not made the default: a topic-specific profile, no speed gain. T14 (drive traffic at or under 17 MB per token): NOT isolated: the runner's 75 to 77 MB per generated token includes each request's prompt read (a short prompt alone reads about 20 GB), so decode-only traffic is unmeasured.
Ledger status after today (the ones this document did not already mark): T3 swap budget: no setting clears the spread, the default 96 per 4 stands (finer rounds lose); T5 kept rows only and T6 grace: built behind switches, no gain, off; T7 per-layer counts: partly tested (above), a gain in hit rate and evenness, none in speed or accuracy; T8 live rebalancing: untested; T9 and T17 the RAM tier: 12 GiB 40.5 and 20 GiB 39.2 tok/s against the default 45.0, no gain here, but it is what makes Foresight and the PCIe path possible (9a); T10 untested; T11 and T13 default stands; T12 untested; T14 not isolated; T2 untested. The HumanEval second control (BC2): 30 of 30.
9g. **Margin at the working floor 2-9 (user's pending item, from snapshot 2 of the page; decode speed prompts, 3 repeats, control 44.6 to 45.0):** margin 2 logits 40.9 tok/s, margin 4 43.7, margin 8 43.6. No speed gain (margin 2 is lower); the artefact scan found nothing in the three arms and the answers read sound (the merge-intervals function; the trains problem 11:36 and 156 km; a fluent lighthouse story). A margin does not help at 2-9 on this sample; a quality gain, if any, is below what 3 samples a kind can show (the benchmark's own noise is about 2 problems in 30), so it is not pursued further unless asked. **Dropped by the user (2026-10-10): the calibration run (`tools/calibrate.py`) and auto-calibration; they are not required, as in solid.** Decode-only drive traffic (requests with at least 20 prompt tokens reused, at most 20 read, 100+ generated; 402 such requests in the log): 54 to 80 MB read from storage per generated token, only 50 to 680 major page faults a request (a long prompt: 11,000 to 35,000), so decode reads are explicit reads, not mapping faults: the page's read-around bug (4 MiB per fault) cannot be the cost there; in the prompt path the drive bytes per expert byte needed are 1.03. At 45 tok/s, 54 to 80 MB a token is 2.4 to 3.6 GB/s, 68 to 100% of the drive's 3.57 GB/s ceiling. The engine's "io prefetch ... used / unused" counters read 2% used, but they count stage-buffer hits only, not reads served from the page cache, so they do NOT show waste for the mapped tier.
9h. **Substitution with MTP, the lookahead, the per-layer budget profile and `--pcie-frac 1.0` (user request 2026-10-10; decode speed prompts, 3 repeats, floor 2-9, profile-B25, 16 GiB pinned RAM tier, memory-capped; default install 44.6 to 45.0, spread about 3):** PL1 (`--pcie-frac` default 0.55) 37.5 tok/s; PL2 (`--pcie-frac 1.0`) 39.2. Both below the default install, so the combination does not beat the plain mapped install here; 1.0 over 0.55 is +1.7, inside the spread. The log of PL2 shows the RAM tier and the prompt path's lookahead on, decode hit rate 95.4%, 10.5% of routed experts read by the GPU over PCIe; the system swapped during both arms (pswpout 154k and 92k pages): a 16 GiB pinned tier takes page cache the 83 GB model needs. Memory cost, from the arms' traces (default-profile arm PB25 against PL1 / PL2): lowest available memory 26.2 GB against 9.0 / 9.5 GB, pages swapped out 49k against 154k / 92k, major faults 45k against 95k / 78k, engine resident 23.9 against 26.3 / 26.7 GB: the 16 GiB pinned copy is unevictable and sits next to the page cache the mapped model needs, so the rest of the 83 GB is read from the drive about twice as often (solid.cpp measured the same: -12% for mlock pinning, and -56% for a dedicated host buffer). Quality, answers READ: PL1 and PL2 gave the same function (merge intervals, correct), the trains problem (11:36 and 156 km in PL1; PL2 worked to the same point and stopped at the 900-token speed cap), and a fluent lighthouse story; the artefact scan found nothing in either. Live rebalancing (13.34) is not built: not required for one model (user).
9i. **The staged PCIe miss path (item 20), the slowness found, and the look-ahead fill of substitute slots (2026-10-10; plain install, floor 2-9, single runs of 300 tokens, spread about 3; the same-length control 34 to 35 tok/s, the 3-repeat default 44.6 to 45.0).** Built (opt-in `STRATA_PCIE_STAGED=1`): a miss that is not in pinned RAM is read after the plan is published and copied by DMA, the GPU computes it. Result: 20.1 tok/s ungated, 19.0 at margin 4 (3 repeats each), 13 to 17 in the small tests; the answers were not read, the speed is the finding. Where the time goes (STRATA_SPLIT_TIMING, per window): the host waits for the GPU 170 to 199 ms against 28 ms in the plain install (pool and plan the same, 45 to 66 against 47 ms). GPU stage profile (STRATA_DECODE_TIMING + STRATA_VERIFY_PROFILE, staged run, per window): waiting for the staged experts to land (waitB) 130 to 230 ms in the GDN layers and 35 to 70 ms in the QSA layers; the PCIe-group kernel about 1 ms; the VRAM groups about 6 ms. So the window is about 85% waiting for the copy, not computing. Measured pieces: a staged read 100 to 560 us in the pool thread (parallel threads), the copy call 290 us through the file mapping and 6 us from an own pinned ring; the mapping route also read about 12 MB per major fault (kernel read-around), the ring route removed the faults but not the stall (13.0 against 16.7 tok/s, one run each). `MADV_RANDOM` alone is the known solid.cpp trap (every miss a 4 KiB read; 76a2a6896 needs a WILLNEED of the exact bytes with it): `STRATA_MADV_RANDOM=1` is opt-in and was not measured. Upstream v0.1.42 measured the opposite direction for the PCIe share on this card: 0.55 to 0.20 gave +30% (47.7 to 62.1 tok/s, IQ3_XXS), because the CPU pool outruns the link. Look-ahead fill without staged compute (user: use the read-ahead, fill the substitute slots, the CPU keeps the exact misses; `--pcie-frac 0`, Foresight slots read from the file tier by the filler): no slots 35.3, 2 slots per layer 31.8, 3 slots 31.4. Foresight counters (whole run): 5,222 copies for 1,699 uses (2 slots), 7,717 copies for 2,376 uses (3 slots), 122k to 153k predictions with no free slot, 32k to 45k over the copy budget: the swap space is too small for the 5 to 7 misses per layer-window and two thirds of its copies are never used, while they cost drive reads. Conclusion for this install: both new paths lose to the plain CPU path; the levers that upstream measured on the same card are the tail skip (+15.6%) and the measured PCIe share, to be tested on the merge branch.
9j. **The plain install is drive-bound, and the decode look-ahead is most of the drive traffic (2026-10-10; decode speed prompts, floor 2-9, 3 repeats x 3 kinds, 900 tokens, hard 29 GB cap, one arm per start; spread about 3).** The NVMe during a plain decode (/proc/diskstats, 0.25 s): median 3.35 GB/s, mean 3.0, max 3.58 against the 3.57 GB/s ceiling, median 18 requests in flight. So the GPU waits on the drive; no path that fetches the same bytes can go faster, and the staged PCIe path (9i) ran at 2.1 GB/s. Read-ahead arms (tok/s median; drive MB per generated token for requests of 400+ tokens): default look-ahead (top 10 per token, depth 2) 44.8, 63.6 MB/token; no decode look-ahead (STRATA_LOOKAHEAD=0) 49.4, 2.8; top 4 per token (STRATA_LOOKAHEAD_K=4) 55.3, 19.4; depth 1 45.5, 47.5; top 2 per token (STRATA_LOOKAHEAD_K=floor, i.e. only the ranks above the floor) 55.0, about 9. The look-ahead reads experts for ranks 2-9 that the floor would replace by a resident stand-in anyway; narrowing it to the ranks the floor cannot replace keeps its use and drops the waste: +23%. Edge gate on the speed prompts: edge 1 logit 42.7, edge 2 logits 47.6 (no gain against 44.8; the old benchmark's E1 +20% does not reproduce here). Order effect of the page cache not yet ruled out (the control was run first; a control rerun at the end was dropped for time). Built, opt-in: STRATA_LOOKAHEAD_K=floor, STRATA_WANT_LOW=<w> (the swap round's demand counts a pick the floor replaces as w).
9k. **All the ideas together, without the staged path (2026-10-10; same method as 9j).** With the staged path added to the narrow look-ahead (K=floor, margin 4, STRATA_WANT_LOW=0, --pcie-frac 1.0): 17.0 tok/s (code 17.3, reasoning 18.4, prose 13.6 at the median repeat), although the decode hit rate was 100% and the drive read only 8 MB a token: the staged path's stall is the copy chain (waitB), not the drive. Without the staged path (narrow look-ahead, margin 4, STRATA_WANT_LOW=0, the per-layer profile): 57.5 tok/s, drive 9.9 MB a token, against 55.0 (K=floor alone) and 56.1 (plus WANT_LOW=0): +1 to +2, inside the spread, so the narrow look-ahead is the whole gain. `STRATA_LOOKAHEAD_K=floor` is now the DEFAULT whenever the stand-in floor is on (STRATA_LOOKAHEAD_K=10 restores the old breadth); it moves pages only, no output changes. The first repeat of each kind is slow (code 38, prose 34 to 38) and the later ones fast (code 63 to 72): the 3-repeat median mixes cache warm-up. The staged PCIe path and Foresight slot fill are dropped for this install. The per-layer budget profile on its own (earlier, default look-ahead): none 43.3, beta 0.25 43.6, beta 0.5 42.2 tok/s.
9l. **CORRECTION to 9j and 9k, and what the swap round counts (2026-10-10).** In the plain install the swap round counts the experts the layers COMPUTED ("decode windows counted as routed", `STRATA_ROUTE_RESIDENT_COUNT` unset): a non-resident pick of rank 2-9 is replaced by a resident stand-in before the host sees it, so it is never counted and never swapped in; rank 0-1 misses (computed exactly) are counted and can be swapped in; the stand-ins' own use counts for them, which already protects them from eviction. `STRATA_WANT_LOW` only acts in the other mode (`STRATA_ROUTE_RESIDENT_COUNT=1`, the router's wishes counted), so the LK2W0 and ALL2 arms that set it alone were NO-OPS: their 56.1 and 57.5 against 55.0 are run-to-run differences, not a measurement of the weighting. More frequent swap rounds, now that the narrow look-ahead leaves the drive free (3 repeats x 3 kinds): a round every window with 24 swaps (SE1) 57.8 tok/s, drive 9 MB a token, against 55.0 for the default every-4-windows round of 96 and the earlier loss of 40.3 in the drive-bound regime (SF1): finer rounds no longer lose.
9m. **Swap rounds sized to the window, and the quality swaps (2026-10-10; speed prompts, 3 repeats x 3 kinds, narrow look-ahead on, floor 2-9; spread about 3).** A round every window with 24 swaps 57.8 tok/s, every 2 windows with 48 swaps 58.0, every window with 64 swaps 58.0 (SE3), against 55.0 for every 4 windows with 96: finer rounds stopped losing (the 40.3 of SPEC 20 9b was the drive-bound regime); a bigger budget than about 24 a window adds nothing. Counting the router's wishes alone (STRATA_ROUTE_RESIDENT_COUNT=1, rank 2-9 weight 0) 56.9: the counting path costs nothing; with rank 2-9 weighted 0.5 (SE3W) 52.6, drive unchanged: swapping rank 2-9 experts in costs speed. SWAP2 (opt-in STRATA_SWAP2_MAX=n: the experts stand-ins replaced are counted by the kernel, and after the demand swaps the most replaced ones are swapped into slots whose own use is below 1.0, at most n a round): n=8 53.8 tok/s, n=32 46.5, against 58.0 for none: about 0.35 ms a swap even into cold slots, so the cost is the swap (copy, synchronisation), not what it evicts. Answers read for n=8 (9): correct merge-intervals code in all three, the trains problem right where it finished (11:36, 156 km) and verified in the others, fluent lighthouse prose; artefact scan clean. User (2026-10-10): the quality swaps show no gain in a handful of answers but pay in reliability; the measure for that is the share of picks a stand-in served (per layer-row, logged per request under STRATA_TRACE_ADAPT).
9n. **Quality swaps lower the stand-in rate; the confidence gate (2026-10-10; every-window rounds with 64 swaps, narrow look-ahead, 2 repeats x 3 kinds, STRATA_TRACE_ADAPT=1 counters; spread about 3).** The kernel's per-request line gives replaced picks per layer-row (of the 8 picks at ranks 2-9 a row can have replaced; more than half are) and the poor ones (stand-in at least 1 logit below the pick). Control (no quality swaps) 56.0 tok/s, 4.68 replaced and 1.006 poor per layer-row (mean of the last 4 requests); 2 quality swaps a round, every replaced pick counted (STRATA_SWAP2_MAX=2): 54.3 tok/s, 4.13 and 0.894 (poor -11%); the same with the confidence gate at 1 logit (STRATA_SWAP2_GAP=1, a replaced pick counts for a quality swap only when its stand-in is poor): 55.2 tok/s, 4.40 and 0.856 (poor -15%). Request by request the quality swaps are lower in every pair (poor -2% to -15%). The gate is not clearly better than the ungated count at 2 a round (inside the request-to-request variation); its point is the same effect with fewer swaps (1 a round not yet tested). An earlier measurement with the first counter (SE3T against SW2, 3 repeats): replaced 4.45 against 4.21 (-6%) at 57.2 against 55.0. Answers of the gated arm read (6): correct merge-intervals code twice, the trains problem right (156 km at 2.6 h, 11:36 in the one that finished) and verified, fluent lighthouse prose; artefact scan clean. Cost of the quality swaps by count (3 repeats, 58.0 with none): 2 a round about -2, 8 -4, 32 -11.5 tok/s.
9o. **The merged version, solid-wins plus upstream v0.1.42, on this install (2026-10-11; speed prompts, 3 repeats x 3 kinds, 900 tokens, hard 29 GB cap; spread about 3).** M0, our floor 2-9 with every default (upstream's tail skip 7 and its measured PCIe share on by default): 58.4 tok/s (code 44.4 / 62.0 / 67.7, reasoning 58.4 / 66.1 / 67.6, prose 38.8 / 44.7 / 46.8); M1, the same with `STRATA_ROUTE_TAIL_SKIP=0`: 56.8 (the tail skip does nothing under the floor: every non-resident pick from rank 2 on is already replaced by a resident stand-in, so there is nothing for it to skip); M2, no floor, upstream's way (tail skip 7, measured PCIe share, our swap-round and look-ahead defaults): 27.1 (code 27.1, reasoning 26.4, prose 24.4 to 36.7): the floor is worth about 2x on this install and upstream's levers cannot replace it. Before the merge the same defaults gave 53.6 (DEF2): the merge costs nothing. Upstream's measured PCIe share logic sets how many misses MAY go over PCIe (the CPU pool took 91 to 120 us per missed expert here, so shares of 0.35 to 0.45) but a miss goes over PCIe only from a pinned RAM arena, which the install has not; the install config now carries `--pcie-frac 0`, so the share logic is off. Answers of M0 read (9): correct merge-intervals code three times, the trains problem right (156 km, verified 156 + 144 = 300) in all three, fluent lighthouse prose three times.
9p. **Foresight retested in the new regime (2026-10-11; merged binary, all defaults, floor 2-9, `STRATA_FS_SLOTS=2` with `STRATA_PCIE_STAGED=1` so the filler may read unpinned experts; 3 repeats x 3 kinds, 900 tokens).** 35.2 tok/s (code 25.0 / 38.0 / 42.1, reasoning 35.2 / 37.3 / 39.9, prose 24.7 / 29.7 / 34.7) against 58.4 for the same defaults without it (M0): -40%, more than the -10% of the old regime (31.8 against 35.3). Drive unchanged at 8 MB a token. Counters of the run: 27,635 copies, 8,312 served from the swap space (30%), 6,553 not landed in time, 114,242 predictions with no free slot, 3,490 over the copy budget. The narrow look-ahead did not rescue it. Suspected, not shown: the filler thread's reads and copies compete with the CPU pool for cores, and 2 slots a layer take about 220 MiB of VRAM from the expert cache; only 30% of the copies are used. The look-ahead's own consumer, the page-cache warming for the CPU (narrowed to the floor), is the part that works (+23%, 9j). The staged GPU-compute path (9i) and Foresight both stay off for this install.
12. **Not in Strata, or not compared:** the empirical calibration search for placement, threads and fit margin (Strata ships `tools/calibrate.py`, not compared today); GreedyDual-Size (cost-aware) eviction; cross-depth-agreement eviction; adaptive draft depth (the page lost 4.2%); lazy KV commit; the reserve floor of 1,500 MiB for two concurrent ~30K-token prompts (Strata has about 456 MiB free after load at 32,768 context; a prompt near the full window is UNTESTED here); f16 recurrent state; the page's MTP-on measurement (+28.1%: not measured for Strata today, and without `--mtp` the native pack still needs `--spec`).

