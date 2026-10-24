# 22 Oct 2026: review, read-ahead work, system simulation - and what to run next

Everything on this branch was written on a PC with no GPU and no compiler. The Python changes are tested; the C++/CUDA changes are
reviewed but NOT compiled and NOT measured; every speed figure below is from a model (tools/sysim, tools/predictor), not from the box.
Build every backend a file touches before relying on it (AGENTS.md).

## What was done

1. **Repository review** (5 parallel reviewers, top findings checked against the code). Highlights: the prefill layer map (700 MiB) is
   sized for 1.4 MB experts, not the 2.33 MB measured since; reads start only after the router sync; per-window `std::thread` and a
   ~29 MB main-thread memcpy in the adaptive tier (`--adapt-async` is off); several faster CUDA kernels are opt-in only on AMD
   (`STRATA_MMVF_ROWS`, `STRATA_GDN_SPLIT`, the hit fusion). Full list: the review in this session; the GPU-side ones are listed below.
2. **Python server** (`serve/`): Nagle off on the SSE handler; the disk prompt cache's save runs after the reply's end is sent, on a
   thread that keeps the engine turn; the prompt cache's prefix scan by slices (60K tokens: 1.86 -> 0.47 ms). 848 existing tests pass
   on both the old and new code; `serve/test_prompt_cache_save.py` added (fails on the old code, passes on the new).
3. **Why predicted rank 0-1 experts arrive late** (code audit, docs/perf-ideas.md "22 Oct"): up to 8 guesses are read at once and share
   the drive with the decode's own reads, so even a right guess lands late; the newest guess is read first; layer 0 is never predicted;
   the trained predictor in tools/predictor was never wired in.
4. **Engine switches, all opt-in (unset = as before)**:
   - `STRATA_IO_ARRIVAL=1`: per request, each file-tier blob by what the read-ahead did for it (landed / still being read / still
     queued / predicted but not kept / never predicted), layers 0-1 and 2+.
   - `STRATA_IO_EDF=1`: decode guesses read next layer first, rank 0 first, stale ones dropped; the prompt's guesses untouched.
   - `STRATA_IO_PF_YIELD=1`: no guess starts while the decode's own reads run.
   - `STRATA_LA_MODEL=<file>`: the next-layer guess from the published linear predictor (`tools/predictor/export_la_model.py`
     writes the file; ~123 MiB of VRAM).
   - The existing `STRATA_IO_PF_THREADS` (default 8) already caps the guesses in flight: only 16 was ever tested, never fewer.
5. **Simulators**: `tools/predictor/sim_readahead.py` (read-ahead and drive), `tools/sysim/sysim.py` (every decode task on a generic
   machine: critical path, who waits on whom, utilisation, 1.5x what-ifs, ideal->real waterfall, Perfetto trace + HTML timeline),
   `tools/sysim/ideas.py` (five ideas from other fields), `tools/sysim/idle.py` (idle time per resource and its cause).
6. **Trace check**: `tools/predictor/promote_check.py` replays a router trace to measure how useful idle-drive RAM-tier promotions
   would be, net of the cold reads their evictions cause.

## What the model says (to be confirmed on the box)

| Change | Short context | Long conversation |
| --- | ---: | ---: |
| `STRATA_IO_PF_THREADS=1` (existing setting, no rebuild) | +3% | +17% |
| + `STRATA_IO_PF_YIELD=1` | +5% | +23% |
| + `STRATA_IO_EDF=1` + `STRATA_LA_MODEL` | +9-10% | +31% |
| cost-aware read rule (confidence >= 0.6) on top | +10% | +15% |
| better RAM-tier caching (25% fewer cold reads) on top | +12% | +25% |
| idle-drive promotions, 0.29 MB pieces, 30% useful | +8-13% | +10-11% |

Correction (23 Oct): the table above came from 3 seeds a step, too few (one seed's spread ~7 tok/s). The waterfall now uses 12
(rtx3060 profile, model): today 70.0 tok/s; PF_THREADS=1 73.9; + PF_YIELD 75.2; + EDF 75.6; + `STRATA_LA_MODEL` with
`STRATA_IO_PF_THREADS=3` 81.7 (+16.6%); + zero-copy reads 84.5 (+20.7%); + the two GPU kernel items 90.9 (+29.7%). With the trained
predictor, 3-6 guess threads beat 1 (16 seeds: 78.1 / 81.6 / 82.3 tok/s at 1 / 3 / 6).

Ideal -> today (model): GPU kernels below VRAM bandwidth -26%, graph-node cost -14%, missed guesses -9%, wrong guesses sharing the
drive -8%, host hand-off -5%. The two GPU terms rest on ASSUMED parameters (kernel efficiency 0.70, 3 us a node): only their sum
(~25 ms a window) is measured. Rejected by the model: longer look-ahead (3 layers), pacing guess starts.

## Next steps, in order

1. **Build** the branch (CUDA; HIP and SYCL too if touched - SYCL's copy of expert_source.cpp was not changed). Check the start log:
   no new line unless a new switch is set.
2. **Measure the GPU split** (calibrates the simulator's biggest terms): one run with `STRATA_DECODE_STAGE_TIMERS=1`, 8.5K speed run.
3. **Arrival counters**: baseline vs baseline + `STRATA_IO_ARRIVAL=1` (check the counters cost nothing), and read where the late
   blobs are: still being read, still queued, never predicted.
4. **No-rebuild A/B**: `STRATA_IO_PF_THREADS=1`, `=2` against 8, alternated pairs, 8.5K speed run and the 45K conversation,
   `STRATA_DECODE_TIMING=1`. Keep `=2` if `=1` loses drive throughput.
5. **The new switches**, one at a time on the best of step 4: `STRATA_IO_PF_YIELD=1`, then `STRATA_IO_EDF=1`.
6. **The trained predictor**: `python tools/predictor/export_la_model.py <models_linear dir> la_model.bin` (it prints the bf16 fold's
   fidelity), then `STRATA_LA_MODEL=la_model.bin`; check VRAM headroom in the start log; if the read-ahead counters look odd, compare
   with `STRATA_LA_EDGE_SKIP=0` (the edge skip compares predicted scores against a 1-logit margin, the model's scale may differ).
   The same file now also drives the CPU look-ahead, which makes the prompt's read-ahead guesses (NOT compiled yet; ~123 MB of RAM;
   the start log says "STRATA_LA_MODEL ... on the CPU look-ahead"). A/B against `STRATA_LA_MODEL_CPU=0` (prompt guesses from the
   routers, as before): 8.5K and 45K prompt read time, alternated pairs, and `STRATA_PREFILL_LOOKAHEAD_STATS=1` for how many guesses
   land (today ~1/3). The predictors were trained on decode tokens only; this measures whether they hold on prompt tokens.
7. **Trace checks** (offline, ~10 min each):
   `python tools/predictor/promote_check.py /mnt/nvme/strata-data/traces/trace1.bin --ram 450` and `--ram 4400`;
   >= ~30% useful: build idle-drive promotions in small pieces; <= ~10%: drop the idea.
   Before building the cost-aware rule: check whether the GPU look-ahead's confidence (route_pred_weights) separates right from wrong
   guesses on the trace - a confidence gate already measured "no net gain" once (perf-ideas, Tested - keep out).
8. **Re-run the simulator with the measured numbers** (`--set gpu_eff=... gpu_node_s=...`), then rank the GPU-side work:
   router + look-ahead fusion, HC norm fold, hit-add fusion on CUDA, `STRATA_MMVF_ROWS`, `STRATA_GDN_SPLIT`, `STRATA_EXPDEC`.
9. **Other review findings worth an A/B** (no code): `STRATA_PREFILL_LM_MAP_MIB=1200` (45K prompt) and `=0` (8.5K prompt);
   `--adapt-async 1`; `STRATA_EXCHANGE_ROTATE=1`; `STRATA_CPU_IQ_ALL_EXACT1=1`.
10. Record every result in docs/perf-ideas.md, as before: alternated pairs, the same binary, numbers with what they were measured on.

## How to run the simulators

    python tools/sysim/sysim.py --waterfall --html trace.html --trace trace.json   # ideal -> today, fixes, timeline
    python tools/sysim/ideas.py                                                     # the five ideas
    python tools/sysim/idle.py                                                      # idle time per resource
    python tools/predictor/sim_readahead.py                                         # read-ahead settings only

`--set key=value` changes any parameter (`python tools/sysim/sysim.py --list`); `--profile rtx4090-gen4` is a second machine.
`--seeds N` sets the waterfall's seeds a step (default 12). tools/sysim/decode_trace.html (open in a browser) and decode_trace.json
(ui.perfetto.dev) are that command's output for the rtx3060 profile after the fixes, as of 23 Oct.

## The Monitor's decode timeline (added later the same day)

Monitor tab, below "Experts in VRAM": a **Record** button. While on, the engine stamps HOST times only - each MoE layer's dispatch
(doorbell, plan published, quantise, demand reads, waits on reads in flight, CPU experts) and each read the I/O threads make - into
fixed rings, and writes the last 4 decode windows every 2 s from the adaptive tier's round (STRATA_TIMELINE_FILE, switched by
STRATA_TIMELINE_CTL; the server sets both; `POST /v1/monitor/timeline {"on": true|false}`). The CUDA graph is not touched, so the GPU
runs the same with it on or off; its work is drawn as the time between the host's hand-offs. Off: one relaxed load per layer and per
read. Tested: the server switch and snapshot (serve/test_timeline.py) and the page against a snapshot in the engine's format. NOT
compiled or run in the engine yet - after the build, check on the box:
1. Record on, one request: the panel fills within a few seconds; the window time it shows matches STRATA_DECODE_TIMING's.
2. Decode tok/s with Record on vs off (alternated pairs): the same within noise.
3. It needs the adaptive tier's rounds (the snapshot is written from them): with `--adapt-swaps 0` nothing is shown.

## Review rounds after that (same day)

- Fixed: the trace check's useful share (warm-up promotions were in its denominator) and its speed (a heap instead of a scan of the
  whole RAM tier per promotion); the timeline's demand-read count (only blobs actually read, not RAM-tier experts); a zoomed timeline
  pauses until Reset view; STRATA_LA_MODEL only on the GPU that loaded it; the API monitor's per-token copy under the status lock
  (650 -> 19 ms over an 80,000-token answer).
- Built, opt-in, NOT compiled: `STRATA_DIRECT_ZEROCOPY=1` (Linux O_DIRECT): an expert whose file offset and size are multiples of the
  drive's O_DIRECT alignment is read straight into its 4 KiB-aligned stage buffer, not into a bounce buffer and copied (2.33 MB of
  memcpy per read; the model's +4-6%). Alignment from statx (STATX_DIOALIGN), else 4096, or `STRATA_DIRECT_ALIGN=512`. With 4096 only
  about a quarter of the experts qualify (a 2,329,600-byte blob is 4,550 x 512 B, not a multiple of 4 KiB); 512 covers all of them if
  the drive reports it. A/B: alternated pairs, decode tok/s and "decode waited ... ms in ... fetches", answers read. Also A/B a long prompt with it on and off: in-place reads are not merged into the prompt path's
  large requests (up to 32 MiB), which may cost prompt speed. If the drive refuses an in-place read, the engine says so once and turns it off.
- Left as is (measured too small for the risk): the Responses API's per-token text growth (~40 ms over a 32K-token answer), the
  triple template render with effort_end (~14 ms a request on a 400-message chat).

## Correctness reviews (engine and server), late on 22 Oct

Fixed (engine NOT compiled - build first): a parked conversation saved with another conversation's K/V (wrong tokens, silent);
mid-prompt checkpoints under layer-major labelled with a chunk's tokens but holding the whole run's state (wrong tokens, default
settings - now skipped while layer-major reports; the better fix is to end the run's parts at those boundaries); hangs on PLE errors
and on released batch windows; split windows counted as new windows; a failed shared read failing a window; max_new overflow;
STRATA_PREFILL_LM_MAP_MIB=0 allocating 700 MiB instead of no map; early prompt exits leaving GPU work queued over refilled slots.
Server (855+ tests pass): an image-job deadlock in parallel mode, a second HTTP response written into a stream on mid-stream errors,
half-written prompt-cache files, Windows temp cleanup, a drive-relative /web/ read, and smaller items.

Deferred (need a build to do safely, or low value): layer-major's extra slot loan is lent even when layer-major cannot run (gate extra_p on the same
conditions as run_impl); STRATA_PF_GROUP_OVERLAP slot release across layer boundaries (opt-in); the drafter's K/V not caught up
after a request's last window (speed only); STOP cleared after a GEN line is read (not reachable with server.py); the pipelined
loop's usage race (swap choice only).

Installer review (setup.py, setup tests pass; test_setup_golden's 24 stale snapshots fail as before): --gguf-dir copies with a wrong
size or SHA-256 are no longer deleted (they are the user's files); a failed engine update no longer stops the model from starting;
the hipBLASLt table no longer replaces the env written before it; run and build .bat files work from a non-ASCII folder; --data-dir
inside the old folder no longer merges a folder into itself; a one-run fallback no longer forgets the remembered data folder; a
damaged HIP archive is dropped. Deferred: the RAM fit checks compare GiB from the OS with "ram_gb" - the values (44, 52, 62, 108,
118, 136) look like measured GiB needs, so no slack was added without a measurement on a PC at the boundary.

Conversation save/restore review (engine NOT compiled - build first): no wrong-token bug found in parks, checkpoints, retained K/V,
streaming or session files. Fixed: SAVE and RESTORE always failed without --mtp (the empty drafter state was passed as the draft
layer; now no draft layer, as parking already did); the session config fingerprint used the auto-resolved --kv-resident, so a
restart with less free RAM refused a compatible file (now the value as given). Left (speed only, every token is verified): the
draft cell at resume-1 is not recomputed; --batch without --batch-mtp leaves the main drafter with another conversation's K/V;
--kv k8v4 with a ring drafter never parks.

Speed work, 22 Oct evening (engine NOT built here; checked with a syntax-only compile, see below):
- Parallel requests without --batch-mtp: a request handed back from its slot to the solo path drafted against another
  conversation's draft K/V. Each slot now keeps the drafter's K/V from admission and every batch window's residual row and next
  token, and the hand-back replays them (docs/BATCHING.md; STRATA_SLOT_DRAFT_KEEP=0 turns it off). Measure: the DONE line's
  draft_accepted / draft_offered after a hand-back, with and without it.
- Simulator: the waterfall's 3 seeds were too few (one seed's spread ~7 tok/s) and showed STRATA_LA_MODEL as a loss. With 16 seeds
  (model, rtx3060 profile, PF_YIELD + EDF on): router reuse 76.5 +- 1.1 tok/s; trained predictor at STRATA_IO_PF_THREADS=1
  78.1 +- 1.7, =3 81.6 +- 1.9, =6 82.3 +- 1.9. The better predictor makes more next-layer guesses right with ~1.1 ms of lead, and
  one guess thread queues them. So step 6 above: try STRATA_LA_MODEL with STRATA_IO_PF_THREADS=3 as well as 1. The waterfall now
  uses 12 seeds (--seeds) and that step.
- Compile check without a GPU or compiler: clang 21 (pip `ziglang`) with NVIDIA's pip headers (cuda-runtime, cuda-nvcc,
  cuda-cccl) and clang's CUDA wrapper headers, -fsyntax-only, host and sm_86. Every file this branch changed compiles: generate.cpp,
  expert_source.cpp, verify.cpp, prefill.cpp, conversation_state.cpp and its test (Linux and Windows targets), native_router.cu and
  sampler.cu (CUDA). 188 of the CUDA tree's 223 sources pass; the rest need ggml, cuBLAS or HIP headers or per-file AVX flags,
  and moe_f16_common.cuh's bit_cast of half is refused by clang, not nvcc (not changed here). Not checked: HIP and SYCL builds,
  and nvcc itself.

## DeepInfer review, the DFlash 2 drafter and the first layers' experts (late 22 Oct; engine NOT built - syntax-checked only)

**What was compared.** deepseek-ai/DeepInfer (9bd9cf2; DeepSeek's serving engine, vLLM-based) against this branch. Most of what it
has is for many GPUs and many users at once (expert parallelism and EPLB replicas, context parallelism, the batch's verify budget);
the rest this engine already has (seeded sampling, the NaN guard, the sync guard, logit bias, the speculation cost model, prompt
checkpoints, the PLE n-gram embedding). Two things are not here: grammar-constrained decoding (serve/structured.py validates after
the fact) and a block drafter. DeepInfer's confidence-scheduled verification is a many-user feature: its own table
(doc/inference_deployment_qa.md) gives -1.5% at 16 sequences a GPU and +5.4% at 60, and solid.cpp's single-user version of the idea
(LLAMA_SPEC_GOVERNOR, ae03bf5) measured -4.2%. Not pursued.

**A block drafter exists for this model.** incoai/Qwen3.8-Flash-Next-DFlash2 (Inco AI, 19 Oct 2026; GGUF: incoai/Qwen3.8-Flash-Next-
DFlash2-GGUF, 362 MB at Q4_K_M). Neither Qwen nor ISTA-DASLab ship one: Qwen's checkpoint has only the one-layer MTP. Facts from its
config and card: 5 dense layers (hidden 2560, sliding window 2048, no experts), block 8 (tested at 4: SGLang's QSA limit), reads the
completed outputs of layers 5, 15, 25, 35, 45 averaged over the 4 hyper-connection streams (12,800 features a token; its SGLang
capture patch), keeps the top 16 candidates a position and a small selector picks one path. Inco's numbers (one GB300, temperature
1.0, 3 drafts a step, one request at a time), tokens per verify step MTP -> DFlash 2: GSM8K 3.08 -> 3.26, MATH-500 2.93 -> 3.17,
HumanEval 2.52 -> 2.94, MBPP 2.65 -> 3.06, MT-Bench 2.50 -> 2.83; throughput +13-24% over the MTP. Our MTP already makes ~3.3 tokens
a window greedy at --spec 4, so the gain here is unknown until measured. llama.cpp cannot drive this drafter yet (its card: no
stream mean for the draft).

**Decided order (user, 22 Oct):** measure before porting anything. The MTP stays the drafter until DFlash 2 beats it on this PC. If
it does, the MTP either goes (its ~0.9 GB to expert slots) or keeps one job: predicting the first layers' experts a window early,
which only matters if a token table cannot (below).

| Token table good enough for layers 0-1 | DFlash 2 beats the MTP | The MTP |
| --- | --- | --- |
| yes | yes | removed (DFlash 2 drafts, the table covers layers 0-1) |
| no | yes | kept for the layer 0-1 expert guesses only (heads on its hidden state, trained offline, MTP frozen) |
| either | no | stays the drafter |

**Built for that (opt-in, measurement only unless said; default path unchanged):**
1. `STRATA_LA_MODEL` on the CPU look-ahead too (src/core/expert_source.cpp `RouterLookahead::set_model`, `load_la_model_host`;
   generate.cpp). The prompt's read-ahead guesses (512 sampled tokens a layer) now take the trained next-layer predictor instead of
   the next layer's router when the file is given; two layers ahead stays on the router (the predictors look one ahead). ~123 MB of
   RAM. `STRATA_LA_MODEL_CPU=0` is the A/B arm (prompt on the routers as before, decode on the predictor). Step 6 above says how to
   measure it. Trained on decode tokens only: this checks whether it holds on prompt tokens.
2. Token records in `STRATA_ROUTER_TRACE` (`router_trace_tokens`, called where verify.cpp launches a window on the first stage):
   each window's token ids beside its routing. tools/predictor/trace_reader.py pairs them with the windows (in order, checking the
   row count) and gains `want_x=False`; older readers stop at the first token record. tools/predictor/token_table_check.py: does a
   table token id -> its usual experts find layers 0-3's rank 0-1 picks? Trains on the first 70% of windows, scores the rest, against
   a static most-used set; the reference is the predictor's 89.4% at K=4. Tested on synthetic traces (test_token_table_check.py).
3. `STRATA_DRAFT_TRACE=<file>` (verify.cpp; one GPU, single-sequence windows): every window's tokens, the model's picks and the
   residual after layers `STRATA_DRAFT_TRACE_LAYERS` (default 5,15,25,35,45) averaged over the streams, bf16 (25.6 KB a row). A
   device copy per traced layer inside the window graph, written by the host after the window; nothing is allocated without it.
   tools/drafter/dflash_offline.py rebuilds each request's text and features, prints today's tokens per generating window (MTP and
   lookup), and with the drafter replays DFlash 2 round by round as z-lab's dflash_generate does and prints its tokens per round at
   blocks 4 and 8. Greedy decoding is lossless, so the replay sees the text a DFlash run would write. Tested on synthetic traces
   (test_dflash_offline.py) and once end to end with z-lab's code on a tiny random DFlash 2 (CPU).

**To run on the box, in order:**
1. Build (CUDA). Start log, nothing set: no new lines.
2. `STRATA_IO_ARRIVAL=1` (step 3 above): how late layers 0-1's blobs are. If decode barely waits there, the layer 0-1 work stops.
3. Router trace with tokens (stand-ins off, as tools/predictor/README.md says), then
   `python tools/predictor/token_table_check.py <trace> --layers 0-3`.
4. Draft trace, greedy, one GPU, the prompt through the windows (no `--prefill`, so the prompt's features are traced), fresh
   requests (a resumed conversation's earlier positions are not in the trace):
   `STRATA_DRAFT_TRACE=/mnt/nvme/strata-data/traces/draft1.bin <engine command>`, the 44 prompts of tools/predictor/trace_prompts.py.
5. Download once: `hf download incoai/Qwen3.8-Flash-Next-DFlash2 --local-dir dflash2`;
   `hf download Qwen/Qwen3.8-Flash-Next model-00130-of-00131.safetensors model-00131-of-00131.safetensors --local-dir qwen-head`
   (the embedding and the head the drafter shares: 4.2 GB); `python -m pip install "dflash[local]" safetensors` (it pins
   transformers 5.15; older ones lack `DynamicCache.activate_past_recording`).
6. `python tools/drafter/dflash_offline.py draft1.bin --drafter dflash2 --head qwen-head --block 4 8`.
   DFlash 2 tokens per round clearly above today's tokens per window: port it (CUDA: the 5 layers, the 2-tap convolution, the
   selector; the stream-mean capture at the 5 layers; a switch beside --mtp). Not above: drop it.

**Assumptions to check on the first trace.** The residual captured is R after `post(l)` (the layer's gr_write), taken as "the completed
output of layer l", and R is read as 4 streams of 2560, stream-major. If DFlash 2's tokens per round come out near 1 (nothing
accepted), re-trace with `STRATA_DRAFT_TRACE_LAYERS=4,14,24,34,44` (the other reading of "completed") before concluding.

**Checked here (no GPU):** zig 0.16's clang (pip `ziglang`) with NVIDIA's pip headers, -fsyntax-only, Linux and Windows targets:
expert_source.cpp, verify.cpp and generate.cpp compile (generate.cpp's two Windows `getpid` errors are the same on the last commit).
Not checked: nvcc, HIP, SYCL (the SYCL tree has its own older look-ahead and was not changed), any run.

## The expert predictor as a setup option (23 Oct; opt-in, NOT measured on the engine)

Until now Aswin's predictors were never used: the engine installed on 19 Oct has no support for them, and on this branch
`STRATA_LA_MODEL` is opt-in, setup did not write it, and the file had to be made by hand. Now:

- `setup --setup --expert-predictor on` downloads the published predictors once (aswinkumar99/qwen3.8-flash-next-expert-traces at
  13969e68, `compact/heads/197/197_models_d48_linear.tar`, 774 MB, SHA-256 92d8a164...d2; CC-BY-4.0), converts them into
  `data/expert-predictor/la_model.bin` (~123 MB) and deletes the tar, then writes `STRATA_LA_MODEL` into the config's `env`.
  `--expert-predictor FILE` takes an export of one's own (a fine-tuned one, say); `off` removes it. A setup run again without the
  flag keeps the earlier choice. Offered for the original model's families (qwen, unsloth, nvfp4); Swift 1.5 and the Coder are left
  off with a note (other weights). Off by default: nothing changes for an install that does not ask.
- What the engine then does, with the install's stand-in rule (ranks 3-9 stand-ins, rank 2 within 1 logit): the look-ahead guesses
  ranks 0-2 of the next layer with the predictor instead of the router, in decode (the GPU look-ahead node) and while a prompt is
  read (the CPU look-ahead); two layers ahead stays on the router. The router still decides every expert: only speed can change.
- The tar holds `L00`..`L47_d1_linear_state.pt` and a `.json` per layer. `L00` is twice the size of the others: the dataset's table
  marks layer 0 as a cross-token cell (the previous token's state predicting layer 0), a candidate for the layer 0 gap above; the
  converter skips layer 0 as before.
- tools/predictor/export_la_model.py no longer needs torch: tools/predictor/pt_reader.py reads the .pt files with the same
  allow-list as torch.load(weights_only=True) and refuses anything else; the exporter takes the folder or the tar as published.
- Tests (no GPU): tools/predictor/test_export_la_model.py (the reader against torch's own files, the tar, the fold, a refused file),
  tools/test_setup_expert_predictor.py (the option end to end on setup's mocked install, the fetch and its SHA-256 check).
  The other setup tests fail the same way with and without this change (24 golden configs predate the Hot Rod env).

To test on the box: `START-HERE.bat --setup --expert-predictor on` (or setup.sh), then the A/B of step 6 (decode with and without,
`STRATA_IO_PF_THREADS` 1 and 3; prompt reading with `STRATA_LA_MODEL_CPU=0` and unset), alternated pairs, answers read.

## Review of the 23 Oct commit (f64249e1), the same day

Read in full and tested here (no GPU). The default path is unchanged: every addition waits on its switch (`STRATA_LA_MODEL`,
`STRATA_ROUTER_TRACE`, `STRATA_DRAFT_TRACE`, `--expert-predictor`), and the draft trace takes no arena memory when off.
- Checked by reading: the CPU look-ahead applies layer l's predictor to layer l - 1's MoE input (as trained), only for the next
  layer; the host loader reads the same LAM1 layout as the GPU one; the draft trace's buffer is sized and indexed by the same row
  count (max_t); setup pins the download's revision and SHA-256 and offers it only for the original weights' families.
- Fixed (d83a5f9f): tools/predictor/pt_reader.py's allow-list stops a file from running code, but it passed each tensor's size,
  stride and offset to numpy's as_strided, which does not check bounds; a cut or hostile .pt could read memory outside its storage.
  Such records are now refused, with a test that needs no torch.
- Tests: with torch 2.14 (CPU) the reader matches torch's own files (non-contiguous view, scalar, bf16, half) and the export test
  passes; token_table_check, dflash_offline and test_setup_expert_predictor pass; the setup suite fails only the 24 stale golden
  configs, as before. expert_source.cpp, verify.cpp and generate.cpp compile (clang -fsyntax-only, Linux and Windows; generate.cpp's
  two Windows getpid errors are older than this branch), as do native_router.cu and sampler.cu (CUDA, host and sm_86).
- Not tested: the real 774 MB predictor download and its conversion, nvcc, HIP, SYCL, any run on a GPU.

## Measured on the box, 24 Oct (incorporation into main)

Every C++ change above now compiles with nvcc. One build break (a raw line break inside a string literal in read_direct)
was fixed. The fixes went in first, then each feature on its own, with three prompts after each (code / reasoning / prose,
tok/s, every answer read). The stand-in budget was pinned at 0.2, so only the switch differed.

| Step | Result | Decision |
| --- | --- | --- |
| 13 correctness fixes | all answers right; serve tests 853/854 (NoLeakedSampler fails only inside the full suite, passes alone 3/3) | main |
| `STRATA_IO_ARRIVAL` | ~25,000 read ahead per answer, ~3,000 used | stays opt-in (counters) |
| `STRATA_IO_PF_THREADS` 8 -> 2 | 36.6-37.8 / 41.1-45.9 / 28.1-29.6 -> 39.5-39.9 / 49.2-50.6 / 30.6-32.3; drive wait 3.8-4.7 -> 2.8-3.3 s | **default 2** |
| `STRATA_IO_EDF` (with 2 threads) | 40.1-41.9 / 52.1-53.1 / 31.9-32.2 | **default on** |
| `STRATA_IO_PF_YIELD` | 39.2-41.5 / 49.6-49.9 / 29.8-31.6: no gain over 2 threads | opt-in |
| `STRATA_DIRECT_ZEROCOPY` | off 42.3 / 41.8 code, 52.8 / 52.0 reasoning; on 41.6 / 46.1, 53.6 / 54.4: +2-5% reasoning, the rest within noise | opt-in, re-test later |
| `STRATA_LA_MODEL` (la_model.bin 118 MiB, top-2 kept 99.7%) | three prompts: 43.6 / 53.1 / 31.5 vs control 43.0 / 52.7 / 31.6; guesses used +24% (2,592 vs 2,094); code's wait 2.4 -> 1.9 s. 45K conversation, 4 turns x 2 rounds: 31.8-47.8 vs control 33.6-47.9 tok/s, same wait | opt-in: better guesses, but after 2 threads + EDF the drive wait is ~10% of decode, too little left to win |
| Monitor decode timeline | in; the switch's write fixed for Windows (in place, 503 on failure) | main |
| DFlash 2, offline replay on the engine's own trace (greedy; the three prompts + a chat) | MTP + lookup 2.58 tokens per window; DFlash 2 block 4: 3.03 per round, block 8: **4.26** per round | port it |
| DFlash 2 as a Python helper process | the helper takes ~4.8 GB of the 12 GB card (bf16 drafter + its own embedding and head + torch): the engine is left 0.99 GiB, 61 expert slots, and does not start | native port instead: Q4_K_M drafter (362 MB), the engine's embedding and head shared |
| Window cost by size (the MTP drafting 4 / always 4 / always 8) | ~3.3-token windows 71-81 ms, 4-token 80-93, 8-token 129-156 ms: an 8-token window costs 1.8-2x, mostly compute (more experts per window), not drive wait (28 vs 31 ms) | block 8 cannot win here; test block 4 |
| DFlash 2 native (branch dflash-port: Q8_0 drafter in the engine, the engine's embedding and head, the MTP not loaded), block 4, live | check against bf16: 36 of 42 draft tokens equal (Q4_K_M 25 of 42), layer cos >= 0.997. Speed 22.1-26.8 / 33.5-34.1 / 19.9-20.2 against the MTP's 29.8-30.6 / 37.7-39.5 / 23.2-23.7: 2.0-3.15 tokens a round (about the MTP's) at 12.5 ms a propose | parked; the MTP stays the drafter |
| The MTP itself (with against without, --mtp dropped) | 34.8-38.4 / 41.2-45.6 / 30.8-31.4 against 23.8-25.8 / 28.7-31.6 / 26.0-28.7: +40-50% | the bar for any drafter |
| Token table for layers 0-3 (`token_table_check.py`, a router trace with tokens, 8,564 decode tokens, stand-ins off) | rank 0-1 experts in the table's top 4: layer 0 57.0%, 1 42.1%, 2 43.3%, 3 39.7% (68.7% of tokens seen) against the trained predictor's 89.4% | not enough: with DFlash 2 drafting, the MTP stays for the layer 0-1 expert guesses (row 2 of the table above) |
