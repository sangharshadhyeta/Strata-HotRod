# Citations and credits

Every project, paper, model and person this work draws on, so nothing is presented as original that is not. Where a role is
stated it comes from the author's own notes in this repository; where it says "cited as related work" the source is only
named in the notes. Figures and counts are as of November 2026.

## 1. Projects this work is built on

| Project | By | Licence | How it is used |
|---|---|---|---|
| [Strata](https://github.com/Niko1221/Strata) | Niko1221 and contributors | MIT | The engine, server and installer that Strata-HotRod modifies. Its copyright notice is kept in [`LICENSE`](LICENSE) |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml) | ggml-org and contributors | MIT | The base of solid.cpp, and parts used inside Strata (the picture encoder uses llama.cpp's `mtmd`). Licence: [`solid-cpp/LICENSE-llama.cpp`](solid-cpp/LICENSE-llama.cpp); authors: [`solid-cpp/AUTHORS-llama.cpp`](solid-cpp/AUTHORS-llama.cpp) |
| [Project Maya](https://github.com/mw00/project-maya) (maya-ds4) | Peasant Smith (mw00) and the Project Maya contributors, including needmorevram | MIT | The DeepSeek-V4.1 engine (`src/dsv41`, v1.1.17, commit 8d9a9b5d, with later releases v1.1.18 to v1.1.26 re-written onto Strata). Project Maya is itself built on Strata. Notice and licence: [`hotrod-deepseek/third_party/project-maya/`](hotrod-deepseek/third_party/project-maya) |

## 2. Prior art and code the ideas come from

| Project | What it contributed |
|---|---|
| [leloch/llama.cpp](https://github.com/leloch/llama.cpp) (moe-cache-pr) | The original GPU-resident MoE expert cache |
| [thecodacus/llama.cpp](https://github.com/thecodacus/llama.cpp) (prefetch-experts), from the [Codacus](https://www.youtube.com/@Codacus) channel | Expert prefetch overlapped with compute (the +36.0% prompt-processing result in solid.cpp is credited to it) |
| [JustVugg/colibri](https://github.com/JustVugg/colibri) | LFRU eviction and the Expert Atlas method; its [router-prediction experiment](https://github.com/JustVugg/colibri/issues/200) showed where a predictor stops paying |
| [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) | The hit/miss split for expert uploads |
| [antirez/ds4](https://github.com/antirez/ds4) (DwarfStar) | The three-tier layout and explicit reads instead of mmap |
| [vLLM](https://github.com/vllm-project/vllm) | A reference for the model's code, and not reserving memory for unused context |
| [aswinkumar99/qwen3.8-flash-next-expert-traces](https://huggingface.co/datasets/aswinkumar99/qwen3.8-flash-next-expert-traces) (CC-BY-4.0) | The expert-predictor study on this model: the learned look-ahead uses its training target, and `--expert-predictor on` downloads its published predictor. Attribution required by the licence |
| [SGLang](https://github.com/sgl-project/sglang), [oMLX](https://github.com/jundot/omlx), [llamafile](https://github.com/mozilla-ocho/llamafile) | Cited in the notes as related projects |

## 3. Papers

Titles and authors are as returned by arXiv. Three more works are named in the HotRod README without an identifier:
Stream2LLM (reading the prompt while it is written), ProMoE and PowerInfer-2 (expert prefetch), and FineMoE
([EuroSys 2026](https://intellisys.haow.us/assets/pdf/Hanfei_FineMoE_EuroSys26.pdf)).

| arXiv | Title | First author | What it was used for | Cited in |
|---|---|---|---|---|
| [2110.03742](https://arxiv.org/abs/2110.03742) | Beyond Distillation: Task-level Mixture-of-Experts for Efficient Inference | Sneha Kudugunta et al. (2021) | Cited as related work | hotrod-qwen/docs/DESIGN-planned-moe.md |
| [2303.06865](https://arxiv.org/abs/2303.06865) | FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU | Ying Sheng et al. (2023) | Layer-by-layer prompt reading (HotRod README) | hotrod-qwen/PRODUCT-README.md |
| [2308.12066](https://arxiv.org/abs/2308.12066) | Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference | Ranggi Hwang et al. (2023) | Cited as related work | hotrod-qwen/docs/DESIGN-planned-moe.md |
| [2309.06180](https://arxiv.org/abs/2309.06180) | Efficient Memory Management for Large Language Model Serving with PagedAttention | Woosuk Kwon et al. (2023) | vLLM / PagedAttention: read as a design reference, not copied | solid.cpp's original web page |
| [2310.17157](https://arxiv.org/abs/2310.17157) | Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time | Zichang Liu et al. (2023) | Cited as related work | solid-cpp/docs/plan.md |
| [2312.12456](https://arxiv.org/abs/2312.12456) | PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU | Yixin Song et al. (2023) | Cited as related work | solid-cpp/docs/plan.md |
| [2402.13516](https://arxiv.org/abs/2402.13516) | ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models | Chenyang Song et al. (2024) | Cited as related work | solid-cpp/docs/plan.md |
| [2409.12186](https://arxiv.org/abs/2409.12186) | Qwen2.5-Coder Technical Report | Binyuan Hui et al. (2024) | Cited as related work | solid-cpp/tools/server/README.md |
| [2410.19123](https://arxiv.org/abs/2410.19123) | Read-ME: Refactorizing LLMs as Router-Decoupled Mixture of Experts with System Co-Design | Ruisi Cai et al. (2024) | Cited as related work | hotrod-qwen/docs/DESIGN-planned-moe.md |
| [2411.01433](https://arxiv.org/abs/2411.01433) | HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference | Peng Tang et al. (2024) | Importance by cumulative gate mass, with a skip class (LEARNINGS L2; scoped) | solid.cpp's original web page |
| [2411.11217](https://arxiv.org/abs/2411.11217) | MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs | Shiyi Cao et al. (2024) | Layer-by-layer prompt reading (HotRod README) | hotrod-qwen/PRODUCT-README.md, solid.cpp's original web page |
| [2412.00099](https://arxiv.org/abs/2412.00099) | Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference | Andrii Skliar et al. (2024) | Router bias toward experts already on the card (solid.cpp LEARNINGS L1; implemented and tested) | solid.cpp's original web page |
| [2502.12224](https://arxiv.org/abs/2502.12224) | Fate: Fast Edge Inference of Mixture-of-Experts Models via Cross-Layer Gate | Zhiyuan Fang et al. (2025) | Cross-layer gate, cache and prefetch details (LEARNINGS L4); also cited as PILOT / Fate | hotrod-qwen/PRODUCT-README.md, solid.cpp's original web page |
| [2504.09345](https://arxiv.org/abs/2504.09345) | MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints | Yichao Yuan et al. (2025) | Cited as related work | solid.cpp's original web page |
| [2510.10302](https://arxiv.org/abs/2510.10302) | SP-MoE: Speculative Decoding and Prefetching for Accelerating MoE-based Model Inference | Liangkun Chen et al. (2025) | Expert-prefetch literature (HotRod README) | hotrod-qwen/PRODUCT-README.md, solid.cpp's original web page |
| [2511.06494](https://arxiv.org/abs/2511.06494) | Route Experts by Sequence, not by Token | Tiansheng Wen et al. (2025) | Cited as related work | hotrod-qwen/docs/DESIGN-planned-moe.md |
| [2511.10054](https://arxiv.org/abs/2511.10054) | BuddyMoE: Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference | Yun Wang et al. (2025) | Buddy-expert pairs as stand-ins (compared against solid.cpp's atlas-similarity stand-ins) | solid.cpp's original web page |
| [2511.14102](https://arxiv.org/abs/2511.14102) | Specter: Fast MoE Inference via Quantization-Enabled Multi-Step Speculative Offloading | Wenfeng Wang et al. (2025) | Adaptive draft length from a cost model (LEARNINGS L3). The solid.cpp notes call it MoE-SpeQ | solid.cpp's original web page |
| [2602.06036](https://arxiv.org/abs/2602.06036) | DFlash: Block Diffusion for Flash Speculative Decoding | Jian Chen et al. (2026) | DFlash, the basis of the DFlash 2 drafter tried in the dflash-port work | solid.cpp's original web page |
| [2603.19289](https://arxiv.org/abs/2603.19289) | Speculating Experts Accelerates Inference for Mixture-of-Experts | Vivan Madan et al. (2026) | Expert-prefetch literature (HotRod README) | hotrod-qwen/PRODUCT-README.md, solid.cpp's original web page |
| [2605.08575](https://arxiv.org/abs/2605.08575) | Uncovering Intra-expert Activation Sparsity for Efficient Mixture-of-Expert Model Execution | Jongseok Park et al. (2026) | Cited as related work | solid-cpp/docs/plan.md |
| [2605.23872](https://arxiv.org/abs/2605.23872) | Training-Free Looped Transformers | Lizhang Chen et al. (2026) | Looped layers: tested as the experimental STRATA_GDN_LOOP, kept off | - |
| [2605.27081](https://arxiv.org/abs/2605.27081) | ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference | Xiongwei Zhu et al. (2026) | Cited as related work | hotrod-qwen/docs/DESIGN-planned-moe.md |
| [2606.02091](https://arxiv.org/abs/2606.02091) | DFlare: Scaling Up Draft Capacity for Block Diffusion Speculative Decoding | Jiebin Zhang et al. (2026) | Cited as related work | solid.cpp's original web page |
| [2607.05147](https://arxiv.org/abs/2607.05147) | DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation | Xin Cheng et al. (2026) | The DSpark drafter used with DeepSeek-V4.1-Flash | solid.cpp's original web page |
| [2608.07911](https://arxiv.org/abs/2608.07911) | Reproducible Evaluation of MoE Expert Caching: Replay Semantics, Workload Contamination, and Operating Regimes | Yu Zhang (2026) | Cited as related work | - |
| [2608.12103](https://arxiv.org/abs/2608.12103) | Who Should Own the Expert Cache? Kernel-Managed Tiering for Trillion-Parameter MoE Inference | Yuan Si et al. (2026) | Cited as related work | hotrod-qwen/docs/DESIGN-planned-moe.md |
| [2608.18261](https://arxiv.org/abs/2608.18261) | Cacheable by Design? Training Mixture-of-Experts Routers for Locality Against the Edge Memory-Bandwidth Wall: A Pre-Registered Negative Result with a Systems Measurement Study | Shriniwas Ramesh Suram (2026) | Cited as related work | hotrod-qwen/docs/DESIGN-planned-moe.md |
| [2609.18110](https://arxiv.org/abs/2609.18110) | SSD-LLaMA: SSD-Native Inference for Trillion-Parameter MoE at 1+ Token/s on a Consumer PC | Fangzhou Liang et al. (2026) | Cited as related work | hotrod-qwen/docs/DESIGN-planned-moe.md, solid.cpp's original web page |

## 4. Models and data

| Model or data | By | Use |
|---|---|---|
| Qwen3.8-Flash-Next | the Qwen team | The model HotRod's Qwen build runs |
| [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | ISTA-DASLab | The compressed release used (IQ3_S, 78 GB) |
| Unsloth quantisations (UD-Q4_K_XL, UD-IQ4_XS, UD-IQ1_S) | Unsloth | Larger and smaller variants measured |
| DeepSeek-V4.1-Flash-1017 (GSQ-RCO IQ2_XXS backbone) | Starw1, with DeepSeek's own MXFP8 Engram | The DeepSeek model the port ran |
| DSpark drafter | arXiv 2607.05147 authors | The drafter used with DeepSeek |
| Gemma-4 26B-A4B, GLM-5.3-Flash | their publishers | Models measured in solid.cpp |

## 5. Benchmarks used in testing

Cited here as standard references; the notes name the benchmarks without links.

| Benchmark | Reference |
|---|---|
| HumanEval | Chen et al., Evaluating Large Language Models Trained on Code, [arXiv:2107.03374](https://arxiv.org/abs/2107.03374) |
| GSM8K | Cobbe et al., Training Verifiers to Solve Math Word Problems, [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) |
| MMLU-Pro | Wang et al., MMLU-Pro, [arXiv:2406.01574](https://arxiv.org/abs/2406.01574) |

## 6. Libraries bundled in llama.cpp's server (kept in `solid-cpp`)

[cpp-httplib](https://github.com/yhirose/cpp-httplib) · [stb](https://github.com/nothings/stb) ·
[nlohmann/json](https://github.com/nlohmann/json) · [miniaudio](https://github.com/mackron/miniaudio) ·
[subprocess.h](https://github.com/sheredom/subprocess.h) ·
[zod-to-json-schema](https://github.com/stefanterdell/zod-to-json-schema) · [OpenAI OpenAPI spec](https://github.com/openai/openai-openapi)

## 7. People

**Strata contributors** (public git author names, by number of commits in upstream Strata; Niko1221 is the author of the
project): Niko1221 (5973), maxfridbe (69), sergiywith (52), Jakub Luwierski (30), Laurent Zuijdwijk (30), havok2-htwo (29), Jeremiah Ritchey (20), Baptiste Rajaut (18), linxuhao (18), noon-at-cgn (18), Rafa (17), Andy (14), libratechw (14), David (13), Guillaume PUTIER (13), Francesco Albano (12), nekomario28 (12), ruibeikaa (12), Maurizio Verde (11), pipeob0 (11), xjc10 (11), JeanP00l (10), duncanmcqueen (10), BlueKingMuch (9), SuperDistance (9), pete (9), xrip (9), ATIVX928 (8), Andy Slater (8), Norihiro Nishimura (8). The full list is on the [contributors page of Strata](https://github.com/Niko1221/Strata/graphs/contributors).

**llama.cpp** contributors are listed in [`solid-cpp/AUTHORS-llama.cpp`](solid-cpp/AUTHORS-llama.cpp).

**Project Maya:** Peasant Smith (mw00), needmorevram and the Project Maya contributors, as named in their notice.

## 8. The author's own work

solid.cpp: about 540 commits, 13 August to 3 October 2026. Strata-HotRod: about 630 commits on the Qwen line and about 120 on
the DeepSeek line, 10 October to 15 November 2026. These counts are from the fork's history; the fork's history holds about
3,600 commits by Niko1221, and upstream Strata has grown since. The work was done with Claude Code, an AI coding assistant,
directed by the author, who set the decisions and read the results.
