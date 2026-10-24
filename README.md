<p align="center"><img src="docs/media/hotrod-hero.gif" width="380" alt="Strata Hot Rod: the Strata hexagon in fire and ice"></p>

<h1 align="center">Strata-HotRod</h1>

<p align="center"><b>A 125-billion-parameter model on a 12 GB gaming card.</b><br>
Qwen3.8-Flash-Next on one RTX 3060 with 30 GB of RAM. Nothing leaves your PC.</p>

<p align="center">
  <img alt="GPU: RTX 3060 12 GB" src="https://img.shields.io/badge/GPU-RTX%203060%2012%20GB-76b900?style=flat-square">
  <img alt="Model: 125B MoE" src="https://img.shields.io/badge/model-125B%20MoE%20%C2%B7%20IQ3__S-8a2be2?style=flat-square">
  <img alt="Context: 262K" src="https://img.shields.io/badge/context-262K%20tokens-0a7ea4?style=flat-square">
  <img alt="Decode: 50-70 tokens/s" src="https://img.shields.io/badge/decode-50--70%20tok%2Fs-e8590c?style=flat-square">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-555?style=flat-square"></a>
</p>

<p align="center">
  <a href="#how-fast">Speed</a> ·
  <a href="#what-you-see">The app</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#install">Install</a> ·
  <a href="#credits-and-license">Credits</a>
</p>

---

<div align="justify">

A fork of [Strata](https://github.com/Niko1221/Strata) by Niko1221 and contributors (MIT). Strata runs a model that normally
needs a server by keeping the most-used parts on the graphics card and streaming the rest from the SSD. This fork makes
that path faster and keeps answers correct while doing it.

</div>

> **Tested on one machine, with one model.** Other cards, AMD, Windows, macOS and other model sizes may work in upstream
> Strata; we have not run them, so we do not describe them here.

<table>
<tr>
<td align="center" width="25%"><h3>50-70</h3>tokens/s writing code, reasoning and prose</td>
<td align="center" width="25%"><h3>262K</h3>tokens of context on a 12 GB card</td>
<td align="center" width="25%"><h3>0.4 s</h3>to pick up a 45K-token conversation</td>
<td align="center" width="25%"><h3>85-88%</h3>of the needed experts guessed a layer ahead</td>
</tr>
</table>

## How fast

<div align="justify">

Measured on the machine below; a token is about ¾ of a word. Every answer in these runs was read and checked.

</div>

<table width="100%">
<tr><th align="left">Answer</th><th align="right">tokens/s (24 Oct, 131K context)</th></tr>
<tr><td align="left">Code (a Python function, 500-650 tokens)</td><td align="right">52 - 55</td></tr>
<tr><td align="left">Reasoning (a two-train problem, ~1,000 tokens)</td><td align="right">66 - 71</td></tr>
<tr><td align="left">Story (a paragraph, 100-150 tokens)</td><td align="right">50 - 53</td></tr>
</table>

<table width="100%">
<tr><th align="left">Long prompts (262K context, 20 Oct)</th><th align="right">Read in</th><th align="right">Answer written at</th></tr>
<tr><td align="left">8,554 tokens</td><td align="right">16.9 s</td><td align="right">52 - 54 tokens/s</td></tr>
<tr><td align="left">45,398 tokens</td><td align="right">67 - 68 s</td><td align="right">38 - 41 tokens/s</td></tr>
<tr><td align="left">The same 45K conversation, next turn</td><td align="right">0.4 - 0.5 s</td><td align="right">34 - 45 tokens/s</td></tr>
</table>

<div align="justify">

The first message of a chat is the slow part: it waits on the SSD for the model's experts. Follow-ups reuse what was
already read. A 262K context costs about 8% against 131K, because more of the attention's memory streams from RAM.

</div>

<details>
<summary><b>The machine</b></summary>

<table width="100%">
<tr><th align="left"></th><th align="left"></th></tr>
<tr><td align="left"><b>Graphics card</b></td><td align="left">NVIDIA GeForce RTX 3060, 12 GB (driver 610.57, CUDA 13.3)</td></tr>
<tr><td align="left"><b>Processor, RAM</b></td><td align="left">Intel Core i5-12400F (6 cores, 12 threads), 30 GB</td></tr>
<tr><td align="left"><b>Disk</b></td><td align="left">500 GB NVMe (WD Blue SN5000) on a PCIe 3.0 link; the model files live on it</td></tr>
<tr><td align="left"><b>System</b></td><td align="left">Rocky Linux 9.6, built with gcc-toolset-13</td></tr>
<tr><td align="left"><b>Engine</b></td><td align="left">Strata 0.1.45 (upstream, merged 17 Oct 2026) plus this fork's changes</td></tr>
<tr><td align="left"><b>Model</b></td><td align="left"><a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Qwen3.8-Flash-Next</a>, IQ3_S (about 3.5 bits per weight, 78 GB of files) by <a href="https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF">ISTA-DASLab</a></td></tr>
<tr><td align="left"><b>Context</b></td><td align="left">262K tokens (the longest prompt served so far: 114K)</td></tr>
</table>
</details>

<details>
<summary><b>Answer quality</b></summary>

<div align="justify">

Strata replaces some of a token's experts with similar ones already on the card to stay fast. In a small test (50 maths
questions, 30 coding questions, answers read by hand): 47 of 50 and 30 of 30 correct, against 48 of 50 and 30 of 30 with
nothing replaced. A test this small cannot show a difference of one or two answers either way. Since 24 Oct every JSON and
code answer in the structured-output checks has been valid. Detail: [SPEC.md](SPEC.md).

</div>
</details>

## What you see

<p align="center"><img src="docs/media/portal-monitor.png" width="900" alt="The Strata app's Monitor tab while the model is generating: 44 tokens per second, the RTX 3060 at 97% load, 11.7 of 12 GB of video memory, 2,793 experts on the card, and a table of recent requests with a 92 to 94% expert hit rate"><br>
<sub>The <b>Monitor</b>: speed, card load and memory, the experts on the card, each request's hit rate.</sub></p>

<p align="center"><img src="docs/media/portal-chat.png" width="900" alt="The Strata chat: a finished answer with '1,805 tokens, decode 55.1 tok/s' under it"><br>
<sub>The <b>Chat</b>: under every answer, the number of tokens and the speed.</sub></p>

## How it works

<div align="justify">

The model has 512 experts per layer and picks 10 for every token; only some fit on the card. Everything this fork adds is
about the ones that do not.

</div>

<table>
<tr>
<td width="50%" valign="top">

<div align="justify">

**Stand-in experts.** A missing expert is replaced by one already on the card only when
`router weight × logit gap ≤ budget`. The budget tunes itself by measured speed, a token a stand-in could have flipped is
recomputed exactly, and each request brakes itself when it sees flips.

</div>

</td>
<td width="50%" valign="top">

<div align="justify">

**A look-ahead that learns.** Experts for 1, 2 and 4 layers ahead are guessed on the card and read from the SSD before they
are needed. Each guess is the router's own score plus a small correction learned live from the real routing; its size is
chosen per layer by a live contest, and it falls back to the plain router if it ever does worse.

</div>

</td>
</tr>
<tr>
<td valign="top">

<div align="justify">

**Three tiers: card → RAM → SSD.** The hottest experts on the card, the next in a RAM tier sized from free memory, the
rest on the SSD. Swap rounds move a chat's experts onto the card in the background.

</div>

</td>
<td valign="top">

<div align="justify">

**Long prompts, long memory.** Prompts are read layer by layer, each layer's experts once, straight from the SSD.
Past 64K the attention's memory streams from RAM, and a long conversation resumes from a saved state in under a second.

</div>

</td>
</tr>
<tr>
<td valign="top">

<div align="justify">

**Draft and check.** A small draft head guesses the next words; the big model checks several at once.

</div>

</td>
<td valign="top">

<div align="justify">

**Nothing hand-set.** Budgets, reserves of RAM and video memory, and what the look-ahead learned are measured on your PC
and kept across restarts.

</div>

</td>
</tr>
</table>

<table width="100%">
<tr><th align="left">Needed experts guessed (live, 24 Oct)</th><th align="center">Plain router</th><th align="center">Learned look-ahead</th></tr>
<tr><td align="left">1 layer ahead</td><td align="center">77-79%</td><td align="center"><b>85-88%</b></td></tr>
<tr><td align="left">2 layers ahead</td><td align="center">66-69%</td><td align="center"><b>76-79%</b></td></tr>
<tr><td align="left">4 layers ahead</td><td align="center">55-58%</td><td align="center"><b>66-71%</b></td></tr>
</table>

<div align="justify">

Every change and its measurement: [docs/HOTROD-DETAILS.md](docs/HOTROD-DETAILS.md) · the predictor study:
[docs/PREDICTOR-EXPERIMENTS.md](docs/PREDICTOR-EXPERIMENTS.md) · the full log of ideas tried: [docs/perf-ideas.md](docs/perf-ideas.md).

</div>

## Install

<div align="justify">

You need a recent NVIDIA driver, the CUDA 13 toolkit, gcc-toolset-13 (Rocky Linux 9; other distributions are untested)
and about 80 GB free on an NVMe.

</div>

```bash
sudo dnf install -y git gcc-toolset-13
git clone https://github.com/sangharshadhyeta/Strata-HotRod.git
cd Strata-HotRod
scl enable gcc-toolset-13 -- ./setup.sh --model IQ3_S --context 131072 --build --yes --no-start
```

<div align="justify">

- `--build` compiles this fork's engine; without it setup downloads upstream's, which lacks these changes. The compiled
  engine was checked against the one measured here: the same speed.
- For 262K context set `--max-context 262144` in the config's `args`.
- `--model IQ3_S` is the size tested here. Setup says it wants a 64 GB PC; naming it is the consent, and it runs on 30 GB.
- `--models-dir /path/on/nvme` puts the model files elsewhere.

</div>

<div align="justify">

**Run** inside a memory limit (94% of RAM; two uncapped runs once froze this PC):

</div>

```bash
systemd-run --scope -p MemoryMax=$(awk '/MemTotal/ {printf "%dK", $2 * 0.94}' /proc/meminfo) -p MemorySwapMax=0 \
  .venv/bin/python serve/server.py --engine strata --config strata-iq3_s.json \
  --port 8095 --host 0.0.0.0 --api-key YOUR_KEY
```

<div align="justify">

When it prints `ready: http://127.0.0.1:8095/v1`, open `http://127.0.0.1:8095`. For this PC only, leave out `--host` and
`--api-key`.

</div>

<div align="justify">

**Update:** `scl enable gcc-toolset-13 -- ./update.sh --build` (new code, engine rebuilt where it changed, model kept).

</div>

## Using it

<div align="justify">

Any OpenAI-compatible client works with **`http://127.0.0.1:8095/v1`** and your API key:

</div>

```bash
curl http://127.0.0.1:8095/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"strata","temperature":1.0,"top_p":0.95,"max_tokens":300,
       "messages":[{"role":"user","content":"Name the three largest cities in India."}]}'
```

<div align="justify">

- **Thinking** is on by default; the chat menu picks off, low, medium or high.
- **Sampling:** with no temperature set, decoding is greedy. The model card recommends temperature 1.0, top-p 0.95, top-k 20
  with thinking on, and 0.7 / 0.8 / 20 with it off.
- **Reading while you type:** the chat page sends your draft as you pause, so Send only reads what changed. Programs can do
  the same with `"strata_prefill": true` (or `X-Strata-Prefill: 1`, or `/v1/prefill`).
- **Pictures (optional):** add a `"vision"` section to the config; a message can then carry an `image_url`
  ([docs/IMAGES.md](docs/IMAGES.md)).
- **One request at a time;** others wait their turn.

</div>

<div align="justify">

Every option: [docs/DETAILS.md](docs/DETAILS.md) (upstream's reference; it covers more than we tested).

</div>

<details>
<summary><b>Not tested</b></summary>

<div align="justify">

- Any other graphics card, any AMD card, Windows, macOS or another Linux.
- Any other model or size (Q2_0, IQ2_XS, IQ3_XXS, Coder, Swift, Unsloth's files).
- Several requests at once, or two graphics cards.
- A prompt that fills the whole 262K window (the longest served: 114K tokens).

</div>
</details>

## Credits and license

<div align="justify">

The model is Qwen3.8-Flash-Next by the Qwen team, compressed by ISTA-DASLab. Strata is by Niko1221 and its contributors;
this fork keeps its [MIT License](LICENSE) and copyright notice. The engine uses parts of
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp), and the picture encoder uses llama.cpp's `mtmd`. A few parts and
every model have their own licenses: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#license). Security: [SECURITY.md](SECURITY.md).

</div>

<details>
<summary><b>Ideas this fork builds on</b></summary>

<div align="justify">

Several changes were first built and measured in [solid.cpp](https://github.com/sangharshadhyeta/solid.cpp), a llama.cpp
fork for MoE expert caching, and ported here one concept at a time. Prior art behind them:

</div>

<div align="justify">

- [leloch/llama.cpp](https://github.com/leloch/llama.cpp) (moe-cache-pr): the original GPU-resident MoE expert cache.
- [thecodacus/llama.cpp](https://github.com/thecodacus/llama.cpp) (prefetch-experts, from the
  [Codacus](https://www.youtube.com/@Codacus) channel): expert prefetch overlapped with compute.
- [JustVugg/colibri](https://github.com/JustVugg/colibri): LFRU eviction and the Expert Atlas method; its
  [router-prediction experiment](https://github.com/JustVugg/colibri/issues/200) showed where a predictor stops paying.
- [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken): the hit/miss split for expert uploads.
- [antirez/ds4](https://github.com/antirez/ds4): the three-tier layout and explicit reads instead of mmap.
- [aswinkumar99/qwen3.8-flash-next-expert-traces](https://huggingface.co/datasets/aswinkumar99/qwen3.8-flash-next-expert-traces)
  (CC-BY-4.0): the expert-predictor study on this model; the learned look-ahead uses its training target (the router's top-32
  scores plus a hinge), and `--expert-predictor on` downloads its published predictor.
- [vLLM](https://github.com/vllm-project/vllm): a reference for the model's code, and not reserving memory for unused context.
- Papers: FlexGen ([2303.06865](https://arxiv.org/abs/2303.06865)) and MoE-Lightning
  ([2411.11217](https://arxiv.org/abs/2411.11217)) for layer-by-layer prompts; Stream2LLM for reading the prompt while it is
  written; the expert-prefetch literature (Speculating Experts [2603.19289](https://arxiv.org/abs/2603.19289), PILOT / Fate
  [2502.12224](https://arxiv.org/abs/2502.12224), FineMoE, SP-MoE [2510.10302](https://arxiv.org/abs/2510.10302), ProMoE,
  PowerInfer-2).

</div>
</details>
