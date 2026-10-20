<p align="center"><img src="docs/media/hotrod-hero.gif" width="400" alt="Strata Hot Rod: the Strata hexagon in fire and ice"></p>

<h1 align="center">Strata-HotRod</h1>

<p align="center"><b>Qwen3.8-Flash-Next, a 125-billion-parameter model, on one RTX 3060 (12 GB) with 30 GB of RAM</b><br>
Linux · NVMe SSD · nothing leaves your PC</p>

This is a fork of [Strata](https://github.com/Niko1221/Strata) by Niko1221 and contributors (MIT). Strata runs a model that
normally needs a server by keeping the most-used parts on the graphics card and streaming the rest from the SSD.

**Everything in this README was tested on one machine, with one model.** Other graphics cards, AMD, Windows, macOS and
other model sizes may work in upstream Strata, but we have not run them, so we do not describe them here.

## What was tested

| | |
| --- | --- |
| **Graphics card** | NVIDIA GeForce RTX 3060, 12 GB (driver 610.57, CUDA 13.3) |
| **Processor, RAM** | Intel Core i5-12400F (6 cores, 12 threads), 30 GB |
| **Disk** | 500 GB NVMe (WD Blue SN5000); the model files live on it |
| **System** | Rocky Linux 9.6, built with gcc-toolset-13 |
| **Engine** | Strata 0.1.45 (upstream, merged 17 Oct 2026) plus the changes below |
| **Model** | [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) in the IQ3_S size (about 3.5 bits per weight; 78 GB of files), compressed by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) |
| **Context** | 262K tokens (the longest prompt served so far: 114K) |

## How fast is it?

Measured on the machine above, with the settings the install below writes. A token is about ¾ of a word.

**Writing answers** (17 Oct 2026). Three kinds of prompt, each asked three times in a row; answers stop at 900 tokens.
The first time a prompt is asked is what a new question gets. The repeats run faster, because that prompt's experts are
already on the card.

| Prompt (answer length) | 128K context, first ask | 128K, repeats | 32K context, first ask (5 runs) |
| --- | ---: | ---: | ---: |
| Code (240 - 320 tokens) | 43.9 tokens/s | 67.9, 76.1 | 43.0 - 44.0 |
| Reasoning (790 - 900 tokens) | 53.7 tokens/s | 64.1, 71.7 | 53.7 - 61.8 |
| Story (140 - 180 tokens) | 41.8 tokens/s | 43.1, 51.1 | 36.1 - 39.8 |

The 128K column is one run; the 32K column is five runs with the same settings, which shows how much one run varies.

**Long prompts, 262K context** (20 Oct 2026; greedy, 1,000-token answer after the 8.5K prompt, 300 after the 45K one;
each number is one of two or more runs that agree within about 5%):

| Prompt | Read in | Answer written at |
| --- | ---: | ---: |
| 8,554 tokens | 16.9 s | 52 - 54 tokens/s |
| 45,398 tokens | 67 - 68 s | 38 - 41 tokens/s |
| The same 45K conversation, next turn | 0.4 - 0.5 s (only the new message is read) | 34 - 45 tokens/s |

**Short prompts** (seconds until the first word; a different text each time; one run on 10 Oct 2026, before the 0.1.45
merge):

| Prompt | First word after |
| --- | ---: |
| 150 - 260 tokens | 3.8 - 5.8 s |
| 430 - 500 tokens | 7.5 - 10.5 s |
| 1,600 - 2,100 tokens | 14 - 16 s |

The first message of a chat is the slow part: it is limited by how fast the SSD delivers the model's experts. Follow-up
messages reuse what was already read. Where the time goes, and every change measured on this PC, is in
[docs/perf-ideas.md](docs/perf-ideas.md); it also compares the engine with a week earlier (decode 17 -> 54 tokens/s,
a 45K prompt 120 -> 68 s, context 32K -> 262K).

**Quality of the answers.** Strata replaces some of a token's experts with similar ones already on the card to stay fast.
In a small test (50 maths questions, 30 coding questions, answers read by hand): 47 of 50 and 30 of 30 correct, against
48 of 50 and 30 of 30 with nothing replaced. A test this small cannot show a difference of one or two answers either way.
The detail is in [SPEC.md](SPEC.md).

## What you see

<p align="center"><img src="docs/media/portal-monitor.png" width="900" alt="The Strata app's Monitor tab while the model is generating: 44 tokens per second, the RTX 3060 at 97% load, 11.7 of 12 GB of video memory, 2,793 experts on the card, and a table of recent requests with a 92 to 94% expert hit rate"><br>
<sub>The <b>Monitor</b> tab while the model writes: speed, graphics card load and memory, the experts on the card, and each request's hit rate.</sub></p>

<p align="center"><img src="docs/media/portal-chat.png" width="900" alt="The Strata chat: a finished answer with '1,805 tokens, decode 55.1 tok/s' under it"><br>
<sub>The <b>Chat</b> tab. Under every answer: the number of tokens and the speed.</sub></p>

## What this fork changes

Upstream Strata, plus (the details and the measurements are in [SPEC.md](SPEC.md)):

- **Stand-in experts.** Each token uses ten experts per layer. The two best are replaced by an expert that is on the card
  only when that one scores within 1 logit of them (close enough that answers stay the same); otherwise they are the real
  experts, computed from RAM or read from the SSD. Ranks 3 to 10 that are not on the card are replaced by the most similar
  one that is, so the card does not wait for the SSD. Measured on this PC: 57.7 against 54.4 tokens/s on a long prompt and
  47.4 against 38.6 across short ones, with clean code, reasoning and prose; the 210-question accuracy check is running. Swap rounds move the experts a chat asks for onto the card in the background. The install writes
  these settings for IQ3_S. Measured: a wider margin than 1 logit drops tokens (2 logits) or makes answers collapse into
  loops (3 and more); details in [docs/perf-ideas.md](docs/perf-ideas.md). The settings, in the config's `env`:

  | Setting | What it does | Install value |
  | --- | --- | --- |
  | `STRATA_ROUTE_RESIDENT_RANKS` | the ranks that always take a stand-in when not on the card | `2-9` |
  | `STRATA_ROUTE_RAM_FLOOR` | the two best experts may take a close stand-in (`0`: always the real ones) | `1` |
  | `STRATA_ROUTE_RAM_FLOOR_MARGIN` | how close that stand-in must score, in logits | `1` (default) |
  | `STRATA_ROUTE_RAM_FLOOR_ANYTIER` | also when the expert is in RAM (`0`: only when it would come from the SSD) | `1` |
  | `STRATA_ROUTE_RAM_FLOOR_RANK0` | the best expert too (`0`: only the second best may take a stand-in) | `1` |
- **Guessing ahead.** A small draft head guesses the next words and the big model checks them together, with
  probabilistic acceptance of the guesses (`--no-spec-prob-accept` turns it off).
- **Three tiers of experts: card -> RAM -> SSD.** Every expert lives in exactly one place. The hottest are on the card, the
  next ones in a RAM tier sized from this PC's free memory, the rest on the SSD; swaps exchange experts between the card
  and RAM, and the look-ahead promotes the experts it predicts from the SSD into RAM.
- **Reading long prompts layer by layer.** The whole prompt passes one layer at a time, so each layer's experts are read
  from the SSD once, not once per piece of the prompt; the reads bypass the page cache (O_DIRECT), and the next layer's
  predicted experts are read ahead while the current one computes.
- **262K context with the K/V in RAM.** From 64K of context the attention's K/V streams from RAM; 32K tokens of it stay
  on the card.
- **Look-ahead on the card.** The next layers' experts are predicted on the graphics card (not recomputed by the CPU)
  while the current layer runs, and the predictions start their reads early.
- **Starting the answer warm.** The experts the end of the prompt used most are put on the card before the first word
  of the answer, so decode does not start cold.
- **Picking up a conversation.** A long conversation's state is saved to the SSD, and its next turn reloads it instead of
  reading the whole conversation again.
- **Memory safety.** The engine keeps a floor of RAM free before each request; when something else needs RAM, the RAM
  tier gives back its least-used experts until there is room again. The amount it keeps in reserve is measured on this
  PC and learned across restarts (`data/ram-reserve.json`), not a fixed number; the video-memory reserve is learned the
  same way (`vram-reserve.json`).
- **The app:** live speeds while it works, and the experts-on-the-card map in the Monitor.
- **Experimental, off by default:** `STRATA_GDN_LOOP=1 STRATA_GDN_LOOP_NORM=1` runs the middle linear-attention layers
  twice per token (a training-free "looped" model). Measured on 210 MMLU-Pro questions: 180 correct against 175 without
  it (not significant: only 9 questions differ), and decode 10% slower (38.5 against 42.7 tokens/s). Not worth it on a
  model that was not trained to loop, so it stays off.

## Using it

Open the app at `http://127.0.0.1:8095` (the port is set with `--port`). Your own programs can use the
OpenAI-compatible address **`http://127.0.0.1:8095/v1`** (`/v1/chat/completions`), with the API key you set.

```bash
curl http://127.0.0.1:8095/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"strata","temperature":1.0,"top_p":0.95,"max_tokens":300,
       "messages":[{"role":"user","content":"Name the three largest cities in India."}]}'
```

- **Thinking:** the model thinks before it answers by default. The chat menu chooses off, low, medium or high.
- **Sampling:** a request that sets no temperature is decoded greedily. The model's card
  ([Best Practices](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices)) recommends temperature 1.0, top-p 0.95
  and top-k 20 with thinking on, and temperature 0.7, top-p 0.8 and top-k 20 with thinking off. Set them in your client.
- **Reading while you type.** The chat page sends your draft while you pause typing, so the engine reads it before you
  press Send; Send then reads only what changed. Your own programs and agent harnesses can do the same through the
  endpoint they already use: send the request so far with `"strata_prefill": true` in the body or the header
  `X-Strata-Prefill: 1` (or POST it to `/v1/prefill`); the answer is a small JSON (`prompt_tokens`, `reused`, `seconds`)
  instead of a completion. An agent can send its growing context, a tool's output included, while the tool still runs.
  Answers do not change: the real request is read exactly as sent. `"prefill_warm": false` in the config turns it off.
- **One request at a time.** While one request runs, the others wait.
- **Run it inside a memory limit.** Two uncapped runs once froze this 30 GB PC; the run command below sets one.

## Setting it up

This is how this PC was set up. It has not been repeated on another machine.

You need a recent NVIDIA driver, the CUDA 13 toolkit and gcc-toolset-13 (Rocky Linux 9; other distributions are
untested), and about 80 GB free on the NVMe for the model.

**Install** (once; it makes the Python environment, compiles the engine, downloads the model and writes the config
`strata-iq3_s.json`):

```bash
sudo dnf install -y git gcc-toolset-13
git clone https://github.com/sangharshadhyeta/Strata-HotRod.git
cd Strata-HotRod
scl enable gcc-toolset-13 -- ./setup.sh --model IQ3_S --context 131072 --build --yes --no-start
```

- `--build` matters: without it, setup downloads upstream's ready-made engine, which does not have this fork's changes.
- The speeds above use a 262K context: after setup, set `--max-context 262144` in the config's `args`. From 64K the K/V
  streams from RAM by itself.
- `--model IQ3_S` is the size tested here. Setup says IQ3_S wants a 64 GB PC; naming it explicitly is the consent, and
  it runs on this 30 GB PC.
- Add `--models-dir /path/on/nvme` to put the model files somewhere other than the default.

**Run** (inside a memory limit; `--host 0.0.0.0` and a key let other devices on your network use it):

```bash
systemd-run --scope -p MemoryMax=29G -p MemorySwapMax=0 \
  .venv/bin/python serve/server.py --engine strata --config strata-iq3_s.json \
  --port 8095 --host 0.0.0.0 --api-key YOUR_KEY
```

When it prints `ready: http://127.0.0.1:8095/v1`, open `http://127.0.0.1:8095` in a browser. For this PC only, leave out
`--host` and `--api-key`. Stop it with Ctrl+C.

**Update** (the newest code, then the engine compiled again where its source changed; the model files are kept):

```bash
scl enable gcc-toolset-13 -- ./update.sh --build
```

**Pictures (optional):** build the encoder (`tools/vision`, the CPU build) and add a `"vision"` section to the config
with the encoder, the model's `mmproj` file and the first model file. The engine then starts with `--vision`, and a
message can carry an `image_url` part (base64 or a link; more in [docs/IMAGES.md](docs/IMAGES.md)). A new picture adds a
few seconds before the answer starts.

Every option is described in [docs/DETAILS.md](docs/DETAILS.md) (upstream's reference; it covers more than we tested).

## Not tested

- Any other graphics card, any AMD card, Windows, macOS or another Linux.
- Any other model or size (Q2_0, IQ2_XS, IQ3_XXS, Coder, Swift, Unsloth's files).
- Several requests at once, or two graphics cards.
- A prompt that fills the whole 262K window (the longest served: 114K tokens).
- Accuracy beyond the small test above (a 210-question MMLU-Pro run is in progress).

## Credits and license

The model is Qwen3.8-Flash-Next by the Qwen team, compressed by ISTA-DASLab. Strata is by Niko1221 and its contributors;
this fork keeps its [MIT License](LICENSE) and copyright notice. The engine uses parts of
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp), and the picture encoder uses llama.cpp's `mtmd`. A few parts
and every model have their own licenses: see [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#license). Security problems:
[SECURITY.md](SECURITY.md). The Hot Rod emblem and landing page (`docs/index.html`) are from
[AAO-IS2/Strata-HotRod](https://github.com/aao-is2/Strata-HotRod) under their MIT license.
