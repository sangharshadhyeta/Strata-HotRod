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
| **Context** | 128K tokens |

## How fast is it?

Measured on the machine above. A token is about ¾ of a word.

**Writing answers** (the same install config, 3 repeats of 3 kinds of prompt, 900 tokens each, 17 Oct 2026):

| Context window | Median | Range |
| --- | ---: | ---: |
| 128K tokens | **53.7 tokens/s** | 51.8 - 59.4 |
| 32K tokens | 57.6 tokens/s | |

While a coding agent used it for real work (long prompts, thinking on), the server log shows 28 to 45 tokens per second,
with 89 to 93% of the experts found on the graphics card.

**Reading a prompt** (seconds until the first word; a different text each time, 10 Oct 2026):

| Prompt | First word after |
| --- | ---: |
| 150 - 260 tokens | 4 - 9 s |
| 430 - 500 tokens | 7.5 - 10.5 s |
| 1,600 - 2,100 tokens | 14 - 16 s |

The first message of a chat is the slow part: it is limited by how fast the SSD delivers the model's experts. Follow-up
messages reuse what was already read and start in a few seconds. Long prompts are read in pieces of up to 8,192 tokens.

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

- **Stand-in experts.** For each token the two best experts are always the real ones. Lower-ranked experts that are not on
  the card can be replaced by a similar one that is, so the card does not wait for the SSD. Swap rounds move the experts a
  chat asks for onto the card in the background.
- **Guessing ahead.** A small draft head guesses the next words and the big model checks them together. Probabilistic
  acceptance of the guesses is on by default (`--no-spec-prob-accept` turns it off).
- **A look-ahead ring** for fetching experts early (opt-in with `STRATA_RING=1`; it was slower on this machine).
- **The app:** live speeds while it works, and the experts-on-the-card map in the Monitor.
- **Pictures.** The model reads images through a processor-based encoder, so the graphics card keeps all its memory for
  experts. Tested on 18 Oct 2026 with generated images: text, shapes and colours were read correctly. A new picture adds a
  few seconds before the answer starts.

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
- **Pictures:** send an `image_url` part in the message (base64 or a link). More in [docs/IMAGES.md](docs/IMAGES.md).
- **One request at a time.** While one request runs, the others wait.
- **Run it inside a memory limit.** Two uncapped runs once froze this 30 GB PC. We start the server in a limited scope:
  `systemd-run --scope -p MemoryMax=29G -p MemorySwapMax=0 ...`

## Setting it up

This is how this PC was set up. It has not been repeated on another machine.

1. A recent NVIDIA driver, the CUDA 13 toolkit and gcc-toolset-13 (Rocky Linux 9; other distributions are untested).
2. `git clone` this repository and run **`./setup.sh`**. It makes the Python environment, downloads the model
   (about 80 GB: allow the time and the disk) and writes the config `strata-<model>.json`.
3. Start it with `./setup.sh` again, or run `serve/server.py --engine strata --port 8095 --config strata-iq3_s.json`.
4. For pictures, build the encoder (`tools/vision`, the CPU build) and add a `"vision"` section to the config with
   the encoder, the model's `mmproj` file and the first model file. The engine then starts with `--vision`.

Every option is described in [docs/DETAILS.md](docs/DETAILS.md) (upstream's reference; it covers more than we tested).

## Not tested

- Any other graphics card, any AMD card, Windows, macOS or another Linux.
- Any other model or size (Q2_0, IQ2_XS, IQ3_XXS, Coder, Swift, Unsloth's files).
- Several requests at once, or two graphics cards.
- A prompt that fills the whole 128K window.
- Accuracy beyond the small test above.

## Credits and license

The model is Qwen3.8-Flash-Next by the Qwen team, compressed by ISTA-DASLab. Strata is by Niko1221 and its contributors;
this fork keeps its [MIT License](LICENSE) and copyright notice. The engine uses parts of
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp), and the picture encoder uses llama.cpp's `mtmd`. A few parts
and every model have their own licenses: see [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#license). Security problems:
[SECURITY.md](SECURITY.md). The Hot Rod emblem and landing page (`docs/index.html`) are from
[AAO-IS2/Strata-HotRod](https://github.com/aao-is2/Strata-HotRod) under their MIT license.
