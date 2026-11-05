<p align="center"><img src="docs/media/hotrod-hero.gif" width="380" alt="Strata Hot Rod: the Strata hexagon in fire and ice"></p>

<h1 align="center">Strata-HotRod</h1>

<h2 align="center">A 341 GB model. 31 GB of RAM. 12 GB of VRAM. Zero cloud.</h2>

<p align="center"><b>DeepSeek-V4.1-Flash on one gaming card, streaming from your SSD, at about 5 tokens/s.</b><br>
Nothing leaves your PC.</p>

<p align="center">
🔥 <b>hot experts</b> on the card &nbsp;→&nbsp; 🌡️ <b>warm</b> in RAM &nbsp;→&nbsp; ❄️ <b>cold</b> on the SSD
</p>

<p align="center">
  <img alt="Streams from SSD" src="https://img.shields.io/badge/STREAMS%20FROM-SSD-1c7ed6?style=for-the-badge">
  <img alt="31 GB RAM, 341 GB model" src="https://img.shields.io/badge/31%20GB%20RAM-341%20GB%20MODEL-d6336c?style=for-the-badge">
  <img alt="12 GB VRAM" src="https://img.shields.io/badge/12%20GB-VRAM-76b900?style=for-the-badge">
  <img alt="About 5 tokens per second" src="https://img.shields.io/badge/~5-TOK%2FS-e8590c?style=for-the-badge">
</p>

<table>
<tr>
<td align="center" width="33%"><h2>🐋 341 GB</h2>of DeepSeek-V4.1-Flash files on a <b>31 GB</b> PC</td>
<td align="center" width="33%"><h2>⚡ about 5</h2>tokens/s on one RTX 3060 12 GB</td>
<td align="center" width="33%"><h2>💾 2.8 GB/s</h2>the NVMe is the limit: each token still reads about 2 GB of experts</td>
</tr>
</table>

---

<div align="justify">

This is the **DeepSeek branch** (`upstream-dsv41`) of Strata-HotRod: upstream [Strata](https://github.com/Niko1221/Strata)
0.2.7.1's DeepSeek engine (`strata-dsv41`) with the Strata-HotRod features ported. The Qwen3.8-Flash-Next page is on
[`main`](https://github.com/sangharshadhyeta/Strata-HotRod). What was ported, with every measurement:
[docs/HOTROD-DSV41-PORT.md](docs/HOTROD-DSV41-PORT.md). Upstream's own README for this branch is kept in the history
([the last version](https://github.com/sangharshadhyeta/Strata-HotRod/blob/6c02a2e1/README.md)).

</div>

> **Tested on one machine, with one model.** Other cards, AMD, Windows, macOS and other model sizes may work in upstream
> Strata; we have not run them, so we do not describe them here.

## How fast

<div align="justify">

Measured on the machine below (answer tokens/s, the same 3 questions, 32K context, every answer read by hand).

</div>

<table width="100%">
<tr><th align="left">Setting</th><th align="right">tokens/s</th></tr>
<tr><td align="left">The default here: the VRAM ring + stand-ins</td><td align="right"><b>5.2 / 5.3 / 5.4</b> (three runs)</td></tr>
<tr><td align="left">The same sessions without the ring and stand-ins</td><td align="right">4.8 - 5.0</td></tr>
</table>

<div align="justify">

Why it is not faster: 270 of the 15,360 routed experts (2.65 GB) fit the card next to 6.2 GB of dense weights, so about 75% of
the picks are computed on the CPU lane, from the RAM store or in place from the files. Each token still needs about 2 GB of
expert weights from RAM and the NVMe, and the NVMe (about 2.8 GB/s) is the limit on this PC. All the rows, including what did
not help: [docs/HOTROD-DSV41-PORT.md](docs/HOTROD-DSV41-PORT.md).

</div>

<details>
<summary><b>The machine and the files</b></summary>

<table width="100%">
<tr><th align="left"></th><th align="left"></th></tr>
<tr><td align="left"><b>Graphics card</b></td><td align="left">NVIDIA GeForce RTX 3060, 12 GB</td></tr>
<tr><td align="left"><b>Processor, RAM</b></td><td align="left">Intel Core i5-12400F, 31 GB</td></tr>
<tr><td align="left"><b>Disk</b></td><td align="left">NVMe on a PCIe 3.0 link (about 2.8 GB/s); the model files live on it</td></tr>
<tr><td align="left"><b>Model</b></td><td align="left">Starw1's DeepSeek-V4.1-Flash-1017 uncensored: the GSQ-RCO IQ2_XXS backbone (138.6 GB) with DeepSeek's MXFP8 Engram (202.8 GB), which is the 341 GB; plus the DSpark drafter (8.0 GB, a separate file) and a 5.8 MB metadata shard from Unsloth</td></tr>
<tr><td align="left"><b>Engine</b></td><td align="left">Strata 0.2.7.1 (upstream) plus this fork's DeepSeek work, release <code>v0.2.7.1-hotrod.2</code></td></tr>
</table>
</details>

## What this branch adds

<div align="justify">

To upstream's DeepSeek engine, on by default when the ring is on (`=0` turns each off):

- **A VRAM ring** (`DS4_RING=32`): the learned look-ahead's guesses for the next 1-2 layers are copied from a page-locked
  RAM store into card slots; a pick uses its slot only if the copy has landed in time.
- **Sticky ring** (`DS4_RING_STICKY`): a landed copy stays until its slot is needed, so a later token that picks the same
  expert reuses it (about 80,000 reuses in one answer).
- **The prompt pass's buffers as ring slots** (`DS4_LEND`, `DS4_LEND_RING`): the 1 GB the chunked prompt pass keeps on
  the card holds 99-100 more ring slots while the model answers, and is handed back before a long prompt.
- **A self-adjusting copy limit** (`DS4_RING_INFLIGHT`, auto): copies in flight follow the measured decode speed
  (128-token averages, a 3% dead band; it settles at 11-12 on this PC). `=fixed` keeps 12.
- **Stand-ins with their re-check and self-tuning budget**, the learned look-ahead and the learned VRAM reserve.
- **The RAM store gives back its coldest idle chunk** (unpinned first) when RAM runs short, and takes chunks back later.
- **Pictures**: with Unsloth's `mmproj-BF16.gguf` in the pack (`--mmproj` to `pack_gguf.py`) and `--vision on`, the
  engine's own vision tower reads images (319 MiB of VRAM; checked on a known picture: shapes, colours, positions and text
  right). Known issue: with the tower on, the RAM store is sized 0 and the ring turns off (a free-RAM drop is taken for a
  unified-memory machine); the fix is pending.

Measured and left opt-in: `STRATA_LA_TAIL_READ=3` (a learned read-ahead of lower-ranked guesses: 23% slower, the drive is
already at its limit here) and `DS4_PCIE_BALANCE=1` (Qwen's dynamic balance ported: its copies arrive too late and it is
not yet safe - do not turn it on).

</div>

## Install

<div align="justify">

About **345 GB free on an NVMe** (139 GB of model, the 203 GB Engram table, 1 GB for pictures), a recent NVIDIA driver, the
CUDA 13 toolkit and gcc-toolset-13 (Rocky Linux 9; other distributions are untested). The DeepSeek engine builds against the
`third_party/llama.cpp` of the Qwen checkout on [`main`](https://github.com/sangharshadhyeta/Strata-HotRod), so clone that first.
These are the commands this PC was set up with; the branch's own `setup.py --model Starw1-2.00bpw-MXFP8` downloads the same files
and makes the pack, but that route is untested here.

</div>

```bash
# the DeepSeek branch beside the Qwen checkout (run from inside Strata-HotRod, where the Qwen install left you)
git fetch origin upstream-dsv41 && git worktree add ../Strata-HotRod-ds upstream-dsv41
cd ../Strata-HotRod-ds
PY=../Strata-HotRod/.venv/bin/python   # the Qwen install's venv runs this branch's server and tools too

# the engine (sm_86 = RTX 30-series; set yours)
scl enable gcc-toolset-13 -- bash -c 'cmake -G Ninja -S . -B build-dsv41 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86-real -DSTRATA_GGML_DIR=$PWD/../Strata-HotRod/third_party/llama.cpp \
  && ninja -C build-dsv41 strata-dsv41'

# the model (M: the folder for the files): Starw1's 4 shards + the MXFP8 Engram, Unsloth's metadata shard,
# and for pictures Unsloth's mmproj-BF16.gguf in $M/..
M=/mnt/nvme/models/deepseek41-flash/GSQ-RCO-IQ2_XXS
$PY - "$M" <<'EOF'
import sys
from huggingface_hub import hf_hub_download, snapshot_download
m = sys.argv[1]
snapshot_download("Starw1/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-IQ2_XXS-MXFP8-Engram-GGUF", local_dir=m)
hf_hub_download("unsloth/DeepSeek-V4.1-Flash-1017-GGUF", "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf",
                local_dir=m + "/unsloth")
hf_hub_download("unsloth/DeepSeek-V4.1-Flash-1017-GGUF", "mmproj-BF16.gguf", local_dir=m + "/..")
EOF

# the pack (a few MB of index beside the files; no weights are copied)
$PY tools/dsv41/pack_gguf.py --model $M \
  --shards DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-0000{1,2,3,4}-of-00004.gguf \
  --engram $M/DeepSeek-V4.1-Flash-Engram-MXFP8.gguf \
  --engram-meta $M/unsloth/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf \
  --mmproj $M/../mmproj-BF16.gguf --out /mnt/nvme/strata-data/packs/ds41v
cp data/dsv41/profile-2.00.bin /mnt/nvme/strata-data/packs/ds41v/expert-profile.bin
```

<div align="justify">

**The config** `strata-ds41.json` (the ring and stand-ins are switched on here; everything above follows from them):

</div>

```json
{
 "exe": "/path/to/Strata-HotRod-ds/build-dsv41/strata-dsv41",
 "args": ["--pack", "/mnt/nvme/strata-data/packs/ds41v", "--max-context", "262144", "--eos", "1",
          "--vision", "on", "--dspark", "auto"],
 "cwd": "/path/to/Strata-HotRod-ds",
 "tokenizer": "/mnt/nvme/strata-data/packs/ds41v/tokenizer",
 "model_name": "deepseek-v4.1-flash",
 "arch": "deepseek41",
 "engine_ready_s": 3600,
 "env": {"DS4_RING": "32", "DS4_RING_PIN": "1", "STRATA_STANDIN": "1"},
 "answer_room": false,
 "reasoning_budget_tokens": 0
}
```

<div align="justify">

- `"answer_room": false` and `"reasoning_budget_tokens": 0`: no limit but the context (otherwise DeepSeek's thinking
  stops at 32,768 tokens and a client's cap keeps a quarter for the answer).
- `--vision off` (or leaving out `--mmproj` when packing) gives text only, and today the faster setup (see the known issue).

**Run** it inside a memory limit (94% of RAM; two uncapped runs once froze this PC), from `Strata-HotRod-ds`:

</div>

```bash
systemd-run --scope -p MemoryMax=$(awk '/MemTotal/ {printf "%dK", $2 * 0.94}' /proc/meminfo) -p MemorySwapMax=0 \
  $PY serve/server.py --engine strata --config strata-ds41.json --port 8095 --host 0.0.0.0 --api-key YOUR_KEY
```

<div align="justify">

The first start at a new context size learns its VRAM reserve; if it prints `ring: off (... would not fit)`, start it once
more. When it prints `ready: http://127.0.0.1:8095/v1`, open `http://127.0.0.1:8095`. For this PC only, leave out `--host`
and `--api-key`.

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

Pictures (optional): add a `"vision"` section to the config; a message can then carry an `image_url`
([docs/IMAGES.md](docs/IMAGES.md)). Every switch, flag and server key: [docs/FEATURES-deepseek.md](docs/FEATURES-deepseek.md).
Upstream's reference for the rest: [docs/DETAILS.md](docs/DETAILS.md) (it covers more than we tested).

</div>

<details>
<summary><b>Not tested</b></summary>

<div align="justify">

- Any other graphics card, any AMD card, Windows, macOS or another Linux.
- Any other DeepSeek set or size.
- Several requests at once, or two graphics cards.

</div>
</details>

## Credits and license

<div align="justify">

The model is DeepSeek-V4.1-Flash by DeepSeek, compressed by Starw1 (the GSQ-RCO IQ2_XXS backbone with DeepSeek's MXFP8
Engram), with metadata and the picture projector from Unsloth. Strata is by Niko1221 and its contributors; the DeepSeek
engine comes from Project Maya's maya-ds4. This fork keeps Strata's [MIT License](LICENSE) and copyright notice. The engine
uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). A few parts and every model have their own
licenses: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#license). Security: [SECURITY.md](SECURITY.md).

</div>
