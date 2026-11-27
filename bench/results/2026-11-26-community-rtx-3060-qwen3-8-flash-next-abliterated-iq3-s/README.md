# Community benchmark: RTX 3060 12 GB (Proxmox LXC), Ryzen 9 5900X, Strata-HotRod ratrod.v1.2

Reported by `VenimK` on [issue #7](https://github.com/sangharshadhyeta/Strata-HotRod/issues/7), 2026-11-26 ([the comment](https://github.com/sangharshadhyeta/Strata-HotRod/issues/7#issuecomment-7183051584)). This is the second RTX 3060 box reported on issue #7; the first is [2026-11-23-community-rtx-3060-ratrod](../2026-11-23-community-rtx-3060-ratrod/README.md). Numbers below are copied from the report; nothing here was re-measured by us.

The per-run numbers, ranges and the agent-turn table are not in this repository. The report refers to them in a PR folder of the same name as this one, which was not included here. They are in the issue comment linked above; the summary medians the comment prints are copied below.

## Hardware and software (as reported)

- Proxmox LXC with 64 GiB assigned; GPU passed through.
- RTX 3060 12 GB, run at x4 (downgraded link).
- Ryzen 9 5900X.
- 64 GB DDR4-3600.
- WD SN850X 1 TB, Gen 4 x4.
- Engine: `ratrod.v1.2` (commit `4ea40f1d`).
- Model: Qwen3.8-Flash-Next Swift-1.5 abliterated IQ3_S pack. The reporter notes this is not directly comparable with the stock IQ3_S numbers in the 2026-11-23 report: the quantization is the same, but the model has been through abliteration, and this box has more RAM and a faster drive and link.

## Settings (as reported)

Environment variables on top of the defaults:

- `STRATA_ROUTE_RESIDENT_RANKS=2-9`
- `STRATA_HC_REQ8=1`
- `STRATA_TAIL_CPU=0` (the rank rule)

Shipped settings, listed by the reporter as in force:

- `STRATA_ROUTE_RESIDENT=1e30`
- `STRATA_ROUTE_RESIDENT_EDGE=1`
- `STRATA_SWAP2_MAX=2`
- `STRATA_SWAP2_GAP=1`
- `STRATA_KV_GROW_INIT=4096`
- `STRATA_LA_LEARN=1`

Also as posted: "Spec 4", KV cache `int8`, context `131072`.

## `./bench.sh` medians (3 runs each)

| Test | Result |
|---|---:|
| Fresh prompt, 4,019 tokens | 1041.8 prompt tok/s |
| Fresh prompt, 16,058 tokens | 720.9 prompt tok/s |
| Short chat, 43-51 tokens | 47.7 generation tok/s |
| Agent follow-up, up to about 15.5K tokens | 50.2 generation tok/s (median); agent cache hit 69.9% |

## Quality audit (`tools/quality_audit.py`)

Run as asked: reference = `tail_weight=0, tail_skip=0` (exact tail rules), saved as `audit-ref.json`. The engine's own resident and stand-in settings above were in force, which is what changes which experts are missing on this machine. The server was restarted with `STRATA_LOGPOS` and `STRATA_LOGPOS_TOPK=64`, `--short-read 640 --adapt-every 100000`.

| setting | positions | dNLL | top1 | KL | tok/s |
|---|---:|---:|---:|---:|---:|
| base (reference) | 2542 | +0.00000 | 1.0000 | 0.00000 | 8.0 |
| base-repeat (noise floor) | 2142 | 0.29573 | 0.7106 | 0.70475 | 14.3 |
| rank7 | 2526 | 0.07516 | 0.6465 | 0.97810 | 74.6 |
| rank3 | 2526 | 0.02182 | 0.6536 | 0.93382 | 74.5 |
| w02 | 2526 | -0.03744 | 0.6746 | 0.73299 | 74.6 |
| w05 | 2526 | 0.00869 | 0.6037 | 1.26294 | 74.5 |
| w10 | 2526 | -0.47980 | 0.6445 | 0.83308 | 74.5 |
| w20 | 2526 | -0.05171 | 0.6837 | 0.74592 | 74.5 |
| w30 | 2526 | -0.56031 | 0.6599 | 0.78051 | 74.5 |

The reporter's reading: the noise floor is |dNLL| 0.296, and every tail setting sits within about that of the reference (a couple beyond), with top1 agreement a little under the noise floor's 0.71. The settings' divergence is small but not zero on this machine. The tok/s column is the verify-window read with the scorer on, so it is not comparable with the bench numbers above; all settings ran at the same speed there.
