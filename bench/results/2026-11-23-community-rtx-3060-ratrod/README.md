# Community benchmark: RTX 3060 12 GB (Proxmox VM), Strata-HotRod latest release

Reported by `mpisat` on [issue #7](https://github.com/sangharshadhyeta/Strata-HotRod/issues/7), 2026-11-23, after the `ratrod.v1` release. Numbers below are copied from the reports; nothing here was re-measured by us.

## Hardware and software (as reported)

- One RTX 3060 12 GB with GPU pass-through into a KVM guest on Proxmox.
- 64 GB DDR4-3200 (59 GB visible inside the guest), Gen 4 NVMe SSD.
- Model: Qwen3.8-Flash-Next IQ3_S.
- Engine: "the latest version released" when reported. The report does not name the tag; `ratrod.v1` (2026-11-22) was the latest release at the time.

## One 120K-token request (the reporter's standard comparison)

| Measurement | Result |
|---|---:|
| Prompt | synthetic coding prompt, 120,000 tokens, none cached |
| Output | 8,192 tokens |
| Allocated context | 131,072 tokens |
| Prompt processing | 781.4 tok/s (153.6 s) |
| Generation | 40.0 tok/s (204.9 s) |
| Total request | 358.6 s |

A synthetic coding prompt is predictable, so the drafter's guesses are accepted more often than on real work; 40 tok/s is likely above what a real prompt gives. Prompt processing is the average over the whole 120K prompt read at once.

## `./bench.sh` medians (IQ3_S)

The comment with these numbers does not say which machine ran them.

| Test | Actual prompt | Result |
|---|---:|---:|
| Fresh prompt | 4,019 tokens | 997.4 prompt tok/s |
| Fresh prompt | 16,058 tokens | 928.6 prompt tok/s |
| Short chat | 43-51 tokens | 33.6 generation tok/s |
| Agent follow-up | up to about 15.5K tokens | 33.4 generation tok/s |

The reporter's reading: the lower generation speed here than in the 120K run may come from the smaller KV cache and smaller prompts. We have not checked that.

## Second machine: Intel Arc Pro B50 16 GB, 96 GB DDR5-5200

Reported as "worse than stock Strata", with no numbers. This fork's SYCL code was last synced with upstream in mid-October and has no Battlemage-specific kernels, so this is the expected direction. The upstream SYCL work is to be ported; the reporter offered to re-run when it is.
