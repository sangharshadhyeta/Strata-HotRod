# Porting plan: one engine for Qwen, DeepSeek and Edge0 (from the Strataq fork's design)

Status: plan, 21 Nov 2026. Nothing here is built yet. Source read: `qwertie/Strataq` (MIT, David Piepgrass),
`docs/MEMORY_MANAGER_PLAN.md`, `src/common/serve/paged_start.cpp`, the start of `paged_engine.cpp`. Not read yet:
the rest of `paged_engine.cpp` (3,077 lines), his AMD docs, his panel code. Nothing of his was built or run.

## Rules (decided 21 Nov)

- **No foundation switch.** Upstream's layout and file names stay our foundation. His restructure (`src/common`, `src/models`, `src/base`, the paged
  engine) exists only in his fork; upstream has none of it. Adopting it would break path-based porting of upstream changes and tie us to one person's
  fork (1,951 commits behind upstream, last commit 14 Nov).
- **Hand-port, copy nothing.** We read his design and write our own version on our tree. No code of his is copied, so his licence notice does not
  enter our files. We name him in the README/docs for an idea once it ships (held until then).
- **Not merging upstream either.** Upstream changes are hand-ported too; `tools/upstream_triage.py` sorts the backlog (4,729 commits since our base
  on 21 Nov: 2,913 clean, 115 light, 1,318 heavy, 383 docs, by file overlap) and `docs/UPSTREAM_PORTS.md` is the ledger.
- **No PRs to upstream** (they accept none). Anyone, including qwertie, may send PRs to us.
- **Gate every port:** tokens identical, speed within noise, and the tuner, stand-ins, pool, peer tier and K/V streaming still on. A port that turns
  one of them off does not go in (rule: integrate, don't replace).
- **Issue #3 reply:** held, not posted.

## His five reasons (issue #3) and where each lands

| His reason | Our test / step |
|---|---|
| Two conversations allocate three full context windows (~10 GB at 200K from the start) | Step 2. Our code: `--batch N` carves N slot sessions of `session_bytes(max_context)` besides the main session (`generate.cpp` ~5868-5925). Measured live 21 Nov (result below) |
| Start serving once the dense weights are loaded | Step 1 |
| Three engines, repeated code | Step 4: shared interface as new files, adapters per model |
| One low-RAM mode that scales | Step 5, with the single scheduler |
| Live settings, each with a tooltip | Step 3: read-only view first, then editable |

### Measured 21 Nov: the three-windows claim holds for our `--batch`

Qwen UD-Q4_K_XL, 12 GB card, `--max-context 200000`, K/V streaming on (32,768 cells per layer resident), 29 GiB memory scope, same flags both runs:

| | no `--batch` | `--batch 2` |
|---|---|---|
| slot sessions | none | 2 x 0.84 GiB of VRAM |
| expert cache | 1,473 slots (4.27 GiB) | 885 slots (2.57 GiB): 40% fewer |
| RAM tier | 21.74 GiB | 17.06 GiB (4.7 GiB less) |
| RAM available to the engine (cgroup) | 24.24 GiB | 19.56 GiB: 4.68 GiB less = two more 2.36 GiB K/V host copies |

So two batch slots cost two more full-size context windows besides the main session (3 in all), in VRAM and in RAM, taken from the experts, as he
described. What the logs do not show: the main session's VRAM size, and whether it sits idle while the slots decode. Next: read that, then size slot
sessions by use (the elastic K/V that the single session already has; `--kv-grow` is switched off with `--batch`) and reuse the main session as slot 0.
Decode speed with `--batch 2` was not measured here (no requests were sent).

## What we take, and what we keep

His idea is one engine that owns memory and the decode loop, and a small adapter per model. We take that shape. We keep what we measured and built:
the online tuner, calibration, stand-ins, route tail skip, the learned look-ahead, the CPU pool, the peer tier, K/V streaming.

## Why the three branches are not a `git merge`

| Branch | Relation to main | Dry-run merge |
|---|---|---|
| `hotrod-edge0` | main is its ancestor (34 commits ahead) | 0 conflicts; main can fast-forward to it |
| `upstream-dsv41` (DeepSeek) | common base is 16 Oct; 2,378 commits on it, 478 on main | 34 conflicts, 2,433 files differ |
| `main` (Qwen) | | |

Merging DeepSeek textually would bury our Qwen work under another snapshot of upstream. So the unification is done by one interface, in steps,
with each model passing its own test before the next step.

## Steps (smallest and most testable first)

0. **Edge0 into main.** Fast-forward `main` to `hotrod-edge0` once the Edge0 model has had a live smoke run after the last merge (not yet done).
   Until then Edge0's model is a branch-only feature.
1. **Fast start.** Say READY once the dense weights are loaded; fill the expert pools while serving (GPU pools first in the profile's order, then
   RAM), with `EXPERT_FILL` in the memory report. His numbers (Strix Halo): Qwen READY after 9-10 s instead of 38 s. Ours takes about 4-5 minutes.
   Test here: time to READY and first-token speed before and after, same prompt, tokens identical. Risk: a request during the fill meets cold
   experts; the fill must yield to a request (our tuner already skips windows while a request is not running).
2. **Check the conversation-memory claim.** His finding: two conversation slots allocate three context windows. Measure our allocation per slot at
   128K and 262K (VRAM and RAM) before changing anything.
3. **Read-only live memory view** in our server: per memory (each GPU, RAM, files) what is held now, the storage profile line, the tuner's
   current settings. Then make a few settings editable, "this session" or "save as default".
4. **One model interface.** Extract what Qwen and DeepSeek share into one adapter interface (load dense weights, one prompt chunk, one decode step
   over pages, expert key space, state checkpoint). Start with Qwen as the first adapter (no behaviour change; tokens identical, speed within noise),
   then port DeepSeek onto it from its branch, feature by feature, each with its own A/B (the DeepSeek ring/lend/stand-in results of 31 Oct-6 Nov are
   the bar to keep).
5. **One scheduler on that engine** (docs/SCHEDULER.md): the single owner of slot/RAM/file placement, the drive model, admission. Large-RAM setups
   skip file read-ahead when `all_in_ram()`; needs a big-RAM machine to test.
6. **Edge0** as a third adapter (latent attention path), after step 4 passes for both others.

## AMD

Our fork keeps upstream's HIP path. His unified engine also runs on AMD, and he works mainly on AMD. We have no AMD hardware here, so an AMD part
of any step is untested by us. Rule: shared code compiled for HIP where the build allows it; AMD-specific changes go up as draft PRs marked untested
for AMD users to try (as with the peer-reserve PR). His `docs/AMD_HIP.md` and `AMD_HIP_PERFORMANCE.md` are the next things to read.

## Not decided

- Whether to move our base snapshot of upstream forward (we are at 16 Oct; upstream has 5,000+ commits since). Separate decision.
- How much of his memory-manager (no expert-cache setting, O_DIRECT file tier, pausing conversations to SSD) we adopt; steps 1-3 do not depend on it.
