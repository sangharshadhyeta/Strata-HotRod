# Porting plan: one engine for Qwen, DeepSeek and Edge0 (from the Strataq fork's design)

Status: plan, 21 Nov 2026. Nothing here is built yet. Source read: `qwertie/Strataq` (MIT, David Piepgrass),
`docs/MEMORY_MANAGER_PLAN.md`, `src/common/serve/paged_start.cpp`, the start of `paged_engine.cpp`. Not read yet:
the rest of `paged_engine.cpp` (3,077 lines), his AMD docs, his panel code. Nothing of his was built or run.

## What we take, and what we keep

His idea is one engine that owns memory and the decode loop, and a small adapter per model. We take that shape. We keep what we measured and built:
the online tuner, calibration, stand-ins, route tail skip, the learned look-ahead, the CPU pool, the peer tier, K/V streaming. A port must not switch
any of these off (rule: integrate, don't replace). If we copy his code we keep his MIT notice; if we take only an idea we name him in the docs.

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
