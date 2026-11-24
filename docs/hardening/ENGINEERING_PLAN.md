# Hardening pass - engineering plan

Started 24 Nov 2026 on branch `hardening` (worktree /home/Projects/Strata-harden, from 9279a5e90 = ratrod.v1.2).

## Scope

Strata-HotRod: the C++/CUDA engine (`src/`, `include/`), the Python server (`serve/`), tools and tests.
There is no web frontend or database in this repo; the directive's frontend/database roles map to:
- "frontend" -> the HTTP/OpenAI-compatible surface of `serve/server.py` and `serve/frontend.py`;
- "database" -> the engine's on-disk state (learned profiles, `*.bin` learner states, prompt-cache sessions, layout weights).

## Hard rules for every agent

1. Never touch the user's server on port 8095, its process, or its data dir `/home/Projects/Strata/data`. No test may read or write there.
2. Never `pkill -f`; kill exact PIDs you started.
3. Builds: only `cmake --build` in `/home/Projects/Strata-harden/build-*`, one at a time (`flock /root/.claude/jobs/743230e4/tmp/build.lock`), under `systemd-run --scope -p MemoryMax=4G -p MemorySwapMax=0 nice -n 15`. The box has 31 GB and froze twice on uncapped runs.
4. No feature may be removed or turned off to make something pass; no test may be weakened. No regex/pattern code in engine decisions.
5. Every fix: root cause stated, a test that fails before and passes after (where practical), no change to a default unless asked.
6. No pushes, no tags, no releases. Commit on `hardening` only.

## Phases

0. Reconnaissance and baseline (build, C++ tests, Python tests, lint) - architecture map.
1. Integration: server <-> engine protocol, config/env handling, state files.
2. Debugging and security: crash paths, malformed input, auth on the HTTP surface, file paths.
3. Reduction: dead code, duplicate utilities (measured, never a feature).
4. Integration + regression.
5. Adversarial verification.
