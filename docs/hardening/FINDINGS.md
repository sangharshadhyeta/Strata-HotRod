# Findings

Severity: critical / high / medium / low. Status: open / fixed (commit) / wontfix (reason).

## F-01 (high, fixed (eb8762752)) - double slot survives a runtime cache shrink with its spill unmapped
- Where: `src/program/generate.cpp` VRAM resize, the loop after `xcache.shrink(keep, e)` (~12517): it evicts `host_res[i] >= live` only.
- Scenario: `--vram-elastic` serve, a VRAM request shrinks the cache so that `live - 1` is a doubled slot. Its owner stays resident; its spill lies in slot `live`, now unmapped. The next window reads it: illegal address (engine death) or garbage weights.
- Reachability: needs `--vram-elastic`, and with the RAM tier on also `STRATA_VRAM_COMPLEMENT=1`. Not the default setup; the default build of ratrod.v1.2 has double slots on.
- Fix: evict also when `xcache.spills(host_res[i]) && host_res[i] + 1 >= live`; clear the spill flag of a slot whose owner was evicted there. Test: unit-level on ExpertCache is not enough (the loop is in generate.cpp); add a check to the existing resize test if one exists, else a small extracted helper with a test.

## F-02 (low, open) - a dissolved double's neighbour stays reserved
- `cross_consumed` / the spill flag are cleared only by the integrity monitor. If a doubled slot later receives a small expert, its neighbour stays empty until a big expert takes the slot again (best-fit ranks it last for small candidates, so this is rare). Efficiency only, no corruption.

## F-03 (low, open) - the integrity monitor uses a synchronous `cudaMemcpy` on the serve thread
- 2 x ~4 MB device-to-host every 8th adapt round; a legacy-stream copy waits for the device. Cost not measured; it may be part of the unexplained 19.1 tok/s batch. Measure before changing (the check itself must stay).

## From P0-B (architecture map), each checked against the code by the lead
- **F-04 (medium, open) - stdin writes bypass the write lock.** `serve/server.py` 1170 (VRAM), 2092 (GEN/GENI), 2175 (STOP), 2317 (SAVE/RESTORE) write `proc.stdin` directly; only `_send` (1281) takes `wlock`, whose own comment says writes come "from several request threads". A `/v1/vram` or session save arriving while a GEN line is written can interleave two lines into one malformed command. Fix: route them through `_send` (keep each one's own error handling). Checked: 2281 is `quit_line` at shutdown (single writer) - fine.
- **F-05 (medium, fixed (9a5a15ce4, 23e5ff166: no fsync - the shadow save runs under g_sh_mu on the decode path; whole-file rejection covers a short file after a power cut)) - learner state files: unchecked writes, no fsync.** `layer-shadow.bin`, `slot-share.bin`, `layer-live.bin`, `.layer-weights` writers (expert_source.cpp ~6036-6052, 6199-6208, 6240-6252, 6550-6560) ignore `fwrite`/`fclose` results and never fsync before `rename`, so a full disk or a power cut (one happened this week) can leave a short or empty file in place; the readers then keep a partial state silently (layer-live logs "starting fresh" while keeping part). Fix: check every write and `fclose`, `fsync` before rename, remove the temp file on failure; readers reject a short file as a whole and say so. Not a feature change. The prompt-cache session writer already does this right (conversation_file.cpp 274-302) - reuse its pattern.
- **F-06 (low, fixed (980316ff2)) - stale comment:** expert_source.cpp ~6646 says STRATA_LAYER_CAPTURE is "default on ... =0 turns it off"; the code (generate.cpp 10633) and serve/lowrank_maint.py require `=1`. Fix the comment, not the default.
- **F-07 (low, fixed (4092a4a21)) - `STRATA_ADAPT_CHECKRES` is presence-based** (generate.cpp 10971, 18994): `=0` turns the debug check ON. Make it `[0]=='1'` like its siblings.
- Rejected after checking: "STRATA_ADAPT_OVERLAP is presence-based" - it is `e && e[0]=='1'` (generate.cpp 1946); the presence test at 10906 only chooses a log line. "pf_parts not found" - it is at generate.cpp 9006.
- Noted, not acted on: RouterLookahead reads `host_res` without a lock (a stale read only changes a prefetch guess, not computed results); 174 env names read at several sites (audit later, only where defaults differ); the 700 MiB literal in three places.

## From P0-C (security recon), checked by the lead
- **S-01 (medium, open) - Docker example publishes a keyless server on every host interface.** docker-entrypoint.sh:24-26 (`HOST=0.0.0.0`, empty `API_KEY`) is right inside a container, but the Dockerfile's run example uses `-p 8080:8080`. Fix: the example uses `-p 127.0.0.1:8080:8080` and says how to open it with a key. No behaviour change in the entrypoint (users rely on it).
- **S-02 (medium, open) - config files holding the API key are created with the default umask** (setup.py write_config ~4647, serve/runconfig.py save). Fix: create them 0600 (os.open with mode, then replace); keep the .bak 0600 too.
- **S-03 (low, open) - `max_tokens: 1e400` -> `int(inf)` OverflowError** (server.py ~6816, 7067), uncaught, after the model is loaded. Fix: validate finite non-negative integers in one helper, 400 on bad input.
- **S-04 (low, open) - predictable /tmp names opened "wb"** for TIMELINE_CTL / RESIDENCY (server.py ~3111-3114, 3187): symlinks are followed on a shared host. Fix: open with O_NOFOLLOW|O_CREAT and 0600, or a per-user directory.
- Not a defect (by design, documented): with no api_key every client is trusted; "no Origin = not a browser" (server.py 6681-6688) is the CSRF guard for browsers only. SECURITY.md's wording "only as JSON from Strata's own page" should say it guards against other web pages, not other programs - doc fix only.
- Noted, not acted on: local-path and http(s) image sources for an authorised client (the key holder is the owner; SSRF is blind); /health unauthenticated detail (document it); decompression bombs in pass-through formats depend on the external vision decoder (not in repo); supply-chain pins (llama.cpp zip without SHA-256).

## Checked and fine (double slots, 24 Nov)
- Sized caches are one contiguous allocation (`open_sized` -> one `open` of the summed size; VMM reserves one VA range), and fill/verify clamp to the LARGEST blob (`blob_ = mx`), so the spill copy is not truncated.
- The second victim goes through the same protocol as the first (`host_res` cleared and `res_dirty_idx` pushed in the same places, dma-batch path included); cross-layer swaps run only with no overlap, no fence, one GPU.
- The prompt-loan boundary `pf_parts[0].first` is fixed at start, so the straddle check at formation is effective; a double inside the loan region is overwritten during the prompt and restored by the refill, which writes the owner's full blob.
- `admit()` only advances a cursor, so it never hands out a reserved neighbour slot.
- The monitor only checks slots whose owner is in `host_res`, i.e. whose copy has landed: no false alarms from copies in flight.
