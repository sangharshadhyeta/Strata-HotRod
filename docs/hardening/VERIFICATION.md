# Verification: branch `hardening` (505e2065c)

Integrated regression check of the merged `hardening` branch against its base `9279a5e90` (ratrod.v1.2).
Source was not changed by this check. Findings and fixes are listed in [FINDINGS.md](FINDINGS.md).

## Summary

| Area | Status | Evidence |
|---|---|---|
| Production build | PASS | CUDA `strata` target built on the engine branch (same engine sources). CPU test build of this branch: `cmake --build` exit 0, no `error` or `warning:` lines. |
| Type checking | BLOCKED | No type checker installed. Byte-compile only (see Python tests). |
| Linting | BLOCKED | No linter installed. |
| Unit tests (C++) | PASS | ctest: 61 tests, 60 passed, 1 skipped (`expert_multi_test`, no AVX-512), 0 failed. |
| Python tests | PASS (no regressions) | Every failure on merged also fails on base (same test names); the two V-1 regressions were a flaky pre-existing test and a test injecting at a call S-02 removed (fixed). Details below. |
| Integration tests | BLOCKED | The live-server test (`tools/test_conversation_cache_http_smoke.py`) and `bench/results/**/test_replay.py` need a live server and are skipped by design. The mock-engine tests are counted under Python tests. |
| E2E tests | BLOCKED | Needs a real model on a GPU. The only GPU serves the user's live server. |
| Security audit | PASS (open items) | 8 of 11 findings fixed; S-04 partly fixed; F-02 and F-03 open (both low). See [FINDINGS.md](FINDINGS.md). |
| Runtime verification | BLOCKED | Needs a real model. The only GPU serves the user's live server. |
| Byte-compile | PASS | 284 `.py` files under `serve/` and `tools/` parse (see note on `tools/pack_index.py`). |

## Python tests

Each file was run as `python <file>` with `unshare -rn` (no network), a fresh empty HOME/XDG/APPDATA directory
per run, a 600 s timeout, and a 4 GB memory scope. Totals below count only the files run in this form.

| | Base `9279a5e90` | Merged `505e2065c` |
|---|---|---|
| Files run | 135 | 136 (includes new `serve/test_write_lock.py`) |
| Exit 0 | 89 | 88 |
| Exit 1 | 46 | 48 |
| Tests in files that print `Ran N tests` | 1413 (89 files) | 1413 (89 files) |

Exit 1 in the launch form is partly an invocation artifact: 41 `serve/` and `tools/` files import the `serve` or
`tools` package and fail with `ModuleNotFoundError` when run as `python serve/x.py`. Their own docstrings give the
`python -m unittest serve.x` form, so they were also run that way on both trees (cwd = tree root):

| | Base | Merged |
|---|---|---|
| Files run as `-m unittest` | 40 | 41 |
| Files OK | 40 (331 tests) | 41 (339 tests) |
| Files FAILED | 0 | 0 |

The 8 extra tests on merged are new tests in `serve/test_write_lock.py` (3), `serve/test_runconfig.py` (+2),
`serve/test_timeline.py` (+2) and `serve/test_request_hardening.py` (+1).

### Regressions (pass on base, fail on merged) - none after the lead's follow-up

The two reported by V-1, resolved by the lead:

1. **`serve/test_responses.py`** `NoLeakedSampler.test_server_close_ends_the_sampler` (line 284, `1 != 2`): **not a
   regression - flaky on the base too.** Four runs each (offline, fresh HOME): base 3 OK / 1 FAILED, merged 4 OK. The
   server agent's own baseline run on the base also failed it once. Pre-existing flaky thread-count test; open.

2. **`tools/test_setup_risk.py`** `test_write_config_leaves_no_partial_file`: a real gap - the test injected the full
   disk through `Path.write_text`, which S-02's `write_config` no longer calls, so the failure path went untested.
   Fixed in the follow-up commit: the test now injects the full disk at the write `write_config` really does
   (`os.fdopen(...).write`), and also asserts that no partial `.tmp` is left; `write_config` removes its temp file
   when the write fails. That test passes on merged; the file's other 5 failures are the same 5 test names on base and
   merged (pre-existing, below).

### Pre-existing failures (fail on base and merged)

| File | Result (same on both trees unless noted) | First failing assertion / reason |
|---|---|---|
| `bench/test_gfx906_qsa_query_swizzle_guard.py` | exit 1, no `Ran` line | `RuntimeError: production query-swizzle dispatch changed; update the test` (raised at import) |
| `tools/test_iq_pack.py` | exit 1, no `Ran` line | `llama.cpp's gguf-py was not found` (third_party not populated in this checkout) |
| `tools/test_setup_golden.py` | `Ran 4`, 24 failures | config dict mismatch, e.g. `--pack <T>/data/packs/iq3_s` expected, different args produced (`test_enter_for_every_question`, `test_yes`, many config variants) |
| `tools/test_setup_engine_hash.py` | `Ran 30`, 2 failures | `SystemExit not raised`; `'keeping the engine that is installed' not found` (the run printed `urlopen ... Name or service not known`; probably the sandbox has no network) |
| `tools/test_setup_hotrod_predictor.py` | `Ran 5`, 2 failures, 1 error | `FileNotFoundError` for the predictor the test expects to download; `'wrong SHA-256' not found` (needs a download; probably the sandbox has no network) |
| `tools/test_setup_risk.py` | `Ran 40`, 5 failures (base) | `test_an_empty_config_is_skipped` (3 variants), `test_the_earlier_install_keeps_its_context`, `test_the_newest_readable_config_is_used` |

The network-dependent rows were not rerun with network access, so their cause is not confirmed.

### Other files

- `tools/drafter/test_dflash_offline.py`, `tools/predictor/test_export_la_model.py`,
  `tools/predictor/test_token_table_check.py`, `tools/predictor/test_trace_reader.py`: exit 0 on both trees, script
  style (no unittest summary).
- `serve/test_security.py` was run. Its grep matched the string `8095` and it has no `/home/Projects/Strata/data`
  hit. The matches are only Host/Origin header literals; the test binds `port=0` and uses a mock engine. Under
  `unshare -rn` the test has no route to any host port, so it cannot reach the live server.

### Skipped by design or by filter

- `serve/test_winjob.py`: Windows-only.
- `tools/test_conversation_cache_http_smoke.py`: needs a live server.
- `bench/results/**/test_replay.py`: needs a live server (excluded by the `bench/results` filter).

## C++ tests

`cmake -DSTRATA_BUILD_TESTS=ON` on the CPU-only configuration, built under the capped, locked build wrapper
(`MemoryMax=4G`, `nice -n 15`, `-j4`). Then `ctest -j2 --output-on-failure` under `unshare -n`:

```
100% tests passed, 0 tests failed out of 61
Skipped: 52 - expert_multi_test (no AVX-512)
```

## Byte-compile

`ast.parse` over the 284 `.py` files under `serve/` and `tools/` (merged, excluding `__pycache__`):

- Run literally as `open(f).read()`: fails on `tools/pack_index.py`, which starts with a UTF-8 BOM. The BOM is
  unchanged from base (same first bytes at `9279a5e90`), so this is not introduced by the merge.
- Run with a byte read (`open(f, 'rb')`, which accepts the BOM the way the interpreter does): all 284 files parse.
  This is the result reported.

## Notes on method

- Scripts and logs are kept outside the repository (in the job's scratch directory); none are committed.
- Each test run was wrapped in `systemd-run --scope` with `MemoryMax=4G` outside the spec'd
  `timeout 600 unshare -rn` command, because the box had about 3 GB free with the live server running.
- `PYTHONDONTWRITEBYTECODE=1` was set, so the runs wrote no `__pycache__` into the tree.
- The base worktree was created detached at `9279a5e90` and removed after the runs.
