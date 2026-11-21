# Port ledger

One line per upstream commit (or group) and per idea from another fork: what was done and why. `tools/upstream_triage.py` reads this file and
leaves out any commit whose full or 9-character hash appears here. Verdicts: ported, adapted, skipped. Always give the reason and the test result.
Skipped items are recorded so they are not weighed again. Nothing is copied from other forks; ideas are written by hand (docs/PORTING_STRATAQ.md).

Upstream = Niko1221/Strata (`origin`). Our base: 477ac13cb (16 Oct 2026). First triage 21 Nov: 4,729 commits after the base.

| Source | Item | Verdict | Reason / test |
|---|---|---|---|
| upstream | ADAPT_FETCH | skipped | the resident-RAM exchange invariant forbids a local port (see docs/SCHEDULER.md) |
| qwertie/Strataq | two conversations allocate three context windows | confirmed for `--batch` (21 Nov) | `--batch 2` at 200K: expert cache 1,473 -> 885 slots, 4.7 GiB less RAM; decode 25 -> 15 tok/s for one request, 20 in all for two at once; fix = elastic slot K/V + main session as slot 0 (docs/PORTING_STRATAQ.md) |
| qwertie/Strataq | serve once the dense weights are loaded | planned | step 1 |
| qwertie/Strataq | live settings with tooltips | planned | step 3 (read-only first) |
| qwertie/Strataq | one shared engine, adapter per model | planned | step 4 (new files only) |
| qwertie/Strataq | one low-RAM mode that scales | planned | step 5 |
