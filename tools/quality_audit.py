#!/usr/bin/env python3
"""quality_audit.py - a cheap, per-model quality table for the tail rules (and, later, any per-request setting).

It scores one fixed text (tools/audit_text.json, about 4,000 tokens of prose, docs, code and worked arithmetic) through a RUNNING server under several settings and
compares each setting with a reference: the engine's teacher-forced scorer (STRATA_LOGPOS, docs/UNSLOTH_Q4.md) records the model's distribution at every position, so the
result is deterministic and has far less noise than sampling a few dozen problems.

The scorer only sees text the engine reads through its verify windows, which is the NEW tail of a conversation (at most --short-read tokens).  So the text goes in as a
growing conversation: request k holds messages 1..k, the engine resumes from the checkpoint after message k-1 and reads message k through the windows, where it is scored.
Every setting gets its own conversation, started with a short numbered system line of the same length, so no cached state is shared between settings.

The server must have been started with
    STRATA_LOGPOS=<file> STRATA_LOGPOS_TOPK=64   and   --short-read 640 --adapt-every 100000
(--short-read so every message of the audit text is read through the verify windows, where the scorer lives; --adapt-every so the expert cache does not move between settings).
For an exact reference (no stand-ins, no skips) run once against an engine started that way with --save-ref FILE, then later runs compare against it with --ref FILE.
The settings are switched per request with the strata_tune keys tail_weight / tail_skip (docs/HOTROD-DETAILS.md), no restart.

Columns, against the first setting (the reference; it is run twice, the second run is the noise floor):
  dNLL   mean of -log p(target) for the setting minus the reference: the perplexity cost in nats per token (positive = worse)
  top1   share of positions where the most likely token is the reference's
  KL     mean KL(reference || setting) over the reference's top candidates plus one bucket for the rest
  tok/s  how fast the audit text was read through the verify windows

Usage: quality_audit.py --logpos FILE [--url http://127.0.0.1:8095] [--key KEY] [--out quality-table.json] [--configs "name:tail_weight=0.05,tail_skip=0;..."] [--save-ref F | --ref F]
"""
import argparse, json, math, os, sys, time, urllib.request, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIGS = ("base:tail_weight=0,tail_skip=0;base-repeat:tail_weight=0,tail_skip=0;rank7:tail_weight=0,tail_skip=7;rank3:tail_weight=0,tail_skip=3;"
                   "w02:tail_weight=0.02,tail_skip=0;w05:tail_weight=0.05,tail_skip=0;w10:tail_weight=0.10,tail_skip=0;w20:tail_weight=0.20,tail_skip=0;w30:tail_weight=0.30,tail_skip=0")


def parse_logpos(text):
    """Rows of a STRATA_LOGPOS file: position -> (target id, log p(target), most likely id, {id: log p} of the top candidates)."""
    rows = {}
    for line in text.splitlines():
        f = line.split("\t")
        if len(f) < 9:
            continue
        rows[int(f[0])] = (int(f[1]), float(f[2]), int(f[3]), {int(a): float(b) for a, b in (x.split(":") for x in f[8:])})
    return rows


def compare(ref, got):
    """(positions, dNLL, top1, KL) of `got` against `ref`, over the positions both scored."""
    pos = sorted(set(ref) & set(got))
    if not pos:
        return 0, float("nan"), float("nan"), float("nan")
    dn = sum(-got[p][1] + ref[p][1] for p in pos) / len(pos)
    t1 = sum(got[p][2] == ref[p][2] for p in pos) / len(pos)
    kl = 0.0
    for p in pos:
        pe, pa = ref[p][3], got[p][3]
        floor = min(pa.values()) if pa else -30.0
        rest_e = max(1e-12, 1.0 - sum(math.exp(v) for v in pe.values()))
        rest_a = max(1e-12, 1.0 - sum(math.exp(v) for v in pa.values()))
        k = sum(math.exp(lpe) * (lpe - pa.get(i, floor)) for i, lpe in pe.items())
        kl += k + rest_e * math.log(rest_e / rest_a)
    return len(pos), dn, t1, kl / len(pos)


def parse_configs(spec):
    out = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        name, _, kv = item.partition(":")
        tune = {}
        for pair in kv.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                tune[k.strip()] = float(v) if k.strip() == "tail_weight" else int(v)
        out.append((name.strip(), tune))
    return out


def tag_for(name):
    """A four-digit number from the setting's name: the same token count for every setting, a different cached prefix for each."""
    return 1000 + zlib.crc32(name.encode()) % 9000


def score_once(url, key, texts, tune, logpos, name):
    """Scores the whole text as a growing conversation of user messages; returns (the log-probability lines of all the requests, seconds, prompt tokens of the last request)."""
    lines, secs, ntok = [], 0.0, 0
    system = {"role": "system", "content": "Session %d." % tag_for(name)}
    for k in range(1, len(texts) + 1):
        start = os.path.getsize(logpos) if os.path.exists(logpos) else 0
        msgs = [system] + [{"role": "user", "content": t} for t in texts[:k]]
        body = json.dumps({"model": "audit", "messages": msgs, "max_tokens": 1, "temperature": 0, "stream": False,
                           "chat_template_kwargs": {"enable_thinking": False}, "strata_tune": tune}).encode()
        req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", body, {"Content-Type": "application/json", "Authorization": "Bearer " + key})
        t0 = time.time()
        reply = json.load(urllib.request.urlopen(req, timeout=3600))
        secs += time.time() - t0
        ntok = reply["usage"]["prompt_tokens"]
        if os.path.exists(logpos):
            with open(logpos, "rb") as f:
                f.seek(start)
                lines.append(f.read().decode("utf-8", "replace"))
    return "".join(lines), secs, ntok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logpos", required=True, help="the file the server writes STRATA_LOGPOS to")
    ap.add_argument("--url", default="http://127.0.0.1:8095")
    ap.add_argument("--key", default=os.environ.get("STRATA_API_KEY", ""))
    ap.add_argument("--text", default=os.path.join(HERE, "audit_text.json"))
    ap.add_argument("--configs", default=DEFAULT_CONFIGS)
    ap.add_argument("--out", default="")
    ap.add_argument("--save-ref", default="", help="write the first setting's log-probability lines here (for an exact reference run)")
    ap.add_argument("--ref", default="", help="compare every setting with the lines saved by an earlier --save-ref run")
    a = ap.parse_args(argv)
    messages = [m["content"] for m in json.load(open(a.text, encoding="utf-8")) if m.get("content")]
    configs = parse_configs(a.configs)
    if len(configs) < 2 and not a.ref:
        sys.exit("need at least two settings (or --ref)")
    ref_rows, table = (parse_logpos(open(a.ref).read()) if a.ref else None), []
    print("%-12s | %9s | %-9s | %-7s | %-7s | %-6s" % ("setting", "positions", "dNLL", "top1", "KL", "tok/s"))
    for name, tune in configs:
        text, secs, ntok = score_once(a.url, a.key, messages, tune, a.logpos, name)
        rows = parse_logpos(text)
        if ref_rows is None:
            ref_rows = rows
            if a.save_ref:
                open(a.save_ref, "w").write(text)
        n, dn, t1, kl = compare(ref_rows, rows)
        table.append({"setting": name, "tune": tune, "positions": n, "dnll": dn, "top1": t1, "kl": kl, "prompt_tokens": ntok, "tok_per_s": ntok / secs})
        print("%-12s | %9d | %+.5f | %.4f  | %.5f | %6.1f" % (name, n, dn, t1, kl, ntok / secs), flush=True)
    noise = next((abs(r["dnll"]) for r in table[1:] if r["tune"] == table[0]["tune"] and not a.ref), None)
    if noise is not None:
        print("noise floor (the reference repeated): |dNLL| = %.5f nats per token" % noise)
    if a.out:
        json.dump({"made": time.strftime("%Y-%m-%d %H:%M:%S"), "reference": table[0]["setting"], "noise_dnll": noise, "table": table}, open(a.out, "w"), indent=1)
        print("table written to", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
