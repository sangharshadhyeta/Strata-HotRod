"""Would the DFlash 2 drafter beat the MTP on this PC?  Replays it offline on an engine trace (22 Oct 2026).

The drafter: incoai/Qwen3.8-Flash-Next-DFlash2 (Inco AI, 19 Oct 2026; 5 small layers, 1.2 GB in bf16), trained on the completed outputs of
layers 5, 15, 25, 35 and 45 averaged over the four hyper-connection streams.  It drafts a whole block in one pass from those features
of the context and the block's first token; a selector picks one path through its top-16 candidates.

The engine's STRATA_DRAFT_TRACE=<file> records, for every verify window, its rows' tokens, the model's pick after each row and the
residual after those layers (src/core/verify.cpp).  From it this script rebuilds each request's text and the features of every position,
then
  * measures today's drafting (the MTP, and prompt lookup where it fired): tokens per window = accepted drafts + 1, over the windows
    that generate (the prompt's windows are left out);
  * replays the drafter over the same text, round by round as z-lab's dflash_generate runs it (the context's features, the block's
    first token, mask tokens after it), and counts what greedy verification would accept: tokens per round.
Greedy decoding is lossless, so the replay sees exactly the text a DFlash run would have produced; trace greedy requests only.

Record the trace (on the box; one GPU; greedy; the prompt through the verify windows so its features are traced too - no --prefill):
    STRATA_DRAFT_TRACE=/mnt/nvme/strata-data/traces/draft1.bin  <the usual engine command without --prefill>
Download (once):
    hf download incoai/Qwen3.8-Flash-Next-DFlash2 --local-dir dflash2
    hf download Qwen/Qwen3.8-Flash-Next model-00130-of-00131.safetensors model-00131-of-00131.safetensors --local-dir qwen-head
      (the token embedding and the output head the drafter shares with the model: 4.2 GB)
    python -m pip install "dflash[local]" safetensors
Run:
    python dflash_offline.py draft1.bin --drafter dflash2 --head qwen-head --block 4 8
Without --drafter it only parses the trace and prints today's numbers.
"""
import argparse
import os
import struct
import sys

import numpy as np


def read_trace(path):
    """-> (layers, n_embd, records): records are dicts pos0, T, tokens, picks, feats (uint16 bf16 [T, L * n_embd])."""
    with open(path, "rb") as f:
        buf = f.read()
    if len(buf) < 16 or buf[:4] != b"DFT1":
        raise ValueError("not a STRATA_DRAFT_TRACE file")
    L, N, _hc = struct.unpack_from("<III", buf, 4)
    layers = list(struct.unpack_from("<%di" % L, buf, 16))
    off = 16 + 4 * L
    recs = []
    while off + 12 <= len(buf):
        pos0, T = struct.unpack_from("<qi", buf, off)
        size = 12 + 8 * T + 2 * T * L * N
        if T < 1 or off + size > len(buf):
            break                                   # a window cut by the end of a still-growing file
        p = off + 12
        tokens = np.frombuffer(buf, np.int32, T, p); p += 4 * T
        picks = np.frombuffer(buf, np.int32, T, p); p += 4 * T
        feats = np.frombuffer(buf, np.uint16, T * L * N, p).reshape(T, L * N)
        recs.append(dict(pos0=pos0, T=T, tokens=tokens, picks=picks, feats=feats))
        off += size
    return layers, N, recs


def accepted(rec):
    """Drafts the window's verification kept: rows 1.. that equal the model's pick after the row before."""
    a = 0
    while a + 1 < rec["T"] and rec["tokens"][a + 1] == rec["picks"][a]:
        a += 1
    return a


def sequences(recs):
    """Split the windows into requests (a window that starts before the previous one is a new request) and rebuild each:
    dict(tokens [n] int32 committed text, feats [n, L*N] uint16, first, gen_start, windows (that request's records), gen_windows)."""
    groups, cur = [], []
    for r in recs:
        if cur and r["pos0"] < cur[-1]["pos0"]:
            groups.append(cur)
            cur = []
        cur.append(r)
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        first = g[0]["pos0"]
        last = g[-1]
        end = last["pos0"] + accepted(last) + 1           # the last window commits rows 0..accepted
        n = end - first
        D = g[0]["feats"].shape[1]
        toks = np.full(n, -1, np.int32)
        feats = np.zeros((n, D), np.uint16)
        for r in g:                                       # later windows rewrite the positions rejected drafts held
            lo = r["pos0"] - first
            k = min(r["T"], n - lo)
            if k <= 0 or lo < 0:
                continue
            toks[lo:lo + k] = r["tokens"][:k]
            feats[lo:lo + k] = r["feats"][:k]
        # the first generating window: from it on, every window starts with the pick its predecessor made after the row before
        def cont(i):
            a, b = g[i - 1], g[i]
            j = b["pos0"] - a["pos0"] - 1
            return 0 <= j < a["T"] and b["tokens"][0] == a["picks"][j]
        k0 = len(g)
        while k0 - 1 >= 1 and cont(k0 - 1):
            k0 -= 1
        if k0 == 1 and first > 0:                         # no prompt windows traced (--prefill): the first window generates too
            k0 = 0
        gen_windows = g[k0:] if k0 < len(g) else []
        gen_start = gen_windows[0]["pos0"] - first if gen_windows else n
        out.append(dict(tokens=toks, feats=feats, first=first, gen_start=gen_start, windows=g, gen_windows=gen_windows))
    return out


def today_stats(seqs):
    """Tokens per generating window today (accepted drafts + 1) and the windows' sizes."""
    per, sizes = [], []
    for s in seqs:
        for r in s["gen_windows"]:
            per.append(accepted(r) + 1)
            sizes.append(r["T"])
    return dict(windows=len(per), tokens_per_window=float(np.mean(per)) if per else 0.0,
                mean_size=float(np.mean(sizes)) if sizes else 0.0)


def replay(seq, propose, block, max_rounds=None):
    """Rounds as dflash_generate runs them, over the committed text.  propose(ctx_lo, ctx_hi, start, anchor, block) returns block - 1
    draft tokens for positions start+1..; the context features new to the drafter are those of [ctx_lo, ctx_hi).  -> tokens per round."""
    toks = seq["tokens"]
    n = len(toks)
    s = seq["gen_start"]
    if s == 0:                                            # no prompt features traced (--prefill): the first position is the context
        s = 1
    ctx = 0
    produced_all = []
    while s + 1 < n and (max_rounds is None or len(produced_all) < max_rounds):
        b = min(block, n - s)
        draft = propose(ctx, s, s, int(toks[s]), b)
        acc = 0
        while acc < b - 1 and int(draft[acc]) == int(toks[s + 1 + acc]):
            acc += 1
        produced = min(acc + 1, n - s - 1)
        produced_all.append(produced)
        ctx = s
        s += produced
    return produced_all


def load_dflash(drafter_dir, head_dir, layers, device):
    """The drafter, the embedding and the head, once (z-lab's dflash package; imported here so parsing needs no torch)."""
    import torch
    from dflash.model import DFlash2DraftModel, _draft_value
    from safetensors import safe_open

    model = DFlash2DraftModel.from_pretrained(drafter_dir, torch_dtype=torch.bfloat16).to(device).eval()
    want = list(model.target_layer_ids)
    if want != list(layers):
        print("NOTE: the drafter reads layers %s, the trace holds %s - the replay feeds the trace's as they are" % (want, layers))
    tensors = {}
    for fn in sorted(os.listdir(head_dir)):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(head_dir, fn), "pt", device="cpu") as f:
                for name in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
                    if name in f.keys():
                        tensors[name] = f.get_tensor(name).to(device=device, dtype=torch.bfloat16)
    if len(tensors) != 2:
        raise SystemExit("--head: the token embedding and lm_head were not found in %s" % head_dir)
    embed = tensors["model.language_model.embed_tokens.weight"]
    head = torch.nn.Linear(embed.shape[1], embed.shape[0], bias=False, device=device, dtype=torch.bfloat16)
    head.weight.data = tensors["lm_head.weight"]
    return dict(model=model, embed=embed, head=head, scale=float(_draft_value(model.config, "input_embedding_scale", 1.0)),
                mask_id=int(model.mask_token_id), device=device)


def make_dflash_proposer(st, seq):
    """A proposer over one request's features, with its own draft cache."""
    import torch
    import torch.nn.functional as F
    from dflash.model import _crop_to, _make_cache

    model, embed, head, scale, mask_id, device = st["model"], st["embed"], st["head"], st["scale"], st["mask_id"], st["device"]
    feats = torch.from_numpy((seq["feats"].astype(np.uint32) << 16).view(np.float32))   # bf16 bits -> float32 [n, L*N]
    cache = _make_cache(model.config)

    @torch.inference_mode()
    def propose(ctx_lo, ctx_hi, start, anchor, block):
        th = feats[ctx_lo:ctx_hi].to(device=device, dtype=torch.bfloat16)[None]
        ids = torch.full((1, block), mask_id, dtype=torch.long, device=device)
        ids[0, 0] = anchor
        noise = F.embedding(ids, embed) * scale
        pos = torch.arange(start - th.shape[1], start + block, device=device)[None]
        hidden = model(target_hidden=th, noise_embedding=noise, position_ids=pos, past_key_values=cache, use_cache=True)[:, 1 - block:, :]
        _crop_to(cache, start)
        draft, _, _ = model.propose(hidden, ids[:, 0], head, 0.0)
        return draft[0].tolist()

    return propose


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace")
    ap.add_argument("--drafter", help="the downloaded incoai/Qwen3.8-Flash-Next-DFlash2 folder")
    ap.add_argument("--head", help="folder with the model's shards holding embed_tokens and lm_head")
    ap.add_argument("--block", type=int, nargs="+", default=[4, 8], help="block sizes to replay (4 = Inco's test, 8 = trained)")
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    layers, N, recs = read_trace(a.trace)
    seqs = sequences(recs)
    print("trace: %d windows, %d requests, layers %s, n_embd %d" % (len(recs), len(seqs), layers, N))
    for i, s in enumerate(seqs):
        print("  request %d: positions %d..%d, generated from %d (%d generating windows)%s" % (
            i, s["first"], s["first"] + len(s["tokens"]) - 1, s["first"] + s["gen_start"], len(s["gen_windows"]),
            "" if s["first"] == 0 else "  (no prompt features: run without --prefill)"))
    t = today_stats(seqs)
    print("today (MTP and lookup): %.2f tokens per window over %d generating windows, mean window %.2f tokens" % (
        t["tokens_per_window"], t["windows"], t["mean_size"]))
    if not a.drafter:
        return
    if not a.head:
        sys.exit("--head is needed with --drafter")
    st = load_dflash(a.drafter, a.head, layers, a.device)
    for block in a.block:
        rounds = []
        for s in seqs:
            if s["gen_start"] >= len(s["tokens"]) - 1:
                continue
            rounds += replay(s, make_dflash_proposer(st, s), block, a.max_rounds)
        if rounds:
            print("DFlash 2, block %d: %.2f tokens per round over %d rounds (today: %.2f per window)" % (
                block, float(np.mean(rounds)), len(rounds), t["tokens_per_window"]))


if __name__ == "__main__":
    main()
