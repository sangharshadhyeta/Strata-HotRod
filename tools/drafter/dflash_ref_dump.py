"""Stage A of the native DFlash 2 drafter: the reference model's rounds, dumped for the C++ port to be checked against (22 Oct 2026).

Runs z-lab's DFlash2DraftModel on the CPU (dflash_offline.load_dflash, the drafter and the shared embedding and head from
qwen-head) over the first request of an engine trace, replaying its first rounds as dflash_offline.replay drives make_dflash_proposer.
Per round k, writes <out>/round<k>.npz with:
  start, anchor, ctx_lo, ctx_hi, cache_len   the round's positions and the draft cache's length before the forward
  ctx_bits      uint16 [ctx, L*N]   the features fed (bf16 bits, exactly as the trace holds them)
  ids           int64 [block]       the noise ids (anchor, then mask tokens)
  pos           int64 [ctx+block]   the position ids
  th_norm       float32 [ctx, H]    hidden_norm(fc(target_hidden)): the context every layer's k/v projections read
  layer<i>      float32 [block, H]  the output of decoder layer i over all noise rows (i = 0..4)
  final         float32 [block-1, H] norm(last layer) over rows 1.. : what propose() reads
  cand          int64 [block-1, K]  the selector's top-K candidate ids per row (topk over the head's logits)
  draft         int64 [block-1]     the selector's path: the draft tokens
Usage (CPU only; the GPU is busy):
    /home/models/dflash2/venv/bin/python -I tools/drafter/dflash_ref_dump.py /mnt/nvme/strata-data/traces/draft1.bin \
        --out /mnt/nvme/strata-data/traces/dflash_ref --rounds 6 --block 8
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dflash_offline as off  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace")
    ap.add_argument("--drafter", default="/home/models/dflash2/drafter")
    ap.add_argument("--head", default="/home/models/dflash2/qwen-head")
    ap.add_argument("--out", default="/mnt/nvme/strata-data/traces/dflash_ref")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--block", type=int, default=8)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from dflash.model import _crop_to, _make_cache

    layers, N, recs = off.read_trace(a.trace)
    seqs = off.sequences(recs)
    seq = seqs[0]
    print("trace layers %s, n_embd %d, request 0: %d positions, generated from %d" % (
        layers, N, len(seq["tokens"]), seq["gen_start"]))
    st = off.load_dflash(a.drafter, a.head, layers, "cpu")
    model, embed, head, scale, mask_id = st["model"], st["embed"], st["head"], st["scale"], st["mask_id"]
    print("model layers %d, block %d, selector top-k %d, scale %s, mask %d" % (
        len(model.layers), model.block_size, model.candidate_selector.top_k, scale, mask_id))
    os.makedirs(a.out, exist_ok=True)

    feats = torch.from_numpy((seq["feats"].astype(np.uint32) << 16).view(np.float32))
    cache = _make_cache(model.config)
    captured = {}
    for i, layer in enumerate(model.layers):
        layer.register_forward_hook(lambda m, inp, out, i=i: captured.__setitem__("layer%d" % i, out.detach()))
    counter = {"k": 0}

    @torch.inference_mode()
    def propose(ctx_lo, ctx_hi, start, anchor, block):
        k = counter["k"]
        counter["k"] += 1
        th_bits = seq["feats"][ctx_lo:ctx_hi]
        th = feats[ctx_lo:ctx_hi].to(dtype=torch.bfloat16)[None]
        ids = torch.full((1, block), mask_id, dtype=torch.long)
        ids[0, 0] = anchor
        noise = F.embedding(ids, embed) * scale
        pos = torch.arange(start - th.shape[1], start + block)[None]
        cache_len = cache.get_seq_length()
        th_norm = model.hidden_norm(model.fc(th))
        captured.clear()
        hidden_full = model(target_hidden=th, noise_embedding=noise, position_ids=pos, past_key_values=cache, use_cache=True)
        hidden = hidden_full[:, 1 - block:, :]
        _crop_to(cache, start)
        logits = model.compute_logits(hidden, head)
        cand = torch.topk(logits, model.candidate_selector.top_k, dim=-1, sorted=False)[1]
        draft, _, _ = model.propose(hidden, ids[:, 0], head, 0.0)
        out = dict(start=start, anchor=anchor, ctx_lo=ctx_lo, ctx_hi=ctx_hi, cache_len=cache_len,
                   ctx_bits=th_bits, ids=ids[0].numpy(), pos=pos[0].numpy(), th_norm=th_norm[0].float().numpy())
        for i in range(len(model.layers)):
            out["layer%d" % i] = captured["layer%d" % i][0].float().numpy()
        out["final"] = hidden[0].float().numpy()
        out["cand"] = cand[0].numpy()
        out["draft"] = draft[0].numpy()
        path = os.path.join(a.out, "round%d.npz" % k)
        np.savez(path, **out)
        print("round %d: start %d anchor %d ctx %d..%d cache %d -> draft %s (%s)" % (
            k, start, anchor, ctx_lo, ctx_hi, cache_len, out["draft"].tolist(), path))
        return draft[0].tolist()

    rounds = off.replay(seq, propose, a.block, max_rounds=a.rounds)
    print("replayed %d rounds, tokens produced per round %s" % (len(rounds), rounds))


if __name__ == "__main__":
    main()
