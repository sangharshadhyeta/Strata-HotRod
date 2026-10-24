"""The DFlash 2 drafter as a helper process for the live engine (STRATA_DFLASH=<socket>, 22 Oct 2026).

Run with the drafter's own venv, one per box (one GPU):
    /home/models/dflash2/venv/bin/python tools/drafter/dflash_server.py /run/user/$UID/strata-dflash.sock
Options: --drafter (default /home/models/dflash2/drafter), --head (default /home/models/dflash2/qwen-head), --device cuda.

The math is dflash_offline.py's make_dflash_proposer, streamed: the engine sends the features of its COMMITTED positions as they
are committed (FEED), and asks for a block of drafts from an anchor token (PROPOSE).  The draft KV cache is cropped to `start` after
every proposal, so the noise rows of a proposal never stay in it; the engine feeds only committed rows, so a rejected draft's
features never reach the context either.

Protocol (one client at a time; little endian; every message starts with a 20-byte header "<Iqii": op, a, b, c):
    op 1 RESET                              -> a new request: the context and the draft cache are emptied.   reply: i32 status
    op 2 FEED   a = pos0, b = n            + payload n * 5*2560 bf16 (uint16)  the features of positions pos0.. (in order)
                                                                             reply: i32 status
    op 3 PROPOSE a = start, b = anchor, c = block                             reply: i32 status, then (block-1) i32 tokens
Status: 0 ok, 1 bad request, 2 FEED is not at the next position, 3 PROPOSE asks for positions not fed yet.
The features of positions ctx..start-1 (ctx = the previous PROPOSE's start) are the context of a PROPOSE; the anchor is the token at
`start`, the block's rows are start..start+block-1 (mask tokens after the anchor), as in dflash_offline.py.
"""
import argparse
import os
import socket
import struct
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dflash_offline  # noqa: E402  (load_dflash: the reference loader, the same drafter and head)

LAYERS = [5, 15, 25, 35, 45]
HDR = struct.Struct("<Iqii")
STATUS = struct.Struct("<i")
OP_RESET, OP_FEED, OP_PROPOSE = 1, 2, 3


def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


class Session:
    """One request's context: the features fed so far (bf16 on the device, grown by doubling) and the draft cache."""

    def __init__(self, st):
        import torch
        from dflash.model import _make_cache
        self.torch = torch
        self.st = st
        self.D = int(st["embed"].shape[1]) * len(LAYERS)   # the drafter reads every traced layer, each n_embd wide
        self.feats = torch.empty((0, self.D), dtype=torch.bfloat16, device=st["device"])
        self.fed = 0      # positions with features
        self.ctx = 0      # positions the draft cache holds (the last PROPOSE's start)
        self.cache = _make_cache(st["model"].config)

    def reset(self):
        self.__init__(self.st)

    def feed(self, pos0, n, payload):
        torch = self.torch
        if pos0 != self.fed:
            return 2
        rows = np.frombuffer(payload, dtype=np.uint16, count=n * self.D).reshape(n, self.D)
        new = torch.from_numpy(rows.view(np.int16).copy()).view(torch.bfloat16).to(self.st["device"])
        if self.fed + n > self.feats.shape[0]:
            cap = max(1024, 2 * (self.fed + n))
            grown = torch.empty((cap, self.D), dtype=torch.bfloat16, device=self.st["device"])
            grown[: self.fed] = self.feats[: self.fed]
            self.feats = grown
        self.feats[self.fed: self.fed + n] = new
        self.fed += n
        return 0

    def propose(self, start, anchor, block):
        """dflash_offline.make_dflash_proposer's propose, over the features fed so far -> (status, block-1 tokens)."""
        from dflash.model import _crop_to
        torch = self.torch
        import torch.nn.functional as F
        st = self.st
        model, embed, head, scale, mask_id, device = st["model"], st["embed"], st["head"], st["scale"], st["mask_id"], st["device"]
        if block < 2 or start < self.ctx:
            return 1, []
        if start > self.fed:
            return 3, []
        with torch.inference_mode():
            th = self.feats[self.ctx: start][None]
            if th.shape[1] == 0:
                return 1, []
            ids = torch.full((1, block), mask_id, dtype=torch.long, device=device)
            ids[0, 0] = anchor
            noise = F.embedding(ids, embed) * scale
            pos = torch.arange(self.ctx, start + block, device=device)[None]
            hidden = model(target_hidden=th, noise_embedding=noise, position_ids=pos, past_key_values=self.cache,
                           use_cache=True)[:, 1 - block:, :]
            _crop_to(self.cache, start)
            draft, _, _ = model.propose(hidden, ids[:, 0], head, 0.0)
        self.ctx = start
        return 0, [int(t) for t in draft[0].tolist()]


def handle(conn, st):
    session = Session(st)
    D = session.D
    while True:
        head = recv_exact(conn, HDR.size)
        if head is None:
            return
        op, a, b, c = HDR.unpack(head)
        if op == OP_RESET:
            session.reset()
            conn.sendall(STATUS.pack(0))
        elif op == OP_FEED:
            payload = recv_exact(conn, max(0, b) * D * 2) if b > 0 else b""
            if payload is None:
                return
            status = session.feed(a, b, payload) if b > 0 else 1
            conn.sendall(STATUS.pack(status))
        elif op == OP_PROPOSE:
            status, toks = session.propose(a, b, c)
            out = STATUS.pack(status)
            if status == 0:
                out += struct.pack("<%di" % len(toks), *toks)
            conn.sendall(out)
        else:
            print("dflash_server: unknown op %d" % op, file=sys.stderr, flush=True)
            conn.sendall(STATUS.pack(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("socket", help="the Unix socket path to listen on")
    ap.add_argument("--drafter", default="/home/models/dflash2/drafter")
    ap.add_argument("--head", default="/home/models/dflash2/qwen-head")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    st = dflash_offline.load_dflash(a.drafter, a.head, LAYERS, a.device)
    if os.path.exists(a.socket):
        os.unlink(a.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(a.socket)
    srv.listen(1)
    print("dflash_server: listening on %s (drafter %s, device %s)" % (a.socket, a.drafter, a.device), flush=True)
    while True:
        conn, _ = srv.accept()
        try:
            handle(conn, st)
        except Exception:
            traceback.print_exc()
        finally:
            conn.close()


if __name__ == "__main__":
    main()
