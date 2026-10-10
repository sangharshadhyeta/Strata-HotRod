"""tools/layer_budget_profile.py - a profile whose per-layer budgets follow what each layer's router requests leave unmet (SPEC 13.17 to 13.25).

The shipped profile ranks all 24,576 (layer, expert) pairs globally by routing frequency; the cache holds its first N pairs, so each layer's
budget is however many of its pairs rank in the first N.  Measured (SPEC 20, LAYERS trace): layer 0 holds 22 experts and leaves 16% of the router's
requests unmet, layers 1, 2, 22 and 38 leave 7 to 11%, the median layer leaves 1 to 2%: the misses are far from homogeneous (13.25's aim).

This tool moves budget toward the layers that miss most, within the floor of 10 resident experts and a cap (13.21, 13.24):
    w_l  ~  resident_l * (unmet_l / mean unmet) ** beta       (clamped to [floor, cap], rescaled so the weights sum to N)
and re-ranks the pairs by the key (j + 0.5) / w_l, j = the pair's place in its own layer's base order, so the first N pairs hold about w_l of layer l
for ANY N near the measured one (the cache size varies with the free VRAM) and each layer keeps its own best experts.

    python tools/layer_budget_profile.py --log strata-iq3_s.log --beta 0.5 --out /path/profile-b05.bin     (the last LAYERVEC line of the log)
    python tools/layer_budget_profile.py --unmet "160 110 ..." --resident "22 35 ..." --beta 1.0 --out ...
"""
import argparse
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_LAYER, N_EXPERT = 48, 512
MAGIC = b"STRP"


def read_profile(path):
    blob = Path(path).read_bytes()
    if blob[:4] != MAGIC:
        raise SystemExit(f"{path}: not a Strata profile")
    ver, nl, ne, slots, n = struct.unpack_from("<5I", blob, 4)
    pairs = [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n)]
    return (ver, nl, ne, slots, n), pairs


def last_layervec(log):
    unmet = res = None
    for line in Path(log).read_text(errors="replace").splitlines():
        if "LAYERVEC unmet_permille" in line:
            a, b = line.split("LAYERVEC unmet_permille")[1].split("| resident")
            unmet, res = [int(x) for x in a.split()], [int(x) for x in b.split()]
    if unmet is None:
        raise SystemExit(f"{log}: no LAYERVEC line (run the engine with STRATA_TRACE_ADAPT=1)")
    return unmet, res


def budgets(unmet, resident, beta, floor, cap_mult):
    n = sum(resident)
    cap = cap_mult * n / len(resident)
    mean_u = sum(max(u, 0) for u in unmet) / len(unmet) or 1.0
    w = [r * (max(u, 0.5) / mean_u) ** beta for u, r in zip(unmet, resident)]
    for _ in range(50):                       # clamp to [floor, cap] and rescale to the same total
        s = sum(w)
        w = [min(cap, max(floor, x * n / s)) for x in w]
        if abs(sum(w) - n) < 0.5:
            break
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "data" / "expert-profile.bin"))
    ap.add_argument("--log")
    ap.add_argument("--unmet", help="48 numbers: unmet share x1000 per layer")
    ap.add_argument("--resident", help="48 numbers: resident experts per layer")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--floor", type=float, default=10.0)
    ap.add_argument("--cap", type=float, default=2.0, help="cap as a multiple of the equal share (13.24: about 2x)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    unmet, resident = last_layervec(a.log) if a.log else ([int(x) for x in a.unmet.split()], [int(x) for x in a.resident.split()])
    if len(unmet) != N_LAYER or len(resident) != N_LAYER:
        raise SystemExit("48 values per layer are needed")
    hdr, pairs = read_profile(a.base)
    w = budgets(unmet, resident, a.beta, a.floor, a.cap)
    place, seen = {}, [0] * N_LAYER
    for l, e in pairs:                         # j: the pair's place in its own layer's base order
        place[(l, e)] = seen[l]
        seen[l] += 1
    ranked = sorted(pairs, key=lambda p: ((place[p] + 0.5) / w[p[0]], p[0], p[1]))
    blob = MAGIC + struct.pack("<5I", *hdr) + b"".join(struct.pack("<HH", l, e) for l, e in ranked)
    Path(a.out).write_bytes(blob)
    n = sum(resident)
    new = [0] * N_LAYER
    for l, e in ranked[:n]:
        new[l] += 1
    print(f"N = {n}; budgets per layer (old -> new), beta {a.beta}:")
    print("  " + " ".join(f"L{l}:{resident[l]}->{new[l]}" for l in range(N_LAYER) if abs(new[l] - resident[l]) >= 3))
    print(f"  min/median/max old {min(resident)}/{sorted(resident)[24]}/{max(resident)}, new {min(new)}/{sorted(new)[24]}/{max(new)}; wrote {a.out}")


if __name__ == "__main__":
    main()
