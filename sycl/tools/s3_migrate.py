"""s3_migrate.py STEP: S3 of the SYCL port (sycl/PORT-C.md) - our .cu changes since the copies' base into sycl/.
  list     : the .cu files changed or added since fde1d795e
  base     : a worktree at fde1d795e ($SYCL_S3_DIR/base) and SYCLomatic over its changed .cu -> base_out
  new      : SYCLomatic over the same .cu in this checkout (HEAD) -> new_out
  merge    : 3-way merge (merge_upstream.py's rule, .cu only) into sycl/; new .cu get the dpct output as is
"""
import os, re, shutil, subprocess, sys

W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository
OUT = os.environ.get("SYCL_S3_DIR", os.path.join(os.path.dirname(W), "sycl-s3"))   # work directory
BASE = "fde1d795e"
env = dict(os.environ)   # set CUDA_HEADERS (the CUDA include directory) and put SYCLomatic's dpct on PATH


def git(*a, cwd=W):
    return subprocess.run(["git", "-C", cwd] + list(a), capture_output=True, text=True, check=True).stdout


def changed():
    files = [f for f in git("diff", "--name-only", BASE, "HEAD", "--", "src").split() if f.endswith(".cu")]
    skip = {"src/prefill/moe_mmq.cu", "src/prefill/ggml_cuda_host.cu"}   # (migrate.sh's own skip list: ggml-cuda sources)
    return [f for f in files if f not in skip and os.path.exists(f"{W}/{f}")]


def migrate(repo, out, files):
    os.makedirs(out, exist_ok=True)
    e = dict(env, MIGRATE_ONLY=" ".join(files))
    r = subprocess.run(["nice", "-n", "15", "bash", f"{W}/sycl/tools/migrate.sh", repo, out], env=e,
                       capture_output=True, text=True)
    open(f"{out}.log", "w").write(r.stdout + r.stderr)
    print((r.stdout + r.stderr)[-1500:])


step = sys.argv[1]
if step == "list":
    fs = changed()
    print(len(fs), "changed/new .cu:")
    for f in fs:
        print(("  NEW  " if not os.path.exists(f"{W}/sycl/{f[:-3]}.dp.cpp") else "       ") + f)
elif step == "base":
    wt = f"{OUT}/base"
    if not os.path.exists(wt):
        os.makedirs(OUT, exist_ok=True)
        git("worktree", "add", "--detach", wt, BASE)
    fs = [f for f in changed() if os.path.exists(f"{wt}/{f}")]
    migrate(wt, f"{OUT}/base_out", fs)
elif step == "new":
    migrate(W, f"{OUT}/new_out", changed())
elif step == "merge":
    B, N = f"{OUT}/base_out", f"{OUT}/new_out"
    pat = re.compile(r"\b([A-Za-z0-9_]+)_([0-9a-f]{6})\b")
    H = lambda t: pat.sub(lambda m: m.group(1) + "_HASH", t)
    def hashes(t):
        m = {}
        for a, h in pat.findall(t): m.setdefault(a, set()).add(h)
        return {a: next(iter(v)) for a, v in m.items() if len(v) == 1}
    for f in changed():
        rel = f[:-3] + ".dp.cpp"
        pf, bf, nf = f"{W}/sycl/{rel}", f"{B}/{rel}", f"{N}/{rel}"
        if not os.path.exists(nf):
            print("NO-DPCT-OUTPUT", rel); continue
        if not os.path.exists(pf):
            os.makedirs(os.path.dirname(pf), exist_ok=True); shutil.copy(nf, pf); print("NEWCOPY", rel); continue
        if not os.path.exists(bf):
            print("NO-BASE (new to the dpct output)", rel); continue
        bt, nt = open(bf).read(), open(nf).read()
        if H(bt) == H(nt):
            print("NOCHANGE", rel); continue
        ph, bh = hashes(open(pf).read()), hashes(bt)
        nt = pat.sub(lambda m: f"{m.group(1)}_{bh[m.group(1)]}" if m.group(1) in bh else m.group(0), nt)
        canon = lambda t: pat.sub(lambda m: f"{m.group(1)}_{ph[m.group(1)]}" if m.group(1) in ph else m.group(0), t)
        open(f"{OUT}/_b", "w").write(canon(bt))
        open(f"{OUT}/_n", "w").write(canon(nt))
        r = subprocess.run(["git", "merge-file", "--diff3", "-L", "sycl", "-L", "base", "-L", "ours-now", pf,
                            f"{OUT}/_b", f"{OUT}/_n"]).returncode
        print("MERGED" if r == 0 else f"CONFLICT {r}", rel)
