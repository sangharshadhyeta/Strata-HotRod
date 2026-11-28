"""s3_merge2.py: the kernel 3-way merge with a consistent base: base = SYCLomatic(fde1d795e) + fixups.py, theirs =
SYCLomatic(HEAD) + fixups.py (same tool version, so formatting matches), ours = the SYCL copy (its hand work kept).
New .cu (no copy) get SYCLomatic(HEAD) + fixups as they are.  Prints the conflict count per file."""
import os, shutil, subprocess

W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository
S3 = os.environ.get("SYCL_S3_DIR", os.path.join(os.path.dirname(W), "sycl-s3"))   # work directory
for name in ("base", "new"):
    T = f"{S3}/{name}_fixed"
    shutil.rmtree(T, ignore_errors=True)
    shutil.copytree(f"{S3}/{name}_out", T)
    os.makedirs(f"{T}/tools", exist_ok=True)
    shutil.copy(f"{W}/sycl/tools/fixups.py", f"{T}/tools/fixups.py")
    subprocess.run(["python3", "-I", f"{T}/tools/fixups.py"], capture_output=True, text=True)
B, N = f"{S3}/base_fixed", f"{S3}/new_fixed"
import re
NUM = re.compile(r"DPCT(\d{4}):\d+:")
HASH = re.compile(r"\b([A-Za-z0-9_]+)_([0-9a-f]{6})\b")
def hashes(t):
    m = {}
    for a, h in HASH.findall(t): m.setdefault(a, set()).add(h)
    return {a: next(iter(v)) for a, v in m.items() if len(v) == 1}
def canon(t, to):   # each kernel name's hash -> the copy's hash for that name (merge_upstream.py's rule)
    return HASH.sub(lambda m: f"{m.group(1)}_{to[m.group(1)]}" if m.group(1) in to else m.group(0), t)
for root, _, files in list(os.walk(B)) + list(os.walk(N)):
    for f in files:
        if f.endswith(".dp.cpp"):
            q = os.path.join(root, f); t = open(q, errors="replace").read(); t = NUM.sub(r"DPCT\1:", t); open(q, "w").write(t)
total = 0
for root, _, files in os.walk(f"{N}/src"):
    for f in files:
        if not f.endswith(".dp.cpp"):
            continue
        rel = os.path.relpath(os.path.join(root, f), N)
        pf, bf, nf = f"{W}/sycl/{rel}", f"{B}/{rel}", os.path.join(root, f)
        if not os.path.exists(pf):
            os.makedirs(os.path.dirname(pf), exist_ok=True)
            shutil.copy(nf, pf)
            print(f"  NEW      {rel}")
            continue
        if not os.path.exists(bf):
            print(f"  NO-BASE  {rel}")
            continue
        to = hashes(open(pf, errors="replace").read())
        tb = canon(open(bf, errors="replace").read(), to); open(bf, "w").write(tb)
        tn = canon(open(nf, errors="replace").read(), to); open(nf, "w").write(tn)
        if open(bf).read() == open(nf).read():
            print(f"  same     {rel}")
            continue
        subprocess.run(["git", "merge-file", "--diff3", "-L", "sycl", "-L", "base", "-L", "ours-now", pf, bf, nf])
        n = open(pf, errors="replace").read().count("<<<<<<< sycl")
        total += n
        print(f"  {n:4d}     {rel}")
print("conflicts", total)
