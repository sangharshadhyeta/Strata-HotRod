#!/usr/bin/env python3
"""upstream_triage.py - sort upstream commits by how much they overlap what we changed, so a hand port starts from a short list.

Read-only: it only runs git log / diff / merge-base.  It never fetches, merges or writes into the repo.
Run `git fetch origin` yourself first (origin = Niko1221/Strata).

  python3 tools/upstream_triage.py [--upstream origin/main] [--ours HEAD] [--base REF] [--light-lines 30] [--tsv out.tsv] [--ledger docs/UPSTREAM_PORTS.md]

For each upstream commit after the merge-base (or --base) it looks at the files the commit touches and at how much WE changed each of them since the base:
  clean  - touches no file we changed: the patch can be taken as it is (git cherry-pick -x, or git apply --3way)
  light  - overlaps only files where our change is at most --light-lines lines: re-apply the hook by hand
  heavy  - overlaps a file we changed more than that: a real port, read both sides
  docs   - only docs / translations / CI / data files (usually skipped or taken as it is)
Commits already named in the ledger (a line containing the short or full hash) are marked `done` and left out of the counts.
"""
import argparse
import collections
import subprocess
import sys


def git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        sys.exit("git %s failed: %s" % (" ".join(args), r.stderr.strip()[:300]))
    return r.stdout


DOC_PREFIXES = ("docs/", ".github/", "bench/results/", "data/")
DOC_SUFFIXES = (".md", ".png", ".svg", ".jpg", ".json", ".txt", ".yml", ".yaml")
DOC_NAMES = ("LICENSE", "VERSION", ".gitignore", ".gitattributes", ".dockerignore")


def is_doc(path):
    return path.startswith(DOC_PREFIXES) or path.endswith(DOC_SUFFIXES) or path in DOC_NAMES


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", default="origin/main")
    ap.add_argument("--ours", default="HEAD")
    ap.add_argument("--base", default=None, help="the upstream commit our tree last took (default: merge-base)")
    ap.add_argument("--light-lines", type=int, default=30)
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--ledger", default="docs/UPSTREAM_PORTS.md")
    ap.add_argument("--top", type=int, default=25, help="how many heavy-overlap files to list")
    a = ap.parse_args()

    base = a.base or git("merge-base", a.upstream, a.ours).strip()
    print("base %s  upstream %s  ours %s" % (base[:9], git("rev-parse", "--short", a.upstream).strip(), a.ours))

    churn = collections.Counter()   # path -> lines we added+deleted since the base
    for line in git("diff", "--numstat", base, a.ours).splitlines():
        add, dele, path = line.split("\t", 2)
        churn[path] = (int(add) if add != "-" else 10 ** 6) + (int(dele) if dele != "-" else 10 ** 6)

    try:
        ledger = open(a.ledger).read()
    except OSError:
        ledger = ""

    commits = git("log", "--no-merges", "--reverse", "--format=%H\t%as\t%an\t%s", "%s..%s" % (base, a.upstream)).splitlines()
    rows, counts, hot = [], collections.Counter(), collections.Counter()
    for c in commits:
        h, day, who, subj = c.split("\t", 3)
        files = [f for f in git("diff-tree", "--no-commit-id", "--name-only", "-r", "-m", "--first-parent", h).splitlines() if f]
        if h in ledger or h[:9] in ledger:
            kind = "done"
        elif files and all(is_doc(f) for f in files):
            kind = "docs"
        else:
            over = [f for f in files if churn.get(f, 0) > 0]
            if not over:
                kind = "clean"
            elif all(churn[f] <= a.light_lines for f in over):
                kind = "light"
            else:
                kind = "heavy"
                for f in over:
                    if churn[f] > a.light_lines:
                        hot[f] += 1
        counts[kind] += 1
        rows.append((kind, h[:9], day, who, subj[:100], len(files)))

    print("upstream commits since the base (merges left out): %d" % len(commits))
    for k in ("clean", "light", "heavy", "docs", "done"):
        print("  %-6s %5d" % (k, counts[k]))
    print("\nfiles we changed a lot that upstream keeps changing (where the real ports are), by upstream commits touching them:")
    for f, n in hot.most_common(a.top):
        print("  %4d commits  our change %5d lines  %s" % (n, churn[f], f))
    if a.tsv:
        with open(a.tsv, "w") as o:
            o.write("kind\thash\tdate\tauthor\tsubject\tfiles\n")
            for r in rows:
                o.write("\t".join(str(x) for x in r) + "\n")
        print("\nper-commit list written to %s" % a.tsv)


if __name__ == "__main__":
    main()
