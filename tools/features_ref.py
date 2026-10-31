#!/usr/bin/env python3
"""Write docs/FEATURES-<profile>.md: every environment switch, engine flag and server
config key that this repo's sources read.

Standard library only.  Usage (from any directory):

    python3 tools/features_ref.py [--profile qwen|deepseek]

The profile defaults to the repo folder name (Strata -> qwen, Strata-ds -> deepseek).
This file is the same in both repos; the PROFILES table holds each repo's source list.

Regular expressions only extract literal names (getenv("NAME"), a == "--flag",
cfg.get("key")).  Defaults come from string scans of the line; Python sources are read
with ast and comments with tokenize; the HotRod/upstream origin comes from git log -S.
Nothing here builds or runs an engine.
"""

import argparse
import ast
import concurrent.futures
import datetime
import io
import json
import os
import re
import subprocess
import sys
import tokenize

CPP_EXTS = (".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp", ".inc")

PROFILES = {
    "qwen": {
        "title": "Qwen engine (Strata)",
        "repo_dir": "Strata",
        "branch": "main",
        "doc": "docs/FEATURES-qwen.md",
        "config": "strata-iq3_s.json",
        "cpp_roots": ["src", "include"],
        "flags_files": ["src/program/generate.cpp"],
        "usage_func": None,
    },
    "deepseek": {
        "title": "DeepSeek engine (Strata-ds, DeepSeek-V4.1)",
        "repo_dir": "Strata-ds",
        "branch": "upstream-dsv41",
        "doc": "docs/FEATURES-deepseek.md",
        "config": "strata-ds41.json",
        "cpp_roots": ["src/dsv41"],
        "flags_files": ["src/dsv41/ds4_main.cpp"],
        "usage_func": "void usage()",
    },
}

SERVER_FILE = "serve/server.py"
TEST_WORDS = ("measurement", "a/b", "test", "diagnostic", "trace", "debug")

# Name extraction only (no meaning is decided by these patterns).
CPP_GETENV = re.compile(r'(?:std::)?getenv\(\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*\)')
FLAG_BRANCH = re.compile(r'\b\w+ == "(--[A-Za-z0-9][A-Za-z0-9_-]*)"')
USAGE_FLAG = re.compile(r"--[a-z][a-z0-9-]*")


# ---------------------------------------------------------------- comments

def read_lines(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return [ln.rstrip("\r") for ln in f.read().split("\n")]


def split_cpp_comment(line):
    """(code, trailing // comment or None), skipping // inside string and char literals."""
    quote = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "/" and line[i + 1:i + 2] == "/":
            return line[:i], line[i + 2:].strip()
        i += 1
    return line, None


def cpp_above(lines, i):
    """Up to 3 consecutive // comment-only lines directly above line index i, top-down."""
    out = []
    j = i - 1
    while j >= 0 and len(out) < 3:
        s = lines[j].strip()
        if not s.startswith("//"):
            break
        out.append(s[2:].strip())
        j -= 1
    return list(reversed(out))


def py_above(lines, i):
    out = []
    j = i - 1
    while j >= 0 and len(out) < 3:
        s = lines[j].strip()
        if not s.startswith("#"):
            break
        out.append(s[1:].strip())
        j -= 1
    return list(reversed(out))


def compose(above, trailing):
    parts = [p for p in above if p]
    if trailing:
        parts.append(trailing)
    return " ".join(parts)


def python_comment_map(text):
    comments = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                comments[tok.start[0]] = tok.string[1:].strip()
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return comments


# ---------------------------------------------------------------- C++ defaults

def is_ident(c):
    return c.isalnum() or c == "_"


def assigned_var(before):
    """`const char* v = ` at the end of `before` -> 'v'; else None."""
    b = before.rstrip()
    if not b.endswith("="):
        return None
    b = b[:-1].rstrip()
    j = len(b)
    while j > 0 and is_ident(b[j - 1]):
        j -= 1
    return b[j:] or None


def word_phrase_present(text, phrase):
    start = 0
    while True:
        k = text.find(phrase, start)
        if k < 0:
            return False
        if k == 0 or not is_ident(text[k - 1]):
            return True
        start = k + 1


def literal_token(tok):
    if tok == "":
        return "see code"
    if len(tok) >= 2 and tok[0] == '"' and tok[-1] == '"' and '"' not in tok[1:-1]:
        return tok[1:-1] if tok[1:-1] else '""'
    if tok in ("true", "false"):
        return tok
    if tok in ("nullptr", "NULL"):
        return "unset"
    try:
        float(tok)
        return tok
    except ValueError:
        return "see code"


def cpp_default(code, end, var, before):
    """Default read from the getenv expression on this line.

    1. ternary `v ? ... : LITERAL` (the condition is the variable or the call itself);
    2. presence/nullptr tests (`!= nullptr`, `if (getenv(..))`, `v &&`): "unset";
    3. anything else: "see code".
    """
    rest = code[end:]
    depth = 0
    q = None
    for k, ch in enumerate(rest):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                break
        elif ch == "?" and depth == 0:
            q = k
            break
    if q is not None:
        core = rest[:q]
        for sep in ";{(,":
            core = core.split(sep)[-1]
        core = core.strip()
        if core.startswith("return "):
            core = core[len("return "):].strip()
        if core == "" or (var and core == var):
            colon = None
            depth = 0
            for k in range(q + 1, len(rest)):
                ch = rest[k]
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                    if depth < 0:
                        break
                elif ch == ":" and depth == 0 and rest[k - 1:k] != ":" and rest[k + 1:k + 2] != ":":
                    colon = k
                    break
            if colon is not None:
                tail = rest[colon + 1:]
                depth = 0
                cut = len(tail)
                for k, ch in enumerate(tail):
                    if ch in "([{":
                        depth += 1
                    elif ch in ")]}":
                        if depth == 0:
                            cut = k
                            break
                        depth -= 1
                    elif ch in ";," and depth == 0:
                        cut = k
                        break
                return literal_token(tail[:cut].strip())
    stripped = rest.lstrip()
    for pat in ("== nullptr", "!= nullptr", "== NULL", "!= NULL"):
        if stripped.startswith(pat):
            return "unset"
    if before.rstrip().endswith(("if (", "if(")):
        return "unset"
    if var:
        for pat in ("== nullptr", "!= nullptr", "== NULL", "!= NULL", "&&"):
            if word_phrase_present(rest, var + " " + pat):
                return "unset"
        if word_phrase_present(rest, "!" + var):
            return "unset"
    return "see code"


# ---------------------------------------------------------------- file lists

def cpp_files(repo, roots):
    for root in roots:
        base = os.path.join(repo, root)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames.sort()
            for fn in sorted(filenames):
                if fn.endswith(CPP_EXTS):
                    yield os.path.relpath(os.path.join(dirpath, fn), repo).replace(os.sep, "/")


def python_files(repo):
    """serve/*.py (not tests) plus the tools/*.py modules they import, transitively."""
    serve = os.path.join(repo, "serve")
    queue = []
    for fn in sorted(os.listdir(serve)):
        if fn.endswith(".py") and not fn.startswith("test_") and not fn.endswith("_test.py"):
            queue.append("serve/" + fn)
    seen = set(queue)
    out = []
    while queue:
        rel = queue.pop(0)
        out.append(rel)
        try:
            tree = ast.parse("\n".join(read_lines(os.path.join(repo, rel))))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module.split(".")[0]]
            for mod in mods:
                cand = "tools/" + mod + ".py"
                if os.path.isfile(os.path.join(repo, cand)) and cand not in seen:
                    seen.add(cand)
                    queue.append(cand)
    return sorted(out)


# ---------------------------------------------------------------- collection

def is_environ(node):
    return isinstance(node, ast.Attribute) and node.attr == "environ" and \
        isinstance(node.value, ast.Name) and node.value.id == "os"


def literal_py(node):
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, str):
            return v if v != "" else '""'
        return str(v)
    return "see code"


def collect_env(repo, P):
    uses = []
    unextracted = []
    unparsed = []
    for rel in cpp_files(repo, P["cpp_roots"]):
        lines = read_lines(os.path.join(repo, rel))
        for i, raw in enumerate(lines):
            code, trailing = split_cpp_comment(raw)
            if "getenv(" not in code:
                continue
            matches = list(CPP_GETENV.finditer(code))
            if code.count("getenv(") > len(matches):
                unextracted.append("%s:%d: %s" % (rel, i + 1, raw.strip()))
            for m in matches:
                before = code[:m.start()]
                uses.append({
                    "name": m.group(1), "rel": rel, "line": i + 1,
                    "default": cpp_default(code, m.end(), assigned_var(before), before),
                    "comment": compose(cpp_above(lines, i), trailing),
                })
    for rel in python_files(repo):
        text = "\n".join(read_lines(os.path.join(repo, rel)))
        try:
            tree = ast.parse(text)
        except SyntaxError:
            unparsed.append(rel)
            continue
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child._parent = parent
        comments = python_comment_map(text)
        lines = text.split("\n")
        for node in ast.walk(tree):
            name_node = None
            default = None
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                f = node.func
                owns = (f.attr == "get" and is_environ(f.value)) or \
                    (f.attr == "getenv" and isinstance(f.value, ast.Name) and f.value.id == "os")
                if not owns or not node.args:
                    continue
                name_node = node.args[0]
                if len(node.args) >= 2:
                    default = literal_py(node.args[1])
                else:
                    parent = getattr(node, "_parent", None)
                    if isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.Or) and \
                            parent.values[0] is node and len(parent.values) == 2:
                        default = literal_py(parent.values[1])
                    else:
                        default = "unset"
            elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and is_environ(node.value):
                name_node = node.slice
                index_cls = getattr(ast, "Index", None)
                if index_cls is not None and isinstance(name_node, index_cls):
                    name_node = name_node.value
                default = "see code"
            else:
                continue
            if not (isinstance(name_node, ast.Constant) and isinstance(name_node.value, str)):
                unextracted.append("%s:%d: non-literal environment name" % (rel, node.lineno))
                continue
            i = node.lineno - 1
            uses.append({
                "name": name_node.value, "rel": rel, "line": node.lineno,
                "default": default,
                "comment": compose(py_above(lines, i), comments.get(node.lineno)),
            })
    return uses, unextracted, unparsed


def collect_flags(repo, P):
    recs = []
    for rel in P["flags_files"]:
        lines = read_lines(os.path.join(repo, rel))
        for i, raw in enumerate(lines):
            code, trailing = split_cpp_comment(raw)
            for m in FLAG_BRANCH.finditer(code):
                recs.append({"name": m.group(1), "rel": rel, "line": i + 1, "kind": "branch",
                             "comment": compose(cpp_above(lines, i), trailing)})
        if P["usage_func"]:
            start = next((i for i, ln in enumerate(lines) if ln.strip().startswith(P["usage_func"])), None)
            if start is None:
                continue
            end = next((j for j in range(start + 1, len(lines)) if lines[j].rstrip() == "}"), len(lines))
            for j in range(start + 1, end):
                if '"' not in lines[j]:
                    continue
                for nm in USAGE_FLAG.findall(lines[j]):
                    recs.append({"name": nm, "rel": rel, "line": j + 1, "kind": "usage", "comment": ""})
    return recs


def collect_keys(repo, P):
    text = "\n".join(read_lines(os.path.join(repo, SERVER_FILE)))
    tree = ast.parse(text)
    comments = python_comment_map(text)
    lines = text.split("\n")
    recs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                node.func.attr == "get" and isinstance(node.func.value, ast.Name) and \
                node.func.value.id == "cfg" and node.args and \
                isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            recs.append({"name": node.args[0].value, "rel": SERVER_FILE, "line": node.lineno,
                         "comment": compose(py_above(lines, node.lineno - 1), comments.get(node.lineno))})
    return recs


# ---------------------------------------------------------------- git origin

def classify(repo, name):
    """HotRod if the oldest commit that changed the quoted name is ours (author root, or a
    Co-Authored-By: Claude trailer); upstream otherwise; unclassified if git finds none."""
    out = subprocess.run(["git", "-C", repo, "log", '-S"%s"' % name, "--format=%x1e%H%x1f%an%x1f%B"],
                         capture_output=True, text=True).stdout
    records = [r for r in out.split("\x1e") if r.strip()]
    if not records:
        return {"origin": "unclassified", "sha": "", "author": "", "subject": ""}
    sha, author, body = (records[-1].split("\x1f", 2) + ["", ""])[:3]
    ours = author.strip() == "root" or "Co-Authored-By: Claude" in body
    return {"origin": "HotRod" if ours else "upstream", "sha": sha.strip()[:10],
            "author": author.strip(), "subject": body.strip().split("\n")[0]}


def classify_all(repo, names):
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        return dict(zip(names, ex.map(lambda n: classify(repo, n), names)))


# ---------------------------------------------------------------- output

def cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def row(*cells):
    return "| " + " | ".join(cell(c) for c in cells) + " |"


def group_of(name):
    if name.startswith("STRATA_"):
        return "STRATA_"
    if name.startswith("DS4_"):
        return "DS4_"
    return "Other"


def is_test(comment):
    low = comment.lower()
    return any(w in low for w in TEST_WORDS)


def merged_default(values):
    literal = []
    for v in values:
        if v != "see code" and v not in literal:
            literal.append(v)
    return " / ".join(literal) if literal else "see code"


def git_output(repo, *args):
    return subprocess.run(["git", "-C", repo] + list(args), capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", choices=sorted(PROFILES))
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prof = args.profile or next((k for k, p in PROFILES.items()
                                 if p["repo_dir"] == os.path.basename(repo)), None)
    if prof is None:
        sys.exit("cannot tell the profile from the folder name %r; pass --profile" % os.path.basename(repo))
    P = PROFILES[prof]

    # ---- environment switches
    uses, env_unextracted, env_unparsed = collect_env(repo, P)
    by_name = {}
    for u in sorted(uses, key=lambda u: (u["rel"], u["line"])):
        ent = by_name.setdefault(u["name"], {"uses": [], "defaults": [], "comment": None})
        ent["uses"].append("%s:%d" % (u["rel"], u["line"]))
        ent["defaults"].append(u["default"])
        if ent["comment"] is None:
            ent["comment"] = u["comment"]  # the comment at the first use
    origins = classify_all(repo, sorted(by_name))

    # ---- engine flags
    flag_recs = collect_flags(repo, P)
    flags = {}
    for r in sorted(flag_recs, key=lambda r: (r["kind"] != "branch", r["rel"], r["line"])):
        ent = flags.setdefault(r["name"], {"where": [], "comment": None})
        ent["where"].append("%s:%d%s" % (r["rel"], r["line"], " (usage text)" if r["kind"] == "usage" else ""))
        if ent["comment"] is None:
            ent["comment"] = r["comment"] or "(no comment; listed in the usage text)"

    # ---- server keys
    keys = {}
    for r in sorted(collect_keys(repo, P), key=lambda r: r["line"]):
        ent = keys.setdefault(r["name"], {"where": [], "comment": None})
        ent["where"].append("%s:%d" % (r["rel"], r["line"]))
        if ent["comment"] is None:
            ent["comment"] = r["comment"]

    # ---- the model config, read at generation time
    cfg_path = os.path.join(repo, P["config"])
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg_env = cfg.get("env") or {}
    cfg_args = cfg.get("args") or []

    # ---- markdown
    today = datetime.date.today().isoformat()
    head = git_output(repo, "rev-parse", "--short", "HEAD")
    out = []
    w = out.append
    w("# FEATURES reference: %s" % P["title"])
    w("")
    w("Generated by `tools/features_ref.py` on %s from branch `%s` (source commit `%s`). "
      "Do not edit by hand. Regenerate with:" % (today, P["branch"], head))
    w("")
    w("    python3 tools/features_ref.py")
    w("")
    w("It lists every environment switch the engine and the server read (`getenv(\"NAME\")` in the "
      "C/C++/CUDA sources; `os.environ` and `os.getenv` in `serve/*.py` and the `tools/` modules the "
      "server imports), every engine flag, and every server config key (`cfg.get(\"key\")` in "
      "`serve/server.py`). Each row gives the default as the code reads it on that line, the comment "
      "at the first use, and where it is used.")
    w("")
    w("## How to use a switch")
    w("")
    w("- Environment switch: put `\"NAME\": \"value\"` into the model config's `\"env\"` object. A blank "
      "value means the engine's default; the server drops it.")
    w("- Engine flag: put `\"--name\"` and its value (if it takes one) into the model config's `\"args\"` list.")
    w("- Server key: put `\"key\": value` at the top level of the model config.")
    w("")
    w("Reading the tables:")
    w("")
    w("- **Default**: the value written literally on the use's line (a ternary fallback, or a literal "
      "default argument). `unset` means the line only tests whether the variable is set or null, so the "
      "value when unset is whatever that test selects; read the comment. `see code` means no literal "
      "default is on that line. A name with several uses shows its distinct literal readings joined by `/`.")
    w("- **HotRod / upstream**: from `git log -S\"NAME\"`. The oldest commit that added the quoted name "
      "decides it: HotRod if its author is `root` (our commits) or its message has a `Co-Authored-By: Claude` "
      "trailer; upstream otherwise. `unclassified` means git found no commit with that name.")
    w("- **test/measurement**: the comment contains `measurement`, `A/B`, `test`, `diagnostic`, `trace` or "
      "`debug` (case-insensitive substring check).")
    w("")

    # In use
    w("## In use on this machine")
    w("")
    w("Read from `%s` at generation time." % P["config"])
    w("")
    w("Environment (`env`):")
    w("")
    if cfg_env:
        w(row("Name", "Value", "Found in sources"))
        w(row("---", "---", "---"))
        for k, v in cfg_env.items():
            shown = "(blank: engine default)" if v in ("", None) else v
            w(row(k, shown, "yes" if k in by_name else "no"))
    else:
        w("None.")
    w("")
    w("Arguments (`args`):")
    w("")
    w("```")
    w(json.dumps(cfg_args, indent=1))
    w("```")
    w("")

    # Environment switches
    w("## Environment switches")
    w("")
    groups = {"STRATA_": [], "DS4_": [], "Other": []}
    for name in by_name:
        groups[group_of(name)].append(name)
    counts = {"names": len(by_name), "uses": len(uses), "HotRod": 0, "upstream": 0,
              "unclassified": 0, "test": 0, "nocomment": 0, "seecode": 0}
    for name in by_name:
        o = origins[name]["origin"]
        counts[o] += 1
        if is_test(by_name[name]["comment"] or ""):
            counts["test"] += 1
        if not by_name[name]["comment"]:
            counts["nocomment"] += 1
        if merged_default(by_name[name]["defaults"]) == "see code":
            counts["seecode"] += 1
    w("%d names from %d uses: HotRod %d, upstream %d, unclassified %d; test/measurement %d; "
      "no comment at first use %d; no literal default %d." % (
          counts["names"], counts["uses"], counts["HotRod"], counts["upstream"], counts["unclassified"],
          counts["test"], counts["nocomment"], counts["seecode"]))
    w("")
    for gname in ("STRATA_", "DS4_", "Other"):
        names = sorted(groups[gname])
        if not names:
            continue
        w("### %s (%d)" % (gname if gname != "Other" else "Other prefixes", len(names)))
        w("")
        w(row("Name", "Default", "Origin", "test/measurement", "What it does (comment at first use)", "Where"))
        w(row("---", "---", "---", "---", "---", "---"))
        for name in names:
            ent = by_name[name]
            comment = ent["comment"] or "(no comment)"
            w(row("`%s`" % name, merged_default(ent["defaults"]), origins[name]["origin"],
                  "yes" if is_test(ent["comment"] or "") else "",
                  comment, ", ".join(ent["uses"])))
        w("")

    # Engine flags
    w("## Engine flags")
    w("")
    sources = list(P["flags_files"])
    if P["usage_func"]:
        sources.append("the usage text of `%s`" % P["usage_func"])
    w("%d flags, from %s." % (len(flags), ", ".join("`%s`" % s for s in sources)))
    w("")
    w(row("Flag", "Where", "What it does (comment on or above the first branch)"))
    w(row("---", "---", "---"))
    for name in sorted(flags):
        w(row("`%s`" % name, ", ".join(flags[name]["where"]), flags[name]["comment"] or "(no comment)"))
    w("")

    # Server keys
    w("## Server config keys")
    w("")
    w("%d keys read with `cfg.get(\"key\")` in `%s`." % (len(keys), SERVER_FILE))
    w("")
    w(row("Key", "Where", "What it does (comment on or above the first use)"))
    w(row("---", "---", "---"))
    for name in sorted(keys):
        w(row("`%s`" % name, ", ".join(keys[name]["where"]), keys[name]["comment"] or "(no comment)"))
    w("")

    # Not extracted
    w("## Not extracted")
    w("")
    w("Call sites whose name is not a string literal, and files that did not parse. These are "
      "not in the tables above.")
    w("")
    unext = env_unextracted + (["%s: did not parse" % f for f in env_unparsed])
    if unext:
        for item in unext:
            w("- `%s`" % item)
    else:
        w("None.")
    w("")

    text = "\n".join(out)
    dest = os.path.join(repo, P["doc"])
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)

    print("wrote %s" % P["doc"])
    print("env names %d, uses %d; HotRod %d, upstream %d, unclassified %d, test/measurement %d"
          % (counts["names"], counts["uses"], counts["HotRod"], counts["upstream"],
             counts["unclassified"], counts["test"]))
    print("engine flags %d, server keys %d, not extracted %d" % (len(flags), len(keys), len(unext)))


if __name__ == "__main__":
    main()
