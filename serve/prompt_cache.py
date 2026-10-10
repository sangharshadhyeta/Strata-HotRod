"""prompt_cache.py - a prompt cache on disk, built on the engine's own SAVE / RESTORE (disk sessions).

What it adds to the manual `--slot-save-path` save and restore: the saving and the restoring happen by themselves.  After a request
that had to READ a long prompt, the conversation the engine now holds is saved; before a request whose prompt starts with a saved
conversation that the engine does not hold, that file is restored, so the long start is not read again (after a restart too).
Taken from solid.cpp's disk prompt cache (a 3,623-token prompt: 5,775 ms re-read, 271 ms from disk): one file per cached prompt
named by a hash of its tokens, a fingerprint of the model in every entry (an entry of another model is wrong, not stale), writes that
are whole or absent, a size limit (an eighth of the free disk by default), the oldest entries dropped first.

Strata specifics, MEASURED (SPEC 19.9): this model keeps a recurrent state that cannot be rewound to an arbitrary token, so an entry
helps only a prompt that CONTINUES it (a conversation's next turn, a shared system prompt followed by more text), never one that
changes the text inside it; so an entry is used only when ALL its tokens are the start of the new prompt, and a saved entry that a
longer one continues is dropped.  A restore costs about 0.6 s for 268 MB (2,087 tokens); a read costs about 11 ms per token, so
restoring pays from about 100 tokens up and saving pays after a read of about as many.

The engine checks every file itself (fingerprint, tokens, size): a wrong guess here costs a refused restore, never a wrong answer.
"""
import hashlib
import json
import os
import shutil
import time


class PromptCache:
    def __init__(self, directory: str, fingerprint: str, min_read: int = 256, min_gain: int = 256, limit_bytes: int | None = None):
        self.dir = directory
        self.fingerprint = fingerprint
        self.min_read = min_read          # a request that read fewer prompt tokens than this is not worth a save
        self.min_gain = min_gain          # a restore must bring at least this many tokens the engine does not hold
        self.limit_bytes = limit_bytes    # None: an eighth of the free disk, between 1 GiB and 256 GiB (solid's rule)
        self.held: list[int] | None = None   # what the engine holds now, as far as this server knows (a hint: the engine checks)
        self.held_gen = None                 # ... and the engine process it was true of
        self.entries: dict[str, dict] = {}
        self.stats = {"saves": 0, "restores": 0, "restore_refused": 0, "dropped": 0}
        os.makedirs(self.dir, mode=0o700, exist_ok=True)
        self._scan()

    # ---- the index: the small sidecar files only, never the state files
    def _scan(self):
        for name in os.listdir(self.dir):
            if not name.endswith(".json"):
                continue
            key = name[:-5]
            try:
                with open(os.path.join(self.dir, name), encoding="utf-8") as f:
                    meta = json.load(f)
                size = os.path.getsize(os.path.join(self.dir, key + ".bin"))
            except (OSError, ValueError):
                self._remove(key)         # a sidecar without its state, or an unreadable one
                continue
            if meta.get("fingerprint") == self.fingerprint and isinstance(meta.get("tokens"), list):
                self.entries[key] = {"tokens": meta["tokens"], "bytes": size, "used": meta.get("used", 0)}

    def _remove(self, key: str):
        self.entries.pop(key, None)
        for ext in (".json", ".bin"):
            try:
                os.remove(os.path.join(self.dir, key + ext))
            except OSError:
                pass

    def _limit(self) -> int:
        if self.limit_bytes:
            return self.limit_bytes
        free = shutil.disk_usage(self.dir).free + sum(e["bytes"] for e in self.entries.values())
        return max(1 << 30, min(256 << 30, free // 8))

    @staticmethod
    def _prefix_len(a: list[int], b: list[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _engine_reuse(self, ids: list[int], gen) -> int:
        """How many tokens of `ids` the engine already holds (a hint): all of what it holds when that starts the prompt, else none
        (the recurrent state cannot go back to the middle of it)."""
        if self.held is None or self.held_gen != gen:
            return 0
        return len(self.held) if self._prefix_len(self.held, ids) == len(self.held) else 0

    # ---- before a request: bring in the best entry, when it saves a read
    def before(self, engine, ids: list[int]) -> str | None:
        """Restores the longest entry that starts `ids` when it brings at least min_gain tokens the engine does not hold.  Returns
        a line to log, or None.  Called with the service's fifo held, after the engine is loaded."""
        gen = getattr(engine, "gen", None)
        have = self._engine_reuse(ids, gen)
        best_key, best_len = None, 0
        for key, e in self.entries.items():
            t = e["tokens"]
            if len(t) < len(ids) and len(t) > best_len and self._prefix_len(t, ids) == len(t):
                best_key, best_len = key, len(t)
        if best_key is None or best_len - have < self.min_gain:
            return None
        t0 = time.time()
        try:
            engine.session_file("restore", os.path.join(self.dir, best_key + ".bin"))
        except ValueError as e:           # SessionRefused is a ValueError: the engine is in step, the entry is no good
            self.stats["restore_refused"] += 1
            self._remove(best_key)
            self.held = None
            return f"prompt cache: restore of {best_len} tokens refused ({str(e)[:120]}); entry dropped"
        self.held, self.held_gen = list(self.entries[best_key]["tokens"]), gen
        self.entries[best_key]["used"] = time.time()
        self.stats["restores"] += 1
        return f"prompt cache: restored {best_len} tokens in {time.time() - t0:.2f} s (the engine held {have})"

    # ---- after a request: remember what the engine holds, and save it when the read was long
    def after(self, engine, ids: list[int], generated: list[int], prompt_read: int | None) -> str | None:
        """`generated`: every token the engine produced, the stop token included.  MEASURED (10 Oct 2026, 3 saves): the engine holds the
        prompt, all of those, and ONE more (the token after the stop token, which this server is not told: the template's newline
        in the cases seen).  The entry records the tokens this server knows; the extra one is the engine's own and is checked by the
        engine against the next prompt.  Called with the fifo held.  Returns a line to log, or None."""
        gen = getattr(engine, "gen", None)
        tokens = list(ids) + list(generated)
        self.held, self.held_gen = tokens, gen
        if not prompt_read or prompt_read < self.min_read or any(self._prefix_len(e["tokens"], tokens) == len(tokens) for e in self.entries.values()):
            return None                   # a short read, or an entry already holds all of it
        key = hashlib.sha1(json.dumps(tokens).encode()).hexdigest()[:24]
        path = os.path.join(self.dir, key + ".bin")
        t0 = time.time()
        try:
            r = engine.session_file("save", path)
        except ValueError as e:
            self._remove(key)
            return f"prompt cache: save refused ({str(e)[:120]})"
        saved = int((r or {}).get("tokens", -1))
        if not 0 <= saved - len(tokens) <= 2:   # a count further off than the engine's one extra token: the lists no longer agree
            self._remove(key)
            return f"prompt cache: the engine saved {saved} tokens, this server expected {len(tokens)} to {len(tokens) + 2}; entry dropped"
        meta = {"fingerprint": self.fingerprint, "tokens": tokens, "used": time.time()}
        tmp = os.path.join(self.dir, key + ".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        os.replace(tmp, os.path.join(self.dir, key + ".json"))   # the sidecar last and whole: an entry exists only with both
        self.entries[key] = {"tokens": tokens, "bytes": os.path.getsize(path), "used": meta["used"]}
        self.stats["saves"] += 1
        for k in [k for k, e in self.entries.items() if k != key and self._prefix_len(e["tokens"], tokens) == len(e["tokens"])]:
            self._remove(k)               # a shorter entry that this one continues is no longer needed
            self.stats["dropped"] += 1
        limit = self._limit()
        while sum(e["bytes"] for e in self.entries.values()) > limit and len(self.entries) > 1:
            oldest = min((k for k in self.entries if k != key), key=lambda k: self.entries[k]["used"])
            self._remove(oldest)
            self.stats["dropped"] += 1
        return f"prompt cache: saved {len(tokens)} tokens in {time.time() - t0:.2f} s ({self.entries[key]['bytes'] >> 20} MiB, {len(self.entries)} entries)"
