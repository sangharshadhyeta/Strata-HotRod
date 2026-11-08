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
    def __init__(self, directory: str, fingerprint: str, min_read: int = 256, min_gain: int = 256, limit_bytes: int | None = None,
                 turn_id: int | None = None, max_age_s: float = 0):
        self.dir = directory
        self.fingerprint = fingerprint
        self.min_read = min_read          # a request that read fewer prompt tokens than this is not worth a save
        self.min_gain = min_gain          # a restore must bring at least this many tokens the engine does not hold
        self.limit_bytes = limit_bytes    # None: an eighth of the free disk, between 1 GiB and 256 GiB (solid's rule)
        self.turn_id = turn_id            # <|im_start|>: the engine's turn checkpoint sits at an entry's last one (None: whole entries only)
        self.max_age_s = max_age_s        # an entry unused (its "used" time) for longer goes, the last own one too (0: no age limit)
        self.held: list[int] | None = None   # what the engine holds now, as far as this server knows (a hint: the engine checks)
        self.held_gen = None                 # ... and the engine process it was true of
        self.entries: dict[str, dict] = {}
        self.foreign: dict[str, dict] = {}   # entries another engine build or setting wrote: counted against the limit, dropped first
        self.stats = {"saves": 0, "restores": 0, "restore_refused": 0, "dropped": 0}
        os.makedirs(self.dir, mode=0o700, exist_ok=True)
        # owner-only, as solid.cpp's cache is: the entries hold the prompts (the sidecar lists their token ids).  makedirs does not change a directory
        # that exists, and a file made with open() takes the umask, so both are set here; entries from an earlier version are tightened too
        try:
            os.chmod(self.dir, 0o700)
            for name in os.listdir(self.dir):
                os.chmod(os.path.join(self.dir, name), 0o600)
        except OSError:
            pass
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
                self.entries[key] = {"tokens": meta["tokens"], "bytes": size, "used": meta.get("used", 0),
                                     "prompt_len": int(meta.get("prompt_len") or 0)}
            else:   # another engine build or setting wrote it: never restored here, but it takes disk space (21 Oct: 83 GB of them)
                self.foreign[key] = {"bytes": size, "used": meta.get("used", 0)}
        # a state file with no sidecar: a save the engine or this process did not finish (it died, or the user closed the server
        # right after an answer while the save ran).  Nothing names it, nothing counts it against the limit: removed - once it is
        # an hour old, so a save another server is writing into the same folder right now is left alone
        known = set(self.entries) | set(self.foreign)
        for name in os.listdir(self.dir):
            key = next((name[:-len(x)] for x in (".bin.conv", ".bin", ".json.tmp") if name.endswith(x)), None)
            if key is None or key in known:
                continue
            path = os.path.join(self.dir, name)
            try:
                if time.time() - os.path.getmtime(path) > 3600:
                    os.remove(path)
            except OSError:
                pass
        self._trim()

    def _age_out(self):
        """E1 (8 Nov): an entry unused for longer than max_age_s goes - another build's or this engine's, the last own one too
        (unlike the size rule): a save nobody has used for that long is dropped.  Logged as one line."""
        if not self.max_age_s:
            return
        cutoff = time.time() - self.max_age_s
        old = [k for k, e in self.entries.items() if e["used"] < cutoff] + [k for k, e in self.foreign.items() if e["used"] < cutoff]
        if not old:
            return
        freed = sum((self.entries.get(k) or self.foreign[k])["bytes"] for k in old)
        for k in old:
            self._remove(k)
            self.stats["dropped"] += 1
        print(f"[strata] prompt cache: dropped {len(old)} entries unused for over {self.max_age_s / 3600:g} h "
              f"({freed / (1 << 30):.2f} GiB)", flush=True)

    def _trim(self, keep: str | None = None):
        """Under the limit: the entries other engine builds or settings wrote go first (oldest first: this engine never restores
        them, but a rollback to the build that wrote them can while they are kept), then this engine's own, least recently
        used first; `keep` (the entry just saved) and the last own entry stay.  Entries unused past max_age_s go first of all."""
        self._age_out()
        limit = self._limit()
        own = lambda: sum(e["bytes"] for e in self.entries.values())   # noqa: E731
        while self.foreign and own() + sum(e["bytes"] for e in self.foreign.values()) > limit:
            self._remove(min(self.foreign, key=lambda k: self.foreign[k]["used"]))
            self.stats["dropped"] += 1
        while own() > limit and len(self.entries) > 1:
            self._remove(min((k for k in self.entries if k != keep), key=lambda k: self.entries[k]["used"]))
            self.stats["dropped"] += 1

    def _remove(self, key: str):
        self.entries.pop(key, None)
        self.foreign.pop(key, None)
        for ext in (".json", ".bin", ".bin.conv"):
            try:
                os.remove(os.path.join(self.dir, key + ext))
            except OSError:
                pass

    def _limit(self) -> int:
        if self.limit_bytes:
            return self.limit_bytes
        free = (shutil.disk_usage(self.dir).free + sum(e["bytes"] for e in self.entries.values())
                + sum(e["bytes"] for e in self.foreign.values()))
        return max(1 << 30, min(256 << 30, free // 8))

    @staticmethod
    def _prefix_len(a: list[int], b: list[int]) -> int:
        """Length of the longest common prefix, by slices (C speed) rather than a loop over tokens (strata_tokenizer's
        common_prefix_len, for lists): a 60K-token compare 2.1 ms -> under 1 ms, and it runs per entry on every request."""
        n = min(len(a), len(b))
        lo, step = 0, 4096
        while lo < n:                                       # whole chunks first, doubling
            hi = min(n, lo + step)
            if a[lo:hi] != b[lo:hi]:
                break
            lo, step = hi, min(step * 2, 1 << 20)
        else:
            return n
        while hi - lo > 1:                                  # the first difference is in [lo, hi): bisect it
            mid = (lo + hi) // 2
            if a[lo:mid] == b[lo:mid]:
                lo = mid
            else:
                hi = mid
        return lo

    def _turn_at(self, e: dict) -> int:
        """The position of an entry's last turn token (0: none, or whole entries only), found once per entry."""
        k = e.get("turn_at")
        if k is None:
            t = e["tokens"]
            k = 0
            if self.turn_id is not None:
                try:
                    k = len(t) - 1 - t[::-1].index(self.turn_id)
                except ValueError:
                    pass
            e["turn_at"] = k
        return k

    def _engine_reuse(self, ids: list[int], gen) -> int:
        """How many tokens of `ids` the engine already holds (a hint): all of what it holds when that starts the prompt, else none
        (the recurrent state cannot go back to the middle of it)."""
        if self.held is None or self.held_gen != gen:
            return 0
        return len(self.held) if self._prefix_len(self.held, ids) == len(self.held) else 0

    def covers(self, ids: list[int], gen) -> bool:
        """True when the engine holds all of `ids` (a hint, as _engine_reuse): a warm read of such a prompt has nothing to read."""
        return self.held is not None and self.held_gen == gen and self._prefix_len(self.held, ids) == len(ids)

    # ---- before a request: bring in the best entry, when it saves a read
    def before(self, engine, ids: list[int]) -> str | None:
        """Restores the longest entry that starts `ids` when it brings at least min_gain tokens the engine does not hold.  Returns
        a line to log, or None.  Called with the service's fifo held, after the engine is loaded."""
        gen = getattr(engine, "gen", None)
        have = self._engine_reuse(ids, gen)
        best_key, best_len = None, 0
        for key, e in self.entries.items():
            t = e["tokens"]
            if len(t) <= best_len:        # neither the whole entry nor its turn (shorter) can beat the best
                continue
            p = self._prefix_len(t, ids)   # once per entry: both checks below read it
            if len(t) < len(ids) and p == len(t):
                best_key, best_len = key, len(t)
            # 20 Oct: a reasoning model's next turn does not repeat the reasoning the engine generated (the template drops it
            # from the history), so a whole entry is never the start of it - but its turn is, and the saved session carries
            # the engine's turn checkpoint, which the engine rewinds to.  A wrong guess is a refused or partial
            # reuse (the engine checks the tokens itself), never a wrong answer.
            # the engine's turn checkpoint holds the tokens before the entry's last <|im_start|> (the end of the user's turn;
            # server.py pick_slot's held_prefix uses the same rule) - an id-equality on the model's own control token
            k = self._turn_at(e)
            if 0 < k < len(ids) and k > best_len and p >= k:
                best_key, best_len = key, k
        if best_key is None or best_len - have < self.min_gain:
            return None
        t0 = time.time()
        try:
            engine.session_file("restore", os.path.join(self.dir, best_key + ".bin"))
        except ValueError as e:           # SessionRefused is a ValueError: the engine is in step, the refusal's kind says what is wrong
            self.stats["restore_refused"] += 1
            # "invalid" is the engine's kind for a file that fails its checks (a mismatch); memory, storage and io are about the
            # machine, not the file, so the entry is kept for them.  (The engine also uses "invalid" for a few refusals of its
            # own settings, which this cannot tell apart: those drop the entry too.)
            bad = getattr(e, "kind", None) == "invalid"
            if bad:
                self._remove(best_key)
            self.held = None
            return f"prompt cache: restore of {best_len} tokens refused ({str(e)[:120]}); entry {'dropped' if bad else 'kept'}"
        self.held, self.held_gen = list(self.entries[best_key]["tokens"]), gen
        self.entries[best_key]["used"] = time.time()
        self.stats["restores"] += 1
        return f"prompt cache: restored {best_len} tokens in {time.time() - t0:.2f} s (the engine held {have})"

    # ---- after a request: remember what the engine holds, and save it when the read was long
    def after(self, engine, ids: list[int], generated: list[int], prompt_read: int | None, save: bool = True) -> str | None:
        """`generated`: every token the engine produced, the stop token included.  MEASURED (10 Oct 2026, 3 saves): the engine holds the
        prompt, all of those, and ONE more (the token after the stop token, which this server is not told: the template's newline
        in the cases seen).  The entry records the tokens this server knows; the extra one is the engine's own and is checked by the
        engine against the next prompt.  Called with the fifo held.  Returns a line to log, or None.
        save=False (20 Oct: a warm read, POST /v1/prefill) records what the engine holds and writes nothing to disk."""
        gen = getattr(engine, "gen", None)
        tokens = list(ids) + list(generated)
        self.held, self.held_gen = tokens, gen
        if not save or not prompt_read or prompt_read < self.min_read or any(self._prefix_len(e["tokens"], tokens) == len(tokens) for e in self.entries.values()):
            return None                   # a short read, or an entry already holds all of it
        return self._save(engine, tokens, ids=ids)

    # ---- before a request that switches conversation: the one the engine holds is saved first (8 Nov)
    def before_switch(self, engine, ids: list[int]) -> str | None:
        """A long conversation the engine holds is replaced by this request (its prompt does not start with what the engine holds):
        it is saved first, so its next turn restores instead of reading it again (8 Nov: a 180K-token agent conversation was
        read again from token 0 after one other request).  Not when the request continues it, when it is shorter than min_read,
        when the held process is not this engine, or when an entry already holds all of it.  Called with the fifo held, before
        before().  Returns a line to log, or None."""
        held = self.held
        if held is None or len(held) < self.min_read or self.held_gen != getattr(engine, "gen", None):
            return None
        if self._prefix_len(self.held, ids) >= len(self.held) - 2:   # the request continues it (the engine's one or two extra tokens allowed)
            return None
        if any(self._prefix_len(e["tokens"], held) == len(held) for e in self.entries.values()):
            return None
        return self._save(engine, list(held), replaced=True)

    # ---- the save itself, shared by after() and before_switch()
    def _save(self, engine, tokens: list[int], ids: list[int] | None = None, replaced: bool = False) -> str | None:
        """Saves what the engine holds (`tokens`, as this server knows it) as one entry.  `ids` (the prompt) only goes into the
        entry's prompt_len; `replaced`: the log line says the conversation was replaced.  The entry is written whole, the entries
        it continues are dropped, and so are the other entries of the same conversation (B2: its key is its tokens up to and
        including the third turn token, so only the latest save of a conversation is kept).  Returns a line to log, or None."""
        key = hashlib.sha1(json.dumps(tokens).encode()).hexdigest()[:24]
        path = os.path.join(self.dir, key + ".bin")
        t0 = time.time()
        try:
            r = engine.session_file("save", path)
        except ValueError as e:
            self._remove(key)
            return f"prompt cache: save refused ({str(e)[:120]})"
        except BaseException:
            self._remove(key)             # the engine died mid-save (or the server is stopping): no half-written state is kept
            raise
        saved = int((r or {}).get("tokens", -1))
        # 20 Oct: an answer that ended on max_tokens (no stop token) leaves the engine one token SHORT of this list - the
        # last sampled token was never fed back - so the entry records the tokens the engine holds (the engine checks the
        # next prompt against its own copy: a wrong guess is a refused restore, never a wrong answer)
        if saved == len(tokens) - 1:
            tokens = tokens[:saved]
        if not 0 <= saved - len(tokens) <= 2:   # a count further off than the engine's one extra token: the lists no longer agree
            self._remove(key)
            return f"prompt cache: the engine saved {saved} tokens, this server expected {len(tokens)} to {len(tokens) + 2}; entry dropped"
        meta = {"fingerprint": self.fingerprint, "tokens": tokens, "used": time.time(), "prompt_len": len(ids or tokens)}
        tmp = os.path.join(self.dir, key + ".json.tmp")
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        os.replace(tmp, os.path.join(self.dir, key + ".json"))   # the sidecar last and whole: an entry exists only with both
        self.entries[key] = {"tokens": tokens, "bytes": os.path.getsize(path), "used": meta["used"], "prompt_len": meta["prompt_len"]}
        self.stats["saves"] += 1
        for k in [k for k, e in self.entries.items() if k != key and self._prefix_len(e["tokens"], tokens) == len(e["tokens"])]:
            self._remove(k)               # a shorter entry that this one continues is no longer needed
            self.stats["dropped"] += 1
        conv = self._conv_len(tokens)     # B2: the same conversation's other saves go (only the latest is kept)
        if conv is not None:
            for k in [k for k, e in self.entries.items() if k != key and self._conv_len(e["tokens"]) == conv
                      and self._prefix_len(e["tokens"], tokens) >= conv]:
                self._remove(k)
                self.stats["dropped"] += 1
        self._trim(keep=key)
        what = f"the conversation being replaced ({len(tokens)} tokens)" if replaced else f"{len(tokens)} tokens"
        return f"prompt cache: saved {what} in {time.time() - t0:.2f} s ({self.entries[key]['bytes'] >> 20} MiB, {len(self.entries)} entries)"

    def _conv_len(self, t: list[int]) -> int | None:
        """B1: the length of a conversation's key - its tokens up to and including the third turn token (the system turn and the
        first user turn).  None with fewer than three turn tokens, or no turn_id.  Id equality only."""
        if self.turn_id is None:
            return None
        i = -1
        try:
            for _ in range(3):
                i = t.index(self.turn_id, i + 1)
        except ValueError:
            return None
        return i + 1
