# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""tokenizer: prompt text -> token ids the way llama.cpp does it, for byte-level BPE
vocabularies (gpt2-style: SmolLM2, Llama 3, Qwen2), so a reference re-execution can start
from the certified prompt text instead of the runtime's token ids.

Chat template (the GGUF's) rendered with jinja2 for messages=[{"role": "user", ...}] and
add_generation_prompt=True, as llama-cli does for -p; special tokens in the rendered text
matched literally (longest first); pre-tokenisation with llama.cpp's regexes per
tokenizer.ggml.pre (the `regex` module, for \\p{L} / \\p{N}); byte-level encoding; BPE by merge
rank; a leading BOS is not doubled. Optional dependencies: regex, jinja2 (pip install
'anomly-invar[reexec]')."""
from __future__ import annotations

import datetime

try:
    import regex as _re
except ImportError:  # pragma: no cover
    _re = None

_PRE_REGEX = {
    # llama.cpp llama-vocab.cpp: each list is applied in sequence (every regex splits the pieces of the previous one)
    "smollm": [r"\p{N}", r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"],
    "gpt2": [r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"],
    "llama-bpe": [r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"],
    "qwen2": [r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"],
}


def _byte_encoder() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class BPETokenizer:
    def __init__(self, kv: dict):
        self.kv = kv
        if kv.get("tokenizer.ggml.model") != "gpt2":
            raise ValueError("only byte-level BPE (gpt2-style) vocabularies are supported")
        self.pre = kv.get("tokenizer.ggml.pre", "gpt2")
        if self.pre not in _PRE_REGEX:
            raise ValueError(f"pre-tokenizer {self.pre!r} not supported")
        if _re is None:
            raise ImportError("the tokenizer needs the 'regex' module")
        self.tokens = kv["tokenizer.ggml.tokens"]
        self.types = kv.get("tokenizer.ggml.token_type", [1] * len(self.tokens))
        self.id_of = {t: i for i, t in enumerate(self.tokens)}
        self.ranks = {tuple(m.split(" ", 1)): i for i, m in enumerate(kv.get("tokenizer.ggml.merges", []))}
        self.enc = _byte_encoder()
        self.patterns = [_re.compile(p) for p in _PRE_REGEX[self.pre]]
        # special tokens (control 3 / user-defined 4) are matched literally in the text, longest first
        self.specials = sorted((t for t, ty in zip(self.tokens, self.types) if ty in (3, 4)), key=len, reverse=True)
        self.bos_id = int(kv.get("tokenizer.ggml.bos_token_id", -1))
        self.add_bos = bool(kv.get("tokenizer.ggml.add_bos_token", False))

    # ---- pieces
    def _split_specials(self, text: str) -> list[tuple[bool, str]]:
        out: list[tuple[bool, str]] = []
        i = 0
        buf = ""
        while i < len(text):
            hit = None
            for sp in self.specials:
                if text.startswith(sp, i):
                    hit = sp
                    break
            if hit is None:
                buf += text[i]
                i += 1
                continue
            if buf:
                out.append((False, buf))
                buf = ""
            out.append((True, hit))
            i += len(hit)
        if buf:
            out.append((False, buf))
        return out

    def _pre_tokenize(self, text: str) -> list[str]:
        pieces = [text]
        for pat in self.patterns:
            nxt: list[str] = []
            for p in pieces:
                pos = 0
                for m in pat.finditer(p):
                    if m.start() > pos:
                        nxt.append(p[pos:m.start()])
                    nxt.append(m.group(0))
                    pos = m.end()
                if pos < len(p):
                    nxt.append(p[pos:])
            pieces = nxt
        return pieces

    def _bpe(self, piece: str) -> list[int]:
        word = [self.enc[b] for b in piece.encode("utf-8")]
        while len(word) > 1:
            best = None
            for j in range(len(word) - 1):
                r = self.ranks.get((word[j], word[j + 1]))
                if r is not None and (best is None or r < best[0]):
                    best = (r, j)
            if best is None:
                break
            j = best[1]
            word = word[:j] + [word[j] + word[j + 1]] + word[j + 2:]
        try:
            return [self.id_of[w] for w in word]
        except KeyError as e:
            raise ValueError(f"piece {e.args[0]!r} is not in the vocabulary (byte-level vocabularies carry every byte)") from None

    def encode(self, text: str, add_bos: bool | None = None) -> list[int]:
        ids: list[int] = []
        for is_special, chunk in self._split_specials(text):
            if is_special:
                ids.append(self.id_of[chunk])
            else:
                for piece in self._pre_tokenize(chunk):
                    ids += self._bpe(piece)
        want_bos = self.add_bos if add_bos is None else add_bos
        if want_bos and self.bos_id >= 0 and (not ids or ids[0] != self.bos_id):
            ids.insert(0, self.bos_id)
        return ids

    # ---- chat template
    def render_chat(self, messages: list[dict], add_generation_prompt: bool = True, now: datetime.date | None = None) -> str:
        import jinja2
        tpl = self.kv.get("tokenizer.chat_template")
        if not tpl:
            raise ValueError("GGUF carries no chat template")
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        env.globals["strftime_now"] = lambda fmt: (now or datetime.date.today()).strftime(fmt)
        bos = self.tokens[self.bos_id] if self.bos_id >= 0 else ""
        eos_id = int(self.kv.get("tokenizer.ggml.eos_token_id", -1))
        eos = self.tokens[eos_id] if eos_id >= 0 else ""
        return env.from_string(tpl).render(messages=messages, add_generation_prompt=add_generation_prompt,
                                           bos_token=bos, eos_token=eos, tools=None)

    def prompt_ids(self, prompt: str, now: datetime.date | None = None) -> list[int]:
        """The token ids llama-cli feeds for `-p prompt` in single-turn chat mode."""
        return self.encode(self.render_chat([{"role": "user", "content": prompt}], True, now))


class SPMTokenizer:
    """llama.cpp's sentencepiece (LLAMA_VOCAB_TYPE_SPM) tokeniser: one symbol per UTF-8
    character, greedy bigram merges by token score (ties: leftmost), byte fallback
    "<0xXX>" for pieces not in the vocabulary; spaces escaped to U+2581; special tokens
    matched literally; BOS prepended once when add_bos_token."""

    def __init__(self, kv: dict):
        self.kv = kv
        if kv.get("tokenizer.ggml.model") != "llama":
            raise ValueError("not a sentencepiece (llama) vocabulary")
        self.tokens = kv["tokenizer.ggml.tokens"]
        self.scores = kv.get("tokenizer.ggml.scores", [0.0] * len(self.tokens))
        self.types = kv.get("tokenizer.ggml.token_type", [1] * len(self.tokens))
        self.id_of = {t: i for i, t in enumerate(self.tokens)}
        self.specials = sorted((t for t, ty in zip(self.tokens, self.types) if ty in (3, 4)), key=len, reverse=True)
        self.bos_id = int(kv.get("tokenizer.ggml.bos_token_id", -1))
        self.add_bos = bool(kv.get("tokenizer.ggml.add_bos_token", False))
        self.add_space_prefix = bool(kv.get("tokenizer.ggml.add_space_prefix", False))
        self.pre = "spm"

    def _split_specials(self, text: str):
        return BPETokenizer._split_specials(self, text)

    def _spm(self, text: str) -> list[int]:
        import heapq
        text = text.replace(" ", "\u2581")
        syms = [ch for ch in text]                       # one symbol per code point
        n = len(syms)
        if n == 0:
            return []
        length = [len(ch.encode("utf-8")) for ch in syms]  # merged byte length, as llama.cpp tracks it
        prev = [i - 1 for i in range(n)]
        nxt = [i + 1 if i + 1 < n else -1 for i in range(n)]
        text_of = list(syms)
        heap: list = []
        rev: dict[str, tuple[int, int]] = {}

        def try_add(left: int, right: int):
            if left == -1 or right == -1:
                return
            t = text_of[left] + text_of[right]
            tid = self.id_of.get(t)
            if tid is None:
                return
            heapq.heappush(heap, (-float(self.scores[tid]), left, length[left] + length[right], right, t))
            rev[t] = (left, right)

        for i in range(1, n):
            try_add(i - 1, i)
        while heap:
            _, left, size, right, t = heapq.heappop(heap)
            if length[left] == 0 or length[right] == 0 or length[left] + length[right] != size or text_of[left] + text_of[right] != t:
                continue
            text_of[left] = t
            length[left] += length[right]
            length[right] = 0
            nxt[left] = nxt[right]
            if nxt[right] >= 0:
                prev[nxt[right]] = left
            try_add(prev[left], left)
            try_add(left, nxt[left])
        out: list[int] = []

        def reseg(i: int):
            t = text_of[i]
            tid = self.id_of.get(t)
            if tid is not None:
                out.append(tid)
                return
            p = rev.get(t)
            if p is None:
                for b in t.encode("utf-8"):
                    out.append(self.id_of[f"<0x{b:02X}>"])
                return
            reseg(p[0])
            reseg(p[1])
        i = 0
        while i != -1:
            if length[i] > 0:
                reseg(i)
            i = nxt[i]
        return out

    def encode(self, text: str, add_bos: bool | None = None) -> list[int]:
        ids: list[int] = []
        first = True
        for is_special, chunk in self._split_specials(text):
            if is_special:
                ids.append(self.id_of[chunk])
            else:
                if first and self.add_space_prefix and not chunk.startswith(" "):
                    chunk = " " + chunk
                ids += self._spm(chunk)
            first = False
        want_bos = self.add_bos if add_bos is None else add_bos
        if want_bos and self.bos_id >= 0 and (not ids or ids[0] != self.bos_id):
            ids.insert(0, self.bos_id)
        return ids

    render_chat = BPETokenizer.render_chat
    prompt_ids = BPETokenizer.prompt_ids


def make_tokenizer(kv: dict):
    """The tokeniser for a GGUF vocabulary: byte-level BPE or sentencepiece."""
    model = kv.get("tokenizer.ggml.model")
    if model == "gpt2":
        return BPETokenizer(kv)
    if model == "llama":
        return SPMTokenizer(kv)
    raise ValueError(f"vocabulary model {model!r} not supported")
