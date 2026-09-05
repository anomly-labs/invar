# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
invar.tlog — an append-only transparency log of Signed Statements with inclusion and
consistency proofs (RFC 6962 Merkle tree hashing, the construction SCITT registries use).

OpenPCC's transparency log holds software images. This one holds COMPUTATIONS: each leaf
is a SCITT-style Signed Statement (invar.scitt) whose payload is a canonical CR manifest,
digests only, so registering an inference never leaks the prompt or the answer. A client
who holds a statement and an inclusion proof can check, offline, that the log committed
to it under a given tree head; anyone holding two tree heads can check the log only ever
appended (consistency proof). Registering returns a *receipt* in the IETF sense: the
tree head plus the inclusion path.

Hashing (RFC 6962 §2.1): leaf = sha256(0x00 || data), node = sha256(0x01 || L || R).
Storage: one file, leaves appended as base64 lines; the tree is recomputed on load
(fine to millions of leaves; a persisted node cache is a later optimisation).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import threading


def _leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _root(hashes: list[bytes]) -> bytes:
    """Merkle Tree Hash of a leaf-hash list (RFC 6962 §2.1)."""
    n = len(hashes)
    if n == 0:
        return hashlib.sha256(b"").digest()
    if n == 1:
        return hashes[0]
    k = _largest_pow2_lt(n)
    return _node_hash(_root(hashes[:k]), _root(hashes[k:]))


def _largest_pow2_lt(n: int) -> int:
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def _inclusion_path(hashes: list[bytes], m: int) -> list[bytes]:
    """Audit path for leaf m in a tree of len(hashes) leaves (RFC 6962 §2.1.1)."""
    n = len(hashes)
    if n <= 1:
        return []
    k = _largest_pow2_lt(n)
    if m < k:
        return _inclusion_path(hashes[:k], m) + [_root(hashes[k:])]
    return _inclusion_path(hashes[k:], m - k) + [_root(hashes[:k])]


def verify_inclusion(leaf_data: bytes, index: int, tree_size: int,
                     path: list[bytes], root: bytes) -> bool:
    """RFC 6962 §2.1.1 verification, iterative."""
    if index >= tree_size or tree_size <= 0:
        return False
    h = _leaf_hash(leaf_data)
    fn, sn = index, tree_size - 1
    for p in path:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            h = _node_hash(p, h)
            while not (fn & 1) and fn != 0:
                fn >>= 1
                sn >>= 1
        else:
            h = _node_hash(h, p)
        fn >>= 1
        sn >>= 1
    return sn == 0 and h == root


def _consistency_path(hashes: list[bytes], m: int) -> list[bytes]:
    """Consistency proof between the first m leaves and all n (RFC 6962 §2.1.2)."""
    n = len(hashes)
    if m == n or m == 0:
        return []
    return _subproof(hashes, m, True)


def _subproof(hashes: list[bytes], m: int, b: bool) -> list[bytes]:
    n = len(hashes)
    if m == n:
        return [] if b else [_root(hashes)]
    k = _largest_pow2_lt(n)
    if m <= k:
        return _subproof(hashes[:k], m, b) + [_root(hashes[k:])]
    return _subproof(hashes[k:], m - k, False) + [_root(hashes[:k])]


def verify_consistency(old_size: int, old_root: bytes, new_size: int, new_root: bytes,
                       path: list[bytes]) -> bool:
    """RFC 9162 §2.1.4.2 consistency verification."""
    if old_size == new_size:
        return old_root == new_root and not path
    if old_size == 0:
        return not path                    # empty tree is consistent with anything
    if old_size > new_size or not path:
        return False
    path = list(path)
    fn, sn = old_size - 1, new_size - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1
    if fn == 0:                            # old_size is a power of two: first node is the old root
        fr = sr = old_root
    else:
        fr = sr = path.pop(0)
    for p in path:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            fr = _node_hash(p, fr)
            sr = _node_hash(p, sr)
            while not (fn & 1) and fn != 0:
                fn >>= 1
                sn >>= 1
        else:
            sr = _node_hash(sr, p)
        fn >>= 1
        sn >>= 1
    return sn == 0 and fr == old_root and sr == new_root


class TransparencyLog:
    """Append-only leaf store with Merkle proofs. One log per Ledger."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._leaves: list[bytes] = []
        self._hashes: list[bytes] = []
        if os.path.exists(path):
            with open(path, "rb") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = base64.b64decode(line)
                        self._leaves.append(d)
                        self._hashes.append(_leaf_hash(d))

    @property
    def size(self) -> int:
        return len(self._leaves)

    def root(self) -> bytes:
        return _root(self._hashes)

    def head(self) -> dict:
        return {"tree_size": self.size, "root": "sha256:" + self.root().hex()}

    def append(self, data: bytes) -> dict:
        """Register a statement; returns the registration receipt (tree head +
        inclusion path), which the registrant keeps."""
        with self._lock:
            with open(self.path, "ab") as f:
                f.write(base64.b64encode(data) + b"\n")
            self._leaves.append(data)
            self._hashes.append(_leaf_hash(data))
            idx = len(self._leaves) - 1
            return self._receipt(idx)

    def _receipt(self, idx: int) -> dict:
        path = _inclusion_path(self._hashes, idx)
        return {"index": idx, "tree_size": self.size, "root": "sha256:" + self.root().hex(),
                "leaf_hash": "sha256:" + self._hashes[idx].hex(),
                "path": ["sha256:" + p.hex() for p in path]}

    def inclusion(self, idx: int) -> dict:
        with self._lock:
            if not 0 <= idx < self.size:
                raise IndexError(idx)
            return self._receipt(idx)

    def leaf(self, idx: int) -> bytes:
        return self._leaves[idx]

    def consistency(self, old_size: int) -> dict:
        with self._lock:
            if not 0 <= old_size <= self.size:
                raise IndexError(old_size)
            path = _consistency_path(self._hashes, old_size)
            return {"old_size": old_size, "tree_size": self.size,
                    "root": "sha256:" + self.root().hex(),
                    "path": ["sha256:" + p.hex() for p in path]}

    def find(self, data: bytes) -> int:
        """Index of an exact leaf, or -1 (linear; fine for a per-Ledger log)."""
        try:
            return self._leaves.index(data)
        except ValueError:
            return -1


def _hexes(xs: list[str]) -> list[bytes]:
    return [bytes.fromhex(x.split(":", 1)[1]) for x in xs]


def check_receipt(leaf_data: bytes, receipt: dict) -> bool:
    """Offline check of a registration receipt against the leaf bytes you hold."""
    return verify_inclusion(leaf_data, receipt["index"], receipt["tree_size"],
                            _hexes(receipt["path"]),
                            bytes.fromhex(receipt["root"].split(":", 1)[1]))


def check_consistency(old_head: dict, proof: dict) -> bool:
    return verify_consistency(old_head["tree_size"],
                              bytes.fromhex(old_head["root"].split(":", 1)[1]),
                              proof["tree_size"], bytes.fromhex(proof["root"].split(":", 1)[1]),
                              _hexes(proof["path"]))


def dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)
