"""Exact frozen prefix index for SCRATCH supports beyond the eager trie cap.

Sorted big-endian uint16 token strings encode every leaf. Only prefixes visited
by actual likelihood/sampling calls materialize child tables. The decoder is
the unchanged BatchedTrie.decode_many, with the same row seeds/float64 softmax.
"""

import hashlib
import struct
from bisect import bisect_left
from collections.abc import Mapping

from .successor_support import digest


def encode(tokens):
    if any(type(t) is not int or not 0 <= t < 65535 for t in tokens):
        raise ValueError("bounded uint16 tokens required")
    return struct.pack(f">{len(tokens)}H", *tokens)


def decode(key):
    return struct.unpack(f">{len(key) // 2}H", key)


class _Leaves(Mapping):
    def __init__(self, index):
        self.index = index

    def __len__(self):
        return len(self.index.keys)

    def __iter__(self):
        return (decode(key) for key in self.index.keys)

    def __getitem__(self, tokens):
        key = encode(tokens)
        pos = bisect_left(self.index.keys, key)
        if pos == len(self.index.keys) or self.index.keys[pos] != key:
            raise KeyError(tokens)
        return self.index.program_rows[pos]


class _Nodes:
    def __init__(self, index):
        self.index = index
        self.prefixes = [encode([1])]
        self.cursors = {self.prefixes[0]: 0}
        self.tables = {}

    def __getitem__(self, cursor):
        if cursor in self.tables:
            return self.tables[cursor]
        prefix = self.prefixes[cursor]
        keys = self.index.keys
        lo = bisect_left(keys, prefix)
        hi = bisect_left(keys, prefix + b"\xff\xff")
        children = {}
        while lo < hi:
            key = keys[lo]
            if len(key) <= len(prefix):
                lo += 1
                continue
            child_prefix = key[: len(prefix) + 2]
            token = decode(child_prefix[-2:])[0]
            child = self.cursors.get(child_prefix)
            if child is None:
                child = len(self.prefixes)
                self.cursors[child_prefix] = child
                self.prefixes.append(child_prefix)
            children[token] = child
            lo = bisect_left(keys, child_prefix + b"\xff\xff", lo, hi)
        self.tables[cursor] = children
        return children


class CompactSupportTrie:
    """Input tokens must have already passed native identity/canonical checks."""

    def __init__(self, programs, *, encoded_by_sha, vocabulary_sha256, max_seq):
        if not programs or len(programs) > 4_000_000:
            raise ValueError("bounded nonempty scratch support required")
        pairs = []
        for program in programs:
            key = encoded_by_sha[program["program_sha256"]]
            tokens = decode(key)
            if (
                not 2 <= len(tokens) <= max_seq
                or tokens[0] != 1
                or tokens[-1] != 2
                or 2 in tokens[:-1]
                or 1 in tokens[1:]
            ):
                raise ValueError("complete representable BOS/EOS leaves required")
            pairs.append((key, program))
        pairs.sort(key=lambda pair: pair[0])
        self.keys = [key for key, _ in pairs]
        self.program_rows = [program for _, program in pairs]
        if any(a == b for a, b in zip(self.keys, self.keys[1:])):
            raise ValueError("duplicate raw leaf refused")
        self.max_seq = max_seq
        tokens_hash = hashlib.sha256()
        for key in self.keys:
            tokens_hash.update(key)
        self.identity = digest(
            {
                "schema": "jope.rlfw-compact-scratch-trie/v1",
                "vocabulary_sha256": vocabulary_sha256,
                "max_seq": max_seq,
                "programs_sha256": digest(programs),
                "tokens_sha256": tokens_hash.hexdigest(),
            }
        )
        self.programs = _Leaves(self)
        self.nodes = _Nodes(self)

    def get(self, prefix, default=()):
        if not prefix or prefix[0] != 1:
            return default
        node = 0
        for token in prefix[1:]:
            table = self.nodes[node]
            if token not in table:
                return default
            node = table[token]
        return self.nodes[node]

    def decode_many(self, *args, **kwargs):
        from neural_decode import BatchedTrie

        return BatchedTrie.decode_many(self, *args, **kwargs)
