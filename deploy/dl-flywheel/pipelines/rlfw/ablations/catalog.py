"""Compressed trie over complete BOS..EOS programs.

``get`` matches ``rlfw.trie_logprob``: the prefix includes BOS, and children
are the tokens that follow it in the catalog.
"""

from __future__ import annotations


class CatalogTrie:
    def __init__(self, programs):
        if not programs:
            raise ValueError("trie requires at least one program")
        self.programs = {}
        self.nodes = [{}]
        for program in programs:
            tokens = tuple(program)
            if len(tokens) < 2:
                raise ValueError("program needs BOS and EOS")
            if tokens in self.programs:
                raise ValueError("duplicate program")
            self.programs[tokens] = tokens
            node = 0
            for token in tokens[1:]:
                if type(token) is not int:
                    raise ValueError("token ids must be ints")
                children = self.nodes[node]
                if token not in children:
                    children[token] = len(self.nodes)
                    self.nodes.append({})
                node = children[token]

    def __len__(self):
        return len(self.programs)

    def get(self, prefix):
        node = 0
        for token in tuple(prefix)[1:]:
            children = self.nodes[node]
            if token not in children:
                return ()
            node = children[token]
        return self.nodes[node]
