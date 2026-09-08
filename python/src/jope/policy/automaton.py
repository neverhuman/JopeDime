"""Grammar-masking automaton for policy_v3 decoding.

DecoderState tracks the finite-state position inside a genome sequence plus an
arity/depth/node-budget stack for embedded expressions. ``legal_ids()`` is the exact
set of next tokens; sampling within it makes every decode valid by construction:
- non-leaf expr tokens are masked unless the children's depth fits max_depth AND the
  node budget can still cover one leaf per pending child (termination by construction);
- ``map``/``derive`` are masked once max_expr_stages is reached;
- the elementwise var ``x`` is legal only inside ``map`` expressions.
"""

from __future__ import annotations

from typing import Sequence

from .tokens import BOS, EOS, ActionVocab, OpInfo

# Phases
_ENGINE = "ENGINE"
_TASK = "TASK"
_STAGE = "STAGE"
_VAL = "VAL"
_APPEND = "APPEND"
_EXPR = "EXPR"
_EXPR_END = "EXPR_END"
_EOS_PHASE = "EOS"
_DONE = "DONE"


class IllegalToken(ValueError):
    pass


class DecoderState:
    __slots__ = (
        "vocab",
        "phase",
        "engine",
        "task",
        "stage_count",
        "expr_stages",
        "cur_op",
        "slot_idx",
        "expr_stack",
        "expr_nodes",
        "strict_cols",
        "sampling",
    )

    def __init__(self, vocab: ActionVocab, *, strict_cols: bool = True, sampling: bool = True) -> None:
        self.vocab = vocab
        self.strict_cols = strict_cols
        # sampling=True (generation): parse-only ops (generate=False) are ILLEGAL — the
        # model must never propose quarantined/compat ops. sampling=False (replay):
        # permissive, so historical rows containing them still encode for training.
        self.sampling = sampling
        self.phase = _ENGINE
        self.engine: str | None = None
        self.task: str | None = None
        self.stage_count = 0
        self.expr_stages = 0
        self.cur_op: OpInfo | None = None
        self.slot_idx = 0
        self.expr_stack: list[tuple[int, int]] = []  # (children_remaining, node_depth)
        self.expr_nodes = 0

    def clone(self) -> "DecoderState":
        out = DecoderState(self.vocab, strict_cols=self.strict_cols, sampling=self.sampling)
        out.phase = self.phase
        out.engine = self.engine
        out.task = self.task
        out.stage_count = self.stage_count
        out.expr_stages = self.expr_stages
        out.cur_op = self.cur_op
        out.slot_idx = self.slot_idx
        out.expr_stack = list(self.expr_stack)
        out.expr_nodes = self.expr_nodes
        return out

    def is_done(self) -> bool:
        return self.phase == _DONE

    # -- legality -----------------------------------------------------------
    def legal_ids(self) -> list[int]:
        vocab = self.vocab
        if self.phase == _ENGINE:
            return [vocab.engine_ids[e] for e in vocab.engines]
        if self.phase == _TASK:
            return [vocab.task_ids[t] for t in vocab.tasks]
        if self.phase == _STAGE:
            assert self.engine is not None
            legal: list[int] = []
            if self.stage_count < vocab.max_transforms[self.engine]:
                for op_id in vocab.engine_transforms[self.engine]:
                    op = vocab.ops[op_id]
                    if op.takes_expr and self.expr_stages >= vocab.max_expr_stages[self.engine]:
                        continue
                    if self.sampling and not op.generate:
                        continue  # parse-only transforms never sampled
                    legal.append(op_id)
            # heads masked by task (e.g. featmath gpr is regression-only); sampling also
            # drops parse-only heads (replay stays permissive for historical rows)
            legal.extend(
                vocab.heads_for(self.engine, self.task or "regression", generate_only=self.sampling)
            )
            return legal
        if self.phase == _VAL:
            assert self.cur_op is not None
            slot = self.cur_op.slots[self.slot_idx]
            if slot.is_expr:
                # macro bins OR the root of an EXPR block (stack [(1,1)], 0 nodes used).
                # A bare const root is excluded: it would render as a number (not a
                # string param) and a constant-only expression is useless anyway.
                legal = list(slot.val_ids)
                const_ids = set(self.vocab.expr.const_ids)
                legal.extend(tid for tid in self._expr_legal([(1, 1)], 0) if tid not in const_ids)
                return legal
            legal = list(slot.val_ids)
            if slot.absent_id is not None:
                legal.append(slot.absent_id)
            return legal
        if self.phase == _APPEND:
            return list(self.vocab.append_ids)
        if self.phase == _EXPR:
            return self._expr_legal(self.expr_stack, self.expr_nodes)
        if self.phase == _EXPR_END:
            return [vocab.expr.end_id]
        if self.phase == _EOS_PHASE:
            return [EOS]
        return []

    def _expr_legal(self, stack: list[tuple[int, int]], nodes: int) -> list[int]:
        table = self.vocab.expr
        elementwise = self.cur_op is not None and self.cur_op.name == "map"
        # col(i) refs crash newmath at fit time when an earlier stage shrank the matrix
        # below i columns (RF4: 1,304 corpus failures) — newmath decodes never emit them.
        # Replay of existing corpus rows stays permissive via strict_cols=False.
        allow_col = not (self.engine == "newmath" and self.strict_cols)
        pending = sum(children for children, _ in stack)
        depth = stack[-1][1] if stack else 1
        legal = list(table.leaf_ids(elementwise, allow_col=allow_col))
        for tid, arity in table.func_arity.items():
            if depth + 1 > table.max_depth:
                continue  # candidate's children would exceed the depth cap
            if nodes + 1 + (pending - 1 + arity) > table.max_nodes:
                continue  # budget could no longer cover one leaf per pending child
            legal.append(tid)
        return legal

    # -- transition ---------------------------------------------------------
    def advance(self, token_id: int) -> None:
        if token_id not in self.legal_ids():
            token = self.vocab.tokens[token_id] if 0 <= token_id < self.vocab.size else token_id
            raise IllegalToken(f"illegal token {token!r} in phase {self.phase}")
        vocab = self.vocab
        if self.phase == _ENGINE:
            for engine, tid in vocab.engine_ids.items():
                if tid == token_id:
                    self.engine = engine
            self.phase = _TASK
            return
        if self.phase == _TASK:
            for task, tid in vocab.task_ids.items():
                if tid == token_id:
                    self.task = task
            self.phase = _STAGE
            return
        if self.phase == _STAGE:
            op = vocab.ops[token_id]
            self.cur_op = op
            self.slot_idx = 0
            if op.takes_expr:
                self.expr_stages += 1
            self._enter_slot_or_finish()
            return
        if self.phase == _VAL:
            assert self.cur_op is not None
            slot = self.cur_op.slots[self.slot_idx]
            if slot.is_expr and token_id not in slot.val_ids:
                # entering an EXPR block: this token is the root node at depth 1
                self.expr_stack = [(1, 1)]
                self.expr_nodes = 0
                self._consume_expr_token(token_id)
                return
            self.slot_idx += 1
            self._enter_slot_or_finish()
            return
        if self.phase == _APPEND:
            self._finish_stage()
            return
        if self.phase == _EXPR:
            self._consume_expr_token(token_id)
            return
        if self.phase == _EXPR_END:
            self.slot_idx += 1
            self._enter_slot_or_finish()
            return
        if self.phase == _EOS_PHASE:
            self.phase = _DONE
            return
        raise IllegalToken(f"advance called in phase {self.phase}")

    def _consume_expr_token(self, token_id: int) -> None:
        table = self.vocab.expr
        children, depth = self.expr_stack[-1]
        if children == 1:
            self.expr_stack.pop()
        else:
            self.expr_stack[-1] = (children - 1, depth)
        self.expr_nodes += 1
        arity = table.func_arity.get(token_id)
        if arity is not None:
            self.expr_stack.append((arity, depth + 1))
        self.phase = _EXPR if self.expr_stack else _EXPR_END

    def _enter_slot_or_finish(self) -> None:
        assert self.cur_op is not None
        if self.slot_idx < len(self.cur_op.slots):
            self.phase = _VAL
            return
        if self.cur_op.allow_append:
            self.phase = _APPEND
            return
        self._finish_stage()

    def _finish_stage(self) -> None:
        assert self.cur_op is not None
        if self.cur_op.role == "head":
            self.phase = _EOS_PHASE
        else:
            self.stage_count += 1
            self.phase = _STAGE
        self.cur_op = None
        self.slot_idx = 0


_REPLAY_CACHE: dict[tuple[int, tuple[int, ...]], list[list[int]]] = {}
_REPLAY_CACHE_MAX = 65536


def replay_masks(token_ids: Sequence[int], vocab: ActionVocab) -> list[list[int]]:
    """Legal-id lists for predicting each token at positions 1..len-1.

    ``token_ids`` must start with BOS (given, not predicted). Memoized — corpus rows
    repeat by canonical genome, so replay cost amortizes to ~0 after the first epoch.
    """
    key = (id(vocab), tuple(token_ids))
    cached = _REPLAY_CACHE.get(key)
    if cached is not None:
        return cached
    if not token_ids or token_ids[0] != BOS:
        raise ValueError("token sequence must start with BOS")
    # Replay is PERMISSIVE (sampling=False): historical rows containing parse-only ops
    # (generate=False) must still encode for training; only free generation masks them.
    state = DecoderState(vocab, strict_cols=False, sampling=False)
    masks: list[list[int]] = []
    for tid in token_ids[1:]:
        masks.append(state.legal_ids())
        state.advance(tid)
    if not state.is_done():
        raise ValueError("token sequence is incomplete (no EOS after head)")
    if len(_REPLAY_CACHE) >= _REPLAY_CACHE_MAX:
        _REPLAY_CACHE.clear()
    _REPLAY_CACHE[key] = masks
    return masks
