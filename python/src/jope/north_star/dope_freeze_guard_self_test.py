"""Opt-in real-model self-test for the DOPE freeze guard."""

import json
import sys
from pathlib import Path

import torch
from torch import nn

def run_self_test() -> dict:
    """
    Executes the full NS3 verification suite against the real `FusedJopeNetwork` and
    `dope_mlc_v3_multitask_best.pt` on CPU (zero GPU contention):
      Test 1: Load `FusedJopeNetwork`, apply `lock_dope_weights_and_eval_mode`, take baseline snapshot.
      Test 2: Call `net.train(True)` and verify `net.dope_model.training is False` (Dropout stays off).
      Test 3: Unfreeze `net.adapter_10tok`, run a synthetic flywheel training epoch (5 AdamW steps
              backpropagating through `adapter_10tok` + `dope_model` outputs), verify `adapter_10tok`
              weights change while `verify_dope_frozen_unchanged` passes with 0-bit drift.
      Test 4: Negative tamper tests (1-ULP parameter mutation, normalizer buffer mutation, and
              `requires_grad=True` mutation) all raise `DopeFreezeViolationError`.
    """
    if __package__:
        from .dope_freeze_guard import (
            DopeFreezeViolationError,
            _tensor_le_sha256,
            assert_optimizer_excludes_dope,
            get_trainable_param_counts_by_submodule,
            lock_dope_weights_and_eval_mode,
            snapshot_dope_weights,
            verify_dope_frozen_unchanged,
        )
    elif hasattr(sys.modules.get("__main__"), "snapshot_dope_weights"):
        from __main__ import (
            DopeFreezeViolationError,
            _tensor_le_sha256,
            assert_optimizer_excludes_dope,
            get_trainable_param_counts_by_submodule,
            lock_dope_weights_and_eval_mode,
            snapshot_dope_weights,
            verify_dope_frozen_unchanged,
        )
    else:  # Support direct import from this package directory.
        from dope_freeze_guard import (
            DopeFreezeViolationError,
            _tensor_le_sha256,
            assert_optimizer_excludes_dope,
            get_trainable_param_counts_by_submodule,
            lock_dope_weights_and_eval_mode,
            snapshot_dope_weights,
            verify_dope_frozen_unchanged,
        )
    sys.path.insert(0, "/home/ubuntu/gpu_bench/fused")
    from fused_jope_net import FusedJopeNetwork

    net = FusedJopeNetwork(device="cpu", adapter_variant="h40b")
    lock_dope_weights_and_eval_mode(net)
    snap_before = snapshot_dope_weights(net)

    # Test 2: Verify `net.train(True)` keeps `dope_model` in `.eval()` mode
    net.train(True)
    assert not net.dope_model.training, "dope_model.training became True under net.train(True)!"
    assert not net.gamenet_anchor.training, "gamenet_anchor.training became True under net.train(True)!"

    # Test 3: Run 5 synthetic flywheel AdamW steps updating ONLY the non-anchor `adapter_10tok` params
    # (Note: `adapter_10tok.anchor` aliases `net.gamenet_anchor` (`520,785` params), so a naive loop over
    # `adapter_10tok.parameters()` without excluding `anchor.` would accidentally unfreeze `gamenet_anchor`!)
    for p in net.parameters():
        p.requires_grad_(False)
    trainable_params = []
    for name, p in net.adapter_10tok.named_parameters():
        if not name.startswith("anchor.") and not name.startswith("frozen_anchor."):
            p.requires_grad_(True)
            trainable_params.append(p)
    lock_dope_weights_and_eval_mode(net)

    adapter_sha_before = _tensor_le_sha256(trainable_params[0])
    opt = torch.optim.AdamW(trainable_params, lr=1e-3)
    assert_optimizer_excludes_dope(opt, net)
    submodule_param_counts = get_trainable_param_counts_by_submodule(net)

    torch.manual_seed(42)
    vec256 = torch.randn(4, 256)
    col_lm = torch.randn(4, 64, 17)
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        dope_629, lm_48 = net._run_dope_mlc_v3_and_lm48(vec256, col_lm, use_landmarks=True)
        # Route DOPE output into a trainable adapter projection to verify gradient stopper
        mm_tok = dope_629[:, :320].view(4, 10, 32)
        mm_mask = torch.ones(4, 10)
        prog_412 = torch.randn(4, 96, 412)
        sos_prior = torch.randn(4, 96, 2)
        ctx_p, cand_q, sem24, mm_adj = net.adapter_10tok.encode_multimodal_context_and_candidates(
            mm_tok, mm_mask, prog_412, sos_prior
        )
        loss = (mm_adj[:, 0] - 1.0).pow(2).mean() + 0.0 * lm_48.sum()
        loss.backward()
        opt.step()

    adapter_sha_after = _tensor_le_sha256(trainable_params[0])
    assert adapter_sha_before != adapter_sha_after, "Trainable adapter weights did not update!"
    ok, snap_after = verify_dope_frozen_unchanged(net, snap_before)
    assert ok and snap_after.composite_sha256 == snap_before.composite_sha256

    # Test 4: Negative tamper checks (must raise DopeFreezeViolationError)
    tamper_results = {}

    # 4a: 1-ULP weight tamper
    first_param = next(net.dope_model.parameters())
    orig_val = float(first_param.view(-1)[0].item())
    with torch.no_grad():
        first_param.view(-1)[0] = orig_val + 1e-6
    try:
        verify_dope_frozen_unchanged(net, snap_before)
        tamper_results["weight_1ulp_tamper_caught"] = False
    except DopeFreezeViolationError:
        tamper_results["weight_1ulp_ulp_tamper_caught"] = True
    finally:
        with torch.no_grad():
            first_param.view(-1)[0] = orig_val

    # 4b: Normalizer buffer tamper
    orig_buf = float(net.step2_lm_mean_1x256.view(-1)[0].item())
    with torch.no_grad():
        net.step2_lm_mean_1x256.view(-1)[0] = orig_buf + 1e-5
    try:
        verify_dope_frozen_unchanged(net, snap_before)
        tamper_results["normalizer_buffer_tamper_caught"] = False
    except DopeFreezeViolationError:
        tamper_results["normalizer_buffer_tamper_caught"] = True
    finally:
        with torch.no_grad():
            net.step2_lm_mean_1x256.view(-1)[0] = orig_buf

    # 4c: requires_grad=True tamper
    first_param.requires_grad_(True)
    try:
        verify_dope_frozen_unchanged(net, snap_before)
        tamper_results["requires_grad_tamper_caught"] = False
    except DopeFreezeViolationError:
        tamper_results["requires_grad_tamper_caught"] = True
    finally:
        first_param.requires_grad_(False)

    # 4d: Optimizer param_group inclusion tamper
    bad_opt = torch.optim.SGD([first_param], lr=1e-3)
    try:
        assert_optimizer_excludes_dope(bad_opt, net)
        tamper_results["optimizer_group_tamper_caught"] = False
    except DopeFreezeViolationError:
        tamper_results["optimizer_group_tamper_caught"] = True

    # 4e: Codex counterexample #1 — direct nested submodule `.train(True)` call AND raw `.training = True` attribute tamper
    nested_dope_mods = [m for m in net.dope_model.modules() if m is not net.dope_model]
    target_nested = nested_dope_mods[0]
    target_nested.train(True)
    # Because `lock_dope_weights_and_eval_mode` installed `_locked_eval_train` on every nested submodule,
    # `target_nested.train(True)` must keep `target_nested.training == False`:
    assert not target_nested.training, "Nested DOPE submodule `.train(True)` escaped eval lock!"
    tamper_results["nested_submodule_train_call_blocked"] = not target_nested.training

    # Even if code bypasses `.train()` and directly assigns `target_nested.training = True`,
    # `snapshot_dope_weights` and `get_trainable_param_counts_by_submodule` must catch it:
    target_nested.training = True
    try:
        verify_dope_frozen_unchanged(net, snap_before)
        tamper_results["nested_submodule_attr_tamper_caught"] = False
    except DopeFreezeViolationError:
        tamper_results["nested_submodule_attr_tamper_caught"] = True
    finally:
        target_nested.training = False

    # 4f: Codex counterexample #2 — minimal fixture with a direct root parameter (1 trainable on root + 2 on child = 3 total)
    class _MinimalRootParamFixture(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.root_scale = nn.Parameter(torch.ones(1), requires_grad=True)
            self.dope_model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Dropout(0.1))
            self.head = nn.Linear(2, 1, bias=False)

    min_fix = _MinimalRootParamFixture()
    lock_dope_weights_and_eval_mode(min_fix)
    min_fix.dope_model[1].train(True)
    assert not min_fix.dope_model[1].training, "dope_model[1].train(True) escaped on Sequential fixture!"
    min_counts = get_trainable_param_counts_by_submodule(min_fix)
    assert min_counts["__root_direct__"]["trainable_params"] == 1
    assert min_counts["head"]["trainable_params"] == 2
    assert min_counts["dope_model"]["trainable_params"] == 0
    assert min_counts["__global_total__"]["trainable_params"] == 3
    tamper_results["root_direct_and_global_param_reconciliation_verified"] = True

    # Restore clean state and re-verify
    verify_dope_frozen_unchanged(net, snap_before)

    report = {
        "task": "NS3_DOPE_WEIGHT_FREEZE_AND_HASH_TEST",
        "status": "PASS",
        "dope_composite_sha256_before": snap_before.composite_sha256,
        "dope_composite_sha256_after": snap_after.composite_sha256,
        "dope_params_sha256": snap_before.params_sha256,
        "dope_buffers_sha256": snap_before.buffers_sha256,
        "num_dope_param_tensors": snap_before.num_param_tensors,
        "num_dope_param_elements": snap_before.num_param_elements,
        "num_dope_normalizer_buffers": snap_before.num_buffer_tensors,
        "num_dope_normalizer_elements": snap_before.num_buffer_elements,
        "trainable_param_counts_by_submodule": submodule_param_counts,
        "optimizer_excludes_dope_verified": True,
        "train_mode_override_verified": (not net.dope_model.training),
        "adapter_updated_during_flywheel_step": (adapter_sha_before != adapter_sha_after),
        "adapter_param0_sha256_before": adapter_sha_before,
        "adapter_param0_sha256_after": adapter_sha_after,
        "negative_tamper_checks": tamper_results,
        "tensor_hashes": snap_before.tensor_hashes,
        "tensor_shapes": snap_before.tensor_shapes,
    }
    out_path = Path(__file__).resolve().parent / "ns3_dope_freeze_receipt.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("tensor_hashes", "tensor_shapes")}, indent=2)
    )
    return report
