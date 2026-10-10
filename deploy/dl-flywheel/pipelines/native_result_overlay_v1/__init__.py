"""native_result_overlay_v1 revision 2 (W3/X3): verified native results -> a DELTA chain of train-only overlay packets.

Interface the C10b boundary inbox calls (see README.md):
  build_packet(rows_iter, *, native, store, authority, ..., predecessor_chain=None, predecessor_head=None, root,
               deadline) -> receipt
  validate_chain(packet_refs_in_order, *, expected_head) -> ChainView (cumulative view)
  validate_packet(packet_ref, *, predecessor_head) -> packet dict (one packet, no whole-chain properties)
  head_sha256(packet_raw, predecessor_head) -> str
  overlay_snapshot(base_snapshot, chain_view, *, max_padded_tokens) -> snapshot
Exports are resolved lazily so that importing the pinned runner (``offline``) imports no base module before the
runner has fixed and pinned sys.path.
"""
import importlib

_EXPORTS = {'build_packet': 'build', 'validate_packet': 'packet', 'validate_chain': 'packet',
            'completed_packet': 'packet', 'head_sha256': 'packet', 'ChainView': 'packet',
            'overlay_snapshot': 'loader', 'overlay_ref_digest': 'loader', 'OverlayLoader': 'loader'}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(importlib.import_module(__name__ + '.' + _EXPORTS[name]), name)
