"""Synthetic adversarial tests of the tensor inventory verifier (W2-T06).

The accepted asset is not available in CI; these fixtures exercise the same parser, digest,
role and forward-reference rules on miniature safetensors. The real-asset check runs on the
host through verify_inventory.py itself and is retained as custody evidence.
"""
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import zipfile

import verify_inventory as inventory

SOURCE = '''
fn encode_context(&self) {}
fn decoder_states(&self) { self.g("ctx_proj.weight"); self.encode_context("encoder", c); }
fn reward_runtime(&self) { self.g("runtime_head.0.weight"); }
#[cfg(feature = "inference-cuda")]
fn qualification() { self.g("stability_head.0.weight"); }
'''
ROLES = {"decode": ("encoder", "ctx_proj"), "rank": ("runtime_head",), "training_only": ("stability_head",)}
SHAPES = {"encoder.0.weight": [2, 3], "ctx_proj.weight": [3, 3], "ctx_proj.bias": [3],
          "runtime_head.0.weight": [3, 1], "stability_head.0.weight": [2, 3]}


def safetensors(shapes, values=None, trailing=b"", offsets=None):
    header, payload, cursor = {}, b"", 0
    for index, (name, shape) in enumerate(shapes.items()):
        count = math.prod(shape)
        data = struct.pack(f"<{count}f", *((values or {}).get(name) or [index + 0.5] * count))
        begin, end = (offsets or {}).get(name, (cursor, cursor + len(data)))
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [begin, end]}
        payload += data
        cursor += len(data)
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + payload + trailing


def record(raw):
    header, payload = inventory.parse_safetensors(raw)
    tensors, roles = [], {role: 0 for role in ROLES}
    for name, entry in header.items():
        begin, end = entry["data_offsets"]
        count = math.prod(entry["shape"])
        role = next(r for r, roots in ROLES.items() if name.split(".")[0] in roots)
        roles[role] += count
        tensors.append({"name": name, "shape": entry["shape"], "dtype": "F32", "parameters": count,
                        "role": role, "payload_sha256": hashlib.sha256(payload[begin:end]).hexdigest()})
    total = sum(t["parameters"] for t in tensors)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "tensors": tensors, "parameters": total,
            "tensor_count": len(tensors), "role_parameters": roles}


class InventoryTests(unittest.TestCase):
    def patched(self):
        total = sum(math.prod(s) for s in SHAPES.values())
        return [mock.patch.object(inventory, "ROLES", ROLES),
                mock.patch.object(inventory, "expected_geometry", lambda source: dict(SHAPES)),
                mock.patch.object(inventory, "EXPECTED_PARAMETERS", total),
                mock.patch.object(inventory, "EXPECTED_TENSORS", len(SHAPES))]

    def check(self, record_value, raw, source=SOURCE, reason=None):
        patches = self.patched()
        for patch in patches:
            patch.start()
        try:
            if reason is None:
                return inventory.verify(record_value, raw, source)
            with self.assertRaisesRegex(ValueError, reason):
                inventory.verify(record_value, raw, source)
        finally:
            for patch in patches:
                patch.stop()

    def test_valid_fixture_accounts_every_role(self):
        raw = safetensors(SHAPES)
        summary = self.check(record(raw), raw)
        self.assertEqual(summary["parameters"]["stored"], sum(math.prod(s) for s in SHAPES.values()))
        self.assertEqual(summary["forward_references"], ["ctx_proj", "encoder", "runtime_head"])

    def test_payload_bit_flip_is_localised(self):
        raw = safetensors(SHAPES)
        pinned = record(raw)
        flipped = bytearray(raw)
        flipped[-1] ^= 1  # last byte belongs to stability_head.0.weight
        pinned["sha256"] = hashlib.sha256(flipped).hexdigest()
        self.check(pinned, bytes(flipped), reason="stability_head.0.weight: payload digest differs")

    def test_same_shape_value_substitution_is_rejected(self):
        raw = safetensors(SHAPES)
        pinned = record(raw)
        swapped = safetensors(SHAPES, values={"encoder.0.weight": [9.0] * 6})
        pinned["sha256"] = hashlib.sha256(swapped).hexdigest()
        self.check(pinned, swapped, reason="encoder.0.weight: payload digest differs")

    def test_whole_file_digest_is_bound(self):
        raw = safetensors(SHAPES)
        self.check(record(raw), safetensors(SHAPES, values={"ctx_proj.bias": [1, 2, 3]}),
                   reason="asset digest differs")

    def test_layout_mutations_are_rejected(self):
        cases = {
            "trailing": (safetensors(SHAPES, trailing=b"\0\0\0\0"), "trailing payload bytes"),
            "overlap": (safetensors(SHAPES, offsets={"ctx_proj.bias": (0, 12)}), "gap or overlap"),
            "short": (safetensors(SHAPES, offsets={"ctx_proj.bias": (60, 68)}), "byte length differs"),
            "outside": (safetensors(SHAPES, offsets={"ctx_proj.bias": (1000, 1012)}), "outside payload"),
            "truncated": (b"\x10\0\0\0\0\0\0\0{}", "truncated header"),
        }
        for label, (raw, reason) in cases.items():
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, reason):
                inventory.parse_safetensors(raw)

    def test_dtype_and_fields_are_closed(self):
        raw = safetensors(SHAPES)
        size = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + size])
        for label, change, reason in [("dtype", ("dtype", "F16"), "dtype F16"),
                                      ("field", ("extra", 1), "header fields")]:
            with self.subTest(label=label):
                changed = json.loads(json.dumps(header))
                changed["ctx_proj.bias"][change[0]] = change[1]
                encoded = json.dumps(changed).encode()
                mutated = struct.pack("<Q", len(encoded)) + encoded + raw[8 + size:]
                with self.assertRaisesRegex(ValueError, reason):
                    inventory.parse_safetensors(mutated)

    def test_role_and_forward_reference_rules(self):
        raw = safetensors(SHAPES)
        pinned = record(raw)
        wrong = json.loads(json.dumps(pinned))
        wrong["tensors"][0]["role"] = "rank"
        self.check(wrong, raw, reason="record role")
        wrong = json.loads(json.dumps(pinned))
        wrong["role_parameters"]["decode"] += 1
        self.check(wrong, raw, reason="record role accounting")
        # Training-only weights used by inference, or inference weights left unused, both fail.
        leaking = SOURCE.replace('self.g("runtime_head.0.weight");',
                                 'self.g("runtime_head.0.weight"); self.g("stability_head.0.weight");')
        self.check(pinned, raw, source=leaking, reason="Candle forward roots")
        unused = SOURCE.replace('self.g("runtime_head.0.weight");', "")
        self.check(pinned, raw, source=unused, reason="Candle forward roots")
        with mock.patch.object(inventory, "ROLES", ROLES), self.assertRaisesRegex(ValueError, "no single role"):
            inventory.role_of("mystery.weight")

    def test_optimizer_moments_must_cover_every_stored_parameter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimizer.ot"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("optimizer/data.pkl", b"pickle")
                archive.writestr("optimizer/data/0", b"\0" * 24)
                archive.writestr("optimizer/data/1", b"\0" * 24)
            self.assertEqual(inventory.optimizer_moment_bytes(path), 48)


if __name__ == "__main__":
    unittest.main()
