"""Offline proofs for dltrain.family_screen (PLAN §5 P2, spec SCREEN). Synthetic tables only: no S3, no SQLite.

Test data come from sha256 (det_unit), never from RNG state.
"""
import ast
import gzip
import hashlib
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dltrain import common
from dltrain import family_screen as fs

P = (1 << 61) - 1
SRC = Path(fs.__file__)


def det_unit(n: int, label: str) -> np.ndarray:
    """n deterministic floats in [0, 1) derived from sha256(label:i)."""
    buf = bytearray()
    i = 0
    while len(buf) < 8 * n:
        buf += hashlib.sha256(f"{label}:{i}".encode()).digest()
        i += 1
    words = np.frombuffer(bytes(buf[:8 * n]), dtype="<u8") >> np.uint64(11)
    return words.astype(np.float64) / float(1 << 53)


def det_perm(n: int, label: str) -> np.ndarray:
    return np.argsort(det_unit(n, label), kind="stable")


def base_table(label: str, n_rows: int = 400, n_cols: int = 5) -> np.ndarray:
    return np.column_stack([np.round(det_unit(n_rows, f"{label}:{c}"), 4) for c in range(n_cols)])


def epoch(identity: str, values) -> fs.Table:
    return fs.make_table(identity, "epoch", values)


def pair_of(links, a, b):
    return [r for r in links if (r["a"], r["b"]) == tuple(sorted((a, b)))]


class TokensAndHashes(unittest.TestCase):
    def test_tokens_round_to_five_and_fold_negative_zero(self):
        x = np.array([[-0.0, 0.177777782, 0.177777778, 1e-7, -1e-7, 0.123455, np.inf, np.nan]])
        t = fs.tokens(x, 5)
        self.assertEqual(t[0, :5].tolist(), [0.0, 0.17778, 0.17778, 0.0, 0.0])
        self.assertFalse(np.signbit(t[0, :5]).any())
        self.assertEqual(t[0, 0].tobytes(), np.float64(0.0).tobytes())
        self.assertEqual(t[0, 5], np.round(np.float64(0.123455), 5) + 0.0)
        self.assertTrue(np.isnan(t[0, 6]) and np.isnan(t[0, 7]))

    def test_row_multiset_sha256_matches_the_spec_formula(self):
        a = base_table("rms", 50, 3)
        tok = fs.tokens(a, 5)
        ref = hashlib.sha256(b"".join(sorted(hashlib.sha256(r.astype("<f8").tobytes()).digest() for r in tok))).hexdigest()
        self.assertEqual(fs.row_multiset_sha256(tok), (ref, 50))
        self.assertEqual(fs.row_multiset_sha256(tok[det_perm(50, "rms:p")])[0], ref)          # order-free
        dup = np.vstack([tok, tok[:1]])
        self.assertNotEqual(fs.row_multiset_sha256(dup)[0], ref)                              # a multiset
        self.assertEqual(fs.row_multiset_sha256(np.empty((5, 0))), (None, 0))

    def test_minhash_params_follow_the_domain_rule(self):
        a, b = fs.minhash_params(b"jope.dl-family-nds/v1", 64)
        self.assertEqual((a, b), fs.minhash_params("jope.dl-family-nds/v1", 64))
        for j in (0, 1, 63):
            d = hashlib.sha256(b"jope.dl-family-nds/v1" + j.to_bytes(2, "big")).digest()
            self.assertEqual(a[j], int.from_bytes(d[0:8], "big") % P)
            self.assertEqual(b[j], int.from_bytes(d[8:16], "big") % P)
        self.assertTrue(all(0 < x < P for x in a) and all(0 <= x < P for x in b))
        self.assertEqual(len(set(zip(a, b))), 64)

    def test_mulmod61_is_exact_against_python_integers(self):
        words = np.frombuffer(b"".join(hashlib.sha256(f"mm:{i}".encode()).digest() for i in range(500)), dtype="<u8")
        edge = np.array([0, 1, 2, P - 1, P - 2, (1 << 32) - 1, 1 << 32, 1 << 60, (1 << 64) - 1, P, P + 1],
                        dtype=np.uint64)
        h = np.concatenate([edge, words.astype(np.uint64)])
        a_list, b_list = fs.minhash_params("jope.dl-family-nds/v1", 64)
        for a, b in list(zip(a_list, b_list))[:16] + [(1, 0), (P - 1, P - 1), ((1 << 32) - 1, 7), (1 << 32, 3)]:
            got = fs.affine_mod61(h, a, b)
            self.assertEqual(got.tolist(), [(a * int(v) + b) % P for v in h.tolist()])

    def test_signatures_match_the_reference_definition(self):
        cols = [np.array([0.1, 0.25, 0.5, 0.75, 1.0, 0.33333, 0.0, 0.66667, 0.2, 0.4, 0.6, 0.8, 0.9]),
                np.round(det_unit(40, "sig"), 5), np.array([0.5, 0.5, 0.25, 0.125])]
        sig = fs.minhash_signatures(cols, "jope.dl-family-nds/v1", 64)
        a_list, b_list = fs.minhash_params("jope.dl-family-nds/v1", 64)

        def h(t):
            return int.from_bytes(hashlib.blake2b(struct.pack("<d", t), digest_size=8).digest(), "big")

        for c, col in enumerate(cols):
            ref = [min((a * h(float(t)) + b) % P for t in set(col.tolist())) for a, b in zip(a_list, b_list)]
            self.assertEqual(sig[c].tolist(), ref)
            self.assertEqual(fs.token_hash64(float(col[0])), h(float(col[0])))
        with self.assertRaises(fs.ScreenInputError):
            fs.minhash_signatures([np.array([])], "d", 64)


class ScreenLinks(unittest.TestCase):
    def test_exact_copy_is_content_equal(self):
        a = base_table("copy")
        r = fs.near_dup_screen([epoch("item:a", a), epoch("item:b", a.copy())])
        self.assertEqual(len(r.content_equal), 1)
        link = r.content_equal[0]
        self.assertEqual((link["a"], link["b"], link["kind"]), ("item:a", "item:b", "content"))
        self.assertEqual(link["value"], r.row_multiset_sha256["item:a"])
        self.assertEqual(r.row_multiset_sha256["item:a"], r.row_multiset_sha256["item:b"])
        self.assertIn(link, r.union_links())
        self.assertEqual(len(r.value_strong), 1)                       # a copy is also value-strong

    def test_row_permuted_copy_is_still_content_equal(self):
        a = base_table("rowperm")
        r = fs.near_dup_screen([epoch("item:a", a), epoch("item:b", a[det_perm(400, "rp")])])
        self.assertEqual(len(r.content_equal), 1)

    def test_permuted_rows_and_columns_are_strong_not_content(self):
        a = base_table("perm")
        cols = [3, 0, 4, 1, 2]
        b = a[det_perm(400, "perm:rows")][:, cols]
        r = fs.near_dup_screen([epoch("item:b", b), epoch("item:a", a)])
        self.assertEqual(r.content_equal, [])
        self.assertEqual(r.value_weak, [])
        self.assertEqual(len(r.value_strong), 1)
        ev = r.value_strong[0]["evidence"]
        self.assertEqual((r.value_strong[0]["a"], r.value_strong[0]["b"]), ("item:a", "item:b"))
        self.assertEqual(sorted(map(tuple, ev["pairs"])), sorted((cols[j], j) for j in range(5)))
        self.assertEqual(ev["J"], [1.0] * 5)
        self.assertEqual((ev["n_mapped"], ev["join_cols"], ev["containment"], ev["T"]), (5, 5, 1.0, 5))
        self.assertTrue(ev["min_tuples_ok"])
        self.assertIn(r.value_strong[0], r.union_links())

    def test_unrelated_tables_have_no_link(self):
        r = fs.near_dup_screen([epoch("item:a", base_table("u1")), epoch("item:b", base_table("u2")),
                                epoch("item:c", base_table("u3", 300, 7))])
        self.assertEqual(r.links(), [])
        self.assertEqual(r.stop_columns, {})
        self.assertEqual(r.counts["lsh_matches"], 0)
        for s in r.sensitivity.values():
            self.assertEqual((s["strong_pairs"], s["weak_pairs_total"]), (0, 0))

    def test_shared_column_sets_without_joint_rows_are_weak(self):
        x, y = np.round(det_unit(300, "w:x"), 4), np.round(det_unit(300, "w:y"), 4)
        a = np.column_stack([x, y])
        b = np.column_stack([x, y[det_perm(300, "w:p")]])
        r = fs.near_dup_screen([epoch("item:a", a), epoch("item:b", b)])
        self.assertEqual((r.value_strong, r.content_equal), ([], []))
        self.assertEqual(len(r.value_weak), 1)
        ev = r.value_weak[0]["evidence"]
        self.assertEqual(r.value_weak[0]["kind"], "nds:value_weak")
        self.assertEqual(ev["n_mapped"], 2)
        self.assertTrue(ev["min_tuples_ok"])
        self.assertLess(ev["containment"], 0.5)
        self.assertNotIn(r.value_weak[0], r.union_links())

    def test_fewer_than_min_tuples_is_never_strong(self):
        k = np.arange(30)
        rows = np.column_stack([k / 29.0, ((k * 7) % 30) / 29.0])
        a = np.vstack([rows] * 4)                                 # 120 rows, 30 distinct tuples
        r = fs.near_dup_screen([epoch("item:a", a), epoch("item:b", a.copy())])
        self.assertEqual(r.value_strong, [])
        self.assertEqual(len(r.value_weak), 1)
        ev = r.value_weak[0]["evidence"]
        self.assertFalse(ev["min_tuples_ok"])
        self.assertEqual((ev["n_tuples_a"], ev["containment"]), (30, 1.0))
        self.assertEqual(len(r.content_equal), 1)                 # identical content still unions

    def test_containment_uses_the_smaller_tuple_set(self):
        a = base_table("sub", 400, 3)
        r = fs.near_dup_screen([epoch("item:a", a), epoch("item:b", a[:120])])
        self.assertEqual(len(r.value_strong), 0)                  # column J = 120/400 < 0.8: no column match
        b = np.vstack([a[:380], np.round(det_unit(60, "sub:x"), 4).reshape(20, 3)])
        r = fs.near_dup_screen([epoch("item:a", a), epoch("item:b", b[:390])])
        ev = r.value_strong[0]["evidence"]
        self.assertEqual((ev["n_mapped"], ev["n_tuples_a"], ev["n_tuples_b"], ev["n_shared_tuples"]), (3, 400, 390, 380))
        self.assertEqual(ev["containment"], round(380 / 390, 6))     # divided by the smaller tuple set

    def test_greedy_mapping_is_one_to_one(self):
        x, y = np.round(det_unit(200, "g:x"), 4), np.round(det_unit(200, "g:y"), 4)
        r = fs.near_dup_screen([epoch("item:a", np.column_stack([x, x, y])), epoch("item:b", np.column_stack([x, y]))])
        self.assertEqual(len(r.value_strong), 1)
        self.assertEqual(r.value_strong[0]["evidence"]["pairs"], [[0, 0], [2, 1]])

    def test_jaccard_boundary_is_exact(self):
        col_a = np.arange(18) / 100.0
        col_b = np.concatenate([np.arange(16), [100, 101]]) / 100.0           # |&| 16, |U| 20: J = 0.8
        other = np.round(det_unit(18, "jb"), 4)
        r = fs.near_dup_screen([epoch("item:a", np.column_stack([col_a, other])),
                                epoch("item:b", np.column_stack([col_b, other]))])
        self.assertEqual(len(r.value_weak), 1)                    # 18 tuples < 50: weak
        self.assertEqual(sorted(r.value_weak[0]["evidence"]["J"]), [0.8, 1.0])
        col_c = np.concatenate([np.arange(15), [100, 101, 102]]) / 100.0      # J = 15/21 < 0.8
        r = fs.near_dup_screen([epoch("item:a", np.column_stack([col_a, other])),
                                epoch("item:c", np.column_stack([col_c, other]))])
        self.assertEqual(r.links(), [])

    def test_tables_without_features_never_link_on_content(self):
        r = fs.near_dup_screen([epoch("item:a", np.empty((10, 0))), epoch("item:b", np.empty((10, 0)))])
        self.assertEqual(r.links(), [])
        self.assertIsNone(r.row_multiset_sha256["item:a"])


class StopColumns(unittest.TestCase):
    def tables(self):
        n = 200
        lattice = (np.arange(n) % 20) / 19.0
        s = np.round(det_unit(n, "stop:s"), 4)
        out = [epoch("item:t0", np.column_stack([lattice, s, np.round(det_unit(n, "stop:u0"), 4)])),
               epoch("item:t1", np.column_stack([lattice, s, np.round(det_unit(n, "stop:u1"), 4)]))]
        for i in range(2, 7):
            out.append(epoch(f"item:t{i}", np.column_stack([lattice[det_perm(n, f"stop:p{i}")],
                                                             np.round(det_unit(n, f"stop:u{i}"), 4)])))
        return out

    def test_a_column_matching_more_than_stop_t_identities_is_stop(self):
        r = fs.near_dup_screen(self.tables())
        self.assertEqual(r.stop_columns, {f"item:t{i}": [0] for i in range(7)})
        self.assertEqual({d["n_other_identities"] for d in r.stop_column_detail}, {6})
        self.assertEqual({d["n_distinct"] for d in r.stop_column_detail}, {20})
        # t0/t1 share the lattice and s; without the stop column only 1 column maps: no link at all
        self.assertEqual((r.value_strong, r.value_weak), ([], []))
        self.assertEqual(r.tables["item:t0"]["n_stop_columns"], 1)

    def test_sensitivity_recomputes_at_ten_and_twenty(self):
        r = fs.near_dup_screen(self.tables())
        self.assertEqual(sorted(r.sensitivity), ["10", "20", "5"])
        self.assertTrue(r.sensitivity["5"]["primary"])
        self.assertFalse(r.sensitivity["10"]["primary"] or r.sensitivity["20"]["primary"])
        self.assertEqual((r.sensitivity["5"]["stop_columns"], r.sensitivity["5"]["strong_pairs"]), (7, 0))
        for t in ("10", "20"):
            self.assertEqual((r.sensitivity[t]["stop_columns"], r.sensitivity[t]["strong_pairs"]), (0, 1))
            self.assertEqual(r.sensitivity[t]["max_strong_component"], 2)
        self.assertEqual(r.value_strong, [])                      # sensitivity is report-only

    def test_stop_threshold_is_strictly_more_than_t(self):
        r = fs.near_dup_screen(self.tables(), policy={"STOP_T": 6})
        self.assertEqual(r.stop_columns, {})
        self.assertEqual([(x["a"], x["b"]) for x in r.value_strong], [("item:t0", "item:t1")])
        self.assertEqual(r.value_strong[0]["evidence"]["T"], 6)


class Names(unittest.TestCase):
    def test_normalization_and_generic_names(self):
        self.assertEqual(fs.normalize_name("  Sale Price ($) "), "sale_price")
        self.assertEqual(fs.normalize_name("Unnamed: 0"), "unnamed_0")
        got = fs.name_set(["X1", "V2", "col_3", "Feature10", "f_1", "var7", "Unnamed: 0", "ID", "x", "Year", "###",
                           "Price", "carat", "Cut", "cut"])
        self.assertEqual(got, frozenset({"price", "carat", "cut"}))
        self.assertIsNone(fs.name_set(None))
        self.assertEqual(fs.names_from_meta({"feature_columns": ["a", "b"], "target_column": "t"}), ["a", "b", "t"])
        self.assertIsNone(fs.names_from_meta({"feature_columns": "a"}))
        self.assertIsNone(fs.names_from_meta(None))

    def screen(self, na, nb):
        tabs = [epoch("item:a", base_table("n1", 50, 2)), epoch("item:b", base_table("n2", 50, 2))]
        return fs.near_dup_screen(tabs, names={"item:a": na, "item:b": nb})

    def test_name_link_rules(self):
        r = self.screen(["carat", "cut", "color", "clarity", "depth", "price", "x", "y"],
                        ["Carat", "Cut", "Color", "Clarity", "Depth", "Price", "z"])
        self.assertEqual(len(r.name_links), 1)
        ev = r.name_links[0]["evidence"]
        self.assertEqual((ev["n_shared"], ev["n_union"], ev["J"]), (6, 7, round(6 / 7, 6)))
        self.assertEqual(ev["shared"], ["carat", "clarity", "color", "cut", "depth", "price"])
        self.assertIn(r.name_links[0], r.union_links())
        self.assertEqual(self.screen(["alpha", "beta"], ["alpha", "beta"]).name_links, [])            # 2 shared
        self.assertEqual(self.screen(["alpha", "beta", "gamma", "delta"], ["alpha", "beta", "gamma", "eps"]).name_links,
                         [])                                                                         # J = 0.6
        self.assertEqual(len(self.screen(["a", "b", "c", "d"], ["a", "b", "c", "d", "e"]).name_links), 1)  # J = 0.8
        self.assertEqual(self.screen(["x1", "x2", "x3", "id"], ["x1", "x2", "x3", "id"]).name_links, [])  # generic
        self.assertEqual(self.screen(None, ["a", "b", "c"]).name_links, [])

    def test_beyondarena_tables_bring_parquet_names(self):
        t = fs.make_table("beyondarena:z", "beyondarena", base_table("bz", 50, 2), names=("Alpha", "Beta", "Gamma"))
        r = fs.near_dup_screen([t, epoch("item:a", base_table("n3", 50, 2))], names={"item:a": ["alpha", "beta", "gamma"]})
        self.assertEqual([(x["a"], x["b"], x["kinds"]) for x in r.name_links],
                         [("beyondarena:z", "item:a", ["beyondarena", "epoch"])])

    def test_sealed_and_redacted_names_never_reach_outputs(self):
        secret = ["zebrafish_mass", "orbital_period", "kelvin_reading", "lumen_count"]
        sealed = fs.make_table("sealed_ref:00aa", "sealed_ref", base_table("s1", 60, 2))
        r = fs.near_dup_screen([sealed, epoch("item:a", base_table("s2", 60, 2)), epoch("item:b", base_table("s3", 60, 2))],
                               names={"sealed_ref:00aa": secret, "item:a": secret, "item:b": secret})
        self.assertEqual(len(r.name_links), 3)
        self.assertEqual(r.name_links[0]["evidence"]["shared"], sorted(secret))      # in memory only
        blobs = list(r.files(redact=["item:a"]).values()) + [common.canonical(r.as_dict(redact=["item:a"]))]
        for blob in blobs:
            for name in secret:
                self.assertNotIn(name.encode(), blob)
        links = [json.loads(x) for x in r.files(redact=["item:a"])["links.jsonl"].splitlines()]
        self.assertEqual(links[0]["evidence"]["shared_sha256"], sorted(common.sha256_bytes(n.encode()) for n in secret))
        # without redaction only the epoch-epoch link stays plaintext; sealed links are always hashed
        plain = [json.loads(x) for x in r.files()["links.jsonl"].splitlines()]
        self.assertEqual([("shared" in x["evidence"]) for x in plain], [True, False, False])


class Coverage(unittest.TestCase):
    def test_coverage_classes(self):
        one_inf = np.column_stack([np.round(det_unit(60, "c1"), 4), np.arange(60) % 3])
        two_inf = base_table("c2", 60, 2)
        r = fs.near_dup_screen([epoch("item:full", two_inf), epoch("item:names", one_inf),
                                epoch("item:values", base_table("c3", 60, 2)), epoch("item:exact", one_inf * 0.5)],
                               names={"item:full": ["aa", "bb", "cc"], "item:names": ["aa2", "bb2", "cc2"],
                                      "item:exact": ["aa3", "bb3"]})
        self.assertEqual(r.coverage(), {"item:exact": "exact_only", "item:full": "full", "item:names": "names_only",
                                        "item:values": "values_only"})
        self.assertEqual(r.tables["item:names"]["n_informative"], 1)
        self.assertEqual((r.tables["item:values"]["names_known"], r.tables["item:values"]["n_names"]), (False, None))
        self.assertEqual(r.tables["item:exact"]["n_names"], 2)
        self.assertEqual([fs.coverage_class(v, n) for v, n in ((1, 1), (0, 1), (1, 0), (0, 0))],
                         ["full", "names_only", "values_only", "exact_only"])


class Determinism(unittest.TestCase):
    def tables(self):
        a = base_table("d:a")
        lattice = (np.arange(400) % 20) / 19.0
        tabs = [epoch("item:a", a), epoch("item:b", a[det_perm(400, "d:r")][:, [4, 3, 2, 1, 0]]),
                epoch("item:c", a.copy()), epoch("item:d", base_table("d:d")),
                fs.make_table("sealed_ref:01", "sealed_ref", np.column_stack([a[:, :2], lattice])),
                fs.make_table("beyondarena:e", "beyondarena", np.column_stack([a[:, 3], a[:, 1], lattice]),
                              names=("p", "q", "r", "s"))]
        for i in range(6):
            tabs.append(epoch(f"item:l{i}", np.column_stack([lattice[det_perm(400, f"d:l{i}")],
                                                              np.round(det_unit(400, f"d:u{i}"), 4)])))
        names = {"item:a": ["p", "q", "r", "s"], "item:d": ["p", "q", "r"], "sealed_ref:01": ["p", "q", "r", "s"]}
        return tabs, names

    def test_shuffled_inputs_give_identical_outputs(self):
        tabs, names = self.tables()
        r1 = fs.near_dup_screen(tabs, names=names, ba_pairs=[("beyondarena:e", "item:a")])
        order = det_perm(len(tabs), "d:order")
        shuffled = {tabs[i].identity: tabs[i] for i in order}
        names2 = {k: list(reversed(v)) for k, v in reversed(list(names.items()))}
        r2 = fs.near_dup_screen(shuffled, names=names2, ba_pairs=[("item:a", "beyondarena:e")])
        self.assertEqual(r1.files(), r2.files())
        self.assertEqual(common.canonical(r1.as_dict()), common.canonical(r2.as_dict()))
        self.assertEqual(r1.tables_digest, r2.tables_digest)
        self.assertEqual(set(r1.files()), set(fs.SCREEN_FILES))
        for name, blob in r1.files().items():
            if name.endswith(".json"):
                self.assertEqual(json.loads(blob)["schema"], fs.SCHEMA)
        self.assertTrue(r1.content_equal and r1.value_strong and r1.name_links)

    def test_tables_digest_binds_values(self):
        a = base_table("bind")
        b = a.copy()
        b[0, 0] = 0.5
        d1 = fs.near_dup_screen([epoch("item:a", a)]).tables_digest
        d2 = fs.near_dup_screen([epoch("item:a", b)]).tables_digest
        self.assertNotEqual(d1, d2)

    def test_module_keeps_no_rng_and_imports_no_forbidden_module(self):
        tree = ast.parse(SRC.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
                if node.module == "dltrain":
                    imported |= {"dltrain." + a.name for a in node.names}
        forbidden = {"jope", "gp_worker", "local_gateway", "reuse_cache", "intake_registry", "family_protocol",
                     "random", "secrets", "sqlite3", "boto3", "botocore", "time", "dltrain.s3io"}
        self.assertEqual(imported & forbidden, set())
        self.assertNotIn("np.random", SRC.read_text())


class Loaders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def csv(self, rows, fmt="%.9g") -> bytes:
        return "".join(",".join(fmt % v for v in r) + "\n" for r in rows).encode()

    def test_data_object_drops_the_target_and_checks_its_pin(self):
        rows = np.column_stack([np.float32(det_unit(30, "lo:a")), det_unit(30, "lo:b"), det_unit(30, "lo:t")])
        data = gzip.compress(self.csv(rows), mtime=0)
        p = self.d / "obj.csv.gz"
        p.write_bytes(data)
        t = fs.load_data_object("item:x", p, expected_sha256=common.sha256_bytes(data), expected_bytes=len(data))
        self.assertEqual((t.kind, t.n_rows, t.n_cols), ("epoch", 30, 2))
        self.assertEqual(t.values[:, 1].tolist(), [float("%.9g" % v) for v in rows[:, 1]])
        self.assertEqual(t.source["data"]["sha256"], common.sha256_bytes(data))
        with self.assertRaises(fs.ScreenInputError):
            fs.load_data_object("item:x", p, expected_sha256="0" * 64)
        with self.assertRaises(fs.ScreenInputError):
            fs.load_data_object("item:x", p, expected_bytes=1)
        with self.assertRaises(ValueError):
            t.values[0, 0] = 1.0                                  # tables are read-only

    def test_malformed_csv_stops_without_echoing_cells(self):
        for bad in (b"secretname,price\n1,2\n", b"1,2,3\n4,5\n", b"", b"1,secretcell\n"):
            p = self.d / "bad.csv"
            p.write_bytes(bad)
            with self.assertRaises(fs.ScreenInputError) as cm:
                fs.load_data_object("item:bad", p)
            self.assertNotIn("secret", str(cm.exception))
        p.write_bytes(b"1\n2\n")
        with self.assertRaises(fs.ScreenInputError):
            fs.load_data_object("item:one", p)

    def test_sealed_ref_concatenates_train_then_test(self):
        tr = np.column_stack([det_unit(20, "sr:a"), det_unit(20, "sr:t")])
        te = np.column_stack([det_unit(10, "sr:b"), det_unit(10, "sr:u")])
        (self.d / "train").write_bytes(self.csv(tr))
        (self.d / "test").write_bytes(gzip.compress(self.csv(te), mtime=0))
        t = fs.load_sealed_ref("sealed_ref:h", self.d / "train", self.d / "test")
        self.assertEqual((t.kind, t.n_rows, t.n_cols), ("sealed_ref", 30, 1))
        self.assertEqual(t.values[:20, 0].tolist(), [float("%.9g" % v) for v in tr[:, 0]])
        self.assertEqual(t.values[20:, 0].tolist(), [float("%.9g" % v) for v in te[:, 0]])
        (self.d / "test3").write_bytes(self.csv(np.column_stack([te, te])))
        with self.assertRaises(fs.ScreenInputError):
            fs.load_sealed_ref("sealed_ref:h", self.d / "train", self.d / "test3")

    def test_beyondarena_parquet_numeric_features_scaled_over_the_full_table(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        n = 8
        tbl = pa.table({
            "a": pa.array([3, 5, 7, 9, 11, 13, 15, 17], pa.int64()),
            "b": pa.array([1.0, None, 3.0, float("inf"), 5.0, 6.0, 7.0, 9.0], pa.float64()),
            "cat": pa.array(list("abababab")).dictionary_encode(),
            "s": pa.array(list("xyzxyzxy")),
            "flag": pa.array([True, False] * 4),
            "k": pa.array([2.0] * n),
            "Target": pa.array([float(i) for i in range(n)]),
        })
        p = self.d / "dataset.parquet"
        pq.write_table(tbl, p)
        t = fs.load_beyondarena("beyondarena:toy", p, "Target", expected_sha256=common.sha256_file(p))
        self.assertEqual(t.column_names, ("a", "b"))
        self.assertEqual(t.names, ("a", "b", "cat", "s", "flag", "k", "Target"))
        self.assertEqual(t.values[:, 0].tolist(), [(v - 3) / 14 for v in range(3, 18, 2)])
        b = t.values[:, 1]
        self.assertTrue(np.isnan(b[1]) and np.isnan(b[3]))
        self.assertEqual(b[[0, 2, 4, 5, 6, 7]].tolist(), [0.0, 0.25, 0.5, 0.625, 0.75, 1.0])
        self.assertEqual(t.raw_values[:, 0].tolist(), list(range(3, 18, 2)))
        self.assertEqual(t.source["parquet"]["n_numeric_features"], 2)
        with self.assertRaises(fs.ScreenInputError):
            fs.load_beyondarena("beyondarena:toy", p, "missing")


class BeyondArenaCheck(unittest.TestCase):
    """A BeyondArena table is the full source; an epoch data object is a sample of it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.d = Path(self.tmp.name)
        n = 3000
        self.raw = np.column_stack([np.round(det_unit(n, "ba:0") * 1000, 2), np.round(det_unit(n, "ba:1") * 50 + 10, 3),
                                    np.round(det_unit(n, "ba:2"), 4)])
        target = det_unit(n, "ba:t")
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.table({"c0": self.raw[:, 0], "c1": self.raw[:, 1], "c2": self.raw[:, 2], "t": target}),
                       self.d / "ba.parquet")
        self.ba = fs.load_beyondarena("beyondarena:src", self.d / "ba.parquet", "t")
        self.sample = det_perm(n, "ba:sample")[:1000]

    def tearDown(self):
        self.tmp.cleanup()

    def epoch_from(self, lo, hi, single=False) -> fs.Table:
        x = (self.raw[self.sample] - lo) / (hi - lo)
        rows = np.column_stack([np.float32(x) if single else x, det_unit(1000, "ba:y")])
        p = self.d / f"e{lo[0]}.csv.gz"
        p.write_bytes(gzip.compress("".join(",".join("%.9g" % v for v in r) + "\n" for r in rows).encode(), mtime=0))
        return fs.load_data_object("item:e", p)

    def test_sample_of_the_source_is_confirmed_in_the_scaled_domain(self):
        lo, hi = self.raw.min(axis=0), self.raw.max(axis=0)
        e = self.epoch_from(lo, hi)
        r = fs.near_dup_screen([self.ba, e], ba_pairs=[("item:e", "beyondarena:src")])
        self.assertEqual(r.links(), [])                           # column J ~ 1/3: the spec screen cannot see it
        chk = r.beyondarena_check("item:e", "beyondarena:src")
        self.assertEqual((chk["a"], chk["b"], chk["requested"], chk["class"]), ("beyondarena:src", "item:e", True,
                                                                                 "strong"))
        self.assertEqual(chk["scaled"]["n_mapped"], 3)
        self.assertGreaterEqual(chk["scaled"]["containment"], 0.99)
        self.assertEqual(chk["scaled"]["n_tuples_b"], 1000)
        self.assertTrue(all(c >= 0.99 for c in chk["scaled"]["col_containment"]))
        self.assertTrue(all(j < 0.8 for j in chk["scaled"]["J"]))
        self.assertEqual(chk["raw"], {"status": "no_meta"})
        self.assertEqual(r.counts["beyondarena_checks_strong"], 1)
        self.assertNotIn("beyondarena_checks", [x["kind"] for x in r.union_links()])

    def test_raw_domain_recovers_a_differently_scaled_copy(self):
        lo, hi = self.raw.min(axis=0) - 5.0, self.raw.max(axis=0) + 7.0
        e = self.epoch_from(lo, hi, single=True)                  # the frozen cohort stores float32
        meta = {"feature_columns": ["c0", "c1", "c2"], "target_column": "t", "output_headers": ["c0", "gone", "c1", "c2", "t"],
                "feature_scaling": {"method": "minmax", "min": [lo[0], -1.0, lo[1], lo[2]], "max": [hi[0], 9.0, hi[1], hi[2]]}}
        r = fs.near_dup_screen([self.ba, e], metas={"item:e": meta}, ba_pairs=[("beyondarena:src", "item:e")])
        chk = r.beyondarena_check("beyondarena:src", "item:e")
        self.assertEqual(chk["scaled"]["class"], "none")
        self.assertEqual((chk["raw"]["status"], chk["raw"]["class"], chk["raw"]["n_mapped"]), ("applied", "strong", 3))
        self.assertGreaterEqual(chk["raw"]["containment"], 0.95)        # float32 storage, 5-decimal grid
        self.assertEqual(sorted(map(tuple, chk["raw"]["pairs"])), [(0, 0), (1, 1), (2, 2)])
        self.assertEqual(chk["class"], "strong")
        bad = dict(meta, output_headers=["c0", "c1", "c2", "t"])
        r = fs.near_dup_screen([self.ba, e], metas={"item:e": bad}, ba_pairs=[("beyondarena:src", "item:e")])
        self.assertEqual(r.beyondarena_check("beyondarena:src", "item:e")["raw"], {"status": "lengths_do_not_validate"})

    def test_unrelated_pair_is_none_and_linked_pairs_are_checked_automatically(self):
        r = fs.near_dup_screen([self.ba, epoch("item:u", base_table("ba:u", 500, 3))],
                               ba_pairs=[("beyondarena:src", "item:u")])
        self.assertEqual(r.beyondarena_check("beyondarena:src", "item:u")["class"], "none")
        copy = fs.make_table("item:copy", "epoch", self.ba.values)
        r = fs.near_dup_screen([self.ba, copy])
        self.assertEqual(len(r.value_strong), 1)
        chk = r.beyondarena_check("beyondarena:src", "item:copy")
        self.assertEqual((chk["requested"], chk["class"]), (False, "strong"))


class Validation(unittest.TestCase):
    def test_policy_defaults_and_overrides(self):
        p = fs.screen_policy()
        self.assertEqual((p["ROUND"], p["INFORMATIVE_MIN_DISTINCT"], p["MINHASH_K"], p["LSH"], p["COL_J"], p["MIN_COLS"],
                          p["STOP_T"], p["SENS_T"], p["CONTAIN"], p["MIN_TUPLES"], p["MAX_JOIN_COLS"], p["NAME_J"],
                          p["NAME_MIN_SHARED"], p["NDS_DOMAIN"]),
                         (5, 12, 64, [16, 4], 0.8, 2, 5, [10, 20], 0.5, 50, 8, 0.8, 3, "jope.dl-family-nds/v1"))
        self.assertEqual(set(p["GENERIC_NAMES"]), {"x", "y", "id", "index", "value", "date", "time", "year", "month",
                                                   "day", "target", "label", "unnamed_0"})
        q = fs.screen_policy({"STOP_T": 10, "LSH": {"bands": 32, "rows": 2}, "NDS_DOMAIN": b"jope.dl-family-nds/v1",
                              "FID_DOMAIN": "ignored", "SENS_T": [20, 10, 20]})
        self.assertEqual((q["STOP_T"], q["LSH"], q["NDS_DOMAIN"], q["SENS_T"]), (10, [32, 2], "jope.dl-family-nds/v1",
                                                                                   [10, 20]))
        self.assertNotIn("FID_DOMAIN", q)
        for bad in ({"LSH": [16, 3]}, {"COL_J": 1.5}, {"COL_J": True}, {"ROUND": -1}, {"GENERIC_NAME_RE": "("},
                    {"SENS_T": "x"}, {"NDS_DOMAIN": ""}):
            with self.assertRaises(fs.ScreenInputError):
                fs.screen_policy(bad)
        json.dumps(p)

    def test_bad_inputs_stop(self):
        t = epoch("item:a", base_table("v", 20, 2))
        with self.assertRaises(fs.ScreenInputError):
            fs.near_dup_screen([t, epoch("item:a", base_table("v2", 20, 2))])
        with self.assertRaises(fs.ScreenInputError):
            fs.near_dup_screen([t], names={"item:zz": ["a"]})
        with self.assertRaises(fs.ScreenInputError):
            fs.near_dup_screen({"item:b": t})
        with self.assertRaises(fs.ScreenInputError):
            fs.near_dup_screen([t], ba_pairs=[("item:a", "beyondarena:none")])
        with self.assertRaises(fs.ScreenInputError):
            fs.make_table("item:a", "mystery", [[1.0]])
        with self.assertRaises(fs.ScreenInputError):
            fs.make_table("item:a", "epoch", [1.0, 2.0])
        self.assertTrue(np.isnan(fs.make_table("item:a", "epoch", [[np.inf, 1.0]]).values[0, 0]))

    def test_candidate_guard_stops_instead_of_capping(self):
        a = base_table("guard", 100, 2)
        with self.assertRaises(fs.ScreenLimitError):
            fs.near_dup_screen([epoch(f"item:{i}", a) for i in range(3)], policy={"MAX_CANDIDATE_PAIRS": 1})


if __name__ == "__main__":
    unittest.main()
