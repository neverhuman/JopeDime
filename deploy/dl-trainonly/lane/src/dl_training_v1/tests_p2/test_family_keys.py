"""Offline proofs for dltrain.family_keys (PLAN §5 P2 keys; spec union_keys / review_triggers / FORBIDDEN).

Synthetic inputs only, except PinnedReadinessCounts, which re-derives the frozen cohort from the
sha-pinned readiness file (skipped when absent) and asserts counts only. Nothing here prints values.
"""
import ast
import copy
import hashlib
import json
import random
import unittest
from pathlib import Path

from dltrain import family_keys as fk
from dltrain.common import canonical, sha256_file

SRC = Path(__file__).resolve().parents[1]
MODULE = SRC / "dltrain" / "family_keys.py"
FAMILY_PROTOCOL = Path("/home/ubuntu/JopeDime/.agent/experiments/lambda-training/local_lambda/full_regression/"
                       "research_2k/per_fold_hwm/worst50_v1/next100_v1/family_protocol.py")
FAMILY_PROTOCOL_SHA256 = "20381727b39035b73dde59c2058ec44005f27fa1e72d7c0ea90694cf65738026"
READINESS = Path("/home/ubuntu/scratch/jopedime-next100-gp-audit-v1/readiness-v1/family-readiness.json")
READINESS_SHA256 = "0b27b80e4bbec8526e8432f47d509b8680bd7f3189c4732c4a67176e7a86d233"

H16 = "0123456789abcdef"
C64 = "c" * 64
M64 = "d" * 64
E64 = "e" * 64
F40 = "f" * 40
NULLS = (None, "", "   ", "unknown", "Unknown", " none ", "NONE", "null", "NULL", "n/a", "N/A")


def union(attrs, kind=None):
    u, _ = fk.derive(attrs)
    return u if kind is None else [v for k, v in u if k == kind]


def triggers(attrs, kind=None):
    _, t = fk.derive(attrs)
    return t if kind is None else [v for k, v in t if k == kind]


def url(u):
    return {"provenance": {"url": u}}


class CopiedRuleTests(unittest.TestCase):
    def test_known_null_tokens(self):
        for v in NULLS + (1, 0.0, [], {}, b"x"):
            self.assertFalse(fk.known(v), repr(v))
        for v in ("x", "na", "0", "unknownish", " n/a2 "):
            self.assertTrue(fk.known(v), repr(v))

    def test_source_url_rules(self):
        cases = {
            "http://Example.COM/A/b.csv?X=1&y=2#frag": "https://example.com/A/b.csv?X=1&y=2",
            "https://example.com:443/x": "https://example.com/x",
            "http://example.com:80/x": "https://example.com/x",
            "https://example.com:8443/x": "https://example.com:8443/x",
            "https://example.com": "https://example.com/",
            "https://[2001:db8::1]/x": "https://[2001:db8::1]/x",
            "ftp://example.com/x": None,
            "s3://bucket/key": None,
            "https://user:pw@example.com/x": None,
            "https://example.com:99999/x": None,
            "not a url": None,
            "unknown": None,
            None: None,
        }
        for raw, want in cases.items():
            self.assertEqual(fk.source_url(raw), want, repr(raw))

    def test_known_and_source_url_are_verbatim_copies(self):
        if not FAMILY_PROTOCOL.exists() or sha256_file(FAMILY_PROTOCOL) != FAMILY_PROTOCOL_SHA256:
            self.skipTest("pinned family_protocol.py is not available")

        def functions(path):
            tree = ast.parse(Path(path).read_text())
            return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}

        theirs, ours = functions(FAMILY_PROTOCOL), functions(MODULE)
        for name in ("known", "source_url"):
            self.assertEqual(ours[name], theirs[name], name)


class SocrataTests(unittest.TestCase):
    KEY = "socrata:data.cityofnewyork.us:5uug-f49n"
    VARIANTS = (
        "https://data.cityofnewyork.us/api/views/5uug-f49n/rows.csv?accessType=DOWNLOAD",
        "http://data.cityofnewyork.us/api/views/5UUG-F49N/rows.csv",
        "https://DATA.CITYOFNEWYORK.US:443/api/v3/views/5uug-f49n/export.csv?accessType=DOWNLOAD",
        "https://data.cityofnewyork.us/resource/5uug-f49n.csv",
        "https://data.cityofnewyork.us/resource/5uug-f49n.json?$limit=50000#top",
        "https://www.data.cityofnewyork.us/api/views/5uug-f49n",
        "https://data.cityofnewyork.us/api/views/5uug-f49n/",
    )

    def test_url_variants_give_one_key(self):
        for variant in self.VARIANTS:
            for platform in ("socrata", "nyc open data / socrata", "NYC OpenData", "unknown", None):
                attrs = {"provenance": {"platform": platform, "url": variant}}
                self.assertEqual(union(attrs, "socrata"), [self.KEY], variant)
                self.assertEqual(triggers(attrs, "socrata_domain"), ["socrata:data.cityofnewyork.us"])
                self.assertEqual(triggers(attrs, "socrata_4x4_cross_host"), ["socrata:*:5uug-f49n"])
                self.assertIn("cityofnewyork.us", triggers(attrs, "publisher"))

    def test_hunt_source_key_maps_to_the_same_key(self):
        gate = {"source_keys": ["socrata:Data.CityOfNewYork.us:5UUG-F49N", "unknown", None]}
        attrs = {"meta": {"quality_checks": {"gates": {"deduplication": gate}}}}
        self.assertEqual(union(attrs, "socrata"), [self.KEY])
        self.assertEqual(triggers(attrs, "socrata_domain"), ["socrata:data.cityofnewyork.us"])

    def test_detected_by_url_never_by_platform_text(self):
        attrs = {"provenance": {"platform": "socrata", "dataset_id": "5uug-f49n", "url": "https://example.org/x.csv"}}
        self.assertEqual(union(attrs, "socrata"), [])
        self.assertEqual(union(attrs, "platform_dataset_id"), ["socrata:5uug-f49n"])
        for not_socrata in (
            "https://data.cityofnewyork.us/Transportation/Some-Name/5uug-f49n",   # landing page
            "https://data.cityofnewyork.us/resource/5uug-f49nx.csv",
            "https://data.cityofnewyork.us/views/5uug-f49n",
            "https://github.com/resource/5uug-f49n.csv",                          # hosting host
            "https://zenodo.org/api/views/5uug-f49n/rows.csv",
        ):
            self.assertEqual(union(url(not_socrata), "socrata"), [], not_socrata)

    def test_other_tables_and_hosts_stay_apart(self):
        self.assertEqual(union(url("https://data.cityofnewyork.us/resource/4aki-r3np.csv"), "socrata"),
                         ["socrata:data.cityofnewyork.us:4aki-r3np"])
        federated = url("https://data.ny.gov/api/views/5uug-f49n/rows.csv")
        self.assertEqual(union(federated, "socrata"), ["socrata:data.ny.gov:5uug-f49n"])
        self.assertEqual(triggers(federated, "socrata_4x4_cross_host"), ["socrata:*:5uug-f49n"])


class RdatasetsTests(unittest.TestCase):
    ITEM = "rdatasets:ecdat/wages"

    def test_case_and_csv_doc_give_one_item_key(self):
        sources = (
            url("https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Wages.csv"),
            url("http://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Wages.html"),
            url("https://vincentarelbundock.github.io/Rdatasets/csv/ecdat/WAGES.CSV"),
            url("https://vincentarelbundock.github.io/rdatasets/doc/Ecdat/Wages"),
            url("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/Ecdat/Wages.csv"),
            url("https://github.com/vincentarelbundock/Rdatasets/blob/main/doc/Ecdat/Wages.html"),
            {"provenance": {"platform": " RDatasets ", "dataset_id": " Ecdat/Wages "}},
        )
        for attrs in sources:
            self.assertEqual(union(attrs, "rdatasets_item"), [self.ITEM], attrs)
            self.assertEqual(triggers(attrs, "rdatasets_pkg"), ["rdatasets:ecdat"])
            self.assertEqual(union(attrs, "github_repo"), [])
            self.assertEqual(triggers(attrs, "github_owner"), [])
            self.assertEqual(triggers(attrs, "publisher"), [])

    def test_item_level_union_package_is_only_a_trigger(self):
        a = url("https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Wages.csv")
        b = url("https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Computers.csv")
        self.assertFalse(set(union(a)) & set(union(b)))
        self.assertEqual(set(fk.derive(a)[1]) & set(fk.derive(b)[1]), {("rdatasets_pkg", "rdatasets:ecdat")})
        self.assertEqual(union(url("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/ability.cov.csv"),
                               "rdatasets_item"), ["rdatasets:datasets/ability.cov"])

    def test_rdatasets_item_key_for_sealed_joins(self):
        for text in ("rdatasets:Ecdat/Wages", "RDATASETS:ecdat/WAGES",
                     "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Wages.html",
                     "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Wages.csv"):
            self.assertEqual(fk.rdatasets_item_key(text), self.ITEM, text)
        for text in ("/home/someone/data/Wages.csv", "Ecdat/Wages", "rdatasets:Ecdat", "rdatasets:unknown/x",
                     "https://example.org/Rdatasets/csv/Ecdat/Wages.csv", None, "n/a"):
            self.assertIsNone(fk.rdatasets_item_key(text), text)


class AggregatorTests(unittest.TestCase):
    TT = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"

    def test_tidytuesday_keyed_by_date_never_by_repo(self):
        same_day = (self.TT + "master/data/2018/2018-10-30/r_downloads_year.csv",
                    self.TT + "main/data/2018/2018-10-30/r_downloads.csv",
                    "https://github.com/rfordatascience/tidytuesday/tree/master/data/2018/2018-10-30")
        for u in same_day:
            self.assertEqual(union(url(u), "tidytuesday_date"), ["tidytuesday:2018-10-30"], u)
            self.assertEqual(triggers(url(u), "tidytuesday_year"), ["tidytuesday:2018"])
            self.assertEqual(triggers(url(u), "github_owner"), ["github:rfordatascience"])
            self.assertEqual(union(url(u), "github_repo"), [])
            self.assertEqual(union(url(u), "github_file"), [])
        other = url("https://github.com/rfordatascience/tidytuesday/blob/master/data/2018/2018-11-13/malaria.csv")
        self.assertEqual(union(other, "tidytuesday_date"), ["tidytuesday:2018-11-13"])
        self.assertEqual(triggers(other, "tidytuesday_year"), ["tidytuesday:2018"])
        refs = url(self.TT + "refs/heads/main/data/2021/2021-01-05/transit_cost.csv")
        self.assertEqual(union(refs, "tidytuesday_date"), ["tidytuesday:2021-01-05"])
        bad_date = url(self.TT + "master/data/2018/2018-02-30/x.csv")
        self.assertEqual(union(bad_date, "tidytuesday_date"), [])
        self.assertEqual(union(bad_date, "github_file"), ["github:rfordatascience/tidytuesday/data/2018/2018-02-30/x.csv"])
        self.assertEqual(union(url("https://github.com/rfordatascience/tidytuesday"), "github_repo"), [])

    def test_pmlb_keyed_by_name(self):
        raw = url("https://raw.githubusercontent.com/EpistasisLab/pmlb/master/datasets/197_cpu_act/197_cpu_act.tsv.gz")
        tree = url("https://github.com/EpistasisLab/pmlb/tree/master/datasets/197_cpu_act")
        plat = {"provenance": {"platform": "pmlb", "dataset_id": "197_cpu_act"}}
        for attrs in (raw, tree, plat):
            self.assertEqual(union(attrs, "pmlb_name"), ["pmlb:197_cpu_act"])
            self.assertEqual(triggers(attrs, "pmlb_stem"), ["openml:197", "pmlb_stem:cpu_act"])
            self.assertEqual(union(attrs, "openml_id"), [])      # inferred ids are never union keys
            self.assertEqual(union(attrs, "github_repo"), [])
        self.assertEqual(triggers(plat, "publisher"), ["pmlb"])
        self.assertEqual(triggers(raw, "publisher"), [])
        self.assertEqual(triggers(raw, "github_owner"), ["github:epistasislab"])
        other = {"provenance": {"platform": "pmlb", "dataset_id": "573_cpu_act"}}
        self.assertNotEqual(union(other, "pmlb_name"), union(plat, "pmlb_name"))
        self.assertIn("pmlb_stem:cpu_act", triggers(other, "pmlb_stem"))
        churn = url("https://github.com/EpistasisLab/pmlb/tree/master/datasets/churn")
        self.assertEqual(union(churn, "pmlb_name"), ["pmlb:churn"])
        self.assertFalse(set(union(churn)) & set(union(raw)))

    def test_github_file_aggregators_are_file_level(self):
        a = url("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv")
        b = url("https://raw.githubusercontent.com/mwaskom/seaborn-data/main/diamonds.csv")
        c = url("https://github.com/mwaskom/seaborn-data/blob/master/diamonds.csv?raw=true")
        d = url("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
        for attrs in (a, b, c):
            self.assertEqual(union(attrs, "github_file"), ["github:mwaskom/seaborn-data/diamonds.csv"])
            self.assertEqual(union(attrs, "github_repo"), [])
            self.assertEqual(triggers(attrs, "github_owner"), ["github:mwaskom"])
        self.assertEqual(union(d, "github_file"), ["github:mwaskom/seaborn-data/tips.csv"])
        e = url("https://github.com/fivethirtyeight/data/blob/master/airline-safety/airline-safety.csv")
        f = url("https://raw.githubusercontent.com/fivethirtyeight/data/master/airline-safety/airline-safety.csv")
        self.assertEqual(union(e, "github_file"), union(f, "github_file"))
        self.assertEqual(union(e, "github_file"), ["github:fivethirtyeight/data/airline-safety/airline-safety.csv"])
        repo_page = url("https://github.com/fivethirtyeight/data")
        self.assertEqual([k for k, _ in union(repo_page)], ["normalized_url"])
        self.assertEqual(triggers(repo_page, "github_owner"), ["github:fivethirtyeight"])

    def test_other_repos_are_repo_level(self):
        files = (url("https://raw.githubusercontent.com/SomeOwner/Exp-Repo/main/data/exp1.csv"),
                 url("https://raw.githubusercontent.com/someowner/exp-repo/0a1b2c/data/exp2.csv"),
                 url("https://github.com/SomeOwner/Exp-Repo.git"),
                 url("https://github.com/someowner/exp-repo/blob/main/data/exp3.csv"),
                 url("https://codeload.github.com/someowner/exp-repo/zip/refs/heads/main"),
                 url("https://media.githubusercontent.com/media/someowner/exp-repo/main/big.csv"))
        for attrs in files:
            self.assertEqual(union(attrs, "github_repo"), ["github:someowner/exp-repo"], attrs)
            self.assertEqual(triggers(attrs, "github_owner"), ["github:someowner"])
            self.assertEqual(triggers(attrs, "publisher"), [])
        self.assertEqual(union(url("https://github.com/topics/data"), "github_repo"), [])
        self.assertEqual(triggers(url("https://github.com/someowner"), "github_owner"), ["github:someowner"])

    def test_gist_and_pages(self):
        for u in ("https://gist.githubusercontent.com/Owner/ABC123/raw/0f0f/data.csv",
                  "https://gist.github.com/owner/abc123"):
            self.assertEqual(union(url(u), "gist"), ["gist:owner/abc123"])
            self.assertEqual(triggers(url(u), "github_owner"), ["github:owner"])
        cars = url("https://corgis-edu.github.io/corgis/datasets/csv/cars/cars.csv")
        food = url("https://corgis-edu.github.io/corgis/datasets/csv/food/food.csv")
        self.assertEqual(union(cars, "ghpages_file"), ["corgis-edu.github.io/corgis/datasets/csv/cars/cars.csv"])
        self.assertFalse(set(union(cars)) & set(union(food)))
        self.assertEqual(triggers(cars, "github_owner"), ["github:corgis-edu"])
        self.assertEqual(triggers(cars, "publisher"), [])
        self.assertEqual(union(cars, "github_repo"), [])
        project = url("https://someone.github.io/proj/data/a.csv")
        self.assertEqual(union(project, "github_repo"), ["github:someone/proj"])
        self.assertEqual(union(project, "ghpages_file"), [])


class OtherSourceAwareTests(unittest.TestCase):
    def test_zenodo_record(self):
        for u in ("https://zenodo.org/records/1234567/files/x.csv?download=1", "https://zenodo.org/record/1234567",
                  "https://zenodo.org/api/records/01234567", "https://doi.org/10.5281/zenodo.1234567"):
            self.assertEqual(union(url(u), "zenodo_record"), ["zenodo:1234567"], u)
            self.assertEqual(triggers(url(u), "publisher"), [])
        self.assertEqual(union(url("https://zenodo.org/communities/x"), "zenodo_record"), [])

    def test_uci(self):
        for u in ("https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength",
                  "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"):
            self.assertEqual(union(url(u), "uci_id"), ["uci:165"])
            self.assertEqual(triggers(url(u), "publisher"), ["uci.edu"])
        legacy = url("https://archive.ics.uci.edu/ml/machine-learning-databases/Parkinsons/parkinsons.data")
        self.assertEqual(union(legacy, "uci_legacy_folder"), ["uci_folder:parkinsons"])
        self.assertEqual(union(legacy, "uci_id"), [])

    def test_openml_direct_urls_only(self):
        for u in ("https://www.openml.org/d/43342", "https://www.openml.org/search?type=data&id=43342",
                  "https://api.openml.org/api/v1/json/data/43342"):
            self.assertEqual(union(url(u), "openml_id"), ["openml:43342"], u)
            self.assertIn("openml:43342", triggers(url(u), "pmlb_stem"))
        for u in ("https://www.openml.org/data/download/22044777/x.arff", "https://www.openml.org/search?type=task&id=7"):
            self.assertEqual(union(url(u), "openml_id"), [], u)

    def test_huggingface_and_kaggle(self):
        hf = url("https://huggingface.co/datasets/Owner/Repo/resolve/main/data.csv?download=true")
        self.assertEqual(union(hf, "hf_repo"), ["hf:owner/repo"])
        self.assertEqual(triggers(hf, "hf_owner"), ["hf:owner"])
        self.assertEqual(triggers(hf, "publisher"), [])
        self.assertEqual(union(url("https://huggingface.co/datasets/squad/resolve/main/x.json"), "hf_repo"), ["hf:squad"])
        self.assertEqual(union(url("https://huggingface.co/Owner/model"), "hf_repo"), [])
        kd = url("https://www.kaggle.com/datasets/ElvinRustam/Wine-Dataset/data")
        self.assertEqual(union(kd, "kaggle"), ["kaggle:elvinrustam/wine-dataset"])
        self.assertEqual(triggers(kd, "publisher"), [])
        for u in ("https://www.kaggle.com/c/california-house-prices",
                  "https://www.kaggle.com/competitions/California-House-Prices/data?select=train.csv"):
            self.assertEqual(union(url(u), "kaggle_competition"), ["kaggle_competition:california-house-prices"])

    def test_doi(self):
        for raw in ("10.24432/C51S4N", "doi:10.24432/C51S4N", "https://dx.doi.org/10.24432/c51s4n",
                    "https://doi.org/10.24432/C51S4N"):
            self.assertEqual(union(url(raw), "doi"), ["doi:10.24432/c51s4n"], raw)
            self.assertEqual(triggers(url(raw), "publisher"), [])
        self.assertEqual(union(url("10.1/x"), "doi"), [])


class PublisherTests(unittest.TestCase):
    def test_registrable_domain(self):
        cases = {
            "data.cityofnewyork.us": "cityofnewyork.us", "www150.statcan.gc.ca": "gc.ca",
            "data.gov.au": "data.gov.au", "files.science-data.qld.gov.au": "qld.gov.au",
            "www.ons.gov.uk": "ons.gov.uk", "api.worldbank.org": "worldbank.org", "datos.gob.cl": "datos.gob.cl",
            "www.emi.ea.govt.nz": "ea.govt.nz", "example.com:8443": "example.com", "Example.COM.": "example.com",
            "192.168.1.1": "192.168.1.1", "2001:db8::1": "[2001:db8::1]", "localhost": "localhost",
            None: None, "unknown": None,
        }
        for host, want in cases.items():
            self.assertEqual(fk.registrable_domain(host), want, host)

    def test_hosting_hosts(self):
        for host in sorted(fk.HOSTING) + ["www.kaggle.com", "corgis-edu.github.io", "media.githubusercontent.com",
                                          "files.osf.io", "cdn-lfs.huggingface.co", "gist.github.com"]:
            self.assertTrue(fk.is_hosting(host), host)
        for host in ("data.cityofnewyork.us", "www.openml.org", "doi.org", "example.com", None):
            self.assertFalse(fk.is_hosting(host), host)

    def test_publisher_trigger(self):
        self.assertEqual(triggers(url("https://data.ny.gov/x.csv"), "publisher"), ["ny.gov"])
        self.assertEqual(triggers(url("https://raw.githubusercontent.com/o/r/main/x.csv"), "publisher"), [])
        self.assertEqual(triggers(url("https://www.kaggle.com/datasets/o/s"), "publisher"), [])
        for platform in ("pmlb", "WorldBank", " seaborn ", "sklearn"):
            self.assertEqual(triggers({"provenance": {"platform": platform}}, "publisher"), [fk.norm(platform)])
        wb = {"provenance": {"platform": "worldbank", "url": "https://api.worldbank.org/v2/country/all/indicator/X"}}
        self.assertEqual(triggers(wb, "publisher"), ["worldbank", "worldbank.org"])
        self.assertEqual(triggers({"provenance": {"platform": "socrata"}}, "publisher"), [])
        self.assertEqual(union(url("https://data.ny.gov/x.csv"), "publisher"), [])   # never a union key


class ExactKeyTests(unittest.TestCase):
    def test_exact_keys(self):
        attrs = {
            "dataset_hash": f" {H16} ", "canonical": C64, "meta_sha256": M64,
            "provenance": {"platform": "  NYC  Open Data / Socrata ", "dataset_id": " 5uug-f49n ",
                           "parent_dataset_id": "p1", "source_table_id": "t1", "source_family": "f1",
                           "dedupe_fingerprint": F40, "url": "https://example.org/a.csv#x"},
        }
        u = set(union(attrs))
        platform = "nyc open data / socrata"
        for want in (("dataset_hash", H16), ("canonical_sha256", C64), ("lineage_b_metadata_sha256", M64),
                     ("platform_dataset_id", f"{platform}:5uug-f49n"), ("platform_parent_dataset_id", f"{platform}:p1"),
                     ("platform_source_table_id", f"{platform}:t1"), ("platform_source_family", f"{platform}:f1"),
                     ("dedupe_fingerprint", f"{platform}:{F40}"), ("normalized_url", "https://example.org/a.csv")):
            self.assertIn(want, u)
        self.assertTrue({k for k, _ in u} <= fk.EXACT_KINDS)

    def test_canonical_and_meta_aliases(self):
        self.assertEqual(union({"canonical_sha256": C64}, "canonical_sha256"), [C64])
        self.assertEqual(union({"canonical": C64, "canonical_sha256": C64}, "canonical_sha256"), [C64])
        with self.assertRaises(fk.MalformedKey):
            fk.derive({"canonical": C64, "canonical_sha256": E64})
        self.assertEqual(union({"canonical": C64.upper()}, "canonical_sha256"), [])   # copied rule: 64 lowercase hex
        self.assertEqual(union({"files": {"meta": {"sha256": M64}}}, "lineage_b_metadata_sha256"), [M64])
        with self.assertRaises(fk.MalformedKey):
            fk.derive({"meta_sha256": M64, "files": {"meta": {"sha256": E64}}})

    def test_dedupe_fingerprint_rules(self):
        self.assertEqual(union({"provenance": {"dedupe_fingerprint": E64}}, "dedupe_fingerprint"), [f"none:{E64}"])
        self.assertEqual(union({"provenance": {"platform": "n/a", "dedupe_fingerprint": F40}}, "dedupe_fingerprint"),
                         [f"none:{F40}"])
        for bad in (F40.upper(), F40[:-1], F40 + "f", " " + F40, "g" * 40):
            self.assertEqual(union({"provenance": {"platform": "x", "dedupe_fingerprint": bad}}, "dedupe_fingerprint"),
                             [], bad)

    def test_context_keys(self):
        ctx = {"input_raw_sha256": C64, "evaluator_sha256": E64, "candidate_timeout_ms": 90000, "train_rows": 1600,
               "test_rows": 399, "encoded_features": 20, "schema": "jope.native-context/v5", "task": "regression"}
        expected = hashlib.sha256(json.dumps({k: ctx[k] for k in fk.REUSE_FIELDS}, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(fk.reuse_identity(ctx), expected)
        self.assertEqual(set(union({"contexts": [ctx]})), {("input_raw_sha256", C64), ("reuse_identity", expected)})
        row = {"input_ref": {"raw_sha256": C64}}
        self.assertEqual(union({"contexts": [row]}), [("input_raw_sha256", C64)])
        with self.assertRaises(fk.MalformedKey):
            fk.derive({"contexts": [dict(ctx, input_ref={"raw_sha256": E64})]})
        with self.assertRaises(fk.MalformedKey):
            fk.derive({"contexts": [dict(ctx, reuse_identity=M64)]})
        for broken in (dict(ctx, train_rows=True), dict(ctx, test_rows=None), {"input_raw_sha256": C64}):
            self.assertEqual(union({"contexts": [broken]}, "reuse_identity"), [])
            self.assertEqual(union({"contexts": [broken]}, "input_raw_sha256"), [C64])
        self.assertEqual(union({"input_raw_sha256": [E64, C64, "x"], "reuse_identity": M64}),
                         [("input_raw_sha256", C64), ("input_raw_sha256", E64), ("reuse_identity", M64)])

    def test_hunt_gate(self):
        gate = {"normalized_url": "http://Data.CityOfNewYork.us/resource/5uug-f49n.csv#x",
                "platform_id": "Socrata:5UUG-F49N", "source_keys": ["socrata:data.cityofnewyork.us:5uug-f49n"],
                "content_sha256": E64, "semantic_signature": {"exact": " sig-1 "}}
        for meta in ({"quality_checks": {"gates": {"deduplication": gate}}, "source_family": " Prices-Of "},
                     {"quality_checks.gates.deduplication": gate, "source_family": "prices-of"}):
            attrs = {"canonical": C64, "meta": meta}
            u, t = fk.derive(attrs)
            self.assertIn(("normalized_url", "https://data.cityofnewyork.us/resource/5uug-f49n.csv"), u)
            self.assertIn(("hunt_platform_id", "socrata:5uug-f49n"), u)
            self.assertIn(("socrata", "socrata:data.cityofnewyork.us:5uug-f49n"), u)
            self.assertIn(("hunt_content_mismatch", E64), t)
            self.assertIn(("hunt_semantic_signature", "sig-1"), t)
            self.assertIn(("legacy_source_family", "prices-of"), t)
        same = {"canonical": C64, "meta": {"quality_checks.gates.deduplication": {"content_sha256": C64}}}
        self.assertEqual(triggers(same, "hunt_content_mismatch"), [])
        self.assertEqual(union({"meta": {"quality_checks.gates.deduplication": {"platform_id": "socrata:unknown"}}}), [])

    def test_beyondarena_identity(self):
        self.assertEqual(union({"identity": "beyondarena:diamonds"}), [("beyondarena", "beyondarena:diamonds")])
        self.assertEqual(union({"source_family_id": "beyondarena:houses"}), [("beyondarena", "beyondarena:houses")])
        self.assertEqual(union({"identity": "item:" + C64}), [])
        pointer = {"name": "diamonds", "source_family_id": "beyondarena:diamonds", "schema": "x", "task": "regression",
                   "original_source": "https://github.com/tidyverse/ggplot2/blob/main/data-raw/diamonds.csv",
                   "license": "MIT License", "split_regime": "x", "files": {}}
        attrs = fk.attrs_from_beyondarena_pointer(pointer)
        self.assertEqual(set(attrs), {"identity", "provenance", "license"})
        u = union(attrs)
        self.assertIn(("beyondarena", "beyondarena:diamonds"), u)
        self.assertIn(("github_repo", "github:tidyverse/ggplot2"), u)
        nc = fk.attrs_from_beyondarena_pointer({"name": "miami_housing", "license": "CC BY-NC-SA",
                                                "original_source": "https://www.openml.org/d/43093"})
        self.assertEqual(triggers(nc, "rights_note"), ["CC BY-NC-SA"])
        self.assertEqual(union(nc, "openml_id"), ["openml:43093"])

    def test_rights_note(self):
        for note in ("reported by discovery agent: CC BY-NC 4.0", "Non-commercial use only", "resale prohibited",
                     "redistribution is restricted", "Data Use Agreement required", "(c) copyright 2020"):
            self.assertEqual(triggers({"license": {"status": "unknown", "note": note}}, "rights_note"),
                             [" ".join(note.split())], note)
        self.assertEqual(triggers({"provenance": {"license": {"note": "CC BY-NC-SA 4.0"}}}, "rights_note"),
                         ["CC BY-NC-SA 4.0"])
        self.assertEqual(triggers({"license": {"status": "recorded", "spdx": "CC-BY-NC-4.0"}}, "rights_note"),
                         ["CC-BY-NC-4.0"])
        for note in ("CC0 1.0 (TidyTuesday)", "source_license: GPL (R package X)", "MIT", "unknown", None):
            self.assertEqual(triggers({"license": {"note": note}}, "rights_note"), [], note)

    def test_builder_joined_triggers(self):
        attrs = {"legacy_source_family": " Daily-Closing ", "lineage_b_audit_family": ["fam-a", "fam-b", "unknown"],
                 "lineage_b_related_class": "class-x"}
        u, t = fk.derive(attrs)
        self.assertEqual(u, [])
        self.assertEqual(t, [("legacy_source_family", "daily-closing"), ("lineage_b_audit_family", "fam-a"),
                             ("lineage_b_audit_family", "fam-b"), ("lineage_b_related_class", "class-x")])

    def test_strict_mode_rejects_unrecognized_fields(self):
        with self.assertRaises(ValueError):
            fk.derive({"dataset_hash": H16, "canonical_sha": C64}, strict=True)
        self.assertEqual(fk.derive({"dataset_hash": H16, "canonical_sha": C64}), ([("dataset_hash", H16)], []))
        fk.derive({k: None for k in fk.ATTR_FIELDS}, strict=True)


class NullSafetyTests(unittest.TestCase):
    @staticmethod
    def everything(t):
        gate = {"normalized_url": t, "platform_id": t, "source_keys": [t], "content_sha256": t,
                "semantic_signature": {"exact": t}}
        return {
            "identity": t, "source_family_id": t, "dataset_hash": t, "canonical": t, "canonical_sha256": t,
            "meta_sha256": t, "files": {"meta": {"sha256": t}},
            "provenance": {"platform": t, "dataset_id": t, "url": t, "dedupe_fingerprint": t,
                           "parent_dataset_id": t, "source_table_id": t, "source_family": t,
                           "license": {"status": t, "note": t, "spdx": t}},
            "meta": {"source_family": t, "quality_checks": {"gates": {"deduplication": gate}}},
            "license": t, "contexts": [{"input_raw_sha256": t, "reuse_identity": t}], "input_raw_sha256": t,
            "reuse_identity": t, "legacy_source_family": t, "lineage_b_audit_family": [t],
            "lineage_b_related_class": t,
        }

    def test_null_values_never_produce_keys(self):
        for t in NULLS:
            self.assertEqual(fk.derive(self.everything(t)), ([], []), repr(t))
        self.assertEqual(fk.derive({}), ([], []))

    def test_null_never_links_two_identities(self):
        a, b = fk.derive(self.everything(None)), fk.derive(self.everything("unknown"))
        self.assertFalse(set(a[0]) & set(b[0]))
        self.assertFalse(set(a[1]) & set(b[1]))

    def test_partially_null_pairs_emit_nothing(self):
        cases = (
            {"platform": "socrata", "dataset_id": "unknown"},
            {"platform": "n/a", "dataset_id": "5uug-f49n"},
            {"platform": None, "parent_dataset_id": "p1", "source_table_id": "t1", "source_family": "f1"},
            {"platform": "rdatasets", "dataset_id": "none"},
        )
        for prov in cases:
            self.assertEqual(fk.derive({"provenance": prov}), ([], []), prov)
        # A known publisher platform still names its publisher; the null dataset_id gives no union key.
        self.assertEqual(fk.derive({"provenance": {"platform": "pmlb", "dataset_id": " "}}),
                         ([], [("publisher", "pmlb")]))
        for ident in ("beyondarena:", "beyondarena:unknown", "beyondarena: n/a "):
            self.assertEqual(fk.derive({"identity": ident}), ([], []), ident)
        self.assertEqual(fk.derive({"meta": {"quality_checks.gates.deduplication": {"platform_id": "unknown:x"}}}),
                         ([], []))

    def test_pin_redacted_values_never_produce_keys(self):
        redacted = {"value_sha256": E64}
        meta = {"source_family": redacted, "source_url": redacted, "feature_columns": ["a", "b"],
                "quality_checks": {"gates": {"deduplication": {
                    "normalized_url": redacted, "platform_id": redacted, "source_keys": [redacted],
                    "content_sha256": redacted, "semantic_signature": {"exact": redacted}}}}}
        attrs = {"canonical": C64, "meta": meta, "provenance": {"url": redacted, "platform": redacted},
                 "legacy_source_family": redacted}
        self.assertEqual(fk.derive(attrs), ([("canonical_sha256", C64)], []))

    def test_non_mapping_input_raises(self):
        for bad in (None, [], "attrs", 1):
            with self.assertRaises(TypeError):
                fk.derive(bad)


class ForbiddenTests(unittest.TestCase):
    FIELDS = ("collections", "ingest", "ingest.source", "ingest.tool_version", "ingest.relpath", "origin", "aliases",
              "target.column_order_sha256", "schema", "shape", "split", "task", "display_name", "positional_names",
              "materialized_columns", "Collections", "DISPLAY_NAME")

    def test_every_forbidden_field_raises_whatever_its_value(self):
        for name in self.FIELDS:
            for value in (None, [], "", "registry-real-v1", {"tool_version": "x"}, ["jailgun-regression-hunt"]):
                with self.assertRaises(fk.ForbiddenKey, msg=f"{name}={value!r}"):
                    fk.derive({"dataset_hash": H16, name: value})

    def test_nested_forbidden_fields_raise(self):
        for attrs in ({"target": {"index": 3, "column_order_sha256": "31770f25" + "0" * 56}},
                      {"provenance": {"url": "https://example.org/", "collections": ["registry-real-v1"]}},
                      {"provenance": {"ingest.source": "quant"}},
                      {"meta": {"display_name": "x"}}, {"meta": {"schema": "jope.meta/v1"}},
                      {"license": {"origin": "real"}}, {"files": {"ingest": {}}}):
            with self.assertRaises(fk.ForbiddenKey, msg=repr(attrs)):
                fk.derive(attrs)

    @staticmethod
    def catalog_row(h, canon, url, display):
        return {
            "id": h, "dataset_hash": h, "canonical_sha256": canon, "display_name": display, "aliases": [],
            "collections": ["registry-real-v1"], "origin": "real", "schema": "jope.dataset-catalog/v1",
            "task": "regression", "shape": {"rows": 2000, "columns": 9, "features": 8, "categorical_features": 0},
            "split": {"method": "random", "seed": 0, "train_rows": 1600, "test_rows": 400},
            "target": {"index": 8, "name": "y", "type": "float", "column_order_sha256": "31770f25" + "0" * 56},
            "ingest": {"source": "quant", "tool_version": "jope-datasets/0.1.0", "relpath": "a/b",
                       "ingested_at": "2026-06-27T00:00:00Z"},
            "files": {"meta": {"bytes": 10, "sha256": hashlib.sha256(h.encode()).hexdigest()}},
            "provenance": {"license": {"status": "unknown"}, "retrieved_at": "2026-06-27", "url": url},
        }

    def test_catalog_projection_drops_forbidden_fields(self):
        row = self.catalog_row(H16, C64, "https://example.org/a.csv", "Some Display Name")
        with self.assertRaises(fk.ForbiddenKey):
            fk.derive(row)
        attrs = fk.attrs_from_catalog_row(row)
        self.assertEqual(set(attrs), {"dataset_hash", "canonical", "provenance", "meta_sha256"})
        keys = fk.derive_exact_and_source_aware(row)
        text = json.dumps(keys)
        for leaked in ("registry-real-v1", "jope-datasets/0.1.0", "quant", "real\"", "31770f25", "Some Display Name",
                       "regression", "a/b"):
            self.assertNotIn(leaked, text, leaked)
        self.assertIn(("dataset_hash", H16), keys)
        self.assertEqual(fk.derive_exact_and_source_aware({"dataset_hash": "synth_0d380248e9016ba7"}),
                         [("dataset_hash", "synth_0d380248e9016ba7")])

    def test_uniform_catalog_fields_never_union(self):
        a = fk.derive_exact_and_source_aware(self.catalog_row(H16, C64, "https://example.org/a.csv", "Same Name"))
        b = fk.derive_exact_and_source_aware(self.catalog_row("fedcba9876543210", E64, "https://example.net/b.csv",
                                                              "Same Name"))
        self.assertFalse(set(a) & set(b))
        self.assertFalse({v for _, v in a} & {v for _, v in b})

    def test_check_edge(self):
        for kind in ("collections", "ingest.source", "target.column_order_sha256", "display_name", "origin",
                     "aliases", "schema", "collections:registry"):
            with self.assertRaises(fk.ForbiddenKey, msg=kind):
                fk.check_edge(kind, "x")
        for value in NULLS:
            with self.assertRaises(fk.ForbiddenKey):
                fk.check_edge("normalized_url", value)
        for kind in sorted(fk.UNION_KINDS | {"nds:value_strong", "curated_alias", "content_equal"}):
            fk.check_edge(kind, "x")

    def test_kind_registries(self):
        self.assertFalse(fk.UNION_KINDS & fk.TRIGGER_KINDS)
        for kind in fk.UNION_KINDS | fk.TRIGGER_KINDS:
            fk.check_edge(kind, "x")
        self.assertTrue(fk.SPLITTABLE_KINDS <= fk.SOURCE_AWARE_KINDS)
        self.assertEqual(fk.UNION_KINDS, fk.EXACT_KINDS | fk.CONTEXT_KINDS | fk.SOURCE_AWARE_KINDS | fk.IDENTITY_KINDS)


class DeterminismTests(unittest.TestCase):
    RICH = {
        "identity": "item:" + C64, "dataset_hash": H16, "canonical": C64, "meta_sha256": M64,
        "provenance": {"platform": "nyc open data / socrata", "dataset_id": "5uug-f49n", "dedupe_fingerprint": F40,
                       "url": "https://data.cityofnewyork.us/api/views/5uug-f49n/rows.csv?accessType=DOWNLOAD",
                       "license": {"status": "unknown", "note": "reported by discovery agent: CC BY-NC 4.0"}},
        "meta": {"source_family": "nyc-311",
                 "quality_checks": {"gates": {"deduplication": {
                     "normalized_url": "https://data.cityofnewyork.us/resource/5uug-f49n.csv",
                     "platform_id": "socrata:5uug-f49n",
                     "source_keys": ["socrata:data.cityofnewyork.us:5uug-f49n", "socrata:data.ny.gov:5uug-f49n"],
                     "content_sha256": E64, "semantic_signature": {"exact": "sig"}}}}},
        "contexts": [
            {"input_raw_sha256": "1" * 64, "evaluator_sha256": E64, "candidate_timeout_ms": 90000, "train_rows": 1600,
             "test_rows": 399, "encoded_features": 20},
            {"input_raw_sha256": "2" * 64, "evaluator_sha256": E64, "candidate_timeout_ms": 90000, "train_rows": 1607,
             "test_rows": 393, "encoded_features": 10},
        ],
        "legacy_source_family": "nyc-311", "lineage_b_audit_family": ["b", "a"],
    }

    def shuffled(self, obj, rng):
        if isinstance(obj, dict):
            items = list(obj.items())
            rng.shuffle(items)
            return {k: self.shuffled(v, rng) for k, v in items}
        if isinstance(obj, list):
            items = [self.shuffled(v, rng) for v in obj]
            rng.shuffle(items)
            return items
        return obj

    def test_output_is_sorted_unique_and_stable(self):
        first = fk.derive(self.RICH)
        for part in first:
            self.assertEqual(part, sorted(set(part)))
            self.assertTrue(all(isinstance(k, str) and isinstance(v, str) and fk.known(v) for k, v in part))
        # hash, canonical, meta, platform id, fingerprint, hunt platform id, 2 URLs, 2 socrata hosts, 2 raw, 2 reuse
        self.assertEqual(len(first[0]), 14)
        for _ in range(3):
            self.assertEqual(fk.derive(self.RICH), first)

    def test_insertion_and_list_order_do_not_matter(self):
        expected = fk.derive(self.RICH)
        rng = random.Random(20261005)
        for _ in range(25):
            self.assertEqual(fk.derive(self.shuffled(copy.deepcopy(self.RICH), rng)), expected)

    def test_derive_is_pure(self):
        before = copy.deepcopy(self.RICH)
        fk.derive(self.RICH)
        fk.derive_exact_and_source_aware(ForbiddenTests.catalog_row(H16, C64, "https://example.org/a.csv", "x"))
        self.assertEqual(self.RICH, before)

    def test_policy_is_canonical_and_matches_the_spec(self):
        self.assertEqual(fk.KEYS_POLICY_SHA256, hashlib.sha256(canonical(fk.KEYS_POLICY)).hexdigest())
        self.assertEqual(fk.HOSTING, {
            "vincentarelbundock.github.io", "raw.githubusercontent.com", "github.com", "gist.githubusercontent.com",
            "zenodo.org", "huggingface.co", "osf.io", "figshare.com", "ndownloader.figshare.com", "kaggle.com"})
        self.assertEqual(fk.AGGREGATOR_REPOS, {
            "rfordatascience/tidytuesday": "tidytuesday_date", "epistasislab/pmlb": "pmlb_name",
            "mwaskom/seaborn-data": "github_file", "fivethirtyeight/data": "github_file",
            "vincentarelbundock/rdatasets": "rdatasets_item"})
        self.assertEqual(fk.PAGES_FILE_LEVEL, {"corgis-edu"})
        self.assertEqual(fk.PUBLISHER_PLATFORMS, {"pmlb", "worldbank", "seaborn", "sklearn"})
        self.assertEqual(fk.SECOND_LEVEL_LABELS, {"gov", "co", "ac", "org", "com", "net", "edu", "govt", "gob", "gouv",
                                                  "gv", "go"})
        self.assertEqual(fk.RESTRICTIVE_NOTE.pattern,
                         r"(?i)(^|[^a-z])nc([^a-z]|$)|non[- ]?commercial|by-nc|redistribut[a-z]* (is )?"
                         r"(not|restricted|prohibited)|resale|proprietary|all rights reserved|prohibit|restricted"
                         r"|data use agreement|copyright")


class HygieneTests(unittest.TestCase):
    def test_imports_are_stdlib_and_common_only(self):
        tree = ast.parse(MODULE.read_text())
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                modules.add(node.module)
        banned = {"jope", "gp_worker", "local_gateway", "reuse_cache", "intake_registry", "family_protocol", "sqlite3",
                  "boto3", "botocore", "dltrain.s3io", "requests", "urllib.request", "socket"}
        self.assertFalse({m for m in modules if m in banned or m.split(".")[0] in banned})
        self.assertEqual(modules, {"__future__", "datetime", "hashlib", "re", "collections.abc", "urllib.parse",
                                   "dltrain.common"})


class PinnedReadinessCounts(unittest.TestCase):
    """The frozen cohort re-derived from the pinned readiness file reproduces the spec's survey counts."""

    def test_frozen_source_aware_counts(self):
        if not READINESS.exists() or sha256_file(READINESS) != READINESS_SHA256:
            self.skipTest("pinned readiness-v1 file is not available")
        records = [r for r in json.loads(READINESS.read_text())["records"] if r["cohort"] == "frozen"]
        self.assertEqual(len(records), 570)
        per_kind, values = {}, {}
        source_aware = 0
        for rec in records:
            u, _ = fk.derive(fk.attrs_from_catalog_row(rec))
            kinds = {k for k, _ in u}
            source_aware += bool(kinds & fk.SOURCE_AWARE_KINDS)
            for k, v in u:
                per_kind[k] = per_kind.get(k, 0) + 1
                values.setdefault(k, set()).add(v)
        self.assertEqual(source_aware, 562)
        expected = {"rdatasets_item": 464, "socrata": 44, "pmlb_name": 21, "tidytuesday_date": 16, "uci_id": 9,
                    "uci_legacy_folder": 5, "github_file": 3, "dataset_hash": 570, "canonical_sha256": 570,
                    "normalized_url": 570, "dedupe_fingerprint": 497, "platform_dataset_id": 497}
        for kind, count in expected.items():
            self.assertEqual(per_kind.get(kind, 0), count, kind)
        self.assertEqual(len(values["rdatasets_item"]), 464)   # item level: no two frozen datasets share one


if __name__ == "__main__":
    unittest.main()
