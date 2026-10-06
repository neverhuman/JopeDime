"""P2 family keys: derive(attrs) -> (union_keys, triggers) (PLAN §5 P2; spec union_keys / review_triggers).

One call per identity. Both results are sorted, de-duplicated lists of ``(kind, value)`` pairs:

- union keys: the family builder unions every identity that shares an equal ``(kind, value)``;
- triggers: never union; a trigger shared by >= 2 families becomes one review row.

Values are self-describing (for example ``rdatasets:ecdat/wages``, ``socrata:<host>:<4x4>``), so a
value-only join (the exposure fit keys) cannot confuse two kinds.

Rules:

- **Null safety.** A value takes part only when ``known()``: None, '', 'unknown', 'none', 'null' and
  'n/a' never produce a key, so null == null never links.
- **FORBIDDEN.** derive() never reads the catalog ``collections``, ``ingest.*``, ``origin``,
  ``aliases``, ``target.column_order_sha256``, ``schema``/``shape``/``split``/``task``, positional
  materialization names or ``display_name``. If any of them is present in its input (top level, or one
  level inside ``provenance``/``meta``/``license``/``target``/``files``, or as a dotted name) it raises
  ForbiddenKey, whatever the value. Project raw rows first with ``attrs_from_catalog_row()`` or
  ``attrs_from_beyondarena_pointer()``, which drop those fields unread.
- **Copied rules, never imported.** ``known()`` and ``source_url()`` are copied verbatim from
  ``next100_v1/family_protocol.py`` (sha256 20381727…); the fingerprint and canonical patterns follow
  its ``evidence_keys()``.
- **Source-aware keys** come from normalized URLs (and, for Rdatasets and PMLB only, from the platform
  dataset_id). Socrata is detected by URL path on non-hosting hosts, never by platform text.
  Aggregator repositories are keyed per file and never per repository.

Attributes read by derive() (anything else is ignored; see ATTR_FIELDS):

- ``identity`` / ``source_family_id``: 'beyondarena:<name>' gives the beyondarena identity key;
- ``dataset_hash``; ``canonical`` or ``canonical_sha256`` (64 hex);
- ``meta_sha256`` or ``files.meta.sha256``: the catalog meta blob (lineage_b_metadata_sha256 join);
- ``provenance``: platform, dataset_id, url, dedupe_fingerprint, parent_dataset_id, source_table_id,
  source_family, license;
- ``meta``: ``source_family`` (legacy trigger) and the hunt gate ``quality_checks.gates.deduplication``
  {normalized_url, platform_id, source_keys, content_sha256, semantic_signature} (nested or dotted);
  the meta copies of source_url/source_platform/source_dataset_id/dedupe_fingerprint are cross-check
  fields for the builder and are not keys;
- ``license``: {status, spdx, note} or a licence string (rights_note trigger);
- ``contexts``: prepared.json contexts (or epoch rows) giving input_raw_sha256 and reuse_identity;
  ``input_raw_sha256`` / ``reuse_identity`` may also be given directly (a string or a list);
- ``legacy_source_family``, ``lineage_b_audit_family``, ``lineage_b_related_class``: trigger values the
  builder joined from the legacy TSV and lineage-B (a string or a list).

Values redacted by the pin step (``{'value_sha256': ...}`` leaves in sealed members' metas) are not strings,
so they never produce a key; sealed members keep their keys through the plaintext provenance held in memory.
Returned values are plaintext: the builder writes sealed members' key and trigger values only as sha256.

Standard library only, plus ``dltrain.common.canonical``. Never imports jope, gp_worker,
local_gateway, reuse_cache, intake_registry or family_protocol.
"""
from __future__ import annotations

import datetime
import hashlib
import re
from collections.abc import Mapping
from urllib.parse import parse_qs, unquote, urlsplit, urlunsplit

from dltrain.common import canonical

SCHEMA = "jope.dl-family-keys/v1"


class ForbiddenKey(RuntimeError):
    """A FORBIDDEN field reached derive(). A union through a forbidden key is a bug (PLAN §5 P2)."""


class MalformedKey(ValueError):
    """One identity carries two different values for a single-valued key."""


# --- Copied rules (next100_v1/family_protocol.py, sha256 20381727b39035b73dde59c2058ec44005f27fa1e72d7c0ea90694cf65738026) ---

def known(value):
    return isinstance(value, str) and value.strip().lower() not in ('', 'unknown', 'none', 'null', 'n/a')


def source_url(value):
    """Keep case-sensitive path/query; remove fragment, normalize host/default port.

    HTTP and HTTPS delivery of the same resource are conservatively one family.
    Never drop query parameters: these can identify genuinely different tables.
    """
    if not known(value):
        return None
    try:
        u = urlsplit(value)
        if u.scheme.lower() not in ('http', 'https') or not u.hostname or u.username or u.password:
            return None
        host = u.hostname.lower()
        if ':' in host:
            host = '[' + host + ']'
        if u.port and u.port not in (80 if u.scheme.lower() == 'http' else 443,):
            host += ':' + str(u.port)
        return urlunsplit(('https', host, u.path or '/', u.query, ''))
    except ValueError:
        return None


# --- Policy (copied into alias-sources.json['policy'] through KEYS_POLICY; a change gives a new sha) ---

HOSTING = frozenset({
    "vincentarelbundock.github.io", "raw.githubusercontent.com", "github.com", "gist.githubusercontent.com",
    "zenodo.org", "huggingface.co", "osf.io", "figshare.com", "ndownloader.figshare.com", "kaggle.com",
})
AGGREGATOR_REPOS = {
    "rfordatascience/tidytuesday": "tidytuesday_date",
    "epistasislab/pmlb": "pmlb_name",
    "mwaskom/seaborn-data": "github_file",
    "fivethirtyeight/data": "github_file",
    "vincentarelbundock/rdatasets": "rdatasets_item",
}
PAGES_FILE_LEVEL = frozenset({"corgis-edu"})
PUBLISHER_PLATFORMS = frozenset({"pmlb", "worldbank", "seaborn", "sklearn"})
SECOND_LEVEL_LABELS = frozenset({"gov", "co", "ac", "org", "com", "net", "edu", "govt", "gob", "gouv", "gv", "go"})
# DOI resolvers redirect to a publisher; they are not one, so they carry no publisher trigger.
RESOLVER_HOSTS = frozenset({"doi.org", "dx.doi.org"})
RESTRICTIVE_NOTE = re.compile(
    r"(?i)(^|[^a-z])nc([^a-z]|$)|non[- ]?commercial|by-nc|redistribut[a-z]* (is )?(not|restricted|prohibited)"
    r"|resale|proprietary|all rights reserved|prohibit|restricted|data use agreement|copyright")
FORBIDDEN = ("collections", "ingest.*", "origin", "aliases", "target.column_order_sha256", "schema", "shape",
             "split", "task", "positional names", "display_name-alone")
# Field names that raise ForbiddenKey (any dotted component matches, case-insensitive).
FORBIDDEN_FIELDS = frozenset({
    "collections", "ingest", "origin", "aliases", "column_order_sha256", "schema", "shape", "split", "task",
    "display_name", "positional_names", "positional_columns", "materialized_columns", "materialization_names",
})
SCANNED_BLOCKS = ("provenance", "meta", "license", "target", "files")
ATTR_FIELDS = frozenset({
    "identity", "source_family_id", "dataset_hash", "canonical", "canonical_sha256", "meta_sha256", "files",
    "provenance", "meta", "license", "contexts", "input_raw_sha256", "reuse_identity", "legacy_source_family",
    "lineage_b_audit_family", "lineage_b_related_class",
    # carried for the builder, never keys:
    "item_id", "sample", "cohort", "registry_state", "epoch_status",
})
PROVENANCE_FIELDS = ("platform", "dataset_id", "url", "dedupe_fingerprint", "parent_dataset_id",
                     "source_table_id", "source_family", "license")
REUSE_FIELDS = ("input_raw_sha256", "evaluator_sha256", "candidate_timeout_ms", "train_rows", "test_rows",
                "encoded_features")

EXACT_KINDS = frozenset({
    "dataset_hash", "canonical_sha256", "lineage_b_metadata_sha256", "platform_dataset_id", "hunt_platform_id",
    "platform_parent_dataset_id", "platform_source_table_id", "platform_source_family", "dedupe_fingerprint",
    "normalized_url",
})
CONTEXT_KINDS = frozenset({"input_raw_sha256", "reuse_identity"})
SOURCE_AWARE_KINDS = frozenset({
    "rdatasets_item", "github_repo", "github_file", "gist", "ghpages_file", "socrata", "tidytuesday_date",
    "zenodo_record", "uci_id", "uci_legacy_folder", "pmlb_name", "openml_id", "hf_repo", "kaggle",
    "kaggle_competition", "doi",
})
IDENTITY_KINDS = frozenset({"beyondarena"})
UNION_KINDS = EXACT_KINDS | CONTEXT_KINDS | SOURCE_AWARE_KINDS | IDENTITY_KINDS
TRIGGER_KINDS = frozenset({
    "rdatasets_pkg", "socrata_domain", "socrata_4x4_cross_host", "github_owner", "tidytuesday_year", "hf_owner",
    "publisher", "pmlb_stem", "legacy_source_family", "lineage_b_audit_family", "lineage_b_related_class",
    "hunt_semantic_signature", "hunt_content_mismatch", "rights_note",
})
# Union kinds an owner decision may split (spec: github_repo, nds:*, curated_alias; never exact or same-source keys).
SPLITTABLE_KINDS = frozenset({"github_repo"})

_SHA256 = re.compile(r"[0-9a-f]{64}")
_FINGERPRINT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")   # family_protocol.evidence_keys
_DIGITS = re.compile(r"[0-9]+")
_RDATASETS_PAGE = re.compile(r"(?i)^/rdatasets/(csv|doc)/([^/]+)/([^/?#]+?)(\.csv|\.html)?$")
_RDATASETS_REPO_PATH = re.compile(r"(?i)^(csv|doc)/([^/]+)/([^/?#]+?)(\.csv|\.html)?$")
_RDATASETS_SOURCE_KEY = re.compile(r"(?i)rdatasets:([^/\s:]+)/([^/\s]+)")
_RDATASETS_DATASET_ID = re.compile(r"([^/\s]+)/([^/\s]+)")
_SOCRATA_PATH = re.compile(r"^/(?:api/views|api/v3/views|resource)/([A-Za-z0-9]{4}-[A-Za-z0-9]{4})(?:[/.]|$)")
_SOCRATA_SOURCE_KEY = re.compile(r"(?i)socrata:([a-z0-9][a-z0-9.-]*):([a-z0-9]{4}-[a-z0-9]{4})")
_DOI = re.compile(r"10\.[0-9]{4,9}/\S+")
_ZENODO_DOI = re.compile(r"10\.5281/zenodo\.([0-9]+)")
_PMLB_ID = re.compile(r"([0-9]+)_(.+)")
_YEAR = re.compile(r"[0-9]{4}")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_IPV4 = re.compile(r"[0-9]{1,3}(?:\.[0-9]{1,3}){3}")

_GITHUB_RESERVED = frozenset({
    "about", "apps", "collections", "contact", "customer-stories", "enterprise", "events", "explore", "features",
    "login", "marketplace", "new", "notifications", "organizations", "orgs", "pricing", "search", "security",
    "settings", "site", "sponsors", "topics", "trending", "users",
})
_GITHUB_FILE_VIEWS = frozenset({"blob", "raw", "tree", "blame", "edit"})
_HF_VIEWS = frozenset({"resolve", "blob", "tree", "raw", "viewer", "discussions", "commits", "commit", "blame",
                       "edit", "upload", "delete", "settings", "refs", "parquet"})
_UCI_HOSTS = frozenset({"archive.ics.uci.edu", "archive-beta.ics.uci.edu"})
_OPENML_HOSTS = frozenset({"openml.org", "api.openml.org", "new.openml.org"})


def norm(value):
    """Spec norm(): lowercase, trim, collapse whitespace."""
    return " ".join(value.split()).lower()


def registrable_domain(host):
    """Last 2 labels; last 3 when the TLD has 2 letters and the second-level label is in SECOND_LEVEL_LABELS.

    An approximation without a public suffix list (spec risk). IP literals are returned unchanged.
    """
    if not known(host):
        return None
    h = host.strip().lower()
    if h.startswith("["):
        return h[: h.index("]") + 1] if "]" in h else None
    if h.count(":") >= 2:   # a bare IPv6 literal
        return "[" + h + "]"
    h = h.split(":")[0].rstrip(".")
    if _IPV4.fullmatch(h):
        return h
    labels = [label for label in h.split(".") if label]
    if not labels:
        return None
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-1].isalpha() and labels[-2] in SECOND_LEVEL_LABELS:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


_HOSTING_DOMAINS = frozenset(registrable_domain(h) for h in HOSTING)


def is_hosting(host):
    """A HOSTING host, or any host under a HOSTING host's registrable domain (www., subdomains, <owner>.github.io)."""
    if not known(host):
        return False
    h = host.strip().lower()
    if h.count(":") <= 1 and not h.startswith("["):
        h = h.split(":")[0].rstrip(".")
    return h in HOSTING or registrable_domain(h) in _HOSTING_DOMAINS


def restrictive_note(text):
    return known(text) and RESTRICTIVE_NOTE.search(text) is not None


def reuse_identity(context):
    """sha256(canonical({input_raw_sha256, evaluator_sha256, candidate_timeout_ms, train_rows, test_rows,
    encoded_features})) of one prepared.json context, or None when a field is missing or mistyped."""
    if not isinstance(context, Mapping):
        return None
    fields = {k: context.get(k) for k in REUSE_FIELDS}
    if not all(isinstance(fields[k], str) and _SHA256.fullmatch(fields[k]) for k in REUSE_FIELDS[:2]):
        return None
    if not all(isinstance(fields[k], int) and not isinstance(fields[k], bool) for k in REUSE_FIELDS[2:]):
        return None
    return hashlib.sha256(canonical(fields)).hexdigest()


# --- internal helpers ---

def _add(bucket, kind, value):
    if known(value):
        bucket.add((kind, value))


def _strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [v for v in value if isinstance(v, str)]
    return []


def _mappings(value):
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, Mapping)]
    return []


def _forbidden_name(name):
    return isinstance(name, str) and any(part.strip() in FORBIDDEN_FIELDS for part in name.lower().split("."))


def _check_forbidden(attrs):
    for name, value in attrs.items():
        if _forbidden_name(name):
            raise ForbiddenKey(f"forbidden field {name!r} passed to derive()")
        if name in SCANNED_BLOCKS and isinstance(value, Mapping):
            for inner in value:
                if _forbidden_name(inner):
                    raise ForbiddenKey(f"forbidden field {name}.{inner} passed to derive()")


def _single_sha256(*candidates, what):
    values = {v for v in candidates if isinstance(v, str) and _SHA256.fullmatch(v)}
    if len(values) > 1:
        raise MalformedKey(f"identity carries {len(values)} different {what} values")
    return values.pop() if values else None


def _rdatasets(pkg, item, union, triggers):
    p, i = pkg.strip().lower(), item.strip().lower()
    if known(p) and known(i):
        _add(union, "rdatasets_item", f"rdatasets:{p}/{i}")
        _add(triggers, "rdatasets_pkg", f"rdatasets:{p}")
        return True
    return False


def _pmlb(name, union, triggers):
    if not known(name):
        return False
    n = name.strip()
    _add(union, "pmlb_name", "pmlb:" + n)
    m = _PMLB_ID.fullmatch(n)
    _add(triggers, "pmlb_stem", "pmlb_stem:" + (m.group(2) if m else n).lower())
    if m:   # an id inferred from a PMLB name is never a union key (spec)
        _add(triggers, "pmlb_stem", f"openml:{int(m.group(1))}")
    return True


def _doi(doi, union):
    d = doi.strip().lower()
    if not _DOI.fullmatch(d):
        return
    _add(union, "doi", "doi:" + d)
    m = _ZENODO_DOI.fullmatch(d)
    if m:   # a Zenodo DOI names one record id (version or concept); ids never collide across records
        _add(union, "zenodo_record", f"zenodo:{int(m.group(1))}")


def _bare_doi(text):
    if not known(text):
        return None
    t = text.strip()
    if t.lower().startswith("doi:"):
        t = t[4:].strip()
    return t if _DOI.fullmatch(t.lower()) else None


def _tidytuesday_date(path):
    for i in range(len(path) - 2):
        if path[i] == "data" and _YEAR.fullmatch(path[i + 1]) and _DATE.fullmatch(path[i + 2]):
            date = path[i + 2]
            if date[:4] != path[i + 1]:
                continue
            try:
                datetime.date.fromisoformat(date)
            except ValueError:
                continue
            return date
    return None


def _github_repo_path(owner, repo, path, union, triggers):
    """github_repo, or per-file keys for AGGREGATOR_REPOS (never a repository-level key for those)."""
    o = owner.strip().lower()
    r = repo.strip().lower()
    if r.endswith(".git"):
        r = r[:-4]
    if not (known(o) and known(r)) or o in _GITHUB_RESERVED:
        return
    key = f"{o}/{r}"
    rule = AGGREGATOR_REPOS.get(key)
    if rule is None:
        _add(union, "github_repo", "github:" + key)
        _add(triggers, "github_owner", "github:" + o)
        return
    keyed = False
    if rule == "rdatasets_item":
        m = _RDATASETS_REPO_PATH.match("/".join(path))
        if m and _rdatasets(m.group(2), m.group(3), union, triggers):
            return   # the package trigger replaces the owner trigger for Rdatasets
    elif rule == "tidytuesday_date":
        date = _tidytuesday_date(path)
        if date:
            _add(union, "tidytuesday_date", "tidytuesday:" + date)
            _add(triggers, "tidytuesday_year", "tidytuesday:" + date[:4])
            keyed = True
    elif rule == "pmlb_name":
        if len(path) >= 2 and path[0] == "datasets":
            keyed = _pmlb(path[1], union, triggers)
    _add(triggers, "github_owner", "github:" + o)
    if not keyed and path:
        _add(union, "github_file", "github:" + key + "/" + "/".join(path))


def _github(host, segs, union, triggers):
    if host == "media.githubusercontent.com":
        if not segs or segs[0] != "media":
            return
        segs, host = segs[1:], "raw.githubusercontent.com"
    if len(segs) < 1:
        return
    if host == "github.com" and len(segs) == 1:
        if segs[0].lower() not in _GITHUB_RESERVED:
            _add(triggers, "github_owner", "github:" + segs[0].strip().lower())
        return
    if len(segs) < 2:
        return
    owner, repo, rest = segs[0], segs[1], segs[2:]
    path = []
    if host == "raw.githubusercontent.com":
        if len(rest) >= 3 and rest[0] == "refs" and rest[1] in ("heads", "tags"):
            path = rest[3:]
        else:
            path = rest[1:]
    elif host == "github.com" and rest:
        if rest[0] in _GITHUB_FILE_VIEWS and len(rest) >= 2:
            if len(rest) >= 4 and rest[1] == "refs" and rest[2] in ("heads", "tags"):
                path = rest[4:]
            else:
                path = rest[2:]
        elif rest[0] == "releases" and len(rest) >= 3 and rest[1] == "download":
            path = rest
    _github_repo_path(owner, repo, path, union, triggers)


def _pages(host, path, segs, union, triggers):
    owner = host[: -len(".github.io")]
    if not known(owner):
        return
    if owner == "vincentarelbundock":
        m = _RDATASETS_PAGE.match(path)
        if m and _rdatasets(m.group(2), m.group(3), union, triggers):
            return
        _add(union, "ghpages_file", host + path)
        _add(triggers, "github_owner", "github:" + owner)
        return
    _add(triggers, "github_owner", "github:" + owner)
    if owner in PAGES_FILE_LEVEL:
        _add(union, "ghpages_file", host + path)
    else:   # project page <owner>.github.io/<repo>/...; a single-segment path belongs to the user-site repo
        repo = segs[0].lower() if len(segs) >= 2 else host
        _add(union, "github_repo", f"github:{owner}/{repo}")


def _url(raw, union, triggers):
    """normalized_url plus the source-aware keys and publisher trigger of one URL field."""
    nurl = source_url(raw)
    if nurl is None:
        doi = _bare_doi(raw)
        if doi:
            _doi(doi, union)
        return
    _add(union, "normalized_url", nurl)
    parts = urlsplit(nurl)
    host = (parts.hostname or "").rstrip(".")
    path = parts.path
    segs = [s for s in path.split("/") if s]
    bare = host[4:] if host.startswith("www.") else host
    hosting = is_hosting(host)

    if bare in RESOLVER_HOSTS:
        _doi(unquote(path.lstrip("/")), union)
    elif bare.endswith(".github.io"):
        _pages(bare, path, segs, union, triggers)
    elif bare in ("raw.githubusercontent.com", "media.githubusercontent.com", "github.com"):
        _github(bare, segs, union, triggers)
    elif bare == "codeload.github.com":
        if len(segs) >= 2:
            _github_repo_path(segs[0], segs[1], [], union, triggers)
    elif bare in ("gist.githubusercontent.com", "gist.github.com"):
        if len(segs) >= 2:
            o, g = segs[0].strip().lower(), segs[1].strip().lower()
            if known(o) and known(g):
                _add(union, "gist", f"gist:{o}/{g}")
                _add(triggers, "github_owner", "github:" + o)
    elif bare == "zenodo.org":
        rec = None
        if len(segs) >= 2 and segs[0] in ("record", "records") and _DIGITS.fullmatch(segs[1]):
            rec = segs[1]
        elif len(segs) >= 3 and segs[0] == "api" and segs[1] in ("record", "records") and _DIGITS.fullmatch(segs[2]):
            rec = segs[2]
        if rec:
            _add(union, "zenodo_record", f"zenodo:{int(rec)}")
    elif bare == "huggingface.co":
        s = segs[1:] if segs[:1] == ["api"] else segs
        if len(s) >= 2 and s[0] == "datasets":
            if len(s) >= 3 and s[2] not in _HF_VIEWS:
                o, r = s[1].lower(), s[2].lower()
                _add(union, "hf_repo", f"hf:{o}/{r}")
                _add(triggers, "hf_owner", "hf:" + o)
            else:   # legacy canonical dataset without an owner namespace
                _add(union, "hf_repo", "hf:" + s[1].lower())
    elif bare == "kaggle.com":
        if len(segs) >= 3 and segs[0] == "datasets":
            _add(union, "kaggle", f"kaggle:{segs[1].lower()}/{segs[2].lower()}")
        elif len(segs) >= 2 and segs[0] in ("c", "competitions"):
            _add(union, "kaggle_competition", "kaggle_competition:" + segs[1].lower())
        elif segs[:4] == ["api", "v1", "datasets", "download"] and len(segs) >= 6:
            _add(union, "kaggle", f"kaggle:{segs[4].lower()}/{segs[5].lower()}")
        elif (segs[:4] == ["api", "v1", "competitions", "data"] and len(segs) >= 6
              and segs[4] in ("download", "download-all")):
            _add(union, "kaggle_competition", "kaggle_competition:" + segs[5].lower())
    elif bare in _UCI_HOSTS:
        if len(segs) >= 2 and segs[0] == "dataset" and _DIGITS.fullmatch(segs[1]):
            _add(union, "uci_id", f"uci:{int(segs[1])}")
        elif len(segs) >= 3 and segs[:2] == ["static", "public"] and _DIGITS.fullmatch(segs[2]):
            _add(union, "uci_id", f"uci:{int(segs[2])}")
        elif len(segs) >= 3 and segs[:2] == ["ml", "machine-learning-databases"]:
            _add(union, "uci_legacy_folder", "uci_folder:" + segs[2].lower())
    elif bare in _OPENML_HOSTS:
        oid = None
        if len(segs) == 2 and segs[0] == "d" and _DIGITS.fullmatch(segs[1]):
            oid = segs[1]
        elif segs[:2] == ["api", "v1"]:
            rest = segs[2:]
            if rest[:1] in (["json"], ["xml"]):
                rest = rest[1:]
            if len(rest) == 2 and rest[0] == "data" and _DIGITS.fullmatch(rest[1]):
                oid = rest[1]
        elif segs == ["search"]:
            q = parse_qs(parts.query)
            ids = q.get("id", [])
            if q.get("type") == ["data"] and len(ids) == 1 and _DIGITS.fullmatch(ids[0]):
                oid = ids[0]
        if oid:   # direct OpenML URLs only; ids inferred from PMLB names stay triggers
            _add(union, "openml_id", f"openml:{int(oid)}")
            _add(triggers, "pmlb_stem", f"openml:{int(oid)}")

    if not hosting and bare not in RESOLVER_HOSTS:
        m = _SOCRATA_PATH.match(path)
        if m:
            four = m.group(1).lower()
            _add(union, "socrata", f"socrata:{bare}:{four}")
            _add(triggers, "socrata_domain", "socrata:" + bare)
            _add(triggers, "socrata_4x4_cross_host", "socrata:*:" + four)
        _add(triggers, "publisher", registrable_domain(host))


def _socrata_source_key(text, union, triggers):
    m = _SOCRATA_SOURCE_KEY.fullmatch(text.strip())
    if not m:
        return
    host = m.group(1).lower().rstrip(".")
    host = host[4:] if host.startswith("www.") else host
    four = m.group(2).lower()
    _add(union, "socrata", f"socrata:{host}:{four}")
    _add(triggers, "socrata_domain", "socrata:" + host)
    _add(triggers, "socrata_4x4_cross_host", "socrata:*:" + four)


def _provenance(prov, union, triggers):
    platform = prov.get("platform")
    pn = norm(platform) if known(platform) else None
    did = prov.get("dataset_id")
    if pn and known(did):
        d = did.strip()
        _add(union, "platform_dataset_id", f"{pn}:{d}")
        if pn == "rdatasets":
            m = _RDATASETS_DATASET_ID.fullmatch(d)
            if m:
                _rdatasets(m.group(1), m.group(2), union, triggers)
        elif pn == "pmlb":
            _pmlb(d, union, triggers)
    if pn:
        for field in ("parent_dataset_id", "source_table_id", "source_family"):
            value = prov.get(field)
            if known(value):
                _add(union, "platform_" + field, f"{pn}:{value.strip()}")
        if pn in PUBLISHER_PLATFORMS:
            _add(triggers, "publisher", pn)
    fp = prov.get("dedupe_fingerprint")
    if isinstance(fp, str) and _FINGERPRINT.fullmatch(fp):
        _add(union, "dedupe_fingerprint", f"{pn or 'none'}:{fp}")
    _url(prov.get("url"), union, triggers)


def _dedupe_gate(meta):
    gate = meta.get("quality_checks.gates.deduplication")
    if isinstance(gate, Mapping):
        return gate
    qc = meta.get("quality_checks")
    if isinstance(qc, Mapping):
        gates = qc.get("gates")
        if isinstance(gates, Mapping) and isinstance(gates.get("deduplication"), Mapping):
            return gates["deduplication"]
        if isinstance(qc.get("gates.deduplication"), Mapping):
            return qc["gates.deduplication"]
    return None


def _meta(meta, canonical_sha, union, triggers):
    source_family = meta.get("source_family")
    if known(source_family):
        _add(triggers, "legacy_source_family", norm(source_family))
    gate = _dedupe_gate(meta)
    if not gate:
        return
    _url(gate.get("normalized_url"), union, triggers)   # hunt_normalized_url maps to the normalized_url key
    pid = gate.get("platform_id")
    if known(pid):
        head, sep, tail = pid.partition(":")
        if not sep or (known(head) and known(tail)):
            _add(union, "hunt_platform_id", norm(pid))
    for key in _strings(gate.get("source_keys")):
        _socrata_source_key(key, union, triggers)
    content = gate.get("content_sha256")
    if known(content) and canonical_sha is not None and content.strip() != canonical_sha:
        _add(triggers, "hunt_content_mismatch", content.strip())
    signature = gate.get("semantic_signature")
    if isinstance(signature, Mapping):
        signature = signature.get("exact")
    if known(signature):
        _add(triggers, "hunt_semantic_signature", signature.strip())


def _rights(license_value, triggers):
    if isinstance(license_value, str):
        texts = [license_value]
    elif isinstance(license_value, Mapping):
        texts = [license_value.get("note"), license_value.get("spdx")]
    else:
        texts = []
    for text in texts:
        if restrictive_note(text):
            _add(triggers, "rights_note", " ".join(text.split()))


def _contexts(attrs, union):
    for ctx in _mappings(attrs.get("contexts")):
        ref = ctx.get("input_ref")
        raw = _single_sha256(ctx.get("input_raw_sha256"),
                             ref.get("raw_sha256") if isinstance(ref, Mapping) else None,
                             what="context input_raw_sha256")
        if raw:
            _add(union, "input_raw_sha256", raw)
        computed = reuse_identity(ctx)
        given = ctx.get("reuse_identity")
        if computed and isinstance(given, str) and _SHA256.fullmatch(given) and given != computed:
            raise MalformedKey("context reuse_identity does not match its prepared fields")
        rid = computed or (given if isinstance(given, str) and _SHA256.fullmatch(given) else None)
        if rid:
            _add(union, "reuse_identity", rid)
    for value in _strings(attrs.get("input_raw_sha256")):
        if _SHA256.fullmatch(value):
            _add(union, "input_raw_sha256", value)
    for value in _strings(attrs.get("reuse_identity")):
        if _SHA256.fullmatch(value):
            _add(union, "reuse_identity", value)


# --- public API ---

def derive(attrs, *, strict=False):
    """(union_keys, triggers) of one identity; each a sorted list of unique (kind, value) pairs.

    Raises ForbiddenKey when a FORBIDDEN field is present, MalformedKey when a single-valued key is
    contradictory and, with strict=True, ValueError for a top-level field outside ATTR_FIELDS (so a
    misspelled field cannot silently drop a key). Pure: the input is never modified.
    """
    if not isinstance(attrs, Mapping):
        raise TypeError("derive() takes a mapping of identity attributes")
    _check_forbidden(attrs)
    if strict:
        unknown = sorted(str(k) for k in attrs if k not in ATTR_FIELDS)
        if unknown:
            raise ValueError(f"derive(strict=True): unrecognized attribute(s) {unknown}")
    union, triggers = set(), set()

    for name in ("identity", "source_family_id"):
        value = attrs.get(name)
        if isinstance(value, str) and value.startswith("beyondarena:") and known(value[len("beyondarena:"):]):
            _add(union, "beyondarena", "beyondarena:" + value[len("beyondarena:"):].strip())

    dataset_hash = attrs.get("dataset_hash")
    if known(dataset_hash):
        _add(union, "dataset_hash", dataset_hash.strip())
    canonical_sha = _single_sha256(attrs.get("canonical"), attrs.get("canonical_sha256"), what="canonical_sha256")
    if canonical_sha:
        _add(union, "canonical_sha256", canonical_sha)
    files = attrs.get("files")
    files_meta = files.get("meta") if isinstance(files, Mapping) else None
    meta_sha = _single_sha256(attrs.get("meta_sha256"),
                              files_meta.get("sha256") if isinstance(files_meta, Mapping) else None,
                              what="meta sha256")
    if meta_sha:
        _add(union, "lineage_b_metadata_sha256", meta_sha)

    prov = attrs.get("provenance")
    if isinstance(prov, Mapping):
        _provenance(prov, union, triggers)
        _rights(prov.get("license"), triggers)
    meta = attrs.get("meta")
    if isinstance(meta, Mapping):
        _meta(meta, canonical_sha, union, triggers)
    _rights(attrs.get("license"), triggers)
    _contexts(attrs, union)

    for value in _strings(attrs.get("legacy_source_family")):
        if known(value):
            _add(triggers, "legacy_source_family", norm(value))
    for name in ("lineage_b_audit_family", "lineage_b_related_class"):
        for value in _strings(attrs.get(name)):
            if known(value):
                _add(triggers, name, value.strip())

    for kind, _ in union:
        if kind not in UNION_KINDS:
            raise RuntimeError(f"derive() produced an unregistered union kind {kind!r}")
    for kind, _ in triggers:
        if kind not in TRIGGER_KINDS:
            raise RuntimeError(f"derive() produced an unregistered trigger kind {kind!r}")
    return sorted(union), sorted(triggers)


def url_keys(url):
    """(union_keys, triggers) of a single URL: normalized_url, source-aware keys and the publisher trigger."""
    union, triggers = set(), set()
    _url(url, union, triggers)
    return sorted(union), sorted(triggers)


def rdatasets_item_key(text):
    """'rdatasets:<pkg>/<item>' (lower-cased) from a lineage-B source_key or any Rdatasets csv/doc URL, else None.

    Local file paths (non-portable lineage-B source_keys) never match.
    """
    if not known(text):
        return None
    m = _RDATASETS_SOURCE_KEY.fullmatch(text.strip())
    if m:
        p, i = m.group(1).lower(), m.group(2).lower()
        return f"rdatasets:{p}/{i}" if known(p) and known(i) else None
    union, _ = url_keys(text)
    items = sorted(v for k, v in union if k == "rdatasets_item")
    return items[0] if len(items) == 1 else None


def check_edge(kind, value):
    """Verify-time guard: no edge kind names a FORBIDDEN field and no edge value is null/unknown."""
    if _forbidden_name(kind) or any(_forbidden_name(part) for part in str(kind).split(":")):
        raise ForbiddenKey(f"edge kind {kind!r} names a forbidden field")
    if not known(value):
        raise ForbiddenKey(f"edge {kind!r} has a null or unknown value")


def attrs_from_catalog_row(row):
    """derive() attributes of a catalog v1 row, an incoming catalog_row or a readiness record.

    Keeps dataset_hash, canonical_sha256, the provenance fields and files.meta.sha256; FORBIDDEN fields
    (collections, ingest, origin, aliases, schema, shape, split, task, target, display_name) are dropped
    without being read.
    """
    if not isinstance(row, Mapping):
        raise TypeError("attrs_from_catalog_row() takes a mapping")
    out = {}
    if "dataset_hash" in row:
        out["dataset_hash"] = row["dataset_hash"]
    canon = row.get("canonical_sha256", row.get("canonical"))
    if canon is not None:
        out["canonical"] = canon
    prov = row.get("provenance")
    if isinstance(prov, Mapping):
        out["provenance"] = {k: prov[k] for k in PROVENANCE_FIELDS if k in prov}
    files = row.get("files")
    if isinstance(files, Mapping) and isinstance(files.get("meta"), Mapping) and "sha256" in files["meta"]:
        out["meta_sha256"] = files["meta"]["sha256"]
    return out


def attrs_from_beyondarena_pointer(pointer):
    """derive() attributes of a BeyondArena source pointer: the identity, original_source and licence text."""
    if not isinstance(pointer, Mapping):
        raise TypeError("attrs_from_beyondarena_pointer() takes a mapping")
    out = {}
    sfid, name = pointer.get("source_family_id"), pointer.get("name")
    if isinstance(sfid, str) and sfid.startswith("beyondarena:") and known(sfid[len("beyondarena:"):]):
        out["identity"] = sfid.strip()
    elif known(name):
        out["identity"] = "beyondarena:" + name.strip()
    if known(pointer.get("original_source")):
        out["provenance"] = {"url": pointer["original_source"]}
    if known(pointer.get("license")):
        out["license"] = {"note": pointer["license"]}
    return out


def derive_exact_and_source_aware(row):
    """Union keys of a fit-set or external-alias row (catalog v1 row, readiness record or {'dataset_hash': id})."""
    return derive(attrs_from_catalog_row(row))[0]


KEYS_POLICY = {
    "schema": SCHEMA,
    "known_null_tokens": ["", "unknown", "none", "null", "n/a"],
    "source_url_rule": "family_protocol.source_url sha256 20381727b39035b73dde59c2058ec44005f27fa1e72d7c0ea90694cf65738026",
    "HOSTING": sorted(HOSTING),
    "hosting_registrable_domains": sorted(_HOSTING_DOMAINS),
    "AGGREGATOR_REPOS": dict(sorted(AGGREGATOR_REPOS.items())),
    "PAGES_FILE_LEVEL": sorted(PAGES_FILE_LEVEL),
    "PUBLISHER_PLATFORMS": sorted(PUBLISHER_PLATFORMS),
    "SECOND_LEVEL_LABELS": sorted(SECOND_LEVEL_LABELS),
    "RESOLVER_HOSTS": sorted(RESOLVER_HOSTS),
    "RESTRICTIVE_NOTE": RESTRICTIVE_NOTE.pattern,
    "FORBIDDEN": list(FORBIDDEN),
    "FORBIDDEN_FIELDS": sorted(FORBIDDEN_FIELDS),
    "union_kinds": sorted(UNION_KINDS),
    "trigger_kinds": sorted(TRIGGER_KINDS),
    "splittable_kinds": sorted(SPLITTABLE_KINDS),
}
KEYS_POLICY_SHA256 = hashlib.sha256(canonical(KEYS_POLICY)).hexdigest()

__all__ = [
    "SCHEMA", "ForbiddenKey", "MalformedKey", "known", "norm", "source_url", "registrable_domain", "is_hosting",
    "restrictive_note", "reuse_identity", "derive", "url_keys", "rdatasets_item_key", "check_edge",
    "attrs_from_catalog_row", "attrs_from_beyondarena_pointer", "derive_exact_and_source_aware",
    "HOSTING", "AGGREGATOR_REPOS", "PAGES_FILE_LEVEL", "PUBLISHER_PLATFORMS", "SECOND_LEVEL_LABELS",
    "RESOLVER_HOSTS", "RESTRICTIVE_NOTE", "FORBIDDEN", "FORBIDDEN_FIELDS", "ATTR_FIELDS", "PROVENANCE_FIELDS",
    "REUSE_FIELDS", "EXACT_KINDS", "CONTEXT_KINDS", "SOURCE_AWARE_KINDS", "IDENTITY_KINDS", "UNION_KINDS",
    "TRIGGER_KINDS", "SPLITTABLE_KINDS", "KEYS_POLICY", "KEYS_POLICY_SHA256",
]
