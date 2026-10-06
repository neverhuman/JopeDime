"""L7a: original T1 source rows, content match, and train-only refit (owner decisions L6(a) and L7a).

T1 rows (research_2k data objects) hold values the acquisition tool fitted over every cleaned source row: numeric
median imputation, category codes, min-max scaling, float32 '%.9g' printing (t1prep). L7a goes back to the re-fetched
original source file and refits what T1 cannot undo, on outer-train rows only.

1. read_source(): the fetched file (plain, gzip or zip, as found; HTML/XML/JSON refused) is parsed as delimited text.
   The header row, delimiter and encoding are the ones that hold every column the meta names; nothing else is guessed.
2. Only the meta's own NON-fitted steps are applied to every raw row: column selection by name (original_headers /
   output_headers / feature_columns, target = target_original_name or source_target_column_original), the category
   value -> code mapping by key (meta mapping or categories list), and the 'datetime -> days since 1970-01-01' derivation
   the meta records. Missing cells stay NaN (numeric) or -1 (categorical). Nothing is imputed or scaled.
3. match(): every T1 row of A is de-scaled with the meta's feature_scaling / target_scaling and compared with every
   raw row. A T1 row matches a raw row when every non-missing raw cell equals the de-scaled T1 cell within the float32
   '%.9g' printing error (MATCH_TOL_REL * span, plus float64 slack), every categorical code is equal, and every missing
   raw cell sits where T1 holds that column's impute value. Raw rows with identical reconstructed values are one value
   class (they are indistinguishable and give identical L7a values). Every row of A must match exactly ONE value class,
   and no class may be claimed by more A rows than the raw file holds (one-to-one); otherwise the dataset is refused
   and stays on L7b. The tolerance is never widened to make rows match.
4. refit()/prepare(): for a matched dataset, per (dataset, outer fold): numeric missing cells are imputed with the
   median of the non-missing OUTER-TRAIN values of that column (outer-test rows get that train median), category codes
   are refitted by t1prep's train-only recode (missing and outer-train-unseen share one extra code), target copies are
   detected on outer-train rows only, then trainonly.prepare_split (train-only min-max, inner folds, JTT1).

Numpy and the standard library only. No S3, network, GPU or evaluator access.
"""
from __future__ import annotations

import array
import csv
import datetime as _dt
import gzip
import hashlib
import io
import math
import re
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dltrain import t1prep, trainonly

SCHEMA = "jope.dl-t1raw/v1"
MATCH_TOL_REL = t1prep.F32_PRINT_REL      # float32 '%.9g' printing error of a value in [0, 1]: <= 2^-25; 4x margin
D9_PRINT_REL = 1e-8                       # float64 '%.9g' printing error: <= 5e-9 of the scaled value; 2x margin
F64_SLACK = 8.0 * 2.0 ** -52              # float64 subtract/divide/multiply error relative to the column magnitude
NA_STRINGS = frozenset(["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>",
                        "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null"])          # pandas read_csv defaults
BOOL_TRUE = frozenset(["True", "TRUE", "true"])
BOOL_FALSE = frozenset(["False", "FALSE", "false"])
DELIMITERS = (",", "\t", ";", "|")
FALLBACK_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
HEADER_SCAN_RECORDS = 200
INDEX_PROBE_RECORDS = 2000
DIAG_ROWS = 200
MISSING_CODE = -1
UNMAPPED_CODE = -2
EPOCH = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)


class T1RawError(ValueError):
    """A source that cannot be read or reconstructed; `reason` is a short stable code for the receipts."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# ---------------------------------------------------------------------------------------------- column specs

def target_source_name(meta: Mapping[str, Any]) -> str:
    name = meta.get("target_original_name") or meta.get("source_target_column_original")
    if not isinstance(name, str) or not name:
        raise T1RawError("meta_no_target_name")
    return name


def columns(meta: Mapping[str, Any], width: int) -> list[dict]:
    """t1prep.layout of the T1 file (features, file order) plus the target spec (last). Each spec carries the raw
    column name and the non-fitted derivation recorded in the meta."""
    specs = t1prep.layout(meta, width)
    enc = meta.get("feature_encodings") or {}
    stamp = str((meta.get("acquisition") or {}).get("acquired_at") or meta.get("retrieved_at") or meta.get("created_at") or "")
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", stamp)
    today = (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
    for s in specs:
        e = enc.get(s["name"]) or {}
        s["source_name"] = s["name"]
        s["derive"] = None
        if "derived_from" in e:
            if e.get("derived_from") == "datetime" and e.get("unit") == "days since 1970-01-01":
                if today is None:
                    raise T1RawError("meta_no_acquisition_date", s["name"])
                s["derive"] = "datetime_days"
                s["today"], s["ref_year"] = today, today[0]   # pandas: a time-only cell takes the current date
            else:
                raise T1RawError("unsupported_derivation", f"{s['name']}: {e.get('derived_from')}/{e.get('unit')}")
        if s["kind"] == "numeric":
            imp = s.get("impute_value")
            s["impute_value"] = float(imp) if isinstance(imp, (int, float)) and not isinstance(imp, bool) and math.isfinite(float(imp)) else None
        elif s.get("unseen_code") is not None:
            raise T1RawError("unsupported_encoding", f"{s['name']}: unseen-code map (category lost in T1)")
        else:
            imp = s.get("impute_value")
            if not isinstance(imp, (int, float)) or isinstance(imp, bool) or not float(imp).is_integer() or int(imp) not in s["inverse"]:
                raise T1RawError("unsupported_encoding", f"{s['name']}: categorical impute_value {imp!r} is not a code")
            s["impute_code"] = int(imp)
    ts = meta.get("target_scaling") or {}
    if "min" not in ts or "max" not in ts or list(ts.get("range") or [0.0, 1.0]) != [0.0, 1.0]:
        raise T1RawError("meta_target_scaling", repr(ts)[:200])
    specs.append({"index": len(specs), "name": "target", "source_name": target_source_name(meta), "kind": "numeric",
                  "lo": float(ts["min"]), "hi": float(ts["max"]), "impute_value": None, "derive": None, "target": True})
    return specs


def tolerance(spec: Mapping[str, Any], descaled, tol_rel: float = MATCH_TOL_REL, decimal9: bool = False) -> np.ndarray:
    """Largest |raw - de-scaled T1| the printing of (raw - lo) / span can produce, per cell: the float32 '%.9g' bound
    (tol_rel * span, coarser above 1), or, for a column whose T1 values prove a float64 '%.9g' printing (decimal9), the
    tighter 9-significant-digit bound D9_PRINT_REL * |raw - lo|. Both add float64 slack for the de-scaling."""
    span = abs(spec["hi"] - spec["lo"])
    d = np.asarray(descaled, dtype=np.float64)
    mag = np.maximum(np.abs(d), max(abs(spec["lo"]), abs(spec["hi"])))
    if decimal9:
        return D9_PRINT_REL * np.abs(d - spec["lo"]) + F64_SLACK * mag
    rel = np.maximum(1.0, np.abs(d - spec["lo"]) / span) if span > 0 else 1.0   # |scaled value| > 1: coarser float32 grid
    return tol_rel * span * rel + F64_SLACK * mag


def printed_decimal9(col) -> bool:
    """True when a T1 column was printed '%.9g' from float64 rather than from float32. A float32-printed value v always
    survives v -> float32 -> '%.9g' -> float64; a float64-printed value with more than float32 precision does not. The
    column counts as decimal9 when more than half of its values that are not exact float32 numbers fail that trip."""
    v = np.asarray(col, dtype=np.float64)
    v = v[np.isfinite(v)]
    f32 = v.astype(np.float32)
    inexact = f32.astype(np.float64) != v
    if not bool(np.any(inexact)):
        return False
    trip = np.asarray([float("%.9g" % x) for x in f32[inexact]], dtype=np.float64)
    return int(np.sum(trip != v[inexact])) * 2 > int(np.sum(inexact))


# ---------------------------------------------------------------------------------------------- reading the source

def _container(meta: Mapping[str, Any]) -> dict:
    c = (((meta.get("acquisition") or {}).get("download") or {}).get("container")) or ""
    out = {"declared": c or None, "member": None, "sep": None, "encoding": None}
    parts = c.split(";") if c else []
    if parts and parts[0].startswith("zip:"):
        out["member"] = parts[0][4:]
    for p in parts[1:]:
        if p.startswith("sep="):
            v = p[4:]
            out["sep"] = "\t" if v in ("TAB", "\\t") else v
        elif p.startswith("enc="):
            out["encoding"] = p[4:]
    enc = ((meta.get("acquisition") or {}).get("download") or {}).get("encoding")
    if enc and not out["encoding"]:
        out["encoding"] = enc
    return out


def _opener(path: Path, member_hint):
    """(factory returning a fresh binary stream, layer list). gzip and zip layers are unwrapped by magic bytes."""
    with open(path, "rb") as f:
        head = f.read(4)
    if head[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(path) as zf:
            names = [i.filename for i in zf.infolist() if not i.is_dir()]
        pick = None
        if member_hint:
            hits = [n for n in names if n == member_hint or n.rsplit("/", 1)[-1] == member_hint]
            pick = hits[0] if len(hits) == 1 else None
        if pick is None:
            tables = [n for n in names if n.lower().endswith((".csv", ".tsv", ".txt"))]
            if len(tables) == 1:
                pick = tables[0]
            elif len(names) == 1:
                pick = names[0]
        if pick is None:
            raise T1RawError("zip_member_ambiguous", f"{len(names)} members, hint {member_hint!r}")

        def factory(pick=pick):
            zf = zipfile.ZipFile(path)
            return zf.open(pick)
        return factory, [f"zip:{pick}"]
    if head[:2] == b"\x1f\x8b":
        with gzip.open(path, "rb") as g:
            inner = g.read(2)
        if inner == b"\x1f\x8b":
            return (lambda: gzip.GzipFile(fileobj=gzip.open(path, "rb"), mode="rb")), ["gzip", "gzip"]
        return (lambda: gzip.open(path, "rb")), ["gzip"]
    return (lambda: open(path, "rb")), ["plain"]


def _mangle(names: Sequence[str]) -> list[str]:
    """pandas read_csv header names: an empty name becomes 'Unnamed: <i>', a repeated name 'name.<k>'."""
    out, seen = [], {}
    for i, n in enumerate(names):
        n = n if n != "" else f"Unnamed: {i}"
        if n in seen:
            seen[n] += 1
            n2 = f"{n}.{seen[n]}"
            while n2 in seen:
                seen[n] += 1
                n2 = f"{n}.{seen[n]}"
            seen[n2] = 0
            out.append(n2)
        else:
            seen[n] = 0
            out.append(n)
    return out


def _resolve(header: Sequence[str], wanted: Sequence[str]) -> tuple[dict, list]:
    """Column index per wanted name: exact (pandas-mangled) name first, then a unique whitespace-stripped name."""
    names = _mangle([h.lstrip("\ufeff") if i == 0 else h for i, h in enumerate(header)])
    exact = {}
    for i, n in enumerate(names):
        exact.setdefault(n, i)
    stripped = {}
    for i, n in enumerate(names):
        stripped.setdefault(n.strip(), []).append(i)
    got, missing = {}, []
    for w in wanted:
        if w in exact:
            got[w] = exact[w]
        elif len(stripped.get(w.strip(), [])) == 1:
            got[w] = stripped[w.strip()][0]
        else:
            missing.append(w)
    return got, missing


REPLACE_DECODE = "utf-8/replace"         # the tool's own decode when the meta's column names hold U+FFFD


def _text(factory, encoding):
    if encoding == REPLACE_DECODE:
        return io.TextIOWrapper(factory(), encoding="utf-8", errors="replace", newline="")
    return io.TextIOWrapper(factory(), encoding=encoding, errors="strict", newline="")


def _locate_header(factory, encoding, delims, wanted):
    best = None
    for di, d in enumerate(delims):
        stream = _text(factory, encoding)
        try:
            reader = csv.reader(stream, delimiter=d)
            for line in range(HEADER_SCAN_RECORDS):
                try:
                    rec = next(reader)
                except StopIteration:
                    break
                if not rec:
                    continue
                got, missing = _resolve(rec, wanted)
                key = (len(got), -di, -line)
                if best is None or key > best[0]:
                    best = (key, d, line, got, missing, len(rec))
                if not missing:
                    break
        finally:
            stream.close()
        if best is not None and not best[4] and best[1] == d:
            break
    return best


_NUM_CLEAN = re.compile(r"[\s,$%]")


def _to_float(s: str) -> float:
    """Numeric cell as the acquisition tool cleaned it: a float literal (no '_' digit separators), else the text with
    every comma, whitespace, '$' and '%' removed ('$30,000' -> 30000, '40, 50' -> 4050, '70.2%' -> 70.2: observed in
    T1 against the raw text); non-finite or anything else is missing (pd.to_numeric(errors='coerce'))."""
    if "_" in s:
        return math.nan
    try:
        v = float(s)
    except ValueError:
        t = _NUM_CLEAN.sub("", s)
        if not t or t == s:
            return math.nan
        try:
            v = float(t)
        except ValueError:
            return math.nan
    return v if math.isfinite(v) else math.nan


# --- datetime -> days since 1970-01-01: element-wise, after pandas' to_datetime string path (string_to_dts ISO 8601,
# does_string_look_like_time -> today's date, _parse_delimited_date, _parse_dateabbr_string, dateutil with default
# 0001-01-01 and dayfirst=yearfirst=False). Out-of-range (pandas ns bounds) or unparseable text is missing (NaT).
_FLOATLIKE = re.compile(r"^\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\s*$")
_ISO = re.compile(r"^(\d{4})(?:([-/.\\ ]?)(\d{1,2})(?:\2(\d{1,2})(?:[T ](\d{1,2})(?::?(\d{2})(?::?(\d{2})(?:[.,](\d{1,9}))?)?)?)?"
                  r"\s*(Z|[+-]\d{2}(?::?\d{2})?)?)?)?$")
_ISO_BASIC = re.compile(r"^(\d{4})(\d{2})(\d{2})(?:T(\d{2})(?::?(\d{2})(?::?(\d{2})(?:\.(\d{1,9}))?)?)?)?(Z|[+-]\d{2}(?::?\d{2})?)?$")
_TIME = re.compile(r"(?<![\d.:])(\d{1,2}):(\d{2})(?::(\d{2})(?:[.,](\d{1,9}))?)?(?![\d:])")
_AMPM = re.compile(r"(?<![A-Za-z])([AaPp])\.?[Mm]\.?(?![A-Za-z])")
_TZ_NUM = re.compile(r"(?:^|(?<=[\s\d]))([+-])(\d{2})(?::?(\d{2}))?\s*$")
_MONTHS = {m: i + 1 for i, ms in enumerate([("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"),
                                             ("may",), ("jun", "june"), ("jul", "july"), ("aug", "august"),
                                             ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
                                             ("dec", "december")]) for m in ms}
_WEEKDAYS = {"mon", "monday", "tue", "tues", "tuesday", "wed", "wednesday", "thu", "thur", "thurs", "thursday", "fri", "friday",
             "sat", "saturday", "sun", "sunday"}
_JUMP = {"at", "on", "and", "ad", "m", "t", "of", "st", "nd", "rd", "th"}
_UTC_NAMES = {"z", "utc", "gmt"}
_NS_LO, _NS_HI = _dt.datetime(1677, 9, 22, tzinfo=_dt.timezone.utc), _dt.datetime(2262, 4, 11, tzinfo=_dt.timezone.utc)


def _days(y, mo, d, h=0, mi=0, s=0, frac=None, off_s=0) -> float:
    try:
        t = _dt.datetime(int(y), int(mo), int(d), int(h), int(mi), int(s), tzinfo=_dt.timezone.utc)
    except ValueError:
        return math.nan
    if not _NS_LO <= t <= _NS_HI:
        return math.nan
    delta = t - EPOCH
    ns = ((delta.days * 86400 + delta.seconds - off_s) * 1_000_000_000 + int((frac or "0")[:9].ljust(9, "0")))
    return ns / 86_400_000_000_000


def _tz_seconds(tz):
    if not tz or tz.upper() in ("Z", "UTC", "GMT"):
        return 0
    sign = -1 if tz[0] == "-" else 1
    digits = tz[1:].replace(":", "")
    return sign * (int(digits[:2]) * 3600 + int(digits[2:4] or 0) * 60)


def _iso(s):
    m = _ISO.match(s)
    if m:
        y, _sep, mo, d, hh, mi, ss, frac, tz = m.groups()
        if hh is not None and d is None:
            return None
        return _days(y, mo or 1, d or 1, hh or 0, mi or 0, ss or 0, frac, _tz_seconds(tz))
    m = _ISO_BASIC.match(s)
    if m:
        y, mo, d, hh, mi, ss, frac, tz = m.groups()
        return _days(y, mo, d, hh or 0, mi or 0, ss or 0, frac, _tz_seconds(tz))
    return None


def _delimited(s):
    n = len(s)
    if n == 10 and s[2] in " /-." and s[5] in " /-.":
        a, b, y = s[0:2], s[3:5], s[6:10]
    elif n == 9 and s[1] in " /-." and s[4] in " /-.":
        a, b, y = s[0:1], s[2:4], s[5:9]
    elif n == 9 and s[2] in " /-." and s[4] in " /-.":
        a, b, y = s[0:2], s[3:4], s[5:9]
    elif n == 8 and s[1] in " /-." and s[3] in " /-.":
        a, b, y = s[0:1], s[2:3], s[4:8]
    elif n == 7 and s[2] in " /-":
        if not (s[:2].isdigit() and s[3:].isdigit()):
            return None
        mo, y = int(s[:2]), int(s[3:])
        return _days(y, mo, 1) if y >= 1000 and 1 <= mo <= 12 else None
    else:
        return None
    if not (a.isdigit() and b.isdigit() and y.isdigit()):
        return None
    mo, d, yr = int(a), int(b), int(y)
    if yr < 1000:
        return None
    if 1 <= mo <= 31 and 1 <= d <= 31 and (mo <= 12 or d <= 12):
        if mo > 12:
            mo, d = d, mo
        return _days(yr, mo, d)
    return math.nan


def _dateabbr(s):
    u = s.upper()
    n = len(u)
    if n == 4 and u.isdigit():
        return _days(int(u), 1, 1)
    if 4 <= n <= 7 and "Q" in u[1:6]:
        i = u.index("Q", 1, 6)
        try:
            if i == 1:
                q = int(u[0])
                if n == 4 or (n == 5 and u[i + 1] == "-"):
                    y = 2000 + int(u[-2:])
                elif n == 6 or (n == 7 and u[i + 1] == "-"):
                    y = int(u[-4:])
                else:
                    return None
            elif i in (2, 3):
                if n == 4 or (n == 5 and u[i - 1] == "-"):
                    q, y = int(u[-1]), 2000 + int(u[:2])
                else:
                    return None
            elif i in (4, 5):
                if n == 6 or (n == 7 and u[i - 1] == "-"):
                    q, y = int(u[-1]), int(u[:4])
                else:
                    return None
            else:
                return None
        except ValueError:
            return None
        return _days(y, (q - 1) * 3 + 1, 1) if 1 <= q <= 4 else math.nan
    for fmt in ("%Y-%m", "%b %Y", "%b-%Y"):
        try:
            t = _dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
        return _days(t.year, t.month, 1)
    return None


def _dateutil(s, default_ymd, ref_year):
    """dateutil.parser._parse + resolve_ymd (dayfirst=yearfirst=False), missing fields from default_ymd."""
    off = 0
    h = mi = sec = 0
    frac = None
    rest = s
    tm = _TIME.search(rest)
    if tm:
        h, mi = int(tm.group(1)), int(tm.group(2))
        sec = int(tm.group(3) or 0)
        frac = tm.group(4)
        head, tail = rest[:tm.start()], rest[tm.end():]
        ap = _AMPM.search(tail)
        if ap:
            if not 0 < h <= 12:
                return math.nan
            h = (h % 12) + (12 if ap.group(1) in "Pp" else 0)
            tail = tail[:ap.start()] + " " + tail[ap.end():]
        tzm = _TZ_NUM.search(tail)                # an offset only after the time
        if tzm:
            off = _tz_seconds(tzm.group(1) + tzm.group(2) + (tzm.group(3) or ""))
            tail = tail[:tzm.start()]
        rest = head + " " + tail
    ymd, ystr, mstr = [], None, None
    for tok in re.findall(r"\d+|[A-Za-z]+|[^\sA-Za-z\d,;/\-.']", rest):
        if tok.isdigit():
            if len(tok) > 4:
                return math.nan
            if len(tok) >= 3:                     # dateutil: a year written with 3+ digits (century specified)
                if ystr is not None:
                    return math.nan
                ystr = len(ymd)
            ymd.append(int(tok))
        elif tok.isalpha():
            low = tok.lower()
            if low in _MONTHS:
                if mstr is not None:
                    return math.nan
                mstr = len(ymd)
                ymd.append(_MONTHS[low])
            elif low in _WEEKDAYS or low in _JUMP:
                continue
            elif low in _UTC_NAMES and tm:
                off = 0
            elif tm and tok.isupper() and len(tok) <= 5:
                continue                      # unknown tz name: dateutil leaves the time naive
            else:
                return math.nan
        else:
            return math.nan
    n = len(ymd)
    if n > 3 or (n == 0 and not tm):
        return math.nan
    yi = mi_ = di = None                      # indices into ymd (dateutil resolve_ymd, dayfirst=yearfirst=False)
    strids = {k: v for k, v in (("y", ystr), ("m", mstr)) if v is not None}
    if n and (n == len(strids) or (n == 3 and len(strids) == 2)):
        if n == 3 and len(strids) == 2:
            strids[[k for k in "ymd" if k not in strids][0]] = [x for x in range(3) if x not in strids.values()][0]
        yi, mi_, di = strids.get("y"), strids.get("m"), strids.get("d")
    elif n == 1 or (mstr is not None and n == 2):
        if mstr is not None:
            mi_ = mstr
            other = (mstr - 1) % n
        else:
            other = 0
        if n > 1 or mstr is None:
            if ymd[other] > 31:
                yi = other
            else:
                di = other
    elif n == 2:
        if ymd[0] > 31:
            yi, mi_ = 0, 1
        elif ymd[1] > 31:
            mi_, yi = 0, 1
        else:
            mi_, di = 0, 1
    elif n == 3:
        a, b, c = ymd
        if mstr == 0:
            mi_, yi, di = (0, 1, 2) if b > 31 else (0, 2, 1)
        elif mstr == 1:
            yi, mi_, di = (0, 1, 2) if a > 31 else (2, 1, 0)
        elif mstr == 2:
            di, yi, mi_ = (0, 1, 2) if b > 31 else (1, 0, 2)
        elif a > 31 or ystr == 0:
            yi, mi_, di = (0, 2, 1) if b > 12 else (0, 1, 2)
        elif a > 12:
            di, mi_, yi = 0, 1, 2
        else:
            mi_, di, yi = 0, 1, 2
    year = ymd[yi] if yi is not None else None
    month = ymd[mi_] if mi_ is not None else None
    day = ymd[di] if di is not None else None
    if year is not None and year < 100 and yi != ystr:
        year += (ref_year // 100) * 100
        if year >= ref_year + 50:
            year -= 100
        elif year < ref_year - 50:
            year += 100
    y0, m0, d0 = default_ymd
    return _days(year if year is not None else y0, month if month is not None else m0, day if day is not None else d0,
                 h, mi, sec, frac, off)


def datetime_days(s: str, today=(1970, 1, 1), ref_year: int = 2026) -> float:
    """Days since 1970-01-01 of one datetime cell; a timezone offset is honoured (UTC instant), a naive time is UTC.
    `today` is the date a time-only cell takes (pandas uses the current date: the meta's acquisition date)."""
    if s in NA_STRINGS:
        return math.nan
    t = s.strip()
    if not t:
        return math.nan
    if _FLOATLIKE.match(t):
        try:
            if float(t) < 1000:
                return math.nan                    # pandas: a number below 1000 does not look like a datetime
        except ValueError:
            pass
    v = _iso(t)
    if v is not None:
        return v
    if len(t) >= 4 and ((t[1] == ":" and t[0].isdigit() and t[2:4].isdigit()) or
                        (len(t) >= 5 and t[2] == ":" and t[:2].isdigit() and t[3:5].isdigit())):
        hh = int(t[:t.index(":")])
        mm = int(t[t.index(":") + 1:t.index(":") + 3])
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return _dateutil(t, today, ref_year)
    v = _delimited(t)
    if v is not None:
        return v
    v = _dateabbr(t)
    if v is not None:
        return v
    return _dateutil(t, (1, 1, 1), ref_year)


def _category_lookup(spec: Mapping[str, Any]):
    """raw cell text -> meta code: exact key, unique stripped key, pandas bool text, then numeric equality of keys
    (a float-typed column prints 3400 as '3400.0'). Missing (pandas NA text) -> MISSING_CODE; any other text that
    names no key -> UNMAPPED_CODE (such a raw row cannot match a T1 row)."""
    inv = spec["inverse"]
    key_code = {k: c for c, k in inv.items()}
    strip_code = {}
    for k, c in key_code.items():
        strip_code.setdefault(k.strip(), []).append(c)
    num_code, num_dup = {}, set()
    for k, c in key_code.items():
        v = t1prep._num(k)
        if v is not None:
            if v in num_code:
                num_dup.add(v)
            num_code[v] = c
    for v in num_dup:
        del num_code[v]

    def lookup(s: str) -> int:
        if s in key_code:
            return key_code[s]
        if s in NA_STRINGS or s.strip() in NA_STRINGS:
            return MISSING_CODE
        cs = strip_code.get(s.strip())
        if cs is not None and len(cs) == 1:
            return cs[0]
        if s in BOOL_TRUE and "True" in key_code:
            return key_code["True"]
        if s in BOOL_FALSE and "False" in key_code:
            return key_code["False"]
        v = _to_float(s)
        if not math.isnan(v) and v in num_code:
            return num_code[v]
        return UNMAPPED_CODE
    return lookup


def read_source(path, meta: Mapping[str, Any], specs: Sequence[Mapping[str, Any]]) -> dict:
    """Every data row of the raw source, reconstructed per spec: values (n, C) float64 with NaN for a missing numeric
    cell and the meta code (MISSING_CODE / UNMAPPED_CODE) for a categorical cell, plus the parse report."""
    path = Path(path)
    hint = _container(meta)
    factory, layers = _opener(path, hint["member"])
    with factory() as s:
        head = s.read(65536)
    lead = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    if lead[:1] == b"<":
        raise T1RawError("markup_not_table", lead[:60].decode("latin-1"))
    if lead[:1] in (b"{", b"["):
        raise T1RawError("json_not_supported", lead[:60].decode("latin-1"))
    wanted = []
    for sp in specs:
        if sp["source_name"] not in wanted:
            wanted.append(sp["source_name"])
    delims = ([hint["sep"]] if hint["sep"] else []) + [d for d in DELIMITERS if d != hint["sep"]]
    encs = ([hint["encoding"]] if hint["encoding"] else []) + [e for e in FALLBACK_ENCODINGS if e != hint["encoding"]]
    if any("\ufffd" in w for w in wanted):          # the tool decoded with replacement characters: do the same, first
        encs = [REPLACE_DECODE] + encs
    last_err = None
    csv.field_size_limit(2 ** 31 - 1)
    for enc in encs:
        try:
            return _read_with(factory, enc, delims, wanted, specs, layers, hint)
        except UnicodeDecodeError as e:
            last_err = f"{enc}: {e}"[:200]
        except LookupError as e:
            last_err = f"{enc}: {e}"[:200]
    raise T1RawError("undecodable", last_err or "")


def _read_with(factory, enc, delims, wanted, specs, layers, hint):
    best = _locate_header(factory, enc, delims, wanted)
    if best is None:
        raise T1RawError("empty_source")
    _, delim, hline, got, missing, hwidth = best
    if missing:
        raise T1RawError("header_missing_columns", f"{len(missing)}/{len(wanted)} missing, e.g. {missing[:4]}")
    C = len(specs)
    stream = _text(factory, enc)
    try:
        reader = csv.reader(stream, delimiter=delim)
        for _ in range(hline + 1):
            next(reader)
        # pandas: when every data row has exactly one field more than the header, the first field is an implicit index
        probe, lens = [], {}
        for rec in reader:
            probe.append(rec)
            if rec and rec != [""]:
                lens[len(rec)] = lens.get(len(rec), 0) + 1
            if len(probe) >= INDEX_PROBE_RECORDS:
                break
        implicit_index = bool(lens) and set(lens) == {hwidth + 1}
        shift = 1 if implicit_index else 0
        width = hwidth + shift
        idx = [got[sp["source_name"]] + shift for sp in specs]
        kinds = []
        cat_cache = []
        for sp in specs:
            if sp["kind"] == "categorical_codes":
                kinds.append(2)
            elif sp.get("derive") == "datetime_days":
                kinds.append(1)
            else:
                kinds.append(0)
            cat_cache.append({})
        cols = [array.array("d") for _ in range(C)]
        nonnum = [dict() for _ in range(C)]          # numeric column: non-numeric, non-NA text -> count
        bool_rows = [(array.array("q"), array.array("q")) for _ in range(C)]   # numeric column: rows of true / false text
        lookups = [(_category_lookup(sp) if k == 2 else None) for sp, k in zip(specs, kinds)]
        dt_cache = [dict() if k == 1 else None for k in kinds]
        today = [sp.get("today", (1970, 1, 1)) for sp in specs]
        ref_year = [sp.get("ref_year", 2026) for sp in specs]
        nrows = blank = long_rows = short_rows = 0

        def rows():
            yield from probe
            yield from reader
        for rec in rows():
            n = len(rec)
            if n == 0 or (n == 1 and rec[0] == ""):
                blank += 1
                continue
            if n > width:
                long_rows += 1
                continue
            if n < width:
                short_rows += 1
            for c in range(C):
                i = idx[c]
                s = rec[i] if i < n else ""
                k = kinds[c]
                if k == 0:
                    v = _to_float(s)
                    if v != v and s not in NA_STRINGS and s.strip() not in NA_STRINGS:
                        d = nonnum[c]
                        if len(d) < 1000 or s in d:
                            d[s] = d.get(s, 0) + 1
                        if s in BOOL_TRUE:
                            bool_rows[c][0].append(nrows)
                        elif s in BOOL_FALSE:
                            bool_rows[c][1].append(nrows)
                    cols[c].append(v)
                elif k == 1:
                    cache = dt_cache[c]
                    v = cache.get(s)
                    if v is None:
                        v = datetime_days(s, today[c], ref_year[c])
                        if len(cache) < 2_000_000:
                            cache[s] = v
                    cols[c].append(v)
                else:
                    cache = cat_cache[c]
                    code = cache.get(s)
                    if code is None:
                        code = lookups[c](s)
                        cache[s] = code
                    cols[c].append(float(code))
            nrows += 1
    finally:
        stream.close()
    values = np.empty((nrows, C), dtype=np.float64)
    bool_cols = []
    for c in range(C):
        values[:, c] = np.frombuffer(cols[c], dtype=np.float64) if nrows else np.empty(0)
        if kinds[c] == 0 and nonnum[c] and set(nonnum[c]) <= (BOOL_TRUE | BOOL_FALSE) and not np.any(~np.isnan(values[:, c])):
            # pandas: a column whose every non-missing cell is bool text is read as bool; to_numeric gives 1.0 / 0.0
            values[np.frombuffer(bool_rows[c][0], dtype=np.int64), c] = 1.0
            values[np.frombuffer(bool_rows[c][1], dtype=np.int64), c] = 0.0
            bool_cols.append(specs[c]["name"])
    report = {"layers": layers, "encoding": enc, "delimiter": delim, "header_record": hline, "header_width": hwidth,
              "implicit_index": implicit_index, "data_rows": nrows, "blank_rows": blank, "long_rows_skipped": long_rows,
              "short_rows_padded": short_rows, "declared_container": hint["declared"], "bool_columns": bool_cols,
              "nonnumeric_cells": {specs[c]["name"]: int(sum(nonnum[c].values())) for c in range(C) if nonnum[c]},
              "nonnumeric_examples": {specs[c]["name"]: sorted(nonnum[c], key=lambda t: -nonnum[c][t])[:3]
                                      for c in range(C) if nonnum[c]},
              "unmapped_category_cells": {specs[c]["name"]: int(np.sum(values[:, c] == UNMAPPED_CODE))
                                          for c in range(C) if kinds[c] == 2 and np.any(values[:, c] == UNMAPPED_CODE)}}
    return {"values": values, "report": report}


# ---------------------------------------------------------------------------------------------- matching

def t1_cells(A, specs) -> tuple[np.ndarray, np.ndarray]:
    """De-scaled T1 values (numeric columns) and decoded codes (categorical columns; refused when not integral)."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[1] != len(specs):
        raise T1RawError("t1_width", f"A has {A.shape} for {len(specs)} columns")
    d = np.empty_like(A)
    for s in specs:
        j = s["index"]
        span = s["hi"] - s["lo"]
        col = A[:, j] * span + s["lo"] if span != 0 else np.full(A.shape[0], s["lo"])
        if s["kind"] == "categorical_codes":
            code = np.rint(col)
            if bool(np.any(np.abs(col - code) > t1prep.CODE_TOL)):
                raise T1RawError("t1_code_not_integral", s["name"])
            col = code
        d[:, j] = col
    return d, A


def _canonical(values: np.ndarray) -> np.ndarray:
    v = np.ascontiguousarray(values, dtype=np.float64).copy()
    v[np.isnan(v)] = np.nan
    v = v + 0.0                                            # -0.0 -> 0.0
    return v.view(np.uint64)


def match(A, specs, raw_values, tol_rel: float = MATCH_TOL_REL) -> dict:
    """One value class of raw rows per row of A, or the reasons there is none. Returns assignment (A row -> class),
    class values and the statistics. `matched` is True only when every row is assigned uniquely and one-to-one."""
    d, _ = t1_cells(A, specs)
    nA, C = d.shape
    raw = np.asarray(raw_values, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != C:
        raise T1RawError("raw_width", f"{raw.shape} for {C} columns")
    tgt = C - 1
    keep = ~np.isnan(raw[:, tgt])                           # a row without a target is not a T1 row
    u64, inverse, counts = np.unique(_canonical(raw[keep]), axis=0, return_inverse=True, return_counts=True)
    U = u64.view(np.float64)
    order = np.argsort(U[:, tgt], kind="mergesort")
    t_sorted = U[order, tgt]
    dec9 = [s["kind"] == "numeric" and printed_decimal9(np.asarray(A, dtype=np.float64)[:, s["index"]]) for s in specs]
    tols = np.column_stack([tolerance(s, d[:, s["index"]], tol_rel, dec9[s["index"]]) if s["kind"] == "numeric" else np.zeros(nA)
                            for s in specs])
    imp_ok = np.zeros((nA, C), dtype=bool)
    for s in specs:
        j = s["index"]
        if s["kind"] == "numeric" and s.get("impute_value") is not None:
            imp_ok[:, j] = np.abs(s["impute_value"] - d[:, j]) <= tols[:, j]
        elif s["kind"] == "categorical_codes":
            imp_ok[:, j] = d[:, j] == s["impute_code"]
    feat_order = [s["index"] for s in specs[:-1]]
    assign = np.full(nA, -1, dtype=np.int64)
    status = np.zeros(nA, dtype=np.int8)            # 0 matched, 1 no candidate, 2 ambiguous
    blame: dict[str, int] = {}
    diag_rows = 0
    missing_matched = 0
    for i in range(nA):
        a = np.searchsorted(t_sorted, d[i, tgt] - tols[i, tgt], side="left")
        b = np.searchsorted(t_sorted, d[i, tgt] + tols[i, tgt], side="right")
        cand = order[a:b]
        window = cand
        for j in feat_order:
            if cand.size == 0:
                break
            cand = cand[_cell_ok(U[cand, j], specs[j], d[i, j], tols[i, j], imp_ok[i, j])]
        if cand.size == 1:
            assign[i] = cand[0]
            missing_matched += int(np.sum(np.isnan(U[cand[0], :]) | (U[cand[0], :] == MISSING_CODE) & _cat_mask(specs)))
        elif cand.size == 0:
            status[i] = 1
            if diag_rows < DIAG_ROWS:
                diag_rows += 1
                for name in _blame(U, window, specs, d[i], tols[i], imp_ok[i], feat_order):
                    blame[name] = blame.get(name, 0) + 1
        else:
            status[i] = 2
    claimed = np.bincount(assign[assign >= 0], minlength=U.shape[0]) if U.shape[0] else np.zeros(0, dtype=np.int64)
    over = int(np.sum(np.maximum(claimed - counts, 0))) if U.shape[0] else 0
    unmatched, ambiguous = int(np.sum(status == 1)), int(np.sum(status == 2))
    ok = nA > 0 and unmatched == 0 and ambiguous == 0 and over == 0
    resid = _residuals(U, assign, specs, d, tols) if np.any(assign >= 0) else {}
    stats = {"schema": SCHEMA, "rows": int(nA), "matched_rows": int(np.sum(assign >= 0)), "unmatched_rows": unmatched,
             "ambiguous_rows": ambiguous, "oversubscribed_rows": over, "raw_rows": int(raw.shape[0]),
             "raw_rows_without_target": int(np.sum(~keep)), "raw_value_classes": int(U.shape[0]),
             "duplicate_classes_claimed": int(np.sum((claimed > 0) & (counts > 1))) if U.shape[0] else 0,
             "matched_missing_cells": missing_matched, "tol_rel": tol_rel,
             "print_class": {"decimal9": sum(dec9), "float32": sum(1 for s in specs if s["kind"] == "numeric") - sum(dec9)},
             "blame_columns_first_unmatched": dict(sorted(blame.items(), key=lambda kv: -kv[1])[:20]),
             "diag_rows": diag_rows, "residual": resid, "matched": ok}
    return {"matched": ok, "assign": assign, "classes": U, "class_counts": counts, "stats": stats}


def _cat_mask(specs):
    return np.asarray([s["kind"] == "categorical_codes" for s in specs])


def _cell_ok(x, spec, dij, tol, imp_ok):
    if spec["kind"] == "categorical_codes":
        return (x == dij) | ((x == MISSING_CODE) & imp_ok)
    miss = np.isnan(x)
    return np.where(miss, imp_ok, np.abs(x - dij) <= tol)


def _blame(U, window, specs, di, toli, imp_oki, feat_order):
    if window.size == 0:
        return ["target"]
    w = window[:200_000]
    fails = np.zeros((w.size, len(feat_order)), dtype=bool)
    for k, j in enumerate(feat_order):
        fails[:, k] = ~_cell_ok(U[w, j], specs[j], di[j], toli[j], imp_oki[j])
    best = int(np.argmin(fails.sum(axis=1)))
    return [specs[feat_order[k]]["name"] for k in np.nonzero(fails[best])[0]]


def _residuals(U, assign, specs, d, tols) -> dict:
    """max |raw - de-scaled T1| / tolerance over matched non-missing numeric cells (diagnostic: <= 1 by construction)."""
    rows = np.nonzero(assign >= 0)[0]
    worst, worst_rel_span = 0.0, 0.0
    for s in specs:
        if s["kind"] != "numeric":
            continue
        j = s["index"]
        x = U[assign[rows], j]
        m = ~np.isnan(x)
        if not np.any(m):
            continue
        dev = np.abs(x[m] - d[rows[m], j])
        tol = tols[rows[m], j]
        worst = max(worst, float(np.max(dev / np.maximum(tol, 1e-300))))
        span = abs(s["hi"] - s["lo"])
        if span > 0:
            worst_rel_span = max(worst_rel_span, float(np.max(dev)) / span)
    return {"max_dev_over_tol": worst, "max_dev_over_span": worst_rel_span}


# ---------------------------------------------------------------------------------------------- L7a values

def matched_values(m: Mapping[str, Any]) -> np.ndarray:
    """Reconstructed raw values of every row of A (NaN = missing numeric, MISSING_CODE = missing category)."""
    if not m["matched"]:
        raise T1RawError("not_matched")
    return np.ascontiguousarray(m["classes"][m["assign"]])


def refit(values, specs: Sequence[Mapping[str, Any]], outer_train: Sequence[int]) -> tuple[np.ndarray, dict]:
    """Reconstructed raw values -> fitted matrix from outer-train rows only: numeric missing cells get the median of the
    column's non-missing outer-train values (0.0 when it has none, counted), categories are re-coded by t1prep's
    train-only map, target copies are found on outer-train rows. Returns (matrix, report)."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != len(specs):
        raise T1RawError("values_width", f"{v.shape} for {len(specs)} columns")
    n = v.shape[0]
    tr = [int(i) for i in outer_train]
    tr_set = set(tr)
    if len(tr_set) != len(tr) or (tr and (min(tr) < 0 or max(tr) >= n)):
        raise ValueError("outer-train indices must be unique and in range")
    te = [i for i in range(n) if i not in tr_set]
    out = np.empty_like(v)
    num_rep, cat_rep, no_train_value = {}, {}, []
    imputed = {"train": 0, "test": 0}
    for s in specs[:-1]:
        j = s["index"]
        col = v[:, j]
        if s["kind"] == "categorical_codes":
            codes_raw = col.astype(np.int64)
            if bool(np.any((codes_raw < MISSING_CODE) | ((codes_raw >= 0) & ~np.isin(codes_raw, list(s["inverse"]))))):
                raise T1RawError("values_bad_code", s["name"])
            keys = [None if c == MISSING_CODE else s["inverse"][int(c)] for c in codes_raw.tolist()]
            codes = t1prep.fit_category_codes([keys[i] for i in tr])
            out[:, j] = t1prep.apply_category_codes(keys, codes)
            cat_rep[s["name"]] = {
                "train": {"categories": len(codes), "missing": sum(1 for i in tr if keys[i] is None),
                          "map_sha256": hashlib.sha256(repr(sorted(codes.items())).encode()).hexdigest()},
                "test": {"missing": sum(1 for i in te if keys[i] is None),
                         "unseen": sum(1 for i in te if keys[i] is not None and keys[i] not in codes)}}
        else:
            miss = np.isnan(col)
            train_vals = col[tr][~miss[tr]]
            if train_vals.size:
                med = float(np.median(train_vals))
            else:
                med = 0.0
                no_train_value.append(s["name"])
            out[:, j] = np.where(miss, med, col)
            mt, ms = int(np.sum(miss[tr])), int(np.sum(miss[te])) if te else 0
            imputed["train"] += mt
            imputed["test"] += ms
            if mt or ms:
                num_rep[s["name"]] = {"train_missing": mt, "test_missing": ms, "train_median": med.hex()}
    y = v[:, -1]
    if bool(np.any(np.isnan(y))):
        raise T1RawError("values_missing_target")
    out[:, -1] = y
    drop = t1prep.target_copy_columns(out[tr])
    keep = [j for j in range(len(specs) - 1) if j not in set(drop)] + [len(specs) - 1]
    report = {"schema": SCHEMA, "values_mode": "L7a",
              "imputation": {"policy": "median_of_nonmissing_outer_train_values_per_column; outer-test rows get the train median",
                             "imputed_cells": imputed, "columns": num_rep, "no_train_value_columns": no_train_value},
              "categorical": cat_rep, "target_copy_dropped": [specs[j]["name"] for j in drop], "kept_columns": keep[:-1]}
    return np.ascontiguousarray(out[:, keep]), report


def prepare(values, specs, outer_train: Sequence[int], outer_test: Sequence[int], **kw) -> dict:
    """refit() then trainonly.prepare_split(); the report rides along as split['t1raw']."""
    fitted, report = refit(values, specs, outer_train)
    split = trainonly.prepare_split(fitted, outer_train, outer_test, **kw)
    split["t1raw"] = report
    split["fitted"] = fitted
    return split


# ---------------------------------------------------------------------------------------------- one dataset

def match_dataset(raw_path, meta: Mapping[str, Any], A, tol_rel: float = MATCH_TOL_REL) -> dict:
    """read_source + match for one dataset. Never raises for a data problem: returns status 'matched' or 'unmatched'
    with a reason code and the statistics (the caller falls back to L7b)."""
    A = np.asarray(A, dtype=np.float64)
    try:
        specs = columns(meta, A.shape[1])
        src = read_source(raw_path, meta, specs)
        m = match(A, specs, src["values"], tol_rel)
    except T1RawError as e:
        return {"status": "unmatched", "reason": e.reason, "detail": e.detail[:300], "stats": None, "read": None}
    st = m["stats"]
    if m["matched"]:
        return {"status": "matched", "reason": None, "detail": "", "stats": st, "read": src["report"], "specs": specs,
                "values": matched_values(m)}
    if st["unmatched_rows"]:
        nonnum = set(src["report"]["nonnumeric_cells"]) & set(st["blame_columns_first_unmatched"])
        reason = "rows_unmatched_nonnumeric_raw" if nonnum else (
            "rows_unmatched_target" if set(st["blame_columns_first_unmatched"]) == {"target"} else "rows_unmatched")
    elif st["ambiguous_rows"]:
        reason = "rows_ambiguous"
    else:
        reason = "rows_oversubscribed"
    return {"status": "unmatched", "reason": reason, "detail": "", "stats": st, "read": src["report"], "specs": specs}
