# JopeDime matrix format, version 1

SVC-11 defines this initial format for spec §4.5 and §5.3. Registration is
target-independent: every source column is retained, in header order. Selecting
features, targets, tasks and FIT/HOLDOUT/TEST rosters happens later. No fitted
preprocessing or classification-label encoding is learned during registration.

All integers and floats are little-endian. Offsets are absolute file offsets.
There is no native Rust struct layout, pointer, platform-sized integer or padding
whose contents are unspecified. Readers must validate the complete file before
returning cells. Version changes require a new reader contract and review.

## File layout

| Offset | Bytes | Meaning |
| --- | --- | --- |
| 0 | 8 | Magic `JMAT\r\n\x1a\n` |
| 8 | 8 | Format version, exactly 1 |
| 16 | 8 | Total file length, including checksum |
| 24 | 8 | JSON metadata length |
| 32 | 8 | Number of rows |
| 40 | 8 | Number of columns |
| 48 | 8 | Row-ID vector offset |
| 56 | 8 | Reserved, exactly zero |
| 64 | variable | UTF-8 JSON metadata |
| next multiple of 64 | `rows * 8` | Row IDs: u64 ordinals `0..rows` |
| aligned blocks | variable | Each column's values, then presence bytes |
| final 32 | 32 | SHA-256 of every preceding byte |

Every vector starts on a 64-byte boundary. Padding is zero. There is no trailing
content after the checksum. Metadata is the serialized `Metadata` struct:
`{"columns":[{"name":...,"kind":"numeric"|"categorical","dictionary":[...],"values":offset,"present":offset},...]}`.
Writers use this field order and compact JSON. Offset resolution is deterministic;
metadata is padded to a fixed reserved area before vectors are written. The
reserved metadata area is its serialized length plus 1024 bytes per column,
rounded up to 64 bytes. This headroom accommodates all decimal u64 offsets without
a self-referential variable-length layout.

## Column encoding and exactness

* Numeric: every nonmissing source cell parses, after trimming, as a finite f64.
  Values use IEEE-754 binary64, preserving the sign of zero. Missing slots have
  all-zero value bytes. The dictionary is empty.
* Categorical: nonmissing strings are preserved exactly, including whitespace.
  The dictionary contains unique strings in UTF-8 byte order. Values are u32
  indices into that dictionary. Missing slots contain zero and are distinguished
  from dictionary index zero by presence bytes.
* Presence: exactly one byte per row, either zero (missing) or one (present).
  Empty/ASCII-whitespace-only cells are missing, matching runtime table ingress;
  literal `NA`, `null` and `?` remain values. All-missing columns are numeric.

Round-trip exactness means header names/order, column kinds, dictionaries, row IDs,
missingness, categorical strings and numeric f64 bits. Original CSV bytes are a
separate immutable CAS blob: quoting, numeric lexemes, line endings and BOM remain
recoverable there. No byte-level CSV reconstruction is claimed from a matrix.

## Validation, limits and publication

Readers reject unknown versions/fields, inconsistent counts/lengths, overlapping or
unaligned vectors, nonzero padding/reserved fields, invalid row IDs, invalid UTF-8,
unsorted/duplicate dictionaries, invalid category codes, nonfinite numbers,
invalid presence bytes and checksum mismatch. Checked arithmetic precedes slicing
and allocation. Numeric reads copy eight bytes and use `from_le_bytes`; they do
not cast potentially unaligned pointers.

The source ID is `ds_` followed by the first 24 lowercase hex digits of SHA-256 of
the complete original CSV. Metadata retains the full digest to detect prefix
collisions. CSV and matrix are published through the SVC-01 store before one
dataset metadata transaction records both blob references. Failed registration
may leave unreferenced CAS blobs; it never leaves a dataset referring to missing
content. Ordinary store GC owns orphan reclamation.

`RegistrationLimits` explicitly bounds dataset bytes (default 4 GiB), rows
(2,000,000), columns (10,000), decoded cell bytes (1 MiB), and categorical
cardinality per column (65,536). Every violated policy returns `DATASET_INVALID`
with its exact field name. Config overrides are supplied by the engine owner;
this module does not change config or select a target. Invalid CSV diagnostics
never include raw cells or runtime parser messages.

The read-only mapping API requires an immutable regular file that remains
unmodified and untruncated for the entire mapping lifetime. This is an explicit
unsafe caller contract: read-only mapping permissions cannot prevent an external
process from truncating the underlying file. Engine-owned CAS publication and
worker lifetimes must uphold it. Separate-process tests exercise the actual
mapping; loading a byte vector is not reported as a mapped read.
