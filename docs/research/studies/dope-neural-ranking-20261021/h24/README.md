# H24 critical completion report

The two catalog metadata locators are now bound to observed non-null S3 version
identifiers. Both independent metadata checks passed. No object body was fetched,
and no new training example, source-equivalence proof or neural lift was shown.

The cache-first route avoided fresh bucket, configuration and object listing.
Five complete, receipt-bound current-key caches supplied 1,166,070 metadata
objects over 1,169 pages and 78,496,603 compressed bytes. All compressed hashes,
lengths, gzip CRC/EOF, pagination tokens, page/object denominators and exact target
role bindings were independently replayed. Each of the two selected full catalog
digests had exactly one key-literal match with the expected size. This was a
locator result, not a content checksum result. The sixth configured source's
partial cache was excluded and remains a gap; no global absence claim is valid.

Two new conditional HEAD requests then returned HTTP 200, zero observed SDK
retries, matching opaque ETags and lengths of 10,524 and 11,508 bytes, and two
non-null version identifiers. Independent saved-response QA verified the exact
request tuples and returned metadata. HEAD did not verify the catalog bodies,
their raw SHA256, JSON schema, row semantics, original acquisition, preprocessing,
GP fitness context or admission. The version observations also do not establish
future retention or durability. ETags were never treated as MD5 or SHA256.

Two implementation failures remain visible. First, a launch preflight found that
the prospective review receipt had not been mirrored; the exact receipt was
copied before any worker/cache/API execution. Second, the same-author saved HEAD
checker failed in its initial source-hash loop because its helper expected a
Path and received a string. It failed before response validation. The successful
HEAD worker and its immutable closure were preserved, the checker was not edited
or rerun, and a separate independent checker passed. The stopped service's failed
terminal state reflects that checker failure; no process remains running.

Measured costs are phase-specific, not a complete preparation-session total:

| Phase | Wall seconds | CPU seconds |
|---|---:|---:|
| Offline lookup worker | 18.691557 | 18.408076 |
| Enclosing lookup launcher, including worker | 18.748055 | 18.460517 child + 0.001299 launcher |
| Independent full five-cache replay | 18.176804 | 18.141308 |
| HEAD worker | 1.377454 | 0.334869 |
| Enclosing HEAD worker subprocess phase | 1.482044 | 0.450205 |
| Failed same-author saved-response checker | 0.110791 | 0.106625 |
| Independent saved HEAD QA | 0.002752 | 0.002426 |
| Twelve offline toy controls | 0.006132 | 0.004143 |
| Seven new HEAD scope/hash-domain toy controls | 0.000651 | 0.000615 |

Worker and enclosing launcher rows overlap and must not be added. The larger
enclosing HEAD CPU figure includes child-process work. Interactive preparation,
metadata inspection, SSH, copying and custody costs are not fully captured by
these measurements. Actual resource receipts retain Ubuntu identity, two CPUs,
8 GiB memory, zero swap, 64 tasks and no compute timeout. The offline lookup and
independent replay used private networking; HEAD alone used the approved SDK
route. No API or compute failure was hidden by a new budget or a reset.

The earlier audit already supplied six dated report-manifest metadata pointers
and 24 Parquet file/footer descriptors accounting for 35,912,671 historical
inventory records. Those are not additional training examples. All 24 full
Parquet payload-hash flags are false; a footer hash does not verify the whole
file. H24 read no Parquet rows or report/data/model/outcome bodies. Configured
all-version coverage and these dated snapshots are distinct from current,
complete delivered inventory and original-source authority.

The next useful step is only the proposed two exact metadata GETs, totaling
22,032 bytes, under a new source-bound allowlist, parser review and root grant.
Their expected raw hashes come from catalog metadata roles, not filename or
ETag guesses. If the bodies resolve a valid direct GP-input semantic route,
that route can guide qualification of the 1,909 epoch-included metadata
potentials. Those potentials still have zero numeric admissions. Native writer
history is an optional mechanism investigation where relevant; it is not a
universal substitute for direct row/cut/transform/fitness authority, nor a
universal barrier when direct semantic proof is available. No new fitting or
body retrieval is authorized by this report.

Exact protocol, result and independent-QA receipt hashes are recorded in the
sanitized aggregate. Private locator, version and expected catalog hash values
remain only in private evidence. Historical adverse neural results are unchanged.
