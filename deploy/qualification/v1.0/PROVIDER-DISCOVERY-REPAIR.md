# Complete M3 provider discovery and input transport

Current merged source cannot discover its own complete provider set: the
28-gate inventory rejects `providers/m3_full.py` as an unregistered provider,
even though `m3.py` imports that tracked helper. Independently, discovery has no
M3 input binding and would invoke the bounded source-control path instead of
the complete four-clause comparison. Both failures were reproduced before this
repair, without a candidate or dataset payload.

Register only `common.py` and `m3_full.py` as known helpers and require regular
non-symlink files. Keep unknown modules, missing gate IDs and unsafe paths as
refusals. Add the closed M3 binding `{full_inputs, full_inputs_sha256}` to the
existing explicit input schema and pass both as argv to the maintained M3 CLI.
Incomplete binding means unbound discovery and refusal before evidence creation;
ambient environment variables cannot substitute the manifest or its pin.

Regression controls preserve the old failing discovery and exercise the real M3
argument parser through the restricted child environment. They also reject an
unknown helper, a helper symlink, bad/relative pins and source/resource overrides.
Existing all-28 completeness tests now require all five external input groups.

Alternatives rejected: ignoring every unrecognized module would hide accidental
providers; allowing partial M3 source controls would overstate parity; routing
full-input bindings through inherited environment would bypass the existing
explicit boundary. No new dependency, gate ID, release schema, signed-candidate
verification, tolerance, proposer algorithm or release-reference change occurs.

This fixes source orchestration under spec §14 and REL-07's all-28 execution
requirement. It does not turn M3's historical proposer FAIL into PASS, invent
missing candidate/platform/named-client evidence, sign bytes or approve release.
Final qualification still requires all actual exact-byte gates and OWNER approval.
