"""PROVENANCE-01: resolve every code hash recorded by a v2 publish run, binding manifest, GPU encode or build summary to exact
bytes in the content-addressed custody dir (RUN/receipts/trainonly-v2/code-custody/<file>.<sha256>), publish those bytes
create-only under v2-trainonly/provenance/code/, and publish a provenance document under v2-trainonly/provenance/provenance-v*/.
The lane source is git-ignored (not committed), so the bytes themselves are the record until a reviewed PR lands them.

Usage (lane slice, online): v2_provenance.py --name provenance-v2 [--publish]
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
HEX64 = re.compile(r"[0-9a-f]{64}")
C = RUN / "receipts/trainonly-v2/code-custody"
PUB = RUN / "receipts/trainonly-v2/publish"
# documents built by an inline script (no code_sha256 field): manifest name -> sha256 of the exact script bytes in custody
INLINE = {"superseded-raw-split-v1": "cd51465f47ff0aaf0da7bd0d84b82bb5e576f016a71a189d6385121d56710029"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args(argv)
    spec = importlib.util.spec_from_file_location("vp", Path(__file__).resolve().parent / "v2_publish.py")
    vp = importlib.util.module_from_spec(spec); spec.loader.exec_module(vp)
    custody = {}
    for p in C.iterdir():
        name, h = p.name.rsplit(".", 1)
        if sha(p.read_bytes()) != h:
            raise SystemExit(f"custody file {p} does not hash to its name")
        custody[h] = {"file": name, "path": p}

    def resolve(h):
        """Bind a recorded code hash to custody bytes by its FULL sha256 only; a prefix never binds (PROVENANCE-01, #218 finding 9)."""
        if h is None or h == "" or h == {}:
            return {"sha256_prefix": "", "status": "MISSING CODE HASH"}
        if not isinstance(h, str) or not HEX64.fullmatch(h):
            return {"sha256_prefix": str(h), "status": "NOT A FULL SHA256"}
        if h not in custody:
            return {"sha256_prefix": h, "status": "NOT FOUND"}
        return {"sha256": h, "file": custody[h]["file"], "code_key": f"{vp.s3io.V2_PREFIX_KEY}provenance/code/{custody[h]['file']}.{h}"}

    man = []
    for m in sorted(PUB.glob("v2-publish-manifest-*.json")):
        if "-check" in m.name:
            continue
        d = json.loads(m.read_text())
        man.append({"local": m.name, "sha256": sha(m.read_bytes()), "code": resolve(d.get("code_sha256")), "publisher": resolve(d.get("publisher_code_sha256"))})
    # F3: a run receipt, GPU context or build summary without its code hashes is unresolved (MISSING CODE HASH), never skipped
    runs = []
    for f in sorted(PUB.glob("v2-publish-real-*.json")):
        d = json.loads(f.read_text())
        cs = d.get("code_sha256")
        code = {k: resolve(v) for k, v in cs.items()} if isinstance(cs, dict) and cs else {"<code_sha256>": resolve(None)}
        runs.append({"receipt": f.name, "sha256": sha(f.read_bytes()), "code": code})
    gpu, gpu_missing = set(), []
    for p in sorted(glob.glob(str(RUN / "build-v2/**/out/*/context.json"), recursive=True)):
        g = json.loads(Path(p).read_text()).get("gpu")
        h = ((g or {}).get("code_sha256") or {}).get("v2_gpu_encode.py") if isinstance(g, dict) else None
        if h:
            gpu.add(h)
        else:
            gpu_missing.append(str(Path(p).relative_to(RUN)))
    # build summaries are classified by schema, not file name: "...-summary/vN" must carry its code hashes; a
    # "...-summary-correction/vN" receipt is an addendum to a summary (listed, bound to the summary it corrects); any other
    # *summary*.json is unclassified and fails closed
    builds, addenda = {}, {}
    for s in sorted((RUN / "receipts/trainonly-v2/build").glob("*summary*.json")):
        d = json.loads(s.read_text()); schema = str(d.get("schema", ""))
        if re.search(r"-summary-correction/v\d+$", schema):
            addenda[s.name] = {"schema": schema, "sha256": sha(s.read_bytes())}
            continue
        if not re.search(r"-summary/v\d+$", schema):
            builds[s.name] = {"<schema>": {"sha256_prefix": "", "status": f"UNCLASSIFIED SUMMARY {schema!r}"}}
            continue
        cs = d.get("code_sha256")
        builds[s.name] = {k: resolve(v) for k, v in cs.items()} if isinstance(cs, dict) and cs else {"<code_sha256>": resolve(None)}
    # every document verified in the publish ledger (denylists, S1 lists, supplements, superseded, provenance), not only
    # context manifests: each code hash it records, at any depth; per-context/per-row arrays are scanned at EVERY row (F2),
    # the distinct hashes deduplicated with the generic path they occur at
    bulky = {"contexts", "rows", "datasets", "object_keys", "assignment_by_view", "manifests", "publish_runs", "build_summaries", "gpu_encode_versions"}

    def code_hashes(d, path=""):
        if isinstance(d, dict):
            for k, v in d.items():
                if k.endswith("code_sha256"):
                    if not v:                              # F3: present but empty is unresolved, not skipped
                        yield f"{path}/{k}", None
                    for kk, vv in (v.items() if isinstance(v, dict) else [("", v)] if v else []):
                        yield f"{path}/{k}" + (f"/{kk}" if kk else ""), vv if isinstance(vv, str) and vv else None
                elif k not in bulky:
                    yield from code_hashes(v, f"{path}/{k}")

    by_size = {}
    for p in PUB.glob("*.json"):
        by_size.setdefault(p.stat().st_size, []).append(p)
    docs = {}
    for line in vp.DEFAULT_PUBLISH_LEDGER.read_bytes().splitlines():
        r = json.loads(line)
        if r.get("event") == "verified" and ("/manifests/" in r["key"] or "/provenance/provenance-" in r["key"]):
            docs[r["key"]] = r
    published_docs = []
    for key, r in sorted(docs.items()):
        local = [p for p in by_size.get(r.get("bytes"), []) if sha(p.read_bytes()) == r["sha256"]]
        if not local:
            published_docs.append({"key": key, "sha256": r["sha256"], "status": "NO LOCAL COPY"})
            continue
        d = json.loads(local[0].read_bytes())
        found = set(code_hashes(d))
        for fld in ("contexts", "rows"):
            for row in (d.get(fld) if isinstance(d.get(fld), list) else []):
                found.update(code_hashes(row, f"/{fld}[*]"))
        code = {f"{p}@{h if h else 'MISSING'}": resolve(h) for p, h in sorted(found, key=lambda t: (t[0], t[1] or ""))}
        name = key.split("v2-trainonly/", 1)[1].split("/")[1]
        if not found:
            code = {"inline": resolve(INLINE[name])} if name in INLINE else {"<code_sha256>": resolve(None)}
        published_docs.append({"key": key, "version_id": r["version_id"], "sha256": r["sha256"], "local": local[0].name, "code": code})
    unresolved = sorted({f"{e['local']}:{x['sha256_prefix'] or x['status']}" for e in man for x in (e["code"], e["publisher"]) if x and x.get("status")}
                        | {f"{r['receipt']}:{k}:{x['sha256_prefix'] or x['status']}" for r in runs for k, x in r["code"].items() if x and x.get("status")}
                        | {f"{b}:{k}:{x['sha256_prefix'] or x['status']}" for b, v in builds.items() for k, x in v.items() if x and x.get("status")}
                        | {f"{e['key']}:{p}:{x['sha256_prefix'] or x['status']}" for e in published_docs for p, x in (e.get("code") or {}).items() if x and x.get("status")}
                        | {f"{e['key']}:{e['status']}" for e in published_docs if e.get("status")}
                        | {f"gpu_encode_versions:{h}:{x['status']}" for h, x in ((h, resolve(h)) for h in gpu) if x and x.get("status")}   # F1
                        | {f"{p}:gpu.code_sha256:MISSING CODE HASH" for p in gpu_missing})                                              # F3
    doc = {"schema": "jope.dl-v2-provenance/v1", "name": a.name,
           "purpose": "PROVENANCE-01: every recorded code hash resolved to exact bytes; lane source is git-ignored, so the bytes are published here create-only",
           "notes": ["18eac154 = dltrain/rowsel_raw.py (module); 3951fabf = tools/v2_rowsel_raw.py (tool that built ROWSEL-RAW-v1)",
                     "2751260c, 0519ea30 (tools/v2_rowsel_manifest.py) and cfa1d899 (tools/v2_publish.py) were reconstructed by reverse-applying "
                     "recorded edits and verified byte-exact by sha256",
                     "provenance-v1 covered context manifests, publish runs, GPU encodes and build summaries only; v2 adds every verified "
                     "published document. 8c957d9b (tools/v2_holdout_denylist.py, denylist v1), 40ba1c69 (tools/v2_s1_input_list.py, S1 lists "
                     "v2) were reconstructed by replaying the recorded writes and patch scripts and verified byte-exact by sha256 (the replay "
                     "also reproduces the current 63eeea98); superseded-raw-split-v1 was built by an inline script, kept byte-exact (cd51465f)"],
           "manifests": man, "publish_runs": runs, "gpu_encode_versions": {h: resolve(h) for h in sorted(gpu)}, "gpu_contexts_missing_code_hash": gpu_missing,
           "build_summaries": builds, "summary_addenda": addenda,
           "published_documents": published_docs,
           "unresolved": unresolved, "verdict": "FAIL" if unresolved else "PASS", "code_sha256": sha(Path(__file__).read_bytes())}
    body = (json.dumps(doc, sort_keys=True, indent=1) + "\n").encode(); dg = sha(body)
    (PUB / f"v2-{a.name}-{dg[:12]}.json").write_bytes(body)
    out = {"sha256": dg, "unresolved": unresolved, "custody_files": len(custody), "verdict": doc["verdict"]}
    if unresolved:                                     # fail closed: nothing is published while any recorded hash is unbound
        print(json.dumps(out, sort_keys=True))
        print(f"FAIL: {len(unresolved)} recorded code hashes do not resolve to custody bytes", file=sys.stderr)
        return 2
    if a.publish:
        writer, reader = vp.real_clients(); ledger = vp.PublishLedger(vp.DEFAULT_PUBLISH_LEDGER)
        try:
            pub = vp.Publisher(writer, reader, ledger)
            for h, c in custody.items():
                pub.publish_bytes(f"{vp.s3io.V2_PREFIX_KEY}provenance/code/{c['file']}.{h}", c["path"].read_bytes(), "text/x-python", "provenance", "provenance")
            out["published"] = pub.publish_bytes(f"{vp.s3io.V2_PREFIX_KEY}provenance/{a.name}/{dg[:12]}.json", body, "application/json", "manifest", "provenance")
        finally:
            ledger.close()
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
