"""Durable coverage ledger and GP job bundle producer. Owns no execution workers."""
import json
import math
import sqlite3
import time
from pathlib import Path

from continuation import digest, encoded, plan
from run_episode import read_rows, summarize


def sha(path): return digest(Path(path).read_bytes())


def immutable(path, value):
    import os
    import tempfile
    path = Path(path)
    raw = encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as out:
            temporary=Path(out.name)
            out.write(raw);out.flush();os.fsync(out.fileno())
        try:
            try: os.link(temporary,path)
            except FileExistsError: pass
            directory=os.open(path.parent,os.O_RDONLY)
            try: os.fsync(directory)
            finally: os.close(directory)
        finally: temporary.unlink()
    if path.read_bytes()!=raw: raise ValueError("Immutable artifact changed: " + str(path))
    return digest(raw)


def within(root, path):
    path = Path(path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    path.relative_to(root)
    return path


def live_process(owner):
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        ticks = Path(f"/proc/{int(owner['pid'])}/stat").read_text().rsplit(") ",1)[1].split()[19]
        return boot == owner["boot_id"] and ticks == str(owner["start_ticks"])
    except (OSError, ValueError, KeyError, IndexError): return False


def validate_native_source(root, payload, rows, receipt, vocabulary_sha):
    """Validate original complete ledgers; never rewrite row fitness or receipts."""
    identity=payload["identity"];job=payload["job"]
    prepared_path=within(root,job["prepared"])
    if sha(prepared_path/"prepared.json")!=identity["prepared_sha256"]: raise ValueError("Admission prepared bytes changed")
    prepared=json.loads((prepared_path/"prepared.json").read_text())
    cut=next(c for c in prepared["cuts"] if c["index"]==identity["cut"])
    preprocessing_path=prepared_path/f"cut-{identity['cut']}"/"preprocessing.json"
    if sha(preprocessing_path)!=cut["preprocessing_sha256"]: raise ValueError("Admission preprocessing changed")
    preprocessing=json.loads(preprocessing_path.read_text())
    if prepared["meta_partition"]!="meta_train" or prepared["family"]!=identity["family"]: raise ValueError("Admission partition/family mismatch")
    if any(prepared["family"].removeprefix("rdatasets-")==f.removeprefix("rdatasets-") for f in identity["excluded_families"]): raise ValueError("Excluded family source")
    receipt_expected={k:identity[k] for k in ("evaluator_sha256","prepared_sha256","cut","vocabulary_sha256","seeds_sha256","rng_seed")}
    receipt_expected["evaluator_protocol"]="featmath_training_fold_normalized_v1"
    if any(receipt.get(k)!=v for k,v in receipt_expected.items()): raise ValueError("Admission search receipt identity mismatch")
    if not isinstance(preprocessing["target_scale"],(int,float)) or not math.isfinite(preprocessing["target_scale"]) or preprocessing["target_scale"]<=0: raise ValueError("Invalid target scale")
    feature_view=digest(json.dumps(dict(protocol="raw_training_fold_v1",training_sha256=cut["training_sha256"],preprocessing_sha256=cut["preprocessing_sha256"]),sort_keys=True,separators=(",",":")).encode())
    keys=("program_sha256","family","partition","prepared_sha256","cut","training_sha256","validation_sha256","evaluator_sha256","evaluator_protocol","preprocessing_sha256","target_scale","metric_units","vocabulary_sha256","feature_view_sha256","evaluation_seed","resource_protocol")
    for row in rows:
        expected=dict(family=prepared["family"],dataset=prepared["dataset"],partition="meta_train",evaluator_sha256=identity["evaluator_sha256"],
            evaluator_protocol="featmath_training_fold_normalized_v1",resource_protocol=identity["resource_protocol"],vocabulary_sha256=vocabulary_sha,
            training_sha256=cut["training_sha256"],validation_sha256=cut["validation_sha256"],preprocessing_sha256=cut["preprocessing_sha256"],
            target_scale=preprocessing["target_scale"],metric_units="cut_normalized_target",evaluation_seed=0,feature_view_sha256=feature_view)
        if expected["resource_protocol"]!="single_candidate_2s_8gib_v1" or any(row.get(k)!=v for k,v in expected.items()): raise ValueError("Admission execution identity conflict")
        canonical=json.dumps({key:row[key] for key in keys},sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        if digest(canonical)!=row.get("evaluation_identity_sha256"): raise ValueError("Admission evaluation identity digest changed")
        runtime=row.get("runtime_seconds")
        if not isinstance(runtime,(int,float)) or not math.isfinite(runtime) or runtime<0: raise ValueError("Admission runtime invalid")
        if "failure" not in row: raise ValueError("Explicit native outcome absent")
        if row.get("failure") is None:
            if row.get("metrics",{}).get("units")!="cut_normalized_target": raise ValueError("Admission metric units changed")
        elif not isinstance(row["failure"],str) or not row["failure"].strip(): raise ValueError("Admission failure outcome invalid")


class Dispatcher:
    def __init__(self, root, directory, run_identity, run_output, vocabulary, evaluator):
        self.root = Path(root).resolve()
        self.directory = within(self.root, directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_output = within(self.root, run_output)
        self.vocabulary = within(self.root, vocabulary)
        self.evaluator = within(self.root, evaluator)
        self.db = sqlite3.connect(self.directory / "coverage.sqlite", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS episodes(id TEXT PRIMARY KEY,payload TEXT NOT NULL,status TEXT NOT NULL,owner TEXT,deadline REAL,result TEXT);
            CREATE TABLE IF NOT EXISTS emissions(generation INTEGER,population_sha TEXT,slot INTEGER,episode TEXT,status TEXT,PRIMARY KEY(generation,population_sha,slot));
            CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,episode TEXT,status TEXT,at REAL,detail TEXT);""")
        identity = dict(run_identity=run_identity,run_output=str(self.run_output),vocabulary_sha256=sha(self.vocabulary),evaluator_sha256=sha(self.evaluator))
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES('identity',?)", (json.dumps(identity,sort_keys=True),))
            existing=self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
            if json.loads(existing[0])!=identity: raise ValueError("Dispatcher run/evaluator/vocabulary identity changed")
        self.identity = identity

    def ingest(self, generation, jobs_path, jobs_sha256, generic, exclusions, population_sha256=None):
        """Consume only a completed proposal job manifest inside this current learning run."""
        jobs_path = within(self.root,jobs_path)
        jobs_path.relative_to(self.run_output)
        if sha(jobs_path) != jobs_sha256: raise ValueError("Proposal jobs changed")
        jobs = json.loads(jobs_path.read_text())
        if jobs["schema"] != "parallel.gp_jobs.v1": raise ValueError("Wrong proposal jobs schema")
        for job in jobs["jobs"]:
            if job.get("generation") != generation: raise ValueError("Generation identity mismatch")
            source = within(self.root,job["seeds"])
            source.relative_to(self.run_output)
            # Shuffled comparison files retain neural_policy origins; their arm is not neural.
            if source.relative_to(jobs_path.parent).parts[0]!="neural": continue
            if population_sha256 is not None and job["seeds_sha256"]!=population_sha256: continue
            if sha(source) != job["seeds_sha256"]: raise ValueError("Emission bytes changed")
            population = json.loads(source.read_text())
            if not any(c["origin"]=="neural_policy" for c in population["candidates"]): continue
            parent = within(self.root,job["parent_checkpoint"])
            if sha(parent / "state.json") != job["parent_checkpoint_sha256"] or population["checkpoint_sha256"] != job["parent_checkpoint_sha256"]:
                raise ValueError("Producing checkpoint changed")
            prepared_path = within(self.root,job["prepared"])
            prepared = json.loads((prepared_path / "prepared.json").read_text())
            if sha(prepared_path / "prepared.json") != job["prepared_sha256"] or population["cut"] != job["cut"]:
                raise ValueError("Prepared cut changed")
            planned = plan(population,prepared,generic,prepared_sha=job["prepared_sha256"],population_sha=job["seeds_sha256"],
                           generic_sha=digest(encoded(generic)),evaluator_sha=self.identity["evaluator_sha256"],
                           vocabulary_sha=self.identity["vocabulary_sha256"],excluded_families=exclusions,rng_seed=job["rng_seed"])
            with self.db:
                for episode in planned["episodes"]:
                    directory = self.directory / "episodes" / episode["episode_id"]
                    identity_sha = immutable(directory / "identity.json",episode["identity"])
                    seeds_sha = immutable(directory / "seeds.json",episode["seeds"])
                    immutable(directory / "manifest.json",dict(schema="parallel.seed_experiment.v2",continuation_identity_sha256=identity_sha,final_tests_opened=False))
                    native_job = dict(prepared=job["prepared"],prepared_sha256=job["prepared_sha256"],cut=job["cut"],
                        output=str((directory / "search").relative_to(self.root)),seeds=str((directory / "seeds.json").relative_to(self.root)),seeds_sha256=seeds_sha,
                        rng_seed=job["rng_seed"],generation=generation,parent_checkpoint=job["parent_checkpoint"],parent_checkpoint_sha256=job["parent_checkpoint_sha256"],
                        continuation=dict(identity=str((directory / "identity.json").relative_to(self.root)),identity_sha256=identity_sha))
                    payload = dict(identity=episode["identity"],job=native_job,source_jobs_sha256=jobs_sha256,source_population=str(source.relative_to(self.root)))
                    old = self.db.execute("SELECT payload FROM episodes WHERE id=?",(episode["episode_id"],)).fetchone()
                    if old and json.loads(old[0])["identity"] != payload["identity"]: raise ValueError("Episode identity collision")
                    self.db.execute("INSERT OR IGNORE INTO episodes(id,payload,status) VALUES(?,?,'pending')",(episode["episode_id"],json.dumps(payload,sort_keys=True)))
                for emission in planned["emissions"]:
                    value=(generation,job["seeds_sha256"],emission["slot"],emission.get("episode_id"),emission["status"])
                    self.db.execute("INSERT OR IGNORE INTO emissions VALUES(?,?,?,?,?)",value)

    def pending_bundle(self, limit=104):
        if not 0 <= limit <= 104: raise ValueError("Bundle cannot exceed shared104-slot allocation")
        rows=self.db.execute("SELECT payload FROM episodes WHERE status='pending' ORDER BY id LIMIT ?",(limit,))
        return dict(schema="parallel.gp_jobs.v1",qualification=True,vocabulary=str(self.vocabulary.relative_to(self.root)),jobs=[json.loads(r[0])["job"] for r in rows])

    def transition(self, episode, status, *, owner=None, deadline=None, result=None):
        with self.db:
            self.db.execute("UPDATE episodes SET status=?,owner=?,deadline=?,result=? WHERE id=?",(status,json.dumps(owner) if owner else None,deadline,json.dumps(result) if result else None,episode))
            self.db.execute("INSERT INTO events(episode,status,at,detail) VALUES(?,?,?,?)",(episode,status,time.time(),json.dumps(result or owner)))

    def reconcile(self, now=None, process_live=live_process):
        now=time.time() if now is None else now
        for record in list(self.db.execute("SELECT * FROM episodes WHERE status!='completed'")):
            payload=json.loads(record["payload"]);job=payload["job"];identity=payload["identity"]
            output=within(self.root,job["output"]);episode=record["id"]
            identity_sha=job["continuation"]["identity_sha256"]
            if sha(within(self.root,job["continuation"]["identity"]))!=identity_sha or sha(within(self.root,job["seeds"]))!=job["seeds_sha256"]:
                raise ValueError("Queued episode source changed")
            owner=None
            if (output / "dispatch-process.json").exists():
                owner=json.loads((output / "dispatch-process.json").read_text())
                if owner["continuation_sha256"]!=identity_sha or owner["deadline_at"]-owner["started_at"]!=36:
                    raise ValueError("Worker deadline/identity changed")
                if record["owner"] and json.loads(record["owner"])!=owner: raise ValueError("Running episode owner changed")
                if record["deadline"] is not None and record["deadline"]!=owner["deadline_at"]: raise ValueError("Frozen worker deadline changed")
            deadline=owner["deadline_at"] if owner else record["deadline"]
            if (output / "search.json").exists():
                receipt=json.loads((output / "search.json").read_text())
                if receipt.get("continuation_sha256")!=identity_sha or receipt.get("final_tests_opened") is not False:
                    raise ValueError("Completed continuation receipt changed")
                if json.loads((output / "job-binding.json").read_text()) != job: raise ValueError("Pool job binding mismatch")
                if owner is None:
                    # A very short search can publish before its service records the child identity.
                    # Never use absence of an earlier running sample as permission to skip the deadline.
                    continue
                completed_at=receipt.get("completed_at")
                if not isinstance(completed_at,(int,float)) or not math.isfinite(completed_at): raise ValueError("Completion timestamp absent or invalid")
                if not owner["started_at"]<=completed_at<=deadline:
                    if record["status"]!="deadline": self.transition(episode,"deadline",owner=owner,deadline=deadline,result=dict(reason="completion outside frozen worker window; retained, not admitted"))
                    continue
                rows=read_rows(output)
                result=summarize(identity,identity_sha,rows)
                validate_native_source(self.root,payload,rows,receipt,self.identity["vocabulary_sha256"])
                # Keep original full ledger and receipt. Packer may include parent's OWN score too.
                result["source_admission"]=dict(source=str(output.relative_to(self.root)),observations_sha256=sha(output / "observations.json"),
                    search_sha256=sha(output / "search.json"),dispatch_process_sha256=sha(output / "dispatch-process.json"),job_binding_sha256=sha(output / "job-binding.json"),eligible_offspring_rows=[i for i,r in enumerate(rows) if i>0 and r["genetic_parents"] and r.get("metrics") is not None],
                    evaluator_sha256=identity["evaluator_sha256"],resource_protocol=identity["resource_protocol"],parent_relabeling=False,requires_normal_provenance_validation=True)
                self.transition(episode,"completed",owner=owner,deadline=deadline,result=result)
            elif record["status"] in ("failed","deadline"):
                continue
            elif (output / "dispatch-result.json").exists():
                result=json.loads((output / "dispatch-result.json").read_text())
                if result["continuation_sha256"] != identity_sha: raise ValueError("Worker result identity changed")
                self.transition(episode,"deadline" if result.get("deadline_exceeded") else "failed",result=result)
            elif owner is not None:
                if now>deadline: self.transition(episode,"deadline",owner=owner,deadline=deadline,result=dict(reason="bound worker deadline elapsed"))
                elif not process_live(owner): self.transition(episode,"failed",owner=owner,deadline=deadline,result=dict(reason="worker identity lost; budget not reset"))
                elif record["status"]!="running": self.transition(episode,"running",owner=owner,deadline=deadline)

    def sources(self, admitted_evaluator_sha256):
        if sha(self.evaluator)!=self.identity["evaluator_sha256"] or sha(self.vocabulary)!=self.identity["vocabulary_sha256"]:
            raise ValueError("Admitted executable/vocabulary bytes changed")
        if admitted_evaluator_sha256 != self.identity["evaluator_sha256"]: raise ValueError("New evaluator not explicitly admitted")
        sources=[]
        for record in self.db.execute("SELECT result,payload FROM episodes WHERE status='completed'"):
            source=json.loads(record[0])["source_admission"]
            payload=json.loads(record[1]);job=payload["job"]
            if sha(within(self.root,job["parent_checkpoint"])/"state.json")!=job["parent_checkpoint_sha256"]:
                raise ValueError("Admitted producing checkpoint changed")
            output=within(self.root,source["source"])
            if any(sha(output/name)!=source[key] for name,key in (("observations.json","observations_sha256"),("search.json","search_sha256"),("dispatch-process.json","dispatch_process_sha256"),("job-binding.json","job_binding_sha256"))):
                raise ValueError("Completed source changed after admission")
            if source["eligible_offspring_rows"]: sources.append(source)
        return dict(schema="parallel.continuation_sources.v1",run_identity=self.identity["run_identity"],sources=sources,
                    admission_policy=dict(schema="parallel.label_admission.v1",evaluator=str(self.evaluator.relative_to(self.root)),evaluator_sha256=admitted_evaluator_sha256,evaluator_protocol="featmath_training_fold_normalized_v1",resource_protocol="single_candidate_2s_8gib_v1"),
                    parent_relabeling=False,seed_search_value_consumed=False,normal_packer_validation_required=True)

    def additive_pack_request(self, existing_sources, admitted_evaluator_sha256):
        """Retain reference_v1 and the exact prior source prefix; validate new ledgers separately."""
        admitted=self.sources(admitted_evaluator_sha256)
        sources=list(existing_sources)
        for source in admitted["sources"]:
            if source["source"] not in sources: sources.append(source["source"])
        return dict(schema="parallel.additive_continuation_pack.v1",run_identity=self.identity["run_identity"],
            label_policy="reference_v1",existing_sources=list(existing_sources),existing_prefix_sha256=digest(encoded(existing_sources)),
            sources=sources,new_source_admission=admitted,existing_evaluator_identities_preserved=True,
            pass_strict_midrank_admission_flag=False,parent_relabeling=False,seed_search_value_consumed=False)
