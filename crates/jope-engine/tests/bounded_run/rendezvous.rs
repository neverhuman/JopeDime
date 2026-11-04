// ---------------------------------------------------------------- W7-T04 rendezvous

struct Rv {
    run: Run,
    c: Coordinator<FakeSupervisor>,
    fake: FakeSupervisor,
    rv: Rendezvous,
    closure: String,
    now: u64,
}
/// A planned, preflighted run whose three ranks are launched (fake supervisor) and whose
/// rendezvous is open: ranks exist but none may start work before admission.
fn rendezvous_run() -> Rv {
    let now = now_ms();
    let run = planned(now);
    let fake = FakeSupervisor::default();
    let mut c = Coordinator::open(&run.dir, &run.sha, fake.clone(), now).unwrap();
    c.preflight(
        &serde_json::to_vec(&census(&run.contract, now)).unwrap(),
        now,
    )
    .unwrap();
    c.dispatch_all(now).unwrap();
    let rv = c
        .open_rendezvous(
            &run.keys.coordinator,
            "fixture-coordinator-boot",
            &rendezvous::fresh_nonce().unwrap(),
            now,
        )
        .unwrap();
    let closure =
        rendezvous::observed_closure_digest(&run.root, &run.contract.artifact_closure).unwrap();
    Rv {
        run,
        c,
        fake,
        rv,
        closure,
        now,
    }
}
impl Rv {
    fn honest(&self, rank: u32) -> HelloBody {
        hello_body(
            &self.run.contract,
            &self.run.sha,
            self.rv.challenge(),
            rank,
            self.now,
            &self.closure,
        )
    }
    fn frame(&self, body: HelloBody) -> Vec<u8> {
        hello_frame(&self.run.keys, body)
    }
    /// Persist the refusal and prove nothing started and nothing was committed.
    fn refused_with_zero_commits(mut self, refusal: GroupRefusal, cause: &RefusalCause) {
        assert_eq!(&refusal.cause, cause);
        assert_eq!((refusal.started_ranks, refusal.committed_updates), (0, 0));
        self.c.record_refusal(&refusal, self.now).unwrap();
        assert_eq!(
            self.fake.running(),
            0,
            "launched rank work survived refusal"
        );
        let late_admission = dummy_admission(&self);
        assert!(self.c.record_admission(late_admission, self.now).is_err());
        assert!(self.c.record_dispatch_intent(0, self.now).is_err());
        let dir = self.run.dir.clone();
        drop(self.c);
        let s = controller::status(&dir).unwrap();
        assert_eq!(s.outcome, Some(Outcome::Refused));
        assert_eq!(s.committed_updates_total, 60_000);
        assert!(s
            .ranks
            .iter()
            .all(|r| r.disposition == Some(Disposition::Refused)));
        assert!(!dir.join("results").exists());
        controller::verify(&dir, Some(&self.run.sha)).unwrap();
    }
}
fn dummy_admission(r: &Rv) -> rendezvous::GroupAdmission {
    rendezvous::GroupAdmission {
        body: rendezvous::AdmissionBody {
            schema: "jope.native_group_admission.v1".into(),
            run: r.run.contract.run.clone(),
            attempt: 1,
            plan_sha256: r.run.sha.clone(),
            rendezvous_id: r.rv.rendezvous_id().into(),
            admitted_unix_ms: r.now,
            original_deadline_unix_ms: r.run.contract.original_deadline_unix_ms,
            grants_sha256: String::new(),
            members: Vec::new(),
            communicator_bootstrap: String::new(),
        },
        signature: String::new(),
    }
}
