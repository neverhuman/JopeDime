import importlib.util
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

spec=importlib.util.spec_from_file_location("admission",Path(__file__).with_name("deferred_proposal_admission.py"))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.out=self.root/"runs/admission";self.out.mkdir(parents=True)
        self.cycle=self.root/"runs/campaign/cycle-0000";self.cycle.mkdir(parents=True)
        m.write(self.root/"runs/campaign-progress.json",{"current_cycle":str(self.cycle)})
        m.write(self.root/"runs/cycle-progress.json",{"neural_seeded_gp":{"completed":2,"total":2}})
        self.jobs=[{"output":"runs/library/a/search","prepared":"data/a","cut":3,"seeds":"runs/library/a/seeds.json"}]
        m.write(self.out/"combined-jobs.json",{"jobs":self.jobs})
        m.write(self.root/"contexts.json",{"sealed":True})
        self.plan={"cycle":"runs/campaign/cycle-0000","expected_comparisons":2,"deadline":time.time()+600,
                   "pins":{"runs/admission/combined-jobs.json":m.digest(self.out/"combined-jobs.json"),
                           "contexts.json":m.digest(self.root/"contexts.json")},
                   "contexts":"contexts.json","executable":"bin/pinned-cpu"}
        self.calls=0
    def enqueue(self,command,remaining,log):
        self.calls+=1
        self.assertLessEqual(remaining,30)
        self.assertIsNone(m.lease(self.root/"runs/encoding-service.lock"))
        m.write(self.root/"runs/gp-priority-jobs.json",{"jobs":self.jobs})
    def test_phase_barrier_is_held_during_atomic_admission(self):
        result=m.protected_admit(self.root,self.out,self.plan,self.enqueue)
        self.assertEqual(result["outcome"],"admitted");self.assertEqual(result["newly_admitted_jobs"],1)
        self.assertEqual(self.calls,1)
        handle=m.lease(self.root/"runs/encoding-service.lock");self.assertIsNotNone(handle);handle.close()
    def test_unfinished_or_advanced_or_completed_cycle_never_enqueues(self):
        for kind in ["unfinished","advanced","train2"]:
            with self.subTest(kind=kind):
                m.write(self.root/"runs/cycle-progress.json",{"neural_seeded_gp":{"completed":1 if kind=="unfinished" else 2,"total":2}})
                m.write(self.root/"runs/campaign-progress.json",{"current_cycle":str(self.cycle.parent/"cycle-0001") if kind=="advanced" else str(self.cycle)})
                if kind=="train2":m.write(self.cycle/"train-2/complete.json",{})
                result=m.protected_admit(self.root,self.out,self.plan,self.enqueue)
                self.assertEqual(result["outcome"],"skipped")
        self.assertEqual(self.calls,0)
    def test_busy_lease_does_not_wait_or_admit(self):
        held=m.lease(self.root/"runs/encoding-service.lock")
        try:self.assertIsNone(m.protected_admit(self.root,self.out,self.plan,self.enqueue))
        finally:held.close()
        self.assertEqual(self.calls,0)
    def test_mutated_input_aborts_before_enqueue_and_releases_barrier(self):
        m.write(self.root/"contexts.json",{"changed":True})
        with self.assertRaises(ValueError):m.protected_admit(self.root,self.out,self.plan,self.enqueue)
        self.assertEqual(self.calls,0)
        held=m.lease(self.root/"runs/encoding-service.lock");self.assertIsNotNone(held);held.close()
    def test_timeout_after_commit_reports_actual_admission(self):
        def committed_then_timeout(*args):
            self.enqueue(*args)
            raise subprocess.TimeoutExpired(args[0],1)
        result=m.protected_admit(self.root,self.out,self.plan,committed_then_timeout)
        self.assertEqual(result["outcome"],"admitted");self.assertTrue(result["cli_error"])
    def test_already_committed_manifest_does_not_reorder_queue(self):
        m.write(self.root/"runs/gp-priority-jobs.json",{"jobs":[{"output":"future-priority"}]+self.jobs})
        before=(self.root/"runs/gp-priority-jobs.json").read_bytes()
        result=m.protected_admit(self.root,self.out,self.plan,self.enqueue)
        self.assertEqual(result["newly_admitted_jobs"],0);self.assertEqual(self.calls,0)
        self.assertEqual(before,(self.root/"runs/gp-priority-jobs.json").read_bytes())
    def test_deadline_margin_skips_without_admission(self):
        self.plan["deadline"]=time.time()+5
        self.assertEqual(m.protected_admit(self.root,self.out,self.plan,self.enqueue)["outcome"],"skipped")
        self.assertEqual(self.calls,0)
    def next_boundary(self):
        boundary=self.cycle.parent/"cycle-0001";boundary.mkdir()
        m.write(self.root/"runs/campaign-progress.json",{"current_cycle":str(boundary)})
        m.write(self.cycle/"complete.json",{"comparisons":2})
        self.plan["boundary_cycle"]=str(boundary.relative_to(self.root))
        return boundary
    def test_next_boundary_holds_barrier_before_remaining_gpu_stage(self):
        boundary=self.next_boundary()
        for stage in ["train-1","neural"]:m.write(boundary/stage/"complete.json",{})
        self.assertEqual(m.protected_admit(self.root,self.out,self.plan,self.enqueue)["outcome"],"admitted")
        self.assertEqual(self.calls,1)
    def test_next_boundary_rejects_finished_gpu_stages_before_queue_file(self):
        boundary=self.next_boundary()
        for stage in ["train-1","neural","shuffled"]:m.write(boundary/stage/"complete.json",{})
        self.assertEqual(m.protected_admit(self.root,self.out,self.plan,self.enqueue)["outcome"],"skipped")
        self.assertEqual(self.calls,0)
    def test_next_boundary_rejects_search_admission_and_later_cycle(self):
        boundary=self.next_boundary()
        m.write(boundary/"search-jobs.json",{})
        self.assertEqual(m.protected_admit(self.root,self.out,self.plan,self.enqueue)["outcome"],"skipped")
        (boundary/"search-jobs.json").unlink()
        m.write(self.root/"runs/campaign-progress.json",{"current_cycle":str(boundary.parent/"cycle-0002")})
        self.assertEqual(m.protected_admit(self.root,self.out,self.plan,self.enqueue)["outcome"],"skipped")
        self.assertEqual(self.calls,0)

if __name__=="__main__":unittest.main()
