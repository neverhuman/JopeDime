from test_provenance_support import *


class ManualPrBindTests(unittest.TestCase):
    def facts(self):
        repository = dict(id=1376159324, full_name="neverhuman/JopeDime")
        env = dict(GITHUB_REPOSITORY="neverhuman/JopeDime", GITHUB_REPOSITORY_ID="1376159324",
            GITHUB_SHA="b"*40, GITHUB_REF="refs/heads/codex/runner",
            GITHUB_WORKFLOW_REF="neverhuman/JopeDime/.github/workflows/ci.yml@refs/heads/codex/runner")
        pr = dict(number=420, state="open", merged=False,
            head=dict(sha="b"*40, ref="codex/runner", repo=repository.copy()),
            base=dict(sha="a"*40, ref="main", repo=repository.copy()))
        return env,pr

    def test_exact_open_manual_pr_uses_current_merged_base(self):
        env,pr=self.facts()
        with mock.patch.object(bind,"github_public_json",side_effect=[pr,dict(object=dict(sha="a"*40))]) as api:
            self.assertEqual("a"*40,bind.manual_pr_bind(env,"420"))
            self.assertEqual(["pulls/420","git/ref/heads/main"],[c.args[0] for c in api.call_args_list])

    def test_wrong_head_fork_workflow_state_base_and_stale_main_refuse(self):
        import copy
        mutations=[("state","closed"),("merged",True),("number",421),
            ("head.sha","c"*40),("head.ref","foreign"),("head.repo.id",1),
            ("head.repo.full_name","foreign/repo"),("base.ref","other"),
            ("base.sha","HEAD"),("base.repo.id",1),("base.repo.full_name","foreign/repo")]
        for key,value in mutations:
            with self.subTest(key=key):
                env,pr=self.facts();changed=copy.deepcopy(pr);dst=changed;parts=key.split(".")
                for part in parts[:-1]: dst=dst[part]
                dst[parts[-1]]=value
                with mock.patch.object(bind,"github_public_json",return_value=changed),self.assertRaises(ValueError):
                    bind.manual_pr_bind(env,"420")
        env,pr=self.facts()
        with mock.patch.object(bind,"github_public_json",side_effect=[pr,dict(object=dict(sha="c"*40))]),self.assertRaises(ValueError):
            bind.manual_pr_bind(env,"420")
        for key,value in [("GITHUB_SHA","c"*40),("GITHUB_REF","refs/heads/main"),
            ("GITHUB_WORKFLOW_REF","neverhuman/JopeDime/.github/workflows/full-main.yml@refs/heads/codex/runner"),
            ("GITHUB_REPOSITORY_ID","1"),("GITHUB_REPOSITORY","foreign/repo")]:
            with self.subTest(key=key):
                env,pr=self.facts();env[key]=value
                with mock.patch.object(bind,"github_public_json",return_value=pr),self.assertRaises(ValueError):
                    bind.manual_pr_bind(env,"420")
        for number in ["0","-1","420/path"," 420","0420","1"*10]:
            with self.subTest(number=number),mock.patch.object(bind,"github_public_json") as api,self.assertRaises(ValueError):
                bind.manual_pr_bind(self.facts()[0],number)
            api.assert_not_called()

    def test_dispatch_payload_routes_through_validation_and_default_stays_exact(self):
        scratch=ROOT/'target/tmp';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as name:
            path=Path(name)/'event.json';env,_=self.facts()
            env.update(GITHUB_EVENT_NAME='workflow_dispatch',GITHUB_EVENT_PATH=str(path),BASE_SHA='a'*40)
            path.write_text(json.dumps(dict(inputs=dict(provenance_pr='420'))))
            with mock.patch.object(bind,'manual_pr_bind',return_value='a'*40) as validate:
                self.assertEqual('a'*40,bind.bind_rev_from_env(env))
                validate.assert_called_once_with(env,'420')
            path.write_text(json.dumps(dict(inputs=dict(provenance_pr=''))))
            with mock.patch.object(bind,'github_public_json') as api:
                self.assertIsNone(bind.bind_rev_from_env(env))
                api.assert_not_called()

    def test_dispatch_without_inputs_retains_exact_head_and_wrong_types_refuse(self):
        scratch=ROOT/'target/tmp';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as name:
            path=Path(name)/'event.json'
            env=dict(GITHUB_EVENT_NAME='workflow_dispatch',GITHUB_EVENT_PATH=str(path),BASE_SHA='a'*40)
            for payload in [{},{'inputs':None},{'inputs':{}},{'inputs':{'provenance_pr':''}}]:
                path.write_text(json.dumps(payload))
                with self.subTest(payload=payload),mock.patch.object(bind,'github_public_json') as api:
                    self.assertIsNone(bind.bind_rev_from_env(env));api.assert_not_called()
            for inputs in [False,0,'',[],['420']]:
                path.write_text(json.dumps({'inputs':inputs}))
                with self.subTest(inputs=inputs),self.assertRaises(ValueError):
                    bind.bind_rev_from_env(env)

    def test_api_missing_token_and_failure_are_not_exact_head_fallback(self):
        with self.assertRaises(ValueError):bind.github_public_json("pulls/420",{})
        with mock.patch.object(bind,"github_public_json",side_effect=OSError("unavailable")),self.assertRaises(OSError):
            bind.manual_pr_bind(self.facts()[0],"420")
