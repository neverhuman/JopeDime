import json
import tempfile
import unittest
from pathlib import Path

from extend_component_sampling import load_registries, sha, validate_coverage


class RegistryUnion(unittest.TestCase):
    def registry(self, root, name, recipes, module=b'frozen', window=32):
        directory = root/name
        directory.mkdir()
        (directory/'module.pt').write_bytes(module)
        (directory/'sampling.json').write_text(json.dumps(dict(window=window, views=2, recipes=recipes)))
        parity = dict(passed=True, module_sha256=sha(directory/'module.pt'), sampling_sha256=sha(directory/'sampling.json'), tests=[dict(context=16, query=8, width=1, batch=1, passed=True)])
        (directory/'parity.json').write_text(json.dumps(parity))
        spec = {}
        for key, filename in [('module','module.pt'),('sampling','sampling.json'),('parity','parity.json')]:
            spec[key] = str((directory/filename).relative_to(root))
            spec[key+'_sha256'] = sha(directory/filename)
        path = directory/'spec.json'
        path.write_text(json.dumps(spec))
        return path.relative_to(root)

    def test_union_preserves_parents_and_qualified_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            base=self.registry(root,'base',{'1:100':{'rows':[1,2]}})
            extra=self.registry(root,'extra',{'1:100':{'rows':[1,2]},'1:229':{'rows':[3,4]}})
            before={p:sha(p) for p in root.rglob('*') if p.is_file()}
            _, sampler, parity, records=load_registries(root,'target',[base,extra])
            self.assertEqual(set(sampler['recipes']),{'1:100','1:229'})
            self.assertEqual(len(parity['tests']),1)
            self.assertEqual(len(records),2)
            self.assertEqual(before,{p:sha(p) for p in before})

    def test_conflicting_recipe_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            base=self.registry(root,'base',{'1:100':{'rows':[1,2]}})
            extra=self.registry(root,'extra',{'1:100':{'rows':[2,1]}})
            with self.assertRaisesRegex(ValueError,'conflicting existing recipe'):
                load_registries(root,'target',[base,extra])

    def test_different_frozen_module_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            base=self.registry(root,'base',{})
            extra=self.registry(root,'extra',{},module=b'changed')
            with self.assertRaisesRegex(ValueError,'frozen modules differ'):
                load_registries(root,'target',[base,extra])

    def test_changed_sampler_configuration_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            base=self.registry(root,'base',{})
            extra=self.registry(root,'extra',{},window=64)
            with self.assertRaisesRegex(ValueError,'sampler configurations differ'):
                load_registries(root,'target',[base,extra])

    def test_parity_merge_requires_complete_unique_coverage(self):
        request=dict(target_new_lengths=[[1,80]], foundation_new_shapes=[[60,20,7,2]])
        target=[dict(task=1,length=80,kind=k,passed=True) for k in ['random','constant','endpoints']]
        foundation=[dict(context=60,query=20,width=7,batch=2,passed=True)]
        validate_coverage(request,target,foundation)
        with self.assertRaisesRegex(ValueError,'Target parity coverage'):
            validate_coverage(request,target[:-1],foundation)
        with self.assertRaisesRegex(ValueError,'Foundation parity coverage'):
            validate_coverage(request,target,foundation*2)
        with self.assertRaisesRegex(ValueError,'Frozen module parity failed'):
            validate_coverage(request,target,[dict(foundation[0],passed=False)])


if __name__ == '__main__': unittest.main()
