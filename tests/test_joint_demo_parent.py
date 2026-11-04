"""Internal parent orchestration fault tests. Gates mocked, not actual closure."""
import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import test_joint_demo as test_joint_candidate
from harness import joint_demo_parent as p
class Parent(unittest.TestCase):
 def fixture(self,d):
  root=Path(d);(root/'harness').mkdir();(root/'tests').mkdir();control=root/'var/joint-demo-control';control.mkdir(parents=True)
  (control/'ownership.json').write_text('{}')
  (root/'harness/joint_demo_source_manifest.json').write_text('{}')
  (root/'harness/joint_demo_profile.json').write_bytes((test_joint_candidate.ROOT/'harness/joint_demo_profile.json').read_bytes())
  (control/'offline-receipt-source-fedcba9876543210.json').write_text('{"windows_owned_lifecycle_verified":true,"validation_id":"fedcba9876543210"}')
  for n in ['test_joint_demo.py','test_joint_demo_control.py','test_joint_demo_parent.py']:(root/'tests'/n).write_text('')
  return root
 def runparent(self,root,worker,allow=True):
  output=io.StringIO()
  with patch.object(p,'ROOT',root),patch.object(p.sys,'platform','win32'),patch.object(p.sys,'executable',str(root/'.venv/python.exe')),patch.object(p,'verify_source',return_value='source'),patch.object(p,'verify_offline_receipt'),patch.object(p,'supervise',side_effect=worker),patch.dict(p.os.environ,{},clear=True),contextlib.redirect_stdout(output):
   rc=p.main(['--attempt','0123456789abcdef','--validation-id','fedcba9876543210']+(['--allow-live'] if allow else []))
  return rc,json.loads(output.getvalue())
 def test_parent_success_reads_exact_closure(self):
  with tempfile.TemporaryDirectory() as d:
   root=self.fixture(d)
   def worker(attempt,env):
    e=attempt/'evidence';e.mkdir();(e/'a.txt').write_text('safe')
    (e/'manifest.json').write_text(json.dumps({'a.txt':p.digest(e/'a.txt')}))
    return {'complete':True,'run_context':'real_model_and_shopify'},{'cleanup_status':'verified'}
   rc,out=self.runparent(root,worker);self.assertEqual(rc,0);self.assertEqual(out['status'],'COMPLETE')
 def test_parent_worker_error_has_stage_hash_diagnostics_no_leak(self):
  with tempfile.TemporaryDirectory() as d:
   root=self.fixture(d)
   def worker(attempt,env):
    (attempt/'stderr.txt').write_text('SECRET-VALUE');raise ValueError('SECRET-VALUE')
   rc,out=self.runparent(root,worker);self.assertEqual(rc,2);self.assertEqual(out['stage'],'owned_worker')
   self.assertEqual(out['error_class'],'ValueError');self.assertNotIn('SECRET-VALUE',json.dumps(out))
   self.assertEqual(out['diagnostics']['stderr.txt']['bytes'],12)
 def test_parent_changed_closure_stop(self):
  with tempfile.TemporaryDirectory() as d:
   root=self.fixture(d)
   def worker(attempt,env):
    e=attempt/'evidence';e.mkdir();(e/'a.txt').write_text('safe');(e/'manifest.json').write_text('{"a.txt":"bad"}')
    return {'complete':True,'run_context':'real_model_and_shopify'},{}
   rc,out=self.runparent(root,worker);self.assertEqual(rc,2);self.assertEqual(out['reason'],'evidence_hash_changed')
 def test_no_live_flag_no_worker(self):
  with tempfile.TemporaryDirectory() as d:
   root=self.fixture(d)
   def worker(*a):raise AssertionError('must not run')
   rc,out=self.runparent(root,worker,allow=False);self.assertEqual(rc,2);self.assertEqual(out['reason'],'explicit_live_flag_required')
if __name__=='__main__':unittest.main(verbosity=2)
