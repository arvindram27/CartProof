import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import test_joint_demo as test_joint_candidate
from harness import joint_demo_control as c
from harness.joint_demo_worker import run_worker
class Controls(unittest.TestCase):
 def profile(self):return json.loads((test_joint_candidate.ROOT/'harness/joint_demo_profile.json').read_text())
 def owned(self,d):
  p=Path(d);(p/'ownership.json').write_text('{}');return p
 def test_profile_exact(self):c.validate_profile(self.profile())
 def test_profile_changed(self):
  p=self.profile();p['shop']='other.invalid'
  with self.assertRaises(Exception):c.validate_profile(p)
 def test_reservation_reuse_refused(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.owned(d);c.reserve_attempt(p,'0123456789abcdef')
   with self.assertRaises(FileExistsError):c.reserve_attempt(p,'0123456789abcdef')
 def test_partial_receipt_refused(self):
  with self.assertRaises(Exception):c.verify_offline_receipt({'complete':True},'a','b','c')
 def test_stub_receipt_refused(self):
  r={'complete':True,'real_dependency_import':True,'source_digest':'a','tests_sha256':'b',
    'profile_sha256':'c','network_mode':'mock_only','live_requests':0,'tests_failed':0,
    'tests_run':22,'mcp_stub_used':True,'platform':'win32'}
  with self.assertRaises(Exception):c.verify_offline_receipt(r,'a','b','c')
 def test_barrier_timeout(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.owned(d)
   with self.assertRaises(Exception):c.wait_release(p,timeout=0)
 def test_supervision_platform_stop(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.owned(d)
   with patch.object(c.sys,'platform','linux'):
    with self.assertRaises(Exception):c.supervise(p,{})
 def test_lifecycle_cleanup_required(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.owned(d)
   with patch.object(c.sys,'platform','win32'):
    with self.assertRaises(Exception):c.supervise(p,{},lifecycle=lambda *a:({'complete':True},{'cleanup_status':'unverified'}))
 def test_lifecycle_env_and_args(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.owned(d)
   def fake(ctx,cmd,env,info,barrier):
    self.assertEqual(cmd[1:4],['-B','-m','harness.joint_demo_worker'])
    self.assertNotIn('EVIL=',env);self.assertNotIn('GIT_CONFIG',env)
    self.assertIn('MODEL_API_KEY=SYNTHETIC',env)
    self.assertIn('CARTPROOF_JOINT_PARENT_LAUNCH=reviewed-parent-v1',env)
    return {'complete':True},{'job_assigned':True,'cleanup_status':'verified','timed_out':False}
   with patch.object(c.sys,'platform','win32'):
    c.supervise(p,{'MODEL_API_KEY':'SYNTHETIC','EVIL':'secret','GIT_CONFIG_COUNT':'1'},lifecycle=fake)
    def fake_offline(ctx,cmd,env,info,barrier):
     self.assertIn('--offline-owned-probe',cmd)
     keys={x.split('=',1)[0] for x in env.split('\0') if x}
     self.assertEqual(keys,{'SYSTEMROOT','PYTHONDONTWRITEBYTECODE','CARTPROOF_JOINT_PARENT_LAUNCH'})
     self.assertIn('SYSTEMROOT=SYNTHETIC',env)
     self.assertIn('PYTHONDONTWRITEBYTECODE=1',env)
     self.assertIn('CARTPROOF_JOINT_PARENT_LAUNCH=reviewed-parent-v1',env)
     self.assertNotIn('MODEL_API_KEY=',env)
     self.assertNotIn('CARTPROOF_SHOPIFY_STOREFRONT_TOKEN=',env)
     self.assertNotIn('CARTPROOF_SHOPIFY_STOREFRONT_TOKEN_FILE=',env)
     self.assertNotIn('EVIL=',env)
     return {'complete':True},{'job_assigned':True,'cleanup_status':'verified','timed_out':False}
    c.supervise(p,{'SYSTEMROOT':'SYNTHETIC','MODEL_API_KEY':'SHOULD-DROP','CARTPROOF_SHOPIFY_STOREFRONT_TOKEN':'SHOULD-DROP','CARTPROOF_SHOPIFY_STOREFRONT_TOKEN_FILE':'SHOULD-DROP','EVIL':'secret'},offline_probe=True,lifecycle=fake_offline)
    def fake_empty(ctx,cmd,env,info,barrier):
     self.assertEqual(env,'CARTPROOF_JOINT_PARENT_LAUNCH=reviewed-parent-v1\0PYTHONDONTWRITEBYTECODE=1\0\0')
     self.assertNotIn('SYSTEMROOT=',env)
     return {'complete':True},{'job_assigned':True,'cleanup_status':'verified','timed_out':False}
    c.supervise(p,{},offline_probe=True,lifecycle=fake_empty)
 def test_real_worker_orchestration_mock_injected(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.owned(d);(p/'profile.json').write_text(json.dumps(self.profile()))
   fixture_dir=p/'fixture';fixture_dir.mkdir()
   j,f,calls,shown=test_joint_candidate.Case().fixture(fixture_dir)
   r=run_worker(p,backend_factory=lambda _: (j.backend,test_joint_candidate.TOKEN),
     model_factory=lambda _:(j.model,test_joint_candidate.KEY),barrier=False)
   self.assertTrue(r['complete'],r);self.assertEqual(r['shopify_requests'],16)
   self.assertEqual(r['run_context'],'offline_injected')
 def test_evidence_disk_failure_no_complete(self):
  with tempfile.TemporaryDirectory() as d:
   j,f,calls,shown=test_joint_candidate.Case().fixture(d)
   original=Path.write_text
   def fail(path,*a,**kw):
    if path.name=='demo-evidence.json':raise OSError('SECRET-DO-NOT-PRINT')
    return original(path,*a,**kw)
   with patch.object(Path,'write_text',fail):r=j.execute()
   self.assertFalse(r['complete']);self.assertEqual(r['reason'],'finalization_or_interrupt_failure')
   self.assertFalse((j.output/'manifest.json').exists())
 def test_unknown_worker_exception_no_leak(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.owned(d);(p/'profile.json').write_text(json.dumps(self.profile()))
   def fail(_):raise RuntimeError('SECRET')
   with self.assertRaises(RuntimeError):run_worker(p,backend_factory=fail,barrier=False)
   self.assertFalse((p/'run.json').exists())
 def test_timeout_after_shopify_applied_no_retry(self):
  import httpx
  with tempfile.TemporaryDirectory() as d:
   j,f,calls,shown=test_joint_candidate.Case().fixture(d)
   original=f.transport.handler
   def timeout_after_apply(request):
    result=original(request)
    body=json.loads(request.content)
    if 'cartLinesAdd' in body.get('query',''):
     raise httpx.ReadTimeout('secret-echo',request=request)
    return result
   f.transport.handler=timeout_after_apply
   r=j.execute();self.assertFalse(r['complete'])
   self.assertEqual(r['effects']['add_attempts'],1)
   # Remote fake did mutate before timeout. Failure must never trigger replay.
   self.assertEqual(sum(x['quantity'] for x in f.lines.values()),1)
   self.assertEqual(sum('cartLinesAdd' in q['query'] for q in f.requests),1)
 def test_unknown_sku_echo_not_followed(self):
  with tempfile.TemporaryDirectory() as d:
   j,f,calls,shown=test_joint_candidate.Case().fixture(d)
   original=j.backend.acquire_state
   def foreign():
    state=original()
    if len(calls)>=2 and state.get('lines') is not None:
     state['lines']['FOREIGN-SKU']=1
    return state
   j.backend.acquire_state=foreign;r=j.execute()
   self.assertFalse(r['complete']);self.assertEqual(r['effects']['recovery_attempts'],0)
 def test_final_guard_refuses_exact_key_echo(self):
  with tempfile.TemporaryDirectory() as d:
   j,f,calls,shown=test_joint_candidate.Case().fixture(d)
   original=j.check
   def inject(*a,**kw):
    result=original(*a,**kw)
    if a[2]=='recovered':
     (j.output/'echo.txt').write_text(test_joint_candidate.KEY)
    return result
   j.check=inject;r=j.execute()
   self.assertFalse(r['complete']);self.assertEqual(r['reason'],'archive_guard_refused')
   self.assertFalse((j.output/'manifest.json').exists())
 @unittest.skipUnless(c.sys.platform=='win32','real Windows lifecycle requires Windows')
 def test_REAL_WINDOWS_owned_lifecycle_offline(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.owned(d)
   sr=os.environ.get('SYSTEMROOT')
   self.assertTrue(sr,'SYSTEMROOT required for real Windows lifecycle')
   result,info=c.supervise(p,{'SYSTEMROOT':sr},offline_probe=True)
   self.assertTrue(result['offline_owned_probe']);self.assertEqual(result['network_requests'],0)
   self.assertEqual(info['cleanup_status'],'verified')
 def test_empty_seed_evidence_stop(self):
  import store.mcp_server_v2 as module
  original=module.run_check
  def missing(state):
   r=original(state);r['seed_lines']=[];return r
  with tempfile.TemporaryDirectory() as d:
   j,f,calls,shown=test_joint_candidate.Case().fixture(d)
   with patch.object(module,'run_check',missing):r=j.execute()
   self.assertFalse(r['complete']);self.assertEqual(r['reason'],'preserved_item_evidence_missing_or_changed')
   self.assertFalse(any('unrelated_item_preserved' in x for x in shown))
 def test_unexpected_exception_fixed_class_no_raw_echo(self):
  with tempfile.TemporaryDirectory() as d:
   j,f,calls,shown=test_joint_candidate.Case().fixture(d)
   def boom():raise ValueError(test_joint_candidate.KEY)
   j.state=boom;r=j.execute()
   self.assertEqual(r['error_class'],'ValueError');self.assertNotIn(test_joint_candidate.KEY,json.dumps(r))
 def test_offline_network_denies_dns_socket_http(self):
  import socket,httpx
  from harness.joint_demo_offline_guard import offline_boundary
  with offline_boundary() as blocked:
   with self.assertRaises(RuntimeError):socket.getaddrinfo('api.meta.ai',443)
   with self.assertRaises(RuntimeError):socket.create_connection(('cartproof.myshopify.com',443))
   with self.assertRaises(RuntimeError):
    with httpx.Client(trust_env=False) as client:client.get('https://api.meta.ai/v1')
   self.assertEqual(len(blocked),3)
 def test_offline_strips_credentials_and_fallback(self):
  import os
  from harness.joint_demo_offline_guard import offline_boundary
  from agents.env_loader import load_root_dotenv
  with patch.dict(os.environ,{'MODEL_API_KEY':'SECRET','CARTPROOF_SHOPIFY_STOREFRONT_TOKEN':'SECRET',
       'CARTPROOF_SHOPIFY_STOREFRONT_TOKEN_FILE':'private-path'}):
   with offline_boundary():
    self.assertNotIn('MODEL_API_KEY',os.environ)
    self.assertNotIn('CARTPROOF_SHOPIFY_STOREFRONT_TOKEN_FILE',os.environ)
    from agents import env_loader
    with self.assertRaises(RuntimeError):env_loader.load_root_dotenv(Path('/unused'))
   self.assertEqual(os.environ['MODEL_API_KEY'],'SECRET')
 def test_offline_mock_http_still_works(self):
  import httpx
  from harness.joint_demo_offline_guard import offline_boundary
  with offline_boundary():
   with httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'safe':True}))) as client:
    self.assertTrue(client.get('https://synthetic.invalid').json()['safe'])
 def test_worker_model_construction_closes_backend(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.owned(d);(p/'profile.json').write_text(json.dumps(self.profile()))
   fd=p/'fixture';fd.mkdir();j,f,calls,shown=test_joint_candidate.Case().fixture(fd)
   def fail(_):raise ValueError('SECRET')
   with self.assertRaises(ValueError):run_worker(p,backend_factory=lambda _:(j.backend,'SYNTHETIC'),model_factory=fail,barrier=False)
   self.assertTrue(j.backend.client.is_closed)
if __name__=='__main__':unittest.main(verbosity=2)
