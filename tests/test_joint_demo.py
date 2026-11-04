"""Internal partial-closure tests. MCP import stub explicitly disclosed.
No installed MCP transport/server/runtime behavior is tested.
"""
import contextlib, copy, io, json, os, sys, tempfile, types, unittest
from pathlib import Path
ROOT=Path(__file__).parent/'source'
if not ROOT.is_dir():ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
# Explicit internal diagnostic switch, forbidden in a real offline receipt.
MCP_STUB_USED=os.environ.get('CARTPROOF_INTERNAL_MCP_IMPORT_STUB')=='1'
if MCP_STUB_USED:
 mcp=types.ModuleType('mcp');server=types.ModuleType('mcp.server')
 class MCPServer:
  def __init__(self,*a,**kw):raise AssertionError('MCP stub MUST NOT execute')
 server.MCPServer=MCPServer;mcp.server=server
 sys.modules['mcp']=mcp;sys.modules['mcp.server']=server
else:
 from mcp.server import MCPServer
import httpx
from agents.model_client import BudgetedModelClient, ClientConfig
from agents.meta_client import MetaModelClient
from harness.shopify_live_backend import ShopifyLiveBackend
from harness.joint_demo_candidate import JointDemo
from shopify_live_fake import FakeLiveShopify
A='CARTPROOF-TEST-A';B='CARTPROOF-TEST-B'
MAP={'variants':{A:'gid://shopify/ProductVariant/1',B:'gid://shopify/ProductVariant/2'}}
TOKEN='synthetic-prep-shop-token-000000';KEY='synthetic-prep-model-secret-000000'
def responses():
 intent={'items':[{'sku':A,'quantity':1}],'budget_max_cents':None,'substitution_allowed':False}
 add={'action':'add_to_cart','query':None,'product_id':None,'sku':A,'quantity':1}
 return [intent,add,intent,add,add]
class Case(unittest.TestCase):
 def fixture(self,d,modify=None,mode=None,limit=6,tokens=20000,products=None):
  fake=FakeLiveShopify(mode=mode,products=products or [
   {'id':'gid://shopify/Product/1','title':'Synthetic','variants':[
    {'id':v,'sku':k} for k,v in MAP['variants'].items()]}])
  mapping=Path(d)/'mapping.json';mapping.write_text(json.dumps(MAP))
  be=ShopifyLiveBackend('cartproof.myshopify.com',str(mapping),token=TOKEN,transport=fake.transport)
  pro=responses()
  if modify:modify(pro)
  calls=[]
  def handler(req):
   self.assertEqual(req.url.host,'api.meta.ai');calls.append(json.loads(req.content))
   n=len(calls)-1
   item=pro[n]
   if item=='timeout':raise httpx.ReadTimeout('SENSITIVE '+KEY,request=req)
   usage={'prompt_tokens':40,'completion_tokens':20,'total_tokens':60}
   if item=='missing_usage':payload=responses()[0];usage=None
   elif item=='inconsistent_usage':payload=responses()[0];usage['total_tokens']=999
   elif item=='overrun':payload=responses()[0];usage={'prompt_tokens':99999,'completion_tokens':20,'total_tokens':100019}
   else:payload=item
   return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{
      'content':json.dumps(payload)}}], 'usage':usage})
  cfg=ClientConfig(enabled=True,provider='meta',base_url='https://api.meta.ai/v1',
   model_name='muse-spark-1.3',max_calls=limit,token_budget=tokens,
   max_output_tokens_per_call=1500,live_enabled=True,
   reasoning_effort_by_kind={x:'low' for x in ['intent','action','review']},
   price_table={'per_1m_tokens':{'muse-spark-1.3':{'input':1.25,'output':4.25,'cached_input':.15}}})
  inner=MetaModelClient(cfg,allow_live=True,environ={'MODEL_API_KEY':KEY},transport=httpx.MockTransport(handler))
  model=BudgetedModelClient(inner,cfg,allow_live=True)
  shown=[];job=JointDemo(be,model,MAP,Path(d)/'run',secrets=[TOKEN,KEY],emit=shown.append)
  return job,fake,calls,shown
 def runjob(self,modify=None,mode=None,limit=6,tokens=20000):
  with tempfile.TemporaryDirectory() as d:
   j,f,c,s=self.fixture(d,modify,mode,limit,tokens);r=j.execute()
   raw='\n'.join(s)+'\n'+json.dumps(r)
   for p in (Path(d)/'run').rglob('*'):
    if p.is_file():raw+=p.read_text()
   for secret in [TOKEN,KEY,f.cart_id,'fakekeycapability000000']:self.assertNotIn(secret,raw)
   return r,f,c,s
 def test_complete_real_modules_fake_transports_partial_dependency_closure(self):
  r,f,c,s=self.runjob();self.assertTrue(r['complete'],r)
  self.assertEqual(r['model_calls'],5);self.assertEqual(r['effects']['drop_hook_uses'],1)
  self.assertEqual(r['effects']['add_attempts'],3);self.assertEqual(r['effects']['recovery_attempts'],1)
  self.assertLessEqual(r['shopify_requests'],20)
  self.assertEqual([x['quantity'] for x in f.lines.values()],[1,2])
  self.assertTrue(any('controlled_drop_NO_REQUEST_SENT' in x for x in s))
 def test_invalid_intent_no_cart_or_writes(self):
  r,f,c,s=self.runjob(lambda p:p[0]['items'][0].update(sku='UNAPPROVED-SKU'))
  self.assertFalse(r['complete']);self.assertEqual(r['shopify_requests'],0)
 def test_invalid_quantity_no_cart(self):
  r,f,c,s=self.runjob(lambda p:p[0]['items'][0].update(quantity=True))
  self.assertFalse(r['complete']);self.assertEqual(r['shopify_requests'],0)
 def test_invalid_action_stops_after_setup(self):
  r,f,c,s=self.runjob(lambda p:p[1].update(sku=B));self.assertFalse(r['complete'])
  self.assertEqual(r['effects']['add_attempts'],1)
 def test_missing_usage_stops_before_cart(self):
  r,f,c,s=self.runjob(lambda p:p.__setitem__(0,'missing_usage'))
  self.assertEqual(r['reason'],'model_usage_unverified');self.assertEqual(r['shopify_requests'],0)
 def test_inconsistent_usage_stops_before_cart(self):
  r,f,c,s=self.runjob(lambda p:p.__setitem__(0,'inconsistent_usage'))
  self.assertEqual(r['reason'],'model_cost_unverified');self.assertEqual(r['shopify_requests'],0)
 def test_usage_overrun_stops_before_cart(self):
  r,f,c,s=self.runjob(lambda p:p.__setitem__(0,'overrun'))
  self.assertFalse(r['complete']);self.assertEqual(r['shopify_requests'],0)
 def test_timeout_secret_never_rendered(self):
  r,f,c,s=self.runjob(lambda p:p.__setitem__(0,'timeout'));self.assertFalse(r['complete'])
  self.assertEqual(r['shopify_requests'],0)
 def test_model_budget_exhausted_no_request(self):
  r,f,c,s=self.runjob(tokens=1);self.assertFalse(r['complete']);self.assertEqual(len(c),0)
 def test_call_cap_blocks_recovery(self):
  r,f,c,s=self.runjob(limit=4);self.assertFalse(r['complete'])
  self.assertEqual(r['effects']['recovery_attempts'],0)
 def test_truncated_never_mutates(self):
  r,f,c,s=self.runjob(mode='truncated');self.assertFalse(r['complete'])
  self.assertEqual(r['effects']['add_attempts'],0)
 def test_wrong_cart_never_mutates(self):
  r,f,c,s=self.runjob(mode='wrong_cart');self.assertFalse(r['complete'])
  self.assertEqual(r['effects']['add_attempts'],0)
 def test_mutation_error_no_retry(self):
  r,f,c,s=self.runjob(mode='add_http_500');self.assertFalse(r['complete'])
  self.assertEqual(r['effects']['add_attempts'],1)
 def test_existing_output_refused(self):
  with tempfile.TemporaryDirectory() as d:
   j,f,c,s=self.fixture(d);j.output.mkdir();sentinel=j.output/'demo-evidence.json';sentinel.write_text('PRESERVE');r=j.execute();self.assertEqual(sentinel.read_text(),'PRESERVE')
   self.assertFalse(r['complete']);self.assertEqual(len(c),0);self.assertEqual(f.requests,[])
 def test_request_cap_no_retry(self):
  with tempfile.TemporaryDirectory() as d:
   j,f,c,s=self.fixture(d);j.backend.set_request_cap(2);r=j.execute()
   self.assertFalse(r['complete']);self.assertEqual(r['shopify_requests'],2)
 def test_alias_mapping_refused(self):
  with tempfile.TemporaryDirectory() as d:
   j,f,c,s=self.fixture(d);j.mapping={'variants':{A:MAP['variants'][A],B:MAP['variants'][A]}}
   r=j.execute();self.assertEqual(r['reason'],'mapping_alias');self.assertEqual(c,[])
 def test_model_finish_not_mutation(self):
  r,f,c,s=self.runjob(lambda p:p.__setitem__(1,{'action':'finish','query':None,'product_id':None,'sku':None,'quantity':None}))
  self.assertEqual(r['reason'],'action_not_exact_allowed_add');self.assertEqual(r['effects']['add_attempts'],1)
 def test_missing_output_on_incomplete_sequence(self):
  with tempfile.TemporaryDirectory() as d:
   j,f,c,s=self.fixture(d,modify=lambda p:p[1].update(sku=B));r=j.execute()
   self.assertFalse(r['complete']);self.assertFalse((j.output/'manifest.json').exists())
 def test_recovery_changed_state_stop(self):
  with tempfile.TemporaryDirectory() as d:
   j,f,c,s=self.fixture(d);original=j.choose_add
   def choose(*a,**kw):
    original(*a,**kw)
    if a[-1]=='bounded_recovery':next(iter(f.lines.values()))['quantity']=2
   j.choose_add=choose;r=j.execute();self.assertFalse(r['complete'])
   self.assertEqual(r['effects']['recovery_attempts'],0)
 def test_progress_secret_echo_refused(self):
  with tempfile.TemporaryDirectory() as d:
   j,f,c,s=self.fixture(d)
   with self.assertRaises(Exception):j.publish('echo',payload=KEY)
   self.assertEqual(s,[])
 def test_progress_capability_echo_refused(self):
  with tempfile.TemporaryDirectory() as d:
   j,f,c,s=self.fixture(d)
   with self.assertRaises(Exception):j.publish('echo',payload=f.cart_id)
   self.assertEqual(s,[])
 def test_deadline_no_mutation(self):
  with tempfile.TemporaryDirectory() as d:
   j,f,c,s=self.fixture(d);j.deadline=-1;r=j.execute()
   self.assertFalse(r['complete']);self.assertEqual(f.requests,[])
 def test_action_context_serialized_for_each_scenario_and_recovery(self):
  r,f,c,s=self.runjob();self.assertTrue(r['complete'],r)
  actions=[json.loads(x['messages'][1]['content']) for x in c if json.loads(x['messages'][1]['content'])['kind']=='action']
  self.assertEqual(len(actions),3)
  for row,phase,baseline,target in zip(actions,['matching','controlled_drop','bounded_recovery'],[{B:1},{A:1,B:1},{A:1,B:1}],[{A:1,B:1},{A:2,B:1},{A:2,B:1}]):
   obs=row['observed_state']
   self.assertEqual(obs['phase'],phase);self.assertEqual(obs['scenario_baseline_lines'],baseline)
   self.assertEqual(obs['requested_additional_lines'],{A:1});self.assertEqual(obs['scenario_target_lines'],target)
   self.assertIn('existing units are not completion',obs['action_semantics'])
   self.assertEqual(row['available_actions'],[{'action':'add_to_cart','sku':A,'quantity':1},{'action':'finish'}])
   self.assertEqual(row['completed_items'],[])
  self.assertEqual(actions[1]['observed_state']['scenario_target_lines'],actions[2]['observed_state']['scenario_target_lines'])
 def reject_at_both_later_actions(self,payload):
  for idx,phase,drop in [(3,'controlled_drop',0),(4,'recovery',1)]:
   with self.subTest(index=idx):
    r,f,c,s=self.runjob(lambda p:p.__setitem__(idx,copy.deepcopy(payload)))
    self.assertFalse(r['complete']);self.assertEqual(r['phase'],phase)
    self.assertEqual(r['reason'],'action_not_exact_allowed_add')
    self.assertEqual(r['effects']['add_attempts'],2);self.assertEqual(r['effects']['recovery_attempts'],0)
    self.assertEqual(r['effects']['drop_hook_uses'],drop)
    self.assertEqual(len(c),idx+1)
 def test_finish_still_refused_at_drop_and_recovery(self):
  self.reject_at_both_later_actions({'action':'finish'})
 def test_wrong_sku_still_refused_at_drop_and_recovery(self):
  self.reject_at_both_later_actions({'action':'add_to_cart','sku':B,'quantity':1})
 def test_wrong_quantity_still_refused_at_drop_and_recovery(self):
  self.reject_at_both_later_actions({'action':'add_to_cart','sku':A,'quantity':2})
if __name__=='__main__':unittest.main(verbosity=2)
