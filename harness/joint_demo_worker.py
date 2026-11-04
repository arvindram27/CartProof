"""Candidate fixed local worker. Parent source/offline/authority gates required.
No shell, no recording automation, no retries. Not approved for user execution.
"""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
from harness.joint_demo_candidate import JointDemo,DemoStop,need
from harness.joint_demo_control import wait_release,validate_profile,regular_chain

def discover(backend):
    # Only synthetic SKU/GID pairs enter the catalog shown to the model.
    data=backend.read_catalog();variants={};seen=set()
    edges=(data.get('products') or {}).get('edges')
    need(isinstance(edges,list),'catalog_malformed')
    for edge in edges:
        node=edge.get('node',{})
        for v in (node.get('variants') or {}).get('edges',[]):
            item=v.get('node',{});sku=item.get('sku');gid=item.get('id')
            if isinstance(sku,str) and sku.startswith('CARTPROOF-TEST-'):
                need(sku not in variants and gid not in seen,'catalog_duplicate')
                variants[sku]=gid;seen.add(gid)
    need(len(variants)>=2,'catalog_shortfall')
    return {'variants':variants}

def run_worker(directory,*,backend_factory=None,model_factory=None,barrier=True):
    d=Path(directory);regular_chain(d/'ownership.json')
    if barrier:wait_release(d)
    # Parent creates immutable controls before launch. No user-provided endpoint.
    regular_chain(d/'profile.json')
    profile=validate_profile(json.loads((d/'profile.json').read_text(encoding='utf-8')))
    # Live opt-in is child-local and only reached behind parent+barrier gates.
    # This is a mechanical opt-in, never a source of owner authority.
    if backend_factory is None:
        from harness.shopify_live_backend import ShopifyLiveBackend,resolve_token
        token=resolve_token();need(bool(token),'shopify_token_unavailable')
        os.environ['CARTPROOF_SHOPIFY_LIVE']='1'
        mapping=d/'empty-mapping.json';mapping.write_text('{"variants":{}}',encoding='utf-8')
        backend=ShopifyLiveBackend(profile['shop'],str(mapping),token=token,
                                   api_version=profile['api_version'])
    else:
        backend,token=backend_factory(profile)
    model=None
    try:
        model,key=construct_model(profile,model_factory)
    except BaseException:
        backend.client.close()
        raise
    try:
        mapping=discover(backend)
        job=JointDemo(backend,model,mapping,d/'evidence',secrets=[token,key],
            emit=lambda line:print(line,flush=True),timeout_seconds=profile['timeout_seconds']-20)
        result=job.execute()
        result['run_context']='real_model_and_shopify' if backend_factory is None else 'offline_injected'
        with (d/'run.json').open('x',encoding='utf-8') as f:
            json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
        return result
    finally:
        backend.client.close()
        inner=getattr(model,'inner',None)
        client=getattr(inner,'_client',None)
        if client is not None:client.close()

def construct_model(profile,model_factory):
    if model_factory is None:
        from agents.model_client import ClientConfig,BudgetedModelClient
        from agents.meta_client import MetaModelClient
        from agents.env_loader import load_root_dotenv
        root=Path(__file__).absolute().parents[1]
        key=os.environ.get('MODEL_API_KEY') or load_root_dotenv(root)
        need(isinstance(key,str) and bool(key.strip()),'model_key_unavailable')
        cfg=ClientConfig(enabled=True,provider='meta',base_url=profile['provider_origin'],
            model_name=profile['model'],max_calls=profile['max_calls'],
            token_budget=profile['token_budget'],max_output_tokens_per_call=profile['max_output_tokens'],
            live_enabled=True,reasoning_effort_by_kind={k:profile['reasoning_effort']
                 for k in ('intent','action','review')},
            price_table={'as_of':profile['pricing_as_of'],'per_1m_tokens':{
                profile['model']:{'input':1.25,'cached_input':.15,'output':4.25}}})
        inner=MetaModelClient(cfg,allow_live=True,environ={'MODEL_API_KEY':key})
        model=BudgetedModelClient(inner,cfg,allow_live=True)
    else:model,key=model_factory(profile)
    return model,key

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker-dir',required=True)
    parser.add_argument('--offline-owned-probe',action='store_true')
    args=parser.parse_args()
    # Deliberately refuses standalone launch, even on Windows. Parent wrapper
    # must exist and be reviewed before a live command can be enabled.
    if os.environ.get('CARTPROOF_JOINT_PARENT_LAUNCH')!='reviewed-parent-v1':
        print('{"status":"STOP","reason":"parent_launch_required"}')
        return 2
    stage='branch_select'
    try:
        if args.offline_owned_probe:
            stage='offline_guard_import'
            from harness.joint_demo_offline_guard import offline_boundary
            stage='offline_boundary_enter'
            with offline_boundary():
                stage='offline_wait_release'
                d=Path(args.worker_dir);wait_release(d)
                stage='offline_released'
                result={'complete':True,'offline_owned_probe':True,'network_requests':0}
            stage='offline_run_write'
            with (d/'run.json').open('x',encoding='utf-8') as f:json.dump(result,f)
        else:
            stage='live_worker'
            result=run_worker(Path(args.worker_dir))
        return 0 if result.get('complete') else 1
    except BaseException as exc:
        # Exact fixed metadata only. Never output arbitrary exception text.
        detail={'status':'STOP','reason':'worker_failed_preserve_attempt','stage':stage,
                'error_class':type(exc).__name__ if type(exc).__module__=='builtins' else 'GateError'}
        allowed={'linked_path_refused','regular_file_required','startup_barrier_timeout','startup_barrier_invalid',
                 'offline_network_denied','offline_credential_fallback_denied'}
        detail['detail_reason']=str(exc) if str(exc) in allowed else 'not_disclosed'
        print(json.dumps(detail,sort_keys=True))
        return 2
if __name__=='__main__':raise SystemExit(main())
