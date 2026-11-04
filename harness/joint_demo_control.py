"""Local demo control primitives. Candidate only; reviewed runtime still needed.

Source/config/offline receipts are integrity checks, not user consent.
The operator still needs the separately approved exact code and live run.
"""
from __future__ import annotations
import hashlib,json,os,stat,sys,time
from pathlib import Path
from harness.joint_demo_candidate import DemoStop,need

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def regular_chain(path):
    p=Path(path).absolute()
    for part in [p,*p.parents]:
        st=part.lstat()
        need(not stat.S_ISLNK(st.st_mode) and not getattr(st,'st_file_attributes',0)&0x400,
             'linked_path_refused')
    need(p.is_file(),'regular_file_required')

def write_fresh(path,data):
    p=Path(path)
    regular_chain(p.parent/'ownership.json')
    with p.open('x',encoding='utf-8') as f:
        f.write(json.dumps(data,indent=2,sort_keys=True));f.flush();os.fsync(f.fileno())

def verify_source(root,manifest):
    need(manifest.get('schema')==1,'manifest_schema')
    entries=manifest.get('files')
    need(isinstance(entries,dict) and len(entries)>=181,'source_manifest_incomplete')
    need(set(entries).issuperset({'harness/joint_demo_parent.py', 'harness/joint_demo_worker.py',
         'harness/joint_demo_candidate.py','harness/joint_demo_control.py',
         'harness/joint_demo_offline.py','harness/joint_demo_offline_guard.py','harness/joint_demo_profile.json',
         'tests/test_joint_demo.py','tests/test_joint_demo_control.py','tests/test_joint_demo_parent.py'}), 'source_manifest_missing_payload')
    for rel,want in entries.items():
        need(isinstance(rel,str) and not rel.startswith(('/','\\')) and
             '\\' not in rel and ':' not in rel and all(p not in ('','..','.') for p in rel.split('/')),
             'source_path')
        p=Path(root)/rel;regular_chain(p)
        need(isinstance(want,str) and len(want)==64 and digest(p)==want,'source_hash_changed')
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def validate_profile(config):
    need(config.get('schema')==1,'profile_schema')
    need(config.get('shop')=='cartproof.myshopify.com','shop_not_fixed')
    need(config.get('api_version')=='2026-07','api_version_not_fixed')
    need(config.get('model')=='muse-spark-1.3','model_not_fixed')
    need(config.get('provider_origin')=='https://api.meta.ai/v1','provider_origin')
    need(config.get('mode')=='fixed_joint_demo_controlled_drop','mode_not_fixed')
    need(config.get('request_cap')==20,'request_cap')
    need(config.get('max_calls')==6,'max_calls')
    need(config.get('token_budget')==100000,'token_budget')
    need(config.get('max_output_tokens')==8192,'max_output_tokens')
    need(config.get('timeout_seconds')==360,'timeout')
    need(config.get('reasoning_effort')=='low','reasoning_effort')
    need(config.get('pricing_as_of')=='2026-11-04','pricing_date')
    return config

def verify_offline_receipt(receipt,source_digest,test_digest,profile_digest,require_ui=False):
    need(receipt.get('complete') is True and receipt.get('real_dependency_import') is True,
         'offline_real_closure_required')
    need(receipt.get('source_digest')==source_digest and receipt.get('tests_sha256')==test_digest
         and receipt.get('profile_sha256')==profile_digest,'offline_identity_mismatch')
    need(receipt.get('network_mode')=='mock_only' and receipt.get('live_requests')==0
         and receipt.get('network_denied') is True,
         'offline_network_mode')
    need(receipt.get('tests_failed')==0 and receipt.get('tests_run',0)>=48,'offline_tests_failed')
    need(receipt.get('mcp_stub_used') is False,'mcp_stub_refused')
    need(receipt.get('platform')=='win32','offline_windows_required')
    if require_ui:
        need(receipt.get('ui_route') is True,'ui_route_unverified')

def reserve_attempt(parent,name):
    need(isinstance(name,str) and len(name)==16 and all(c in '0123456789abcdef' for c in name),
         'attempt_id')
    base=Path(parent);need(base.is_dir(),'control_parent_missing')
    regular_chain(base/'ownership.json')
    out=base/name
    out.mkdir(exist_ok=False)
    # No reuse even after failure. Parent must never remove this reservation.
    ownership={'schema':1,'reserved':True,'attempt_id':name,'pid':os.getpid()}
    with (out/'ownership.json').open('x',encoding='utf-8') as f:
        json.dump(ownership,f);f.flush();os.fsync(f.fileno())
    return out

def wait_release(directory, *, timeout=30,clock=time.monotonic,sleep=time.sleep):
    d=Path(directory);regular_chain(d/'ownership.json')
    write_fresh(d/'barrier-ready.json',{'ready':True})
    end=clock()+timeout
    while not (d/'barrier-release').exists():
        need(clock()<end,'startup_barrier_timeout');sleep(.05)
    regular_chain(d/'barrier-release')
    need((d/'barrier-release').read_text(encoding='utf-8')=='release','startup_barrier_invalid')

def supervise(directory,env,*,lifecycle=None,offline_probe=False):
    # Real implementation reuses exact reviewed Windows suspended-create and
    # assign-job-before-resume ownership, never bare Popen.
    need(sys.platform=='win32','windows_supervision_required')
    if lifecycle is None:
        from harness.model_smoke import _owned_lifecycle
        lifecycle=_owned_lifecycle
    d=Path(directory);regular_chain(d/'ownership.json')
    ctx={'rundir':d,'timeout_seconds':360.0}
    info={'cleanup_status':'not_started','timed_out':False,'job_assigned':False}
    barrier={'ready':str(d/'barrier-ready.json'),'release':str(d/'barrier-release')}
    cmd=[sys.executable,'-B','-m','harness.joint_demo_worker','--worker-dir',str(d)]
    if offline_probe:cmd.append('--offline-owned-probe')
    # Explicit child environment. No env/config values are printed or saved.
    names=('PATH','SYSTEMROOT','SYSTEMDRIVE','WINDIR','TEMP','TMP','COMSPEC','PATHEXT',
           'USERPROFILE','LOCALAPPDATA','APPDATA','VIRTUAL_ENV','MODEL_API_KEY',
           'CARTPROOF_SHOPIFY_STOREFRONT_TOKEN','CARTPROOF_SHOPIFY_STOREFRONT_TOKEN_FILE')
    allowed={k:env[k] for k in names if k in env and (not offline_probe or k not in ('MODEL_API_KEY','CARTPROOF_SHOPIFY_STOREFRONT_TOKEN','CARTPROOF_SHOPIFY_STOREFRONT_TOKEN_FILE'))}
    allowed['PYTHONDONTWRITEBYTECODE']='1'
    allowed['CARTPROOF_JOINT_PARENT_LAUNCH']='reviewed-parent-v1'
    block='\0'.join(k+'='+v for k,v in sorted(allowed.items(),key=lambda x:x[0].lower()))+'\0\0'
    run,info=lifecycle(ctx,cmd,block,info,barrier)
    # Exact successful result + verified ownership closure, not just exit code.
    need(info.get('job_assigned') is True and info.get('cleanup_status')=='verified'
         and info.get('timed_out') is False,'worker_cleanup_unverified')
    need(isinstance(run,dict) and run.get('complete') is True,'worker_not_complete')
    return run,info
