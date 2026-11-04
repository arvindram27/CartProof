"""Fixed joint-demo parent candidate. Live command needs separate owner approval.

No consent is inferred from flags, source manifests or offline receipts.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
from harness.joint_demo_control import (digest,regular_chain,validate_profile,
    verify_source,verify_offline_receipt,reserve_attempt,supervise,write_fresh)
from harness.joint_demo_candidate import need,DemoStop
ROOT=Path(__file__).absolute().parents[1]

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--attempt',required=True)
    parser.add_argument('--validation-id',required=True)
    parser.add_argument('--allow-live',action='store_true')
    args=parser.parse_args(argv)
    attempt=None
    stage="parent_preflight"
    try:
        need(args.allow_live,'explicit_live_flag_required')
        need(sys.platform=='win32','windows_only')
        need(os.environ.get('CARTPROOF_INTERNAL_MCP_IMPORT_STUB') is None,'internal_stub_env_refused')
        need(Path(sys.executable).resolve().is_relative_to((ROOT/'.venv').resolve()),
             'reviewed_project_interpreter_required')
        source_path=ROOT/'harness/joint_demo_source_manifest.json'
        profile_path=ROOT/'harness/joint_demo_profile.json'
        need(len(args.validation_id)==16 and all(c in '0123456789abcdef' for c in args.validation_id),'validation_id')
        for p in (source_path,profile_path):regular_chain(p)
        stage='source_verify'
        source=verify_source(ROOT,json.loads(source_path.read_text(encoding='utf-8')))
        receipt_path=ROOT/'var/joint-demo-control'/('offline-receipt-'+source[:12]+'-'+args.validation_id+'.json')
        regular_chain(receipt_path)
        profile=validate_profile(json.loads(profile_path.read_text(encoding='utf-8')))
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
        tests_digest=hashlib.sha256((digest(ROOT/'tests/test_joint_demo.py')+
                    digest(ROOT/'tests/test_joint_demo_control.py')+digest(ROOT/'tests/test_joint_demo_parent.py')).encode()).hexdigest()
        stage='offline_receipt_verify'
        verify_offline_receipt(receipt,source,tests_digest,digest(profile_path))
        need(receipt.get('validation_id')==args.validation_id,'validation_identity_mismatch')
        need(receipt.get('windows_owned_lifecycle_verified') is True,
             'windows_owned_lifecycle_unverified')
        # Fresh reservation before any credentials are read or transmitted.
        stage='attempt_reserve'
        attempt=reserve_attempt(ROOT/'var/joint-demo-control',args.attempt)
        write_fresh(attempt/'profile.json',profile)
        write_fresh(attempt/'run-context.json',{'schema':1,'source_digest':source,
            'mode':'real_model_and_shopify','controlled_drop':True,
            'approval_is_external_to_artifact':True})
        stage='owned_worker'
        result,lifecycle=supervise(attempt,os.environ)
        # Parent checks the exact evidence closure after the worker is gone.
        stage='parent_evidence_closure'
        manifest_path=attempt/'evidence/manifest.json';regular_chain(manifest_path)
        entries=json.loads(manifest_path.read_text(encoding='utf-8'))
        need(isinstance(entries,dict) and bool(entries),'evidence_manifest_missing')
        actual={p.relative_to(attempt/'evidence').as_posix() for p in (attempt/'evidence').rglob('*')
                if p.is_file() and p.name!='manifest.json'}
        need(actual==set(entries),'evidence_closure_changed')
        for rel,want in entries.items():
            need(not rel.startswith(('/','\\')) and '..' not in rel.split('/')
                 and ':' not in rel and '\\' not in rel,'evidence_path')
            p=attempt/'evidence'/rel;regular_chain(p);need(digest(p)==want,'evidence_hash_changed')
        need(result.get('run_context')=='real_model_and_shopify','worker_context')
        write_fresh(attempt/'parent-result.json',{'complete':True,'source_digest':source,
                    'lifecycle':lifecycle,'worker_result':result})
        print('{"status":"COMPLETE","live_run":true,"retry":false}')
        return 0
    except BaseException as exc:
        # Fixed taxonomy and byte/hash diagnostics only, never raw output.
        cls=type(exc).__name__ if type(exc).__module__=='builtins' else 'GateError'
        reason=str(exc) if isinstance(exc,DemoStop) else 'parent_gate_or_outcome_unverified'
        report={'status':'STOP','stage':stage,'reason':reason,'error_class':cls,'retry':False}
        if attempt is not None:
            diagnostics={}
            for name in ('stdout.txt','stderr.txt','run.json'):
                p=attempt/name
                if p.is_file():
                    raw=p.read_bytes();diagnostics[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            report['diagnostics']=diagnostics
            try:write_fresh(attempt/'parent-failure.json',report)
            except Exception:report['failure_report_write']='failed'
        print(json.dumps(report,sort_keys=True))
        return 2
if __name__=='__main__':raise SystemExit(main())
