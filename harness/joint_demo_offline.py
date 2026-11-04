"""Run exact offline suite and write a receipt only with real Windows closure.

No live transports; tests inject both HTTP surfaces. Not an approval for live.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,unittest
from pathlib import Path
from harness.joint_demo_control import digest,verify_source,regular_chain,need
ROOT=Path(__file__).absolute().parents[1]

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--validation-id',required=True)
    args=parser.parse_args(argv)
    stage='offline_preflight'
    try:
        need(len(args.validation_id)==16 and all(c in '0123456789abcdef' for c in args.validation_id),
             'validation_id')
        need('CARTPROOF_INTERNAL_MCP_IMPORT_STUB' not in os.environ,'internal_stub_refused')
        # Actual imports before tests, so no fake MCP package can pass closure.
        from mcp.server import MCPServer
        import importlib.metadata as metadata
        pins={}
        for line in (ROOT/'requirements.txt').read_text(encoding='utf-8').splitlines():
            value=line.split('#')[0].strip()
            if not value or '==' not in value:continue
            if ';' in value:
                value,marker=value.split(';',1)
                if 'win32' in marker and sys.platform!='win32':continue
            name,version=value.strip().split('==')
            need(metadata.version(name)==version,'installed_pin_mismatch')
            pins[name]=version
        manifest=json.loads((ROOT/'harness/joint_demo_source_manifest.json').read_text())
        source=verify_source(ROOT,manifest)
        profile=ROOT/'harness/joint_demo_profile.json'
        sys.path.insert(0,str(ROOT/'tests'))
        suite=unittest.defaultTestLoader.loadTestsFromNames(['test_joint_demo','test_joint_demo_control','test_joint_demo_parent','test_demo_ui'])
        from harness.joint_demo_offline_guard import offline_boundary
        stage='offline_suite_network_denied'
        with offline_boundary() as blocked:
            result=unittest.TextTestRunner(verbosity=2).run(suite)
        # Boundary regressions intentionally exercise denied calls. No real
        # transport can send; attempted calls are separately counted.
        blocked_count=len(blocked)
        need(result.wasSuccessful(),'offline_suite_failed')
        need(sys.platform=='win32' and not result.skipped,'windows_lifecycle_not_verified')
        # Browser proof (after offline_boundary so loopback is available): one
        # automated single-page run. Receipt is only minted after it passes.
        import hashlib as _hl
        from harness import demo_browser as _db
        from harness.demo_ui import REQUIRED_BROWSER_CHECKS
        screens_dir=ROOT/'var/demo-ui-screenshots'
        browser_report=_db.run_page_test(ROOT,screens_dir)
        browser_report_path=screens_dir/'browser-report.json'
        need(browser_report_path.is_file(),'browser_report_missing')
        need(set(browser_report.get('checks') or []) >= REQUIRED_BROWSER_CHECKS,
             'browser_named_checks_missing')
        network=browser_report.get('network') or {}
        parent_canary=(network.get('parent') or {}).get('egress_canary') or {}
        for key in ('dns_egress','numeric_ip_egress','connect_egress','nonowned_loopback','httpx'):
            need(parent_canary.get(key,0)>=1,'browser_parent_canary_missing:'+key)
        need(parent_canary.get('token_fallback') is True and parent_canary.get('dotenv_fallback') is True,
             'browser_parent_credential_canary_missing')
        browser_canary=(network.get('browser') or {}).get('blocked_egress_attempts') or []
        need(bool(browser_canary),'browser_egress_canary_missing')
        worker_net=network.get('worker') or {}
        need(worker_net.get('network_denied') is True,'browser_worker_network_unverified')
        need((worker_net.get('egress_canary_blocked') or {}).get('socket',0)>=1,
             'browser_worker_canary_missing')
        cleanup=browser_report.get('cleanup') or {}
        need(cleanup.get('verified') is True,'browser_cleanup_unverified')
        need(bool(cleanup.get('browser_closes')) and all(c.get('ok') for c in cleanup['browser_closes']),
             'browser_close_unverified')
        shots=browser_report.get('screenshots') or {}
        for name in ('initial.png','partial.png','final.png','failure.png','disconnection.png'):
            need(shots.get(name)==_hl.sha256((screens_dir/name).read_bytes()).hexdigest(),
                 'browser_screenshot_hash_mismatch')
        browser_report_sha=_hl.sha256(browser_report_path.read_bytes()).hexdigest()
        browser_harness_sha=_hl.sha256((ROOT/'harness/demo_browser.py').read_bytes()).hexdigest()
        template_sha=_hl.sha256((ROOT/'harness/demo_templates/demo_cart.html').read_bytes()).hexdigest()
        browser_bindings={'report_sha256':browser_report_sha,'harness_sha256':browser_harness_sha,
            'template_sha256':template_sha,'screenshots':shots,
            'checks':sorted(browser_report['checks']),
            'network':{'parent':(network.get('parent') or {}),
                       'browser_blocked_egress_attempts':browser_canary,
                       'worker':worker_net},
            'cleanup_verified':True,'browser_closes':cleanup.get('browser_closes')}
        receipt={'schema':1,'complete':True,'real_dependency_import':True,'mcp_stub_used':False,
            'source_digest':source,'profile_sha256':digest(profile),
            'tests_sha256':hashlib.sha256((digest(ROOT/'tests/test_joint_demo.py')+
                 digest(ROOT/'tests/test_joint_demo_control.py')+digest(ROOT/'tests/test_joint_demo_parent.py')+
                 digest(ROOT/'tests/test_demo_ui.py')).encode()).hexdigest(),
            'network_mode':'mock_only','live_requests':0,'network_denied':True,
            'blocked_network_attempts':blocked_count,'validation_id':args.validation_id,'tests_failed':0,
            'tests_run':result.testsRun,'platform':sys.platform,
            'windows_owned_lifecycle_verified':True,'ui_route':True,'browser_proof':True,
            'browser_report_sha256':browser_report_sha,'browser_harness_sha256':browser_harness_sha,
            'browser':browser_bindings,'installed_pins':pins}
        directory=ROOT/'var/joint-demo-control'
        directory.mkdir(parents=True,exist_ok=True)
        owner=directory/'ownership.json'
        if not owner.exists():
            with owner.open('x',encoding='utf-8') as f:json.dump({'offline_control':True},f)
        regular_chain(owner)
        target=directory/('offline-receipt-'+source[:12]+'-'+args.validation_id+'.json')
        need(not target.exists(),'validation_receipt_exists_re_review_required')
        with target.open('x',encoding='utf-8') as f:
            json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
        print(json.dumps({'status':'OFFLINE_VERIFIED','live_requests':0,
            'validation_id':args.validation_id,'receipt_filename':target.name}))
        return 0
    except BaseException as exc:
        from harness.joint_demo_candidate import DemoStop
        reason=str(exc) if isinstance(exc,DemoStop) else 'offline_closure_unverified'
        print(json.dumps({'status':'STOP','stage':stage,'reason':reason,'live_requests':0}))
        return 2
if __name__=='__main__':raise SystemExit(main())
