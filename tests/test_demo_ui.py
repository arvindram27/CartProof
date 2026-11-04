"""Offline tests for the same-cart local shopping UI worker, control, CLI and
the checked owner-launch path.

Uses injected fake model/Shopify transports under the installed network-denied
boundary; never real credentials or network. LABELED OFFLINE / FAKE TRANSPORTS.
"""
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import test_joint_demo as _tj
from harness import joint_demo_candidate as jdc
from harness.joint_demo_candidate import DemoStop
from harness import demo_worker as dw
from harness import demo_ui as du

PROFILE = json.loads((ROOT / "harness/joint_demo_profile.json").read_text(encoding="utf-8"))
MAPPING = {"variants": {
    "CARTPROOF-TEST-A": "gid://shopify/ProductVariant/1",
    "CARTPROOF-TEST-B": "gid://shopify/ProductVariant/2"}}


def _rundir(tmp):
    d = Path(tmp)
    (d / "profile.json").write_text(json.dumps(PROFILE), encoding="utf-8")
    (d / "mapping.json").write_text(json.dumps(MAPPING), encoding="utf-8")
    return d


def _launch_parent():
    p = Path(tempfile.mkdtemp(prefix="cartproof-test-parent-"))
    (p / "ownership.json").write_text(json.dumps({"offline_control": True}), encoding="utf-8")
    return p


def _snap(**over):
    base = {"schema": "cartproof-demo-ui-v1", "status": "complete", "phase": "complete",
            "request": {"alias": "A", "quantity": 1}, "target": {"A": 2, "B": 1},
            "observed": {"A": 2, "B": 1}, "verdict": "COMPLETE", "missing": {},
            "action": "add_to_cart", "controlled_drop": False,
            "controlled_drop_no_request": True, "recovery": True,
            "read_seq": 9, "checked_read_seq": 9, "event_seq": 8,
            "read_at_ms": 100, "check_at_ms": 120, "model_calls": 5,
            "shopify_requests": 16, "saved_check": None, "freshness": 40}
    base.update(over)
    return base


class SnapshotAllowlist(unittest.TestCase):
    def _valid(self):
        return {"schema": "cartproof-demo-ui-v1", "status": "running",
                "phase": "controlled_drop", "request": {"alias": "A", "quantity": 1},
                "target": {"A": 2, "B": 1}, "observed": {"A": 1, "B": 1},
                "verdict": "PARTIAL", "missing": {"A": 1}, "action": "add_to_cart",
                "controlled_drop": False, "controlled_drop_no_request": True,
                "recovery": False, "read_seq": 3, "checked_read_seq": 3, "event_seq": 2,
                "read_at_ms": 1, "check_at_ms": 2, "model_calls": 2,
                "shopify_requests": 4, "saved_check": None, "freshness": 5}

    def test_accepts_valid_snapshot(self):
        self.assertIsNotNone(du.validate_snapshot(self._valid()))

    def test_rejects_unknown_key(self):
        s = self._valid(); s["evil"] = "x"
        self.assertIsNone(du.validate_snapshot(s))

    def test_rejects_unknown_alias(self):
        s = self._valid(); s["observed"] = {"A": 1, "C": 2}
        self.assertIsNone(du.validate_snapshot(s))

    def test_rejects_bad_quantity(self):
        for bad in (True, -1, "2", 100, 1.5):
            s = self._valid(); s["observed"] = {"A": bad, "B": 1}
            self.assertIsNone(du.validate_snapshot(s), bad)

    def test_rejects_bad_verdict(self):
        s = self._valid(); s["verdict"] = "COMPLETE_BAD"
        self.assertIsNone(du.validate_snapshot(s))

    def test_rejects_bad_phase(self):
        s = self._valid(); s["phase"] = "hacked"
        self.assertIsNone(du.validate_snapshot(s))

    def test_rejects_bad_request(self):
        s = self._valid(); s["request"] = {"alias": "A", "quantity": 2}
        self.assertIsNone(du.validate_snapshot(s))

    def test_valid_complete_accepted(self):
        self.assertIsNotNone(du.validate_snapshot(_snap()))

    def test_complete_with_null_maps_rejected(self):
        self.assertIsNone(du.validate_snapshot(_snap(target=None, observed=None)))

    def test_complete_mismatch_rejected(self):
        self.assertIsNone(du.validate_snapshot(_snap(target={"A": 1, "B": 1})))

    def test_complete_with_missing_rejected(self):
        self.assertIsNone(du.validate_snapshot(_snap(missing={"A": 1})))

    def test_status_complete_with_partial_rejected(self):
        self.assertIsNone(du.validate_snapshot(_snap(verdict="PARTIAL")))

    def test_status_complete_with_null_verdict_rejected(self):
        self.assertIsNone(du.validate_snapshot(_snap(verdict=None)))

    def test_complete_checked_read_mismatch_rejected(self):
        self.assertIsNone(du.validate_snapshot(_snap(checked_read_seq=8)))

    def test_complete_bad_times_rejected(self):
        self.assertIsNone(du.validate_snapshot(_snap(check_at_ms=0)))
        self.assertIsNone(du.validate_snapshot(_snap(read_at_ms=0)))

    def test_failed_status_with_verdict_rejected(self):
        self.assertIsNone(du.validate_snapshot(_snap(status="failed")))


class SnapshotWriterState(unittest.TestCase):
    """B: current COMPLETE never survives a newer read / delayed old check /
    wrong cart / missing maps / sqlite resolution."""

    def _writer(self, tmp):
        be = types.SimpleNamespace(request_count=0, _scan_unsafe=lambda s: False)
        mo = types.SimpleNamespace(calls_made=0)
        aliases = {"CARTPROOF-TEST-A": "A", "CARTPROOF-TEST-B": "B"}
        return dw.SnapshotWriter(Path(tmp), aliases, be, mo)

    def _state(self, lines, cart_ref="ref-1", resolution="remote-cart"):
        return {"lines": lines, "resolution": resolution,
                "provenance": {"cart_ref": cart_ref}}

    def _joint(self, verdict, expected):
        return types.SimpleNamespace(
            results=[{"verdict": verdict, "expected_lines": expected}],
            phase="controlled_drop")

    def _snap(self, tmp):
        return json.loads((Path(tmp) / "snapshot.json").read_text(encoding="utf-8"))

    def _check_row(self, w):
        return {"event": "independent_check", "read_token": w.current_read_token()}

    def test_new_read_clears_complete_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = self._writer(tmp)
            w.on_read(self._state({"CARTPROOF-TEST-A": 2, "CARTPROOF-TEST-B": 1}))
            w.on_event(self._check_row(w),
                       self._joint("COMPLETE", {"CARTPROOF-TEST-A": 2, "CARTPROOF-TEST-B": 1}))
            self.assertEqual(self._snap(tmp)["verdict"], "COMPLETE")
            w.on_read(self._state({"CARTPROOF-TEST-A": 1, "CARTPROOF-TEST-B": 1}))
            snap = self._snap(tmp)
            self.assertIsNone(snap["verdict"])
            self.assertIsNone(snap["target"])
            self.assertEqual(snap["read_seq"], 2)

    def test_delayed_old_check_with_same_quantities_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = self._writer(tmp)
            w.on_read(self._state({"CARTPROOF-TEST-A": 2, "CARTPROOF-TEST-B": 1}))
            old_token = w.current_read_token()
            w.on_event({"event": "independent_check", "read_token": old_token},
                       self._joint("COMPLETE", {"CARTPROOF-TEST-A": 2, "CARTPROOF-TEST-B": 1}))
            self.assertEqual(self._snap(tmp)["verdict"], "COMPLETE")
            # A NEW read with the SAME quantities gets a NEW token.
            w.on_read(self._state({"CARTPROOF-TEST-A": 2, "CARTPROOF-TEST-B": 1}))
            self.assertIsNone(self._snap(tmp)["verdict"])
            # Replay the OLD check (same quantities, older read token).
            w.on_event({"event": "independent_check", "read_token": old_token},
                       self._joint("COMPLETE", {"CARTPROOF-TEST-A": 2, "CARTPROOF-TEST-B": 1}))
            snap = self._snap(tmp)
            self.assertNotEqual(snap["verdict"], "COMPLETE")

    def test_old_check_from_different_cart_same_quantities_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = self._writer(tmp)
            w.on_read(self._state({"CARTPROOF-TEST-A": 2, "CARTPROOF-TEST-B": 1}, cart_ref="ref-1"))
            w.on_event(self._check_row(w),
                       self._joint("COMPLETE", {"CARTPROOF-TEST-A": 2, "CARTPROOF-TEST-B": 1}))
            self.assertEqual(self._snap(tmp)["verdict"], "COMPLETE")
            # A read from a DIFFERENT cart (same quantities) is fatal: no COMPLETE
            # survives, and the attempt stops rather than resetting identity.
            with self.assertRaises(DemoStop):
                w.on_read(self._state({"CARTPROOF-TEST-A": 2, "CARTPROOF-TEST-B": 1}, cart_ref="ref-2"))

    def test_missing_read_token_check_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = self._writer(tmp)
            w.on_read(self._state({"CARTPROOF-TEST-A": 2, "CARTPROOF-TEST-B": 1}))
            # A check event without a read token cannot bind to the read.
            w.on_event({"event": "independent_check"},
                       self._joint("COMPLETE", {"CARTPROOF-TEST-A": 2, "CARTPROOF-TEST-B": 1}))
            snap = self._snap(tmp)
            self.assertNotEqual(snap["verdict"], "COMPLETE")

    def test_valid_exact_current_read_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = self._writer(tmp)
            w.on_read(self._state({"CARTPROOF-TEST-A": 2, "CARTPROOF-TEST-B": 1}))
            w.on_event(self._check_row(w),
                       self._joint("COMPLETE", {"CARTPROOF-TEST-A": 2, "CARTPROOF-TEST-B": 1}))
            snap = self._snap(tmp)
            self.assertEqual(snap["verdict"], "COMPLETE")
            self.assertEqual(snap["checked_read_seq"], snap["read_seq"])

    def test_wrong_cart_raises_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = self._writer(tmp)
            w.on_read(self._state({"CARTPROOF-TEST-A": 1, "CARTPROOF-TEST-B": 1}, cart_ref="ref-1"))
            with self.assertRaises(DemoStop):
                w.on_read(self._state({"CARTPROOF-TEST-A": 1, "CARTPROOF-TEST-B": 1}, cart_ref="ref-2"))

    def test_sqlite_resolution_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = self._writer(tmp)
            with self.assertRaises(DemoStop):
                w.on_read(self._state({"CARTPROOF-TEST-A": 1, "CARTPROOF-TEST-B": 1},
                                      resolution="sqlite-snapshot"))

    def test_unknown_item_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = self._writer(tmp)
            with self.assertRaises(DemoStop):
                w.on_read(self._state({"CARTPROOF-TEST-A": 1, "FOREIGN-SKU": 1}))

    def test_missing_identity_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = self._writer(tmp)
            with self.assertRaises(DemoStop):
                w.on_read(self._state({"CARTPROOF-TEST-A": 1, "CARTPROOF-TEST-B": 1},
                                      cart_ref=None))

    def test_finish_false_clears_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = self._writer(tmp)
            w.on_read(self._state({"CARTPROOF-TEST-A": 2, "CARTPROOF-TEST-B": 1}))
            w.on_event(self._check_row(w),
                       self._joint("COMPLETE", {"CARTPROOF-TEST-A": 2, "CARTPROOF-TEST-B": 1}))
            self.assertEqual(self._snap(tmp)["verdict"], "COMPLETE")
            w.finish(False, "cancel")
            snap = self._snap(tmp)
            self.assertEqual(snap["status"], "failed")
            self.assertIsNone(snap["verdict"])


class CheckTokenSeam(unittest.TestCase):
    """The check carries the token captured for its OWN read (via run_check ->
    PublishingBackend -> JointDemo.check -> emit), never an emitter-time latest
    token. Tests go through the actual candidate check/emit seam."""

    def _seam(self, tmp):
        d = Path(tmp)
        j, f, calls, shown = _tj.Case().fixture(d)
        aliases = {"CARTPROOF-TEST-A": "A", "CARTPROOF-TEST-B": "B"}
        writer = dw.SnapshotWriter(d, aliases, j.backend, j.model)
        pbackend = dw.PublishingBackend(j.backend, writer.on_read)
        writer.write()
        j.backend = pbackend
        j.validate()
        j.mutate(j.b)  # seed {B:1}
        state = j.make_check_state("Add 1 x CARTPROOF-TEST-A.", "matching")
        j.mutate(j.a)  # {A:1, B:1}
        return d, j, writer, pbackend, state

    def test_delayed_check_same_quantities_through_seam(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, j, writer, pbackend, state = self._seam(tmp)
            captured = {}

            def emit(line):
                row = json.loads(line)
                if row.get("event") == "independent_check" and "token" not in captured:
                    captured["token"] = row.get("read_token")
                    # read2 with the SAME quantities before delivering check1
                    pbackend.acquire_state()
                writer.on_event(row, j)
            j.emit = emit
            j.check(state, {"CARTPROOF-TEST-A": 1, "CARTPROOF-TEST-B": 1},
                    "matching", "COMPLETE")
            snap = json.loads((d / "snapshot.json").read_text(encoding="utf-8"))
            self.assertIsNotNone(captured.get("token"))
            self.assertNotEqual(snap.get("verdict"), "COMPLETE")

    def test_valid_check_exact_read_through_seam(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, j, writer, pbackend, state = self._seam(tmp)
            j.emit = lambda line: writer.on_event(json.loads(line), j)
            j.check(state, {"CARTPROOF-TEST-A": 1, "CARTPROOF-TEST-B": 1},
                    "matching", "COMPLETE")
            snap = json.loads((d / "snapshot.json").read_text(encoding="utf-8"))
            self.assertEqual(snap.get("verdict"), "COMPLETE")

    def test_missing_token_through_seam(self):
        # Raw backend (no PublishingBackend) injects no token -> read_token None.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            j, f, calls, shown = _tj.Case().fixture(d)
            aliases = {"CARTPROOF-TEST-A": "A", "CARTPROOF-TEST-B": "B"}
            writer = dw.SnapshotWriter(d, aliases, j.backend, j.model)
            writer.write()
            j.validate()
            j.mutate(j.b)
            state = j.make_check_state("Add 1 x CARTPROOF-TEST-A.", "matching")
            j.mutate(j.a)
            j.emit = lambda line: writer.on_event(json.loads(line), j)
            j.check(state, {"CARTPROOF-TEST-A": 1, "CARTPROOF-TEST-B": 1},
                    "matching", "COMPLETE")
            snap = json.loads((d / "snapshot.json").read_text(encoding="utf-8"))
            self.assertNotEqual(snap.get("verdict"), "COMPLETE")

    def test_seam_adds_no_extra_shopify_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, j, writer, pbackend, state = self._seam(tmp)
            before = j.backend.request_count
            j.emit = lambda line: writer.on_event(json.loads(line), j)
            j.check(state, {"CARTPROOF-TEST-A": 1, "CARTPROOF-TEST-B": 1},
                    "matching", "COMPLETE")
            # The check performs exactly the reads run_check already does; the
            # token seam adds zero provider requests.
            self.assertEqual(j.backend.request_count - before, 1)


class WorkerSequence(unittest.TestCase):
    def setUp(self):
        os.environ["CARTPROOF_DEMO_HOLD"] = "0"

    def tearDown(self):
        os.environ.pop("CARTPROOF_DEMO_HOLD", None)

    def test_offline_sequence_completes_final_A2_B1(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = _rundir(tmp)
            run = dw.run_worker_dir(d, offline=True)
            self.assertTrue(run["complete"])
            self.assertEqual(run["model_calls"], 5)
            self.assertEqual(run["run_context"], "offline_injected")
            snap = json.loads((d / "snapshot.json").read_text(encoding="utf-8"))
            self.assertEqual(snap["status"], "complete")
            self.assertEqual(snap["verdict"], "COMPLETE")
            self.assertEqual(snap["observed"], {"A": 2, "B": 1})
            self.assertEqual(snap["target"], {"A": 2, "B": 1})
            self.assertEqual(snap["missing"], {})
            self.assertEqual(snap["checked_read_seq"], snap["read_seq"])
            self.assertTrue(snap["controlled_drop_no_request"])
            self.assertEqual(snap["saved_check"]["verdict"], "COMPLETE")

    def test_offline_partial_visible_during_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = _rundir(tmp)
            os.environ["CARTPROOF_DEMO_HOLD"] = "3"
            result = {}

            def go():
                result["run"] = dw.run_worker_dir(d, offline=True)
            t = threading.Thread(target=go, daemon=True)
            t.start()
            seen_partial = False
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and t.is_alive():
                if (d / "snapshot.json").is_file():
                    s = json.loads((d / "snapshot.json").read_text(encoding="utf-8"))
                    if s.get("verdict") == "PARTIAL":
                        seen_partial = True
                        self.assertEqual(s["observed"], {"A": 1, "B": 1})
                        self.assertEqual(s["target"], {"A": 2, "B": 1})
                        self.assertEqual(s["missing"], {"A": 1})
                        break
                time.sleep(0.02)
            t.join(timeout=30)
            self.assertTrue(seen_partial, "PARTIAL state was never observed")

    def test_snapshot_has_no_raw_sku_or_cart_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = _rundir(tmp)
            dw.run_worker_dir(d, offline=True)
            raw = (d / "snapshot.json").read_text(encoding="utf-8")
            for canary in ("CARTPROOF-TEST-", "gid://shopify/", "cartref:", "?key=",
                           "Cart/"):
                self.assertNotIn(canary, raw)

    def test_api_only_default_gate_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, f, calls, shown = _tj.Case().fixture(tmp)
            r = j.execute()
            self.assertTrue(r["complete"], r)
            self.assertEqual(r["model_calls"], 5)


class DemoControllerOwned(unittest.TestCase):
    def setUp(self):
        os.environ["CARTPROOF_DEMO_HOLD"] = "0"

    def tearDown(self):
        os.environ.pop("CARTPROOF_DEMO_HOLD", None)
        os.environ.pop("CARTPROOF_DEMO_BLOCK", None)

    def _ctl(self, parent, **kw):
        return du.DemoController(mode="offline", root=ROOT, port=8765, hold="0",
                                 offline_mapping=MAPPING, launch_parent=parent, **kw)

    def test_offline_start_completes_and_cleans_up(self):
        parent = _launch_parent()
        ctl = self._ctl(parent)
        r = ctl.start()
        self.assertTrue(r["started"])
        ctl.thread.join(timeout=90)
        self.assertTrue(ctl.finished)
        self.assertTrue(ctl.info.get("job_assigned"))
        self.assertEqual(ctl.info.get("cleanup_status"), "verified")
        self.assertEqual(ctl.info.get("active_before_close"), 0)
        self.assertIsNotNone(ctl.run)
        self.assertTrue(ctl.run.get("complete"))
        self.assertTrue(ctl.run.get("network_denied"))
        # exactly one owned attempt with correct controls
        attempts = [p for p in parent.iterdir() if p.is_dir()]
        self.assertEqual(len(attempts), 1)
        ctx = json.loads((attempts[0] / "run-context.json").read_text(encoding="utf-8"))
        self.assertEqual(ctx["route"], "cartproof-demo-ui-v1")
        self.assertEqual(ctx["mode"], "offline")
        snap = ctl.snapshot()
        self.assertEqual(snap["status"], "complete")
        self.assertEqual(snap["verdict"], "COMPLETE")

    def test_single_start_refused(self):
        parent = _launch_parent()
        ctl = self._ctl(parent)
        self.assertTrue(ctl.start()["started"])
        self.assertFalse(ctl.start()["started"])
        ctl.thread.join(timeout=90)

    def test_concurrent_start_single_launch(self):
        parent = _launch_parent()
        ctl = self._ctl(parent)
        results = []

        def go():
            results.append(ctl.start())
        t1 = threading.Thread(target=go); t2 = threading.Thread(target=go)
        t1.start(); t2.start(); t1.join(); t2.join()
        self.assertEqual(sum(1 for r in results if r.get("started")), 1)
        ctl.thread.join(timeout=90)

    def test_cancel_terminates_stuck_worker(self):
        parent = _launch_parent()
        os.environ["CARTPROOF_DEMO_BLOCK"] = "1"
        ctl = self._ctl(parent, timeout_s=30)
        ctl.start()
        time.sleep(2.0)
        stopped = ctl.stop()
        self.assertTrue(stopped["stopped"], stopped)
        self.assertEqual(ctl.info.get("cleanup_status"), "terminated")
        self.assertEqual(ctl.info.get("active_after_terminate"), 0)
        snap = ctl.snapshot()
        self.assertEqual(snap["status"], "failed")
        self.assertIsNone(snap["verdict"])

    def test_snapshot_polling_adds_no_shopify_requests(self):
        parent = _launch_parent()
        ctl = self._ctl(parent)
        ctl.start()
        ctl.thread.join(timeout=90)
        before = ctl.snapshot()["shopify_requests"]
        for _ in range(5):
            ctl.snapshot()
        self.assertEqual(ctl.snapshot()["shopify_requests"], before)

    def test_snapshot_regression_rejected(self):
        parent = _launch_parent()
        ctl = self._ctl(parent)
        attempt = Path(tempfile.mkdtemp(prefix="cartproof-fake-attempt-"))
        ctl.ctx = {"attempt_dir": attempt}

        def write_snap(freshness, read_seq):
            snap = _snap(status="running", verdict=None, target=None,
                         observed=None, missing={}, freshness=freshness,
                         read_seq=read_seq, checked_read_seq=0,
                         read_at_ms=100, check_at_ms=120)
            (attempt / "snapshot.json").write_text(json.dumps(snap), encoding="utf-8")

        write_snap(10, 5)
        self.assertEqual(ctl.snapshot()["freshness"], 10)
        write_snap(5, 3)  # regressed freshness/read_seq
        out = ctl.snapshot()
        self.assertIn(out["status"], ("unavailable", "unknown"))
        self.assertIsNone(out["verdict"])


class DemoUISecurity(unittest.TestCase):
    BOUND = "http://127.0.0.1:8765"

    def setUp(self):
        os.environ["CARTPROOF_DEMO_HOLD"] = "0"

    def tearDown(self):
        os.environ.pop("CARTPROOF_DEMO_HOLD", None)

    def test_authorize_start_rejects_foreign_origin(self):
        err = du.authorize_start({"host": "127.0.0.1:8765",
                                  "origin": "https://evil.example",
                                  "x-demo-csrf": "n"}, "n", self.BOUND)
        self.assertEqual(err, "origin_rejected")

    def test_authorize_start_rejects_null_origin(self):
        err = du.authorize_start({"host": "127.0.0.1:8765",
                                  "x-demo-csrf": "n"}, "n", self.BOUND)
        self.assertEqual(err, "origin_rejected")

    def test_authorize_start_rejects_prefix_spoofed_origin(self):
        for spoof in ("http://localhost.attacker", "http://127.0.0.1.attacker",
                      "http://127.0.0.1:87650", "http://localhost:8765"):
            err = du.authorize_start({"host": "127.0.0.1:8765", "origin": spoof,
                                      "x-demo-csrf": "n"}, "n", self.BOUND)
            self.assertEqual(err, "origin_rejected", spoof)

    def test_authorize_start_rejects_foreign_host(self):
        err = du.authorize_start({"host": "evil.example",
                                  "origin": "http://127.0.0.1:8765",
                                  "x-demo-csrf": "n"}, "n", self.BOUND)
        self.assertEqual(err, "origin_rejected")

    def test_authorize_start_rejects_csrf(self):
        err = du.authorize_start({"host": "127.0.0.1:8765",
                                  "origin": "http://127.0.0.1:8765",
                                  "x-demo-csrf": "wrong"}, "right", self.BOUND)
        self.assertEqual(err, "csrf_rejected")

    def test_authorize_start_rejects_forwarding(self):
        for h in ("x-forwarded-for", "x-forwarded-host", "x-forwarded-proto",
                  "cf-connecting-ip"):
            err = du.authorize_start({"host": "127.0.0.1:8765",
                                      "origin": "http://127.0.0.1:8765",
                                      "x-demo-csrf": "n", h: ""}, "n", self.BOUND)
            self.assertEqual(err, "forwarding_header_rejected", h)

    def test_authorize_start_accepts_loopback(self):
        err = du.authorize_start({"host": "127.0.0.1:8765",
                                  "origin": "http://127.0.0.1:8765",
                                  "x-demo-csrf": "n"}, "n", self.BOUND)
        self.assertIsNone(err)

    def test_no_get_execution_and_no_raw_mutation_api(self):
        parent = _launch_parent()
        ctl = du.DemoController(mode="offline", root=ROOT, port=8765, hold="0",
                                offline_mapping=MAPPING, launch_parent=parent)
        app = du.create_demo_app(ctl, ROOT / "harness" / "demo_templates")
        methods = {}
        for route in app.routes:
            path = getattr(route, "path", None)
            meth = getattr(route, "methods", None)
            if path is None:
                continue
            methods[path] = set(meth) if meth else None
        self.assertEqual(methods.get("/demo/start"), {"POST"})
        self.assertEqual(methods.get("/demo/stop"), {"POST"})
        self.assertEqual(methods.get("/demo/state"), {"GET"})
        self.assertIn("/", methods)
        for path in methods:
            self.assertNotIn("/api/cart", path)
            self.assertNotIn("mutation", path.lower())

    def test_index_html_has_no_raw_sku_or_capability(self):
        from fastapi.templating import Jinja2Templates
        templates = Jinja2Templates(directory=str(ROOT / "harness" / "demo_templates"))
        snap = {"status": "not_started", "phase": "not_started",
                "request": {"alias": "A", "quantity": 1}, "target": None,
                "observed": None, "verdict": None, "missing": {},
                "action": None, "controlled_drop": False,
                "controlled_drop_no_request": False, "recovery": False,
                "read_seq": 0, "checked_read_seq": 0, "event_seq": 0,
                "read_at_ms": 0, "check_at_ms": 0, "model_calls": 0,
                "shopify_requests": 0, "saved_check": None, "freshness": 0}
        html = templates.get_template("demo_cart.html").render(
            snap=snap, offline=True, csrf="test", bound_origin="http://127.0.0.1:8765")
        for canary in ("CARTPROOF-TEST-", "gid://shopify/", "?key=", "shpat_",
                       "cartref:", "Bearer "):
            self.assertNotIn(canary, html)
        # The fallback that invented COMPLETE from status is gone.
        self.assertNotIn("status === 'complete' ? 'COMPLETE'", html)


class WorkerLaunchGate(unittest.TestCase):
    def setUp(self):
        os.environ["CARTPROOF_DEMO_HOLD"] = "0"
        self._saved = {k: os.environ.get(k) for k in
                       (dw.PARENT_MARKER, dw.OFFLINE_ENV, dw.LIVE_RELEASE_ENV)}

    def tearDown(self):
        os.environ.pop("CARTPROOF_DEMO_HOLD", None)
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_worker_refuses_standalone(self):
        os.environ.pop(dw.PARENT_MARKER, None)
        os.environ.pop(dw.OFFLINE_ENV, None)
        os.environ.pop(dw.LIVE_RELEASE_ENV, None)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            rc = dw.main(['--worker-dir', str(d)])
            self.assertEqual(rc, 2)
            self.assertFalse((d / dw.RUN_NAME).exists())

    def test_worker_refuses_live_default(self):
        os.environ[dw.PARENT_MARKER] = dw.PARENT_VALUE
        os.environ.pop(dw.OFFLINE_ENV, None)
        os.environ.pop(dw.LIVE_RELEASE_ENV, None)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            rc = dw.main(['--worker-dir', str(d)])
            self.assertEqual(rc, 2)
            self.assertFalse((d / dw.RUN_NAME).exists())

    def test_worker_refuses_mixed_flags(self):
        os.environ[dw.PARENT_MARKER] = dw.PARENT_VALUE
        os.environ[dw.OFFLINE_ENV] = "1"
        os.environ[dw.LIVE_RELEASE_ENV] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            rc = dw.main(['--worker-dir', str(d)])
            self.assertEqual(rc, 2)
            self.assertFalse((d / dw.RUN_NAME).exists())

    def test_worker_malformed_controls_refuses_before_clients(self):
        os.environ[dw.PARENT_MARKER] = dw.PARENT_VALUE
        os.environ[dw.OFFLINE_ENV] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "profile.json").write_text(json.dumps(PROFILE), encoding="utf-8")
            (d / "run-context.json").write_text(json.dumps(
                {"schema": 1, "route": "wrong", "mode": "offline", "source_digest": "0" * 64}),
                encoding="utf-8")
            (d / "ownership.json").write_text("{}", encoding="utf-8")
            with patch("harness.shopify_live_backend.resolve_token") as rt, \
                 patch("harness.joint_demo_worker.construct_model") as cm:
                rc = dw.main(['--worker-dir', str(d)])
            self.assertEqual(rc, 2)
            rt.assert_not_called()
            cm.assert_not_called()

    def test_worker_forged_source_controls_refuses(self):
        os.environ[dw.PARENT_MARKER] = dw.PARENT_VALUE
        os.environ[dw.OFFLINE_ENV] = "1"
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "profile.json").write_text(json.dumps(PROFILE), encoding="utf-8")
            (d / "run-context.json").write_text(json.dumps(
                {"schema": 1, "route": "cartproof-demo-ui-v1", "mode": "offline",
                 "source_digest": "0" * 64}), encoding="utf-8")
            (d / "ownership.json").write_text("{}", encoding="utf-8")
            with patch("harness.shopify_live_backend.resolve_token") as rt, \
                 patch("harness.joint_demo_worker.construct_model") as cm:
                rc = dw.main(['--worker-dir', str(d)])
            self.assertEqual(rc, 2)
            rt.assert_not_called()
            cm.assert_not_called()

    def test_verify_receipt_require_ui(self):
        from harness.joint_demo_control import verify_offline_receipt
        r = {'complete': True, 'real_dependency_import': True,
             'source_digest': 's', 'tests_sha256': 't', 'profile_sha256': 'p',
             'network_mode': 'mock_only', 'live_requests': 0, 'network_denied': True,
             'tests_failed': 0, 'tests_run': 75, 'mcp_stub_used': False,
             'platform': 'win32', 'ui_route': True}
        verify_offline_receipt(dict(r), 's', 't', 'p', require_ui=True)
        r_no_ui = dict(r); r_no_ui.pop('ui_route')
        with self.assertRaises(DemoStop):
            verify_offline_receipt(r_no_ui, 's', 't', 'p', require_ui=True)


class LaunchPreflight(unittest.TestCase):
    """A: the real preflight/start path refuses before any forbidden stage."""

    def _temp_control(self):
        p = Path(tempfile.mkdtemp(prefix="cartproof-ctrl-"))
        (p / "ownership.json").write_text("{}", encoding="utf-8")
        return p

    def _rejected_no_spawn(self, ctl):
        with patch("harness.demo_ui._demo_env_block") as deb, \
             patch("harness.model_smoke._SuspendedProcess") as sp, \
             patch("harness.shopify_live_backend.resolve_token") as rt, \
             patch("harness.joint_demo_worker.construct_model") as cm, \
             patch("agents.env_loader.load_root_dotenv") as lrd:
            r = ctl.start()
        self.assertFalse(r["started"])
        deb.assert_not_called()
        sp.assert_not_called()
        rt.assert_not_called()
        cm.assert_not_called()
        lrd.assert_not_called()
        return r

    def test_start_live_no_receipt_zero_spawn(self):
        from harness.joint_demo_control import verify_source
        man = json.loads((ROOT / "harness/joint_demo_source_manifest.json").read_text(encoding="utf-8"))
        source = verify_source(ROOT, man)
        tests = du._compute_tests_digest(ROOT)
        parent = self._temp_control()
        ctl = du.DemoController(mode="live-release", root=ROOT, port=8765, hold="0",
                                validation_id="0" * 16, source_digest=source,
                                tests_digest=tests, launch_parent=parent,
                                control_dir=parent)
        self._rejected_no_spawn(ctl)

    def test_start_live_mismatched_source_zero_spawn(self):
        parent = self._temp_control()
        ctl = du.DemoController(mode="live-release", root=ROOT, port=8765, hold="0",
                                validation_id="0" * 16, source_digest="s",
                                tests_digest="t", launch_parent=parent,
                                control_dir=parent)
        r = self._rejected_no_spawn(ctl)
        self.assertEqual(r["reason"], "source_digest_mismatch")

    def test_start_offline_requires_mapping(self):
        parent = self._temp_control()
        ctl = du.DemoController(mode="offline", root=ROOT, port=8765, hold="0",
                                offline_mapping=None, launch_parent=parent)
        r = self._rejected_no_spawn(ctl)
        self.assertEqual(r["reason"], "offline_mapping_required")

    def test_start_invokes_prepare_launch_once(self):
        parent = self._temp_control()
        ctl = du.DemoController(mode="offline", root=ROOT, port=8765, hold="0",
                                offline_mapping=MAPPING, launch_parent=parent)
        with patch.object(du, "prepare_launch", wraps=du.prepare_launch) as pp:
            r = ctl.start()
            self.assertTrue(r["started"])
            self.assertEqual(pp.call_count, 1)
        ctl.thread.join(timeout=90)

    def test_prepare_launch_live_calls_require_ui(self):
        from harness import joint_demo_control
        from harness.joint_demo_control import verify_source, digest
        man = json.loads((ROOT / "harness/joint_demo_source_manifest.json").read_text(encoding="utf-8"))
        source = verify_source(ROOT, man)
        tests = du._compute_tests_digest(ROOT)
        profile_path = ROOT / "harness/joint_demo_profile.json"
        ctrl = self._temp_control()
        receipt = {"complete": True, "real_dependency_import": True,
                   "source_digest": source, "tests_sha256": tests,
                   "profile_sha256": digest(profile_path),
                   "network_mode": "mock_only", "live_requests": 0, "network_denied": True,
                   "tests_failed": 0, "tests_run": 80, "mcp_stub_used": False,
                   "platform": "win32", "ui_route": True, "browser_proof": True,
                   "validation_id": "0" * 16}
        rp = ctrl / ("offline-receipt-" + source[:12] + "-" + "0" * 16 + ".json")
        rp.write_text(json.dumps(receipt), encoding="utf-8")
        captured = {}

        def fake_verify(receipt, s, t, p, require_ui=False):
            captured["require_ui"] = require_ui

        with patch.object(joint_demo_control, "verify_offline_receipt", side_effect=fake_verify), \
             patch.object(du, "_verify_browser_proof", return_value={}):
            du.prepare_launch(ROOT, mode="live-release", port=8765, hold="0",
                              timeout_s=360, validation_id="0" * 16,
                              source_digest=source, tests_digest=tests,
                              control_dir=ctrl, launch_parent=ctrl)
        self.assertIs(captured["require_ui"], True)


class CliEntry(unittest.TestCase):
    def test_cli_refuses_no_mode(self):
        with patch("sys.stdout", new=io.StringIO()) as out:
            rc = du.main([])
        self.assertEqual(rc, 2)
        self.assertIn("STOP", out.getvalue())

    def test_cli_live_requires_identity(self):
        with patch("sys.stdout", new=io.StringIO()) as out:
            rc = du.main(["--mode", "live-release"])
        self.assertEqual(rc, 2)
        self.assertIn("STOP", out.getvalue())

    def _spawn(self, port, parent, block=False):
        from harness.model_smoke import _JobObject, _SuspendedProcess, _pilot_env_block, PROJECT_ROOT
        extra = ["CARTPROOF_DEMO_BLOCK=1"] if block else []
        base = _pilot_env_block()
        env_block = (base[:-1] + "\0".join(extra) + "\0\0") if extra else base
        proc = _SuspendedProcess(
            [sys.executable, "-B", "-m", "harness.demo_ui", "--mode", "offline",
             "--port", str(port), "--hold", "0", "--launch-parent", str(parent)],
            PROJECT_ROOT, parent / "stdout.txt", parent / "stderr.txt",
            env_block=env_block)
        job = _JobObject()
        job.assign(proc.handle)
        proc.resume()
        return proc, job

    def _wait_ready(self, proc, parent, timeout=30.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("module_command_exited")
            out = parent / "stdout.txt"
            if out.is_file() and "READY" in out.read_text(encoding="utf-8", errors="ignore"):
                return True
            time.sleep(0.1)
        raise RuntimeError("module_command_not_ready")

    def test_module_command_serves_and_stays_alive_without_attempt(self):
        parent = _launch_parent()
        proc, job = self._spawn(8901, parent)
        try:
            self._wait_ready(proc, parent)
            self.assertIsNone(proc.poll(), "module command exited early")
            # no attempt reserved before an accepted Start
            self.assertEqual([p for p in parent.iterdir() if p.is_dir()], [])
            # shutdown verifies zero child/descendant
            job.terminate()
            proc.wait(timeout=10)
            self.assertEqual(job.active_processes(), 0)
        finally:
            job.close()
            proc.close()

    def test_module_command_occupied_port_stop(self):
        parent = _launch_parent()
        proc, job = self._spawn(8902, parent)
        try:
            self._wait_ready(proc, parent)
            parent2 = _launch_parent()
            proc2, job2 = self._spawn(8902, parent2)
            try:
                # second bind fails -> STOP, exits, no worker
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline and proc2.poll() is None:
                    time.sleep(0.1)
                self.assertIsNotNone(proc2.poll(), "second server did not exit")
                out2 = (parent2 / "stdout.txt").read_text(encoding="utf-8", errors="ignore")
                self.assertIn("STOP", out2)
            finally:
                job2.close()
                proc2.close()
            # first server still alive
            self.assertIsNone(proc.poll())
        finally:
            job.terminate()
            proc.wait(timeout=10)
            job.close()
            proc.close()


class WorkerControls(unittest.TestCase):
    def setUp(self):
        os.environ["CARTPROOF_DEMO_HOLD"] = "0"
        self._saved = {k: os.environ.get(k) for k in
                       (dw.PARENT_MARKER, dw.OFFLINE_ENV, dw.LIVE_RELEASE_ENV)}

    def tearDown(self):
        os.environ.pop("CARTPROOF_DEMO_HOLD", None)
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _attempt(self, tmp, **ctx_over):
        from harness.joint_demo_control import verify_source, digest
        name = "0123456789abcdef"
        d = Path(tmp) / name
        d.mkdir()
        (d / "ownership.json").write_text(json.dumps(
            {"schema": 1, "reserved": True, "attempt_id": name, "pid": 12345}),
            encoding="utf-8")
        (d / "profile.json").write_text(json.dumps(PROFILE), encoding="utf-8")
        man = json.loads((ROOT / "harness/joint_demo_source_manifest.json").read_text(encoding="utf-8"))
        source = verify_source(ROOT, man)
        tests = du._compute_tests_digest(ROOT)
        ctx = {"schema": 1, "route": "cartproof-demo-ui-v1", "mode": "offline",
               "source_digest": source, "tests_digest": tests,
               "profile_sha256": digest(ROOT / "harness/joint_demo_profile.json"),
               "release": False}
        ctx.update(ctx_over)
        (d / "run-context.json").write_text(json.dumps(ctx), encoding="utf-8")
        return d

    def _refuse(self, d):
        os.environ[dw.PARENT_MARKER] = dw.PARENT_VALUE
        os.environ[dw.OFFLINE_ENV] = "1"
        with patch("harness.shopify_live_backend.resolve_token") as rt, \
             patch("harness.joint_demo_worker.construct_model") as cm:
            rc = dw.main(['--worker-dir', str(d)])
        self.assertEqual(rc, 2)
        rt.assert_not_called()
        cm.assert_not_called()

    def test_forged_ownership_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._attempt(tmp)
            (d / "ownership.json").write_text(json.dumps(
                {"schema": 1, "reserved": False, "attempt_id": "0123456789abcdef", "pid": 1}),
                encoding="utf-8")
            self._refuse(d)

    def test_ownership_attempt_id_mismatch_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._attempt(tmp)
            (d / "ownership.json").write_text(json.dumps(
                {"schema": 1, "reserved": True, "attempt_id": "different", "pid": 1}),
                encoding="utf-8")
            self._refuse(d)

    def test_release_mismatch_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._attempt(tmp, release=True)  # offline but release true
            self._refuse(d)

    def test_wrong_tests_digest_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._attempt(tmp, tests_digest="0" * 64)
            self._refuse(d)

    def test_altered_profile_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._attempt(tmp)
            altered = dict(PROFILE); altered["request_cap"] = 999
            (d / "profile.json").write_text(json.dumps(altered), encoding="utf-8")
            self._refuse(d)


class ParentClosure(unittest.TestCase):
    """B: final success is withheld until parent verification really passes."""

    def setUp(self):
        os.environ["CARTPROOF_DEMO_HOLD"] = "0"

    def tearDown(self):
        os.environ.pop("CARTPROOF_DEMO_HOLD", None)

    def test_complete_withheld_until_parent_verifies(self):
        gate = threading.Event()
        parent = _launch_parent()
        ctl = du.DemoController(mode="offline", root=ROOT, port=8765, hold="0",
                                offline_mapping=MAPPING, launch_parent=parent)
        real_verify = du.verify_worker_result

        def blocking_verify(attempt_dir, profile, mode):
            gate.wait(timeout=60)  # hold the parent verifier behind the event
            return real_verify(attempt_dir, profile, mode)

        with patch.object(du, "verify_worker_result", side_effect=blocking_verify):
            ctl.start()
            # Wait until the worker wrote its final snapshot to disk.
            attempt = ctl.ctx["attempt_dir"]
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                sp = attempt / "snapshot.json"
                if sp.is_file() and json.loads(sp.read_text(encoding="utf-8")).get("status") == "complete":
                    break
                time.sleep(0.05)
            # Parent has NOT verified yet: no COMPLETE may be presented.
            for _ in range(5):
                snap = ctl.snapshot()
                self.assertNotEqual(snap.get("verdict"), "COMPLETE")
                self.assertNotEqual(snap.get("status"), "complete")
            gate.set()
            ctl.thread.join(timeout=90)
            self.assertTrue(ctl.finished)
            self.assertEqual(ctl._terminal, "complete")
            self.assertTrue(ctl._final_ok)
            self.assertEqual(ctl.snapshot()["verdict"], "COMPLETE")
            self.assertEqual(ctl.snapshot()["status"], "complete")


if __name__ == "__main__":
    unittest.main(verbosity=2)
