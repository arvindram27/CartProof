"""INTERNAL CANDIDATE. No CLI/live entry until dependency and supervision proof.

Real model proposes; explicit-SKU grammar is deterministic intent authority.
The dropped-action hook is confined to this fixed demo sequence.
"""
from __future__ import annotations
import hashlib
import json
import re
import time
from pathlib import Path
from agents.intent import validate_proposal
from agents.model_client import ModelRequest

class DemoStop(Exception):
    pass

def need(value, reason):
    if not value:
        raise DemoStop(reason)

class JointDemo:
    """One fresh backend, frozen synthetic mapping, two fixed scenarios.

    Caller owns isolation/deadline/client construction. This class never creates
    network transports, changes budgets, retries, or enables live mode.
    """
    def __init__(self, backend, model, mapping, output, *, secrets=(), emit=None,
                 timeout_seconds=300, clock=time.monotonic):
        self.backend, self.model = backend, model
        self.mapping, self.output = mapping, Path(output)
        self.secrets = tuple(s for s in secrets if s)
        self.emit = emit or (lambda line: None)
        self.clock = clock
        self.deadline = clock() + timeout_seconds
        self.events = []
        self.results = []
        self.effects = {'seed_attempted': False, 'add_attempts': 0,
                        'drop_hook_uses': 0, 'recovery_attempts': 0}
        self.phase = 'not_started'
        self.reason = None
        self.error_class = None
        self.owns_output = False
        self.a = self.b = None

    def remaining(self):
        need(self.clock() < self.deadline, 'deadline_exhausted')

    def publish(self, event, **fields):
        # Only caller-selected scalar metadata. Never render provider payloads.
        row = {'event': event, **fields}
        line = json.dumps(row, sort_keys=True)
        need(not any(s in line for s in self.secrets), 'progress_secret_refused')
        need(not self.backend._scan_unsafe(line), 'progress_capability_refused')
        self.events.append(row)
        self.emit(line)

    def state(self):
        self.remaining()
        result = self.backend.acquire_state()
        need(isinstance(result, dict) and isinstance(result.get('lines'), dict),
             'state_unavailable')
        lines = result['lines']
        need(all(isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool)
                 and 0 < v <= 99 for k, v in lines.items()), 'state_malformed')
        return result

    def equal(self, actual, expected):
        need(actual.get('lines') == expected.get('lines'), 'state_changed')
        a = actual.get('provenance', {}).get('cart_ref')
        b = expected.get('provenance', {}).get('cart_ref')
        need(a is not None and a == b, 'cart_identity_changed')

    def propose(self, kind, **fields):
        self.remaining()
        proposal = self.model.propose(ModelRequest(
            kind=kind, max_output_tokens=self.model.config.max_output_tokens_per_call,
            **fields))
        self.remaining()
        need(proposal.status == 'ok', 'model_failed')
        record = self.model.records[-1] if self.model.records else {}
        need(record.get('status') == 'made', 'model_usage_unverified')
        need(not self.model.overrun, 'model_budget_overrun')
        sidecar = self.model.sidecars.get(self.model.calls_attempted)
        need(isinstance(sidecar, dict) and sidecar.get('usage_present') is True
             and sidecar.get('usage_detail_trustworthy') is True
             and isinstance(sidecar.get('estimated_cost_usd'), (int, float)),
             'model_cost_unverified')
        need(isinstance(proposal.payload, dict), 'model_payload_malformed')
        return proposal.payload

    def intent(self):
        text = 'Add 1 x ' + self.a + '.'
        catalog = [{'sku': s, 'product_name': s, 'size': '', 'color': ''}
                   for s in (self.a, self.b)]
        payload = self.propose('intent', request_text=text, catalog=catalog)
        result = validate_proposal(text, catalog, payload)
        need(result['outcome_kind'] == 'proceed', 'intent_not_agreed')
        intent = result['executable_intent']
        need([(x['sku'], x['quantity']) for x in intent['items']] == [(self.a, 1)],
             'intent_scope')
        self.publish('intent_crosscheck_agreed')
        return text, intent

    def choose_add(self, intent, observed, phase, *, scenario_baseline):
        add = {'action': 'add_to_cart', 'sku': self.a, 'quantity': 1}
        stop = {'action': 'finish'}
        baseline = dict(scenario_baseline)
        target = dict(baseline)
        target[self.a] = target.get(self.a, 0) + 1
        payload = self.propose('action', executable_intent=intent,
            observed_state={'phase': phase, 'lines': dict(observed['lines']),
                'action_semantics': 'Add the requested quantity to this scenario baseline; existing units are not completion of this new request.',
                'scenario_baseline_lines': baseline,
                'requested_additional_lines': {self.a: 1},
                'scenario_target_lines': target},
            available_actions=[add, stop], completed_items=[])
        # MetaModelClient normalizes relevant action fields. Other shapes STOP.
        need(payload == add, 'action_not_exact_allowed_add')
        self.publish('model_add_selected', phase=phase, quantity=1)

    def mutate(self, sku, *, recovery=False):
        self.remaining()
        if recovery:
            self.effects['recovery_attempts'] += 1
        self.effects['add_attempts'] += 1
        result = self.backend.add_line(sku, 1, expected_version=None)
        self.remaining()
        need(isinstance(result, dict) and result.get('ok') is True,
             'mutation_outcome_unverified')
        self.publish('mutation_response_ok', recovery=recovery)

    def make_check_state(self, text, name):
        from store.mcp_server_v2 import ServerState
        p = self.output / name
        p.mkdir()
        task = {'runtime': {'request_text': text,
                            'seed_lines': [{'sku': self.b, 'quantity': 1}]},
                'expected_interpretation': {'items': [{'sku': self.a, 'quantity': 1}],
                    'budget_max_cents': None, 'substitution_allowed': False}}
        task_path = p / 'task.json'
        task_path.write_text(json.dumps(task), encoding='utf-8')
        self.remaining()
        state = ServerState(self.backend, str(task_path), str(p / 'log'))
        self.remaining()
        return state

    def check(self, state, expected_lines, label, expected_verdict):
        from store.mcp_server_v2 import run_check
        self.remaining()
        result = run_check(state)
        self.remaining()
        need(result.get('verdict') == expected_verdict, 'unexpected_verdict')
        need(result.get('final_lines') == expected_lines, 'unexpected_final_lines')
        seeds = result.get('seed_lines')
        need(isinstance(seeds, list) and len(seeds) == 1
             and isinstance(seeds[0], dict) and seeds[0].get('item_key') == self.b
             and seeds[0].get('survived_unchanged') is True,
             'preserved_item_evidence_missing_or_changed')
        need(result.get('baseline_provenance', {}).get('cart_ref') ==
             result.get('check_provenance', {}).get('cart_ref'), 'check_identity')
        self.results.append(self.backend.redact(result))
        self.publish('independent_check', scenario=label, verdict=expected_verdict,
                     unrelated_item_preserved=True)
        return result

    def validate(self):
        need(not self.output.exists(), 'output_exists')
        need(getattr(self.backend, '_cart_id', None) is None, 'fresh_cart_required')
        variants = self.mapping.get('variants', {})
        need(isinstance(variants, dict) and len(variants) >= 2, 'catalog_shortfall')
        need(all(isinstance(k, str) and re.fullmatch(r'CARTPROOF-TEST-[A-Z0-9-]{1,40}', k)
                 and isinstance(v, str) and re.fullmatch(r'gid://shopify/ProductVariant/[0-9]+', v)
                 for k, v in variants.items()), 'mapping_invalid')
        need(len(set(variants.values())) == len(variants), 'mapping_alias')
        self.a, self.b = sorted(variants)[:2]
        self.backend.install_mapping(self.mapping)
        self.output.mkdir(parents=True, exist_ok=False)
        self.owns_output = True
        self.publish('demo_mode', evidence='API state at fetch time',
            model_role='intent crosscheck and constrained action selection',
            controlled_fault=True, atomic_concurrency_guard=False)

    def run_sequence(self):
        self.phase = 'validate'
        self.validate()
        # Resolve ambiguity BEFORE creating or mutating any cart.
        self.phase = 'matching_intent'
        text, intent = self.intent()
        self.phase = 'fresh_cart'
        before = self.state()
        need(before['lines'] == {}, 'fresh_cart_not_empty')
        self.phase = 'seed'
        self.effects['seed_attempted'] = True
        self.mutate(self.b)
        seed = self.state()
        need(seed['lines'] == {self.b: 1}, 'seed_state')
        first = self.make_check_state(text, 'matching')
        self.equal(first.baseline, seed)
        self.phase = 'matching_action'
        self.choose_add(intent, first.baseline, 'matching',
                        scenario_baseline=first.baseline['lines'])
        self.equal(self.state(), first.baseline)
        self.mutate(self.a)
        self.check(first, {self.a: 1, self.b: 1}, 'matching', 'COMPLETE')
        self.phase = 'fault_intent'
        text, intent = self.intent()
        second = self.make_check_state(text, 'controlled_drop')
        need(second.baseline['lines'] == {self.a: 1, self.b: 1}, 'second_baseline_changed')
        self.equal(second.baseline, {'lines': {self.a: 1, self.b: 1},
            'provenance': first.baseline['provenance']})
        self.phase = 'controlled_drop'
        self.choose_add(intent, second.baseline, 'controlled_drop',
                        scenario_baseline=second.baseline['lines'])
        self.equal(self.state(), second.baseline)
        self.effects['drop_hook_uses'] += 1
        self.publish('controlled_drop_NO_REQUEST_SENT')
        mismatch = self.check(second, {self.a: 1, self.b: 1}, 'controlled_drop', 'PARTIAL')
        need(mismatch['mismatches'] == [{'sku': self.a, 'expected_quantity': 2,
                                       'actual_quantity': 1}], 'wrong_discrepancy')
        self.phase = 'recovery'
        observed = self.state()
        self.equal(observed, second.baseline)
        self.choose_add(intent, observed, 'bounded_recovery',
                        scenario_baseline=second.baseline['lines'])
        self.equal(self.state(), second.baseline)
        self.mutate(self.a, recovery=True)
        self.check(second, {self.a: 2, self.b: 1}, 'recovered', 'COMPLETE')
        self.phase = 'complete'
        self.publish('sequence_complete')

    def _execute(self):
        try:
            self.run_sequence()
            complete = True
        except DemoStop as exc:
            self.reason = str(exc)
            complete = False
        except Exception as exc:
            self.error_class = type(exc).__name__ if type(exc).__module__ == 'builtins' else 'ExternalError'
            self.reason = 'unexpected_exception'
            complete = False
        summary = {'complete': complete, 'phase': self.phase,
            'reason': self.reason, 'error_class': self.error_class, 'effects': self.effects,
            'model_calls': self.model.calls_made,
            'consumed_tokens': self.model.consumed_tokens,
            'shopify_requests': self.backend.request_count,
            'estimated_cost_usd': round(sum(x.get('estimated_cost_usd') or 0
                 for x in self.model.sidecars.values()), 8),
            'cost_status': ('estimated' if self.model.calls_made == len(self.model.sidecars)
                 and all(x.get('estimated_cost_usd') is not None
                         and x.get('usage_detail_trustworthy') is True
                         for x in self.model.sidecars.values()) else 'unknown'),
            'not_live_proof_without_external_run_context': True,
            'no_automatic_retry': True}
        # Final output uses fixed labels/counts only, not raw model summaries.
        if self.owns_output and self.output.is_dir():
            accounting = [{k: row.get(k) for k in (
                'kind', 'model_name', 'duration_ms', 'http_status', 'finish_reason',
                'provider_usage', 'usage_present', 'usage_detail_trustworthy',
                'max_completion_tokens', 'estimated_cost_usd')}
                for _, row in sorted(self.model.sidecars.items())]
            data = {'summary': summary, 'events': self.events, 'checks': self.results,
                    'model_accounting': accounting,
                    'shopify_request_log': self.backend.request_log}
            text = json.dumps(self.backend.redact(data), indent=2, sort_keys=True)
            unsafe = any(s in text for s in self.secrets) or self.backend._scan_unsafe(text)
            if unsafe:
                summary['complete'] = False
                summary['reason'] = 'final_export_refused'
            else:
                (self.output / 'demo-evidence.json').write_text(text, encoding='utf-8')
            offenders = []
            for p in self.output.rglob('*'):
                if p.is_file():
                    raw = p.read_text(encoding='utf-8', errors='replace')
                    if any(s in raw for s in self.secrets) or self.backend._scan_unsafe(raw):
                        offenders.append(p.name)
            if offenders:
                summary['complete'] = False
                summary['reason'] = 'archive_guard_refused'
                # No manifest on refusal. Safe summary replaces any stale success.
                (self.output / 'export-status.json').write_text(
                    json.dumps({'complete': False, 'reason': 'archive_guard_refused'}),
                    encoding='utf-8')
            if summary['complete']:
                files = {p.relative_to(self.output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(self.output.rglob('*')) if p.is_file()}
                manifest_text = json.dumps(files, indent=2, sort_keys=True)
                need(not any(s in manifest_text for s in self.secrets)
                     and not self.backend._scan_unsafe(manifest_text), 'manifest_guard_refused')
                (self.output / 'manifest.json').write_text(manifest_text, encoding='utf-8')
                actual = {p.relative_to(self.output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(self.output.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
                need(actual == files, 'archive_closure_changed')
                for p in self.output.rglob('*'):
                    if p.is_file():
                        raw=p.read_text(encoding='utf-8',errors='replace')
                        need(not any(s in raw for s in self.secrets)
                             and not self.backend._scan_unsafe(raw), 'post_manifest_guard_refused')
        return summary

    def execute(self):
        try:
            return self._execute()
        except BaseException as exc:
            # Includes disk/finalization failure, never print raw exceptions.
            return {'complete': False, 'phase': self.phase,
                    'reason': 'finalization_or_interrupt_failure',
                    'error_class': type(exc).__name__ if type(exc).__module__ == 'builtins' else 'ExternalError',
                    'effects': dict(self.effects),
                    'model_calls': self.model.calls_made,
                    'shopify_requests': self.backend.request_count,
                    'no_automatic_retry': True}
