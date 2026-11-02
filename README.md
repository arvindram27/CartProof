# CartProof — offline foundation and gated model pilot

CartProof is a verification-and-recovery layer for shopping agents. The
repository contains:

- a **fault-injectable mock storefront** (FastAPI + SQLite + Jinja2),
- a **Playwright browser adapter** that acts on and observes the store,
- a **scripted (non-LLM) actor** that assembles a cart through the adapter,
- an intent-ledger **verification layer** and an opt-in bounded **recovery
  layer** for the owned mock store, and
- a **gated model pilot** (`harness/model_pilot.py`) that can run a fixed,
  budgeted comparison of model profiles against pinned tasks, with
  independent scoring and frozen evidence.

The foundation runs entirely on your machine. The model pilot's live
transport is separately gated (see "Model pilot" below); the pilot evidence
in this repository's review record is offline and does not come from live
model calls.

## Status

- A prior live smoke exists on record, with a prior smoke cost estimate of
  $0.0132445. It is therefore not accurate to say this project has never
  made live calls.
- The new 4-cell Stage 1 comparison (2 Spark + 2 Wonder cells) has **not**
  been live-validated. Its evidence is offline, produced with FAKE injected
  transport through the real pilot cell path.
- One trial per cell. The pilot readout is a plumbing/scoring readout of
  those cells only, not a statistical or model-capability claim.
- Stage 2 is not admitted: its predecessor-receipt seam is UNSET
  (`ACCEPTED_STAGE1_RECEIPT_ID` / `ACCEPTED_STAGE1_MANIFEST_ID` are `None`).

## Layout

```
cartproof/
  requirements.txt
  store/           # FastAPI app, SQLite schema, faults, templates, seed catalog
  adapter/         # Playwright actions + observations + Evidence
  agents/          # scripted actor, budgeted model client, Meta transport, env loader
  harness/         # run_task.py (foundation runner), model pilot + scoring harness
  examples/        # task1.json
  tests/           # pytest suite
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m playwright install chromium
```

## Run the store (manual)

The store is normally booted by the harness or the tests, but you can run the
two apps yourself if you want to browse:

```powershell
.\.venv\Scripts\python -c "from store.app import create_app, create_control_app; import uvicorn; uvicorn.run(create_app('var/cartproof.db','store/seed_catalog.json'), host='127.0.0.1', port=8000)"
```

The `/_control/` endpoints are served by `create_control_app`, which should be
bound to a different 127.0.0.1 port and is not linked from any page.

## Run a task

```powershell
.\.venv\Scripts\python -m harness.run_task --task examples/task1.json
.\.venv\Scripts\python -m harness.run_task --task examples/task1.json --fault dom-api-mismatch
.\.venv\Scripts\python -m harness.run_task --task examples/task1.json --fault slow-consistency --delay-ms 800
.\.venv\Scripts\python -m harness.run_task --task examples/task1.json --fault wrong-variant --param to_sku=BTL-HIK-750-STEEL
```

The runner boots the store and control endpoint on ephemeral 127.0.0.1 ports,
resets, applies the requested fault, runs the actor, prints the evidence log to
stdout, and shuts down.

## Run the tests

```powershell
.\.venv\Scripts\python -m pytest
```

## Model pilot

Architecture diagram: [`docs/architecture.svg`](docs/architecture.svg).

The pilot compares two fixed profiles — `spark-medium` (`muse-spark-1.3`)
and `wonder-medium` (`muse-wonder-1.0`) — over pinned tasks in a fixed
cell order, defined in `harness/pilot_manifest.py` (the manifest of
record: limits, profiles, tasks with pinned hashes, stages, and a pinned
price table). Stage 1 is 4 CA-arm cells (2 Spark + 2 Wonder); Stage 2
stays UNSET until a Stage 1 batch is reviewed and accepted.

Manifest prices were verified against
https://dev.meta.ai/docs/pricing-rate-limits on 2026-10-31.

### Gates and budgets

- Per-cell budgets: 12 calls / 20,000 local tokens / 4,096 output tokens,
  360 s worker timeout, 2,000 ms settling. `BudgetedModelClient`
  (`agents/model_client.py`) reserves budget per call and latches on
  overrun; uncertainty counts against the budget.
- Cost gate: a $3 new-pilot review threshold and a $5 total planning
  allowance that includes the prior smoke estimate ($0.0132445). The
  Stage 1 local allowance is $0.97 (2 Spark + 2 Wonder cells at the
  manifest's per-profile allowances).
- Content gate: live runs require a fully clean worktree and require the
  on-disk bytes of the 43-path live-execution closure (`DIGEST_PATHS` in
  `harness/model_pilot.py`) to match the digest pinned in
  `harness/reviewed_source.sha256`. `HEAD` is recorded in the attempt
  reservation as provenance, never compared, so publication no longer
  blocks a live run. The earlier `BASE_HEAD` pin allowed seven files to be
  dirty, so it never byte-pinned them; the digest covers them like every
  other reviewed byte. The pin is checkout-specific: this working tree is a
  mix of LF and CRLF bytes (`core.autocrlf=true`), so the digest commits to
  those exact bytes and must be recomputed if the EOL policy or the checkout
  changes. Review, not cryptography, is the boundary: the pin
  is data, and the reviewer recomputes the digest from the delivered bytes
  before approving any commit.

  Source-v2 recipe: for each path in `DIGEST_PATHS`, in listed order,
  record `rel|len|sha256hex` over the raw on-disk bytes (`read_bytes()`,
  no newline handling); seed
  `"cartproof|" + MANIFEST_ID + "|" + ATTEMPT_PROTOCOL_ID + "|source-v" +
  str(SOURCE_DIGEST_VERSION) + "\n" + records joined by newline`; the
  digest is the SHA-256 of the UTF-8 seed.
- Transport: `agents/meta_client.py` fails closed before reading the key
  unless every live gate passes, and never retries. The API key comes from
  the `MODEL_API_KEY` environment variable first; the sanitized loader in
  `agents/env_loader.py` is a fallback used only when it is absent
  (failures surface as fixed categories, never values).
- Write safety: the harness-enumerated stage and control mutations
  (reservation latch, export copies, batch staging and publication,
  control-root handling) use leaf and ancestor validation. Batches freeze
  `batch.json`, `ledger.json` and `summary.md` before publication, and
  export withholds anything that fails its checks.

### Attempt successor (default denied)

`SUCCESSOR_GRANT` in `harness/pilot_manifest.py` is reviewed data inside
the digest-covered source. It is **UNSET** by default, so a consumed
attempt still refuses with `attempt_already_reserved`. No grant is wired
for the halted attempt `627eb19e50084072`.

When a later review sets it, the grant names exactly one prior attempt id
and the prior terminal states it accepts. At runtime the harness validates
that grant against the prior attempt's on-disk `reservation.json`, refusing
with explicit categories (`successor_grant_invalid`,
`successor_prior_missing`, `successor_prior_mismatch`,
`successor_prior_unreadable`, `successor_prior_ambiguous`,
`successor_prior_not_terminal`, `successor_prior_state_rejected`). Prior
attempt files are read only. A successful successor reservation records
`successor_of` and `successor_grant_id`.

Reserving a successor is atomic across processes: the transition is guarded
by an exclusive per-grant create (`successor-claim-<hash>.guard`), in the
same style as the original latch, not a check-then-write. However many
invocations run concurrently under one reviewed grant, exactly one takes the
guard and reserves the successor; every other invocation refuses with
`successor_already_taken` (the successor exists) or
`successor_guard_incomplete` (the guard was taken but no complete successor
is on disk). Failures after the guard are fail-closed: a crash or partial
write leaves the guard in place, so every later run under that grant refuses
with `successor_guard_incomplete` and nothing is repaired, reset, or
deleted. Recovery is a **new** reviewed grant after inspection, which gets
its own guard; the stuck guard keeps blocking the old one. A replay under a
consumed grant refuses the same way, and the original latch plus all prior
attempt files stay byte-untouched. The mechanism does not unlock Stage 2,
retry inside an attempt, or change any cost gate.

### No-model dry-run (separately run runtime check)

The dry-run is a deliberately separate mode that runs **one** Stage 1 cell
through the real worker spawn machinery, local store boot and Chromium
launch, with the model transport replaced by the offline fake client. It
never constructs the Meta client and never reads `.env` or
`MODEL_API_KEY`; if the offline transport is unavailable it refuses rather
than falling back. It never writes an attempt latch, reservation, receipt
or control-root state. Output goes to its own labeled
`var\dry-run-stage1-<timestamp>\` area (`dry-run.json` plus `summary.md`),
records zero provider calls, the observed offline fixture call count and no
known cost, and says it is a plumbing check, not a measurement.

Its offline responses come from `eval/model/dev/fake_scripts.json`, which is
hash-pinned in `harness/pilot_manifest.py` (`DRY_RUN_FIXTURE`, the same style
as the pinned task files). That manifest module is inside the reviewed
digest closure, so the fixture is tamper-evident under the review loop: a
replaced or corrupted fixture refuses with `dry_run_fixture_mismatch`, a
missing/unreadable/malformed one with `dry_run_responses_unavailable`, and a
malformed pin with `dry_run_fixture_pin_invalid`. There is no fallback. The
live Meta path never reads the fixture and `SUCCESSOR_GRANT` never consults
it.

Exact command, run from the repository root, **separately and
deliberately**, never as part of the unit test suite:

```powershell
.\.venv\Scripts\python.exe -m harness.model_pilot --stage 1 --dry-run
```

This command spawns a real worker, boots the local store and launches
Chromium. It refuses when combined with `--live`, for any stage other than
1, or when the offline transport is not importable. Neither the successor
mechanism nor the dry-run changes the fresh-approval requirement for any
paid attempt.

On 2026-11-01, on this machine, the Playwright driver's temp-artifact
creation under the profile Temp directory was denied. For one invocation,
setting the session-scoped TEMP/TMP to a fresh project-local directory let
that no-model dry-run reach `execution_status: normal` with
`cleanup_status: verified`; it then halted on the accounting defect this
brief corrects (`accounting_sidecar_correspondence_mismatch`). This is a
single observed configuration, not a pass, a guarantee, or a permanent
environment change.

### Scoring readout

Each cell's worker output is measured independently
(`harness/model_compare.py`, `harness/evaluate.py`,
`harness/model_metrics.py`, `harness/snapshots.py`); the model's declared
outcome is never substituted for the measurement. The frozen readout
splits declared COMPLETE into verified / known-false / unverified, and
reports per-profile counts beside the existing cost fields.
`correct_clarification` is reported as a behavior count, distinct from
task appropriateness.

Readout completeness is explicit: a cell whose evaluated record should
exist but cannot be read is counted as `evaluated_unreadable` (never
silently scored as zero), a cell with no worker output is counted
separately as `halted_before_evaluation`, and the batch readout status is
`COMPLETE` only when every evaluated record was readable. Readout
COMPLETE means the evaluation records were readable; it is not task
success, and traversal complete is not task success either.

### Test evidence

Two separate saved suite runs (kept separate, never summed), from review
evidence; independent review of them was byte/hash level, not a re-run:

- Strict suite: 123 passed / 5 skipped. The 5 skips are Windows symlink
  cases and remain UNVERIFIED in this account.
- Regression suite: 76 passed / 1 deselected.

### Known limits

- Dollar figures are local estimates at Meta's published 2026-10-31
  prices. They are NOT billing caps, and Meta offers no provider-side
  hard spending cap; the budget wrapper is a best-effort fail-closed
  guard, not a hard token or dollar guarantee.
- `harness.metrics` and the MC arm are outside the reviewed scope: they
  were not reviewed, and nothing here asserts whether they are
  implemented or unimplemented.
- Windows junction/reparse rows pass; the 5 symbolic-link creation rows
  are skipped and remain UNVERIFIED in this account.
- Checks are sequential and non-following; no hostile
  concurrent-replacement immunity is claimed.
- Offline test evidence does not authorize or imply any live stage. A
  live run still needs its own separate approval.

## Faults

| name | behaviour |
|---|---|
| `silent-drop` | add returns success but writes no line and does not bump the version |
| `wrong-variant` | add writes the confusable sibling sku (`to_sku` to override) |
| `quantity-drift` | add writes `quantity + delta` (default `delta=1`) |
| `stale-total` | lines write, but the read total is the pre-write total |
| `dom-api-mismatch` | `/cart` renders one line fewer than the API reports |
| `slow-consistency` | writes commit immediately; reads serve a snapshot lagged by `delay_ms` |
| `external-mutation` | on the next read, bump one line's quantity by `delta`, once |

## Import rule

Runtime code outside `store/` (the adapter, actor, and `core/`) never imports
from `store/`; it reaches the store only over HTTP. `tests/` and `harness/` may
import the store app factory to boot it, and only they call `/_control/`.
This is a scoped trust assumption for a local demo harness, not a security
boundary.

## Verification layer

This phase adds the intent ledger and the action-outcome verification loop
(`core/`). It verifies and reports; it does **not** correct anything.

- A task JSON with a `constraints` list is verified. Constraints are typed and
  immutable: `item_quantity` (final absolute quantity of one sku),
  `count_distinct` (exactly N distinct skus from a named set), and `budget_max`
  (recomputed cart total in cents). Each constraint carries a state from
  `unverified`, `confirmed`, `contradicted`, or `unknown`.
- The verifier orders reads against writes using the store's per-cart
  `version`. A read below `known_version` is provably stale and never counts as
  confirmation; a view must be fresh (version equals `known_version`) and
  settled (three consecutive identical DOM+API rereads) to confirm or contradict
  a constraint. This version-ordering rule exists only because this owned mock
  store offers a version contract; on a store without one, uncertain constraints
  stay `unknown`.
- `unknown` blocks completion. A run that ends `VERDICT: COMPLETE` has every
  constraint confirmed *and* the final-minus-baseline delta equal to the
  declared additions; anything else is `VERDICT: PARTIAL (n/m constraints
  confirmed)` with the failed checks and anomalies named. Exit codes: 0 for
  COMPLETE, 1 for PARTIAL, 2 for hard errors.

## Recovery layer (opt-in)

`--recover` turns on bounded, deterministic correction for the owned mock store.
Without it, behaviour is unchanged. It corrects only a proven, safely
attributable mismatch; it never repairs a total or makes uncertain evidence
agree.

- Usage: `python -m harness.run_task --task examples/task1.json --fault
  quantity-drift --recover`. `--recover` with no `constraints` key is a hard
  configuration error.
- Supported corrections: a proven wrong variant (remove the owned wrong line,
  then re-add the intended SKU — two separately guarded, non-atomic steps), a
  proven excess quantity on a newly created line (PATCH to the intended absolute
  quantity), and a proven no-commit silent drop (one bounded retry of the same
  intended add). Timeout / 5xx / malformed response is `unknown_commit`, never a
  silent drop.
- Conservative ownership: automatic quantity reduction or removal applies only
  to a line absent from the original baseline *and* from the immediately
  preceding snapshot, newly observed after one tracked action, with a stable
  line ID and an exact explained delta. A merge into a seeded/pre-existing SKU
  stops with `ownership_uncertain` — some repairable-looking merged carts stay
  `PARTIAL` by design. Baseline lines, quantities, SKUs and unit prices, and all
  unrelated lines, are preserved.
- Hard bounds: at most 2 correction plans per original target and 4 attempted
  corrective HTTP mutations per run; a plan is not started unless its remaining
  steps fit both caps. External mutation, unknown commit, ambiguous ownership,
  unresolved diagnosis, cap exhaustion, a rejected correction, a 409 version
  conflict, stale totals, DOM/API conflict and deadline expiry all stop the run.
- Completion: `COMPLETE` requires every original constraint confirmed by a final
  fresh settled view, the final-minus-original-baseline net delta equal to the
  original declared additions, all protected items preserved, and no blocking
  incident. Historical failures remain visible even after a successful recovery.
- Scope: this is the owned mock store only. A transient fault fixture that is
  cleared between the initial add and the corrective add recovers; a persistent
  fault that remains active on every write ends `PARTIAL` — this is not a claim
  of real-merchant reliability or model-agent improvement.

## Project Meta key (private, after review)

After implementation, source, and test review — never before — the human may
privately create `.env` in the project root containing a single
`MODEL_API_KEY` assignment line, using a local editor with AI assistance
disabled. Saving `.env` never authorizes live use; a live smoke still needs
its own separate approval.

## MCP tools and Shopify adapter status

CartProof exposes raw mock-store cart tools and a callable cartproof_check over MCP. The check compares the cart with the frozen starting state and requested changes, reporting COMPLETE, PARTIAL or UNKNOWN. Calling the check is optional; the MCP tools do not automatically enforce CartProof's protection rules.

The new backend interface supports an offline Shopify adapter. Its responses and failure handling are tested using a stateful fake and adapted documentation fixtures, not a live Shopify connection. The v2 tool responses use canonical cart state and include provenance; they are not drop-in copies of the original server's responses.

Saved Windows test logs from November 2, 2026 record 33 focused tests passed and 594 passed / 5 skipped in the full suite.

Still pending: live Shopify compatibility, agent integration over the new interface, and latency measurement. The Shopify adapter has no atomic cart-version guard; its check is after-the-fact detection, not prevention.
