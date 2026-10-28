# CartProof — offline foundation

The first phase of CartProof, a verification-and-recovery layer for shopping
agents. This phase builds only the local foundation:

- a **fault-injectable mock storefront** (FastAPI + SQLite + Jinja2),
- a **Playwright browser adapter** that acts on and observes the store,
- a **scripted (non-LLM) actor** that assembles a cart through the adapter.

Everything runs on your machine. No model calls, no API keys or secrets, no
external services, no git remotes, no checkout/payment flow.

## Layout

```
cartproof/
  requirements.txt
  store/           # FastAPI app, SQLite schema, faults, templates, seed catalog
  adapter/         # Playwright actions + observations + Evidence
  agents/          # scripted_actor.py (deterministic cart assembly)
  harness/         # run_task.py (boots store + control, applies fault, runs actor)
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


