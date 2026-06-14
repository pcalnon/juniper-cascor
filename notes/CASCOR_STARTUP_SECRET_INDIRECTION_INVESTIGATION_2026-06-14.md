# CasCor Startup / Secret-Indirection / FSM / API / Canopy — Investigation & Fix

**Project**: Juniper — juniper-cascor
**Author**: Paul Calnon (investigation driven via Claude Code)
**Date**: 2026-06-14
**Status**: Root-caused. Critical fix implemented in this branch; related issues triaged below.
**Branch**: `fix/cascor-data-api-key-file-indirection`

---

## 1. Executive summary

While verifying the CW-05 dual-path remote-worker fix (juniper-cascor#319) end-to-end on the live
deploy stack, an attempt to load a fresh dataset and grow the network surfaced a cluster of
**service-config / startup-behaviour defects** in juniper-cascor. They are individually small but
collectively a *major blocker* for any workflow that drives cascor through its API or through the
Canopy dashboard (iterative training, live dataset swap, automated/CI runs).

The central defect — and the one this branch fixes — is a **silent secret-indirection failure**:
cascor never sends its juniper-data API key on the live-dataset-swap path because the code imports a
secret helper from a **module that does not exist**, and an `except ImportError` silently degrades it
to a `None`-returning stub. Both linters that would have caught it were suppressed inline. The
remaining findings (auto-start-on-boot, FSM-409-after-stop, experimental-gate-default-off,
client-side `_FILE`-blindness, opaque Canopy error surfacing) are documented with root cause, blast
radius, and a recommended fix each.

**This branch ships the critical fix + a regression test.** The related issues are triaged in §5 with
a recommended PR split.

---

## 2. How this surfaced (work completed so far)

Context: CW-05 published `juniper-cascor-model` 0.1.0 to PyPI; the distributed worker adopted it
(juniper-cascor-worker#102); the deploy worker image was rebuilt and both workers were cut over off
the `--cascor-path` stopgap. The **#319 dual-path was verified** end-to-end:

```
cascor   _dispatch_to_remote_workers: Dispatching 2 tasks to remote workers
worker   Received task … / Task completed: candidate 16 / Sent result … success=True   (×2)
cascor   Collected 2/2 results from remote workers          (2/2, 0 failed, torch 2.12.0+cpu)
```

The *only* unproven #319 criterion was a live unit-*add* (network growth). Attempting to drive that
through the cascor REST API hit, in order, **four gates** — none of which are part of CW-05:

| # | Gate observed | HTTP | Where |
|---|---------------|------|-------|
| G1 | Training auto-started on boot on a fresh 0-unit network with no real dataset (`epoch` climbed to 7576, `hidden_units=0`, `dataset_name=""`) | — | FSM / lifespan |
| G2 | `POST /v1/training/start` rejected after `stop`/`reset` | 409 | training FSM |
| G3 | `POST /v1/training/dataset/live` rejected — experimental functions disabled | 403 | admin gate |
| G4 | After enabling G3, the live swap failed: cascor could not authenticate to juniper-data | 502 / 401 | **secret indirection** |

G4 is the headline bug; G1–G3 are contributing startup/FSM defects. Each is root-caused below.

---

## 3. Root-cause findings

### 3.1 [CRITICAL — FIXED HERE] cascor sends no juniper-data API key on dataset swap (G4)

**Symptom.** `POST /v1/training/dataset/live` →
`502 juniper-data fetch failed: Request failed (401): Missing API key. Provide X-API-Key header.`
cascor's container env has `JUNIPER_DATA_API_KEY_FILE=/run/secrets/juniper_data_api_keys` (readable,
43 chars) but the resolved `JUNIPER_DATA_API_KEY` is empty.

**Root cause.** `src/api/lifecycle/manager.py` `_reload_dataset` (the code behind the live swap)
resolved the key like this:

```python
try:
    from secrets_util import get_secret   # ← module does NOT exist in this repo
except ImportError:
    get_secret = lambda _key: None        # ← always taken → api_key is always None
...
api_key = get_secret("JUNIPER_DATA_API_KEY")          # → None
client = JuniperDataClient(base_url=data_url, api_key=api_key)   # no X-API-Key sent
```

`find src -name 'secrets_util*'` returns nothing. The only `get_secret` in the tree is
`src/api/secrets.py:13`, which **correctly** honors `JUNIPER_DATA_API_KEY_FILE`:

```python
def get_secret(env_var, file_env_var=None):
    file_env_var = file_env_var or f"{env_var}_FILE"
    file_path = os.environ.get(file_env_var)
    if file_path and Path(file_path).is_file():
        return Path(file_path).read_text().strip()
    return os.environ.get(env_var)
```

**The asymmetry that proves it.** The same key is resolved correctly everywhere else:
- inbound cascor auth — `src/api/settings.py:9,196` `from api.secrets import get_secret`
- auto-start outbound fetch — `src/api/app.py:20,316` `from api.secrets import get_secret`

Only `manager._reload_dataset` imported from the bogus `secrets_util` → only the live-swap path
fails. Same env var, same secret file, three call sites, one wired to a nonexistent module.

**Why it shipped silently.** The broken line carried `# type: ignore[import-not-found]` **and** the
fallback carried `# noqa: E731` — i.e. **both mypy and flake8 were suppressed inline**, exactly the
two linters that would have flagged an unresolved import / a lambda assignment. No test exercised the
live-swap key path. Introduced in PR #242 / commit `30b2f0f` ("Issue #3 Phase 1, PR-6").

**Fix (this branch).** Replace the `try/except` block with the correct import:
`from api.secrets import get_secret`. Now `get_secret("JUNIPER_DATA_API_KEY")` resolves the `_FILE`
secret, the client gets a real key, and `X-API-Key` is sent. Plus a regression test
(`src/tests/integration/api/test_pending_dataset.py::test_reload_dataset_resolves_juniper_data_api_key_from_secret_file`)
that asserts the resolved key reaches the `JuniperDataClient` constructor (fails on the old `None`).

**Blast radius of the bug:** every Canopy/API-driven live dataset swap; any cascor→juniper-data fetch
routed through `_reload_dataset`. (The auto-start fetch via `app.py` was unaffected — different,
correct import.)

### 3.2 [CLASS — partially addressed] silent `ImportError` degradation + suppressed linters

The real story of 3.1 is not a typo, it's a **failure mode that should be impossible to ship**: a
first-party helper imported from a guessed module name, wrapped in `try/except ImportError` that
substitutes a *wrong-but-non-crashing* stub, with the two relevant linters silenced on the same
lines. This converts "module not found" (a hard, obvious failure) into "feature silently returns
`None`/411s in production."

**Prevention (recommended):**
- Never `try/except ImportError` around **first-party** modules to substitute a degraded stub. Import
  them directly; let an import error be loud.
- Treat inline `# type: ignore[import-not-found]` on a first-party import as a review red flag.
- CI guard: an import-resolution check (mypy in non-suppressed mode, or `python -c "import …"` smoke,
  or a small AST lint) for first-party modules. (Deferred — see §5.)
- The regression test added here covers the specific path; the class guard is the durable fix.

### 3.3 [HIGH] cascor auto-starts training on boot — default ON (G1)

`src/api/settings.py:68,330` defaults `auto_start = True`
(`_JUNIPER_CASCOR_API_AUTO_START_ENABLED = True`). On startup the lifespan handler
(`src/api/app.py:244–248`) fires `_auto_start_training` as a background task, which creates a default
(empty `{}`) network and a default dataset and immediately calls `lifecycle.start_training(...)`. The
code itself warns at `app.py:245`: *"Auto-start training is ENABLED — this should only be used in
demo/dev environments"* — yet it is the **default**.

`juniper-deploy/docker-compose.yml:355` sets `JUNIPER_CASCOR_AUTO_START: "true"` explicitly, so the
deploy demo opts in regardless of the code default.

**Effect.** Every fresh cascor boots into an already-`STARTED` training run on a 0-unit network
(this is why the #319 probe saw `epoch=7576, hidden_units=0`). It violates the Canopy/automation
assumption of a clean `STOPPED` initial state, and it interacts badly with the FSM guard in 3.4.

**Recommended fix.** Flip the **code default to `False`** (auto-start becomes opt-in; deploy already
sets it explicitly so the demo is unaffected). Separately reconsider whether the *deploy demo* should
auto-start onto a **real** dataset rather than an empty network, or not at all. (Deferred — §5.)

### 3.4 [HIGH] iterative training is blocked: 409 after stop/reset (G2)

`POST /v1/training/start` surfaces a `RuntimeError`/`ValueError` from
`lifecycle.start_training` as **409 "Training cannot be started in the current state"**
(`src/api/routes/training.py:73–78`). Two contributors:
1. After `stop`/`reset` the loaded training tensors are cleared, so a bare `start` (no `inline_data`,
   no successfully-fetched staged dataset) hits the *"Training data not provided"* guard
   (`manager.py` `start_training`) → 409.
2. The FSM `_handle_start` (`src/api/lifecycle/state_machine.py:155–196`) returns `False` for some
   states **without the route checking the return value**, so a rejected transition can look like a
   silent no-op rather than a clear error.

**Effect.** The natural loop *boot → load dataset → train → stop → load new dataset → train again* is
not achievable through the API without re-supplying data or restarting the container. Canopy's
start/stop/reset buttons can dead-end (see 3.7). Note: this is *aggravated* by 3.1 — the cold-stage +
start path that *should* re-fetch a dataset on start also routes through the broken `_reload_dataset`
key resolution, so even the "correct" recovery path 401s.

**Recommended fix.** (a) distinguish the 409 causes with actionable detail ("no dataset loaded — stage
one first" vs "invalid state"); (b) make `stop`/`reset` deterministically leave a *startable* state;
(c) have the route honour the FSM transition return value. (Deferred — §5.)

### 3.5 [MEDIUM] live dataset swap gated off by default (G3)

`manager.py:1133` initialises `_experimental_functions_enabled` from
`CASCOR_EXPERIMENTAL_FUNCTIONS_ENABLED == "1"` (default `False`); `swap_dataset_live`
(`manager.py:2485`) raises `PermissionError("experimental_functions_disabled")` → **403**, and the
DELETE counterpart 403s too (`training.py:255`). Toggle: `POST /v1/admin/experimental_functions
{enabled:true}` (`src/api/routes/admin.py:40`).

**Effect.** Canopy's live-swap feature is invisible/non-functional until an operator opens the gate.
This is intended (Phase-2, equal-dim-only, server-authoritative), but undocumented at the
ops/dashboard level and easy to mistake for a bug. **Recommended:** document the gate + ensure Canopy
surfaces the closed state explicitly (it largely does — see 3.7).

### 3.6 [LOW / defense-in-depth] juniper-data-client is `_FILE`-blind

`juniper-data-client/juniper_data_client/client.py:154`:
`resolved_api_key = api_key or os.environ.get("JUNIPER_DATA_API_KEY")` — the client's own env
fallback does **not** honour `JUNIPER_DATA_API_KEY_FILE`. So any consumer that sets *only* the `_FILE`
form and passes `api_key=None` gets an unauthenticated client. Not the cause of 3.1 (cascor must pass
the resolved key), but teaching the client to honour `_FILE` would close the broader class for every
consumer. **Cross-repo (juniper-data-client) — separate PR.**

### 3.7 [MEDIUM] Canopy surfaces these cascor failures inconsistently

Canopy drives cascor training via `src/backend/cascor_service_adapter.py` (start/stop/pause/
resume/reset, stage/swap dataset) behind `src/main.py` `/api/train/*` routes.
- **Training-control errors are opaque.** `dashboard_manager.py:4717` catches and *logs only*
  (`"Training control failed: …"`) — no toast. A 409 (3.4) leaves the button disabled for ~5s with no
  user-facing reason. This is the "why is this button dead?" class of dashboard confusion.
- **Dataset-swap / experimental-gate errors are surfaced** as danger alerts
  (`dashboard_manager.py:4095`, `:3640`) — so 3.5/G4 at least show *something*, though the message is
  generic ("Backend rejected").
- Canopy has **no special handling for cascor's auto-start-on-boot** (3.3): it relays whatever cascor
  broadcasts, so the dashboard shows a phantom in-progress run on a fresh stack.

**Recommended (canopy PR):** surface training-control failures (start/stop/reset) as alerts with
cascor's error detail; treat auto-start-on-boot defensively.

---

## 4. What this branch changes

| File | Change |
|------|--------|
| `src/api/lifecycle/manager.py` | `_reload_dataset`: replace `from secrets_util import get_secret` + `except ImportError` `None`-lambda with `from api.secrets import get_secret`. Fixes 3.1. |
| `src/tests/integration/api/test_pending_dataset.py` | Add `test_reload_dataset_resolves_juniper_data_api_key_from_secret_file` — asserts the resolved `JUNIPER_DATA_API_KEY_FILE` value reaches `JuniperDataClient(api_key=…)`. Guards 3.1/3.2. |
| `notes/…` | This document. |

Scope is deliberately tight: the one-line-class import fix is definitive, low-risk, and unblocks
every API/Canopy-driven dataset fetch. The behaviour-changing and cross-repo items (3.3, 3.4, 3.6,
3.7, and the 3.2 CI guard) are triaged below for follow-up PRs so each can be reviewed on its own
blast radius.

---

## 5. Recommended follow-up (triaged)

| Item | Repo | Risk | Recommendation |
|------|------|------|----------------|
| 3.3 auto-start default → `False` | juniper-cascor | low (deploy sets it explicitly) | small PR; flip default + add a test asserting fresh boot is `STOPPED` |
| 3.4 FSM 409 messaging + startable-after-stop | juniper-cascor | medium (FSM semantics) | design-first; clarify states + route return-value check |
| 3.2 import-resolution CI guard | juniper-cascor (+ ecosystem) | low | add a first-party import smoke / mypy lane so this class fails CI |
| 3.6 juniper-data-client `_FILE` awareness | juniper-data-client | low | defense-in-depth PR |
| 3.7 Canopy training-control error surfacing | juniper-canopy | low | UX PR |
| 3.5 experimental-gate docs | juniper-cascor / deploy | low | doc-only |

---

## 6. Verification

- `manager._reload_dataset` now imports the real `api.secrets.get_secret`; `secrets_util` no longer
  referenced anywhere (`grep -r secrets_util src` → empty).
- New regression test passes and fails on the pre-fix code (asserts a non-`None` resolved key reaches
  the client). Run: `pytest src/tests/integration/api/test_pending_dataset.py --integration`.
- End-to-end (post-deploy, after this lands + image rebuild): `POST /v1/training/dataset/live`
  against a populated `secrets/juniper_data_api_keys` should fetch from juniper-data with `X-API-Key`
  and return 200 instead of 502/401.

---

## 7. References

- Bug origin: PR #242 / commit `30b2f0f` (Issue #3 Phase 1, PR-6).
- Correct helper: `src/api/secrets.py`; correct call sites: `src/api/settings.py:196`,
  `src/api/app.py:316`.
- FSM: `src/api/lifecycle/state_machine.py`, `src/api/lifecycle/manager.py`,
  `src/api/routes/training.py`, `src/api/routes/admin.py`.
- Canopy integration: `juniper-canopy/src/backend/cascor_service_adapter.py`,
  `juniper-canopy/src/main.py` (`/api/train/*`), `juniper-canopy/src/frontend/dashboard_manager.py`.
- Canonical `_FILE` resolution precedents: `juniper-cascor-worker` `config._resolve` (#94/#95),
  `juniper-canopy` `src/secrets_util.py`, `juniper-data` `core/secrets.py`.
- CW-05 / #319 context: `notes` in juniper-ml + juniper-cascor-worker#102, juniper-deploy#115.
