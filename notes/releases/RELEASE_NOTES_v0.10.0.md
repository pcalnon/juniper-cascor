# Juniper CasCor v0.10.0 Release Notes

**Release Date:** 2026-08-30
**Version:** 0.10.0
**Codename:** Process Hygiene
**Release Type:** MINOR

---

## Overview

The first juniper-cascor release since v0.9.0 (2026-08-14), covering 63 merged commits. Its centre of gravity is **process and artifact hygiene**: forkserver children that outlived teardown, workers that hung at exit, a shutdown that let uvicorn's SIGTERM re-raise kill the process before training was joined, and several shared-path collisions between concurrent cascor instances. It also carries `JUNIPER_CASCOR_LOG_DIR` (#523), which downstream tooling has been waiting on.

> **Status:** STABLE – no known regressions.

---

## Release Summary

- **Release type:** MINOR
- **Primary focus:** process lifecycle + concurrent-instance isolation
- **Breaking changes:** NO for HTTP callers; **see Upgrade Notes** if you generate an SDK
- **Priority summary:** unblocks juniper-ml's Q-6 (`run_suite` cascor-parallel refusal)

---

## Features Summary

| Area | Change | Reference |
|---|---|---|
| Logging | `JUNIPER_CASCOR_LOG_DIR` override, service + direct CLI | #523 (Q-6 / H-7) |
| Lifecycle | shutdown joins training and releases the pool before SIGTERM re-raise | #589 |
| Pool | workers no longer hang at exit; shutdown joins share one deadline | #586/#587 |
| Pool | trainer preloaded in the forkserver | #592 (F3) |
| CLI | forkserver children no longer re-run `main.py`'s body | #588 |
| Security | docs surface no longer auth-exempt | #599 |

---

## What's New

**`JUNIPER_CASCOR_LOG_DIR` (#523, Q-6 / H-7).** Mirrors the W-6 `JUNIPER_CASCOR_SNAPSHOTS_DIR` override in shape and semantics, against the same class of problem: a checkout-shared path that concurrent cascor processes collide on. The direct-CLI tier resolves the default from the env var at import time; the service tier reads it at call time.

**This is the release that makes it consumable.** `JUNIPER_CASCOR_LOG_DIR` merged on `main` but appeared in **no tag** — so every downstream that installs a released cascor still collided on `logs/juniper_cascor.log`. juniper-ml's `run_suite.py` refuses `app: cascor` with `max_parallel > 1` for exactly this reason, and its refusal comment states the condition for lifting it: *"a cascor version floor asserted at suite load"*. That floor now has a version to point at.

---

## Bug Fixes

- **Shutdown joins training and releases the candidate pool before uvicorn's SIGTERM re-raise kills the process** (#589). The re-raise skips `atexit` entirely, so cleanup registered there never ran under any fleet stop tool.
- **Forkserver children no longer re-run `main.py`'s body** (#588); the plotter import is lazy.
- **Workers no longer hang at exit on advisory-queue flush**; shutdown joins share one deadline (#586, #587).
- **The trainer is preloaded in the forkserver** (#592, F3), plus a corrected fork-context comment and four guards.
- **`start` while paused, reset replay, and create-while-resume-ready are rejected** (#584).
- **`phase_started_at` is emitted tz-aware UTC, not naive local** (#594, F-CANOPY-026).
- **An absent snapshot `format` attribute names itself** instead of reading as the string `'None'` (#575).

---

## Improvements

- Every route declares an `operation_id`; every envelope route a `response_model` (#593).
- One snapshot root for every stack origin, `<repo>/cascor-snapshots/`; `snapshot_cli cleanup` is dry-run by default and refuses the shared root.
- `namespaces = false` in `pyproject.toml` — the actual guard keeping artifacts off PyPI.
- Snapshot `.gitignore` rules are directory-anchored.
- Memory-budget gate promoted to BLOCKING.

---

## API Changes

**The documentation surface is no longer auth-exempt (#599).** `/docs`, `/openapi.json` and `/redoc` are removed from `EXEMPT_PATHS`. This was **not** a live exposure — `docs_enabled = not settings.api_keys` already un-mounts them whenever auth is on — but the exempt entries made *"docs enabled"* and *"docs public"* the same switch, since `_is_exempt()` ignores whether a key is configured. Removing them decouples the two, matching juniper-data (APD-DATA-024) and juniper-service-core 0.6.0.

Observable change, auth-enabled deployments only: an unauthenticated `GET /docs` returns **401 instead of 404**.

---

## Test Results

Full suite green on `main` at the release commit. The lifecycle and pool fixes each ship with the regression that pins them.

---

## Upgrade Notes

**If you generate a client SDK from the OpenAPI document, regenerate it.** Every route now declares an explicit `operation_id` (#593), so **generated method names will change**. Derived ids were unstable under refactoring; this is the intended fix, but it is the change most likely to break a downstream build.

**`JUNIPER_CASCOR_LOG_DIR` is opt-in.** Unset or blank keeps the previous behaviour exactly, so no deployment changes on upgrade.

---

## Known Issues

- juniper-ml's `run_suite.py` still refuses `app: cascor` with `max_parallel > 1`. This release supplies the *released* version its comment requires, but lifting the refusal is a separate juniper-ml change — it must assert a cascor floor at suite load, because against an older cascor the env export is silently ignored and parallel cells would race the shared log exactly as before, with no signal.

---

## What's Next

- Assert the cascor `>=0.10.0` floor at suite load in juniper-ml's `run_suite.py` and lift the Q-6 refusal.

---

## Version History

| Version | Date | Focus |
|---|---|---|
| 0.10.0 | 2026-08-30 | process lifecycle + concurrent-instance isolation |
| 0.9.0 | 2026-08-14 | — |
| 0.8.0 | 2026-08-10 | — |

---

## Links

- Changelog: [`CHANGELOG.md`](../../CHANGELOG.md)
