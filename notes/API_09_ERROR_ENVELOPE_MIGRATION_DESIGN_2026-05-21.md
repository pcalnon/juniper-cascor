# API-09: HTTPException → ResponseEnvelope Migration Design

**Date**: 2026-05-21
**Status**: Design (pre-implementation)
**Tracks**: API-09 in the v7 outstanding-development roadmap (juniper-ml/notes/JUNIPER_2026-05-25_JUNIPER-ECOSYSTEM_OUTSTANDING-DEVELOPMENT-ITEMS-V7-IMPLEMENTATION-ROADMAP.md §21)
**Author**: Claude Code (session 2026-05-21)
**Predecessor**: CFG-03 (#287), CFG-05 (#289) — both shipped in this session

---

## 1. Problem statement

`juniper-cascor`'s API has **two coexisting error response shapes**, and which one a caller receives depends on which Python exception the route raised, not on any property of the request or the route itself:

| Exception raised in route | Response shape | Source |
| --- | --- | --- |
| `raise HTTPException(status_code=..., detail=...)` | `{"detail": "..."}` | FastAPI default handler (built-in) |
| `raise ValueError(...)` | `{"status": "error", "error": {"code": "VALIDATION_ERROR", "message": "...", "detail": null}, "meta": {...}}` | Custom handler in [`api/app.py:482-488`](../src/api/app.py#L482-L488) |
| `raise Exception(...)` (or anything else uncaught) | `{"status": "error", "error": {"code": "INTERNAL_ERROR", "message": "...", "detail": null}, "meta": {...}}` | Custom handler in [`api/app.py:490-496`](../src/api/app.py#L490-L496) |
| Successful response | `{"status": "success", "data": {...}, "meta": {...}}` (`ResponseEnvelope`) | Per-route wrapping via `success_response()` |

Clients have to parse two error shapes to extract a human-readable message. This shows up concretely at [`juniper-cascor-client/juniper_cascor_client/client.py:398`](../../juniper-cascor-client/juniper_cascor_client/client.py#L398):

```python
error_msg = body.get("detail", response.text)
```

— the `body.get("detail", ...)` half is the `HTTPException` shape; the `response.text` fallback is the degraded path that fires when a `ValueError`/unhandled exception returns the envelope shape (it then reports the entire JSON blob as the error message, which is hostile).

API-09 in the v7 roadmap proposes registering a custom `HTTPException` handler in `create_app()` so all error responses share the `ResponseEnvelope` shape and clients can rely on a single parse path.

---

## 2. Current behavior (concrete code references)

### 2.1 Server side — cascor

`src/api/app.py:399 create_app(...)` already registers two custom exception handlers (validated against `origin/main`):

```python
# src/api/app.py:481-496
# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    logger.debug("Validation error: %s", exc)
    return JSONResponse(
        status_code=400,
        content=error_response("VALIDATION_ERROR", "Invalid request parameters"),
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content=error_response("INTERNAL_ERROR", "Internal server error"),
    )
```

`HTTPException` is **not** registered, so FastAPI's built-in handler emits the bare `{"detail": "..."}` body. `HTTPException` is more specific than `Exception`, so the general `Exception` handler does **not** intercept it.

### 2.2 Server side — raise sites

`grep -rn "raise HTTPException" src/api/routes/ | wc -l` → **66 call sites** across:

| File | Approx. count |
| --- | --- |
| `routes/snapshots.py` | ~20 |
| `routes/network.py` | ~10 |
| `routes/training.py` | ~10 |
| `routes/workers.py` | ~5 |
| `routes/dataset.py` | ~3 |
| `routes/admin.py` | ~3 |
| `routes/health.py` | ~3 |
| `routes/metrics.py`, `routes/history.py`, `routes/decision_boundary.py` | remainder |

Status codes in use: 400, 403, 404, 409, 422, 500, 503.

### 2.3 Existing schema — `ResponseEnvelope` + `ErrorResponse`

[`src/api/models/common.py:53-105`](../src/api/models/common.py#L53-L105):

```python
class ResponseEnvelope(BaseModel):
    """Standard API response envelope."""
    status: str = "success"
    data: Any = None
    meta: Meta = Field(default_factory=Meta)

class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: str | None = None

class ErrorResponse(BaseModel):
    status: str = "error"
    error: ErrorDetail
    meta: Meta = Field(default_factory=Meta)

def error_response(code: str, message: str, detail: str | None = None) -> dict:
    return ErrorResponse(
        error=ErrorDetail(code=code, message=message, detail=detail),
    ).model_dump()
```

The schema for the migrated `HTTPException` body is therefore already defined — we just need to populate it from the exception.

### 2.4 Test-side blast radius — cascor

`grep -rn 'response.json()["detail"]|...' src/tests/ | wc -l` → **36 assertions** across **7 files**:

| File | Assertion count |
| --- | --- |
| `tests/unit/api/test_snapshot_route_coverage.py` | 15 |
| `tests/unit/api/test_phase2_routes.py` | 7 |
| `tests/unit/api/test_training_route_coverage.py` | 5 |
| `tests/unit/api/test_network_route_coverage.py` | 4 |
| `tests/unit/test_remaining_coverage_deep.py` | 3 |
| `tests/unit/api/test_worker_routes.py` | 1 |
| `tests/integration/api/test_candidate_pool_invariants.py` | 1 |

All assertions look like `assert "<expected-msg>" in response.json()["detail"]`.

### 2.5 Test-side blast radius — wire-compat snapshots

No existing wire-compat snapshot (`test_*wire*.py`) explicitly pins the `HTTPException` response shape. The four wire-compat tests
(`test_messages_wire_compat.py`, `test_metrics_obs_wire_01.py`, `test_metrics_obs_wire_02.py`, `test_r2_1_4_wire_compat.py`) all pin WebSocket message envelopes or successful-path HTTP responses. No pre-existing snapshot needs to be updated, but **a new snapshot pinning the migrated error shape should be added** so we don't regress in the other direction.

### 2.6 Client-side blast radius — juniper-cascor-client

[`juniper-cascor-client/juniper_cascor_client/client.py:394-414`](../../juniper-cascor-client/juniper_cascor_client/client.py#L394-L414):

```python
try:
    body = response.json()
except (ValueError, JSONDecodeError):
    error_msg = response.text
else:
    error_msg = body.get("detail", response.text)
status = response.status_code
if status == 422 or status == 400:
    raise JuniperCascorValidationError(error_msg)
elif status == 404:
    raise JuniperCascorNotFoundError(error_msg)
elif status == 409:
    raise JuniperCascorConflictError(error_msg)
elif status == 503:
    raise JuniperCascorServiceUnavailableError(error_msg)
else:
    raise JuniperCascorClientError(f"HTTP {status}: {error_msg}")
```

If cascor switches its `HTTPException` body to `ResponseEnvelope` without a client-side update, `body.get("detail", response.text)` returns `None` (because the new shape's top-level keys are `status`, `error`, `meta` — no `detail`), and the client falls back to `response.text` (the whole JSON blob as a string). Error messages degrade from `"No network loaded"` to `'{"status":"error","error":{"code":"HTTP_404","message":"No network loaded","detail":null},"meta":{...}}'`.

`test_client_update_params.py:64` uses `json={"detail": "..."}` to mock a cascor 404 response — a single client-side test would need updating to mock the new shape (or rather, mock **both** shapes during the deprecation window).

### 2.7 Downstream consumers — juniper-canopy

`grep` for `response.json()["detail"]` in canopy returned **only canopy's own tests testing canopy's own API surface** — not cascor responses. Canopy reaches cascor exclusively through `juniper-cascor-client`'s `JuniperCascor*Error` exception types (not by parsing JSON directly). **Updating cascor-client therefore covers canopy with no canopy-repo changes.**

### 2.8 Other downstream consumers

- `juniper-deploy`'s Docker health checks use HTTP status codes only (200 vs. ≥400), not response bodies. **No impact.**
- Browser dashboards / hand-rolled HTTP curl scripts in `notes/` or `docs/` may quote the old `{"detail": ...}` shape. **Documentation-only impact, low risk.**

---

## 3. Approach matrix

### Approach A — Big-bang single PR (roadmap's implicit recommendation)

Add `@app.exception_handler(HTTPException)` to `create_app()`, rewrite all 36 cascor test assertions, update cascor-client `_request()` to read the new shape, update cascor-client tests, ship as one PR.

| | |
| --- | --- |
| **Pros** | Single atomic change; no transitional code; easiest to reason about end-state. |
| **Cons** | 36-test rewrite + cross-repo PR coordination + version pin update in cascor-client; high blast radius; conflict-prone; clients on older cascor-client versions break the moment they upgrade cascor without upgrading client; tight coordination across **3 repos** (cascor, cascor-client, possibly cascor-client release-and-pin in juniper-ml's `[clients]` extra). |
| **Risk** | **High.** Any downstream user pinned to an older cascor-client breaks silently (degraded error messages). |

### Approach B — Dual-shape envelope (client-friendly transitional)

Make cascor emit a response that satisfies **both** shapes during the deprecation window:

```json
{
  "status": "error",
  "error": {"code": "HTTP_404", "message": "No network loaded", "detail": null},
  "meta": {...},
  "detail": "No network loaded"
}
```

The `"detail"` top-level key is added as a deprecated alias of `error.message` for the duration of the transition. Clients reading either shape continue to work. After cascor-client is updated and pinned, drop the top-level `"detail"` in a follow-up.

| | |
| --- | --- |
| **Pros** | Zero client breakage during rollout; clients on older cascor-client versions keep getting useful error messages; gives cascor-client time to be released and pinned independently; tests assert on **either** key as appropriate. |
| **Cons** | Schema carries a known-temporary field; needs a removal-tracking issue or CHANGELOG note for the follow-up cleanup; minor wire-format weirdness during the transition. |
| **Risk** | **Low.** Worst case is forgetting to clean up the `"detail"` alias — annoying, not breaking. |

### Approach C — Per-exception-subclass opt-in

Define a `EnvelopedHTTPException(HTTPException)` subclass; register the handler only for the subclass; migrate each `raise HTTPException` call site to `raise EnvelopedHTTPException` in subsequent PRs (one per route module).

| | |
| --- | --- |
| **Pros** | Smallest per-PR blast radius (1 route module at a time); each migration PR independently revertible; lowest risk of merge conflicts. |
| **Cons** | Schema-divergence period stretches across many PRs (months); during the migration, the API has **three** error shapes coexisting (raw `HTTPException`, `EnvelopedHTTPException`, custom-handler `Exception`/`ValueError`); harder to reason about; client side has to handle all three until the last route migrates. |
| **Risk** | **Medium.** Sustained schema divergence period is itself a problem; loses the unification benefit the migration was supposed to provide. |

### Approach D — Defer indefinitely (do nothing)

Document the dual-shape reality, close API-09 as "won't fix", move on.

| | |
| --- | --- |
| **Pros** | Zero work; zero risk. |
| **Cons** | The original API-09 defect (dual error formats, hostile clients) remains; cascor-client's `body.get("detail", response.text)` continues to require manual maintenance every time the schema changes. |
| **Risk** | **Low** in the short term; the technical debt continues to accrue. |

---

## 4. Recommendation: Approach B (dual-shape envelope) with 3-PR rollout

Approach B delivers the API-09 unification benefit (single client parse path post-deprecation) without the cross-repo coordination risk of Approach A or the prolonged divergence of Approach C.

### 4.1 PR sequence

**PR 1 (cascor, M)** — Add `HTTPException` handler emitting the dual-shape envelope.

- Add `@app.exception_handler(HTTPException)` in `create_app()` immediately after the existing `ValueError` handler.
- Handler:

  ```python
  @app.exception_handler(HTTPException)
  async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
      envelope = error_response(
          code=f"HTTP_{exc.status_code}",
          message=str(exc.detail) if exc.detail is not None else f"HTTP {exc.status_code}",
      )
      # API-09 deprecation alias — top-level "detail" key for clients on
      # the pre-migration shape (notably cascor-client < X.Y.0 which reads
      # body.get("detail", response.text)). To be removed in a future
      # release after cascor-client adopts the envelope-aware parser.
      envelope["detail"] = envelope["error"]["message"]
      return JSONResponse(
          status_code=exc.status_code,
          content=envelope,
          headers=exc.headers,
      )
  ```

- New regression test `src/tests/unit/api/test_api_09_http_exception_envelope.py` covering:
  - 400 / 401 / 403 / 404 / 409 / 422 / 500 / 503 each return the dual shape with the right `status_code`, `error.code`, `error.message`, AND the legacy `detail` top-level alias.
  - `headers=exc.headers` passthrough preserves headers (`WWW-Authenticate` for 401, `Retry-After` for 429 etc.).
- The 36 existing `response.json()["detail"]` assertions in cascor's test suite **continue to pass unchanged** because the top-level `"detail"` alias is preserved.
- CHANGELOG entry: `Added` (new envelope shape) + `Deprecated` (top-level `detail` alias) with the same deprecation-period framing as CFG-03/CFG-05.
- A new wire-compat test (`test_api_09_http_exception_wire_compat.py`) pins both the envelope AND the alias to prevent silent regression on either.

**PR 2 (cascor-client, S)** — Teach `_request()` to read both shapes.

- Update [`client.py:398`](../../juniper-cascor-client/juniper_cascor_client/client.py#L398):

  ```python
  # API-09: cascor switched HTTPException responses to the ResponseEnvelope
  # shape. During the deprecation window the server emits both a top-level
  # "detail" alias and the envelope's nested error.message; prefer the
  # envelope's message when present so the alias can eventually be removed
  # server-side without breaking us.
  if isinstance(body, dict):
      error_obj = body.get("error")
      if isinstance(error_obj, dict) and "message" in error_obj:
          error_msg = error_obj["message"]
      else:
          error_msg = body.get("detail", response.text)
  else:
      error_msg = response.text
  ```

- Regression test mocking **both** shapes (legacy `{"detail": ...}` and new `{"status":"error","error":{"message":...}}`) and asserting both produce the same `JuniperCascor*Error` with the same `.args[0]` message.
- Minor `juniper-cascor-client` version bump (patch) + CHANGELOG entry.
- Update `juniper-ml`'s `[clients]` extra in `pyproject.toml` to the new minimum cascor-client version.

**PR 3 (cascor, S, post-soak)** — Drop the top-level `detail` alias.

- Remove the `envelope["detail"] = envelope["error"]["message"]` line.
- Update the new wire-compat test to assert the alias is **absent**.
- Update the new regression test to assert the alias is **absent**.
- CHANGELOG entry under `Removed` (or `Changed (potentially breaking)` if any external consumers still depend on the alias).
- Bump cascor's minimum cascor-client pin to the version shipped in PR 2.

PR 3 only ships after enough deployment cycles have passed that PR 2's cascor-client release has been adopted (≥1 release cycle, similar to CFG-03/CFG-05).

### 4.2 Why the dual-shape envelope (and not just a custom handler) is the right call

- The 36 cascor test assertions are not the real cost — they could be rewritten in one sitting. The real cost is the **silent client-side degradation** on deployments where cascor and cascor-client release cadences are not aligned. This is exactly the failure mode the CFG-03/CFG-05 deprecation pattern was designed to avoid.
- The dual-shape envelope is wire-format-equivalent to the eventual end state on every key clients care about (`status`, `error.code`, `error.message`, `meta`) — the temporary `"detail"` alias is a single extra field that costs ~8 bytes per error response and is removable without further client changes.
- Treating the change as 3 small PRs across 2 repos with a soak period in the middle matches the operating convention established by CFG-03 (#287) and CFG-05 (#289) earlier in this session.

### 4.3 Out-of-scope for this design

- Migrating `juniper-canopy`'s own routes to the same envelope (API-09 is scoped to juniper-cascor in §21 of the roadmap).
- Migrating `juniper-data`'s error responses (separate roadmap entry: API-05 / XREPO-15).
- Replacing the `error.code` namespace (`HTTP_NNN`) with semantic codes (e.g. `NETWORK_NOT_FOUND`, `LIFECYCLE_NOT_INITIALIZED`) — that is desirable but properly belongs to a follow-up "API-09b: semantic error codes" entry. The `HTTP_NNN` shape is intentionally chosen as a low-risk default so the PR doesn't require touching the body of each `raise HTTPException(...)` site.

---

## 5. Open questions / things to confirm with Paul

1. **OK to commit the 3-PR plan?** PR 1 is shippable on its own (the cascor handler change is self-contained and backward-compatible for clients). PR 2 + PR 3 require explicit follow-through; if abandoned mid-rollout, cascor ends up emitting the dual shape forever, which is the second-worst outcome (better than no migration but worse than completing the migration).
2. **`headers=exc.headers` passthrough — preserve or strip?** FastAPI's default handler passes them through; the design preserves that. The alternative is to strip headers (cleaner but loses `WWW-Authenticate` / `Retry-After`), which would be a behavior regression vs. the FastAPI default.
3. **`error.code` namespace — `HTTP_NNN` (recommended) vs. uppercase string?** Roadmap suggested `code = exc.status_code` (integer); this design uses `f"HTTP_{exc.status_code}"` (string) to match the existing `error.code` convention (`VALIDATION_ERROR`, `INTERNAL_ERROR`). Confirm.
4. **CHANGELOG categorization for PR 1?** This is genuinely "Added + Deprecated" (new envelope shape added; top-level `detail` alias deprecated from day one). Roadmap convention from CFG-03/CFG-05 puts the entry under `Deprecated`. Confirm we want the same here, OR a split entry under both `Added` and `Deprecated`.

---

## 6. Verification checklist (for whoever picks up PR 1)

- [ ] `@app.exception_handler(HTTPException)` registered in `create_app()` after the `ValueError` handler.
- [ ] Handler returns dual-shape envelope (envelope + `"detail"` alias).
- [ ] `headers=exc.headers` passthrough preserved.
- [ ] `error.code = f"HTTP_{exc.status_code}"`.
- [ ] `error.message = str(exc.detail) if exc.detail is not None else f"HTTP {exc.status_code}"`.
- [ ] New regression test `test_api_09_http_exception_envelope.py` covers 400/401/403/404/409/422/500/503 + header passthrough + alias-presence.
- [ ] New wire-compat snapshot `test_api_09_http_exception_wire_compat.py` pins both shapes byte-for-byte.
- [ ] All 36 existing `response.json()["detail"]` test assertions continue to pass unchanged.
- [ ] CHANGELOG entry under `[Unreleased] > Deprecated` (top-level `detail` alias) AND under `Added` (envelope-shape `HTTPException` response) — or one combined entry, per Paul's call on §5 Q4.
- [ ] `pre-commit run --files src/api/app.py src/tests/unit/api/test_api_09_http_exception_envelope.py src/tests/unit/api/test_api_09_http_exception_wire_compat.py CHANGELOG.md` passes.
- [ ] `conda run -n JuniperCascor1 python -m pytest src/tests/unit/api/test_api_09_http_exception_envelope.py src/tests/unit/api/test_api_09_http_exception_wire_compat.py -q --timeout=60` passes.
- [ ] Full unit suite passes: `conda run -n JuniperCascor1 python -m pytest src/tests/unit/ -q --timeout=120` (must not regress on the 36 `detail`-key assertions).

---

## 7. References

- v7 roadmap §21 API-09 (juniper-ml/notes/JUNIPER_2026-05-25_JUNIPER-ECOSYSTEM_OUTSTANDING-DEVELOPMENT-ITEMS-V7-IMPLEMENTATION-ROADMAP.md)
- CFG-03 (#287, juniper-cascor) — sibling migration: `SENTRY_SDK_DSN` → `JUNIPER_CASCOR_SENTRY_DSN`, established the deprecation-with-stderr-drift pattern this design follows.
- CFG-05 (#289, juniper-cascor) — sibling migration: `CASCOR_LOG_LEVEL` → `JUNIPER_CASCOR_LOG_LEVEL`, same pattern.
- `src/api/app.py:399-498` `create_app()` — handler-registration site.
- `src/api/models/common.py:53-105` — `ResponseEnvelope` / `ErrorResponse` / `error_response()` definitions.
- `juniper-cascor-client/juniper_cascor_client/client.py:394-414` — downstream parser to update in PR 2.
