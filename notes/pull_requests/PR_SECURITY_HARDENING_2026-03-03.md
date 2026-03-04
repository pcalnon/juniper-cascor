# Pull Request: Security Hardening — Middleware, WebSocket Auth, Pickle Verification, and Scanning

**Date:** 2026-03-03
**Version(s):** 0.3.17 → 0.4.0
**Author:** Paul Calnon
**Status:** READY_FOR_MERGE

---

## Summary

Comprehensive security hardening for juniper-cascor as part of the cross-ecosystem security audit. Adds security headers, request body limits, error sanitization, restrictive defaults, WebSocket authentication with message validation, HMAC pickle verification, /metrics auth, conditional API docs, CI hardening, and scheduled security scanning.

---

## Context / Motivation

A full security audit of the Juniper ecosystem identified 24 findings across 7 repositories. This PR addresses the juniper-cascor portion, including the most critical finding: WebSocket endpoints bypassing authentication entirely.

---

## Changes

### Security

- Added `SecurityHeadersMiddleware` with X-Content-Type-Options, X-Frame-Options, Referrer-Policy, Permissions-Policy, conditional HSTS
- Added `RequestBodyLimitMiddleware` with configurable max body size (default 10MB)
- Sanitized error responses in ValueError handler, training routes (`HTTPException(detail=str(e))`), and network routes
- Changed CORS origins default from `["*"]` to `[]`
- Changed rate limiting default from disabled to enabled
- Added WebSocket authentication — API key validation at connection accept, close code 4001 on invalid/missing key
- Added WebSocket message size limits and Pydantic schema validation for control commands
- Added HMAC signature verification before `pickle.loads()` in snapshot serializer
- Removed `/metrics` from authentication-exempt paths
- Added conditional API docs — disabled when API keys configured
- Removed `|| true` from Bandit CI step

### Added

- `.github/workflows/security-scan.yml` — Weekly Bandit SAST and pip-audit dependency scanning

### Changed

- Updated test fixtures for new security defaults

---

## Impact & SemVer

- **SemVer impact:** MINOR (0.3.17 → 0.4.0)
- **User-visible behavior change:** YES — CORS restrictive by default; rate limiting enabled; WS requires auth; API docs hidden when keys configured
- **Breaking changes:** NO — All configurable via environment variables
- **Performance impact:** NONE
- **Security/privacy impact:** HIGH — Addresses 11 of 24 ecosystem-wide findings including CRITICAL WebSocket auth bypass
- **Guarded by feature flag:** YES — All features configurable via env vars

---

## Testing & Results

### Test Summary

| Test Type   | Passed | Failed | Skipped | Notes              |
| ----------- | ------ | ------ | ------- | ------------------ |
| API Tests   | 264    | 0      | 0       | All tests passing  |

### Environments Tested

- JuniperCascor conda environment: All tests pass

---

## Verification Checklist

- [x] Security headers present on all responses
- [x] Error responses do not leak internal details
- [x] CORS rejects unknown origins by default
- [x] Rate limiting active by default
- [x] WebSocket connections require valid API key
- [x] WebSocket messages validated against Pydantic schema
- [x] Pickle deserialization verifies HMAC signature
- [x] `/metrics` requires authentication
- [x] Bandit CI failures now fail the build
- [x] All existing tests pass

---

## Files Changed

### New Components

- `.github/workflows/security-scan.yml` — Scheduled security scanning workflow

### Modified Components

**Backend:**

- `src/api/middleware.py` — Added SecurityHeadersMiddleware, RequestBodyLimitMiddleware
- `src/api/app.py` — Registered middleware, sanitized error handlers, conditional docs
- `src/api/settings.py` — Changed CORS and rate limiting defaults
- `src/api/routes/training.py` — Sanitized error responses
- `src/api/routes/network.py` — Sanitized error responses
- `src/api/websocket/control_stream.py` — Added auth, message validation, sanitized errors
- `src/snapshots/snapshot_serializer.py` — Added HMAC verification before pickle.loads()

**CI/CD:**

- `.github/workflows/ci.yml` — Removed `|| true` from Bandit step

**Tests:**

- `src/tests/unit/api/test_api_app.py` — Updated for new middleware and security defaults
- `src/tests/unit/api/test_api_settings.py` — Updated for new setting defaults
- `src/tests/unit/api/test_network_route_coverage.py` — Updated for sanitized errors
- `src/tests/unit/api/test_training_route_coverage.py` — Updated for sanitized errors

---

## Risks & Rollback Plan

- **Key risks:** Existing deployments without API keys configured will have WebSocket connections rejected; CORS wildcard users need explicit config
- **Rollback plan:** Set `CORS_ORIGINS=*`, `RATE_LIMIT_ENABLED=false`, `DOCS_ENABLED=true` to restore previous behavior; WebSocket auth bypassed when no API keys configured

---

## Related Issues / Tickets

- Related PRs: Security hardening PRs across all 7 Juniper repositories
- Phase Documentation: `juniper-ml/notes/SECURITY_AUDIT_PLAN.md`

---

## Notes for Release

**v0.4.0** — Security hardening release. Adds security headers, WebSocket authentication, HMAC pickle verification, restrictive defaults, and scheduled scanning. Part of cross-ecosystem audit addressing 24 findings.
