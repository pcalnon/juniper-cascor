# juniper-cascor — Control-Surface Bind-Guard & NAT-Inert IP Controls (SEC-F22 / SEC-F19)

**Project**: Juniper Cascade Correlation Neural Network
**Repository**: pcalnon/juniper-cascor
**Author**: Paul Calnon
**License**: MIT License
**Version**: 0.5.0
**Last Updated**: 2026-07-04

---

This note records the two defensive controls added to cascor as the symmetric
counterpart to the canopy remediation, and the invariants they enforce. It is
the local companion to the ecosystem design of record: juniper-ml
`notes/JUNIPER_CANOPY_CONTROL_SURFACE_AUTH_AND_NAT_DESIGN_2026-07-03.md`
(§4 Option A / §5 Option B / §7 Phases 1–2 / §8 D2+D4). Both findings are an
authorized defensive audit of the platform owner's own containerized stack.

## Threat model in one paragraph

cascor's control/worker WebSocket surface (`/ws/control`, `/ws/v1/workers`,
`/ws/training`) has **no app-layer authentication that a browser can hold**;
machine callers present an `X-API-Key`, but the effective perimeter for the
containerized stack is **network position** — the compose-level loopback
host-publish and `internal:true` networks. Inside that perimeter, Docker NAT
collapses every client to the bridge-gateway IP, so any control keyed on the
raw socket peer degrades to a single shared bucket. The two controls below
harden that reality without pretending the IP is an identity.

## D2 — startup bind-guard (SEC-F22)

**Invariant:** cascor refuses to start when it is configured to bind a
**non-loopback** interface unless the operator has provided at least one bind
attestation. The scheme is a **two-flag** attestation (identical across canopy /
cascor / juniper-deploy), each naming a distinct reason a non-loopback bind is
safe:

- Settings (both `bool = False`, `src/api/settings.py`):
  - `loopback_publish_attested`
    (env `JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED`) — the port is reachable
    **only** via a loopback-only host publish (the containerized default:
    `127.0.0.1:8200:8200` in compose; verifiable by the juniper-deploy
    preflight).
  - `auth_proxy_attested`
    (env `JUNIPER_CASCOR_AUTH_PROXY_ATTESTED`) — a fronting authenticating
    reverse proxy terminates access before the port (Phase-4; attestation only,
    no in-process verification).
- Enforcement: `enforce_bind_attestation_guard(settings)` in `src/api/app.py`,
  called at the top of the `lifespan` startup — **before** uvicorn binds the
  socket or any background thread is spawned. Fail-closed and **loud** (CRITICAL
  log), raising `NonLoopbackBindError`.
- Loopback (`127.0.0.0/8`, `::1`, `localhost`, IPv4-mapped-IPv6 loopback) →
  always start. Non-loopback + **neither** attestation → refuse (hard-fail;
  **no warning-only mode**). Non-loopback + **either** attestation (or both) →
  start, with a WARNING that names which attestation permitted the bind.
- Host is read from cascor's own settings (`settings.host`, from
  `JUNIPER_CASCOR_HOST`; port `JUNIPER_CASCOR_PORT=8200`), matching the runtime
  path: the container runs `python src/server.py` with `JUNIPER_CASCOR_HOST` set,
  so `settings.host` genuinely carries the bind host.
- The documented uvicorn factory entry point
  (`uvicorn api.app:create_app --factory --host ...`) passes the bind host to
  uvicorn rather than `JUNIPER_CASCOR_HOST`. `create_app()` mirrors those CLI
  `--host` / `--port` values into a transient settings copy before the lifespan
  guard runs so this path is covered by the same invariant.

This converts the auth design's load-bearing precondition ("do not ship the
un-fronted control surface on a public interface") from prose into an enforced
invariant, and closes the silent `JUNIPER_CASCOR_HOST=0.0.0.0` footgun.

**Deploy roll-out is owner-gated (Phase 1).** In `juniper-deploy` the container
binds `JUNIPER_CASCOR_HOST=0.0.0.0` behind a **loopback host-publish**
(`${BIND_HOST:-127.0.0.1}:…:8200`). Because the guard keys on `settings.host`
(`0.0.0.0`), enabling it in the deploy requires setting
`JUNIPER_CASCOR_LOOPBACK_PUBLISH_ATTESTED=true` there — the operator attesting
that the loopback host-publish fronts the port (a fronting proxy later would
instead attest `JUNIPER_CASCOR_AUTH_PROXY_ATTESTED=true`). That env/deploy change
is approved separately by the platform owner and is **not** part of the code PR
that adds the guard.

## D4 — WebSocket connection caps (SEC-F19)

Two caps are added alongside the existing per-IP cap, all in
`src/api/websocket/manager.py`:

- **Stack-absolute GLOBAL cap** (`ws_max_connections_global`, env
  `JUNIPER_CASCOR_WS_MAX_CONNECTIONS_GLOBAL`, default 200). One counter spanning
  **all** WS endpoints: `/ws/training` reserves via `connect`/`connect_pending`;
  `/ws/control` and `/ws/v1/workers` reserve via the new `try_admit` admission
  gate (they accept their own sockets and are not broadcast-eligible via the
  manager's active set). Over-cap → close `1013`. This is the availability /
  DoS-dampening backstop that survives NAT.
- **Per-identity cap** (`ws_max_connections_per_identity`, env
  `JUNIPER_CASCOR_WS_MAX_CONNECTIONS_PER_IDENTITY`, default 5), enforced on
  `/ws/control`, keyed on a non-reversible SHA-256 digest of the caller's
  `X-API-Key` token (`ws_identity_key`). Anonymous callers (auth disabled) are
  exempt and rely on the global + per-IP caps. Restores per-principal fairness
  independent of the (NAT-collapsed) source IP.

**Per-IP cap is inert-behind-NAT (documented, not removed).** The existing
`ws_max_connections_per_ip` (default 5) keys on `websocket.client[0]`. Behind
Docker NAT every client presents as the bridge gateway, so it becomes a single
shared bucket (one client's 5 sockets exhaust it for everyone — the HO-3
self-DoS). It is **DoS-dampening, not authentication**, and is kept unchanged;
the global + per-identity caps are the controls that survive NAT.

**Why `/ws/v1/workers` is GLOBAL-only (per-identity deferred).** A worker fleet
shares one auth token, so keying the worker WS per-identity on the token would
cap horizontal scaling; the unique server-assigned `worker_id` is only known
**after** registration (post-accept), so it is unavailable at the admission
point and, being unique per worker, would always count as one. Meaningful worker
per-identity keying is therefore not cleanly available; the global cap is the
worker minimum (design §8 OQ-2). Worker fleet size is additionally bounded by
the existing registry capacity cap and the optional per-IP connection rate
limiter. Worker per-identity keying is a documented follow-up.

## Explicitly deferred (Phase 4, owner-gated)

The fronting authenticating reverse proxy and trusted-`X-Forwarded-For`
client-IP resolution (design §5 Option A / §6 / §8 D6) are **not** built here.
Until a proxy exists, honoring `X-Forwarded-For` would be a footgun (any client
could forge identity), so it stays deferred with the invariant written down:
XFF is trusted **only** from a configured proxy IP once one is deployed. The
deterministic-metrics-subnet work (§8 D5) is a `juniper-deploy` change and is
also out of scope for this cascor code change.

## Tests

- `src/tests/unit/api/test_bind_guard.py` — loopback detection; the D2
  refuse/allow matrix (non-loopback+no-attest→refuse, +attest→start,
  loopback→start); the loud CRITICAL log; and the lifespan wiring (starting a
  non-loopback+no-attest app raises before bind).
- `src/tests/unit/api/test_ws_connection_caps.py` — the `ws_identity_key`
  digest; global-cap saturation across training + admission and its release on
  disconnect; per-identity rejection and cross-principal fairness under a shared
  peer IP; the anonymous-exempt path; and no global-slot leak on a per-identity
  rejection or failed `/ws/training` accept.
