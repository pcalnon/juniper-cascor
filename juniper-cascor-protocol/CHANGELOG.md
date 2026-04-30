# Changelog

All notable changes to the `juniper-cascor-protocol` package are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
with [PEP 440](https://peps.python.org/pep-0440/) pre-release identifiers.

## [Unreleased]

## [0.1.0] - 2026-04-30

### Changed

- **First stable promotion** (METRICS-MON R2.2.3 / seed-05). Promoted from pre-release to stable now that the first consumer (juniper-cascor server, [pcalnon/juniper-cascor#159](https://github.com/pcalnon/juniper-cascor/pull/159)) has shipped without surfacing a wire-format regression. **No public-API changes** vs `0.1.0a0` — same surface, same behavior; only the version string and trove classifier change.
- Trove classifier moved from `Development Status :: 3 - Alpha` to `Development Status :: 4 - Beta` to reflect the 0.1.x stability commitment.
- Consumers should pin `juniper-cascor-protocol>=0.1.0` going forward. Existing pins of `>=0.1.0a0` continue to resolve to the latest published version, which is now `0.1.0`.

### Notes

- The previous alpha (`0.1.0a0`) remains on PyPI for reproducibility of historical builds. Yanking is intentionally avoided; consumers can downgrade in a hotfix scenario by pinning explicitly.

## [0.1.0a0] - 2026-04-29

Initial publishable alpha. METRICS-MON R2.2.1 / seed-05.

### Added

- **Envelope subpackage** (`juniper_cascor_protocol.envelope`) — Pydantic v2 schemas for the ten typed `/ws/training` and `/ws/control` envelopes:
  - Training: `MetricsEnvelope`, `StateEnvelope`, `TopologyEnvelope`, `EventEnvelope`, `CascadeAddEnvelope`, `CandidateProgressEnvelope`, `InitialMetricsEnvelope` (with `InitialMetricsData` typed payload), `ChunkedMessageEnvelope` (with `ChunkedMessageData` typed payload — GAP-WS-18).
  - Control: `CommandResponseEnvelope` (with `CommandResponseData` typed payload), `ConnectionEstablishedEnvelope` (with `ConnectionEstablishedData` typed payload).
- **`validate_envelope(frame: dict) -> BaseEnvelope`** — consumer-facing validation helper. Returns the typed envelope when `frame["type"]` is recognized; returns `UnknownEnvelope` with a cardinality-bounded `type` label otherwise.
- **R1.1 cardinality bound** — unknown `type` strings tracked verbatim up to `UNKNOWN_TYPE_BUDGET = 16` distinct values per process, then collapsed to the `UNMATCHED_TYPE_LABEL = "_unmatched"` literal (mirrors the `juniper_observability.UNMATCHED_ENDPOINT_LABEL` strategy).
- **Worker subpackage** (`juniper_cascor_protocol.worker`) — `WorkerMessageType` `StrEnum` covering all `/ws/v1/workers` message types + `BinaryFrame` numpy codec for the side-channel tensor frames. **Numpy-only; no Pydantic at runtime** so the cascor-worker can adopt without violating the [METRICS-MON R2 exit-gate decision](https://github.com/pcalnon/juniper-ml/blob/main/notes/code-review/METRICS_MONITORING_R2_EXIT_GATE_WORKER_ADOPTION_2026-04-29.md).
- **Lazy subpackage layout** — the top-level `juniper_cascor_protocol` namespace re-exports only worker symbols. Pydantic is loaded only when a caller explicitly imports `juniper_cascor_protocol.envelope`. Tests pin both invariants (`test_worker_subpackage_does_not_import_pydantic`, `test_top_level_does_not_load_pydantic`).
- **Wire-compat snapshots** — every typed envelope has a byte-for-byte test against the pre-migration shape produced by `juniper-cascor/src/api/websocket/messages.py::create_*_message` so the cascor server's R2.2.2 migration cannot silently drift the contract.

### Notes

- This release ships the schemas only. The cascor server adopts them in R2.2.2, then the alpha is promoted to `0.1.0` in R2.2.3 after a soak. Consumers (cascor-client R2.2.4, canopy R2.2.5, worker R2.2.6) pin `>=0.1.0` once stable.
- See the design at [`notes/code-review/METRICS_MONITORING_R2.2_WS_FRAME_SCHEMA_DESIGN_2026-04-29.md`](https://github.com/pcalnon/juniper-ml/blob/main/notes/code-review/METRICS_MONITORING_R2.2_WS_FRAME_SCHEMA_DESIGN_2026-04-29.md) in juniper-ml.

[Unreleased]: https://github.com/pcalnon/juniper-cascor/compare/juniper-cascor-protocol-v0.1.0...HEAD
[0.1.0]: https://github.com/pcalnon/juniper-cascor/releases/tag/juniper-cascor-protocol-v0.1.0
[0.1.0a0]: https://github.com/pcalnon/juniper-cascor/releases/tag/juniper-cascor-protocol-v0.1.0a0
