# Log-record envelope and named-marker contract (P0.4, cascor#573)

Fixtures for `src/tests/unit/test_log_record_envelope_contract.py`, the test that enforces, in
**cascor's** CI, the log-format surface that **juniper-ml's** analysis tooling parses. Roadmap:
juniper-ml `notes/JUNIPER_2026-09-02_JUNIPER-CASCOR_LOGGING-REDESIGN-ROADMAP.md` §3.1 (P0.4).

A cascor change that renames a message, drops the `+` sentinel or adds milliseconds to a
timestamp leaves every other cascor test green, while juniper-ml's scripts silently stop matching.
A test in juniper-ml cannot fail a cascor PR, so the contract lives here.

## What a record looks like, per writer (observed, not documented)

Three writers share `logs/juniper_cascor.log`. The shapes below were measured on real captures
(`reference_*.txt`), and they correct the design documents in one respect: **Path B is second
resolution; the millisecond records are Path C**.

| shape | writer | example | where |
| --- | --- | --- | --- |
| `A_FILE` | Path A, the classmethod `Logger` | `+[f.py: func:LINE] (YYYY-MM-DD HH:MM:SS) [LEVEL] msg` | file sink |
| `A_CONSOLE` | Path A | `+[f.py: LINE] (YYYY-MM-DD HH:MM:SS) [LEVEL] msg` | stdout (redirected to `juniper-cascor.log` / `direct_cli.log` by the juniper-ml launchers) |
| `STD_FILE_S` | Path B, `logging.getLogger("juniper")` via `conf/logging_config.yaml` | `[f.py: func:LINE] (YYYY-MM-DD HH:MM:SS) [LEVEL] msg` | file sink |
| `STD_FILE_MS` | Path C, `api/observability.configure_logging` (RotatingFileHandler, no `datefmt`) | `[f.py: func:LINE] (YYYY-MM-DD HH:MM:SS,mmm) [LEVEL] msg` | file sink (service only) |
| `STD_CONSOLE` | Path B console handler, ERROR and above | `[f.py:LINE] (YY-MM-DD HH:MM:SS) [LEVEL] msg` — two-digit year | stdout |

- **Only Path A writes the `+` sentinel, and it is written by CODE** (`print(f"+{...}")` in
  `Logger._log_at_level`), not by any formatter string. A golden of formatter strings alone cannot
  see it drop; the emit-path tests can.
- **The run-verdict markers are Path B records.** `fit: Training completed.`,
  `train_candidates: Executing candidate training with N processes` and `Completed solving
  SpiralProblem instance` land in the file sink with no `+`. The candidate-worker markers
  (`CandidateUnit: train: ...`) are Path A.
- `conf/logging_config-CANOPY.yaml` carries a fourth copy of the prefix. **No loader reads it** at
  the time of writing (`git grep` finds it named only in `notes/history/`), so it is deliberately
  not pinned. If something starts loading it, pin it.

## Files

| file | what | produced by |
| --- | --- | --- |
| `envelope_golden.json` | Path A formatter constants and date format; the whole parsed `conf/logging_config.yaml`; the handlers Path B and Path C actually INSTALL (formats, `datefmt`, levels, file name, rotation); the shapes each reference capture carries | this test under `LOG_ENVELOPE_CAPTURE=1` |
| `marker_inventory.json` | 34 message markers juniper-ml anchors on: their literal fragments (in order), the file that emits each, the level a consumer can rely on, and every consumer by path | juniper-ml `util/ad-hoc/2026-09-22_p04_log_marker_census.py --emit-inventory` |
| `reference_*.txt` | excerpts of four REAL sinks: the P0.1 corpus run (direct CLI, cascor `8065ca0f`) file and stdout, and a 2026-09-01 service run's file and stdout | juniper-ml `util/ad-hoc/2026-09-22_p04_reference_capture_excerpt.py` |
| `reference_captures.json` | provenance for each excerpt: source path, size, sha256, cascor revision (or `null` where the run did not record one), and the source line numbers kept | the same script |

The only transformation applied to the captures is `/home/<user>/` → `~/`.

## When the test fails

The failure message names the juniper-ml consumers that will stop matching. Then:

1. **If the change is not intended**, it is a regression — fix it.
2. **If it is intended**, update the consumers in juniper-ml **in the same change set**, then:
   - message markers: re-run the census (`--emit-inventory` into this directory) and commit the
     new `marker_inventory.json`;
   - envelope / formatters / YAML: `LOG_ENVELOPE_CAPTURE=1 pytest src/tests/unit/test_log_record_envelope_contract.py`
     rewrites `envelope_golden.json`; review the diff before committing it.

Detecting a juniper-ml consumer that starts anchoring on a marker that is **not** in the
inventory is the other direction, and it runs on the juniper-ml side:
`util/ad-hoc/2026-09-22_p04_log_marker_census.py --check-inventory <this dir>/marker_inventory.json`
exits 1 on drift.

## How the test reaches the real emit path

`src/tests/conftest.py`'s session-scoped `_cache_logging_system` replaces `Logger._log_at_level`
with a no-op for the whole pytest session, so nothing in-process emits. The test therefore runs
`src/tests/helpers/log_envelope_emitter.py` in a **child interpreter**, which has no conftest,
drives all three writers into one file under a private `JUNIPER_CASCOR_LOG_DIR`, and reports the
handlers it installed. Each record's `func:LINE` is checked against the emitter's own AST, which
makes this the detector for roadmap P2.1 (moving frame capture inside `_log_at_level`).

## Proven to fail

juniper-ml `util/ad-hoc/2026-09-22_p04_harness_mutation_check.py` applies 14 mutations to the real
source and requires every one to fail this test, plus two controls to pass — the unmutated tree,
and P2.1 implemented as prescribed (`frame = cls._frm().f_back` inside `_log_at_level`). The
mutations: `+` dropped on each sink; microseconds on the Path A timestamp; the console record sent
to stderr; the console/file closures swapped; lazy `%`-args interpolation removed; P2.1 with
`.f_back` forgotten and with one hop too many; a marker's text edited; a marker's INFO emit site
demoted to DEBUG while its DEBUG twin keeps the literal alive; the `juniper` logger raised to
WARNING; the rotation count changed; Path C's file renamed; Path B's `datefmt` given milliseconds.
