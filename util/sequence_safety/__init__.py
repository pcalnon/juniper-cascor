"""Juniper sequence-safety gates -- compositional-loss screens for the PR flood.

Ecosystem port (juniper-cascor) of juniper-ml's 2026-07-28 Cursor-fleet flood census
(Proposal P2; the flood-remediation analysis
``notes/JUNIPER_2026-07-28_JUNIPER-ML_CURSOR-PR-FLOOD-REMEDIATION-ANALYSIS.md`` S3 / S4
item 8, which lives in the juniper-ml repo). The flood damage was *compositional*: every
PR was individually green, but serial same-file merges into main fused or silently
deleted sibling content, and a deleted test cannot fail -- so flake8 / mypy / pre-commit
all stayed green while whole test classes and doc sections disappeared. The 2026-08-05
cascor storm triage found this same compositional-loss net to be cascor's one remaining
gap, which this package closes (ADVISORY, non-blocking).

Two ref-diff screens answer the one question the per-PR checks could not:

  * ``symbol_loss_check.py``    -- AST symbol inventory of BASE vs HEAD for the
    in-scope Python surface (``src/**/*.py``, which includes ``src/tests/**``); FAIL on
    a silently deleted (LOST) def/class/method, a shrunk-past-threshold (WEAKENED)
    body, or a duplicated (DUPLICATED) member, with a qualified-name / body-similarity
    relocation downgrade and a ``Allow-Symbol-Loss:`` commit-trailer escape hatch.
  * ``docs_additions_check.py`` -- markdown deletion-magnitude screen of BASE vs HEAD
    for ``AGENTS.md`` + ``docs/**`` + ``notes/**``; FAIL on a deleted heading or a run
    of >= N consecutive deleted lines, WARN on small in-place swaps, with a
    ``Allow-Docs-Rewrite:`` commit-trailer escape hatch.

Both are pure git + stdlib (no network, no gh, no pip) and path-invoked. cascor's
pre-commit scopes black / isort / flake8 to ``^src/``, so ``util/`` has no lint gate of
its own; these modules are behaviourally identical to juniper-ml's, whose dedicated
``tests/test_symbol_loss_check.py`` + ``tests/test_docs_additions_check.py`` remain the
authoritative regression suite (see this PR's body for the cascor test-deferral note).
The per-PR ``sequence-safety.yml`` workflow runs both ADVISORY over a PR's
``base..HEAD``, and the post-merge ``main-verify.yml`` workflow runs both over a
catch-up base .. ``<merge>`` so the compositional-loss net fires no matter who merged or
what they bypassed -- the only gate the always-bypass actors cannot skip.

Project: juniper-cascor
Sub-Project: flood-remediation sequence-safety gates
Application: juniper_cascor
Author: Paul Calnon
Created: 2026-08-07
Status: permanent utility
Provenance: ecosystem port of juniper-ml util/sequence_safety/ (ml#873 / #880 / #928).
"""
