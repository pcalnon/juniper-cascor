#!/usr/bin/env python3
"""Docs deletion-magnitude screen (G2 / G3 step 4) -- BASE vs HEAD markdown.

Ported from juniper-ml's docs census screen (Proposal P2 S2), itself productionized
(deletion-magnitude only, not the full LOST-IN-MERGE reconstruction) from
``util/ad-hoc/2026-07-28_docs_census_v2_c2.py``. juniper-ml's flood docs class (its
PR #801 / #803) was a *net section deletion* that no existing check could see: the
doc-links validator only catches dangling anchors, and markdownlint excludes ``notes/``
+ ``docs/`` (cascor's ``.pre-commit-config.yaml`` excludes exactly the same). So a merge
that dropped a whole runbook section stayed green -- the same gap the 2026-08-05 cascor
storm triage flagged as this repo's one remaining compositional-loss exposure.

A bare "any ``-`` hunk fails" rule is too blunt -- the UPDATE-target docs
(``REFERENCE.md``, the runbooks, the cheatsheet) take legitimate line *replacements* on
nearly every edit, so it would paint honest docs PRs red or train a reflex
``docs-rewrite`` waiver. Instead a **magnitude-gated** rule:

  * FAIL on a deleted Markdown **heading** line (a ``-`` hunk whose content matches
    ``^\\s{0,3}#{1,6}\\s``) UNLESS the same hunk also adds a heading (a retitle -> WARN).
  * FAIL on a run of **>= N consecutive deleted lines with no adjacent addition**
    (``added == 0 and deleted >= min_run``; default N = 5) -- the net-section-removal
    signature of the juniper-ml #801 / #803 incident.
  * WARN (annotate, not fail) on smaller deletions and small in-place swaps (a few
    deleted lines bracketed by additions -- a normal edit).

Blind spot (stated honestly, mirrors the symbol screen's WEAKENED note). A *lopsided
swap* that deletes a large block but adds a line or two in the same hunk evades the
pure-run rule (added > 0). The heading-deletion rule usually still catches it (sections
carry headings), but a section-body gut that removes no heading and adds one filler line
can slip to WARN. That residue is for human review, not this magnitude screen.

Escape hatch. A ``Allow-Docs-Rewrite: <path>[, ...]`` trailer in any commit of the
BASE..HEAD range waives the enumerated files (``*`` waives all docs in the diff); it
travels in git history so it works for both the per-PR and the post-merge gate. The
ergonomic ``docs-rewrite`` PR label is WARN-only and lives in the per-PR (G2) job, not
this ref-based module.

Scope. Docs cluster = ``AGENTS.md`` (and its ``CLAUDE.md`` symlink), ``docs/**/*.md``,
and ``notes/**/*.md``. An explicit ``--files`` list bypasses the scope filter (any
``.md`` path).

CLI::

    python util/sequence_safety/docs_additions_check.py --base <ref> --head <ref> \
        [--files PATH ...] [--repo-root DIR] [--min-run N] [--advisory] [--json]

Exit codes: 0 = clean (no unwaived FAIL), 1 = >= 1 unwaived FAIL, 2 = usage /
invocation error. WARN / WAIVED never fail.

Advisory mode (``--advisory``). Print every finding as usual but exit 0 even on an
unwaived FAIL (a top-level ``advisory: true`` is recorded and an ADVISORY note printed;
finding severities are left intact so the artifact keeps the ground truth). This is the
demotion the per-PR CI job applies when the owner attaches the ``docs-rewrite`` label --
a blanket per-PR override downgraded to WARN-only (P2 SF5), so the auditable enumerated
``Allow-Docs-Rewrite`` commit trailer stays the primary waiver. Exit 2 (invocation
error) is NEVER masked by ``--advisory``.

Project: juniper-cascor
Sub-Project: flood-remediation sequence-safety gates
Application: juniper_cascor
Author: Paul Calnon
Created: 2026-08-07
Status: permanent utility
Provenance: ports juniper-ml ``util/sequence_safety/docs_additions_check.py`` (ml#873 /
    #880 / #928; flood-remediation analysis §4 item 8) unchanged in behaviour -- the docs
    cluster scope (AGENTS.md + docs/** + notes/**) is identical for cascor, so only this
    docstring's repo-specific provenance text differs from the juniper-ml original.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_MIN_RUN = 5  # >= this many consecutive deleted lines (no adjacent add) -> FAIL

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_HUNK_RE = re.compile(r"^@@ ")


def in_docs_scope(path: str) -> bool:
    """AGENTS.md (+ its CLAUDE.md symlink), docs/**/*.md, notes/**/*.md."""
    if path in ("AGENTS.md", "CLAUDE.md"):
        return True
    if path.endswith(".md") and (path.startswith("docs/") or path.startswith("notes/")):
        return True
    return False


# ---- git helpers (standalone so the script is path-invokable) --------------


def _git(root: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)


def resolve_ref(root: str, ref: str) -> Optional[str]:
    cp = _git(root, "rev-parse", "--verify", "-q", f"{ref}^{{commit}}")
    out = cp.stdout.strip()
    return out if cp.returncode == 0 and out else None


def changed_files(root: str, base: str, head: str) -> list[str]:
    cp = _git(root, "diff", "--name-only", f"{base}...{head}")
    return [ln for ln in cp.stdout.splitlines() if ln]


def file_diff(root: str, base: str, head: str, path: str) -> str:
    """Unified=0 base->head diff for one file (minimal, tight hunks)."""
    return _git(root, "diff", "--unified=0", "--no-color", base, head, "--", path).stdout


def range_messages(root: str, base: str, head: str) -> str:
    return _git(root, "log", "--format=%B", f"{base}..{head}").stdout


# ---- hunk parsing ----------------------------------------------------------


@dataclass
class Hunk:
    deleted: list[str] = field(default_factory=list)  # content of '-' lines
    added: list[str] = field(default_factory=list)  # content of '+' lines


def parse_hunks(diff_text: str) -> list[Hunk]:
    """Split a unified diff into hunks, collecting deleted / added line contents.

    With ``--unified=0`` each hunk's deleted lines are contiguous in the old file, so a
    hunk with ``added == 0`` is a run of that many consecutive deletions with no
    adjacent addition -- exactly the P2 S2 magnitude signal.
    """
    hunks: list[Hunk] = []
    cur: Optional[Hunk] = None
    for line in diff_text.splitlines():
        if _HUNK_RE.match(line):
            cur = Hunk()
            hunks.append(cur)
            continue
        if cur is None:
            continue  # pre-hunk file header (diff --git / index / --- / +++)
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("-"):
            cur.deleted.append(line[1:])
        elif line.startswith("+"):
            cur.added.append(line[1:])
    return hunks


# ---- classification --------------------------------------------------------


@dataclass
class Finding:
    path: str
    reason: str  # heading-deletion | deletion-run | small-deletion
    severity: str  # FAIL | WARN | WAIVED
    detail: dict = field(default_factory=dict)


def classify_file(path: str, hunks: list[Hunk], min_run: int) -> list[Finding]:
    findings: list[Finding] = []
    for h in hunks:
        deleted, added = len(h.deleted), len(h.added)
        if deleted == 0:
            continue  # pure addition -- the additions-only happy path
        del_headings = [ln for ln in h.deleted if _HEADING_RE.match(ln)]
        add_headings = [ln for ln in h.added if _HEADING_RE.match(ln)]
        if del_headings and not add_headings:
            findings.append(Finding(path, "heading-deletion", "FAIL", {"headings": [ln.strip()[:120] for ln in del_headings], "deleted": deleted, "added": added}))
        elif added == 0 and deleted >= min_run:
            findings.append(Finding(path, "deletion-run", "FAIL", {"deleted": deleted, "min_run": min_run}))
        else:
            findings.append(Finding(path, "small-deletion", "WARN", {"deleted": deleted, "added": added}))
    return findings


# ---- escape-hatch trailer parsing ------------------------------------------

_ALLOW_RE = re.compile(r"^\s*Allow-Docs-Rewrite:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_allow_trailers(messages: str) -> tuple[set[str], bool]:
    """Return (enumerated file tokens, wildcard_seen). A ``*`` waives all docs files."""
    allowed: set[str] = set()
    wildcard = False
    for m in _ALLOW_RE.finditer(messages or ""):
        for tok in re.split(r"[,\s]+", m.group(1).strip()):
            tok = tok.strip()
            if not tok:
                continue
            if tok == "*":
                wildcard = True
                continue
            allowed.add(tok)
    return allowed, wildcard


def _waives(path: str, allowed: set[str], wildcard: bool) -> bool:
    if wildcard:
        return True
    return path in allowed or path.rsplit("/", 1)[-1] in allowed


def apply_waivers(findings: list[Finding], allowed: set[str], wildcard: bool) -> None:
    for f in findings:
        if f.severity == "FAIL" and _waives(f.path, allowed, wildcard):
            f.severity = "WAIVED"
            f.detail = {**f.detail, "waived_by": "Allow-Docs-Rewrite trailer"}


# ---- driver ----------------------------------------------------------------


def run(root: str, base: str, head: str, files: Optional[list[str]], min_run: int) -> tuple[int, dict]:
    base_sha = resolve_ref(root, base)
    head_sha = resolve_ref(root, head)
    if base_sha is None or head_sha is None:
        bad = base if base_sha is None else head
        return 2, {"error": f"could not resolve ref: {bad!r}"}

    if files:
        scoped = [p for p in files if p.endswith(".md")]
        skipped = [p for p in files if p not in scoped]
    else:
        discovered = changed_files(root, base, head)
        scoped = [p for p in discovered if in_docs_scope(p)]
        skipped = [p for p in discovered if not in_docs_scope(p)]

    findings: list[Finding] = []
    for path in sorted(set(scoped)):
        hunks = parse_hunks(file_diff(root, base_sha, head_sha, path))
        findings.extend(classify_file(path, hunks, min_run))

    allowed, wildcard = parse_allow_trailers(range_messages(root, base_sha, head_sha))
    apply_waivers(findings, allowed, wildcard)

    fails = [f for f in findings if f.severity == "FAIL"]
    by_reason: dict[str, int] = {}
    for f in findings:
        by_reason[f.reason] = by_reason.get(f.reason, 0) + 1

    report = {
        "base": base_sha,
        "head": head_sha,
        "min_run": min_run,
        "stats": {
            "files_screened": len(scoped),
            "skipped_out_of_scope": sorted(set(skipped)),
            "findings_total": len(findings),
            "fail_count": len(fails),
            "by_reason": by_reason,
            "waived_files": sorted(allowed),
            "wildcard_waiver": wildcard,
        },
        "findings": [{"path": f.path, "reason": f.reason, "severity": f.severity, "detail": f.detail} for f in sorted(findings, key=lambda x: (x.path, x.reason))],
    }
    return (1 if fails else 0), report


def _print_human(report: dict) -> None:
    st = report["stats"]
    print("=== sequence-safety: docs deletion-magnitude screen ===")
    print(f"base={report['base'][:12]} head={report['head'][:12]} min_run={report['min_run']}")
    print(f"files_screened={st['files_screened']} findings={st['findings_total']} " f"fail={st['fail_count']} by_reason={st['by_reason']}")
    if st["waived_files"] or st["wildcard_waiver"]:
        waived = "*" if st["wildcard_waiver"] else ", ".join(st["waived_files"])
        print(f"waived (Allow-Docs-Rewrite): {waived}")
    print()
    for f in report["findings"]:
        print(f"    [{f['severity']}/{f['reason']}] {f['path']}  {f['detail']}")
    if st["fail_count"]:
        print(f"\nFAIL: {st['fail_count']} unwaived docs-deletion finding(s). " "Declare intentional rewrites with a `Allow-Docs-Rewrite: <path>` commit trailer.")
    else:
        print("\nOK: no unwaived docs-deletion findings.")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="docs_additions_check.py",
        description="Docs deletion-magnitude screen: FAIL on a deleted heading or a run of >= N consecutive deleted lines between BASE and HEAD.",
        epilog=("Default scope (auto-discovery): AGENTS.md (+ CLAUDE.md symlink), docs/**/*.md, notes/**/*.md. " "An explicit --files list bypasses the scope filter (any .md path). Exit 0=clean, 1=findings, " "2=usage. Escape hatch: a `Allow-Docs-Rewrite: <path>[, ...]` commit trailer in the BASE..HEAD " "range waives the enumerated files (`*` waives all)."),
    )
    ap.add_argument("--base", required=True, help="base ref (e.g. origin/main, <merge>^1, github.event.before)")
    ap.add_argument("--head", required=True, help="head ref (e.g. HEAD, <merge>, github.sha)")
    ap.add_argument("--files", nargs="*", default=None, help="explicit .md files to screen (bypasses the scope filter)")
    ap.add_argument("--repo-root", default=".", help="repository root the git commands run in (default: cwd)")
    ap.add_argument("--min-run", type=int, default=DEFAULT_MIN_RUN, help=f"consecutive-deletion FAIL threshold (default: {DEFAULT_MIN_RUN})")
    ap.add_argument(
        "--advisory",
        action="store_true",
        help="advisory mode: print findings but exit 0 even on an unwaived FAIL (the per-PR docs-rewrite label hatch, demoted to WARN-only; the Allow-Docs-Rewrite commit trailer stays the primary waiver). Exit 2 (invocation error) is never masked.",
    )
    ap.add_argument("--json", action="store_true", help="emit the machine-readable report to stdout")
    args = ap.parse_args(argv)

    if args.min_run < 1:
        print("ERROR: --min-run must be >= 1", file=sys.stderr)
        return 2

    code, report = run(args.repo_root, args.base, args.head, args.files, args.min_run)
    if code == 2:
        print(f"ERROR: {report.get('error', 'invocation error')}", file=sys.stderr)
        return 2
    report["advisory"] = args.advisory
    if args.json:
        print(json.dumps(report, indent=1, sort_keys=True))
    else:
        _print_human(report)
        if args.advisory and code == 1:
            print("\nADVISORY (--advisory): the FAIL finding(s) above are downgraded to WARN-only for this run; exit 0. The auditable `Allow-Docs-Rewrite: <path>` commit trailer remains the primary waiver.")
    return 0 if (args.advisory and code == 1) else code


if __name__ == "__main__":
    sys.exit(main())
