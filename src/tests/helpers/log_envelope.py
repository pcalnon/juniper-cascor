"""The log-record envelope and named-marker contract, as code (cascor#573 roadmap step P0.4).

Project:     juniper-cascor
Sub-Project: logging
Author:      Paul Calnon
License:     MIT License

WHY THIS FILE EXISTS
Every cascor log record is consumed by scripts in ANOTHER repository (juniper-ml), which parse
the envelope -- the ``+`` sentinel, the bracket prefix, the ``(timestamp)`` and its precision,
the ``[LEVEL]`` -- and anchor on message text. None of that breakage is visible to cascor's own
CI: a change that renames a message, drops the sentinel or adds milliseconds to a timestamp
leaves every cascor test green while juniper-ml's analysis tooling silently stops matching
(RECON N-4 in juniper-ml ``notes/JUNIPER_2026-09-02_JUNIPER-CASCOR_LOGGING-CURRENT-STATE-RECONCILIATION.md``).

This module is the single definition of that contract, so the test that enforces it and any
later phase that needs it (P2 moves frame capture, P3 re-plumbs the sinks) share one checker.

THE SHAPES, as observed in real captures (not as documented -- the documentation was wrong about
which writer carries milliseconds; see ``fixtures/log_envelope/README.md``)::

    A_FILE       +[f.py: func:LINE] (YYYY-MM-DD HH:MM:SS) [LEVEL] msg      Path A, file sink
    A_CONSOLE    +[f.py: LINE] (YYYY-MM-DD HH:MM:SS) [LEVEL] msg           Path A, stdout sink
    STD_FILE_S   [f.py: func:LINE] (YYYY-MM-DD HH:MM:SS) [LEVEL] msg       Path B (dictConfig FileHandler)
    STD_FILE_MS  [f.py: func:LINE] (YYYY-MM-DD HH:MM:SS,mmm) [LEVEL] msg   Path C (api/observability RotatingFileHandler)
    STD_CONSOLE  [f.py:LINE] (YY-MM-DD HH:MM:SS) [LEVEL] msg               Path B console handler (ERROR+, stdout)

Path A (the custom classmethod ``Logger``) is ~98 % of all records and the only writer that
carries the ``+`` sentinel; juniper-ml uses ``line.startswith("+")`` to split worker records from
parent ones. Its timestamp is SECOND resolution, and three juniper-ml parsers accept only that.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

_TS_SECONDS = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
_LEVEL = r"[A-Z]+"
_FILE = r"[^\]:\s][^\]:]*"
_FUNC = r"[^\]:\s][^\]:]*"

#: Full-line, anchored. Order matters only for readability; the shapes are mutually exclusive.
SHAPES: dict[str, re.Pattern[str]] = {
    "A_FILE": re.compile(rf"^\+\[(?P<file>{_FILE}): (?P<func>{_FUNC}):(?P<line>\d+)\] \((?P<ts>{_TS_SECONDS})\) \[(?P<level>{_LEVEL})\] (?P<msg>.*)$"),
    "A_CONSOLE": re.compile(rf"^\+\[(?P<file>{_FILE}): (?P<line>\d+)\] \((?P<ts>{_TS_SECONDS})\) \[(?P<level>{_LEVEL})\] (?P<msg>.*)$"),
    "STD_FILE_S": re.compile(rf"^\[(?P<file>{_FILE}): (?P<func>{_FUNC}):(?P<line>\d+)\] \((?P<ts>{_TS_SECONDS})\) \[(?P<level>{_LEVEL})\] (?P<msg>.*)$"),
    "STD_FILE_MS": re.compile(rf"^\[(?P<file>{_FILE}): (?P<func>{_FUNC}):(?P<line>\d+)\] \((?P<ts>{_TS_SECONDS},\d{{3}})\) \[(?P<level>{_LEVEL})\] (?P<msg>.*)$"),
    # conf/logging_config.yaml ``formatter_console``: no space after the colon, no function name,
    # and a TWO-digit year (``%y``). Path B's ERROR-and-above records reach stdout in this shape.
    "STD_CONSOLE": re.compile(rf"^\[(?P<file>{_FILE}):(?P<line>\d+)\] \((?P<ts>\d{{2}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}})\) \[(?P<level>{_LEVEL})\] (?P<msg>.*)$"),
}
STD_CONSOLE = SHAPES["STD_CONSOLE"]

#: The level names Path A renders. ``[LEVEL]`` is the NAME, not the number.
PATH_A_LEVEL_NAMES = ("TRACE", "VERBOSE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "FATAL")

#: Logger-method name -> level number, for the marker level check. ``exception`` logs at ERROR.
METHOD_LEVELS = {"trace": 1, "verbose": 5, "debug": 10, "info": 20, "warning": 30, "warn": 30, "error": 40, "exception": 40, "critical": 50, "fatal": 60}


def classify(line: str) -> tuple[str | None, re.Match[str] | None]:
    """Return ``(shape, match)`` for a record line, or ``(None, None)`` when it is not a record."""
    for name, pattern in SHAPES.items():
        match = pattern.match(line)
        if match:
            return name, match
    return None, None


def render_path_a(sink: str, *, file: str, func: str, line: int, ts: str, level: str, msg: str) -> str:
    """The exact bytes Path A writes for one record (without the trailing newline)."""
    if sink == "file":
        return f"+[{file}: {func}:{line}] ({ts}) [{level}] {msg}"
    if sink == "console":
        return f"+[{file}: {line}] ({ts}) [{level}] {msg}"
    raise ValueError(f"unknown sink {sink!r}")


# ---------------------------------------------------------------------------------------------------
# Named markers: message text juniper-ml anchors on, resolved statically against the source tree
# ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkerSite:
    """One string literal that carries every fragment of a marker, in order."""

    file: str
    line: int
    method: str  # the enclosing logger method, "print", or "UNKNOWN"


def _skeletons(source: str) -> list[tuple[int, str, str]]:
    """(line, skeleton, enclosing method) for every string literal in *source*.

    A skeleton is the literal's text with each f-string placeholder replaced by ``\\x00``, so a
    marker's fragments either side of a placeholder still match in order. ``%``-format strings
    need no special handling: their placeholders are literal text (``%s``), which is a gap too.
    """
    tree = ast.parse(source)
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    out: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            skeleton = "".join(v.value if isinstance(v, ast.Constant) else "\x00" for v in node.values)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and not isinstance(parents.get(id(node)), ast.JoinedStr):
            skeleton = node.value
        else:
            continue
        method = "UNKNOWN"
        cur = parents.get(id(node))
        for _ in range(4):
            if cur is None:
                break
            if isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute) and cur.func.attr in METHOD_LEVELS:
                method = cur.func.attr
                break
            if isinstance(cur, ast.Call) and isinstance(cur.func, ast.Name) and cur.func.id == "print":
                method = "print"
                break
            cur = parents.get(id(cur))
        out.append((node.lineno, skeleton, method))
    return out


def _in_order(skeleton: str, fragments: list[str]) -> bool:
    pos = 0
    for fragment in fragments:
        at = skeleton.find(fragment, pos)
        if at < 0:
            return False
        pos = at + len(fragment)
    return True


def find_marker_sites(src_root: Path, relpath: str, fragments: list[str]) -> list[MarkerSite]:
    """Every literal in ``src_root/relpath`` that carries all *fragments*, in order."""
    path = src_root / relpath
    source = path.read_text(encoding="utf-8")
    return [MarkerSite(relpath, line, method) for line, skel, method in _skeletons(source) if _in_order(skel, fragments)]
