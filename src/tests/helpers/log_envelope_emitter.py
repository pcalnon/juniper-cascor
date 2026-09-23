#!/usr/bin/env python
"""Emit reference records through ALL THREE real log writers, for the P0.4 envelope contract test.

Project:     juniper-cascor
Sub-Project: logging
Author:      Paul Calnon
License:     MIT License

WHY THIS IS A SCRIPT, RUN IN A CHILD PROCESS
The test session's ``conftest.py`` replaces ``Logger._log_at_level`` with a no-op for the WHOLE
session (the autouse, session-scoped ``_cache_logging_system`` fixture). An in-process emit would
therefore exercise nothing: the filter, the ``print`` and the ``open``/``write`` are all stubbed
out, and a test built on them passes whatever the envelope looks like. A fresh interpreter has no
conftest, so this is the real emit path end to end -- including two production conditions the
in-process path cannot reproduce at all:

* stdout is a PIPE here, exactly as it is a redirected file under juniper-ml's launchers, so
  ``print()`` is block-buffered rather than line-buffered; and
* ``JUNIPER_CASCOR_LOG_DIR`` is read at IMPORT time (``cascor_constants/constants.py``), so the
  file sink can only be redirected before the first cascor import -- i.e. in a new process.

THE THREE WRITERS, ALL INTO ONE FILE (as a real run does -- RECON N-9)

1. **Path A** -- the classmethod ``Logger``. Worker/candidate records; the only ``+`` writer.
2. **Path B** -- ``logging.getLogger("juniper")`` configured by ``conf/logging_config.yaml``
   through a real ``LogConfig``, built exactly as ``CascadeCorrelationNetwork._init_logging_system``
   builds it. **The run-verdict markers are Path B records**, not Path A: ``fit: Training
   completed.``, ``train_candidates: Executing candidate training with N processes``,
   ``Completed solving SpiralProblem instance`` all land in the file with no ``+``.
3. **Path C** -- ``api.observability.configure_logging``, the service tier's RotatingFileHandler
   (the only writer that rotates the shared file, and the only one with millisecond timestamps).

THE CONTRACT WITH THE TEST
Every ``emit_*`` function makes one single-line call per record. The test derives each record's
expected ``func:LINE`` by parsing THIS file's AST, so the resolved caller is checked against the
real call site, not a hard-coded number. That makes it a detector for roadmap P2.1 (moving frame
capture inside ``_log_at_level``): get the frame depth wrong and every Path A record names
``_log_at_level`` or an emit method instead of ``emit_path_a``.

Keep every call on ONE line, inside its ``emit_*`` function.

Usage (the test does this): ``python log_envelope_emitter.py <report.json>``
"""

import json
import logging
import logging.config
import logging.handlers
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from log_config.logger.logger import Logger  # noqa: E402  -- sys.path must be set first

#: The %-args record: ``_log_at_level`` interpolates only AFTER the level filter passes.
PERCENT_ARGS_TEMPLATE = "P0.4 reference record: %s-args %d"
PERCENT_ARGS_VALUES = ("percent", 2)


def emit_path_a() -> None:
    Logger.trace("P0.4 reference record: path A TRACE")
    Logger.verbose("P0.4 reference record: path A VERBOSE")
    Logger.debug("P0.4 reference record: path A DEBUG")
    Logger.info("P0.4 reference record: path A INFO")
    Logger.warning("P0.4 reference record: path A WARNING")
    Logger.error("P0.4 reference record: path A ERROR")
    Logger.critical("P0.4 reference record: path A CRITICAL")
    Logger.fatal("P0.4 reference record: path A FATAL")
    Logger.info(PERCENT_ARGS_TEMPLATE, *PERCENT_ARGS_VALUES)


def emit_path_b(log: logging.Logger) -> None:
    log.debug("P0.4 reference record: path B DEBUG")
    log.info("P0.4 reference record: path B INFO")
    log.warning("P0.4 reference record: path B WARNING")
    log.error("P0.4 reference record: path B ERROR")


def emit_path_c(log: logging.Logger) -> None:
    log.debug("P0.4 reference record: path C DEBUG")
    log.info("P0.4 reference record: path C INFO")


def _formatter(handler: logging.Handler) -> dict:
    fmt = handler.formatter
    return {"format": getattr(fmt, "_fmt", None), "datefmt": getattr(fmt, "datefmt", None)}


def _handler(handler: logging.Handler) -> dict:
    out = {"class": f"{type(handler).__module__}.{type(handler).__name__}", "level": logging.getLevelName(handler.level), **_formatter(handler)}
    if isinstance(handler, logging.FileHandler):
        out["filename"] = Path(handler.baseFilename).name
    if isinstance(handler, logging.handlers.RotatingFileHandler):
        out["maxBytes"] = handler.maxBytes
        out["backupCount"] = handler.backupCount
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
        out["stream"] = {sys.stdout: "stdout", sys.stderr: "stderr"}.get(handler.stream, "other")
    return out


def main(argv: list[str]) -> int:
    report_path = Path(argv[1])
    report: dict = {}

    # --- Path B's configuration FIRST, at the default level, the way
    # CascadeCorrelationNetwork._init_logging_system builds it. It must precede the TRACE switch
    # below: LogConfig.__init__ copies the custom ``verbose``/``trace`` closures onto the stdlib
    # ``juniper`` logger with ``__get__``, which binds the LOGGER as the closure's ``message``
    # argument; at VERBOSE/TRACE every such call then raises inside logging's formatter
    # (``--- Logging error ---`` on stderr, record lost). That is a real, separate defect -- it is
    # latent at the default INFO, and this harness must not depend on it either way.
    from log_config.log_config import LogConfig

    log_config = LogConfig(_LogConfig__log_config=logging.config)
    path_b = log_config.get_logger()

    # --- Path A. TRACE, so every level emits. P1.1 made set_level the single writer of the one
    # level both the guard and the emit filter read; before it, this call was a no-op for emission.
    Logger.set_level("TRACE")
    emit_path_a()
    report["path_a"] = {"logging_file": str(Logger._logging_file), "log_level": Logger.get_level()}

    emit_path_b(path_b)
    report["path_b"] = {"logger": path_b.name, "level": logging.getLevelName(path_b.level), "propagate": path_b.propagate, "handlers": [_handler(h) for h in path_b.handlers]}

    # --- Path C, the service tier's writer. It REMOVES every root handler first, then installs its
    # own; ``juniper`` (Path B) has propagate: False and its own handlers, so it is unaffected.
    from api.observability import configure_logging

    configure_logging("INFO", "text")
    path_c = logging.getLogger("p04.path_c")
    emit_path_c(path_c)
    root = logging.getLogger()
    report["path_c"] = {"root_level": logging.getLevelName(root.level), "handlers": [_handler(h) for h in root.handlers]}

    for handler in [*root.handlers, *path_b.handlers]:
        handler.flush()
    sys.stdout.flush()
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
