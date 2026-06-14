"""Drift-guard for the juniper-cascor-model candidate-core (CW-05 plan §5/§7; placement D3).

``juniper-cascor-model`` is extracted verbatim from this repo's ``src`` and consumed by
``juniper-cascor-worker``, while ``juniper-cascor`` itself keeps its inline ``src`` copies
until it adopts the package (plan Wave 2, deferred). The only correctness risk of that
deferral is **drift** between the package copy and the ``src`` copy. This test fails if any
extracted candidate-core module diverges from its ``src`` counterpart, so divergence is a
conscious choice (re-extract, or extend the allowlist).

The logger implementation intentionally diverges for the deployment-agnostic logging fix
(CW-05 gap #3) and is allowlisted until it is backported to ``src`` in Wave 2. The
``cascor_constants/constants.py`` log-dir override is normalized before comparison so
unrelated drift in that file is still caught.

Both copies live in the same repo (``juniper-cascor/juniper-cascor-model`` and
``juniper-cascor/src``), so no sibling-repo lookup is needed; the test skips only when
``src`` is absent (e.g. the package built/extracted standalone, or isolated CI).
"""

from __future__ import annotations

import unittest
from pathlib import Path

# The candidate-core module trees extracted into juniper-cascor-model/ (top-level names,
# CW-05 plan §3.1 option (i)).
_EXTRACTED_DIRS = ("candidate_unit", "utils", "log_config", "cascor_constants")

# Files intentionally modified in the package vs src (CW-05 gap #3: env-overridable +
# best-effort logging). NOT byte-checked; to be backported to src in Wave 2.
_INTENTIONAL_DIVERGENCE = frozenset({Path("log_config/logger/logger.py")})

# Files whose only intended divergence is normalized away before comparison, so any other
# drift in them is still caught.
_NORMALIZED_DIVERGENCE = frozenset({Path("cascor_constants/constants.py")})

_PACKAGE_LOG_DIR_OVERRIDE = """# juniper-cascor-model: honor JUNIPER_CASCOR_LOG_DIR so deployments (e.g. the distributed
# worker, where this package lives in site-packages and the source-relative logs/ dir is
# not writable) can redirect file logging without code changes. Unset -> source default.
_PROJECT_LOG_DIR_DEFAULT = pathlib.Path(os.environ.get("JUNIPER_CASCOR_LOG_DIR") or pathlib.Path(_PROJECT_DIR, _PROJECT_LOG_DIR_NAME_DEFAULT))"""
_CASCOR_LOG_DIR_DEFAULT = "_PROJECT_LOG_DIR_DEFAULT = pathlib.Path(_PROJECT_DIR, _PROJECT_LOG_DIR_NAME_DEFAULT)"


def _drift_bytes(path: Path, rel: Path, *, is_package: bool) -> bytes:
    text = path.read_text()
    if is_package and rel in _NORMALIZED_DIVERGENCE:
        text = text.replace(_PACKAGE_LOG_DIR_OVERRIDE, _CASCOR_LOG_DIR_DEFAULT)
    return text.encode()


class JuniperCascorModelDriftTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # tests/ -> juniper-cascor-model/ (package root) -> juniper-cascor/ (repo root)
        cls.package_root = Path(__file__).resolve().parent.parent
        cls.cascor_src = cls.package_root.parent / "src"

    def test_package_present(self):
        self.assertTrue(
            (self.package_root / "candidate_unit").is_dir(),
            f"juniper-cascor-model candidate core not found at {self.package_root}",
        )

    def test_extracted_modules_match_cascor_src(self):
        if not self.cascor_src.is_dir():
            self.skipTest("juniper-cascor/src not on disk -- drift check skipped (standalone / isolated context)")

        missing: list[str] = []
        mismatches: list[str] = []
        checked = 0
        for d in _EXTRACTED_DIRS:
            for pkg_file in sorted((self.package_root / d).rglob("*.py")):
                rel = pkg_file.relative_to(self.package_root)
                src_file = self.cascor_src / rel
                if rel in _INTENTIONAL_DIVERGENCE:
                    if not src_file.is_file():
                        missing.append(f"{rel} (allowlisted, but missing upstream)")
                    continue
                if not src_file.is_file():
                    missing.append(str(rel))
                    continue
                checked += 1
                if _drift_bytes(pkg_file, rel, is_package=True) != _drift_bytes(src_file, rel, is_package=False):
                    mismatches.append(str(rel))

        problems: list[str] = []
        if missing:
            problems.append("missing upstream counterpart: " + ", ".join(missing))
        if mismatches:
            problems.append(
                "DRIFT -- these juniper-cascor-model files differ from juniper-cascor/src "
                "(re-extract them, or add to _INTENTIONAL_DIVERGENCE if the change is deliberate): "
                + ", ".join(mismatches)
            )
        self.assertEqual([], problems, "\n".join(problems))
        self.assertGreater(checked, 0, "no files were drift-checked -- extraction set may be empty")

    def test_intentional_divergences_actually_differ(self):
        """Sanity: allowlisted files SHOULD differ from src; else the allowlist is stale."""
        if not self.cascor_src.is_dir():
            self.skipTest("juniper-cascor/src not present")
        for rel in _INTENTIONAL_DIVERGENCE:
            pkg_file = self.package_root / rel
            src_file = self.cascor_src / rel
            if pkg_file.is_file() and src_file.is_file():
                self.assertNotEqual(
                    pkg_file.read_bytes(),
                    src_file.read_bytes(),
                    f"{rel} is allowlisted as intentionally-diverged but is byte-identical to "
                    "cascor src -- remove it from _INTENTIONAL_DIVERGENCE.",
                )


if __name__ == "__main__":
    unittest.main()
