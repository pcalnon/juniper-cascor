"""
Pin ``util/check_image_no_secrets.py`` and its wiring into ``publish-image.yml``.

The script is the check that does not depend on someone getting a ``.dockerignore`` pattern
right. ``filepath.Match`` is root-anchored, so ``cascor_snapshots/`` never matches
``src/cascor_snapshots/``, and a ``COPY src/`` directory allowlist ships whatever is beneath
it (juniper-cascor#661). Walking only ``/app`` is a vacuous pass on images whose code lives
in site-packages, and a scan that finds no root or walks zero files must not report success.

These tests need no Docker. Roots are redirected at ``/app`` and at the purelib path, and the
workflow tests pin that both the PR arm and the publish arm run the script without swallowing
its exit status. The real execution happens in CI, inside the image just built.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

# CI's unit lane runs ``pytest -m "unit and not slow"``. A test file with no ``unit`` marker
# is collected and then deselected, and the job still reports success.
pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
WORKFLOW = REPO / ".github" / "workflows" / "publish-image.yml"
SCRIPT = REPO / "util" / "check_image_no_secrets.py"
SCRIPT_REL = "util/check_image_no_secrets.py"
MISSING_APP = Path("/no-such-app-root-for-check-image-no-secrets")

# One name per credential glob, plus the allow-list suffixes that must beat those globs.
BAD_NAMES = (
    ".env",
    ".env.local",
    ".env.",
    "server.key",
    "server.pem",
    "store.p12",
    "store.pfx",
    "store.jks",
    "app.keystore",
    "id_rsa",
    "id_rsa.pub",
    "id_ecdsa",
    "id_ed25519",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials",
    "credentials.json",
    "vault.kdbx",
)
ALLOWED_NAMES = (
    ".env.example",
    ".env.sample",
    ".env.template",
    ".env.dist",
    "server.pem.example",
    "server.key.sample",
    "credentials.template",
    "credentials.dist",
)
INNOCENT_NAMES = (
    "cascade_correlation.py",
    "README.md",
    ".ENV",
    "server.PEM",
    "my.credentials",
    "notes.txt",
    "secrets",
)
BAD_DIRS = ("secrets", ".git", ".ssh", ".aws", ".gnupg", "private")


def _load():
    spec = importlib.util.spec_from_file_location("check_image_no_secrets", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bind(monkeypatch, module, app: Path | None, site: Path | None) -> None:
    """Point the script's hard-coded ``/app`` and purelib lookups at temp directories."""
    real = Path

    def fake(value, *args, **kwargs):
        if value == "/app":
            return app if app is not None else MISSING_APP
        return real(value, *args, **kwargs)

    monkeypatch.setattr(module, "Path", fake)
    if site is None:
        monkeypatch.setattr(module.sysconfig, "get_paths", lambda: {})
    else:
        monkeypatch.setattr(module.sysconfig, "get_paths", lambda: {"purelib": str(site)})


def _run(monkeypatch, capsys, app: Path | None, site: Path | None) -> tuple[int, str]:
    module = _load()
    _bind(monkeypatch, module, app, site)
    code = module.main()
    return code, capsys.readouterr().out


def _finding_lines(out: str) -> list[str]:
    lines: list[str] = []
    capture = False
    for line in out.splitlines():
        if "credential-shaped path" in line:
            capture = True
            continue
        if capture and line.startswith("    "):
            lines.append(line[4:])
    return lines


def _workflow() -> dict:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML (YAML 1.1) reads the bare `on:` key as boolean True.
    data["on"] = data.pop(True, data.get("on"))
    return data


def _step(job: str, name_prefix: str) -> dict:
    matches = [s for s in _workflow()["jobs"][job]["steps"] if str(s.get("name", "")).startswith(name_prefix)]
    assert len(matches) == 1, f"expected exactly one step in job {job!r} named {name_prefix!r}..., found {len(matches)}"
    return matches[0]


class TestIsBadFile:
    @pytest.mark.parametrize("name", BAD_NAMES)
    def test_credential_shaped_names_match(self, name):
        assert _load().is_bad_file(name) is True

    @pytest.mark.parametrize("name", ALLOWED_NAMES)
    def test_template_suffix_beats_the_glob(self, name):
        """``.env.example`` matches ``.env.*``; the suffix is checked first, or the allow-list is a comment."""
        assert _load().is_bad_file(name) is False

    @pytest.mark.parametrize("name", INNOCENT_NAMES)
    def test_ordinary_and_case_mismatched_names_do_not_match(self, name):
        """Matching is case-sensitive (``fnmatchcase``). A file named ``secrets`` is not the directory."""
        assert _load().is_bad_file(name) is False

    def test_a_later_suffix_does_not_reopen_an_allow_listed_stem(self):
        """.env.example is allowed; .env.example.bak still matches ``.env.*``."""
        assert _load().is_bad_file(".env.example.bak") is True


class TestScanRoots:
    def test_app_plus_juniper_and_the_flat_service_packages_only(self, monkeypatch, tmp_path):
        app = tmp_path / "app"
        site = tmp_path / "site"
        app.mkdir()
        site.mkdir()
        for name in ("candidate_unit", "cascade_correlation", "juniper_cascor", "juniper_cascor-0.11.0.dist-info", "numpy", "JuniperUpper", "cascade"):
            (site / name).mkdir()
        (site / "juniper_note.txt").write_text("not a package tree")

        module = _load()
        _bind(monkeypatch, module, app, site)
        roots = module.scan_roots()

        assert roots[0] == app
        site_names = [p.name for p in roots[1:]]
        assert site_names == sorted(site_names)
        assert "juniper_cascor" in site_names
        assert "candidate_unit" in site_names
        assert "cascade_correlation" in site_names
        assert "numpy" not in site_names
        assert "JuniperUpper" not in site_names
        assert "cascade" not in site_names
        assert "juniper_note.txt" not in site_names

    def test_packages_are_roots_when_app_is_absent(self, monkeypatch, tmp_path):
        site = tmp_path / "site"
        (site / "juniper_cascor").mkdir(parents=True)
        module = _load()
        _bind(monkeypatch, module, None, site)
        roots = module.scan_roots()
        assert [p.name for p in roots] == ["juniper_cascor"]

    def test_app_alone_when_purelib_is_missing_or_not_a_directory(self, monkeypatch, tmp_path):
        app = tmp_path / "app"
        app.mkdir()
        module = _load()
        _bind(monkeypatch, module, app, None)
        assert module.scan_roots() == [app]

        not_a_dir = tmp_path / "purelib-file"
        not_a_dir.write_text("nope")
        _bind(monkeypatch, module, app, not_a_dir)
        assert module.scan_roots() == [app]

    def test_no_roots_when_neither_tree_exists(self, monkeypatch, tmp_path):
        module = _load()
        _bind(monkeypatch, module, None, tmp_path / "missing-site")
        assert module.scan_roots() == []


class TestMain:
    def test_no_root_is_exit_2_and_says_nothing_was_inspected(self, monkeypatch, capsys, tmp_path):
        code, out = _run(monkeypatch, capsys, None, tmp_path / "missing-site")
        assert code == 2
        assert "no scan root found" in out
        assert "inspected NOTHING" in out
        assert "no credential-shaped file" not in out

    def test_a_root_that_walks_zero_files_is_exit_2(self, monkeypatch, capsys, tmp_path):
        app = tmp_path / "app"
        app.mkdir()
        (app / "empty-child").mkdir()
        code, out = _run(monkeypatch, capsys, app, None)
        assert code == 2
        assert "ZERO files" in out
        assert "proved nothing" in out
        assert code != 0

    def test_an_empty_credential_directory_and_nothing_else_is_not_a_pass(self, monkeypatch, capsys, tmp_path):
        """Zero files is checked before findings, so this is exit 2 today. Either failure is closed; a pass is not."""
        app = tmp_path / "app"
        (app / "secrets").mkdir(parents=True)
        code, out = _run(monkeypatch, capsys, app, None)
        assert code != 0
        assert "no credential-shaped file" not in out

    def test_a_clean_tree_passes_and_counts_every_file(self, monkeypatch, capsys, tmp_path):
        app = tmp_path / "app"
        (app / "sub").mkdir(parents=True)
        (app / "cascade_correlation.py").write_text("ok")
        (app / "sub" / "net.py").write_text("ok")
        code, out = _run(monkeypatch, capsys, app, None)
        assert code == 0
        assert "scanned 2 files across 1 root(s):" in out
        assert str(app) in out
        assert "no credential-shaped file" in out

    def test_templates_next_to_code_pass(self, monkeypatch, capsys, tmp_path):
        app = tmp_path / "app"
        app.mkdir()
        for name in ALLOWED_NAMES:
            (app / name).write_text("template")
        (app / "code.py").write_text("ok")
        code, out = _run(monkeypatch, capsys, app, None)
        assert code == 0, out
        assert "credential-shaped path" not in out

    @pytest.mark.parametrize("name", (".env", "server.pem", "id_rsa", "credentials", ".netrc"))
    def test_a_planted_credential_file_is_exit_1_and_its_contents_are_not_printed(self, monkeypatch, capsys, tmp_path, name):
        app = tmp_path / "app"
        app.mkdir()
        (app / "ok.py").write_text("ok")
        (app / name).write_text("SECRET-MATERIAL")
        code, out = _run(monkeypatch, capsys, app, None)
        assert code == 1
        assert str(app / name) in _finding_lines(out)
        assert "SECRET-MATERIAL" not in out
        assert "ZERO files" not in out

    @pytest.mark.parametrize("dirname", BAD_DIRS)
    def test_a_credential_directory_is_reported_and_so_is_a_file_inside_it(self, monkeypatch, capsys, tmp_path, dirname):
        app = tmp_path / "app"
        (app / dirname).mkdir(parents=True)
        (app / "ok.py").write_text("ok")
        (app / dirname / ".env").write_text("SECRET-MATERIAL")
        code, out = _run(monkeypatch, capsys, app, None)
        assert code == 1
        findings = _finding_lines(out)
        assert f"{app / dirname}/  (directory)" in findings
        assert str(app / dirname / ".env") in findings
        assert "SECRET-MATERIAL" not in out
        assert "::error::2 credential-shaped path" in out

    def test_pruned_caches_hide_nested_credentials_and_are_not_counted(self, monkeypatch, capsys, tmp_path):
        app = tmp_path / "app"
        app.mkdir()
        (app / "ok.py").write_text("ok")
        for cache in ("__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "node_modules"):
            cache_dir = app / cache
            (cache_dir / "secrets").mkdir(parents=True)
            (cache_dir / ".env").write_text("SECRET-MATERIAL")
            (cache_dir / "secrets" / "id_rsa").write_text("SECRET-MATERIAL")
        code, out = _run(monkeypatch, capsys, app, None)
        assert code == 0, out
        assert "scanned 1 files across 1 root(s):" in out
        assert "SECRET-MATERIAL" not in out
        assert ".env" not in out
        assert "id_rsa" not in out

    def test_a_secret_in_a_juniper_package_fails_when_app_is_clean(self, monkeypatch, capsys, tmp_path):
        """The vacuous-success class: ``/app`` holds only runtime dirs, and the code is in site-packages."""
        app = tmp_path / "app"
        site = tmp_path / "site"
        app.mkdir()
        (app / "data").mkdir()
        (site / "juniper_cascor").mkdir(parents=True)
        (site / "juniper_cascor" / ".env").write_text("SECRET-MATERIAL")
        (site / "numpy").mkdir()
        (site / "numpy" / ".env").write_text("FOREIGN-SECRET")
        code, out = _run(monkeypatch, capsys, app, site)
        assert code == 1
        assert str(site / "juniper_cascor" / ".env") in out
        assert "FOREIGN-SECRET" not in out
        assert str(site / "numpy") not in out
        assert "scanned 1 files across 2 root(s):" in out

    def test_a_flat_service_package_is_scanned_without_a_juniper_prefix(self, monkeypatch, capsys, tmp_path):
        site = tmp_path / "site"
        (site / "cascade_correlation").mkdir(parents=True)
        (site / "cascade_correlation" / "id_rsa").write_text("SECRET-MATERIAL")
        (site / "candidate_unit").mkdir()
        (site / "candidate_unit" / "unit.py").write_text("ok")
        code, out = _run(monkeypatch, capsys, None, site)
        assert code == 1
        assert str(site / "cascade_correlation" / "id_rsa") in out
        assert "scanned 2 files across 2 root(s):" in out
        assert str(site / "candidate_unit") in out

    def test_findings_are_printed_in_sorted_order_regardless_of_walk_order(self, monkeypatch, capsys, tmp_path):
        app = tmp_path / "app"
        app.mkdir()
        module = _load()
        _bind(monkeypatch, module, app, None)

        def reversed_walk(root):
            yield str(root), [], ["z.pem", "ok.py", "a.key"]

        monkeypatch.setattr(module.os, "walk", reversed_walk)
        code = module.main()
        out = capsys.readouterr().out
        assert code == 1
        assert _finding_lines(out) == [str(app / "a.key"), str(app / "z.pem")]
        assert "scanned 3 files across 1 root(s):" in out

    def test_a_symlinked_credential_directory_is_named_and_not_followed(self, monkeypatch, capsys, tmp_path):
        app = tmp_path / "app"
        outside = tmp_path / "outside"
        app.mkdir()
        outside.mkdir()
        (app / "ok.py").write_text("ok")
        (outside / ".env").write_text("SECRET-MATERIAL")
        (app / "secrets").symlink_to(outside, target_is_directory=True)
        code, out = _run(monkeypatch, capsys, app, None)
        assert code == 1
        assert f"{app / 'secrets'}/  (directory)" in _finding_lines(out)
        assert str(outside / ".env") not in out
        assert "SECRET-MATERIAL" not in out


class TestPublishWorkflowRunsTheSecretCheck:
    def test_paths_filter_covers_the_script(self):
        assert SCRIPT_REL in _workflow()["on"]["pull_request"]["paths"]

    @pytest.mark.parametrize(
        ("step_name", "image"),
        [
            ("Smoke test (build-only runs)", '"${img}"'),
            ("Verify pushed image is CPU-only (publish runs)", '"${ref}"'),
        ],
    )
    def test_both_arms_run_the_script_and_keep_its_exit_status(self, step_name, image):
        run = _step("build", step_name)["run"]
        expected = f"docker run --rm -i {image} python - < {SCRIPT_REL}"
        assert expected in run
        assert "--entrypoint" not in run, "the cascor image is CMD, not ENTRYPOINT; an override changes which interpreter runs"
        secret_lines = [line.strip() for line in run.splitlines() if SCRIPT_REL in line]
        assert secret_lines == [expected]
        assert all("||" not in line for line in secret_lines)
