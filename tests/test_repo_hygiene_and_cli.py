"""Repository hygiene, LF line endings, code-audit, prose/dupes, Stage 0 safety, and CLI smoke tests."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
import urllib.parse

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_BIN = REPO_ROOT / "bin" / "mlos"

SKIP_DIRS: frozenset[str] = frozenset({".git", "__pycache__", ".pytest_cache", ".venv", "venv", ".mde", "build", "dist"})


def _iter_repo_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(REPO_ROOT.rglob("*")):
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts):
            continue
        if path.is_file():
            files.append(path)
    return files


class TestRepoHygieneAndCli(unittest.TestCase):
    def test_zero_crlf_and_zero_forbidden_strings_across_repo(self) -> None:
        # Construct forbidden needles dynamically so this test file itself stays 100% clean.
        forbidden = (
            "/" + "Users" + "/",
            "/" + "home" + "/",
            "ket" + "huda",
            "haf" + "iz",
            "ci@" + "example.invalid",
            "192" + ".168.",
            "100" + ".",
            "fd7a" + ":",
            "Co-authored-" + "by:",
        )
        repo_files = _iter_repo_files()
        self.assertGreater(len(repo_files), 20)

        for path in repo_files:
            raw = path.read_bytes()
            self.assertNotIn(b"\r", raw, msg=f"CR byte found in {path.relative_to(REPO_ROOT)} (LF required)")
            text = raw.decode("utf-8")
            for needle in forbidden:
                self.assertNotIn(
                    needle,
                    text,
                    msg=f"forbidden string {needle!r} found in {path.relative_to(REPO_ROOT)}",
                )

    def test_python_ast_zero_bare_assert_or_bare_except_and_json_unique_keys(self) -> None:
        for path in _iter_repo_files():
            if path.suffix == ".py" or path == CLI_BIN:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assert):
                        self.fail(f"bare assert at {path.relative_to(REPO_ROOT)}:{node.lineno} (stripped under -O)")
                    if isinstance(node, ast.ExceptHandler) and node.type is None:
                        self.fail(f"bare except at {path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif path.suffix == ".json":
                def check_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
                    seen: set[str] = set()
                    out: dict[str, object] = {}
                    for k, v in pairs:
                        if k in seen:
                            self.fail(f"duplicate JSON key {k!r} in {path.relative_to(REPO_ROOT)}")
                        seen.add(k)
                        out[k] = v
                    return out

                json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=check_pairs)

    def test_stage0_powershell_scripts_forbid_destructive_disk_cmdlets(self) -> None:
        ps_files = sorted((REPO_ROOT / "scripts").glob("*.ps1"))
        self.assertEqual(len(ps_files), 2)
        banned_disk_ops = (
            "Resize-Partition",
            "Format-Volume",
            "Clear-Disk",
            "Remove-Partition",
            "Initialize-Disk",
            "diskpart",
        )
        for ps_path in ps_files:
            txt = ps_path.read_text(encoding="utf-8")
            self.assertIn("Set-StrictMode -Version Latest", txt)
            for op in banned_disk_ops:
                self.assertNotIn(op, txt, msg=f"destructive cmdlet {op} found in {ps_path.name}")

    def test_markdown_docs_prose_and_duplicate_reference_gates(self) -> None:
        md_files = [p for p in _iter_repo_files() if p.suffix == ".md"]
        self.assertGreaterEqual(len(md_files), 3)

        gh_re = re.compile(r"github\.com/[^/\s]+/[^/\s]+/(?:pull|issues)/(\d+)", re.IGNORECASE)
        mention_re = re.compile(r"(?<![\w.@])@\w+")

        for md in md_files:
            raw = md.read_text(encoding="utf-8")
            in_fence = False
            for line_no, line in enumerate(raw.splitlines(), 1):
                if line.lstrip().startswith("```"):
                    in_fence = not in_fence
                    continue
                if in_fence:
                    continue
                prose_line = re.sub(r"`[^`]*`", "", line)
                self.assertNotIn("\u2014", prose_line, msg=f"em dash in {md.name}:{line_no}")
                self.assertIsNone(
                    mention_re.search(prose_line),
                    msg=f"standalone @-mention in {md.name}:{line_no}",
                )
                if not prose_line.strip().startswith("|"):
                    self.assertLessEqual(
                        len(prose_line),
                        420,
                        msg=f"paragraph > 420 chars in {md.name}:{line_no}",
                    )

            # Duplicate reference gate per markdown document.
            counts: dict[str, int] = {}
            url_spans: list[tuple[int, int]] = []
            for m in re.finditer(r"https?://\S+", raw, re.IGNORECASE):
                url = m.group(0).rstrip(".,;:!?'\"`)]}>")
                url_spans.append((m.start(), m.end()))
                gh = gh_re.search(url)
                if gh:
                    key = "gh#" + gh.group(1)
                else:
                    parts = urllib.parse.urlparse(url)
                    key = (parts.scheme.lower() + "://" + parts.netloc.lower() + parts.path).lower()
                counts[key] = counts.get(key, 0) + 1
            for m in re.finditer(r"(?:PR|issue)?\s*#(\d+)", raw, re.IGNORECASE):
                if any(s <= m.start() < e for s, e in url_spans):
                    continue
                key = "gh#" + m.group(1)
                counts[key] = counts.get(key, 0) + 1
            dupes = {k: v for k, v in counts.items() if v > 1}
            self.assertEqual(dupes, {}, msg=f"duplicate external references in {md.name}: {dupes}")

    def test_cli_entrypoints_end_to_end(self) -> None:
        ux_prof = str(REPO_ROOT / "profiles" / "ux582hs-rtx3080-8gb.toml")
        ux_bp = str(REPO_ROOT / "blueprints" / "ux582hs-rtx3080-8gb")

        r_adapt = subprocess.run(
            [sys.executable, str(CLI_BIN), "adapt", "--check", "--json", "--profile", ux_prof],
            capture_output=True,
            text=True,
            env={**os.environ},
            check=False,
        )
        self.assertEqual(r_adapt.returncode, 0, msg=r_adapt.stderr)
        self.assertTrue(json.loads(r_adapt.stdout)["passed"])

        r_comp = subprocess.run(
            [sys.executable, str(CLI_BIN), "compile", "--profile", ux_prof, "--verify-dir", ux_bp],
            capture_output=True,
            text=True,
            env={**os.environ},
            check=False,
        )
        self.assertEqual(r_comp.returncode, 0, msg=r_comp.stderr)

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            lock_f = td_path / "cli.lock"
            receipt_f = td_path / "cli-receipt.json"
            r_lease = subprocess.run(
                [
                    sys.executable,
                    str(CLI_BIN),
                    "gpu-lease",
                    "--mode",
                    "eval",
                    "--lock-file",
                    str(lock_f),
                    "--",
                    sys.executable,
                    str(CLI_BIN),
                    "stamp-run",
                    "--profile",
                    ux_prof,
                    "--zone",
                    "A",
                    "--output",
                    str(receipt_f),
                    "--",
                    sys.executable,
                    "-c",
                    "import os, sys; sys.exit(0 if os.environ.get('PYTHONHASHSEED') == '0' else 1)",
                ],
                capture_output=True,
                text=True,
                env={**os.environ},
                check=False,
            )
            self.assertEqual(r_lease.returncode, 0, msg=r_lease.stderr)
            self.assertTrue(receipt_f.is_file())
            rec = json.loads(receipt_f.read_text(encoding="utf-8"))
            self.assertEqual(rec["determinism_zone"], "A")
            self.assertEqual(rec["execution"]["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
