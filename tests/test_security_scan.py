import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.security_scan import PACKAGE_DIR, scan_file


class SecurityScanTests(unittest.TestCase):
    def test_package_has_no_findings(self):
        findings: list[dict[str, str]] = []
        files = sorted(PACKAGE_DIR.rglob("*.py"))
        self.assertTrue(files, "package sources must exist")
        for path in files:
            findings.extend(scan_file(path))
        self.assertEqual(findings, [], findings)

    def _scan_snippet(self, snippet: str) -> list[dict[str, str]]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.py"
            path.write_text(snippet, encoding="utf-8")
            return scan_file(path)

    def test_scanner_detects_network_import(self):
        findings = self._scan_snippet("import socket\n")
        self.assertTrue(any("forbidden import" in item["issue"] for item in findings))

    def test_scanner_detects_subprocess_and_shell_call(self):
        findings = self._scan_snippet("import subprocess\nsubprocess.run(['ls'], shell=True)\n")
        self.assertTrue(any("subprocess" in item["issue"] for item in findings))

    def test_scanner_detects_dynamic_execution(self):
        findings = self._scan_snippet("def run(payload):\n    return eval(payload)\n")
        self.assertTrue(any("eval" in item["issue"] for item in findings))

    def test_scanner_detects_unguarded_rmtree(self):
        findings = self._scan_snippet("import shutil\n\ndef purge(target):\n    shutil.rmtree(target)\n")
        self.assertTrue(any("rmtree" in item["issue"] for item in findings))

    def test_scanner_accepts_guarded_rmtree(self):
        snippet = (
            "import shutil\n"
            "from pathlib import Path\n"
            "\n"
            "def purge(root: Path, target: Path):\n"
            "    resolved = target.resolve()\n"
            "    if resolved.is_relative_to(root):\n"
            "        shutil.rmtree(resolved, ignore_errors=True)\n"
        )
        self.assertEqual(self._scan_snippet(snippet), [])

    def test_scanner_detects_hardcoded_secret(self):
        findings = self._scan_snippet("API_KEY = 'abcdef1234567890abcdef'\n")
        self.assertTrue(any("secret" in item["issue"] for item in findings))


if __name__ == "__main__":
    unittest.main()
