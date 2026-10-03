"""Static audit for the stage-1 reference system.

Checks that the implementation stays inside the agreed boundary:
no network access, no external processes, no dynamic code execution,
no secret material, no unbounded recursive deletion.

Run:  python -m tools.security_scan
Exit code 0 means no findings.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_ROOT / "reference_system"

FORBIDDEN_IMPORTS = {
    "socket",
    "ssl",
    "urllib",
    "urllib.request",
    "http",
    "http.client",
    "httpx",
    "requests",
    "ftplib",
    "smtplib",
    "telnetlib",
    "webbrowser",
    "subprocess",
    "pty",
    "multiprocessing",
    "ctypes",
    "pickle",
    "marshal",
    "shelve",
}

FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "os.system",
    "os.popen",
    "os.spawnl",
    "os.execv",
    "os.remove",
    "os.unlink",
    "shutil.rmtree",
}

# shutil.rmtree is allowed only when the target is proven to live inside the root.
RMTREE_ALLOWED_IF = "is_relative_to"

SECRET_PATTERNS = [
    re.compile(r"(?i)\b(api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


def _qualified_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def scan_file(path: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".")[0]
                if alias.name in FORBIDDEN_IMPORTS or root_module in FORBIDDEN_IMPORTS:
                    findings.append({"file": path.name, "line": str(node.lineno), "issue": f"forbidden import: {alias.name}"})
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root_module = module.split(".")[0]
            if module in FORBIDDEN_IMPORTS or root_module in FORBIDDEN_IMPORTS:
                findings.append({"file": path.name, "line": str(node.lineno), "issue": f"forbidden import: {module}"})
        elif isinstance(node, ast.Call):
            name = _qualified_name(node.func)
            if name in FORBIDDEN_CALLS:
                if name == "shutil.rmtree":
                    window = source.splitlines()[max(0, node.lineno - 4):node.lineno]
                    if any(RMTREE_ALLOWED_IF in line for line in window):
                        continue
                    findings.append({
                        "file": path.name,
                        "line": str(node.lineno),
                        "issue": "shutil.rmtree without an is_relative_to() guard",
                    })
                    continue
                findings.append({"file": path.name, "line": str(node.lineno), "issue": f"forbidden call: {name}"})

    for index, line in enumerate(source.splitlines(), start=1):
        for pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append({"file": path.name, "line": str(index), "issue": "possible secret material"})
    return findings


def main() -> int:
    findings: list[dict[str, str]] = []
    files = sorted(PACKAGE_DIR.rglob("*.py"))
    for path in files:
        findings.extend(scan_file(path))

    print(f"scanned {len(files)} python files under {PACKAGE_DIR}")
    print(f"network imports allowed: none")
    if findings:
        for finding in findings:
            print(f"FINDING {finding['file']}:{finding['line']} {finding['issue']}")
        return 1
    print("no findings: no network, no subprocess, no dynamic execution, no secrets, rmtree guarded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
