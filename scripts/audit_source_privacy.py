"""Reproducible local source/privacy audit for the prototype release.

The audit is deliberately narrow: it detects common committed-secret patterns
and a small set of dangerous Python constructs.  It is not a penetration test,
data-protection impact assessment, or substitute for human security review.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "training_candidate_20261102" / "source_privacy_audit.json"
SCAN_ROOTS = (ROOT / "src", ROOT / "configs", ROOT / "tests")
SECRET_PATTERNS = {
    "private_key_block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "generic_bearer": re.compile(r"(?i)authorization\s*[:=]\s*['\"]bearer\s+[A-Za-z0-9._-]{20,}"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files() -> list[Path]:
    allowed = {".py", ".json", ".html", ".js", ".css", ".toml", ".yaml", ".yml"}
    return sorted(path for root in SCAN_ROOTS for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed)


def ast_findings(path: Path, text: str) -> list[dict]:
    if path.suffix.lower() != ".py":
        return []
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return [{"kind": "syntax_error", "line": exc.lineno, "detail": str(exc)}]
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        # Only bare built-ins are dangerous here.  ``torch.nn.Module.eval()``
        # and similar attribute calls are normal inference-mode operations.
        if isinstance(node.func, ast.Name) and name in {"eval", "exec"}:
            findings.append({"kind": f"dynamic_{name}", "line": node.lineno})
        if name in {"run", "Popen", "call", "check_call", "check_output"}:
            for keyword in node.keywords:
                if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                    findings.append({"kind": "subprocess_shell_true", "line": node.lineno})
    return findings


def main() -> None:
    secret_hits = []
    dangerous_constructs = []
    files = source_files()
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = str(path.relative_to(ROOT))
        for label, pattern in SECRET_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                secret_hits.append({"kind": label, "path": relative, "line": line})
        for finding in ast_findings(path, text):
            dangerous_constructs.append({"path": relative, **finding})

    pip = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, cwd=ROOT, check=False
    )
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": [str(root.relative_to(ROOT)) for root in SCAN_ROOTS],
        "files_scanned": len(files),
        "secret_pattern_hits": secret_hits,
        "dangerous_python_constructs": dangerous_constructs,
        "dependency_check": {
            "exit_code": pip.returncode,
            "stdout": pip.stdout.strip(),
            "stderr": pip.stderr.strip(),
        },
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "interpretation": {
            "secret_scan_pass": not secret_hits,
            "dangerous_construct_scan_pass": not dangerous_constructs,
            "dependency_check_pass": pip.returncode == 0,
        },
        "limitations": [
            "Pattern matching cannot prove that no secret or personal data exists.",
            "The scan excludes datasets, generated artifacts, model weights, and user receipt storage.",
            "No network attack, penetration test, deployment review, or privacy impact assessment was performed.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
