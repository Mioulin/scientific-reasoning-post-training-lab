"""Limited publication hygiene checks, not a comprehensive vulnerability audit."""

import json
import re
import subprocess
from pathlib import Path

patterns = {
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "fine_grained_github_token": re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    "huggingface_token": re.compile(r"hf_[A-Za-z0-9]{25,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
}
files = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
findings = []
size = 0
count = 0
for name in filter(None, files):
    p = Path(name)
    if not p.is_file():
        continue
    count += 1
    size += p.stat().st_size
    if p.stat().st_size > 10 * 1024 * 1024:
        findings.append({"path": name, "kind": "large-file"})
    if any(part in {".venv", ".cache", "checkpoints", "data", ".env"} for part in p.parts):
        findings.append({"path": name, "kind": "excluded-material-tracked"})
    if p.suffix in {".bin", ".safetensors", ".pem", ".key"}:
        findings.append({"path": name, "kind": "weights-or-key"})
    text = p.read_text(errors="replace")
    for kind, pattern in patterns.items():
        if pattern.search(text):
            findings.append({"path": name, "kind": kind})
# Scan historical blobs as well, without printing matched secret material.
objects = subprocess.check_output(["git", "rev-list", "--objects", "--all"]).decode().splitlines()
blobs = 0
for line in objects:
    oid = line.split(" ", 1)[0]
    kind = subprocess.check_output(["git", "cat-file", "-t", oid]).decode().strip()
    if kind != "blob":
        continue
    blobs += 1
    content = subprocess.check_output(["git", "cat-file", "-p", oid]).decode(errors="replace")
    for name, pattern in patterns.items():
        if pattern.search(content):
            findings.append({"object": oid, "kind": name})
report = {
    "status": "COMPLETED" if not findings else "FAILED",
    "tracked_files_checked": count,
    "tracked_bytes": size,
    "historical_blobs_checked": blobs,
    "findings": findings,
    "manual_scope": [
        "Original synthetic data only; no private source files imported",
        "Model downloads pinned to a public immutable revision; no remote custom code",
        "No API keys in configs; no telemetry report_to integration enabled",
        "Weights, environment, caches and generated training data ignored",
        "Public attribution uses GitHub noreply identity",
        "Local paths in diagnostic logs reveal only project location and public author name",
    ],
    "limitations": "Heuristic secret/size checks plus source review. Not a comprehensive dependency-vulnerability audit; remote CI evidence is recorded separately in reports/publication.json.",
}
Path("reports/publication_check.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k not in ["manual_scope"]}, indent=2))
assert not findings
