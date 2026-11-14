"""Secret scanner: rejects work records that look like they contain credentials.

Reports only the names of the patterns found, never the matched text.
"""

from __future__ import annotations

import re

PATTERNS: dict[str, re.Pattern[str]] = {
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(
        r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b"
    ),
    "Atlassian API token": re.compile(r"\bATATT[A-Za-z0-9_\-=]{20,}"),
    "Slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "JSON web token": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    "connection string with password": re.compile(
        r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s@/]+@", re.IGNORECASE
    ),
    "assigned secret": re.compile(
        r"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|personal[_-]?access[_-]?token|pat)"
        r"\s*[:=]\s*['\"]?[A-Za-z0-9_\-/+]{12,}"
    ),
}


class RegexSecretScanner:
    def scan(self, text: str) -> list[str]:
        return [name for name, pattern in PATTERNS.items() if pattern.search(text)]
