"""Work record files, read by path from the folder mounted read-only into the publisher.

Only the path crosses the Knowledge MCP; the document is read here, never sent as a tool argument.
A path outside the folder, or a file that is not Markdown with front matter, is rejected.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import yaml

from aether_core.domain.work_record import WorkRecordRejectedError
from aether_core.ports import RawWorkRecord


class MountedWorkRecordFiles:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def resolve(self, path: str) -> Path:
        relative = PurePosixPath(path.replace("\\", "/"))
        candidate = Path(path) if relative.is_absolute() else self._root / relative
        resolved = candidate.resolve()
        if not resolved.is_relative_to(self._root):
            raise WorkRecordRejectedError(f"path {path!r} is outside the work records folder")
        if resolved.suffix.lower() != ".md":
            raise WorkRecordRejectedError(f"path {path!r} is not a Markdown file")
        if not resolved.is_file():
            raise WorkRecordRejectedError(f"no work record file at {path!r}")
        return resolved

    def read(self, path: str) -> RawWorkRecord:
        text = self.resolve(path).read_text(encoding="utf-8").replace("\r\n", "\n")
        if not text.startswith("---\n"):
            raise WorkRecordRejectedError("the file has no front matter")
        end = text.find("\n---", 4)
        if end == -1:
            raise WorkRecordRejectedError("the front matter is not closed with ---")
        try:
            front_matter = yaml.safe_load(text[4:end]) or {}
        except yaml.YAMLError as error:
            raise WorkRecordRejectedError(f"the front matter is not valid YAML: {error}") from error
        if not isinstance(front_matter, dict):
            raise WorkRecordRejectedError("the front matter must be a mapping of fields")
        body = text[end + 4 :].lstrip("\n")
        return RawWorkRecord({str(k): v for k, v in front_matter.items()}, body)
