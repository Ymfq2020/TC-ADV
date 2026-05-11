"""Repository code exporter for thesis appendices."""

from __future__ import annotations

from pathlib import Path

from tc_adv.utils.io import write_text


def export_repository_code(output_path: str | Path, repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root or Path.cwd()).resolve()
    ordered_groups = [
        root / "configs",
        root / "src",
        root / "scripts",
    ]
    parts: list[str] = []
    for group in ordered_groups:
        if not group.exists():
            continue
        for file_path in sorted(path for path in group.rglob("*") if path.is_file()):
            rel = file_path.relative_to(root)
            if _should_skip_file(file_path):
                continue
            parts.append(f"```text\n# FILE: {rel}\n```")
            parts.append(f"```{_language_for(file_path)}\n{file_path.read_text(encoding='utf-8')}\n```")
    destination = Path(output_path)
    write_text(destination, "\n".join(parts) + "\n")
    return destination


def _language_for(path: Path) -> str:
    return {
        ".py": "python",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".md": "markdown",
        ".txt": "text",
        ".json": "json",
        ".jsonl": "json",
    }.get(path.suffix.lower(), "text")


def _should_skip_file(path: Path) -> bool:
    if "__pycache__" in path.parts:
        return True
    if any(part.endswith(".egg-info") for part in path.parts):
        return True
    try:
        path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return True
    return False
