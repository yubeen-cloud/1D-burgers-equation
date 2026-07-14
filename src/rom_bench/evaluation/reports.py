"""Markdown summary report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rom_bench.paths import resolve_path


def write_markdown_report(path: str | Path, title: str, sections: dict[str, Any]) -> None:
    """Write a simple Markdown report."""
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for heading, content in sections.items():
        lines.extend([f"## {heading}", ""])
        if isinstance(content, dict):
            for key, value in content.items():
                lines.append(f"- **{key}**: {value}")
        elif isinstance(content, list):
            lines.extend([f"- {item}" for item in content])
        else:
            lines.append(str(content))
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
