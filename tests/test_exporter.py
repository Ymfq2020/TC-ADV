from pathlib import Path

from tc_adv.report.exporter import export_repository_code


def test_exporter_covers_source_tree(tmp_path: Path):
    output = tmp_path / "code_bundle.md"
    export_repository_code(output_path=output, repo_root=Path.cwd())
    payload = output.read_text(encoding="utf-8")
    assert "# FILE: src/tc_adv/cli.py" in payload
    assert "```python" in payload
    assert "Summary" not in payload
