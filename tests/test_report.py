from __future__ import annotations

import json
from pathlib import Path

from slimx_rag.cli import main
from slimx_rag.report import build_report, format_report_markdown


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _chunk(text: str, chunk_id: str, **metadata) -> dict:
    md = {"chunk_id": chunk_id, "doc_id": "d1", "kb_relpath": "doc.md", "content_hash": "h", "doc_type": "docs"}
    md.update(metadata)
    return {"page_content": text, "metadata": md}


def test_report_json_stats_duplicates_empty_and_coverage(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "docs.jsonl", [{"page_content": "abcdef", "metadata": {"doc_id": "d1", "kb_relpath": "doc.md", "content_len": 6, "doc_type": "docs"}}])
    _write_jsonl(
        tmp_path / "chunks.jsonl",
        [
            _chunk("", "c-empty"),
            _chunk("abcd", "c-dupe"),
            _chunk("abcd", "c-dupe"),
            _chunk("abcdefghijkl", "c-long", doc_type=""),
        ],
    )
    _write_jsonl(tmp_path / "embeddings.jsonl", [{"chunk_id": "c-empty"}])
    _write_jsonl(tmp_path / "index.jsonl", [{"chunk_id": "c-empty"}, {"chunk_id": "c-dupe"}])

    report = build_report(tmp_path)

    assert report["summary"] == {"doc_count": 1, "chunk_count": 4, "embedding_count": 1, "index_item_count": 2}
    assert report["chunk_stats"]["min_len"] == 0
    assert report["chunk_stats"]["max_len"] == 12
    assert report["chunk_stats"]["avg_len"] == 5.0
    assert report["chunk_stats"]["median_len"] == 4.0
    assert report["empty_or_near_empty_chunks"][0]["chunk_id"] == "c-empty"
    assert report["duplicates"]["duplicate_chunk_ids"] == [{"value": "c-dupe", "count": 2}]
    assert report["duplicates"]["duplicate_chunk_texts"][0]["count"] == 2
    assert report["metadata_coverage"]["chunk_id"] == 4
    assert report["metadata_coverage"]["doc_type_or_parent_doc_type"] == 3
    assert any("Missing manifest.json" in warning for warning in report["warnings"])


def test_report_markdown_contains_main_headings(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "docs.jsonl", [])
    _write_jsonl(tmp_path / "chunks.jsonl", [])

    markdown = format_report_markdown(build_report(tmp_path))

    for heading in [
        "## Build summary",
        "## Document inventory",
        "## Chunk statistics",
        "## Largest documents",
        "## Smallest chunks",
        "## Duplicate chunks",
        "## Empty or near-empty chunks",
        "## Metadata coverage",
        "## Backend information",
        "## Embedding configuration",
        "## Hash policy",
        "## Warnings",
    ]:
        assert heading in markdown


def test_report_cli_json_and_markdown(tmp_path: Path, capsys) -> None:
    _write_jsonl(tmp_path / "docs.jsonl", [{"page_content": "abc", "metadata": {"doc_id": "d1", "content_len": 3}}])
    _write_jsonl(tmp_path / "chunks.jsonl", [_chunk("abc", "c1")])

    assert main(["report", "--out-dir", str(tmp_path), "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema_version"] == "report-v1"
    assert data["summary"]["chunk_count"] == 1

    assert main(["report", "--out-dir", str(tmp_path), "--format", "markdown"]) == 0
    assert "# SlimX-RAG RAGOps Report" in capsys.readouterr().out
