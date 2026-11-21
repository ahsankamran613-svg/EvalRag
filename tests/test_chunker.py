"""Unit tests for our Header-Aware Markdown Chunker (Milestone 1)."""

from src.ingestion.chunker import chunk_document
from src.schemas import PolicyDocument


def _make_sample_doc(content: str) -> PolicyDocument:
    return PolicyDocument(
        doc_id="sample-policy",
        title="Benefits and Perks",
        category="HR & Benefits",
        source_repo="basecamp/handbook",
        source_url="https://github.com/basecamp/handbook/blob/master/benefits-and-perks.md",
        content=content,
    )


def test_preserves_nested_heading_breadcrumbs():
    md = """
# Benefits and Perks

Welcome to our benefits guide.

## Time Off

### Vacation

Everyone gets 3 weeks of paid vacation per year.

### Sabbaticals

Every 3 years you get a 30-day paid sabbatical.
"""
    chunks = chunk_document(_make_sample_doc(md), max_words=350, overlap_words=50)

    assert len(chunks) == 3
    assert chunks[0].section_path == "Benefits and Perks"
    assert chunks[1].section_path == "Benefits and Perks > Time Off > Vacation"
    assert chunks[2].section_path == "Benefits and Perks > Time Off > Sabbaticals"

    # Every chunk must start with the breadcrumb header
    assert chunks[1].text.startswith(
        "[Document: Benefits and Perks | Section: Benefits and Perks > Time Off > Vacation]"
    )
    assert "3 weeks of paid vacation" in chunks[1].text


def test_splits_long_section_with_overlap():
    # Create 3 paragraphs of 40 words each (120 words total) and set max_words=60
    para_a = " ".join([f"alpha{i}" for i in range(40)])
    para_b = " ".join([f"beta{i}" for i in range(40)])
    para_c = " ".join([f"gamma{i}" for i in range(40)])
    md = f"## Long Policy\n\n{para_a}\n\n{para_b}\n\n{para_c}"

    chunks = chunk_document(_make_sample_doc(md), max_words=60, overlap_words=10)

    assert len(chunks) == 3
    # All 3 chunks inherit the same section breadcrumb
    assert all(c.section_path == "Long Policy" for c in chunks)
    # Chunk 2 should carry over the last 10 words of Chunk 1 (alpha30..alpha39)
    assert "alpha39" in chunks[1].text
    assert "beta0" in chunks[1].text
    # Chunk IDs should be sequential and unique
    assert [c.chunk_id for c in chunks] == [
        "sample-policy::chunk_000",
        "sample-policy::chunk_001",
        "sample-policy::chunk_002",
    ]


def test_ignores_empty_headings():
    md = """
# Title Only

## Empty Section

## Real Section

Actual policy details live here.
"""
    chunks = chunk_document(_make_sample_doc(md))
    assert len(chunks) == 1
    assert chunks[0].section_path == "Title Only > Real Section"
