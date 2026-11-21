"""
Markdown corpus loader.

Reads cleaned policy files from `data/corpus/*.md`, parses their YAML frontmatter
metadata, and returns typed `PolicyDocument` objects.
"""

from pathlib import Path
import yaml

from src.schemas import PolicyDocument


def load_policy_file(file_path: Path) -> PolicyDocument:
    """Load a single Markdown policy file with YAML frontmatter."""
    raw = file_path.read_text(encoding="utf-8")

    if not raw.startswith("---"):
        raise ValueError(f"Missing YAML frontmatter in {file_path}")

    # Split into: ["", yaml_block, markdown_body]
    parts = raw.split("---", maxsplit=2)
    if len(parts) < 3:
        raise ValueError(f"Malformed YAML frontmatter in {file_path}")

    metadata = yaml.safe_load(parts[1]) or {}
    body = parts[2].strip()

    return PolicyDocument(
        doc_id=metadata["doc_id"],
        title=metadata["title"],
        category=metadata["category"],
        source_repo=metadata["source_repo"],
        source_url=metadata["source_url"],
        license=metadata.get("license", "CC BY 4.0"),
        content=body,
    )


def load_corpus(corpus_dir: Path) -> list[PolicyDocument]:
    """Load all policy Markdown files from `corpus_dir` (skipping ATTRIBUTION.md)."""
    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {corpus_dir}. "
            "Run `python scripts/fetch_corpus.py` first."
        )

    documents: list[PolicyDocument] = []
    for md_file in sorted(corpus_dir.glob("*.md")):
        if md_file.name.upper() == "ATTRIBUTION.MD":
            continue
        documents.append(load_policy_file(md_file))

    return documents
