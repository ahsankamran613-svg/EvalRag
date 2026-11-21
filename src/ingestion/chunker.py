"""
Header-Aware Markdown Chunker with Breadcrumb Injection.

WHY NOT BLIND FIXED-SIZE CHUNKING?
  If you blindly chop text every 350 words:
  1. You cut right through the middle of a policy section or bullet list.
  2. A chunk might say "Employees get $5,000 per year" without mentioning *which*
     benefit it belongs to (because the heading "## Continuing Education" was
     left behind in the previous chunk!).

HOW THIS SMART CHUNKER FIXES IT:
  1. It parses Markdown headings (#, ##, ###) and tracks a breadcrumb stack
     like: "Benefits and Perks > Vacation & Paid Time Off".
  2. It keeps each section intact if it fits within `max_words`.
  3. If a section is longer than `max_words`, it splits cleanly on paragraph
     boundaries (never mid-word) with a small word overlap (`overlap_words`)
     so consecutive chunks share context.
  4. It prepends `[Document: <title> | Section: <breadcrumb>]` to EVERY chunk.
"""

import re
from src.schemas import DocumentChunk, PolicyDocument


# Matches Markdown headings: "#", "##", "###", "####"
HEADING_PATTERN = re.compile(r"^(#{1,4})\s+(.+?)\s*$")


def _split_markdown_by_headers(markdown_text: str, default_section: str = "Overview") -> list[tuple[str, str]]:
    """
    Walk through a Markdown document line-by-line and group text under its
    hierarchical heading breadcrumb.

    Returns:
        List of `(section_path, section_body)` pairs.
    """
    # Stack of (heading_level, heading_title), e.g., [(1, "Benefits"), (2, "Vacation")]
    heading_stack: list[tuple[int, str]] = []
    sections: list[tuple[str, str]] = []
    current_lines: list[str] = []

    def flush_section() -> None:
        body = "\n".join(current_lines).strip()
        if not body:
            return
        breadcrumb = " > ".join(title for _, title in heading_stack) or default_section
        sections.append((breadcrumb, body))

    for line in markdown_text.splitlines():
        match = HEADING_PATTERN.match(line.strip())
        if match:
            # Save the previous section before starting a new heading
            flush_section()
            current_lines = []

            level = len(match.group(1))
            title = match.group(2).strip("# ").strip()

            # Pop any headings at the same or deeper level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
        else:
            current_lines.append(line)

    # Flush the final section at EOF
    flush_section()
    return sections


def _split_text_with_overlap(text: str, max_words: int, overlap_words: int) -> list[str]:
    """
    Split a section body into chunks of at most `max_words` words.

    Prefers splitting on paragraph breaks (`\\n\\n`). If a single paragraph is
    longer than `max_words`, falls back to word-window splitting with overlap.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current_paras: list[str] = []
    current_word_count = 0

    for para in paragraphs:
        para_words = para.split()

        # Edge case: a single paragraph is already larger than max_words
        if len(para_words) > max_words:
            if current_paras:
                chunks.append("\n\n".join(current_paras))
                current_paras = []
                current_word_count = 0

            step = max(1, max_words - overlap_words)
            for start in range(0, len(para_words), step):
                window = " ".join(para_words[start : start + max_words])
                chunks.append(window)
                if start + max_words >= len(para_words):
                    break
            continue

        # If adding this paragraph exceeds max_words, finalize the current chunk
        if current_paras and (current_word_count + len(para_words) > max_words):
            finished_chunk = "\n\n".join(current_paras)
            chunks.append(finished_chunk)

            # Carry over the last `overlap_words` words as context for the next chunk
            if overlap_words > 0:
                tail_words = finished_chunk.split()[-overlap_words:]
                overlap_text = " ".join(tail_words)
                current_paras = [overlap_text, para]
                current_word_count = len(tail_words) + len(para_words)
            else:
                current_paras = [para]
                current_word_count = len(para_words)
        else:
            current_paras.append(para)
            current_word_count += len(para_words)

    if current_paras:
        chunks.append("\n\n".join(current_paras))

    return chunks


def chunk_document(
    doc: PolicyDocument,
    max_words: int = 350,
    overlap_words: int = 50,
) -> list[DocumentChunk]:
    """Convert one `PolicyDocument` into breadcrumb-enriched `DocumentChunk`s."""
    sections = _split_markdown_by_headers(doc.content, default_section=doc.title)
    chunks: list[DocumentChunk] = []
    chunk_idx = 0

    for section_path, section_body in sections:
        pieces = _split_text_with_overlap(
            section_body,
            max_words=max_words,
            overlap_words=overlap_words,
        )
        for piece in pieces:
            breadcrumb_header = f"[Document: {doc.title} | Section: {section_path}]"
            full_text = f"{breadcrumb_header}\n{piece}"

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc.doc_id}::chunk_{chunk_idx:03d}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    category=doc.category,
                    section_path=section_path,
                    text=full_text,
                    source_url=doc.source_url,
                    word_count=len(full_text.split()),
                )
            )
            chunk_idx += 1

    return chunks


def chunk_corpus(
    docs: list[PolicyDocument],
    max_words: int = 350,
    overlap_words: int = 50,
) -> list[DocumentChunk]:
    """Chunk an entire list of `PolicyDocument`s."""
    all_chunks: list[DocumentChunk] = []
    for doc in docs:
        all_chunks.extend(
            chunk_document(doc, max_words=max_words, overlap_words=overlap_words)
        )
    return all_chunks
