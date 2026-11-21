"""
Core data structures (schemas) used across the ingestion pipeline.

Using Pydantic models gives us:
1. Clear documentation of what fields every document/chunk has.
2. Automatic validation (no silent None/type bugs).
3. Easy JSON serialization for inspection and evaluation later.
"""

from pydantic import BaseModel, Field


class PolicyDocument(BaseModel):
    """A single cleaned corporate policy Markdown document from data/corpus/."""

    doc_id: str = Field(..., description="Unique slug, e.g., 'benefits-and-perks'")
    title: str = Field(..., description="Human-readable title, e.g., 'Benefits and Perks'")
    category: str = Field(..., description="Policy domain, e.g., 'HR & Benefits', 'Security'")
    source_repo: str = Field(..., description="GitHub repo, e.g., 'basecamp/handbook'")
    source_url: str = Field(..., description="Direct link to the original file on GitHub")
    license: str = Field(default="CC BY 4.0")
    content: str = Field(..., description="Cleaned Markdown body (without YAML frontmatter)")


class DocumentChunk(BaseModel):
    """
    A single retrieval-ready chunk produced by our Header-Aware Chunker.

    Notice `text` includes the breadcrumb header:
        [Document: Benefits and Perks | Section: Vacation & Paid Time Off]
        ...body...
    so both keyword search (BM25) and semantic search (Embeddings) always know
    which document and section the chunk came from.
    """

    chunk_id: str = Field(..., description="Unique ID, e.g., 'benefits-and-perks::chunk_003'")
    doc_id: str = Field(..., description="Parent document slug")
    doc_title: str = Field(..., description="Parent document title")
    category: str = Field(..., description="Parent document category")
    section_path: str = Field(..., description="Heading breadcrumb, e.g., 'Benefits > Vacation'")
    text: str = Field(..., description="Breadcrumb-prefixed chunk text used for indexing")
    source_url: str = Field(..., description="Original GitHub URL for citations")
    word_count: int = Field(..., description="Number of words in this chunk")
