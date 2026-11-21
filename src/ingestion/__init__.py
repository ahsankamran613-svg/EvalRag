"""Ingestion package: loading cleaned Markdown policies and smart chunking."""

from src.ingestion.loader import load_corpus, load_policy_file
from src.ingestion.chunker import chunk_corpus, chunk_document

__all__ = ["load_corpus", "load_policy_file", "chunk_corpus", "chunk_document"]
