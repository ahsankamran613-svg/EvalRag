"""
Download and clean our real, open-source corporate policy corpus.

Sources (both licensed under Creative Commons Attribution 4.0 — CC BY 4.0):
  1. basecamp/handbook — Employee Handbook (HR, Benefits, Career, IT, Culture)
  2. basecamp/policies — Company Policies (Security, Privacy, Abuse, Billing, SLA)

Run from the project root:
    .\\.venv\\Scripts\\python.exe scripts\\fetch_corpus.py
"""

from dataclasses import dataclass
from pathlib import Path
import re
import httpx
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"


@dataclass(frozen=True)
class CorpusItem:
    """Metadata describing one real policy file to fetch from GitHub."""

    doc_id: str
    title: str
    category: str
    repo: str
    branch: str
    remote_path: str

    @property
    def raw_url(self) -> str:
        return f"https://raw.githubusercontent.com/{self.repo}/{self.branch}/{self.remote_path}"

    @property
    def github_url(self) -> str:
        return f"https://github.com/{self.repo}/blob/{self.branch}/{self.remote_path}"


# 26 real, substantive policy documents (15 Handbook + 11 Policies)
CURATED_CORPUS: list[CorpusItem] = [
    # --- Basecamp Employee Handbook (basecamp/handbook) ---
    CorpusItem("benefits-and-perks", "Benefits and Perks", "HR & Benefits", "basecamp/handbook", "master", "benefits-and-perks.md"),
    CorpusItem("getting-started", "Getting Started", "Onboarding", "basecamp/handbook", "master", "getting-started.md"),
    CorpusItem("how-we-work", "How We Work", "Culture & Work", "basecamp/handbook", "master", "how-we-work.md"),
    CorpusItem("making-a-career", "Making a Career", "Career & Compensation", "basecamp/handbook", "master", "making-a-career.md"),
    CorpusItem("managing-work-devices", "Managing Work Devices", "IT & Devices", "basecamp/handbook", "master", "managing-work-devices.md"),
    CorpusItem("moonlighting", "Moonlighting Policy", "Ethics & Compliance", "basecamp/handbook", "master", "moonlighting.md"),
    CorpusItem("our-internal-systems", "Our Internal Systems", "IT & Systems", "basecamp/handbook", "master", "our-internal-systems.md"),
    CorpusItem("our-rituals", "Our Rituals", "Culture & Work", "basecamp/handbook", "master", "our-rituals.md"),
    CorpusItem("severance", "Severance Policy", "HR & Benefits", "basecamp/handbook", "master", "severance.md"),
    CorpusItem("state-fmla", "State FMLA and Leave Policies", "HR & Benefits", "basecamp/handbook", "master", "stateFMLA.md"),
    CorpusItem("titles-for-qa", "Titles for QA", "Career & Titles", "basecamp/handbook", "master", "titles-for-QA.md"),
    CorpusItem("titles-for-designers", "Titles for Designers", "Career & Titles", "basecamp/handbook", "master", "titles-for-designers.md"),
    CorpusItem("titles-for-ops", "Titles for Ops", "Career & Titles", "basecamp/handbook", "master", "titles-for-ops.md"),
    CorpusItem("titles-for-programmers", "Titles for Programmers", "Career & Titles", "basecamp/handbook", "master", "titles-for-programmers.md"),
    CorpusItem("titles-for-support", "Titles for Support", "Career & Titles", "basecamp/handbook", "master", "titles-for-support.md"),

    # --- Basecamp Public Policies (basecamp/policies) ---
    CorpusItem("security-overview", "Security Overview", "Security", "basecamp/policies", "master", "security/index.md"),
    CorpusItem("security-incident-response", "Security Incident Response", "Security", "basecamp/policies", "master", "security/response/index.md"),
    CorpusItem("privacy-policy", "Privacy Policy", "Privacy & Legal", "basecamp/policies", "master", "privacy/index.md"),
    CorpusItem("privacy-recruitment", "Recruitment Privacy Policy", "Privacy & Legal", "basecamp/policies", "master", "privacy/recruitment/index.md"),
    CorpusItem("privacy-ccpa", "CCPA Privacy Notice", "Privacy & Legal", "basecamp/policies", "master", "privacy/regulations/ccpa/index.md"),
    CorpusItem("abuse-policy", "Use Restrictions and Abuse Policy", "Compliance & Abuse", "basecamp/policies", "master", "abuse/index.md"),
    CorpusItem("abuse-handling", "How We Handle Abuse Reports", "Compliance & Abuse", "basecamp/policies", "master", "abuse/how-we-handle/index.md"),
    CorpusItem("cancellation-policy", "Cancellation Policy", "Billing & Customer", "basecamp/policies", "master", "cancellation/index.md"),
    CorpusItem("refund-policy", "Refund Policy", "Billing & Customer", "basecamp/policies", "master", "refund/index.md"),
    CorpusItem("copyright-policy", "Copyright and DMCA Policy", "Privacy & Legal", "basecamp/policies", "master", "copyright/index.md"),
    CorpusItem("service-level-agreement", "Service Level Agreement (SLA)", "Security & Operations", "basecamp/policies", "master", "sla/index.md"),
]


def clean_markdown(raw_text: str) -> str:
    """
    Strip Jekyll/Liquid website boilerplate while keeping clean Markdown text
    and headings (#, ##, ###) intact for our Header-Aware Chunker.
    """
    text = raw_text.replace("\r\n", "\n").strip()

    # 1. Remove existing Jekyll YAML frontmatter (--- ... ---) at top of file
    if text.startswith("---"):
        text = re.sub(r"^---\n.*?\n---\n?", "", text, count=1, flags=re.DOTALL)

    # 2. Remove HTML comments (<!-- ... -->)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # 3. Remove Liquid tags like {% include ... %} or {{ site.url }}
    text = re.sub(r"\{%.*?%\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)

    # 4. Remove inline Kramdown attribute tags like {: .note } or {: #id }
    text = re.sub(r"\{:.*?\}", "", text)

    # 5. Collapse 3+ consecutive blank lines into at most 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def format_document_with_frontmatter(item: CorpusItem, cleaned_body: str) -> str:
    """Attach clean, standardized YAML frontmatter to the top of each Markdown file."""
    metadata = {
        "doc_id": item.doc_id,
        "title": item.title,
        "category": item.category,
        "source_repo": item.repo,
        "source_url": item.github_url,
        "license": "CC BY 4.0",
    }
    yaml_header = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{yaml_header}\n---\n\n{cleaned_body}\n"


def write_attribution_file(items: list[CorpusItem]) -> None:
    """Create ATTRIBUTION.md to properly honor the CC BY 4.0 license."""
    lines = [
        "# Corpus Attribution & License",
        "",
        "All corporate policy documents in `data/corpus/` are real, open-source",
        "policies authored by **37signals / Basecamp** and published under the",
        "[Creative Commons Attribution 4.0 International License (CC BY 4.0)]"
        "(https://creativecommons.org/licenses/by/4.0/).",
        "",
        "Minimal formatting cleanup (removing Jekyll/Liquid site templates and adding",
        "standardized YAML metadata headers) was applied for RAG indexing.",
        "",
        "## Included Documents",
        "",
        "| Doc ID | Title | Category | Original Source |",
        "|---|---|---|---|",
    ]
    for item in items:
        lines.append(
            f"| `{item.doc_id}` | {item.title} | {item.category} | "
            f"[{item.repo}/{item.remote_path}]({item.github_url}) |"
        )

    (CORPUS_DIR / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(CURATED_CORPUS)} real policy documents into {CORPUS_DIR}...\n")

    fetched_items: list[CorpusItem] = []

    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        for item in CURATED_CORPUS:
            response = client.get(item.raw_url)
            response.raise_for_status()

            cleaned = clean_markdown(response.text)
            word_count = len(cleaned.split())

            out_path = CORPUS_DIR / f"{item.doc_id}.md"
            out_path.write_text(
                format_document_with_frontmatter(item, cleaned),
                encoding="utf-8",
            )
            fetched_items.append(item)
            print(f"  [OK] {item.doc_id:<28} | {item.category:<21} | {word_count:>5} words")

    write_attribution_file(fetched_items)
    print(f"\nDone! Saved {len(fetched_items)} policy docs + ATTRIBUTION.md.")


if __name__ == "__main__":
    main()
