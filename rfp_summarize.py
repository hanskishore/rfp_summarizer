"""RFP Summarizer — extract structured summaries from Request for Proposal documents."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

SYSTEM_PROMPT = """You are an expert procurement analyst specializing in RFP (Request for Proposal) analysis.

Your task is to read the provided RFP document and produce a structured summary in valid JSON.

Return ONLY a JSON object with these fields (omit any field for which no information is found):

{
  "title": "Official RFP title",
  "issuing_organization": "Name of the organization issuing the RFP",
  "rfp_number": "RFP reference or solicitation number",
  "issue_date": "Date the RFP was issued (ISO 8601 if determinable)",
  "due_date": "Proposal submission deadline (ISO 8601 if determinable)",
  "project_overview": "2–4 sentence plain-English description of what is being procured",
  "scope_of_work": ["Bullet list of key deliverables or work items"],
  "requirements": {
    "mandatory": ["List of must-have requirements"],
    "preferred": ["List of nice-to-have requirements"]
  },
  "evaluation_criteria": [
    {"criterion": "Criterion name", "weight": "Weight or percentage if stated"}
  ],
  "budget": {
    "amount": "Budget ceiling or estimated value if stated",
    "notes": "Any budget-related notes (e.g. fixed-price, T&M)"
  },
  "contract_type": "Type of contract (e.g. Fixed-price, Time & Materials, IDIQ)",
  "contract_period": "Performance period or contract duration",
  "eligibility": ["Any eligibility restrictions (e.g. small business set-aside)"],
  "submission_requirements": ["What must be included in the proposal"],
  "submission_format": "Preferred submission method (e.g. email, portal, hard copy)",
  "contact": {
    "name": "Point of contact name",
    "email": "Contact email",
    "phone": "Contact phone"
  },
  "key_dates": [
    {"event": "Event name", "date": "Date"}
  ],
  "questions_deadline": "Deadline for submitting questions",
  "amendments": "Note if this is an amendment to a prior RFP",
  "attachments": ["List of attachments or exhibits referenced"],
  "red_flags": ["Any unusual clauses, aggressive timelines, or concerns worth flagging"]
}"""


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_docx_file(path: Path) -> str:
    try:
        from docx import Document  # type: ignore[import-untyped]
    except ImportError:
        sys.exit(
            "python-docx is required to read .docx files.\n"
            "Install it with: pip install python-docx"
        )
    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                paragraphs.append(" | ".join(cells))
    return "\n".join(paragraphs)


def _read_pdf_file(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import-untyped]
    except ImportError:
        sys.exit(
            "pypdf is required to read .pdf files.\n"
            "Install it with: pip install pypdf"
        )
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def summarize_rfp(
    source: str,
    *,
    model: str = "gpt-4o",
    stream: bool = True,
    output_format: str = "both",  # "json" | "text" | "both"
) -> dict:
    """
    Summarize an RFP from a file path or raw text string.

    Args:
        source: Path to an RFP file (.pdf, .docx, .txt) or raw text content.
        model: OpenAI model ID to use.
        stream: Whether to stream the response (recommended for large docs).
        output_format: Controls console output. "json" prints only JSON,
                       "text" prints only the formatted summary, "both" prints both.

    Returns:
        Parsed summary dict.
    """
    client = OpenAI()

    # --- Resolve input ---
    path = Path(source)
    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            print(f"[*] Extracting text from PDF: {path.name}", file=sys.stderr)
            text_content = _read_pdf_file(path)
        elif suffix in {".docx", ".doc"}:
            print(f"[*] Extracting text from DOCX: {path.name}", file=sys.stderr)
            text_content = _read_docx_file(path)
        else:
            text_content = _read_text_file(path)
    else:
        text_content = source

    user_message = (
        f"{text_content}\n\n"
        "Please analyze the RFP document above and return a structured JSON summary "
        "following the schema in your instructions. Be thorough and extract all "
        "relevant dates, requirements, and evaluation criteria."
    )

    print(f"[*] Sending request to {model}…", file=sys.stderr)

    raw_text = _call_openai(client, model, user_message, stream=stream)

    summary = _extract_json(raw_text)

    if output_format in {"json", "both"}:
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    if output_format in {"text", "both"}:
        if output_format == "both":
            print("\n" + "─" * 60)
        _print_readable(summary)

    return summary


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _call_openai(
    client: OpenAI,
    model: str,
    user_message: str,
    *,
    stream: bool,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    if stream:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
            stream=True,
        )
        text = ""
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                text += delta
        return text
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
            stream=False,
        )
        return response.choices[0].message.content or ""


def _extract_json(raw: str) -> dict:
    """Extract the first JSON object from a string."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return {"raw_response": raw}
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        return {"raw_response": raw}


def _print_readable(summary: dict) -> None:
    """Print a human-friendly version of the summary."""
    if "raw_response" in summary:
        print(summary["raw_response"])
        return

    def _h(text: str) -> None:
        print(f"\n{'═' * 60}")
        print(f"  {text.upper()}")
        print(f"{'═' * 60}")

    def _section(title: str, content) -> None:
        if not content:
            return
        print(f"\n▸ {title}")
        if isinstance(content, str):
            print(f"  {content}")
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    parts = [f"{k}: {v}" for k, v in item.items() if v]
                    print(f"  • {' | '.join(parts)}")
                else:
                    print(f"  • {item}")
        elif isinstance(content, dict):
            for k, v in content.items():
                if v:
                    print(f"  {k}: {v}")

    title = summary.get("title", "RFP Summary")
    org = summary.get("issuing_organization", "")
    _h(f"{title}{' — ' + org if org else ''}")

    _section("RFP Number", summary.get("rfp_number"))
    _section("Issue Date", summary.get("issue_date"))
    _section("Submission Due", summary.get("due_date"))
    _section("Questions Deadline", summary.get("questions_deadline"))
    _section("Project Overview", summary.get("project_overview"))
    _section("Scope of Work", summary.get("scope_of_work"))

    reqs = summary.get("requirements", {})
    if reqs:
        print("\n▸ Requirements")
        for kind, items in reqs.items():
            if items:
                print(f"  [{kind.upper()}]")
                for item in items:
                    print(f"    • {item}")

    _section("Evaluation Criteria", summary.get("evaluation_criteria"))
    _section("Budget", summary.get("budget"))
    _section("Contract Type", summary.get("contract_type"))
    _section("Contract Period", summary.get("contract_period"))
    _section("Eligibility", summary.get("eligibility"))
    _section("Submission Requirements", summary.get("submission_requirements"))
    _section("Submission Format", summary.get("submission_format"))
    _section("Contact", summary.get("contact"))
    _section("Key Dates", summary.get("key_dates"))
    _section("Attachments", summary.get("attachments"))

    red_flags = summary.get("red_flags", [])
    if red_flags:
        print("\n▸ ⚠  Red Flags / Notes")
        for flag in red_flags:
            print(f"  ⚠  {flag}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize an RFP document using the OpenAI API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rfp_summarize.py rfp.pdf
  python rfp_summarize.py rfp.docx --format text
  python rfp_summarize.py rfp.txt --format json --no-stream
  cat rfp.txt | python rfp_summarize.py -
""",
    )
    parser.add_argument(
        "source",
        help="Path to an RFP file (.pdf, .docx, .txt) or '-' to read from stdin.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model ID (default: gpt-4o)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming (collect full response before output)",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Write JSON summary to this file (in addition to stdout)",
    )

    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(
            "Error: OPENAI_API_KEY environment variable is not set.\n"
            "Export it before running: export OPENAI_API_KEY=sk-..."
        )

    source = args.source
    if source == "-":
        source = sys.stdin.read()

    summary = summarize_rfp(
        source,
        model=args.model,
        stream=not args.no_stream,
        output_format=args.format,
    )

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[*] JSON summary written to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
