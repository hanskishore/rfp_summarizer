"""FastAPI web server for the RFP Summarizer UI."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="RFP Summarizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def root():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
async def status():
    return {
        "openai_key": bool(os.environ.get("OPENAI_API_KEY")),
        "sam_key": bool(os.environ.get("SAM_GOV_API_KEY")),
    }


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #


class SAMSearchRequest(BaseModel):
    query: str
    limit: int = 5
    posted_from: str | None = None
    posted_to: str | None = None
    model: str = "gpt-4o"


class SAMNoticeRequest(BaseModel):
    notice_id: str
    use_attachments: bool = False
    model: str = "gpt-4o"


# --------------------------------------------------------------------------- #
# SSE helper
# --------------------------------------------------------------------------- #


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@app.post("/api/search")
async def search_sam(req: SAMSearchRequest):
    """Stream SAM.gov search results via SSE — one summary per event."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(400, "OPENAI_API_KEY not configured in .env")
    if not os.environ.get("SAM_GOV_API_KEY"):
        raise HTTPException(400, "SAM_GOV_API_KEY not configured in .env")

    def generate():
        try:
            from sam_gov import search_opportunities, opportunity_to_text
            from rfp_summarize import _summarize_text

            notices = search_opportunities(
                req.query,
                limit=req.limit,
                posted_from=req.posted_from,
                posted_to=req.posted_to,
            )

            yield _sse({"type": "total", "count": len(notices)})

            if not notices:
                yield _sse({"type": "done"})
                return

            for i, notice in enumerate(notices):
                title = notice.get("title", f"Opportunity {i + 1}")
                yield _sse({"type": "progress", "index": i, "title": title})

                text = opportunity_to_text(notice)
                summary = _summarize_text(
                    text, model=req.model, stream=False, output_format="none"
                )
                summary["_notice_id"] = notice.get("noticeId", "")
                summary["_sam_link"] = notice.get("uiLink", "")
                summary["_posted_date"] = notice.get("postedDate", "")
                summary["_source"] = "sam"

                yield _sse({"type": "summary", "index": i, "data": summary})

            yield _sse({"type": "done"})

        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/notice")
async def get_notice(req: SAMNoticeRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(400, "OPENAI_API_KEY not configured in .env")

    def run():
        from rfp_summarize import summarize_from_sam_id
        return summarize_from_sam_id(
            req.notice_id,
            model=req.model,
            stream=False,
            output_format="none",
            use_attachments=req.use_attachments,
        )

    loop = asyncio.get_event_loop()
    summary = await loop.run_in_executor(None, run)
    summary["_notice_id"] = req.notice_id
    summary["_source"] = "sam"
    return {"summary": summary}


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    model: str = Form(default="gpt-4o"),
):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(400, "OPENAI_API_KEY not configured in .env")

    suffix = Path(file.filename or "rfp.txt").suffix.lower()
    if suffix not in {".pdf", ".docx", ".doc", ".txt"}:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    content = await file.read()

    def run():
        from rfp_summarize import summarize_rfp
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            return summarize_rfp(
                tmp_path, model=model, stream=False, output_format="none"
            )
        finally:
            os.unlink(tmp_path)

    loop = asyncio.get_event_loop()
    summary = await loop.run_in_executor(None, run)
    summary["_source"] = "upload"
    summary["_filename"] = file.filename
    return {"summary": summary}
