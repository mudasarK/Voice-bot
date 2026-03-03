import asyncio
import tempfile
import os
from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from backend.api.dependencies import (
    get_rag,
    get_vector_store,
    get_voice_processor,
)
from backend.core.interfaces import VectorStoreProvider
from backend.rag.chain import RAGChain
from backend.rag.document_loader import DocumentIngestor
from backend.voice.processor import VoiceProcessor

router = APIRouter()

# Map filename extension / content-type to OpenRouter audio format
AUDIO_FORMAT_MAP = {
    "wav": "wav",
    "wave": "wav",
    "mp3": "mp3",
    "mpeg": "mp3",
    "ogg": "ogg",
    "flac": "flac",
    "m4a": "m4a",
}


def _detect_audio_format(filename: Optional[str], content_type: Optional[str]) -> str:
    if filename:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in AUDIO_FORMAT_MAP:
            return AUDIO_FORMAT_MAP[ext]
    if content_type and "audio/" in content_type:
        subtype = content_type.split("/")[-1].split(";")[0].strip().lower()
        return AUDIO_FORMAT_MAP.get(subtype, "wav")
    return "wav"


class ChatRequest(BaseModel):
    message: str


@router.post("/chat")
async def text_chat(
    request: ChatRequest,
    rag: RAGChain = Depends(get_rag),
):
    """Text-only RAG interactions. Runs in thread pool to avoid blocking."""
    response = await asyncio.to_thread(rag.invoke, request.message)
    return {"response": response}


@router.post("/voice")
async def voice_chat(
    audio_file: UploadFile = File(...),
    voice_processor: VoiceProcessor = Depends(get_voice_processor),
):
    """Full audio -> RAG -> audio pipeline. Runs in thread pool to avoid blocking."""
    audio_bytes = await audio_file.read()
    audio_format = _detect_audio_format(
        audio_file.filename,
        audio_file.content_type,
    )
    user_text, bot_response_text, bot_audio_bytes = await asyncio.to_thread(
        voice_processor.process_audio,
        audio_bytes,
        audio_format,
    )
    return Response(
        content=bot_audio_bytes,
        media_type="audio/mpeg",
        headers={
            "X-User-Text": user_text.replace("\n", " "),
            "X-Bot-Text": bot_response_text.replace("\n", " "),
        },
    )


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    store: VectorStoreProvider = Depends(get_vector_store),
):
    """Upload a document (PDF or TXT); split and add to the vector store."""
    if not file.filename:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing filename"},
        )
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ("pdf", "txt", "text"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only PDF and TXT files are supported"},
        )
    suffix = f".{ext}"
    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        ingestor = DocumentIngestor()
        docs = ingestor.load_and_split(tmp_path)
        if not docs:
            return JSONResponse(
                status_code=400,
                content={"error": "No content could be extracted from the file"},
            )
        store.add_documents(docs)
        return {"status": "ok", "chunks_added": len(docs), "filename": file.filename}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
