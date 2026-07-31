"""FastAPI application for PDF upload and vector storage."""

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.pdf_processor import process_pdf
from app.vector_store import VectorStore

app = FastAPI(title="PDF Vector Store", version="1.0.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"
vector_store = VectorStore()


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5


class UploadResponse(BaseModel):
    message: str
    document_id: str
    filename: str
    chunks_stored: int


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        chunks = process_pdf(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    document_id = vector_store.new_document_id()
    count = vector_store.add_chunks(chunks, document_id)

    return UploadResponse(
        message="PDF processed and stored successfully.",
        document_id=document_id,
        filename=file.filename,
        chunks_stored=count,
    )


@app.post("/api/search")
async def search_documents(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    results = vector_store.search(request.query, request.n_results)
    return {"query": request.query, "results": results}


@app.get("/api/documents")
async def list_documents():
    return {"documents": vector_store.list_documents()}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
