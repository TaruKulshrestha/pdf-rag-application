"""Vector database operations using FAISS and sentence-transformers."""

import json
import uuid
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "vector_data"
INDEX_FILE = DATA_DIR / "index.faiss"
METADATA_FILE = DATA_DIR / "metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(MODEL_NAME)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = self._load_or_create_index()
        self.metadata: list[dict] = self._load_metadata()

    def _load_or_create_index(self) -> faiss.IndexFlatIP:
        if INDEX_FILE.exists():
            return faiss.read_index(str(INDEX_FILE))
        return faiss.IndexFlatIP(self.dimension)

    def _load_metadata(self) -> list[dict]:
        if METADATA_FILE.exists():
            return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        return []

    def _save(self):
        faiss.write_index(self.index, str(INDEX_FILE))
        METADATA_FILE.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    def _embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)

    def add_chunks(self, chunks: list[dict], document_id: str) -> int:
        """Store tokenized chunks with embeddings in the vector database."""
        texts = [c["text"] for c in chunks]
        vectors = self._embed(texts)
        self.index.add(vectors)

        for i, chunk in enumerate(chunks):
            self.metadata.append(
                {
                    "id": f"{document_id}_{i}",
                    "text": chunk["text"],
                    "metadata": {**chunk["metadata"], "document_id": document_id},
                }
            )

        self._save()
        return len(chunks)

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Search for similar chunks by query text."""
        if self.index.ntotal == 0:
            return []

        query_vector = self._embed([query])
        k = min(n_results, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        matches = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = self.metadata[idx]
            matches.append(
                {
                    "text": entry["text"],
                    "metadata": entry["metadata"],
                    "score": float(score),
                }
            )
        return matches

    def list_documents(self) -> list[dict]:
        """List unique documents stored in the vector database."""
        docs: dict[str, dict] = {}
        for entry in self.metadata:
            meta = entry["metadata"]
            doc_id = meta.get("document_id", "unknown")
            source = meta.get("source", "unknown")
            if doc_id not in docs:
                docs[doc_id] = {"document_id": doc_id, "source": source, "chunks": 0}
            docs[doc_id]["chunks"] += 1
        return list(docs.values())

    @staticmethod
    def new_document_id() -> str:
        return str(uuid.uuid4())
