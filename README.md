# pdf-semantic-search
# 📄 PDF Semantic Search Application

A FastAPI-based web application that allows users to upload PDF documents and perform semantic search using natural language queries. The application extracts text from PDFs, generates vector embeddings, stores them in a FAISS vector database, and retrieves the most relevant content based on user queries.

## 🚀 Features

- 📂 Upload PDF documents
- 📖 Extract text from PDFs
- 🔍 Semantic search using natural language
- 🧠 Vector embeddings for accurate retrieval
- ⚡ Fast similarity search with FAISS
- 🌐 Interactive web interface
- 🐍 FastAPI backend

---

## 🛠️ Tech Stack

### Backend
- FastAPI
- Uvicorn

### AI & NLP
- Sentence Transformers
- FAISS
- NumPy

### PDF Processing
- PyPDF

### Frontend
- HTML
- CSS
- JavaScript

---

## 📁 Project Structure

```
pdf-rag-application/
│
├── app/
│   ├── main.py
│   ├── pdf_processor.py
│   ├── vector_store.py
│   └── static/
│       └── index.html
│
├── vector_data/
│   ├── index.faiss
│   └── metadata.json
│
├── requirements.txt
├── run.py
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/TaruKulshrestha/pdf-rag-application.git
```

Move into the project directory:

```bash
cd pdf-rag-application
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the FastAPI server:

```bash
python run.py
```

Open your browser and visit:

```
http://localhost:8000
```

---

## 📖 How It Works

1. Upload one or more PDF files.
2. The application extracts text from each document.
3. Text is converted into vector embeddings.
4. Embeddings are stored in a FAISS vector database.
5. Enter a natural language query.
6. The application returns the most semantically relevant content from the uploaded PDFs.

---

## 🔮 Future Improvements

- User authentication
- Chat with multiple PDFs
- Conversation history
- Support for DOCX and TXT files
- Cloud storage integration
- Deployment on cloud platforms
- Citation highlighting
- OCR support for scanned PDFs

---

## 👨‍💻 Author

**Taru Kulshrestha**

GitHub: https://github.com/TaruKulshrestha

---

## 📄 License

This project is licensed under the MIT License.
