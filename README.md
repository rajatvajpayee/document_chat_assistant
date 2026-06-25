# Document Chat Assistant (RAG)

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about PDF documents using semantic search and a local Large Language Model (LLM) powered by Ollama.

This project was built to understand the complete RAG pipeline from first principles while leveraging LangChain and LlamaIndex where appropriate.

---

## Features

* Load PDF documents
* Split documents into semantic chunks
* Generate embeddings using Sentence Transformers
* Store embeddings in a FAISS vector index
* Perform semantic retrieval
* Generate grounded answers using Qwen3 running locally with Ollama
* Display source document and page number for retrieved context
* Separate ingestion and query pipelines

---

## Architecture

```
                  PDF Documents
                        │
                        ▼
             LlamaIndex Document Loader
                        │
                        ▼
                 Sentence Splitter
                        │
                        ▼
                      Nodes
                        │
                        ▼
          SentenceTransformer Embeddings
                        │
                        ▼
                  FAISS Vector Store
                        │
                        ▼
                    Retriever
                        │
                        ▼
                Top-K Relevant Nodes
                        │
                        ▼
                Prompt Construction
                        │
                        ▼
             Ollama (Qwen3:8B)
                        │
                        ▼
                   Final Answer
```

---

## Project Structure

```
document_chat_assistant/

│
├── data/
│   └── attention.pdf
│
├── storage/
│   ├── faiss.index
│   └── nodes.pkl
│
├── config.py
├── embeddings.py
├── vector_store.py
├── retriever.py
├── prompt.py
├── llm.py
├── ingest.py
├── chat.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

* Python
* Ollama
* Qwen3:8B
* LlamaIndex
* Sentence Transformers
* FAISS
* PyMuPDF

---

## Installation

Clone the repository.

```bash
git clone <repository_url>

cd document_chat_assistant
```

Create a virtual environment.

```bash
python -m venv .venv

source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Make sure Ollama is installed and the model is available.

```bash
ollama list
```

Expected:

```
qwen3:8b
```

If not available:

```bash
ollama pull qwen3:8b
```

---

## Add Documents

Place one or more PDF files inside

```
data/
```

Example

```
data/
    attention.pdf
```

---

## Build the Vector Index

Generate chunks, embeddings and the FAISS index.

```bash
python ingest.py
```

This creates

```
storage/
    faiss.index
    nodes.pkl
```

This step only needs to be rerun when documents change.

---

## Chat with Documents

Start the chatbot.

```bash
python chat.py
```

Example

```
Ask>

How many attention heads are used?
```

Example output

```
Answer

The Transformer base model uses eight attention heads.

Sources

attention.pdf (Page 5)
```

---

## Retrieval Pipeline

1. Load PDF
2. Split into chunks (Nodes)
3. Generate embeddings
4. Store embeddings in FAISS
5. Embed user query
6. Retrieve Top-K relevant chunks
7. Build prompt
8. Generate grounded response using Qwen3

---

## Components

### LlamaIndex

Used for

* Document loading
* Node creation
* Metadata management

---

### Sentence Transformers

Generates dense vector embeddings for document chunks and user queries.

Default model:

```
sentence-transformers/all-MiniLM-L6-v2
```

---

### FAISS

Responsible for

* Vector indexing
* Similarity search
* Fast nearest-neighbor retrieval

---

### Retriever

Converts retrieved vector indices into rich documents containing

* text
* source file
* page number
* similarity score

This abstraction hides the underlying vector database from the rest of the application.

---

### Ollama

Runs the LLM locally.

Current model:

```
qwen3:8b
```

---

## Metadata-Aware Retrieval

Each retrieved node includes metadata such as

* source file
* page number

This allows generated answers to include citations, improving transparency and user trust.

---

## Current Features

* PDF ingestion
* Node-based chunking
* Dense embeddings
* FAISS semantic retrieval
* Metadata-aware retrieval
* Local LLM inference
* Source citations
* Modular retriever architecture
* Conversation memory
* Cross-Encoder reranking

---

## Planned Improvements

* History-aware retrieval
* Hybrid Search (BM25 + Dense Retrieval)
* Metadata filtering
* Incremental indexing
* Streaming responses
* Web UI
* RAGAS evaluation
* Experiment tracking
* Support for multiple embedding models

---

## License

This project is intended for learning and educational purposes.
