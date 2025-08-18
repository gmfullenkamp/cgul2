# Repo Structure
offline-llm-rag/
│── models/                  # local LLM + embeddings
│── docs/                    # your documents
│── data/                    # processed vector store
│── src/
│   ├── main.py              # CLI entry point
│   ├── ingest.py            # loads & chunks documents
│   ├── retriever.py         # FAISS retriever
│   ├── llm.py               # local LLM wrapper
│   └── rag.py               # retrieval-augmented generation
│── requirements.txt
│── README.md

# TODOs
1. Implement more usable docs. Currently only supports .txt, .md, and .pdf files. Need to add .py and .docx at the very least.
2. Implement user conversation logging. This will help with looking at previous queries and answers.
3. Colored logging to help pretty-print in terminal use.
4. Tqdm for think time? If not, at least print resulting thought time.
5. Clean repo with ruff.
6. Test on larger docs and larger prompts.
