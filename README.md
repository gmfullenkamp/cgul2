# CGUL2 - Comprehensive Generation Using LLMs v2

This project allows you to build a local question-answering system over documents (PDF, TXT, MD) using embeddings and a large language model. It supports offline workflows with local embeddings and LLMs.

## Features

- Load documents from a folder (docs/) in PDF, TXT, or Markdown format.
- Split documents into manageable chunks for embeddings.
- Generate vector embeddings locally using SentenceTransformers.
- Store and retrieve embeddings efficiently with FAISS.
- Query documents interactively with a Hugging Face text-generation model.
- Provides concise, citation-aware answers.

## Repo Structure
```html
cgul2/
│── models/                  # local LLMs
│   └── embeddings/          # local word to vec embeddings
│── docs/                    # your documents
│── data/                    # processed vector store
│── src/
│   ├── main.py              # CLI entry point
│   └── ingest.py            # loads & chunks documents
│── .gitignore
│── requirements.txt
└── README.md
```

## Installation

Using Python 3.8+:
```bash
pip install -r requirements.txt
```
Make sure you have your document files in the docs/ fodler and modesl in the models/ directory.

## Usage

1. Build the vector store
```bash
python ingest.py
```
This will:
- Load documents from docs/.
- Split them into chunks (default 1000 characters with 100 overlap).
- Embed chunks with SentenceTransformer.
- Save the FAISS vector store to data/.
2. Run interactive Q&A
```bash
python main.py
```
- Loads teh FAISS vector store and your local LLM.
- Retrieves relevant documents based on your query.
- Generate concise, citation-aware answers.
Sample Interaction:
```
Ask a question (or type 'exit'): What is the capital of France?
Answer: Paris is the capital of France. (source.pdf)
Sources: ['source.pdf']
```
Type exit to quit.

## Configuration

- Embedding model path: Set in ingest.py with model_path="models/embeddings/all-MiniLM-L6-v2".
- LLM path: Set in main.py with model_name="models/gpt-oss-20b".
- Document directory: docs/ by default.
- Vector store persistence: data/ by default.

## Notes

- The system works fully offline if you have downloaded models locally.
- Only PDF, TXT, and Markdown files are supported.
- You can adjust chunk size, overlap, and top-k retrieval in the scripts.

## Dependencies

- Python 3.8+.
- See requirements.txt.
- (Optional) A machine with GPUs for faster inferencing times.

## TODOs
1. Implement more usable docs. Currently only supports .txt, .md, and .pdf files. Need to add .py and .docx at the very least.
2. Implement user conversation logging. This will help with looking at previous queries and answers.
3. Colored logging to help pretty-print in terminal use.
4. Tqdm for think time? If not, at least print resulting thought time.
5. Clean repo with ruff.
6. Test on larger docs and larger prompts.
