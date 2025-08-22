# CGUL2 - Comprehensive Generation Using LLMs v2

This project allows you to build a local question-answering system over documents (PDF, TXT, MD, DOCX, PY) using embeddings and a large language model. It supports offline workflows with local embeddings and LLMs.

## Features

- Load documents from a folder (docs/) in PDF, TXT, DOCX, Python, or Markdown format.
- Split documents into manageable chunks for embeddings.
- Generate vector embeddings locally using SentenceTransformers.
- Store and retrieve embeddings efficiently with FAISS.
- Query documents interactively with a Hugging Face text-generation model.
- Provides concise, citation-aware answers.

## Repo Structure
```html
cgul2/
│── docs/                    # your documents
│── src/
│   ├── models/              # local LLMs and word to vec embeddings
│   ├── vector_store/        # processed vector store
│   ├── core/
│   │   ├── auto_doc.py      # CLI entry point for auto doc string python repo
│   │   ├── main.py          # CLI entry point for chatting with doc citation
│   │   └── ingest.py        # CLI entry point for loading & chunking documents
│   ├── gui/                 # will contain gui
│   ├── constants.py         # folder location constants
│   └── utils.py             # extra utilities
│── tests/
│   ├── test_core.py
│   └── test_gui.py
│── .gitignore
│── pyproject.toml
│── requirements.txt
└── README.md
```

## Installation

Using Python 3.8+:
```bash
pip install -r requirements.txt
pip install -e .
```
Make sure you have your document files in the docs/ fodler and modesl in the models/ directory.

For faster vector store search:
```bash
pip install faiss-gpu-cu12
```

## Usage

1. Build the vector store
```bash
python ingest.py
```
This will:
- Load documents from docs/.
- Split them into chunks (default 1000 characters with 100 overlap).
- Embed chunks with SentenceTransformer.
- Save the FAISS vector store to src/vector_store/.
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

- Document directory: docs/ by default.
- TODO: Document CLI configurations for all files.

## Repo Cleaning and Testing

Run the following commands and fix any issues they show:
```bash
ruff check .\src\ --fix
python .\tests\test_smoke.py
```

## Notes

- The system works fully offline if you have downloaded models locally.
- Only PDF, TXT, DOCX, Python, and Markdown files are supported.
- You can adjust chunk size, overlap, and top-k retrieval in the scripts.

## Dependencies

- Python 3.8+.
- See requirements.txt.
- (Optional) A machine with GPUs for faster inferencing times.

## TODOs
1. Implement user conversation logging. This will help with looking at previous queries and answers.
2. Add line specifiers or page specifiers for each doc for more precise citations.
3. Add a simple UI for nicer chatting and document uploading? (Similar to ChatGPT premium with doc uploading?)
4. Quantization using bits and bytes?
5. More customizable model use and answer parsing: (https://cookbook.openai.com/articles/gpt-oss/run-transformers)
