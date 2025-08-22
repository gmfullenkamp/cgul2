"""Main for creating a vector store.

This script runs a basic sentence transformer on a directory of given
documents and creates a vector store that can be queried by an LLM.
"""

from pathlib import Path

from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

from constants import vector_store_dir
from utils import clogger, download_model


class LocalEmbeddingFunction(Embeddings):
    """Wrapper so we can use SentenceTransformer like LangChain's embedding API."""

    def __init__(self, model_path: str =
                 "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Initialize the local embedding function for the vector store."""
        model_path = download_model(model_path)
        self.model = SentenceTransformer(str(model_path))

    def embed_documents(self, texts: list) -> list:
        """Embed a list of documents into a list of vectors."""
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list:
        """Embed a text query to a vector."""
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    def __call__(self, text: str) -> str:
        """Call the object itself for query embedding."""
        return self.embed_query(text)

def load_documents(doc_dir: str) -> list:
    """Load the documents into langchain readable formats."""
    docs = []
    for path in Path(doc_dir).iterdir():
        if path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(path).load())
        elif path.suffix.lower() in [".txt", ".md", ".py"]:
            docs.extend(TextLoader(path).load())
        elif path.suffix.lower() == ".docx":
            docs.extend(UnstructuredWordDocumentLoader(path).load())
        else:
            msg = f"File {path} has unsupported extension {str(path).split('.')[-1]}"
            raise ValueError(msg)

    if not docs:
        msg = (
            f"No documents found in the {doc_dir} folder. "
            "Be sure to populate your documents with pdf, txt, md, py, or docx files."
        )
        raise ValueError(
            msg,
        )
    return docs

def build_vectorstore(doc_dir: str, persist_dir: str, model_path: str, chunk_size: int,
                      chunk_overlap: int) -> None:
    """Build the vector store from the given docs folder."""
    docs = load_documents(doc_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    embedder = LocalEmbeddingFunction(model_path=model_path)

    vectordb = FAISS.from_documents(chunks, embedding=embedder)
    vectordb.save_local(persist_dir)
    clogger.info(f"""Finished building vector store for {len(docs)} docs
                 and {len(chunks)} chunks.""")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="""Create a vector store for document citing
        using an embedding model.""",
    )
    parser.add_argument("--doc_dir", type=str, default="docs",
                        help="Path to the Python repository.")
    parser.add_argument("--model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model to use.")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Size of vector store chunks for referencing.")
    parser.add_argument("--chunk_overlap", type=int, default=100,
                        help="Overlap between vector store chunks.")

    args = parser.parse_args()
    build_vectorstore(args.doc_dir, vector_store_dir, args.model, args.chunk_size,
                      args.chunk_overlap)
