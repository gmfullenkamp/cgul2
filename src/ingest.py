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
from utils import download_model


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

def load_documents(doc_dir: str = "docs") -> list:
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
            msg = f"File {path} has unsupported extension {path.split('.')[-1]}"
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

def build_vectorstore(doc_dir: str = "docs",
                      persist_dir: str = vector_store_dir) -> None:
    """Build the vector store from the given docs folder."""
    docs = load_documents(doc_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embedder = LocalEmbeddingFunction()

    vectordb = FAISS.from_documents(chunks, embedding=embedder)
    vectordb.save_local(persist_dir)


if __name__ == "__main__":
    build_vectorstore()
