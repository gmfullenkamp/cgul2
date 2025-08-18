import os
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader

class LocalEmbeddingFunction:
    """Wrapper so we can use SentenceTransformer like LangChain's embedding API."""
    def __init__(self, model_path="models/embeddings/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_path)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    def __call__(self, text):
        # FAISS calls the object itself for query embedding
        return self.embed_query(text)

def load_documents(doc_dir="docs"):
    docs = []
    for file in os.listdir(doc_dir):
        path = os.path.join(doc_dir, file)
        if file == "__init__.py":
            pass
        elif file.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif file.endswith(".txt") or file.endswith(".md"):
            docs.extend(TextLoader(path).load())
        else:
            raise ValueError(f"File {path} unsupported extension {path.split('.')[-1]}")
    return docs

def build_vectorstore(doc_dir="docs", persist_dir="data"):
    docs = load_documents(doc_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # ✅ Use SentenceTransformer directly (local path OR HF ID)
    # model_path = 'sentence-transformers/all-MiniLM-L6-v2'
    model_path = "models/embeddings/all-MiniLM-L6-v2"  # <-- set this to local folder if offline
    embedder = LocalEmbeddingFunction(model_path)

    vectordb = FAISS.from_documents(chunks, embedding=embedder)
    vectordb.save_local(persist_dir)

    print(f"✅ Vector store built with {len(chunks)} chunks and saved to {persist_dir}")

if __name__ == "__main__":
    build_vectorstore()
