"""Main for running LLM doc citing.

This script runs a basic sentence transformer on a directory of given
documents and creates a vector store that can be queried by ChatGPT4
locally to create citations when answering questions.
"""

import time
from datetime import datetime, timezone
from pathlib import Path

from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import GenerationConfig, pipeline

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
    clogger.info(f"Finished building vector store for {len(docs)}"
                 f" docs and {len(chunks)} chunks.")

def load_llm(model_name: str = "openai/gpt-oss-20b") -> pipeline:
    """Load Hugging Face text-generation pipeline."""
    model_name = download_model(model_name)
    return pipeline(
        "text-generation",
        model=str(model_name),
        torch_dtype="auto",
        device_map="auto",
    )

def generate_response(pipe: pipeline, prompt: PromptTemplate,
                      max_new_tokens: int = 256) -> str:
    """Generate text using the pipeline."""
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,  # deterministic
    )
    outputs = pipe(prompt, generation_config=gen_config)
    return outputs[0]["generated_text"]

def chat(embedding_model: str, model: str, reasoning_level: str, knowledge_cutoff: str,
         vector_store: str, k_nearest: int, max_new_tokens: int) -> None:
    """Talk with LLM that cites documents."""
    if reasoning_level not in ["high", "medium", "low"]:
        exception_str = f"""Reasing level must be high, medium, or low.
        Not '{reasoning_level}'."""
        raise ValueError(exception_str)

    current_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    # Load vector store
    embeddings = LocalEmbeddingFunction(embedding_model)
    db = FAISS.load_local(vector_store, embeddings,
                          allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_kwargs={"k": k_nearest})

    template = """
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: {knowledge_cutoff}
Current date: {current_date}

Reasoning: {reasoning_level}

# Valid channels: analysis, commentary, final.
# Channel must be included for every message.<|end|>

<|start|>developer<|message|># Instructions

Use the context below to answer the question as concisely as possible.
Include a citation in parentheses indicating the source of the information.

# Tools (none required for this task)

<|end|><|start|>user<|message|>Context:
{context}

Question:
{question}

Answer (concisely, include citation):<|end|>
"""

    # Prompt template for concise, citation-aware answers
    prompt_template = PromptTemplate(
        template=template,  # using openai harmony format
        input_variables=["knowledge_cutoff", "current_date", "reasoning_level",
                         "context", "question"],
    )

    pipe = load_llm(model)

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        # Measure reference time
        start_time = time.time()
        # Retrieve top-k relevant documents
        context_docs = retriever.invoke(query)
        elapsed_time = time.time() - start_time
        clogger.info(f"[Reference Time: {elapsed_time:.2f} seconds]")

        # Label each context snippet with its source
        context_text = "\n".join(
            [f"[{doc.metadata.get('source', 'Context')}] {doc.page_content}" \
             for doc in context_docs],
        )

        # Fill prompt
        prompt_text = prompt_template.format(
            question=query, context=context_text, reasoning_level=reasoning_level,
            knowledge_cutoff=knowledge_cutoff, current_date=current_date,
        )

        # Measure thought time
        start_time = time.time()
        # Generate answer
        answer = generate_response(pipe, prompt_text, max_new_tokens=max_new_tokens)
        elapsed_time = time.time() - start_time
        clogger.info(f"[Thought Time: {elapsed_time:.2f} seconds]")

        # Remove prompt and thought
        if "assistantanalysis" in answer:
            prompt, answer = answer.split("assistantanalysis")
        if "assistantfinal" in answer:
            thought, answer = answer.split("assistantfinal")

        # Display results
        sources = [doc.metadata.get("source", "Context") for doc in context_docs]

        clogger.info("\nAnswer: " + answer)
        clogger.info(f"Sources: {sources}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="""Create a vector store for document citing using an
        embedding model then chat using an LLM with a citation ability.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--doc_dir", type=str, default="docs",
                        help="Path to the Python repository.")
    parser.add_argument("--embedding_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model to use.")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Size of vector store chunks for referencing.")
    parser.add_argument("--chunk_overlap", type=int, default=100,
                        help="Overlap between vector store chunks.")
    # Optional: let users skip ingest if they know the store already exists
    parser.add_argument("--skip_ingest", action="store_true",
                        help="""If set, do not rebuild the vector
                        store before chatting.""")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b",
                        help="LLM model to use.")
    parser.add_argument("--reasoning_level", type=str, default="low",
                        help="""Level of reasoning to use when responding.
                        Must be high, medium, or low. The level of reasoning
                        is directly tied to the amount of tokens needed to
                        get to a final answer.""")
    parser.add_argument("--knowledge_cutoff", type=str, default="2024-06",
                        help="How far back in time the model can think.")
    parser.add_argument("--k_nearest", type=int, default=1,
                        help="Number of documents to reference.")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="""Amount of characters the model can make
                        when generating a response. This includes context
                        and analysis of the problem. This is directly tied
                        to the amount of time the model takes to generate
                        a response.""")

    args = parser.parse_args()

    if not args.skip_ingest:
        clogger.info("[1/2] Building vector store.")
        build_vectorstore(args.doc_dir, vector_store_dir, args.embedding_model,
                        args.chunk_size, args.chunk_overlap)
    else:
        clogger.info("[1/2] Skipping ingest as requested.")

    clogger.info("[2/2] Launching chat.")
    chat(args.embedding_model, args.model, args.reasoning_level, args.knowledge_cutoff,
         vector_store_dir, args.k_nearest, args.max_new_tokens)
