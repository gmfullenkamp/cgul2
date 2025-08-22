"""Main for running LLM doc citing.

This script runs ChatGPT4 locally and uses a vector store
of given documents to create citations when answering questions.
"""

from datetime import date
import time

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
from transformers import GenerationConfig, pipeline

from constants import vector_store_dir
from ingest import LocalEmbeddingFunction
from utils import download_model


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
        raise ValueError(f"Reasing level must be high, medium, or low. Not '{reasoning_level}'.")

    current_date = date.today().strftime("%Y-%m-%d")

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

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>

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
        template=template,  # using harmony format as that is what openai used for prompt training
        input_variables=["knowledge_cutoff", "current_date", "reasoning_level", "context", "question"],
    )

    pipe = load_llm(model)

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        # Measure reference time
        start_time = time.time()
        with tqdm(total=1, desc="Referencing...", bar_format="{desc} {bar}") as pbar:
            # Retrieve top-k relevant documents
            context_docs = retriever.invoke(query)
            pbar.update(1)
        elapsed_time = time.time() - start_time
        print(f"[Reference Time: {elapsed_time:.2f} seconds]")  # noqa: T201

        # Label each context snippet with its source
        context_text = "\n".join(
            [f"[{doc.metadata.get('source', 'Context')}] {doc.page_content}" \
             for doc in context_docs],
        )

        # Fill prompt
        prompt_text = prompt_template.format(question=query, context=context_text,
                                             reasoning_level=reasoning_level, knowledge_cutoff=knowledge_cutoff,
                                             current_date=current_date)

        # Measure thought time
        start_time = time.time()
        with tqdm(total=1, desc="Thinking...", bar_format="{desc} {bar}") as pbar:
            # Generate answer
            answer = generate_response(pipe, prompt_text, max_new_tokens=max_new_tokens)
            pbar.update(1)
        elapsed_time = time.time() - start_time
        print(f"[Thought Time: {elapsed_time:.2f} seconds]")  # noqa: T201

        # Remove prompt and thought
        if "assistantanalysis" in answer:
            prompt, answer = answer.split("assistantanalysis")
        if "assistantfinal" in answer:
            thought, answer = answer.split("assistantfinal")

        # Display results
        sources = [doc.metadata.get("source", "Context") for doc in context_docs]

        print("\nAnswer:", answer)  # noqa: T201
        print("Sources:", sources)  # noqa: T201

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-document Python repo using an LLM.")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b",
                        help="Local LLM model to use.")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model to use.")
    parser.add_argument("--reasoning_level", type=str, default="high",
                        help="Level of reasoning to use when responding. Must be high, medium, or low.")
    parser.add_argument("--knowledge_cutoff", type=str, default="2024-06",
                        help="How far back in time the model can think.")
    parser.add_argument("--k_nearest", type=int, default=1, help="Number of documents to reference.")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Amount of characters the model can make when generating a response. " \
                            "This includes context and analysis of the problem.")

    args = parser.parse_args()
    chat(args.embedding_model, args.model, args.reasoning_level, args.knowledge_cutoff, vector_store_dir,
         args.k_nearest, args.max_new_tokens)
