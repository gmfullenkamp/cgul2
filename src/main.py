"""Main for running LLM doc citing.

This script runs ChatGPT4 locally and uses a vector store
of given documents to create citations when answering questions.
"""

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from transformers import GenerationConfig, pipeline

from ingest import LocalEmbeddingFunction


def load_llm(model_name: str = "openai/gpt-oss-20b") -> pipeline:
    """Load Hugging Face text-generation pipeline."""
    return pipeline(
        "text-generation",
        model=model_name,
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

def main() -> None:
    """Talk with LLM that cites documents."""
    # Load vector store
    embeddings = LocalEmbeddingFunction()
    db = FAISS.load_local("data", embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_kwargs={"k": 1})

    # Prompt template for concise, citation-aware answers
    prompt_template = PromptTemplate(
        template="""
Use the context below to answer the question in exactly one sentence.
Include a citation in parentheses indicating the source of the information.

Context:
{context}

Question:
{question}

Answer (one sentence, include citation):
""",
        input_variables=["context", "question"],
    )

    # ✅ Use chat gpt 4 directly (local path OR HF ID)
    model_path = "models/gpt-oss-20b"  # <-- set this to local folder if offline
    model_path = "openai/gpt-oss-20b"
    pipe = load_llm(model_path)

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        # Retrieve top-k relevant documents
        context_docs = retriever.invoke(query)

        # Label each context snippet with its source
        context_text = "\n".join(
            [f"[{doc.metadata.get('source', 'Context')}] {doc.page_content}" \
             for doc in context_docs],
        )

        # Fill prompt
        prompt_text = prompt_template.format(question=query, context=context_text)

        # Generate answer
        answer = generate_response(pipe, prompt_text, max_new_tokens=512)

        # Remove thought
        if "assistantfinal" in answer:
            thought, answer = answer.split("assistantfinal")

        # Display results
        [doc.metadata.get("source", "Context") for doc in context_docs]

if __name__ == "__main__":
    main()

