from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

from ingest import LocalEmbeddingFunction
from transformers import pipeline

# --- Load Hugging Face text-generation pipeline ---
def load_llm(model_name="models/gpt-oss-20b"):
    return pipeline(
        "text-generation",
        model=model_name,
        torch_dtype="auto",
        device_map="auto",
    )

# --- Generate text using the pipeline ---
def generate_response(pipe, prompt, max_new_tokens=256):
    outputs = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # deterministic for consistent answers
    )
    return outputs[0]["generated_text"]

def main():
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

    pipe = load_llm()

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        # Retrieve top-k relevant documents
        context_docs = retriever.get_relevant_documents(query)

        # Label each context snippet with its source
        context_text = "\n".join(
            [f"[{doc.metadata.get('source', 'Context')}] {doc.page_content}" for doc in context_docs]
        )

        # Fill prompt
        prompt_text = prompt_template.format(question=query, context=context_text)

        # Generate answer
        answer = generate_response(pipe, prompt_text, max_new_tokens=512)

        # Remove thought
        if "assistantfinal" in answer:
            thought, answer = answer.split("assistantfinal")

        # Display results
        sources = [doc.metadata.get("source", "Context") for doc in context_docs]
        print("\nAnswer:", answer)
        print("Sources:", sources)

if __name__ == "__main__":
    main()

