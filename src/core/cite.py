"""Langextract integration helpers.
"""

from pathlib import Path
import os
import textwrap
from typing import Any, Dict, List, Tuple

try:
    import langextract as lx
    _LANGEXTRACT_OK = True
except Exception as _e:
    lx = None  # type: ignore
    _LANGEXTRACT_OK = False

def _langextract_example() -> Any:
    """Few-shot example that nudges LangExtract to return short, verbatim spans."""
    if not _LANGEXTRACT_OK:
        return None
    example_text = (
        "TEXT: The Eiffel Tower is located in Paris, France.\n"
        "QUESTION: Where is the Eiffel Tower?"
    )
    return lx.data.ExampleData(
        text=example_text,
        extractions=[
            lx.data.Extraction(
                extraction_class="answer_span",
                extraction_text="Paris, France",
                attributes={"why": "direct answer"},
            )
        ],
    )

def run_langextract_on_docs(
    question: str,
    docs: List[Any],
    model_id: str = "gemini-2.5-flash",
    provider_is_openai: bool = False,
    save_visualization_to: str | None = None,
) -> Dict[str, Any]:
    """
    Run LangExtract over retrieved chunks to find *minimal, verbatim* evidence spans.

    Returns:
      {
        "quotes": [{"text": str, "source": str, "page": Optional[int]}...],
        "raw_results": <library result objects>
      }
    """
    if not _LANGEXTRACT_OK:
        raise RuntimeError(
            "LangExtract is not installed. Run: "
            "pip install langextract  (or)  pip install 'langextract[openai]'"
        )

    if not docs:
        return {"quotes": [], "raw_results": []}

    prompt_desc = textwrap.dedent(f"""
        You are an EVIDENCE EXTRACTOR.
        Given a TEXT and a QUESTION, extract the minimal verbatim span(s) from TEXT
        that directly answer the QUESTION. Use exact quotes from TEXT; do not paraphrase.
        Prefer short spans (phrase or single sentence). If no answer is present, extract nothing.

        QUESTION: {question}
    """)

    example = _langextract_example()
    results = []
    quotes: List[Dict[str, Any]] = []

    for d in docs:
        res = lx.extract(
            text_or_documents=d.page_content,
            prompt_description=prompt_desc,
            examples=[example] if example else None,
            model_id=model_id,
            # OpenAI models benefit from fenced output and relaxed schema:
            fence_output=True if provider_is_openai else False,
            use_schema_constraints=False if provider_is_openai else True,
        )
        results.append(res)

        # Collect extracted spans and map them back to file + page via loader metadata.
        for item in getattr(res, "extractions", []) or []:
            span_text = getattr(item, "extraction_text", None)
            if not span_text:
                continue
            quotes.append({
                "text": span_text,
                "source": d.metadata.get("source", "Context"),
                "page": d.metadata.get("page", None),
            })

    if save_visualization_to:
        os.makedirs(save_visualization_to, exist_ok=True)
        lx.io.save_annotated_documents(
            results,
            output_name="extractions.jsonl",
            output_dir=save_visualization_to,
        )
        html = lx.visualize(os.path.join(save_visualization_to, "extractions.jsonl"))
        with open(os.path.join(save_visualization_to, "evidence.html"), "w", encoding="utf-8") as f:
            f.write(getattr(html, "data", html))

    return {"quotes": quotes, "raw_results": results}

def build_grounded_context_from_quotes(
    quotes: List[Dict[str, Any]],
    max_quotes: int = 6,
) -> str:
    """
    Turn extracted quotes into a compact, labeled context block for the LLM.
    """
    if not quotes:
        return ""
    lines = []
    for q in quotes[:max_quotes]:
        src = Path(q["source"]).name if q.get("source") else "Context"
        page = f", p.{q['page']}" if q.get("page") is not None else ""
        lines.append(f'• "{q["text"]}" — {src}{page}')
    return "Grounded evidence:\n" + "\n".join(lines) + "\n"

def build_fallback_context_from_docs(docs: List[Any]) -> str:
    """
    Your original fallback: label each chunk with its source and concatenate.
    """
    if not docs:
        return ""
    return "\n".join(
        f"[{d.metadata.get('source', 'Context')}] {d.page_content}" for d in docs
    )

def apply_grounding(
    question: str,
    retrieved_docs: List[Any],
    use_langextract: bool = False,
    langextract_model: str = "gemini-2.5-flash",
    langextract_provider_openai: bool = False,
    save_evidence_html: str = "",
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Decide whether to run LangExtract and return:
      context_text, quotes

    - If LangExtract finds quotes, we provide a compact evidence section.
    - Otherwise we fall back to your original concatenated chunks.
    """
    if use_langextract:
        try:
            out = run_langextract_on_docs(
                question=question,
                docs=retrieved_docs,
                model_id=langextract_model,
                provider_is_openai=langextract_provider_openai,
                save_visualization_to=save_evidence_html or None,
            )
            quotes = out["quotes"]
            if quotes:
                return build_grounded_context_from_quotes(quotes), quotes
        except Exception as e:
            # Non-fatal: log and fall back to chunk context
            try:
                from utils import clogger  # reuse your logger if available
                clogger.warning(f"LangExtract failed; falling back to chunks. {e}")
            except Exception:
                pass

    # Fallback path
    return build_fallback_context_from_docs(retrieved_docs), []
