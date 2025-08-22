"""Auto-generate docstrings for Python in a given repository using an LLM.

This script holds all the functions and ability to run an LLM on
a repo of python files and auto documents all methods, functions,
classes, and etc. so that you don't have too!
"""

import ast
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm
from transformers import GenerationConfig, pipeline

from utils import download_model, clogger


def load_llm(model_name: str = "openai/gpt-oss-20b") -> pipeline:
    """Load Hugging Face text-generation pipeline for docstring generation."""
    model_name = download_model(model_name)
    return pipeline(
        "text-generation",
        model=str(model_name),
        torch_dtype="auto",
        device_map="auto",
    )


def generate_docstring(pipe: pipeline, signature: str, code: str, max_new_tokens: int,
                       knowledge_cutoff: str, reasoning_level: str) -> str:
    """Generate a docstring for a function/method/class."""
    if reasoning_level not in ["high", "medium", "low"]:
        exception_str = f"""Reasing level must be high, medium, or low.
        Not '{reasoning_level}'."""
        raise ValueError(exception_str)

    current_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    prompt = f"""
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: {knowledge_cutoff}
Current date: {current_date}

Reasoning: {reasoning_level}

# Valid channels: analysis, commentary, final.
# Channel must be included for every message.<|end|>

<|start|>developer<|message|># Instructions

Generate a concise, well-structured Python docstring for the given function or class.
- Summarize what the code does.
- Clearly describe parameters and return values if applicable.
- Keep it concise and in standard Python docstring style.

# Tools (none required for this task)

<|end|><|start|>user<|message|>Signature:
{signature}

Code:
{code}

Docstring:<|end|>
"""
    gen_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    output = pipe(prompt, generation_config=gen_config)
    output = output[0]["generated_text"].strip()
    if "assistantfinal" in output:
        answer = output.split("assistantfinal")[-1]
    elif "final" in output:
        answer = output.split("final")[-1]
    return answer


def extract_functions_classes(file_path: Path) -> tuple:
    """Return list of tuples (node, node_type, start_lineno, end_lineno)."""
    with Path.open(file_path, encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    results = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            results.append((node, "function", node.lineno, node.end_lineno))
        elif isinstance(node, ast.ClassDef):
            results.append((node, "class", node.lineno, node.end_lineno))

    return results, source


def update_file_with_docstrings(file_path: Path, pipe: pipeline, max_new_tokens: int,
                                knowledge_cutoff: str, reasoning_level: str) -> None:
    """Update a single Python file with generated docstrings."""
    results, source = extract_functions_classes(file_path)
    lines = source.split("\n")
    offset = 0  # Line offset due to inserted docstrings

    for node, node_type, start, end in tqdm(results, desc="Documenting functions",
                                            colour="green"):
        docstring = ast.get_docstring(node)
        if docstring:
            continue  # Skip if docstring already exists

        # Extract code snippet of the function/class
        snippet_lines = lines[start - 1 + offset:end + offset]
        code_snippet = "\n".join(snippet_lines)

        # Build signature string
        if node_type == "function":
            to_join = [arg.arg for arg in node.args.args]
            signature = f"def {node.name}({', '.join(to_join)})"
        elif node_type == "class":
            signature = f"class {node.name}"
        else:
            signature = node.name

        # Generate docstring
        generated_doc = generate_docstring(
            pipe, signature, code_snippet, max_new_tokens,
            knowledge_cutoff, reasoning_level,
        )
        for quote in ('"""', "'''"):
            if generated_doc.startswith(quote) and generated_doc.endswith(quote):
                generated_doc = generated_doc[len(quote):-len(quote)].strip()
                break

        # Insert docstring
        indent = " " * (node.col_offset + 4)
        doc_lines = [f'{indent}"""' + generated_doc + '"""']
        lines.insert(start + offset, "\n".join(doc_lines))
        offset += len(doc_lines)

    # Write back updated file
    with Path.open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def process_repository(repo_path: str, model_name: str, max_new_tokens: int,
                       knowledge_cutoff: str, reasoning_level: str) -> None:
    """Process all Python files in a repository and add docstrings."""
    pipe = load_llm(model_name)
    py_files = list(Path(repo_path).rglob("*.py"))

    for file_path in py_files:
        clogger.info(f"Documenting {Path(file_path).name}")
        update_file_with_docstrings(file_path, pipe, max_new_tokens, knowledge_cutoff,
                                    reasoning_level)
        clogger.info(f"Updated doc strings of {Path(file_path).name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-document Python repo using an LLM.",
    )
    parser.add_argument("repo_path", type=str,
                        help="Path to the Python repository.")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b",
                        help="LLM model to use.")
    parser.add_argument("--reasoning_level", type=str, default="high",
                        help="""Level of reasoning to use when responding.
                        Must be high, medium, or low.""")
    parser.add_argument("--knowledge_cutoff", type=str, default="2024-06",
                        help="How far back in time the model can think.")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="""Amount of characters the model can make when
                        generating a response. This includes context and analysis
                        of the problem.""")

    args = parser.parse_args()
    process_repository(args.repo_path, args.model, args.max_new_tokens,
                       args.knowledge_cutoff, args.reasoning_level)
