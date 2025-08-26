"""Auto-generate docstrings for Python in a given repository using an LLM.

This script holds all the functions and ability to run an LLM on
a repo of python files and auto documents all methods, functions,
classes, and etc. so that you don't have too!
"""

import ast
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import clogger, download_model


def load_llm(model_name: str) -> tuple:
    """Load Hugging Face text-generation pipeline for docstring generation."""
    model_name = download_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def generate_docstring(model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                       code: str, max_new_tokens: int) -> str:
    """Generate a docstring for a function/method/class."""
    prompt = f'''<|endoftext|>
def add(a, b):
    return a + b

# docstring
"""
Calculate numbers add.

Args:
    a: the first number to add
    b: the second number to add

Return:
    The result of a + b
"""
<|endoftext|>
{code}

# docstring
"""'''
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=len(prompt) + max_new_tokens,
                             do_sample=False, return_dict_in_generate=True, num_return_sequences=1,
                             output_scores=True, pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
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


def update_file_with_docstrings(file_path: Path, model: AutoModelForCausalLM,
                                tokenizer: AutoTokenizer, max_new_tokens: int) -> None:
    """Update a single Python file with generated docstrings."""
    results, source = extract_functions_classes(file_path)
    lines = source.split("\n")
    offset = 0  # Line offset due to inserted docstrings

    for node, _, start, end in tqdm(results, desc="Documenting functions",
                                    colour="green"):
        docstring = ast.get_docstring(node)
        if docstring:
            continue  # Skip if docstring already exists

        # Extract code snippet of the function/class
        snippet_lines = lines[start - 1 + offset:end + offset]
        code_snippet = "\n".join(snippet_lines)

        # Generate docstring
        generated_doc = generate_docstring(
            model, tokenizer, code_snippet, max_new_tokens,
        )
        generated_doc = generated_doc.split("# docstring")[-1].split('"""')[1]

        # Insert docstring
        indent = " " * (node.col_offset + 4)
        doc_lines = [f'{indent}"""' + generated_doc + '"""']
        lines.insert(start + offset, "\n".join(doc_lines))
        offset += len(doc_lines)

    # Write back updated file
    with Path.open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def process_repository(repo_path: str, model_name: str, max_new_tokens: int) -> None:
    """Process all Python files in a repository and add docstrings."""
    model, tokenizer = load_llm(model_name)
    py_files = list(Path(repo_path).rglob("*.py"))

    for file_path in py_files:
        clogger.info(f"Documenting {Path(file_path).name}")
        update_file_with_docstrings(file_path, model, tokenizer, max_new_tokens)
        clogger.info(f"Updated doc strings of {Path(file_path).name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-document Python repo using an LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("repo_path", type=str,
                        help="Path to the Python repository.")
    parser.add_argument("--model", type=str, default="kdf/python-docstring-generation",
                        help="LLM model to use.")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="""Amount of characters the model can make when
                        generating a response. This includes context and analysis
                        of the problem.""")

    args = parser.parse_args()
    process_repository(args.repo_path, args.model, args.max_new_tokens)
