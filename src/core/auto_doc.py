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
def add(a: int = 5, b: int = 7):
    return a + b

# docstring
"""
Adds together two numbers a and b.

Parameters:
    a (int): the first number to add (default: 5)
    b (int): the second number to add (default: 7)

Returns:
    int: the result of a + b
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
    clogger.info(f"Updated doc strings of {Path(file_path).name}")


def summarize_python_file(p: Path) -> tuple:
    """
    Return a summary tuple:
      (module_name, module_doc, classes[(name, doc)], functions[(name, doc)])
    """
    try:
        src = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        src = p.read_text(errors="ignore")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return (p.stem, None, [], [])

    module_doc = ast.get_docstring(tree)
    classes = []
    functions = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append((node.name, ast.get_docstring(node)))
        elif isinstance(node, ast.FunctionDef):
            functions.append((node.name, ast.get_docstring(node)))

    return (p.stem, module_doc, classes, functions)


def create_readme_for_folder(folder: Path, py_files: list, model, tokenizer,
                             max_new_tokens: int) -> None:
    """
    Creates or overwrites folder/README.md with a structured summary of the folder’s Python files.
    Uses lightweight AST parsing; you can optionally plug LLM formatting in the marked section.
    """
    if not py_files:
        return
    readme_path = folder / "README.md"
    if readme_path.exists():
        clogger.info(f"README.md already exists in {folder}; skipping")
        return

    # Gather structured summaries
    summaries = [summarize_python_file(p) for p in sorted(py_files)]

    # ---- Optional: LLM-enhanced intro (plug your model in here if desired) ----
    # If you want to use your `model` / `tokenizer` to produce a higher-level overview,
    # build a prompt from `summaries` and generate a paragraph. For now we keep it deterministic.
    folder_title = folder.name if folder.name != "" else "Repository"
    intro = (
        f"# {folder_title}\n\n"
        f"This folder contains the following Python modules. "
        f"Auto-generated README based on module/class/function docstrings.\n\n"
    )

    # Build markdown content
    lines = [intro, "## Contents\n"]
    for mod_name, mod_doc, classes, functions in summaries:
        lines.append(f"### `{mod_name}.py`")
        if mod_doc:
            lines.append(f"{mod_doc.strip()}\n")
        else:
            lines.append("_No module docstring found._\n")

        if classes:
            lines.append("**Classes**")
            for cname, cdoc in classes:
                desc = cdoc.strip().splitlines()[0] if cdoc else "No class docstring."
                lines.append(f"- `{cname}` — {desc}")
            lines.append("")  # spacer

        if functions:
            lines.append("**Functions**")
            for fname, fdoc in functions:
                desc = fdoc.strip().splitlines()[0] if fdoc else "No function docstring."
                lines.append(f"- `{fname}()` — {desc}")
            lines.append("")  # spacer

    # TODO: Implement model and tokenizer with file summaries for better readme

    # Write README.md
    readme_path = folder / "README.md"
    readme_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    clogger.info(f"Created README.md in {folder}")


def process_repository(repo_path: str, model_name: str, max_new_tokens: int) -> None:
    """Process all Python files in a repository and add docstrings, then create README.md in each folder."""
    model, tokenizer = load_llm(model_name)
    py_files = list(Path(repo_path).rglob("*.py"))

    for file_path in py_files:
        clogger.info(f"Documenting {Path(file_path).name}")
        update_file_with_docstrings(file_path, model, tokenizer, max_new_tokens)

    # Unique directories that contain at least one .py
    folders_with_py = sorted({p.parent for p in py_files})

    for folder in folders_with_py:
        # Collect only the .py files directly inside this folder (not recursively)
        immediate_py_files = sorted([p for p in folder.glob("*.py") if p.is_file()])
        if not immediate_py_files:
            continue

        clogger.info(f"Creating README.md for {folder}")
        create_readme_for_folder(
            folder=folder,
            py_files=immediate_py_files,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )


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
