"""Auto-generate docstrings for Python in a given repository using an LLM.

This module provides an AutoDoc class that can traverse a repository,
generate docstrings for functions/classes using an LLM, and add a
module-level docstring per file using harmony-style prompts.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from transformers import pipeline

from utils import clogger, download_model


HARMONY_FUNC_TEMPLATE = '''
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: {knowledge_cutoff}
Current date: {current_date}

Reasoning: {reasoning_level}

# Valid channels: analysis, commentary, final.
# Channel must be included for every message.<|end|>

<|start|>developer<|message|># Instructions

Use the context below to complete the task as concisely as possible.

For Python docstring tasks (functions/classes):
- Write clear, PEP 257–style docstrings.
- One-line summary, then a blank line, then details (Args/Returns/Raises/Examples if useful).
- Use type hints when appropriate in the text.
- Use triple double quotes (""").
- **Return only the docstring content (without the surrounding quotes)**.

# Tools (none required for this task)

<|end|><|start|>user<|message|>Context:
{context}

Question:
{question}

Answer (docstring content only):<|end|>
'''.strip()


HARMONY_MODULE_TEMPLATE = '''
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: {knowledge_cutoff}
Current date: {current_date}

Reasoning: {reasoning_level}

# Valid channels: analysis, commentary, final.
# Channel must be included for every message.<|end|>

<|start|>developer<|message|># Instructions

Use the context below to write a **module-level** docstring.

For Python module docstrings:
- Give a concise overview of what the module provides.
- Mention key classes/functions and notable behaviors or side effects.
- Note any external dependencies or configuration if obvious from code.
- Keep it high-level; do not restate every symbol.
- Use triple double quotes (""").
- **Return only the docstring content (without the surrounding quotes)**.

# Tools (none required for this task)

<|end|><|start|>user<|message|>Context:
{context}

Question:
{question}

Answer (docstring content only):<|end|>
'''.strip()


@dataclass
class GenConfig:
    model_name: str
    max_new_tokens: int = 1024
    reasoning_level: str = "low"
    knowledge_cutoff: str = "2024-06"
    current_date: str = date.today().isoformat()


class AutoDoc:
    """Encapsulates the auto-documentation workflow."""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 1024,
        reasoning_level: str = "low",
        knowledge_cutoff: str = "2024-06",
        current_date: Optional[str] = None,
    ) -> None:
        """Initialize the AutoDoc engine and load the model pipeline."""
        self.config = GenConfig(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            reasoning_level=reasoning_level,
            knowledge_cutoff=knowledge_cutoff,
            current_date=current_date or date.today().isoformat(),
        )
        self.pipe = self._load_llm(self.config.model_name)

    @staticmethod
    def _load_llm(model_name) -> pipeline:
        """Load Hugging Face text-generation pipeline."""
        model_name = download_model(model_name)
        return pipeline(
            "text-generation",
            model=str(model_name),
            torch_dtype="auto",
            device_map="auto",
        )

    def _build_prompt(self, *, template: str, context: str, question: str) -> str:
        """Fill a harmony-style template with run-time metadata, context, and question."""
        return template.format(
            knowledge_cutoff=self.config.knowledge_cutoff,
            current_date=self.config.current_date,
            reasoning_level=self.config.reasoning_level,
            context=context,
            question=question,
        )

    def _generate(self, prompt: str) -> str:
        # Use the pipeline to generate only the continuation (exclude the prompt)
        outputs = self.pipe(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            return_full_text=False,
            eos_token_id=getattr(self.pipe.tokenizer, "eos_token_id", None),
            pad_token_id=getattr(self.pipe.tokenizer, "eos_token_id", None),
        )
        # HF text-generation returns a list of dicts with 'generated_text'
        text = outputs[0].get("generated_text", outputs[0].get("text", ""))
        return self._extract_final(text)

    @staticmethod
    def _extract_final(text: str) -> str:
        """
        If the model emits channel markers like 'analysis' / 'final',
        return only the final message. Otherwise, return the decoded text.
        """
        if "final" in text:
            try:
                text = text.split("final", 1)[1]
            except Exception:
                pass
        return text.strip()

    @staticmethod
    def _clean_docstring_body(s: str) -> str:
        """
        Return only the inner docstring content (no surrounding quotes),
        and collapse accidental enclosing triple quotes if included by the model.
        """
        s = s.strip()
        if '"""' in s:
            parts = s.split('"""')
            if len(parts) >= 3:
                s = parts[1].strip()
        return s.strip().strip('"').strip("'").strip()

    def generate_docstring(self, code: str, symbol_name: Optional[str] = None) -> str:
        """
        Generate a docstring body for a function/class code snippet.
        Returns only the inner text (no triple quotes).
        """
        q = (
            f'Write a complete Python docstring for the following '
            f'{"object" if not symbol_name else symbol_name}. '
            f'Return only the docstring content (no quotes).'
        )
        prompt = self._build_prompt(
            template=HARMONY_FUNC_TEMPLATE,
            context=code,
            question=q,
        )
        raw = self._generate(prompt)
        return self._clean_docstring_body(raw)

    def generate_module_docstring(self, full_source: str, filename: str | None = None) -> str:
        """
        Generate the *module-level* docstring body from the entire file.
        Returns only the inner text (no triple quotes).
        """
        fname = filename or "module.py"
        q = (
            f"Write a concise, PEP 257–style module docstring summarizing {fname}: "
            f"what it provides, key components, and noteworthy behavior. "
            f"Return only the docstring content (no quotes)."
        )
        prompt = self._build_prompt(
            template=HARMONY_MODULE_TEMPLATE,
            context=full_source,
            question=q,
        )
        raw = self._generate(prompt)
        body = self._clean_docstring_body(raw)
        return body if body else f"{fname} module."

    @staticmethod
    def extract_functions_classes(file_path: Path):
        """Return list of tuples (node, node_type, start_lineno, end_lineno) plus file source."""
        with Path.open(file_path, encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        results: list[tuple[ast.AST, str, int, int]] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                results.append((node, "function", node.lineno, node.end_lineno))
            elif isinstance(node, ast.ClassDef):
                results.append((node, "class", node.lineno, node.end_lineno))

        return results, source

    def add_module_docstring(self, file_path: Path) -> bool:
        """
        Ensure a module-level docstring exists.

        If missing, generate one from the entire file content and insert it after any shebang
        and/or encoding cookie. Returns True if a docstring was inserted, else False.
        """
        try:
            source = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            source = file_path.read_text(errors="ignore")

        # If the file can't be parsed, don't attempt modification.
        try:
            tree = ast.parse(source)
        except SyntaxError:
            clogger.info(f"Skipping module docstring for {file_path.name}: syntax error.")
            return False

        if ast.get_docstring(tree) is not None:
            return False  # Already has a module docstring.

        # Generate the docstring body from the full source.
        body = self.generate_module_docstring(source, file_path.name).strip()
        body = self._clean_docstring_body(body)

        doc_block = f'"""\n{body}\n"""\n\n'

        lines = source.splitlines()
        insert_idx = 0

        # Shebang on first line
        if lines and lines[0].startswith("#!"):
            insert_idx = 1

        # Encoding cookie can be on line 0 or 1 (PEP 263)
        enc_re = re.compile(r'^[ \t\f]*#.*coding[:=][ \t]*([-_.a-zA-Z0-9]+)')
        for idx in (0, 1):
            if idx < len(lines) and enc_re.match(lines[idx]):
                insert_idx = max(insert_idx, idx + 1)

        # Insert and write back
        lines.insert(insert_idx, doc_block)
        file_path.write_text("\n".join(lines), encoding="utf-8")
        clogger.info(f'Inserted module docstring into {file_path.name}')
        return True

    def update_file_with_docstrings(self, file_path: Path) -> None:
        """Update a single Python file with generated docstrings (logic matches original)."""
        results, source = self.extract_functions_classes(file_path)
        lines = source.split("\n")
        offset = 0  # Line offset due to inserted docstrings

        for node, node_type, start, end in tqdm(results, desc="Documenting functions", colour="green"):
            if ast.get_docstring(node):
                continue  # Skip if docstring already exists

            # Extract code snippet of the function/class from the current lines buffer
            snippet_lines = lines[start - 1 + offset : end + offset]
            code_snippet = "\n".join(snippet_lines)

            # Generate docstring body using the function/class prompt
            generated_body = self.generate_docstring(
                code_snippet,
                symbol_name=getattr(node, "name", None),
            )

            # Insert docstring after the signature line, indented properly
            indent = " " * (node.col_offset + 4)
            doc_lines = [f'{indent}"""' + generated_body.strip() + '"""']
            lines.insert(start + offset, "\n".join(doc_lines))

            # Adjust offset by +1 logical insertion
            offset += 1

        # Write back updated file
        with Path.open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        clogger.info(f"Updated doc strings of {Path(file_path).name}")

    def process_repository(self, repo_path: str | Path) -> None:
        """Process all Python files in a repository, add module + function/class docstrings."""
        repo_path = Path(repo_path)
        py_files = list(repo_path.rglob("*.py"))

        for file_path in py_files:
            clogger.info(f"Documenting {Path(file_path).name}")

            # First ensure the module has a docstring (uses full file context).
            self.add_module_docstring(file_path)

            # Then fill in any function/class docstrings.
            self.update_file_with_docstrings(file_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-document Python repo using an LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("repo_path", type=str, help="Path to the Python repository.")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-20b",
        help="LLM model to use.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help=(
            "Amount of characters the model can make when generating a response. "
            "This includes context and analysis of the problem."
        ),
    )
    parser.add_argument(
        "--reasoning_level",
        type=str,
        default="low",
        help="high, medium, or low (controls verbosity/steps in the prompt).",
    )
    parser.add_argument(
        "--knowledge_cutoff",
        type=str,
        default="2024-06",
        help="How far back in time the model can think (used in the prompt).",
    )

    args = parser.parse_args()
    AutoDoc(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        reasoning_level=args.reasoning_level,
        knowledge_cutoff=args.knowledge_cutoff,
    ).process_repository(args.repo_path)
