#!/usr/bin/env python3
import argparse
import importlib
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

DEFAULT_TARGETS = [
    "src/core/main.py",
    "src/core/auto_doc.py",
    "src/utils.py",
]


def as_module_from_path(path: Path):
    """Import a module from a filesystem path (no permanent sys.modules entry)."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def try_import(target: str) -> Tuple[bool, str]:
    """Try to import a module or file path. Returns (ok, message)."""
    try:
        p = Path(target)
        if p.exists():
            as_module_from_path(p)
            return True, f"Imported from path: {p}"
        mod = importlib.import_module(target)
        return True, f"Imported module: {mod.__name__}"
    except Exception as e:
        return False, f"Import failed: {e.__class__.__name__}: {e}"


def _run_cmd(cmd: list[str], *, timeout: int, env: dict | None = None, cwd: Path | None = None) -> Tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=env,
        cwd=str(cwd) if cwd else None,
    )
    return proc.returncode, proc.stdout.decode(errors="ignore"), proc.stderr.decode(errors="ignore")


def try_help(target: str, timeout: int = 10) -> Tuple[bool, str]:
    """Run `<script> --help` (or `-m module --help`). Exit code 0 or 2 is OK."""
    try:
        p = Path(target)
        env = os.environ.copy()
        env.setdefault("CGUL2_SMOKE_TEST", "1")
        cmd = [sys.executable, str(p), "--help"] if p.exists() else [sys.executable, "-m", target, "--help"]
        code, out, err = _run_cmd(cmd, timeout=timeout, env=env)
        if code in (0, 2):
            return True, f"Help ran (exit={code})."
        return False, f"Help returned {code}\nSTDERR:\n{err}"
    except Exception as e:
        return False, f"Help run failed: {e.__class__.__name__}: {e}"


def _prep_docs_dir() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="cgul2_docs_"))
    (tmp / "a.txt").write_text("alpha beta gamma\n" * 5, encoding="utf-8")
    (tmp / "b.md").write_text("# heading\n" + ("content\n" * 5), encoding="utf-8")
    return tmp


def try_run(target: str, timeout: int = 30) -> Tuple[bool, str]:
    """
    Run a target end-to-end.

    Special handling:
      - ingest.py: create temp docs and try --doc-dir / --docs; fallback to creating ./docs if flags unsupported.
      - auto_doc.py: create temp repo with a .py and pass it as positional argument.
      - others: run with no args.
    """
    p = Path(target)
    env = os.environ.copy()
    env.setdefault("CGUL2_SMOKE_TEST", "1")

    # Helper to build base command
    def base_cmd():
        return [sys.executable, str(p)] if p.exists() else [sys.executable, "-m", target]

    # Strategy by basename
    name = p.name if p.exists() else target.split(".")[-1]  # crude but fine for smoke

    try:
        # ingest.py: supply docs
        if "ingest" in name:
            docs = _prep_docs_dir()

            # Try common flags
            for flag in ("--doc-dir", "--docs"):
                cmd = base_cmd() + [flag, str(docs)]
                code, out, err = _run_cmd(cmd, timeout=timeout, env=env)
                if code == 0:
                    return True, f"Ran successfully with {flag}={docs} (exit=0)."
            # Fallback: some scripts read ./docs; create one in CWD and run
            cwd_docs = Path.cwd() / "docs"
            try:
                cwd_docs.mkdir(exist_ok=True, parents=True)
                (cwd_docs / "smoke.txt").write_text("smoke test content\n", encoding="utf-8")
                cmd = base_cmd()
                code, out, err = _run_cmd(cmd, timeout=timeout, env=env)
                if code == 0:
                    return True, "Ran successfully with ./docs fallback (exit=0)."
                return False, f"Run returned {code}\nSTDERR:\n{err}"
            finally:
                # Clean only the file we created; keep dir if user had content
                try:
                    (cwd_docs / "smoke.txt").unlink(missing_ok=True)
                    # do not rmdir to avoid nuking user's real docs folder
                except Exception:
                    pass

        # auto_doc.py: needs a repo_path positional
        if "auto_doc" in name:
            repo = Path(tempfile.mkdtemp(prefix="cgul2_repo_"))
            sample = repo / "sample.py"
            sample.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            cmd = base_cmd() + [str(repo)]
            code, out, err = _run_cmd(cmd, timeout=timeout, env=env)
            if code == 0:
                return True, f"Ran successfully with repo_path={repo} (exit=0)."
            return False, f"Run returned {code}\nSTDERR:\n{err}"

        # main.py & others: basic run
        cmd = base_cmd()
        code, out, err = _run_cmd(cmd, timeout=timeout, env=env)
        if code == 0:
            return True, "Ran successfully (exit=0)."
        return False, f"Run returned {code}\nSTDERR:\n{err}"

    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout}s."
    except Exception as e:
        return False, f"Run failed: {e.__class__.__name__}: {e}"


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke tester: ensure scripts import, --help, and run.")
    parser.add_argument("targets", nargs="*", default=DEFAULT_TARGETS,
                        help="Files or dotted modules to test.")
    parser.add_argument("--skip-import", action="store_true", help="Skip import check.")
    parser.add_argument("--skip-help", action="store_true", help="Skip --help check.")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout for full run (seconds).")
    parser.add_argument("--help-timeout", type=int, default=10, help="Timeout for --help run (seconds).")
    args = parser.parse_args(argv)

    print("== Smoke Test ==")
    print("Targets:", *args.targets, sep="\n  - ")
    print()

    failures = []

    for tgt in args.targets:
        if not args.skip_import:
            print(f"[{tgt}] Import check...", end=" ")
            ok, msg = try_import(tgt)
            print("OK" if ok else "FAIL")
            print("   ", msg)
            if not ok:
                failures.append((tgt, f"import: {msg}"))

        if not args.skip_help:
            print(f"[{tgt}] --help check...", end=" ")
            ok, msg = try_help(tgt, timeout=args.help_timeout)
            print("OK" if ok else "FAIL")
            print("   ", msg)
            if not ok:
                failures.append((tgt, f"--help: {msg}"))

        print(f"[{tgt}] Full run...", end=" ")
        ok, msg = try_run(tgt, timeout=args.timeout)
        print("OK" if ok else "FAIL")
        print("   ", msg)
        if not ok:
            failures.append((tgt, f"run: {msg}"))
        print()

    if failures:
        print("== FAILURES ==")
        for tgt, reason in failures:
            print(f"- {tgt}: {reason}")
        return 1

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
