#!/usr/bin/env python3
import argparse
import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

DEFAULT_TARGETS = [
    "src/core/main.py",
    "src/core/ingest.py",
    "src/core/auto_doc.py",
    "src/utils.py",
]

def as_module_from_path(path: Path):
    """Import a module from a filesystem path
    (without adding it to sys.modules permanently).
    """
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
        # Try dotted module
        mod = importlib.import_module(target)
        return True, f"Imported module: {mod.__name__}"
    except Exception as e:
        return False, f"Import failed: {e.__class__.__name__}: {e}"

def try_help(target: str, timeout: int = 10) -> Tuple[bool, str]:
    """Try to run `<script> --help` or `python -m <module> --help` for dotted names.
    Accepts exit codes 0 or 2 (argparse often returns 2 for parse errors).
    """
    try:
        p = Path(target)
        env = os.environ.copy()
        # Mark this as a smoke run so your scripts can choose
        # to short-circuit if they want.
        env.setdefault("CGUL2_SMOKE_TEST", "1")

        if p.exists():
            cmd = [sys.executable, str(p), "--help"]
        else:
            # dotted module
            cmd = [sys.executable, "-m", target, "--help"]

        proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              env=env, timeout=timeout)
        code = proc.returncode
        if code in (0, 2):
            return True, f"Help ran (exit={code})."
        return False, f"Help returned non-OK exit code {code}.\nSTDERR:" \
            "\n{proc.stderr.decode(errors='ignore')}"
    except FileNotFoundError as e:
        return False, f"Not found: {e}"
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout}s."
    except Exception as e:
        return False, f"Help run failed: {e.__class__.__name__}: {e}"

def main(argv: List[str] = None) -> int:
    """Test main scripts."""
    parser = argparse.ArgumentParser(description="Ultra-light smoke tester: ensure scripts import " \
                                     "and optionally --help without crashing.")
    parser.add_argument("targets", nargs="*", default=DEFAULT_TARGETS,
                        help="Files or dotted modules to test.")
    parser.add_argument("--call-help", action="store_true",
                        help="Also run each target with --help via a subprocess.")
    parser.add_argument("--timeout", type=int, default=10,
                        help="Timeout for each --help call (seconds).")
    args = parser.parse_args(argv)

    print("== CGUL2 Smoke Test ==")
    print("Targets:", *args.targets, sep="\n  - ")
    print()

    failures = []

    for tgt in args.targets:
        print(f"[{tgt}] Import check...", end=" ")
        ok, msg = try_import(tgt)
        print("OK" if ok else "FAIL")
        print("   ", msg)
        if not ok:
            failures.append((tgt, f"import: {msg}"))

        if args.call_help:
            print(f"[{tgt}] --help check...", end=" ")
            ok, msg = try_help(tgt, timeout=args.timeout)
            print("OK" if ok else "FAIL")
            print("   ", msg)
            if not ok:
                failures.append((tgt, f"--help: {msg}"))

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
