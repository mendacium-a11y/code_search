"""
Language-aware code chunkers.
Supported: Python (AST-based), JavaScript / TypeScript (regex-based).
Each chunk is a dict: {code, function_name, start_line, end_line, language}
"""

import ast
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
}


def get_language(filepath: str) -> Optional[str]:
    return SUPPORTED_EXTENSIONS.get(Path(filepath).suffix.lower())


def _source_lines(filepath: str) -> list[str]:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()


# ---------------------------------------------------------------------------
# Python — AST based
# ---------------------------------------------------------------------------

def chunk_python(filepath: str) -> list[dict]:
    """
    Extract top-level and class-level functions/methods from a Python file.
    Uses the built-in `ast` module for precise boundary detection.
    """
    try:
        source = Path(filepath).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines()
    chunks = []

    def extract_nodes(node, parent_name=None):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{parent_name}.{child.name}" if parent_name else child.name
                start = child.lineno - 1          # 0-indexed
                # end_lineno is available in Python 3.8+
                end = getattr(child, "end_lineno", start + 30)
                code = "\n".join(lines[start:end])
                if code.strip():
                    chunks.append({
                        "code": code,
                        "function_name": name,
                        "start_line": child.lineno,
                        "end_line": end,
                        "language": "python",
                    })
            elif isinstance(child, ast.ClassDef):
                extract_nodes(child, parent_name=child.name)
            else:
                extract_nodes(child, parent_name)

    extract_nodes(tree)
    return chunks


# ---------------------------------------------------------------------------
# JavaScript / TypeScript — regex based
# ---------------------------------------------------------------------------

# Patterns to find the START of a function or class declaration.
# We capture: keyword and name, then scan forward for the matching closing brace.
_JS_PATTERNS = [
    # async function foo(...) {
    re.compile(r"^(?P<indent>\s*)(?:export\s+)?(?:default\s+)?async\s+function\s*\*?\s*(?P<name>\w+)\s*\(", re.MULTILINE),
    # function foo(...) {
    re.compile(r"^(?P<indent>\s*)(?:export\s+)?(?:default\s+)?function\s*\*?\s*(?P<name>\w+)\s*\(", re.MULTILINE),
    # const/let/var foo = (...) => {   OR   const foo = function(...) {
    re.compile(r"^(?P<indent>\s*)(?:export\s+)?(?:const|let|var)\s+(?P<name>\w+)\s*=\s*(?:async\s+)?(?:\(.*?\)\s*=>|function\s*\*?\s*(?:\w+)?\s*\()", re.MULTILINE),
    # class Foo {
    re.compile(r"^(?P<indent>\s*)(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(?P<name>\w+)", re.MULTILINE),
    # export default function(...) {   (anonymous)
    re.compile(r"^(?P<indent>\s*)export\s+default\s+(?:async\s+)?function\s*\(", re.MULTILINE),
]


def _find_block_end(lines: list[str], start_idx: int) -> int:
    """
    Starting from start_idx, scan forward and return the line index (inclusive)
    where the top-level brace block ends. Falls back gracefully.
    """
    depth = 0
    found_open = False
    for i, line in enumerate(lines[start_idx:], start=start_idx):
        for ch in line:
            if ch == "{":
                depth += 1
                found_open = True
            elif ch == "}":
                depth -= 1
        if found_open and depth <= 0:
            return i
    # If braces unbalanced, return a reasonable slice
    return min(start_idx + 60, len(lines) - 1)


def chunk_js_ts(filepath: str, language: str = "javascript") -> list[dict]:
    """
    Extract named functions, arrow functions, and classes from JS/TS files
    using regex to find declaration starts and brace-matching to find ends.
    """
    source = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    lines = source.splitlines()
    chunks = []
    covered_lines: set[int] = set()   # avoid duplicating overlapping matches

    matches = []
    for pat in _JS_PATTERNS:
        for m in pat.finditer(source):
            # Convert char offset → line index
            start_line = source[:m.start()].count("\n")
            name = m.groupdict().get("name") or "<anonymous>"
            matches.append((start_line, name))

    # Sort by line and deduplicate close starts
    matches.sort(key=lambda x: x[0])

    for start_line, name in matches:
        if start_line in covered_lines:
            continue
        end_line = _find_block_end(lines, start_line)
        # Skip if this range is already captured by a parent
        if any(l in covered_lines for l in range(start_line, min(start_line + 3, end_line))):
            continue
        code = "\n".join(lines[start_line: end_line + 1])
        if len(code.strip()) < 20:   # skip trivial stubs
            continue
        chunks.append({
            "code": code,
            "function_name": name,
            "start_line": start_line + 1,   # 1-indexed for display
            "end_line": end_line + 1,
            "language": language,
        })
        covered_lines.update(range(start_line, end_line + 1))

    return chunks


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def chunk_file(filepath: str) -> list[dict]:
    """Route a file to the correct chunker based on its extension."""
    lang = get_language(filepath)
    if lang is None:
        return []
    if lang == "python":
        return chunk_python(filepath)
    else:
        return chunk_js_ts(filepath, language=lang)
