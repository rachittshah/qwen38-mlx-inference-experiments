"""Deterministic prompts for the benchmarks.

The headline bottleneck is that an agentic client (OpenCode) resends a large,
*constant* system+tools prompt every turn. We reproduce that with a synthetic
~13-15K-token system prompt made of realistic tool schemas, built deterministically
so it is byte-identical across runs (a requirement for prefix-cache / APC hits).
"""
from __future__ import annotations

# A single fake tool schema block, ~ a few hundred tokens, similar in shape to what
# agentic coding clients serialize into the system prompt.
_TOOL_TEMPLATE = """
### tool: {name}
{desc}
Parameters (JSON schema):
{{
  "type": "object",
  "properties": {{
    "path": {{"type": "string", "description": "Absolute path to the target file or directory."}},
    "content": {{"type": "string", "description": "UTF-8 payload to write, when applicable."}},
    "pattern": {{"type": "string", "description": "Regular expression or glob to match against."}},
    "recursive": {{"type": "boolean", "description": "Whether to descend into subdirectories."}},
    "max_results": {{"type": "integer", "description": "Cap on the number of results returned."}},
    "timeout_ms": {{"type": "integer", "description": "Abort the call after this many milliseconds."}}
  }},
  "required": ["path"]
}}
Usage notes: {desc} Always prefer the dedicated tool over a raw shell command. Validate
inputs, quote paths, and never operate outside the working directory without confirmation.
"""

_TOOL_NAMES = [
    ("read_file", "Read the contents of a file from the workspace."),
    ("write_file", "Create or overwrite a file with new contents."),
    ("edit_file", "Apply an exact string replacement to an existing file."),
    ("list_dir", "List the entries of a directory."),
    ("grep", "Search file contents by regular expression."),
    ("glob", "Find files by glob pattern."),
    ("run_bash", "Execute a shell command in the workspace."),
    ("git_status", "Show the working-tree status."),
    ("git_diff", "Show the staged or unstaged diff."),
    ("git_commit", "Create a commit with the given message."),
    ("web_fetch", "Fetch a URL and return its readable content."),
    ("web_search", "Search the web and return ranked results."),
    ("todo_write", "Record or update the task list."),
    ("run_tests", "Run the project's test suite and report results."),
    ("format_code", "Run the project's formatter over changed files."),
    ("type_check", "Run the project's static type checker."),
]

_PREAMBLE = """You are an autonomous software-engineering agent operating in a terminal.
Follow the user's instructions precisely, prefer dedicated tools over raw shell, keep edits
minimal and reversible, and confirm before any irreversible or outward-facing action. Below
is the full catalog of tools available to you, each with its JSON schema and usage notes.
Read them carefully; you must call tools by their exact names and respect their required
parameters. Think step by step, but keep your visible output concise.
"""


def opencode_system_prompt(target_tokens: int = 15000) -> str:
    """Build a constant, ~target_tokens system prompt out of repeated tool schemas.

    Roughly 4 chars/token, so we size by characters and let the tokenizer land near
    the target. The result is deterministic (same bytes every call).
    """
    approx_chars = target_tokens * 4
    parts = [_PREAMBLE]
    i = 0
    while sum(len(p) for p in parts) < approx_chars:
        name, desc = _TOOL_NAMES[i % len(_TOOL_NAMES)]
        # Suffix the name so repeats stay plausible and the block stays constant.
        parts.append(_TOOL_TEMPLATE.format(name=f"{name}_v{i // len(_TOOL_NAMES)}", desc=desc))
        i += 1
    return "".join(parts)


# A few user turns of the sort an agent loop produces, kept short so decode length
# (not prompt length) is controlled by max_tokens.
USER_TURNS = [
    "List the Python files in the repo and tell me which one defines the entrypoint.",
    "Now add a docstring to the main() function explaining what it does.",
    "Run the tests and summarize any failures in one sentence.",
]

# A short, prompt-independent question for pure decode-speed / speculative tests.
SHORT_PROMPT = "Write a Python function that returns the nth Fibonacci number iteratively."
