#!/usr/bin/env python3
"""
session_logger.py  —  AGENTS.md-compliant logging helper.

AGENTS.md (§2, §5) requires an append-only log at:
    macOS/Linux : $HOME/hackerrank_orchestrate/log.txt
    Windows     : %USERPROFILE%\\hackerrank_orchestrate\\log.txt

This helper writes §5.2 per-turn entries to that path. It is cross-platform
(uses Path.home()), UTF-8 / '\\n', append-only, and never logs secrets.

Usage (one call per conversation turn you want recorded):

    python session_logger.py \
        --title "Phase 2: architecture + prompts" \
        --prompt "Show me the two-stage prompts" \
        --summary "Designed vision + fusion stages; wrote both prompt files." \
        --actions "wrote prompts/vision_prompt.txt" "wrote prompts/fusion_prompt.txt" \
        --agent "claude-chat (manual)"

You can also pipe a verbatim prompt via --prompt-file to avoid shell-escaping.
"""
import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path


def log_path() -> Path:
    return Path.home() / "hackerrank_orchestrate" / "log.txt"


def git_branch(repo_root: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def iso_now() -> str:
    # Local time with timezone offset, ISO-8601.
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True, help="Short title (<=80 chars).")
    ap.add_argument("--prompt", default="", help="Verbatim user prompt.")
    ap.add_argument("--prompt-file", help="File with the verbatim prompt (overrides --prompt).")
    ap.add_argument("--summary", required=True, help="2-5 sentence response summary.")
    ap.add_argument("--actions", nargs="*", default=[], help="Files/commands/tools touched.")
    ap.add_argument("--agent", default="claude-chat (manual)")
    ap.add_argument("--parent-agent", default="none")
    ap.add_argument("--repo-root", default=os.getcwd())
    args = ap.parse_args()

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    actions = "\n".join(f"* {a}" for a in args.actions) or "* (none)"

    entry = (
        f"\n## [{iso_now()}] {args.title[:80]}\n\n"
        f"User Prompt (verbatim, secrets redacted):\n{prompt or '(not recorded)'}\n\n"
        f"Agent Response Summary:\n{args.summary}\n\n"
        f"Actions:\n{actions}\n\n"
        f"Context:\n"
        f"tool={args.agent}\n"
        f"branch={git_branch(args.repo_root)}\n"
        f"repo_root={args.repo_root}\n"
        f"worktree=main\n"
        f"parent_agent={args.parent_agent}\n"
    )

    p = log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(entry)
    print(f"Appended log entry to {p}")


if __name__ == "__main__":
    main()
