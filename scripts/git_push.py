"""Utility to stage, commit, and push repository changes over SSH."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_MESSAGE = "auto update: portfolio credit model"


def project_root() -> Path:
    """Return the repository root based on this script location."""

    return Path(__file__).resolve().parents[1]


def run_git_command(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git command and return the completed process."""

    return subprocess.run(
        args,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=check,
    )


def print_step(message: str) -> None:
    """Print a consistent step log."""

    print(f"[git-push] {message}")


def current_remote_url(cwd: Path) -> str:
    """Return the configured origin remote URL."""

    result = run_git_command(["git", "remote", "get-url", "origin"], cwd=cwd)
    return result.stdout.strip()


def working_tree_status(cwd: Path) -> str:
    """Return git status in porcelain format."""

    result = run_git_command(["git", "status", "--short"], cwd=cwd)
    return result.stdout.strip()


def latest_commit_hash(cwd: Path) -> str:
    """Return the latest commit hash."""

    result = run_git_command(["git", "rev-parse", "HEAD"], cwd=cwd)
    return result.stdout.strip()


def branch_name(cwd: Path) -> str:
    """Return the current branch name."""

    result = run_git_command(["git", "branch", "--show-current"], cwd=cwd)
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage, commit, and push changes to origin/main over SSH.")
    parser.add_argument("--message", help="Commit message to use.")
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Append a UTC timestamp to the commit message.",
    )
    return parser.parse_args()


def build_commit_message(message: str | None, include_timestamp: bool) -> str:
    """Build the final commit message."""

    commit_message = message or DEFAULT_MESSAGE
    if include_timestamp:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        commit_message = f"{commit_message} | {timestamp}"
    return commit_message


def main() -> int:
    args = parse_args()
    cwd = project_root()

    print_step(f"Repository root: {cwd}")

    try:
        remote_url = current_remote_url(cwd)
        print_step(f"Origin remote: {remote_url}")
        if remote_url.startswith("http://") or remote_url.startswith("https://"):
            print_step("Warning: origin remote uses HTTPS. Switch to an SSH remote before using this utility.")
            return 1

        initial_status = working_tree_status(cwd)
        if not initial_status:
            print("Nothing to commit")
            return 0

        print_step("Current git status:")
        print(initial_status)

        print_step("Staging changes with: git add .")
        run_git_command(["git", "add", "."], cwd=cwd)

        staged_status = working_tree_status(cwd)
        if not staged_status:
            print("Nothing to commit")
            return 0

        commit_message = build_commit_message(args.message, args.timestamp)
        print_step(f'Creating commit: "{commit_message}"')
        commit_result = run_git_command(["git", "commit", "-m", commit_message], cwd=cwd)
        if commit_result.stdout.strip():
            print(commit_result.stdout.strip())

        current_branch = branch_name(cwd)
        print_step(f"Current branch: {current_branch}")
        print_step("Git status before push:")
        final_status = working_tree_status(cwd)
        print(final_status if final_status else "(clean working tree)")

        print_step("Pushing to origin main over SSH")
        push_result = run_git_command(["git", "push", "origin", "main"], cwd=cwd)
        if push_result.stdout.strip():
            print(push_result.stdout.strip())
        if push_result.stderr.strip():
            print(push_result.stderr.strip())

        commit_hash = latest_commit_hash(cwd)
        print_step("Push succeeded")
        print_step(f"Latest commit hash: {commit_hash}")
        return 0
    except subprocess.CalledProcessError as exc:
        print_step("Git command failed")
        print(f"Command: {' '.join(exc.args[0]) if isinstance(exc.args[0], list) else exc.cmd}")
        if exc.stdout:
            print(exc.stdout.strip())
        if exc.stderr:
            print(exc.stderr.strip())
        return exc.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
