#!/usr/bin/env python3
# -*- coding: ascii -*-

import os
import sys
import json
import time
import re
import argparse
import concurrent.futures
import queue
import threading
import subprocess
import csv
import tempfile
import requests
import uuid
import shutil
import urllib.parse
import signal
import random
import atexit
import ipaddress
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Tuple, List, Dict, Set, Optional, Union
from openai import OpenAI

# ==============================================================================
# Global Configuration & Endpoints
# ==============================================================================

# Phase 0: Git Repository Intake Config
GIT_CLONE_DEPTH = 1
GIT_CLONE_TIMEOUT = 600
REPO_MAX_FILE_BYTES = 200000        # Skip single files larger than this
REPO_MAX_TOTAL_CHARS = 4000000      # Hard cap on total ingested source characters
REPO_MANIFEST_MAX_ENTRIES = 400     # Cap manifest listing length in the intake doc
REPO_SUMMARY_REDUCE_DEPTH = 3       # Max recursive reduce passes over batch summaries

REPO_CODE_EXTENSIONS = {
    ".py", ".pyx", ".pyi", ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".cu", ".cuh",
    ".cl", ".rs", ".go", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".java", ".kt",
    ".swift", ".rb", ".php", ".cs", ".sh", ".bash", ".zsh", ".ps1", ".pl", ".lua",
    ".r", ".jl", ".scala", ".sql", ".m", ".mm", ".v", ".vhd", ".proto", ".cmake",
    ".mk", ".gradle", ".tf", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".json",
    ".md", ".rst", ".txt", ".dockerfile"
}
REPO_SPECIAL_FILENAMES = {
    "dockerfile", "makefile", "cmakelists.txt", "requirements.txt", "setup.py",
    "setup.cfg", "pyproject.toml", "package.json", "cargo.toml", "go.mod",
    "readme", "license", "gemfile", "rakefile", "justfile"
}
REPO_EXCLUDE_FILENAMES = {
    "package-lock.json", "yarn.lock", "poetry.lock", "cargo.lock", "pnpm-lock.yaml",
    "composer.lock", "gemfile.lock"
}
REPO_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "vendor", "dist", "build", "target",
    "__pycache__", ".venv", "venv", "env", ".tox", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "site-packages", ".idea", ".vscode", "third_party", "external",
    ".eggs", "htmlcov", ".ipynb_checkpoints"
}

GIT_URL_PATTERNS = [
    r'^https?://[\w.\-]+(:\d+)?/[\w.\-~+/%]+(\.git)?/?$',
    r'^git@[\w.\-]+:[\w.\-~+/%]+(\.git)?$',
    r'^ssh://(git@)?[\w.\-]+(:\d+)?/[\w.\-~+/%]+(\.git)?$',
    r'^git://[\w.\-]+/[\w.\-~+/%]+(\.git)?$',
]

_REPO_EXT_LANG_MAP = {
    ".py": "python", ".pyx": "python", ".pyi": "python", ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".cu": "cuda",
    ".cuh": "cuda", ".cl": "c", ".rs": "rust", ".go": "go", ".js": "javascript",
    ".mjs": "javascript", ".cjs": "javascript", ".ts": "typescript",
    ".tsx": "tsx", ".jsx": "jsx", ".java": "java", ".kt": "kotlin",
    ".swift": "swift", ".rb": "ruby", ".php": "php", ".cs": "csharp",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash", ".ps1": "powershell",
    ".pl": "perl", ".lua": "lua", ".r": "r", ".jl": "julia", ".scala": "scala",
    ".sql": "sql", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".json": "json", ".md": "markdown", ".rst": "rst", ".proto": "protobuf",
    ".cmake": "cmake", ".tf": "hcl"
}

# Phase 1: Raw Generation Config
GEN_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:8081/v1")
GEN_API_KEY = os.getenv("OPENAI_API_KEY", "sk-local")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.5-122B-A10B-UD-IQ1_M.gguf")

# Unified Context Limits
MAX_CONTEXT_CHARS = 60000
MAX_CHUNK_CHARS = 40000

# Phase 2: Distillation Config
DISTILLER_URL = os.getenv("DISTILLER_URL", "http://localhost:8081/v1")
DISTILLER_MODEL = os.getenv("DISTILLER_MODEL", "Qwen3.5-122B-A10B-UD-IQ1_M.gguf")
DISTILLER_API_KEY = os.getenv("DISTILLER_API_KEY", "local-sk")

# Phase 3 & 4: Orchestrator Map-Reduce & Post-Processing Config
ORCHESTRATOR_ENDPOINTS = [
    "http://localhost:8080/v1",
    "http://localhost:8079/v1"
]
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "nvidia_Orchestrator-8B-Q5_K.gguf")
ORCH_API_KEY = os.getenv("ORCH_API_KEY", "local-sk")
MAX_RETRIES = 3

# Phase 3b: Dedicated Stitcher Cluster (tree-reduction merge nodes)
STITCHER_ENDPOINTS = [
    "http://localhost:8070/v1",
    "http://localhost:8071/v1"
]
STITCHER_MODEL = os.getenv("STITCHER_MODEL", "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf")
STITCHER_API_KEY = os.getenv("STITCHER_API_KEY", "local-sk")
STITCH_PARALLEL_SLOTS = 2
MAX_RETRIES = 3

# Tuned for ~8k context window models. 16,000 chars is ~4,000 tokens,
# leaving a comfortable 4,096 tokens for the completion payload.
MAX_STITCH_TOKENS = int(os.getenv("MAX_STITCH_TOKENS", "4096"))
MAX_STITCH_CONTEXT_CHARS = int(os.getenv("MAX_STITCH_CONTEXT_CHARS", "16000"))

WORKER_ENDPOINTS = [
    "http://localhost:8033/v1",
    "http://localhost:8034/v1"
]
WORKER_MODEL = os.getenv("WORKER_MODEL", "Qwen3.5-9B-IQ4_XS.gguf")
WORKER_API_KEY = os.getenv("WORKER_API_KEY", "local-sk")

WORKER_PARALLEL_SLOTS = 2
WORKER_RETRIES = 3
WORKER_TIMEOUT_SECS = float(os.getenv("WORKER_TIMEOUT_SECS", "1800.0"))
ORCH_PARALLEL_SLOTS = 1
SYNTHESIS_CHUNK_SIZE = 3

CONCURRENT_SLOTS_PER_ENDPOINT = 1
MAX_WORKER_TOKENS = 12288

# Phase 5: Automatic Unittests Config
TEST_WORKER_ENDPOINTS = [
    "http://localhost:8033/v1/chat/completions",
    "http://localhost:8034/v1/chat/completions"
]
CONCURRENT_REQS_PER_ENDPOINT = 2
MAX_OUTPUT_TOKENS = 16384
LLM_TEMPERATURE = 0.1
LLM_TOP_P = 0.95
LLM_FREQUENCY_PENALTY = 0.5
LLM_PRESENCE_PENALTY = 0.2
RETRY_BASE_DELAY = 2.0
RETRY_JITTER = 0.5
MAX_EXEC_WORKERS = 4
EXECUTION_RESULT_FIELDS = ["filename", "language", "status", "message"]


# ==============================================================================
# Global Prompts
# ==============================================================================

_PROMPT_PHASE1_GEN = (
    "You are an expert researcher and technical writer.\n"
    "Your task is to write a comprehensive, detailed, and highly informative document based on the user's prompt.\n"
    "Write clearly, use markdown formatting (headings, bullet points, bold text), and provide deep insights.\n"
    "Do not include any conversational filler. Just output the raw document content."
)

_PROMPT_PHASE0_SUMMARIZE = (
    "You are a senior staff engineer performing a rigorous code audit of a repository batch. "
    "For EVERY file provided, output a markdown section starting with '### <file path>' containing: "
    "1. Purpose of the file. "
    "2. Key classes and functions with one-line descriptions (include signatures where useful). "
    "3. External dependencies and relationships to other files. "
    "4. Notable issues, bugs, TODOs, or architectural concerns. "
    "Be dense and technical. Do not omit any file. Do not add conversational filler. "
    "Output strictly in standard ASCII."
)

_PROMPT_PHASE0_REDUCE = (
    "You are a consolidation node merging per-file code audit notes from a large repository. "
    "Merge and deduplicate the notes into a compressed but information-dense markdown analysis. "
    "Preserve every distinct file path as a '### <file path>' header. "
    "Retain concrete technical detail: function names, dependencies, issues, TODOs. "
    "Remove repetition and filler. Output strictly in standard ASCII."
)

_PROMPT_PHASE2_DISTILL = (
    "You are a ruthless, highly technical Lead Engineer and Project Manager. "
    "Your job is to read dense, fluffy, or theoretical technical documents and extract ONLY "
    "a succinct, actionable list of explicit TO-DOs, architectural requirements, and implementation tasks. "
    "STRIP AWAY all marketing fluff, academic rambling, metaphors, and context setting. "
    "Output a clean, highly structured Markdown list of tasks that a developer can immediately start building. "
    "Do not include conversational filler."
)

_PROMPT_PHASE3_DECOMPOSE = (
    "You are an algorithmic micro-task decomposer.\n"
    "Your sole purpose is to take a large, complex query or task and shatter it into atomic, independent pieces for parallel processing.\n"
    "Output ONLY a valid, flat JSON array of strings. No markdown formatting, no conversational text."
)

_PROMPT_PHASE3_WORKER = (
    "You are an autonomous, highly-capable worker agent equipped with advanced reasoning. "
    "Think step-by-step to formulate a plan. You must execute your specific objective fully. "
    "CRITICAL: If your task involves writing code, creating configurations, or generating files, "
    "you MUST output the file contents wrapped exactly in these XML tags:\n"
    '<file path="filename.ext">\n[YOUR FILE CONTENT HERE]\n</file>\n'
)

_PROMPT_PHASE3_SYNTHESIS = (
    "You are a Level-1 Synthesis Node in a distributed cluster. "
    "Merge the following sequential worker reports into a coherent, deduplicated section. "
    "Retain all code blocks, configurations, and critical technical data. "
    "Output strictly in standard ASCII."
)

_PROMPT_PHASE3_STITCH = (
    "You are the Final Assembly Layer. Seamlessly weave the new sequential section into the existing master document. "
    "Expand the document logically. Do not drop existing data or code. Output strictly in standard ASCII."
)

_PROMPT_PHASE4_DEDUP = (
    "You are a strict, highly logical Technical Editor processing a section of a larger technical document. "
    "Your job is to take this messy, repetitive draft and rewrite it into cohesive, succinct markdown. "
    "\n\nRULES:"
    "\n1. Remove all repetitive statements, redundant introductions, and duplicated concepts."
    "\n2. Organize the content into logical, non-repeating markdown headers."
    "\n3. Use bullet points for readability where applicable."
    "\n4. Do not add conversational filler. Be direct and professional."
)

_PROMPT_PHASE4_SMOOTH = (
    "You are the Executive Technical Editor (Pass 1). "
    "Smooth out any jarring transitions between sections in this stitched document. "
    "Do NOT delete any major technical concepts, instructions, or features."
)

_PROMPT_PHASE4_UNIFY = (
    "You are the Executive Technical Editor (Pass 2). "
    "Unify the tone and organize the Markdown headers logically from start to finish. "
    "Do NOT delete any major technical concepts, instructions, or features."
)

_PROMPT_PHASE5_UNITTEST = (
    "You are a highly efficient code testing assistant. "
    "Write succinct, compact, and boilerplate-free unit tests. "
    "Use parametrization to consolidate test cases where applicable. Do not explain your code.\n\n"
    "CRITICAL RULES:\n"
    "1. DO NOT hallucinate imports or use non-existent modules.\n"
    "2. Keep the code as short as possible while ensuring it runs and passes.\n"
    "3. Group assertions and use parametrization where possible to save space.\n"
    "4. Output ONLY valid test code inside a single markdown code block. No explanations.\n"
    "5. If C/C++, #include the provided filename directly and write your own main()."
)

_PROMPT_PHASE6_DISTILL = (
    "You are a ruthless, highly technical Lead Engineer and Project Manager. "
    "Read the provided raw documentation and extract a succinct, actionable "
    "list of explicit TO-DOs, architectural requirements, and implementation tasks. "
    "\n\nCRITICAL DIRECTIVES: "
    "\n1. TEST TELEMETRY: Hunt for any unit test execution logs, reports, or telemetry. "
    "Create a distinct 'Test Execution Status' section detailing passes and failures. "
    "Convert any failures into high-priority TO-DO items."
    "\n2. EMBED ARTIFACTS: For EVERY task or failed test, you MUST extract and embed the relevant "
    "source artifact directly beneath the task description. If a task involves a specific function, "
    "include the code snippet. If it involves an error, include the traceback. "
    "Format these artifacts using proper markdown code fences and explicitly label the source filename."
    "\n3. STRICT ASCII ONLY: The generated markdown MUST consist entirely of standard ASCII characters. "
    "Do NOT use unicode symbols. "
    "Use standard hyphens (-) or asterisks (*) for bullet points. "
    "\n\nOutput a clean, highly structured Markdown document. Do not include conversational filler."
)

# ==============================================================================
# Global Utilities & State
# ==============================================================================

_active_clone_dirs: Set[Path] = set()
_clone_dirs_lock = threading.Lock()
_shutdown_event = threading.Event()

def cleanup_clones():
    with _clone_dirs_lock:
        for clone_dir in list(_active_clone_dirs):
            if clone_dir.exists():
                shutil.rmtree(clone_dir, ignore_errors=True)
        _active_clone_dirs.clear()

atexit.register(cleanup_clones)

def enforce_ascii(text: str) -> str:
    """Strip out non-ASCII characters immediately from LLM responses."""
    return text.encode("ascii", "ignore").decode("ascii")

def read_file_content_safe(file_path: Path) -> Optional[str]:
    encodings = ['utf-8', 'latin-1', 'ascii']
    for enc in encodings:
        try:
            errors = "ignore" if enc == 'ascii' else "strict"
            with open(file_path, "r", encoding=enc, errors=errors) as f:
                return f.read()
        except Exception:
            continue
    return None

def split_into_logical_chunks(text: str, max_chars: int) -> List[str]:
    chunks = []
    current_chunk = ""
    sections = [s for s in re.split(r'(?=\n#{2,3} )', text) if s.strip()]

    for section in sections:
        if len(current_chunk) + len(section) <= max_chars:
            current_chunk += section
        else:
            if len(section) > max_chars:
                paragraphs = section.split('\n\n')
                for p in paragraphs:
                    safe_p = p[:max_chars] if len(p) > max_chars else p
                    if len(current_chunk) + len(safe_p) + 2 <= max_chars:
                        current_chunk += safe_p + "\n\n"
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = safe_p + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = section

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def _format_eta(start_time: float, completed: int, total: int) -> str:
    if completed == 0 or total == 0:
        return "--:--"
    elapsed = time.time() - start_time
    eta_secs = (elapsed / completed) * (total - completed)
    mins, secs = divmod(int(eta_secs), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours}h {mins:02d}m"
    return f"{mins:02d}:{secs:02d}"

# ==============================================================================
# Phase 1: Raw Content Generation
# ==============================================================================

def generate_safe_filename(prompt_text: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    words = re.findall(r'[a-zA-Z0-9]+', prompt_text)[:5]
    slug = "-".join(words).lower()
    if not slug:
        slug = "generated-content"
    short_uuid = uuid.uuid4().hex[:6]
    return f"{timestamp}_{slug}_{short_uuid}.md"

def generate_content(prompt: str, target_dir: Path) -> Path:
    print(f"\n[PHASE 1] [*] Generating content for: '{prompt[:50]}...'")
    gen_client = OpenAI(base_url=GEN_API_BASE, api_key=GEN_API_KEY, timeout=WORKER_TIMEOUT_SECS)

    full_content = ""
    start_time = time.time()

    try:
        response = gen_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _PROMPT_PHASE1_GEN},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4096,
            stream=True
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                full_content += chunk.choices[0].delta.content

        elapsed = round(time.time() - start_time, 2)
        print(f"[+] Generation complete in {elapsed} seconds.")

        ascii_content = enforce_ascii(full_content)
        if len(ascii_content) < len(full_content):
            print(f"    [!] Warning: Dropped {len(full_content) - len(ascii_content)} non-ASCII characters during generation.", flush=True)

        filename = generate_safe_filename(prompt)
        filepath = target_dir / filename

        with open(filepath, "w", encoding="ascii") as f:
            f.write(ascii_content.strip())

        print(f"[+] Saved raw content to: {filepath.absolute()}")
        return filepath

    except Exception as e:
        print(f"\n[!] Fatal Error during generation: {e}")
        sys.exit(1)

# ==============================================================================
# Phase 0: Git Repository Intake
# ==============================================================================

def _is_private_host(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(host)
        return addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved
    except ValueError:
        blocked = {"metadata.google.internal", "metadata.azure.internal", "localhost"}
        return host.lower() in blocked

def validate_git_url(git_url: str) -> bool:
    if not any(re.match(p, git_url) for p in GIT_URL_PATTERNS):
        return False

    parsed = urllib.parse.urlparse(git_url)
    host = parsed.hostname or ""

    if not host:
        scp_match = re.match(r'^git@([\w.\-]+):', git_url)
        if scp_match:
            host = scp_match.group(1)
        else:
            return False

    if _is_private_host(host):
        return False

    return True

def clone_git_repository(git_url: str) -> tuple:
    if shutil.which("git") is None:
        print("[!] Fatal: 'git' executable not found on PATH.", flush=True)
        sys.exit(1)

    clone_dir = Path(tempfile.mkdtemp(prefix="autoresearch_repo_"))
    with _clone_dirs_lock:
        _active_clone_dirs.add(clone_dir)

    print(f"[*] Cloning repository (depth={GIT_CLONE_DEPTH}): {git_url}", flush=True)

    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    cmd = ["git", "clone", "--depth", str(GIT_CLONE_DEPTH), "--single-branch",
           "--", git_url, str(clone_dir)]
    try:
        start_time = time.time()
        res = subprocess.run(cmd, capture_output=True, encoding="ascii",
                             errors="replace", timeout=GIT_CLONE_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        shutil.rmtree(clone_dir, ignore_errors=True)
        with _clone_dirs_lock:
            _active_clone_dirs.discard(clone_dir)
        print(f"[!] Fatal: git clone timed out after {GIT_CLONE_TIMEOUT}s.", flush=True)
        sys.exit(1)

    if res.returncode != 0:
        shutil.rmtree(clone_dir, ignore_errors=True)
        with _clone_dirs_lock:
            _active_clone_dirs.discard(clone_dir)
        err_lines = [l for l in (res.stderr or "").strip().splitlines() if l.strip()]
        err_tail = err_lines[-1] if err_lines else "unknown error"
        print(f"[!] Fatal: git clone failed: {err_tail}", flush=True)
        sys.exit(1)

    commit_hash, branch_name = "unknown", "unknown"
    try:
        h = subprocess.run(["git", "-C", str(clone_dir), "rev-parse", "HEAD"],
                           capture_output=True, encoding="ascii", errors="replace", timeout=30)
        if h.returncode == 0:
            commit_hash = h.stdout.strip()
        b = subprocess.run(["git", "-C", str(clone_dir), "rev-parse", "--abbrev-ref", "HEAD"],
                           capture_output=True, encoding="ascii", errors="replace", timeout=30)
        if b.returncode == 0:
            branch_name = b.stdout.strip()
    except Exception:
        pass

    elapsed = round(time.time() - start_time, 2)
    print(f"    [+] Clone complete in {elapsed}s. HEAD: {commit_hash[:12]} (branch: {branch_name})", flush=True)
    return clone_dir, commit_hash, branch_name

def collect_repo_code_files(repo_dir: Path, sub_path: str = "") -> tuple:
    entries = []
    stats = {"ingested": 0, "skipped_large": 0, "skipped_binary": 0,
             "skipped_unreadable": 0, "total_chars": 0, "capped": False}

    candidates = []
    search_target = repo_dir / sub_path.strip("/") if sub_path else repo_dir

    if sub_path and not search_target.exists():
        print(f"    [!] Error: Target path '{sub_path}' not found in the cloned repository.", flush=True)
        return entries, stats

    if search_target.is_file():
        candidates.append((search_target.relative_to(repo_dir), search_target))
    else:
        for root, dirs, files in os.walk(search_target, followlinks=False):
            root_path = Path(root)
            dirs[:] = [d for d in dirs if d not in REPO_EXCLUDE_DIRS and not (root_path / d).is_symlink()]
            for f in files:
                file_path = root_path / f
                if file_path.is_symlink():
                    continue
                rel = file_path.relative_to(repo_dir)
                name_lower = f.lower()
                stem_lower = file_path.stem.lower()

                if name_lower in REPO_EXCLUDE_FILENAMES:
                    continue
                if (file_path.suffix.lower() not in REPO_CODE_EXTENSIONS
                        and name_lower not in REPO_SPECIAL_FILENAMES
                        and stem_lower not in REPO_SPECIAL_FILENAMES):
                    continue

                candidates.append((rel, file_path))

    def sort_key(item):
        rel, _ = item
        name_lower = rel.name.lower()
        is_priority = (name_lower.startswith("readme")
                       or name_lower in REPO_SPECIAL_FILENAMES
                       or rel.stem.lower() in REPO_SPECIAL_FILENAMES)
        return (0 if is_priority else 1, len(rel.parts), str(rel).lower())

    candidates.sort(key=sort_key)

    for rel, path in candidates:
        try:
            size = path.stat().st_size
        except OSError:
            stats["skipped_unreadable"] += 1
            continue
        if size == 0:
            continue
        if size > REPO_MAX_FILE_BYTES:
            stats["skipped_large"] += 1
            continue

        try:
            with open(path, "rb") as fb:
                if b"\x00" in fb.read(8192):
                    stats["skipped_binary"] += 1
                    continue
        except OSError:
            stats["skipped_unreadable"] += 1
            continue

        content = read_file_content_safe(path)
        if content is None or not content.strip():
            stats["skipped_unreadable"] += 1
            continue

        char_len = len(content)
        if stats["total_chars"] + char_len > REPO_MAX_TOTAL_CHARS:
            stats["capped"] = True
            print(f"    [!] WARNING: Total ingest cap of {REPO_MAX_TOTAL_CHARS:,} characters reached. Remaining files skipped.", flush=True)
            break

        stats["total_chars"] += char_len
        stats["ingested"] += 1
        rel_str = str(rel).replace("\\", "/")

        if len(content) > MAX_CHUNK_CHARS:
            part_count = (len(content) + MAX_CHUNK_CHARS - 1) // MAX_CHUNK_CHARS
            for p_idx in range(part_count):
                segment = content[p_idx * MAX_CHUNK_CHARS:(p_idx + 1) * MAX_CHUNK_CHARS]
                entries.append({
                    "path": f"{rel_str} (part {p_idx + 1}/{part_count})",
                    "suffix": path.suffix.lower(),
                    "content": segment, "chars": len(segment)
                })
        else:
            entries.append({"path": rel_str, "suffix": path.suffix.lower(),
                            "content": content, "chars": len(content)})

    return entries, stats

def batch_repo_entries(entries: list) -> list:
    batches, current, current_chars = [], [], 0
    for entry in entries:
        if current and current_chars + entry["chars"] >= MAX_CHUNK_CHARS:
            batches.append(current)
            current, current_chars = [], 0
        current.append(entry)
        current_chars += entry["chars"]
    if current:
        batches.append(current)
    return batches

def build_repo_manifest(git_url: str, repo_name: str, commit_hash: str,
                        branch_name: str, entries: list, stats: dict, focus: str, git_path: str = "") -> str:
    lines = [f"# Git Repository Analysis: {repo_name}", ""]
    lines.append(f"- **Source URL:** {git_url}")
    lines.append(f"- **Branch:** {branch_name}")
    lines.append(f"- **HEAD Commit:** {commit_hash}")
    lines.append(f"- **Cloned:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if git_path:
        lines.append(f"- **Target Path:** `{git_path}`")
    lines.append(f"- **Files ingested:** {stats['ingested']} ({stats['total_chars']:,} characters)")
    skipped_total = stats["skipped_large"] + stats["skipped_binary"] + stats["skipped_unreadable"]
    lines.append(f"- **Files skipped:** {skipped_total} ({stats['skipped_large']} oversized, "
                 f"{stats['skipped_binary']} binary, {stats['skipped_unreadable']} unreadable)")
    if stats.get("capped"):
        lines.append(f"- **NOTE:** Ingestion stopped at the {REPO_MAX_TOTAL_CHARS:,} character cap; the target path was not fully ingested.")
    if focus:
        lines.append(f"- **Analysis focus:** {focus}")
    lines.append("")
    lines.append("## Ingested File Manifest")
    lines.append("")
    manifest_paths = [e["path"] for e in entries]
    for p in manifest_paths[:REPO_MANIFEST_MAX_ENTRIES]:
        lines.append(f"- {p}")
    if len(manifest_paths) > REPO_MANIFEST_MAX_ENTRIES:
        lines.append(f"- ... and {len(manifest_paths) - REPO_MANIFEST_MAX_ENTRIES} more file segments")
    lines.append("")
    return enforce_ascii("\n".join(lines))

def render_inline_source(entries: list) -> str:
    sections = ["## Source Files", ""]
    for entry in entries:
        lang = _REPO_EXT_LANG_MAP.get(entry.get("suffix", ""), "")
        max_ticks = max((len(m.group(0)) for m in re.finditer(r'`+', entry["content"])), default=2)
        fence = '`' * max(3, max_ticks + 1)
        sections.append(f"### {entry['path']}")
        sections.append(f"{fence}{lang}")
        sections.append(entry["content"].rstrip())
        sections.append(fence)
        sections.append("")
    return "\n".join(sections)

def _repo_worker_call(system_prompt: str, user_prompt: str, endpoint: str) -> str:
    client = OpenAI(base_url=endpoint, api_key=WORKER_API_KEY, timeout=WORKER_TIMEOUT_SECS, max_retries=0)
    response = client.chat.completions.create(
        model=WORKER_MODEL,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=0.2,
        max_tokens=MAX_WORKER_TOKENS
    )
    raw = response.choices[0].message.content.strip()
    return enforce_ascii(raw)

def _parallel_repo_jobs(jobs: list, job_fn, fallback_fn, label: str) -> list:
    total = len(jobs)
    slot_queue = queue.Queue()
    for ep in WORKER_ENDPOINTS:
        for _ in range(WORKER_PARALLEL_SLOTS):
            slot_queue.put(ep)

    results = [""] * total
    completed_count = 0
    start_time_progress = time.time()

    def wrapper(idx: int, payload):
        for attempt in range(1, WORKER_RETRIES + 1):
            try:
                endpoint = slot_queue.get(timeout=5.0)
            except queue.Empty:
                continue

            try:
                output = job_fn(idx + 1, total, payload, endpoint)
                if output and len(output.strip()) >= 20:
                    return output.strip()
            except Exception as e:
                if attempt < WORKER_RETRIES:
                    print(f"        [!] {label} {idx+1}/{total} attempt {attempt} failed ({str(e)}), retrying...", flush=True)
            finally:
                slot_queue.put(endpoint)
            time.sleep(2)
        return fallback_fn(payload)

    pool_size = max(1, len(WORKER_ENDPOINTS) * WORKER_PARALLEL_SLOTS)
    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
        future_to_idx = {executor.submit(wrapper, i, job): i for i, job in enumerate(jobs)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            completed_count += 1
            eta_str = _format_eta(start_time_progress, completed_count, total)
            try:
                results[idx] = future.result()
                print(f"        [+] {label} {completed_count}/{total} complete. | ETC: {eta_str}", flush=True)
            except Exception as e:
                print(f"        [!] {label} {completed_count}/{total} future raised exception: {e}. Using fallback. | ETC: {eta_str}", flush=True)
                results[idx] = fallback_fn(jobs[idx])
    return results

def summarize_repo_batches(batches: list, focus: str) -> list:
    def job_fn(batch_id: int, total: int, files: list, endpoint: str) -> str:
        corpus = "\n\n".join([f"===== FILE: {f['path']} =====\n{f['content']}" for f in files])
        system_prompt = _PROMPT_PHASE0_SUMMARIZE + (f"\n\nANALYSIS FOCUS: Prioritize findings relevant to: {focus}" if focus else "")
        user_prompt = f"Repository batch {batch_id}/{total}. Analyse these files:\n\n{corpus}"
        return _repo_worker_call(system_prompt, user_prompt, endpoint)

    def fallback_fn(files: list) -> str:
        return "\n\n".join(
            [f"### {f['path']}\n(Summarization failed; truncated raw preview below.)\n\n"
             f"{f['content'][:2000]}" for f in files]
        )

    print(f"    [*] Summarizing {len(batches)} batches across "
          f"{len(WORKER_ENDPOINTS)} worker endpoint(s) x {WORKER_PARALLEL_SLOTS} slots...", flush=True)
    return _parallel_repo_jobs(batches, job_fn, fallback_fn, "Repo batch")

def reduce_repo_summaries(summaries: list, focus: str, char_budget: int) -> str:
    combined = "\n\n".join(summaries)

    def job_fn(chunk_id: int, total: int, chunk_text: str, endpoint: str) -> str:
        system_prompt = _PROMPT_PHASE0_REDUCE + (f"\n\nANALYSIS FOCUS: Prioritize findings relevant to: {focus}" if focus else "")
        user_prompt = f"Consolidation chunk {chunk_id}/{total}:\n\n{chunk_text}"
        return _repo_worker_call(system_prompt, user_prompt, endpoint)

    def fallback_fn(chunk_text: str) -> str:
        return chunk_text[:MAX_CHUNK_CHARS // 2]

    depth = 0
    last_len = len(combined)
    while len(combined) > char_budget and depth < REPO_SUMMARY_REDUCE_DEPTH:
        depth += 1
        print(f"    [*] Reduce pass {depth}: consolidating {len(combined):,} chars "
              f"toward {char_budget:,} char budget...", flush=True)
        chunks = split_into_logical_chunks(combined, MAX_CHUNK_CHARS)
        merged = _parallel_repo_jobs(chunks, job_fn, fallback_fn, "Reduce chunk")
        new_combined = "\n\n".join(merged)

        if len(new_combined) >= last_len * 0.95:
            print("    [!] Reduce pass produced < 5% compression. Truncation will occur if budget is exceeded.", flush=True)
            combined = new_combined
            break
        combined = new_combined
        last_len = len(combined)

    if len(combined) > char_budget:
        combined = combined[:char_budget] + "\n\n...[REPO ANALYSIS TRUNCATED FOR CONTEXT LIMITS]..."
    return combined

def ingest_git_repository(git_url: str, target_dir: Path, focus: str = "", git_path: str = "") -> Path:
    print(f"\n[PHASE 0] GIT REPOSITORY INTAKE", flush=True)

    if not validate_git_url(git_url):
        print(f"[!] Fatal: '{git_url}' does not look like a valid git URL or references blocked subnets.", flush=True)
        sys.exit(1)

    clone_dir = None
    try:
        clone_dir, commit_hash, branch_name = clone_git_repository(git_url)
        entries, stats = collect_repo_code_files(clone_dir, git_path)
        if not entries:
            path_err = f" at path '{git_path}'" if git_path else ""
            print(f"[!] Fatal: No ingestible code or documentation files found{path_err} (or repository is empty).", flush=True)
            sys.exit(1)

        repo_tail = git_url.rstrip('/').split('/')[-1].split(':')[-1]
        repo_name = re.sub(r'\.git$', '', repo_tail) or "repository"

        header = build_repo_manifest(git_url, repo_name, commit_hash, branch_name,
                                     entries, stats, focus, git_path)
        body_budget = max(10000, MAX_CONTEXT_CHARS - len(header))

        if stats["total_chars"] + len(header) <= MAX_CONTEXT_CHARS:
            print(f"    [*] Repository fits in context ({stats['total_chars']:,} chars). Embedding source directly.", flush=True)
            body = render_inline_source(entries)
        else:
            print(f"    [*] Repository exceeds context ({stats['total_chars']:,} chars). Engaging worker map-reduce summarization.", flush=True)
            batches = batch_repo_entries(entries)
            print(f"    [*] Packed {len(entries)} file segments into {len(batches)} batches "
                  f"(<= {MAX_CHUNK_CHARS:,} chars each).", flush=True)
            summaries = summarize_repo_batches(batches, focus)
            body = "## Repository Analysis\n\n" + reduce_repo_summaries(summaries, focus, body_budget)

        document = f"{header}\n{body}"
        filename = generate_safe_filename(f"git repo analysis {repo_name}")
        filepath = target_dir / filename
        with open(filepath, "w", encoding="ascii") as f:
            f.write(document.strip() + "\n")

        print(f"[+] Repository intake document saved to: {filepath.absolute()} ({len(document):,} chars)", flush=True)
        return filepath
    finally:
        if clone_dir and clone_dir.exists():
            shutil.rmtree(clone_dir, ignore_errors=True)
        with _clone_dirs_lock:
            _active_clone_dirs.discard(clone_dir)

# ==============================================================================
# Phase 2: Fluff-to-Action Technical Distillation
# ==============================================================================

def distill_document(raw_text: str) -> str:
    distill_client = OpenAI(base_url=DISTILLER_URL, api_key=DISTILLER_API_KEY)
    char_count = len(raw_text)
    print(f"\n[PHASE 2] [*] Ingesting document ({char_count:,} characters)...", flush=True)

    if char_count > MAX_CONTEXT_CHARS:
        print(f"    [!] WARNING: Document size exceeds {MAX_CONTEXT_CHARS:,} characters. Truncating.", flush=True)
        raw_text = raw_text[:MAX_CONTEXT_CHARS] + "\n\n...[TRUNCATED]..."

    try:
        response = distill_client.chat.completions.create(
            model=DISTILLER_MODEL,
            messages=[
                {"role": "system", "content": _PROMPT_PHASE2_DISTILL},
                {"role": "user", "content": f"Extract the actionable tasks from this document:\n\n{raw_text}"}
            ],
            temperature=0.3,
            max_tokens=8192
        )
        return enforce_ascii(response.choices[0].message.content.strip())

    except Exception as e:
        print(f"[!] Error during distillation: {e}", flush=True)
        sys.exit(1)

def save_distilled_output(output_text: str, original_path: Path) -> Path:
    output_filename = f"{original_path.stem}_distilled.md"
    output_path = original_path.parent / output_filename

    try:
        with open(output_path, "w", encoding="ascii") as f:
            f.write(output_text)
        return output_path
    except Exception as e:
        print(f"[!] Error saving output file: {e}", flush=True)
        sys.exit(1)

# ==============================================================================
# Phase 3: Unified Distributed Orchestrator Cluster (Map-Reduce)
# ==============================================================================

def estimate_tokens(text: str) -> int:
    return len(str(text)) // 4

def extract_json_array(raw_text: str) -> str:
    cleaned_text = re.sub(r'```json\s*', '', raw_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\n?```\s*', '', cleaned_text).strip()

    start_idx = cleaned_text.find('[')
    if start_idx == -1:
        return ""

    depth = 0
    in_string = False
    i = start_idx
    while i < len(cleaned_text):
        char = cleaned_text[i]

        if char == '\\':
            if i + 1 >= len(cleaned_text):
                break
            if cleaned_text[i+1] == 'u':
                i += 6
            else:
                i += 2
            continue

        if char == '"':
            in_string = not in_string
        elif not in_string:
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth == 0:
                    candidate = cleaned_text[start_idx:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        pass
        i += 1
    return ""

def decompose_to_atomic_pieces(large_query: str) -> tuple:
    print(f"\n[PHASE 3] [1] INGRESS: Analyzing massive query...\n    Length: {len(large_query)} characters", flush=True)

    client = OpenAI(base_url=ORCHESTRATOR_ENDPOINTS[0], api_key=ORCH_API_KEY, timeout=WORKER_TIMEOUT_SECS, max_retries=0)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[2] DECOMPOSITION: Engaging atomic breakdown via {ORCHESTRATOR_ENDPOINTS[0]} (Attempt {attempt}/{MAX_RETRIES})...", flush=True)
        raw_output = ""
        prompt_tokens, comp_tokens = 0, 0

        try:
            start_time = time.time()
            try:
                response = client.chat.completions.create(
                    model=ORCHESTRATOR_MODEL,
                    messages=[
                        {"role": "system", "content": _PROMPT_PHASE3_DECOMPOSE},
                        {"role": "user", "content": f"Decompose this to the atomic level:\n\n{large_query}"}
                    ],
                    temperature=0.7, max_tokens=40960, stream=True,
                    stream_options={"include_usage": True}
                )
            except Exception as e:
                if "stream_options" in str(e).lower() or "unrecognized" in str(e).lower():
                    response = client.chat.completions.create(
                        model=ORCHESTRATOR_MODEL,
                        messages=[
                            {"role": "system", "content": _PROMPT_PHASE3_DECOMPOSE},
                            {"role": "user", "content": f"Decompose this to the atomic level:\n\n{large_query}"}
                        ],
                        temperature=0.7, max_tokens=40960, stream=True
                    )
                else:
                    raise e

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    raw_output += chunk.choices[0].delta.content
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                    comp_tokens = chunk.usage.completion_tokens

            raw_output = enforce_ascii(raw_output)
            cleaned_output = extract_json_array(raw_output)
            if not cleaned_output: raise ValueError("Could not locate JSON array.")

            atomic_pieces = json.loads(cleaned_output)
            if not isinstance(atomic_pieces, list) or not all(isinstance(x, str) for x in atomic_pieces):
                raise ValueError(f"Decomposition produced non-string items: {type(atomic_pieces)}")

            if prompt_tokens == 0 and comp_tokens == 0:
                prompt_tokens = estimate_tokens(_PROMPT_PHASE3_DECOMPOSE + large_query)
                comp_tokens = estimate_tokens(raw_output)

            elapsed = round(time.time() - start_time, 2)
            print(f"    [+] Success! Shattered into {len(atomic_pieces)} distinct micro-pieces in {elapsed}s.", flush=True)
            return atomic_pieces, prompt_tokens, comp_tokens

        except Exception as e:
            print(f"    [!] Decomposition Error: {e}", flush=True)
            time.sleep(2)

    return [large_query[:MAX_CHUNK_CHARS]], estimate_tokens(large_query[:MAX_CHUNK_CHARS]), 0

def export_to_split_files(pieces: list, work_dir: Path) -> None:
    if len(pieces) <= 1: return
    tasks_dir = work_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)

    for idx, piece in enumerate(pieces, start=1):
        filepath = tasks_dir / f"task{idx:03d}.md"
        with open(filepath, "w", encoding="ascii") as f:
            f.write(f"{piece.strip()}\n")

def process_subtask(task_id: int, task_prompt: str, endpoint: str, slot_name: str, original_query: str, run_dir: Path) -> dict:
    worker_client = OpenAI(base_url=endpoint, api_key=WORKER_API_KEY, timeout=WORKER_TIMEOUT_SECS, max_retries=0)
    start_time = time.time()

    user_instruction = f"BACKGROUND CONTEXT:\n{original_query}\n\nYOUR SPECIFIC OBJECTIVE:\n{task_prompt}"

    saved_artifacts = []
    status = "success"
    prompt_tokens, comp_tokens = 0, 0
    is_estimated = True
    malformed_tags = False

    try:
        response = worker_client.chat.completions.create(
            model=WORKER_MODEL,
            messages=[{"role": "system", "content": _PROMPT_PHASE3_WORKER}, {"role": "user", "content": user_instruction}],
            temperature=0.4,
            max_tokens=MAX_WORKER_TOKENS,
            frequency_penalty=1.1,
            presence_penalty=0.5
        )
        result_text = enforce_ascii(response.choices[0].message.content.strip())

        open_tags = len(re.findall(r'<file\b', result_text, re.IGNORECASE))
        close_tags = len(re.findall(r'</file>', result_text, re.IGNORECASE))
        if open_tags != close_tags:
            print(f"        [!] Warning: Thread{task_id:02d} emitted unbalanced <file> tags ({open_tags} open, {close_tags} close). Skipping artifact extraction to prevent corruption.", flush=True)
            malformed_tags = True

        prompt_tokens = response.usage.prompt_tokens if response.usage else estimate_tokens(_PROMPT_PHASE3_WORKER + user_instruction)
        comp_tokens = response.usage.completion_tokens if response.usage else estimate_tokens(result_text)
        is_estimated = response.usage is None

        if not malformed_tags:
            file_matches = re.finditer(r'<file\s+path="([^"]+)">([\s\S]*?)</file>', result_text, re.IGNORECASE)
            for match in file_matches:
                file_path, file_content = match.group(1).strip(), match.group(2).strip()
                safe_filename = os.path.basename(file_path)
                artifact_dir = run_dir / "artifacts" / f"thread{task_id:02d}"
                artifact_dir.mkdir(parents=True, exist_ok=True)
                with open(artifact_dir / safe_filename, "w", encoding="ascii") as af:
                    af.write(file_content)
                saved_artifacts.append(safe_filename)

        if len(result_text) < 20: status = "failed_validation"

    except Exception as e:
        result_text, status = f"Worker Error: {str(e)}", "error"
        is_estimated = False

    elapsed = round(time.time() - start_time, 2)
    return {
        "id": task_id, "prompt": task_prompt, "result": result_text,
        "artifacts": saved_artifacts, "status": status,
        "prompt_tokens": prompt_tokens, "completion_tokens": comp_tokens,
        "total_tokens": prompt_tokens + comp_tokens, "elapsed": elapsed,
        "tps": round(comp_tokens / elapsed, 2) if elapsed > 0 else 0,
        "slot": slot_name,
        "is_estimated": is_estimated
    }

def parallel_chunk_synthesis(batch_id: int, tasks: list, endpoint: str, original_query: str) -> tuple:
    client = OpenAI(base_url=endpoint, api_key=ORCH_API_KEY, timeout=WORKER_TIMEOUT_SECS, max_retries=0)
    batch_context = "\n\n".join([f"--- TASK {t['id']}: {t['prompt']} ---\n{t['result']}" for t in tasks])
    user_prompt = f"ORIGINAL QUERY: {original_query}\n\nREPORTS TO MERGE:\n{batch_context}"

    start_time = time.time()
    response = client.chat.completions.create(
        model=ORCHESTRATOR_MODEL,
        messages=[{"role": "system", "content": _PROMPT_PHASE3_SYNTHESIS}, {"role": "user", "content": user_prompt}],
        temperature=0.3, max_tokens=40960
    )
    elapsed = round(time.time() - start_time, 2)

    res_content = enforce_ascii(response.choices[0].message.content.strip())
    p_tok = response.usage.prompt_tokens if response.usage else estimate_tokens(_PROMPT_PHASE3_SYNTHESIS + user_prompt)
    c_tok = response.usage.completion_tokens if response.usage else estimate_tokens(res_content)
    return batch_id, res_content, p_tok, c_tok, elapsed

def rolling_master_stitch(chunk_id: int, left_doc: str, right_doc: str, endpoint: str, original_query: str) -> tuple:
    """Merge right_doc into left_doc via one stitcher LLM call.

    Runs on the dedicated STITCHER_ENDPOINTS pool (STITCHER_MODEL), not the
    orchestrator pool. This is the atomic merge unit consumed by
    tree_reduce_stitch. Both text arguments are treated symmetrically so the
    tree caller can pass any pair of nodes without special-casing left vs.
    right.

    Context budgeting is sized against MAX_STITCH_CONTEXT_CHARS rather than
    the global MAX_CONTEXT_CHARS, because the stitcher model runs a much
    smaller window than the 8B orchestrator it replaces.
    """
    client = OpenAI(base_url=endpoint, api_key=STITCHER_API_KEY, timeout=WORKER_TIMEOUT_SECS, max_retries=0)

    # Dynamic symmetric budget split, sized to the stitcher context window.
    budget_each = max(2000, (MAX_STITCH_CONTEXT_CHARS - 2000) // 2)

    if len(right_doc) > budget_each:
        right_doc = right_doc[:budget_each] + "\n\n...[NEW CHUNK TRUNCATED]..."

    if len(left_doc) > budget_each:
        stitch_context = "...[EARLIER CONTENT TRUNCATED FOR CONTEXT LIMITS]...\n\n" + left_doc[-budget_each:]
    else:
        stitch_context = left_doc

    user_prompt = (
        f"ORIGINAL QUERY: {original_query}\n\n"
        f"--- CURRENT MASTER DOCUMENT ---\n{stitch_context}\n\n"
        f"--- NEW SECTION {chunk_id} TO INTEGRATE ---\n{right_doc}"
    )

    start_time = time.time()
    response = client.chat.completions.create(
        model=STITCHER_MODEL,
        messages=[{"role": "system", "content": _PROMPT_PHASE3_STITCH}, {"role": "user", "content": user_prompt}],
        temperature=0.3, max_tokens=MAX_STITCH_TOKENS
    )
    elapsed = round(time.time() - start_time, 2)

    res_content = enforce_ascii(response.choices[0].message.content.strip())
    p_tok = response.usage.prompt_tokens if response.usage else estimate_tokens(_PROMPT_PHASE3_STITCH + user_prompt)
    c_tok = response.usage.completion_tokens if response.usage else estimate_tokens(res_content)
    return res_content, p_tok, c_tok, elapsed


def tree_reduce_stitch(ordered_chunks: List[str], original_query: str) -> Tuple[str, int, int]:
    """Hierarchical parallel document assembly via pairwise tree reduction.

    Runs entirely on the dedicated stitcher cluster (STITCHER_ENDPOINTS x
    STITCH_PARALLEL_SLOTS). Every round merges all available pairs
    simultaneously across the full stitcher endpoint pool, then feeds the
    merged nodes into the next round.

    Complexity: O(log2 N) sequential rounds, each fully parallelised across
    len(STITCHER_ENDPOINTS) * STITCH_PARALLEL_SLOTS concurrent slots.
    Total merge count is always exactly N-1 (each merge removes one node),
    so overall progress and ETA are computable up front.

    Example with 8 chunks and 4 stitcher slots:
        Round 1 : 4 merges in parallel  -> 4 nodes
        Round 2 : 2 merges in parallel  -> 2 nodes
        Round 3 : 1 merge               -> 1 node  (final document)

    Args:
        ordered_chunks : Synthesized chunk texts in logical (chunk_id) order.
        original_query : Forwarded verbatim to each rolling_master_stitch call.

    Returns:
        (master_document, total_stitch_p_tok, total_stitch_c_tok)
    """
    if not ordered_chunks:
        return "", 0, 0

    if len(ordered_chunks) == 1:
        print("    [*] Single chunk - no stitching required.", flush=True)
        return ordered_chunks[0], 0, 0

    # Build a flat, rotating endpoint pool shared across all rounds.
    endpoint_pool: List[str] = [
        ep for ep in STITCHER_ENDPOINTS for _ in range(STITCH_PARALLEL_SLOTS)
    ]

    if not endpoint_pool:
        print("    [!] WARNING: No stitcher endpoints defined. Falling back to raw concatenation.", flush=True)
        return "\n\n".join(ordered_chunks), 0, 0

    ep_queue: queue.Queue = queue.Queue()
    for ep in endpoint_pool:
        ep_queue.put(ep)

    # ---- Pre-compute the reduction plan so progress/ETA are exact ----
    total_merges = len(ordered_chunks) - 1
    total_rounds = 0
    _nodes = len(ordered_chunks)
    while _nodes > 1:
        total_rounds += 1
        _nodes = (_nodes // 2) + (_nodes % 2)

    print(
        f"    [*] Stitch plan: {total_merges} merge(s) over {total_rounds} round(s) "
        f"using {len(endpoint_pool)} concurrent stitcher slot(s).",
        flush=True,
    )

    total_stitch_p_tok = 0
    total_stitch_c_tok = 0
    merges_done = [0]            # global merge counter (drives progress bar)
    merge_counter = [0]          # monotonic merge ID for logging
    progress_lock = threading.Lock()
    stitch_start = time.time()
    current_nodes = list(ordered_chunks)
    round_num = 0

    def _render_progress(active_round: int) -> None:
        """Redraw the single-line stitch progress bar. Caller holds progress_lock."""
        bar_len = 30
        done = merges_done[0]
        filled = int((done / total_merges) * bar_len) if total_merges else bar_len
        bar = "#" * filled + "-" * (bar_len - filled)
        percent = int((done / total_merges) * 100) if total_merges else 100
        eta_str = _format_eta(stitch_start, done, total_merges)
        sys.stdout.write(
            f"\r    [+] Stitch Progress: [{bar}] {percent}% "
            f"(Merges: {done}/{total_merges} | Round: {active_round}/{total_rounds}) "
            f"| ETC: {eta_str}"
        )
        sys.stdout.flush()

    while len(current_nodes) > 1:
        round_num += 1
        n = len(current_nodes)

        # Pair adjacent nodes; the odd one out passes through untouched.
        pairs = [(current_nodes[i], current_nodes[i + 1]) for i in range(0, n - 1, 2)]
        passthrough = [current_nodes[-1]] if n % 2 == 1 else []

        print(
            f"    [STITCH] Round {round_num}/{total_rounds}: {len(pairs)} parallel merge(s)"
            + (" + 1 passthrough node" if passthrough else "")
            + f" ({n} -> {len(pairs) + len(passthrough)} nodes)",
            flush=True,
        )

        merge_results: List[Optional[str]] = [None] * len(pairs)
        round_p_acc = [0]
        round_c_acc = [0]
        round_start = time.time()

        with progress_lock:
            _render_progress(round_num)

        def _merge_pair(pair_idx: int, left: str, right: str, current_round: int) -> None:
            """Merge one pair; falls back to concatenation if all retries fail."""
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    endpoint = ep_queue.get(timeout=5.0)
                except queue.Empty:
                    continue
                try:
                    with progress_lock:
                        merge_counter[0] += 1
                        cid = merge_counter[0]

                    merged_text, p, c, elapsed = rolling_master_stitch(
                        cid, left, right, endpoint, original_query
                    )

                    with progress_lock:
                        merge_results[pair_idx] = merged_text
                        round_p_acc[0] += p
                        round_c_acc[0] += c
                        merges_done[0] += 1
                        _render_progress(current_round)
                    return

                except Exception as exc:
                    with progress_lock:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        print(
                            f"        [!] Merge round {current_round} pair "
                            f"{pair_idx + 1}/{len(pairs)} attempt {attempt}/{MAX_RETRIES} "
                            f"failed on {endpoint}: {exc}",
                            flush=True,
                        )
                    if attempt == MAX_RETRIES:
                        # Hard fallback: plain concatenation so nothing is lost.
                        with progress_lock:
                            merge_results[pair_idx] = left + "\n\n" + right
                            print(
                                f"        [!] Merge round {current_round} pair "
                                f"{pair_idx + 1} exhausted retries - "
                                f"falling back to raw concatenation.",
                                flush=True,
                            )
                            merges_done[0] += 1
                            _render_progress(current_round)
                finally:
                    ep_queue.put(endpoint)

                if attempt < MAX_RETRIES:
                    time.sleep(2)

            # Terminal guard: every attempt hit queue.Empty, so the retry loop
            # never acquired an endpoint and never reached its own fallback.
            # Without this the slot stays None and BOTH input nodes are
            # silently dropped from the document by the caller.
            with progress_lock:
                if merge_results[pair_idx] is None:
                    merge_results[pair_idx] = left + "\n\n" + right
                    print(
                        f"        [!] Merge round {current_round} pair {pair_idx + 1} "
                        f"never acquired a stitcher slot - falling back to raw concatenation.",
                        flush=True,
                    )
                    merges_done[0] += 1
                    _render_progress(current_round)

        # Fan out every pair in this round simultaneously.
        max_workers = max(1, min(len(pairs), len(endpoint_pool)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_merge_pair, i, left, right, round_num)
                for i, (left, right) in enumerate(pairs)
            ]
            concurrent.futures.wait(futures)

        round_elapsed = round(time.time() - round_start, 2)
        total_stitch_p_tok += round_p_acc[0]
        total_stitch_c_tok += round_c_acc[0]

        sys.stdout.write("\n")
        sys.stdout.flush()
        print(
            f"    [STITCH] Round {round_num}/{total_rounds} complete in {round_elapsed}s "
            f"({round_c_acc[0]} completion tokens).",
            flush=True,
        )

        current_nodes = [r if r is not None else "" for r in merge_results] + passthrough

    total_elapsed = round(time.time() - stitch_start, 2)
    print(
        f"    [+] Stitch complete: {total_merges} merge(s) in {total_elapsed}s. "
        f"Final document: {len(current_nodes[0]):,} chars.",
        flush=True,
    )

    return current_nodes[0], total_stitch_p_tok, total_stitch_c_tok

def execute_continuous_map_reduce(sub_tasks: list, original_query: str, run_dir: Path, decomp_p_tok: int = 0, decomp_c_tok: int = 0) -> tuple:
    """Full Phase 3 map-reduce pipeline with parallelised tree-reduction stitching.

    Map phase    : workers execute all sub-tasks in parallel.
    Reduce phase : chunk-synthesis orchestrators merge worker outputs into
                   synthesized chunks in parallel.
    Stitch phase : tree_reduce_stitch assembles synthesized chunks into the
                   master document on the dedicated stitcher cluster, using
                   O(log N) parallel merge rounds.

    Returns:
        (master_document, worker_p, worker_c, orch_p, orch_c,
         stitch_p, stitch_c, worker_stats_log)

    Stitcher token counts are reported separately from the orchestrator
    totals because they come from a different model on different hardware;
    folding them together would make both figures meaningless.
    """
    if not ORCHESTRATOR_ENDPOINTS or not WORKER_ENDPOINTS or not STITCHER_ENDPOINTS:
        print("\n[!] FATAL: Endpoints not defined for map-reduce cluster (need orchestrator, worker, and stitcher pools).", flush=True)
        sys.exit(1)

    if not sub_tasks:
        return "", 0, 0, decomp_p_tok, decomp_c_tok, 0, 0, []

    total_tasks = len(sub_tasks)
    total_chunks = (total_tasks + SYNTHESIS_CHUNK_SIZE - 1) // SYNTHESIS_CHUNK_SIZE

    print(f"\n[4] CONTINUOUS MAP-REDUCE: Launching parallel workers, chunk synthesis, and tree-reduction stitch...", flush=True)

    worker_queue = queue.Queue()
    w_slot_idx = 1
    for ep in WORKER_ENDPOINTS:
        for _ in range(WORKER_PARALLEL_SLOTS):
            worker_queue.put((ep, f"W-Slot{w_slot_idx:02d}"))
            w_slot_idx += 1

    orch_queue = queue.Queue()
    o_slot_idx = 1
    for ep in ORCHESTRATOR_ENDPOINTS:
        for _ in range(ORCH_PARALLEL_SLOTS):
            orch_queue.put((ep, f"O-Slot{o_slot_idx:02d}"))
            o_slot_idx += 1

    event_queue: queue.Queue = queue.Queue()

    # synthesized_chunks collects chunk texts as they arrive from the orch pool.
    # tree_reduce_stitch assembles them once all chunks are complete.
    synthesized_chunks: Dict[int, str] = {}

    def _map_reduce_worker(tid: int, prompt: str):
        last_result = {
            "id": tid, "status": "error", "result": "Worker execution failed completely.",
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "elapsed": 0, "tps": 0, "slot": "Unknown", "is_estimated": False
        }
        accum_p_tok = 0
        accum_c_tok = 0

        for _ in range(WORKER_RETRIES):
            try:
                endpoint, slot_name = worker_queue.get(timeout=5.0)
            except queue.Empty:
                continue

            try:
                res = process_subtask(tid, prompt, endpoint, slot_name, original_query, run_dir)
                accum_p_tok += res.get("prompt_tokens", 0)
                accum_c_tok += res.get("completion_tokens", 0)

                if res["status"] == "success":
                    if res.get("completion_tokens", 0) > MAX_WORKER_TOKENS:
                        safe_char_limit = MAX_WORKER_TOKENS * 4
                        res["result"] = res["result"][:safe_char_limit] + "\n\n...[OUTPUT TRUNCATED DUE TO LENGTH LIMIT]..."
                        res["completion_tokens"] = MAX_WORKER_TOKENS

                    res["prompt_tokens"] = accum_p_tok
                    res["total_tokens"] = accum_p_tok + res.get("completion_tokens", 0)

                    event_queue.put(("worker", res))
                    return

                last_result = res
            except Exception as e:
                last_result["result"] = f"Failed: {str(e)}"
                last_result["slot"] = slot_name
            finally:
                worker_queue.put((endpoint, slot_name))
            time.sleep(2)

        last_result["prompt_tokens"] = accum_p_tok
        last_result["completion_tokens"] = accum_c_tok
        last_result["total_tokens"] = accum_p_tok + accum_c_tok
        event_queue.put(("worker", last_result))

    def chunk_wrapper(batch_id: int, tasks: list):
        tasks = [t for t in tasks if t is not None]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                endpoint, slot_name = orch_queue.get(timeout=5.0)
            except queue.Empty:
                continue

            try:
                b_id, text, p_tok, c_tok, elap = parallel_chunk_synthesis(batch_id, tasks, endpoint, original_query)
                event_queue.put(("chunk", b_id, text, p_tok, c_tok, elap, slot_name))
                return
            except Exception as e:
                print(f"    [!] Chunk synthesis error on attempt {attempt}: {e}", flush=True)
            finally:
                orch_queue.put((endpoint, slot_name))
            time.sleep(2)

        batch_context = "\n\n".join([f"--- TASK {t['id']}: {t['prompt']} ---\n{t['result']}" for t in tasks])
        fallback_text = f"\n--- [RAW CHUNK {batch_id}] ---\n" + batch_context

        user_prompt = f"ORIGINAL QUERY: {original_query}\n\nREPORTS TO MERGE:\n{batch_context}"
        est_p_tok = estimate_tokens(_PROMPT_PHASE3_SYNTHESIS + user_prompt)
        est_c_tok = estimate_tokens(fallback_text)
        event_queue.put(("chunk", batch_id, fallback_text, est_p_tok, est_c_tok, 0, "Fallback"))

    worker_p_tok, worker_c_tok = 0, 0
    chunk_p_tok, chunk_c_tok = 0, 0
    results_dict: Dict[int, dict] = {}
    worker_stats_log: list = []

    chunks_completed = 0
    workers_finished = 0
    submitted_chunks: Set[int] = set()

    worker_threads = max(1, min(total_tasks, len(WORKER_ENDPOINTS) * WORKER_PARALLEL_SLOTS))
    orch_threads = max(1, min(total_chunks, len(ORCHESTRATOR_ENDPOINTS) * ORCH_PARALLEL_SLOTS))

    start_time_progress = time.time()
    last_event_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_threads) as worker_exec, \
         concurrent.futures.ThreadPoolExecutor(max_workers=orch_threads) as orch_exec:

        for i, task in enumerate(sub_tasks):
            worker_exec.submit(_map_reduce_worker, i + 1, task)

        while chunks_completed < total_chunks:
            try:
                event = event_queue.get(timeout=2.0)
                last_event_time = time.time()
            except queue.Empty:
                if _shutdown_event.is_set():
                    break
                if workers_finished >= total_tasks and chunks_completed < total_chunks:
                    pending_chunks = total_chunks - chunks_completed
                    if pending_chunks > 0:
                        if time.time() - last_event_time > 60.0:
                            print(f"    [!] Watchdog: Stalled for 60s with {pending_chunks} chunks missing. Forcing exit.", flush=True)
                            break
                        # Sweep for any missed chunks during event race condition
                        for chunk_idx in range(1, total_chunks + 1):
                            if chunk_idx not in submitted_chunks:
                                expected_start = (chunk_idx - 1) * SYNTHESIS_CHUNK_SIZE + 1
                                expected_end = min(expected_start + SYNTHESIS_CHUNK_SIZE, total_tasks + 1)
                                available = [results_dict.get(i) for i in range(expected_start, expected_end) if i in results_dict]
                                if available:
                                    submitted_chunks.add(chunk_idx)
                                    orch_exec.submit(chunk_wrapper, chunk_idx, available)
                continue

            if event[0] == "worker":
                workers_finished += 1
                task_res = event[1]

                if task_res is not None:
                    tid = task_res["id"]
                    results_dict[tid] = task_res
                    worker_stats_log.append(task_res)

                    worker_p_tok += task_res["prompt_tokens"]
                    worker_c_tok += task_res["completion_tokens"]

                    chunk_idx = (tid - 1) // SYNTHESIS_CHUNK_SIZE + 1
                    expected_start = (chunk_idx - 1) * SYNTHESIS_CHUNK_SIZE + 1
                    expected_end = min(expected_start + SYNTHESIS_CHUNK_SIZE, total_tasks + 1)

                    chunk_ready = all(i in results_dict for i in range(expected_start, expected_end))

                    if chunk_ready and chunk_idx not in submitted_chunks:
                        submitted_chunks.add(chunk_idx)
                        chunk_tasks = [results_dict.get(i) for i in range(expected_start, expected_end)]
                        orch_exec.submit(chunk_wrapper, chunk_idx, chunk_tasks)

                if workers_finished == total_tasks:
                    for chunk_idx in range(1, total_chunks + 1):
                        if chunk_idx not in submitted_chunks:
                            expected_start = (chunk_idx - 1) * SYNTHESIS_CHUNK_SIZE + 1
                            expected_end = min(expected_start + SYNTHESIS_CHUNK_SIZE, total_tasks + 1)
                            available = [results_dict.get(i) for i in range(expected_start, expected_end) if i in results_dict]
                            if available:
                                submitted_chunks.add(chunk_idx)
                                orch_exec.submit(chunk_wrapper, chunk_idx, available)

            elif event[0] == "chunk":
                _, b_id, text, p_tok, c_tok, elap, slot_name = event
                synthesized_chunks[b_id] = text
                chunk_p_tok += p_tok
                chunk_c_tok += c_tok
                chunks_completed += 1

            bar_len = 30
            total_tasks_safe = total_tasks if total_tasks > 0 else 1
            filled = int((workers_finished / total_tasks_safe) * bar_len)
            bar = '#' * filled + '-' * (bar_len - filled)
            percent = int((workers_finished / total_tasks_safe) * 100)
            eta_str = _format_eta(start_time_progress, workers_finished, total_tasks)
            sys.stdout.write(f"\r    [+] Map-Reduce Progress: [{bar}] {percent}% (Workers: {workers_finished}/{total_tasks} | Chunks: {chunks_completed}/{total_chunks}) | ETC: {eta_str}")
            sys.stdout.flush()

    print()

    # ------------------------------------------------------------------
    # Stitch phase: hierarchical parallel tree reduction on the dedicated
    # stitcher cluster. Chunks are assembled in chunk_id order so document
    # coherence is maintained; the tree reduction halves the node list each
    # round and uses every available stitcher endpoint concurrently.
    # ------------------------------------------------------------------
    ordered_chunk_texts = [
        synthesized_chunks[cid] for cid in sorted(synthesized_chunks.keys())
    ]

    if len(ordered_chunk_texts) < total_chunks:
        print(f"    [!] WARNING: Map-Reduce completed but {total_chunks - len(ordered_chunk_texts)} chunks were lost.", flush=True)

    print(
        f"\n[5] TREE-REDUCTION STITCH: Assembling {len(ordered_chunk_texts)} chunk(s) "
        f"across {len(STITCHER_ENDPOINTS)} stitcher endpoint(s) x {STITCH_PARALLEL_SLOTS} slot(s) "
        f"[{STITCHER_MODEL}]...",
        flush=True,
    )

    master_document, stitch_p_tok, stitch_c_tok = tree_reduce_stitch(
        ordered_chunk_texts, original_query
    )

    total_orch_p = chunk_p_tok + decomp_p_tok
    total_orch_c = chunk_c_tok + decomp_c_tok

    return (master_document, worker_p_tok, worker_c_tok, total_orch_p, total_orch_c,
            stitch_p_tok, stitch_c_tok, worker_stats_log)

# ==============================================================================
# Phase 4: Distributed Parallel Post-Processing
# ==============================================================================

def extract_and_protect_blocks(markdown_text: str) -> Tuple[str, Dict[str, str]]:
    protected_blocks = {}
    block_counter = 0

    def replacer(match: re.Match) -> str:
        nonlocal block_counter
        placeholder = f"[[PROTECTED_CODE_BLOCK{block_counter:03d}]]"
        protected_blocks[placeholder] = match.group(0)
        block_counter += 1
        return placeholder

    code_pattern = re.compile(r'(`{3,})[^\r\n]*\r?\n[\s\S]*?\r?\n\1(?!`)')
    text_without_code = code_pattern.sub(replacer, markdown_text)

    table_pattern = re.compile(r'## Execution Telemetry[\s\S]*?(?=\r?\n## |\Z)')
    table_match = table_pattern.search(text_without_code)

    if table_match:
        placeholder = "[[PROTECTED_TELEMETRY_TABLE]]"
        protected_blocks[placeholder] = table_match.group(0)
        text_without_code = text_without_code[:table_match.start()] + f"\n\n{placeholder}\n"

    return text_without_code, protected_blocks

def semantic_deduplication(chunk_text: str, chunk_id: int, total_chunks: int, endpoint: str, slot_name: str) -> str:
    placeholder_pattern = r'\[\[PROTECTED_[A-Z_]+?\d{3,}\]\]|\[\[PROTECTED_TELEMETRY_TABLE\]\]'
    chunk_inventory = re.findall(placeholder_pattern, chunk_text)

    if len(chunk_text) > MAX_CONTEXT_CHARS:
        chunk_text = chunk_text[:MAX_CONTEXT_CHARS] + "\n\n...[CONTENT TRUNCATED FOR CONTEXT LIMITS]..."

    client = OpenAI(base_url=endpoint, api_key=ORCH_API_KEY, timeout=1200.0, max_retries=2)

    system_prompt = _PROMPT_PHASE4_DEDUP

    if chunk_inventory:
        inventory_str = ", ".join(chunk_inventory)
        system_prompt += (
            f"\n\nCRITICAL ARTIFACT INVENTORY:\n"
            f"This specific text section contains the following protected placeholders: {inventory_str}\n"
            f"You MUST include EVERY SINGLE ONE of these exact placeholder strings in your rewritten output. "
            f"Even if you summarize the surrounding text, do NOT drop these placeholders. They represent vital code blocks."
        )

    try:
        response = client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": chunk_text}],
            temperature=0.1, max_tokens=16384, presence_penalty=0.2
        )
        distilled_text = enforce_ascii(response.choices[0].message.content.strip())

        if chunk_inventory:
            missing_placeholders = [p for p in chunk_inventory if p not in distilled_text]
            if missing_placeholders:
                distilled_text += "\n\n### Recovered Chunk Artifacts\n" + "\n\n".join(missing_placeholders)

        return distilled_text
    except Exception:
        return chunk_text

def parallel_edit_chunks(chunks: List[str]) -> str:
    if not ORCHESTRATOR_ENDPOINTS:
        print("    [!] WARNING: No orchestrator endpoints defined. Skipping deduplication.", flush=True)
        return "\n\n".join(chunks)

    total_chunks = len(chunks)
    endpoint_queue = queue.Queue()
    for ep in ORCHESTRATOR_ENDPOINTS:
        parsed = urllib.parse.urlparse(ep)
        ip_tail = parsed.hostname if parsed.hostname else "local"
        if parsed.port:
            ip_tail += f":{parsed.port}"

        for i in range(CONCURRENT_SLOTS_PER_ENDPOINT):
            slot_name = f"Node-{ip_tail}-Slot-{i+1}"
            endpoint_queue.put((ep, slot_name))

    results = [""] * total_chunks
    def _edit_chunk_worker(chunk_idx: int, chunk_content: str):
        endpoint, slot_name = None, None
        while not _shutdown_event.is_set():
            try:
                endpoint, slot_name = endpoint_queue.get(timeout=2.0)
                break
            except queue.Empty:
                continue
        if endpoint is None:
            return chunk_content
        try:
            return semantic_deduplication(chunk_content, chunk_idx + 1, total_chunks, endpoint, slot_name)
        except Exception:
            return chunk_content
        finally:
            endpoint_queue.put((endpoint, slot_name))

    total_slots = len(ORCHESTRATOR_ENDPOINTS) * CONCURRENT_SLOTS_PER_ENDPOINT
    pool_size = max(1, total_slots)

    print(f"    [*] Semantic Deduplication: Refining {total_chunks} chunk(s) across {pool_size} slot(s)...", flush=True)

    start_time_progress = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
        future_to_idx = {executor.submit(_edit_chunk_worker, i, chunk): i for i, chunk in enumerate(chunks)}
        completed = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

            completed += 1
            bar_len = 30
            total_chunks_safe = total_chunks if total_chunks > 0 else 1
            filled = int((completed / total_chunks_safe) * bar_len)
            bar = '#' * filled + '-' * (bar_len - filled)
            percent = int((completed / total_chunks_safe) * 100)
            eta_str = _format_eta(start_time_progress, completed, total_chunks)
            sys.stdout.write(f"\r    [+] Progress: [{bar}] {percent}% ({completed}/{total_chunks}) | ETC: {eta_str}")
            sys.stdout.flush()

    print()

    return "\n\n".join(results)

def section_boundary_smoothing(full_skeleton: str, global_inventory: List[str], endpoint: str) -> str:
    client = OpenAI(base_url=endpoint, api_key=ORCH_API_KEY, timeout=WORKER_TIMEOUT_SECS, max_retries=1)
    system_prompt = _PROMPT_PHASE4_SMOOTH

    if global_inventory:
        system_prompt += f"\n\nCRITICAL ARTIFACT INVENTORY: {', '.join(global_inventory)}\nYou MUST retain EVERY SINGLE placeholder."

    if len(full_skeleton) > MAX_CONTEXT_CHARS:
        full_skeleton = full_skeleton[:MAX_CONTEXT_CHARS] + "\n\n...[CONTENT TRUNCATED FOR CONTEXT LIMITS]..."

    try:
        response = client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": full_skeleton}],
            temperature=0.2, max_tokens=32768, presence_penalty=0.1
        )
        return enforce_ascii(response.choices[0].message.content.strip())
    except Exception:
        return full_skeleton

def header_unification_pass(smoothed_skeleton: str, global_inventory: List[str], endpoint: str) -> str:
    client = OpenAI(base_url=endpoint, api_key=ORCH_API_KEY, timeout=WORKER_TIMEOUT_SECS, max_retries=1)
    system_prompt = _PROMPT_PHASE4_UNIFY

    if global_inventory:
        system_prompt += f"\n\nCRITICAL ARTIFACT INVENTORY: {', '.join(global_inventory)}\nYou MUST retain EVERY SINGLE placeholder."

    if len(smoothed_skeleton) > MAX_CONTEXT_CHARS:
        smoothed_skeleton = smoothed_skeleton[:MAX_CONTEXT_CHARS] + "\n\n...[CONTENT TRUNCATED FOR CONTEXT LIMITS]..."

    try:
        response = client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": smoothed_skeleton}],
            temperature=0.2, max_tokens=32768, presence_penalty=0.1
        )
        return enforce_ascii(response.choices[0].message.content.strip())
    except Exception:
        return smoothed_skeleton

def global_consolidation_pass(full_skeleton: str, global_inventory: List[str], endpoint: str) -> str:
    print("    [*] Executive Editor Pass 1: Section Boundary Smoothing...", flush=True)
    smoothed_text = section_boundary_smoothing(full_skeleton, global_inventory, endpoint)

    print("    [*] Executive Editor Pass 2: Header Unification...", flush=True)
    final_text = header_unification_pass(smoothed_text, global_inventory, endpoint)

    missing_placeholders = [p for p in global_inventory if p not in final_text]
    if missing_placeholders:
        final_text += "\n\n### Recovered Global Artifacts\n" + "\n\n".join(missing_placeholders)

    return final_text

def reassemble_document(distilled_text: str, protected_blocks: Dict[str, str]) -> str:
    final_text = distilled_text
    for placeholder, original_content in protected_blocks.items():
        if placeholder in final_text:
            final_text = final_text.replace(placeholder, original_content)
        else:
            final_text += f"\n\n### Orphaned Artifact\n{original_content}"
    return final_text

# ==============================================================================
# Phase 5: Automatic Unittests
# ==============================================================================

def _format_execution_report_as_markdown(report_data: list) -> str:
    if not report_data: return ""
    lines = ["## Test Execution Status\n", "| Artifact | Language | Status | Detail |", "|---|---|---|---|"]
    for res in report_data:
        lines.append(f"| {res.get('filename', '')} | {res.get('language', '')} | **{res.get('status', '')}** | {res.get('message', '')} |")
    return "\n".join(lines) + "\n\n"

def _safe_output_path(detected_filename: str, output_dir: Path, seen_paths: Set[str]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalised = detected_filename.replace("\\", "/")
    parts = [p for p in PurePosixPath(normalised).parts if p not in ("", ".", "..")]
    if not parts:
        parts = ["artifact.txt"]
    safe_parts = parts[-2:] if len(parts) >= 2 else parts
    candidate = output_dir.joinpath(*safe_parts)
    candidate.parent.mkdir(parents=True, exist_ok=True)

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1

    while str(candidate.resolve()) in seen_paths or candidate.exists():
        candidate = candidate.parent / f"{stem}_{counter}{suffix}"
        counter += 1

    seen_paths.add(str(candidate.resolve()))
    return candidate

def _strip_markdown_fences(text: str) -> str:
    return re.sub(r'^```[^\r\n]*\r?\n?|^```\s*$', '', text, flags=re.MULTILINE).strip()

def _extract_error_line(output: str, lang: str) -> str:
    lines = [l for l in output.splitlines() if l.strip()]
    if not lines:
        return "no output"
    filtered_lines = [l for l in lines if not re.match(r'^\d+ (failed|error|passed|warning|deselected)', l.strip())]
    if not filtered_lines:
        return "(pytest summary only - no error detail captured)"
    if lang in ("python", "py"):
        for line in filtered_lines:
            if line.strip().startswith("E "):
                return line.strip()[2:].strip()
        err_regex = re.compile(r'^([A-Z][a-zA-Z0-9_]+Error|[A-Z][a-zA-Z0-9_]+Exception|Exception|FAIL:|ERROR:)( |:)')
        for line in filtered_lines:
            if err_regex.match(line.strip()):
                return line.strip()
        for line in filtered_lines:
            if line.strip().startswith("FAILED "):
                return line.strip()
    if lang in ("c", "cpp"):
        return filtered_lines[0].strip()
    if len(filtered_lines) >= 2:
        return f"{filtered_lines[-2].strip()} | {filtered_lines[-1].strip()}"
    return filtered_lines[-1].strip()

def _sanitize_requirements_file(filepath: Path) -> None:
    try:
        with open(filepath, 'r', encoding='utf-8', errors="ignore") as f:
            lines = f.readlines()
        cleaned_lines = []
        valid_req_pattern = re.compile(r'^([A-Za-z0-9_\-\.\[\]]+\s*(==|>=|<=|~=|!=|<|>|@).*|-[re]\s+.*|#.*)$')
        for line in lines:
            s_line = line.strip()
            if not s_line:
                cleaned_lines.append(line)
                continue
            if valid_req_pattern.match(s_line) or (s_line.isalnum() or re.match(r'^[A-Za-z0-9_\-\.\[\]]+$', s_line)):
                cleaned_lines.append(line)
        with open(filepath, 'w', encoding='utf-8', errors="ignore") as f:
            f.writelines(cleaned_lines)
    except Exception as e:
        print(f"Warning: Could not sanitize requirements file {filepath}: {e}")

def extract_code_blocks(md_content: str, output_dir: Union[str, Path]) -> list:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lines = md_content.splitlines()
    i = 0
    in_block = False
    current_block: List[str] = []
    current_lang = ""
    detected_filename: Optional[str] = None
    last_header: Optional[str] = None
    file_counter = 1
    extracted_artifacts = []
    seen_paths = set()
    active_fence_len = 0

    while i < len(lines):
        line = lines[i]

        if "</file>" in line:
            detected_filename = None
            i += 1
            continue

        xml_match = re.search(r'<file path="([^"]+)">', line)
        if xml_match:
            if detected_filename is not None:
                print(f"    [!] Warning: Missing </file>. Overwriting filename '{detected_filename}' with '{xml_match.group(1)}'.", flush=True)
            detected_filename = xml_match.group(1)
            i += 1
            continue

        header_match = re.match(r'^###?\s+([a-zA-Z0-9_\-\.]+.*)$', line)
        if header_match and not in_block:
            potential_name = header_match.group(1).strip()
            embedded_match = re.search(r'[\(\`]([a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]+)[\)\`]', potential_name)
            if embedded_match:
                candidate = embedded_match.group(1)
            elif "." in potential_name or potential_name.lower() in ("dockerfile", "makefile"):
                candidate = potential_name
            else:
                candidate = None
            if candidate is not None:
                detected_filename = candidate
            last_header = potential_name
            i += 1
            continue

        stripped_line = line.lstrip()
        if not in_block and stripped_line.startswith("```"):
            in_block = True
            active_fence_len = len(stripped_line) - len(stripped_line.lstrip('`'))
            current_lang = stripped_line.lstrip('`').strip()
            current_block = []
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith("# ") and "." in next_line:
                    detected_filename = next_line[2:].strip()
                    i += 1
        elif in_block:
            if re.match(rf'^`{{{active_fence_len}}}\s*$', stripped_line):
                in_block = False
                content = "\n".join(current_block)
                if not detected_filename:
                    ext = current_lang if current_lang else "txt"
                    ext_map = {"python": "py", "bash": "sh", "yaml": "yml", "dockerfile": "Dockerfile"}
                    ext = ext_map.get(ext.lower(), ext)
                    base_name = last_header.replace(" ", "_").lower() if last_header else f"artifact_{file_counter}"
                    detected_filename = f"{base_name}.{ext}" if ext.lower() != "dockerfile" else "Dockerfile"
                    file_counter += 1
                detected_filename = re.sub(r'[()\[\]{}]', '', detected_filename)
                detected_filename = detected_filename.replace(" ", "_")

                file_path = _safe_output_path(detected_filename, output_path, seen_paths)

                with open(file_path, "w", encoding="ascii") as f:
                    f.write(content + "\n")
                extracted_artifacts.append({
                    "filename": file_path.name,
                    "relative_path": str(file_path.relative_to(output_path)),
                    "language": current_lang,
                    "filepath": str(file_path),
                    "content": content,
                })
                detected_filename = None
                last_header = None
                current_lang = ""
                current_block = []
                active_fence_len = 0
            else:
                current_block.append(line)
        i += 1

    if in_block and current_block:
        content = "\n".join(current_block)
        fallback_name = f"artifact_{file_counter}_partial.txt"
        file_path = output_path / fallback_name
        with open(file_path, "w", encoding="ascii") as f:
            f.write(content + "\n")
    return extracted_artifacts

def request_unittests_from_worker(artifact: dict, endpoint_queue: queue.Queue, test_output_dir: Path, progress_lock: threading.Lock, progress_state: dict) -> Optional[dict]:
    if artifact["language"].lower() not in {"python", "py", "cpp", "c", "bash", "sh"}:
        print(f"    [!] Skipping unsupported unit-test language generation: {artifact['language']}", flush=True)
        return None

    try:
        endpoint_url = endpoint_queue.get(timeout=5.0)
    except queue.Empty:
        return None

    try:
        code_content = artifact['content']
        if len(code_content) > MAX_CONTEXT_CHARS:
            code_content = code_content[:MAX_CONTEXT_CHARS] + "\n\n...[CONTENT TRUNCATED FOR CONTEXT LIMITS]..."

        prompt = (
            f"File: {artifact['filename']}\n"
            f"```{artifact['language']}\n{code_content}\n```"
        )
        payload = {
            "model": WORKER_MODEL,
            "messages": [
                {"role": "system", "content": _PROMPT_PHASE5_UNITTEST},
                {"role": "user", "content": prompt},
            ],
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
            "frequency_penalty": LLM_FREQUENCY_PENALTY,
            "presence_penalty": LLM_PRESENCE_PENALTY,
            "max_tokens": MAX_OUTPUT_TOKENS,
        }
        generation_metadata = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(endpoint_url, json=payload, timeout=WORKER_TIMEOUT_SECS)
                response.raise_for_status()
                result = response.json()
                choices = result.get("choices")

                if not choices:
                    print(f"    [!] Endpoint {endpoint_url} returned no choices (attempt {attempt}). Raw: {result}", flush=True)
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER))
                        continue
                    break

                test_code = choices[0].get("message", {}).get("content", "")
                if not test_code:
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER))
                    continue

                test_code = enforce_ascii(_strip_markdown_fences(test_code))
                test_output_dir.mkdir(parents=True, exist_ok=True)
                test_filename = f"test_{artifact['filename']}"
                test_filepath = test_output_dir / test_filename

                with open(test_filepath, "w", encoding="ascii") as f:
                    f.write(test_code + "\n")

                with progress_lock:
                    progress_state["done"] += 1
                    eta_str = _format_eta(progress_state["start_time"], progress_state["done"], progress_state["total"])
                    print(f"    [+] [{endpoint_url}] Generated tests ({progress_state['done']}/{progress_state['total']}) | ETC: {eta_str} -> {test_filename}")

                generation_metadata = {
                    "filename": test_filename,
                    "test_filepath": str(test_filepath),
                    "language": artifact["language"],
                    "artifact_filepath": artifact["filepath"],
                }
                break

            except requests.exceptions.RequestException:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER))

        return generation_metadata
    finally:
        endpoint_queue.put(endpoint_url)

def execute_test_artifact(test_meta: dict) -> dict:
    lang = test_meta["language"].lower()
    test_path = Path(test_meta["test_filepath"]).resolve()
    artifact_path = Path(test_meta["artifact_filepath"]).resolve()
    result = {"filename": test_meta["filename"], "language": lang, "status": "UNKNOWN", "message": ""}
    try:
        if lang in ("python", "py"):
            env = os.environ.copy()
            src_dir = str(test_path.parent)
            existing_pypath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(artifact_path.parent), src_dir, existing_pypath]))

            cmd = ["python", "-m", "pytest", "-p", "no:cacheprovider", "--no-header", "--tb=short", "-q", str(test_path)]
            start_time = time.time()
            res = subprocess.run(cmd, capture_output=True, encoding="ascii", errors="ignore", timeout=45, cwd=str(test_path.parent), env=env)
            duration = time.time() - start_time
            if res.returncode == 0:
                result["status"], result["message"] = "PASSED", f"OK ({duration:.2f}s)"
            else:
                result["status"], result["message"] = "FAILED", _extract_error_line(res.stderr + res.stdout, lang)
        elif lang in ("bash", "sh"):
            start_time = time.time()
            res = subprocess.run(["bash", str(test_path)], capture_output=True, encoding="ascii", errors="ignore", timeout=30)
            duration = time.time() - start_time
            if res.returncode == 0:
                result["status"], result["message"] = "PASSED", f"OK ({duration:.2f}s)"
            else:
                result["status"], result["message"] = "FAILED", _extract_error_line(res.stderr + res.stdout, lang)
        elif lang in ("c", "cpp"):
            compiler = "gcc" if lang == "c" else "g++"
            binary_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
                    binary_path = Path(tmp.name)

                compile_cmd = [compiler, "-I", str(artifact_path.parent), str(test_path), "-o", str(binary_path)]
                comp_res = subprocess.run(compile_cmd, capture_output=True, encoding="ascii", errors="ignore", timeout=20)
                if comp_res.returncode != 0:
                    result["status"], result["message"] = "COMPILE_ERROR", _extract_error_line(comp_res.stderr, lang)
                    return result

                start_time = time.time()
                res = subprocess.run([str(binary_path)], capture_output=True, encoding="ascii", errors="ignore", timeout=30)
                duration = time.time() - start_time
                if res.returncode == 0:
                    result["status"], result["message"] = "PASSED", f"OK ({duration:.2f}s)"
                else:
                    result["status"], result["message"] = "FAILED", _extract_error_line(res.stderr + res.stdout, lang)
            finally:
                if binary_path and binary_path.exists():
                    binary_path.unlink()
        else:
            result["status"], result["message"] = "SKIPPED", f"No environment definition for: {lang}"
    except subprocess.TimeoutExpired:
        result["status"], result["message"] = "TIMEOUT", "Execution threshold exceeded"
    except Exception as exc:
        result["status"], result["message"] = "ERROR", str(exc)
    return result

def run_phase5_automatic_unittests(source_path: Path):
    print(f"\n[PHASE 5] STARTING AUTOMATED UNITTEST PIPELINE", flush=True)
    if not source_path.exists():
        print(f"Error: Could not find {source_path}.", flush=True)
        return
    OUTPUT_WORKSPACE = source_path.parent
    EXTRACTED_ARTIFACTS_DIR = OUTPUT_WORKSPACE / "artifacts" / "extracted"
    TEST_OUTPUT_DIR = OUTPUT_WORKSPACE / "tests"
    REPORT_OUTPUT_DIR = OUTPUT_WORKSPACE / "reports"

    endpoint_concurrency = CONCURRENT_REQS_PER_ENDPOINT
    total_gen_workers = len(TEST_WORKER_ENDPOINTS) * endpoint_concurrency
    endpoint_queue: queue.Queue = queue.Queue()
    for ep in TEST_WORKER_ENDPOINTS:
        for _ in range(endpoint_concurrency):
            endpoint_queue.put(ep)

    md_content = read_file_content_safe(source_path)
    if md_content is None:
        print(f"Error: Could not read {source_path}.", flush=True)
        return

    if re.search(r'\[\[PROTECTED_[A-Z_]+\d+\]\]', md_content):
        print("    [!] WARNING: POLISHED_SYNTHESIS.md contains unresolved placeholder tokens. Phase 4 reassembly may have failed.", flush=True)

    artifacts = extract_code_blocks(md_content, EXTRACTED_ARTIFACTS_DIR)
    valid_langs = {"python", "py", "cpp", "c", "bash", "sh"}
    testable_artifacts = [a for a in artifacts if a["language"].lower() in valid_langs]

    progress_lock = threading.Lock()
    progress_state = {"done": 0, "total": len(testable_artifacts), "start_time": time.time()}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=total_gen_workers)
    generated_tests: list = []

    try:
        futures = [executor.submit(request_unittests_from_worker, artifact, endpoint_queue, TEST_OUTPUT_DIR, progress_lock, progress_state) for artifact in testable_artifacts]
        for future in concurrent.futures.as_completed(futures):
            try:
                test_meta = future.result()
                if test_meta:
                    generated_tests.append(test_meta)
            except Exception:
                pass
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    req_files = [a for a in artifacts if "requirements" in a["filename"].lower() and a["filename"].endswith(".txt")]
    for req in req_files:
        req_path = Path(req["filepath"]).resolve()
        _sanitize_requirements_file(req_path)
        try:
            res = subprocess.run(["python", "-m", "pip", "install", "--break-system-packages", "-r", str(req_path)], capture_output=True, encoding="ascii", errors="ignore", timeout=120)
            if res.returncode != 0:
                print(f"    [!] Warning: pip install failed for {req['filename']}. Stderr:\n{res.stderr}", flush=True)
        except Exception as e:
            print(f"    [!] Warning: pip install exception for {req['filename']}: {e}", flush=True)

    execution_results: list = []
    if generated_tests:
        exec_executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_EXEC_WORKERS)
        try:
            exec_futures = [exec_executor.submit(execute_test_artifact, tm) for tm in generated_tests]
            for future in concurrent.futures.as_completed(exec_futures):
                try:
                    res = future.result()
                    execution_results.append(res)
                except Exception:
                    pass
        finally:
            exec_executor.shutdown(wait=True, cancel_futures=True)

    if execution_results:
        REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        json_report_path = REPORT_OUTPUT_DIR / "execution_report.json"
        csv_report_path  = REPORT_OUTPUT_DIR / "execution_report.csv"
        with open(json_report_path, "w", encoding="ascii") as f:
            json.dump(execution_results, f, indent=4)
        with open(csv_report_path, "w", newline="", encoding="ascii") as f:
            writer = csv.DictWriter(f, fieldnames=EXECUTION_RESULT_FIELDS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(execution_results)

# ==============================================================================
# Phase 6: Todo Project Distillation
# ==============================================================================

def extract_project_tasks(project_name: str, raw_content: str) -> Optional[str]:
    client = OpenAI(base_url=ORCHESTRATOR_ENDPOINTS[0], api_key=ORCH_API_KEY, timeout=WORKER_TIMEOUT_SECS, max_retries=0)
    try:
        response = client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[
                {"role": "system", "content": _PROMPT_PHASE6_DISTILL},
                {"role": "user", "content": f"Extract actionable tasks, test outcomes, and relevant artifacts from this {project_name} documentation:\n\n{raw_content}"}
            ],
            temperature=0.2, max_tokens=8192
        )
        return enforce_ascii(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"[!] Orchestrator inference failed: {e}", flush=True)
        return None

def run_phase6_project_distillation(project_dir: Path):
    print(f"\n[PHASE 6] STARTING PROJECT DISTILLATION", flush=True)
    project_name = project_dir.name

    exclude_dirs = {"tests", "tasks", "artifacts", "reports"}
    p6_exclude_names = {"DISTILLED_TASKS", "project_state", "FINAL_SYNTHESIS", "POLISHED_SYNTHESIS"}

    raw_files = []
    seen_paths = set()

    report_json_path = project_dir / "reports" / "execution_report.json"
    if report_json_path.exists():
        raw_files.append(report_json_path)
        seen_paths.add(report_json_path.resolve())

    for root, dirs, files in os.walk(project_dir, followlinks=False):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not (root_path / d).is_symlink()]
        for f in files:
            file_path = root_path / f
            if file_path.is_symlink():
                continue

            if file_path.stem in p6_exclude_names:
                continue

            # If a report JSON exists, it was added above. Skip any OTHER json files.
            if file_path.suffix == ".json" and file_path.resolve() != report_json_path.resolve():
                continue

            if file_path.suffix in {".txt", ".md", ".csv", ".json"}:
                resolved_path = file_path.resolve()
                if resolved_path not in seen_paths:
                    raw_files.append(file_path)
                    seen_paths.add(resolved_path)

    if not raw_files:
        print(f"[{project_name}] No raw documentation or test logs found. Skipping.", flush=True)
        return

    raw_files.sort(key=lambda x: 0 if "execution_report" in x.name else 1)

    aggregated_content = []
    for file_path in raw_files:
        try:
            if file_path.suffix == '.json' and 'execution_report' in file_path.name:
                with open(file_path, "r", encoding="utf-8", errors="strict") as f:
                    report_data = json.load(f)
                    md_table = _format_execution_report_as_markdown(report_data)
                    aggregated_content.append(f"--- SOURCE: {file_path.name} ---\n{md_table}")
            else:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    aggregated_content.append(f"--- SOURCE: {file_path.name} ---\n{f.read()}")
        except Exception as e:
            print(f"    [!] Could not parse {file_path.name}: {e}", flush=True)

    full_text = "\n\n".join(aggregated_content)
    if len(full_text) > MAX_CONTEXT_CHARS:
        full_text = full_text[:MAX_CONTEXT_CHARS] + "\n\n...[CONTENT TRUNCATED FOR CONTEXT LIMITS]..."

    distilled_markdown = extract_project_tasks(project_name, full_text)
    if distilled_markdown:
        output_file = project_dir / "DISTILLED_TASKS.md"
        try:
            with open(output_file, "w", encoding="ascii") as f:
                f.write(f"# Distilled Tasks: {project_name}\n\n{distilled_markdown}\n")
            print(f"[{project_name}] Successfully saved distilled tasks to {output_file.name}", flush=True)
        except Exception as e:
            print(f"[{project_name}] Failed to save output file: {e}", flush=True)

# ==============================================================================
# Pipeline Executor (Main)
# ==============================================================================

def signal_handler(sig, frame):
    if _shutdown_event.is_set():
        print("\n[!] Force exit triggered.", flush=True)
        os._exit(1)

    _shutdown_event.set()
    print("\n[!] Graceful shutdown requested (SIGINT/SIGTERM). Awaiting active threads to abort... (Press Ctrl+C again to force exit)", flush=True)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="End-to-End Agentic Content Generation Pipeline")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-p", "--prompt", type=str, help="Direct prompt for the full pipeline.")
    group.add_argument("-f", "--file", type=str, help="Path to a text file containing the prompt.")
    group.add_argument("-g", "--git", type=str, help="Git repository URL to clone and analyse as the pipeline input.")

    parser.add_argument("--focus", type=str, default="", help="Optional analysis focus applied during git repository intake (used with -g).")
    parser.add_argument("--git-path", type=str, default="", help="Specific file or folder path within the repository to process (used with -g).")
    parser.add_argument("-d", "--dir", type=str, default="run_data", help="Base directory for outputs.")
    parser.add_argument("-c", "--category", type=str, default="projects", help="Category folder.")
    parser.add_argument("-r", "--resume", action="store_true", help="Resume pipeline from the furthest completed artifact in the target directory.")

    args = parser.parse_args()

    # manual check required since -r is outside the mutually exclusive group
    if args.resume and (args.prompt or args.file or args.git):
        parser.error("--resume cannot be combined with -p, -f, or -g.")
    if not args.resume and not args.prompt and not args.file and not args.git:
        parser.error("Must provide a prompt (-p), a prompt file (-f), a git URL (-g), or use the resume flag (-r).")
    if args.focus and not args.git:
        parser.error("--focus can only be used together with -g/--git.")
    if args.git_path and not args.git:
        parser.error("--git-path can only be used together with -g/--git.")
    if args.git and not validate_git_url(args.git):
        parser.error(f"'{args.git}' does not look like a valid git URL or resolves to a blocked private/IMDS network address.")

    target_prompt = ""
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"[!] Error: Prompt file '{args.file}' does not exist.")
            sys.exit(1)
        target_prompt = read_file_content_safe(file_path)
        if target_prompt is None:
            print(f"[!] Error: Could not read prompt file '{args.file}'.")
            sys.exit(1)
        target_prompt = enforce_ascii(target_prompt.strip())
    elif args.prompt:
        target_prompt = args.prompt

    work_dir = Path(args.dir).resolve()
    category_dir = work_dir / args.category

    if args.resume:
        if not category_dir.exists():
            print(f"[!] Resume failed: Category directory {category_dir} does not exist.")
            sys.exit(1)

        run_dirs = [d for d in category_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        valid_run_dirs = [d for d in run_dirs if list(d.glob("*.md"))]

        if not valid_run_dirs:
            print(f"[!] Resume failed: No valid run directories with artifacts found in {category_dir}.")
            sys.exit(1)

        target_directory = max(valid_run_dirs, key=os.path.getmtime)
        print(f"[*] Resume detected. Binding to existing run directory: {target_directory.name}")
    else:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        target_directory = category_dir / run_id
        target_directory.mkdir(parents=True, exist_ok=True)

    # State tracking
    raw_filepath = None
    distilled_filepath = None
    final_file_path = target_directory / "FINAL_SYNTHESIS.md"
    output_path = target_directory / "POLISHED_SYNTHESIS.md"
    report_path = target_directory / "reports" / "execution_report.json"
    distilled_tasks_path = target_directory / "DISTILLED_TASKS.md"

    if args.resume:
        md_files = list(target_directory.glob("*.md"))
        resume_exclude_names = {"FINAL_SYNTHESIS.md", "POLISHED_SYNTHESIS.md", "DISTILLED_TASKS.md"}
        valid_raw = [f for f in md_files if re.match(r'^\d{8}_\d{6}_.*\.md$', f.name) and f.name not in resume_exclude_names]

        if valid_raw:
            raw_filepath = max(valid_raw, key=os.path.getmtime)
            print(f"[*] Base raw file detected: {raw_filepath.name}")

            expected_distilled = raw_filepath.parent / f"{raw_filepath.stem}_distilled.md"
            if expected_distilled.exists():
                distilled_filepath = expected_distilled
            else:
                print(f"[*] Note: Distilled file missing. Will re-distill from {raw_filepath.name}.")
        else:
            print("[!] Resume flag passed but no valid base raw file found in target directory. Aborting.")
            sys.exit(1)

    # ---------------------------------------------------------
    # PHASE 0/1: INTAKE (GIT REPOSITORY OR GENERATED CONTENT)
    if args.resume and raw_filepath and raw_filepath.exists():
        print(f"[PHASE 1] Bypassed. Resuming from existing raw file: {raw_filepath.name}")
    elif args.git:
        raw_filepath = ingest_git_repository(args.git, target_directory, args.focus, args.git_path)
    else:
        raw_filepath = generate_content(target_prompt, target_directory)

    # ---------------------------------------------------------
    # PHASE 2: DISTILL
    if args.resume and distilled_filepath and distilled_filepath.exists():
        print(f"[PHASE 2] Bypassed. Resuming from existing distilled file: {distilled_filepath.name}")
    else:
        raw_content = read_file_content_safe(raw_filepath)
        if raw_content is None:
            print(f"[!] Fatal: Could not read {raw_filepath}.", flush=True)
            sys.exit(1)
        actionable_tasks = distill_document(raw_content)
        distilled_filepath = save_distilled_output(actionable_tasks, raw_filepath)

    # ---------------------------------------------------------
    # PHASE 3: ORCHESTRATE MAP-REDUCE
    if args.resume and final_file_path.exists():
        print(f"[PHASE 3] Bypassed. Resuming from existing FINAL_SYNTHESIS.md")
    else:
        target_query = read_file_content_safe(distilled_filepath)
        if target_query is None:
            print(f"[!] Fatal: Could not read {distilled_filepath}.", flush=True)
            sys.exit(1)

        master_start_time = time.time()
        fragments, p_tok, c_tok = decompose_to_atomic_pieces(target_query)

        export_to_split_files(fragments, target_directory)
        (final_output, w_p, w_c, o_p, o_c, s_p, s_c, worker_stats) = execute_continuous_map_reduce(
            fragments, target_query, target_directory, decomp_p_tok=p_tok, decomp_c_tok=c_tok)

        master_elapsed_time = time.time() - master_start_time

        stats_md = "\n\n---\n## Execution Telemetry\n### Worker Execution Statistics\n"
        stats_md += "| Worker ID | Slot | Status | Elapsed (s) | Task TPS | Prompt Tokens | Comp Tokens | Total Tokens | Estimated |\n"
        stats_md += "|-----------|------|--------|-------------|----------|---------------|-------------|--------------|-----------|\n"
        for stat in sorted(worker_stats, key=lambda x: x['id']):
            estimated_flag = "Yes" if stat.get('is_estimated') else "No"
            stats_md += f"| Thread{stat['id']:02d} | {stat.get('slot', 'N/A')} | {stat['status']} | {stat.get('elapsed', 0)} | {stat.get('tps', 0)} | {stat.get('prompt_tokens', 0)} | {stat.get('completion_tokens', 0)} | {stat.get('total_tokens', 0)} | {estimated_flag} |\n"

        agg_md = "\n### Cluster Aggregate Statistics\n"
        agg_md += f"- **Total Wall-Clock Time:** {master_elapsed_time:.2f} seconds\n"
        agg_md += f"- **Worker Prompt Tokens:** {w_p}\n"
        agg_md += f"- **Worker Completion Tokens:** {w_c}\n"
        agg_md += f"- **Total Orchestrator + Decomposition Prompt Tokens:** {o_p}\n"
        agg_md += f"- **Total Orchestrator + Decomposition Completion Tokens:** {o_c}\n"
        agg_md += f"- **Stitcher Prompt Tokens:** {s_p}\n"
        agg_md += f"- **Stitcher Completion Tokens:** {s_c}\n"
        agg_md += f"- **Stitcher Model:** {STITCHER_MODEL}\n"
        agg_md += f"- **Stitcher Endpoints:** {', '.join(STITCHER_ENDPOINTS)}\n"

        final_output += stats_md + agg_md

        with open(final_file_path, "w", encoding="ascii") as f:
            f.write(final_output)

    # ---------------------------------------------------------
    # PHASE 4: POST-PROCESS EDITORIAL
    if args.resume and output_path.exists():
        print(f"[PHASE 4] Bypassed. Resuming from existing POLISHED_SYNTHESIS.md")
    else:
        print("\n[PHASE 4] STARTING DISTRIBUTED SYNTHESIS POLISH", flush=True)
        raw_markdown = read_file_content_safe(final_file_path)
        if raw_markdown is None:
            print(f"[!] Fatal: Could not read {final_file_path}.", flush=True)
            sys.exit(1)

        skeleton_text, protected_assets = extract_and_protect_blocks(raw_markdown)
        chunks = split_into_logical_chunks(skeleton_text, MAX_CHUNK_CHARS)

        if len(chunks) <= 1:
            with open(output_path, "w", encoding="ascii") as f:
                f.write(raw_markdown)
            print(f"[+] Document fits in single chunk. Bypassed LLM refinement. Saved to {output_path}")
        else:
            distilled_skeleton = parallel_edit_chunks(chunks)
            global_inventory = [k for k in protected_assets.keys() if k != "[[PROTECTED_TELEMETRY_TABLE]]"]
            final_skeleton = global_consolidation_pass(distilled_skeleton, global_inventory, ORCHESTRATOR_ENDPOINTS[0])
            final_polished_markdown = reassemble_document(final_skeleton, protected_assets)

            with open(output_path, "w", encoding="ascii") as f:
                f.write(final_polished_markdown)
            print(f"[+] Cleaned File Saved To: {output_path.absolute()}")

    # ---------------------------------------------------------
    # PHASE 5: AUTOMATED UNITTESTS
    if args.resume and report_path.exists():
        print(f"[PHASE 5] Bypassed. Resuming from existing execution_report.json")
    else:
        run_phase5_automatic_unittests(output_path)

    # ---------------------------------------------------------
    # PHASE 6: PROJECT DISTILLATION
    if args.resume and distilled_tasks_path.exists():
        print(f"[PHASE 6] Bypassed. DISTILLED_TASKS.md already exists.")
    else:
        run_phase6_project_distillation(output_path.parent)

    print("\n==============================================================================")
    print("PIPELINE COMPLETE")
    print("==============================================================================\n")

if __name__ == "__main__":
    main()
