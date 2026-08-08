#!/usr/bin/env python3
# -*- coding: ascii -*-

import os
import socket
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

# ==============================================================================
# Server Context Alignment
# ------------------------------------------------------------------------------
# These constants MUST mirror the -c / -np flags in start-zerg-all.sh.
#
# llama-server semantics:
#   * -c N is the TOTAL KV budget for the process, not a per-request guarantee.
#   * With --kv-unified (default on recent builds) the KV cells are SHARED
#     across the -np slots. A lone sequence may address up to N, but the SUM
#     over concurrently active slots must stay <= N.
#   * The concurrency-safe per-request window is therefore N // np.
#   * That window is additionally capped by the model's native n_ctx_train;
#     going past it is rope extrapolation, not free context.
#
# Every downstream char budget in this file is derived from these numbers.
# Change the server flags and these constants together, never one alone.
# ==============================================================================

# Apex / generation / distillation node (port 8081): -c 65536 -np 1
APEX_SERVER_CTX = int(os.getenv("APEX_SERVER_CTX", "65536"))
APEX_SERVER_NP = int(os.getenv("APEX_SERVER_NP", "1"))

# Stitcher cluster (ports 8070, 8071): -c 131072 -np 2 --kv-unified
STITCH_SERVER_CTX = int(os.getenv("STITCH_SERVER_CTX", "131072"))
STITCH_SERVER_NP = int(os.getenv("STITCH_SERVER_NP", "2"))
# gemma-4-E4B native training window == actual context window here, no extrapolation
STITCH_MODEL_NATIVE_CTX = int(os.getenv("STITCH_MODEL_NATIVE_CTX", "131072"))

# Worker cluster (ports 8033, 8034): -c 196608 -np 2 --kv-unified
WORKER_SERVER_CTX = int(os.getenv("WORKER_SERVER_CTX", "196608"))
WORKER_SERVER_NP = int(os.getenv("WORKER_SERVER_NP", "2"))

# Concurrency-safe per-request windows.
APEX_CONTEXT_TOKENS = max(4096, APEX_SERVER_CTX // max(1, APEX_SERVER_NP))
WORKER_CONTEXT_TOKENS = max(4096, WORKER_SERVER_CTX // max(1, WORKER_SERVER_NP))
STITCH_CONTEXT_TOKENS = int(os.getenv("STITCH_CONTEXT_TOKENS", str(min(
    max(4096, STITCH_SERVER_CTX // max(1, STITCH_SERVER_NP)),
    STITCH_MODEL_NATIVE_CTX
))))

# Phase 1: Raw Generation Config
GEN_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:8081/v1")
GEN_API_KEY = os.getenv("OPENAI_API_KEY", "sk-local")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.6-27B-UD-IQ3_XXS.gguf")

# Unified Context Limits
CHARS_PER_TOKEN = float(os.getenv("CHARS_PER_TOKEN", "3.5"))
APEX_MAX_OUTPUT_TOKENS = int(os.getenv("APEX_MAX_OUTPUT_TOKENS", "8192"))
APEX_GEN_TOKENS = int(os.getenv("APEX_GEN_TOKENS", "4096"))
APEX_RESERVE_TOKENS = int(os.getenv("APEX_RESERVE_TOKENS", "2048"))

_apex_input_chars = int(
    max(4096, APEX_CONTEXT_TOKENS - APEX_MAX_OUTPUT_TOKENS - APEX_RESERVE_TOKENS)
    * CHARS_PER_TOKEN
)

MAX_CONTEXT_CHARS = min(
    int(os.getenv("MAX_CONTEXT_CHARS", "60000")),
    _apex_input_chars
)
MAX_CHUNK_CHARS = min(
    int(os.getenv("MAX_CHUNK_CHARS", "40000")),
    MAX_CONTEXT_CHARS
)

# Phase 2: Distillation Config
DISTILLER_URL = os.getenv("DISTILLER_URL", "http://localhost:8081/v1")
DISTILLER_MODEL = os.getenv("DISTILLER_MODEL", "Qwen3.6-27B-UD-IQ3_XXS.gguf")
DISTILLER_API_KEY = os.getenv("DISTILLER_API_KEY", "local-sk")

# Phase 3 & 4: Unified Stitcher Cluster
STITCHER_ENDPOINTS = [
    ep.strip() for ep in os.getenv(
        "STITCHER_ENDPOINTS",
        "http://localhost:8070/v1,http://localhost:8071/v1"
    ).split(",") if ep.strip()
]
STITCHER_MODEL = os.getenv("STITCHER_MODEL", "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf")
STITCHER_API_KEY = os.getenv("STITCHER_API_KEY", "local-sk")
STITCH_PARALLEL_SLOTS = min(
    int(os.getenv("STITCH_PARALLEL_SLOTS", str(STITCH_SERVER_NP))),
    STITCH_SERVER_NP
)
MAX_RETRIES = 3

MAX_STITCH_TOKENS = int(os.getenv("MAX_STITCH_TOKENS", "32768"))
# Chunk-level synthesis output ceiling. Each Level-1 synthesis node merges
# SYNTHESIS_CHUNK_SIZE worker reports; it never needs a full 32k emission and a
# smaller ceiling keeps per-chunk wall time bounded.
MAX_STITCH_MERGE_TOKENS = int(os.getenv("MAX_STITCH_MERGE_TOKENS", str(min(MAX_STITCH_TOKENS, 16384))))
STITCH_CHARS_PER_TOKEN = float(os.getenv("STITCH_CHARS_PER_TOKEN", str(CHARS_PER_TOKEN)))
STITCH_RESERVE_TOKENS = int(os.getenv("STITCH_RESERVE_TOKENS", "4096"))

# Restrict decomposer to not bloat out pipeline operations on simple prompts
MAX_DECOMPOSE_TASKS = int(os.getenv("MAX_DECOMPOSE_TASKS", "20"))

_stitch_input_tokens = max(
    4096, STITCH_CONTEXT_TOKENS - MAX_STITCH_TOKENS - STITCH_RESERVE_TOKENS
)

MAX_STITCH_CONTEXT_CHARS = int(os.getenv(
    "MAX_STITCH_CONTEXT_CHARS",
    str(int(_stitch_input_tokens * STITCH_CHARS_PER_TOKEN))
))

# Echo passes must fit within the input window, not the output window.
STITCH_ECHO_MAX_CHARS = int(os.getenv(
    "STITCH_ECHO_MAX_CHARS",
    str(MAX_STITCH_CONTEXT_CHARS)
))

# Phase 3 & 4: Worker Config
WORKER_ENDPOINTS = [
    "http://localhost:8033/v1",
    "http://localhost:8034/v1"
]
WORKER_MODEL = os.getenv("WORKER_MODEL", "Qwen3.5-9B-IQ4_XS.gguf")
WORKER_API_KEY = os.getenv("WORKER_API_KEY", "local-sk")

WORKER_PARALLEL_SLOTS = min(
    int(os.getenv("WORKER_PARALLEL_SLOTS", str(WORKER_SERVER_NP))),
    WORKER_SERVER_NP
)
WORKER_RETRIES = 3

WORKER_TIMEOUT_SECS = float(os.getenv("WORKER_TIMEOUT_SECS", "300.0"))
STITCH_TIMEOUT_SECS = float(os.getenv("STITCH_TIMEOUT_SECS", "300.0"))

SYNTHESIS_CHUNK_SIZE = int(os.getenv("SYNTHESIS_CHUNK_SIZE", "5"))

WORKER_RESERVE_TOKENS = int(os.getenv("WORKER_RESERVE_TOKENS", "2048"))
MAX_WORKER_TOKENS = min(
    int(os.getenv("MAX_WORKER_TOKENS", "8192")),
    max(1024, WORKER_CONTEXT_TOKENS - WORKER_RESERVE_TOKENS
        - int(MAX_CONTEXT_CHARS / CHARS_PER_TOKEN))
)

WORKER_MIN_DECODE_TPS = float(os.getenv("WORKER_MIN_DECODE_TPS", "4.0"))
WORKER_MAX_WALL_SECS = float(os.getenv(
    "WORKER_MAX_WALL_SECS",
    str(max(600.0, MAX_WORKER_TOKENS / WORKER_MIN_DECODE_TPS + 120.0))))

REPO_WORKER_WALL_SECS = float(os.getenv(
    "REPO_WORKER_WALL_SECS",
    str(WORKER_TIMEOUT_SECS * WORKER_RETRIES)))

# Phase 5: Automatic Unittests Config
TEST_WORKER_ENDPOINTS = [
    "http://localhost:8033/v1/chat/completions",
    "http://localhost:8034/v1/chat/completions"
]
CONCURRENT_REQS_PER_ENDPOINT = WORKER_PARALLEL_SLOTS
MAX_OUTPUT_TOKENS = min(
    int(os.getenv("MAX_OUTPUT_TOKENS", "4096")),
    max(1024, WORKER_CONTEXT_TOKENS - WORKER_RESERVE_TOKENS
        - int(MAX_CONTEXT_CHARS / CHARS_PER_TOKEN))
)
LLM_TEMPERATURE = 0.1
LLM_TOP_P = 0.95
LLM_FREQUENCY_PENALTY = 0.5
LLM_PRESENCE_PENALTY = 0.2
RETRY_BASE_DELAY = 2.0
RETRY_JITTER = 0.5
MAX_EXEC_WORKERS = 4
EXECUTION_RESULT_FIELDS = ["filename", "language", "status", "message", "chunk"]

TEST_MIN_DECODE_TPS = float(os.getenv("TEST_MIN_DECODE_TPS", "10.0"))
TEST_TIMEOUT_SECS = float(os.getenv(
    "TEST_TIMEOUT_SECS",
    str(max(300.0, MAX_OUTPUT_TOKENS / TEST_MIN_DECODE_TPS + 60.0))
))

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
    "Output ONLY a valid, flat JSON array of strings. No markdown formatting, no conversational text.\n"
    "Generate between 3 and 20 tasks. Do not generate fewer than 3 or more than 20 tasks regardless of input size."
)

_PROMPT_PHASE3_WORKER = (
    "You are an autonomous, highly-capable worker agent equipped with advanced reasoning. "
    "Think step-by-step to formulate a plan. You must execute your specific objective fully. "
    "CRITICAL: If your task involves writing code, creating configurations, or generating files, "
    "you MUST output the file contents wrapped exactly in these XML tags:\n"
    '<file path="filename.ext">\n[YOUR FILE CONTENT HERE]\n</file>\n'
    "RULES FOR FILE TAGS:\n"
    "1. Every opening <file path=\"...\"> tag MUST have exactly one matching </file> closing tag.\n"
    "2. Do NOT use <file> tags in prose, explanations, or code examples. "
    "Only use them to wrap actual file output.\n"
    "3. If you reference a filename in text, write it as plain text or in backticks, "
    "never as an XML tag.\n"
    "4. Never nest <file> tags inside other <file> tags.\n"
)

_PROMPT_PHASE3_SYNTHESIS = (
    "You are a Level-1 Synthesis Node in a distributed cluster. "
    "Merge the following sequential worker reports into a coherent, deduplicated section. "
    "Retain all code blocks, configurations, and critical technical data. "
    "Output strictly in standard ASCII.\n"
    "IMPORTANT: If the input contains <file path=\"...\">...</file> blocks, "
    "preserve them exactly as-is including both opening and closing tags. "
    "Do not add new <file> tags in your synthesis output."
)

_PROMPT_PHASE4_DEDUP = (
    "You are a strict, highly logical Technical Editor processing a section of a larger technical document. "
    "Your job is to take this messy, repetitive draft and rewrite it into cohesive, succinct markdown. "
    "\n\nRULES:"
    "\n1. Remove all repetitive statements, redundant introductions, and duplicated concepts."
    "\n2. Organize the content into logical, non-repeating markdown headers."
    "\n3. Use bullet points for readability where applicable."
    "\n4. Preserve every fenced code block and configuration verbatim."
    "\n5. Do not add conversational filler. Be direct and professional."
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

_PROMPT_PHASE6_ITERATE = (
    "You are a Lead Engineer iterating on an existing project distillation. "
    "You will be provided with the CURRENT DISTILLED TASKS, followed by NEW TELEMETRY AND DOCS. "
    "Your job is to update, refine, and append to the existing tasks based on the new execution results. "
    "Resolve tasks that passed their tests. Prioritize tasks whose tests failed. "
    "Do NOT throw away the existing tasks unless they are demonstrably completed. "
    "Output the updated clean, highly structured Markdown document."
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
                    safe_p = p[:max_chars - 50] + "\n\n...[CHUNK TRUNCATED]..." if len(p) > max_chars else p
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

# ------------------------------------------------------------------
# Stitcher cluster fabric
# ------------------------------------------------------------------

_stitcher_rr_lock = threading.Lock()
_stitcher_rr_idx = 0

def next_stitcher_endpoint() -> str:
    global _stitcher_rr_idx
    if not STITCHER_ENDPOINTS:
        print("\n[!] FATAL: STITCHER_ENDPOINTS is empty - no inference tier available.", flush=True)
        sys.exit(1)
    with _stitcher_rr_lock:
        endpoint = STITCHER_ENDPOINTS[_stitcher_rr_idx % len(STITCHER_ENDPOINTS)]
        _stitcher_rr_idx += 1
    return endpoint

def stitcher_client(endpoint: Optional[str] = None, timeout: float = STITCH_TIMEOUT_SECS,
                    max_retries: int = 0) -> OpenAI:
    return OpenAI(
        base_url=endpoint or next_stitcher_endpoint(),
        api_key=STITCHER_API_KEY,
        timeout=timeout,
        max_retries=max_retries,
    )

def _stitch_completion(client: OpenAI, system_prompt: str, user_prompt: str,
                       max_tokens: int, temperature: float,
                       presence_penalty: Optional[float] = None) -> Tuple[str, int, int]:
    """Streaming stitcher call. Returns (text, prompt_tokens, completion_tokens)."""
    kwargs = dict(
        model=STITCHER_MODEL,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=temperature, max_tokens=max_tokens, stream=True,
        stream_options={"include_usage": True},
    )
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        low = str(e).lower()
        if "stream_options" in low or "unrecognized" in low or "400" in low:
            kwargs.pop("stream_options")
            response = client.chat.completions.create(**kwargs)
        else:
            raise

    text, p_tok, c_tok = "", 0, 0
    try:
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                text += chunk.choices[0].delta.content
            if getattr(chunk, "usage", None) is not None:
                p_tok, c_tok = chunk.usage.prompt_tokens, chunk.usage.completion_tokens
    finally:
        try:
            response.close()
        except Exception:
            pass

    text = enforce_ascii(text.strip())
    if not p_tok and not c_tok:
        p_tok, c_tok = estimate_tokens(system_prompt + user_prompt), estimate_tokens(text)
    return text, p_tok, c_tok

def verify_server_props(endpoints: List[str], label: str, expect_ctx: int, expect_np: int) -> None:
    for ep in endpoints:
        for attempt in range(3):
            try:
                base = ep.rsplit("/v1", 1)[0]
                props_url = f"{base}/props"
                resp = requests.get(props_url, timeout=5.0)
                if resp.status_code == 200:
                    data = resp.json()
                    default_props = data.get("default_generation_settings", {})
                    n_ctx = default_props.get("n_ctx", 0)
                    slots = data.get("total_slots") or default_props.get("n_parallel") or 0
                    
                    # Handle false MISMATCH during unified KV startup where n_ctx defaults briefly
                    per_slot = int(expect_ctx) // max(1, int(expect_np))
                    if expect_ctx and n_ctx and int(n_ctx) not in (int(expect_ctx), per_slot) and int(n_ctx) in (512, 2048) and attempt < 2:
                        time.sleep(2)
                        continue
                    
                    note = "ok"
                    if expect_ctx and n_ctx and int(n_ctx) not in (int(expect_ctx), per_slot):
                        note = "MISMATCH: constant says -c {} (per-slot {})".format(expect_ctx, per_slot)
                    elif expect_np and slots and int(slots) != int(expect_np):
                        note = "MISMATCH: constant says -np {}".format(expect_np)
                    print("    [+] {} {} n_ctx={} slots={} :: {}".format(label, ep, n_ctx, slots, note), flush=True)
                    break
                else:
                    if attempt == 2:
                        print(f"    [!] {label} {ep} /props check returned HTTP {resp.status_code}", flush=True)
                    time.sleep(2)
            except Exception as exc:
                if attempt == 2:
                    print(f"    [!] {label} {ep} /props check failed: {str(exc)[:120]}", flush=True)
                time.sleep(2)

def ping_tier(endpoints: List[str], model: str, api_key: str, label: str, timeout: float = 90.0) -> bool:
    ok = True
    for ep in endpoints:
        try:
            c = OpenAI(base_url=ep, api_key=api_key, timeout=timeout, max_retries=0)
            t0 = time.time()
            c.chat.completions.create(model=model,
                                      messages=[{"role": "user", "content": "ping"}],
                                      max_tokens=4, temperature=0.0)
            print("    [+] {} {} responded in {:.1f}s".format(label, ep, time.time() - t0), flush=True)
        except Exception as exc:
            print("    [!] WARNING: {} {} smoke test failed: {}".format(label, ep, str(exc)[:160]), flush=True)
            ok = False
    return ok

def clamp_stitch_tokens(requested: int) -> int:
    return max(256, min(int(requested), MAX_STITCH_TOKENS))

def fit_stitch_context(text: str, budget: Optional[int] = None,
                       note: str = "...[CONTENT TRUNCATED FOR CONTEXT LIMITS]...") -> str:
    limit = MAX_STITCH_CONTEXT_CHARS if budget is None else budget
    if len(text) <= limit:
        return text
    print("    [!] fit_stitch_context dropped {:,} of {:,} chars.".format(
        len(text) - limit, len(text)), flush=True)
    return text[:limit] + f"\n\n{note}"

def build_stitcher_slot_queue(prefix: str = "S-Slot") -> Tuple[queue.Queue, int]:
    slot_queue: queue.Queue = queue.Queue()
    slot_idx = 1
    for ep in STITCHER_ENDPOINTS:
        parsed = urllib.parse.urlparse(ep)
        host_tail = parsed.hostname or "local"
        if parsed.port:
            host_tail += f":{parsed.port}"
        for _ in range(STITCH_PARALLEL_SLOTS):
            slot_queue.put((ep, f"{prefix}{slot_idx:02d}-{host_tail}"))
            slot_idx += 1
    return slot_queue, max(0, slot_idx - 1)

def describe_budget_alignment() -> str:
    lines = []
    lines.append("[BUDGET] Server context alignment")
    lines.append(
        f"    apex    :8081  -c {APEX_SERVER_CTX} -np {APEX_SERVER_NP}"
        f"  -> {APEX_CONTEXT_TOKENS//1024}k tok/req"
        f" | input<={MAX_CONTEXT_CHARS:,} chars"
        f" (~{int(MAX_CONTEXT_CHARS/CHARS_PER_TOKEN)//1024}k tok)"
        f" | out<={APEX_GEN_TOKENS//1024}k gen / {APEX_MAX_OUTPUT_TOKENS//1024}k distil"
    )
    lines.append(
        f"    stitcher:8070/1 -c {STITCH_SERVER_CTX} -np {STITCH_SERVER_NP}"
        f"  -> {STITCH_CONTEXT_TOKENS//1024}k tok/req"
        f" (native cap {STITCH_MODEL_NATIVE_CTX//1024}k)"
        f" | input<={MAX_STITCH_CONTEXT_CHARS:,} chars"
        f" | synth<={MAX_STITCH_MERGE_TOKENS//1024}k tok"
        f" | edit<={MAX_STITCH_TOKENS//1024}k tok"
        f" | echo<={STITCH_ECHO_MAX_CHARS:,} chars"
        f" | gap_to={int(STITCH_TIMEOUT_SECS)}s"
    )
    lines.append(
        f"    worker  :8033/4 -c {WORKER_SERVER_CTX} -np {WORKER_SERVER_NP}"
        f"  -> {WORKER_CONTEXT_TOKENS//1024}k tok/req"
        f" | input<={MAX_CONTEXT_CHARS:,} chars"
        f" | out<={MAX_WORKER_TOKENS//1024}k tok"
        f" | gap={int(WORKER_TIMEOUT_SECS)}s wall={int(WORKER_MAX_WALL_SECS)}s"
    )
    lines.append(
        f"    repo    :batch  wall={int(REPO_WORKER_WALL_SECS)}s/attempt "
        f"| outer={int(WORKER_RETRIES * REPO_WORKER_WALL_SECS + 30)}s"
    )
    
    calc_worker_tok = WORKER_CONTEXT_TOKENS - WORKER_RESERVE_TOKENS - int(MAX_CONTEXT_CHARS / CHARS_PER_TOKEN)
    if calc_worker_tok < 1024:
        lines.append(
            f"    [!] WARNING: Calculated worker output budget collapsed to {calc_worker_tok} "
            f"and hit the 1024 floor. Most worker responses will truncate!"
        )

    lines.append(
        f"    tests   :8033/4 out<={MAX_OUTPUT_TOKENS//1024}k tok"
        f" | min_tps={TEST_MIN_DECODE_TPS} | timeout={int(TEST_TIMEOUT_SECS)}s"
    )
    lines.append(
        "    [i] Tree-reduction stitch DISABLED: chunks are preserved individually. "
        "No cross-chunk content loss."
    )

    peak_stitch = (
        int(MAX_STITCH_CONTEXT_CHARS / STITCH_CHARS_PER_TOKEN)
        + MAX_STITCH_TOKENS + STITCH_RESERVE_TOKENS
    )
    concurrent_stitch = peak_stitch * STITCH_PARALLEL_SLOTS
    if concurrent_stitch > STITCH_SERVER_CTX:
        lines.append(
            f"    [!] Stitcher over-subscribed: {STITCH_PARALLEL_SLOTS} slot(s) x "
            f"{peak_stitch} tok = {concurrent_stitch} > -c {STITCH_SERVER_CTX}. "
            "Lower MAX_STITCH_TOKENS, lower -np, or raise -c."
        )
    if STITCH_SERVER_CTX // max(1, STITCH_SERVER_NP) > STITCH_MODEL_NATIVE_CTX:
        lines.append(
            f"    [!] Stitcher slot ({STITCH_SERVER_CTX // max(1, STITCH_SERVER_NP)}) "
            f"exceeds native {STITCH_MODEL_NATIVE_CTX}; excess is rope extrapolation."
        )
    return "\n".join(lines)

def describe_stitch_budget() -> str:
    return (
        f"window={STITCH_CONTEXT_TOKENS//1024}k tok | "
        f"input<={MAX_STITCH_CONTEXT_CHARS:,} chars (~{int(MAX_STITCH_CONTEXT_CHARS/STITCH_CHARS_PER_TOKEN)//1024}k tok) | "
        f"synth<={MAX_STITCH_MERGE_TOKENS//1024}k tok | "
        f"edit<={MAX_STITCH_TOKENS//1024}k tok | "
        f"{len(STITCHER_ENDPOINTS)} node(s) x {STITCH_PARALLEL_SLOTS} slot(s)"
    )

def _format_eta(start_time: float, completed: int, total: int) -> str:
    if completed == 0 or total == 0 or completed == total:
        return "--:--"
    elapsed = time.time() - start_time
    eta_secs = (elapsed / completed) * (total - completed)
    mins, secs = divmod(int(eta_secs), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours}h {mins:02d}m"
    return f"{mins:02d}:{secs:02d}"

def _render_map_progress(workers_finished: int, total_tasks: int, chunks_completed: int, total_chunks: int, start_time: float) -> None:
    bar_len = 30
    total_tasks_safe = total_tasks if total_tasks > 0 else 1
    filled = int((workers_finished / total_tasks_safe) * bar_len)
    bar = '#' * filled + '-' * (bar_len - filled)
    percent = int((workers_finished / total_tasks_safe) * 100)
    eta_str = _format_eta(start_time, workers_finished, total_tasks)
    sys.stdout.write(f"\r    [+] Map-Reduce Progress: [{bar}] {percent}% (Workers: {workers_finished}/{total_tasks} | Chunks: {chunks_completed}/{total_chunks}) | ETC: {eta_str}")
    sys.stdout.flush()

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
            max_tokens=APEX_GEN_TOKENS,
            stream=True
        )

        try:
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    full_content += chunk.choices[0].delta.content
        finally:
            try:
                response.close()
            except Exception:
                pass

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
    """
    Advisory check to prevent basic SSRF against local metadata/loopback.
    NOTE: This is subject to TOCTOU (Time-of-Check to Time-of-Use) DNS rebinding
    attacks. Real mitigation requires running the clone behind a network namespace
    or using GIT_PROXY_COMMAND with an enforced allowlist.
    """
    try:
        orig_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(3.0)
        try:
            ip = socket.gethostbyname(host)
        finally:
            socket.setdefaulttimeout(orig_timeout)
        addr = ipaddress.ip_address(ip)
        return addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved
    except Exception:
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
    
    def _consume():
        stream = client.chat.completions.create(
            model=WORKER_MODEL,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=0.2, max_tokens=MAX_WORKER_TOKENS, stream=True)
        try:
            out = "".join(c.choices[0].delta.content for c in stream 
                          if c.choices and c.choices[0].delta.content is not None)
            return enforce_ascii(out.strip())
        finally:
            try:
                stream.close()
            except Exception:
                pass
                
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_consume)
        try:
            return future.result(timeout=REPO_WORKER_WALL_SECS)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Stream stall: exceeded {REPO_WORKER_WALL_SECS}s wall clock limit.")

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
        endpoint = None
        while not _shutdown_event.is_set():
            try:
                endpoint = slot_queue.get(timeout=5.0)
                break
            except queue.Empty:
                continue
        if endpoint is None:
            return fallback_fn(payload)

        try:
            for attempt in range(1, WORKER_RETRIES + 1):
                try:
                    output = job_fn(idx + 1, total, payload, endpoint)
                    if output and len(output.strip()) >= 20:
                        return output.strip()
                except Exception as e:
                    if attempt < WORKER_RETRIES:
                        print(f"        [!] {label} {idx+1}/{total} attempt {attempt} failed ({str(e)}), retrying...", flush=True)
                    time.sleep(2)
            return fallback_fn(payload)
        finally:
            slot_queue.put(endpoint)

    pool_size = max(1, len(WORKER_ENDPOINTS) * WORKER_PARALLEL_SLOTS)
    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
        future_to_idx = {executor.submit(wrapper, i, job): i for i, job in enumerate(jobs)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            completed_count += 1
            eta_str = _format_eta(start_time_progress, completed_count, total)
            try:
                # Add strict wall clock boundary here to prevent fully blocked futures
                results[idx] = future.result(timeout=WORKER_RETRIES * REPO_WORKER_WALL_SECS + 30)
                print(f"        [+] {label} {completed_count}/{total} complete. | ETC: {eta_str}", flush=True)
            except concurrent.futures.TimeoutError:
                print(f"        [!] {label} {completed_count}/{total} future timed out entirely. Using fallback. | ETC: {eta_str}", flush=True)
                results[idx] = fallback_fn(jobs[idx])
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
        print("    [!] Warning: Reduce chunk synthesis failed. Truncating to safe length.", flush=True)
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
    distill_client = OpenAI(base_url=DISTILLER_URL, api_key=DISTILLER_API_KEY, timeout=WORKER_TIMEOUT_SECS)
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
            max_tokens=APEX_MAX_OUTPUT_TOKENS,
            stream=True
        )
        out = ""
        try:
            for c in response:
                if c.choices and c.choices[0].delta.content is not None:
                    out += c.choices[0].delta.content
        finally:
            try:
                response.close()
            except Exception:
                pass

        return enforce_ascii(out.strip())

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
# Phase 3: Distributed Map-Reduce (chunk-preserving, no tree reduction)
# ------------------------------------------------------------------------------
# The tree-reduction stitch was removed deliberately. Collapsing N synthesis
# chunks into one document forces every merge through a single
# MAX_STITCH_MERGE_TOKENS emission, so each round silently truncated the
# accumulated left-hand side. With 8 chunks over 3 rounds that discarded
# roughly 7/8 of the corpus and produced a final document containing only the
# last section. Chunks are now persisted individually; Phase 4 polishes each
# in place and Phase 5 harvests artifacts from all of them.
# ==============================================================================

def estimate_tokens(text: str) -> int:
    return int(len(str(text)) / CHARS_PER_TOKEN)

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

        if in_string and char == '\\':
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

    fitted_query = fit_stitch_context(large_query)
    user_content = f"Decompose this to the atomic level:\n\n{fitted_query}"

    for attempt in range(1, MAX_RETRIES + 1):
        endpoint = next_stitcher_endpoint()
        client = stitcher_client(endpoint)
        print(f"[2] DECOMPOSITION: Engaging atomic breakdown via {endpoint} [{STITCHER_MODEL}] (Attempt {attempt}/{MAX_RETRIES})...", flush=True)
        try:
            start_time = time.time()
            raw_output, prompt_tokens, comp_tokens = _stitch_completion(
                client, _PROMPT_PHASE3_DECOMPOSE, user_content, MAX_STITCH_TOKENS, 0.7
            )

            cleaned_output = extract_json_array(raw_output)
            if not cleaned_output: raise ValueError("Could not locate JSON array.")

            atomic_pieces = json.loads(cleaned_output)
            if not isinstance(atomic_pieces, list) or not all(isinstance(x, str) for x in atomic_pieces):
                raise ValueError(f"Decomposition produced non-string items: {type(atomic_pieces)}")

            if len(atomic_pieces) > MAX_DECOMPOSE_TASKS:
                print(f"    [!] Decomposer emitted {len(atomic_pieces)} tasks; clamping to {MAX_DECOMPOSE_TASKS}.", flush=True)
                atomic_pieces = atomic_pieces[:MAX_DECOMPOSE_TASKS]

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

def process_subtask(task_id: int, task_prompt: str, endpoint: str, slot_name: str,
                    original_query: str, run_dir: Path, on_progress=None) -> dict:
    worker_client = OpenAI(base_url=endpoint, api_key=WORKER_API_KEY, timeout=WORKER_TIMEOUT_SECS, max_retries=0)
    start_time = time.time()

    user_instruction = f"BACKGROUND CONTEXT:\n{original_query}\n\nYOUR SPECIFIC OBJECTIVE:\n{task_prompt}"

    saved_artifacts = []
    status = "success"
    prompt_tokens, comp_tokens = 0, 0
    is_estimated = True
    result_text = ""
    truncated = False

    try:
        try:
            response = worker_client.chat.completions.create(
                model=WORKER_MODEL,
                messages=[{"role": "system", "content": _PROMPT_PHASE3_WORKER}, {"role": "user", "content": user_instruction}],
                temperature=0.4,
                max_tokens=MAX_WORKER_TOKENS,
                frequency_penalty=1.1,
                presence_penalty=0.5,
                stream=True,
                stream_options={"include_usage": True}
            )
        except Exception as e:
            if "stream_options" in str(e).lower() or "unrecognized" in str(e).lower() or "400" in str(e):
                response = worker_client.chat.completions.create(
                    model=WORKER_MODEL,
                    messages=[{"role": "system", "content": _PROMPT_PHASE3_WORKER}, {"role": "user", "content": user_instruction}],
                    temperature=0.4,
                    max_tokens=MAX_WORKER_TOKENS,
                    frequency_penalty=1.1,
                    presence_penalty=0.5,
                    stream=True
                )
            else:
                raise e

        first_token_at = None
        last_ping = time.time()

        try:
            for chunk in response:
                now = time.time()
                if now - start_time > WORKER_MAX_WALL_SECS:
                    result_text += "\n\n...[OUTPUT TRUNCATED: MAX WALL CLOCK EXCEEDED]..."
                    if on_progress:
                        on_progress(task_id, "wallclock", f"Max wall clock exceeded after {now - start_time:.0f}s, truncating stream.")
                    truncated = True
                    break

                if chunk.choices and chunk.choices[0].delta.content is not None:
                    result_text += chunk.choices[0].delta.content
                    if first_token_at is None:
                        first_token_at = now
                        if on_progress:
                            on_progress(task_id, "ttft", f"ttft {now - start_time:.1f}s")
                    elif on_progress and now - last_ping > 15.0:
                        last_ping = now
                        on_progress(task_id, "ping", f"{len(result_text)} chars")

                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                    comp_tokens = chunk.usage.completion_tokens
                    is_estimated = False
        finally:
            try:
                response.close()
            except Exception:
                pass

        result_text = enforce_ascii(result_text.strip())

        # Strip fenced code blocks before counting structural tags.
        # <file> strings inside code fences are content, not pipeline tags.
        _text_stripped = re.sub(r'```[\s\S]*?```', '', result_text)
        _text_stripped = re.sub(r'`[^`\n]+`', '', _text_stripped)

        open_tags = len(re.findall(r'<file\s+path="[^"]*"', _text_stripped, re.IGNORECASE))
        close_tags = len(re.findall(r'</file\s*>', _text_stripped, re.IGNORECASE))

        if is_estimated:
            prompt_tokens = estimate_tokens(_PROMPT_PHASE3_WORKER + user_instruction)
            comp_tokens = estimate_tokens(result_text)

        if open_tags != close_tags:
            print(f"        [!] Warning: Thread{task_id:02d} tag count mismatch "
                  f"({open_tags} open, {close_tags} close). Skipping artifact extraction to prevent corrupted slurring.", flush=True)
            result_text += "\n\n...[ARTIFACT EXTRACTION FAILED: UNBALANCED FILE TAGS]..."
            status = "failed_validation"
        else:
            file_matches = re.finditer(
                r'<file\s+path="([^"]+)">([\s\S]*?)</file>', result_text, re.IGNORECASE
            )
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
        is_estimated = True

    elapsed = round(time.time() - start_time, 2)
    return {
        "id": task_id, "prompt": task_prompt, "result": result_text,
        "artifacts": saved_artifacts, "status": status,
        "prompt_tokens": prompt_tokens, "completion_tokens": comp_tokens,
        "total_tokens": prompt_tokens + comp_tokens, "elapsed": elapsed,
        "tps": round(comp_tokens / elapsed, 2) if elapsed > 0 else 0,
        "slot": slot_name,
        "is_estimated": is_estimated,
        "truncated": truncated
    }

def parallel_chunk_synthesis(batch_id: int, tasks: list, endpoint: str, original_query: str) -> tuple:
    client = stitcher_client(endpoint)
    batch_context = "\n\n".join([f"--- TASK {t['id']}: {t['prompt']} ---\n{t['result']}" for t in tasks])
    user_prompt = fit_stitch_context(
        f"ORIGINAL QUERY: {original_query}\n\nREPORTS TO MERGE:\n{batch_context}"
    )

    start_time = time.time()
    res_content, p_tok, c_tok = _stitch_completion(
        client, _PROMPT_PHASE3_SYNTHESIS, user_prompt, MAX_STITCH_MERGE_TOKENS, 0.3
    )
    elapsed = round(time.time() - start_time, 2)
    return batch_id, res_content, p_tok, c_tok, elapsed

# ------------------------------------------------------------------
# Chunk persistence (replaces tree_reduce_stitch)
# ------------------------------------------------------------------

CHUNKS_DIRNAME = "chunks"

def chunks_dir_for(run_dir: Path) -> Path:
    return run_dir / CHUNKS_DIRNAME

def list_chunk_files(run_dir: Path, polished: bool = False) -> List[Path]:
    """Return sorted chunk files. If polished=True, prefer *_polished.md and
    fall back to the raw chunk when a polished twin is missing."""
    cdir = chunks_dir_for(run_dir)
    if not cdir.exists():
        return []
    raw = sorted(cdir.glob("chunk_[0-9][0-9][0-9].md"))
    if not polished:
        return raw
    out = []
    for r in raw:
        p = r.parent / f"{r.stem}_polished.md"
        out.append(p if p.exists() else r)
    return out

def save_chunk_files(ordered_chunks: List[str], run_dir: Path) -> List[Path]:
    """Persist each synthesis chunk to its own file. Nothing is merged, so
    nothing is lost."""
    cdir = chunks_dir_for(run_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for idx, text in enumerate(ordered_chunks, start=1):
        p = cdir / f"chunk_{idx:03d}.md"
        body = enforce_ascii(text.strip())
        header = f"<!-- chunk {idx:03d} of {len(ordered_chunks)} -->\n\n"
        with open(p, "w", encoding="ascii") as f:
            f.write(header + body + "\n")
        paths.append(p)
    return paths

def build_synthesis_manifest(chunk_paths: List[Path], worker_stats: list,
                             original_query: str, elapsed: float,
                             w_p: int, w_c: int, syn_p: int, syn_c: int,
                             truncated_count: int) -> str:
    """FINAL_SYNTHESIS.md is now an index + telemetry document. The corpus
    itself lives in chunks/ and is never collapsed."""
    total_chars = sum(p.stat().st_size for p in chunk_paths if p.exists())

    lines = ["# Final Synthesis Manifest", ""]
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Chunks preserved:** {len(chunk_paths)}")
    lines.append(f"- **Total corpus:** {total_chars:,} bytes across all chunks")
    lines.append(f"- **Assembly mode:** chunk-preserving (tree-reduction stitch disabled)")
    lines.append("")
    lines.append("## Query")
    lines.append("")
    q = original_query.strip()
    lines.append("```")
    lines.append(q[:4000] + ("\n...[QUERY TRUNCATED IN MANIFEST]..." if len(q) > 4000 else ""))
    lines.append("```")
    lines.append("")
    lines.append("## Chunk Index")
    lines.append("")
    lines.append("| # | File | Bytes |")
    lines.append("|---|------|-------|")
    for idx, p in enumerate(chunk_paths, start=1):
        size = p.stat().st_size if p.exists() else 0
        lines.append(f"| {idx} | `{CHUNKS_DIRNAME}/{p.name}` | {size:,} |")
    lines.append("")

    lines.append("## Execution Telemetry")
    lines.append("")
    lines.append("### Worker Execution Statistics")
    lines.append("")
    lines.append("| Worker ID | Slot | Status | Elapsed (s) | Task TPS | Prompt Tokens | Comp Tokens | Total Tokens | Estimated | Truncated |")
    lines.append("|-----------|------|--------|-------------|----------|---------------|-------------|--------------|-----------|-----------|")
    for stat in sorted(worker_stats, key=lambda x: x['id']):
        est = "Yes" if stat.get('is_estimated') else "No"
        trc = "Yes" if stat.get('truncated') else "No"
        lines.append(
            f"| Thread{stat['id']:02d} | {stat.get('slot', 'N/A')} | {stat['status']} | "
            f"{stat.get('elapsed', 0)} | {stat.get('tps', 0)} | {stat.get('prompt_tokens', 0)} | "
            f"{stat.get('completion_tokens', 0)} | {stat.get('total_tokens', 0)} | {est} | {trc} |"
        )
    lines.append("")
    lines.append("### Cluster Aggregate Statistics")
    lines.append("")
    lines.append(f"- **Total Wall-Clock Time:** {elapsed:.2f} seconds")
    lines.append(f"- **Worker Prompt Tokens:** {w_p}")
    lines.append(f"- **Worker Completion Tokens:** {w_c}")
    lines.append(f"- **Stitcher Synthesis + Decomposition Prompt Tokens:** {syn_p}")
    lines.append(f"- **Stitcher Synthesis + Decomposition Completion Tokens:** {syn_c}")
    lines.append(f"- **Stitcher Tree-Reduction Tokens:** 0 (stage removed)")
    lines.append(f"- **Worker Model:** {WORKER_MODEL}")
    lines.append(f"- **Stitcher Model:** {STITCHER_MODEL}")
    lines.append(f"- **Stitcher Endpoints:** {', '.join(STITCHER_ENDPOINTS)} (x{STITCH_PARALLEL_SLOTS} slot(s))")
    lines.append(f"- **Stitcher Budget:** {describe_stitch_budget()}")
    lines.append(f"- **Workers Truncated (wall clock):** {truncated_count} of {len(worker_stats)}")
    lines.append("")

    return enforce_ascii("\n".join(lines))

def execute_continuous_map_reduce(sub_tasks: list, original_query: str, run_dir: Path,
                                  decomp_p_tok: int = 0, decomp_c_tok: int = 0) -> tuple:
    if not WORKER_ENDPOINTS or not STITCHER_ENDPOINTS:
        print("\n[!] FATAL: Endpoints not defined for map-reduce cluster (need worker and stitcher pools).", flush=True)
        sys.exit(1)

    if not sub_tasks:
        return [], 0, 0, decomp_p_tok, decomp_c_tok, []

    total_tasks = len(sub_tasks)
    total_chunks = (total_tasks + SYNTHESIS_CHUNK_SIZE - 1) // SYNTHESIS_CHUNK_SIZE

    print(f"\n[4] CONTINUOUS MAP-REDUCE: Launching parallel workers and chunk synthesis...", flush=True)
    print(f"    [*] Stitcher budget: {describe_stitch_budget()}", flush=True)

    worker_queue = queue.Queue()
    w_slot_idx = 1
    for ep in WORKER_ENDPOINTS:
        for _ in range(WORKER_PARALLEL_SLOTS):
            worker_queue.put((ep, f"W-Slot{w_slot_idx:02d}"))
            w_slot_idx += 1

    synth_queue, synth_slot_count = build_stitcher_slot_queue(prefix="Stitch-Slot")

    event_queue: queue.Queue = queue.Queue()

    synthesized_chunks: Dict[int, str] = {}

    def _map_reduce_worker(tid: int, prompt: str, on_progress=None):
        last_result = {
            "id": tid, "status": "error", "result": "Worker execution failed completely.",
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "elapsed": 0, "tps": 0, "slot": "Unknown", "is_estimated": True, "truncated": False
        }
        accum_p_tok = 0
        accum_c_tok = 0

        for attempt in range(WORKER_RETRIES):
            endpoint, slot_name = None, None
            while not _shutdown_event.is_set():
                try:
                    endpoint, slot_name = worker_queue.get(timeout=5.0)
                    break
                except queue.Empty:
                    continue

            if not endpoint:
                last_result["result"] = "Aborted due to shutdown event."
                event_queue.put(("worker", last_result))
                return

            try:
                res = process_subtask(tid, prompt, endpoint, slot_name, original_query, run_dir, on_progress=on_progress)
                accum_p_tok += res.get("prompt_tokens", 0)
                accum_c_tok += res.get("completion_tokens", 0)

                if res["status"] == "success":
                    if res.get("completion_tokens", 0) > MAX_WORKER_TOKENS:
                        safe_char_limit = int(MAX_WORKER_TOKENS * CHARS_PER_TOKEN)
                        res["result"] = res["result"][:safe_char_limit] + "\n\n...[OUTPUT TRUNCATED DUE TO LENGTH LIMIT]..."
                        res["completion_tokens"] = MAX_WORKER_TOKENS

                    res["prompt_tokens"] = accum_p_tok
                    res["total_tokens"] = accum_p_tok + res.get("completion_tokens", 0)

                    event_queue.put(("worker", res))
                    return

                if on_progress and res["status"] != "success":
                    on_progress(tid, "retry", "retry {}/{} on {} status={} :: {}".format(
                        attempt + 1, WORKER_RETRIES, slot_name, res["status"],
                        str(res.get("result", ""))[:80]))
                last_result = res
            except Exception as e:
                last_result["result"] = f"Failed: {str(e)}"
                last_result["slot"] = slot_name
                last_result["is_estimated"] = True
                if on_progress:
                    on_progress(tid, "retry", "retry {}/{} after {}".format(attempt + 1, WORKER_RETRIES, str(e)[:40]))
            finally:
                worker_queue.put((endpoint, slot_name))
            time.sleep(2)

        last_result["prompt_tokens"] = accum_p_tok
        last_result["completion_tokens"] = accum_c_tok
        last_result["total_tokens"] = accum_p_tok + accum_c_tok
        last_result["is_estimated"] = True
        event_queue.put(("worker", last_result))

    def _emit_chunk_fallback(batch_id: int, tasks: list):
        batch_context = "\n\n".join([f"--- TASK {t['id']}: {t['prompt']} ---\n{t['result']}" for t in tasks])
        fallback_text = f"\n--- [RAW CHUNK {batch_id}] ---\n" + batch_context
        user_prompt = f"ORIGINAL QUERY: {original_query}\n\nREPORTS TO MERGE:\n{batch_context}"
        est_p_tok = estimate_tokens(_PROMPT_PHASE3_SYNTHESIS + user_prompt)
        est_c_tok = estimate_tokens(fallback_text)
        event_queue.put(("chunk", batch_id, fallback_text, est_p_tok, est_c_tok, 0, "Stitch-Fallback"))

    def chunk_wrapper(batch_id: int, tasks: list):
        tasks = [t for t in tasks if t is not None]

        endpoint, slot_name = None, None
        deadline = time.time() + (MAX_RETRIES * STITCH_TIMEOUT_SECS)
        while time.time() < deadline:
            if _shutdown_event.is_set():
                break
            try:
                endpoint, slot_name = synth_queue.get(timeout=5.0)
                break
            except queue.Empty:
                continue

        if endpoint is None:
            _emit_chunk_fallback(batch_id, tasks)
            return

        try:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    b_id, text, p_tok, c_tok, elap = parallel_chunk_synthesis(batch_id, tasks, endpoint, original_query)
                    event_queue.put(("chunk", b_id, text, p_tok, c_tok, elap, slot_name))
                    return
                except Exception as e:
                    print(f"    [!] Chunk synthesis error on attempt {attempt} ({slot_name}): {e}", flush=True)
                    time.sleep(2)
            _emit_chunk_fallback(batch_id, tasks)
        finally:
            synth_queue.put((endpoint, slot_name))

    worker_p_tok, worker_c_tok = 0, 0
    chunk_p_tok, chunk_c_tok = 0, 0
    results_dict: Dict[int, dict] = {}
    worker_stats_log: list = []

    chunks_completed = 0
    workers_finished = 0
    submitted_chunks: Set[int] = set()

    worker_threads = max(1, min(total_tasks, len(WORKER_ENDPOINTS) * WORKER_PARALLEL_SLOTS))
    synth_threads = max(1, min(total_chunks, synth_slot_count))

    MAP_HEARTBEAT_SECS = 30.0
    MAP_STALL_TIMEOUT = float(os.getenv(
        "MAP_STALL_TIMEOUT",
        str(WORKER_RETRIES * WORKER_TIMEOUT_SECS + 120.0)))

    last_heartbeat = time.time()
    start_time_progress = time.time()
    last_event_time = time.time()
    last_completion_time = time.time()

    aborted = False
    worker_exec = concurrent.futures.ThreadPoolExecutor(max_workers=worker_threads)
    synth_exec = concurrent.futures.ThreadPoolExecutor(max_workers=synth_threads)

    try:
        for i, task in enumerate(sub_tasks):
            p_cb = lambda t_id, kind, note: event_queue.put(("progress", t_id, kind, note))
            worker_exec.submit(_map_reduce_worker, i + 1, task, p_cb)

        while chunks_completed < total_chunks:
            try:
                event = event_queue.get(timeout=2.0)
                last_event_time = time.time()
            except queue.Empty:
                if _shutdown_event.is_set():
                    break

                idle = time.time() - last_event_time
                if time.time() - last_heartbeat > MAP_HEARTBEAT_SECS:
                    last_heartbeat = time.time()
                    sys.stdout.write("\n")
                    last_done = time.time() - last_completion_time
                    print("    [~] Map status: workers {}/{} chunks {}/{} | {:.0f}s since last completion | "
                          "free worker slots {} | free stitch slots {}".format(
                              workers_finished, total_tasks, chunks_completed,
                              total_chunks, last_done, worker_queue.qsize(),
                              synth_queue.qsize()), flush=True)

                    _render_map_progress(workers_finished, total_tasks, chunks_completed, total_chunks, start_time_progress)

                if idle > MAP_STALL_TIMEOUT:
                    print("\n    [!] Watchdog: no map/reduce events for {}s. Aborting phase.".format(
                        int(idle)), flush=True)
                    aborted = True
                    break

                if workers_finished >= total_tasks and chunks_completed < total_chunks:
                    for chunk_idx in range(1, total_chunks + 1):
                        if chunk_idx not in submitted_chunks:
                            expected_start = (chunk_idx - 1) * SYNTHESIS_CHUNK_SIZE + 1
                            expected_end = min(expected_start + SYNTHESIS_CHUNK_SIZE, total_tasks + 1)
                            
                            chunk_tasks = []
                            for i in range(expected_start, expected_end):
                                if i in results_dict:
                                    chunk_tasks.append(results_dict[i])
                                    
                            if len(chunk_tasks) == (expected_end - expected_start):
                                submitted_chunks.add(chunk_idx)
                                synth_exec.submit(chunk_wrapper, chunk_idx, chunk_tasks)
                continue

            if event[0] == "progress":
                _, t_id, kind, note = event
                if kind != "ping":
                    sys.stdout.write("\n")
                    print(f"    [~] Thread{t_id:02d} {note}", flush=True)
                    _render_map_progress(workers_finished, total_tasks, chunks_completed, total_chunks, start_time_progress)
                continue

            if event[0] == "worker":
                last_completion_time = time.time()
                last_heartbeat = time.time()
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
                        synth_exec.submit(chunk_wrapper, chunk_idx, chunk_tasks)

                if workers_finished == total_tasks:
                    for chunk_idx in range(1, total_chunks + 1):
                        if chunk_idx not in submitted_chunks:
                            expected_start = (chunk_idx - 1) * SYNTHESIS_CHUNK_SIZE + 1
                            expected_end = min(expected_start + SYNTHESIS_CHUNK_SIZE, total_tasks + 1)
                            
                            chunk_tasks = []
                            for i in range(expected_start, expected_end):
                                if i in results_dict:
                                    chunk_tasks.append(results_dict[i])
                            
                            if len(chunk_tasks) == (expected_end - expected_start):
                                submitted_chunks.add(chunk_idx)
                                synth_exec.submit(chunk_wrapper, chunk_idx, chunk_tasks)

            elif event[0] == "chunk":
                last_completion_time = time.time()
                last_heartbeat = time.time()
                _, b_id, text, p_tok, c_tok, elap, slot_name = event
                synthesized_chunks[b_id] = text
                chunk_p_tok += p_tok
                chunk_c_tok += c_tok
                chunks_completed += 1

            _render_map_progress(workers_finished, total_tasks, chunks_completed, total_chunks, start_time_progress)

    finally:
        worker_exec.shutdown(wait=False, cancel_futures=True)
        synth_exec.shutdown(wait=False, cancel_futures=True)

    if aborted:
        print("\n    [!] Map-reduce aborted after {} of {} chunks. Waiting up to {}s for "
              "in-flight requests to time out, then exiting; re-run with -r to resume.".format(
                  chunks_completed, total_chunks, int(max(WORKER_TIMEOUT_SECS, STITCH_TIMEOUT_SECS))),
              flush=True)
        sys.exit(2)

    print()

    ordered_chunk_texts = [
        synthesized_chunks[cid] for cid in sorted(synthesized_chunks.keys())
    ]

    if len(ordered_chunk_texts) < total_chunks:
        print(f"    [!] WARNING: Map-Reduce completed but {total_chunks - len(ordered_chunk_texts)} chunks were lost.", flush=True)

    if not ordered_chunk_texts:
        print("    [!] Map-reduce produced no output chunks. Aborting.", flush=True)
        sys.exit(2)

    print(f"\n[5] CHUNK PERSISTENCE: Writing {len(ordered_chunk_texts)} chunk(s) to disk "
          f"(tree-reduction stitch disabled - no content is merged away)...", flush=True)

    chunk_paths = save_chunk_files(ordered_chunk_texts, run_dir)
    total_bytes = sum(p.stat().st_size for p in chunk_paths)
    print(f"    [+] Preserved {len(chunk_paths)} chunk(s), {total_bytes:,} bytes total, "
          f"in {chunks_dir_for(run_dir)}", flush=True)

    total_synth_p = chunk_p_tok + decomp_p_tok
    total_synth_c = chunk_c_tok + decomp_c_tok

    return (chunk_paths, worker_p_tok, worker_c_tok, total_synth_p, total_synth_c,
            worker_stats_log)

# ==============================================================================
# Phase 4: Per-Chunk Post-Processing
# ------------------------------------------------------------------------------
# Each chunk is polished independently and written back as chunk_NNN_polished.md.
# Sub-chunks from every chunk are flattened into a single parallel dedup pass so
# the stitcher slots stay saturated instead of idling between chunk files.
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

    return text_without_code, protected_blocks

def semantic_deduplication(chunk_text: str, chunk_id: int, total_chunks: int, endpoint: str, slot_name: str) -> Tuple[str, bool]:
    original = chunk_text
    placeholder_pattern = r'\[\[PROTECTED_[A-Z_]+?\d{3,}\]\]'
    chunk_inventory = re.findall(placeholder_pattern, chunk_text)

    fitted_chunk_text = fit_stitch_context(chunk_text)

    client = stitcher_client(endpoint, max_retries=0)

    system_prompt = _PROMPT_PHASE4_DEDUP

    if chunk_inventory:
        inventory_str = ", ".join(chunk_inventory)
        system_prompt += (
            f"\n\nCRITICAL ARTIFACT INVENTORY:\n"
            f"This specific text section contains the following protected placeholders: {inventory_str}\n"
            f"You MUST include EVERY SINGLE ONE of these exact placeholder strings in your rewritten output. "
            f"Even if you summarize the surrounding text, do NOT drop these placeholders. They represent vital code blocks."
        )

    for attempt in range(1, 4):
        try:
            distilled_text, _, _ = _stitch_completion(
                client, system_prompt, fitted_chunk_text, clamp_stitch_tokens(16384), 0.1, presence_penalty=0.2
            )

            if chunk_inventory:
                missing_placeholders = [p for p in chunk_inventory if p not in distilled_text]
                if missing_placeholders:
                    distilled_text += "\n\n### Recovered Chunk Artifacts\n" + "\n\n".join(missing_placeholders)

            return distilled_text, True
        except Exception as exc:
            print(f"\n    [!] Dedup segment {chunk_id}/{total_chunks} failed on {slot_name} (attempt {attempt}/3): {str(exc)[:120]}", flush=True)
            time.sleep(2)

    print(f"    [!] Dedup segment {chunk_id}/{total_chunks} exhausted retries. Passing through unedited.", flush=True)
    return original, False

def parallel_edit_chunks(chunks: List[str]) -> Tuple[List[str], int]:
    """Dedup a flat list of text segments in parallel. Returns (results, failures)."""
    if not STITCHER_ENDPOINTS:
        print("    [!] WARNING: No stitcher endpoints defined. Skipping deduplication.", flush=True)
        return list(chunks), len(chunks)

    total_chunks = len(chunks)
    endpoint_queue, total_slots = build_stitcher_slot_queue(prefix="Edit-Slot")

    results: List[str] = [""] * total_chunks
    failures = 0

    def _edit_chunk_worker(chunk_idx: int, chunk_content: str) -> Tuple[str, bool]:
        endpoint, slot_name = None, None
        deadline = time.time() + (MAX_RETRIES * STITCH_TIMEOUT_SECS)
        while time.time() < deadline:
            if _shutdown_event.is_set():
                break
            try:
                endpoint, slot_name = endpoint_queue.get(timeout=2.0)
                break
            except queue.Empty:
                continue
        if endpoint is None:
            return chunk_content, False
        try:
            return semantic_deduplication(chunk_content, chunk_idx + 1, total_chunks, endpoint, slot_name)
        except Exception as exc:
            print(f"\n    [!] Edit segment worker failed unexpectedly: {str(exc)[:120]}", flush=True)
            return chunk_content, False
        finally:
            endpoint_queue.put((endpoint, slot_name))

    pool_size = max(1, total_slots)

    print(f"    [*] Semantic Deduplication: Refining {total_chunks} segment(s) across {pool_size} stitcher slot(s) [{STITCHER_MODEL}]...", flush=True)

    start_time_progress = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
        future_to_idx = {executor.submit(_edit_chunk_worker, i, chunk): i for i, chunk in enumerate(chunks)}
        completed = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            res, success = future.result()
            results[idx] = res
            if not success:
                failures += 1

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

    if failures > 0:
        print(f"    [!] {failures} of {total_chunks} segments passed through unedited.", flush=True)

    return results, failures

def section_boundary_smoothing(full_skeleton: str, global_inventory: List[str], endpoint: str) -> Tuple[str, bool]:
    client = stitcher_client(endpoint, max_retries=0)
    system_prompt = _PROMPT_PHASE4_SMOOTH

    if global_inventory:
        system_prompt += f"\n\nCRITICAL ARTIFACT INVENTORY: {', '.join(global_inventory)}\nYou MUST retain EVERY SINGLE placeholder."

    fitted_skeleton = fit_stitch_context(full_skeleton)

    for attempt in range(1, 3):
        try:
            text, _, _ = _stitch_completion(
                client, system_prompt, fitted_skeleton, clamp_stitch_tokens(32768), 0.2, presence_penalty=0.1
            )
            return text, True
        except Exception as exc:
            print(f"\n    [!] Section boundary smoothing failed on {endpoint} (attempt {attempt}/2): {str(exc)[:120]}", flush=True)
            time.sleep(2)

    print("    [!] Section boundary smoothing exhausted retries. Passing through unedited.", flush=True)
    return full_skeleton, False

def header_unification_pass(smoothed_skeleton: str, global_inventory: List[str], endpoint: str) -> Tuple[str, bool]:
    client = stitcher_client(endpoint, max_retries=0)
    system_prompt = _PROMPT_PHASE4_UNIFY

    if global_inventory:
        system_prompt += f"\n\nCRITICAL ARTIFACT INVENTORY: {', '.join(global_inventory)}\nYou MUST retain EVERY SINGLE placeholder."

    fitted_skeleton = fit_stitch_context(smoothed_skeleton)

    for attempt in range(1, 3):
        try:
            text, _, _ = _stitch_completion(
                client, system_prompt, fitted_skeleton, clamp_stitch_tokens(32768), 0.2, presence_penalty=0.1
            )
            return text, True
        except Exception as exc:
            print(f"\n    [!] Header unification failed on {endpoint} (attempt {attempt}/2): {str(exc)[:120]}", flush=True)
            time.sleep(2)

    print("    [!] Header unification exhausted retries. Passing through unedited.", flush=True)
    return smoothed_skeleton, False

def global_consolidation_pass(full_skeleton: str, global_inventory: List[str],
                              endpoint: Optional[str] = None) -> Tuple[str, bool, bool]:
    endpoint = endpoint or next_stitcher_endpoint()

    # The STITCH_ECHO_MAX_CHARS check is deliberately lifted to the caller 
    # to better record telemetry.

    smoothed_text, smooth_ok = section_boundary_smoothing(full_skeleton, global_inventory, endpoint)
    final_text, unify_ok = header_unification_pass(smoothed_text, global_inventory, endpoint)

    missing_placeholders = [p for p in global_inventory if p not in final_text]
    if missing_placeholders:
        final_text += "\n\n### Recovered Global Artifacts\n" + "\n\n".join(missing_placeholders)

    return final_text, smooth_ok, unify_ok

def reassemble_document(distilled_text: str, protected_blocks: Dict[str, str]) -> str:
    final_text = distilled_text
    for placeholder, original_content in protected_blocks.items():
        if placeholder in final_text:
            final_text = final_text.replace(placeholder, original_content)
        else:
            final_text += f"\n\n### Orphaned Artifact\n{original_content}"
    return final_text

def run_phase4_polish_chunks(chunk_paths: List[Path], run_dir: Path,
                             skip_executive: bool = False) -> Tuple[List[Path], dict]:
    """Polish each chunk independently. Returns (polished_paths, stats)."""
    print("\n[PHASE 4] STARTING PER-CHUNK SYNTHESIS POLISH", flush=True)

    stats = {"chunks": len(chunk_paths), "segments": 0, "dedup_failures": 0,
             "exec_ok": 0, "exec_skipped": 0, "exec_skipped_size": 0, "passthrough": 0}

    # 1. Read + protect + split every chunk, remembering ownership.
    per_chunk_protected: List[Dict[str, str]] = []
    per_chunk_segments: List[List[str]] = []
    owner_index: List[int] = []
    flat_segments: List[str] = []

    for c_idx, p in enumerate(chunk_paths):
        raw = read_file_content_safe(p)
        if raw is None:
            print(f"    [!] Could not read {p.name}; skipping.", flush=True)
            per_chunk_protected.append({})
            per_chunk_segments.append([])
            continue
        skeleton, protected = extract_and_protect_blocks(raw)
        segments = split_into_logical_chunks(skeleton, MAX_CHUNK_CHARS)
        if not segments:
            segments = [skeleton]
        per_chunk_protected.append(protected)
        per_chunk_segments.append(segments)
        for s in segments:
            flat_segments.append(s)
            owner_index.append(c_idx)

    stats["segments"] = len(flat_segments)
    print(f"    [*] {len(chunk_paths)} chunk(s) -> {len(flat_segments)} editable segment(s).", flush=True)

    if not flat_segments:
        print("    [!] Nothing to polish.", flush=True)
        return list(chunk_paths), stats

    # 2. One flat parallel dedup pass across all segments.
    edited_flat, dedup_failures = parallel_edit_chunks(flat_segments)
    stats["dedup_failures"] = dedup_failures

    # 3. Regroup by owning chunk, optionally run the executive editor, reassemble.
    regrouped: Dict[int, List[str]] = {}
    for seg_text, owner in zip(edited_flat, owner_index):
        regrouped.setdefault(owner, []).append(seg_text)

    polished_paths: List[Path] = []
    for c_idx, p in enumerate(chunk_paths):
        segs = regrouped.get(c_idx, [])
        protected = per_chunk_protected[c_idx]
        if not segs:
            polished_paths.append(p)
            stats["passthrough"] += 1
            continue

        skeleton = "\n\n".join(segs)
        inventory = list(protected.keys())

        if skip_executive or len(segs) <= 1:
            final_skeleton = skeleton
            stats["exec_skipped"] += 1
        elif len(skeleton) > STITCH_ECHO_MAX_CHARS:
            print(f"    [!] Skipping executive editor passes for {p.name}: skeleton is {len(skeleton):,} chars "
                  f"but the stitcher can only ingest ~{STITCH_ECHO_MAX_CHARS:,}.", flush=True)
            final_skeleton = skeleton
            stats["exec_skipped_size"] += 1
            stats["exec_skipped"] += 1
        else:
            final_skeleton, smooth_ok, unify_ok = global_consolidation_pass(skeleton, inventory)
            if smooth_ok and unify_ok:
                stats["exec_ok"] += 1
            else:
                stats["exec_skipped"] += 1

        final_md = reassemble_document(final_skeleton, protected)
        out_path = p.parent / f"{p.stem}_polished.md"
        with open(out_path, "w", encoding="ascii") as f:
            f.write(enforce_ascii(final_md).strip() + "\n")
        polished_paths.append(out_path)
        print(f"    [+] Polished {p.name} -> {out_path.name} ({out_path.stat().st_size:,} bytes)", flush=True)

    return polished_paths, stats

def build_polish_manifest(polished_paths: List[Path], stats: dict) -> str:
    total = sum(p.stat().st_size for p in polished_paths if p.exists())
    lines = ["# Polished Synthesis Manifest", ""]
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Chunks polished:** {len(polished_paths)}")
    lines.append(f"- **Editable segments processed:** {stats.get('segments', 0)}")
    lines.append(f"- **Segments passed through unedited:** {stats.get('dedup_failures', 0)}")
    lines.append(f"- **Executive editor applied:** {stats.get('exec_ok', 0)} chunk(s)")
    lines.append(f"- **Executive editor skipped:** {stats.get('exec_skipped', 0)} chunk(s)")
    lines.append(f"- **Executive editor skipped (size limit):** {stats.get('exec_skipped_size', 0)} chunk(s)")
    lines.append(f"- **Total polished corpus:** {total:,} bytes")
    lines.append("")
    lines.append("## Polished Chunk Index")
    lines.append("")
    lines.append("| # | File | Bytes |")
    lines.append("|---|------|-------|")
    for idx, p in enumerate(polished_paths, start=1):
        size = p.stat().st_size if p.exists() else 0
        lines.append(f"| {idx} | `{CHUNKS_DIRNAME}/{p.name}` | {size:,} |")
    lines.append("")
    return enforce_ascii("\n".join(lines))

# ==============================================================================
# Phase 5: Automatic Unittests (harvests artifacts from every chunk)
# ==============================================================================

def _format_execution_report_as_markdown(report_data: list) -> str:
    if not report_data: return ""
    lines = ["## Test Execution Status\n", "| Chunk | Artifact | Language | Status | Detail |", "|---|---|---|---|---|"]
    for res in report_data:
        lines.append(
            f"| {res.get('chunk', '')} | {res.get('filename', '')} | {res.get('language', '')} | "
            f"**{res.get('status', '')}** | {res.get('message', '')} |"
        )
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

        if "</file>" in line and not in_block:
            detected_filename = None
            i += 1
            continue

        xml_match = re.search(r'<file path="([^"]+)">', line)
        if xml_match and not in_block:
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
                comment_match = re.match(r'^(?:#|//)\s*([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]{1,10})\s*$', next_line)
                if comment_match:
                    detected_filename = comment_match.group(1).strip()
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

def request_unittests_from_worker(artifact: dict, endpoint_queue: queue.Queue, test_output_dir: Path,
                                  progress_lock: threading.Lock, progress_state: dict) -> Optional[dict]:
    if artifact["language"].lower() not in {"python", "py", "cpp", "c", "bash", "sh"}:
        return None

    endpoint_url = None
    deadline = time.time() + (MAX_RETRIES * TEST_TIMEOUT_SECS)
    while time.time() < deadline:
        if _shutdown_event.is_set():
            break
        try:
            endpoint_url = endpoint_queue.get(timeout=5.0)
            break
        except queue.Empty:
            continue
    if endpoint_url is None:
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
                response = requests.post(endpoint_url, json=payload, timeout=TEST_TIMEOUT_SECS)
                response.raise_for_status()
                result = response.json()
                choices = result.get("choices")

                if not choices:
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER))
                        continue
                    return None

                test_code = choices[0].get("message", {}).get("content", "")
                if not test_code:
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER))
                        continue
                    return None

                test_code = enforce_ascii(_strip_markdown_fences(test_code))
                chunk_tag = artifact.get("chunk", "chunk")
                chunk_test_dir = test_output_dir / chunk_tag
                chunk_test_dir.mkdir(parents=True, exist_ok=True)
                test_filename = f"test_{artifact['filename']}"
                test_filepath = chunk_test_dir / test_filename

                with open(test_filepath, "w", encoding="ascii") as f:
                    f.write(test_code + "\n")

                with progress_lock:
                    progress_state["done"] += 1
                    eta_str = _format_eta(progress_state["start_time"], progress_state["done"], progress_state["total"])
                    print(f"    [+] Generated tests ({progress_state['done']}/{progress_state['total']}) "
                          f"| ETC: {eta_str} -> {chunk_tag}/{test_filename}", flush=True)

                generation_metadata = {
                    "filename": test_filename,
                    "test_filepath": str(test_filepath),
                    "language": artifact["language"],
                    "artifact_filepath": artifact["filepath"],
                    "chunk": chunk_tag,
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
    result = {"filename": test_meta["filename"], "language": lang,
              "status": "UNKNOWN", "message": "", "chunk": test_meta.get("chunk", "")}
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

def run_phase5_automatic_unittests(source_paths: List[Path], workspace: Path):
    print(f"\n[PHASE 5] STARTING AUTOMATED UNITTEST PIPELINE", flush=True)

    source_paths = [p for p in source_paths if p.exists()]
    if not source_paths:
        print("Error: No source chunks available for artifact extraction.", flush=True)
        return

    EXTRACTED_ARTIFACTS_DIR = workspace / "artifacts" / "extracted"
    TEST_OUTPUT_DIR = workspace / "tests"
    REPORT_OUTPUT_DIR = workspace / "reports"

    endpoint_concurrency = CONCURRENT_REQS_PER_ENDPOINT
    total_gen_workers = len(TEST_WORKER_ENDPOINTS) * endpoint_concurrency
    endpoint_queue: queue.Queue = queue.Queue()
    for ep in TEST_WORKER_ENDPOINTS:
        for _ in range(endpoint_concurrency):
            endpoint_queue.put(ep)

    # Harvest artifacts from EVERY chunk, keeping them namespaced per chunk.
    all_artifacts: list = []
    for p in source_paths:
        md_content = read_file_content_safe(p)
        if md_content is None:
            print(f"    [!] Could not read {p.name}; skipping.", flush=True)
            continue

        if re.search(r'\[\[PROTECTED_[A-Z_]+\d+\]\]', md_content):
            print(f"    [!] WARNING: {p.name} contains unresolved placeholder tokens. "
                  "Phase 4 reassembly may have failed for this chunk.", flush=True)

        chunk_tag = re.sub(r'_polished$', '', p.stem)
        chunk_out = EXTRACTED_ARTIFACTS_DIR / chunk_tag
        found = extract_code_blocks(md_content, chunk_out)
        for a in found:
            a["chunk"] = chunk_tag
        all_artifacts.extend(found)
        print(f"    [+] {p.name}: extracted {len(found)} code artifact(s).", flush=True)

    valid_langs = {"python", "py", "cpp", "c", "bash", "sh"}
    testable_artifacts = [a for a in all_artifacts if a["language"].lower() in valid_langs]
    print(f"    [*] {len(all_artifacts)} artifact(s) total, {len(testable_artifacts)} testable.", flush=True)

    if not testable_artifacts:
        print("    [!] No testable artifacts found across any chunk.", flush=True)
        return

    progress_lock = threading.Lock()
    progress_state = {"done": 0, "total": len(testable_artifacts), "start_time": time.time()}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=total_gen_workers)
    generated_tests: list = []

    try:
        futures = [executor.submit(request_unittests_from_worker, artifact, endpoint_queue,
                                   TEST_OUTPUT_DIR, progress_lock, progress_state)
                   for artifact in testable_artifacts]
        for future in concurrent.futures.as_completed(futures):
            try:
                test_meta = future.result()
                if test_meta:
                    generated_tests.append(test_meta)
            except Exception:
                pass
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    req_files = [a for a in all_artifacts if "requirements" in a["filename"].lower() and a["filename"].endswith(".txt")]
    for req in req_files:
        req_path = Path(req["filepath"]).resolve()
        _sanitize_requirements_file(req_path)
        try:
            res = subprocess.run(["python", "-m", "pip", "install", "--break-system-packages", "-r", str(req_path)],
                                 capture_output=True, encoding="ascii", errors="ignore", timeout=120)
            if res.returncode != 0:
                print(f"    [!] Warning: pip install failed for {req['filename']}.", flush=True)
        except Exception as e:
            print(f"    [!] Warning: pip install exception for {req['filename']}: {e}", flush=True)

    execution_results: list = []
    if generated_tests:
        print(f"    [*] Executing {len(generated_tests)} generated test file(s)...", flush=True)
        exec_executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_EXEC_WORKERS)
        try:
            exec_futures = [exec_executor.submit(execute_test_artifact, tm) for tm in generated_tests]
            for future in concurrent.futures.as_completed(exec_futures):
                try:
                    execution_results.append(future.result())
                except Exception:
                    pass
        finally:
            exec_executor.shutdown(wait=True, cancel_futures=True)

    if execution_results:
        REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        json_report_path = REPORT_OUTPUT_DIR / "execution_report.json"
        csv_report_path = REPORT_OUTPUT_DIR / "execution_report.csv"
        with open(json_report_path, "w", encoding="ascii") as f:
            json.dump(execution_results, f, indent=4)
        with open(csv_report_path, "w", newline="", encoding="ascii") as f:
            writer = csv.DictWriter(f, fieldnames=EXECUTION_RESULT_FIELDS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(execution_results)

        passed = sum(1 for r in execution_results if r["status"] == "PASSED")
        print(f"    [+] Test execution complete: {passed}/{len(execution_results)} passed. "
              f"Reports in {REPORT_OUTPUT_DIR}", flush=True)

# ==============================================================================
# Phase 6: Todo Project Distillation
# ==============================================================================

def run_phase6_project_distillation(project_dir: Path, iterate: bool = False):
    print(f"\n[PHASE 6] STARTING PROJECT DISTILLATION", flush=True)
    project_name = project_dir.name

    # Phase 6 bases its tasks on the raw ingested source (Phase 0/1) plus test
    # telemetry. The synthesized chunks are excluded: they are derived output and
    # would swamp the stitcher input window.
    exclude_dirs = {"tests", "tasks", "artifacts", "reports", CHUNKS_DIRNAME}
    p6_exclude_names = {"DISTILLED_TASKS", "project_state", "FINAL_SYNTHESIS", "POLISHED_SYNTHESIS"}

    existing_tasks = ""
    tasks_path = project_dir / "DISTILLED_TASKS.md"
    if iterate and tasks_path.exists():
        existing_tasks = read_file_content_safe(tasks_path) or ""
        print(f"    [*] Found existing DISTILLED_TASKS.md. Operating in iterative refinement mode.", flush=True)

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

            if file_path.name.endswith("_distilled.md"):
                continue

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
    client = stitcher_client()

    if existing_tasks:
        half_budget = MAX_STITCH_CONTEXT_CHARS // 2
        existing_tasks_trimmed = fit_stitch_context(
            existing_tasks, budget=half_budget - 1000,
            note="...[EXISTING TASKS TRUNCATED FOR CONTEXT LIMITS]..."
        )
        trimmed_telemetry = fit_stitch_context(
            full_text, budget=half_budget - 1000,
            note="...[NEW TELEMETRY TRUNCATED FOR CONTEXT LIMITS]..."
        )
        sys_prompt = _PROMPT_PHASE6_ITERATE
        user_content = (
            f"--- CURRENT DISTILLED TASKS ---\n{existing_tasks_trimmed}\n\n"
            f"--- NEW TELEMETRY AND DOCS ---\n{trimmed_telemetry}"
        )
    else:
        sys_prompt = _PROMPT_PHASE6_DISTILL
        trimmed_telemetry = fit_stitch_context(
            full_text, budget=MAX_STITCH_CONTEXT_CHARS - 1000,
            note="...[CONTENT TRUNCATED FOR CONTEXT LIMITS]..."
        )
        user_content = (
            f"Extract actionable tasks, test outcomes, and relevant artifacts "
            f"from this {project_name} documentation:\n\n{trimmed_telemetry}"
        )

    distilled_markdown = None
    try:
        distilled_markdown, _, _ = _stitch_completion(
            client, sys_prompt, user_content, clamp_stitch_tokens(8192), 0.2
        )
    except Exception as e:
        print(f"[!] Stitcher inference failed: {e}", flush=True)

    if distilled_markdown:
        try:
            with open(tasks_path, "w", encoding="ascii") as f:
                f.write(f"# Distilled Tasks: {project_name}\n\n{distilled_markdown}\n")
            print(f"[{project_name}] Successfully saved distilled tasks to {tasks_path.name}", flush=True)
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
    parser.add_argument("--iterate", action="store_true", help="Enable iterative distillation for Phase 6.")
    parser.add_argument("--no-polish", action="store_true", help="Skip Phase 4 entirely; test the raw synthesis chunks.")
    parser.add_argument("--no-executive", action="store_true", help="Run per-segment dedup but skip the two executive editor passes.")

    args = parser.parse_args()

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

    print(describe_budget_alignment(), flush=True)

    verify_server_props([GEN_API_BASE], "Apex", APEX_SERVER_CTX, APEX_SERVER_NP)
    verify_server_props(WORKER_ENDPOINTS, "Worker", WORKER_SERVER_CTX, WORKER_SERVER_NP)
    verify_server_props(STITCHER_ENDPOINTS, "Stitcher", STITCH_SERVER_CTX, STITCH_SERVER_NP)

    apex_ok = ping_tier([GEN_API_BASE], LLM_MODEL, GEN_API_KEY, "Apex", timeout=90.0)
    worker_ok = ping_tier(WORKER_ENDPOINTS, WORKER_MODEL, WORKER_API_KEY, "Worker", timeout=90.0)
    stitch_ok = ping_tier(STITCHER_ENDPOINTS, STITCHER_MODEL, STITCHER_API_KEY, "Stitcher", timeout=90.0)

    if not worker_ok or not stitch_ok:
        print("\n[!] Critical cluster tiers (Worker/Stitcher) failed smoke tests. Aborting.", flush=True)
        sys.exit(1)

    raw_filepath = None
    distilled_filepath = None
    final_file_path = target_directory / "FINAL_SYNTHESIS.md"
    output_path = target_directory / "POLISHED_SYNTHESIS.md"
    report_path = target_directory / "reports" / "execution_report.json"
    distilled_tasks_path = target_directory / "DISTILLED_TASKS.md"

    if args.resume:
        md_files = list(target_directory.glob("*.md"))
        valid_raw = [
            f for f in md_files
            if re.match(r'^\d{8}_\d{6}_.*\.md$', f.name) and not f.name.endswith('_distilled.md')
        ]

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

    # ---------------- Phase 1 / Phase 0 ----------------
    if args.resume and raw_filepath and raw_filepath.exists():
        print(f"[PHASE 1] Bypassed. Resuming from existing raw file: {raw_filepath.name}")
    elif args.git:
        raw_filepath = ingest_git_repository(args.git, target_directory, args.focus, args.git_path)
    else:
        if not apex_ok:
            print("\n[!] Apex tier offline and generation required. Aborting.", flush=True)
            sys.exit(1)
        raw_filepath = generate_content(target_prompt, target_directory)

    # ---------------- Phase 2 ----------------
    if args.resume and distilled_filepath and distilled_filepath.exists():
        print(f"[PHASE 2] Bypassed. Resuming from existing distilled file: {distilled_filepath.name}")
    else:
        if not apex_ok:
            print("\n[!] Apex tier offline; Phase 2 distillation cannot run. "
                  "Re-run with -r once 8081 is healthy.", flush=True)
            sys.exit(1)

        raw_content = read_file_content_safe(raw_filepath)
        if raw_content is None:
            print(f"[!] Fatal: Could not read {raw_filepath}.", flush=True)
            sys.exit(1)
        actionable_tasks = distill_document(raw_content)
        distilled_filepath = save_distilled_output(actionable_tasks, raw_filepath)

    # ---------------- Phase 3 ----------------
    existing_chunks = list_chunk_files(target_directory, polished=False)
    if args.resume and existing_chunks and final_file_path.exists():
        print(f"[PHASE 3] Bypassed. Resuming from {len(existing_chunks)} existing chunk(s).")
        chunk_paths = existing_chunks
    else:
        target_query = read_file_content_safe(distilled_filepath)
        if target_query is None:
            print(f"[!] Fatal: Could not read {distilled_filepath}.", flush=True)
            sys.exit(1)

        master_start_time = time.time()
        fragments, p_tok, c_tok = decompose_to_atomic_pieces(target_query)

        export_to_split_files(fragments, target_directory)
        (chunk_paths, w_p, w_c, syn_p, syn_c, worker_stats) = execute_continuous_map_reduce(
            fragments, target_query, target_directory, decomp_p_tok=p_tok, decomp_c_tok=c_tok)

        master_elapsed_time = time.time() - master_start_time

        truncated_count = sum(1 for s in worker_stats if s.get("truncated"))
        if truncated_count > 0:
            print(f"    [!] WARNING: {truncated_count} workers were truncated due to exceeding MAX_WALL_SECS.", flush=True)

        manifest = build_synthesis_manifest(
            chunk_paths, worker_stats, target_query, master_elapsed_time,
            w_p, w_c, syn_p, syn_c, truncated_count
        )
        with open(final_file_path, "w", encoding="ascii") as f:
            f.write(manifest)
        print(f"[+] Synthesis manifest written to {final_file_path.name} "
              f"({len(chunk_paths)} chunk(s) indexed).", flush=True)

    # ---------------- Phase 4 ----------------
    existing_polished = [p for p in chunks_dir_for(target_directory).glob("chunk_*_polished.md")]
    if args.no_polish:
        print("\n[PHASE 4] Skipped by --no-polish. Using raw synthesis chunks downstream.")
        source_paths = chunk_paths
    elif args.resume and existing_polished and output_path.exists():
        print(f"[PHASE 4] Bypassed. Resuming from {len(existing_polished)} polished chunk(s).")
        source_paths = list_chunk_files(target_directory, polished=True)
    else:
        source_paths, polish_stats = run_phase4_polish_chunks(
            chunk_paths, target_directory, skip_executive=args.no_executive
        )
        with open(output_path, "w", encoding="ascii") as f:
            f.write(build_polish_manifest(source_paths, polish_stats))
        print(f"[+] Polish manifest written to {output_path.name}", flush=True)

    # ---------------- Phase 5 ----------------
    if args.resume and report_path.exists():
        print(f"[PHASE 5] Bypassed. Resuming from existing execution_report.json")
    else:
        run_phase5_automatic_unittests(source_paths, target_directory)

    # ---------------- Phase 6 ----------------
    if args.resume and distilled_tasks_path.exists() and not args.iterate:
        print(f"\n[PHASE 6] Bypassed. DISTILLED_TASKS.md already exists. Use --iterate to force re-run.")
    else:
        run_phase6_project_distillation(target_directory, iterate=args.iterate)

    print("\n==============================================================================")
    print("PIPELINE COMPLETE")
    print(f"  Chunks:    {chunks_dir_for(target_directory)}")
    print(f"  Manifests: {final_file_path.name}, {output_path.name}")
    print("==============================================================================\n")

if __name__ == "__main__":
    main()
