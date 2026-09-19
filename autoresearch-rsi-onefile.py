#!/usr/bin/env python3
# -*- coding: ascii -*-

import os
import socket
import sys
import json
import time
import re
import argparse
import ast
import concurrent.futures
import queue
import threading
import subprocess
import csv
import hashlib
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

# Extension -> executable test language, used by Phase 5 on real work/ files.
_TESTABLE_EXT_LANG = {
    ".py": "python", ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    ".cc": "cpp", ".cxx": "cpp", ".sh": "bash", ".bash": "bash",
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
# TIER POLICY (post-stitcher-removal):
#   * APEX (8081) runs non-worker, non-agent tasks only: Phase 1 generation,
#     Phase 2 distillation, Phase 3 decomposition (planning), Phase 6
#     distillation. It is -np 1 and therefore strictly serial.
#   * WORKERS (8033, 8034, 8070, 8071) run every agent assignment plus Phase 0
#     repo map-reduce. 8070/8071 are the former stitcher nodes, folded into the
#     agent pool.
#   * There is NO stitcher tier. Consolidation is mechanical (filesystem walk +
#     hashing), never a model merge.
#
# The worker pool is now HETEROGENEOUS in -c (8033/8034 were 196608, 8070/8071
# were 131072). Budget from the SMALLEST node in the pool or the larger nodes
# will silently over-subscribe. WORKER_SERVER_CTX defaults to the 131072 floor.
# ==============================================================================

# Apex / planning / generation / distillation node (port 8081): -c 65536 -np 1
APEX_SERVER_CTX = int(os.getenv("APEX_SERVER_CTX", "65536"))
APEX_SERVER_NP = int(os.getenv("APEX_SERVER_NP", "1"))

# Agent worker cluster (8033, 8034, 8070, 8071): -np 2 --kv-unified each.
# Budget against the smallest -c in the pool.
WORKER_SERVER_CTX = int(os.getenv("WORKER_SERVER_CTX", "131072"))
WORKER_SERVER_NP = int(os.getenv("WORKER_SERVER_NP", "2"))

# Concurrency-safe per-request windows.
APEX_CONTEXT_TOKENS = max(4096, APEX_SERVER_CTX // max(1, APEX_SERVER_NP))
WORKER_CONTEXT_TOKENS = max(4096, WORKER_SERVER_CTX // max(1, WORKER_SERVER_NP))

# Phase 1: Raw Generation Config
GEN_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:9931/v1")
GEN_API_KEY = os.getenv("OPENAI_API_KEY", "sk-local")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.8-Flash-Next-UD-IQ4_XS")

# Unified Context Limits
CHARS_PER_TOKEN = float(os.getenv("CHARS_PER_TOKEN", "3.5"))
APEX_MAX_OUTPUT_TOKENS = int(os.getenv("APEX_MAX_OUTPUT_TOKENS", "8192"))
APEX_GEN_TOKENS = int(os.getenv("APEX_GEN_TOKENS", "4096"))
APEX_RESERVE_TOKENS = int(os.getenv("APEX_RESERVE_TOKENS", "2048"))

# Planning (decomposition) emits a <=20 element JSON array; it never needs the
# full apex output ceiling.
APEX_PLAN_TOKENS = min(int(os.getenv("APEX_PLAN_TOKENS", "4096")), APEX_MAX_OUTPUT_TOKENS)
APEX_DISTILL_TOKENS = min(int(os.getenv("APEX_DISTILL_TOKENS", "8192")), APEX_MAX_OUTPUT_TOKENS)
# Policy revisions are short programs, not documents.
APEX_POLICY_TOKENS = min(int(os.getenv("APEX_POLICY_TOKENS", "4096")), APEX_MAX_OUTPUT_TOKENS)

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

# Phase 2: Distillation Config (apex tier)
DISTILLER_URL = os.getenv("DISTILLER_URL", "http://localhost:9931/v1")
DISTILLER_MODEL = os.getenv("DISTILLER_MODEL", "Qwen3.8-Flash-Next-UD-IQ4_XS")
DISTILLER_API_KEY = os.getenv("DISTILLER_API_KEY", "local-sk")

# Restrict decomposer to not bloat out pipeline operations on simple prompts
MAX_DECOMPOSE_TASKS = int(os.getenv("MAX_DECOMPOSE_TASKS", "20"))
MAX_RETRIES = 3

# Phase 3: Agent worker cluster
WORKER_ENDPOINTS = [
    ep.strip() for ep in os.getenv(
        "WORKER_ENDPOINTS",
        "http://localhost:8030/v1,http://localhost:8031/v1,"
        "http://localhost:8032/v1,http://localhost:8033/v1,"
        "http://localhost:8034/v1,http://localhost:8035/v1"
    ).split(",") if ep.strip()
]
WORKER_MODEL = os.getenv("WORKER_MODEL", "Qwen3.8-9B-Q4_K_M.gguf")
WORKER_API_KEY = os.getenv("WORKER_API_KEY", "local-sk")

WORKER_PARALLEL_SLOTS = min(
    int(os.getenv("WORKER_PARALLEL_SLOTS", str(WORKER_SERVER_NP))),
    WORKER_SERVER_NP
)
WORKER_RETRIES = 3
WORKER_TIMEOUT_SECS = float(os.getenv("WORKER_TIMEOUT_SECS", "300.0"))
WORKER_RESERVE_TOKENS = int(os.getenv("WORKER_RESERVE_TOKENS", "2048"))

# ------------------------------------------------------------------------------
# Agent input budget.
#
# An agent prompt is no longer just "background + objective". It now carries the
# roster (every sibling agent and the scope that is explicitly NOT this agent's)
# and a digest of prior-wave comms. Those are what replace the old shared-context
# flooding, so they need hard sub-budgets or MAX_WORKER_TOKENS collapses to its
# floor and every agent truncates mid-file.
# ------------------------------------------------------------------------------
WORKER_INPUT_CHARS = min(
    int(os.getenv("WORKER_INPUT_CHARS", "90000")),
    int(max(4096, WORKER_CONTEXT_TOKENS - WORKER_RESERVE_TOKENS - 8192) * CHARS_PER_TOKEN)
)

MAX_WORKER_TOKENS = min(
    int(os.getenv("MAX_WORKER_TOKENS", "8192")),
    max(1024, WORKER_CONTEXT_TOKENS - WORKER_RESERVE_TOKENS
        - int(WORKER_INPUT_CHARS / CHARS_PER_TOKEN))
)

AGENT_CONTEXT_BUDGET = int(os.getenv("AGENT_CONTEXT_BUDGET", str(int(WORKER_INPUT_CHARS * 0.35))))
AGENT_ROSTER_BUDGET = int(os.getenv("AGENT_ROSTER_BUDGET", str(int(WORKER_INPUT_CHARS * 0.15))))
AGENT_COMMS_BUDGET = int(os.getenv("AGENT_COMMS_BUDGET", str(int(WORKER_INPUT_CHARS * 0.35))))
AGENT_OBJECTIVE_BUDGET = int(os.getenv("AGENT_OBJECTIVE_BUDGET", str(int(WORKER_INPUT_CHARS * 0.15))))
# Continuations: the parent's deliverables are carved OUT of the context budget
# (background shrinks), so the total input window is unchanged.
AGENT_PARENT_BUDGET = min(AGENT_CONTEXT_BUDGET,
                          int(os.getenv("AGENT_PARENT_BUDGET", str(int(AGENT_CONTEXT_BUDGET * 0.7)))))

WORKER_MIN_DECODE_TPS = float(os.getenv("WORKER_MIN_DECODE_TPS", "4.0"))
WORKER_MAX_WALL_SECS = float(os.getenv(
    "WORKER_MAX_WALL_SECS",
    str(max(600.0, MAX_WORKER_TOKENS / WORKER_MIN_DECODE_TPS + 120.0))))

REPO_WORKER_WALL_SECS = float(os.getenv(
    "REPO_WORKER_WALL_SECS",
    str(WORKER_TIMEOUT_SECS * WORKER_RETRIES)))

COMMS_PEER_SUMMARY_CHARS = int(os.getenv("COMMS_PEER_SUMMARY_CHARS", "700"))

# Directory layout
WORK_DIRNAME = "work"
COMMS_DIRNAME = "comms"
TREES_DIRNAME = "trees"
POLICY_DIRNAME = "policy"
DREAM_DIRNAME = "dream"
ABORTED_DIRNAME = "aborted"

# ------------------------------------------------------------------------------
# Recursive self-improvement at the exploration layer (Dream-RSI).
#
# ROUND_BUDGET is the currency the exploration policy spends: one unit = one
# agent call. It is what makes online rounds and dreamed replays comparable, so
# a policy cannot win by quietly spending more.
# ------------------------------------------------------------------------------
DEFAULT_ROUNDS = int(os.getenv("RSI_ROUNDS", "1"))
MAX_ROUNDS = int(os.getenv("RSI_MAX_ROUNDS", "12"))
ROUND_BUDGET_PER_TASK = float(os.getenv("ROUND_BUDGET_PER_TASK", "2.0"))
ROUND_BUDGET_MIN = int(os.getenv("ROUND_BUDGET_MIN", "3"))
ROUND_BUDGET_MAX = int(os.getenv("ROUND_BUDGET_MAX", "60"))

# Offline policy improvement. Candidates are cheap (zero agent calls), but each
# is one serial apex generation, so this is the real wall-clock cost of dreaming.
DREAM_CANDIDATES = int(os.getenv("DREAM_CANDIDATES", "3"))
# Coverage weighting: a policy that reaches a few tasks brilliantly must not beat
# one that reaches the whole roster well.
DREAM_COVERAGE_FLOOR = float(os.getenv("DREAM_COVERAGE_FLOOR", "0.4"))

# Policy sandbox limits.
POLICY_MAX_CHARS = int(os.getenv("POLICY_MAX_CHARS", "20000"))
POLICY_MAX_STEPS = int(os.getenv("POLICY_MAX_STEPS", "400"))
POLICY_MAX_FANOUT = int(os.getenv("POLICY_MAX_FANOUT", "16"))
# Cap on ALL interface calls (reads included), so a loop that only polls
# budget_left() and swallows errors still terminates promptly.
POLICY_MAX_RPC = int(os.getenv("POLICY_MAX_RPC", str(POLICY_MAX_STEPS * 10)))

# Evaluator weights. Deterministic and cheap by requirement: replay reads stored
# node scores rather than recomputing, so nondeterminism here would make
# dreaming lie about the past.
EVAL_W_STATUS = float(os.getenv("EVAL_W_STATUS", "0.35"))
EVAL_W_FILES = float(os.getenv("EVAL_W_FILES", "0.25"))
EVAL_W_LOG = float(os.getenv("EVAL_W_LOG", "0.15"))
EVAL_W_NOVELTY = float(os.getenv("EVAL_W_NOVELTY", "0.25"))
EVAL_W_VIOLATION = float(os.getenv("EVAL_W_VIOLATION", "0.20"))
EVAL_W_TRUNCATED = float(os.getenv("EVAL_W_TRUNCATED", "0.15"))
# Mix of creation-time heuristic vs Phase 5 pass rate once tests have run.
EVAL_HEURISTIC_MIX = float(os.getenv("EVAL_HEURISTIC_MIX", "0.6"))
# Untested nodes are blended against this prior instead of being left on the raw
# heuristic scale. Without it a tested node (<= heuristic unless every test
# passes) is systematically outranked by its untested siblings.
EVAL_UNTESTED_PRIOR = float(os.getenv("EVAL_UNTESTED_PRIOR", "0.5"))
# Status credit for an attempt that hit its output limit but still yielded
# complete, closed <file> blocks (salvaged).
EVAL_PARTIAL_STATUS_FRAC = float(os.getenv("EVAL_PARTIAL_STATUS_FRAC", "0.5"))

# Replay accounting. A request for a branch history never recorded would have
# cost a real agent call online, so replay charges for it as well. Otherwise
# probing the edge of the dream is free and replay cost stops meaning anything.
REPLAY_CHARGE_UNRECORDED = os.getenv("REPLAY_CHARGE_UNRECORDED", "1") == "1"

# Dream selection. Scores within DREAM_TIE_EPS of the incumbent are ties; ties
# keep the incumbent unless DREAM_PREFER_CHEAPER=1. Cheaper-on-tie is what turns
# a saturated evaluator into a ratchet toward shallower search.
DREAM_TIE_EPS = float(os.getenv("DREAM_TIE_EPS", "0.001"))
DREAM_PREFER_CHEAPER = os.getenv("DREAM_PREFER_CHEAPER", "0") == "1"

# Replay cost term (the paper's beta_1). Without it the objective is blind to
# spend: under a shared budget cap a policy that exhausts the cap weakly
# dominates one that stops early, so "conserve while progress is good" can never
# win a dream. Cost is normalised by the policy budget: 0.05 means spending the
# whole budget costs 0.05 score. 0 restores the old spend-blind objective.
DREAM_COST_WEIGHT = float(os.getenv("DREAM_COST_WEIGHT", "0.05"))

# Replay evaluates untested nodes against the pool's empirical test pass rate
# instead of the fixed EVAL_UNTESTED_PRIOR. With a fixed 0.5 and a real pass
# rate below it, replay rewards policies that route AROUND tested nodes.
# "fixed" restores the old behaviour. Below the minimum sample size the fixed
# prior is used.
EVAL_UNTESTED_PRIOR_MODE = os.getenv("EVAL_UNTESTED_PRIOR_MODE", "empirical").lower()
EVAL_EMPIRICAL_MIN_TESTED = int(os.getenv("EVAL_EMPIRICAL_MIN_TESTED", "4"))

# Off-policy support probes. A fraction of each live round is held back from the
# deployed policy and spent mechanically on expansions the policy family tends
# not to make (refine each task's best node; open a further independent root
# attempt). Without them the recorded pool only contains branches the logging
# policy chose, and replay can only ever confirm that policy. Applied in BOTH
# arms (dream and --no-dream) so the comparison stays at equal budget.
SUPPORT_PROBE_FRAC = float(os.getenv("SUPPORT_PROBE_FRAC", "0.15"))

# Policy isolation. Policies run in a child process that talks to the explorer
# over a line-JSON RPC; these are that child's resource limits.
POLICY_CPU_SECS = int(os.getenv("POLICY_CPU_SECS", "20"))
POLICY_MEM_MB = int(os.getenv("POLICY_MEM_MB", "512"))
POLICY_REPLAY_WALL_SECS = float(os.getenv("POLICY_REPLAY_WALL_SECS", "60"))

# Phase 5 hardening.
TEST_NODES_PER_TASK = max(1, int(os.getenv("TEST_NODES_PER_TASK", "2")))
TEST_PIP_INSTALL = os.getenv("TEST_PIP_INSTALL", "1") == "1"
TEST_PIP_ALLOWLIST = {p.strip().lower().replace("_", "-")
                      for p in os.getenv("TEST_PIP_ALLOWLIST", "").split(",") if p.strip()}
TEST_CPU_SECS = int(os.getenv("TEST_CPU_SECS", "60"))
TEST_MEM_MB = int(os.getenv("TEST_MEM_MB", "4096"))
TEST_FSIZE_MB = int(os.getenv("TEST_FSIZE_MB", "128"))

# Phase 0: permit LAN git hosts (e.g. a self-hosted Gitea). Off = fail closed.
GIT_ALLOW_PRIVATE_HOSTS = os.getenv("GIT_ALLOW_PRIVATE_HOSTS", "0") == "1"

# Phase 5: Automatic Unittests Config (runs on the agent worker pool)
TEST_WORKER_ENDPOINTS = [ep.rstrip("/") + "/chat/completions" for ep in WORKER_ENDPOINTS]
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
EXECUTION_RESULT_FIELDS = ["agent", "node", "filename", "language", "status", "message"]

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

# Planning runs on APEX. Tasks must be disjoint by construction: the whole point
# of the roster is that an agent can be told what is NOT its job, which is only
# meaningful if the decomposition drew real boundaries in the first place.
_PROMPT_PHASE3_DECOMPOSE = (
    "You are an algorithmic work-partitioner for a team of parallel agents.\n"
    "Shatter the incoming task into atomic assignments that are MUTUALLY EXCLUSIVE.\n"
    "CRITICAL PARTITIONING RULES:\n"
    "1. No two assignments may produce the same file, module, or artifact.\n"
    "2. Each assignment must name the concrete deliverable it owns.\n"
    "3. Prefer splitting by component or file boundary, never by 'do the same thing from another angle'.\n"
    "4. If two pieces of work must touch the same artifact, merge them into ONE assignment.\n"
    "Output ONLY a valid, flat JSON array of strings. No markdown formatting, no conversational text.\n"
    "Generate between 3 and 20 assignments. Do not generate fewer than 3 or more than 20 regardless of input size."
)

# The agent prompt replaces both the old worker prompt and the stitcher entirely.
# The agent writes its own deliverables and its own log; nothing downstream
# merges its prose with anyone else's.
_PROMPT_PHASE3_AGENT = (
    "You are one agent in a coordinated team. Each teammate has a DIFFERENT assignment "
    "and its own output directory. You will be shown the full team roster and the log of "
    "what previous waves already produced.\n"
    "\n"
    "SCOPE DISCIPLINE (most important rule):\n"
    "1. Do ONLY your own objective. Never produce a file that the roster assigns to another agent.\n"
    "2. If your work depends on a teammate's deliverable, DO NOT rebuild it. Reference it by path "
    "and state the dependency in your log.\n"
    "3. If you believe a teammate's deliverable is wrong or missing, do not fix it yourself. "
    "Raise it with a note to that agent.\n"
    "4. Broader context is given for orientation only. It is not a licence to widen your scope.\n"
    "\n"
    "OUTPUT CONTRACT - you MUST use these tags:\n"
    '<file path="relative/name.ext">\n[FULL FILE CONTENT]\n</file>\n'
    "  One per deliverable you own. Paths are relative to YOUR OWN directory.\n"
    "<log>\n[what you built, what you deliberately did NOT build, dependencies, open questions]\n</log>\n"
    "  Exactly one. This is read by the next wave of agents.\n"
    '<note to="tNN">\n[message to one specific teammate]\n</note>\n'
    "  Optional, zero or more. Use for hand-offs, conflicts, and corrections.\n"
    "\n"
    "TAG RULES:\n"
    "1. Every opening tag MUST have exactly one matching closing tag.\n"
    "2. Never use these tags inside prose, explanations, or code examples. Only as real output.\n"
    "3. Never nest tags inside other tags.\n"
    "4. Reference filenames in prose as plain text or in backticks, never as a tag.\n"
    "5. Output strictly in standard ASCII."
)

# The policy-development agent. Runs on APEX: writing the exploration policy is a
# planning task, not an agent assignment. It is shown replay scores, never the
# domain content - the policy governs search shape, not what the agents build.
_PROMPT_POLICY_DEV = (
    "You are a policy-development engineer improving the EXPLORATION POLICY of a "
    "multi-agent discovery system. You do not do the discovery work yourself and you "
    "never change what the agents are assigned.\n"
    "\n"
    "The policy is executable Python that decides where the agent team continues "
    "searching, which attempts run concurrently, and when exploration stops. It is "
    "scored by replaying it over recorded discovery trees: every outcome it asks for is "
    "already on disk, so evaluation costs no agent calls, and a branch that history never "
    "recorded is simply unavailable.\n"
    "\n"
    "What actually moves the score:\n"
    "1. Allocate depth where continuation has been paying and breadth where it has not.\n"
    "2. Cover the whole roster - unreached assignments are scored as zero.\n"
    "3. Group expansions into parallel batches; serial expansion wastes slots.\n"
    "4. Stop lines that stop improving rather than spending the budget evenly.\n"
    "5. Be adaptive, not uniformly greedier or broader. Conserving calls while progress "
    "is good and spending them when it plateaus is a real strategy.\n"
    "\n"
    "Output ONLY Python source for the policy module. No prose, no markdown fences."
)

_PROMPT_PHASE6_DISTILL = (
    "You are a ruthless, highly technical Lead Engineer and Project Manager. "
    "Read the provided raw documentation, agent reconciliation reports, and test telemetry, "
    "and extract a succinct, actionable list of explicit TO-DOs, architectural requirements, "
    "and implementation tasks. "
    "\n\nCRITICAL DIRECTIVES: "
    "\n1. TEST TELEMETRY: Hunt for any unit test execution logs, reports, or telemetry. "
    "Create a distinct 'Test Execution Status' section detailing passes and failures. "
    "Convert any failures into high-priority TO-DO items."
    "\n2. OWNERSHIP CONFLICTS: Any duplicate-deliverable or scope-violation finding in a "
    "reconciliation report becomes a high-priority TO-DO naming the agents involved."
    "\n3. EMBED ARTIFACTS: For EVERY task or failed test, extract and embed the relevant "
    "source artifact directly beneath the task description. If a task involves a specific function, "
    "include the code snippet. If it involves an error, include the traceback. "
    "Format these artifacts using proper markdown code fences and explicitly label the source filename."
    "\n4. STRICT ASCII ONLY: The generated markdown MUST consist entirely of standard ASCII characters. "
    "Do NOT use unicode symbols. Use standard hyphens (-) or asterisks (*) for bullet points. "
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

# ==============================================================================
# Global Utilities & State
# ==============================================================================

_active_clone_dirs: Set[Path] = set()
_clone_dirs_lock = threading.Lock()
_events_lock = threading.Lock()
_tree_lock = threading.Lock()
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


def estimate_tokens(text: str) -> int:
    return int(len(str(text)) / CHARS_PER_TOKEN)


def fit_context(text: str, budget: int,
                note: str = "...[CONTENT TRUNCATED FOR CONTEXT LIMITS]...") -> str:
    if text is None:
        return ""
    if len(text) <= budget:
        return text
    return text[:max(0, budget)] + f"\n\n{note}"


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
# Apex tier (planning / generation / distillation only)
# ------------------------------------------------------------------

def apex_client(base_url: Optional[str] = None, api_key: Optional[str] = None,
                timeout: float = WORKER_TIMEOUT_SECS, max_retries: int = 0) -> OpenAI:
    return OpenAI(
        base_url=base_url or GEN_API_BASE,
        api_key=api_key or GEN_API_KEY,
        timeout=timeout,
        max_retries=max_retries,
    )


def _is_stream_options_rejection(err_lower: str) -> bool:
    """Only a server that rejects stream_options/include_usage gets the retry
    without it. A generic HTTP 400 (context overflow, bad params) is re-raised
    instead of being retried into the same failure."""
    return ("stream_options" in err_lower or "include_usage" in err_lower
            or ("unrecognized" in err_lower and "stream" in err_lower))


def _apex_completion(client: OpenAI, system_prompt: str, user_prompt: str,
                     max_tokens: int, temperature: float,
                     model: Optional[str] = None,
                     presence_penalty: Optional[float] = None) -> Tuple[str, int, int]:
    """Streaming apex call. Returns (text, prompt_tokens, completion_tokens)."""
    kwargs = dict(
        model=model or LLM_MODEL,
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
        if _is_stream_options_rejection(low):
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
                    if expect_ctx and n_ctx and int(n_ctx) < per_slot:
                        note = ("UNDER-PROVISIONED: node window {} < budgeted per-slot {}"
                                .format(n_ctx, per_slot))
                    elif expect_ctx and n_ctx and int(n_ctx) not in (int(expect_ctx), per_slot):
                        note = ("larger than budget (-c {} / per-slot {}); headroom unused"
                                .format(expect_ctx, per_slot))
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


def build_worker_slot_queue(prefix: str = "W-Slot") -> Tuple[queue.Queue, int]:
    slot_queue: queue.Queue = queue.Queue()
    slot_idx = 1
    for ep in WORKER_ENDPOINTS:
        parsed = urllib.parse.urlparse(ep)
        host_tail = parsed.hostname or "local"
        if parsed.port:
            host_tail += f":{parsed.port}"
        for _ in range(WORKER_PARALLEL_SLOTS):
            slot_queue.put((ep, f"{prefix}{slot_idx:02d}-{host_tail}"))
            slot_idx += 1
    return slot_queue, max(0, slot_idx - 1)


def describe_budget_alignment() -> str:
    lines = []
    lines.append("[BUDGET] Server context alignment (stitcher tier removed)")
    lines.append(
        f"    apex    :8081  -c {APEX_SERVER_CTX} -np {APEX_SERVER_NP}"
        f"  -> {APEX_CONTEXT_TOKENS//1024}k tok/req"
        f" | input<={MAX_CONTEXT_CHARS:,} chars"
        f" (~{int(MAX_CONTEXT_CHARS/CHARS_PER_TOKEN)//1024}k tok)"
        f" | gen<={APEX_GEN_TOKENS//1024}k"
        f" / plan<={APEX_PLAN_TOKENS//1024}k"
        f" / distil<={APEX_DISTILL_TOKENS//1024}k"
        f"  [non-agent tasks only]"
    )
    lines.append(
        f"    agents  :{len(WORKER_ENDPOINTS)} node(s) x {WORKER_PARALLEL_SLOTS} slot(s)"
        f"  -c {WORKER_SERVER_CTX} -np {WORKER_SERVER_NP}"
        f"  -> {WORKER_CONTEXT_TOKENS//1024}k tok/req"
        f" | out<={MAX_WORKER_TOKENS//1024}k tok"
        f" | gap={int(WORKER_TIMEOUT_SECS)}s wall={int(WORKER_MAX_WALL_SECS)}s"
    )
    lines.append(
        f"            input<={WORKER_INPUT_CHARS:,} chars ="
        f" context {AGENT_CONTEXT_BUDGET:,}"
        f" + roster {AGENT_ROSTER_BUDGET:,}"
        f" + comms {AGENT_COMMS_BUDGET:,}"
        f" + objective {AGENT_OBJECTIVE_BUDGET:,}"
    )
    lines.append(
        f"            endpoints: {', '.join(WORKER_ENDPOINTS)}"
    )
    lines.append(
        f"    repo    :batch  wall={int(REPO_WORKER_WALL_SECS)}s/attempt "
        f"| outer={int(WORKER_RETRIES * REPO_WORKER_WALL_SECS + 30)}s"
    )
    lines.append(
        f"    tests   :agent pool out<={MAX_OUTPUT_TOKENS//1024}k tok"
        f" | min_tps={TEST_MIN_DECODE_TPS} | timeout={int(TEST_TIMEOUT_SECS)}s"
    )

    sub_budget_total = (AGENT_CONTEXT_BUDGET + AGENT_ROSTER_BUDGET
                        + AGENT_COMMS_BUDGET + AGENT_OBJECTIVE_BUDGET)
    if sub_budget_total > WORKER_INPUT_CHARS:
        lines.append(
            f"    [!] Agent sub-budgets total {sub_budget_total:,} chars but the input window is "
            f"{WORKER_INPUT_CHARS:,}. Prompts will be truncated at assembly."
        )

    calc_worker_tok = (WORKER_CONTEXT_TOKENS - WORKER_RESERVE_TOKENS
                       - int(WORKER_INPUT_CHARS / CHARS_PER_TOKEN))
    if calc_worker_tok < 1024:
        lines.append(
            f"    [!] WARNING: Calculated agent output budget collapsed to {calc_worker_tok} "
            f"and hit the 1024 floor. Most agent deliverables will truncate! "
            f"Lower WORKER_INPUT_CHARS or raise -c."
        )

    peak_worker = (int(WORKER_INPUT_CHARS / CHARS_PER_TOKEN)
                   + MAX_WORKER_TOKENS + WORKER_RESERVE_TOKENS)
    concurrent_worker = peak_worker * WORKER_PARALLEL_SLOTS
    if concurrent_worker > WORKER_SERVER_CTX:
        lines.append(
            f"    [!] Agent node over-subscribed: {WORKER_PARALLEL_SLOTS} slot(s) x "
            f"{peak_worker} tok = {concurrent_worker} > -c {WORKER_SERVER_CTX}. "
            "Lower WORKER_INPUT_CHARS, lower -np, or raise -c."
        )

    lines.append(
        "    [i] Consolidation is mechanical: agents write their own deliverables into "
        f"{WORK_DIRNAME}/ and their own logs into {COMMS_DIRNAME}/. No model merges output."
    )
    lines.append(
        f"    [i] Policies run isolated (cpu {POLICY_CPU_SECS}s, mem {POLICY_MEM_MB}MB, no fs/fds); "
        f"replay charges unrecorded branches: {'yes' if REPLAY_CHARGE_UNRECORDED else 'no'}; "
        f"ties keep incumbent{' unless cheaper' if DREAM_PREFER_CHEAPER else ''}"
    )
    lines.append(
        f"    [i] Tests: top-{TEST_NODES_PER_TASK} node(s)/task, run-scoped venv, wheels only"
        f"{', allowlist ' + str(len(TEST_PIP_ALLOWLIST)) + ' pkg(s)' if TEST_PIP_ALLOWLIST else ''}"
        f"{'' if TEST_PIP_INSTALL else ', installs OFF'}; untested prior {EVAL_UNTESTED_PRIOR}"
    )
    lines.append(
        f"    [i] RSI: budget {ROUND_BUDGET_PER_TASK} agent call(s)/task per round"
        f" (clamped {ROUND_BUDGET_MIN}-{ROUND_BUDGET_MAX})"
        f" | {DREAM_CANDIDATES} policy revision(s) per dream on apex"
        f" | replay costs 0 agent calls"
    )
    return "\n".join(lines)


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


def _render_wave_progress(wave: int, finished: int, total: int, start_time: float) -> None:
    bar_len = 30
    total_safe = total if total > 0 else 1
    filled = int((finished / total_safe) * bar_len)
    bar = '#' * filled + '-' * (bar_len - filled)
    percent = int((finished / total_safe) * 100)
    eta_str = _format_eta(start_time, finished, total)
    sys.stdout.write(f"\r    [+] Wave {wave:02d}: [{bar}] {percent}% "
                     f"(agents {finished}/{total}) | ETC: {eta_str}")
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
    gen_client = apex_client(timeout=WORKER_TIMEOUT_SECS)

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

_CGNAT_NET = ipaddress.ip_network("100.64.0.0/10")


def _is_private_host(host: str) -> bool:
    """
    Advisory SSRF check against local/metadata addresses. Resolves EVERY address
    (IPv4 and IPv6) and fails CLOSED: a host that cannot be resolved is treated
    as blocked. Still subject to DNS-rebinding TOCTOU; the clone itself is further
    constrained with redirects disabled and a protocol allowlist. Set
    GIT_ALLOW_PRIVATE_HOSTS=1 to clone from a LAN git server.
    """
    blocked_names = {"metadata.google.internal", "metadata.azure.internal", "localhost"}
    if host.lower() in blocked_names:
        return True
    try:
        orig_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(3.0)
        try:
            infos = socket.getaddrinfo(host, None)
        finally:
            socket.setdefaulttimeout(orig_timeout)
    except Exception:
        return True
    addrs = {info[4][0] for info in infos}
    if not addrs:
        return True
    for raw in addrs:
        try:
            addr = ipaddress.ip_address(raw.split("%", 1)[0])
        except ValueError:
            return True
        if getattr(addr, "ipv4_mapped", None):
            addr = addr.ipv4_mapped
        if (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved
                or addr.is_multicast or addr.is_unspecified
                or (addr.version == 4 and addr in _CGNAT_NET)):
            return True
    return False


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

    if not GIT_ALLOW_PRIVATE_HOSTS and _is_private_host(host):
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
    env["GIT_ALLOW_PROTOCOL"] = "https:http:ssh:git"
    # No redirects (a public URL cannot bounce the clone onto a private host), no
    # file:// or ext:: transports, no submodules.
    cmd = ["git", "-c", "http.followRedirects=false",
           "-c", "protocol.file.allow=never", "-c", "protocol.ext.allow=never",
           "clone", "--depth", str(GIT_CLONE_DEPTH), "--single-branch", "--no-recurse-submodules",
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
        ascii_document = enforce_ascii(document)
        if len(ascii_document) < len(document):
            print(f"    [!] Warning: Dropped {len(document) - len(ascii_document)} non-ASCII characters from repository source.", flush=True)
        document = ascii_document
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
# Phase 2: Fluff-to-Action Technical Distillation (apex)
# ==============================================================================

def distill_document(raw_text: str) -> str:
    client = apex_client(base_url=DISTILLER_URL, api_key=DISTILLER_API_KEY,
                         timeout=WORKER_TIMEOUT_SECS)
    char_count = len(raw_text)
    print(f"\n[PHASE 2] [*] Ingesting document ({char_count:,} characters)...", flush=True)

    if char_count > MAX_CONTEXT_CHARS:
        print(f"    [!] WARNING: Document size exceeds {MAX_CONTEXT_CHARS:,} characters. Truncating.", flush=True)
        raw_text = fit_context(raw_text, MAX_CONTEXT_CHARS, note="...[TRUNCATED]...")

    try:
        text, _, _ = _apex_completion(
            client, _PROMPT_PHASE2_DISTILL,
            f"Extract the actionable tasks from this document:\n\n{raw_text}",
            APEX_DISTILL_TOKENS, 0.3, model=DISTILLER_MODEL
        )
        return text
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
# Phase 3: Planning (apex) + Policy-Driven Discovery Tree (workers) + Dreaming
# ------------------------------------------------------------------------------
# Structure follows Dream-RSI (Zheng et al., 2026): recursive self-improvement is
# applied at the EXPLORATION layer, not to the agents themselves.
#
#   round t: deploy policy pi_t online -> agents expand a discovery TREE, every
#            node carrying its realized outcome -> append the tree to the pool ->
#            a policy-development agent (APEX) writes M revisions of the policy's
#            CODE -> each is scored by REPLAY over the whole pool at zero agent
#            calls -> the winner becomes pi_{t+1}.
#
# Two properties this buys, both load-bearing:
#   * Replay is exact, not approximate. The simulator IS the realized search
#     space; every outcome a candidate policy asks for is already on disk. Its
#     limit is equally sharp - a policy can only be dreamt where history went,
#     which is why this must be a loop rather than a one-off tuning pass.
#   * Monotonicity. The deployed policy pi^0 is itself in the candidate set, so
#     the winner can never score worse than the policy it replaces.
#
# The exploration policy is executable Python controlling WHERE the agent
# continues, WHAT runs in parallel, and WHEN to stop. The same policy source runs
# unchanged against LiveExplorer (real agent calls) and ReplayExplorer (recorded
# nodes, no calls) - that identity is what makes dreaming meaningful.
#
# Deliberately NOT implemented: abstracting prior trajectories into high-level
# directional "insights" injected into agent prompts. The paper's ablation found
# that consistently underperforms its unguided counterpart at equal budget -
# strong semantic priors over-constrain long-horizon parallel search. The comms
# digest stays raw structured log, and --semantic-guidance exists only to
# reproduce the negative result.
# ==============================================================================

def work_dir_for(run_dir: Path) -> Path:
    return run_dir / WORK_DIRNAME


def comms_dir_for(run_dir: Path) -> Path:
    return run_dir / COMMS_DIRNAME


def trees_dir_for(run_dir: Path) -> Path:
    return run_dir / TREES_DIRNAME


def policy_dir_for(run_dir: Path) -> Path:
    return run_dir / POLICY_DIRNAME


def dream_dir_for(run_dir: Path) -> Path:
    return run_dir / DREAM_DIRNAME


def round_dir_for(run_dir: Path, rnd: int) -> Path:
    return comms_dir_for(run_dir) / f"round{rnd:02d}"


def events_path_for(run_dir: Path) -> Path:
    return comms_dir_for(run_dir) / "events.jsonl"


def roster_path_for(run_dir: Path) -> Path:
    return comms_dir_for(run_dir) / "roster.json"


def tree_path_for(run_dir: Path, rnd: int) -> Path:
    return trees_dir_for(run_dir) / f"round{rnd:02d}.jsonl"


def round_done_marker(run_dir: Path, rnd: int) -> Path:
    return round_dir_for(run_dir, rnd) / ".round_complete"


def completed_rounds(run_dir: Path) -> List[int]:
    cdir = comms_dir_for(run_dir)
    if not cdir.exists():
        return []
    done = []
    for d in sorted(cdir.glob("round[0-9][0-9]")):
        if (d / ".round_complete").exists():
            try:
                done.append(int(d.name.replace("round", "")))
            except ValueError:
                continue
    return sorted(done)


def append_event(run_dir: Path, event: dict) -> None:
    """Append-only structured log. Shared by agents, reconciliation, dreaming and
    Phase 5, so later rounds can replay what happened without re-reading every
    artifact."""
    event = dict(event)
    event.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
    path = events_path_for(run_dir)
    try:
        with _events_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="ascii") as f:
                f.write(enforce_ascii(json.dumps(event, ensure_ascii=True)) + "\n")
    except Exception as exc:
        print(f"    [!] Failed to append event: {str(exc)[:120]}", flush=True)


def read_events(run_dir: Path) -> List[dict]:
    path = events_path_for(run_dir)
    out: List[dict] = []
    if not path.exists():
        return out
    content = read_file_content_safe(path) or ""
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ------------------------------------------------------------------
# Roster
# ------------------------------------------------------------------

def _slugify(text: str, max_words: int = 4) -> str:
    words = re.findall(r'[a-zA-Z0-9]+', text)[:max_words]
    slug = "-".join(w.lower() for w in words)
    return slug[:48] or "task"


def build_roster(tasks: List[str]) -> List[dict]:
    roster = []
    for idx, objective in enumerate(tasks, start=1):
        agent_id = f"t{idx:02d}"
        roster.append({
            "id": agent_id,
            "dir": f"{agent_id}_{_slugify(objective)}",
            "objective": objective.strip(),
        })
    return roster


def save_roster(run_dir: Path, roster: List[dict]) -> None:
    path = roster_path_for(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as f:
        f.write(enforce_ascii(json.dumps(roster, indent=2, ensure_ascii=True)) + "\n")


def load_roster(run_dir: Path) -> List[dict]:
    path = roster_path_for(run_dir)
    if not path.exists():
        return []
    content = read_file_content_safe(path)
    if not content:
        return []
    try:
        data = json.loads(content)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def render_roster(roster: List[dict], self_id: str, budget: int) -> str:
    """The roster is the mechanism that stops scope creep. Sibling objectives are
    rendered as explicit exclusions, not as background reading."""
    self_entry = next((r for r in roster if r["id"] == self_id), None)
    lines = ["TEAM ROSTER", ""]
    if self_entry:
        lines.append(f"YOU ARE {self_entry['id']}. Your output directory is {WORK_DIRNAME}/{self_entry['dir']}/")
        lines.append("")
    lines.append("ASSIGNMENTS OWNED BY OTHER AGENTS - DO NOT PRODUCE THESE:")
    others = [r for r in roster if r["id"] != self_id]
    if not others:
        lines.append("  (none)")
    per_entry = max(120, (budget - 400) // max(1, len(others))) if others else budget
    for r in others:
        obj = r["objective"].replace("\n", " ")
        if len(obj) > per_entry:
            obj = obj[:per_entry] + "..."
        lines.append(f"  [{r['id']}] owns {WORK_DIRNAME}/{r['dir']}/ :: {obj}")
    lines.append("")
    lines.append("If your objective seems to overlap one of the above, the overlap belongs to THEM. "
                 "State the dependency in your <log> and move on.")
    return fit_context("\n".join(lines), budget, note="...[ROSTER TRUNCATED]...")


def export_to_split_files(roster: List[dict], run_dir: Path) -> None:
    tasks_dir = run_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    for entry in roster:
        filepath = tasks_dir / f"{entry['id']}.md"
        with open(filepath, "w", encoding="ascii") as f:
            f.write(f"# {entry['id']}\n\n"
                    f"Output directory: {WORK_DIRNAME}/{entry['dir']}/\n\n"
                    f"{entry['objective'].strip()}\n")


# ------------------------------------------------------------------
# Discovery tree persistence
# ------------------------------------------------------------------

def new_node_id(rnd: int, seq: int) -> str:
    return f"r{rnd:02d}n{seq:04d}"


def append_node(run_dir: Path, rnd: int, node: dict) -> None:
    path = tree_path_for(run_dir, rnd)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _tree_lock:
        with open(path, "a", encoding="ascii") as f:
            f.write(enforce_ascii(json.dumps(node, ensure_ascii=True)) + "\n")


def load_tree(run_dir: Path, rnd: int) -> List[dict]:
    path = tree_path_for(run_dir, rnd)
    if not path.exists():
        return []
    nodes = []
    content = read_file_content_safe(path) or ""
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            nodes.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    # A node id may be rewritten by score writeback; last write wins.
    merged: Dict[str, dict] = {}
    for n in nodes:
        if "id" in n:
            merged[n["id"]] = n
    return [_normalise_node(n) for n in merged.values()]


def _normalise_node(n: dict) -> dict:
    """Bring nodes recorded by earlier revisions onto the current score scale.

    score_online is what a policy SAW when it made its decisions (heuristic +
    untested prior); score is the final, test-informed value that replay
    EVALUATES against. Legacy nodes carried a single raw-heuristic score."""
    if not n.get("task"):
        n.setdefault("score_online", n.get("score", 0.0))
        return n
    if "score_online" not in n:
        h = n.get("heuristic_score", n.get("score", 0.0))
        n["heuristic_score"] = h
        has_files = bool(n.get("files"))
        n["score_online"] = blend_test_score(h, None, has_files)
        n["score"] = blend_test_score(h, n.get("test_pass_rate"), has_files)
    n.setdefault("cost", 1)
    n.setdefault("files", [])
    n.setdefault("file_hashes", [])
    n.setdefault("gain", None)
    return n


def load_pool(run_dir: Path) -> List[Tuple[int, List[dict]]]:
    """Every recorded tree. This is the simulator pool the agent dreams in."""
    tdir = trees_dir_for(run_dir)
    if not tdir.exists():
        return []
    pool = []
    for p in sorted(tdir.glob("round[0-9][0-9].jsonl")):
        try:
            rnd = int(p.stem.replace("round", ""))
        except ValueError:
            continue
        nodes = load_tree(run_dir, rnd)
        if nodes:
            pool.append((rnd, nodes))
    return pool


def rewrite_tree(run_dir: Path, rnd: int, nodes: List[dict]) -> None:
    path = tree_path_for(run_dir, rnd)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _tree_lock:
        with open(path, "w", encoding="ascii") as f:
            for n in sorted(nodes, key=lambda x: x.get("seq", 0)):
                f.write(enforce_ascii(json.dumps(n, ensure_ascii=True)) + "\n")


def children_of(nodes: List[dict], node_id: str) -> List[dict]:
    return [n for n in nodes if n.get("parent") == node_id]


def ancestors_of(nodes: List[dict], node_id: str) -> List[dict]:
    index = {n["id"]: n for n in nodes}
    out, cur = [], index.get(node_id)
    while cur is not None and cur.get("parent"):
        cur = index.get(cur["parent"])
        if cur is None:
            break
        out.append(cur)
    return out


# ------------------------------------------------------------------
# Evaluator
# ------------------------------------------------------------------
# A fixed evaluator scoring each node in [0,1]. It must be cheap and
# deterministic: replay reads these scores rather than recomputing anything, so
# any nondeterminism here would make dreaming lie about the past.
#
# Every node carries two scores:
#   score_online - heuristic blended with EVAL_UNTESTED_PRIOR. This is what the
#                  policy saw when it made its decisions, and what replay SHOWS
#                  a policy, so replaying the incumbent over its own tree stays
#                  exact even after test telemetry lands.
#   score        - heuristic blended with the real pass rate once Phase 5 has
#                  run. This is what replay EVALUATES a policy against.
# Tested and untested nodes share one scale, so testing a node can no longer
# only ever push it below its untested siblings.
# ------------------------------------------------------------------

def _normalise_for_hash(body: str) -> str:
    return re.sub(r'\s+', ' ', body or "").strip()


def content_hash(body: str) -> str:
    """Whitespace-insensitive digest: reflowing a file is not new work."""
    return hashlib.sha256(_normalise_for_hash(body).encode("ascii", "ignore")).hexdigest()


def score_node_heuristic(status: str, files: List[str], violations: List[str],
                         truncated: bool, log_len: int, novel_frac: float) -> Tuple[float, dict]:
    parts = {}
    if status == "success":
        parts["status"] = EVAL_W_STATUS
    elif status == "partial":
        parts["status"] = EVAL_W_STATUS * EVAL_PARTIAL_STATUS_FRAC
    else:
        parts["status"] = 0.0
    parts["deliverables"] = EVAL_W_FILES * (1.0 if files else 0.0)
    parts["violations"] = -EVAL_W_VIOLATION * min(len(violations), 3)
    parts["truncated"] = -EVAL_W_TRUNCATED if truncated else 0.0
    if log_len >= 200:
        parts["log"] = EVAL_W_LOG
    elif log_len > 0:
        parts["log"] = EVAL_W_LOG * 0.5
    else:
        parts["log"] = 0.0
    parts["novelty"] = EVAL_W_NOVELTY * max(0.0, min(1.0, novel_frac))
    score = max(0.0, min(1.0, sum(parts.values())))
    return round(score, 6), {k: round(v, 6) for k, v in parts.items()}


def blend_test_score(heuristic: float, pass_rate: Optional[float],
                     has_files: bool = True) -> float:
    """A node with nothing on disk has nothing to test: its rate is 0, not the
    optimistic prior, so failed attempts do not earn score for free."""
    if pass_rate is not None:
        rate = float(pass_rate)
    else:
        rate = EVAL_UNTESTED_PRIOR if has_files else 0.0
    return round(EVAL_HEURISTIC_MIX * heuristic + (1.0 - EVAL_HEURISTIC_MIX) * rate, 6)


# ------------------------------------------------------------------
# Explorer interface - identical surface for live and replay
# ------------------------------------------------------------------
# The policy never touches an explorer object directly: it runs in a child
# process and reaches these methods through a line-JSON RPC (see run_policy).
# Only the names in _POLICY_API are dispatchable, and every return value is a
# plain JSON projection, so there is no Path, queue or roster to reach through.
# ------------------------------------------------------------------

_POLICY_API = {"tasks", "root", "budget_left", "spent", "nodes", "frontier", "best",
               "best_per_task", "note", "expand", "expand_parallel"}


class PolicyAborted(Exception):
    """Raised inside the PARENT when a policy exceeds its interface-call cap.
    The child re-raises it as a BaseException subclass, so a policy's
    `except Exception` cannot swallow it."""
    pass


def _node_view(n: dict) -> dict:
    """What a policy is allowed to see of a node: decision-time score only."""
    return {
        "id": n.get("id"), "parent": n.get("parent"), "task": n.get("task"),
        "depth": n.get("depth", 0), "score": n.get("score_online", n.get("score", 0.0)),
        "gain": n.get("gain"), "status": n.get("status"), "cost": n.get("cost", 1),
        "files": list(n.get("files", [])),
    }


class ExplorerBase:
    """What an exploration policy is allowed to do. Live and replay implement the
    same methods, so one policy source runs in both worlds unchanged."""

    def __init__(self, tasks: List[str], budget: int):
        self._tasks = list(tasks)
        self._budget = int(budget)
        self._spent = 0
        self._nodes: List[dict] = []
        self._steps = 0
        self._log: List[str] = []
        self._root_id = "root"

    # --- read-only views (decision-time scores) ---
    def tasks(self) -> List[str]:
        return list(self._tasks)

    def root(self) -> str:
        return self._root_id

    def budget_left(self) -> int:
        return max(0, self._budget - self._spent)

    def spent(self) -> int:
        return self._spent

    def _real(self) -> List[dict]:
        return [n for n in self._nodes if n.get("task")]

    def nodes(self) -> List[dict]:
        return [_node_view(n) for n in self._real()]

    def frontier(self) -> List[dict]:
        """Expanded nodes with no expanded children yet."""
        parents = {n.get("parent") for n in self._nodes}
        return [_node_view(n) for n in self._real() if n["id"] not in parents]

    def best(self) -> Optional[dict]:
        real = self._real()
        if not real:
            return None
        return _node_view(max(real, key=lambda n: _node_view(n)["score"]))

    def best_per_task(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for n in self._real():
            v = _node_view(n)
            cur = out.get(n["task"])
            if cur is None or v["score"] > cur["score"]:
                out[n["task"]] = v
        return out

    def note(self, msg) -> None:
        if len(self._log) < 200:
            self._log.append(str(msg)[:200])

    # --- evaluation views (final scores; never exposed to the policy) ---
    def final_best_per_task(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for n in self._real():
            cur = out.get(n["task"])
            if cur is None or n.get("score", 0.0) > cur.get("score", 0.0):
                out[n["task"]] = n
        return out

    # --- actions ---
    def _tick(self):
        self._steps += 1
        if self._steps > POLICY_MAX_STEPS:
            raise PolicyAborted(f"policy exceeded {POLICY_MAX_STEPS} interface calls")

    @staticmethod
    def _normalise_request(req) -> Optional[Tuple[str, str]]:
        try:
            return (str(req[0]), str(req[1]))
        except Exception:
            return None

    def _valid(self, req: Optional[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
        if req is None:
            return None
        parent_id, task = req
        if task not in self._tasks:
            return None
        if not any(n["id"] == parent_id for n in self._nodes):
            return None
        return req

    def expand(self, parent_id: str, task: str) -> Optional[dict]:
        res = self.expand_parallel([(parent_id, task)])
        return res[0] if res else None

    def expand_parallel(self, requests) -> List[Optional[dict]]:
        """Always returns a list aligned 1:1 with `requests`. Requests beyond
        POLICY_MAX_FANOUT are run in further batches (each batch is one interface
        call) instead of being silently dropped."""
        reqs = [self._valid(self._normalise_request(r)) for r in list(requests)]
        results: List[Optional[dict]] = [None] * len(reqs)
        for start in range(0, len(reqs), POLICY_MAX_FANOUT):
            self._tick()
            chunk = reqs[start:start + POLICY_MAX_FANOUT]
            for off, res in enumerate(self._expand_batch(chunk)):
                results[start + off] = res
            if self.budget_left() <= 0:
                break
        return results

    def _expand_batch(self, reqs: List[Optional[Tuple[str, str]]]) -> List[Optional[dict]]:
        raise NotImplementedError


class ReplayExplorer(ExplorerBase):
    """Dreaming. Resolves each requested expansion against a recorded node and
    charges its recorded cost; nothing is executed and no agent is called.

    A valid request for a branch history never recorded returns None - that is
    the edge of the dream - and, with REPLAY_CHARGE_UNRECORDED, costs one unit,
    because online it would have cost at least one real agent call."""

    def __init__(self, recorded: List[dict], tasks: List[str], budget: int):
        super().__init__(tasks, budget)
        self._by_parent: Dict[str, List[dict]] = {}
        for n in recorded:
            self._by_parent.setdefault(n.get("parent") or "", []).append(n)
        for v in self._by_parent.values():
            v.sort(key=lambda n: n.get("seq", 0))
        roots = [n for n in recorded if not n.get("parent")]
        self._root_id = roots[0]["id"] if roots else "root"
        self._consumed: Set[str] = set()
        self._unrecorded = 0
        if roots:
            self._nodes.append(dict(roots[0]))
        else:
            self._nodes.append({"id": "root", "parent": None, "task": None, "depth": 0,
                                "score": 0.0, "score_online": 0.0})

    def _expand_batch(self, reqs):
        out: List[Optional[dict]] = []
        for req in reqs:
            if req is None or self.budget_left() <= 0:
                out.append(None)
                continue
            parent_id, task = req
            match = None
            for cand in self._by_parent.get(parent_id, []):
                if cand.get("task") == task and cand["id"] not in self._consumed:
                    match = cand
                    break
            if match is None:
                self._unrecorded += 1
                if REPLAY_CHARGE_UNRECORDED:
                    self._spent += 1
                out.append(None)
                continue
            self._consumed.add(match["id"])
            self._spent += int(match.get("cost", 1))
            self._nodes.append(dict(match))
            out.append(_node_view(match))
        return out


class LiveExplorer(ExplorerBase):
    """Online deployment. Each expansion is a real agent call whose outcome is
    recorded into the tree, so this round's exploration becomes next round's
    simulator."""

    def __init__(self, tasks: List[str], budget: int, roster: List[dict], rnd: int,
                 background: str, run_dir: Path, slot_queue: queue.Queue,
                 prior_hashes: Optional[Dict[str, Set[str]]] = None,
                 semantic_guidance: bool = False):
        super().__init__(tasks, budget)
        self.roster = roster
        self.rnd = rnd
        self.background = background
        self.run_dir = run_dir
        self.slot_queue = slot_queue
        self.semantic_guidance = semantic_guidance
        self.prior_hashes = prior_hashes or {}
        self._seq = 0
        self._lock = threading.Lock()
        self._support_mode = False
        self._root_id = new_node_id(rnd, 0)
        root = {"id": self._root_id, "seq": 0, "round": rnd, "parent": None,
                "depth": 0, "task": None, "status": "root", "score": 0.0,
                "score_online": 0.0, "cost": 0, "files": [], "violations": []}
        self._nodes.append(root)
        append_node(run_dir, rnd, root)

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def _expand_batch(self, reqs):
        results: List[Optional[dict]] = [None] * len(reqs)
        runnable = [(i, v) for i, v in enumerate(reqs) if v is not None]
        with self._lock:
            allowance = self.budget_left()
        runnable = runnable[:allowance]
        if not runnable:
            return results
        pool = max(1, min(len(runnable), POLICY_MAX_FANOUT))
        with concurrent.futures.ThreadPoolExecutor(max_workers=pool) as ex:
            futs = {ex.submit(self._run_one, v[0], v[1]): i for i, v in runnable}
            for fut in concurrent.futures.as_completed(futs):
                i = futs[fut]
                try:
                    node = fut.result()
                    results[i] = _node_view(node) if node else None
                except Exception as exc:
                    print(f"\n    [!] expansion failed: {str(exc)[:120]}", flush=True)
                    results[i] = None
        return results

    def run_support_probes(self, n: int) -> int:
        """Spend up to n calls on off-policy expansions, chosen deterministically
        from the tree the policy grew:
          deep - refine each task's best node that has no same-task child yet
                 (pi_0-style policies refine the WEAKEST half and never do this)
          wide - a further independent root attempt, fewest-roots tasks first
                 (replay can only serve one root child per recorded attempt)
        The two lists are interleaved so both kinds of support accumulate."""
        if n <= 0:
            return 0
        with self._lock:
            real = [dict(x) for x in self._nodes if x.get("task")]
        has_child = {(x.get("parent"), x["task"]) for x in real}
        roots: Dict[str, int] = {}
        best: Dict[str, dict] = {}
        for x in real:
            if x.get("parent") == self._root_id:
                roots[x["task"]] = roots.get(x["task"], 0) + 1
            if not x.get("files"):
                continue
            cur = best.get(x["task"])
            if cur is None or x.get("score_online", 0.0) > cur.get("score_online", 0.0):
                best[x["task"]] = x
        deep = [(b["id"], t) for t, b in sorted(best.items(),
                                                key=lambda kv: (-kv[1].get("score_online", 0.0), kv[0]))
                if (b["id"], t) not in has_child]
        wide = [(self._root_id, t) for t in sorted(self._tasks, key=lambda t: (roots.get(t, 0), t))]
        reqs: List[Tuple[str, str]] = []
        di = wi = 0
        while len(reqs) < n and (di < len(deep) or wi < len(wide)):
            if di < len(deep):
                reqs.append(deep[di]); di += 1
            if len(reqs) < n and wi < len(wide):
                reqs.append(wide[wi]); wi += 1
        if not reqs:
            return 0
        self._support_mode = True
        self._steps = 0
        before = self.spent()
        try:
            self.expand_parallel(reqs)
        finally:
            self._support_mode = False
        return self.spent() - before

    def _reserve(self) -> bool:
        with self._lock:
            if self.budget_left() <= 0:
                return False
            self._spent += 1
            return True

    def _run_one(self, parent_id: str, task: str) -> Optional[dict]:
        # One unit per REAL agent call: the first attempt is reserved here, every
        # retry reserves again and is refused once the budget is gone.
        if not self._reserve():
            return None
        seq = self._next_seq()
        node_id = new_node_id(self.rnd, seq)
        with self._lock:
            index = {n["id"]: n for n in self._nodes}
        parent = index.get(parent_id)
        depth = (parent or {}).get("depth", 0) + 1
        agent = next((r for r in self.roster if r["id"] == task), None)
        if agent is None:
            return None
        parent_task_node = parent if (parent and parent.get("task") == task) else None
        if parent_task_node is not None:
            reference = set(parent_task_node.get("file_hashes", []))
        else:
            reference = set(self.prior_hashes.get(task, set()))

        node = None
        attempts = 0
        slot_name = None
        for attempt in range(1, WORKER_RETRIES + 1):
            if attempt > 1 and not self._reserve():
                break
            attempts += 1
            endpoint, slot_name = None, None
            while not _shutdown_event.is_set():
                try:
                    endpoint, slot_name = self.slot_queue.get(timeout=5.0)
                    break
                except queue.Empty:
                    continue
            if endpoint is None:
                break
            try:
                node = run_agent(agent, self.roster, self.rnd, node_id, seq, parent_id,
                                 depth, endpoint, slot_name, self.background,
                                 self.run_dir, self._nodes, self._lock,
                                 parent_task_node=parent_task_node, reference_hashes=reference,
                                 attempt=attempt, semantic_guidance=self.semantic_guidance)
                if node["status"] in ("success", "partial"):
                    break
            except Exception as exc:
                print(f"\n    [!] {task} attempt {attempt} raised {str(exc)[:80]}", flush=True)
            finally:
                self.slot_queue.put((endpoint, slot_name))
            if _shutdown_event.is_set():
                break
            time.sleep(2)

        if node is None:
            h, parts = score_node_heuristic("error", [], [], False, 0, 0.0)
            node = {"id": node_id, "seq": seq, "round": self.rnd, "parent": parent_id,
                    "depth": depth, "task": task, "dir": agent["dir"], "status": "error",
                    "heuristic_score": h, "score_parts": parts,
                    "score_online": blend_test_score(h, None, False),
                    "score": blend_test_score(h, None, False),
                    "test_pass_rate": None, "files": [], "file_hashes": [], "violations": [],
                    "notes": [], "elapsed": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "truncated": False, "slot": slot_name or "", "log_path": None}
        node["cost"] = max(1, attempts)
        node["attempts"] = attempts
        node["support"] = bool(self._support_mode)
        if parent_task_node is not None:
            node["gain"] = round(node["score_online"] - parent_task_node.get("score_online", 0.0), 6)
        else:
            node["gain"] = None

        with self._lock:
            self._nodes.append(node)
        append_node(self.run_dir, self.rnd, node)
        append_event(self.run_dir, {
            "round": self.rnd, "event": "node", "node": node_id, "parent": parent_id,
            "task": task, "depth": depth, "status": node["status"], "cost": node["cost"],
            "score": node["score_online"], "gain": node["gain"], "files": len(node["files"]),
        })
        with self._lock:
            real = [n for n in self._nodes if n.get("task")]
            spent = self._spent
        sys.stdout.write("\r    [+] round {:02d}: {} node(s), budget {}/{}, best {:.3f}   ".format(
            self.rnd, len(real), spent, self._budget,
            max([n["score_online"] for n in real] or [0.0])))
        sys.stdout.flush()
        return node


# ------------------------------------------------------------------
# Agent output parsing and ownership enforcement
# ------------------------------------------------------------------

_SCAN_RE = re.compile(
    r'```|`'
    r'|<(file)\s+path="([^"]*)"\s*>'
    r'|<(note)\s+to="([^"]*)"\s*>'
    r'|<(log)\s*>'
    r'|</(file|note|log)\s*>',
    re.IGNORECASE)


def scan_agent_output(text: str) -> dict:
    """Single-pass scanner shared by validation and parsing, so the two can never
    disagree. Outside a block, fenced and inline code is skipped (a tag in a code
    example is content). Inside a block, only that block's own closer ends it, so
    file content may contain fences freely. An unclosed <file> at the end is
    reported as `dangling` - the signature of an output cut off at its limit."""
    files: List[dict] = []
    notes: List[dict] = []
    logs: List[str] = []
    spans: List[Tuple[int, int]] = []
    counts = {k: 0 for k in ("file_open", "file_close", "log_open", "log_close",
                             "note_open", "note_close")}
    dangling = None
    pos, n = 0, len(text)
    while pos < n:
        m = _SCAN_RE.search(text, pos)
        if not m:
            break
        tok = m.group(0)
        if tok == "```":
            close = text.find("```", m.end())
            if close == -1:
                break
            pos = close + 3
            continue
        if tok == "`":
            close = text.find("`", m.end())
            nl = text.find("\n", m.end())
            pos = close + 1 if (close != -1 and (nl == -1 or close < nl)) else m.end()
            continue
        if m.group(6):
            counts[m.group(6).lower() + "_close"] += 1
            pos = m.end()
            continue
        kind = (m.group(1) or m.group(3) or m.group(5)).lower()
        counts[kind + "_open"] += 1
        closer = re.compile(r'</%s\s*>' % kind, re.IGNORECASE).search(text, m.end())
        if not closer:
            if kind == "file":
                dangling = {"path": m.group(2).strip(), "content": text[m.end():]}
            spans.append((m.start(), n))
            break
        counts[kind + "_close"] += 1
        body = text[m.end():closer.start()].strip()
        if kind == "file":
            files.append({"path": m.group(2).strip(), "content": body})
        elif kind == "note":
            notes.append({"to": m.group(4).strip().lower(), "body": body})
        else:
            logs.append(body)
        spans.append((m.start(), closer.end()))
        pos = closer.end()

    if logs:
        log = logs[0]
    else:
        residue, last = [], 0
        for a, b in spans:
            residue.append(text[last:a])
            last = b
        residue.append(text[last:])
        log = "".join(residue).strip()

    balanced = (dangling is None
                and counts["file_open"] == counts["file_close"]
                and counts["log_open"] == counts["log_close"]
                and counts["note_open"] == counts["note_close"])
    return {"files": files, "notes": notes, "log": log, "balanced": balanced,
            "counts": counts, "dangling": dangling}


def parse_agent_output(text: str) -> dict:
    """Split raw agent output into deliverables, log and notes."""
    s = scan_agent_output(text)
    return {"files": s["files"], "notes": s["notes"], "log": s["log"]}


def _agent_output_path(declared: str, agent_dir: Path, other_dirs: Set[str],
                       seen: Set[str], inherited: Optional[Set[str]] = None
                       ) -> Tuple[Path, Optional[str]]:
    """Resolve a declared path INSIDE the agent's own node directory.

    Escape is impossible by construction (the path is rebuilt relative to
    agent_dir). The violation we actually care about is an agent claiming a path
    that names a teammate's directory - that is the signal it drifted into a
    neighbour's assignment. Those land in claimed/ and are reported, not silently
    accepted into the shared tree. A path inherited from the parent attempt may
    be overwritten once: that is what a continuation is for."""
    violation = None
    normalised = declared.replace("\\", "/").strip()
    if normalised.startswith("/") or ".." in PurePosixPath(normalised).parts:
        violation = f"declared path '{declared}' attempted to escape its own directory"
    parts = [p for p in PurePosixPath(normalised).parts if p not in ("", ".", "..", "/")]
    parts = [re.sub(r'[^A-Za-z0-9_.\-]', '_', p) for p in parts]
    if not parts:
        parts = ["artifact.txt"]

    if parts[0] in other_dirs or re.match(r'^t\d{2}(_|$)', parts[0]):
        violation = f"declared path '{declared}' addresses another agent's directory"
        parts = ["claimed"] + parts
    elif parts[0] == WORK_DIRNAME:
        parts = parts[1:] or ["artifact.txt"]

    parts = parts[-3:]
    candidate = agent_dir.joinpath(*parts)
    key = str(candidate)
    if inherited is not None and key in inherited and key not in seen:
        inherited.discard(key)
        seen.add(key)
        return candidate, violation
    stem, suffix = candidate.stem, candidate.suffix
    counter = 1
    while str(candidate) in seen or candidate.exists():
        candidate = candidate.parent / f"{stem}_{counter}{suffix}"
        counter += 1
    seen.add(str(candidate))
    return candidate, violation


def _node_files(run_dir: Path, node: dict) -> Dict[str, str]:
    """Relative path (inside the node dir) -> content, for a recorded node."""
    out: Dict[str, str] = {}
    ndir = work_dir_for(run_dir) / node.get("dir", "") / node["id"]
    if not ndir.exists():
        return out
    for p in sorted(ndir.rglob("*")):
        if p.is_file() and not p.is_symlink():
            content = read_file_content_safe(p)
            if content is not None:
                out[str(p.relative_to(ndir)).replace("\\", "/")] = content
    return out


def render_parent_deliverables(files: Dict[str, str], budget: int) -> str:
    if not files:
        return "(the parent attempt produced no files)"
    blocks, used = [], 0
    for rel, content in files.items():
        max_ticks = max((len(m.group(0)) for m in re.finditer(r'`+', content)), default=2)
        fence = "`" * max(3, max_ticks + 1)
        block = f"--- {rel} ---\n{fence}\n{content.rstrip()}\n{fence}"
        remaining = budget - used
        if remaining <= 200:
            blocks.append(f"...[{len(files) - len(blocks)} more inherited file(s) omitted; "
                          "they are still on disk in your directory]...")
            break
        if len(block) > remaining:
            block = block[:remaining] + f"\n...[FILE TRUNCATED IN PROMPT; full copy is on disk]...\n{fence}"
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks)


def build_comms_digest(run_dir: Path, task: str, rnd: int, live_nodes: List[dict],
                       parent_id: Optional[str], budget: int,
                       semantic_guidance: bool = False) -> str:
    """Awareness channel. Priority order inside the budget: notes addressed to
    this agent, the lineage this node continues, the latest reconciliation
    report, then sibling summaries from this round.

    Raw structured log by design - see the semantic-guidance note at the top of
    this section."""
    sections: List[str] = []

    inbox: List[str] = []
    for r in range(rnd, 0, -1):
        npath = round_dir_for(run_dir, r) / "notes" / f"{task}.md"
        if npath.exists():
            body = (read_file_content_safe(npath) or "").strip()
            if body:
                inbox.append(f"--- notes to you from round {r:02d} ---\n{body}")
    if inbox:
        sections.append("MESSAGES ADDRESSED TO YOU\n\n" + "\n\n".join(inbox))

    if parent_id:
        lineage = ancestors_of(live_nodes, parent_id)
        index = {n["id"]: n for n in live_nodes}
        chain = ([index[parent_id]] if parent_id in index else []) + lineage
        blocks = []
        for n in chain[:3]:
            if not n.get("task"):
                continue
            lp = n.get("log_path")
            body = ""
            if lp:
                body = (read_file_content_safe(run_dir / lp) or "").strip()
            if body:
                blocks.append(f"--- attempt {n['id']} (score {n.get('score_online', n['score']):.3f}) ---\n"
                              f"{body[:COMMS_PEER_SUMMARY_CHARS * 2]}")
        if blocks:
            sections.append("THE ATTEMPT YOU ARE CONTINUING (LOGS)\n\n" + "\n\n".join(blocks)
                            + "\n\nImprove on it. Do not restart from nothing.")

    for r in range(rnd, 0, -1):
        rpath = round_dir_for(run_dir, r) / "RECONCILE.md"
        if rpath.exists():
            body = (read_file_content_safe(rpath) or "").strip()
            if body:
                sections.append(f"RECONCILIATION REPORT (round {r:02d})\n\n{body}")
                break

    peers = []
    best = {}
    for n in live_nodes:
        if not n.get("task") or n["task"] == task:
            continue
        cur = best.get(n["task"])
        if cur is None or n.get("score_online", 0) > cur.get("score_online", 0):
            best[n["task"]] = n
    for tid in sorted(best):
        n = best[tid]
        lp = n.get("log_path")
        body = (read_file_content_safe(run_dir / lp) or "").strip() if lp else ""
        if body:
            summary = body[:COMMS_PEER_SUMMARY_CHARS]
            if len(body) > COMMS_PEER_SUMMARY_CHARS:
                summary += "..."
            peers.append(f"[{tid}] {summary}")
    if peers:
        sections.append(f"WHAT TEAMMATES REPORTED THIS ROUND\n\n" + "\n\n".join(peers))

    if semantic_guidance:
        # Reproduces the paper's ablation. Off by default: explicit directional
        # guidance underperformed unguided replay at equal budget.
        sections.insert(0, "DIRECTIONAL GUIDANCE\n\nPrior rounds suggest prioritising "
                           "breadth early and depth once your deliverable exists. "
                           "Follow this guidance.")

    if not sections:
        return "(first attempt - no prior agent activity)"

    out, used = [], 0
    for sec in sections:
        remaining = budget - used
        if remaining <= 200:
            out.append("...[REMAINING COMMS OMITTED FOR CONTEXT LIMITS]...")
            break
        if len(sec) > remaining:
            sec = sec[:remaining] + "\n...[SECTION TRUNCATED]..."
        out.append(sec)
        used += len(sec) + 2
    return "\n\n".join(out)


def run_agent(agent: dict, roster: List[dict], rnd: int, node_id: str, seq: int,
              parent_id: str, depth: int, endpoint: str, slot_name: str,
              background: str, run_dir: Path, live_nodes: List[dict],
              lock: threading.Lock, parent_task_node: Optional[dict] = None,
              reference_hashes: Optional[Set[str]] = None, attempt: int = 1,
              semantic_guidance: bool = False) -> dict:
    """One agent, one assignment, one tree node. Writes its own deliverables and
    log, and returns the recorded outcome that becomes replayable history.

    A continuation starts from its parent's deliverables: they are copied into
    this node's directory and shown in the prompt, and the agent only emits the
    files it changes or adds."""
    task = agent["id"]
    reference_hashes = set(reference_hashes or set())
    client = OpenAI(base_url=endpoint, api_key=WORKER_API_KEY,
                    timeout=WORKER_TIMEOUT_SECS, max_retries=0)
    start_time = time.time()

    wroot = work_dir_for(run_dir)
    node_out = wroot / agent["dir"] / node_id
    # Retry safety: a previous attempt at this node id must not leak files into
    # this one.
    if node_out.exists():
        shutil.rmtree(node_out, ignore_errors=True)

    parent_files: Dict[str, str] = {}
    if parent_task_node is not None:
        parent_files = _node_files(run_dir, parent_task_node)

    roster_block = render_roster(roster, task, AGENT_ROSTER_BUDGET)
    with lock:
        snapshot = [dict(n) for n in live_nodes]
    comms_block = build_comms_digest(run_dir, task, rnd, snapshot, parent_id,
                                     AGENT_COMMS_BUDGET, semantic_guidance)
    objective_block = fit_context(agent["objective"], AGENT_OBJECTIVE_BUDGET)

    if parent_task_node is not None:
        parent_block = render_parent_deliverables(parent_files, AGENT_PARENT_BUDGET)
        context_budget = max(2000, AGENT_CONTEXT_BUDGET - len(parent_block))
        stage_note = (
            f"This is a CONTINUATION (depth {depth}) of attempt {parent_task_node['id']}. "
            "Its deliverables are shown below and are ALREADY IN YOUR DIRECTORY under the same "
            "relative paths. Emit ONLY the files you change or add: a <file> with an existing path "
            "replaces that file; files you do not emit are kept as they are. Improve and complete "
            "YOUR OWN deliverables only.\n\n"
            f"===== INHERITED DELIVERABLES =====\n{parent_block}"
        )
    else:
        context_budget = AGENT_CONTEXT_BUDGET
        stage_note = "This is a fresh attempt at your objective. Produce your deliverables from scratch."
    context_block = fit_context(background, context_budget)

    user_instruction = (
        f"{roster_block}\n\n"
        f"===== BROADER CONTEXT (ORIENTATION ONLY - NOT YOUR SCOPE) =====\n{context_block}\n\n"
        f"===== TEAM COMMUNICATION LOG =====\n{comms_block}\n\n"
        f"===== STAGE =====\n{stage_note}\n\n"
        f"===== YOUR OBJECTIVE ({task}) =====\n{objective_block}\n"
    )
    user_instruction = fit_context(user_instruction, WORKER_INPUT_CHARS)

    result_text = ""
    status = "success"
    prompt_tokens, comp_tokens = 0, 0
    is_estimated = True
    truncated = False
    finish_reason = None
    violations: List[str] = []
    saved_files: List[str] = []
    file_hashes: List[str] = []
    notes_out: List[dict] = []
    novel_frac = 0.0
    emitted = 0
    log_body = ""
    log_rel = None

    try:
        base_kwargs = dict(
            model=WORKER_MODEL,
            messages=[{"role": "system", "content": _PROMPT_PHASE3_AGENT},
                      {"role": "user", "content": user_instruction}],
            temperature=0.4,
            max_tokens=MAX_WORKER_TOKENS,
            frequency_penalty=1.1,
            presence_penalty=0.5,
            stream=True,
        )
        try:
            response = client.chat.completions.create(
                stream_options={"include_usage": True}, **base_kwargs)
        except Exception as e:
            if _is_stream_options_rejection(str(e).lower()):
                response = client.chat.completions.create(**base_kwargs)
            else:
                raise

        try:
            for chunk in response:
                now = time.time()
                if now - start_time > WORKER_MAX_WALL_SECS:
                    truncated = True
                    finish_reason = "wall_clock"
                    break
                if chunk.choices:
                    ch = chunk.choices[0]
                    if ch.delta is not None and ch.delta.content is not None:
                        result_text += ch.delta.content
                    if getattr(ch, "finish_reason", None):
                        finish_reason = ch.finish_reason
                if getattr(chunk, "usage", None) is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                    comp_tokens = chunk.usage.completion_tokens
                    is_estimated = False
        finally:
            try:
                response.close()
            except Exception:
                pass

        if finish_reason == "length":
            truncated = True

        result_text = enforce_ascii(result_text.strip())
        if is_estimated:
            prompt_tokens = estimate_tokens(_PROMPT_PHASE3_AGENT + user_instruction)
            comp_tokens = estimate_tokens(result_text)

        scan = scan_agent_output(result_text)
        counts = scan["counts"]
        log_body = scan["log"]
        emit_files = scan["files"]

        if not scan["balanced"]:
            tag_detail = "file {}/{}, log {}/{}, note {}/{}".format(
                counts["file_open"], counts["file_close"], counts["log_open"],
                counts["log_close"], counts["note_open"], counts["note_close"])
            if truncated and emit_files:
                dangling = scan["dangling"]["path"] if scan["dangling"] else None
                violations.append(
                    f"output cut off ({finish_reason}); {len(emit_files)} complete file(s) salvaged"
                    + (f", unfinished '{dangling}' discarded" if dangling else "")
                    + f" ({tag_detail})")
                status = "partial"
            else:
                violations.append(f"unbalanced output tags ({tag_detail}) - deliverables not written"
                                  + (f" [cut off: {finish_reason}]" if truncated else ""))
                status = "failed_validation"
                emit_files = []

        if len(result_text) < 20:
            status = "failed_validation"
            emit_files = []

        other_dirs = {r["dir"] for r in roster if r["id"] != task}
        accepted = status in ("success", "partial")

        # Inherit the parent's deliverables first (baseline), then apply the
        # agent's emitted files on top.
        inherited: Set[str] = set()
        if accepted and parent_files:
            for rel, content in parent_files.items():
                dest = node_out.joinpath(*PurePosixPath(rel).parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "w", encoding="ascii", errors="ignore") as fh:
                    fh.write(content)
                inherited.add(str(dest))

        novel = 0
        if accepted and emit_files:
            node_out.mkdir(parents=True, exist_ok=True)
            seen: Set[str] = set()
            for f in emit_files:
                path, violation = _agent_output_path(f["path"], node_out, other_dirs, seen, inherited)
                if violation:
                    violations.append(violation)
                path.parent.mkdir(parents=True, exist_ok=True)
                body = enforce_ascii(f["content"])
                with open(path, "w", encoding="ascii") as fh:
                    fh.write(body + "\n")
                if content_hash(body) not in reference_hashes:
                    novel += 1
            emitted = len(emit_files)
            novel_frac = novel / max(1, emitted)

        if node_out.exists():
            for p in sorted(node_out.rglob("*")):
                if p.is_file() and not p.is_symlink():
                    saved_files.append(str(p.relative_to(wroot)))
                    file_hashes.append(content_hash(read_file_content_safe(p) or ""))

        rdir = round_dir_for(run_dir, rnd)
        rdir.mkdir(parents=True, exist_ok=True)
        log_rel = f"{COMMS_DIRNAME}/round{rnd:02d}/{node_id}_{task}.md"
        with open(run_dir / log_rel, "w", encoding="ascii") as fh:
            fh.write(f"# {task} - node {node_id} (round {rnd:02d}, depth {depth}, attempt {attempt})\n\n")
            fh.write(f"- parent: {parent_id}\n")
            fh.write(f"- slot: {slot_name}\n")
            fh.write(f"- status: {status}\n")
            if parent_files:
                fh.write(f"- inherited: {len(parent_files)} file(s) from {parent_task_node['id']}\n")
            fh.write(f"- emitted: {emitted} file(s)\n")
            fh.write(f"- deliverables: {', '.join(saved_files) if saved_files else '(none)'}\n")
            if violations:
                fh.write(f"- violations: {'; '.join(violations)}\n")
            fh.write("\n## Log\n\n")
            fh.write(enforce_ascii(log_body or "(agent produced no log)").strip() + "\n")

        # Notes are only delivered from accepted output, so a rejected attempt
        # (which will be retried) cannot put duplicate or stale messages in a
        # teammate's inbox.
        if accepted:
            valid_ids = {r["id"] for r in roster}
            ndir = rdir / "notes"
            for note in scan["notes"]:
                target = note["to"]
                if target not in valid_ids:
                    violations.append(f"note addressed to unknown agent '{target}'")
                    continue
                ndir.mkdir(parents=True, exist_ok=True)
                with open(ndir / f"{target}.md", "a", encoding="ascii") as fh:
                    fh.write(f"\n### from {task} (node {node_id})\n\n"
                             + enforce_ascii(note["body"]).strip() + "\n")
                notes_out.append({"to": target, "chars": len(note["body"])})

        for v in violations:
            append_event(run_dir, {"round": rnd, "node": node_id, "agent": task,
                                   "attempt": attempt, "event": "violation", "detail": v})

    except Exception as e:
        status = "error"
        violations.append(f"agent error: {str(e)[:160]}")
        log_rel = None
        is_estimated = True

    elapsed = round(time.time() - start_time, 2)
    h, parts = score_node_heuristic(status, saved_files, violations, truncated,
                                    len(log_body or ""), novel_frac)
    online = blend_test_score(h, None, bool(saved_files))
    return {
        "id": node_id, "seq": seq, "round": rnd, "parent": parent_id, "depth": depth,
        "task": task, "dir": agent["dir"], "status": status,
        "score": online, "score_online": online, "heuristic_score": h, "score_parts": parts,
        "test_pass_rate": None, "cost": 1,
        "files": saved_files, "file_hashes": sorted(set(file_hashes)),
        "emitted": emitted, "inherited": len(parent_files),
        "violations": violations, "notes": notes_out,
        "log_path": log_rel if status != "error" else None,
        "elapsed": elapsed, "prompt_tokens": prompt_tokens,
        "completion_tokens": comp_tokens, "tps": round(comp_tokens / elapsed, 2) if elapsed > 0 else 0,
        "slot": slot_name, "is_estimated": is_estimated, "truncated": truncated,
        "finish_reason": finish_reason, "raw_chars": len(result_text),
    }


# ------------------------------------------------------------------
# Exploration policy: executable code, isolated
# ------------------------------------------------------------------

DEFAULT_POLICY_SOURCE = '''\
# Exploration policy pi_0: hand-written parallel-refine.
# Both Dream-RSI and the fixed-exploration control start from this, so round 1 is
# identical by construction and later divergence is attributable to dreaming.

def explore(ctx):
    root = ctx.root()
    opened = ctx.expand_parallel([(root, t) for t in ctx.tasks()])
    frontier = [n for n in opened if n]

    while ctx.budget_left() > 0 and frontier:
        frontier.sort(key=lambda n: n["score"])
        half = max(1, len(frontier) // 2)
        weakest = frontier[:half]
        nxt = ctx.expand_parallel([(n["id"], n["task"]) for n in weakest])
        nxt = [n for n in nxt if n]
        if not nxt:
            break
        frontier = nxt
'''

_POLICY_BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "exit", "quit", "breakpoint",
    "memoryview", "object", "super", "type", "BaseException", "print", "help",
    "dir", "id", "hasattr", "classmethod", "staticmethod", "property",
}

# Attribute names with no leading underscore that still reach interpreter
# internals: frames (and through them module globals), code objects, tracebacks.
_POLICY_BLOCKED_ATTRS = {
    "gi_frame", "gi_code", "gi_yieldfrom", "gi_running", "gi_suspended",
    "cr_frame", "cr_code", "cr_await", "cr_running", "cr_origin",
    "ag_frame", "ag_code", "ag_await", "ag_running",
    "f_back", "f_globals", "f_locals", "f_builtins", "f_code", "f_trace", "f_lineno",
    "tb_frame", "tb_next", "tb_lasti", "tb_lineno", "with_traceback",
    "co_code", "co_consts", "co_names", "func_globals", "func_code", "mro",
}

_POLICY_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "int",
    "len", "list", "map", "max", "min", "pow", "range", "reversed", "round", "set",
    "sorted", "str", "sum", "tuple", "zip", "True", "False", "None", "Exception",
    "ValueError", "KeyError", "IndexError", "TypeError", "ZeroDivisionError",
    "ArithmeticError", "StopIteration", "isinstance", "frozenset",
]


def validate_policy_source(source: str) -> Tuple[bool, str]:
    """Static check before any execution. Defence in depth only: the policy also
    runs in a resource-limited child process that can reach the explorer solely
    through the RPC surface in _POLICY_API."""
    if len(source) > POLICY_MAX_CHARS:
        return False, f"policy source exceeds {POLICY_MAX_CHARS} chars"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"

    has_explore = False
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            if stmt.name == "explore":
                has_explore = True
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue  # docstring / bare constant
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            continue  # module-level constants
        return False, f"top-level {type(stmt).__name__} is not permitted; only defs and constants"
    if not has_explore:
        return False, "no top-level explore(ctx) function"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "imports are not permitted in an exploration policy"
        if isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith)):
            return False, "async constructs are not permitted"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                return False, f"dunder/private attribute access '{node.attr}'"
            if node.attr in _POLICY_BLOCKED_ATTRS:
                return False, f"interpreter-internal attribute '{node.attr}'"
        if isinstance(node, ast.Name) and node.id in _POLICY_BLOCKED_NAMES:
            return False, f"blocked name '{node.id}'"
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return False, f"dunder name '{node.id}'"
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return False, "global/nonlocal are not permitted"
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            return False, "bare 'except:' is not permitted (it would swallow the step cap)"
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            return False, "string constants containing '__' are not permitted"
    return True, "ok"


# The child process. It sets its own resource limits before it ever sees policy
# source, exposes a ctx proxy whose only reach into the parent is the RPC, and
# re-raises the parent's step-cap abort as a BaseException subclass.
_POLICY_CHILD_SRC = r'''
import sys, json, resource, builtins

def _lim(res, soft, hard=None):
    try:
        resource.setrlimit(res, (soft, soft if hard is None else hard))
    except Exception:
        pass

_cpu, _mem = int(sys.argv[1]), int(sys.argv[2])
_lim(resource.RLIMIT_CPU, _cpu, _cpu + 1)
_lim(resource.RLIMIT_AS, _mem * 1024 * 1024)
_lim(resource.RLIMIT_FSIZE, 0)
_lim(resource.RLIMIT_CORE, 0)
if hasattr(resource, "RLIMIT_NPROC"):
    _lim(resource.RLIMIT_NPROC, 0)

_rd, _wr, _fl = sys.stdin.readline, sys.stdout.write, sys.stdout.flush
_loads, _dumps = json.loads, json.dumps
_boot = _loads(_rd())
_src, _names = _boot["source"], _boot["builtins"]
_safe = {n: getattr(builtins, n) for n in _names if hasattr(builtins, n)}
# No new file descriptors from here on: no files, sockets or pipes.
_lim(resource.RLIMIT_NOFILE, 3)


class PolicyAborted(BaseException):
    pass


def _call(m, *a):
    _wr(_dumps({"m": m, "a": list(a)}) + "\n")
    _fl()
    line = _rd()
    if not line:
        raise PolicyAborted("parent closed the channel")
    r = _loads(line)
    if "abort" in r:
        raise PolicyAborted(r["abort"])
    if "fail" in r:
        raise TypeError(r["fail"])
    return r.get("r")


def _norm(req):
    try:
        return [str(req[0]), str(req[1])]
    except BaseException:
        return None


class Ctx:
    __slots__ = ()
    def tasks(self): return _call("tasks")
    def root(self): return _call("root")
    def budget_left(self): return _call("budget_left")
    def spent(self): return _call("spent")
    def nodes(self): return _call("nodes")
    def frontier(self): return _call("frontier")
    def best(self): return _call("best")
    def best_per_task(self): return _call("best_per_task")
    def note(self, msg): return _call("note", str(msg)[:200])
    def expand(self, parent_id, task): return _call("expand", str(parent_id), str(task))
    def expand_parallel(self, requests):
        return _call("expand_parallel", [_norm(r) for r in list(requests)])


_ns = {"__builtins__": _safe}
try:
    exec(compile(_src, "<policy>", "exec"), _ns)
    _fn = _ns.get("explore")
    if not callable(_fn):
        raise ValueError("explore is not callable")
    _fn(Ctx())
    _out = {"done": True, "ok": True, "detail": "ok"}
except PolicyAborted as _e:
    _out = {"done": True, "ok": True, "detail": "aborted: %s" % _e}
except BaseException as _e:
    _out = {"done": True, "ok": False, "detail": "%s: %s" % (type(_e).__name__, str(_e)[:160])}
_wr(_dumps(_out) + "\n")
_fl()
'''


def run_policy(source: str, explorer: ExplorerBase,
               wall_secs: Optional[float] = None) -> Tuple[bool, str]:
    """Execute a policy in an isolated child process against `explorer`.

    The child gets CPU/memory/file-size/fd/process limits and a clean env, and
    can only call the methods in _POLICY_API. Pure-compute runaways die on
    RLIMIT_CPU (waiting on the parent's agent calls costs the child no CPU, so
    live rounds are unaffected); replay additionally gets a wall-clock cap."""
    ok, reason = validate_policy_source(source)
    if not ok:
        return False, reason

    sandbox = Path(tempfile.mkdtemp(prefix="autoresearch_policy_"))
    err_path = sandbox / "stderr.txt"
    killed = {"why": None}
    done = threading.Event()
    result: Optional[Tuple[bool, str]] = None
    try:
        with open(err_path, "w") as err_fh:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", _POLICY_CHILD_SRC,
                 str(POLICY_CPU_SECS), str(POLICY_MEM_MB)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err_fh,
                text=True, encoding="ascii", errors="replace", bufsize=1,
                cwd=str(sandbox), env={"PATH": "/usr/bin:/bin", "LANG": "C"},
                start_new_session=True)

        def _kill(why: str):
            if killed["why"] is None:
                killed["why"] = why
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        def _watchdog():
            t0 = time.time()
            while not done.wait(0.5):
                if _shutdown_event.is_set():
                    _kill("shutdown requested")
                    return
                if wall_secs and time.time() - t0 > wall_secs:
                    _kill(f"wall clock exceeded ({wall_secs:.0f}s)")
                    return

        threading.Thread(target=_watchdog, daemon=True).start()

        try:
            proc.stdin.write(json.dumps({"source": source,
                                         "builtins": _POLICY_SAFE_BUILTIN_NAMES}) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

        aborted_msg = None
        post_abort = 0
        rpc_calls = 0
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                _kill("malformed RPC from policy")
                break
            if msg.get("done"):
                result = (bool(msg.get("ok")), str(msg.get("detail", ""))[:200])
                break
            m, args = msg.get("m"), msg.get("a") or []
            rpc_calls += 1
            if aborted_msg is None and rpc_calls > POLICY_MAX_RPC:
                aborted_msg = f"policy exceeded {POLICY_MAX_RPC} interface calls (reads included)"
            if aborted_msg is not None:
                post_abort += 1
                if post_abort > 50:
                    _kill("kept calling after abort")
                    break
                resp = {"abort": aborted_msg}
            elif m not in _POLICY_API:
                resp = {"fail": f"unknown interface method '{m}'"}
            else:
                try:
                    resp = {"r": getattr(explorer, m)(*args)}
                except PolicyAborted as exc:
                    aborted_msg = str(exc)
                    resp = {"abort": aborted_msg}
                except TypeError as exc:
                    resp = {"fail": f"{m}: {exc}"}
                except Exception as exc:
                    resp = {"fail": f"{m}: {type(exc).__name__}: {str(exc)[:120]}"}
            try:
                proc.stdin.write(json.dumps(resp) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                break

        done.set()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _kill("did not exit")
            proc.wait(timeout=5)

        if result is not None:
            return result
        if killed["why"]:
            return False, f"policy killed: {killed['why']}"
        rc = proc.returncode
        if rc is not None and rc < 0:
            sig = -rc
            if sig == getattr(signal, "SIGXCPU", 24) or sig == signal.SIGKILL:
                return False, f"policy killed: CPU limit ({POLICY_CPU_SECS}s)"
            if sig == getattr(signal, "SIGXFSZ", 25):
                return False, "policy killed: attempted file write"
            return False, f"policy killed by signal {sig}"
        tail = ""
        try:
            tail = (err_path.read_text(errors="replace").strip().splitlines() or [""])[-1]
        except Exception:
            pass
        return False, f"policy process exited ({rc}) {tail[:160]}".strip()
    finally:
        done.set()
        shutil.rmtree(sandbox, ignore_errors=True)


def policy_path(run_dir: Path, rnd: int) -> Path:
    return policy_dir_for(run_dir) / f"pi_r{rnd:02d}.py"


def load_or_init_policy(run_dir: Path, rnd: int) -> str:
    path = policy_path(run_dir, rnd)
    if path.exists():
        src = read_file_content_safe(path)
        if src and src.strip():
            return src
    if rnd > 1:
        print(f"    [!] WARNING: {path.name} not found; deploying pi_0 for round {rnd:02d}. "
              f"The dreamed policy lineage is broken here.", flush=True)
        append_event(run_dir, {"round": rnd, "event": "policy_missing", "fallback": "pi_0"})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as f:
        f.write(DEFAULT_POLICY_SOURCE)
    return DEFAULT_POLICY_SOURCE


# ------------------------------------------------------------------
# Replay scoring and dreaming
# ------------------------------------------------------------------

def pool_untested_prior(pool: List[Tuple[int, List[dict]]]) -> Tuple[float, int]:
    """Prior pass rate for untested nodes during replay evaluation. Tested nodes
    are the top-k by online score, so this mean is if anything optimistic for
    the untested remainder - but it tracks reality, unlike a constant."""
    rates = [n["test_pass_rate"] for _, nodes in pool for n in nodes
             if n.get("task") and n.get("test_pass_rate") is not None]
    if EVAL_UNTESTED_PRIOR_MODE != "empirical" or len(rates) < EVAL_EMPIRICAL_MIN_TESTED:
        return EVAL_UNTESTED_PRIOR, len(rates)
    return sum(rates) / len(rates), len(rates)


def _replay_eval_score(n: dict, prior: float) -> float:
    """Final score of a node for replay EVALUATION. Tested nodes keep their
    test-informed score; untested nodes with files are re-blended against the
    pool prior; nodes with no files keep their (zero-rate) score."""
    if n.get("test_pass_rate") is not None or not n.get("files"):
        return float(n.get("score", 0.0))
    h = n.get("heuristic_score")
    if h is None:
        return float(n.get("score", 0.0))
    return EVAL_HEURISTIC_MIX * float(h) + (1.0 - EVAL_HEURISTIC_MIX) * prior


def replay_score(source: str, pool: List[Tuple[int, List[dict]]], tasks: List[str],
                 budget: int, prior: Optional[float] = None) -> dict:
    """Score a candidate policy by dreaming it over every recorded tree. Zero
    agent calls: each expansion resolves to a node whose outcome is already on
    disk. The policy decides on score_online (what it would have seen live); the
    result is evaluated on the final, test-informed score."""
    if prior is None:
        prior, _ = pool_untested_prior(pool)
    per_tree = []
    for rnd, nodes in pool:
        ex = ReplayExplorer(nodes, tasks, budget)
        ok, detail = run_policy(source, ex, wall_secs=POLICY_REPLAY_WALL_SECS)
        if not ok:
            return {"valid": False, "detail": detail, "score": -1.0,
                    "cost": 0, "per_tree": []}
        covered: Dict[str, float] = {}
        for n in ex._real():
            v = _replay_eval_score(n, prior)
            if v > covered.get(n["task"], -1.0):
                covered[n["task"]] = v
        coverage = len(covered) / max(1, len(tasks))
        per_tree.append({
            "round": rnd, "best": round(max(covered.values(), default=0.0), 6),
            "cost": ex.spent(), "unrecorded": ex._unrecorded,
            "coverage": round(coverage, 4),
            "mean_best": round(sum(covered.values()) / len(covered), 6) if covered else 0.0,
        })

    if not per_tree:
        return {"valid": False, "detail": "empty simulator pool", "score": -1.0,
                "cost": 0, "per_tree": []}

    mean_best = sum(t["mean_best"] for t in per_tree) / len(per_tree)
    mean_cov = sum(t["coverage"] for t in per_tree) / len(per_tree)
    mean_cost = sum(t["cost"] for t in per_tree) / len(per_tree)
    quality = mean_best * (DREAM_COVERAGE_FLOOR + (1 - DREAM_COVERAGE_FLOOR) * mean_cov)
    cost_pen = DREAM_COST_WEIGHT * mean_cost / max(1, budget)
    score = quality - cost_pen
    return {
        "valid": True, "detail": "ok", "score": round(score, 6),
        "quality": round(quality, 6), "cost_penalty": round(cost_pen, 6),
        "prior": round(prior, 4),
        "mean_best": round(mean_best, 6), "coverage": round(mean_cov, 4),
        "cost": round(mean_cost, 2),
        "unrecorded": sum(t["unrecorded"] for t in per_tree),
        "efficiency": round(score / mean_cost, 6) if mean_cost else 0.0,
        "per_tree": per_tree,
    }


def _policy_interface_doc() -> str:
    return (
        "POLICY INTERFACE (this is the whole API; nothing else is available)\n"
        "  ctx.tasks() -> list of task ids, e.g. ['t01','t02']\n"
        "  ctx.root() -> id of the tree root\n"
        "  ctx.expand(parent_id, task) -> node dict or None\n"
        "  ctx.expand_parallel([(parent_id, task), ...]) -> list of node-or-None, aligned 1:1\n"
        f"      with the requests. Runs concurrently in batches of {POLICY_MAX_FANOUT}; each batch\n"
        "      counts as one interface call.\n"
        "  ctx.frontier() -> expanded nodes with no expanded children\n"
        "  ctx.nodes() -> every node expanded so far this run\n"
        "  ctx.best() -> highest-scoring node so far, or None\n"
        "  ctx.best_per_task() -> dict task -> best node\n"
        "  ctx.budget_left() / ctx.spent() -> ints, budget is in agent calls\n"
        "  ctx.note(msg) -> record a short diagnostic string\n"
        "\n"
        "A node dict has: id, parent, task, depth, score (0..1), gain, status, cost, files.\n"
        "  gain = score minus the parent attempt's score for a continuation (None at depth 1).\n"
        "  A continuation starts from its parent's files and only changes what it improves.\n"
        "  cost can exceed 1: failed attempts are retried and every real call is charged.\n"
        "expand returns None when the budget is exhausted, the request is malformed, or\n"
        "- during replay - history never recorded that branch. An unrecorded branch still\n"
        "costs 1 budget unit, exactly as a real call would online. None is information.\n"
        "\n"
        "HARD RULES\n"
        "  1. Define exactly one top-level function: explore(ctx). Helper defs and constants are ok.\n"
        "  2. No imports, no print, no file/network/system access. Pure control flow.\n"
        "  3. No attribute or name starting with '_', no bare 'except:'.\n"
        f"  4. At most {POLICY_MAX_STEPS} interface calls total; exceeding it ends the run.\n"
        "  5. Terminate. Every loop must make progress toward budget exhaustion or break.\n"
        "  6. Output ONLY Python source, no markdown fences, no commentary."
    )


def _pool_summary(pool: List[Tuple[int, List[dict]]], tasks: List[str]) -> str:
    lines = []
    for rnd, nodes in pool:
        real = [n for n in nodes if n.get("task")]
        if not real:
            continue
        depths: Dict[int, int] = {}
        gains: Dict[int, List[float]] = {}
        for n in real:
            d = n.get("depth", 0)
            depths[d] = depths.get(d, 0) + 1
            if isinstance(n.get("gain"), (int, float)):
                gains.setdefault(d, []).append(n["gain"])
        by_task: Dict[str, List[float]] = {}
        for n in real:
            by_task.setdefault(n["task"], []).append(n.get("score_online", n.get("score", 0.0)))
        best = []
        for t in sorted(by_task):
            s = sorted(by_task[t])
            best.append(f"{t}:{s[-1]:.2f}(n={len(s)})")
        tested = [n for n in real if n.get("test_pass_rate") is not None]
        gain_s = ", ".join(f"d{d}:{sum(g)/len(g):+.3f}(n={len(g)})" for d, g in sorted(gains.items()))
        lines.append(
            f"round {rnd:02d}: {len(real)} nodes, depth histogram "
            + ", ".join(f"d{d}={c}" for d, c in sorted(depths.items()))
            + (f" | mean gain by depth {gain_s}" if gain_s else "")
            + (f" | tested {len(tested)} node(s), mean pass "
               f"{sum(n['test_pass_rate'] for n in tested)/len(tested):.2f}" if tested else "")
            + " | best per task " + " ".join(best)
        )
    return "\n".join(lines) or "(pool is empty)"


def _select_winner(results: List[dict]) -> dict:
    """Incumbent-stable selection. A challenger must beat the incumbent by more
    than DREAM_TIE_EPS; ties keep the incumbent (or, with DREAM_PREFER_CHEAPER,
    the cheapest tied policy)."""
    valid = [r for r in results if r["valid"]]
    incumbent = results[0] if results and results[0]["valid"] else None
    if not valid:
        return results[0]
    best = max(valid, key=lambda r: r["score"])
    if incumbent is None:
        return best
    if best["score"] > incumbent["score"] + DREAM_TIE_EPS:
        return best
    if DREAM_PREFER_CHEAPER:
        tied = [r for r in valid if r["score"] >= incumbent["score"] - DREAM_TIE_EPS]
        return min(tied, key=lambda r: (r.get("cost", 0), -r["score"], r["index"]))
    return incumbent


def dream_policy_improvement(run_dir: Path, rnd: int, current_source: str,
                             pool: List[Tuple[int, List[dict]]],
                             tasks: List[str], budget: int) -> Tuple[str, dict]:
    """Offline policy improvement. The policy-development agent runs on APEX -
    it is a planning task, not an agent assignment and not a merge.

    The deployed policy is entered as candidate 0 and wins ties, so the winner
    is never worse than what it replaces on the recorded pool."""
    # Candidates are replayed under the cap the policy actually gets online.
    budget = policy_budget(budget, len(tasks))
    prior, n_tested = pool_untested_prior(pool)
    print(f"\n[DREAM] Round {rnd:02d}: replaying candidate policies over "
          f"{len(pool)} recorded tree(s) at zero agent calls "
          f"(policy budget {budget}, untested prior {prior:.3f} from {n_tested} tested node(s), "
          f"cost weight {DREAM_COST_WEIGHT})...", flush=True)

    def _board(name: str, r: dict) -> str:
        if not r["valid"]:
            return f"{name}: REJECTED - {r['detail'][:80]}"
        trees = "; ".join(f"r{t['round']:02d} cost {t['cost']} off-map {t['unrecorded']} "
                          f"cov {t['coverage']:.2f}" for t in r.get("per_tree", []))
        return (f"{name}: score {r['score']:.4f} (quality {r.get('quality', 0):.4f} "
                f"- cost {r.get('cost_penalty', 0):.4f}), mean cost {r.get('cost', 0)}, "
                f"off-map requests {r.get('unrecorded', 0)} [{trees}]")

    ddir = dream_dir_for(run_dir) / f"round{rnd:02d}"
    ddir.mkdir(parents=True, exist_ok=True)

    candidates = [{"name": "pi_0 (deployed)", "source": current_source}]
    base = replay_score(current_source, pool, tasks, budget, prior)
    results = [dict(base, name="pi_0 (deployed)", index=0)]
    if base["valid"]:
        print(f"    [+] pi_0 (deployed): score {base['score']:.4f} "
              f"| mean_best {base.get('mean_best', 0):.4f} "
              f"| coverage {base.get('coverage', 0):.2f} | cost {base.get('cost', 0)}", flush=True)
        leaderboard = _board("pi_0 (deployed)", base)
    else:
        print(f"    [!] Deployed policy does not replay ({base['detail'][:80]}); "
              f"any valid revision will replace it.", flush=True)
        leaderboard = f"pi_0 (deployed): INVALID IN REPLAY - {base['detail'][:80]}"
    client = apex_client(timeout=WORKER_TIMEOUT_SECS)

    for m in range(1, DREAM_CANDIDATES + 1):
        if _shutdown_event.is_set():
            break
        user = (
            f"{_policy_interface_doc()}\n\n"
            f"===== CURRENT POLICY SOURCE =====\n{fit_context(current_source, 6000)}\n\n"
            f"===== RECORDED DISCOVERY HISTORY =====\n{fit_context(_pool_summary(pool, tasks), 6000)}\n\n"
            f"===== REPLAY LEADERBOARD SO FAR =====\n{leaderboard}\n\n"
            f"===== BUDGET AND OBJECTIVE =====\nEach replay is capped at {budget} agent calls. "
            f"score = quality - {DREAM_COST_WEIGHT} * (calls spent / {budget}), where quality is "
            f"mean best-per-task score scaled by roster coverage. Unspent calls are saved, so "
            f"stopping a line that has plateaued is rewarded. 'off-map' counts requests for "
            f"branches history never recorded: each returned None and still cost a call - "
            f"a high count means the policy is steering where the recorded trees cannot follow. "
            f"A revision must beat the incumbent by more than {DREAM_TIE_EPS} to be adopted.\n\n"
            f"Write revision {m} of the exploration policy. Change the search SHAPE - "
            f"branching, parallel grouping, depth allocation, stopping - not the agents' "
            f"objectives. Output only Python."
        )
        try:
            raw, _, _ = _apex_completion(client, _PROMPT_POLICY_DEV, user,
                                         APEX_POLICY_TOKENS, 0.8)
        except Exception as exc:
            print(f"    [!] Candidate {m} generation failed: {str(exc)[:120]}", flush=True)
            continue

        source = re.sub(r'^```[a-zA-Z]*\s*|```\s*$', '', raw.strip(), flags=re.MULTILINE).strip()
        with open(ddir / f"candidate_{m:02d}.py", "w", encoding="ascii") as f:
            f.write(enforce_ascii(source) + "\n")

        res = replay_score(source, pool, tasks, budget, prior)
        res.update({"name": f"pi_{m}", "index": m})
        results.append(res)
        candidates.append({"name": f"pi_{m}", "source": source})

        if res["valid"]:
            print(f"    [+] pi_{m}: score {res['score']:.4f} "
                  f"| mean_best {res.get('mean_best', 0):.4f} "
                  f"| coverage {res.get('coverage', 0):.2f} | cost {res.get('cost', 0)}", flush=True)
        else:
            print(f"    [!] pi_{m}: rejected ({res['detail'][:80]})", flush=True)
        leaderboard += "\n" + _board(f"pi_{m}", res)

    winner = _select_winner(results)
    if not winner["valid"]:
        # Nothing replays - not even the incumbent. Keep deploying it rather than
        # crash; the live round falls back to pi_0 if it fails there too.
        winner = results[0]
    winner_source = next(c["source"] for c in candidates if c["name"] == winner["name"])

    with open(ddir / "scores.json", "w", encoding="ascii") as f:
        json.dump(results, f, indent=2)

    next_path = policy_path(run_dir, rnd + 1)
    next_path.parent.mkdir(parents=True, exist_ok=True)
    body = enforce_ascii(winner_source).rstrip()
    # Strip a previous selection header so headers do not accumulate.
    body = re.sub(r'\A(# Deployed for round .*\n# Replay score .*\n)+', '', body + "\n").rstrip()
    with open(next_path, "w", encoding="ascii") as f:
        f.write(f"# Deployed for round {rnd + 1}. Selected by replay over {len(pool)} tree(s).\n"
                f"# Replay score {winner['score']:.4f} (pi_0 baseline {base['score']:.4f}).\n"
                + body + "\n")

    improved = winner["name"] != "pi_0 (deployed)"
    print(f"    [+] Winner: {winner['name']} (score {winner['score']:.4f} vs "
          f"pi_0 {base['score']:.4f}) -> {next_path.name}"
          + ("" if improved else "  [no revision beat the incumbent; policy unchanged]"), flush=True)

    append_event(run_dir, {
        "round": rnd, "event": "dream", "candidates": len(results),
        "valid": len([r for r in results if r["valid"]]), "winner": winner["name"],
        "winner_score": winner["score"], "baseline_score": base["score"],
        "improved": improved,
    })

    return winner_source, {
        "round": rnd, "winner": winner["name"], "winner_score": winner["score"],
        "baseline_score": base["score"], "candidates": len(results),
        "valid": len([r for r in results if r["valid"]]), "improved": improved,
    }


# ------------------------------------------------------------------
# Planning (apex)
# ------------------------------------------------------------------

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
    """Planning. Runs on APEX: it is neither an agent assignment nor a merge."""
    print(f"\n[PHASE 3] [1] INGRESS: Analyzing query...\n    Length: {len(large_query)} characters", flush=True)

    fitted_query = fit_context(large_query, MAX_CONTEXT_CHARS)
    user_content = f"Partition this into mutually exclusive agent assignments:\n\n{fitted_query}"

    for attempt in range(1, MAX_RETRIES + 1):
        client = apex_client(timeout=WORKER_TIMEOUT_SECS)
        print(f"[2] PARTITION: Planning agent assignments via apex {GEN_API_BASE} [{LLM_MODEL}] "
              f"(Attempt {attempt}/{MAX_RETRIES})...", flush=True)
        try:
            start_time = time.time()
            raw_output, prompt_tokens, comp_tokens = _apex_completion(
                client, _PROMPT_PHASE3_DECOMPOSE, user_content, APEX_PLAN_TOKENS, 0.7
            )
            cleaned_output = extract_json_array(raw_output)
            if not cleaned_output:
                raise ValueError("Could not locate JSON array.")

            atomic_pieces = json.loads(cleaned_output)
            if not isinstance(atomic_pieces, list) or not all(isinstance(x, str) for x in atomic_pieces):
                raise ValueError(f"Partition produced non-string items: {type(atomic_pieces)}")

            atomic_pieces = [p.strip() for p in atomic_pieces if p and p.strip()]
            if not atomic_pieces:
                raise ValueError("Partition produced no usable assignments.")

            if len(atomic_pieces) > MAX_DECOMPOSE_TASKS:
                print(f"    [!] Partitioner emitted {len(atomic_pieces)} assignments; clamping to {MAX_DECOMPOSE_TASKS}.", flush=True)
                atomic_pieces = atomic_pieces[:MAX_DECOMPOSE_TASKS]

            elapsed = round(time.time() - start_time, 2)
            print(f"    [+] Success! Partitioned into {len(atomic_pieces)} agent assignment(s) in {elapsed}s.", flush=True)
            return atomic_pieces, prompt_tokens, comp_tokens

        except Exception as e:
            print(f"    [!] Partition Error: {e}", flush=True)
            time.sleep(2)

    fallback = fit_context(large_query, AGENT_OBJECTIVE_BUDGET)
    return [fallback], estimate_tokens(fallback), 0


# ------------------------------------------------------------------
# Online round + mechanical reconciliation
# ------------------------------------------------------------------

def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()


def best_nodes_for_round(nodes: List[dict]) -> Dict[str, dict]:
    best: Dict[str, dict] = {}
    for n in nodes:
        if not n.get("task"):
            continue
        cur = best.get(n["task"])
        if cur is None or n["score"] > cur["score"]:
            best[n["task"]] = n
    return best


# Boilerplate that legitimately appears in every package; identical copies are
# not duplicate WORK and should not become high-priority TO-DOs downstream.
_RECONCILE_TRIVIAL_NAMES = {"__init__.py", "py.typed", ".gitignore", ".gitkeep", "license",
                            "license.txt", "license.md", "copying", "conftest.py"}
RECONCILE_MIN_DUP_CHARS = int(os.getenv("RECONCILE_MIN_DUP_CHARS", "64"))


def reconcile_round(run_dir: Path, rnd: int, roster: List[dict],
                    nodes: List[dict]) -> Tuple[str, dict]:
    """Filesystem walk plus content hashing over each task's BEST node. Finds the
    failure mode this design targets: two agents producing the same artifact."""
    wroot = work_dir_for(run_dir)
    best = best_nodes_for_round(nodes)
    dir_to_agent = {r["dir"]: r["id"] for r in roster}

    by_hash: Dict[str, List[str]] = {}
    by_basename: Dict[str, List[str]] = {}
    per_agent_counts: Dict[str, int] = {}

    for task, node in best.items():
        ndir = wroot / node.get("dir", "") / node["id"]
        if not ndir.exists():
            continue
        for path in sorted(ndir.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            rel = path.relative_to(wroot)
            owner = dir_to_agent.get(node.get("dir", ""), task)
            per_agent_counts[owner] = per_agent_counts.get(owner, 0) + 1
            if path.name.lower() in _RECONCILE_TRIVIAL_NAMES:
                continue
            body = read_file_content_safe(path) or ""
            if len(_normalise_for_hash(body)) >= RECONCILE_MIN_DUP_CHARS:
                by_hash.setdefault(content_hash(body), []).append(f"{owner}:{rel}")
            by_basename.setdefault(path.name.lower(), []).append(f"{owner}:{rel}")

    dup_content = {h: v for h, v in by_hash.items() if len({e.split(':', 1)[0] for e in v}) > 1}
    dup_names = {n: v for n, v in by_basename.items() if len({e.split(':', 1)[0] for e in v}) > 1}

    real = [n for n in nodes if n.get("task")]
    violations = [(n["task"], v) for n in real for v in n.get("violations", [])]
    failed = sorted({n["task"] for n in real if n["status"] != "success"})
    truncated = sorted({n["task"] for n in real if n.get("truncated")})
    uncovered = sorted({r["id"] for r in roster} - set(best.keys()))

    lines = [f"# Round {rnd:02d} Reconciliation", ""]
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Nodes expanded:** {len(real)}")
    lines.append(f"- **Tasks reached:** {len(best)} of {len(roster)}"
                 + (f" (never expanded: {uncovered})" if uncovered else ""))
    lines.append(f"- **Tasks with a failed best attempt:** {len(failed)} {failed if failed else ''}")
    lines.append(f"- **Attempts cut off (output limit or wall clock):** {len(truncated)} {truncated if truncated else ''}")
    lines.append(f"- **Scope violations recorded:** {len(violations)}")
    lines.append(f"- **Identical files across agents:** {len(dup_content)}")
    lines.append(f"- **Colliding filenames across agents:** {len(dup_names)}")
    lines.append("")

    lines.append("## Best Attempt Per Agent")
    lines.append("")
    lines.append("| Agent | Best node | Depth | Score | Files | Status |")
    lines.append("|-------|-----------|-------|-------|-------|--------|")
    for r in roster:
        n = best.get(r["id"])
        if n is None:
            lines.append(f"| {r['id']} | (never expanded) | - | - | 0 | not reached |")
        else:
            lines.append(f"| {r['id']} | `{n['id']}` | {n['depth']} | {n['score']:.3f} | "
                         f"{per_agent_counts.get(r['id'], 0)} | {n['status']} |")
    lines.append("")

    if dup_content:
        lines.append("## DUPLICATE WORK - identical content produced by multiple agents")
        lines.append("")
        for h, entries in dup_content.items():
            lines.append(f"- `{h[:12]}` :: " + ", ".join(f"`{e}`" for e in entries))
        lines.append("")
        lines.append("Next round: the agent listed FIRST retains ownership. Every other listed "
                     "agent must drop its copy and reference the owner's path.")
        lines.append("")

    if dup_names:
        lines.append("## FILENAME COLLISIONS across agent directories")
        lines.append("")
        for n, entries in dup_names.items():
            lines.append(f"- `{n}` :: " + ", ".join(f"`{e}`" for e in entries))
        lines.append("")

    if violations:
        lines.append("## SCOPE VIOLATIONS")
        lines.append("")
        for aid, v in violations:
            lines.append(f"- **{aid}**: {v}")
        lines.append("")

    if not dup_content and not dup_names and not violations:
        lines.append("## Result")
        lines.append("")
        lines.append("No overlapping deliverables or scope violations detected this round.")
        lines.append("")

    report = enforce_ascii("\n".join(lines))
    rdir = round_dir_for(run_dir, rnd)
    rdir.mkdir(parents=True, exist_ok=True)
    with open(rdir / "RECONCILE.md", "w", encoding="ascii") as f:
        f.write(report)

    stats = {
        "nodes": len(real), "dup_content": len(dup_content), "dup_names": len(dup_names),
        "violations": len(violations), "failed": len(failed), "truncated": len(truncated),
        "files": sum(per_agent_counts.values()), "coverage": len(best),
        "best_mean": round(sum(n["score"] for n in best.values()) / max(1, len(best)), 4),
    }
    append_event(run_dir, {"round": rnd, "event": "reconcile", **stats})
    return report, stats


def run_online_round(roster: List[dict], rnd: int, background: str, run_dir: Path,
                     policy_source: str, budget: int,
                     semantic_guidance: bool = False) -> Tuple[List[dict], dict]:
    """Deploy the policy online. It drives the agents; the tree it grows is the
    world the next round dreams in."""
    tasks = [r["id"] for r in roster]
    slot_queue, slot_count = build_worker_slot_queue(prefix="A-Slot")

    print(f"\n[3] ROUND {rnd:02d}: deploying {policy_path(run_dir, rnd).name} over "
          f"{len(tasks)} assignment(s), budget {budget} agent call(s) "
          f"({budget - support_reserve(budget, len(tasks))} policy + "
          f"{support_reserve(budget, len(tasks))} support), "
          f"{slot_count} slot(s) [{WORKER_MODEL}]...", flush=True)

    # Novelty reference for fresh (depth-1) attempts: everything this task has
    # already produced in earlier rounds. Continuations compare to their parent.
    prior_hashes: Dict[str, Set[str]] = {}
    for prnd, pnodes in load_pool(run_dir):
        if prnd >= rnd:
            continue
        for n in pnodes:
            if n.get("task"):
                prior_hashes.setdefault(n["task"], set()).update(n.get("file_hashes", []))

    reserve = support_reserve(budget, len(tasks))
    explorer = LiveExplorer(tasks, budget - reserve, roster, rnd, background, run_dir,
                            slot_queue, prior_hashes=prior_hashes,
                            semantic_guidance=semantic_guidance)
    start = time.time()
    ok, detail = run_policy(policy_source, explorer)
    print()

    if not ok and _shutdown_event.is_set():
        print(f"    [!] Policy stopped for shutdown ({detail}).", flush=True)
    elif not ok:
        print(f"    [!] Deployed policy failed ({detail}). Falling back to pi_0 for the "
              f"remaining budget.", flush=True)
        append_event(run_dir, {"round": rnd, "event": "policy_failure", "detail": detail})
        run_policy(DEFAULT_POLICY_SOURCE, explorer)
        print()
    elif detail != "ok":
        print(f"    [~] Policy {detail}", flush=True)

    support_spent = 0
    if reserve > 0 and not _shutdown_event.is_set():
        # The policy's cap was budget - reserve; lift it by exactly the reserve.
        # Calls the policy chose not to spend are NOT handed to the probes, or
        # stopping early would stop saving anything.
        with explorer._lock:
            explorer._budget = explorer._spent + reserve
        support_spent = explorer.run_support_probes(reserve)
        print()
        print(f"    [+] Support probes: {support_spent}/{reserve} call(s) off-policy.", flush=True)
        append_event(run_dir, {"round": rnd, "event": "support", "reserved": reserve,
                               "spent": support_spent})

    nodes = [n for n in explorer._nodes if n.get("task")]
    elapsed = time.time() - start
    print(f"    [+] Round {rnd:02d} online phase complete: {len(nodes)} node(s), "
          f"{explorer.spent()}/{budget} agent call(s), {elapsed:.1f}s.", flush=True)

    report, stats = reconcile_round(run_dir, rnd, roster, nodes)
    stats["spent"] = explorer.spent()
    stats["support_spent"] = support_spent
    stats["elapsed"] = round(elapsed, 2)
    print(f"    [+] Reconciled round {rnd:02d}: {stats['files']} file(s), "
          f"{stats['dup_content']} duplicate artifact(s), "
          f"{stats['violations']} scope violation(s), mean best score {stats['best_mean']:.3f}.",
          flush=True)

    return nodes, stats


def writeback_test_scores(run_dir: Path, rnd: int, results: List[dict]) -> int:
    """Fold Phase 5 telemetry back onto the tree nodes. The next round's dreaming
    then scores policies against test-informed outcomes rather than the
    creation-time heuristic alone - this is how the worlds improve, not just the
    policy."""
    if not results:
        return 0
    nodes = load_tree(run_dir, rnd)
    by_node: Dict[str, List[dict]] = {}
    for r in results:
        nid = r.get("node")
        if nid:
            by_node.setdefault(nid, []).append(r)
    touched = 0
    for n in nodes:
        rs = by_node.get(n["id"])
        if not rs:
            continue
        passed = sum(1 for r in rs if r.get("status") == "PASSED")
        rate = passed / len(rs)
        n["test_pass_rate"] = round(rate, 4)
        n["test_count"] = len(rs)
        n["score"] = blend_test_score(n.get("heuristic_score", n["score"]), rate,
                                      bool(n.get("files")))
        touched += 1
    if touched:
        rewrite_tree(run_dir, rnd, nodes)
        append_event(run_dir, {"round": rnd, "event": "score_writeback", "nodes": touched})
    return touched


def build_run_manifest(run_dir: Path, roster: List[dict], pool: List[Tuple[int, List[dict]]],
                       round_stats: List[dict], dream_stats: List[dict],
                       original_query: str, elapsed: float,
                       plan_p: int, plan_c: int) -> str:
    """Mechanical index over the tree, the policies and the dreaming record."""
    wroot = work_dir_for(run_dir)
    files: List[Tuple[str, int]] = []
    if wroot.exists():
        for p in sorted(wroot.rglob("*")):
            if p.is_file() and not p.is_symlink():
                files.append((str(p.relative_to(wroot)), p.stat().st_size))

    all_nodes = [n for _, nodes in pool for n in nodes if n.get("task")]
    a_p = sum(n.get("prompt_tokens", 0) for n in all_nodes)
    a_c = sum(n.get("completion_tokens", 0) for n in all_nodes)

    lines = ["# Run Manifest", ""]
    lines.append(f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Agents:** {len(roster)}")
    lines.append(f"- **Rounds:** {len(round_stats)}")
    lines.append(f"- **Tree nodes (agent calls):** {len(all_nodes)}")
    lines.append(f"- **Deliverable files:** {len(files)} ({sum(s for _, s in files):,} bytes)")
    lines.append(f"- **Assembly mode:** agent-owned directories, mechanical reconciliation, "
                 f"replay-driven exploration policy (no stitcher tier)")
    lines.append("")

    lines.append("## Query")
    lines.append("")
    q = original_query.strip()
    lines.append("```")
    lines.append(q[:4000] + ("\n...[QUERY TRUNCATED IN MANIFEST]..." if len(q) > 4000 else ""))
    lines.append("```")
    lines.append("")

    lines.append("## Agent Roster")
    lines.append("")
    lines.append("| Agent | Directory | Objective |")
    lines.append("|-------|-----------|-----------|")
    for r in roster:
        obj = r["objective"].replace("\n", " ").replace("|", "/")
        lines.append(f"| {r['id']} | `{WORK_DIRNAME}/{r['dir']}/` | {obj[:160]} |")
    lines.append("")

    lines.append("## Recursive Rounds")
    lines.append("")
    lines.append("| Round | Policy | Agent calls | Nodes | Mean best score | Duplicates | Violations |")
    lines.append("|-------|--------|-------------|-------|-----------------|------------|------------|")
    for i, s in enumerate(round_stats, start=1):
        lines.append(f"| {i} | `{POLICY_DIRNAME}/pi_r{i:02d}.py` | {s.get('spent', 0)} | "
                     f"{s.get('nodes', 0)} | {s.get('best_mean', 0):.3f} | "
                     f"{s.get('dup_content', 0)} | {s.get('violations', 0)} |")
    lines.append("")

    if dream_stats:
        lines.append("## Dreaming (offline policy improvement)")
        lines.append("")
        lines.append("| After round | Candidates | Valid | Winner | Replay score | pi_0 baseline | Changed |")
        lines.append("|-------------|------------|-------|--------|--------------|---------------|---------|")
        for d in dream_stats:
            lines.append(f"| {d['round']} | {d['candidates']} | {d['valid']} | {d['winner']} | "
                         f"{d['winner_score']:.4f} | {d['baseline_score']:.4f} | "
                         f"{'yes' if d['improved'] else 'no'} |")
        lines.append("")
        lines.append("Replay costs zero agent calls: every outcome a candidate policy asks for "
                     "is already recorded. The deployed policy is always a candidate, so the "
                     "selected policy is never worse than the one it replaces.")
        lines.append("")

    lines.append("## Discovery Tree")
    lines.append("")
    lines.append("| Node | Round | Parent | Depth | Task | Score | Tests | Status | Files |")
    lines.append("|------|-------|--------|-------|------|-------|-------|--------|-------|")
    for n in sorted(all_nodes, key=lambda x: (x.get("round", 0), x.get("seq", 0))):
        tr = n.get("test_pass_rate")
        tr_s = f"{tr:.2f}" if isinstance(tr, (int, float)) else "-"
        lines.append(f"| `{n['id']}` | {n.get('round', '')} | `{n.get('parent', '')}` | "
                     f"{n.get('depth', '')} | {n.get('task', '')} | {n.get('score', 0):.3f} | "
                     f"{tr_s} | {n.get('status', '')} | {len(n.get('files', []))} |")
    lines.append("")

    lines.append("## Deliverable Index")
    lines.append("")
    if files:
        lines.append("| File | Bytes |")
        lines.append("|------|-------|")
        for rel, size in files:
            lines.append(f"| `{WORK_DIRNAME}/{rel}` | {size:,} |")
    else:
        lines.append("(no deliverables written)")
    lines.append("")

    lines.append("## Cluster Aggregate Statistics")
    lines.append("")
    lines.append(f"- **Total Wall-Clock Time:** {elapsed:.2f} seconds")
    lines.append(f"- **Agent Prompt Tokens:** {a_p}")
    lines.append(f"- **Agent Completion Tokens:** {a_c}")
    lines.append(f"- **Apex Planning Prompt Tokens:** {plan_p}")
    lines.append(f"- **Apex Planning Completion Tokens:** {plan_c}")
    lines.append(f"- **Stitcher Tokens:** 0 (tier removed)")
    lines.append(f"- **Replay Executions:** 0 by construction")
    lines.append(f"- **Agent Model:** {WORKER_MODEL}")
    lines.append(f"- **Agent Endpoints:** {', '.join(WORKER_ENDPOINTS)} (x{WORKER_PARALLEL_SLOTS} slot(s))")
    lines.append(f"- **Apex Model:** {LLM_MODEL} @ {GEN_API_BASE}")
    lines.append("")
    lines.append(f"Tree: `{TREES_DIRNAME}/` | Policies: `{POLICY_DIRNAME}/` | "
                 f"Dreaming: `{DREAM_DIRNAME}/` | Comms: `{COMMS_DIRNAME}/`")
    lines.append("")
    return enforce_ascii("\n".join(lines))
# ==============================================================================
# Phase 5: Automatic Unittests over real agent deliverables
# ------------------------------------------------------------------------------
# Agents now write real files, so there is no markdown code-fence extraction
# step. Phase 5 walks work/, selects testable files by extension, and namespaces
# generated tests by owning agent.
# ==============================================================================

def _format_execution_report_as_markdown(report_data: list) -> str:
    if not report_data:
        return ""
    lines = ["## Test Execution Status\n", "| Agent | Node | Artifact | Language | Status | Detail |", "|---|---|---|---|---|---|"]
    for res in report_data:
        lines.append(
            f"| {res.get('agent', '')} | {res.get('node', '')} | {res.get('filename', '')} | {res.get('language', '')} | "
            f"**{res.get('status', '')}** | {res.get('message', '')} |"
        )
    return "\n".join(lines) + "\n\n"


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


_REQ_LINE_RE = re.compile(
    r'^([A-Za-z0-9][A-Za-z0-9._-]*)'                     # name
    r'(\[[A-Za-z0-9._,\s-]+\])?'                          # extras
    r'\s*((?:==|>=|<=|~=|!=|<|>)\s*[A-Za-z0-9.*+!_-]+'    # first specifier
    r'(?:\s*,\s*(?:==|>=|<=|~=|!=|<|>)\s*[A-Za-z0-9.*+!_-]+)*)?\s*$')


def sanitize_requirements(text: str) -> Tuple[List[str], List[str]]:
    """Keep only plain `name[extras] <specifiers>` lines. Everything that can make
    pip fetch or execute something other than a named index package is dropped:
    URLs and `pkg @ url`, -e/-r/-c, --index-url/--extra-index-url, local paths,
    environment markers. Returns (kept, dropped)."""
    kept, dropped = [], []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _REQ_LINE_RE.match(line)
        if not m:
            dropped.append(line)
            continue
        name = m.group(1).lower().replace("_", "-")
        if TEST_PIP_ALLOWLIST and name not in TEST_PIP_ALLOWLIST:
            dropped.append(line)
            continue
        kept.append(line)
    return kept, dropped


def _run_limited(cmd: List[str], timeout: float, cwd: Path, env: Dict[str, str],
                 cpu: int = TEST_CPU_SECS, mem_mb: int = TEST_MEM_MB,
                 fsize_mb: int = TEST_FSIZE_MB) -> Tuple[Optional[int], str, bool]:
    """Run model-written code (or tooling acting on it) under CPU, address-space
    and file-size limits in its own process group; a timeout kills the whole
    group, not just the direct child. Returns (returncode, output, timed_out)."""
    wrapped = ["bash", "-c",
               'ulimit -t %d; ulimit -v %d; ulimit -f %d; ulimit -c 0; exec "$@"'
               % (cpu, mem_mb * 1024, fsize_mb * 1024), "limited"] + cmd
    proc = subprocess.Popen(wrapped, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            encoding="ascii", errors="ignore", start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out or "", False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        out, _ = proc.communicate()
        return None, out or "", True


def _test_env(test_root: Path, venv_bin: Optional[Path], pythonpath: str = "") -> Dict[str, str]:
    """Clean environment for generated code: no inherited API keys or endpoints."""
    home = test_root / ".home"
    home.mkdir(parents=True, exist_ok=True)
    path = "/usr/local/bin:/usr/bin:/bin"
    if venv_bin:
        path = f"{venv_bin}:{path}"
    env = {"PATH": path, "HOME": str(home), "TMPDIR": str(home), "LANG": "C",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    return env


def ensure_test_venv(run_dir: Path) -> Tuple[str, Optional[Path]]:
    """Run-scoped venv for Phase 5 so model-chosen dependencies never land in the
    host interpreter. Inherits system site-packages for pytest and the numeric
    stack. Falls back to the current interpreter WITHOUT installs if it cannot
    be created."""
    venv_dir = run_dir / "tests" / ".venv"
    py = venv_dir / "bin" / "python"
    if not py.exists():
        try:
            subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
                           capture_output=True, timeout=180, check=True)
        except Exception as exc:
            print(f"    [!] Could not create test venv ({str(exc)[:80]}); tests run on the host "
                  f"interpreter and dependency installs are skipped.", flush=True)
            return sys.executable, None
    env = _test_env(run_dir / "tests", venv_dir / "bin")
    rc, _, _ = _run_limited([str(py), "-c", "import pytest"], 60, run_dir / "tests", env)
    if rc != 0 and TEST_PIP_INSTALL:
        _run_limited([str(py), "-m", "pip", "install", "--only-binary=:all:", "--no-input",
                      "--disable-pip-version-check", "pytest"], 300, run_dir / "tests", env)
    return str(py), venv_dir / "bin"


def install_round_requirements(run_dir: Path, nodes_tested: List[dict], rnd: int,
                               py: str, venv_bin: Optional[Path]) -> None:
    """Install ONLY the requirements declared by the nodes under test this round,
    after sanitising a COPY (the agents' deliverables are never rewritten), as
    binary wheels only so no sdist build script runs."""
    if not TEST_PIP_INSTALL or venv_bin is None:
        return
    wroot = work_dir_for(run_dir)
    kept_all: List[str] = []
    dropped_all: List[str] = []
    for node in nodes_tested:
        ndir = wroot / node.get("dir", "") / node["id"]
        if not ndir.exists():
            continue
        for req in sorted(ndir.rglob("requirements*.txt")):
            kept, dropped = sanitize_requirements(read_file_content_safe(req) or "")
            kept_all.extend(k for k in kept if k not in kept_all)
            dropped_all.extend(dropped)
    if dropped_all:
        print(f"    [!] Dropped {len(dropped_all)} requirement line(s) that were not plain "
              f"index packages{' or not allowlisted' if TEST_PIP_ALLOWLIST else ''}: "
              f"{', '.join(dropped_all[:5])}{' ...' if len(dropped_all) > 5 else ''}", flush=True)
    if not kept_all:
        return
    test_root = run_dir / "tests" / f"round{rnd:02d}"
    test_root.mkdir(parents=True, exist_ok=True)
    req_copy = test_root / "requirements.sanitized.txt"
    with open(req_copy, "w", encoding="ascii") as f:
        f.write("\n".join(kept_all) + "\n")
    env = _test_env(run_dir / "tests", venv_bin)
    rc, out, timed_out = _run_limited(
        [py, "-m", "pip", "install", "--only-binary=:all:", "--no-input",
         "--disable-pip-version-check", "-r", str(req_copy)], 600, test_root, env)
    if rc != 0:
        tail = (out.strip().splitlines() or ["(no output)"])[-1]
        print(f"    [!] Warning: pip install {'timed out' if timed_out else 'failed'} "
              f"for round requirements: {tail[:120]}", flush=True)


def top_nodes_per_task(nodes: List[dict], k: int) -> List[dict]:
    """The k best nodes per task by decision-time score (deeper first on ties).
    Testing more than the single best gives replay a test-informed signal on
    whether continuing a line actually paid, not just on the winner."""
    by_task: Dict[str, List[dict]] = {}
    for n in nodes:
        if n.get("task") and n.get("files"):
            by_task.setdefault(n["task"], []).append(n)
    out = []
    for task in sorted(by_task):
        ranked = sorted(by_task[task], key=lambda n: (n.get("score_online", n.get("score", 0.0)),
                                                       n.get("depth", 0)), reverse=True)
        out.extend(ranked[:k])
    return out


def _test_filename(artifact_name: str) -> str:
    stem, suffix = Path(artifact_name).stem, Path(artifact_name).suffix.lower()
    if suffix == ".h":
        return f"test_{stem}_h.c"
    if suffix == ".hpp":
        return f"test_{stem}_hpp.cpp"
    return f"test_{stem}{suffix}"


def collect_testable_artifacts(run_dir: Path, roster: List[dict],
                               nodes: List[dict]) -> List[dict]:
    """Testable files of the top TEST_NODES_PER_TASK nodes per task."""
    wroot = work_dir_for(run_dir)
    if not wroot.exists():
        return []
    artifacts = []
    for node in top_nodes_per_task(nodes, TEST_NODES_PER_TASK):
        ndir = wroot / node.get("dir", "") / node["id"]
        if not ndir.exists():
            continue
        for path in sorted(ndir.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            lang = _TESTABLE_EXT_LANG.get(path.suffix.lower())
            if not lang:
                continue
            content = read_file_content_safe(path)
            if content is None or not content.strip():
                continue
            artifacts.append({
                "agent": node["task"],
                "node": node["id"],
                "filename": path.name,
                "relative_path": str(path.relative_to(wroot)),
                "language": lang,
                "filepath": str(path),
                "content": content,
                "content_hash": content_hash(content),
            })
    return artifacts


def request_unittests_from_worker(artifact: dict, endpoint_queue: queue.Queue, test_output_dir: Path,
                                  progress_lock: threading.Lock, progress_state: dict) -> Optional[str]:
    """Generate one test for an artifact. Returns the generated test SOURCE (the
    caller writes it into every node that carries identical content)."""
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
        code_content = fit_context(artifact['content'], MAX_CONTEXT_CHARS)
        lang = artifact['language']
        extra = ""
        if lang in ("c", "cpp"):
            extra = (f"\nInclude it exactly as: #include \"{artifact['filename']}\". "
                     f"If the file defines its own main(), it is renamed to "
                     f"autoresearch_artifact_main() before compiling, so write your own main().")
        prompt = (
            f"File: {artifact['filename']}{extra}\n"
            f"```{lang}\n{code_content}\n```"
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
        headers = {"Authorization": f"Bearer {WORKER_API_KEY}"}
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(endpoint_url, json=payload, headers=headers,
                                         timeout=TEST_TIMEOUT_SECS)
                response.raise_for_status()
                result = response.json()
                choices = result.get("choices")
                test_code = (choices[0].get("message", {}).get("content", "") if choices else "")
                if not test_code:
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER))
                        continue
                    return None
                test_code = enforce_ascii(_strip_markdown_fences(test_code))
                with progress_lock:
                    progress_state["done"] += 1
                    eta_str = _format_eta(progress_state["start_time"], progress_state["done"], progress_state["total"])
                    print(f"    [+] Generated tests ({progress_state['done']}/{progress_state['total']}) "
                          f"| ETC: {eta_str} -> {artifact['agent']}/{artifact['node']}/"
                          f"{_test_filename(artifact['filename'])}", flush=True)
                return test_code
            except requests.exceptions.RequestException:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER))
        return None
    finally:
        endpoint_queue.put(endpoint_url)


_C_MAIN_RE = re.compile(r'\bint\s+main\s*\(')
_IMPL_SUFFIXES = {"c": [".c"], "cpp": [".cpp", ".cc", ".cxx", ".c"]}


def _shim_main(src: str) -> str:
    return _C_MAIN_RE.sub("int autoresearch_artifact_main(", src)


def execute_test_artifact(test_meta: dict) -> dict:
    lang = test_meta["language"].lower()
    test_path = Path(test_meta["test_filepath"]).resolve()
    artifact_path = Path(test_meta["artifact_filepath"]).resolve()
    py = test_meta.get("python", sys.executable)
    venv_bin = Path(test_meta["venv_bin"]) if test_meta.get("venv_bin") else None
    test_root = Path(test_meta["test_root"])
    result = {"agent": test_meta.get("agent", ""), "node": test_meta.get("node", ""),
              "filename": test_meta["filename"], "language": lang,
              "status": "UNKNOWN", "message": ""}

    def _finish(rc, out, timed_out, t0):
        if timed_out:
            result["status"], result["message"] = "TIMEOUT", "Execution threshold exceeded"
        elif rc == 0:
            result["status"], result["message"] = "PASSED", f"OK ({time.time() - t0:.2f}s)"
        else:
            result["status"], result["message"] = "FAILED", _extract_error_line(out, lang)

    try:
        if lang in ("python", "py"):
            env = _test_env(test_root, venv_bin,
                            os.pathsep.join([str(artifact_path.parent), str(test_path.parent)]))
            cmd = [py, "-m", "pytest", "-p", "no:cacheprovider", "--no-header", "--tb=short", "-q", str(test_path)]
            t0 = time.time()
            rc, out, to = _run_limited(cmd, 45, test_path.parent, env)
            _finish(rc, out, to, t0)
        elif lang in ("bash", "sh"):
            env = _test_env(test_root, venv_bin)
            t0 = time.time()
            rc, out, to = _run_limited(["bash", str(test_path)], 30, test_path.parent, env)
            _finish(rc, out, to, t0)
        elif lang in ("c", "cpp"):
            compiler = "gcc" if lang == "c" else "g++"
            build = test_path.parent / f"build_{test_path.stem}"
            shutil.rmtree(build, ignore_errors=True)
            build.mkdir(parents=True)
            try:
                # The test is compiled from build/, so `#include "artifact"` resolves
                # to the copy placed beside it (main() renamed if present).
                test_copy = build / test_path.name
                shutil.copyfile(test_path, test_copy)
                art_src = read_file_content_safe(artifact_path) or ""
                with open(build / artifact_path.name, "w", encoding="ascii", errors="ignore") as f:
                    f.write(_shim_main(art_src))
                sources = [str(test_copy)]
                # Testing a header: link its implementation sibling, if any.
                if artifact_path.suffix.lower() in (".h", ".hpp"):
                    for suf in _IMPL_SUFFIXES[lang]:
                        impl = artifact_path.with_suffix(suf)
                        if impl.exists():
                            impl_copy = build / f"impl_{impl.name}"
                            with open(impl_copy, "w", encoding="ascii", errors="ignore") as f:
                                f.write(_shim_main(read_file_content_safe(impl) or ""))
                            sources.append(str(impl_copy))
                            break
                binary = build / "test.bin"
                env = _test_env(test_root, None)
                compile_cmd = [compiler, "-O0", "-I", str(build), "-I", str(artifact_path.parent)] \
                    + sources + ["-o", str(binary), "-lm"]
                rc, out, to = _run_limited(compile_cmd, 60, build, env)
                if to or rc != 0:
                    result["status"] = "COMPILE_ERROR"
                    result["message"] = "compile timed out" if to else _extract_error_line(out, lang)
                    return result
                t0 = time.time()
                rc, out, to = _run_limited([str(binary)], 30, build, env)
                _finish(rc, out, to, t0)
            finally:
                shutil.rmtree(build, ignore_errors=True)
        else:
            result["status"], result["message"] = "SKIPPED", f"No environment definition for: {lang}"
    except Exception as exc:
        result["status"], result["message"] = "ERROR", str(exc)[:200]
    return result


def run_phase5_automatic_unittests(run_dir: Path, roster: List[dict],
                                   nodes: List[dict], rnd: int) -> List[dict]:
    print(f"\n[PHASE 5] ROUND {rnd:02d}: AUTOMATED UNITTEST PIPELINE", flush=True)

    TEST_OUTPUT_DIR = run_dir / "tests" / f"round{rnd:02d}"
    REPORT_OUTPUT_DIR = run_dir / "reports"

    artifacts = collect_testable_artifacts(run_dir, roster, nodes)
    if not artifacts:
        print("    [!] No testable deliverables among this round's top nodes.", flush=True)
        return []

    # Continuations inherit files, so identical content recurs across nodes.
    # Generate one test per distinct (content, name, language); execute it in
    # every node that carries it (siblings and imports may differ per node).
    groups: Dict[Tuple[str, str, str], List[dict]] = {}
    for a in artifacts:
        groups.setdefault((a["content_hash"], a["filename"], a["language"]), []).append(a)
    reps = [members[0] for members in groups.values()]

    by_agent: Dict[str, int] = {}
    for a in artifacts:
        by_agent[a["agent"]] = by_agent.get(a["agent"], 0) + 1
    for aid, n in sorted(by_agent.items()):
        print(f"    [+] {aid}: {n} testable file(s).", flush=True)
    print(f"    [*] {len(artifacts)} testable file(s) across top-{TEST_NODES_PER_TASK} node(s)/task; "
          f"{len(reps)} distinct -> {len(reps)} test generation(s).", flush=True)

    endpoint_queue: queue.Queue = queue.Queue()
    for ep in TEST_WORKER_ENDPOINTS:
        for _ in range(CONCURRENT_REQS_PER_ENDPOINT):
            endpoint_queue.put(ep)
    total_gen_workers = max(1, len(TEST_WORKER_ENDPOINTS) * CONCURRENT_REQS_PER_ENDPOINT)

    progress_lock = threading.Lock()
    progress_state = {"done": 0, "total": len(reps), "start_time": time.time()}
    generated: Dict[Tuple[str, str, str], str] = {}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=total_gen_workers)
    try:
        fut_to_key = {executor.submit(request_unittests_from_worker, rep, endpoint_queue,
                                      TEST_OUTPUT_DIR, progress_lock, progress_state):
                      (rep["content_hash"], rep["filename"], rep["language"]) for rep in reps}
        for future in concurrent.futures.as_completed(fut_to_key):
            try:
                code = future.result()
                if code:
                    generated[fut_to_key[future]] = code
            except Exception:
                pass
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    py, venv_bin = ensure_test_venv(run_dir)
    tested_nodes = top_nodes_per_task(nodes, TEST_NODES_PER_TASK)
    install_round_requirements(run_dir, tested_nodes, rnd, py, venv_bin)

    generated_tests: List[dict] = []
    for key, members in groups.items():
        code = generated.get(key)
        if not code:
            continue
        for a in members:
            node_dir = TEST_OUTPUT_DIR / a["agent"] / a["node"]
            node_dir.mkdir(parents=True, exist_ok=True)
            tpath = node_dir / _test_filename(a["filename"])
            with open(tpath, "w", encoding="ascii") as f:
                f.write(code + "\n")
            generated_tests.append({
                "filename": tpath.name, "test_filepath": str(tpath), "language": a["language"],
                "artifact_filepath": a["filepath"], "agent": a["agent"], "node": a["node"],
                "python": py, "venv_bin": str(venv_bin) if venv_bin else "",
                "test_root": str(TEST_OUTPUT_DIR),
            })

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
        # Per-round report plus a cumulative one, since Phase 6 reads the latter.
        round_json = REPORT_OUTPUT_DIR / f"execution_report_round{rnd:02d}.json"
        with open(round_json, "w", encoding="ascii") as f:
            json.dump(execution_results, f, indent=4)

        cumulative: List[dict] = []
        cum_path = REPORT_OUTPUT_DIR / "execution_report.json"
        if cum_path.exists():
            try:
                prior = json.loads(read_file_content_safe(cum_path) or "[]")
                if isinstance(prior, list):
                    cumulative = [r for r in prior
                                  if not str(r.get("node", "")).startswith(f"r{rnd:02d}n")]
            except json.JSONDecodeError:
                cumulative = []
        cumulative.extend(execution_results)
        with open(cum_path, "w", encoding="ascii") as f:
            json.dump(cumulative, f, indent=4)
        with open(REPORT_OUTPUT_DIR / "execution_report.csv", "w", newline="", encoding="ascii") as f:
            writer = csv.DictWriter(f, fieldnames=EXECUTION_RESULT_FIELDS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(cumulative)

        passed = sum(1 for r in execution_results if r["status"] == "PASSED")
        print(f"    [+] Test execution complete: {passed}/{len(execution_results)} passed. "
              f"Reports in {REPORT_OUTPUT_DIR}", flush=True)

        for r in execution_results:
            append_event(run_dir, {
                "round": rnd, "event": "test_result", "agent": r.get("agent", ""),
                "node": r.get("node", ""), "artifact": r.get("filename", ""),
                "status": r.get("status", ""),
            })

    return execution_results


# ==============================================================================
# Phase 6: Todo Project Distillation (apex)
# ==============================================================================

def run_phase6_project_distillation(project_dir: Path, iterate: bool = False):
    print(f"\n[PHASE 6] STARTING PROJECT DISTILLATION", flush=True)
    project_name = project_dir.name

    # Phase 6 reads raw intake (Phase 0/1), test telemetry, and the mechanical
    # reconciliation reports. Agent deliverables under work/ are the product, not
    # the input, and the per-agent wave logs would swamp the apex window.
    exclude_dirs = {"tests", "tasks", "reports", WORK_DIRNAME,
                    TREES_DIRNAME, POLICY_DIRNAME, DREAM_DIRNAME, ABORTED_DIRNAME}
    p6_exclude_names = {"DISTILLED_TASKS", "project_state", "RUN_MANIFEST"}

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

    # Reconciliation reports are high-signal and small; pull them explicitly
    # before the generic walk excludes comms/.
    cdir = comms_dir_for(project_dir)
    if cdir.exists():
        for rpath in sorted(cdir.glob("round[0-9][0-9]/RECONCILE.md")):
            resolved = rpath.resolve()
            if resolved not in seen_paths:
                raw_files.append(rpath)
                seen_paths.add(resolved)

    exclude_dirs.add(COMMS_DIRNAME)

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

    raw_files.sort(key=lambda x: (0 if "execution_report" in x.name
                                  else 1 if x.name == "RECONCILE.md" else 2))

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
                    label = file_path.name
                    if file_path.name == "RECONCILE.md":
                        label = f"{file_path.parent.name}/RECONCILE.md"
                    aggregated_content.append(f"--- SOURCE: {label} ---\n{f.read()}")
        except Exception as e:
            print(f"    [!] Could not parse {file_path.name}: {e}", flush=True)

    full_text = "\n\n".join(aggregated_content)
    client = apex_client(base_url=DISTILLER_URL, api_key=DISTILLER_API_KEY,
                         timeout=WORKER_TIMEOUT_SECS)

    if existing_tasks:
        half_budget = MAX_CONTEXT_CHARS // 2
        existing_tasks_trimmed = fit_context(
            existing_tasks, half_budget - 1000,
            note="...[EXISTING TASKS TRUNCATED FOR CONTEXT LIMITS]..."
        )
        trimmed_telemetry = fit_context(
            full_text, half_budget - 1000,
            note="...[NEW TELEMETRY TRUNCATED FOR CONTEXT LIMITS]..."
        )
        sys_prompt = _PROMPT_PHASE6_ITERATE
        user_content = (
            f"--- CURRENT DISTILLED TASKS ---\n{existing_tasks_trimmed}\n\n"
            f"--- NEW TELEMETRY AND DOCS ---\n{trimmed_telemetry}"
        )
    else:
        sys_prompt = _PROMPT_PHASE6_DISTILL
        trimmed_telemetry = fit_context(
            full_text, MAX_CONTEXT_CHARS - 1000,
            note="...[CONTENT TRUNCATED FOR CONTEXT LIMITS]..."
        )
        user_content = (
            f"Extract actionable tasks, test outcomes, ownership conflicts, and relevant artifacts "
            f"from this {project_name} documentation:\n\n{trimmed_telemetry}"
        )

    distilled_markdown = None
    try:
        distilled_markdown, _, _ = _apex_completion(
            client, sys_prompt, user_content, APEX_DISTILL_TOKENS, 0.2, model=DISTILLER_MODEL
        )
    except Exception as e:
        print(f"[!] Apex inference failed: {e}", flush=True)

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

def archive_partial_round(run_dir: Path, rnd: int) -> bool:
    """A round without its done-marker was interrupted. Move everything it wrote
    aside before re-running it: node ids restart at rNNn0000, so appending to the
    old tree would collide ids and the last-write-wins merge would splice two
    different trees together, while stale node directories would leak files into
    reconciliation and Phase 5."""
    if round_done_marker(run_dir, rnd).exists():
        return False
    prefix = f"r{rnd:02d}n"
    moves: List[Tuple[Path, Path]] = []
    dest = run_dir / ABORTED_DIRNAME / f"round{rnd:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tree = tree_path_for(run_dir, rnd)
    if tree.exists():
        moves.append((tree, dest / TREES_DIRNAME / tree.name))
    rdir = round_dir_for(run_dir, rnd)
    if rdir.exists():
        moves.append((rdir, dest / COMMS_DIRNAME / rdir.name))
    tdir = run_dir / "tests" / f"round{rnd:02d}"
    if tdir.exists():
        moves.append((tdir, dest / "tests" / tdir.name))
    wroot = work_dir_for(run_dir)
    if wroot.exists():
        for adir in wroot.iterdir():
            if adir.is_dir():
                for ndir in adir.glob(f"{prefix}*"):
                    if ndir.is_dir():
                        moves.append((ndir, dest / WORK_DIRNAME / adir.name / ndir.name))
    if not moves:
        return False
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    print(f"[*] Round {rnd:02d} was interrupted; moved {len(moves)} partial artifact(s) to "
          f"{dest.relative_to(run_dir)} before re-running it.", flush=True)
    append_event(run_dir, {"round": rnd, "event": "round_reset", "archived": str(dest.relative_to(run_dir))})
    return True


def signal_handler(sig, frame):
    if _shutdown_event.is_set():
        print("\n[!] Force exit triggered.", flush=True)
        os._exit(1)

    _shutdown_event.set()
    print("\n[!] Graceful shutdown requested (SIGINT/SIGTERM). Awaiting active threads to abort... "
          "(Press Ctrl+C again to force exit)", flush=True)


def round_budget(roster: List[dict]) -> int:
    """Budget is in agent calls and is identical online and in replay, so a
    dreamed policy cannot win by spending more than the one it replaces."""
    raw = int(round(len(roster) * ROUND_BUDGET_PER_TASK))
    return max(ROUND_BUDGET_MIN, min(ROUND_BUDGET_MAX, raw))


def support_reserve(budget: int, n_tasks: int) -> int:
    """Calls held back from the policy for off-policy support probes. Never so
    many that the policy cannot open every task once."""
    if SUPPORT_PROBE_FRAC <= 0:
        return 0
    want = int(round(SUPPORT_PROBE_FRAC * budget))
    return max(0, min(want, budget - n_tasks))


def policy_budget(budget: int, n_tasks: int) -> int:
    """What the deployed policy may spend online - and therefore the cap it is
    replayed under, so live and dream stay on the same footing."""
    return budget - support_reserve(budget, n_tasks)


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
    parser.add_argument("-n", "--rounds", type=int, default=DEFAULT_ROUNDS,
                        help=f"Recursive rounds (default {DEFAULT_ROUNDS}, max {MAX_ROUNDS}). Each round "
                             "deploys the current exploration policy, then improves it by replay.")
    parser.add_argument("--budget", type=int, default=0,
                        help="Agent calls available to the policy per round. Default scales with the roster.")
    parser.add_argument("--no-dream", action="store_true",
                        help="Skip offline policy improvement. Every round redeploys pi_0 - this is the "
                             "Recursive Fixed Exploration control the paper compares against.")
    parser.add_argument("--dream-only", action="store_true",
                        help="Run no agents. Dream over the existing tree pool and write the next policy. "
                             "Requires -r and at least one recorded round.")
    parser.add_argument("--semantic-guidance", action="store_true",
                        help="Inject high-level directional guidance into agent prompts. Off by default: "
                             "the paper's ablation found this underperforms unguided replay at equal budget.")

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
    if args.rounds < 1 or args.rounds > MAX_ROUNDS:
        parser.error(f"--rounds must be between 1 and {MAX_ROUNDS}.")
    if args.dream_only and not args.resume:
        parser.error("--dream-only requires -r/--resume: it dreams over an existing tree pool.")
    if args.dream_only and args.no_dream:
        parser.error("--dream-only and --no-dream are contradictory.")

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

    work_base = Path(args.dir).resolve()
    category_dir = work_base / args.category

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

    manifest_path = target_directory / "RUN_MANIFEST.md"
    distilled_tasks_path = target_directory / "DISTILLED_TASKS.md"

    # ---------------- Dream-only shortcut ----------------
    # Policy improvement costs zero agent calls, so it is worth being able to run
    # on its own against a pool recorded earlier - no cluster required beyond apex.
    if args.dream_only:
        roster = load_roster(target_directory)
        pool = load_pool(target_directory)
        if not roster or not pool:
            print("[!] --dream-only needs a roster and at least one recorded tree.")
            sys.exit(1)
        if not ping_tier([GEN_API_BASE], LLM_MODEL, GEN_API_KEY, "Apex", timeout=90.0):
            print("\n[!] Apex tier offline; policy improvement cannot run.", flush=True)
            sys.exit(1)
        last = max(r for r, _ in pool)
        tasks = [r["id"] for r in roster]
        budget = args.budget if args.budget > 0 else round_budget(roster)
        dream_policy_improvement(target_directory, last,
                                 load_or_init_policy(target_directory, last),
                                 pool, tasks, budget)
        print("\n[+] Dreaming complete. Re-run with -r to deploy the selected policy.")
        return

    verify_server_props([GEN_API_BASE], "Apex", APEX_SERVER_CTX, APEX_SERVER_NP)
    verify_server_props(WORKER_ENDPOINTS, "Agent", WORKER_SERVER_CTX, WORKER_SERVER_NP)

    apex_ok = ping_tier([GEN_API_BASE], LLM_MODEL, GEN_API_KEY, "Apex", timeout=90.0)
    worker_ok = ping_tier(WORKER_ENDPOINTS, WORKER_MODEL, WORKER_API_KEY, "Agent", timeout=90.0)
    if not worker_ok:
        print("\n[!] Agent tier failed smoke tests. Aborting.", flush=True)
        sys.exit(1)

    raw_filepath = None
    distilled_filepath = None

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

    target_query = read_file_content_safe(distilled_filepath)
    if target_query is None:
        print(f"[!] Fatal: Could not read {distilled_filepath}.", flush=True)
        sys.exit(1)

    # ---------------- Phase 3: partition ----------------
    master_start_time = time.time()
    roster = load_roster(target_directory) if args.resume else []
    plan_p, plan_c = 0, 0

    if roster:
        print(f"[PHASE 3] Roster loaded from {COMMS_DIRNAME}/roster.json ({len(roster)} agent(s)).")
    else:
        if not apex_ok:
            print("\n[!] Apex tier offline; agent partitioning cannot run. "
                  "Re-run with -r once 8081 is healthy.", flush=True)
            sys.exit(1)
        fragments, plan_p, plan_c = decompose_to_atomic_pieces(target_query)
        roster = build_roster(fragments)
        for d in (comms_dir_for, work_dir_for, trees_dir_for, policy_dir_for, dream_dir_for):
            d(target_directory).mkdir(parents=True, exist_ok=True)
        save_roster(target_directory, roster)
        export_to_split_files(roster, target_directory)
        append_event(target_directory, {"event": "roster", "agents": len(roster)})

    tasks = [r["id"] for r in roster]
    budget = args.budget if args.budget > 0 else round_budget(roster)

    # ---------------- Phase 3-5: recursive rounds ----------------
    done_rounds = completed_rounds(target_directory) if args.resume else []
    if done_rounds:
        print(f"[PHASE 3] Rounds already complete: {done_rounds}. "
              f"Continuing from round {max(done_rounds) + 1:02d}.")

    start_round = (max(done_rounds) + 1) if done_rounds else 1
    end_round = max(args.rounds, max(done_rounds) if done_rounds else 0)

    round_stats: List[dict] = []
    dream_stats: List[dict] = []
    for r in done_rounds:
        nodes = [n for n in load_tree(target_directory, r) if n.get("task")]
        best = best_nodes_for_round(nodes)
        round_stats.append({
            "nodes": len(nodes), "spent": sum(n.get("cost", 1) for n in nodes),
            "dup_content": 0, "violations": sum(len(n.get("violations", [])) for n in nodes),
            "best_mean": round(sum(n["score"] for n in best.values()) / max(1, len(best)), 4),
        })

    if start_round > end_round:
        print(f"[PHASE 3] Bypassed. {end_round} round(s) already complete; "
              f"pass -n {end_round + 1} to run another.")
    else:
        print(f"\n[RSI] {end_round - start_round + 1} round(s) to run, "
              f"{budget} agent call(s) per round, "
              + ("dreaming DISABLED (fixed-exploration control)"
                 if args.no_dream else f"{DREAM_CANDIDATES} policy revision(s) per dream")
              + ".", flush=True)

        # A round that never finished is re-run from a clean slate.
        if args.resume:
            archive_partial_round(target_directory, start_round)

        # Resuming past the last dreamed round: the policy for start_round was
        # never written (no dream follows a run's final round in older runs, or
        # the dream itself was interrupted). Dream now instead of silently
        # redeploying pi_0.
        if (not args.no_dream and start_round > 1
                and not policy_path(target_directory, start_round).exists()):
            pool = load_pool(target_directory)
            if pool and apex_ok:
                print(f"[*] {policy_path(target_directory, start_round).name} missing; "
                      f"dreaming over the recorded pool before round {start_round:02d}.", flush=True)
                _, dstats = dream_policy_improvement(
                    target_directory, start_round - 1,
                    load_or_init_policy(target_directory, start_round - 1),
                    pool, tasks, budget)
                dream_stats.append(dstats)
            elif pool:
                print(f"    [!] Apex offline; cannot dream the missing policy for round "
                      f"{start_round:02d}. Stopping - re-run with -r once 8081 is healthy.", flush=True)
                start_round = end_round + 1

        for rnd in range(start_round, end_round + 1):
            if _shutdown_event.is_set():
                print(f"\n[!] Shutdown requested; stopping before round {rnd:02d}. "
                      "Re-run with -r to continue.", flush=True)
                break

            policy_source = (DEFAULT_POLICY_SOURCE if args.no_dream
                             else load_or_init_policy(target_directory, rnd))
            if args.no_dream:
                path = policy_path(target_directory, rnd)
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="ascii") as f:
                    f.write(DEFAULT_POLICY_SOURCE)

            nodes, stats = run_online_round(roster, rnd, target_query, target_directory,
                                            policy_source, budget,
                                            semantic_guidance=args.semantic_guidance)
            round_stats.append(stats)

            # Test telemetry is part of the world, not a postscript: it is folded
            # back onto the nodes so the next dream scores against real outcomes.
            test_results = run_phase5_automatic_unittests(target_directory, roster, nodes, rnd)
            touched = writeback_test_scores(target_directory, rnd, test_results)
            if touched:
                print(f"    [+] Folded test telemetry back onto {touched} node(s); "
                      f"future replays score against it.", flush=True)

            round_dir_for(target_directory, rnd).mkdir(parents=True, exist_ok=True)
            with open(round_done_marker(target_directory, rnd), "w", encoding="ascii") as f:
                f.write(datetime.now().isoformat(timespec="seconds") + "\n")

            if args.no_dream:
                print(f"    [i] Round {rnd:02d}: dreaming skipped; round {rnd + 1:02d} "
                      f"redeploys pi_0 unchanged.", flush=True)
                continue
            if _shutdown_event.is_set():
                break
            # Dream after EVERY round, including the last, so pi_{rnd+1} always
            # exists and a later `-r -n N+1` continues the lineage.
            if not apex_ok:
                print(f"    [!] Apex offline; cannot improve the policy after round {rnd:02d}. "
                      f"Stopping - re-run with -r (the missing policy is dreamed on resume).",
                      flush=True)
                break

            pool = load_pool(target_directory)
            _, dstats = dream_policy_improvement(target_directory, rnd, policy_source,
                                                 pool, tasks, budget)
            dream_stats.append(dstats)

    master_elapsed_time = time.time() - master_start_time

    pool = load_pool(target_directory)
    manifest = build_run_manifest(target_directory, roster, pool, round_stats,
                                  dream_stats, target_query, master_elapsed_time,
                                  plan_p, plan_c)
    with open(manifest_path, "w", encoding="ascii") as f:
        f.write(manifest)
    print(f"\n[+] Run manifest written to {manifest_path.name}", flush=True)

    # ---------------- Phase 6 ----------------
    if args.resume and distilled_tasks_path.exists() and not args.iterate:
        print(f"\n[PHASE 6] Bypassed. DISTILLED_TASKS.md already exists. Use --iterate to force re-run.")
    else:
        run_phase6_project_distillation(target_directory, iterate=args.iterate)

    print("\n==============================================================================")
    print("PIPELINE COMPLETE")
    print(f"  Deliverables: {work_dir_for(target_directory)}")
    print(f"  Discovery tree: {trees_dir_for(target_directory)}")
    print(f"  Policies: {policy_dir_for(target_directory)}")
    print(f"  Dreaming: {dream_dir_for(target_directory)}")
    print(f"  Comms log: {comms_dir_for(target_directory)}")
    print(f"  Manifest: {manifest_path.name}")
    print("==============================================================================\n")


if __name__ == "__main__":
    main()
