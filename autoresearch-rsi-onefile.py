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
from typing import Any, Tuple, List, Dict, Set, Optional, Union
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
#   * APEX (9931) runs non-worker, non-agent tasks only: Phase 1 generation,
#     Phase 2 distillation, Phase 3 decomposition (planning), Phase 6
#     distillation. It is -np 1 and therefore strictly serial.
#   * WORKERS (8030-8035) run every agent assignment, Phase 0 repo map-reduce
#     and the evaluator's unit-test generation. The former stitcher nodes are
#     folded into this pool.
#   * There is NO stitcher tier. Consolidation is mechanical (filesystem walk +
#     hashing), never a model merge.
#
# Every worker node runs -c 196608 -np 2, i.e. a 98304-token window per slot.
# If the pool ever becomes HETEROGENEOUS in -c, set WORKER_SERVER_CTX to the
# SMALLEST node's value, or the larger nodes' budgets will over-subscribe it.
# ==============================================================================

# Apex / planning / generation / distillation node (port 9931): -c 65536 -np 1
APEX_SERVER_CTX = int(os.getenv("APEX_SERVER_CTX", "65536"))
APEX_SERVER_NP = int(os.getenv("APEX_SERVER_NP", "1"))

# Agent worker cluster (8030-8035): -np 2 --kv-unified each.
# Budget against the smallest -c in the pool.
WORKER_SERVER_CTX = int(os.getenv("WORKER_SERVER_CTX", "196608"))
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
        "http://localhost:9931/v1"
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
# Recursive self-improvement at the exploration layer (Dream-RSI; Zheng et al.,
# "Dream-RSI: Recursive Self-Improvement through Evolving Worlds",
# arXiv:2609.14858, Sec. 3).
#
# The exploration policy acts in DECISION ROUNDS. In each round it selects one
# batch C of at most W legal continuations (W = worker slots), observes their
# evaluated outcomes, and decides again; an empty batch ends the rollout. An
# online rollout allows at most K1 rounds, a replay at most K2.
#
# ROUND_BUDGET is the per-round resource cap in discovery-agent calls (retries
# charged). It is identical online and in replay, matching the paper's equal
# per-round budgets for Dream-RSI and Recursive Fixed Exploration.
# ------------------------------------------------------------------------------
# Default 2: dreaming after round 1 needs a round 2 to deploy into, and the
# pool needs more than one recorded tree before replay scores mean much.
DEFAULT_ROUNDS = int(os.getenv("RSI_ROUNDS", "2"))
MAX_ROUNDS = int(os.getenv("RSI_MAX_ROUNDS", "12"))
ROUND_BUDGET_PER_TASK = float(os.getenv("ROUND_BUDGET_PER_TASK", "2.0"))
ROUND_BUDGET_MIN = int(os.getenv("ROUND_BUDGET_MIN", "3"))
ROUND_BUDGET_MAX = int(os.getenv("ROUND_BUDGET_MAX", "60"))

# W: parallel workers, i.e. the maximum batch size of one decision round.
MAX_PARALLELISM = max(1, int(os.getenv(
    "MAX_PARALLELISM", str(max(1, len(WORKER_ENDPOINTS) * WORKER_PARALLEL_SLOTS)))))
# K1 / K2: maximum decision rounds per online rollout / per replay.
ONLINE_MAX_DECISION_ROUNDS = max(1, int(os.getenv("ONLINE_MAX_DECISION_ROUNDS", "32")))
REPLAY_MAX_DECISION_ROUNDS = max(1, int(os.getenv(
    "REPLAY_MAX_DECISION_ROUNDS", str(ONLINE_MAX_DECISION_ROUNDS))))

# M - 1: policy revisions per offline phase. Version m+1 is derived from version
# m; the deployed policy is version 0, so M = DREAM_CANDIDATES + 1 versions are
# evaluated and the argmax is deployed next.
DREAM_CANDIDATES = int(os.getenv("DREAM_CANDIDATES", "3"))

# Replay objective, Eq. (1):
#   V_i = quality_i - beta_1 * N_i + beta_2 * N_i / max(1, k_i*)
# quality_i: best revealed score per assignment, averaged over the roster (an
# unreached assignment contributes the root's score, 0). N_i: revealed non-root
# nodes. k_i*: completed decision rounds. The policy's score is the mean of V_i
# over every recorded tree.
DREAM_BETA1 = float(os.getenv("DREAM_BETA1", "0.002"))
DREAM_BETA2 = float(os.getenv("DREAM_BETA2", "0.004"))
# Characters of per-round replay trace shown to the policy-development agent.
DREAM_TRACE_CHARS = int(os.getenv("DREAM_TRACE_CHARS", "6000"))
# Headroom gate. Before any apex revision, the replay oracle bound (the best V
# ANY policy could reach on the recorded pool, with hindsight) is compared with
# the deployed policy's replay V. If the gap is below this, no revision can win
# by a meaningful margin, so the M-1 apex calls are skipped and the deployed
# policy carries forward. The bound is non-causal, so it overstates what a real
# policy can reach: a gap just above epsilon is still usually unwinnable.
DREAM_MIN_HEADROOM = float(os.getenv("DREAM_MIN_HEADROOM", "0.01"))
# Set by --force-dream: run the revisions even when the gate says no headroom.
DREAM_FORCE_REVISIONS = False
# pi_0 branching: share of assignments (weakest first-attempt scores) that get a
# second independent root, so recorded trees hold real alternatives (sibling vs
# continuation) instead of pure chains that only teach stop/continue.
PI0_BRANCH_FRACTION = min(1.0, max(0.0, float(os.getenv("PI0_BRANCH_FRACTION", "0.34"))))

# Policy sandbox limits.
POLICY_MAX_CHARS = int(os.getenv("POLICY_MAX_CHARS", "20000"))
# Cap on ALL interface calls (reads included), so a loop that only polls
# budget_left() or keeps submitting illegal batches still terminates promptly.
POLICY_MAX_RPC = int(os.getenv("POLICY_MAX_RPC", "4000"))

# Evaluator. Fixed and applied ONCE per attempt, at creation (Sec. 3: a fixed
# evaluator scores each candidate and returns diagnostic feedback). The stored
# score is final; replay reveals exactly what the online policy observed.
EVAL_W_STATUS = float(os.getenv("EVAL_W_STATUS", "0.35"))
EVAL_W_FILES = float(os.getenv("EVAL_W_FILES", "0.25"))
EVAL_W_LOG = float(os.getenv("EVAL_W_LOG", "0.15"))
EVAL_W_NOVELTY = float(os.getenv("EVAL_W_NOVELTY", "0.25"))
EVAL_W_VIOLATION = float(os.getenv("EVAL_W_VIOLATION", "0.20"))
EVAL_W_TRUNCATED = float(os.getenv("EVAL_W_TRUNCATED", "0.15"))
# Mix of heuristic vs. unit-test pass rate in the evaluator score.
EVAL_HEURISTIC_MIX = float(os.getenv("EVAL_HEURISTIC_MIX", "0.6"))
# Pass-rate term for an attempt that has deliverables but none that could be
# tested (non-testable file types, test generation failed, or tests disabled).
EVAL_UNTESTED_PRIOR = float(os.getenv("EVAL_UNTESTED_PRIOR", "0.5"))
# Status credit for an attempt that hit its output limit but still yielded
# complete, closed <file> blocks (salvaged).
EVAL_PARTIAL_STATUS_FRAC = float(os.getenv("EVAL_PARTIAL_STATUS_FRAC", "0.5"))
# Generate and run unit tests for every attempt as part of its evaluation.
EVAL_INLINE_TESTS = os.getenv("EVAL_INLINE_TESTS", "1") == "1"
EVAL_MAX_TEST_FILES = max(1, int(os.getenv("EVAL_MAX_TEST_FILES", "6")))

# Evaluator part 3: project-level integration. Each attempt is dropped into a
# flat project made of the best known deliverable of every OTHER task, and the
# assembled project is checked as a whole (compile, import, pytest over every
# test file, optional integration command). This is still applied once, at
# creation, so the evaluator stays fixed and replay reveals exactly what the
# live policy saw.
EVAL_INTEGRATION = os.getenv("EVAL_INTEGRATION", "1") == "1"
EVAL_INTEGRATION_MIX = min(1.0, max(0.0, float(os.getenv("EVAL_INTEGRATION_MIX", "0.5"))))
# Credit assignment. The integration part of a node's score also reflects what
# THIS attempt changed: q(project with it) - q(project with its task's previous
# best), both over the whole project. First and test-only attempts: neutral.
# One shared bug otherwise gives every node the same q and the tree is flat.
EVAL_CREDIT = os.getenv("EVAL_CREDIT", "1") == "1"
EVAL_CREDIT_MIX = min(1.0, max(0.0, float(os.getenv("EVAL_CREDIT_MIX", "0.5"))))
EVAL_CREDIT_GAIN = float(os.getenv("EVAL_CREDIT_GAIN", "2.0"))
INTEGRATION_IMPORT_SECS = int(os.getenv("INTEGRATION_IMPORT_SECS", "30"))
INTEGRATION_PYTEST_SECS = int(os.getenv("INTEGRATION_PYTEST_SECS", "180"))
INTEGRATION_CMD = os.getenv("INTEGRATION_CMD", "")
INTEGRATION_CMD_SECS = int(os.getenv("INTEGRATION_CMD_SECS", "300"))
INTEGRATION_DIRNAME = "integration"

# Sections of the ORIGINAL brief passed verbatim to every agent and to the test
# generator. They bypass Phase 1/2 rewriting, which is where interfaces drift.
PINNED_SECTIONS = [x.strip().upper() for x in
                   os.getenv("PINNED_SECTIONS", "INTERFACES,CONSTRAINTS,ACCEPTANCE CRITERIA").split(",")
                   if x.strip()]
AGENT_CONTRACT_BUDGET = int(os.getenv("AGENT_CONTRACT_BUDGET",
                                      str(min(16000, int(WORKER_INPUT_CHARS * 0.18)))))
TEST_CONTRACT_BUDGET = int(os.getenv("TEST_CONTRACT_BUDGET", "8000"))

# Contract synthesis. A brief with no pinned section (a short prose prompt) gets
# a contract written by the apex from the USER'S TEXT ONLY - never from the
# Phase-1 draft, which otherwise becomes the de-facto spec. Saved to CONTRACT.md
# and marked synthesized.
SYNTHESIZE_CONTRACT = os.getenv("SYNTHESIZE_CONTRACT", "1") == "1"
# The original prompt, verbatim, in every agent's input (carved out of the
# background budget, so the total window is unchanged).
AGENT_BRIEF_BUDGET = int(os.getenv("AGENT_BRIEF_BUDGET", str(min(12000, int(WORKER_INPUT_CHARS * 0.12)))))

# The container defines the workload's abilities. Nothing about them is
# configured: every installed distribution and importable module of the
# evaluation interpreter is DISCOVERED (metadata + module path scan, no mass
# imports), shown to every agent, and rescanned at the start of every round so
# packages the container's owner installs mid-run become usable immediately
# (and removed ones stop being allowed). The gate rejects imports of anything
# the container does not have.
ENFORCE_DEPENDENCIES = os.getenv("ENFORCE_DEPENDENCIES", "1") == "1"
ENV_RESCAN_EACH_ROUND = os.getenv("ENV_RESCAN_EACH_ROUND", "1") == "1"
ENV_SECTION_BUDGET = int(os.getenv("ENV_SECTION_BUDGET", "6000"))
EVAL_DEP_REJECT_SCORE = float(os.getenv("EVAL_DEP_REJECT_SCORE", "0.0"))
# Packaging machinery is how abilities get installed, not an ability itself.
_TOOLING_DISTS = {"pip", "setuptools", "wheel", "distribute", "pkg-resources", "pkg_resources"}
# Library API facts are discovered too: modules the brief mentions and every
# third-party name the integrated project actually uses are inspected in the
# evaluation interpreter each round; references that do not exist are listed.
API_FACTS_BUDGET = int(os.getenv("API_FACTS_BUDGET", "6000"))
API_FACTS_MAX_MODULE_NAMES = int(os.getenv("API_FACTS_MAX_MODULE_NAMES", "150"))
QRACK_LIB_PATH = os.getenv("QRACK_LIB_PATH", "/usr/local/lib/qrack/libqrack_pinvoke.so")

# Interfaces. A contract without an INTERFACES section gets one chosen by the
# planner (labelled as such), plus a RUN command for the project's entry point.
SYNTHESIZE_INTERFACES = os.getenv("SYNTHESIZE_INTERFACES", "1") == "1"
# After every round the integrated project's public API is extracted (AST) and
# shown to the next round as CURRENT INTERFACES; removing or breaking a name a
# sibling uses is recorded as a violation.
FREEZE_INTERFACES = os.getenv("FREEZE_INTERFACES", "1") == "1"
# After the final round's grounding run, one extra call (outside the RSI budget
# and the recorded trees) lets the write-up owner rewrite the .md deliverables
# against that final output; the result replaces them in integration/latest/.
FINAL_WRITEUP_REFRESH = os.getenv("FINAL_WRITEUP_REFRESH", "1") == "1"
# Sensitivity probe: a control command (planner-chosen, or PROBE_COMMAND) whose
# change MUST move the reported metrics. Run after every successful grounding
# run; identical numbers mean the metrics do not measure anything.
PROBE_COMMAND = os.getenv("PROBE_COMMAND", "")
# Numeric keys ignored when comparing run and probe output (timing, seeds...).
PROBE_IGNORE_KEYS = re.compile(os.getenv("PROBE_IGNORE_KEYS",
                                         r"(time|elapsed|runtime|seconds|duration|timestamp|seed|pid)"),
                               re.I)
# Skeptic review: one apex call after the final run that reads the code and the
# output and judges whether each reported metric measures what its name says.
FINAL_SKEPTIC_REVIEW = os.getenv("FINAL_SKEPTIC_REVIEW", "1") == "1"
REVIEW_CODE_CHARS = int(os.getenv("REVIEW_CODE_CHARS", "36000"))
AGENT_API_BUDGET = int(os.getenv("AGENT_API_BUDGET", "8000"))
# Grounding run. After every round the RUN command executes in a copy of the
# integrated project; its real output (or traceback) goes to every agent, and a
# write-up may only report what it shows.
RUN_COMMAND = os.getenv("RUN_COMMAND", "")          # overrides the contract's RUN
RUN_COMMAND_SECS = int(os.getenv("RUN_COMMAND_SECS", "300"))
AGENT_RUN_BUDGET = int(os.getenv("AGENT_RUN_BUDGET", "5000"))
_ENTRYPOINT_NAMES = ("run_experiment.py", "runner.py", "run.py", "main.py", "cli.py")

# Off-policy support probes. EXTENSION, not part of Dream-RSI: a fraction of
# each live round is held back from the deployed policy and spent on
# deterministic refine-best / open-new-root continuations, widening the support
# of the recorded pool. Off (0) by default to follow the paper; if enabled it is
# applied in both arms (dream and --no-dream) so budgets stay equal.
SUPPORT_PROBE_FRAC = float(os.getenv("SUPPORT_PROBE_FRAC", "0.0"))

# Policy isolation. Policies run in a child process that talks to the explorer
# over a line-JSON RPC; these are that child's resource limits.
POLICY_CPU_SECS = int(os.getenv("POLICY_CPU_SECS", "20"))
POLICY_MEM_MB = int(os.getenv("POLICY_MEM_MB", "512"))
POLICY_REPLAY_WALL_SECS = float(os.getenv("POLICY_REPLAY_WALL_SECS", "60"))

# Test execution hardening (evaluator).
TEST_PIP_INSTALL = os.getenv("TEST_PIP_INSTALL", "1") == "1"
TEST_PIP_ALLOWLIST = {p.strip().lower().replace("_", "-")
                      for p in os.getenv("TEST_PIP_ALLOWLIST", "").split(",") if p.strip()}
TEST_CPU_SECS = int(os.getenv("TEST_CPU_SECS", "60"))
TEST_MEM_MB = int(os.getenv("TEST_MEM_MB", "4096"))
TEST_FSIZE_MB = int(os.getenv("TEST_FSIZE_MB", "128"))

# Phase 0: permit LAN git hosts (e.g. a self-hosted Gitea). Off = fail closed.
GIT_ALLOW_PRIVATE_HOSTS = os.getenv("GIT_ALLOW_PRIVATE_HOSTS", "0") == "1"

# Evaluator unit-test generation (runs on the agent worker pool)
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
    "2. If your work depends on a teammate's deliverable, DO NOT rebuild it. Import it by the module "
    "name the PINNED CONTRACT gives and state the dependency in your log.\n"
    "3. If you believe a teammate's deliverable is wrong or missing, do not fix it yourself. "
    "Raise it with a note to that agent.\n"
    "4. Broader context is given for orientation only. It is not a licence to widen your scope.\n"
    "\n"
    "INTEGRATION (how your files are used):\n"
    "1. After every attempt, the best file of each agent is copied into ONE shared project root with "
    "the agent directory prefix removed: a file you emit as `pkg/mod.py` lands at `<project>/pkg/mod.py`.\n"
    "2. Import teammates' modules by bare module name (Python: `from module_name import name`). Never "
    "import through work/ or agent directory names and never modify sys.path.\n"
    "3. The PINNED CONTRACT, when shown, is binding: file names, function names, signatures, return "
    "types and data formats must match it exactly. It overrides anything else you are shown.\n"
    "4. Code files contain only code and comments. Never leave reasoning, drafts or self-corrections in a file.\n"
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
    "The policy is executable Python. One decision round at a time, it selects a batch of "
    "legal continuations to run in parallel (at most W per round) and decides when to stop "
    "by submitting no further batch. It is evaluated by replaying it over recorded discovery "
    "trees: every outcome it can reveal is already on disk, so evaluation costs no agent "
    "calls, and only continuations that history recorded are legal in replay.\n"
    "\n"
    "You are given the CURRENT policy version, its replay scores, its per-round execution "
    "traces, and the scores of earlier versions. Examine the traces to identify successful "
    "decisions and recurring failures - serial or undersized batches, premature stops, "
    "over-pruning, calls wasted on lines that stopped improving, abandoned branches that "
    "were repairable, assignments never reached - and revise the code to produce the next "
    "version.\n"
    "\n"
    "Every decision must be derived from what the policy itself has revealed through ctx. "
    "Never hard-code node ids, round numbers, scores or targets seen in the traces.\n"
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
    "5. If C/C++, #include the provided filename directly and write your own main().\n"
    "6. Import the module under test and any teammate module by bare module name; they are all on the path.\n"
    "7. When a CONTRACT is given, assert what the contract specifies. Do not assert conventions, "
    "formats or edge cases that neither the contract nor the file's docstrings define.\n"
    "8. Tests must be deterministic and finish within 30 seconds."
)

# ==============================================================================
# Global Utilities & State
# ==============================================================================

_active_clone_dirs: Set[Path] = set()
_clone_dirs_lock = threading.Lock()
_events_lock = threading.Lock()
_tree_lock = threading.Lock()
_shutdown_event = threading.Event()

# Run-scoped, set once in main() before any agent runs; read-only afterwards.
_RUN_CONTRACT: str = ""
_RUN_INTEGRATION_CMD: str = INTEGRATION_CMD
_RUN_DELIVERABLES: List[str] = []
_RUN_BRIEF: str = ""                              # the user's prompt, verbatim
_RUN_CONTRACT_SYNTHESIZED: bool = False
_RUN_ALLOWED_IMPORTS: Optional[List[str]] = None  # None = gate off; else discovered imports
_RUN_ENV: Dict[str, Any] = {}                      # latest container scan
_RUN_ENV_TEXT: str = ""                            # rendered for agents and tests
_RUN_API_FACTS_TEXT: str = ""                      # discovered library API facts
_ENV_RESCAN_LOCK = threading.Lock()
_RUN_COMMAND: str = ""                             # grounding run command
_RUN_PROBE_COMMAND: str = ""                       # sensitivity control command
_RUN_DIR: Optional[Path] = None                    # set in main(); for diagnostics
_RUN_FROZEN_API: Dict[str, dict] = {}              # module -> name -> {kind, sig, params, used_by}
_RUN_FROZEN_API_TEXT: str = ""
_RUN_LAST_RUN_TEXT: str = ""


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


# ------------------------------------------------------------------
# Token ledger: every model call is recorded (category, tier, prompt and
# completion tokens, wall time, time to first token, generation rate).
# Usage comes from the server when it reports it; otherwise it is estimated
# from characters (CHARS_PER_TOKEN) and marked as such. Persisted to
# tokens.jsonl so a resumed run's summary covers the whole run.
# ------------------------------------------------------------------
_TOKEN_CTX = threading.local()

# Apex calls are categorised by the function that makes them.
_CATEGORY_BY_CALLER = {
    "generate_content": "draft (phase 1)",
    "distill_document": "distil (phase 2)",
    "synthesize_contract": "contract synthesis",
    "synthesize_interfaces": "interface synthesis",
    "decompose_to_atomic_pieces": "partition",
    "dream_policy_improvement": "dream revisions",
    "final_skeptic_review": "skeptic review",
    "run_phase6_project_distillation": "project distillation (phase 6)",
}


class token_category:
    """with token_category("write-up refresh"): ... - overrides the category of
    calls made on this thread inside the block."""
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.prev = getattr(_TOKEN_CTX, "cat", None)
        _TOKEN_CTX.cat = self.name
        return self

    def __exit__(self, *exc):
        _TOKEN_CTX.cat = self.prev
        return False


def _caller_category(default: str) -> str:
    cat = getattr(_TOKEN_CTX, "cat", None)
    if cat:
        return cat
    f = sys._getframe(2)
    for _ in range(4):
        if f is None:
            break
        name = f.f_code.co_name
        if name in _CATEGORY_BY_CALLER:
            return _CATEGORY_BY_CALLER[name]
        f = f.f_back
    return default


class TokenLedger:
    def __init__(self):
        self.lock = threading.Lock()
        self.records: List[dict] = []
        self.path: Optional[Path] = None
        self.session = time.strftime("%Y%m%d_%H%M%S")

    def bind(self, run_dir: Path) -> None:
        self.path = run_dir / "tokens.jsonl"
        if self.path.exists():
            for line in (read_file_content_safe(self.path) or "").splitlines():
                try:
                    self.records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    def add(self, category: str, tier: str, prompt_tokens: int, completion_tokens: int,
            secs: float, ttft: Optional[float] = None, estimated: bool = False,
            rnd: Optional[int] = None, truncated: bool = False) -> dict:
        if rnd is None:
            rnd = _CURRENT_ROUND
        gen_secs = (secs - ttft) if (ttft is not None and secs > ttft) else secs
        rec = {"t": round(time.time(), 2), "session": self.session, "category": category, "tier": tier,
               "round": rnd, "prompt": int(prompt_tokens or 0), "completion": int(completion_tokens or 0),
               "secs": round(secs, 3), "ttft": None if ttft is None else round(ttft, 3),
               "gen_tps": round((completion_tokens or 0) / gen_secs, 2) if gen_secs > 0.05 else None,
               "estimated": bool(estimated), "truncated": bool(truncated)}
        with self.lock:
            self.records.append(rec)
            if self.path is not None:
                try:
                    with open(self.path, "a", encoding="ascii") as f:
                        f.write(json.dumps(rec) + "\n")
                except OSError:
                    pass
        return rec

    def select(self, **match) -> List[dict]:
        with self.lock:
            return [r for r in self.records if all(r.get(k) == v for k, v in match.items())]

    def totals(self, recs: Optional[List[dict]] = None) -> dict:
        recs = self.select() if recs is None else recs
        tps = sorted(r["gen_tps"] for r in recs if r.get("gen_tps"))
        ttfts = [r["ttft"] for r in recs if r.get("ttft") is not None]
        p = sum(r["prompt"] for r in recs)
        c = sum(r["completion"] for r in recs)
        secs = sum(r["secs"] for r in recs)
        gen_secs = sum(max(0.0, r["secs"] - (r.get("ttft") or 0.0)) for r in recs)
        # busy span: wall time during which at least one of these calls was running
        spans = sorted((r["t"] - r["secs"], r["t"]) for r in recs)
        busy, cur_s, cur_e = 0.0, None, None
        for a, b in spans:
            if cur_e is None or a > cur_e:
                if cur_e is not None:
                    busy += cur_e - cur_s
                cur_s, cur_e = a, b
            else:
                cur_e = max(cur_e, b)
        if cur_e is not None:
            busy += cur_e - cur_s
        return {"calls": len(recs), "prompt": p, "completion": c, "total": p + c,
                "secs": secs, "busy_secs": busy,
                # completion tokens per second of generation, weighted by tokens
                "avg_gen_tps": round(c / gen_secs, 2) if gen_secs > 0.05 else None,
                # all tokens (in + out) per second of call time
                "avg_total_tps": round((p + c) / secs, 2) if secs > 0.05 else None,
                # completion tokens per second of wall time with calls in flight (parallelism included)
                "throughput_tps": round(c / busy, 2) if busy > 0.05 else None,
                "median_tps": tps[len(tps) // 2] if tps else None,
                "mean_ttft": (sum(ttfts) / len(ttfts)) if ttfts else None,
                "estimated": sum(1 for r in recs if r.get("estimated")),
                "truncated": sum(1 for r in recs if r.get("truncated"))}

    def by(self, key: str, recs: Optional[List[dict]] = None) -> Dict[str, dict]:
        recs = self.select() if recs is None else recs
        groups: Dict[str, List[dict]] = {}
        for r in recs:
            groups.setdefault(str(r.get(key)), []).append(r)
        return {k: self.totals(v) for k, v in groups.items()}


_LEDGER = TokenLedger()
_CURRENT_ROUND: Optional[int] = None     # set by the round loop; tags calls made during it
_RUN_RUNTIME_SECS: Optional[float] = None  # wall-clock runtime of this invocation, set at the end


def _fmt_tok(n: Optional[float]) -> str:
    if n is None:
        return "-"
    n = float(n)
    return f"{n / 1e6:.2f}M" if n >= 1e6 else (f"{n / 1e3:.1f}k" if n >= 1e3 else f"{n:.0f}")


def _fmt_secs(x: float) -> str:
    return f"{x / 3600:.1f}h" if x >= 3600 else (f"{x / 60:.1f}min" if x >= 90 else f"{x:.0f}s")


def estimate_remaining_tokens(rounds_left: int, calls_per_round: int, n_tasks: int,
                              include_final: bool = True) -> dict:
    """Projection for the rest of the run: expected calls per category times the
    mean tokens per call measured so far (a prior from the configured budgets
    where nothing has been measured yet), and wall time at the measured rate."""
    def per_call(cat: str, prior_p: float, prior_c: float) -> Tuple[float, float, bool]:
        recs = _LEDGER.select(category=cat)
        if recs:
            return (sum(r["prompt"] for r in recs) / len(recs),
                    sum(r["completion"] for r in recs) / len(recs), True)
        return prior_p, prior_c, False
    cpt = CHARS_PER_TOKEN
    plan = [
        ("agent attempts", rounds_left * calls_per_round,
         0.55 * WORKER_INPUT_CHARS / cpt, 0.5 * MAX_WORKER_TOKENS),
        ("unit-test generation", rounds_left * calls_per_round,
         (TEST_CONTRACT_BUDGET + 9000) / cpt, 0.5 * MAX_OUTPUT_TOKENS),
        ("dream revisions", rounds_left * DREAM_CANDIDATES,
         0.5 * MAX_CONTEXT_CHARS / cpt, 0.6 * APEX_PLAN_TOKENS),
    ]
    if include_final:
        plan += [("skeptic review", 1, (REVIEW_CODE_CHARS + 16000) / cpt, 0.4 * APEX_PLAN_TOKENS),
                 ("write-up refresh", 1, 0.55 * WORKER_INPUT_CHARS / cpt, 0.4 * MAX_WORKER_TOKENS),
                 ("project distillation (phase 6)", 1, 0.5 * MAX_CONTEXT_CHARS / cpt, 0.5 * APEX_PLAN_TOKENS)]
    rows, tot_p, tot_c, measured_all = [], 0.0, 0.0, True
    for cat, calls, pp, pc in plan:
        mp, mc, measured = per_call(cat, pp, pc)
        measured_all &= measured or calls == 0
        rows.append({"category": cat, "calls": calls, "prompt": mp * calls, "completion": mc * calls,
                     "measured": measured})
        tot_p += mp * calls
        tot_c += mc * calls
    agent_tps = _LEDGER.totals(_LEDGER.select(tier="agent"))["median_tps"]
    apex_tps = _LEDGER.totals(_LEDGER.select(tier="apex"))["median_tps"]
    agent_par = max(1, min(MAX_PARALLELISM, len(WORKER_ENDPOINTS) * WORKER_PARALLEL_SLOTS))
    secs = 0.0
    for r in rows:
        on_apex = r["category"] in ("dream revisions", "skeptic review", "project distillation (phase 6)")
        tier_tps = apex_tps if on_apex else agent_tps
        if tier_tps:
            secs += r["completion"] / tier_tps / (1 if on_apex else agent_par)
    return {"rows": rows, "prompt": tot_p, "completion": tot_c, "total": tot_p + tot_c,
            "gen_secs": secs if (agent_tps or apex_tps) else None, "all_measured": measured_all}


def print_token_estimate(label: str, est: dict) -> None:
    parts = ", ".join(f"{r['category']} {r['calls']}x~{_fmt_tok((r['prompt'] + r['completion']) / max(1, r['calls']))}"
                      for r in est["rows"] if r["calls"])
    t = est.get("gen_secs")
    print(f"[TOKENS] {label}: ~{_fmt_tok(est['total'])} tok (in ~{_fmt_tok(est['prompt'])}, "
          f"out ~{_fmt_tok(est['completion'])})"
          + (f", ~{_fmt_secs(t)} of generation at measured rates" if t else "")
          + (" [priors from configured budgets until measured]" if not est["all_measured"] else ""),
          flush=True)
    if parts:
        print(f"    {parts}", flush=True)


def print_round_tokens(rnd: int) -> None:
    rt = _LEDGER.totals(_LEDGER.select(round=rnd))
    ag = _LEDGER.totals(_LEDGER.select(round=rnd, tier="agent"))
    allt = _LEDGER.totals()
    ar = _LEDGER.select(round=rnd, tier="agent")
    span = (max(r["t"] for r in ar) - min(r["t"] - r["secs"] for r in ar)) if ar else 0
    agg = (ag["completion"] / span) if span > 1 else None
    print(f"[TOKENS] round {rnd:02d}: {rt['calls']} call(s), in {_fmt_tok(rt['prompt'])}, "
          f"out {_fmt_tok(rt['completion'])} | agent gen {ag['median_tps'] or '-'} tok/s median per stream"
          + (f", {agg:.1f} tok/s aggregate over {_fmt_secs(span)}" if agg else "")
          + (f", ttft {ag['mean_ttft']:.1f}s" if ag["mean_ttft"] is not None else "")
          + f" | run so far {_fmt_tok(allt['total'])}", flush=True)


def token_summary_markdown() -> str:
    allt = _LEDGER.totals()
    rt = _RUN_RUNTIME_SECS
    lines = ["# Token usage", "",
             "## Totals", "",
             "| measure | value |", "|---|---:|",
             f"| model calls | {allt['calls']:,} |",
             f"| prompt tokens | {allt['prompt']:,} |",
             f"| completion tokens | {allt['completion']:,} |",
             f"| total tokens | {allt['total']:,} |",
             f"| pipeline runtime (wall clock, this invocation) | "
             f"{_fmt_secs(rt) + ' (' + format(rt, ',.0f') + ' s)' if rt else '-'} |",
             f"| time with model calls in flight | {_fmt_secs(allt['busy_secs'])} |",
             f"| summed call time (parallel calls counted separately) | {_fmt_secs(allt['secs'])} |",
             f"| average generation rate (completion tok / generation time, token-weighted) | "
             f"{allt['avg_gen_tps'] or '-'} tok/s |",
             f"| median generation rate per call | {allt['median_tps'] or '-'} tok/s |",
             f"| average total rate (prompt + completion tok / call time) | {allt['avg_total_tps'] or '-'} tok/s |",
             f"| throughput (completion tok / time with calls in flight) | {allt['throughput_tps'] or '-'} tok/s |",
             f"| throughput over the whole runtime | "
             f"{round(allt['completion'] / rt, 2) if rt else '-'} tok/s out, "
             f"{round(allt['total'] / rt, 2) if rt else '-'} tok/s in+out |",
             "",
             (f"{allt['estimated']} call(s) had no server usage and were estimated from characters "
              f"({CHARS_PER_TOKEN} chars/token)." if allt["estimated"] else "All counts are server-reported."),
             "",
             "## By category", "",
             "| category | tier | calls | prompt | completion | total | share | per call | out/in "
             "| avg gen tok/s | median gen tok/s | call time | mean ttft | est. | cut off |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    cats: Dict[str, List[dict]] = {}
    for r in _LEDGER.select():
        cats.setdefault(r["category"], []).append(r)
    for cat, recs in sorted(cats.items(), key=lambda kv: -sum(r["prompt"] + r["completion"] for r in kv[1])):
        t = _LEDGER.totals(recs)
        tiers = "/".join(sorted({r["tier"] for r in recs}))
        share = 100.0 * t["total"] / max(1, allt["total"])
        ratio = t["completion"] / max(1, t["prompt"])
        lines.append(f"| {cat} | {tiers} | {t['calls']} | {t['prompt']:,} | {t['completion']:,} | "
                     f"{t['total']:,} | {share:.1f}% | {_fmt_tok(t['total'] / max(1, t['calls']))} | "
                     f"{ratio:.2f} | {t['avg_gen_tps'] or '-'} | {t['median_tps'] or '-'} | "
                     f"{_fmt_secs(t['secs'])} | "
                     f"{'-' if t['mean_ttft'] is None else str(round(t['mean_ttft'], 1)) + 's'} | "
                     f"{t['estimated']} | {t['truncated']} |")
    lines += ["", "## By tier", "",
              "| tier | calls | prompt | completion | avg gen tok/s | median gen tok/s | throughput tok/s "
              "| in flight | mean ttft |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tier, t in sorted(_LEDGER.by("tier").items()):
        lines.append(f"| {tier} | {t['calls']} | {t['prompt']:,} | {t['completion']:,} | "
                     f"{t['avg_gen_tps'] or '-'} | {t['median_tps'] or '-'} | {t['throughput_tps'] or '-'} | "
                     f"{_fmt_secs(t['busy_secs'])} | "
                     f"{'-' if t['mean_ttft'] is None else round(t['mean_ttft'], 1)} |")
    lines += ["", "## By round", "",
              "| round | calls | prompt | completion | total | in flight | avg gen tok/s | throughput tok/s |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for rnd, t in sorted(_LEDGER.by("round").items(), key=lambda kv: (kv[0] == "None", kv[0])):
        lines.append(f"| {'setup / final' if rnd == 'None' else rnd} | {t['calls']} | {t['prompt']:,} | "
                     f"{t['completion']:,} | {t['total']:,} | {_fmt_secs(t['busy_secs'])} | "
                     f"{t['avg_gen_tps'] or '-'} | {t['throughput_tps'] or '-'} |")
    notes = []
    top = max(cats, key=lambda c: sum(r["prompt"] + r["completion"] for r in cats[c])) if cats else None
    if top:
        tt = _LEDGER.totals(cats[top])
        notes.append(f"Largest consumer: {top} ({100.0 * tt['total'] / max(1, allt['total']):.0f}% of all tokens).")
    ag = _LEDGER.totals(_LEDGER.select(category="agent attempts"))
    if ag["calls"]:
        notes.append(f"Agent attempts read {ag['prompt'] / max(1, ag['completion']):.1f} prompt tokens per "
                     f"generated token; prompt context is the lever for agent cost.")
        if ag["truncated"]:
            notes.append(f"{ag['truncated']} agent attempt(s) hit the output cap or wall clock.")
    if allt["estimated"]:
        notes.append("Estimated rows depend on CHARS_PER_TOKEN; enable usage reporting on the server for exact counts.")
    tt = [r["ttft"] for r in _LEDGER.select() if r.get("ttft") is not None]
    if tt and sorted(tt)[len(tt) // 2] < 0.02:
        notes.append("Median time to first token is under 20 ms: the endpoint (e.g. a gateway) sends its first "
                     "chunk before generating, so ttft is not meaningful and generation rates include prefill.")
    notes.append("avg gen tok/s is token-weighted (total completion tokens / total generation time); median is "
                 "per call. Throughput divides by wall time with calls in flight, so parallel slots count.")
    if notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


def print_token_summary() -> None:
    allt = _LEDGER.totals()
    rt = _RUN_RUNTIME_SECS
    print(f"\n[TOKENS] RUN SUMMARY: {allt['calls']} call(s), {allt['total']:,} tokens "
          f"(in {allt['prompt']:,} / out {allt['completion']:,})", flush=True)
    print(f"    runtime {_fmt_secs(rt) if rt else '-'} wall clock, model calls in flight "
          f"{_fmt_secs(allt['busy_secs'])} ({_fmt_secs(allt['secs'])} summed over parallel calls)", flush=True)
    print(f"    average gen {allt['avg_gen_tps'] or '-'} tok/s (token-weighted), median {allt['median_tps'] or '-'} "
          f"tok/s per call, throughput {allt['throughput_tps'] or '-'} tok/s out while busy"
          + (f", {allt['completion'] / rt:.1f} tok/s out / {allt['total'] / rt:.1f} tok/s in+out over the runtime"
             if rt else ""), flush=True)
    cats = _LEDGER.by("category")
    width = max([len(c) for c in cats] + [8])
    print(f"    {'category':<{width}}  {'calls':>5}  {'in':>8}  {'out':>8}  {'share':>6}  {'avg t/s':>8}  "
          f"{'time':>7}", flush=True)
    for cat, t in sorted(cats.items(), key=lambda kv: -kv[1]["total"]):
        print(f"    {cat:<{width}}  {t['calls']:>5}  {_fmt_tok(t['prompt']):>8}  {_fmt_tok(t['completion']):>8}  "
              f"{100.0 * t['total'] / max(1, allt['total']):>5.1f}%  {str(t['avg_gen_tps'] or '-'):>8}  "
              f"{_fmt_secs(t['secs']):>7}", flush=True)
    print(f"    {'TOTAL':<{width}}  {allt['calls']:>5}  {_fmt_tok(allt['prompt']):>8}  "
          f"{_fmt_tok(allt['completion']):>8}  {'100.0%':>6}  {str(allt['avg_gen_tps'] or '-'):>8}  "
          f"{_fmt_secs(allt['secs']):>7}", flush=True)


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

# ------------------------------------------------------------------
# HTTP 400 diagnostics + recovery for non-llama-server OpenAI backends
# (vLLM, LiteLLM, llama-cpp-python, SGLang). The SDK error text is otherwise
# truncated to 160 chars inside agent violations and the cause is lost.
# ------------------------------------------------------------------
_BAD_REQ_SEEN: set = set()
_BAD_REQ_LOCK = threading.Lock()
_DROPPABLE_PARAMS = ("stream_options", "frequency_penalty", "presence_penalty", "top_p")
_PENALTY_PARAMS = ("frequency_penalty", "presence_penalty")
# Params a given endpoint has rejected once; stripped up front on later calls so
# every request doesn't pay a 400 round-trip (e.g. Gemini via gemini2openai:
# "Penalty is not enabled for this model").
_ENDPOINT_DROPS: Dict[str, set] = {}


def _strip_known_rejects(where: str, kwargs: dict) -> None:
    with _BAD_REQ_LOCK:
        drops = set(_ENDPOINT_DROPS.get(where, ()))
    for p in drops:
        kwargs.pop(p, None)


def _remember_drop(where: str, params) -> None:
    with _BAD_REQ_LOCK:
        _ENDPOINT_DROPS.setdefault(where, set()).update(params)


def _bad_request_text(e) -> str:
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        msg = err.get("message") if isinstance(err, dict) else err
        if msg:
            return str(msg)
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            return resp.text
        except Exception:
            pass
    return str(e)


def _log_bad_request(where: str, text: str) -> None:
    key = (where, re.sub(r"\d+", "#", text)[:200])
    with _BAD_REQ_LOCK:
        if key in _BAD_REQ_SEEN:
            return
        _BAD_REQ_SEEN.add(key)
    print(f"    [!] HTTP 400 from {where}: {text[:600]}", flush=True)


def _clamp_max_tokens_from_error(text: str, requested: int) -> Optional[int]:
    """Parse vLLM/SGLang/LiteLLM context-overflow messages -> safe max_tokens."""
    ctx = re.search(r"maximum context length is (\d+)", text) or \
          re.search(r"max(?:imum)?[_ ]model[_ ]len(?:gth)?\D{0,20}(\d+)", text, re.I)
    inp = (re.search(r"(\d+) in the messages", text)
           or re.search(r"(\d+) input tokens", text)
           or re.search(r"prompt (?:is|has|contains) (\d+) tokens", text, re.I))
    if not ctx or not inp:
        return None
    room = int(ctx.group(1)) - int(inp.group(1)) - 64
    if room < 256 or room >= requested:
        return None
    return room


def _fold_system_into_user(messages: list) -> list:
    sys_txt = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    rest = [dict(m) for m in messages if m.get("role") != "system"]
    if sys_txt and rest and rest[0].get("role") == "user":
        rest[0]["content"] = f"{sys_txt}\n\n{rest[0]['content']}"
    elif sys_txt:
        rest.insert(0, {"role": "user", "content": sys_txt})
    return rest


def _fix_payload_for_400(text: str, kwargs: dict, where: str = "") -> Optional[str]:
    """Mutate kwargs to address a 400. Returns a description, or None if unfixable."""
    low = text.lower()
    mt = kwargs.get("max_tokens")
    if mt and ("context length" in low or "max_model_len" in low or "too large" in low
               or "maximum context" in low):
        new_mt = _clamp_max_tokens_from_error(text, int(mt))
        if new_mt:
            kwargs["max_tokens"] = new_mt
            return f"max_tokens {mt} -> {new_mt}"
        return None
    if "system" in low and ("not supported" in low or "alternate" in low):
        if any(m.get("role") == "system" for m in kwargs.get("messages", [])):
            kwargs["messages"] = _fold_system_into_user(kwargs["messages"])
            return "folded system prompt into user turn"
    for p in _DROPPABLE_PARAMS:
        if p in low and p in kwargs:
            kwargs.pop(p)
            _remember_drop(where, [p])
            return f"dropped {p}"
    # Generic penalty rejection that names no parameter (Gemini upstream).
    if "penalty" in low:
        present = [p for p in _PENALTY_PARAMS if p in kwargs]
        if present:
            for p in present:
                kwargs.pop(p)
            _remember_drop(where, _PENALTY_PARAMS)
            return "dropped " + ", ".join(present) + " (endpoint rejects penalties)"
    return None


def _safe_create(client: OpenAI, **kwargs):
    """chat.completions.create with visible 400 reasons and up to 3 targeted fixes."""
    where = str(getattr(client, "base_url", "?")).rstrip("/")
    _strip_known_rejects(where, kwargs)
    for _ in range(4):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            if getattr(e, "status_code", None) != 400:
                raise
            text = _bad_request_text(e)
            _log_bad_request(where, text)
            fix = _fix_payload_for_400(text, kwargs, where)
            if not fix:
                raise
            print(f"    [*] retrying {where}: {fix}", flush=True)
    return client.chat.completions.create(**kwargs)


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
    _category = _caller_category("apex (other)")
    _call_t0 = time.time()
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
        response = _safe_create(client, **kwargs)
    except Exception as e:
        low = str(e).lower()
        if _is_stream_options_rejection(low):
            kwargs.pop("stream_options")
            response = _safe_create(client, **kwargs)
        else:
            raise

    text, p_tok, c_tok = "", 0, 0
    t0, ttft, finish = _call_t0, None, None
    try:
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                if ttft is None and chunk.choices[0].delta.content:
                    ttft = time.time() - t0
                text += chunk.choices[0].delta.content
            if chunk.choices and getattr(chunk.choices[0], "finish_reason", None):
                finish = chunk.choices[0].finish_reason
            if getattr(chunk, "usage", None) is not None:
                p_tok, c_tok = chunk.usage.prompt_tokens, chunk.usage.completion_tokens
    finally:
        try:
            response.close()
        except Exception:
            pass

    text = enforce_ascii(text.strip())
    estimated = False
    if not p_tok and not c_tok:
        p_tok, c_tok = estimate_tokens(system_prompt + user_prompt), estimate_tokens(text)
        estimated = True
    _LEDGER.add(_category, "apex", p_tok, c_tok, time.time() - t0, ttft, estimated,
                rnd=getattr(_TOKEN_CTX, "rnd", None), truncated=(finish == "length"))
    return text, p_tok, c_tok


def _probe_props(root: str):
    """llama-server native endpoint. Returns (n_ctx, slots, source) or None if absent."""
    resp = requests.get(f"{root}/props", timeout=5.0)
    if resp.status_code in (404, 405, 501):
        return None  # permanent: not llama-server (vLLM, LiteLLM, llama-cpp-python, ...)
    resp.raise_for_status()
    data = resp.json()
    dgs = data.get("default_generation_settings", {}) or {}
    n_ctx = dgs.get("n_ctx") or data.get("n_ctx") or 0
    slots = data.get("total_slots") or dgs.get("n_parallel") or 0
    return int(n_ctx or 0), int(slots or 0), "/props"


def _probe_models(ep: str):
    """OpenAI-compatible fallback via /v1/models. Slot count is not exposed here."""
    resp = requests.get(f"{ep}/models", timeout=5.0,
                        headers={"Authorization": f"Bearer {GEN_API_KEY}"})
    resp.raise_for_status()
    models = (resp.json() or {}).get("data") or []
    if not models:
        return 0, 0, "/v1/models (empty)"
    m = next((x for x in models if x.get("id") == LLM_MODEL), models[0])
    meta = m.get("meta") or {}
    # vLLM: max_model_len | SGLang/others: context_length | llama-server: meta.n_ctx_train
    for key, src in (("max_model_len", "max_model_len"),
                     ("context_length", "context_length"),
                     ("context_window", "context_window")):
        if m.get(key):
            return int(m[key]), 0, f"/v1/models:{src}"
    if meta.get("n_ctx"):
        return int(meta["n_ctx"]), 0, "/v1/models:meta.n_ctx"
    if meta.get("n_ctx_train"):
        return int(meta["n_ctx_train"]), 0, "/v1/models:meta.n_ctx_train (train ctx, not -c)"
    return 0, 0, "/v1/models (no ctx field)"


def verify_server_props(endpoints: List[str], label: str, expect_ctx: int, expect_np: int) -> None:
    """Advisory context/slot check. Tries llama-server /props, falls back to
    /v1/models on any OpenAI-compatible backend. Never fatal."""
    if os.getenv("SKIP_PROPS_CHECK", "0") == "1":
        return
    per_slot = int(expect_ctx) // max(1, int(expect_np)) if expect_ctx else 0
    for ep in endpoints:
        ep = ep.rstrip("/")
        root = ep[:-3] if ep.endswith("/v1") else ep
        result, last_err = None, None
        for attempt in range(3):
            try:
                result = _probe_props(root)
                if result is None:          # 404 etc: don't retry, fall back once
                    result = _probe_models(ep)
                n_ctx = result[0]
                # llama-server briefly reports 512/2048 during unified-KV startup
                if (result[2] == "/props" and expect_ctx and n_ctx in (512, 2048)
                        and n_ctx not in (int(expect_ctx), per_slot) and attempt < 2):
                    time.sleep(2)
                    continue
                break
            except Exception as exc:
                last_err, result = exc, None
                time.sleep(2)
        if result is None:
            print(f"    [!] {label} {ep} ctx probe failed: {str(last_err)[:120]}", flush=True)
            continue

        n_ctx, slots, src = result
        if not n_ctx:
            note = "ctx not exposed by backend; trusting {}_SERVER_CTX".format("APEX" if label.lower() == "apex" else "WORKER")
        elif n_ctx < per_slot:
            note = "UNDER-PROVISIONED: node window {} < budgeted per-slot {}".format(n_ctx, per_slot)
        elif n_ctx not in (int(expect_ctx), per_slot):
            note = "larger than budget (-c {} / per-slot {}); headroom unused".format(expect_ctx, per_slot)
        elif expect_np and slots and slots != int(expect_np):
            note = "MISMATCH: constant says -np {}".format(expect_np)
        else:
            note = "ok"
        print("    [+] {} {} n_ctx={} slots={} via {} :: {}".format(
            label, ep, n_ctx or "?", slots or "?", src, note), flush=True)


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
        f"    apex    :{urllib.parse.urlparse(GEN_API_BASE).port or '-'}  -c {APEX_SERVER_CTX} -np {APEX_SERVER_NP}"
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
        f"decision rounds: W={MAX_PARALLELISM} per batch, K1={ONLINE_MAX_DECISION_ROUNDS}, "
        f"K2={REPLAY_MAX_DECISION_ROUNDS}"
    )
    lines.append(
        f"    [i] Evaluator: fixed at creation; inline unit tests {'ON' if EVAL_INLINE_TESTS else 'OFF'}"
        f" (<= {EVAL_MAX_TEST_FILES} file(s)/attempt), run-scoped venv, wheels only"
        f"{', allowlist ' + str(len(TEST_PIP_ALLOWLIST)) + ' pkg(s)' if TEST_PIP_ALLOWLIST else ''}"
        f"{'' if TEST_PIP_INSTALL else ', installs OFF'}; untested prior {EVAL_UNTESTED_PRIOR}"
    )
    lines.append(
        f"    [i] RSI: budget {ROUND_BUDGET_PER_TASK} agent call(s)/task per round"
        f" (clamped {ROUND_BUDGET_MIN}-{ROUND_BUDGET_MAX})"
        f" | M={DREAM_CANDIDATES + 1} chained policy version(s) per dream on apex"
        f" | V = quality - {DREAM_BETA1}*N + {DREAM_BETA2}*N/k | replay costs 0 agent calls"
        f" | revise only if oracle headroom >= {DREAM_MIN_HEADROOM}"
        f" | pi_0 second roots for weakest {PI0_BRANCH_FRACTION:.0%} of assignments"
        + (f" | credit: integration q blended {EVAL_CREDIT_MIX:.2f} with own delta (gain {EVAL_CREDIT_GAIN:g})"
           if EVAL_CREDIT else "")
        + (f" | support probes {SUPPORT_PROBE_FRAC:.0%} (extension)" if SUPPORT_PROBE_FRAC > 0 else "")
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
        response = _safe_create(gen_client,
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _PROMPT_PHASE1_GEN},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=APEX_GEN_TOKENS,
            stream=True
        )

        ttft = None
        try:
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    if ttft is None and chunk.choices[0].delta.content:
                        ttft = time.time() - start_time
                    full_content += chunk.choices[0].delta.content
        finally:
            try:
                response.close()
            except Exception:
                pass

        elapsed = round(time.time() - start_time, 2)
        _LEDGER.add("draft (phase 1)", "apex", estimate_tokens(_PROMPT_PHASE1_GEN + prompt),
                    estimate_tokens(full_content), time.time() - start_time, ttft, estimated=True)
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
        stream = _safe_create(client,
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
#   round t: deploy policy pi_t online -> in decision rounds it selects batches of
#            at most W legal continuations; agents run them and a FIXED evaluator
#            scores each attempt at creation -> the resulting discovery TREE is
#            appended to the history H_t -> offline, a policy-development agent
#            (APEX) derives versions pi_t^1..pi_t^(M-1), each from the previous
#            one, using the replay scores and execution traces of its
#            predecessor -> every version is scored by REPLAY over all of H_t at
#            zero agent calls (Eq. 1) -> the argmax becomes pi_{t+1}.
#
# Two properties this buys, both load-bearing:
#   * Replay is exact, not approximate. The simulator IS the realized search
#     space; every outcome a candidate policy can reveal is already on disk, and
#     the score it reveals is the score the online policy saw. Its limit is
#     equally sharp - only continuations history recorded are legal in replay -
#     which is why this must be a loop rather than a one-off tuning pass.
#   * Monotonicity. The deployed policy pi_t^0 is itself a candidate, so the
#     selected policy is no worse than it in mean replay score on H_t.
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
    slug = "_".join(w.lower() for w in words)
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
        lines.append(f"YOU ARE {self_entry['id']}. Emit paths relative to the PROJECT ROOT (e.g. `mod.py`, "
                     f"`tests/test_mod.py`); the pipeline files them under {WORK_DIRNAME}/{self_entry['dir']}/ "
                     "for bookkeeping only. Never put that directory in a path or an import.")
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
        lines.append(f"  [{r['id']}] owns :: {obj}")
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
    # A node id recorded twice (e.g. by a re-run append); last write wins.
    merged: Dict[str, dict] = {}
    for n in nodes:
        if "id" in n:
            merged[n["id"]] = n
    return [_normalise_node(n) for n in merged.values()]


def _normalise_node(n: dict) -> dict:
    """Bring nodes recorded by earlier revisions onto the current scale.

    Earlier revisions stored two scores: score_online (what the policy saw at
    decision time) and a test-informed score written back after the round. The
    evaluator is now fixed at creation, so a legacy node's creation-time value
    is its evaluator score, and replay reveals exactly that."""
    if "score_online" in n:
        n["score"] = n.pop("score_online")
    n.setdefault("score", 0.0)
    n.setdefault("cost", 1 if n.get("task") else 0)
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
# A fixed evaluator scores every attempt ONCE, at creation, as part of its
# generation-evaluation request. The stored score s_v is final: the online
# policy decides on it and replay reveals the same value, so replay evaluates a
# policy against exactly the outcomes it would have observed.
#
#   s_v = EVAL_HEURISTIC_MIX * h_v + (1 - EVAL_HEURISTIC_MIX) * rho_v
#
#   h_v   - deterministic contract/novelty heuristic (score_node_heuristic)
#   rho_v - pass rate of the unit tests generated and executed for this
#           attempt's testable deliverables (EVAL_INLINE_TESTS=1);
#           EVAL_UNTESTED_PRIOR if it has deliverables but none were tested;
#           0 if it has no deliverables.
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
    """Evaluator score. A node with nothing on disk has nothing to test: its
    rate is 0, not the prior, so failed attempts do not earn score for free."""
    if pass_rate is not None:
        rate = float(pass_rate)
    else:
        rate = EVAL_UNTESTED_PRIOR if has_files else 0.0
    return round(EVAL_HEURISTIC_MIX * heuristic + (1.0 - EVAL_HEURISTIC_MIX) * rate, 6)


# ------------------------------------------------------------------
# Explorer interface - identical surface for live and replay
# ------------------------------------------------------------------
# Decision-round semantics (Dream-RSI, Sec. 3):
#   * The eligible continuation set is A(T) = {root} U {leaves of T}. The root is
#     shared by all assignments, so a root action names the assignment it opens
#     a new branch for; a leaf is continued only by its own assignment.
#   * One call to expand_parallel(C) is ONE decision round. C is a set of at most
#     W distinct actions that are legal in the tree as it stood BEFORE the call.
#     Illegal, duplicate or over-W requests return None and cost nothing; if no
#     request is admissible, no round is consumed.
#   * An online rollout allows at most K1 decision rounds, a replay K2. The
#     rollout also ends when the policy returns (the empty batch) or when the
#     per-round agent-call budget is exhausted.
#   * In replay, the continuation of v reveals Child(v; T), the recorded child of
#     v; an action whose child was never recorded is not legal (legal_actions()
#     lists only recorded continuations).
# The policy never touches an explorer object directly: it runs in a child
# process and reaches these methods through a line-JSON RPC (see run_policy).
# Only the names in _POLICY_API are dispatchable, and every return value is a
# plain JSON projection, so there is no Path, queue or roster to reach through.
# ------------------------------------------------------------------

_POLICY_API = {"tasks", "root", "budget_left", "spent", "max_parallelism", "rounds_left",
               "legal_actions", "nodes", "frontier", "best", "best_per_task", "note",
               "expand", "expand_parallel"}


class PolicyAborted(Exception):
    """Raised inside the PARENT when a policy exceeds its interface-call cap.
    The child re-raises it as a BaseException subclass, so a policy's
    `except Exception` cannot swallow it."""
    pass


def _node_view(n: dict) -> dict:
    """What a policy observes of a revealed node: its evaluator score and the
    evaluator's diagnostics. Deliverable bodies stay on disk; the policy sees
    their paths."""
    return {
        "id": n.get("id"), "parent": n.get("parent"), "task": n.get("task"),
        "depth": n.get("depth", 0), "score": float(n.get("score", 0.0)),
        "gain": float(n.get("gain") or 0.0), "status": n.get("status"), "cost": int(n.get("cost") or 1),
        "files": list(n.get("files", [])),
        "diagnostics": {
            "violations": [str(v)[:160] for v in (n.get("violations") or [])[:5]],
            "truncated": bool(n.get("truncated", False)),
            "emitted": int(n.get("emitted") or 0), "inherited": int(n.get("inherited") or 0),
            "tests_passed": int(n.get("tests_passed") or 0), "tests_total": int(n.get("test_count") or 0),
            "test_failures": [str(x)[:160] for x in (n.get("test_failures") or [])[:5]],
            "integration_delta": float(n.get("integration_delta") or 0.0),
            "integration_delta_known": n.get("integration_delta") is not None,
        },
    }


class ExplorerBase:
    """What an exploration policy is allowed to do. Live and replay implement the
    same methods, so one policy source runs in both worlds unchanged."""

    def __init__(self, tasks: List[str], budget: int, max_parallelism: int, max_rounds: int):
        self._tasks = list(tasks)
        self._budget = int(budget)
        self._spent = 0
        self._nodes: List[dict] = []
        self._log: List[str] = []
        self._root_id = "root"
        self._W = max(1, int(max_parallelism))
        self._K = max(1, int(max_rounds))
        self._rounds = 0
        self._trace: List[dict] = []
        self._stop: Optional[str] = None
        self._support_mode = False

    # --- read-only views ---
    def tasks(self) -> List[str]:
        return list(self._tasks)

    def root(self) -> str:
        return self._root_id

    def budget_left(self) -> int:
        return max(0, self._budget - self._spent)

    def spent(self) -> int:
        return self._spent

    def max_parallelism(self) -> int:
        return self._W

    def rounds_left(self) -> int:
        return max(0, self._K - self._rounds)

    def _real(self) -> List[dict]:
        return [n for n in self._nodes if n.get("task")]

    def _parents(self) -> Set[str]:
        return {n.get("parent") for n in self._nodes if n.get("parent")}

    def _legal_set(self) -> List[Tuple[str, str]]:
        """A(T): one root action per assignment plus every leaf of the tree."""
        parents = self._parents()
        acts = [(self._root_id, t) for t in self._tasks]
        for n in self._real():
            if n["id"] not in parents and n["task"] in self._tasks:
                acts.append((n["id"], n["task"]))
        return acts

    def _available(self, req: Tuple[str, str]) -> bool:
        return True

    def legal_actions(self) -> List[List[str]]:
        return [[p, t] for p, t in self._legal_set() if self._available((p, t))]

    def nodes(self) -> List[dict]:
        return [_node_view(n) for n in self._real()]

    def frontier(self) -> List[dict]:
        """The current leaves: expanded nodes with no expanded children."""
        parents = self._parents()
        return [_node_view(n) for n in self._real() if n["id"] not in parents]

    def best(self) -> Optional[dict]:
        real = self._real()
        if not real:
            return None
        return _node_view(max(real, key=lambda n: float(n.get("score", 0.0))))

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

    # --- actions ---
    @staticmethod
    def _normalise_request(req) -> Optional[Tuple[str, str]]:
        try:
            return (str(req[0]), str(req[1]))
        except Exception:
            return None

    def expand(self, parent_id: str, task: str) -> Optional[dict]:
        res = self.expand_parallel([(parent_id, task)])
        return res[0] if res else None

    def expand_parallel(self, requests) -> List[Optional[dict]]:
        """One decision round. Returns a list aligned 1:1 with `requests`."""
        reqs = [self._normalise_request(r) for r in list(requests)]
        results: List[Optional[dict]] = [None] * len(reqs)
        if self.rounds_left() <= 0:
            if self._stop is None:
                self._stop = f"decision-round cap reached (K={self._K})"
            return results
        if self.budget_left() <= 0:
            if self._stop is None:
                self._stop = "agent-call budget exhausted"
            return results
        legal = set(self._legal_set())
        batch: List[Tuple[int, Tuple[str, str]]] = []
        seen: Set[Tuple[str, str]] = set()
        rejected = {"illegal": 0, "duplicate": 0, "over_w": 0}
        for i, r in enumerate(reqs):
            if r is None or r not in legal or not self._available(r):
                rejected["illegal"] += 1
            elif r in seen:
                rejected["duplicate"] += 1
            elif len(batch) >= self._W:
                rejected["over_w"] += 1
            else:
                seen.add(r)
                batch.append((i, r))
        if not batch:
            return results
        self._rounds += 1
        outs = self._expand_batch([r for _, r in batch])
        for (i, _), o in zip(batch, outs):
            results[i] = o
        revealed = [o for o in outs if o]
        self._trace.append({
            "k": self._rounds,
            "batch": len(batch),
            "roots": sum(1 for _, r in batch if r[0] == self._root_id),
            "refines": sum(1 for _, r in batch if r[0] != self._root_id),
            "rejected": {k: v for k, v in rejected.items() if v},
            "revealed": len(revealed),
            "scores": [round(o["score"], 3) for o in revealed],
            "best": round(max([float(n.get("score", 0.0)) for n in self._real()] or [0.0]), 4),
            "spent": self._spent,
            "support": bool(self._support_mode),
        })
        return results

    def _expand_batch(self, reqs: List[Tuple[str, str]]) -> List[Optional[dict]]:
        raise NotImplementedError


class ReplayExplorer(ExplorerBase):
    """Dreaming. Continuing v reveals Child(v; T) - the recorded child of v for
    that assignment - and charges its recorded cost; nothing is executed and no
    agent is called. A continuation history never recorded is not legal."""

    def __init__(self, recorded: List[dict], tasks: List[str], budget: int,
                 max_parallelism: int, max_rounds: int):
        super().__init__(tasks, budget, max_parallelism, max_rounds)
        self._by_parent: Dict[str, List[dict]] = {}
        for n in recorded:
            self._by_parent.setdefault(n.get("parent") or "", []).append(n)
        for v in self._by_parent.values():
            v.sort(key=lambda n: n.get("seq", 0))
        roots = [n for n in recorded if not n.get("parent")]
        self._root_id = roots[0]["id"] if roots else "root"
        self._consumed: Set[str] = set()
        if roots:
            self._nodes.append(dict(roots[0]))
        else:
            self._nodes.append({"id": "root", "parent": None, "task": None, "depth": 0,
                                "score": 0.0})

    def _next_child(self, parent_id: str, task: str) -> Optional[dict]:
        for cand in self._by_parent.get(parent_id, []):
            if cand.get("task") == task and cand["id"] not in self._consumed:
                return cand
        return None

    def _available(self, req: Tuple[str, str]) -> bool:
        return self._next_child(req[0], req[1]) is not None

    def _expand_batch(self, reqs):
        out: List[Optional[dict]] = []
        for parent_id, task in reqs:
            if self.budget_left() <= 0:
                out.append(None)
                continue
            match = self._next_child(parent_id, task)
            if match is None:
                out.append(None)
                continue
            self._consumed.add(match["id"])
            self._spent += int(match.get("cost", 1))
            self._nodes.append(dict(match))
            out.append(_node_view(match))
        return out


class LiveExplorer(ExplorerBase):
    """Online deployment. Each expansion is a real agent call followed by the
    fixed evaluator; the outcome is recorded into the tree, so this round's
    exploration becomes next round's simulator."""

    def __init__(self, tasks: List[str], budget: int, roster: List[dict], rnd: int,
                 background: str, run_dir: Path, slot_queue: queue.Queue,
                 max_parallelism: int, max_rounds: int,
                 prior_hashes: Optional[Dict[str, Set[str]]] = None,
                 semantic_guidance: bool = False,
                 test_ctx: Optional[dict] = None):
        super().__init__(tasks, budget, max_parallelism, max_rounds)
        self.roster = roster
        self.rnd = rnd
        self.background = background
        self.run_dir = run_dir
        self.slot_queue = slot_queue
        self.semantic_guidance = semantic_guidance
        self.prior_hashes = prior_hashes or {}
        self.test_ctx = test_ctx
        self._seq = 0
        self._lock = threading.Lock()
        self._root_id = new_node_id(rnd, 0)
        root = {"id": self._root_id, "seq": 0, "round": rnd, "parent": None,
                "depth": 0, "task": None, "status": "root", "score": 0.0,
                "cost": 0, "files": [], "violations": [], "max_parallelism": self._W}
        self._nodes.append(root)
        append_node(run_dir, rnd, root)

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def _real(self) -> List[dict]:
        with self._lock:
            return [n for n in self._nodes if n.get("task")]

    def _parents(self) -> Set[str]:
        with self._lock:
            return {n.get("parent") for n in self._nodes if n.get("parent")}

    def _expand_batch(self, reqs):
        results: List[Optional[dict]] = [None] * len(reqs)
        with self._lock:
            allowance = self.budget_left()
        runnable = list(enumerate(reqs))[:allowance]
        if not runnable:
            return results
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(runnable))) as ex:
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
        """EXTENSION (SUPPORT_PROBE_FRAC > 0): spend up to n calls on off-policy
        continuations chosen deterministically from the tree the policy grew,
        using the same legality and batch rules as the policy:
          deep - continue each assignment's best leaf that has deliverables
          wide - open a further root branch, fewest-branches assignments first
        The two lists are interleaved and dispatched in batches of W."""
        if n <= 0:
            return 0
        real = [dict(x) for x in self._real()]
        parents = self._parents()
        roots: Dict[str, int] = {}
        best: Dict[str, dict] = {}
        for x in real:
            if x.get("parent") == self._root_id:
                roots[x["task"]] = roots.get(x["task"], 0) + 1
            if x["id"] in parents or not x.get("files"):
                continue
            cur = best.get(x["task"])
            if cur is None or x.get("score", 0.0) > cur.get("score", 0.0):
                best[x["task"]] = x
        deep = [(b["id"], t) for t, b in sorted(best.items(),
                                                key=lambda kv: (-kv[1].get("score", 0.0), kv[0]))]
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
        self._K = self._rounds + (len(reqs) + self._W - 1) // self._W
        before = self.spent()
        try:
            for i in range(0, len(reqs), self._W):
                if self.budget_left() <= 0 or _shutdown_event.is_set():
                    break
                self.expand_parallel(reqs[i:i + self._W])
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
        # retry reserves again and is refused once the budget is gone. Evaluator
        # calls (test generation) are not discovery-agent calls and are not charged.
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
                    # Fixed evaluator, part of the same generation-evaluation
                    # request; runs on the slot this attempt already holds.
                    if node.get("files") and dependency_gate(node, self.run_dir, self.rnd):
                        pass
                    elif ((EVAL_INLINE_TESTS or EVAL_INTEGRATION) and self.test_ctx is not None
                            and node.get("files")):
                        try:
                            evaluate_node_inline(node, endpoint, self.run_dir, self.rnd,
                                                 self.test_ctx)
                        except Exception as exc:
                            print(f"\n    [!] {task} evaluation raised {str(exc)[:80]}", flush=True)
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
                    "score": blend_test_score(h, None, False),
                    "test_pass_rate": None, "files": [], "file_hashes": [], "violations": [],
                    "notes": [], "elapsed": 0, "prompt_tokens": 0, "completion_tokens": 0,
                    "truncated": False, "slot": slot_name or "", "log_path": None}
        node["cost"] = max(1, attempts)
        node["attempts"] = attempts
        node["support"] = bool(self._support_mode)
        if parent_task_node is not None:
            node["gain"] = round(node["score"] - parent_task_node.get("score", 0.0), 6)
        else:
            node["gain"] = None

        with self._lock:
            self._nodes.append(node)
        append_node(self.run_dir, self.rnd, node)
        append_event(self.run_dir, {
            "round": self.rnd, "event": "node", "node": node_id, "parent": parent_id,
            "task": task, "depth": depth, "status": node["status"], "cost": node["cost"],
            "score": node["score"], "gain": node["gain"], "files": len(node["files"]),
            "tests": node.get("test_pass_rate"),
        })
        with self._lock:
            real = [n for n in self._nodes if n.get("task")]
            spent = self._spent
        rt = _LEDGER.totals(_LEDGER.select(round=self.rnd))
        ag = _LEDGER.totals(_LEDGER.select(round=self.rnd, tier="agent"))
        sys.stdout.write("\r    [+] round {:02d}: {} node(s), budget {}/{}, decision round {}, best {:.3f} | "
                         "tok {} in / {} out, agent {} tok/s   ".format(
            self.rnd, len(real), spent, self._budget, self._rounds,
            max([n["score"] for n in real] or [0.0]),
            _fmt_tok(rt["prompt"]), _fmt_tok(rt["completion"]), ag["median_tps"] or "-"))
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

    # Agents often prefix their own location ("work/t01_x/mod.py"); that is not
    # a claim on anyone else's directory, so strip it before judging ownership.
    if len(parts) > 1 and parts[0] == WORK_DIRNAME:
        parts = parts[1:]
    if len(parts) > 1 and parts[0] == agent_dir.parent.name:
        parts = parts[1:]
    if parts[0] in other_dirs or re.match(r'^t\d{2}(_|$)', parts[0]):
        violation = f"declared path '{declared}' addresses another agent's directory"
        parts = ["claimed"] + parts

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
                blocks.append(f"--- attempt {n['id']} (score {n.get('score', 0.0):.3f}) ---\n"
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
        if cur is None or n.get("score", 0) > cur.get("score", 0):
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
              semantic_guidance: bool = False, extra_stage: str = "") -> dict:
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
    contract_block = (fit_context(_RUN_CONTRACT, AGENT_CONTRACT_BUDGET,
                                  note="...[CONTRACT TRUNCATED]...") if _RUN_CONTRACT else "")

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
    if extra_stage:
        stage_note = f"{extra_stage}\n\n{stage_note}"
    brief_block = (fit_context(_RUN_BRIEF, AGENT_BRIEF_BUDGET, note="...[PROMPT TRUNCATED]...")
                   if _RUN_BRIEF else "")
    api_block = (fit_context(_RUN_FROZEN_API_TEXT, AGENT_API_BUDGET, note="...[API LIST TRUNCATED]...")
                 if _RUN_FROZEN_API_TEXT else "")
    env_block = _RUN_ENV_TEXT or ""
    facts_block = _RUN_API_FACTS_TEXT or ""
    run_block = (fit_context(_RUN_LAST_RUN_TEXT, AGENT_RUN_BUDGET, note="...[RUN OUTPUT TRUNCATED]...")
                 if _RUN_LAST_RUN_TEXT else "")
    context_block = fit_context(background, max(2000, context_budget - len(contract_block)
                                                - len(brief_block) - len(api_block) - len(run_block)
                                                - len(env_block) - len(facts_block)))
    contract_title = ("SYNTHESIZED CONTRACT (FROM THE USER'S PROMPT - BINDING FOR EVERY AGENT)"
                      if _RUN_CONTRACT_SYNTHESIZED else
                      "PINNED CONTRACT (VERBATIM FROM THE BRIEF, PLUS PROBED DEPENDENCIES - "
                      "BINDING FOR EVERY AGENT)")

    user_instruction = (
        f"{roster_block}\n\n"
        f"===== BROADER CONTEXT (ORIENTATION ONLY - NOT YOUR SCOPE) =====\n{context_block}\n\n"
        f"===== TEAM COMMUNICATION LOG =====\n{comms_block}\n\n"
        f"===== STAGE =====\n{stage_note}\n\n"
        + (f"===== ORIGINAL USER PROMPT (VERBATIM - THE INTENT EVERYTHING ELSE SERVES) =====\n"
           f"{brief_block}\n\n" if brief_block else "")
        + (f"===== {contract_title} =====\n"
           f"{contract_block}\n\n" if contract_block else "")
        + (f"===== CONTAINER ENVIRONMENT (DISCOVERED - RESCANNED EVERY ROUND) =====\n"
           f"{env_block}\n\n" if env_block else "")
        + (f"===== LIBRARY API FACTS (INSPECTED THIS ROUND) =====\n"
           f"{facts_block}\n\n" if facts_block else "")
        + (f"===== CURRENT INTERFACES (WHAT YOUR SIBLINGS ACTUALLY CALL) =====\n"
           f"{api_block}\n\n" if api_block else "")
        + (f"===== LATEST REAL RUN OF THE INTEGRATED PROJECT =====\n"
           f"{run_block}\n\n" if run_block else "")
        + f"===== YOUR OBJECTIVE ({task}) =====\n{objective_block}\n"
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
            response = _safe_create(client,
                stream_options={"include_usage": True}, **base_kwargs)
        except Exception as e:
            if _is_stream_options_rejection(str(e).lower()):
                response = _safe_create(client, **base_kwargs)
            else:
                raise

        call_t0, ttft = time.time(), None
        try:
            for chunk in response:
                now = time.time()
                if ttft is None and chunk.choices and chunk.choices[0].delta is not None \
                        and chunk.choices[0].delta.content:
                    ttft = now - call_t0
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
        _LEDGER.add("write-up refresh" if extra_stage else "agent attempts", "agent",
                    prompt_tokens, comp_tokens, time.time() - call_t0, ttft, is_estimated,
                    rnd=rnd, truncated=truncated)

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

        if accepted and node_out.exists():
            violations.extend(frozen_api_violations({"dir": agent["dir"], "id": node_id}, run_dir))

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
    # Heuristic part of the fixed evaluator; the test part is applied by
    # evaluate_node_inline before the node is recorded.
    online = blend_test_score(h, None, bool(saved_files))
    return {
        "id": node_id, "seq": seq, "round": rnd, "parent": parent_id, "depth": depth,
        "task": task, "dir": agent["dir"], "status": status,
        "score": online, "heuristic_score": h, "score_parts": parts,
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
# Exploration policy pi_0: branching parallel refine (the initial policy of
# Dream-RSI Sec. 4, widened so round-1 trees are not pure chains).
#   1. One root branch per assignment, in batches of at most W.
#   2. The weakest BRANCH_FRACTION of assignments (by first-attempt score) get a
#      second independent root: a sibling, recorded as an alternative to
#      continuing the first branch.
#   3. Every remaining decision round continues the current leaf of every open
#      branch, lowest-scoring leaves first (most evaluator headroom), W per batch.
# The recorded tree therefore contains sibling-vs-continuation choices that
# replay can re-decide. Both Dream-RSI and the fixed-exploration control start
# from this policy, so round 1 is identical by construction.

BRANCH_FRACTION = %%BRANCH_FRACTION%%


def chunks(items, size):
    out = []
    for i in range(0, len(items), size):
        out.append(items[i:i + size])
    return out


def can_act(ctx):
    return ctx.budget_left() > 0 and ctx.rounds_left() > 0


def run_batches(ctx, actions, w):
    got = []
    for group in chunks(actions, w):
        if not can_act(ctx):
            break
        got.extend([n for n in ctx.expand_parallel(group) if n])
    return got


def explore(ctx):
    w = ctx.max_parallelism()
    root = ctx.root()
    tasks = ctx.tasks()

    first = {}
    for n in run_batches(ctx, [(root, t) for t in tasks], w):
        first[n["task"]] = n
    if not first or not can_act(ctx):
        return

    n_branch = 0
    if BRANCH_FRACTION > 0:
        n_branch = min(len(first), max(1, int(round(BRANCH_FRACTION * len(first)))))
    weakest = sorted(first, key=lambda t: (first[t]["score"], t))[:n_branch]
    ctx.note("pi_0 second roots: " + ",".join(weakest))

    # Siblings go first in the queue, then continuations of the first branches,
    # lowest score first. They share batches, so no W-slot is left idle.
    queue = [(root, t) for t in weakest]
    queue += [(n["id"], n["task"]) for n in sorted(first.values(), key=lambda n: (n["score"], n["id"]))]
    while queue and can_act(ctx):
        revealed = run_batches(ctx, queue, w)
        if not revealed:
            break
        queue = [(n["id"], n["task"]) for n in sorted(revealed, key=lambda n: (n["score"], n["id"]))]
'''.replace("%%BRANCH_FRACTION%%", repr(PI0_BRANCH_FRACTION))

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
        tree = _quiet_parse(source)
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
    def max_parallelism(self): return _call("max_parallelism")
    def rounds_left(self): return _call("rounds_left")
    def legal_actions(self): return _call("legal_actions")
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

def replay_tree(source: str, nodes: List[dict], tasks: List[str],
                budget: int) -> Tuple[ReplayExplorer, bool, str]:
    """Replay one policy over one recorded world, from its root."""
    ex = ReplayExplorer(nodes, tasks, budget, MAX_PARALLELISM, REPLAY_MAX_DECISION_ROUNDS)
    ok, detail = run_policy(source, ex, wall_secs=POLICY_REPLAY_WALL_SECS)
    if ex._stop is None:
        ex._stop = "policy returned (empty batch)" if (ok and detail == "ok") else detail
    return ex, ok, detail


def replay_value(ex: ExplorerBase, tasks: List[str]) -> dict:
    """Eq. (1) for one replay world:
        V = quality - beta_1 * N + beta_2 * N / max(1, k*)
    quality: best revealed score per assignment, averaged over the roster; an
    assignment never opened contributes the root's score (0). N: revealed
    non-root nodes (generation-evaluation requests represented). k*: completed
    decision rounds."""
    best = {t: 0.0 for t in tasks}
    for n in ex._real():
        t = n.get("task")
        if t in best:
            best[t] = max(best[t], float(n.get("score", 0.0)))
    quality = sum(best.values()) / max(1, len(tasks))
    real = ex._real()
    N, k = len(real), ex._rounds
    parallel = (N / max(1, k)) if N > 0 else 0.0
    V = quality - DREAM_BETA1 * N + DREAM_BETA2 * parallel
    reached = {n.get("task") for n in real}
    return {"V": round(V, 6), "quality": round(quality, 6), "N": N, "k": k,
            "parallelism": round(parallel, 4),
            "coverage": round(len(reached & set(tasks)) / max(1, len(tasks)), 4),
            "cost": ex.spent(), "stop": ex._stop}


def replay_score(source: str, pool: List[Tuple[int, List[dict]]], tasks: List[str],
                 budget: int) -> dict:
    """Score a policy version by dreaming it over every recorded tree at zero
    agent calls: V^m = (1/t) * sum_i V_i^m."""
    per_tree, traces = [], []
    for rnd, nodes in pool:
        ex, ok, detail = replay_tree(source, nodes, tasks, budget)
        if not ok:
            return {"valid": False, "detail": detail, "score": None,
                    "per_tree": per_tree, "traces": traces}
        v = replay_value(ex, tasks)
        v["round"] = rnd
        per_tree.append(v)
        traces.append({"round": rnd, "stop": ex._stop, "decision_rounds": ex._trace,
                       "notes": ex._log[:20]})
    if not per_tree:
        return {"valid": False, "detail": "empty simulator pool", "score": None,
                "per_tree": [], "traces": []}
    T = len(per_tree)

    def _mean(key: str) -> float:
        return sum(float(t[key]) for t in per_tree) / T

    return {
        "valid": True, "detail": "ok", "score": round(_mean("V"), 6),
        "quality": round(_mean("quality"), 6), "N": round(_mean("N"), 2),
        "k": round(_mean("k"), 2), "parallelism": round(_mean("parallelism"), 4),
        "coverage": round(_mean("coverage"), 4), "cost": round(_mean("cost"), 2),
        "per_tree": per_tree, "traces": traces,
    }


def _policy_interface_doc(budget: int) -> str:
    return (
        "POLICY INTERFACE (this is the whole API; nothing else is available)\n"
        "  ctx.tasks() -> assignment ids, e.g. ['t01','t02']\n"
        "  ctx.root() -> id of the shared tree root\n"
        f"  ctx.max_parallelism() -> W = {MAX_PARALLELISM}, the largest batch one decision round may hold\n"
        "  ctx.rounds_left() -> decision rounds remaining in this rollout\n"
        "  ctx.legal_actions() -> list of [parent_id, task] actions legal right now\n"
        "  ctx.expand_parallel([(parent_id, task), ...]) -> ONE decision round; list of\n"
        "      node-or-None aligned 1:1 with the requests\n"
        "  ctx.expand(parent_id, task) -> a decision round with a batch of one\n"
        "  ctx.frontier() -> current leaves; ctx.nodes() -> every revealed node\n"
        "  ctx.best() / ctx.best_per_task() -> best revealed node / dict task -> node\n"
        "  ctx.budget_left() / ctx.spent() -> agent-call units\n"
        "  ctx.note(msg) -> record a short diagnostic string\n"
        "\n"
        "LEGALITY (evaluated against the tree as it stood BEFORE the call)\n"
        "  * (root, task) opens a new independent branch for that assignment.\n"
        "  * (leaf_id, leaf_task) continues a branch from its current leaf; a node that\n"
        "    already has a child is no longer continuable.\n"
        "  * A batch holds at most W distinct legal actions. Illegal, duplicate or over-W\n"
        "    requests return None and cost nothing; a batch with no admissible request\n"
        "    consumes no round. Returning from explore() is the empty batch: it stops.\n"
        "  * In replay only continuations that history recorded are legal; legal_actions()\n"
        "    lists exactly those. Online every root/leaf action is legal.\n"
        "\n"
        "A node dict has: id, parent, task, depth, score (0..1, final evaluator score),\n"
        "  gain (score minus the parent attempt's score; None at depth 1), status, cost\n"
        "  (agent calls incl. retries), files, and diagnostics {violations, truncated,\n"
        "  emitted, inherited, tests_passed, tests_total, test_failures, integration_delta,\n"
        "  integration_delta_known}.\n"
        "  integration_delta (always a float) = project q with this attempt minus q with its\n"
        "  task's previous best; > 0 means the attempt improved the whole project. It is 0.0 with\n"
        "  integration_delta_known False for a task's first attempt and for test-only attempts.\n"
        "  A continuation starts from its parent's files and only changes what it improves.\n"
        "  A failed test or a truncated output is often repairable by continuing the leaf.\n"
        "\n"
        "OBJECTIVE (per recorded tree, averaged over all trees)\n"
        f"  V = quality - {DREAM_BETA1} * N + {DREAM_BETA2} * N / max(1, k)\n"
        "  quality = mean over ALL assignments of the best revealed score (0 if never opened)\n"
        "  N = revealed nodes, k = decision rounds used. Each replay is capped at "
        f"{budget} agent calls and {REPLAY_MAX_DECISION_ROUNDS} decision rounds.\n"
        "  So: cover the roster, reveal only promising continuations, stop lines that\n"
        "  plateau, and batch independent continuations instead of expanding serially.\n"
        "\n"
        "HARD RULES\n"
        "  1. Define exactly one top-level function: explore(ctx). Helper defs and constants are ok.\n"
        "  2. No imports, no print, no file/network/system access. Pure control flow.\n"
        "  3. No attribute or name starting with '_', no bare 'except:'.\n"
        f"  4. At most {POLICY_MAX_RPC} interface calls in total, reads included.\n"
        "  5. Terminate: every loop must reach an empty batch, the budget or the round cap.\n"
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
            by_task.setdefault(n["task"], []).append(float(n.get("score", 0.0)))
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


def _result_line(name: str, r: dict) -> str:
    if not r.get("valid"):
        return f"{name}: INVALID - {str(r.get('detail', ''))[:160]}"
    trees = "; ".join(
        f"r{t['round']:02d} V {t['V']:.4f} q {t['quality']:.3f} N {t['N']} k {t['k']} "
        f"cov {t['coverage']:.2f} stop '{str(t['stop'])[:40]}'" for t in r.get("per_tree", []))
    return (f"{name}: V {r['score']:.4f} (quality {r['quality']:.4f}, N {r['N']}, k {r['k']}, "
            f"N/k {r['parallelism']:.2f}, coverage {r['coverage']:.2f}) [{trees}]")


def _trace_digest(r: dict, limit: int) -> str:
    """Per-decision-round replay trajectories of one policy version, compacted."""
    out = []
    for tr in r.get("traces", []):
        out.append(f"-- world r{tr['round']:02d}: stop = {tr['stop']}")
        for d in tr["decision_rounds"]:
            rej = (" rejected " + ",".join(f"{k}={v}" for k, v in d["rejected"].items())
                   if d.get("rejected") else "")
            out.append(f"   k{d['k']:02d}: batch {d['batch']} ({d['roots']} open, "
                       f"{d['refines']} refine){rej} -> revealed {d['revealed']} "
                       f"scores {d['scores']} best {d['best']:.3f} spent {d['spent']}")
        if tr.get("notes"):
            out.append("   notes: " + " | ".join(tr["notes"][:5]))
    return fit_context("\n".join(out) or "(no trajectories)", limit,
                       note="...[TRACES TRUNCATED]...")


def _select_winner(results: List[dict]) -> dict:
    """argmax over all evaluated versions of the mean replay score; ties go to
    the earliest version, so the deployed policy (version 0) is kept on a tie."""
    valid = [r for r in results if r.get("valid")]
    if not valid:
        return results[0]
    return max(valid, key=lambda r: (r["score"], -r["index"]))


def replay_oracle_bound(nodes: List[dict], tasks: List[str], budget: int) -> dict:
    """Upper bound on Eq. (1) for ONE recorded tree: the best V any policy could
    reach in replay if it knew every recorded score in advance.

    Replay semantics it mirrors: continuing (parent, task) reveals the recorded
    children of that pair in seq order, so reaching a node also reveals (and
    pays for) its ancestors and every earlier-seq sibling under the same
    (parent, task). For each assignment the cheapest way to hold a given best
    score is to target one node, so the optimum is a multiple-choice knapsack
    over assignments on (agent-call cost <= budget, revealed-node count N).
    The parallelism term uses N/k <= min(W, N) since k >= ceil(N / W).
    Continuations stay within their assignment (legality enforces this)."""
    tset = set(tasks)
    T = max(1, len(tasks))
    by_id = {n["id"]: n for n in nodes if n.get("id")}
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for n in nodes:
        if n.get("task"):
            groups.setdefault((n.get("parent") or "", n["task"]), []).append(n)
    for g in groups.values():
        g.sort(key=lambda n: n.get("seq", 0))

    memo: Dict[str, Tuple[int, int, float]] = {}

    def reach(nid: str, depth: int = 0) -> Tuple[int, int, float]:
        """(cost, N, best same-task score) of the minimal reveal set for nid."""
        if nid in memo:
            return memo[nid]
        n = by_id.get(nid)
        if n is None or not n.get("task") or depth > 10000:
            return (0, 0, 0.0)
        parent = n.get("parent") or ""
        pc, pn, ps = reach(parent, depth + 1) if parent in by_id and by_id[parent].get("task") else (0, 0, 0.0)
        cost, cnt, best = pc, pn, ps
        for sib in groups.get((parent, n["task"]), []):
            if sib.get("seq", 0) > n.get("seq", 0):
                break
            cost += int(sib.get("cost", 1))
            cnt += 1
            best = max(best, float(sib.get("score", 0.0)))
        memo[nid] = (cost, cnt, best)
        return memo[nid]

    options: Dict[str, List[Tuple[int, int, float]]] = {t: [(0, 0, 0.0)] for t in tasks}
    for n in nodes:
        if n.get("task") in tset:
            c, k, q = reach(n["id"])
            if c <= budget:
                options[n["task"]].append((c, k, q))

    # dp[(cost, N)] = best sum of per-assignment best scores
    dp: Dict[Tuple[int, int], float] = {(0, 0): 0.0}
    for t in tasks:
        nxt: Dict[Tuple[int, int], float] = {}
        for (c0, n0), q0 in dp.items():
            for c, k, q in options[t]:
                key = (c0 + c, n0 + k)
                if key[0] > budget:
                    continue
                if q0 + q > nxt.get(key, -1.0):
                    nxt[key] = q0 + q
        dp = nxt

    best_v, best_state = float("-inf"), (0, 0, 0.0)
    for (c, N), qsum in dp.items():
        par = min(MAX_PARALLELISM, N) if N > 0 else 0
        v = qsum / T - DREAM_BETA1 * N + DREAM_BETA2 * par
        if v > best_v:
            best_v, best_state = v, (c, N, qsum / T)
    return {"V": round(best_v, 6), "cost": best_state[0], "N": best_state[1],
            "quality": round(best_state[2], 6)}


def replay_oracle_pool(pool: List[Tuple[int, List[dict]]], tasks: List[str],
                       budget: int) -> Optional[float]:
    """Mean oracle bound over the pool (same averaging as replay_score)."""
    if not pool:
        return None
    vals = [replay_oracle_bound(nodes, tasks, budget)["V"] for _, nodes in pool]
    return round(sum(vals) / len(vals), 6)


def dream_policy_improvement(run_dir: Path, rnd: int, current_source: str,
                             pool: List[Tuple[int, List[dict]]],
                             tasks: List[str], budget: int) -> Tuple[str, dict]:
    """Offline policy improvement (Dream-RSI Sec. 3, 'Policy improvement and
    selection'). Version 0 is the deployed policy. For m = 0..M-2 the
    policy-development agent (APEX) examines the replay trajectories and scores
    of version m, together with the scores of all earlier versions, and revises
    version m's code into version m+1. All M versions are replayed on the same
    fixed history and the argmax is deployed next."""
    budget = policy_budget(budget, len(tasks))
    print(f"\n[DREAM] Round {rnd:02d}: replaying policy versions over {len(pool)} recorded "
          f"tree(s) at zero agent calls (budget {budget}, W {MAX_PARALLELISM}, "
          f"K2 {REPLAY_MAX_DECISION_ROUNDS}, beta1 {DREAM_BETA1}, beta2 {DREAM_BETA2})...",
          flush=True)

    ddir = dream_dir_for(run_dir) / f"round{rnd:02d}"
    ddir.mkdir(parents=True, exist_ok=True)
    versions: List[dict] = []

    def _evaluate(name: str, idx: int, src: str) -> dict:
        res = replay_score(src, pool, tasks, budget)
        res.update({"name": name, "index": idx})
        versions.append({"name": name, "index": idx, "source": src, "result": res})
        if res["valid"]:
            print(f"    [+] {name}: V {res['score']:.4f} | quality {res['quality']:.4f} "
                  f"| N {res['N']} | k {res['k']} | N/k {res['parallelism']:.2f} "
                  f"| coverage {res['coverage']:.2f}", flush=True)
        else:
            print(f"    [!] {name}: invalid in replay ({str(res['detail'])[:80]})", flush=True)
        return res

    base0 = _evaluate("pi_0 (deployed)", 0, current_source)

    # Headroom gate: the oracle bound caps what ANY revision could score here.
    oracle = replay_oracle_pool(pool, tasks, budget)
    headroom = (oracle - float(base0["score"])) if (oracle is not None and base0.get("valid")) else None
    skipped = False
    if headroom is not None:
        print(f"    [i] Replay oracle bound V* {oracle:.4f} | headroom {headroom:+.4f} "
              f"(epsilon {DREAM_MIN_HEADROOM})", flush=True)
        if headroom < DREAM_MIN_HEADROOM:
            if DREAM_FORCE_REVISIONS:
                print("    [i] Below epsilon, but --force-dream set; revising anyway.", flush=True)
            else:
                skipped = True
                print(f"    [i] Skipping {DREAM_CANDIDATES} apex revision(s): no revision can beat the "
                      f"deployed policy by >= {DREAM_MIN_HEADROOM} on this pool.", flush=True)

    client = None if skipped else apex_client(timeout=WORKER_TIMEOUT_SECS)

    for m in range(1, 0 if skipped else DREAM_CANDIDATES + 1):
        if _shutdown_event.is_set():
            break
        prev = versions[-1]
        earlier = "\n".join(_result_line(v["name"], v["result"]) for v in versions)
        user = (
            f"{_policy_interface_doc(budget)}\n\n"
            f"===== CURRENT POLICY: VERSION {prev['index']} SOURCE =====\n"
            f"{fit_context(prev['source'], 6000)}\n\n"
            f"===== REPLAY SCORES OF VERSION {prev['index']} =====\n"
            f"{_result_line(prev['name'], prev['result'])}\n\n"
            f"===== REPLAY TRAJECTORIES OF VERSION {prev['index']} (per decision round) =====\n"
            f"{_trace_digest(prev['result'], DREAM_TRACE_CHARS)}\n\n"
            f"===== ALL VERSIONS EVALUATED SO FAR =====\n{earlier}\n\n"
            f"===== RECORDED DISCOVERY HISTORY =====\n"
            f"{fit_context(_pool_summary(pool, tasks), 4000)}\n\n"
            f"Revise version {prev['index']} into version {m}. Keep what the trajectories show "
            f"working, fix what they show failing. Change the search SHAPE - which "
            f"continuations, batch composition, depth versus breadth, stopping - never the "
            f"agents' objectives. Output only Python."
        )
        try:
            raw, _, _ = _apex_completion(client, _PROMPT_POLICY_DEV, user,
                                         APEX_POLICY_TOKENS, 0.8)
        except Exception as exc:
            print(f"    [!] Version {m} generation failed: {str(exc)[:120]}", flush=True)
            break
        source = re.sub(r'^```[a-zA-Z]*\s*|```\s*$', '', raw.strip(), flags=re.MULTILINE).strip()
        source = enforce_ascii(source)
        with open(ddir / f"candidate_{m:02d}.py", "w", encoding="ascii") as f:
            f.write(source + "\n")
        _evaluate(f"pi_{m}", m, source)

    results = [v["result"] for v in versions]
    winner = _select_winner(results)
    winner_source = versions[winner["index"]]["source"]
    base = results[0]

    with open(ddir / "policy_execution_traces.jsonl", "w", encoding="ascii") as f:
        for v in versions:
            for tr in v["result"].get("traces", []):
                f.write(json.dumps({"version": v["index"], "name": v["name"], **tr},
                                   ensure_ascii=True) + "\n")
    with open(ddir / "scores.json", "w", encoding="ascii") as f:
        json.dump({"oracle_bound": oracle,
                   "headroom": None if headroom is None else round(headroom, 6),
                   "min_headroom": DREAM_MIN_HEADROOM, "revisions_skipped": skipped,
                   "versions": [{k: val for k, val in r.items() if k != "traces"} for r in results]},
                  f, indent=2)

    def _fmt(r: dict) -> float:
        return float(r["score"]) if r.get("valid") else -1.0

    next_path = policy_path(run_dir, rnd + 1)
    next_path.parent.mkdir(parents=True, exist_ok=True)
    body = enforce_ascii(winner_source).rstrip()
    # Strip a previous selection header so headers do not accumulate.
    body = re.sub(r'\A(# Deployed for round .*\n# Replay score .*\n)+', '', body + "\n").rstrip()
    with open(next_path, "w", encoding="ascii") as f:
        f.write(f"# Deployed for round {rnd + 1}. Selected by replay over {len(pool)} tree(s).\n"
                f"# Replay score {_fmt(winner):.4f} (deployed version {_fmt(base):.4f}).\n"
                + body + "\n")

    improved = winner["index"] != 0
    print(f"    [+] Selected: {winner['name']} (V {_fmt(winner):.4f} vs deployed "
          f"{_fmt(base):.4f}) -> {next_path.name}"
          + ("" if improved else ("  [revisions skipped: headroom below epsilon; unchanged]" if skipped
                                  else "  [no version beat the deployed policy; unchanged]")), flush=True)

    append_event(run_dir, {
        "round": rnd, "event": "dream", "candidates": len(results),
        "oracle_bound": oracle, "headroom": None if headroom is None else round(headroom, 6),
        "revisions_skipped": skipped,
        "valid": len([r for r in results if r["valid"]]), "winner": winner["name"],
        "winner_score": _fmt(winner), "baseline_score": _fmt(base), "improved": improved,
    })

    return winner_source, {
        "round": rnd, "winner": winner["name"], "winner_score": _fmt(winner),
        "baseline_score": _fmt(base), "candidates": len(results),
        "valid": len([r for r in results if r["valid"]]), "improved": improved,
        "oracle_bound": oracle, "headroom": None if headroom is None else round(headroom, 6),
        "revisions_skipped": skipped,
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


# ------------------------------------------------------------------
# The original brief: pinned contract + required deliverables
# ------------------------------------------------------------------
# Phase 1/2 rewrite the brief and Phase 3 re-partitions it; both are lossy.
# Two things are therefore taken from the ORIGINAL brief and never rewritten:
# the pinned sections (shown verbatim to every agent and test generator) and
# the list of deliverables (every one must be owned by some assignment).
_BRIEF_HEADING_RE = re.compile(r'^([A-Z][A-Z0-9 &/_-]{2,}?)\s*(\([^)]*\))?\s*:?\s*$')
_BRIEF_ITEM_RE = re.compile(r'^\s*(?:\d+[.)]|[-*])\s+`?([A-Za-z0-9_][A-Za-z0-9_./-]*\.[A-Za-z0-9]{1,6})`?(?=[\s:,;(]|$)')


def split_brief_sections(text: str) -> List[Tuple[str, str]]:
    """(HEADING, section text including its heading line). A heading is an
    unindented, all-caps line, optionally followed by a parenthesised note."""
    sections: List[Tuple[str, List[str]]] = [("", [])]
    for line in (text or "").splitlines():
        m = _BRIEF_HEADING_RE.match(line) if line[:1].isalpha() and len(line) <= 100 else None
        if m and any(ch.isalpha() for ch in m.group(1)):
            sections.append((m.group(1).strip().upper(), [line]))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections if "\n".join(body).strip()]


def extract_pinned_contract(text: str) -> str:
    parts = [body for head, body in split_brief_sections(text)
             if head and any(head.startswith(name) for name in PINNED_SECTIONS)]
    return enforce_ascii("\n\n".join(parts).strip())


def extract_deliverables(text: str) -> Dict[str, str]:
    """Deliverable path -> its verbatim item text, taken from every section whose
    heading starts with DELIVERABLES. Items are numbered or bulleted lines whose
    first token is a file path; runtime outputs named inside an item are not
    deliverables."""
    out: Dict[str, str] = {}
    for head, body in split_brief_sections(text):
        if not head.startswith("DELIVERABLES"):
            continue
        current, buf = None, []
        for line in body.splitlines()[1:]:
            m = _BRIEF_ITEM_RE.match(line)
            if m:
                if current:
                    out.setdefault(current, "\n".join(buf).rstrip())
                current, buf = m.group(1), [line]
            elif current:
                buf.append(line)
        if current:
            out.setdefault(current, "\n".join(buf).rstrip())
    return {k: enforce_ascii(v) for k, v in out.items()}


_PROMPT_CONTRACT_SYNTH = (
    "You turn a user's request into a short project contract. Use ONLY what the user's text "
    "states. Do not add libraries, frameworks, algorithms, file formats, parameters or numbers "
    "the user did not state.\n\n"
    "Output plain text with exactly these two sections and nothing else - no preamble, no "
    "markdown fences:\n\n"
    "DELIVERABLES\n"
    "1. relative/path.ext - one line: what the user asked this file to do, in the user's terms\n"
    "2. ...\n\n"
    "CONSTRAINTS\n"
    "- one line per constraint the user stated\n\n"
    "DELIVERABLES rules: one numbered item per distinct thing the user asked to be delivered; "
    "the first token of every item is a relative file path. Python modules live in ONE flat "
    "project directory and import each other by bare module name; tests are tests/test_<name>.py; "
    "a written report or write-up is a .md file. Use short, plain file names. Do not merge "
    "separately requested things into one file, and do not add deliverables the user did not "
    "ask for (no setup, packaging, dependency-check or config files unless requested). Do not "
    "name third-party libraries unless the user named them - the list of available libraries "
    "is appended separately.\n"
    "CONSTRAINTS rules: restate every constraint the user gave - scope, simplicity, "
    "reproducibility, honesty and reporting rules included - in plain words. Invent none."
)


def synthesize_contract(prompt: str) -> Tuple[str, Dict[str, str]]:
    """Contract for a brief with no pinned sections, written by the apex from the
    user's prompt alone. Returns (DELIVERABLES + CONSTRAINTS text, deliverables);
    ("", {}) if the apex output does not parse into at least one deliverable."""
    client = apex_client(timeout=WORKER_TIMEOUT_SECS)
    user = f"USER REQUEST (verbatim):\n\n{fit_context(prompt, MAX_CONTEXT_CHARS)}"
    for attempt in range(1, 3):
        try:
            raw, _, _ = _apex_completion(client, _PROMPT_CONTRACT_SYNTH, user, APEX_PLAN_TOKENS, 0.2)
        except Exception as exc:
            print(f"    [!] Contract synthesis failed (attempt {attempt}/2): {str(exc)[:120]}", flush=True)
            continue
        text = re.sub(r'^```[a-zA-Z]*\s*$', '', enforce_ascii(raw or ""), flags=re.MULTILINE).strip()
        deliverables = {k: v for k, v in extract_deliverables(text).items()
                        if not k.startswith(("/", "~")) and ".." not in PurePosixPath(k).parts}
        sections = {h: body for h, body in split_brief_sections(text) if h}
        constraints = next((b for h, b in sections.items() if h.startswith("CONSTRAINTS")), "")
        if not deliverables:
            print(f"    [!] Contract synthesis produced no parseable deliverables "
                  f"(attempt {attempt}/2).", flush=True)
            continue
        items = "\n".join(v for v in deliverables.values())
        body = "DELIVERABLES\n" + items + (("\n\n" + constraints) if constraints else "")
        return enforce_ascii(body.strip()), deliverables
    return "", {}


def synthesized_contract_header() -> str:
    return ("# SYNTHESIZED CONTRACT - written by the planner from the user's prompt only (the prompt\n"
            "# had no pinned INTERFACES / CONSTRAINTS / ACCEPTANCE CRITERIA). Binding for every agent;\n"
            "# where it and the user's original prompt disagree, the prompt wins.")


_PROMPT_INTERFACES_SYNTH = (
    "You are the planner of a small Python project that several agents build in parallel, one "
    "file each. They cannot see each other's code while writing, so they need ONE shared API. "
    "Write it.\n\n"
    "Output plain text with exactly these two sections and nothing else - no preamble, no "
    "markdown fences:\n\n"
    "INTERFACES\n"
    "module.py\n"
    "  def function_name(arg: type, arg2: type = default) -> return_type   # one-line purpose\n"
    "  class ClassName(init_arg: type, ...)   # one-line purpose\n"
    "      .method(arg: type) -> return_type   # one-line purpose\n"
    "  @dataclass ClassName(field: type = default, ...)\n"
    "  CONSTANT_NAME: type\n"
    "other_module.py\n"
    "  ...\n\n"
    "RUN\n"
    "python3 entry_module.py\n\n"
    "PROBE\n"
    "python3 entry_module.py --some-flag value\n"
    "  effect: which reported metric must change, and in which direction\n\n"
    "KNOWN ANSWERS\n"
    "  metric_name: known-good case -> expected value; known-bad case -> different expected value\n\n"
    "Rules:\n"
    "- One block per NON-TEST .py deliverable, in dependency order (a module only uses names "
    "from blocks above it). List only the public names other modules or the tests need. Keep "
    "it minimal: this is a small prototype.\n"
    "- Module names are the deliverables' file names; modules import each other by bare name "
    "from one flat directory.\n"
    "- Types use only the standard library and the available third-party modules listed below.\n"
    "- Every signature must be complete (all parameters, defaults, return type) so two agents "
    "who never talk can still call each other correctly.\n"
    "- Pin every VALUE vocabulary, not just types: a str that selects between options is written "
    "as Literal[\"a\", \"b\"]; a dict return lists its exact keys and value types; an instruction "
    "or gate list gets its exact tuple layout AND the complete set of op names, e.g.\n"
    "    Gate = tuple[Literal[\"h\", \"x\", \"cx\", \"rz\"], list[int], list[float]]  "
    "# (op, qubits, params); rz params=[theta]\n"
    "  and every producer emits only those names while every consumer accepts all of them. Put "
    "shared vocabularies as named aliases at the top of the module that owns them.\n"
    "- If a module wraps an external library, name the library calls it must use ONLY from the "
    "LIBRARY API FACTS given below, and use only libraries listed in the CONTAINER ENVIRONMENT.\n"
    "- RUN is ONE command, no shell operators, that runs the project's entry point with its "
    "defaults, finishes in under a minute on a CPU, exits 0, and prints a line containing "
    "\"score\": <number> (JSON) for the summary score.\n"
    "- PROBE is a NEGATIVE CONTROL: the same entry point with one flag that introduces a change "
    "whose effect on the reported metrics is known in advance and does NOT depend on the "
    "hypothesis being tested (e.g. deliberately corrupt the transmitted state, skip a required "
    "correction, use an orthogonal target). If the metrics do not move under PROBE, they measure "
    "nothing. The flag must appear in the entry module's interface. Same output format as RUN.\n"
    "- KNOWN ANSWERS: for every metric the RUN output reports, one case with a known expected "
    "value and one case with a DIFFERENT known expected value, stated from first principles "
    "(e.g. ideal teleportation of |1> -> fidelity 1.0; orthogonal state -> 0.0). Never state the "
    "outcome the experiment is meant to measure as a known answer.\n"
    "- Serve the user's request and the deliverables; do not add features."
)

_SHELL_META_RE = re.compile(r'[;&|`$<>\\]')


def synthesize_interfaces(prompt: str, contract: str,
                          deliverables: Dict[str, str]) -> Tuple[str, str, str]:
    """(INTERFACES section incl. RUN/PROBE/KNOWN ANSWERS, RUN command, PROBE
    command). Planner-chosen design, labelled as such. Empty strings if the apex
    output does not parse."""
    modules = [d for d in deliverables if d.endswith(".py") and not _is_test_file(d)]
    if not modules:
        return "", "", ""
    client = apex_client(timeout=WORKER_TIMEOUT_SECS)
    user = (f"USER REQUEST (verbatim):\n{fit_context(prompt, 12000)}\n\n"
            f"CONTRACT SO FAR:\n{fit_context(contract, 12000)}\n\n"
            f"NON-TEST PYTHON DELIVERABLES: {', '.join(modules)}")
    for attempt in range(1, 3):
        try:
            raw, _, _ = _apex_completion(client, _PROMPT_INTERFACES_SYNTH, user, APEX_PLAN_TOKENS, 0.2)
        except Exception as exc:
            print(f"    [!] Interface synthesis failed (attempt {attempt}/2): {str(exc)[:120]}", flush=True)
            continue
        text = re.sub(r'^```[a-zA-Z]*\s*$', '', enforce_ascii(raw or ""), flags=re.MULTILINE)
        body, run_lines, probe_lines, known, cur = [], [], [], [], None
        for line in text.splitlines():
            st = line.strip()
            if re.match(r'^INTERFACES\b', st):
                cur = "i"
                continue
            if re.match(r'^RUN\b\s*:?\s*$', st):
                cur = "r"
                continue
            if re.match(r'^PROBE\b\s*:?\s*$', st):
                cur = "p"
                continue
            if re.match(r'^KNOWN ANSWERS\b\s*:?\s*$', st):
                cur = "k"
                continue
            if cur == "p" and st:
                probe_lines.append(st.strip("`"))
                continue
            if cur == "k" and st:
                known.append("  " + st.lstrip("-* ").strip())
                continue
            if cur == "i" and st:
                # Module headers stay flush left; everything else is indented so
                # an all-caps constant can never be read as a section heading.
                is_mod = re.match(r'^`?[A-Za-z_][A-Za-z0-9_]*\.py`?:?$', st) is not None
                body.append(st.strip("`").rstrip(":") if is_mod
                            else ("  " + line.rstrip() if line[:1].isspace() else "  " + st))
            elif cur == "r" and st:
                run_lines.append(st.strip("`"))
        named = {l for l in body if not l.startswith(" ")}
        if not named & set(modules):
            print(f"    [!] Interface synthesis produced no module blocks (attempt {attempt}/2).", flush=True)
            continue
        run_cmd = validate_run_command(run_lines[0] if run_lines else "", deliverables)
        probe_cmd = ""
        if run_cmd and probe_lines:
            probe_cmd = validate_run_command(probe_lines[0], deliverables)
            if probe_cmd == run_cmd:
                probe_cmd = ""
        effect = " ".join(l for l in probe_lines[1:] if l)[:300]
        section = ("INTERFACES (planner-chosen, not stated in the user's prompt)\n"
                   "  Binding so that modules written in parallel fit together.\n" + "\n".join(body))
        if run_cmd:
            section += f"\n\nRUN (executed by the pipeline after each round)\n  {run_cmd}"
        if probe_cmd:
            section += ("\n\nPROBE (negative control, executed by the pipeline after every successful run)\n"
                        f"  {probe_cmd}\n"
                        "  Must run the same experiment with that one change and print the same JSON keys.\n"
                        + (f"  {effect}\n" if effect else "")
                        + "  If the reported numbers do not change under PROBE, the metrics are flagged as\n"
                          "  measuring nothing.\n"
                          "  The effect above is a PREDICTION the pipeline tests, never a value to emit: the flag\n"
                          "  must change the experiment's input or procedure, and no code may branch on it to set\n"
                          "  a metric or score. The pipeline scans for that and for a score that moves alone.")
        if known:
            section += ("\n\nKNOWN ANSWERS (planner-chosen; the tests must check these)\n"
                        + "\n".join(known[:20]))
        return enforce_ascii(section), run_cmd, probe_cmd
    return "", "", ""


def validate_run_command(cmd: str, deliverables) -> str:
    cmd = (cmd or "").strip()
    parts = cmd.split()
    if len(parts) < 2 or parts[0] not in ("python3", "python") or _SHELL_META_RE.search(cmd):
        return ""
    if not parts[1].endswith(".py") or ".." in parts[1] or parts[1].startswith("/"):
        return ""
    if deliverables and parts[1] not in set(deliverables):
        return ""
    return "python3 " + " ".join(parts[1:])


def default_run_command(deliverables) -> str:
    names = set(deliverables or [])
    for n in _ENTRYPOINT_NAMES:
        if n in names:
            return f"python3 {n}"
    return ""


def _sig_of(fn) -> Tuple[str, List[Tuple[str, bool]]]:
    """Rendered signature and [(param, has_default)] of an ast function."""
    a = fn.args
    params: List[Tuple[str, bool]] = []
    rendered: List[str] = []
    pos = list(a.posonlyargs) + list(a.args)
    defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    for arg, d in zip(pos, defaults):
        ann = f": {ast.unparse(arg.annotation)}" if arg.annotation is not None else ""
        dv = f" = {ast.unparse(d)}" if d is not None else ""
        rendered.append(f"{arg.arg}{ann}{dv}")
        params.append((arg.arg, d is not None))
    if a.vararg:
        rendered.append(f"*{a.vararg.arg}")
    elif a.kwonlyargs:
        rendered.append("*")
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        ann = f": {ast.unparse(arg.annotation)}" if arg.annotation is not None else ""
        dv = f" = {ast.unparse(d)}" if d is not None else ""
        rendered.append(f"{arg.arg}{ann}{dv}")
        params.append((arg.arg, d is not None))
    if a.kwarg:
        rendered.append(f"**{a.kwarg.arg}")
    ret = f" -> {ast.unparse(fn.returns)}" if fn.returns is not None else ""
    return f"({', '.join(rendered)}){ret}", params


def _is_dataclass(cls) -> bool:
    for d in cls.decorator_list:
        name = ast.unparse(d.func if isinstance(d, ast.Call) else d)
        if name.split(".")[-1] == "dataclass":
            return True
    return False


def extract_module_api(source: str) -> Dict[str, dict]:
    """Public names of one module: functions, classes (constructor + public
    methods, dataclass fields), UPPER_CASE constants."""
    try:
        tree = _quiet_parse(source)
    except (SyntaxError, ValueError):
        return {}
    api: Dict[str, dict] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            sig, params = _sig_of(node)
            api[node.name] = {"kind": "def", "sig": sig, "params": params,
                              "varkw": node.args.kwarg is not None}
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods, init_sig, init_params, varkw = [], "()", [], False
            if _is_dataclass(node):
                fields = []
                for st in node.body:
                    if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                        dv = f" = {ast.unparse(st.value)}" if st.value is not None else ""
                        fields.append(f"{st.target.id}: {ast.unparse(st.annotation)}{dv}")
                        init_params.append((st.target.id, st.value is not None))
                init_sig = f"({', '.join(fields)})"
            for st in node.body:
                if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig, params = _sig_of(st)
                    if params and params[0][0] in ("self", "cls"):
                        params = params[1:]
                        sig = re.sub(r'^\((self|cls)(, )?', '(', sig)
                    if st.name == "__init__":
                        init_sig, init_params = re.sub(r' -> None$', '', sig), params
                        varkw = st.args.kwarg is not None
                    elif not st.name.startswith("_"):
                        methods.append(f".{st.name}{sig}")
            api[node.name] = {"kind": "dataclass" if _is_dataclass(node) else "class",
                              "sig": init_sig, "params": init_params, "methods": methods,
                              "varkw": varkw}
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id.isupper() and not t.id.startswith("_"):
                    api[t.id] = {"kind": "const", "sig": "", "params": []}
    return api


def _local_uses(source: str, local: Set[str]) -> List[Tuple[str, str]]:
    """(module, name) pairs this source takes from local modules:
    `from m import a` and `import m` ... `m.a`."""
    try:
        tree = _quiet_parse(source)
    except (SyntaxError, ValueError):
        return []
    uses: List[Tuple[str, str]] = []
    aliases: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in local:
            uses += [(node.module, a.name) for a in node.names if a.name != "*"]
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name in local:
                    aliases[a.asname or a.name] = a.name
    if aliases:
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                    and node.value.id in aliases:
                uses.append((aliases[node.value.id], node.attr))
    return sorted(set(uses))


def _bad_keyword_calls(rel: str, source: str, api: Dict[str, dict]) -> List[str]:
    """Calls into sibling modules that pass a keyword the callee does not take."""
    try:
        tree = _quiet_parse(source)
    except (SyntaxError, ValueError):
        return []
    names: Dict[str, Tuple[str, str]] = {}
    mods: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in api:
            for a in node.names:
                names[a.asname or a.name] = (node.module, a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name in api:
                    mods[a.asname or a.name] = a.name
    out: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.keywords:
            continue
        f, target = node.func, None
        if isinstance(f, ast.Name) and f.id in names:
            target = names[f.id]
        elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in mods:
            target = (mods[f.value.id], f.attr)
        if not target:
            continue
        info = api.get(target[0], {}).get(target[1])
        if not info or info.get("varkw") or info.get("kind") == "const":
            continue
        accepted = {p for p, _ in info.get("params") or []}
        for kw in node.keywords:
            if kw.arg is not None and kw.arg not in accepted:
                out.append(f"{rel} calls {target[0]}.{target[1]}({kw.arg}=...), but it takes no "
                           f"parameter '{kw.arg}' (accepts: {', '.join(sorted(accepted)) or 'none'})")
    return out


def freeze_project_api(proj: Path, rnd: int) -> Tuple[Dict[str, dict], str]:
    """Structured API of the integrated project plus its rendering for agents:
    every public name, who uses it, and every import that does not resolve."""
    files = {str(p.relative_to(proj)).replace("\\", "/"): read_file_content_safe(p) or ""
             for p in sorted(proj.rglob("*.py")) if p.is_file() and "__pycache__" not in p.parts
             and ".home" not in p.parts}
    top = {r: src for r, src in files.items() if "/" not in r}
    local = {r[:-3] for r in top}
    api = {r[:-3]: extract_module_api(src) for r, src in top.items() if not _is_test_file(r)}
    unresolved: List[str] = []
    for rel, src in files.items():
        for mod, name in _local_uses(src, local):
            if mod not in api:
                continue
            if name in api[mod]:
                api[mod][name].setdefault("used_by", [])
                if rel not in api[mod][name]["used_by"]:
                    api[mod][name]["used_by"].append(rel)
            else:
                unresolved.append(f"{rel} uses {mod}.{name}, but {mod}.py defines no such name")
    for rel, src in files.items():
        unresolved += _bad_keyword_calls(rel, src, api)
    lines = [f"CURRENT INTERFACES (extracted from the integrated project after round {rnd:02d} - "
             "what the other modules are actually built against)"]
    for mod in sorted(api):
        lines.append(f"{mod}.py")
        if not api[mod]:
            lines.append("  (no public names)")
        for name, info in api[mod].items():
            head = {"def": "def ", "class": "class ", "dataclass": "@dataclass ", "const": ""}[info["kind"]]
            used = info.get("used_by") or []
            lines.append(f"  {head}{name}{info['sig']}" + (f"   <- used by {', '.join(used)}" if used else ""))
            for m in info.get("methods", [])[:12]:
                lines.append(f"      {m}")
    if unresolved:
        lines.append("")
        lines.append("UNRESOLVED (a module uses a name its owner does not define - one side must change):")
        lines += [f"  - {u}" for u in sorted(set(unresolved))[:20]]
    lines += ["",
              "RULES: keep every name marked 'used by' with a compatible signature. Adding names is "
              "free. If you must rename or change one, keep the old form working (alias or wrapper) "
              "and send a <note> to each agent that uses it. Where this list and the contract's "
              "INTERFACES disagree, the contract wins: move toward it and tell the users."]
    return api, enforce_ascii("\n".join(lines))


def frozen_api_violations(node: dict, run_dir: Path) -> List[str]:
    """Names other modules use that this attempt removed or made incompatible:
    a used name that disappeared, a parameter that disappeared, or a new
    required parameter."""
    if not _RUN_FROZEN_API:
        return []
    ndir = work_dir_for(run_dir) / node.get("dir", "") / node["id"]
    out: List[str] = []
    for p in sorted(ndir.glob("*.py")):
        mod = p.stem
        frozen = _RUN_FROZEN_API.get(mod)
        if not frozen or _is_test_file(p.name):
            continue
        now = extract_module_api(read_file_content_safe(p) or "")
        if not now:
            continue
        for name, info in frozen.items():
            used = info.get("used_by") or []
            if not used:
                continue
            if name not in now:
                out.append(f"breaks frozen interface: {mod}.{name} removed (used by {', '.join(used)})")
                continue
            old = dict(info.get("params") or [])
            new = dict(now[name].get("params") or [])
            gone = [k for k in old if k not in new]
            added_req = [k for k, has_d in new.items() if k not in old and not has_d]
            if gone or added_req:
                what = ", ".join([f"-{k}" for k in gone] + [f"+{k} (required)" for k in added_req])
                out.append(f"breaks frozen interface: {mod}.{name} signature changed ({what}; "
                           f"used by {', '.join(used)})")
    return out[:5]


_RUN_ERROR_PATTERNS = [
    (re.compile(r'"status"\s*:\s*"(error|fail|failed|failure|exception)"', re.I), '"status": "error"'),
    (re.compile(r'"(error|exception|traceback)"\s*:\s*"[^"]', re.I), 'an "error" field'),
    (re.compile(r'^Traceback \(most recent call last\):', re.M), "a Python traceback"),
    (re.compile(r'"score"\s*:\s*(NaN|Infinity|-Infinity|null)', re.I), "a non-numeric score"),
]


def run_output_errors(out: str) -> List[str]:
    """Error reports inside a run's output, whatever its exit code."""
    return [label for rx, label in _RUN_ERROR_PATTERNS if rx.search(out or "")]


def test_rules_section() -> str:
    return "\n".join([
        "TEST RULES (pipeline standard)",
        "- Every metric the run reports gets a known-answer test: one known-good case and one",
        "  known-bad case whose expected values DIFFER (use the KNOWN ANSWERS section if present).",
        "  A metric that cannot fail such a test measures nothing.",
        "- Tests check correctness against cases whose answer is known from first principles.",
        "  They never assert the outcome the experiment is meant to measure (no 'the effect",
        "  exists', no 'the difference is non-zero'): a null result must pass the test suite.",
    ])


def run_rules_section(cmd: str) -> str:
    return "\n".join([
        "RUN RULES (enforced by the pipeline's grounding run after every round)",
        f"- `{cmd}` counts as a successful run only if it exits 0, prints a JSON line with",
        '  "score": <finite number>, and reports no error ("status": "error", an "error" field,',
        "  or a traceback in its output).",
        "- Printed prose (conclusions, notes) is not a result; only measured numbers count.",
        "- Errors must reach the exit code. No catch-all except that prints an error and exits 0,",
        "  and no silent fallback that substitutes another backend or a made-up value when",
        "  something fails. A hidden failure is still scored as a failure.",
    ])


_NUM_KV_RE = re.compile(r'"([A-Za-z0-9_ .\-]{1,60})"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)')


def numeric_fingerprint(out: str) -> List[Tuple[str, float]]:
    """(key, value) for every numeric JSON field in an output, in order, minus
    keys that legitimately vary between runs (timing, seeds)."""
    fp = []
    for k, v in _NUM_KV_RE.findall(out or ""):
        if PROBE_IGNORE_KEYS.search(k):
            continue
        try:
            fp.append((k, float(v)))
        except ValueError:
            pass
    return fp


_SUMMARY_KEY_RE = re.compile(r'(^|_)score$', re.I)


def _flag_tokens(probe_cmd: str) -> Tuple[Set[str], Set[str]]:
    """(identifier forms, literal forms) of the control flag(s) in a probe command."""
    idents, lits = set(), set()
    for tok in (probe_cmd or "").split()[2:]:
        if tok.startswith("--") and len(tok) > 3:
            name = tok[2:].split("=")[0]
            lits.add(tok.split("=")[0])
            idents.add(name.replace("-", "_").lower())
    return idents, lits


def hardcoded_control_branches(proj: Path, probe_cmd: str, metric_keys: List[str]) -> List[str]:
    """'file:line name = constant' for every branch on the control flag that
    assigns a literal number to a reported metric (or to anything named like a
    score). Finds the pattern wherever it lives, entry point or library."""
    idents, lits = _flag_tokens(probe_cmd)
    if not idents and not lits:
        return []
    metric_names = {k.lower() for k in metric_keys} | {"score"}

    # The flag rarely keeps its name on the way down (--corrupt -> args.corrupt ->
    # corrupt_flag -> corrupt_state), so an identifier matches when it contains
    # every word of the flag name as one of its own underscore-separated words.
    flag_words = [set(i.split("_")) for i in idents]

    def ident_matches(name: str) -> bool:
        words = set(name.lower().split("_"))
        return any(fw and fw <= words for fw in flag_words)

    def refs_flag(expr) -> bool:
        for n in ast.walk(expr):
            if isinstance(n, ast.Name) and ident_matches(n.id):
                return True
            if isinstance(n, ast.Attribute) and ident_matches(n.attr):
                return True
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in lits:
                return True
        return False

    def is_metric(name: str) -> bool:
        low = name.lower()
        return low in metric_names or bool(_SUMMARY_KEY_RE.search(low))

    def const_number(v) -> bool:
        if isinstance(v, ast.UnaryOp) and isinstance(v.op, (ast.USub, ast.UAdd)):
            v = v.operand
        return isinstance(v, ast.Constant) and isinstance(v.value, (int, float)) and not isinstance(v.value, bool)

    hits: List[str] = []
    for p in sorted(proj.rglob("*.py")):
        if "__pycache__" in p.parts or _is_test_file(p.name):
            continue
        try:
            tree = _quiet_parse(read_file_content_safe(p) or "")
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            # metric = <measured> if not flag else 0.1   (a flag-keyed constant for a metric)
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.IfExp) \
                    and refs_flag(node.value.test) \
                    and (const_number(node.value.body) or const_number(node.value.orelse)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    name = t.id if isinstance(t, ast.Name) else t.attr if isinstance(t, ast.Attribute) else ""
                    if name and is_metric(name):
                        hits.append(f"{p.name}:{node.lineno} {name} = <expr> if <flag> else constant")
            if not isinstance(node, ast.If) or not refs_flag(node.test):
                continue
            for stmt in list(node.body) + list(node.orelse):
                for sub in ast.walk(stmt):
                    if isinstance(sub, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and const_number(sub.value):
                        targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                        for t in targets:
                            name = (t.id if isinstance(t, ast.Name) else
                                    t.attr if isinstance(t, ast.Attribute) else
                                    t.slice.value if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                                    and isinstance(t.slice.value, str) else "")
                            if name and is_metric(name):
                                hits.append(f"{p.name}:{sub.lineno} {name} = {ast.unparse(sub.value)}")
                    elif isinstance(sub, ast.Dict):
                        for k, v in zip(sub.keys, sub.values):
                            if isinstance(k, ast.Constant) and isinstance(k.value, str) and is_metric(k.value) \
                                    and const_number(v):
                                hits.append(f"{p.name}:{sub.lineno} {{'{k.value}': {ast.unparse(v)}}}")
                    elif isinstance(sub, ast.Return) and sub.value is not None and const_number(sub.value):
                        hits.append(f"{p.name}:{sub.lineno} return {ast.unparse(sub.value)}")
    return sorted(set(hits))[:10]


_FREE_TEXT_RE = re.compile(r'"([A-Za-z0-9_ .\-]{1,60})"\s*:\s*"((?:[^"\\]|\\.){40,})"')


def free_text_fields(out: str) -> List[str]:
    """Keys of long string values in a run's JSON output: prose the code wrote,
    not something it measured."""
    return sorted({k for k, _ in _FREE_TEXT_RE.findall(out or "")})[:10]


def sensitivity_probe(proj: Path, rnd: int, test_ctx: dict, run_dir: Path,
                      run_out: str) -> Optional[dict]:
    """Run the negative-control command and compare its numbers with the main
    run's. RESPONSIVE: something moved. INSENSITIVE: every reported number is
    identical under a change that must affect them."""
    if not _RUN_PROBE_COMMAND:
        return None
    work = run_dir / "tests" / f"round{rnd:02d}" / "grounding_probe"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(proj, work / "project", ignore=shutil.ignore_patterns("__pycache__", ".home"))
    env = _test_env(work, test_ctx.get("venv_bin"), str(work / "project"))
    rc, out, to = _run_limited(_RUN_PROBE_COMMAND.split(), RUN_COMMAND_SECS, work / "project", env,
                               cpu=max(TEST_CPU_SECS, RUN_COMMAND_SECS))
    ptail = (out or "").strip()
    base = {"command": _RUN_PROBE_COMMAND, "rc": rc, "timed_out": to,
            "output_tail": ptail if len(ptail) <= 2500 else "...[cut]...\n" + ptail[-2500:]}
    if to or rc != 0 or run_output_errors(out or ""):
        tailline = (_extract_error_line(out or "", "python") or (out or "").strip()[-160:])
        return {**base, "verdict": "PROBE FAILED",
                "detail": f"the control command did not run cleanly ({'timeout' if to else f'exit {rc}'}): "
                          f"{tailline[:200]} - the metrics' sensitivity is unverified."}
    a, b = numeric_fingerprint(run_out), numeric_fingerprint(out or "")
    if not a or not b:
        return {**base, "verdict": "UNCOMPARABLE",
                "detail": "no numeric JSON fields to compare between the run and the control."}
    def moved(x: float, y: float) -> bool:
        return abs(x - y) > 1e-12 * max(1.0, abs(x))
    if [k for k, _ in a] == [k for k, _ in b]:
        pairs = [(k, x, y) for (k, x), (_, y) in zip(a, b)]
    else:
        da, db = dict(a), dict(b)
        pairs = [(k, da[k], db[k]) for k in da if k in db]
    changed = [(k, x, y) for k, x, y in pairs if moved(x, y)]
    measured = [k for k, _, _ in pairs if not _SUMMARY_KEY_RE.search(k)]
    hard = hardcoded_control_branches(proj, _RUN_PROBE_COMMAND, [k for k, _ in a])
    if hard:
        return {**base, "verdict": "HARDCODED", "hardcoded": hard,
                "detail": "the code assigns a fixed value to a reported metric when the control flag is set ("
                          + "; ".join(hard[:4]) + "). A control that is written in, not measured, proves "
                          "nothing about the metrics."}
    if not changed:
        return {**base, "verdict": "INSENSITIVE",
                "detail": f"all {len(a)} reported number(s) are identical under the control "
                          f"(e.g. {', '.join(f'{k}={v:g}' for k, v in a[:4])}). Either the metrics are "
                          f"constant by construction or the entry point ignores the control flag - "
                          f"both break the contract."}
    if measured and all(_SUMMARY_KEY_RE.search(k) for k, _, _ in changed):
        return {**base, "verdict": "SCORE-ONLY",
                "detail": "only the summary score moved (" + ", ".join(f"{k} {x:g} -> {y:g}" for k, x, y in changed[:4])
                          + "); the measured quantities did not ("
                          + ", ".join(sorted(set(measured))[:6]) + " unchanged). The score does not come from "
                          "the measurement."}
    return {**base, "verdict": "RESPONSIVE",
            "detail": "changed under the control: "
                      + ", ".join(f"{k} {x:g} -> {y:g}" for k, x, y in changed[:6])}


_PROMPT_SKEPTIC_REVIEW = (
    "You are a skeptical reviewer of a small scientific prototype. You receive the user's request, "
    "the project's code, the final run output and the result of a negative-control run. For EACH "
    "metric the output reports, decide from the code whether it actually measures what its name "
    "says. Look for: values that are constant by construction, conditions that are always true, "
    "comparisons against the wrong reference, parameters that never reach the computation, "
    "operations that cannot affect the measured quantity, silent fallbacks, and circuits or models "
    "missing the steps their names imply. Quote the exact code line that decides your verdict.\n"
    "Also trace the negative-control flag from the entry point: what does it actually change? It must "
    "alter the experiment's input or procedure so the metric changes BY MEASUREMENT. If any code "
    "branches on the flag to set a score or metric directly, or the flag never reaches the "
    "computation, say so. Trace each printed key back to the line that computes it, starting from "
    "the RUN entry file.\n\n"
    "Output plain text only, one block per metric:\n"
    "METRIC: <name as printed>\n"
    "VERDICT: valid | suspect | invalid\n"
    "EVIDENCE: <file>:<the exact line>\n"
    "REASON: <one or two sentences>\n\n"
    "Then:\n"
    "CONTROL: genuine | hardcoded | not wired\n"
    "EVIDENCE: <file>:<the exact line>\n"
    "REASON: <one sentence>\n"
    "Then one final line:\n"
    "OVERALL: <one sentence on whether the run's score can be reported as a finding>\n"
    "Judge only what the code does. Do not suggest fixes. Do not praise."
)


def final_skeptic_review(run_dir: Path, rnd: int) -> str:
    """Apex review of the final integrated project and its run. A model's
    judgment, not proof; saved and handed to the write-up refresh as required
    caveats. Empty string if skipped or unparseable."""
    if not FINAL_SKEPTIC_REVIEW or _shutdown_event.is_set():
        return ""
    idir = integration_dir_for(run_dir)
    run_json = idir / f"round{rnd:02d}_run.json"
    if not run_json.exists():
        return ""
    info = json.loads(read_file_content_safe(run_json) or "{}")
    if not info.get("ok"):
        print("\n[REVIEW] Skipped: the final run did not succeed, so there are no metrics to review.",
              flush=True)
        return ""
    proj = idir / "latest"
    code_parts, used = [], 0
    files = sorted((p for p in proj.rglob("*.py") if p.is_file() and "__pycache__" not in p.parts),
                   key=lambda p: (p.name != (_RUN_COMMAND.split()[1] if len(_RUN_COMMAND.split()) > 1 else ""),
                                  _is_test_file(p.name), str(p)))
    for p in files:
        body = read_file_content_safe(p) or ""
        chunk = f"===== {p.relative_to(proj)} =====\n{body}\n"
        if used + len(chunk) > REVIEW_CODE_CHARS:
            chunk = chunk[:max(0, REVIEW_CODE_CHARS - used)] + "\n...[cut]\n"
        code_parts.append(chunk)
        used += len(chunk)
        if used >= REVIEW_CODE_CHARS:
            break
    user = (f"USER REQUEST:\n{fit_context(_RUN_BRIEF, 6000)}\n\n"
            f"RUN command: {_RUN_COMMAND} | negative-control command: {_RUN_PROBE_COMMAND or '(none)'}\n\n"
            f"FINAL RUN (and negative control):\n{fit_context(_RUN_LAST_RUN_TEXT, 10000)}\n\n"
            f"CODE:\n{''.join(code_parts)}")
    print(f"\n[REVIEW] Skeptic review of the final run's metrics on apex "
          f"({len(files)} file(s), {used:,} chars of code)...", flush=True)
    try:
        raw, _, _ = _apex_completion(apex_client(timeout=WORKER_TIMEOUT_SECS), _PROMPT_SKEPTIC_REVIEW,
                                     user, APEX_PLAN_TOKENS, 0.1)
    except Exception as exc:
        print(f"    [!] Review failed: {str(exc)[:120]}", flush=True)
        return ""
    text = enforce_ascii((raw or "").strip())
    verdicts = re.findall(r'^\s*VERDICT:\s*(valid|suspect|invalid)', text, re.I | re.M)
    if not verdicts:
        print("    [!] Review produced no VERDICT lines; ignored.", flush=True)
        return ""
    with open(idir / "final_review.md", "w", encoding="ascii") as f:
        f.write("# Skeptic review (apex model judgment, not proof)\n\n" + text + "\n")
    counts = {v: sum(1 for x in verdicts if x.lower() == v) for v in ("valid", "suspect", "invalid")}
    print(f"    [+] {len(verdicts)} metric(s): {counts['valid']} valid, {counts['suspect']} suspect, "
          f"{counts['invalid']} invalid -> {INTEGRATION_DIRNAME}/final_review.md", flush=True)
    for m in re.finditer(r'METRIC:\s*(.+)\n\s*VERDICT:\s*(suspect|invalid)', text, re.I):
        print(f"    [-] {m.group(1).strip()[:60]}: {m.group(2).lower()}", flush=True)
    ctrl = re.search(r'^\s*CONTROL:\s*(genuine|hardcoded|not wired)', text, re.I | re.M)
    if ctrl:
        print(f"    [{'+' if ctrl.group(1).lower() == 'genuine' else '-'}] negative control: "
              f"{ctrl.group(1).lower()}", flush=True)
    append_event(run_dir, {"round": rnd, "event": "skeptic_review", **counts,
                           "control": ctrl.group(1).lower() if ctrl else None})
    return text


def grounding_run(proj: Path, rnd: int, test_ctx: dict, run_dir: Path) -> Optional[dict]:
    """Run the RUN command in a throwaway copy of the integrated project and keep
    its real output. This is the only source a write-up may report from."""
    if not _RUN_COMMAND:
        return None
    work = run_dir / "tests" / f"round{rnd:02d}" / "grounding"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(proj, work / "project", ignore=shutil.ignore_patterns("__pycache__", ".home"))
    env = _test_env(work, test_ctx.get("venv_bin"), str(work / "project"))
    start = time.time()
    rc, out, to = _run_limited(_RUN_COMMAND.split(), RUN_COMMAND_SECS, work / "project", env,
                               cpu=max(TEST_CPU_SECS, RUN_COMMAND_SECS))
    elapsed = time.time() - start
    m = _CMD_SCORE_RE.findall(out or "")
    score = None
    if m:
        try:
            score = float(m[-1])
        except ValueError:
            score = None
    reported_errors = run_output_errors(out or "")
    exited_ok = rc == 0 and not to
    # A run only counts if it exits 0, prints the contract's score line, and
    # reports no error. A runner that catches everything and exits 0 fails here.
    ok = exited_ok and score is not None and not reported_errors
    produced = sorted(str(p.relative_to(work / "project")) for p in (work / "project").rglob("*")
                      if p.is_file() and not (proj / p.relative_to(work / "project")).exists()
                      and "__pycache__" not in p.parts and ".home" not in p.parts)[:20]
    if ok:
        status = "SUCCEEDED"
    elif to:
        status = "TIMED OUT"
    elif rc != 0:
        status = f"FAILED (exit {rc})"
    elif reported_errors:
        status = f"FAILED (exit 0, but the output reports an error: {reported_errors[0]})"
    else:
        status = "FAILED (exit 0, but no \"score\": <number> line was printed)"
    probe = sensitivity_probe(proj, rnd, test_ctx, run_dir, out or "") if ok else None
    tail = (out or "").strip()
    if len(tail) > 6000:
        tail = "...[earlier output cut]...\n" + tail[-6000:]
    lines = [f"ACTUAL RUN OUTPUT (round {rnd:02d}; the pipeline ran the integrated project)",
             f"command: {_RUN_COMMAND}", f"status: {status} in {elapsed:.1f}s",
             f"score line found: {score if score is not None else 'none'}"]
    if produced:
        lines.append(f"files written: {', '.join(produced)}")
    lines += ["output:", tail or "(no output)", ""]
    if probe:
        lines.append(f"NEGATIVE CONTROL: {probe['command']} -> {probe['verdict']}")
        lines.append(f"  {probe['detail']}")
        lines.append("control output:")
        lines.append(probe.get("output_tail") or "(no output)")
        lines.append("")
    prose = free_text_fields(out or "")
    if prose:
        lines.append(f"UNTRUSTED FREE TEXT: the output field(s) {', '.join(prose)} are prose the code prints, "
                     "not measurements. A write-up must not quote or rely on them as findings.")
        lines.append("")
    if probe and probe["verdict"] in ("INSENSITIVE", "SCORE-ONLY", "HARDCODED"):
        why = {"INSENSITIVE": "the reported metrics did NOT change under a control that must change them",
               "SCORE-ONLY": "only the summary score changed under the control; the measured quantities did not",
               "HARDCODED": "the code writes a fixed value for the control instead of measuring it"}[probe["verdict"]]
        lines.append(f"RULE: {why}, so the metrics do not measure what their names say. A write-up must state "
                     "this plainly and must not present the score as a finding - neither as an effect nor as "
                     "a null result. The owners of the metric code: fix the measurement, and never branch on "
                     "the control flag to set a metric.")
    elif ok:
        lines.append("RULE: any number, table or claim about results in a write-up must appear in this "
                     "output (or in the files it wrote). A score near zero is a valid result: report it as "
                     "such. Do not state results this run did not produce.")
    else:
        lines.append("RULE: no successful run exists yet. A write-up must say so plainly and must not "
                     "state any result, number or trend. The owners of the failing code: fix it.")
    text = enforce_ascii("\n".join(lines))
    idir = integration_dir_for(run_dir)
    with open(idir / f"round{rnd:02d}_run.md", "w", encoding="ascii") as f:
        f.write(text + "\n")
    res = {"round": rnd, "command": _RUN_COMMAND, "ok": ok, "rc": rc, "timed_out": to,
           "score": score, "reported_errors": reported_errors, "status": status,
           "elapsed": round(elapsed, 2), "produced": produced, "probe": probe}
    with open(idir / f"round{rnd:02d}_run.json", "w", encoding="ascii") as f:
        json.dump(res, f, indent=2)
    print(f"[RUN] ROUND {rnd:02d}: {_RUN_COMMAND} -> {status} in {elapsed:.1f}s"
          + (f", score {score:g}" if score is not None else ", no score line"), flush=True)
    if probe:
        print(f"[PROBE] ROUND {rnd:02d}: {probe['command']} -> {probe['verdict']}: {probe['detail'][:160]}",
              flush=True)
    if reported_errors:
        m_err = re.search(r'"(?:error|exception)"\s*:\s*"([^"]{1,200})', out or "", re.I)
        if m_err:
            print(f"    [-] reported error: {m_err.group(1)}", flush=True)
    if not ok and not reported_errors and rc != 0:
        err = _extract_error_line(out or "", "python")
        if err:
            print(f"    [-] {err[:200]}", flush=True)
    return res


def load_round_grounding(run_dir: Path, upto_rnd: int) -> None:
    """Resume: restore the latest frozen API and run output from disk."""
    global _RUN_FROZEN_API, _RUN_FROZEN_API_TEXT, _RUN_LAST_RUN_TEXT
    idir = integration_dir_for(run_dir)
    for r in range(upto_rnd, 0, -1):
        api_json = idir / f"round{r:02d}_api.json"
        if FREEZE_INTERFACES and not _RUN_FROZEN_API and api_json.exists():
            try:
                _RUN_FROZEN_API = json.loads(read_file_content_safe(api_json) or "{}")
                _RUN_FROZEN_API_TEXT = read_file_content_safe(idir / f"round{r:02d}_api.md") or ""
            except json.JSONDecodeError:
                pass
        run_md = idir / f"round{r:02d}_run.md"
        if not _RUN_LAST_RUN_TEXT and run_md.exists():
            _RUN_LAST_RUN_TEXT = read_file_content_safe(run_md) or ""


_ENV_SCAN_SRC = r"""
import sys, json, platform, pkgutil
import importlib.metadata as md
std = set(getattr(sys, "stdlib_module_names", ())) | set(sys.builtin_module_names)
try:
    p2d = md.packages_distributions()
except Exception:
    p2d = {}
dists = {}
for d in md.distributions():
    try:
        name = d.metadata.get("Name") or ""
    except Exception:
        continue
    key = name.lower().replace("_", "-")
    if not name or key in dists:
        continue
    dists[key] = {"name": name, "version": d.version or "",
                  "summary": (d.metadata.get("Summary") or "").strip()[:140], "imports": []}
for mod, dl in p2d.items():
    if not mod.isidentifier() or mod.startswith("_") or mod in std:
        continue
    for dn in dl:
        k = dn.lower().replace("_", "-")
        if k in dists and mod not in dists[k]["imports"]:
            dists[k]["imports"].append(mod)
seen = {m for v in dists.values() for m in v["imports"]}
loose = sorted({m.name for m in pkgutil.iter_modules()
                if m.name.isidentifier() and not m.name.startswith("_")
                and m.name not in std and m.name not in seen})
print(json.dumps({"python": platform.python_version(), "executable": sys.executable,
                  "dists": dists, "loose": loose}))
"""


def scan_environment(run_dir: Path) -> Dict[str, Any]:
    """What the evaluation interpreter can import right now: every installed
    distribution (name, version, summary, import names) plus importable
    top-level modules that belong to no distribution. Nothing is imported."""
    py, venv_bin = ensure_test_venv(run_dir)
    home = run_dir / "tests" / ".envscan"
    home.mkdir(parents=True, exist_ok=True)
    rc, out, to = _run_limited([py, "-c", _ENV_SCAN_SRC], 120, home, _test_env(home, venv_bin))
    if rc != 0 or to:
        print(f"    [!] Environment scan failed ({'timeout' if to else f'exit {rc}'}); "
              f"previous scan kept.", flush=True)
        return {}
    try:
        env = json.loads((out or "").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {}
    env["dists"] = {k: v for k, v in env.get("dists", {}).items()
                    if k not in _TOOLING_DISTS and v.get("imports")}
    env["allowed"] = sorted({m for v in env["dists"].values() for m in v["imports"]}
                            | set(env.get("loose", [])))
    env["scanned_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    env["_ts"] = time.time()
    return env


def env_diff(old: Dict[str, Any], new: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    """(added, removed, upgraded) distributions, rendered."""
    o, n = old.get("dists", {}), new.get("dists", {})
    added = [f"{n[k]['name']} {n[k]['version']}" for k in sorted(set(n) - set(o))]
    removed = [o[k]["name"] for k in sorted(set(o) - set(n))]
    upgraded = [f"{n[k]['name']} {o[k]['version']} -> {n[k]['version']}"
                for k in sorted(set(n) & set(o)) if n[k]["version"] != o[k]["version"]]
    added += [f"{m} (module)" for m in sorted(set(new.get("loose", [])) - set(old.get("loose", [])))]
    removed += [f"{m} (module)" for m in sorted(set(old.get("loose", [])) - set(new.get("loose", [])))]
    return added, removed, upgraded


def _mentioned_modules(env: Dict[str, Any], text: str) -> List[str]:
    """Import names of installed distributions the text names (by import name or
    distribution name, whole word, case-insensitive)."""
    low = (text or "").lower()
    out: List[str] = []
    for v in env.get("dists", {}).values():
        names = {v["name"].lower()} | {m.lower() for m in v["imports"]}
        if any(len(nm) >= 3 and re.search(r'(?<![a-z0-9_])' + re.escape(nm) + r'(?![a-z0-9_])', low)
               for nm in names):
            out += [m for m in v["imports"] if m not in out]
    return out


def render_environment(env: Dict[str, Any], rnd: int, prev: Optional[Dict[str, Any]],
                       focus: List[str], budget: int = ENV_SECTION_BUDGET) -> str:
    """The container's abilities for agents and test generators. Distributions
    the brief or the project touch are listed first with summaries; the rest
    follow compactly, and if the budget runs out, by import name only - the
    list of what is importable is never cut."""
    dists = env.get("dists", {})
    lines = [f"CONTAINER ENVIRONMENT (discovered in the evaluation interpreter; scan for round {rnd:02d})",
             f"  Python {env.get('python', '?')}. {len(dists)} installed distribution(s) provide "
             f"{len(env.get('allowed', []))} importable top-level module(s).",
             "  This is not a fixed list: it is rescanned every round, because the container's owner",
             "  can install more. Use only the standard library and the modules listed here; an",
             "  import of anything else rejects the attempt, and nothing is installed on request.",
             "  Prefer what the task needs; availability is not a reason to use a library."]
    if prev:
        added, removed, upgraded = env_diff(prev, env)
        if added:
            lines.append("  NEW since the previous scan: " + ", ".join(added[:30]))
        if removed:
            lines.append("  REMOVED since the previous scan (no longer importable): " + ", ".join(removed[:30]))
        if upgraded:
            lines.append("  CHANGED version: " + ", ".join(upgraded[:20]))
    if "pyqrack" in env.get("allowed", []):
        lines += ["  pyqrack is installed: every Python file that imports it must contain this line",
                  f'  verbatim and export it to os.environ before the import: QRACK_LIB_PATH = "{QRACK_LIB_PATH}"']
    focus_set = set(focus)
    ordered = sorted(dists.values(), key=lambda v: (not (set(v["imports"]) & focus_set), v["name"].lower()))
    detail, rest = [], []
    used = sum(len(l) + 1 for l in lines)
    for v in ordered:
        imp = ",".join(v["imports"][:6]) + ("..." if len(v["imports"]) > 6 else "")
        ln = f"  {v['name']} {v['version']} [import {imp}]" + (f": {v['summary']}" if v["summary"] else "")
        if used + len(ln) + 1 <= budget * 0.7:
            detail.append(ln[:200])
            used += len(ln[:200]) + 1
        else:
            rest.extend(v["imports"])
    lines += detail
    loose = env.get("loose", [])
    if rest:
        lines.append("  more installed (import names): " + ", ".join(sorted(rest)))
    if loose:
        lines.append("  importable modules outside any distribution: " + ", ".join(loose))
    return enforce_ascii("\n".join(lines))


def dependency_section() -> str:
    return "\n".join([
        "DEPENDENCIES (discovered from the container, not configured)",
        "- Use only the standard library and the modules in the CONTAINER ENVIRONMENT block, which",
        "  the pipeline rescans every round. Importing anything else - even inside try/except as an",
        "  optional backend - rejects the attempt. Nothing is installed on request.",
        "- For every library you call, the LIBRARY API FACTS block lists what actually exists.",
        "  Call nothing else, even if you remember it from another version.",
    ])


def refresh_environment(run_dir: Path, rnd: int, focus_text: str = "",
                        project: Optional[Path] = None, quiet: bool = False) -> None:
    """Rescan the container and rebuild the environment block, the import gate
    and the library API facts. Called before the contract and before each round."""
    global _RUN_ENV, _RUN_ENV_TEXT, _RUN_ALLOWED_IMPORTS, _RUN_API_FACTS_TEXT
    new = scan_environment(run_dir)
    if not new:
        return
    prev = _RUN_ENV or load_last_env(run_dir)
    edir = run_dir / "env"
    edir.mkdir(parents=True, exist_ok=True)
    with open(edir / f"round{rnd:02d}.json", "w", encoding="ascii") as f:
        json.dump(new, f, indent=1, ensure_ascii=True)
    refs = project_library_refs(project, set(new["allowed"])) if project else {}
    focus = _mentioned_modules(new, focus_text) + [m for m in refs.get("modules", []) if m not in focus_text]
    _RUN_ENV = new
    _RUN_ENV_TEXT = render_environment(new, rnd, prev if prev else None, focus)
    if ENFORCE_DEPENDENCIES:
        _RUN_ALLOWED_IMPORTS = list(new["allowed"])
    facts = probe_library_api(run_dir, focus, refs)
    _RUN_API_FACTS_TEXT = render_api_facts(facts)
    with open(edir / f"round{rnd:02d}_api_facts.md", "w", encoding="ascii") as f:
        f.write((_RUN_API_FACTS_TEXT or "(no library API facts)") + "\n")
    if quiet:
        return
    added, removed, upgraded = env_diff(prev, new) if prev else ([], [], [])
    print(f"[ENV] round {rnd:02d} scan: Python {new.get('python')}, {len(new['dists'])} distribution(s), "
          f"{len(new['allowed'])} importable module(s)"
          + (f" | new: {', '.join(added[:6])}" if added else "")
          + (f" | removed: {', '.join(removed[:6])}" if removed else "")
          + (f" | changed: {', '.join(upgraded[:4])}" if upgraded else ""), flush=True)
    if facts:
        bad = sum(len(v.get("missing", [])) for v in facts.values())
        print(f"[API FACTS] inspected {', '.join(sorted(facts))}"
              + (f" | {bad} reference(s) in the project do not exist" if bad else ""), flush=True)


def load_last_env(run_dir: Path) -> Dict[str, Any]:
    edir = run_dir / "env"
    scans = sorted(p for p in edir.glob("round*.json")) if edir.exists() else []
    if not scans:
        return {}
    try:
        return json.loads(read_file_content_safe(scans[-1]) or "{}")
    except json.JSONDecodeError:
        return {}


def project_library_refs(project: Optional[Path], allowed: Set[str]) -> Dict[str, Any]:
    """Third-party usage in a project, from its source: modules imported, names
    imported from them, and attribute chains on module aliases."""
    out = {"modules": [], "names": [], "chains": []}
    if project is None or not project.exists():
        return out
    names, chains, mods = set(), set(), set()
    for p in sorted(project.rglob("*.py")):
        if "__pycache__" in p.parts or ".home" in p.parts:
            continue
        try:
            tree = _quiet_parse(read_file_content_safe(p) or "")
        except (SyntaxError, ValueError):
            continue
        alias: Dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    root = a.name.split(".")[0]
                    if root in allowed:
                        mods.add(root)
                        alias[a.asname or root] = a.name if a.asname else root
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                root = node.module.split(".")[0]
                if root in allowed:
                    mods.add(root)
                    for a in node.names:
                        if a.name != "*":
                            names.add((node.module, a.name))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                parts, cur = [], node
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name) and cur.id in alias:
                    chains.add(tuple(alias[cur.id].split(".") + list(reversed(parts))))
    # keep only maximal chains (a.b.c covers a.b)
    maximal = {c for c in chains if not any(len(o) > len(c) and o[:len(c)] == c for o in chains)}
    out["modules"] = sorted(mods)
    out["names"] = sorted(names)[:80]
    out["chains"] = sorted(maximal)[:120]
    return out


_API_SCAN_SRC = r"""
import os, sys, json, inspect, enum, importlib
os.environ.setdefault("QRACK_LIB_PATH", sys.argv[1])
req = json.loads(sys.argv[2])
MAXN = int(sys.argv[3])
def sig(o):
    try:
        return str(inspect.signature(o)).replace("(self, ", "(").replace("(self)", "()")
    except (TypeError, ValueError):
        return "(...)"
def describe(o):
    if isinstance(o, type) and issubclass(o, enum.Enum):
        return {"kind": "enum", "members": {m.name: repr(m.value) for m in o}}
    if isinstance(o, type):
        mem = {}
        for n in sorted(dir(o)):
            if n.startswith("_"):
                continue
            v = getattr(o, n, None)
            mem[n] = sig(v) if callable(v) else None
        return {"kind": "class", "init": sig(o), "members": mem}
    if callable(o):
        return {"kind": "function", "init": sig(o)}
    return {"kind": type(o).__name__}
out = {}
mods = {}
def mod(name):
    if name not in mods:
        try:
            mods[name] = importlib.import_module(name)
        except Exception as e:
            mods[name] = e
    return mods[name]
for m in req["modules"]:
    r = out.setdefault(m, {"missing": [], "objects": {}})
    mo = mod(m)
    if isinstance(mo, Exception):
        r["import_error"] = f"{type(mo).__name__}: {str(mo)[:200]}"
        continue
    pub = [n for n in dir(mo) if not n.startswith("_")]
    r["public_count"] = len(pub)
    r["version"] = str(getattr(mo, "__version__", ""))
    if len(pub) <= MAXN:
        r["top"] = {n: (lambda o: ("class" if isinstance(o, type) else "function" if callable(o)
                                   else "module" if inspect.ismodule(o) else type(o).__name__))(getattr(mo, n, None))
                    for n in pub}
for modname, name in req["names"]:
    root = modname.split(".")[0]
    r = out.setdefault(root, {"missing": [], "objects": {}})
    mo = mod(modname)
    if isinstance(mo, Exception):
        r.setdefault("import_error", f"{type(mo).__name__}: {str(mo)[:200]}")
        continue
    if not hasattr(mo, name):
        try:
            importlib.import_module(modname + "." + name)
        except Exception:
            r["missing"].append(f"from {modname} import {name}")
        continue
    r["objects"][f"{modname}.{name}"] = describe(getattr(mo, name))
for chain in req["chains"]:
    root = chain[0]
    r = out.setdefault(root, {"missing": [], "objects": {}})
    mo = mod(root)
    if isinstance(mo, Exception):
        continue
    cur, path = mo, root
    for part in chain[1:]:
        nxt = getattr(cur, part, None)
        if nxt is None and inspect.ismodule(cur):
            try:
                nxt = importlib.import_module(path + "." + part)
            except Exception:
                nxt = None
        if nxt is None:
            r["missing"].append(f"{path}.{part}")
            break
        cur, path = nxt, path + "." + part
print(json.dumps(out))
"""


def probe_library_api(run_dir: Path, focus: List[str], refs: Dict[str, Any]) -> Dict[str, dict]:
    """Inspect, in the evaluation interpreter, the libraries in play: modules the
    brief names (top level), every name the project imports from a library
    (full member list for classes), and every attribute chain it uses."""
    mods = sorted(set(focus) | set(refs.get("modules", [])))
    if not mods:
        return {}
    req = {"modules": mods, "names": refs.get("names", []), "chains": [list(c) for c in refs.get("chains", [])]}
    py, venv_bin = ensure_test_venv(run_dir)
    home = run_dir / "tests" / ".envscan"
    home.mkdir(parents=True, exist_ok=True)
    rc, out, to = _run_limited([py, "-c", _API_SCAN_SRC, QRACK_LIB_PATH, json.dumps(req),
                                str(API_FACTS_MAX_MODULE_NAMES)], 180, home, _test_env(home, venv_bin))
    if rc != 0 or to:
        return {}
    try:
        return json.loads((out or "").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {}


# Member names that get a full signature even when the budget is tight.
_API_CORE_PREFIXES = ("out_", "in_", "prob", "measure", "reset", "m_all", "num_qubits", "mtrx",
                      "mcmtrx", "swap", "seed", "run", "apply", "get", "set")


def render_api_facts(facts: Dict[str, dict], budget: int = API_FACTS_BUDGET) -> str:
    """Member lists are always COMPLETE (the block says unlisted names do not
    exist). To fit the budget, detail is shed instead: first signatures beyond
    the short/core ones, then all signatures, then the module overviews."""
    if not facts:
        return ""
    text = ""
    for level in (2, 1, 0):
        text = _render_api_facts_level(facts, level)
        if len(text) <= budget:
            return text
    return text  # names only; over budget but still complete and truthful


def _render_api_facts_level(facts: Dict[str, dict], level: int) -> str:
    lines = ["LIBRARY API FACTS (inspected in the evaluation interpreter this round)",
             "  What these libraries actually contain. For every object listed, the member list is",
             "  complete: a name not listed does not exist - do not call it, even if you remember it."]
    missing = [x for r in facts.values() for x in r.get("missing", [])]
    if missing:
        lines.append("  DOES NOT EXIST (used by the current project - fix these first):")
        lines += [f"    - {x}" for x in missing[:40]]
    for m in sorted(facts):
        r = facts[m]
        if r.get("import_error"):
            lines.append(f"{m}: IMPORT FAILS here ({r['import_error'][:160]})")
            continue
        head = f"{m}" + (f" {r['version']}" if r.get("version") else "")
        if "top" in r and level >= 1:
            cls = [n for n, k in r["top"].items() if k == "class"]
            fns = [n for n, k in r["top"].items() if k == "function"]
            head += f": classes {', '.join(cls) or '-'}; functions {', '.join(fns) or '-'}"
        elif r.get("public_count"):
            head += (f": {r['public_count']} public names; the names this project uses are checked "
                     "(see DOES NOT EXIST)")
        lines.append(head)
        for full, info in r.get("objects", {}).items():
            k = info.get("kind")
            if k == "enum":
                lines.append(f"  {full} (enum): " + ", ".join(f"{a}={b}" for a, b in info["members"].items()))
            elif k == "function":
                lines.append(f"  {full}{info.get('init', '(...)')}")
            elif k == "class":
                mem = info.get("members") or {}
                if level == 2:
                    sig = [n for n, sg in mem.items() if sg is not None
                           and (len(n) <= 4 or n.startswith(_API_CORE_PREFIXES))]
                elif level == 1:
                    sig = [n for n, sg in mem.items() if sg is not None and len(n) <= 3]
                else:
                    sig = []
                rest = [n for n in mem if n not in sig]
                lines.append(f"  {full}{info.get('init', '(...)')}")
                lines += [f"      .{n}{mem[n]}" for n in sig]
                if rest:
                    lines.append(("      other members: " if sig else "      members: ") + ", ".join(rest))
            else:
                lines.append(f"  {full} ({k})")
    return enforce_ascii("\n".join(lines))


_STDLIB_MODULES: Set[str] = set(getattr(sys, "stdlib_module_names", ())) | set(sys.builtin_module_names) \
    | {"__future__"}


def _quiet_parse(source: str):
    """ast.parse without SyntaxWarnings (e.g. a LaTeX '\\_' in an agent's
    docstring) leaking to the console as '<unknown>:N' lines."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source)


def _imported_roots(source: str) -> Set[str]:
    """Top-level module names a Python source imports anywhere (absolute imports,
    plus importlib.import_module / __import__ with a literal name)."""
    try:
        tree = _quiet_parse(source)
    except (SyntaxError, ValueError):
        return set()
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                out.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and node.args:
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            arg = node.args[0]
            if name in ("import_module", "__import__") and isinstance(arg, ast.Constant) \
                    and isinstance(arg.value, str) and arg.value:
                out.add(arg.value.split(".")[0])
    return out


def disallowed_imports(node: dict, run_dir: Path) -> List[str]:
    """'module (file)' for every third-party import outside the probed list.
    Local modules are any .py stem or package directory in the project: this
    attempt's files, every sibling's work, and the required deliverables."""
    if _RUN_ALLOWED_IMPORTS is None:
        return []
    wroot = work_dir_for(run_dir)
    ndir = wroot / node.get("dir", "") / node["id"]
    if not ndir.exists():
        return []
    local: Set[str] = {PurePosixPath(d).stem for d in _RUN_DELIVERABLES}
    local |= {PurePosixPath(d).parts[0] for d in _RUN_DELIVERABLES if len(PurePosixPath(d).parts) > 1}
    for p in wroot.rglob("*"):
        if p.is_file() and p.suffix == ".py":
            local.add(p.stem)
        elif p.is_dir():
            local.add(p.name)
    allowed = set(_RUN_ALLOWED_IMPORTS) | _STDLIB_MODULES | local
    bad: Dict[str, str] = {}
    for p in sorted(ndir.rglob("*.py")):
        if not p.is_file() or p.is_symlink():
            continue
        for mod in sorted(_imported_roots(read_file_content_safe(p) or "")):
            if mod not in allowed and mod not in bad:
                bad[mod] = str(p.relative_to(ndir))
    return [f"{m} ({f})" for m, f in bad.items()]


def _mentions(text: str, deliverable: str) -> Optional[int]:
    """Position of the first mention of a deliverable (full path, or its basename
    when that is what the assignment wrote), else None."""
    best = None
    for token in {deliverable, PurePosixPath(deliverable).name}:
        m = re.search(r'(?<![A-Za-z0-9_.-])' + re.escape(token) + r'(?![A-Za-z0-9_-])', text)
        if m and (best is None or m.start() < best):
            best = m.start()
    return best


def deliverable_coverage(pieces: List[str], required: Dict[str, str]) -> Tuple[Dict[str, int], List[str]]:
    """Owner piece index per deliverable, and the deliverables nobody mentions.
    A piece owns the deliverable it mentions FIRST; a deliverable mentioned only
    as a later dependency is owned by the piece that mentions it earliest."""
    firsts: Dict[int, str] = {}
    for i, piece in enumerate(pieces):
        hits = [(pos, d) for d in required if (pos := _mentions(piece, d)) is not None]
        if hits:
            firsts[i] = min(hits)[1]
    owner: Dict[str, int] = {}
    for i, d in firsts.items():
        owner.setdefault(d, i)
    missing = []
    for d in required:
        if d in owner:
            continue
        cands = [(pos / max(1, len(pieces[i])), i) for i in range(len(pieces))
                 if (pos := _mentions(pieces[i], d)) is not None]
        if cands:
            owner[d] = min(cands)[1]
        else:
            missing.append(d)
    return owner, missing


def attach_deliverable_specs(pieces: List[str], required: Dict[str, str]) -> List[str]:
    """Give each owner the verbatim spec of every deliverable it owns, and add a
    dedicated assignment for any deliverable no piece mentions."""
    owner, missing = deliverable_coverage(pieces, required)
    pieces = list(pieces)
    for d in missing:
        pieces.append(f"Produce `{d}` exactly as specified below and in the PINNED CONTRACT.")
        owner[d] = len(pieces) - 1
    owned: Dict[int, List[str]] = {}
    for d, i in owner.items():
        owned.setdefault(i, []).append(d)
    out = []
    for i, piece in enumerate(pieces):
        ds = owned.get(i, [])
        if ds:
            specs = "\n\n".join(required[d] for d in ds if required.get(d))
            piece = (f"{piece}\n\nYOU OWN: {', '.join(ds)}\n"
                     f"VERBATIM SPEC FROM THE BRIEF (binding):\n{specs}")
        out.append(piece)
    return out


def brief_meta_path_for(run_dir: Path) -> Path:
    return run_dir / COMMS_DIRNAME / "brief_meta.json"


def save_brief_meta(run_dir: Path, meta: dict) -> None:
    path = brief_meta_path_for(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as f:
        json.dump(meta, f, indent=2, ensure_ascii=True)
    if meta.get("contract"):
        with open(run_dir / "CONTRACT.md", "w", encoding="ascii") as f:
            f.write(meta["contract"].rstrip() + "\n")


def load_brief_meta(run_dir: Path) -> Optional[dict]:
    path = brief_meta_path_for(run_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(read_file_content_safe(path) or "")
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def log_partition_failure(attempt: int, error: str, raw: Optional[str], user_content: str) -> None:
    """Keep the planner's raw reply for a failed partition attempt, so a 'Could not
    locate JSON array' can be diagnosed (fences, prose, truncation, refusal)."""
    if _RUN_DIR is None:
        return
    fdir = _RUN_DIR / COMMS_DIRNAME / "partition_failures"
    try:
        fdir.mkdir(parents=True, exist_ok=True)
        path = fdir / f"attempt_{attempt}.txt"
        body = raw if raw is not None else "(no reply: the apex call itself failed)"
        with open(path, "w", encoding="ascii", errors="replace") as f:
            f.write(f"error: {error}\n")
            f.write(f"prompt chars: {len(user_content):,} | reply chars: {len(raw or ''):,}"
                    f" | reply tokens cap: {APEX_PLAN_TOKENS}\n")
            if raw:
                opens, closes = raw.count("["), raw.count("]")
                f.write(f"brackets: [ x{opens}, ] x{closes}"
                        + ("  (unbalanced - likely cut off at the token cap)" if opens != closes else "")
                        + f" | code fences: {raw.count('```')}\n")
            f.write("\n===== RAW REPLY =====\n")
            f.write(enforce_ascii(body) + "\n")
            f.write("\n===== PROMPT SENT =====\n")
            f.write(enforce_ascii(user_content) + "\n")
        head = re.sub(r"\s+", " ", (raw or "").strip())[:160]
        print(f"        raw reply ({len(raw or ''):,} chars) -> {path.relative_to(_RUN_DIR)}"
              + (f" | starts: {head!r}" if head else ""), flush=True)
    except OSError:
        pass


def decompose_to_atomic_pieces(large_query: str, required: Optional[Dict[str, str]] = None,
                               contract: str = "") -> tuple:
    """Planning. Runs on APEX: it is neither an agent assignment nor a merge."""
    print(f"\n[PHASE 3] [1] INGRESS: Analyzing query...\n    Length: {len(large_query)} characters", flush=True)

    required = dict(required or {})
    fitted_query = fit_context(large_query, MAX_CONTEXT_CHARS)
    user_content = f"Partition this into mutually exclusive agent assignments:\n\n{fitted_query}"
    if required:
        user_content += (
            "\n\nREQUIRED DELIVERABLES (from the original brief). Every one of these must be owned by "
            "exactly one assignment, and that assignment must name it verbatim:\n"
            + "\n".join(f"- {d}" for d in required))
    if contract:
        user_content += ("\n\nINTERFACE CONTRACT (for orientation; every agent receives it verbatim):\n"
                         + fit_context(contract, 12000))
    base_content = user_content
    best_pieces: Optional[List[str]] = None
    best_missing: List[str] = []
    tokens = (0, 0)

    for attempt in range(1, MAX_RETRIES + 1):
        client = apex_client(timeout=WORKER_TIMEOUT_SECS)
        print(f"[2] PARTITION: Planning agent assignments via apex {GEN_API_BASE} [{LLM_MODEL}] "
              f"(Attempt {attempt}/{MAX_RETRIES})...", flush=True)
        raw_output = None
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
            if not required:
                return atomic_pieces, prompt_tokens, comp_tokens
            _, missing = deliverable_coverage(atomic_pieces, required)
            if best_pieces is None or len(missing) < len(best_missing):
                best_pieces, best_missing = atomic_pieces, missing
                tokens = (prompt_tokens, comp_tokens)
            if not missing:
                break
            print(f"    [!] Coverage: {len(missing)} required deliverable(s) unowned: "
                  f"{', '.join(missing)}", flush=True)
            if attempt < MAX_RETRIES:
                user_content = (base_content + "\n\nYOUR PREVIOUS PARTITION OMITTED THESE REQUIRED "
                                "DELIVERABLES - include an owner for each: " + ", ".join(missing))

        except Exception as e:
            print(f"    [!] Partition Error: {e}", flush=True)
            log_partition_failure(attempt, str(e), raw_output, user_content)
            time.sleep(2)

    if required:
        if best_pieces is None:
            print("    [!] Partitioning failed; falling back to one assignment per required deliverable.",
                  flush=True)
            best_pieces = []
        elif best_missing:
            print(f"    [*] Adding dedicated assignment(s) for: {', '.join(best_missing)}", flush=True)
        pieces = attach_deliverable_specs(best_pieces, required)[:MAX_DECOMPOSE_TASKS]
        owner, missing = deliverable_coverage(pieces, required)
        print(f"    [+] Deliverable coverage: {len(required) - len(missing)}/{len(required)} owned "
              f"across {len(pieces)} assignment(s).", flush=True)
        if missing:
            print(f"    [!] Still unowned after clamping to {MAX_DECOMPOSE_TASKS}: {', '.join(missing)}",
                  flush=True)
        return pieces, tokens[0], tokens[1]

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


_NUM_RE = re.compile(r'(?<![\d.])-?\d+\.\d{2,}(?:[eE][-+]?\d+)?(?![\d.])')


def ungrounded_numbers(writeup: str, evidence: str) -> List[str]:
    """Decimal numbers in a write-up that match no number in the run output or
    the files it wrote, allowing for rounding to the write-up's precision."""
    ev = []
    for tok in _NUM_RE.findall(evidence or "") + re.findall(r'-?\d+\.\d+(?:[eE][-+]?\d+)?', evidence or ""):
        try:
            ev.append(float(tok))
        except ValueError:
            pass
    out = []
    for tok in _NUM_RE.findall(writeup or ""):
        try:
            v = float(tok)
        except ValueError:
            continue
        decimals = len(tok.split(".")[1].split("e")[0].split("E")[0])
        tol = 0.5 * 10 ** (-decimals) + 1e-12
        if not any(abs(v - e) <= tol for e in ev) and tok not in out:
            out.append(tok)
    return out[:20]


def final_writeup_refresh(run_dir: Path, roster: List[dict], rnd: int, background: str,
                          review: str = "") -> None:
    """One extra call per write-up deliverable after the last grounding run: the
    owner rewrites it against the FINAL output, which no in-round attempt could
    see. Not recorded in the trees (it is not exploration and costs no budget);
    the refreshed file replaces the one in integration/latest/."""
    if not FINAL_WRITEUP_REFRESH or _shutdown_event.is_set():
        return
    idir = integration_dir_for(run_dir)
    run_json = idir / f"round{rnd:02d}_run.json"
    if not run_json.exists() or not _RUN_LAST_RUN_TEXT:
        return
    writeups = [d for d in _RUN_DELIVERABLES if d.lower().endswith(".md")]
    if not writeups:
        return
    owner_idx, _ = deliverable_coverage([r["objective"] for r in roster], {d: "" for d in writeups})
    best = best_known_nodes(run_dir, rnd)
    live = [n for _, nodes in load_pool(run_dir) for n in nodes if n.get("task")]
    run_info = json.loads(read_file_content_safe(run_json) or "{}")
    evidence_dir = run_dir / "tests" / f"round{rnd:02d}" / "grounding" / "project"
    evidence = _RUN_LAST_RUN_TEXT
    for rel in run_info.get("produced", []):
        evidence += "\n" + (read_file_content_safe(evidence_dir / rel) or "")[:20000]
    print(f"\n[WRITE-UP] Final refresh against round {rnd:02d}'s run "
          f"({'succeeded' if run_info.get('ok') else 'failed'}): {', '.join(writeups)}", flush=True)
    slot_queue, _ = build_worker_slot_queue(prefix="F-Slot")
    summary = []
    for d in writeups:
        i = owner_idx.get(d)
        if i is None:
            print(f"    [!] No owner found for {d}; skipped.", flush=True)
            continue
        agent = roster[i]
        parent = best.get(agent["id"])
        if parent is None:
            print(f"    [!] {agent['id']} has no successful attempt to refresh; skipped.", flush=True)
            continue
        extra = (
            "FINAL WRITE-UP REFRESH. Every round is finished and the pipeline has just run the final "
            "integrated project; its real output is in LATEST REAL RUN OF THE INTEGRATED PROJECT below. "
            f"Rewrite {d} so that every number, table and claim about results matches that output "
            "exactly - and states nothing it does not show. Quote numbers as printed. If the run "
            "FAILED, say so plainly and report no results. Keep the prompt's honesty rules: hardware "
            f"figures marked unverified, a null result reported as-is. Emit ONLY {d}."
            + (" If the run output shows a NEGATIVE CONTROL verdict, report it; if it is INSENSITIVE, "
               "SCORE-ONLY or HARDCODED, say the metrics do not measure what their names claim and present "
               "no finding. Never quote the output's UNTRUSTED FREE TEXT fields as results."
               if _RUN_PROBE_COMMAND else "")
            + ("\n\nSKEPTIC REVIEW OF THE METRICS (an apex model read the final code and output; its "
               "judgment, not proof). Every metric marked 'suspect' or 'invalid' below MUST appear in the "
               "write-up as an explicit caveat, with the reason, next to any number it concerns:\n"
               + fit_context(review, 6000) if review else ""))
        endpoint, slot_name = slot_queue.get()
        node_id = f"r{rnd:02d}final{i + 1:02d}"
        try:
            node = run_agent(agent, roster, rnd, node_id, 900000 + i, parent["id"],
                             parent.get("depth", 1) + 1, endpoint, slot_name, background, run_dir,
                             live, threading.Lock(), parent_task_node=parent,
                             reference_hashes=set(parent.get("file_hashes", [])), extra_stage=extra)
        except Exception as exc:
            print(f"    [!] Refresh of {d} raised {str(exc)[:100]}", flush=True)
            continue
        finally:
            slot_queue.put((endpoint, slot_name))
        src = work_dir_for(run_dir) / agent["dir"] / node_id / d
        if node.get("status") not in ("success", "partial") or not src.is_file():
            print(f"    [!] Refresh of {d} produced no file (status {node.get('status')}); "
                  f"integration/latest keeps the previous version.", flush=True)
            continue
        text = read_file_content_safe(src) or ""
        dest = idir / "latest" / d
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        missing = ungrounded_numbers(text, evidence)
        summary.append({"deliverable": d, "agent": agent["id"], "node": node_id,
                        "parent": parent["id"], "ungrounded_numbers": missing})
        print(f"    [+] {d} refreshed by {agent['id']} ({node_id}) -> "
              f"{INTEGRATION_DIRNAME}/latest/{d}", flush=True)
        if missing:
            print(f"    [!] {len(missing)} number(s) in {d} match nothing in the run output: "
                  f"{', '.join(missing[:8])}", flush=True)
        append_event(run_dir, {"round": rnd, "event": "final_writeup_refresh", "deliverable": d,
                               "node": node_id, "ungrounded_numbers": missing})
    with open(idir / "final_writeup_refresh.json", "w", encoding="ascii") as f:
        json.dump({"round": rnd, "run_ok": run_info.get("ok"), "refreshed": summary}, f, indent=2)


def after_round_grounding(run_dir: Path, rnd: int, test_ctx: dict) -> None:
    """Freeze the integrated project's API and run it; both feed the next round."""
    global _RUN_FROZEN_API, _RUN_FROZEN_API_TEXT, _RUN_LAST_RUN_TEXT
    proj = integration_dir_for(run_dir) / f"round{rnd:02d}"
    if not proj.exists():
        return
    idir = integration_dir_for(run_dir)
    if FREEZE_INTERFACES:
        try:
            api, text = freeze_project_api(proj, rnd)
            _RUN_FROZEN_API, _RUN_FROZEN_API_TEXT = api, text
            with open(idir / f"round{rnd:02d}_api.json", "w", encoding="ascii") as f:
                json.dump(api, f, indent=2, ensure_ascii=True)
            with open(idir / f"round{rnd:02d}_api.md", "w", encoding="ascii") as f:
                f.write(text + "\n")
            used = sum(1 for m in api.values() for i in m.values() if i.get("used_by"))
            unres = text.count("\n  - ")
            print(f"[API] ROUND {rnd:02d}: {sum(len(m) for m in api.values())} public name(s) in "
                  f"{len(api)} module(s), {used} used across modules, {unres} unresolved "
                  f"-> frozen for round {rnd + 1:02d}", flush=True)
        except Exception as exc:
            print(f"    [!] Interface freeze failed: {str(exc)[:120]}", flush=True)
    if _RUN_COMMAND:
        try:
            res = grounding_run(proj, rnd, test_ctx, run_dir)
            if res is not None:
                _RUN_LAST_RUN_TEXT = read_file_content_safe(idir / f"round{rnd:02d}_run.md") or ""
        except Exception as exc:
            print(f"    [!] Grounding run failed: {str(exc)[:120]}", flush=True)


def run_online_round(roster: List[dict], rnd: int, background: str, run_dir: Path,
                     policy_source: str, budget: int,
                     semantic_guidance: bool = False) -> Tuple[List[dict], dict]:
    """Deploy the policy online. It drives the agents in decision rounds; every
    attempt is scored by the fixed evaluator at creation; the tree it grows is
    the world the next offline phase dreams in."""
    tasks = [r["id"] for r in roster]
    slot_queue, slot_count = build_worker_slot_queue(prefix="A-Slot")
    reserve = support_reserve(budget, len(tasks))

    print(f"\n[3] ROUND {rnd:02d}: deploying {policy_path(run_dir, rnd).name} over "
          f"{len(tasks)} assignment(s), budget {budget} agent call(s)"
          + (f" ({budget - reserve} policy + {reserve} support)" if reserve else "")
          + f", W {MAX_PARALLELISM}, K1 {ONLINE_MAX_DECISION_ROUNDS}, "
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

    test_ctx = (new_test_context(run_dir, roster)
                if (EVAL_INLINE_TESTS or EVAL_INTEGRATION) else None)
    explorer = LiveExplorer(tasks, budget - reserve, roster, rnd, background, run_dir,
                            slot_queue, MAX_PARALLELISM, ONLINE_MAX_DECISION_ROUNDS,
                            prior_hashes=prior_hashes, semantic_guidance=semantic_guidance,
                            test_ctx=test_ctx)
    start = time.time()
    ok, detail = run_policy(policy_source, explorer)
    print()

    if not ok and _shutdown_event.is_set():
        print(f"    [!] Policy stopped for shutdown ({detail}).", flush=True)
    elif not ok:
        print(f"    [!] Deployed policy failed ({detail}). Falling back to pi_0 for the "
              f"remaining budget and decision rounds.", flush=True)
        append_event(run_dir, {"round": rnd, "event": "policy_failure", "detail": detail})
        fb_ok, fb_detail = run_policy(DEFAULT_POLICY_SOURCE, explorer)
        print()
        if explorer._stop is None:
            explorer._stop = ("pi_0 fallback returned (empty batch)" if (fb_ok and fb_detail == "ok")
                              else f"pi_0 fallback: {fb_detail}")
        explorer._stop = f"{explorer._stop} [deployed policy failed: {detail[:80]}]"
    elif detail != "ok":
        print(f"    [~] Policy {detail}", flush=True)
    if explorer._stop is None:
        explorer._stop = "policy returned (empty batch)" if (ok and detail == "ok") else detail
    policy_rounds, policy_stop = explorer._rounds, explorer._stop

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

    nodes = explorer._real()
    elapsed = time.time() - start
    print(f"    [+] Round {rnd:02d} online phase complete: {len(nodes)} node(s), "
          f"{explorer.spent()}/{budget} agent call(s), {policy_rounds} decision round(s), "
          f"stop: {policy_stop}, {elapsed:.1f}s.", flush=True)

    rdir = round_dir_for(run_dir, rnd)
    rdir.mkdir(parents=True, exist_ok=True)
    with open(rdir / "online_trace.json", "w", encoding="ascii") as f:
        json.dump({"round": rnd, "policy": policy_path(run_dir, rnd).name,
                   "max_parallelism": MAX_PARALLELISM, "max_rounds": ONLINE_MAX_DECISION_ROUNDS,
                   "decision_rounds": policy_rounds, "stop": policy_stop,
                   "trace": explorer._trace}, f, indent=2)

    test_results = list(test_ctx["results"]) if test_ctx else []
    write_round_test_reports(run_dir, rnd, test_results)

    integ = None
    if EVAL_INTEGRATION and test_ctx is not None and not _shutdown_event.is_set():
        try:
            integ = integration_round_report(run_dir, rnd, roster, test_ctx)
        except Exception as exc:
            print(f"    [!] Round integration failed: {str(exc)[:120]}", flush=True)
    if integ and not _shutdown_event.is_set():
        after_round_grounding(run_dir, rnd, test_ctx)

    report, stats = reconcile_round(run_dir, rnd, roster, nodes)
    stats["integration_q"] = integ["q"] if integ else None
    stats["spent"] = explorer.spent()
    stats["support_spent"] = support_spent
    stats["decision_rounds"] = policy_rounds
    stats["stop"] = policy_stop
    stats["elapsed"] = round(elapsed, 2)
    print(f"    [+] Reconciled round {rnd:02d}: {stats['files']} file(s), "
          f"{stats['dup_content']} duplicate artifact(s), "
          f"{stats['violations']} scope violation(s), mean best score {stats['best_mean']:.3f}.",
          flush=True)

    return nodes, stats


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
    lines.append("| Round | Policy | Agent calls | Decision rounds | Nodes | Mean best score | Duplicates | Violations |")
    lines.append("|-------|--------|-------------|-----------------|-------|-----------------|------------|------------|")
    for i, s in enumerate(round_stats, start=1):
        lines.append(f"| {i} | `{POLICY_DIRNAME}/pi_r{i:02d}.py` | {s.get('spent', 0)} | {s.get('decision_rounds', '-')} | "
                     f"{s.get('nodes', 0)} | {s.get('best_mean', 0):.3f} | "
                     f"{s.get('dup_content', 0)} | {s.get('violations', 0)} |")
    lines.append("")

    if dream_stats:
        lines.append("## Dreaming (offline policy improvement)")
        lines.append("")
        lines.append("| After round | Versions | Valid | Selected | Replay score | Deployed version | Changed |")
        lines.append("|-------------|------------|-------|--------|--------------|---------------|---------|")
        for d in dream_stats:
            lines.append(f"| {d['round']} | {d['candidates']} | {d['valid']} | {d['winner']} | "
                         f"{d['winner_score']:.4f} | {d['baseline_score']:.4f} | "
                         f"{'yes' if d['improved'] else 'no'} |")
        lines.append("")
        lines.append("Replay costs zero agent calls: every outcome a policy version can reveal "
                     "is already recorded. Versions are chained (each revises its predecessor) and "
                     "the deployed policy is version 0, so the selected policy is never worse than "
                     "it in mean replay score on the recorded history.")
        lines.append("")

    idir = integration_dir_for(run_dir)
    ireports = []
    for rp in sorted(idir.glob("round[0-9][0-9].json")) if idir.exists() else []:
        try:
            ireports.append(json.loads(read_file_content_safe(rp) or "{}"))
        except json.JSONDecodeError:
            continue
    if ireports:
        lines.append("## Integration (assembled project)")
        lines.append("")
        lines.append("| Round | q | compile | import | pytest | command | command score | missing tasks | conflicts |")
        lines.append("|-------|---|---------|--------|--------|---------|---------------|---------------|-----------|")
        for ir in ireports:
            g = ir.get("groups", {})
            fmt = lambda k: f"{g[k]:.2f}" if k in g else "-"
            q = ir.get("q")
            lines.append(f"| {ir.get('round', 0)} | {q:.3f} | " if q is not None else f"| {ir.get('round', 0)} | - | ")
            lines[-1] += (f"{fmt('compile')} | {fmt('import')} | {fmt('pytest')} | {fmt('command')} | "
                          f"{ir.get('command_score') if ir.get('command_score') is not None else '-'} | "
                          f"{', '.join(ir.get('missing_tasks', [])) or '-'} | {len(ir.get('conflicts', []))} |")
        lines.append("")
        lines.append(f"Latest assembled project: `{INTEGRATION_DIRNAME}/latest/` "
                     "(best known deliverable of every task, agent prefixes removed).")
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
# Phase 5: Unit tests as part of the fixed evaluator
# ------------------------------------------------------------------------------
# Every attempt's testable deliverables are tested at creation, inside the same
# generation-evaluation request (evaluate_node_inline); the pass rate enters the
# node's final score. At the end of a round the executions are written as
# reports for Phase 6.
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
        if _RUN_ALLOWED_IMPORTS is not None and \
                name not in ({a.lower().replace("_", "-") for a in _RUN_ALLOWED_IMPORTS}
                             | set(_RUN_ENV.get("dists", {}))):
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


def new_test_context(run_dir: Path, roster: Optional[List[dict]] = None) -> dict:
    """Per-round evaluator state shared by the concurrent attempts: the test
    venv, a cache of generated tests keyed by (content, name, language), the set
    of already-installed requirement lines, and the collected results."""
    py, venv_bin = ensure_test_venv(run_dir)
    return {"python": py, "venv_bin": venv_bin, "cache": {}, "cache_lock": threading.Lock(),
            "install_lock": threading.Lock(), "installed": set(), "results": [],
            "results_lock": threading.Lock(), "roster": list(roster or []),
            "integration_cache": {}}


def install_node_requirements(run_dir: Path, node: dict, rnd: int, test_ctx: dict) -> None:
    """Install ONLY the requirements declared by this attempt, after sanitising a
    COPY (deliverables are never rewritten), as binary wheels only so no sdist
    build script runs. Serialised: one pip at a time per run."""
    venv_bin = test_ctx.get("venv_bin")
    if not TEST_PIP_INSTALL or venv_bin is None:
        return
    ndir = work_dir_for(run_dir) / node.get("dir", "") / node["id"]
    if not ndir.exists():
        return
    kept: List[str] = []
    dropped: List[str] = []
    for req in sorted(ndir.rglob("requirements*.txt")):
        k, d = sanitize_requirements(read_file_content_safe(req) or "")
        kept.extend(x for x in k if x not in kept)
        dropped.extend(d)
    if dropped:
        print(f"\n    [!] {node['id']}: dropped {len(dropped)} requirement line(s) that were not "
              f"plain index packages{' or not allowlisted' if TEST_PIP_ALLOWLIST else ''}.",
              flush=True)
    with test_ctx["install_lock"]:
        todo = [k for k in kept if k not in test_ctx["installed"]]
        if not todo:
            return
        test_root = run_dir / "tests" / f"round{rnd:02d}" / node["task"] / node["id"]
        test_root.mkdir(parents=True, exist_ok=True)
        req_copy = test_root / "requirements.sanitized.txt"
        with open(req_copy, "w", encoding="ascii") as f:
            f.write("\n".join(todo) + "\n")
        env = _test_env(run_dir / "tests", venv_bin)
        rc, out, timed_out = _run_limited(
            [test_ctx["python"], "-m", "pip", "install", "--only-binary=:all:", "--no-input",
             "--disable-pip-version-check", "-r", str(req_copy)], 600, test_root, env)
        if rc == 0:
            test_ctx["installed"].update(todo)
        else:
            tail = (out.strip().splitlines() or ["(no output)"])[-1]
            print(f"\n    [!] Warning: pip install {'timed out' if timed_out else 'failed'} "
                  f"for {node['id']}: {tail[:120]}", flush=True)


def _test_filename(artifact_name: str) -> str:
    stem, suffix = Path(artifact_name).stem, Path(artifact_name).suffix.lower()
    if suffix == ".h":
        return f"test_{stem}_h.c"
    if suffix == ".hpp":
        return f"test_{stem}_hpp.cpp"
    return f"test_{stem}{suffix}"


def collect_node_artifacts(run_dir: Path, node: dict) -> List[dict]:
    """Testable files of one attempt (its own and inherited deliverables)."""
    ndir = work_dir_for(run_dir) / node.get("dir", "") / node["id"]
    if not ndir.exists():
        return []
    wroot = work_dir_for(run_dir)
    artifacts = []
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
            "agent": node["task"], "node": node["id"], "filename": path.name,
            "relative_path": str(path.relative_to(wroot)), "language": lang,
            "filepath": str(path), "content": content, "content_hash": content_hash(content),
        })
    return artifacts


def generate_unittest(artifact: dict, endpoint: str) -> Optional[str]:
    """One evaluator call on the worker slot the attempt already holds. Returns
    test SOURCE, or None."""
    url = endpoint.rstrip("/") + "/chat/completions"
    code_content = fit_context(artifact["content"], MAX_CONTEXT_CHARS)
    lang = artifact["language"]
    extra = ""
    if lang in ("c", "cpp"):
        extra = (f"\nInclude it exactly as: #include \"{artifact['filename']}\". "
                 f"If the file defines its own main(), it is renamed to "
                 f"autoresearch_artifact_main() before compiling, so write your own main().")
    prompt = f"File: {artifact['filename']}{extra}\n```{lang}\n{code_content}\n```"
    if _RUN_CONTRACT:
        prompt = (f"CONTRACT (binding; test against it):\n"
                  f"{fit_context(_RUN_CONTRACT, TEST_CONTRACT_BUDGET, note='...[CONTRACT TRUNCATED]...')}"
                  f"\n\n{prompt}")
    if _RUN_ENV_TEXT:
        prompt = (f"{fit_context(_RUN_ENV_TEXT, 3000)}\n\n"
                  + (f"{fit_context(_RUN_API_FACTS_TEXT, 3000)}\n\n" if _RUN_API_FACTS_TEXT else "")
                  + "Tests may import only the standard library, the project's modules and the libraries "
                    "above.\n\n" + prompt)
    payload = {
        "model": WORKER_MODEL,
        "messages": [{"role": "system", "content": _PROMPT_PHASE5_UNITTEST},
                     {"role": "user", "content": prompt}],
        "temperature": LLM_TEMPERATURE, "top_p": LLM_TOP_P,
        "frequency_penalty": LLM_FREQUENCY_PENALTY, "presence_penalty": LLM_PRESENCE_PENALTY,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    headers = {"Authorization": f"Bearer {WORKER_API_KEY}"}
    for attempt in range(1, MAX_RETRIES + 1):
        if _shutdown_event.is_set():
            return None
        try:
            _strip_known_rejects(endpoint.rstrip("/"), payload)
            _t_call = time.time()
            response = requests.post(url, json=payload, headers=headers, timeout=TEST_TIMEOUT_SECS)
            if response.status_code == 400:
                text = response.text
                try:
                    err = response.json().get("error", {})
                    text = (err.get("message") if isinstance(err, dict) else err) or text
                except ValueError:
                    pass
                _log_bad_request(endpoint, str(text))
                fix = _fix_payload_for_400(str(text), payload, endpoint.rstrip("/"))
                if not fix:
                    return None          # a 400 will not heal on retry
                print(f"    [*] retrying {endpoint}: {fix}", flush=True)
                continue
            response.raise_for_status()
            body = response.json()
            choices = body.get("choices")
            test_code = choices[0].get("message", {}).get("content", "") if choices else ""
            usage = body.get("usage") or {}
            _LEDGER.add("unit-test generation", "agent",
                        usage.get("prompt_tokens") or estimate_tokens(_PROMPT_PHASE5_UNITTEST + prompt),
                        usage.get("completion_tokens") or estimate_tokens(test_code or ""),
                        time.time() - _t_call, None, not usage, rnd=getattr(_TOKEN_CTX, "rnd", None),
                        truncated=bool(choices and choices[0].get("finish_reason") == "length"))
            if test_code:
                return enforce_ascii(_strip_markdown_fences(test_code))
        except (requests.exceptions.RequestException, ValueError):
            pass
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER))
    return None


# ------------------------------------------------------------------
# Project-level integration (evaluator part 3)
# ------------------------------------------------------------------
# Per-file tests cannot see whether modules written by different agents fit
# together. Integration assembles one flat project (agent directory prefixes
# removed) and checks it as a whole. Weights apply to the groups that exist.
_INTEGRATION_WEIGHTS = {"coverage": 0.20, "compile": 0.10, "import": 0.15, "pytest": 0.35,
                        "command": 0.20}
_PYTEST_COUNT_RE = re.compile(r'(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed)\b')
_CMD_SCORE_RE = re.compile(r'"score"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)')
_COMPILE_SNIPPET = ("import sys\n"
                    "src = open(sys.argv[1], encoding='utf-8', errors='replace').read()\n"
                    "compile(src, sys.argv[1], 'exec')\n")


def integration_dir_for(run_dir: Path) -> Path:
    return run_dir / INTEGRATION_DIRNAME


def _project_rel(rel: str, dir_names: Set[str]) -> Optional[str]:
    """Path a node file takes inside the assembled project, or None to skip it."""
    parts = [x for x in PurePosixPath(rel.replace("\\", "/")).parts if x not in ("", ".", "/")]
    if not parts or parts[0] == "claimed" or ".." in parts:
        return None
    while len(parts) > 1 and (parts[0] == WORK_DIRNAME or parts[0] in dir_names
                              or re.match(r'^t\d{2}(_|$)', parts[0])):
        parts = parts[1:]
    return "/".join(parts)


def _is_test_file(rel: str) -> bool:
    name = PurePosixPath(rel).name
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def best_known_nodes(run_dir: Path, upto_rnd: int) -> Dict[str, dict]:
    """Best recorded attempt per task over every round up to upto_rnd (ties go to
    the most recent). This is the sibling set an attempt is integrated against."""
    best: Dict[str, Tuple[tuple, dict]] = {}
    for prnd, nodes in load_pool(run_dir):
        if prnd > upto_rnd:
            continue
        for n in nodes:
            if not n.get("task") or not n.get("files") or n.get("status") not in ("success", "partial"):
                continue
            key = (float(n.get("score", 0.0)), prnd, n.get("seq", 0))
            cur = best.get(n["task"])
            if cur is None or key > cur[0]:
                best[n["task"]] = (key, n)
    return {t: v[1] for t, v in best.items()}


def assemble_project(run_dir: Path, roster: List[dict], chosen: Dict[str, dict],
                     dest: Path) -> Tuple[Dict[str, str], List[str]]:
    """Copy each chosen node's files into one flat project. Roster order decides
    collisions; within a node the shallowest copy of a path wins."""
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    dir_names = {r["dir"] for r in roster}
    placed: Dict[str, str] = {}
    conflicts: List[str] = []
    for r in roster:
        node = chosen.get(r["id"])
        if node is None:
            continue
        files = _node_files(run_dir, node)
        for rel in sorted(files, key=lambda x: (x.count("/"), x)):
            target = _project_rel(rel, dir_names)
            if not target:
                continue
            if target in placed:
                if placed[target] != r["id"]:
                    conflicts.append(f"{target}: kept {placed[target]}, dropped {r['id']} ({node['id']})")
                continue
            out = dest / target
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="ascii", errors="ignore") as f:
                f.write(files[rel])
            placed[target] = r["id"]
    return placed, conflicts


def _pytest_rate(rc: Optional[int], out: str, timed_out: bool) -> Optional[float]:
    if timed_out:
        return 0.0
    summary = [l for l in out.splitlines() if _PYTEST_COUNT_RE.search(l)]
    counts: Dict[str, int] = {}
    if summary:
        for num, kind in _PYTEST_COUNT_RE.findall(summary[-1]):
            kind = "errors" if kind.startswith("error") else kind
            counts[kind] = counts.get(kind, 0) + int(num)
    denom = counts.get("passed", 0) + counts.get("failed", 0) + counts.get("errors", 0)
    if denom:
        return counts.get("passed", 0) / denom
    if rc == 5:          # no tests collected: the group does not exist
        return None
    return 0.0 if rc != 0 else None


def run_integration_checks(proj: Path, own: Optional[Set[str]], test_ctx: dict,
                           home_root: Path) -> dict:
    """Checks the assembled project. `own` limits compile/import to one attempt's
    files (None = every file); pytest and the command always cover everything."""
    py = test_ctx["python"]
    venv_bin = test_ctx.get("venv_bin")
    env = _test_env(home_root, venv_bin, str(proj))
    all_py = sorted(str(x.relative_to(proj)).replace("\\", "/") for x in proj.rglob("*.py")
                    if x.is_file())
    scope = [r for r in all_py if own is None or r in own]
    groups: Dict[str, float] = {}
    details: Dict[str, List[str]] = {}

    # Without this, a project that compiles and imports but lacks its tests and
    # entry point would look perfect: missing deliverables are a failed check.
    if _RUN_DELIVERABLES:
        present = [d for d in _RUN_DELIVERABLES if (proj / d).is_file()]
        groups["coverage"] = len(present) / len(_RUN_DELIVERABLES)
        details["coverage"] = [f"missing {d}" for d in _RUN_DELIVERABLES if d not in present][:5]

    if scope:
        ok, fails = 0, []
        for r in scope:
            rc, out, to = _run_limited([py, "-c", _COMPILE_SNIPPET, r], 30, proj, env)
            if rc == 0 and not to:
                ok += 1
            else:
                fails.append(f"{r}: {_extract_error_line(out, 'python')}"[:200])
        groups["compile"] = ok / len(scope)
        details["compile"] = fails[:5]

    modules = [r[:-3] for r in scope
               if "/" not in r and not _is_test_file(r) and r not in ("conftest.py", "setup.py")
               and r[:-3].isidentifier()]
    if modules:
        ok, fails = 0, []
        for m in modules:
            rc, out, to = _run_limited([py, "-c", f"import {m}"], INTEGRATION_IMPORT_SECS, proj, env)
            if rc == 0 and not to:
                ok += 1
            else:
                fails.append(f"{m}: " + ("import timed out" if to else _extract_error_line(out, "python"))[:200])
        groups["import"] = ok / len(modules)
        details["import"] = fails[:5]

    tests = [r for r in all_py if _is_test_file(r)]
    if tests:
        rc, out, to = _run_limited(
            [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header", "--tb=line",
             "--rootdir", str(proj)] + tests,
            INTEGRATION_PYTEST_SECS, proj, env, cpu=max(TEST_CPU_SECS, INTEGRATION_PYTEST_SECS))
        rate = _pytest_rate(rc, out, to)
        if rate is not None:
            groups["pytest"] = rate
            fails = [l.strip() for l in out.splitlines()
                     if l.startswith(("FAILED", "ERROR")) or l.strip().startswith("E ")]
            details["pytest"] = (["pytest timed out"] if to else fails[:6]) or \
                ([out.strip().splitlines()[-1][:200]] if out.strip() and rate < 1.0 else [])

    command_score = None
    if _RUN_INTEGRATION_CMD:
        rc, out, to = _run_limited(["bash", "-c", _RUN_INTEGRATION_CMD], INTEGRATION_CMD_SECS,
                                   proj, env, cpu=max(TEST_CPU_SECS, INTEGRATION_CMD_SECS))
        groups["command"] = 1.0 if (rc == 0 and not to) else 0.0
        m = _CMD_SCORE_RE.findall(out)
        if m:
            try:
                command_score = float(m[-1])
            except ValueError:
                command_score = None
        if to:
            details["command"] = ["command timed out"]
        elif rc != 0:
            details["command"] = [f"exit {rc}: {_extract_error_line(out, 'python')}"[:200]]
        else:
            details["command"] = []

    wsum = sum(_INTEGRATION_WEIGHTS[k] for k in groups)
    q = (sum(_INTEGRATION_WEIGHTS[k] * v for k, v in groups.items()) / wsum) if wsum else None
    return {"q": round(q, 4) if q is not None else None,
            "groups": {k: round(v, 4) for k, v in groups.items()},
            "details": details, "command_score": command_score}


def _project_digest(proj: Path, own: Set[str]) -> str:
    h = hashlib.sha256()
    for x in sorted(proj.rglob("*")):
        if x.is_file():
            h.update(str(x.relative_to(proj)).encode())
            h.update(b"\0")
            h.update(x.read_bytes())
            h.update(b"\0")
    h.update(json.dumps(sorted(own)).encode())
    h.update(_RUN_INTEGRATION_CMD.encode())
    return h.hexdigest()


def build_node_project(node: dict, run_dir: Path, rnd: int, test_ctx: dict) -> Tuple[Path, Set[str]]:
    """This attempt plus the best known deliverable of every other task."""
    roster = test_ctx["roster"]
    chosen = {t: n for t, n in best_known_nodes(run_dir, rnd).items() if t != node["task"]}
    chosen[node["task"]] = node
    base = run_dir / "tests" / f"round{rnd:02d}" / "integration" / node["id"]
    proj = base / "project"
    placed, conflicts = assemble_project(run_dir, roster, chosen, proj)
    node["integration_siblings"] = {t: n["id"] for t, n in chosen.items() if t != node["task"]}
    node["integration_conflicts"] = conflicts[:5]
    return proj, {rel for rel, t in placed.items() if t == node["task"]}


def _cached_full_q(proj: Path, test_ctx: dict) -> Optional[float]:
    """Whole-project q (no own-file scoping), cached by project content."""
    key = "full:" + _project_digest(proj, set())
    with test_ctx["cache_lock"]:
        res = test_ctx["integration_cache"].get(key)
    if res is None:
        res = run_integration_checks(proj, None, test_ctx, proj.parent)
        with test_ctx["cache_lock"]:
            test_ctx["integration_cache"][key] = res
    return res["q"]


def integration_credit(node: dict, project: Path, run_dir: Path, rnd: int,
                       test_ctx: dict) -> Optional[float]:
    """What this attempt changed about the whole project: q_with - q_baseline,
    where the baseline swaps in the task's previous best (or leaves the task out
    when it has none). Both sides use the same siblings and the same unscoped
    checks, so a bug shared by every attempt cancels out. Mapped to [0, 1]
    around 0.5 (0.5 = no change) with EVAL_CREDIT_GAIN. First attempts and
    test-only attempts get the neutral 0.5."""
    chosen = {t: n for t, n in best_known_nodes(run_dir, rnd).items()}
    prior = chosen.get(node["task"])
    if prior is not None and prior.get("id") == node["id"]:
        prior = None
    own_py = [f for f in (node.get("files") or []) if f.endswith(".py")]
    # Neutral 0.5 (same scale as everyone else) when there is nothing fair to
    # compare against: a task's FIRST attempt (removing a module would reward
    # merely existing), and test-only attempts (a correct test that exposes a
    # sibling's bug lowers q - that must not count against its author).
    reason = None
    if prior is None:
        reason = "first attempt of this task"
    elif own_py and all(_is_test_file(PurePosixPath(f).name) for f in own_py):
        reason = "test-only attempt"
    if reason:
        node["integration_credit"] = 0.5
        node["integration_delta"] = None
        node["integration_credit_note"] = reason
        return 0.5
    q_with = _cached_full_q(project, test_ctx)
    base = project.parent / "baseline"
    assemble_project(run_dir, test_ctx["roster"], chosen, base)
    q_base = _cached_full_q(base, test_ctx)
    if q_with is None or q_base is None:
        return None
    delta = q_with - q_base
    credit = max(0.0, min(1.0, 0.5 + EVAL_CREDIT_GAIN * delta))
    node["integration_full_q"] = round(q_with, 6)
    node["integration_baseline_q"] = round(q_base, 6)
    node["integration_baseline"] = prior["id"]
    node["integration_delta"] = round(delta, 6)
    node["integration_credit"] = round(credit, 6)
    append_event(run_dir, {"round": rnd, "event": "credit", "agent": node["task"], "node": node["id"],
                           "q_with": q_with, "q_base": q_base, "baseline": node["integration_baseline"],
                           "delta": round(delta, 6)})
    return credit


def evaluate_node_integration(node: dict, proj: Path, own: Set[str], run_dir: Path, rnd: int,
                              test_ctx: dict) -> Optional[float]:
    key = _project_digest(proj, own)
    with test_ctx["cache_lock"]:
        res = test_ctx["integration_cache"].get(key)
    if res is None:
        res = run_integration_checks(proj, own, test_ctx, proj.parent)
        with test_ctx["cache_lock"]:
            test_ctx["integration_cache"][key] = res
    node["integration_q"] = res["q"]
    node["integration"] = {"groups": res["groups"], "command_score": res["command_score"],
                           "details": {k: v[:3] for k, v in res["details"].items() if v}}
    append_event(run_dir, {"round": rnd, "event": "integration", "agent": node["task"],
                           "node": node["id"], "q": res["q"], "groups": res["groups"]})
    return res["q"]


def integration_round_report(run_dir: Path, rnd: int, roster: List[dict],
                             test_ctx: dict) -> Optional[dict]:
    """After a round: assemble the best known deliverable of every task into
    integration/roundNN/ (and integration/latest/), check it, and report."""
    chosen = best_known_nodes(run_dir, rnd)
    idir = integration_dir_for(run_dir)
    print(f"\n[INTEGRATION] ROUND {rnd:02d}", flush=True)
    if not chosen:
        print("    [!] No successful attempt with files yet; nothing to assemble.", flush=True)
        return None
    proj = idir / f"round{rnd:02d}"
    placed, conflicts = assemble_project(run_dir, roster, chosen, proj)
    res = run_integration_checks(proj, None, test_ctx, idir)
    missing = [r["id"] for r in roster if r["id"] not in chosen]
    report = {"round": rnd, "q": res["q"], "groups": res["groups"], "details": res["details"],
              "command": _RUN_INTEGRATION_CMD or None, "command_score": res["command_score"],
              "sources": {t: n["id"] for t, n in sorted(chosen.items())},
              "missing_tasks": missing, "conflicts": conflicts, "files": sorted(placed)}
    with open(idir / f"round{rnd:02d}.json", "w", encoding="ascii") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    latest = idir / "latest"
    shutil.rmtree(latest, ignore_errors=True)
    shutil.copytree(proj, latest, ignore=shutil.ignore_patterns("__pycache__", ".home"))

    grp = " | ".join(f"{k} {v:.2f}" for k, v in res["groups"].items()) or "no checks applied"
    cs = f" (command score {res['command_score']:g})" if res["command_score"] is not None else ""
    qs = f"{res['q']:.3f}" if res["q"] is not None else "n/a"
    print(f"    [+] Assembled {len(placed)} file(s) from {len(chosen)}/{len(roster)} task(s) "
          f"-> {INTEGRATION_DIRNAME}/round{rnd:02d}/", flush=True)
    print(f"    [+] {grp}{cs} -> q {qs}", flush=True)
    if missing:
        print(f"    [!] No usable attempt yet for: {', '.join(missing)}", flush=True)
    for c in conflicts[:3]:
        print(f"    [!] Path conflict: {c}", flush=True)
    shown = 0
    for group, lines in res["details"].items():
        for line in lines:
            if shown >= 8:
                break
            print(f"    [-] {group}: {line[:160]}", flush=True)
            shown += 1
    return report


def dependency_gate(node: dict, run_dir: Path, rnd: int) -> bool:
    """Part of the fixed evaluator, applied before any test or install. Returns
    True if the attempt imports a third-party module outside the contract's
    DEPENDENCIES list; it is then scored EVAL_DEP_REJECT_SCORE and not tested,
    since its tests would only measure the environment."""
    bad = disallowed_imports(node, run_dir)
    if bad and ENFORCE_DEPENDENCIES and time.time() - _RUN_ENV.get("_ts", 0) > 30:
        # Before rejecting, look again: the container's owner may have installed
        # the module since the last scan. A mid-round install counts at once.
        with _ENV_RESCAN_LOCK:
            if time.time() - _RUN_ENV.get("_ts", 0) > 30:
                refresh_environment(run_dir, rnd, focus_text=_RUN_BRIEF + "\n" + _RUN_CONTRACT, quiet=True)
                print(f"\n    [i] {node['id']}: imports {', '.join(bad)}; container rescanned before judging",
                      flush=True)
        bad = disallowed_imports(node, run_dir)
    if bad:
        # Rejected before any test or install: the attempt broke the enforced
        # DEPENDENCIES section, so its tests would measure the environment.
        msg = ("REJECTED by the evaluator: imports module(s) the container does not have: "
               f"{', '.join(bad)}. Only the standard library, the project's own modules and the "
               "modules listed in CONTAINER ENVIRONMENT (rescanned every round) are importable.")
        node["dependency_violations"] = bad
        node.setdefault("violations", []).append(msg[:300])
        node["test_pass_rate"] = 0.0
        node["test_failures"] = [msg[:200]]
        node["score_file_level"] = EVAL_DEP_REJECT_SCORE
        node["score"] = EVAL_DEP_REJECT_SCORE
        lp = node.get("log_path")
        if lp and (run_dir / lp).exists():
            with open(run_dir / lp, "a", encoding="ascii") as fh:
                fh.write(f"\n## Evaluator\n\n{enforce_ascii(msg)}\n")
        append_event(run_dir, {"round": rnd, "node": node["id"], "agent": node.get("task"),
                               "event": "dependency_reject", "modules": bad})
        print(f"\n    [-] {node['id']} ({node.get('task')}) rejected: imports {', '.join(bad)}",
              flush=True)
        return True
    return False


def evaluate_node_inline(node: dict, endpoint: str, run_dir: Path, rnd: int,
                         test_ctx: dict) -> None:
    """The fixed evaluator, applied once at creation: per-file unit tests (with
    every sibling's best deliverable importable), then project-level
    integration of this attempt with those siblings. Sets the final score."""
    project, own = None, set()
    if EVAL_INTEGRATION and test_ctx.get("roster"):
        try:
            project, own = build_node_project(node, run_dir, rnd, test_ctx)
        except Exception as exc:
            print(f"\n    [!] {node['task']} project assembly raised {str(exc)[:80]}", flush=True)
            project = None

    artifacts = (collect_node_artifacts(run_dir, node)[:EVAL_MAX_TEST_FILES]
                 if EVAL_INLINE_TESTS else [])
    if artifacts or project is not None:
        install_node_requirements(run_dir, node, rnd, test_ctx)
    test_root = run_dir / "tests" / f"round{rnd:02d}"
    node_dir = test_root / node["task"] / node["id"]
    venv_bin = test_ctx.get("venv_bin")
    results: List[dict] = []
    for a in artifacts:
        if _shutdown_event.is_set():
            break
        key = (a["content_hash"], a["filename"], a["language"])
        with test_ctx["cache_lock"]:
            code = test_ctx["cache"].get(key)
        if code is None:
            code = generate_unittest(a, endpoint)
            if code:
                with test_ctx["cache_lock"]:
                    test_ctx["cache"][key] = code
        if not code:
            continue
        node_dir.mkdir(parents=True, exist_ok=True)
        tpath = node_dir / _test_filename(a["filename"])
        with open(tpath, "w", encoding="ascii") as f:
            f.write(code + "\n")
        results.append(execute_test_artifact({
            "filename": tpath.name, "test_filepath": str(tpath), "language": a["language"],
            "artifact_filepath": a["filepath"], "agent": a["agent"], "node": a["node"],
            "python": test_ctx["python"], "venv_bin": str(venv_bin) if venv_bin else "",
            "test_root": str(test_root),
            "extra_pythonpath": str(project) if project is not None else "",
        }))

    rate = None
    if results:
        passed = sum(1 for r in results if r["status"] == "PASSED")
        rate = passed / len(results)
        node["test_pass_rate"] = round(rate, 4)
        node["test_count"] = len(results)
        node["tests_passed"] = passed
        node["test_failures"] = [f"{r['filename']}: {r['status']} {r['message']}"[:200]
                                 for r in results if r["status"] != "PASSED"][:5]
        with test_ctx["results_lock"]:
            test_ctx["results"].extend(results)
        for r in results:
            append_event(run_dir, {"round": rnd, "event": "test_result", "agent": r.get("agent", ""),
                                   "node": r.get("node", ""), "artifact": r.get("filename", ""),
                                   "status": r.get("status", "")})

    file_level = blend_test_score(node.get("heuristic_score", 0.0), rate, bool(node.get("files")))
    node["score_file_level"] = file_level
    q = None
    if project is not None and not _shutdown_event.is_set():
        try:
            q = evaluate_node_integration(node, project, own, run_dir, rnd, test_ctx)
        except Exception as exc:
            print(f"\n    [!] {node['task']} integration raised {str(exc)[:80]}", flush=True)
    if q is not None and EVAL_CREDIT and not _shutdown_event.is_set():
        try:
            credit = integration_credit(node, project, run_dir, rnd, test_ctx)
        except Exception as exc:
            print(f"\n    [!] {node['task']} credit assignment raised {str(exc)[:80]}", flush=True)
            credit = None
        if credit is not None:
            q = round((1.0 - EVAL_CREDIT_MIX) * q + EVAL_CREDIT_MIX * credit, 6)
            node["integration_blend"] = q
    node["score"] = (round((1.0 - EVAL_INTEGRATION_MIX) * file_level + EVAL_INTEGRATION_MIX * q, 6)
                     if q is not None else file_level)


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
            # The attempt's own directory first (its files shadow everything),
            # then the assembled project with every sibling's best deliverable.
            pp = [str(artifact_path.parent), str(test_path.parent)]
            if test_meta.get("extra_pythonpath"):
                pp.append(test_meta["extra_pythonpath"])
            env = _test_env(test_root, venv_bin, os.pathsep.join(pp))
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


def write_round_test_reports(run_dir: Path, rnd: int, execution_results: List[dict]) -> None:
    """Phase 5 reporting: the evaluator's test executions of this round, as a
    per-round report plus the cumulative one Phase 6 reads."""
    print(f"\n[PHASE 5] ROUND {rnd:02d}: EVALUATOR TEST TELEMETRY", flush=True)
    if not execution_results:
        print("    [!] No test executions recorded this round.", flush=True)
        return
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / f"execution_report_round{rnd:02d}.json", "w", encoding="ascii") as f:
        json.dump(execution_results, f, indent=4)
    cumulative: List[dict] = []
    cum_path = report_dir / "execution_report.json"
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
    with open(report_dir / "execution_report.csv", "w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=EXECUTION_RESULT_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(cumulative)
    passed = sum(1 for r in execution_results if r["status"] == "PASSED")
    print(f"    [+] {passed}/{len(execution_results)} evaluator test(s) passed. "
          f"Reports in {report_dir}", flush=True)


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
    global _RUN_CONTRACT, _RUN_INTEGRATION_CMD, _RUN_DELIVERABLES, DREAM_FORCE_REVISIONS
    global _RUN_BRIEF, _RUN_CONTRACT_SYNTHESIZED, _RUN_ALLOWED_IMPORTS, _RUN_COMMAND, _RUN_DIR
    global _RUN_PROBE_COMMAND
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
    parser.add_argument("--force-dream", action="store_true",
                        help="Run apex policy revisions even when the replay oracle bound shows less "
                             "than DREAM_MIN_HEADROOM of headroom over the deployed policy.")
    parser.add_argument("--dream-only", action="store_true",
                        help="Run no agents. Dream over the existing tree pool and write the next policy. "
                             "Requires -r and at least one recorded round.")
    parser.add_argument("--integration-cmd", type=str, default=None,
                        help="Shell command run in the assembled project after the built-in checks; "
                             "exit 0 passes, and a JSON \"score\" in its output is reported. Saved "
                             "with the run, so -r reuses it. Pass '' to clear.")
    parser.add_argument("--semantic-guidance", action="store_true",
                        help="Inject high-level directional guidance into agent prompts. Off by default: "
                             "the paper's ablation found this underperforms unguided replay at equal budget.")

    args = parser.parse_args()
    DREAM_FORCE_REVISIONS = bool(args.force_dream)

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
    _LEDGER.bind(target_directory)
    if _LEDGER.records:
        print(f"[TOKENS] resumed ledger: {len(_LEDGER.records)} earlier call(s), "
              f"{_fmt_tok(_LEDGER.totals()['total'])} tokens so far", flush=True)

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
                  f"Re-run with -r once the apex ({GEN_API_BASE}) is healthy.", flush=True)
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

    # ---------------- Pinned contract + required deliverables ----------------
    _RUN_DIR = target_directory
    meta = load_brief_meta(target_directory)
    if meta is None:
        brief_text = target_prompt or (read_file_content_safe(raw_filepath) if raw_filepath else "") or ""
        contract = extract_pinned_contract(brief_text)
        deliverables = extract_deliverables(brief_text)
        synthesized = False
        if not contract and target_prompt and SYNTHESIZE_CONTRACT:
            if apex_ok:
                print("\n[CONTRACT] No pinned section in the prompt; synthesizing DELIVERABLES and "
                      "CONSTRAINTS from the prompt text only (not the Phase-1 draft)...", flush=True)
                s_contract, s_deliv = synthesize_contract(target_prompt)
                if s_contract:
                    contract, synthesized = s_contract, True
                    deliverables = deliverables or s_deliv
            else:
                print("\n[CONTRACT] Apex offline; cannot synthesize a contract.", flush=True)
        allowed = None
        if ENFORCE_DEPENDENCIES:
            refresh_environment(target_directory, 0, focus_text=target_prompt + "\n" + contract)
            allowed = list(_RUN_ENV.get("allowed", [])) or None
            contract = (contract + "\n\n" if contract else "") + dependency_section()
        run_cmd, probe_cmd = "", ""
        interfaces_synth = False
        has_interfaces = any(h.startswith("INTERFACES") for h, _ in split_brief_sections(contract) if h)
        if (SYNTHESIZE_INTERFACES and not has_interfaces and target_prompt and deliverables
                and any(d.endswith(".py") and not _is_test_file(d) for d in deliverables)):
            if apex_ok:
                print("[CONTRACT] No INTERFACES section; the planner chooses one shared API "
                      "(labelled as planner-chosen)...", flush=True)
                section, run_cmd, probe_cmd = synthesize_interfaces(
                    target_prompt, contract + "\n\n" + _RUN_API_FACTS_TEXT + "\n\n"
                    + fit_context(_RUN_ENV_TEXT, 4000), deliverables)
                if section:
                    contract = contract + "\n\n" + section
                    interfaces_synth = True
        run_cmd = validate_run_command(RUN_COMMAND, None) or run_cmd or default_run_command(deliverables)
        probe_cmd = validate_run_command(PROBE_COMMAND, None) or (probe_cmd if run_cmd else "")
        if run_cmd:
            contract = contract + "\n\n" + run_rules_section(run_cmd)
        if any(_is_test_file(PurePosixPath(d).name) for d in deliverables):
            contract = contract + "\n\n" + test_rules_section()
        if synthesized:
            contract = synthesized_contract_header() + "\n\n" + contract
        meta = {"contract": enforce_ascii(contract.strip()), "deliverables": deliverables,
                "integration_cmd": INTEGRATION_CMD, "synthesized": synthesized,
                "allowed_imports": allowed, "brief": target_prompt or "",
                "interfaces_synthesized": interfaces_synth, "run_cmd": run_cmd,
                "probe_cmd": probe_cmd}
    if args.integration_cmd is not None:
        meta["integration_cmd"] = args.integration_cmd.strip()
    save_brief_meta(target_directory, meta)
    _RUN_CONTRACT = meta.get("contract", "") or ""
    _RUN_INTEGRATION_CMD = meta.get("integration_cmd", "") or ""
    required_deliverables: Dict[str, str] = meta.get("deliverables") or {}
    _RUN_DELIVERABLES = list(required_deliverables)
    _RUN_CONTRACT_SYNTHESIZED = bool(meta.get("synthesized"))
    if ENFORCE_DEPENDENCIES and not _RUN_ENV:
        # Resume, or the contract was built earlier: discover the container now.
        refresh_environment(target_directory, max(0, len(completed_rounds(target_directory))),
                            focus_text=(meta.get("brief") or "") + "\n" + (meta.get("contract") or ""),
                            project=integration_dir_for(target_directory) / "latest")
    if not ENFORCE_DEPENDENCIES:
        _RUN_ALLOWED_IMPORTS = None
    _RUN_BRIEF = enforce_ascii(meta.get("brief") or target_prompt or "")
    _RUN_COMMAND = (validate_run_command(RUN_COMMAND, None) or meta.get("run_cmd")
                    or default_run_command(meta.get("deliverables") or {}))
    _RUN_PROBE_COMMAND = validate_run_command(PROBE_COMMAND, None) or meta.get("probe_cmd") or ""
    if _RUN_CONTRACT:
        heads = [h for h, _ in split_brief_sections(_RUN_CONTRACT) if h]
        kind = "Synthesized" if _RUN_CONTRACT_SYNTHESIZED else "Pinned"
        print(f"\n[CONTRACT] {kind} {', '.join(heads) or 'section(s)'} ({len(_RUN_CONTRACT):,} chars) "
              f"-> to every agent and test generator. Saved as CONTRACT.md"
              + (" (marked synthesized)." if _RUN_CONTRACT_SYNTHESIZED else "."), flush=True)
    else:
        print(f"\n[CONTRACT] No pinned section ({', '.join(PINNED_SECTIONS)}) found in the brief; "
              "agents see only their own objective.", flush=True)
    if required_deliverables:
        print(f"[DELIVERABLES] {len(required_deliverables)} required: "
              f"{', '.join(required_deliverables)}", flush=True)
    if _RUN_ALLOWED_IMPORTS is not None:
        print(f"[DEPENDENCIES] discovered, not configured: {len(_RUN_ALLOWED_IMPORTS)} importable module(s) "
              f"in the container; rescanned {'every round' if ENV_RESCAN_EACH_ROUND else 'once'}; "
              f"anything not installed rejects the attempt", flush=True)
    if meta.get("interfaces_synthesized"):
        print("[INTERFACES] planner-chosen API added to the contract (not from the prompt).", flush=True)
    if _RUN_COMMAND:
        print(f"[RUN] grounding command after each round: {_RUN_COMMAND} "
              f"(real output -> every agent; write-ups may only report it)", flush=True)
    if _RUN_PROBE_COMMAND:
        print(f"[PROBE] negative control after each successful run: {_RUN_PROBE_COMMAND} "
              f"(unchanged numbers -> metrics flagged as measuring nothing)", flush=True)
    elif _RUN_COMMAND:
        print("[PROBE] no negative-control command (planner gave none; set PROBE_COMMAND to add one).",
              flush=True)
    if FREEZE_INTERFACES:
        print("[API] after each round the integrated project's public API is frozen for the next; "
              "breaking a used name is a violation.", flush=True)
    if _RUN_BRIEF:
        print(f"[BRIEF] original prompt ({len(_RUN_BRIEF):,} chars) -> verbatim to every agent.", flush=True)
    if EVAL_INTEGRATION:
        print(f"[INTEGRATION] ON: each attempt is checked inside a project of every sibling's best "
              f"deliverable (mix {EVAL_INTEGRATION_MIX:.2f})"
              + (f"; command: {_RUN_INTEGRATION_CMD}" if _RUN_INTEGRATION_CMD else "; no command"),
              flush=True)

    # ---------------- Phase 3: partition ----------------
    master_start_time = time.time()
    roster = load_roster(target_directory) if args.resume else []
    plan_p, plan_c = 0, 0

    if roster:
        print(f"[PHASE 3] Roster loaded from {COMMS_DIRNAME}/roster.json ({len(roster)} agent(s)).")
    else:
        if not apex_ok:
            print("\n[!] Apex tier offline; agent partitioning cannot run. "
                  f"Re-run with -r once the apex ({GEN_API_BASE}) is healthy.", flush=True)
            sys.exit(1)
        fragments, plan_p, plan_c = decompose_to_atomic_pieces(target_query, required_deliverables,
                                                               _RUN_CONTRACT)
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
    last_online_round = 0
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
                 if args.no_dream else f"{DREAM_CANDIDATES} policy revision(s) per dream"
                 + (" (forced)" if DREAM_FORCE_REVISIONS else f" (gated at headroom {DREAM_MIN_HEADROOM})"))
              + ".", flush=True)

        print_token_estimate(f"estimate for {end_round - start_round + 1} round(s) plus final stages",
                             estimate_remaining_tokens(end_round - start_round + 1, budget, len(roster)))

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
                      f"{start_round:02d}. Stopping - re-run with -r once the apex ({GEN_API_BASE}) is healthy.", flush=True)
                start_round = end_round + 1

        # Resume: the next round builds against the last frozen API and run output.
        if start_round > 1:
            load_round_grounding(target_directory, start_round - 1)

        for rnd in range(start_round, end_round + 1):
            if _shutdown_event.is_set():
                print(f"\n[!] Shutdown requested; stopping before round {rnd:02d}. "
                      "Re-run with -r to continue.", flush=True)
                break
            global _CURRENT_ROUND
            _CURRENT_ROUND = rnd
            if ENFORCE_DEPENDENCIES and ENV_RESCAN_EACH_ROUND and \
                    time.time() - _RUN_ENV.get("_ts", 0) > 60:
                prev_proj = integration_dir_for(target_directory) / f"round{rnd - 1:02d}"
                refresh_environment(target_directory, rnd, focus_text=_RUN_BRIEF + "\n" + _RUN_CONTRACT,
                                    project=prev_proj if prev_proj.exists() else None)

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
            last_online_round = rnd

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
            print_round_tokens(rnd)
            if rnd < end_round:
                print_token_estimate(f"remaining ({end_round - rnd} round(s) plus final stages)",
                                     estimate_remaining_tokens(end_round - rnd, budget, len(roster)))
        _CURRENT_ROUND = None

    if last_online_round and not _shutdown_event.is_set():
        try:
            review = final_skeptic_review(target_directory, last_online_round)
            final_writeup_refresh(target_directory, roster, last_online_round, target_query, review)
        except Exception as exc:
            print(f"    [!] Final write-up refresh failed: {str(exc)[:120]}", flush=True)

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

    # ---------------- Token summary (after everything, phase 6 included) ----------------
    global _RUN_RUNTIME_SECS
    _RUN_RUNTIME_SECS = time.time() - master_start_time
    if _LEDGER.records:
        print_token_summary()
        summary_md = token_summary_markdown()
        with open(target_directory / "TOKENS.md", "w", encoding="ascii") as f:
            f.write(summary_md)
        with open(target_directory / "tokens_summary.json", "w", encoding="ascii") as f:
            json.dump({"runtime_secs": round(_RUN_RUNTIME_SECS, 1), "totals": _LEDGER.totals(),
                       "by_category": _LEDGER.by("category"),
                       "by_tier": _LEDGER.by("tier"), "by_round": _LEDGER.by("round")}, f, indent=2)
        with open(manifest_path, "a", encoding="ascii") as f:
            f.write("\n\n" + summary_md.replace("# Token usage", "## Token usage", 1))
        print(f"    -> TOKENS.md, tokens_summary.json (per call: tokens.jsonl); appended to "
              f"{manifest_path.name}", flush=True)

    print("\n==============================================================================")
    print("PIPELINE COMPLETE")
    print(f"  Deliverables: {work_dir_for(target_directory)}")
    print(f"  Discovery tree: {trees_dir_for(target_directory)}")
    print(f"  Policies: {policy_dir_for(target_directory)}")
    print(f"  Dreaming: {dream_dir_for(target_directory)}")
    print(f"  Comms log: {comms_dir_for(target_directory)}")
    print(f"  Manifest: {manifest_path.name}")
    print(f"  Tokens: TOKENS.md")
    print("==============================================================================\n")


if __name__ == "__main__":
    main()
