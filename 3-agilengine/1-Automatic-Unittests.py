# -*- coding: utf-8 -*-
import os
import re
import sys
import time
import random
import queue
import threading
import concurrent.futures
import subprocess
import json
import csv
import argparse
import tempfile
from pathlib import Path, PurePosixPath

import requests

# =====================================================================
# CONFIGURABLE WORKER ENDPOINTS
# Map these ports to your llama.cpp instances bound to specific GPUs.
# =====================================================================
WORKER_ENDPOINTS = [
    "http://192.168.2.137:8034/v1/chat/completions",  # Target: GPU 0
    "http://192.168.2.137:8035/v1/chat/completions",  # Target: GPU 1
]

# Configurable multiplier for concurrent requests per endpoint.
# Increase this if your inference server supports continuous batching.
CONCURRENT_REQS_PER_ENDPOINT = 2

# Maximum output tokens for the generated test suites.
# Scaled up to support large >= 40k context window configurations.
MAX_OUTPUT_TOKENS = 16384

# =====================================================================
# LLM INFERENCE GUARDRAILS
# Tuned to prevent hallucination and looping in local models.
# =====================================================================
LLM_TEMPERATURE = 0.1          # Keep low for precision code generation
LLM_TOP_P = 0.95               # Truncate lowest probability tokens to prevent wild hallucinations
LLM_FREQUENCY_PENALTY = 0.5    # Aggressively penalize repeating the same tokens (anti-looping)
LLM_PRESENCE_PENALTY = 0.2     # Encourage moving on to new concepts/functions

# Retry configuration for worker requests
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0   # seconds; doubles each attempt
RETRY_JITTER = 0.5       # seconds; added as uniform random noise

# Phase 3: max parallel test execution workers (independent of GPU pool size)
MAX_EXEC_WORKERS = 4

# Statically declared so CSV export never depends on a live result existing.
EXECUTION_RESULT_FIELDS = ["filename", "language", "status", "message"]


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _safe_output_path(detected_filename: str, output_dir: Path) -> Path:
    """
    Convert a raw detected filename (possibly containing path components or
    Windows-style separators) into a safe output path beneath output_dir.
    """
    normalised = detected_filename.replace("\\", "/")
    parts = [p for p in PurePosixPath(normalised).parts if p not in ("", ".", "..")]

    if not parts:
        parts = ["artifact.txt"]

    safe_parts = parts[-2:] if len(parts) >= 2 else parts
    candidate = output_dir.joinpath(*safe_parts)
    candidate.parent.mkdir(parents=True, exist_ok=True)

    if candidate.exists():
        stem = candidate.stem
        suffix = candidate.suffix
        counter = 1
        while candidate.exists():
            candidate = candidate.parent / f"{stem}_{counter}{suffix}"
            counter += 1

    return candidate


def _strip_markdown_fences(text: str) -> str:
    """
    Remove all opening and closing Markdown code fences from model output.
    Handles nested or multiple fences that some quantized models emit.
    """
    # Strip all opening fences (``` optionally followed by a language tag)
    text = re.sub(r'^```[^\n]*\n?', '', text.strip(), flags=re.MULTILINE)
    # Strip all closing fences
    text = re.sub(r'^```\s*$', '', text.strip(), flags=re.MULTILINE)
    return text.strip()


def _extract_error_line(output: str, lang: str) -> str:
    """
    Return the most actionable error line from combined stderr+stdout.
    """
    lines = [l for l in output.splitlines() if l.strip()]
    if not lines:
        return "no output"

    # Filter out useless pytest summary lines so they don't hijack the fallback
    filtered_lines = [l for l in lines if not re.match(r'^\d+ (failed|error|passed|warning|deselected)', l.strip())]

    # FIX: if all lines were summary lines, return a meaningful sentinel rather
    # than falling back to a summary line that carries no actionable context.
    if not filtered_lines:
        return "(pytest summary only - no error detail captured)"

    if lang in ("python", "py"):
        # 1. Pytest 'E ' prefix
        for line in filtered_lines:
            if line.strip().startswith("E "):
                return line.strip()[2:].strip()

        # 2. Standard Exception naming conventions
        err_regex = re.compile(r'^([A-Z][a-zA-Z0-9_]+Error|[A-Z][a-zA-Z0-9_]+Exception|Exception|FAIL:|ERROR:)( |:)')
        for line in filtered_lines:
            if err_regex.match(line.strip()):
                return line.strip()

        # 3. Pytest FAILED inline summary
        for line in filtered_lines:
            if line.strip().startswith("FAILED "):
                return line.strip()

    if lang in ("c", "cpp"):
        # Compiler errors: first line is usually the most specific
        return filtered_lines[0].strip()

    # 4. Fallback: grab the last 2 non-empty lines for context
    if len(filtered_lines) >= 2:
        return f"{filtered_lines[-2].strip()} | {filtered_lines[-1].strip()}"
    return filtered_lines[-1].strip()


def _sanitize_requirements_file(filepath: Path) -> None:
    """
    Strips LLM-hallucinated plain text sentences from a requirements.txt file.
    Valid pip lines contain no spaces unless they involve specific operators
    or environment markers.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        cleaned_lines = []
        valid_pip_operators = ['==', '>=', '<=', '~=', '<', '>', '!=', '@', '-r', '-e', '--', ';']

        for line in lines:
            s_line = line.strip()
            # Keep empty lines and standard comments
            if not s_line or s_line.startswith('#'):
                cleaned_lines.append(line)
                continue

            has_space = ' ' in s_line
            has_operator = any(op in s_line for op in valid_pip_operators)

            # Tighten the heuristic: if the line has spaces AND an operator,
            # only keep it when every whitespace-delimited token that contains
            # a space looks like a URL, environment marker, or a known pip flag.
            if has_space and has_operator:
                tokens = s_line.split()
                first_tok = tokens[0]
                looks_like_pip = (
                    first_tok.startswith('-')          # flag like --index-url, -r, -e
                    or '://' in first_tok              # direct URL
                    or re.match(r'^[A-Za-z0-9_\-\.]+', first_tok)  # package name
                )
                extra_tokens = tokens[2:] if len(tokens) > 2 else []
                has_plain_english_suffix = any(
                    not re.search(r'[=<>!~@;:/]', tok) and tok.isalpha() and len(tok) > 2
                    for tok in extra_tokens
                )
                if not looks_like_pip or has_plain_english_suffix:
                    print(f"Stripped hallucinated requirement line: '{s_line}'")
                    continue
            elif has_space and not has_operator:
                # Spaces with no pip operators at all: definitely conversational bleed.
                print(f"Stripped hallucinated requirement line: '{s_line}'")
                continue

            cleaned_lines.append(line)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)

    except Exception as e:
        print(f"Warning: Could not sanitize requirements file {filepath}: {e}")


# ---------------------------------------------------------------------
# Phase 1 - extraction
# ---------------------------------------------------------------------

def extract_code_blocks(md_content: str, output_dir: str | Path) -> list:
    """
    Parses a Markdown string, extracts code blocks, identifies their intended
    filenames based on context heuristics, and writes them to a local directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    lines = md_content.splitlines()
    i = 0
    in_block = False
    current_block: list[str] = []
    current_lang = ""
    detected_filename: str | None = None
    last_header: str | None = None
    file_counter = 1

    extracted_artifacts = []

    while i < len(lines):
        line = lines[i]

        xml_match = re.search(r'<file path="([^"]+)">', line)
        if xml_match:
            detected_filename = xml_match.group(1)
            i += 1
            continue

        header_match = re.match(r'^###?\s+([a-zA-Z0-9_\-\.]+.*)$', line)
        if header_match and not in_block:
            potential_name = header_match.group(1).strip()

            # Look for embedded filenames inside parens or backticks
            embedded_match = re.search(r'[\(\`]([a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]+)[\)\`]', potential_name)
            if embedded_match:
                candidate = embedded_match.group(1)
            elif "." in potential_name or potential_name.lower() in ("dockerfile", "makefile"):
                candidate = potential_name
            else:
                candidate = None

            # Warn when a previously detected filename is about to be clobbered
            if candidate is not None:
                if detected_filename is not None and detected_filename != candidate:
                    print(
                        f"WARNING: detected_filename '{detected_filename}' overwritten by "
                        f"header '{candidate}' before a code fence opened. "
                        f"The earlier filename will be lost."
                    )
                detected_filename = candidate

            last_header = potential_name
            i += 1
            continue

        if line.startswith("```"):
            if not in_block:
                in_block = True
                current_lang = line[3:].strip()
                current_block = []

                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith("# ") and "." in next_line:
                        detected_filename = next_line[2:].strip()
                        i += 1  # consume the comment line
            else:
                in_block = False
                content = "\n".join(current_block)

                if not detected_filename:
                    ext = current_lang if current_lang else "txt"
                    ext_map = {
                        "python": "py", "bash": "sh",
                        "yaml": "yml", "dockerfile": "Dockerfile",
                    }
                    ext = ext_map.get(ext.lower(), ext)

                    base_name = (
                        last_header.replace(" ", "_").lower()
                        if last_header
                        else f"artifact_{file_counter}"
                    )
                    detected_filename = (
                        f"{base_name}.{ext}"
                        if ext.lower() != "dockerfile"
                        else "Dockerfile"
                    )
                    file_counter += 1

                # Sanitize the filename to ensure it is shell-safe and importable
                detected_filename = re.sub(r'[()\[\]{}]', '', detected_filename)
                detected_filename = detected_filename.replace(" ", "_")

                file_path = _safe_output_path(detected_filename, output_path)

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content + "\n")

                print(f"Extracted: {file_path.relative_to(output_path)} ({current_lang})")

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

        elif in_block:
            current_block.append(line)

        i += 1

    if in_block and current_block:
        content = "\n".join(current_block)
        fallback_name = f"artifact_{file_counter}_partial.txt"
        file_path = output_path / fallback_name
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        print(
            f"WARNING: Unclosed code fence at end of document. "
            f"Partial content written to {fallback_name}"
        )

    return extracted_artifacts


# ---------------------------------------------------------------------
# Phase 2 - parallel test generation
# ---------------------------------------------------------------------

def request_unittests_from_worker(
    artifact: dict,
    endpoint_queue: queue.Queue,
    test_output_dir: Path,
    progress_lock: threading.Lock,
    progress_counter: list,
) -> dict | None:
    """
    Checks out an available endpoint from the queue, sends the payload to the
    local worker with exponential-backoff retry, writes the result, and returns
    the endpoint to the queue securely via a finally block.
    """
    valid_langs = {"python", "py", "cpp", "c", "bash", "sh"}
    if artifact["language"].lower() not in valid_langs:
        print(f"Skipping generation for non-code artifact: {artifact['filename']}")
        return None

    endpoint_url = endpoint_queue.get()
    port = endpoint_url.split(":")[-1].split("/")[0]

    try:
        print(f"Thread started for {artifact['filename']} -> Dispatching to worker on port {port}")

        prompt = (
            f"Write highly compact, succinct, and targeted unit tests for the following {artifact['language']} code.\n"
            f"Focus ONLY on core functionality and critical paths. Minimize boilerplate aggressively.\n\n"
            f"CRITICAL RULES:\n"
            f"1. DO NOT hallucinate imports or use non-existent modules.\n"
            f"2. Keep the code as short as possible while ensuring it runs and passes.\n"
            f"3. Group assertions and use parametrization where possible to save space.\n"
            f"4. Output ONLY valid test code inside a single markdown code block. No explanations.\n\n"
            f"File: {artifact['filename']}\n"
            f"```{artifact['language']}\n{artifact['content']}\n```"
        )
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a highly efficient code testing assistant. Write succinct, compact, and boilerplate-free unit tests. Use parametrization to consolidate test cases where applicable. Do not explain your code."
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
            "frequency_penalty": LLM_FREQUENCY_PENALTY,
            "presence_penalty": LLM_PRESENCE_PENALTY,
            "max_tokens": MAX_OUTPUT_TOKENS,
        }

        last_exception: Exception | None = None
        generation_metadata: dict | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(endpoint_url, json=payload, timeout=600)
                response.raise_for_status()
                result = response.json()

                choices = result.get("choices")
                if not choices:
                    print(f"FAILED [{port}]: Unexpected response body for {artifact['filename']}: {result}")
                    break

                test_code = choices[0].get("message", {}).get("content", "")

                # Treat empty content as a transient error and retry
                if not test_code:
                    last_exception = RuntimeError(
                        f"Empty content in response for {artifact['filename']}: {result}"
                    )
                    if attempt < MAX_RETRIES:
                        delay = RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER)
                        print(
                            f"RETRY [{port}] attempt {attempt}/{MAX_RETRIES} for "
                            f"{artifact['filename']} (empty content) in {delay:.1f}s"
                        )
                        time.sleep(delay)
                    continue

                test_code = _strip_markdown_fences(test_code)

                test_output_dir.mkdir(parents=True, exist_ok=True)
                test_filename = f"test_{artifact['filename']}"
                test_filepath = test_output_dir / test_filename

                with open(test_filepath, "w", encoding="utf-8") as f:
                    f.write(test_code + "\n")

                with progress_lock:
                    progress_counter[0] += 1
                    done, total = progress_counter
                    print(f"SUCCESS [{port}] ({done}/{total}): Tests for {artifact['filename']} -> {test_filename}")

                generation_metadata = {
                    "filename": test_filename,
                    "test_filepath": str(test_filepath),
                    "language": artifact["language"],
                    "artifact_filepath": artifact["filepath"],
                }
                last_exception = None
                break

            except requests.exceptions.RequestException as exc:
                last_exception = exc
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER)
                    print(f"RETRY [{port}] attempt {attempt}/{MAX_RETRIES} for {artifact['filename']} in {delay:.1f}s: {exc}")
                    time.sleep(delay)

        if last_exception is not None:
            print(f"FAILED [{port}]: All {MAX_RETRIES} attempts exhausted for {artifact['filename']}: {last_exception}")

        return generation_metadata

    finally:
        # GUARANTEE: The endpoint token is always returned to the queue
        # even if an unhandled Python exception interrupts the block.
        endpoint_queue.put(endpoint_url)


# ---------------------------------------------------------------------
# Phase 3 - execution and validation
# ---------------------------------------------------------------------

def execute_test_artifact(test_meta: dict) -> dict:
    """
    Runs the generated test file using language-appropriate tooling and returns
    a result dict suitable for JSON/CSV export.
    """
    lang = test_meta["language"].lower()

    # Resolve paths to absolute to prevent cwd shifts from breaking file access
    test_path = Path(test_meta["test_filepath"]).resolve()
    artifact_path = Path(test_meta["artifact_filepath"]).resolve()

    result = {
        "filename": test_meta["filename"],
        "language": lang,
        "status": "UNKNOWN",
        "message": "",
    }

    try:
        if lang in ("python", "py"):
            # Use test_path.parent as cwd and inject it into PYTHONPATH so
            # relative imports resolve correctly regardless of nesting depth.
            env = os.environ.copy()
            src_dir = str(test_path.parent)
            existing_pypath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{src_dir}:{existing_pypath}" if existing_pypath else src_dir

            cmd = ["python", "-m", "pytest", "--tb=short", "-q", str(test_path)]
            start_time = time.time()
            res = subprocess.run(
                cmd,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=45,
                cwd=str(test_path.parent),
                env=env,
            )
            duration = time.time() - start_time

            if res.returncode == 0:
                result["status"] = "PASSED"
                result["message"] = f"OK ({duration:.2f}s)"
            else:
                result["status"] = "FAILED"
                combined = res.stderr + res.stdout
                result["message"] = _extract_error_line(combined, lang)

        elif lang in ("bash", "sh"):
            start_time = time.time()
            res = subprocess.run(
                ["bash", str(test_path)],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            duration = time.time() - start_time

            if res.returncode == 0:
                result["status"] = "PASSED"
                result["message"] = f"OK ({duration:.2f}s)"
            else:
                result["status"] = "FAILED"
                result["message"] = _extract_error_line(res.stderr + res.stdout, lang)

        elif lang in ("c", "cpp"):
            compiler = "gcc" if lang == "c" else "g++"

            # Use a temp file for the binary to avoid path collisions when
            # two artifacts share the same stem (e.g. utils.c and utils.cpp).
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
                binary_path = Path(tmp.name)

            try:
                compile_cmd = [compiler, str(artifact_path), str(test_path), "-o", str(binary_path)]
                comp_res = subprocess.run(
                    compile_cmd,
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=20,
                )

                if comp_res.returncode != 0:
                    result["status"] = "COMPILE_ERROR"
                    result["message"] = _extract_error_line(comp_res.stderr, lang)
                    return result

                start_time = time.time()
                res = subprocess.run(
                    [str(binary_path)],
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=30,
                )
                duration = time.time() - start_time

                if res.returncode == 0:
                    result["status"] = "PASSED"
                    result["message"] = f"OK ({duration:.2f}s)"
                else:
                    result["status"] = "FAILED"
                    result["message"] = _extract_error_line(res.stderr + res.stdout, lang)
            finally:
                if binary_path.exists():
                    binary_path.unlink()

        else:
            result["status"] = "SKIPPED"
            result["message"] = f"No environment definition for: {lang}"

    except subprocess.TimeoutExpired:
        result["status"] = "TIMEOUT"
        result["message"] = "Execution threshold exceeded"
    except Exception as exc:
        result["status"] = "ERROR"
        result["message"] = str(exc)

    return result


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallelised LLM Test Generation and Execution Pipeline")
    parser.add_argument(
        "source_file",
        type=str,
        nargs="?",
        default="POLISHED_SYNTHESIS.md",
        help="Path to the input Markdown file (default: POLISHED_SYNTHESIS.md)"
    )
    parser.add_argument(
        "-c", "--concurrency",
        type=int,
        default=CONCURRENT_REQS_PER_ENDPOINT,
        help=f"Concurrent requests allowed per worker endpoint (default: {CONCURRENT_REQS_PER_ENDPOINT})"
    )
    args = parser.parse_args()

    # Resolve the target input file to an absolute path
    source_path = Path(args.source_file).resolve()

    if not source_path.exists():
        print(f"Error: Could not find {source_path}. Please ensure the file exists and the path is correct.")
        sys.exit(1)

    # Bind the workspace entirely to the target directory containing the markdown file
    OUTPUT_WORKSPACE = source_path.parent
    TEST_OUTPUT_DIR = OUTPUT_WORKSPACE / "tests"
    REPORT_OUTPUT_DIR = OUTPUT_WORKSPACE / "reports"

    # Queue tokens dictate how many concurrent threads can hit a given endpoint
    endpoint_concurrency = args.concurrency
    total_gen_workers = len(WORKER_ENDPOINTS) * endpoint_concurrency

    endpoint_queue: queue.Queue = queue.Queue()
    for ep in WORKER_ENDPOINTS:
        for _ in range(endpoint_concurrency):
            endpoint_queue.put(ep)

    with open(source_path, "r", encoding="utf-8") as fh:
        md_content = fh.read()

    # ---------------- Phase 1 ------------------------------
    print(f"Sourcing artifacts from: {source_path}")
    print(f"Workspace mapped to: {OUTPUT_WORKSPACE}")
    print("=== Phase 1: Local Extraction ===")
    artifacts = extract_code_blocks(md_content, OUTPUT_WORKSPACE)

    # ---------------- Phase 2 ------------------------------
    print(f"\n=== Phase 2: Parallelised Test Generation ===")
    print(f"{len(artifacts)} artifacts extracted.")
    print(f"Initialising ThreadPoolExecutor with {total_gen_workers} workers ({endpoint_concurrency} per endpoint)...")

    progress_lock = threading.Lock()
    progress_counter = [0, len(artifacts)]

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=total_gen_workers)
    futures = []
    generated_tests: list[dict] = []

    try:
        futures = [
            executor.submit(
                request_unittests_from_worker,
                artifact,
                endpoint_queue,
                TEST_OUTPUT_DIR,
                progress_lock,
                progress_counter,
            )
            for artifact in artifacts
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                test_meta = future.result()
                if test_meta:
                    generated_tests.append(test_meta)
            except Exception as exc:
                print(f"Unhandled exception in worker thread: {exc}")

    except KeyboardInterrupt:
        print("\nInterrupted - cancelling pending tasks...")
        for f in futures:
            f.cancel()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    # ---------------- Phase 2.5 - Dependency Resolution ----------------
    print("\n=== Phase 2.5: Dependency Resolution ===")
    req_files = [a for a in artifacts if "requirements" in a["filename"].lower() and a["filename"].endswith(".txt")]
    if not req_files:
        print("No requirements.txt files found to install.")
    else:
        for req in req_files:
            req_path = Path(req["filepath"]).resolve()
            print(f"Installing dependencies from {req['filename']}...")

            # Sanitize out any LLM hallucinated conversational sentences before executing pip
            _sanitize_requirements_file(req_path)

            try:
                res = subprocess.run(
                    ["python", "-m", "pip", "install", "--break-system-packages", "-r", str(req_path)],
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=120
                )
                if res.returncode == 0:
                    print(f"Successfully installed {req['filename']}")
                else:
                    print(f"Warning: pip install failed for {req['filename']}\n{res.stderr.strip()}")
            except Exception as e:
                print(f"Failed to execute pip install for {req['filename']}: {e}")

    # ---------------- Phase 3 ------------------------------
    print("\n=== Phase 3: Functional Test Execution ===")
    execution_results: list[dict] = []

    if not generated_tests:
        print("No valid test suites were successfully generated to execute.")
    else:
        print(f"Running {len(generated_tests)} test suites"
              f" (up to {MAX_EXEC_WORKERS} parallel)...")

        exec_executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_EXEC_WORKERS)
        exec_futures = []

        try:
            exec_futures = [
                exec_executor.submit(execute_test_artifact, tm)
                for tm in generated_tests
            ]

            for future in concurrent.futures.as_completed(exec_futures):
                try:
                    res = future.result()
                    execution_results.append(res)
                    print(f"Executed: {res['filename']} -> {res['status']}")
                except Exception as exc:
                    print(f"Unhandled exception during test execution: {exc}")

        except KeyboardInterrupt:
            print("\nInterrupted - cancelling pending executions...")
            for f in exec_futures:
                f.cancel()
        finally:
            exec_executor.shutdown(wait=False, cancel_futures=True)

    # ---------------- Report -------------------------------
    if execution_results:
        passed_count  = sum(1 for r in execution_results if r["status"] == "PASSED")
        failed_count  = sum(1 for r in execution_results if r["status"] in ("FAILED", "COMPILE_ERROR"))
        error_count   = sum(1 for r in execution_results if r["status"] not in ("PASSED", "FAILED", "COMPILE_ERROR"))

        print("\n" + "=" * 90)
        print(f"{'AUTOMATED TEST RUN PIPELINE REPORT':^90}")
        print("=" * 90)
        print(f"{'Generated Test File':<35} | {'Lang':<6} | {'Status':<13} | {'Details / Error Context'}")
        print("-" * 90)

        for r in execution_results:
            msg_summary = r["message"][:32] + "..." if len(r["message"]) > 35 else r["message"]
            print(f"{r['filename']:<35} | {r['language']:<6} | {r['status']:<13} | {msg_summary}")

        print("-" * 90)
        print(
            f"TOTAL RUNS: {len(execution_results)}  |  "
            f"PASSED: {passed_count}  |  "
            f"FAILED/COMPILE: {failed_count}  |  "
            f"ERRORS: {error_count}"
        )
        print("=" * 90)

        REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        json_report_path = REPORT_OUTPUT_DIR / "execution_report.json"
        csv_report_path  = REPORT_OUTPUT_DIR / "execution_report.csv"

        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(execution_results, f, indent=4)

        # extrasaction='ignore' prevents DictWriter from raising if result
        # dicts ever acquire extra debugging keys during future development.
        with open(csv_report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EXECUTION_RESULT_FIELDS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(execution_results)

        print(f"\nReports saved to:")
        print(f"    - {json_report_path}")
        print(f"    - {csv_report_path}")

    print("\n=== Pipeline execution complete ===")

    # ---------------------------------------------------------------------
    # Phase 4 - Automated Handoff to Todo-Project-Distill
    # ---------------------------------------------------------------------
    print("\n=== Phase 4: Automated Handoff ===")
    
    current_script_dir = Path(__file__).resolve().parent
    distill_script = (current_script_dir.parent / "3-agilengine" / "2-Todo-Project-Distill.py").resolve()
    
    # Fallback to the current directory if it couldn't be resolved in the parent structure
    if not distill_script.exists():
        distill_script = (current_script_dir / "2-Todo-Project-Distill.py").resolve()

    if distill_script.exists():
        print(f"Initiating handoff to {distill_script.name}...")
        try:
            # Pass the bound directory directly to the distillation pipeline
            subprocess.run(
                ["python3", str(distill_script), str(OUTPUT_WORKSPACE.absolute())],
                cwd=str(OUTPUT_WORKSPACE),
                check=True
            )
            print("    [+] Todo distillation pipeline completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"    [!] Distillation script failed with exit code: {e.returncode}")
        except Exception as e:
            print(f"    [!] Execution error during handoff: {e}")
    else:
        print(f"    [!] Could not locate {distill_script.name}. Skipping automated handoff.")
