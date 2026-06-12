import os
import re
import time
import random
import queue
import threading
import concurrent.futures
import subprocess
import json
import csv
from pathlib import Path, PurePosixPath

import requests

# =====================================================================
# CONFIGURABLE WORKER ENDPOINTS
# Map these ports to your llama.cpp instances bound to specific GPUs.
# For example, across a dual-socket H11DSi board running a 6-GPU mesh:
# =====================================================================
WORKER_ENDPOINTS = [
    "http://127.0.0.1:8080/v1/chat/completions",  # Target: GPU 0
    "http://127.0.0.1:8081/v1/chat/completions",  # Target: GPU 1
    "http://127.0.0.1:8082/v1/chat/completions",  # Target: GPU 2
    "http://127.0.0.1:8083/v1/chat/completions",  # Target: GPU 3
    "http://127.0.0.1:8084/v1/chat/completions",  # Target: GPU 4
    "http://127.0.0.1:8085/v1/chat/completions",  # Target: GPU 5
]

# Retry configuration for worker requests
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0   # seconds; doubles each attempt
RETRY_JITTER = 0.5       # seconds; added as uniform random noise


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


def extract_code_blocks(md_content: str, output_dir: str) -> list:
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
        if header_match:
            potential_name = header_match.group(1).strip()
            if "." in potential_name or potential_name.lower() in ("dockerfile", "makefile"):
                detected_filename = potential_name
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
                        i += 1
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

                file_path = _safe_output_path(detected_filename, output_path)

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content + "\n")

                print(f"[+] Extracted: {file_path.relative_to(output_path)} ({current_lang})")

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
            f"[!] WARNING: Unclosed code fence at end of document. "
            f"Partial content written to {fallback_name}"
        )

    return extracted_artifacts


def _strip_markdown_fences(text: str) -> str:
    """Remove opening and closing Markdown code fences from model output."""
    return re.sub(r'^```[^\n]*\n?|```\s*$', '', text.strip())


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
    the endpoint to the queue.

    Returns the generated test file metadata dict on success, or None on failure.
    """
    valid_langs = {"python", "py", "cpp", "c", "bash", "sh"}
    if artifact["language"].lower() not in valid_langs:
        print(f"[-] Skipping generation for non-code artifact: {artifact['filename']}")
        return None

    endpoint_url = endpoint_queue.get()
    port = endpoint_url.split(":")[-1].split("/")[0]
    print(f"[*] Thread started for {artifact['filename']} -> Dispatching to worker on port {port}")

    prompt = (
        f"You are an expert software engineer. Write comprehensive unit tests for the following "
        f"{artifact['language']} code. Ensure edge cases are covered. Output ONLY the test code.\n\n"
        f"File: {artifact['filename']}\n"
        f"```{artifact['language']}\n{artifact['content']}\n```"
    )
    payload = {
        "messages": [
            {"role": "system", "content": "You are a specialized code testing assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
    }

    last_exception: Exception | None = None
    generation_metadata: dict | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(endpoint_url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()

            choices = result.get("choices")
            if not choices:
                print(f"[!] FAILED [{port}]: Unexpected response body for {artifact['filename']}: {result}")
                break

            test_code = choices[0].get("message", {}).get("content", "")
            if not test_code:
                print(f"[!] FAILED [{port}]: Empty content in response for {artifact['filename']}: {result}")
                break

            test_code = _strip_markdown_fences(test_code)

            test_output_dir.mkdir(parents=True, exist_ok=True)
            test_filename = f"test_{artifact['filename']}"
            test_filepath = test_output_dir / test_filename

            with open(test_filepath, "w", encoding="utf-8") as f:
                f.write(test_code + "\n")

            with progress_lock:
                progress_counter[0] += 1
                done, total = progress_counter
                print(f"[+] SUCCESS [{port}] ({done}/{total}): Tests for {artifact['filename']} -> {test_filename}")

            generation_metadata = {
                "filename": test_filename,
                "test_filepath": test_filepath,
                "language": artifact["language"],
                "artifact_filepath": artifact["filepath"]
            }
            last_exception = None
            break

        except requests.exceptions.RequestException as exc:
            last_exception = exc
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER)
                print(f"[~] RETRY [{port}] attempt {attempt}/{MAX_RETRIES} for {artifact['filename']} in {delay:.1f}s: {exc}")
                time.sleep(delay)

    if last_exception is not None:
        print(f"[!] FAILED [{port}]: All {MAX_RETRIES} attempts exhausted for {artifact['filename']}: {last_exception}")

    endpoint_queue.put(endpoint_url)
    return generation_metadata


def execute_test_artifact(test_meta: dict) -> dict:
    """
    PHASE 3: Runtime execution and validation engine.
    Runs the generated test files using language-appropriate test hooks.
    """
    lang = test_meta["language"].lower()
    test_path = Path(test_meta["test_filepath"])
    artifact_path = Path(test_meta["artifact_filepath"])

    result = {
        "filename": test_meta["filename"],
        "language": lang,
        "status": "UNKNOWN",
        "message": ""
    }

    start_time = time.time()
    try:
        if lang in ("python", "py"):
            # Execute with python's module suite wrapper to evaluate standalone TestCase classes
            cmd = ["python", "-m", "unittest", str(test_path)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            duration = time.time() - start_time

            if res.returncode == 0:
                result["status"] = "PASSED"
                result["message"] = f"OK ({duration:.2f}s)"
            else:
                result["status"] = "FAILED"
                err_lines = [l for l in (res.stderr + res.stdout).splitlines() if l.strip()]
                result["message"] = err_lines[-1] if err_lines else f"Exit code {res.returncode}"

        elif lang in ("bash", "sh"):
            cmd = ["bash", str(test_path)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            duration = time.time() - start_time

            if res.returncode == 0:
                result["status"] = "PASSED"
                result["message"] = f"OK ({duration:.2f}s)"
            else:
                result["status"] = "FAILED"
                err_lines = [l for l in (res.stderr + res.stdout).splitlines() if l.strip()]
                result["message"] = err_lines[-1] if err_lines else f"Exit code {res.returncode}"

        elif lang in ("c", "cpp"):
            binary_path = test_path.with_suffix(".bin")
            compiler = "gcc" if lang == "c" else "g++"

            # Compiles both components directly. Modify compilation flag array if linking custom harnesses.
            compile_cmd = [compiler, str(artifact_path), str(test_path), "-o", str(binary_path)]
            comp_res = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=20)

            if comp_res.returncode != 0:
                result["status"] = "COMPILE_ERROR"
                err_lines = [l for l in comp_res.stderr.splitlines() if l.strip()]
                result["message"] = err_lines[0] if err_lines else "Compiler failure"
                return result

            res = subprocess.run([str(binary_path)], capture_output=True, text=True, timeout=30)
            duration = time.time() - start_time

            if res.returncode == 0:
                result["status"] = "PASSED"
                result["message"] = f"OK ({duration:.2f}s)"
            else:
                result["status"] = "FAILED"
                err_lines = [l for l in (res.stderr + res.stdout).splitlines() if l.strip()]
                result["message"] = err_lines[-1] if err_lines else f"Exit code {res.returncode}"

            if binary_path.exists():
                binary_path.unlink()
        else:
            result["status"] = "SKIPPED"
            result["message"] = f"No environment definition for: {lang}"

    except subprocess.TimeoutExpired:
        result["status"] = "TIMEOUT"
        result["message"] = "Execution threshold exceeded"
    except Exception as e:
        result["status"] = "ERROR"
        result["message"] = str(e)

    return result


if __name__ == "__main__":
    MARKDOWN_SOURCE = "POLISHED_SYNTHESIS.md"
    OUTPUT_WORKSPACE = "./extracted_workspace"
    TEST_OUTPUT_DIR = Path(OUTPUT_WORKSPACE) / "tests"
    REPORT_OUTPUT_DIR = Path(OUTPUT_WORKSPACE) / "reports"

    endpoint_queue: queue.Queue = queue.Queue()
    for ep in WORKER_ENDPOINTS:
        endpoint_queue.put(ep)

    if not os.path.exists(MARKDOWN_SOURCE):
        print(f"Error: Could not find {MARKDOWN_SOURCE}. Please ensure the file is in the current directory.")
    else:
        with open(MARKDOWN_SOURCE, "r", encoding="utf-8") as fh:
            md_content = fh.read()

        print("=== Phase 1: Local Extraction ===")
        artifacts = extract_code_blocks(md_content, OUTPUT_WORKSPACE)

        print(f"\n=== Phase 2: Parallelised Test Generation ===")
        print(f"[*] {len(artifacts)} artifacts extracted.")
        print(f"[*] Initialising ThreadPoolExecutor with {len(WORKER_ENDPOINTS)} workers...")

        progress_lock = threading.Lock()
        progress_counter = [0, len(artifacts)]

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(WORKER_ENDPOINTS))
        futures = []
        generated_tests = []

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
                    print(f"[!] Unhandled exception in worker thread: {exc}")

        except KeyboardInterrupt:
            print("\n[!] Interrupted - cancelling pending tasks...")
            for f in futures:
                f.cancel()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        print("\n=== Phase 3: Functional Test Execution ===")
        if not generated_tests:
            print("[-] No valid test suites were successfully generated to execute.")
        else:
            print(f"[*] Running {len(generated_tests)} test suites sequentially...")
            execution_results = []
            
            for test_meta in generated_tests:
                print(f"[*] Executing target: {test_meta['filename']}...")
                res = execute_test_artifact(test_meta)
                execution_results.append(res)

            # Execution Output Formatter Table
            print("\n" + "=" * 90)
            print(f"{'AUTOMATED TEST RUN PIPELINE REPORT':^90}")
            print("=" * 90)
            print(f"{'Generated Test File':<35} | {'Lang':<6} | {'Status':<13} | {'Details / Error Context'}")
            print("-" * 90)

            passed_count = 0
            failed_count = 0
            error_count = 0

            for r in execution_results:
                if r["status"] == "PASSED":
                    passed_count += 1
                elif r["status"] in ("FAILED", "COMPILE_ERROR"):
                    failed_count += 1
                else:
                    error_count += 1

                msg_summary = r["message"][:32] + "..." if len(r["message"]) > 35 else r["message"]
                print(f"{r['filename']:<35} | {r['language']:<6} | {r['status']:<13} | {msg_summary}")

            print("-" * 90)
            print(f"TOTAL RUNS: {len(execution_results)}  |  PASSED: {passed_count}  |  FAILED/COMPILE: {failed_count}  |  ERRORS: {error_count}")
            print("=" * 90)

            # Exporting Reports
            REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            json_report_path = REPORT_OUTPUT_DIR / "execution_report.json"
            csv_report_path = REPORT_OUTPUT_DIR / "execution_report.csv"

            # 1. JSON Export
            with open(json_report_path, "w", encoding="utf-8") as f:
                json.dump(execution_results, f, indent=4)

            # 2. CSV Export
            if execution_results:
                keys = execution_results[0].keys()
                with open(csv_report_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(execution_results)
            
            print(f"\n[*] Detailed structural reports saved to:")
            print(f"    - {json_report_path}")
            print(f"    - {csv_report_path}")

        print("\n=== Pipeline execution complete ===")
