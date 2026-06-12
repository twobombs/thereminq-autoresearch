import os
import re
import time
import random
import queue
import threading
import concurrent.futures
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

    Rules:
      - Normalise Windows backslashes before parsing.
      - Preserve up to one level of subdirectory so that files like
        utils/helpers.py and auth/helpers.py land in different places.
      - Refuse any path component that is ".." to block traversal.
    """
    # Normalise Windows separators so PurePosixPath can parse them.
    normalised = detected_filename.replace("\\", "/")
    parts = [p for p in PurePosixPath(normalised).parts if p not in ("", ".", "..")]

    if not parts:
        parts = ["artifact.txt"]

    # Keep at most (parent_dir, filename) to avoid deeply nested trees
    # while still differentiating utils/helpers.py from auth/helpers.py.
    safe_parts = parts[-2:] if len(parts) >= 2 else parts
    candidate = output_dir.joinpath(*safe_parts)
    candidate.parent.mkdir(parents=True, exist_ok=True)

    # Deduplicate: if the candidate already exists, append a counter.
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

    Returns a list of dictionaries containing file metadata for the AI workers.

    Fixed issues:
      - Filename collision: subdirectory structure is preserved; duplicates
        get a numeric suffix rather than silently overwriting.
      - Comment-filename heuristic: the peeked comment line is now consumed
        (skipped) so it does not appear in the extracted file content.
      - Unclosed fence: a warning is emitted and a partial file is written
        with a .partial extension so content is never silently dropped.
      - Stale last_header: cleared after it is consumed as a filename.
      - Windows path separators: normalised before any path operation.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    lines = md_content.splitlines()
    # Use an index-based loop so we can skip the consumed comment line.
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

        # Heuristic 1: Detect XML-like file paths
        xml_match = re.search(r'<file path="([^"]+)">', line)
        if xml_match:
            detected_filename = xml_match.group(1)
            i += 1
            continue

        # Heuristic 2: Detect Markdown headers for filenames
        header_match = re.match(r'^###?\s+([a-zA-Z0-9_\-\.]+.*)$', line)
        if header_match:
            potential_name = header_match.group(1).strip()
            if "." in potential_name or potential_name.lower() in ("dockerfile", "makefile"):
                detected_filename = potential_name
            last_header = potential_name
            i += 1
            continue

        # Block processing
        if line.startswith("```"):
            if not in_block:
                in_block = True
                current_lang = line[3:].strip()
                current_block = []

                # Heuristic 3: Check if the very next line is a comment with
                # the filename.  Consume the line so it is not included in the
                # extracted file content.
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith("# ") and "." in next_line:
                        detected_filename = next_line[2:].strip()
                        i += 1  # skip the comment line - FIX: was only peeked before
            else:
                in_block = False
                content = "\n".join(current_block)

                # Resolve filename if heuristics missed it
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

                # Reset state - FIX: clear last_header so it does not bleed
                # into the next block's filename when heuristics miss.
                detected_filename = None
                last_header = None
                current_lang = ""
                current_block = []

        elif in_block:
            current_block.append(line)

        i += 1

    # FIX: Warn and flush if the document ends with an unclosed fence.
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
    """
    Remove opening and closing Markdown code fences from model output.

    Handles:
      - Opening fence with optional language tag:  ```python
      - Closing fence possibly followed by a trailing newline: ```\\n
    FIX: the previous slice-based approach left the closing fence when the
    model emitted a trailing newline after it.
    """
    return re.sub(r'^```[^\n]*\n?|```\s*$', '', text.strip())


def request_unittests_from_worker(
    artifact: dict,
    endpoint_queue: queue.Queue,
    test_output_dir: Path,          # FIX: explicit output dir, not derived from filepath
    progress_lock: threading.Lock,  # FIX: thread-safe progress reporting
    progress_counter: list,         # [completed, total] - mutable via index
) -> None:
    """
    Checks out an available endpoint from the queue, sends the payload to the
    local worker with exponential-backoff retry, writes the result, and returns
    the endpoint to the queue.

    Fixed issues:
      - Retry logic: up to MAX_RETRIES attempts with exponential backoff + jitter.
      - Fragile markdown strip: uses regex to handle trailing-newline fences.
      - Hard-coded JSON path: .get() with full response body logged on failure.
      - Test output dir: passed explicitly rather than derived from artifact path.
      - Progress tracking: thread-safe counter reported under a lock.
    """
    valid_langs = {"python", "py", "cpp", "c", "bash", "sh"}
    if artifact["language"].lower() not in valid_langs:
        print(f"[-] Skipping generation for non-code artifact: {artifact['filename']}")
        return

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

    # FIX: Retry loop with exponential backoff and jitter.
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(endpoint_url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()

            # FIX: Use .get() at every level; log the full body on failure.
            choices = result.get("choices")
            if not choices:
                print(
                    f"[!] FAILED [{port}]: Unexpected response body for "
                    f"{artifact['filename']}: {result}"
                )
                break

            test_code = choices[0].get("message", {}).get("content", "")
            if not test_code:
                print(
                    f"[!] FAILED [{port}]: Empty content in response for "
                    f"{artifact['filename']}: {result}"
                )
                break

            # FIX: Robust fence stripping.
            test_code = _strip_markdown_fences(test_code)

            test_output_dir.mkdir(parents=True, exist_ok=True)
            test_filename = f"test_{artifact['filename']}"
            test_filepath = test_output_dir / test_filename

            with open(test_filepath, "w", encoding="utf-8") as f:
                f.write(test_code + "\n")

            # FIX: Thread-safe progress reporting.
            with progress_lock:
                progress_counter[0] += 1
                done, total = progress_counter
                print(
                    f"[+] SUCCESS [{port}] ({done}/{total}): "
                    f"Tests for {artifact['filename']} -> {test_filename}"
                )

            last_exception = None
            break  # success - exit retry loop

        except requests.exceptions.RequestException as exc:
            last_exception = exc
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, RETRY_JITTER)
                print(
                    f"[~] RETRY [{port}] attempt {attempt}/{MAX_RETRIES} for "
                    f"{artifact['filename']} in {delay:.1f}s: {exc}"
                )
                time.sleep(delay)
            # else: fall through to the failure log below

    if last_exception is not None:
        print(
            f"[!] FAILED [{port}]: All {MAX_RETRIES} attempts exhausted for "
            f"{artifact['filename']}: {last_exception}"
        )

    # Always return the endpoint regardless of outcome.
    endpoint_queue.put(endpoint_url)


if __name__ == "__main__":
    MARKDOWN_SOURCE = "POLISHED_SYNTHESIS.md"
    OUTPUT_WORKSPACE = "./extracted_workspace"
    TEST_OUTPUT_DIR = Path(OUTPUT_WORKSPACE) / "tests"

    endpoint_queue: queue.Queue = queue.Queue()
    for ep in WORKER_ENDPOINTS:
        endpoint_queue.put(ep)

    if not os.path.exists(MARKDOWN_SOURCE):
        print(
            f"Error: Could not find {MARKDOWN_SOURCE}. "
            "Please ensure the file is in the current directory."
        )
    else:
        with open(MARKDOWN_SOURCE, "r", encoding="utf-8") as fh:
            md_content = fh.read()

        print("=== Phase 1: Local Extraction ===")
        artifacts = extract_code_blocks(md_content, OUTPUT_WORKSPACE)

        print(f"\n=== Phase 2: Parallelised Test Generation ===")
        print(f"[*] {len(artifacts)} artifacts extracted.")
        print(f"[*] Initialising ThreadPoolExecutor with {len(WORKER_ENDPOINTS)} workers...")

        progress_lock = threading.Lock()
        progress_counter = [0, len(artifacts)]  # [completed, total]

        # FIX: Wrap executor in try/finally so Ctrl-C or SIGTERM cancel
        # pending futures instead of blocking indefinitely on shutdown.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(WORKER_ENDPOINTS))
        futures = []
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

            # FIX: Iterate as_completed and call result() so that any
            # uncaught exception inside a thread is surfaced here rather
            # than silently swallowed.
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"[!] Unhandled exception in worker thread: {exc}")

        except KeyboardInterrupt:
            print("\n[!] Interrupted - cancelling pending tasks...")
            for f in futures:
                f.cancel()
        finally:
            # cancel_futures=True (Python 3.9+) drops queued-but-not-started work immediately.
            executor.shutdown(wait=False, cancel_futures=True)

        print("\n=== Pipeline execution complete ===")
