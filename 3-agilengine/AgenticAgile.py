import os
import re
import time
import json
import queue
import fcntl
import logging
import argparse
import threading
import requests
import concurrent.futures
from datetime import datetime
from pathlib import Path

# ==============================================================================
# Configuration & Directory Setup
# ==============================================================================

RAW_DIR = Path(os.getenv("RAW_DIR", "raw"))
WIKI_DIR = Path(os.getenv("WIKI_DIR", "wiki"))

RAW_DIR.mkdir(parents=True, exist_ok=True)
WIKI_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = WIKI_DIR / "project_state.json"
STATE_LOCK_FILE = WIKI_DIR / "project_state.lock"
WIKI_SYNTHESIS_FILE = WIKI_DIR / "DAILY_SYNTHESIS.md"
RAW_INDEX_FILE = WIKI_DIR / "RAWINDEX.md"

VALID_STATES = {"active", "in_progress", "blocked", "completed", "invalid"}

# Approximate character budget for existing-task context sent to the orchestrator.
# Keeps prompt size predictable regardless of how many tasks accumulate.
MAX_CONTEXT_CHARS = 8_000

# Updated logging format for a cleaner CLI interface
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Removed timestamp/level from raw output for cleaner CLI progress tracking
)
log = logging.getLogger(__name__)

def timestamped_log(msg: str) -> str:
    """Helper to prepend just the time for specific logs while keeping formatting clean."""
    return f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"

# ==============================================================================
# State & Index Management
# ==============================================================================

def load_state() -> dict:
    """Load project state under an exclusive file lock to prevent concurrent corruption."""
    with open(STATE_LOCK_FILE, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        if STATE_FILE.exists():
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    log.warning(timestamped_log("[!] State file is corrupt; starting fresh."))
    return {
        "tasks": {},
        "linting_violations": [],
        "failed_chunks": [],
        "last_updated": None,
    }

def save_state(state: dict) -> None:
    """Write project state under an exclusive file lock."""
    state["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(STATE_LOCK_FILE, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)

def load_processed_index() -> set:
    """Return the set of relative paths already recorded in RAWINDEX.md."""
    processed: set = set()
    if RAW_INDEX_FILE.exists():
        with open(RAW_INDEX_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("- "):
                    rel_path = line[2:].split(" | Processed:")[0].strip()
                    processed.add(rel_path)
    return processed

_raw_index_lock = threading.Lock()

def mark_file_processed(file_path: Path) -> None:
    """Append a successfully processed file to RAWINDEX.md (thread-safe, no duplicates)."""
    rel_path = file_path.relative_to(RAW_DIR).as_posix()

    with _raw_index_lock:
        existing = load_processed_index()
        if rel_path in existing:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(RAW_INDEX_FILE, "a", encoding="utf-8") as f:
            f.write(f"- {rel_path} | Processed: {timestamp}\n")

# ==============================================================================
# LLM Cluster & Session Management
# ==============================================================================

WORKER_ENDPOINTS = ["http://localhost:8033/v1/chat/completions"]
ORCHESTRATOR_ENDPOINT = "http://192.168.2.134:8033/v1/chat/completions"
MAX_SESSIONS_PER_ENDPOINT = 2

assert ORCHESTRATOR_ENDPOINT not in WORKER_ENDPOINTS, (
    "ORCHESTRATOR_ENDPOINT must not overlap with WORKER_ENDPOINTS. "
    "Overlapping endpoints cause pool contention and break session isolation."
)

class LLMClusterManager:
    def __init__(self) -> None:
        self.worker_pool: queue.Queue = queue.Queue()
        for endpoint in WORKER_ENDPOINTS:
            for _ in range(MAX_SESSIONS_PER_ENDPOINT):
                self.worker_pool.put(endpoint)

    def query(
        self,
        prompt: str,
        system_prompt: str = "",
        is_orchestrator: bool = False,
        max_retries: int = 3,
        requires_json: bool = False,
    ) -> tuple[bool, str | None]:
        if is_orchestrator:
            return self._send(ORCHESTRATOR_ENDPOINT, prompt, system_prompt, max_retries, requires_json)

        endpoint = self.worker_pool.get()
        try:
            return self._send(endpoint, prompt, system_prompt, max_retries, requires_json)
        finally:
            self.worker_pool.put(endpoint)

    def _send(
        self,
        endpoint: str,
        prompt: str,
        system_prompt: str,
        max_retries: int,
        requires_json: bool,
    ) -> tuple[bool, str | None]:
        payload = {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2 if requires_json else 0.7,
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(endpoint, json=payload, timeout=300)
                response.raise_for_status()
                result = response.json()["choices"][0]["message"]["content"].strip()
                return True, result
            except Exception as e:
                log.warning(
                    timestamped_log(f"      [!] Node {endpoint} failed: {e}. Retrying ({attempt + 1}/{max_retries})...")
                )
                time.sleep(2 ** attempt)

        return False, None

cluster = LLMClusterManager()

# ==============================================================================
# Micro-Task Dispatcher
# ==============================================================================

def chunk_markdown_semantically(text: str, max_chars: int = 4000) -> list[str]:
    """Split *text* on Markdown H1/H2/H3 headers."""
    tokens = re.split(r"(?m)(^#{1,3} [^\n]+\n?)", text)
    sections: list[str] = []
    i = 0
    if tokens and not re.match(r"^#{1,3} ", tokens[0]):
        preamble = tokens[0].strip()
        if preamble:
            sections.append(preamble)
        i = 1

    while i < len(tokens) - 1:
        header = tokens[i].strip()
        body = tokens[i + 1].strip() if (i + 1) < len(tokens) else ""
        section = f"{header}\n{body}".strip() if header else body
        if section:
            sections.append(section)
        i += 2

    chunks: list[str] = []
    for section in sections:
        if len(section) <= max_chars:
            chunks.append(section)
        else:
            sub_sections = section.split("\n\n")
            current_chunk: list[str] = []
            current_len = 0
            for sub in sub_sections:
                if current_len + len(sub) > max_chars and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [sub]
                    current_len = len(sub)
                else:
                    current_chunk.append(sub)
                    current_len += len(sub)
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))

    return chunks

def dispatch_jobs_in_chunks(
    large_text: str, prompt_template: str, system_prompt: str = ""
) -> tuple[list[str], list[dict]]:
    chunks = chunk_markdown_semantically(large_text)
    if not chunks:
        return [], []

    log.info(f"   -> [PHASE 1] Chunking complete. Dispatching {len(chunks)} chunks to worker nodes...")

    results: list[str | None] = [None] * len(chunks)
    failed_chunks: list[dict] = []
    total_capacity = len(WORKER_ENDPOINTS) * MAX_SESSIONS_PER_ENDPOINT

    with concurrent.futures.ThreadPoolExecutor(max_workers=total_capacity) as executor:
        future_to_index = {
            executor.submit(cluster.query, prompt_template.format(chunk=chunk), system_prompt): i
            for i, chunk in enumerate(chunks)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                success, output = future.result()
                if success:
                    results[idx] = output
                else:
                    failed_chunks.append(
                        {"type": "extraction_failure", "payload": chunks[idx][:200] + "..."}
                    )
            except Exception as exc:
                failed_chunks.append({"type": "exception", "error": str(exc)})

    return [r for r in results if r is not None], failed_chunks

# ==============================================================================
# Core Operations
# ==============================================================================

def ingest_and_merge_source(source_path: Path, current_idx: int, total_files: int) -> bool:
    """Read *source_path*, merge extracted insights into state. Returns True on success."""
    rel_path = source_path.relative_to(RAW_DIR)
    
    log.info(timestamped_log(f"--- [FILE {current_idx}/{total_files}] Processing: {rel_path} ---"))

    try:
        with open(source_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
    except Exception as e:
        log.error(f" [!] Error reading {rel_path}: {e}")
        return False

    # Phase 1: Distributed Extraction
    content_prompt = (
        "Extract bullet points of actionable tasks, potential technical blockers, "
        "or core architectural configurations from this artifact segment:\n\n{chunk}"
    )
    raw_tasks_list, extraction_failures = dispatch_jobs_in_chunks(raw_content, content_prompt)

    state = load_state()
    if extraction_failures:
        state["failed_chunks"].extend(extraction_failures)
        log.warning(f"   -> [PHASE 1] Warning: {len(extraction_failures)} chunks failed extraction.")
    else:
        log.info(f"   -> [PHASE 1] Distributed extraction successful.")

    if not raw_tasks_list:
        log.info(f"   -> [SKIP] No actionable signals found in {rel_path}.")
        return True

    # Phase 2: Context-Aware Orchestrator Merge
    log.info(f"   -> [PHASE 2] Reconciling signals with Orchestrator node...")

    existing_tasks = list(state["tasks"].values())
    existing_tasks.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

    existing_tasks_context = json.dumps(existing_tasks[:50], indent=2)[:MAX_CONTEXT_CHARS]
    new_signals_context = "\n---\n".join(raw_tasks_list)

    merge_prompt = f"""
Reconcile these newly extracted technical signals with the current agile project state.
Update existing tasks if the content is related, or create new tasks if the signal is fresh.
Ensure blockers are explicitly flagged.

CURRENT STATE (Recent Tasks):
{existing_tasks_context}

NEW SIGNALS FROM {source_path.name}:
{new_signals_context}
"""
    success, merged_output = cluster.query(
        prompt=merge_prompt,
        system_prompt=(
            "Return ONLY a flat JSON array of objects with keys: 'id' (slug), "
            "'content', 'status', 'confidence'. "
            "No markdown formatting, code blocks, or explanatory text."
        ),
        is_orchestrator=True,
        requires_json=True,
    )

    if success:
        try:
            clean_json = re.search(r"\[.*\]", merged_output, re.DOTALL)
            merged_tasks = json.loads(clean_json.group() if clean_json else merged_output)
            
            updated_count = 0
            for task in merged_tasks:
                slug = task.get("id")
                if slug:
                    task["updated_at"] = datetime.now().isoformat()
                    task["last_source"] = source_path.name
                    raw_status = task.get("status")
                    if raw_status not in VALID_STATES:
                        task["status"] = "in_progress"
                    state["tasks"][slug] = task
                    updated_count += 1

            log.info(f"   -> [SUCCESS] Merged {updated_count} tasks into state.")
            save_state(state)
            return True

        except Exception as e:
            log.error(f"   -> [!] JSON parse error during merge: {e}")
            return False

    log.error(f"   -> [!] Orchestrator failed to merge signals.")
    return False

def generate_daily_synthesis(*, block: bool = True) -> threading.Thread | None:
    """Compile the entire project state into a final Markdown summary."""
    def _run() -> None:
        log.info(timestamped_log("\n[*] SYNTHESIS: Generating Daily Synthesis Report..."))
        state = load_state()

        if not state["tasks"]:
            log.warning(" [!] State is empty. No report generated.")
            return

        synthesis_prompt = (
            "Summarize this agile project state into a readable executive markdown report. "
            "Group cleanly by status (Blocked, Active, Completed). Keep it concise.\n\n"
            f"STATE:\n{json.dumps(state['tasks'])}"
        )
        
        success, response = cluster.query(synthesis_prompt, is_orchestrator=True)

        if success:
            with open(WIKI_SYNTHESIS_FILE, "w", encoding="utf-8") as f:
                f.write(response)
            log.info(timestamped_log(f"[+] Synthesis successfully saved to {WIKI_SYNTHESIS_FILE}"))
        else:
            log.error(timestamped_log(f"[!] Synthesis failed after all retries. {WIKI_SYNTHESIS_FILE} was NOT updated."))

    if block:
        _run()
        return None

    t = threading.Thread(target=_run, name="synthesis", daemon=False)
    t.start()
    return t

# ==============================================================================
# Main Execution Loop
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Deep-scan agentic control loop.")
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help="Only regenerate the daily synthesis from the current state; skip ingestion.",
    )
    parser.add_argument(
        "--async-synthesis",
        action="store_true",
        dest="async_synthesis",
        help="Run synthesis in a background thread instead of blocking after ingestion.",
    )
    args = parser.parse_args()

    if args.synthesize:
        generate_daily_synthesis(block=True)
        return

    log.info("\n==================================================")
    log.info(timestamped_log("STARTING DEEP-SCAN AGENTIC CONTROL LOOP"))
    log.info(f"[*] Target Directory: {RAW_DIR.absolute()}")
    log.info("==================================================\n")

    processed_files = load_processed_index()

    extensions = ("*.txt", "*.md", "*.csv")
    raw_files: list[Path] = []
    for ext in extensions:
        raw_files.extend(RAW_DIR.rglob(ext))

    # Filter files strictly to what needs processing
    files_to_process = [
        f for f in raw_files 
        if f.relative_to(RAW_DIR).as_posix() not in processed_files
    ]
    
    total_files = len(files_to_process)

    if total_files == 0:
        log.info(timestamped_log("[~] No new files to process. Directory is up-to-date."))
    else:
        log.info(timestamped_log(f"[*] Discovered {total_files} new file(s) for processing.\n"))
        new_files_processed = 0

        for current_idx, file_path in enumerate(files_to_process, start=1):
            success = ingest_and_merge_source(file_path, current_idx, total_files)

            if success:
                mark_file_processed(file_path)
                new_files_processed += 1

        if new_files_processed > 0:
            synthesis_thread = generate_daily_synthesis(block=not args.async_synthesis)
            if synthesis_thread is not None:
                log.info(timestamped_log(f"[*] Synthesis running in background (thread: {synthesis_thread.name})."))
                synthesis_thread.join()  
        else:
            log.info("\n[*] No files were successfully processed. Existing synthesis remains valid.")

    log.info("\n==================================================")
    log.info(timestamped_log("PIPELINE FINISHED"))
    log.info("==================================================")

if __name__ == "__main__":
    main()

