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

# These will be dynamically initialized based on the CLI target path
RAW_DIR = None
WIKI_DIR = None
STATE_FILE = None
STATE_LOCK_FILE = None
WIKI_SYNTHESIS_FILE = None
RAW_INDEX_FILE = None

VALID_STATES = {"active", "in_progress", "blocked", "completed", "invalid"}

# Context Limits halved to prevent KV cache thrashing and TTFT bottlenecks
# Orchestrator: Reduced from 120k to 60k chars
ORCHESTRATOR_MAX_CHARS = 60000 
# Worker: Reduced from 200k to 100k chars
WORKER_CHUNK_CHARS = 100000

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
log = logging.getLogger(__name__)

def timestamped_log(msg: str) -> str:
    """Helper to prepend just the time for specific logs while keeping formatting clean."""
    return f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"

def init_paths(base_dir: Path) -> None:
    """Dynamically initialize and create directories bound to the target project path."""
    global RAW_DIR, WIKI_DIR, STATE_FILE, STATE_LOCK_FILE, WIKI_SYNTHESIS_FILE, RAW_INDEX_FILE
    
    RAW_DIR = base_dir
    WIKI_DIR = base_dir / os.getenv("WIKI_DIR", "wiki")
    WIKI_DIR.mkdir(parents=True, exist_ok=True)

    STATE_FILE = WIKI_DIR / "project_state.json"
    STATE_LOCK_FILE = WIKI_DIR / "project_state.lock"
    WIKI_SYNTHESIS_FILE = WIKI_DIR / "DAILY_SYNTHESIS.md"
    RAW_INDEX_FILE = WIKI_DIR / "RAWINDEX.md"

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

raw_worker_endpoints = os.getenv("WORKER_ENDPOINTS", "http://192.168.2.137:8034/v1/chat/completions")
WORKER_ENDPOINTS = [e.strip() for e in raw_worker_endpoints.split(",") if e.strip()]
WORKER_API_KEY = os.getenv("WORKER_API_KEY", "local-sk")

ORCHESTRATOR_ENDPOINT = os.getenv("ORCHESTRATOR_ENDPOINT", "http://192.168.2.137:8080/v1/chat/completions")
ORCH_API_KEY = os.getenv("ORCH_API_KEY", "local-sk")

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

        headers = {"Content-Type": "application/json"}
        if endpoint == ORCHESTRATOR_ENDPOINT:
            headers["Authorization"] = f"Bearer {ORCH_API_KEY}"
        else:
            headers["Authorization"] = f"Bearer {WORKER_API_KEY}"

        for attempt in range(max_retries):
            try:
                # 10 minute timeout to allow KV cache building for large contexts
                response = requests.post(endpoint, headers=headers, json=payload, timeout=600)
                response.raise_for_status()
                
                raw_output = response.json()["choices"][0]["message"]["content"].strip()
                # Strict ASCII enforcement to prevent downstream JSON parse errors
                ascii_output = raw_output.encode("ascii", "ignore").decode("ascii")
                return True, ascii_output
                
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

def chunk_markdown_semantically(text: str, max_chars: int = WORKER_CHUNK_CHARS) -> list[str]:
    """Split *text* on Markdown headers, utilizing the scaled worker context limit."""
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

    log.info(f"   -> [PHASE 1] Dispatching {len(chunks)} chunk(s) to worker nodes...")

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

def extract_json_array(raw_text: str) -> str:
    """Safely extract a JSON array from LLM output, stripping markdown fences."""
    cleaned_text = re.sub(r'```json\s*', '', raw_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\n?```\s*', '', cleaned_text)
    match = re.search(r'\[.*\]', cleaned_text, re.DOTALL)
    return match.group(0) if match else raw_text

def pack_files_into_batches(files_to_process: list[Path], max_chars: int) -> list[tuple[list[Path], str]]:
    """Packs multiple files into grouped batches to maximize worker context windows."""
    batches = []
    current_batch_text = ""
    current_batch_files = []

    for file_path in files_to_process:
        try:
            with open(file_path, "r", encoding="ascii", errors="ignore") as f:
                content = f"--- SOURCE: {file_path.name} ---\n{f.read().strip()}"
        except Exception as e:
            log.warning(timestamped_log(f"[!] Could not read {file_path.name}: {e}"))
            continue

        if len(current_batch_text) + len(content) > max_chars and current_batch_text:
            batches.append((current_batch_files, current_batch_text))
            current_batch_text = content
            current_batch_files = [file_path]
        else:
            current_batch_text += "\n\n" + content if current_batch_text else content
            current_batch_files.append(file_path)

    if current_batch_text:
        batches.append((current_batch_files, current_batch_text))

    return batches

def ingest_and_merge_batch(batch_text: str, batch_files: list[Path], current_idx: int, total_batches: int) -> bool:
    """Dispatches a packed batch of multiple files to the swarm and merges the output."""
    file_names = [f.name for f in batch_files]
    log.info(timestamped_log(f"--- [BATCH {current_idx}/{total_batches}] Processing {len(batch_files)} file(s) ---"))
    
    if len(batch_files) <= 3:
        log.info(f"   -> Included: {', '.join(file_names)}")
    else:
        log.info(f"   -> Included: {', '.join(file_names[:3])} and {len(batch_files) - 3} more...")

    # Phase 1: Distributed Extraction (Workers)
    system_prompt = (
        "You are a ruthless, highly technical Lead Engineer. "
        "Read the provided artifacts and extract a succinct, actionable "
        "list of explicit TO-DOs, architectural requirements, and implementation tasks. "
        "\n\nCRITICAL DIRECTIVES: "
        "\n1. TEST TELEMETRY: Hunt for any unit test execution logs, reports, or telemetry. "
        "Create a distinct 'Test Execution Status' section detailing passes and failures. "
        "Convert any failures into high-priority TO-DO items."
        "\n2. EMBED ARTIFACTS: For EVERY task or failed test, extract and embed the relevant "
        "source artifact directly beneath the task description. Format code or tracebacks using "
        "proper markdown code fences and explicitly label the source filename."
        "\n3. STRICT ASCII ONLY: Do NOT use emojis or specialized unicode symbols."
    )
    
    content_prompt = "Artifacts Payload:\n\n{chunk}"
    
    raw_tasks_list, extraction_failures = dispatch_jobs_in_chunks(batch_text, content_prompt, system_prompt=system_prompt)

    state = load_state()
    if extraction_failures:
        state["failed_chunks"].extend(extraction_failures)
        log.warning(f"   -> [PHASE 1] Warning: {len(extraction_failures)} chunks failed extraction.")
    else:
        log.info(f"   -> [PHASE 1] Distributed extraction successful.")

    if not raw_tasks_list:
        log.info(f"   -> [SKIP] No actionable signals found in this batch.")
        return True

    # Phase 2: Context-Aware Orchestrator Merge
    log.info(f"   -> [PHASE 2] Reconciling signals with Orchestrator node...")

    existing_tasks = list(state["tasks"].values())
    existing_tasks.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

    existing_tasks_context = json.dumps(existing_tasks[:50], indent=2)[:ORCHESTRATOR_MAX_CHARS]
    new_signals_context = "\n---\n".join(raw_tasks_list)

    merge_prompt = f"""
Reconcile these newly extracted technical signals with the current agile project state.
Update existing tasks if the content is related, or create new tasks if the signal is fresh.
Ensure blockers are explicitly flagged.

CURRENT STATE (Recent Tasks):
{existing_tasks_context}

NEW SIGNALS FROM BATCH:
{new_signals_context}
"""
    success, merged_output = cluster.query(
        prompt=merge_prompt,
        system_prompt=(
            "Return ONLY a flat JSON array of objects with keys: 'id' (slug), "
            "'content', 'status', 'confidence'. "
            "No markdown formatting, code blocks, or explanatory text. Strict ASCII only."
        ),
        is_orchestrator=True,
        requires_json=True,
    )

    if success:
        try:
            clean_json_str = extract_json_array(merged_output)
            merged_tasks = json.loads(clean_json_str)
            
            updated_count = 0
            for task in merged_tasks:
                slug = task.get("id")
                if slug:
                    task["updated_at"] = datetime.now().isoformat()
                    task["last_source"] = f"Batch containing {file_names[0]}"
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
            "Group cleanly by status (Blocked, Active, Completed). Keep it concise. Strict ASCII only.\n\n"
            f"STATE:\n{json.dumps(state['tasks'])[:ORCHESTRATOR_MAX_CHARS]}"
        )
        
        success, response = cluster.query(synthesis_prompt, is_orchestrator=True)

        if success:
            with open(WIKI_SYNTHESIS_FILE, "w", encoding="ascii", errors="ignore") as f:
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
        "project_path",
        type=str,
        help="Path to the target project directory to bind the execution context.",
    )
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

    target_dir = Path(args.project_path).resolve()
    
    if not target_dir.exists() or not target_dir.is_dir():
        log.error(f"Fatal: Directory '{target_dir}' does not exist or is not a directory.")
        return

    # Strictly bind operations into the target directory
    os.chdir(target_dir)
    init_paths(target_dir)

    if args.synthesize:
        generate_daily_synthesis(block=True)
        return

    log.info("\n==================================================")
    log.info(timestamped_log("STARTING DEEP-SCAN AGENTIC CONTROL LOOP"))
    log.info(f"[*] Target Directory: {target_dir}")
    log.info("==================================================\n")

    processed_files = load_processed_index()

    extensions = ("*.txt", "*.md", "*.csv", "*.json")
    raw_files: list[Path] = []
    
    for ext in extensions:
        for f in RAW_DIR.rglob(ext):
            # Exclude our internal WIKI_DIR to prevent infinite loops
            if WIKI_DIR in f.parents:
                continue
            raw_files.append(f)

    files_to_process = [
        f for f in raw_files 
        if f.relative_to(RAW_DIR).as_posix() not in processed_files
    ]
    
    total_files = len(files_to_process)

    if total_files == 0:
        log.info(timestamped_log("[~] No new files to process. Directory is up-to-date."))
    else:
        log.info(timestamped_log(f"[*] Discovered {total_files} new file(s) for processing."))
        
        # Pack the files into ~100k character batches for the worker nodes
        batches = pack_files_into_batches(files_to_process, WORKER_CHUNK_CHARS)
        total_batches = len(batches)
        
        log.info(timestamped_log(f"[*] Compacted into {total_batches} processing batch(es).\n"))
        
        new_files_processed = 0

        for current_idx, (batch_files, batch_text) in enumerate(batches, start=1):
            success = ingest_and_merge_batch(batch_text, batch_files, current_idx, total_batches)

            if success:
                for file_path in batch_files:
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
