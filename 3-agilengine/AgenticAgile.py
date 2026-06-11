import os
import re
import time
import json
import queue
import requests
import concurrent.futures
from datetime import datetime
from pathlib import Path

# ==============================================================================
# Configuration & Directory Setup
# ==============================================================================

# Uses environment variables if set, otherwise defaults to local directories.
RAW_DIR = Path(os.getenv("RAW_DIR", "raw"))
WIKI_DIR = Path(os.getenv("WIKI_DIR", "wiki"))

# Ensure base directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
WIKI_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = WIKI_DIR / "project_state.json"
WIKI_SYNTHESIS_FILE = WIKI_DIR / "DAILY_SYNTHESIS.md"
RAW_INDEX_FILE = WIKI_DIR / "RAWINDEX.md"

VALID_STATES = {"active", "in_progress", "blocked", "completed", "invalid"}

# ==============================================================================
# State & Index Management
# ==============================================================================

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass
    return {
        "tasks": {},
        "linting_violations": [],
        "failed_chunks": [],     
        "last_updated": None
    }

def save_state(state):
    state["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4)

def load_processed_index() -> set:
    """Loads the set of already processed file paths from RAWINDEX.md."""
    processed = set()
    if RAW_INDEX_FILE.exists():
        with open(RAW_INDEX_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("- "):
                    # Extract just the relative path before the separator
                    rel_path = line[2:].split(" | Processed:")[0].strip()
                    processed.add(rel_path)
    return processed

def mark_file_processed(file_path: Path):
    """Appends a successfully processed file to RAWINDEX.md."""
    rel_path = file_path.relative_to(RAW_DIR).as_posix()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(RAW_INDEX_FILE, "a", encoding="utf-8") as f:
        f.write(f"- {rel_path} | Processed: {timestamp}\n")

# ==============================================================================
# LLM Cluster & Session Management
# ==============================================================================

WORKER_ENDPOINTS = ["http://localhost:8033/v1/chat/completions"]
ORCHESTRATOR_ENDPOINT = "http://192.168.2.134:8033/v1/chat/completions"
MAX_SESSIONS_PER_ENDPOINT = 2

class LLMClusterManager:
    def __init__(self):
        self.worker_pool = queue.Queue()
        for endpoint in WORKER_ENDPOINTS:
            for _ in range(MAX_SESSIONS_PER_ENDPOINT):
                self.worker_pool.put(endpoint)

    def query(self, prompt, system_prompt="", is_orchestrator=False, max_retries=3, requires_json=False):
        endpoint = ORCHESTRATOR_ENDPOINT if is_orchestrator else self.worker_pool.get()
        
        payload = {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2 if requires_json else 0.7
        }

        for attempt in range(max_retries):
            try:
                print(f"      -> [LLM Cluster] Sending Request to: {endpoint}...")
                
                response = requests.post(endpoint, json=payload, timeout=300)
                response.raise_for_status()
                
                result = response.json()['choices'][0]['message']['content'].strip()
                
                if not is_orchestrator:
                    self.worker_pool.put(endpoint)
                return True, result
                
            except Exception as e:
                print(f"      -> [LLM Cluster] Node {endpoint} failed: {e}. Retrying...")
                time.sleep(2 ** attempt)
        
        if not is_orchestrator:
            self.worker_pool.put(endpoint)
        return False, None

cluster = LLMClusterManager()

# ==============================================================================
# Micro-Task Dispatcher
# ==============================================================================

def chunk_text(text, chunk_size=6, overlap=1):
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    chunks, i = [], 0
    while i < len(lines):
        chunks.append('\n'.join(lines[i:i + chunk_size]))
        i += (chunk_size - overlap)
        if i >= len(lines):
            break
    return chunks

def dispatch_jobs_in_chunks(large_text, prompt_template, system_prompt=""):
    chunks = chunk_text(large_text)
    if not chunks:
        return [], []
        
    print(f"      -> [Dispatcher] Processing {len(chunks)} chunks in parallel.")
    
    results, failed_chunks = [None] * len(chunks), []
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
                    failed_chunks.append({"type": "extraction_failure", "payload": chunks[idx]})
            except Exception as exc:
                failed_chunks.append({"type": "exception", "error": str(exc)})
                
    return [r for r in results if r is not None], failed_chunks

# ==============================================================================
# Core Operations
# ==============================================================================

def ingest_and_merge_source(source_path: Path) -> bool:
    """Reads a file, merges insights, and returns True if successful."""
    rel_path = source_path.relative_to(RAW_DIR)
    print(f"\n[*] Processing file: {rel_path}...")
    
    try:
        with open(source_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
    except Exception as e:
        print(f" [!] Error reading {rel_path}: {e}")
        return False

    # Phase 1: Distributed Extraction
    content_prompt = "Extract bullet points of actionable tasks, blockers, or architectural changes from this transcript segment:\n{chunk}"
    raw_tasks_list, extraction_failures = dispatch_jobs_in_chunks(raw_content, content_prompt)

    state = load_state()
    if extraction_failures:
        state["failed_chunks"].extend(extraction_failures)

    if not raw_tasks_list:
        print(f" [~] No actionable signals found in {rel_path}. Marking as processed.")
        return True 

    # Phase 2: Context-Aware Orchestrator Merge
    print(f"[*] MERGING: Orchestrator reconciling signals from {rel_path}...")
    
    existing_tasks_context = json.dumps(list(state["tasks"].values()), indent=2)
    new_signals_context = "\n".join(raw_tasks_list)

    merge_prompt = f"""
Reconcile these new signals with the current project state. 
Update existing tasks if the content is related, or create new tasks if the signal is fresh.
Return a valid JSON array of objects with keys: id (slug), content, status, confidence.

CURRENT STATE:
{existing_tasks_context}

NEW SIGNALS FROM {source_path.name}:
{new_signals_context}
"""
    success, merged_output = cluster.query(
        prompt=merge_prompt,
        system_prompt="Return ONLY a flat JSON array. No markdown code blocks.",
        is_orchestrator=True,
        requires_json=True
    )

    if success:
        try:
            clean_json = re.sub(r'```json|```', '', merged_output).strip()
            merged_tasks = json.loads(clean_json)
            for task in merged_tasks:
                slug = task.get("id")
                if slug:
                    task["updated_at"] = datetime.now().isoformat()
                    task["last_source"] = source_path.name
                    state["tasks"][slug] = task
                    
            print(f" [+] Reconciliation complete for {rel_path}.")
            save_state(state)
            return True 
            
        except Exception as e:
            print(f" [!] JSON Parse Error during merge: {e}")
            return False
            
    print(f" [!] Orchestrator failed to merge signals for {rel_path}.")
    return False

def generate_daily_synthesis():
    """Compiles the entire project state into a final markdown summary."""
    print("\n[*] SYNTHESIZING: Generating Daily Synthesis Report...")
    state = load_state()
    
    if not state['tasks']:
        print(" [!] State is empty. No report generated.")
        return

    synthesis_prompt = f"Summarize this project state into a readable executive markdown report. Group by status (Blocked, Active, Completed):\n{json.dumps(state['tasks'])}"
    success, response = cluster.query(synthesis_prompt, is_orchestrator=True)
    
    if success:
        with open(WIKI_SYNTHESIS_FILE, "w", encoding="utf-8") as f:
            f.write(response)
        print(f" [+] Synthesis saved to {WIKI_SYNTHESIS_FILE}")

# ==============================================================================
# Main Execution Loop
# ==============================================================================

if __name__ == "__main__":
    print("=== STARTING DEEP-SCAN AGENTIC CONTROL LOOP ===")
    print(f"[*] Target Raw Directory: {RAW_DIR.absolute()}")
    
    processed_files = load_processed_index()
    
    extensions = ("*.txt", "*.md", "*.csv")
    raw_files = []
    for ext in extensions:
        raw_files.extend(RAW_DIR.rglob(ext))
    
    if not raw_files:
        print(f"[!] No content found in {RAW_DIR.absolute()}.")
    else:
        new_files_processed = 0
        
        for file_path in raw_files:
            rel_path = file_path.relative_to(RAW_DIR).as_posix()
            
            # Efficient O(1) set lookup
            if rel_path in processed_files:
                print(f" [~] Skipping {rel_path} (Already in RAWINDEX.md)")
                continue
                
            success = ingest_and_merge_source(file_path)
            
            if success:
                mark_file_processed(file_path)
                new_files_processed += 1
        
        if new_files_processed > 0:
            generate_daily_synthesis()
        else:
            print("\n[*] No new files processed today. Existing Synthesis remains valid.")

    print("\n=== PIPELINE FINISHED ===")
