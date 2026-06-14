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
from datetime import datetime
from pathlib import Path
from openai import OpenAI

# ==============================================================================
# Configuration & Endpoints
# ==============================================================================

ORCHESTRATOR_ENDPOINTS = [
    "http://localhost:8080/v1"
]
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "nvidia_Orchestrator-8B-Q6_K.gguf")
ORCH_API_KEY = os.getenv("ORCH_API_KEY", "local-sk")
MAX_RETRIES = 3

WORKER_ENDPOINTS = [
    "http://localhost:8034/v1"
]
WORKER_MODEL = os.getenv("WORKER_MODEL", "Qwen3.5-9B-IQ4_XS.gguf")
WORKER_API_KEY = os.getenv("WORKER_API_KEY", "local-sk")

WORKER_PARALLEL_SLOTS = 2
WORKER_RETRIES = 3
ORCH_PARALLEL_SLOTS = 1
SYNTHESIS_CHUNK_SIZE = 3  

# ==============================================================================
# Helper: Fallback Token Estimator
# ==============================================================================

def estimate_tokens(text: str) -> int:
    return len(str(text)) // 4

# ==============================================================================
# Phase 1: Hyper-Granular Decomposition
# ==============================================================================

def extract_json_array(raw_text: str) -> str:
    """Robust extraction walking characters and tracking bracket depth."""
    cleaned_text = re.sub(r'```json\s*', '', raw_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\n?```\s*', '', cleaned_text)
    
    start_idx = cleaned_text.find('[')
    if start_idx == -1: 
        return ""
        
    depth = 0
    for i in range(start_idx, len(cleaned_text)):
        if cleaned_text[i] == '[': 
            depth += 1
        elif cleaned_text[i] == ']':
            depth -= 1
            if depth == 0:
                return cleaned_text[start_idx:i+1]
    return ""

def decompose_to_atomic_pieces(large_query: str) -> tuple:
    print(f"\n[1] INGRESS: Analyzing massive query...\n    Length: {len(large_query)} characters", flush=True)

    system_prompt = """You are an algorithmic micro-task decomposer.
Your sole purpose is to take a large, complex query or task and shatter it into atomic, independent pieces for parallel processing.
Output ONLY a valid, flat JSON array of strings. No markdown formatting, no conversational text."""

    client = OpenAI(
        base_url=ORCHESTRATOR_ENDPOINTS[0], 
        api_key=ORCH_API_KEY,
        timeout=1800.0,
        max_retries=0
    )

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[2] DECOMPOSITION: Engaging atomic breakdown via {ORCHESTRATOR_ENDPOINTS[0]} (Attempt {attempt}/{MAX_RETRIES})...", flush=True)
        raw_output = ""
        prompt_tokens, comp_tokens = 0, 0
        
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=ORCHESTRATOR_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Decompose this to the atomic level:\n\n{large_query}"}
                ],
                temperature=0.7, max_tokens=40960, stream=True,
                stream_options={"include_usage": True}
            )
            
            print("    [~] Streaming Live Generation:\n    >> ", end="", flush=True)
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    text_chunk = chunk.choices[0].delta.content
                    print(text_chunk, end="", flush=True)
                    raw_output += text_chunk
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                    comp_tokens = chunk.usage.completion_tokens
                    
            print("\n", flush=True) 
            cleaned_output = extract_json_array(raw_output)
            if not cleaned_output: raise ValueError("Could not locate JSON array.")
            
            atomic_pieces = json.loads(cleaned_output)
            if prompt_tokens == 0 and comp_tokens == 0:
                prompt_tokens = estimate_tokens(system_prompt + large_query)
                comp_tokens = estimate_tokens(raw_output)
                
            elapsed = round(time.time() - start_time, 2)
            print(f"    [+] Success! Shattered into {len(atomic_pieces)} distinct micro-pieces in {elapsed}s.", flush=True)
            return atomic_pieces, prompt_tokens, comp_tokens

        except Exception as e:
            print(f"\n    [!] Decomposition Error: {e}", flush=True)
            time.sleep(2)

    return [large_query], estimate_tokens(large_query), 10

# ==============================================================================
# Phase 2: Audit Trail Setup
# ==============================================================================

def export_to_split_files(pieces: list, work_dir: Path) -> Path:
    if len(pieces) <= 1: return work_dir
        
    print("\n[3] QUEUE EXPORT: Saving task matrix to target directory...", flush=True)
    tasks_dir = work_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    
    for idx, piece in enumerate(pieces, start=1):
        filepath = tasks_dir / f"task{idx:03d}.md"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"{piece.strip()}\n")
            
    return work_dir

# ==============================================================================
# Pipeline Processors
# ==============================================================================

def process_subtask(task_id: int, task_prompt: str, endpoint: str, slot_name: str, original_query: str, run_dir: Path) -> dict:
    print(f"    -> [{slot_name}] Worker{task_id:02d} Dispatched to {endpoint} | Task: '{task_prompt[:40]}...' ", flush=True)
    
    worker_client = OpenAI(
        base_url=endpoint, 
        api_key=WORKER_API_KEY,
        timeout=1800.0,
        max_retries=0
    )
    start_time = time.time()
    
    system_instruction = (
        "You are an autonomous, highly-capable worker agent equipped with advanced reasoning. "
        "Think step-by-step to formulate a plan. You must execute your specific objective fully. "
        "CRITICAL: If your task involves writing code, creating configurations, or generating files, "
        "you MUST output the file contents wrapped exactly in these XML tags:\n"
        '<file path="filename.ext">\n[YOUR FILE CONTENT HERE]\n</file>\n'
    )
    user_instruction = f"BACKGROUND CONTEXT:\n{original_query}\n\nYOUR SPECIFIC OBJECTIVE:\n{task_prompt}"
    
    saved_artifacts = []
    status = "success"
    
    try:
        response = worker_client.chat.completions.create(
            model=WORKER_MODEL,
            messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": user_instruction}],
            temperature=0.4, 
            max_tokens=12288,
            frequency_penalty=1.1,
            presence_penalty=0.5,
            stop=["</file>"]
        )
        result_text = response.choices[0].message.content.strip()
        
        if "<file" in result_text and "</file>" not in result_text:
            result_text += "\n</file>"

        prompt_tokens = response.usage.prompt_tokens if response.usage else estimate_tokens(system_instruction + user_instruction)
        comp_tokens = response.usage.completion_tokens if response.usage else estimate_tokens(result_text)
        
        file_matches = re.finditer(r'<file\s+path="([^"]+)">([\s\S]*?)</file>', result_text, re.IGNORECASE)
        for match in file_matches:
            file_path, file_content = match.group(1).strip(), match.group(2).strip()
            safe_filename = os.path.basename(file_path)
            artifact_dir = run_dir / "artifacts" / f"thread{task_id:02d}"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            with open(artifact_dir / safe_filename, "w", encoding="utf-8") as af: 
                af.write(file_content)
            saved_artifacts.append(safe_filename)

        if len(result_text) < 20: status = "failed_validation"
            
    except Exception as e:
        result_text, status = f"Worker Error: {str(e)}", "error"
        prompt_tokens, comp_tokens = 0, 0

    elapsed = round(time.time() - start_time, 2)
    return {
        "id": task_id, "prompt": task_prompt, "result": result_text,
        "artifacts": saved_artifacts, "status": status,
        "prompt_tokens": prompt_tokens, "completion_tokens": comp_tokens,
        "total_tokens": prompt_tokens + comp_tokens, "elapsed": elapsed, 
        "tps": round(comp_tokens / elapsed, 2) if elapsed > 0 else 0,
        "slot": slot_name
    }

def parallel_chunk_synthesis(batch_id: int, tasks: list, endpoint: str, original_query: str) -> tuple:
    client = OpenAI(
        base_url=endpoint, 
        api_key=ORCH_API_KEY,
        timeout=1800.0,
        max_retries=0
    )
    
    system_prompt = (
        "You are a Level-1 Synthesis Node in a distributed cluster. "
        "Merge the following sequential worker reports into a coherent, deduplicated section. "
        "Retain all code blocks, configurations, and critical technical data. "
        "Output strictly in standard UTF-8."
    )
    batch_context = "\n\n".join([f"--- TASK {t['id']}: {t['prompt']} ---\n{t['result']}" for t in tasks])
    user_prompt = f"ORIGINAL QUERY: {original_query}\n\nREPORTS TO MERGE:\n{batch_context}"
        
    start_time = time.time()
    response = client.chat.completions.create(
        model=ORCHESTRATOR_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.3, max_tokens=40960
    )
    elapsed = round(time.time() - start_time, 2)
    
    res_content = response.choices[0].message.content.strip()
    p_tok = response.usage.prompt_tokens if response.usage else estimate_tokens(system_prompt + user_prompt)
    c_tok = response.usage.completion_tokens if response.usage else estimate_tokens(res_content)
    return batch_id, res_content, p_tok, c_tok, elapsed

def rolling_master_stitch(chunk_id: int, current_master: str, new_chunk: str, endpoint: str, original_query: str) -> tuple:
    client = OpenAI(
        base_url=endpoint, 
        api_key=ORCH_API_KEY,
        timeout=1800.0,
        max_retries=0
    )
    
    system_prompt = (
        "You are the Final Assembly Layer. Seamlessly weave the new sequential section into the existing master document. "
        "Expand the document logically. Do not drop existing data or code. Output strictly in standard UTF-8."
    )
    user_prompt = f"ORIGINAL QUERY: {original_query}\n\n--- CURRENT MASTER DOCUMENT ---\n{current_master}\n\n--- NEW SECTION {chunk_id} TO INTEGRATE ---\n{new_chunk}"
    
    start_time = time.time()
    response = client.chat.completions.create(
        model=ORCHESTRATOR_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.3, max_tokens=65536
    )
    elapsed = round(time.time() - start_time, 2)
    
    res_content = response.choices[0].message.content.strip()
    p_tok = response.usage.prompt_tokens if response.usage else estimate_tokens(system_prompt + user_prompt)
    c_tok = response.usage.completion_tokens if response.usage else estimate_tokens(res_content)
    return res_content, p_tok, c_tok, elapsed

# ==============================================================================
# Phase 3, 4 & 5: Continuous Map-Reduce Event Loop
# ==============================================================================

def execute_continuous_map_reduce(sub_tasks: list, original_query: str, run_dir: Path) -> tuple:
    total_tasks = len(sub_tasks)
    total_chunks = (total_tasks + SYNTHESIS_CHUNK_SIZE - 1) // SYNTHESIS_CHUNK_SIZE
    
    print(f"\n[4] CONTINUOUS MAP-REDUCE: Launching parallel workers, chunks, and rolling master stitch...", flush=True)

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

    event_queue = queue.Queue()
    stitch_queue = queue.Queue()

    # --- Thread Wrappers that emit to the Event Loop ---
    def worker_wrapper(tid: int, prompt: str):
        last_result = None
        for _ in range(WORKER_RETRIES):
            endpoint, slot_name = worker_queue.get()
            try:
                res = process_subtask(tid, prompt, endpoint, slot_name, original_query, run_dir)
                if res["status"] == "success": 
                    MAX_WORKER_TOKENS = 16000
                    if res.get("total_tokens", 0) > MAX_WORKER_TOKENS:
                        print(f"    [TRUNCATE] [{slot_name}] Worker{tid:02d} exceeded limit ({res['total_tokens']} tokens). Truncating to {MAX_WORKER_TOKENS}...", flush=True)
                        safe_char_limit = MAX_WORKER_TOKENS * 4
                        res["result"] = res["result"][:safe_char_limit] + "\n\n...[OUTPUT TRUNCATED DUE TO LENGTH LIMIT]..."
                        
                        res["completion_tokens"] = MAX_WORKER_TOKENS - res["prompt_tokens"]
                        res["total_tokens"] = MAX_WORKER_TOKENS

                    event_queue.put(("worker", res))
                    return  
                last_result = res
            except Exception as e:
                last_result = {"id": tid, "status": "error", "result": f"Failed: {str(e)}", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "elapsed": 0, "tps": 0, "slot": slot_name}
            finally:
                worker_queue.put((endpoint, slot_name))
            time.sleep(2) 
        event_queue.put(("worker", last_result))

    def chunk_wrapper(batch_id: int, tasks: list):
        for attempt in range(1, MAX_RETRIES + 1):
            endpoint, slot_name = orch_queue.get()
            try:
                print(f"    -> [{slot_name}] Chunk{batch_id:02d} Dispatched to {endpoint} | Compressing {len(tasks)} tasks...", flush=True)
                b_id, text, p_tok, c_tok, elap = parallel_chunk_synthesis(batch_id, tasks, endpoint, original_query)
                event_queue.put(("chunk", b_id, text, p_tok, c_tok, elap, slot_name))
                return
            except Exception as e:
                print(f"    [!] [{slot_name}] Chunk {batch_id} synthesis error (Attempt {attempt}): {e}", flush=True)
            finally:
                orch_queue.put((endpoint, slot_name))
            time.sleep(2)
            
        batch_context = "\n\n".join([f"--- TASK {t['id']}: {t['prompt']} ---\n{t['result']}" for t in tasks])
        print(f"    [!] CRITICAL: Chunk {batch_id} failed synthesis. Falling back to raw concatenation.", flush=True)
        event_queue.put(("chunk", batch_id, f"\n--- [RAW CHUNK {batch_id}] ---\n" + batch_context, 0, 0, 0, "Fallback"))

    # --- Level 2 Background Stitching Thread ---
    master_document = ""
    stitch_p_tok, stitch_c_tok = 0, 0
    
    def master_stitch_consumer():
        nonlocal master_document, stitch_p_tok, stitch_c_tok
        while True:
            item = stitch_queue.get()
        
            if item is None: 
                stitch_queue.task_done()
                break
                
            c_id, c_text = item
            
            if master_document == "":
                print(f"    [STITCH] Pipeline trigger: Chunk {c_id}/{total_chunks} is foundation. Establishing Master Document...", flush=True)
                master_document = c_text
            else:
                success = False
                for attempt in range(1, MAX_RETRIES + 1):
                    orch_endpoint, slot_name = orch_queue.get()
                    try:
                        print(f"    [STITCH] [{slot_name}] Pipeline trigger: Weaving Chunk {c_id}/{total_chunks} into Master Document...", flush=True)
                        new_doc, p, c, elap = rolling_master_stitch(c_id, master_document, c_text, orch_endpoint, original_query)
                        master_document = new_doc
                        stitch_p_tok += p
                        stitch_c_tok += c
                        print(f"    [+] [{slot_name}] Master Stitch {c_id} completed in {elap}s.", flush=True)
                        success = True
                        break
                    except Exception as e:
                        print(f"    [!] [{slot_name}] Master Stitch {c_id} error (Attempt {attempt}): {e}", flush=True)
                    finally:
                        orch_queue.put((orch_endpoint, slot_name))
                    time.sleep(2)
                    
                if not success:
                    print(f"    [!] CRITICAL: Master stitch failed on Chunk {c_id}. Falling back to tagged append.", flush=True)
                    master_document += f"\n\n\n" + c_text
                    
            stitch_queue.task_done()

    # Start the stitcher thread
    stitch_thread = threading.Thread(target=master_stitch_consumer, daemon=True)
    stitch_thread.start()

    # --- Main Event Loop Variables ---
    worker_p_tok, worker_c_tok = 0, 0
    chunk_p_tok, chunk_c_tok = 0, 0
    
    results_dict = {}
    chunks_dict = {}
    worker_stats_log = []
    
    next_stitch_id = 1
    chunks_completed = 0
    dispatch_start_time = time.time()
    
    max_useful_threads = (len(WORKER_ENDPOINTS) * WORKER_PARALLEL_SLOTS) + (len(ORCHESTRATOR_ENDPOINTS) * ORCH_PARALLEL_SLOTS)
    thread_pool_size = min(total_tasks, max_useful_threads)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_pool_size) as worker_exec, \
         concurrent.futures.ThreadPoolExecutor(max_workers=thread_pool_size) as orch_exec:
        
        for i, task in enumerate(sub_tasks):
            worker_exec.submit(worker_wrapper, i + 1, task)
            
        while chunks_completed < total_chunks:
            event = event_queue.get()
            
            if event[0] == "worker":
                task_res = event[1]
                tid = task_res["id"]
                results_dict[tid] = task_res
                worker_stats_log.append(task_res)
                
                worker_p_tok += task_res["prompt_tokens"]
                worker_c_tok += task_res["completion_tokens"]
                current_elap = time.time() - dispatch_start_time
                agg_tps = round(worker_c_tok / current_elap, 2) if current_elap > 0 else 0
                
                print(f"    <- [{task_res.get('slot', 'Unknown')}] Worker{tid:02d} Finished in {task_res['elapsed']}s | Status: {task_res['status']} | Agg TPS: {agg_tps}", flush=True)

                chunk_idx = (tid - 1) // SYNTHESIS_CHUNK_SIZE + 1
                expected_start = (chunk_idx - 1) * SYNTHESIS_CHUNK_SIZE + 1
                expected_end = min(expected_start + SYNTHESIS_CHUNK_SIZE, total_tasks + 1)
                
                chunk_ready = True
                for i in range(expected_start, expected_end):
                    if i not in results_dict:
                        chunk_ready = False
                        break
                        
                if chunk_ready:
                    chunk_tasks = [results_dict.pop(i) for i in range(expected_start, expected_end)]
                    print(f"    [BATCH] Multithread trigger: Grouping tasks {chunk_tasks[0]['id']}-{chunk_tasks[-1]['id']} into Chunk {chunk_idx}...", flush=True)
                    orch_exec.submit(chunk_wrapper, chunk_idx, chunk_tasks)

            elif event[0] == "chunk":
                _, b_id, text, p_tok, c_tok, elap, slot_name = event
                chunks_dict[b_id] = text
                chunk_p_tok += p_tok
                chunk_c_tok += c_tok
                chunks_completed += 1
                
                print(f"    <- [{slot_name}] Chunk{b_id:02d} Compressed parallel batch in {elap}s.", flush=True)

                while next_stitch_id in chunks_dict:
                    stitch_text = chunks_dict.pop(next_stitch_id)
                    stitch_queue.put((next_stitch_id, stitch_text))
                    next_stitch_id += 1
                    
    stitch_queue.put(None)
    stitch_queue.join()
    stitch_thread.join()
    
    total_orch_p = chunk_p_tok + stitch_p_tok
    total_orch_c = chunk_c_tok + stitch_c_tok

    return master_document, worker_p_tok, worker_c_tok, total_orch_p, total_orch_c, worker_stats_log

# ==============================================================================
# Main Execution Engine
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Clustered LLM Orchestrator")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-f", "--file", type=str, help="Path to prompt file.")
    group.add_argument("-p", "--prompt", type=str, help="Direct prompt input.")
    args = parser.parse_args()
    
    if args.file:
        target_path = Path(args.file).resolve()
        if not target_path.exists():
            print(f"[!] Fatal: Input file '{target_path}' not found.", flush=True)
            sys.exit(1)
            
        with open(target_path, "r", encoding="utf-8") as f: 
            target_query = f.read()
            
        work_dir = target_path.parent
    elif args.prompt:
        target_query = args.prompt
        work_dir = Path.cwd()
    else:
        target_query = "Create a simple Python HTTP server using Flask that returns 'Hello, Orchestrator!' on the root endpoint. Also, write a standard Dockerfile to containerize it."
        work_dir = Path.cwd()
        
    print("\n=== STARTING UNIFIED DISTRIBUTED ORCHESTRATOR CLUSTER ===", flush=True)
    print(f"[*] Bound Working Directory: {work_dir}", flush=True)
    
    master_start_time = time.time()
    global_input_tokens = 0
    global_output_tokens = 0
    
    fragments, p_tok, c_tok = decompose_to_atomic_pieces(target_query)
    global_input_tokens += p_tok
    global_output_tokens += c_tok
    
    run_directory = export_to_split_files(fragments, work_dir)
    
    final_output, w_p, w_c, o_p, o_c, worker_stats = execute_continuous_map_reduce(fragments, target_query, run_directory)
    global_input_tokens += (w_p + o_p)
    global_output_tokens += (w_c + o_c)
    
    master_elapsed_time = time.time() - master_start_time

    stats_md = "\n\n---\n## Worker Execution Statistics\n"
    stats_md += "| Worker ID | Slot | Status | Elapsed (s) | Task TPS | Prompt Tokens | Comp Tokens | Total Tokens |\n"
    stats_md += "|-----------|------|--------|-------------|----------|---------------|-------------|--------------|\n"
    for stat in sorted(worker_stats, key=lambda x: x['id']):
        stats_md += f"| Thread{stat['id']:02d} | {stat.get('slot', 'N/A')} | {stat['status']} | {stat.get('elapsed', 0)} | {stat.get('tps', 0)} | {stat.get('prompt_tokens', 0)} | {stat.get('completion_tokens', 0)} | {stat.get('total_tokens', 0)} |\n"
    
    agg_md = "\n\n## Cluster Aggregate Statistics\n"
    agg_md += f"- **Total Wall-Clock Time:** {master_elapsed_time:.2f} seconds\n"
    agg_md += f"- **Total Input Tokens:** {global_input_tokens:,}\n"
    agg_md += f"- **Total Output Tokens:** {global_output_tokens:,}\n"
    agg_md += f"- **Total Cluster Tokens:** {global_input_tokens + global_output_tokens:,}\n"
    
    final_output += stats_md + agg_md
    
    print("\n[5] MASTER EXPORT: Saving master synthesis to disk...", flush=True)
    final_file_path = run_directory / "FINAL_SYNTHESIS.md"
    
    with open(final_file_path, "w", encoding="utf-8") as f:
        f.write(final_output)
    
    print("\n==============================================================================", flush=True)
    print("EXECUTION RUN COMPLETE", flush=True)
    print(f"    Total Wall-Clock Time:  {master_elapsed_time:.2f} seconds", flush=True)
    print(f"    Total Input Tokens:     {global_input_tokens:,}", flush=True)
    print(f"    Total Output Tokens:    {global_output_tokens:,}", flush=True)
    print(f"    Total Cluster Tokens:   {global_input_tokens + global_output_tokens:,}", flush=True)
    print(f"    [+] Synthesis Payload Size: {len(final_output):,} characters", flush=True)
    print(f"    Run Master Directory:   {run_directory.absolute()}", flush=True)
    print("==============================================================================\n", flush=True)

    print("[6] AUTOMATED HANDOFF: Post-Processing Synthesis...", flush=True)
    
    current_script_dir = Path(__file__).resolve().parent
    post_process_script = (current_script_dir / "3-post_process_synthesis.py").resolve()
    
    if post_process_script.exists():
        try:
            subprocess.run(
                ["python3", str(post_process_script), str(final_file_path), str(run_directory)], 
                cwd=work_dir, 
                check=True
            )
            print("    [+] Post-processing pipeline completed successfully.", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"    [!] Post-processing script failed with exit code: {e.returncode}", flush=True)
        except Exception as e:
            print(f"    [!] Execution error during handoff: {e}", flush=True)
    else:
        print(f"    [!] Could not locate {post_process_script.name}. Skipping automated handoff.", flush=True)
