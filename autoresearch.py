#!/usr/bin/env python3
# -*- coding: ascii -*-

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
import csv
import tempfile
import requests
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Tuple, List, Dict
from openai import OpenAI

# ==============================================================================
# Global Configuration & Endpoints
# ==============================================================================

# Phase 1: Raw Generation Config
GEN_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:8033/v1")
GEN_API_KEY = os.getenv("OPENAI_API_KEY", "sk-local")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.6-27B-IQ4_XS.gguf")

# Unified Context Limits
MAX_CONTEXT_CHARS = 60000

# Phase 2: Distillation Config
DISTILLER_URL = os.getenv("DISTILLER_URL", "http://localhost:8080/v1")
DISTILLER_MODEL = os.getenv("DISTILLER_MODEL", "nvidia_Orchestrator-8B-Q6_K.gguf")
DISTILLER_API_KEY = os.getenv("DISTILLER_API_KEY", "local-sk")

# Phase 3 & 4: Orchestrator Map-Reduce & Post-Processing Config
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

MAX_CHUNK_CHARS = 100000 
CONCURRENT_SLOTS_PER_ENDPOINT = 1 
MAX_WORKER_TOKENS = 12288

# Phase 5: Automatic Unittests Config
TEST_WORKER_ENDPOINTS = [
    "http://localhost:8034/v1/chat/completions"
]
CONCURRENT_REQS_PER_ENDPOINT = 2
MAX_OUTPUT_TOKENS = 16384
LLM_TEMPERATURE = 0.1          
LLM_TOP_P = 0.95               
LLM_FREQUENCY_PENALTY = 0.5    
LLM_PRESENCE_PENALTY = 0.2     
RETRY_BASE_DELAY = 2.0   
RETRY_JITTER = 0.5       
MAX_EXEC_WORKERS = 4
EXECUTION_RESULT_FIELDS = ["filename", "language", "status", "message"]

# Phase 6: Todo Project Distill Config
ORCHESTRATOR_ENDPOINT_POST = os.getenv("ORCHESTRATOR_ENDPOINT", "http://localhost:8080/v1/chat/completions")

# Global Clients (for single-endpoint phases)
gen_client = OpenAI(base_url=GEN_API_BASE, api_key=GEN_API_KEY)
distill_client = OpenAI(base_url=DISTILLER_URL, api_key=DISTILLER_API_KEY)

# ==============================================================================
# Phase 1: Raw Content Generation
# ==============================================================================

def setup_raw_directory(base_dir: str, category: str) -> Path:
    target_dir = Path(base_dir) / category
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir

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
    
    system_prompt = """You are an expert researcher and technical writer.
Your task is to write a comprehensive, detailed, and highly informative document based on the user's prompt. 
Write clearly, use markdown formatting (headings, bullet points, bold text), and provide deep insights.
Do not include any conversational filler. Just output the raw document content."""

    full_content = ""
    start_time = time.time()
    
    try:
        response = gen_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4096,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                full_content += chunk.choices[0].delta.content
                
        elapsed = round(time.time() - start_time, 2)
        print(f"[+] Generation complete in {elapsed} seconds.")
        
        filename = generate_safe_filename(prompt)
        filepath = target_dir / filename
        
        with open(filepath, "w", encoding="ascii", errors="ignore") as f:
            f.write(full_content.strip())
            
        print(f"[+] Saved raw content to: {filepath.absolute()}")
        return filepath

    except Exception as e:
        print(f"\n[!] Fatal Error during generation: {e}")
        sys.exit(1)

# ==============================================================================
# Phase 2: Fluff-to-Action Technical Distillation
# ==============================================================================

def read_file_content(file_path: Path) -> str:
    encodings = ['utf-8', 'latin-1', 'ascii']
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except Exception:
            continue
    print(f"[!] Fatal: Could not decode '{file_path}'.", flush=True)
    sys.exit(1)

def distill_document(raw_text: str) -> str:
    char_count = len(raw_text)
    print(f"\n[PHASE 2] [*] Ingesting document ({char_count:,} characters)...", flush=True)
    
    if char_count > MAX_CONTEXT_CHARS:
        print(f"    [!] WARNING: Document size exceeds {MAX_CONTEXT_CHARS:,} characters.", flush=True)

    system_prompt = (
        "You are a ruthless, highly technical Lead Engineer and Project Manager. "
        "Your job is to read dense, fluffy, or theoretical technical documents and extract ONLY "
        "a succinct, actionable list of explicit TO-DOs, architectural requirements, and implementation tasks. "
        "STRIP AWAY all marketing fluff, academic rambling, metaphors, and context setting. "
        "Output a clean, highly structured Markdown list of tasks that a developer can immediately start building. "
        "Do not include conversational filler."
    )

    try:
        response = distill_client.chat.completions.create(
            model=DISTILLER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract the actionable tasks from this document:\n\n{raw_text}"}
            ],
            temperature=0.3,
            max_tokens=8192
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[!] Error during distillation: {e}", flush=True)
        sys.exit(1)

def save_distilled_output(output_text: str, original_path: Path) -> Path:
    output_filename = f"{original_path.stem}_distilled.md"
    output_path = original_path.parent / output_filename
    
    try:
        with open(output_path, "w", encoding="ascii", errors="ignore") as f:
            f.write(output_text)
        return output_path
    except Exception as e:
        print(f"[!] Error saving output file: {e}", flush=True)
        sys.exit(1)

# ==============================================================================
# Phase 3: Unified Distributed Orchestrator Cluster (Map-Reduce)
# ==============================================================================

def estimate_tokens(text: str) -> int:
    return len(str(text)) // 4

def extract_json_array(raw_text: str) -> str:
    cleaned_text = re.sub(r'```json\s*', '', raw_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\n?```\s*', '', cleaned_text).strip()
    
    try:
        json.loads(cleaned_text)
        return cleaned_text
    except json.JSONDecodeError:
        pass
    
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
    print(f"\n[PHASE 3] [1] INGRESS: Analyzing massive query...\n    Length: {len(large_query)} characters", flush=True)

    system_prompt = """You are an algorithmic micro-task decomposer.
Your sole purpose is to take a large, complex query or task and shatter it into atomic, independent pieces for parallel processing.
Output ONLY a valid, flat JSON array of strings. No markdown formatting, no conversational text."""

    client = OpenAI(base_url=ORCHESTRATOR_ENDPOINTS[0], api_key=ORCH_API_KEY, timeout=1800.0, max_retries=0)

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
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    raw_output += chunk.choices[0].delta.content
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                    comp_tokens = chunk.usage.completion_tokens
                    
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
            print(f"    [!] Decomposition Error: {e}", flush=True)
            time.sleep(2)

    return [large_query], estimate_tokens(large_query), 10

def export_to_split_files(pieces: list, work_dir: Path) -> Path:
    if len(pieces) <= 1: return work_dir
    tasks_dir = work_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    
    for idx, piece in enumerate(pieces, start=1):
        filepath = tasks_dir / f"task{idx:03d}.md"
        with open(filepath, "w", encoding="ascii", errors="ignore") as f:
            f.write(f"{piece.strip()}\n")
    return work_dir

def process_subtask(task_id: int, task_prompt: str, endpoint: str, slot_name: str, original_query: str, run_dir: Path) -> dict:
    worker_client = OpenAI(base_url=endpoint, api_key=WORKER_API_KEY, timeout=1800.0, max_retries=0)
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
            max_tokens=MAX_WORKER_TOKENS,
            frequency_penalty=1.1,
            presence_penalty=0.5
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
            with open(artifact_dir / safe_filename, "w", encoding="ascii", errors="ignore") as af: 
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
    client = OpenAI(base_url=endpoint, api_key=ORCH_API_KEY, timeout=1800.0, max_retries=0)
    system_prompt = (
        "You are a Level-1 Synthesis Node in a distributed cluster. "
        "Merge the following sequential worker reports into a coherent, deduplicated section. "
        "Retain all code blocks, configurations, and critical technical data. "
        "Output strictly in standard ASCII."
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
    client = OpenAI(base_url=endpoint, api_key=ORCH_API_KEY, timeout=1800.0, max_retries=0)
    system_prompt = (
        "You are the Final Assembly Layer. Seamlessly weave the new sequential section into the existing master document. "
        "Expand the document logically. Do not drop existing data or code. Output strictly in standard ASCII."
    )
    
    tail_limit = 40000
    if len(current_master) > tail_limit:
        stitch_context = "...[EARLIER CONTENT TRUNCATED FOR CONTEXT LIMITS]...\n\n" + current_master[-tail_limit:]
    else:
        stitch_context = current_master
        
    user_prompt = f"ORIGINAL QUERY: {original_query}\n\n--- CURRENT MASTER DOCUMENT ---\n{stitch_context}\n\n--- NEW SECTION {chunk_id} TO INTEGRATE ---\n{new_chunk}"
    
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

def execute_continuous_map_reduce(sub_tasks: list, original_query: str, run_dir: Path) -> tuple:
    if not sub_tasks:
        return "", 0, 0, 0, 0, []
        
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

    def worker_wrapper(tid: int, prompt: str):
        last_result = {
            "id": tid, "status": "error", "result": "Worker execution failed completely.", 
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, 
            "elapsed": 0, "tps": 0, "slot": "Unknown"
        }
        for _ in range(WORKER_RETRIES):
            endpoint, slot_name = worker_queue.get()
            try:
                res = process_subtask(tid, prompt, endpoint, slot_name, original_query, run_dir)
                if res["status"] == "success": 
                    if res.get("completion_tokens", 0) > MAX_WORKER_TOKENS:
                        safe_char_limit = MAX_WORKER_TOKENS * 4
                        res["result"] = res["result"][:safe_char_limit] + "\n\n...[OUTPUT TRUNCATED DUE TO LENGTH LIMIT]..."
                        res["completion_tokens"] = MAX_WORKER_TOKENS
                        res["total_tokens"] = res["prompt_tokens"] + MAX_WORKER_TOKENS
                    event_queue.put(("worker", res))
                    return  
                last_result = res
            except Exception as e:
                last_result["result"] = f"Failed: {str(e)}"
                last_result["slot"] = slot_name
            finally:
                worker_queue.put((endpoint, slot_name))
            time.sleep(2) 
        event_queue.put(("worker", last_result))

    def chunk_wrapper(batch_id: int, tasks: list):
        for attempt in range(1, MAX_RETRIES + 1):
            endpoint, slot_name = orch_queue.get()
            try:
                b_id, text, p_tok, c_tok, elap = parallel_chunk_synthesis(batch_id, tasks, endpoint, original_query)
                event_queue.put(("chunk", b_id, text, p_tok, c_tok, elap, slot_name))
                return
            except Exception as e:
                pass
            finally:
                orch_queue.put((endpoint, slot_name))
            time.sleep(2)
        batch_context = "\n\n".join([f"--- TASK {t['id']}: {t['prompt']} ---\n{t['result']}" for t in tasks])
        event_queue.put(("chunk", batch_id, f"\n--- [RAW CHUNK {batch_id}] ---\n" + batch_context, 0, 0, 0, "Fallback"))

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
                master_document = c_text
            else:
                success = False
                for attempt in range(1, MAX_RETRIES + 1):
                    orch_endpoint, slot_name = orch_queue.get()
                    try:
                        new_doc, p, c, elap = rolling_master_stitch(c_id, master_document, c_text, orch_endpoint, original_query)
                        master_document = new_doc
                        stitch_p_tok += p
                        stitch_c_tok += c
                        success = True
                        break
                    except Exception as e:
                        pass
                    finally:
                        orch_queue.put((orch_endpoint, slot_name))
                    time.sleep(2)
                if not success:
                    master_document += f"\n\n\n" + c_text
            stitch_queue.task_done()

    stitch_thread = threading.Thread(target=master_stitch_consumer, daemon=True)
    stitch_thread.start()

    worker_p_tok, worker_c_tok = 0, 0
    chunk_p_tok, chunk_c_tok = 0, 0
    results_dict, chunks_dict = {}, {}
    worker_stats_log = []
    
    next_stitch_id = 1
    chunks_completed = 0
    workers_finished = 0
    submitted_chunks = set()
    
    max_useful_threads = (len(WORKER_ENDPOINTS) * WORKER_PARALLEL_SLOTS) + (len(ORCHESTRATOR_ENDPOINTS) * ORCH_PARALLEL_SLOTS)
    thread_pool_size = min(total_tasks, max_useful_threads)
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_pool_size) as worker_exec, \
             concurrent.futures.ThreadPoolExecutor(max_workers=thread_pool_size) as orch_exec:
            
            for i, task in enumerate(sub_tasks):
                worker_exec.submit(worker_wrapper, i + 1, task)
                
            while chunks_completed < total_chunks:
                event = event_queue.get()
                if event[0] == "worker":
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
                            chunk_tasks = [results_dict.pop(i) for i in range(expected_start, expected_end)]
                            orch_exec.submit(chunk_wrapper, chunk_idx, chunk_tasks)

                    # Unblock any partial/straggling chunks if all workers have reported in
                    if workers_finished == total_tasks:
                        for chunk_idx in range(1, total_chunks + 1):
                            if chunk_idx not in submitted_chunks:
                                expected_start = (chunk_idx - 1) * SYNTHESIS_CHUNK_SIZE + 1
                                expected_end = min(expected_start + SYNTHESIS_CHUNK_SIZE, total_tasks + 1)
                                available = [results_dict.pop(i) for i in range(expected_start, expected_end) if i in results_dict]
                                if available:
                                    submitted_chunks.add(chunk_idx)
                                    orch_exec.submit(chunk_wrapper, chunk_idx, available)

                elif event[0] == "chunk":
                    _, b_id, text, p_tok, c_tok, elap, slot_name = event
                    chunks_dict[b_id] = text
                    chunk_p_tok += p_tok
                    chunk_c_tok += c_tok
                    chunks_completed += 1
                    
                    while next_stitch_id in chunks_dict:
                        stitch_text = chunks_dict.pop(next_stitch_id)
                        stitch_queue.put((next_stitch_id, stitch_text))
                        next_stitch_id += 1
                        
    finally:
        stitch_queue.put(None)
        stitch_queue.join()
        stitch_thread.join()
    
    total_orch_p = chunk_p_tok + stitch_p_tok
    total_orch_c = chunk_c_tok + stitch_c_tok

    return master_document, worker_p_tok, worker_c_tok, total_orch_p, total_orch_c, worker_stats_log

# ==============================================================================
# Phase 4: Distributed Parallel Post-Processing
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

    code_pattern = re.compile(r'```[\s\S]*?```')
    text_without_code = code_pattern.sub(replacer, markdown_text)

    table_pattern = re.compile(r'##.*Worker Execution Statistics[\s\S]*')
    table_match = table_pattern.search(text_without_code)
    
    if table_match:
        placeholder = "[[PROTECTED_TELEMETRY_TABLE]]"
        protected_blocks[placeholder] = table_match.group(0)
        text_without_code = text_without_code[:table_match.start()] + f"\n\n{placeholder}\n"

    return text_without_code, protected_blocks

def split_into_logical_chunks(text: str, max_chars: int) -> List[str]:
    chunks = []
    current_chunk = ""
    sections = re.split(r'(?=\n## )', text)

    for section in sections:
        if len(current_chunk) + len(section) < max_chars:
            current_chunk += section
        else:
            if len(section) > max_chars:
                paragraphs = section.split('\n\n')
                for p in paragraphs:
                    if len(current_chunk) + len(p) < max_chars:
                        current_chunk += p + "\n\n"
                    else:
                        if current_chunk.strip(): 
                            chunks.append(current_chunk.strip())
                        current_chunk = p + "\n\n"
            else:
                if current_chunk.strip(): 
                    chunks.append(current_chunk.strip())
                current_chunk = section

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def semantic_deduplication(chunk_text: str, chunk_id: int, total_chunks: int, endpoint: str, slot_name: str) -> str:
    placeholder_pattern = r'\[\[PROTECTED_[A-Z_]+?\d{3}\]\]|\[\[PROTECTED_TELEMETRY_TABLE\]\]'
    chunk_inventory = re.findall(placeholder_pattern, chunk_text)
    
    client = OpenAI(base_url=endpoint, api_key=ORCH_API_KEY, timeout=1200.0, max_retries=2)

    system_prompt = (
        "You are a strict, highly logical Technical Editor processing a section of a larger technical document. "
        "Your job is to take this messy, repetitive draft and rewrite it into cohesive, succinct markdown. "
        "\n\nRULES:"
        "\n1. Remove all repetitive statements, redundant introductions, and duplicated concepts."
        "\n2. Organize the content into logical, non-repeating markdown headers."
        "\n3. Use bullet points for readability where applicable."
        "\n4. Do not add conversational filler. Be direct and professional."
    )

    if chunk_inventory:
        inventory_str = ", ".join(chunk_inventory)
        system_prompt += (
            f"\n\nCRITICAL ARTIFACT INVENTORY:\n"
            f"This specific text section contains the following protected placeholders: {inventory_str}\n"
            f"You MUST include EVERY SINGLE ONE of these exact placeholder strings in your rewritten output. "
            f"Even if you summarize the surrounding text, do NOT drop these placeholders. They represent vital code blocks."
        )

    try:
        response = client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": chunk_text}],
            temperature=0.1, max_tokens=16384, presence_penalty=0.2
        )
        distilled_text = response.choices[0].message.content.strip()
        
        if chunk_inventory:
            missing_placeholders = [p for p in chunk_inventory if p not in distilled_text]
            if missing_placeholders:
                distilled_text += "\n\n### Recovered Chunk Artifacts\n" + "\n\n".join(missing_placeholders)

        return distilled_text
    except Exception as e:
        return chunk_text

def parallel_edit_chunks(chunks: List[str]) -> str:
    total_chunks = len(chunks)
    endpoint_queue = queue.Queue()
    for ep in ORCHESTRATOR_ENDPOINTS:
        ip_tail = ep.split('//')[1].split(':')[0].split('.')[-1]
        for i in range(CONCURRENT_SLOTS_PER_ENDPOINT):
            slot_name = f"Node-{ip_tail}-Slot-{i+1}"
            endpoint_queue.put((ep, slot_name))

    results = [""] * total_chunks
    def worker_wrapper(chunk_idx: int, chunk_content: str):
        endpoint, slot_name = endpoint_queue.get()
        try:
            return semantic_deduplication(chunk_content, chunk_idx + 1, total_chunks, endpoint, slot_name)
        except Exception:
            return chunk_content
        finally:
            endpoint_queue.put((endpoint, slot_name))

    pool_size = max(1, endpoint_queue.qsize())
    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
        future_to_idx = {executor.submit(worker_wrapper, i, chunk): i for i, chunk in enumerate(chunks)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
                
    return "\n\n".join(results)

def section_boundary_smoothing(full_skeleton: str, global_inventory: List[str], endpoint: str) -> str:
    client = OpenAI(base_url=endpoint, api_key=ORCH_API_KEY, timeout=1800.0, max_retries=1)
    system_prompt = (
        "You are the Executive Technical Editor (Pass 1). "
        "Smooth out any jarring transitions between sections in this stitched document. "
        "Do NOT delete any major technical concepts, instructions, or features."
    )
    if global_inventory:
        system_prompt += f"\n\nCRITICAL ARTIFACT INVENTORY: {', '.join(global_inventory)}\nYou MUST retain EVERY SINGLE placeholder."
    
    if len(full_skeleton) > MAX_CONTEXT_CHARS:
        full_skeleton = full_skeleton[:MAX_CONTEXT_CHARS] + "\n\n...[CONTENT TRUNCATED FOR CONTEXT LIMITS]..."
        
    try:
        response = client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": full_skeleton}],
            temperature=0.2, max_tokens=32768, presence_penalty=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return full_skeleton

def header_unification_pass(smoothed_skeleton: str, global_inventory: List[str], endpoint: str) -> str:
    client = OpenAI(base_url=endpoint, api_key=ORCH_API_KEY, timeout=1800.0, max_retries=1)
    system_prompt = (
        "You are the Executive Technical Editor (Pass 2). "
        "Unify the tone and organize the Markdown headers logically from start to finish. "
        "Do NOT delete any major technical concepts, instructions, or features."
    )
    if global_inventory:
        system_prompt += f"\n\nCRITICAL ARTIFACT INVENTORY: {', '.join(global_inventory)}\nYou MUST retain EVERY SINGLE placeholder."
    
    if len(smoothed_skeleton) > MAX_CONTEXT_CHARS:
        smoothed_skeleton = smoothed_skeleton[:MAX_CONTEXT_CHARS] + "\n\n...[CONTENT TRUNCATED FOR CONTEXT LIMITS]..."
        
    try:
        response = client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": smoothed_skeleton}],
            temperature=0.2, max_tokens=32768, presence_penalty=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return smoothed_skeleton

def global_consolidation_pass(full_skeleton: str, global_inventory: List[str], endpoint: str) -> str:
    smoothed_text = section_boundary_smoothing(full_skeleton, global_inventory, endpoint)
    final_text = header_unification_pass(smoothed_text, global_inventory, endpoint)
    
    missing_placeholders = [p for p in global_inventory if p not in final_text]
    if missing_placeholders:
        final_text += "\n\n### Recovered Global Artifacts\n" + "\n\n".join(missing_placeholders)

    return final_text

def reassemble_document(distilled_text: str, protected_blocks: Dict[str, str]) -> str:
    final_text = distilled_text
    for placeholder, original_content in protected_blocks.items():
        if placeholder in final_text:
            final_text = final_text.replace(placeholder, original_content)
        else:
            final_text += f"\n\n### Orphaned Artifact\n{original_content}"
    return final_text

# ==============================================================================
# Phase 5: Automatic Unittests
# ==============================================================================

def _safe_output_path(detected_filename: str, output_dir: Path) -> Path:
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
    return re.sub(r'^```[^\n]*\n?|^```\s*$', '', text.strip(), flags=re.MULTILINE).strip()

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
        with open(filepath, 'r', encoding='ascii', errors="ignore") as f:
            lines = f.readlines()
        cleaned_lines = []
        valid_pip_operators = ['==', '>=', '<=', '~=', '<', '>', '!=', '@', '-r', '-e', '--', ';']
        for line in lines:
            s_line = line.strip()
            if not s_line or s_line.startswith('#'):
                cleaned_lines.append(line)
                continue
            has_space = ' ' in s_line
            has_operator = any(op in s_line for op in valid_pip_operators)
            if has_space and has_operator:
                tokens = s_line.split()
                first_tok = tokens[0]
                looks_like_pip = (first_tok.startswith('-') or '://' in first_tok or re.match(r'^[A-Za-z0-9_\-\.]+', first_tok))
                extra_tokens = tokens[2:] if len(tokens) > 2 else []
                has_plain_english_suffix = any(not re.search(r'[=<>!~@;:/]', tok) and tok.isalpha() and len(tok) > 2 for tok in extra_tokens)
                if not looks_like_pip or has_plain_english_suffix:
                    continue
            elif has_space and not has_operator:
                continue
            cleaned_lines.append(line)
        with open(filepath, 'w', encoding='ascii', errors="ignore") as f:
            f.writelines(cleaned_lines)
    except Exception as e:
        print(f"Warning: Could not sanitize requirements file {filepath}: {e}")

def extract_code_blocks(md_content: str, output_dir: str | Path) -> list:
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
                    ext_map = {"python": "py", "bash": "sh", "yaml": "yml", "dockerfile": "Dockerfile"}
                    ext = ext_map.get(ext.lower(), ext)
                    base_name = last_header.replace(" ", "_").lower() if last_header else f"artifact_{file_counter}"
                    detected_filename = f"{base_name}.{ext}" if ext.lower() != "dockerfile" else "Dockerfile"
                    file_counter += 1
                detected_filename = re.sub(r'[()\[\]{}]', '', detected_filename)
                detected_filename = detected_filename.replace(" ", "_")
                file_path = _safe_output_path(detected_filename, output_path)
                with open(file_path, "w", encoding="ascii", errors="ignore") as f:
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
        elif in_block:
            current_block.append(line)
        i += 1
    if in_block and current_block:
        content = "\n".join(current_block)
        fallback_name = f"artifact_{file_counter}_partial.txt"
        file_path = output_path / fallback_name
        with open(file_path, "w", encoding="ascii", errors="ignore") as f:
            f.write(content + "\n")
    return extracted_artifacts

def request_unittests_from_worker(artifact: dict, endpoint_queue: queue.Queue, test_output_dir: Path, progress_lock: threading.Lock, progress_counter: list) -> dict | None:
    valid_langs = {"python", "py", "cpp", "c", "bash", "sh"}
    if artifact["language"].lower() not in valid_langs:
        return None
    endpoint_url = endpoint_queue.get()
    
    # Note: If this future is cancelled mid-flight after queue.get() but before 
    # the finally block, the endpoint slot could theoretically leak. It is safe 
    # here because the queue and the thread executor are discarded immediately 
    # after the block completes.
    try:
        code_content = artifact['content']
        if len(code_content) > MAX_CONTEXT_CHARS:
            code_content = code_content[:MAX_CONTEXT_CHARS] + "\n\n...[CONTENT TRUNCATED FOR CONTEXT LIMITS]..."
            
        prompt = (
            f"Write highly compact, succinct, and targeted unit tests for the following {artifact['language']} code.\n"
            f"Focus ONLY on core functionality and critical paths. Minimize boilerplate aggressively.\n\n"
            f"CRITICAL RULES:\n"
            f"1. DO NOT hallucinate imports or use non-existent modules.\n"
            f"2. Keep the code as short as possible while ensuring it runs and passes.\n"
            f"3. Group assertions and use parametrization where possible to save space.\n"
            f"4. Output ONLY valid test code inside a single markdown code block. No explanations.\n\n"
            f"File: {artifact['filename']}\n"
            f"```{artifact['language']}\n{code_content}\n```"
        )
        payload = {
            "model": WORKER_MODEL,
            "messages": [
                {"role": "system", "content": "You are a highly efficient code testing assistant. Write succinct, compact, and boilerplate-free unit tests. Use parametrization to consolidate test cases where applicable. Do not explain your code."},
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
                response = requests.post(endpoint_url, json=payload, timeout=600)
                response.raise_for_status()
                result = response.json()
                choices = result.get("choices")
                if not choices:
                    break
                test_code = choices[0].get("message", {}).get("content", "")
                if not test_code:
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)))
                    continue
                test_code = _strip_markdown_fences(test_code)
                test_output_dir.mkdir(parents=True, exist_ok=True)
                test_filename = f"test_{artifact['filename']}"
                test_filepath = test_output_dir / test_filename
                with open(test_filepath, "w", encoding="ascii", errors="ignore") as f:
                    f.write(test_code + "\n")
                with progress_lock:
                    progress_counter[0] += 1
                generation_metadata = {
                    "filename": test_filename,
                    "test_filepath": str(test_filepath),
                    "language": artifact["language"],
                    "artifact_filepath": artifact["filepath"],
                }
                break
            except requests.exceptions.RequestException as exc:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)))
        return generation_metadata
    finally:
        endpoint_queue.put(endpoint_url)

def execute_test_artifact(test_meta: dict) -> dict:
    lang = test_meta["language"].lower()
    test_path = Path(test_meta["test_filepath"]).resolve()
    artifact_path = Path(test_meta["artifact_filepath"]).resolve()
    result = {"filename": test_meta["filename"], "language": lang, "status": "UNKNOWN", "message": ""}
    try:
        if lang in ("python", "py"):
            env = os.environ.copy()
            src_dir = str(test_path.parent)
            existing_pypath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{src_dir}:{existing_pypath}" if existing_pypath else src_dir
            cmd = ["python", "-m", "pytest", "--tb=short", "-q", str(test_path)]
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
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
                binary_path = Path(tmp.name)
            try:
                compile_cmd = [compiler, str(artifact_path), str(test_path), "-o", str(binary_path)]
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
                if binary_path.exists():
                    binary_path.unlink()
        else:
            result["status"], result["message"] = "SKIPPED", f"No environment definition for: {lang}"
    except subprocess.TimeoutExpired:
        result["status"], result["message"] = "TIMEOUT", "Execution threshold exceeded"
    except Exception as exc:
        result["status"], result["message"] = "ERROR", str(exc)
    return result

def run_phase5_automatic_unittests(source_path: Path):
    print(f"\n[PHASE 5] STARTING AUTOMATED UNITTEST PIPELINE", flush=True)
    if not source_path.exists():
        print(f"Error: Could not find {source_path}.", flush=True)
        return
    OUTPUT_WORKSPACE = source_path.parent
    TEST_OUTPUT_DIR = OUTPUT_WORKSPACE / "tests"
    REPORT_OUTPUT_DIR = OUTPUT_WORKSPACE / "reports"
    endpoint_concurrency = CONCURRENT_REQS_PER_ENDPOINT
    total_gen_workers = len(TEST_WORKER_ENDPOINTS) * endpoint_concurrency
    endpoint_queue: queue.Queue = queue.Queue()
    for ep in TEST_WORKER_ENDPOINTS:
        for _ in range(endpoint_concurrency):
            endpoint_queue.put(ep)
    with open(source_path, "r", encoding="ascii", errors="ignore") as fh:
        md_content = fh.read()
    artifacts = extract_code_blocks(md_content, OUTPUT_WORKSPACE)
    progress_lock = threading.Lock()
    progress_counter = [0, len(artifacts)]
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=total_gen_workers)
    generated_tests: list[dict] = []
    try:
        futures = [executor.submit(request_unittests_from_worker, artifact, endpoint_queue, TEST_OUTPUT_DIR, progress_lock, progress_counter) for artifact in artifacts]
        for future in concurrent.futures.as_completed(futures):
            try:
                test_meta = future.result()
                if test_meta:
                    generated_tests.append(test_meta)
            except Exception:
                pass
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    req_files = [a for a in artifacts if "requirements" in a["filename"].lower() and a["filename"].endswith(".txt")]
    for req in req_files:
        req_path = Path(req["filepath"]).resolve()
        _sanitize_requirements_file(req_path)
        try:
            subprocess.run(["python", "-m", "pip", "install", "--break-system-packages", "-r", str(req_path)], capture_output=True, encoding="ascii", errors="ignore", timeout=120)
        except Exception:
            pass
    execution_results: list[dict] = []
    if generated_tests:
        exec_executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_EXEC_WORKERS)
        try:
            exec_futures = [exec_executor.submit(execute_test_artifact, tm) for tm in generated_tests]
            for future in concurrent.futures.as_completed(exec_futures):
                try:
                    res = future.result()
                    execution_results.append(res)
                except Exception:
                    pass
        finally:
            exec_executor.shutdown(wait=False, cancel_futures=True)
    if execution_results:
        REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        json_report_path = REPORT_OUTPUT_DIR / "execution_report.json"
        csv_report_path  = REPORT_OUTPUT_DIR / "execution_report.csv"
        with open(json_report_path, "w", encoding="ascii", errors="ignore") as f:
            json.dump(execution_results, f, indent=4)
        with open(csv_report_path, "w", newline="", encoding="ascii", errors="ignore") as f:
            writer = csv.DictWriter(f, fieldnames=EXECUTION_RESULT_FIELDS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(execution_results)

# ==============================================================================
# Phase 6: Todo Project Distillation
# ==============================================================================

def extract_project_tasks(project_name: str, raw_content: str) -> str | None:
    system_prompt = (
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
    headers = {"Authorization": f"Bearer {ORCH_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": ORCHESTRATOR_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract actionable tasks, test outcomes, and relevant artifacts from this {project_name} documentation:\n\n{raw_content}"}
        ],
        "temperature": 0.2, 
        "max_tokens": 8192
    }
    try:
        response = requests.post(ORCHESTRATOR_ENDPOINT_POST, headers=headers, json=payload, timeout=600)
        response.raise_for_status()
        raw_output = response.json()["choices"][0]["message"]["content"].strip()
        return raw_output.encode("ascii", "ignore").decode("ascii")
    except Exception as e:
        print(f"[!] Orchestrator inference failed: {e}", flush=True)
        return None

def run_phase6_project_distillation(project_dir: Path):
    print(f"\n[PHASE 6] STARTING PROJECT DISTILLATION", flush=True)
    project_name = project_dir.name
    
    # Exclude previously distilled tasks, project state files, execution reports, 
    # and files located inside generated tests/reports directories.
    raw_files = list(project_dir.rglob("*.txt")) + list(project_dir.rglob("*.md")) + list(project_dir.rglob("*.csv")) + list(project_dir.rglob("*.json"))
    raw_files = [f for f in raw_files if 
        "DISTILLED_TASKS" not in f.name 
        and "project_state" not in f.name
        and "execution_report" not in f.name
        and f.parent.name != "tests"
        and f.parent.name != "reports"]
        
    if not raw_files:
        print(f"[{project_name}] No raw documentation or test logs found. Skipping.", flush=True)
        return
        
    aggregated_content = []
    for file_path in raw_files:
        try:
            with open(file_path, "r", encoding="ascii", errors="ignore") as f:
                aggregated_content.append(f"--- SOURCE: {file_path.name} ---\n{f.read()}")
        except Exception:
            pass
            
    full_text = "\n\n".join(aggregated_content)
    if len(full_text) > MAX_CONTEXT_CHARS:
        full_text = full_text[:MAX_CONTEXT_CHARS] + "\n\n...[CONTENT TRUNCATED FOR CONTEXT LIMITS]..."
        
    distilled_markdown = extract_project_tasks(project_name, full_text)
    if distilled_markdown:
        output_file = project_dir / "DISTILLED_TASKS.md"
        try:
            with open(output_file, "w", encoding="ascii", errors="ignore") as f:
                f.write(f"# Distilled Tasks: {project_name}\n\n{distilled_markdown}\n")
            print(f"[{project_name}] Successfully saved distilled tasks to {output_file.name}", flush=True)
        except Exception as e:
            print(f"[{project_name}] Failed to save output file: {e}", flush=True)

# ==============================================================================
# Pipeline Executor (Main)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="End-to-End Agentic Content Generation Pipeline")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", "--prompt", type=str, help="Direct prompt for the full pipeline.")
    group.add_argument("-f", "--file", type=str, help="Path to a text file containing the prompt.")
    
    parser.add_argument("-d", "--dir", type=str, default="run_data", help="Base directory for outputs.")
    parser.add_argument("-c", "--category", type=str, default="projects", help="Category folder.")
    
    args = parser.parse_args()
    
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"[!] Error: Prompt file '{args.file}' does not exist.")
            sys.exit(1)
        with open(file_path, "r", encoding="ascii", errors="ignore") as f:
            target_prompt = f.read().strip()
    else:
        target_prompt = args.prompt

    work_dir = Path(args.dir).resolve()
    
    # ---------------------------------------------------------
    # PHASE 1: GENERATE
    target_directory = setup_raw_directory(work_dir, args.category)
    raw_filepath = generate_content(target_prompt, target_directory)

    # ---------------------------------------------------------
    # PHASE 2: DISTILL
    raw_content = read_file_content(raw_filepath)
    actionable_tasks = distill_document(raw_content)
    distilled_filepath = save_distilled_output(actionable_tasks, raw_filepath)
    
    # ---------------------------------------------------------
    # PHASE 3: ORCHESTRATE MAP-REDUCE
    with open(distilled_filepath, "r", encoding="ascii", errors="ignore") as f: 
        target_query = f.read()
        
    master_start_time = time.time()
    fragments, p_tok, c_tok = decompose_to_atomic_pieces(target_query)
    
    run_directory = export_to_split_files(fragments, distilled_filepath.parent)
    final_output, w_p, w_c, o_p, o_c, worker_stats = execute_continuous_map_reduce(fragments, target_query, run_directory)
    
    master_elapsed_time = time.time() - master_start_time

    stats_md = "\n\n---\n## Worker Execution Statistics\n"
    stats_md += "| Worker ID | Slot | Status | Elapsed (s) | Task TPS | Prompt Tokens | Comp Tokens | Total Tokens |\n"
    stats_md += "|-----------|------|--------|-------------|----------|---------------|-------------|--------------|\n"
    for stat in sorted(worker_stats, key=lambda x: x['id']):
        stats_md += f"| Thread{stat['id']:02d} | {stat.get('slot', 'N/A')} | {stat['status']} | {stat.get('elapsed', 0)} | {stat.get('tps', 0)} | {stat.get('prompt_tokens', 0)} | {stat.get('completion_tokens', 0)} | {stat.get('total_tokens', 0)} |\n"
    
    agg_md = "\n\n## Cluster Aggregate Statistics\n"
    agg_md += f"- **Total Wall-Clock Time:** {master_elapsed_time:.2f} seconds\n"
    
    final_output += stats_md + agg_md
    final_file_path = run_directory / "FINAL_SYNTHESIS.md"
    
    with open(final_file_path, "w", encoding="ascii", errors="ignore") as f:
        f.write(final_output)

    # ---------------------------------------------------------
    # PHASE 4: POST-PROCESS EDITORIAL
    print("\n[PHASE 4] STARTING DISTRIBUTED SYNTHESIS POLISH", flush=True)
    with open(final_file_path, "r", encoding="ascii", errors="ignore") as f:
        raw_markdown = f.read()

    skeleton_text, protected_assets = extract_and_protect_blocks(raw_markdown)
    chunks = split_into_logical_chunks(skeleton_text, MAX_CHUNK_CHARS)
    
    output_path = final_file_path.parent / "POLISHED_SYNTHESIS.md"
    
    if len(chunks) <= 1:
        with open(output_path, "w", encoding="ascii", errors="ignore") as f:
            f.write(raw_markdown)
        print(f"[+] Document fits in single chunk. Bypassed LLM refinement. Saved to {output_path}")
    else:
        distilled_skeleton = parallel_edit_chunks(chunks)
        global_inventory = list(protected_assets.keys())
        final_skeleton = global_consolidation_pass(distilled_skeleton, global_inventory, ORCHESTRATOR_ENDPOINTS[0])
        final_polished_markdown = reassemble_document(final_skeleton, protected_assets)

        with open(output_path, "w", encoding="ascii", errors="ignore") as f:
            f.write(final_polished_markdown)
        print(f"[+] Cleaned File Saved To: {output_path.absolute()}")

    # ---------------------------------------------------------
    # PHASE 5: AUTOMATED UNITTESTS
    run_phase5_automatic_unittests(output_path)

    # ---------------------------------------------------------
    # PHASE 6: PROJECT DISTILLATION
    run_phase6_project_distillation(output_path.parent)

    print("\n==============================================================================")
    print("PIPELINE COMPLETE")
    print("==============================================================================\n")
            
if __name__ == "__main__":
    main()
