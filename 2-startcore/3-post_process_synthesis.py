#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distributed Parallel Post-Processing Editorial Node (40k Context Optimized).
Features Parallel Chunk Compression followed by a Two-Phase Global Consolidation Pass 
to holistically smooth the entire document, with Strict Placeholder Auditing.
"""

import os
import re
import argparse
import time
import queue
import concurrent.futures
import subprocess
from pathlib import Path
from typing import Tuple, List, Dict
from openai import OpenAI

# ==============================================================================
# Configuration & Endpoints
# ==============================================================================

ORCHESTRATOR_ENDPOINTS = [
    "http://192.168.2.134:8080/v1",
    "http://192.168.2.137:8080/v1"
]

ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "nvidia_Orchestrator-8B-Q6_K.gguf")
API_KEY = os.getenv("ORCH_API_KEY", "local-sk")

MAX_CHUNK_CHARS = 100000 
CONCURRENT_SLOTS_PER_ENDPOINT = 1 

# ==============================================================================
# Core Processing Logic
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
    
    print(f"    -> [{slot_name}] Processing Chunk {chunk_id:02d}/{total_chunks:02d} ({len(chunk_text):,} chars) | Tracking {len(chunk_inventory)} artifacts...", flush=True)
    
    client = OpenAI(
        base_url=endpoint,
        api_key=API_KEY,
        timeout=1200.0,
        max_retries=2
    )

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

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk_text}
            ],
            temperature=0.1,  
            max_tokens=16384,
            presence_penalty=0.2
        )
        distilled_text = response.choices[0].message.content.strip()
        elapsed = round(time.time() - start_time, 2)
        
        if chunk_inventory:
            missing_placeholders = [p for p in chunk_inventory if p not in distilled_text]
            if missing_placeholders:
                print(f"    [!] [{slot_name}] Warning: Editor dropped {len(missing_placeholders)} artifact(s) in Chunk {chunk_id:02d}. Forcing localized recovery...", flush=True)
                distilled_text += "\n\n### Recovered Chunk Artifacts\n" + "\n\n".join(missing_placeholders)

        print(f"    <- [{slot_name}] Chunk {chunk_id:02d}/{total_chunks:02d} polished in {elapsed}s.", flush=True)
        return distilled_text

    except Exception as e:
        print(f"    [!] [{slot_name}] API Error on Chunk {chunk_id:02d}: {e}")
        return chunk_text

def parallel_edit_chunks(chunks: List[str]) -> str:
    total_chunks = len(chunks)
    
    endpoint_queue = queue.Queue()
    for ep in ORCHESTRATOR_ENDPOINTS:
        ip_tail = ep.split('//')[1].split(':')[0].split('.')[-1]
        for i in range(CONCURRENT_SLOTS_PER_ENDPOINT):
            slot_name = f"Node-{ip_tail}-Slot-{i+1}"
            endpoint_queue.put((ep, slot_name))

    total_workers = endpoint_queue.qsize()
    print(f"[3] Dispatching {total_chunks} massive chunk(s) across {len(ORCHESTRATOR_ENDPOINTS)} endpoints ({total_workers} parallel slots)...", flush=True)
    
    results = [""] * total_chunks
    
    def worker_wrapper(chunk_idx: int, chunk_content: str):
        endpoint, slot_name = endpoint_queue.get()
        try:
            return semantic_deduplication(chunk_content, chunk_idx + 1, total_chunks, endpoint, slot_name)
        except Exception as e:
            print(f"    [!] Catastrophic thread failure on chunk {chunk_idx + 1}: {e}")
            return chunk_content
        finally:
            endpoint_queue.put((endpoint, slot_name))

    with concurrent.futures.ThreadPoolExecutor(max_workers=total_workers) as executor:
        future_to_idx = {
            executor.submit(worker_wrapper, i, chunk): i 
            for i, chunk in enumerate(chunks)
        }
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
                
    return "\n\n".join(results)

def section_boundary_smoothing(full_skeleton: str, global_inventory: List[str], endpoint: str) -> str:
    client = OpenAI(base_url=endpoint, api_key=API_KEY, timeout=1800.0, max_retries=1)
    system_prompt = (
        "You are the Executive Technical Editor (Pass 1). "
        "Smooth out any jarring transitions between sections in this stitched document. "
        "Do NOT delete any major technical concepts, instructions, or features."
    )
    if global_inventory:
        inventory_str = ", ".join(global_inventory)
        system_prompt += f"\n\nCRITICAL ARTIFACT INVENTORY: {inventory_str}\nYou MUST retain EVERY SINGLE placeholder."
    
    try:
        response = client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": full_skeleton}],
            temperature=0.2, max_tokens=32768, presence_penalty=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    [!] Section Smoothing Error: {e}")
        return full_skeleton

def header_unification_pass(smoothed_skeleton: str, global_inventory: List[str], endpoint: str) -> str:
    client = OpenAI(base_url=endpoint, api_key=API_KEY, timeout=1800.0, max_retries=1)
    system_prompt = (
        "You are the Executive Technical Editor (Pass 2). "
        "Unify the tone and organize the Markdown headers logically from start to finish. "
        "Do NOT delete any major technical concepts, instructions, or features."
    )
    if global_inventory:
        inventory_str = ", ".join(global_inventory)
        system_prompt += f"\n\nCRITICAL ARTIFACT INVENTORY: {inventory_str}\nYou MUST retain EVERY SINGLE placeholder."
    
    try:
        response = client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": smoothed_skeleton}],
            temperature=0.2, max_tokens=32768, presence_penalty=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    [!] Header Unification Error: {e}")
        return smoothed_skeleton

def global_consolidation_pass(full_skeleton: str, global_inventory: List[str], endpoint: str) -> str:
    print(f"[4] Executing Two-Phase Global Consolidation Pass...", flush=True)
    
    start_time = time.time()
    
    print(f"    -> Pass 1: Section Boundary Smoothing...", flush=True)
    smoothed_text = section_boundary_smoothing(full_skeleton, global_inventory, endpoint)
    
    print(f"    -> Pass 2: Header Unification...", flush=True)
    final_text = header_unification_pass(smoothed_text, global_inventory, endpoint)
    
    elapsed = round(time.time() - start_time, 2)
    
    missing_placeholders = [p for p in global_inventory if p not in final_text]
    if missing_placeholders:
        print(f"    [!] Warning: Executive Editor dropped {len(missing_placeholders)} artifact(s). Forcing recovery...", flush=True)
        final_text += "\n\n### Recovered Global Artifacts\n" + "\n\n".join(missing_placeholders)

    print(f"    <- Two-Phase Global pass completed successfully in {elapsed}s.", flush=True)
    return final_text

def reassemble_document(distilled_text: str, protected_blocks: Dict[str, str]) -> str:
    final_text = distilled_text
    
    for placeholder, original_content in protected_blocks.items():
        if placeholder in final_text:
            final_text = final_text.replace(placeholder, original_content)
        else:
            print(f"[!] Critical Warning: {placeholder} bypassed all recovery. Appending to absolute bottom.")
            final_text += f"\n\n### Orphaned Artifact\n{original_content}"

    return final_text

# ==============================================================================
# Main Execution Engine
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Distributed Parallel Post-Process and Deduplicate Master Synthesis files (40k Context).")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the FINAL_SYNTHESIS.md file.")
    parser.add_argument("-o", "--output", type=str, help="Path to save the cleaned file (Defaults to POLISHED_SYNTHESIS.md).")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[-] Error: File {input_path} not found.")
        return

    output_path = Path(args.output) if args.output else input_path.parent / "POLISHED_SYNTHESIS.md"

    print("\n=== STARTING DISTRIBUTED SYNTHESIS POLISH ===", flush=True)
    
    with open(input_path, "r", encoding="utf-8") as f:
        raw_markdown = f.read()

    print(f"[1] Extracting and protecting code artifacts from {len(raw_markdown):,} characters...", flush=True)
    skeleton_text, protected_assets = extract_and_protect_blocks(raw_markdown)
    print(f"    [+] Protected {len(protected_assets)} discrete artifacts.")
    
    print(f"[2] Analyzing context footprint...", flush=True)
    chunks = split_into_logical_chunks(skeleton_text, MAX_CHUNK_CHARS)
    
    if len(chunks) <= 1:
        print(f"    [!] Document fits within a single chunk ({len(raw_markdown):,} chars). Skipping LLM refinement.")
        print("[3] Saving unaltered document as-is...", flush=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(raw_markdown)

        print("\n==============================================================================")
        print("POST-PROCESSING COMPLETE (BYPASSED)")
        print(f"    Unaltered File Saved To: {output_path.absolute()}")
        print("==============================================================================\n")
        
    else:
        distilled_skeleton = parallel_edit_chunks(chunks)
        
        global_inventory = list(protected_assets.keys())
        final_skeleton = global_consolidation_pass(distilled_skeleton, global_inventory, ORCHESTRATOR_ENDPOINTS[0])
        
        print("[5] Reassembling final polished document...", flush=True)
        final_polished_markdown = reassemble_document(final_skeleton, protected_assets)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_polished_markdown)

        print("\n==============================================================================")
        print("POST-PROCESSING COMPLETE")
        print(f"    Cleaned File Saved To: {output_path.absolute()}")
        print(f"    Size Reduction:        {len(raw_markdown):,} chars -> {len(final_polished_markdown):,} chars")
        print("==============================================================================\n")

    print("[6] AUTOMATED HANDOFF: Triggering Automatic Unittests...", flush=True)
    
    current_script_dir = Path(__file__).resolve().parent
    unittest_script = (current_script_dir.parent / "3-agilengine" / "1-Automatic-Unittests.py").resolve()
    
    if unittest_script.exists():
        try:
            subprocess.run(
                ["python3", str(unittest_script), str(output_path.absolute())], 
                check=True
            )
            print("    [+] Unittest pipeline completed successfully.", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"    [!] Unittest script failed with exit code: {e.returncode}", flush=True)
        except Exception as e:
            print(f"    [!] Execution error during handoff: {e}", flush=True)
    else:
        print(f"    [!] Could not locate {unittest_script.name}. Skipping automated handoff.", flush=True)

if __name__ == "__main__":
    main()
