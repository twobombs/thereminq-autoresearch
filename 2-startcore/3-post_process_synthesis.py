#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distributed Parallel Post-Processing Editorial Node (40k Context Optimized).
Features Parallel Chunk Compression followed by a Global Consolidation Pass 
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

# Pool of available Orchestrator nodes to load-balance across
ORCHESTRATOR_ENDPOINTS = [
    "http://192.168.2.134:8080/v1",
    "http://192.168.2.137:8080/v1"
]

ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "nvidia_Orchestrator-8B-Q6_K.gguf")
API_KEY = os.getenv("ORCH_API_KEY", "local-sk")

# Safe character limit for a 40k token context window (~28,500 tokens)
MAX_CHUNK_CHARS = 100000 

# Strict 1:1 mapping to prevent KV cache queuing and timeouts
CONCURRENT_SLOTS_PER_ENDPOINT = 1 

# ==============================================================================
# Core Processing Logic
# ==============================================================================

def extract_and_protect_blocks(markdown_text: str) -> Tuple[str, Dict[str, str]]:
    protected_blocks = {}
    block_counter = 0

    # 1. Protect Code Blocks using a robust replacement function
    def replacer(match: re.Match) -> str:
        nonlocal block_counter
        placeholder = f"[[PROTECTED_CODE_BLOCK_{block_counter:03d}]]"
        protected_blocks[placeholder] = match.group(0)
        block_counter += 1
        return placeholder

    code_pattern = re.compile(r'```[\s\S]*?```')
    text_without_code = code_pattern.sub(replacer, markdown_text)

    # 2. Protect Worker Telemetry Table (Strict ASCII regex)
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
    placeholder_pattern = r'\[\[PROTECTED_[A-Z_]+_\d{3}\]\]|\[\[PROTECTED_TELEMETRY_TABLE\]\]'
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

def global_consolidation_pass(full_skeleton: str, global_inventory: List[str], endpoint: str) -> str:
    """
    Takes the fully stitched (but compressed) document and runs it through the LLM 
    one final time to smooth out transitions between chunks and unify the document holistically.
    """
    print(f"[4] Executing Global Consolidation Pass (Holistic Smoothing)...", flush=True)
    print(f"    -> Sending {len(full_skeleton):,} characters to Executive Node...", flush=True)
    
    client = OpenAI(
        base_url=endpoint,
        api_key=API_KEY,
        timeout=1800.0,
        max_retries=1
    )

    system_prompt = (
        "You are the Executive Technical Editor. "
        "You are receiving a master document that has just been stitched together from multiple distinct chunks. "
        "Your sole objective is to perform a holistic final pass: smooth out any jarring transitions between sections, "
        "ensure the tone is unified, and organize the headers logically from start to finish."
        "\n\nRULES:"
        "\n1. Do NOT delete any major technical concepts, instructions, or features."
        "\n2. Focus heavily on fixing disjointed transitions and unifying the narrative flow."
    )

    if global_inventory:
        inventory_str = ", ".join(global_inventory)
        system_prompt += (
            f"\n\nCRITICAL ARTIFACT INVENTORY:\n"
            f"This document contains {len(global_inventory)} protected placeholders: {inventory_str}\n"
            f"You MUST retain EVERY SINGLE placeholder in its exact logical location. "
            f"These represent vital code blocks that will be injected later. DO NOT drop them under any circumstances."
        )

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_skeleton}
            ],
            temperature=0.2,
            max_tokens=32768,
            presence_penalty=0.1
        )
        final_text = response.choices[0].message.content.strip()
        elapsed = round(time.time() - start_time, 2)
        
        # Localized Recovery for the Global Pass
        missing_placeholders = [p for p in global_inventory if p not in final_text]
        if missing_placeholders:
            print(f"    [!] Warning: Executive Editor dropped {len(missing_placeholders)} artifact(s). Forcing recovery...", flush=True)
            final_text += "\n\n### Recovered Global Artifacts\n" + "\n\n".join(missing_placeholders)

        print(f"    <- Global pass completed successfully in {elapsed}s.", flush=True)
        return final_text

    except Exception as e:
        print(f"    [!] Global Pass API Error: {e}")
        print(f"    [-] Falling back to the stitched chunk skeleton.")
        return full_skeleton

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
    
    # --------------------------------------------------------------------------
    # Document Size Bypass: Skip refinement if the file is short enough
    # --------------------------------------------------------------------------
    if len(chunks) <= 1:
        print(f"    [!] Document fits within a single chunk ({len(raw_markdown):,} chars). Skipping LLM refinement.")
        print("[3] Saving unaltered document as-is...", flush=True)
        
        with open(output_path, "w", encoding="ascii", errors="ignore") as f:
            f.write(raw_markdown)

        print("\n==============================================================================")
        print("POST-PROCESSING COMPLETE (BYPASSED)")
        print(f"    Unaltered File Saved To: {output_path.absolute()}")
        print("==============================================================================\n")
        
    else:
        # Phase 3: Parallel Chunk Compression
        distilled_skeleton = parallel_edit_chunks(chunks)
        
        # Phase 4: Global Consolidation Pass
        global_inventory = list(protected_assets.keys())
        final_skeleton = global_consolidation_pass(distilled_skeleton, global_inventory, ORCHESTRATOR_ENDPOINTS[0])
        
        # Phase 5: Code Re-injection
        print("[5] Reassembling final polished document...", flush=True)
        final_polished_markdown = reassemble_document(final_skeleton, protected_assets)

        with open(output_path, "w", encoding="ascii", errors="ignore") as f:
            f.write(final_polished_markdown)

        print("\n==============================================================================")
        print("POST-PROCESSING COMPLETE")
        print(f"    Cleaned File Saved To: {output_path.absolute()}")
        print(f"    Size Reduction:        {len(raw_markdown):,} chars -> {len(final_polished_markdown):,} chars")
        print("==============================================================================\n")

    # ==============================================================================
    # Phase 6: Automated Post-Processing Handoff to Unittests
    # ==============================================================================
    print("[6] AUTOMATED HANDOFF: Triggering Automatic Unittests...", flush=True)
    
    # Dynamically resolve the path to 1-Automatic-Unittests.py
    current_script_dir = Path(__file__).resolve().parent
    unittest_script = (current_script_dir.parent / "3-agilengine" / "1-Automatic-Unittests.py").resolve()
    
    if unittest_script.exists():
        try:
            # Execute the unittest script, passing the generated POLISHED_SYNTHESIS.md file
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
