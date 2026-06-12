import os
import re
import json
import requests
import queue
import concurrent.futures
from pathlib import Path

# =====================================================================
# CONFIGURABLE WORKER ENDPOINTS
# Map these ports to your llama.cpp instances bound to specific GPUs.
# For example, across a dual-socket H11DSi board running a 6-GPU mesh:
# =====================================================================
WORKER_ENDPOINTS = [
    "http://127.0.0.1:8030/v1/chat/completions", # Target: GPU 0
    "http://127.0.0.1:8031/v1/chat/completions", # Target: GPU 1
    "http://127.0.0.1:8032/v1/chat/completions", # Target: GPU 2
    "http://127.0.0.1:8033/v1/chat/completions", # Target: GPU 3
    "http://127.0.0.1:8034/v1/chat/completions", # Target: GPU 4
    "http://127.0.0.1:8035/v1/chat/completions", # Target: GPU 5
]


def extract_code_blocks(md_content: str, output_dir: str) -> list:
    """
    Parses a Markdown string, extracts code blocks, identifies their intended 
    filenames based on context heuristics, and writes them to a local directory.
    
    Returns a list of dictionaries containing file metadata for the AI workers.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    lines = md_content.splitlines()
    in_block = False
    current_block = []
    current_lang = ""
    detected_filename = None
    last_header = None
    file_counter = 1
    
    extracted_artifacts = []

    for i, line in enumerate(lines):
        # Heuristic 1: Detect XML-like file paths
        xml_match = re.search(r'<file path="([^"]+)">', line)
        if xml_match:
            detected_filename = xml_match.group(1)
            continue
            
        # Heuristic 2: Detect Markdown headers for filenames
        header_match = re.match(r'^###?\s+([a-zA-Z0-9_\-\.]+.*)$', line)
        if header_match:
            potential_name = header_match.group(1).strip()
            if "." in potential_name or potential_name.lower() in ["dockerfile", "makefile"]:
                detected_filename = potential_name
            last_header = potential_name
            continue
            
        # Block processing
        if line.startswith("```"):
            if not in_block:
                in_block = True
                current_lang = line[3:].strip()
                current_block = []
                
                # Heuristic 3: Check if the very next line is a comment with the filename
                if i + 1 < len(lines):
                    next_line = lines[i+1].strip()
                    if next_line.startswith("# ") and "." in next_line:
                        detected_filename = next_line[2:].strip()
            else:
                in_block = False
                content = "\n".join(current_block)
                
                # Resolve filename if heuristics missed it
                if not detected_filename:
                    ext = current_lang if current_lang else "txt"
                    ext_map = {"python": "py", "bash": "sh", "yaml": "yml", "dockerfile": "Dockerfile"}
                    ext = ext_map.get(ext.lower(), ext)
                    
                    base_name = last_header.replace(" ", "_").lower() if last_header else f"artifact_{file_counter}"
                    detected_filename = f"{base_name}.{ext}" if ext.lower() != "dockerfile" else "Dockerfile"
                    file_counter += 1
                
                # Clean up paths to prevent directory traversal and write file
                safe_filename = os.path.basename(detected_filename)
                file_path = output_path / safe_filename
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content + "\n")
                
                print(f"[+] Extracted: {safe_filename} ({current_lang})")
                
                extracted_artifacts.append({
                    "filename": safe_filename,
                    "language": current_lang,
                    "filepath": str(file_path),
                    "content": content
                })
                
                # Reset state for the next block
                detected_filename = None
                current_lang = ""
                current_block = []
        elif in_block:
            current_block.append(line)
            
    return extracted_artifacts


def request_unittests_from_worker(artifact: dict, endpoint_queue: queue.Queue):
    """
    Checks out an available endpoint from the queue, sends the payload to the local 
    worker, writes the result, and returns the endpoint to the queue.
    """
    valid_langs = ["python", "py", "cpp", "c", "bash", "sh"]
    if artifact["language"].lower() not in valid_langs:
        print(f"[-] Skipping generation for non-code artifact: {artifact['filename']}")
        return

    # Block until a worker endpoint becomes available
    endpoint_url = endpoint_queue.get()
    port = endpoint_url.split(':')[-1].split('/')[0]
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
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 2048
    }

    try:
        response = requests.post(endpoint_url, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        
        test_code = result["choices"][0]["message"]["content"]
        
        # Strip generic markdown block quotes if the model wraps the output
        if test_code.startswith("```"):
            test_code = "\n".join(test_code.split("\n")[1:-1])
        
        # Save the generated test
        test_filename = f"test_{artifact['filename']}"
        test_filepath = Path(artifact['filepath']).parent / test_filename
        
        with open(test_filepath, "w", encoding="utf-8") as f:
            f.write(test_code + "\n")
            
        print(f"[+] SUCCESS [{port}]: Unit tests for {artifact['filename']} saved to {test_filename}")
        
    except requests.exceptions.RequestException as e:
        print(f"[!] FAILED [{port}]: Could not complete request for {artifact['filename']}: {e}")
    finally:
        # Always return the endpoint to the queue so the next task can use it
        endpoint_queue.put(endpoint_url)


if __name__ == "__main__":
    MARKDOWN_SOURCE = "POLISHED_SYNTHESIS.md"
    OUTPUT_WORKSPACE = "./extracted_workspace"
    
    # Initialize a thread-safe queue with our configured endpoints
    endpoint_queue = queue.Queue()
    for endpoint in WORKER_ENDPOINTS:
        endpoint_queue.put(endpoint)
    
    if not os.path.exists(MARKDOWN_SOURCE):
        print(f"Error: Could not find {MARKDOWN_SOURCE}. Please ensure the file is in the current directory.")
    else:
        with open(MARKDOWN_SOURCE, "r", encoding="utf-8") as file:
            md_content = file.read()
            
        print("=== Phase 1: Local Extraction ===")
        artifacts = extract_code_blocks(md_content, OUTPUT_WORKSPACE)
        
        print("\n=== Phase 2: Parallelized Test Generation ===")
        print(f"[*] Initializing ThreadPoolExecutor with {len(WORKER_ENDPOINTS)} concurrent workers...")
        
        # Use a ThreadPoolExecutor sized exactly to the number of available endpoints
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(WORKER_ENDPOINTS)) as executor:
            # Map the target function across all extracted artifacts
            futures = [
                executor.submit(request_unittests_from_worker, artifact, endpoint_queue)
                for artifact in artifacts
            ]
            
            # Wait for all futures to complete
            concurrent.futures.wait(futures)
            
        print("\n=== Pipeline execution complete ===")
