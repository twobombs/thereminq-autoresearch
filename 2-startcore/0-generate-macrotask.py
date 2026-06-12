# this helper script can be used for testing the agentic pipeline by generating content by means of 2 or 3 words input
#
# its output can then be fed to be distilled into agent compatible mission statements by distill-macrotask.py
#

import os
import sys
import argparse
import time
import re
import subprocess
from datetime import datetime
from pathlib import Path
from openai import OpenAI

# ==============================================================================
# Configuration
# ==============================================================================

# Initialize client to hit your local llama-server
client = OpenAI(
    base_url=os.getenv("OPENAI_API_BASE", "http://localhost:8033/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "sk-local")
)

LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.6-27B-IQ4_XS.gguf")

SYSTEM_PROMPT = """You are an expert researcher and technical writer.
Your task is to write a comprehensive, detailed, and highly informative document based on the user's prompt. 
Write clearly, use markdown formatting (headings, bullet points, bold text), and provide deep insights.
Do not include any conversational filler (e.g., "Here is the article you requested"). Just output the raw document content."""

# ==============================================================================
# Helper Functions
# ==============================================================================

def setup_raw_directory(base_dir: str, category: str) -> Path:
    """Ensures the target raw subdirectory exists."""
    target_dir = Path(base_dir) / category
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir

def generate_safe_filename(prompt_text: str) -> str:
    """Creates a unique, safe filename using a timestamp and a snippet of the prompt."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Grab the first 5 words of the prompt to use as a slug
    words = re.findall(r'[a-zA-Z0-9]+', prompt_text)[:5]
    slug = "-".join(words).lower()
    
    if not slug:
        slug = "generated-content"
        
    return f"{timestamp}_{slug}.md"

def generate_content(prompt: str, target_dir: Path):
    """Generates content from the local LLM silently and saves the final output to a file."""
    print(f"\n[1] 🧠 Generating content for: '{prompt[:50]}...'")
    print("[2] 📡 Requesting response from local LLM (CLI output disabled)...")
    
    full_content = ""
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, # A bit of creativity is good for raw generation
            max_tokens=4096,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                text_chunk = chunk.choices[0].delta.content
                # Appending to string variable without printing to the console
                full_content += text_chunk
                
        elapsed = round(time.time() - start_time, 2)
        print(f"[+] Generation complete in {elapsed} seconds.")
        
        # Save to file
        filename = generate_safe_filename(prompt)
        filepath = target_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_content.strip())
            
        print(f"[3] 💾 Saved raw content to: {filepath.absolute()}")
        
        # Execute the distillation script
        print(f"\n[4] ⚙️ Executing distill-macrotask.py on {filename}...")
        try:
            # sys.executable ensures we use the exact same python environment
            subprocess.run([sys.executable, "distill-macrotask.py", str(filepath.absolute())], check=True)
            print("[+] Distillation complete.")
        except subprocess.CalledProcessError as e:
            print(f"[!] Error: distill-macrotask.py returned a non-zero exit status: {e}")
        except FileNotFoundError:
            print("[!] Error: distill-macrotask.py could not be found. Ensure it is in the same directory.")

    except Exception as e:
        print(f"\n[!] Fatal Error during generation: {e}")

# ==============================================================================
# Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Local LLM Raw Content Generator")
    
    # Input group (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", "--prompt", type=str, help="Direct string prompt for the LLM to write about.")
    group.add_argument("-f", "--file", type=str, help="Path to a text file containing the prompt instructions.")
    
    # Destination arguments
    parser.add_argument("-d", "--dir", type=str, default="raw", 
                        help="Base directory for raw files. (Default: ./raw)")
    parser.add_argument("-c", "--category", type=str, default="articles", 
                        choices=["articles", "papers", "repos", "assets"],
                        help="Subdirectory category inside the raw folder. (Default: articles)")
    
    args = parser.parse_args()
    
    # Resolve the prompt text
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"[!] Error: The prompt file '{args.file}' does not exist.")
            sys.exit(1)
        with open(file_path, "r", encoding="utf-8") as f:
            target_prompt = f.read().strip()
        print(f"[*] Loaded prompt from file: {args.file}")
    else:
        target_prompt = args.prompt
        
    if not target_prompt:
        print("[!] Error: The provided prompt is empty.")
        sys.exit(1)

    # Setup directories
    target_directory = setup_raw_directory(args.dir, args.category)
    
    # Execute generation
    generate_content(target_prompt, target_directory)

if __name__ == "__main__":
    main()
