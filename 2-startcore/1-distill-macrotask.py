import os
import sys
import argparse
from pathlib import Path
from openai import OpenAI

# ==============================================================================
# Configuration
# ==============================================================================

# Using the same local LLM endpoint pattern as the Orchestrator
DISTILLER_URL = os.getenv("DISTILLER_URL", "http://192.168.2.137:8080/v1")
DISTILLER_MODEL = os.getenv("DISTILLER_MODEL", "nvidia_Orchestrator-8B-Q6_K.gguf")
DISTILLER_API_KEY = os.getenv("DISTILLER_API_KEY", "local-sk")

# Safety threshold: Warn if input exceeds typical 8k token context (~30,000 chars)
# Adjust this based on your specific model's context window.
WARN_CHAR_LIMIT = 30000 

client = OpenAI(base_url=DISTILLER_URL, api_key=DISTILLER_API_KEY)

# ==============================================================================
# Core Functions
# ==============================================================================

def read_file_content(file_path: Path) -> str:
    """Reads the content of the target document with fallback encodings."""
    encodings = ['utf-8', 'latin-1', 'windows-1252']
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"[!] Error reading file '{file_path}': {e}", flush=True)
            sys.exit(1)
            
    print(f"[!] Fatal: Could not decode '{file_path}' using standard encodings.", flush=True)
    sys.exit(1)

def distill_document(raw_text: str) -> str:
    """
    Forces the LLM to ruthlessly extract actionable to-dos from fluffy text.
    """
    char_count = len(raw_text)
    print(f"[*] Ingesting document ({char_count:,} characters)...", flush=True)
    
    if char_count > WARN_CHAR_LIMIT:
        print(f"    [!] WARNING: Document size exceeds {WARN_CHAR_LIMIT:,} characters.", flush=True)
        print("    [!] The LLM may truncate context or hallucinate. Consider chunking the input.", flush=True)

    print("[*] Distilling into actionable tasks...", flush=True)

    system_prompt = (
        "You are a ruthless, highly technical Lead Engineer and Project Manager. "
        "Your job is to read dense, fluffy, or theoretical technical documents and extract ONLY "
        "a succinct, actionable list of explicit TO-DOs, architectural requirements, and implementation tasks. "
        "STRIP AWAY all marketing fluff, academic rambling, metaphors, and context setting. "
        "Output a clean, highly structured Markdown list of tasks that a developer can immediately start building. "
        "Do not include conversational filler."
    )

    try:
        response = client.chat.completions.create(
            model=DISTILLER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract the actionable tasks from this document:\n\n{raw_text}"}
            ],
            temperature=0.3, # Low temperature for analytical precision
            max_tokens=8192
        )
        
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[!] Error during distillation: {e}", flush=True)
        sys.exit(1)

def save_distilled_output(output_text: str, original_path: Path) -> Path:
    """Saves the cleaned task list next to the original file."""
    output_filename = f"{original_path.stem}_distilled.md"
    output_path = original_path.parent / output_filename
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        return output_path
    except Exception as e:
        print(f"[!] Error saving output file: {e}", flush=True)
        sys.exit(1)

# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fluff-to-Action Technical Document Distiller")
    parser.add_argument("file", type=str, help="Path to the vague/fluffy text document.")
    args = parser.parse_args()

    input_path = Path(args.file).resolve()
    
    if not input_path.exists():
        print(f"[!] Fatal: File '{input_path}' not found.", flush=True)
        sys.exit(1)

    print("\n=== STARTING INGESTION & DISTILLATION ===", flush=True)
    
    raw_content = read_file_content(input_path)
    actionable_tasks = distill_document(raw_content)
    saved_path = save_distilled_output(actionable_tasks, input_path)
    
    print(f"\n[+] Distillation complete. File saved to: {saved_path}", flush=True)
    print(f"[*] Suggested next step: python full-agentic-workflow.py -f {saved_path}", flush=True)
