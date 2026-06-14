import os
import argparse
import logging
import requests
import subprocess
from pathlib import Path

# ==============================================================================
# Configuration & Endpoints
# ==============================================================================

# Target the Orchestrator node for high-level distillation tasks
ORCHESTRATOR_ENDPOINT = os.getenv("ORCHESTRATOR_ENDPOINT", "http://192.168.2.137:8080/v1/chat/completions")
ORCH_API_KEY = os.getenv("ORCH_API_KEY", "local-sk")

# Tuned for a 40k token context window (~3.5 chars per token = ~140,000 chars max)
# Scaled down to 60k to prevent KV cache thrashing and TTFT bottlenecks
MAX_CONTEXT_CHARS = 60000

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ==============================================================================
# LLM Interaction
# ==============================================================================

def extract_project_tasks(project_name: str, raw_content: str) -> str | None:
    """Sends raw project documents to the local Orchestrator for task distillation."""
    log.info(f"[{project_name}] Dispatching text to Orchestrator for distillation...")
    
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
        "Do NOT use emojis, smart quotes, em-dashes, or specialized unicode symbols. "
        "Use standard hyphens (-) or asterisks (*) for bullet points. "
        "\n\nOutput a clean, highly structured Markdown document. Do not include conversational filler."
    )
    
    headers = {
        "Authorization": f"Bearer {ORCH_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract actionable tasks, test outcomes, and relevant artifacts from this {project_name} documentation:\n\n{raw_content}"}
        ],
        "temperature": 0.2, 
        "max_tokens": 8192
    }

    try:
        response = requests.post(ORCHESTRATOR_ENDPOINT, headers=headers, json=payload, timeout=600)
        response.raise_for_status()
        
        # Strip output to ensure no leading/trailing whitespace anomalies
        raw_output = response.json()["choices"][0]["message"]["content"].strip()
        
        # Hard enforcement: encode to ascii and ignore errors to drop any hallucinated unicode
        ascii_output = raw_output.encode("ascii", "ignore").decode("ascii")
        return ascii_output
        
    except Exception as e:
        log.error(f"[{project_name}] Orchestrator inference failed: {e}")
        return None

# ==============================================================================
# Core Operations
# ==============================================================================

def process_project_directory(project_dir: Path) -> bool:
    """Scans a single project directory, concatenates raw files strictly in read-only mode, and generates a distilled task list."""
    project_name = project_dir.name
    
    # Expanded scope to catch CSV and JSON test reports alongside standard logs
    raw_files = (
        list(project_dir.rglob("*.txt")) + 
        list(project_dir.rglob("*.md")) + 
        list(project_dir.rglob("*.csv")) + 
        list(project_dir.rglob("*.json"))
    )
    
    # Exclude existing distilled files or project state files to prevent recursive ingestion
    raw_files = [f for f in raw_files if "DISTILLED_TASKS" not in f.name and "project_state" not in f.name]

    if not raw_files:
        log.info(f"[{project_name}] No raw documentation or test logs found. Skipping.")
        return False

    log.info(f"[{project_name}] Found {len(raw_files)} raw files. Aggregating context (Read-Only)...")
    
    aggregated_content = []
    for file_path in raw_files:
        try:
            # STRICT READ-ONLY: Artifacts are read into memory, not altered or moved
            # Enforcing ascii reading to prevent unicode bleed from raw logs
            with open(file_path, "r", encoding="ascii", errors="ignore") as f:
                aggregated_content.append(f"--- SOURCE: {file_path.name} ---\n{f.read()}")
        except Exception as e:
            log.warning(f"[{project_name}] Could not read {file_path.name}: {e}")

    full_text = "\n\n".join(aggregated_content)
    
    # Leverage the truncated 60k token context window for KV cache health
    if len(full_text) > MAX_CONTEXT_CHARS:
        log.warning(f"[{project_name}] Aggregated text exceeds {MAX_CONTEXT_CHARS} characters. Truncating.")
        full_text = full_text[:MAX_CONTEXT_CHARS] + "\n\n...[CONTENT TRUNCATED FOR CONTEXT LIMITS]..."

    distilled_markdown = extract_project_tasks(project_name, full_text)
    
    if distilled_markdown:
        # The output artifact is written directly into the target directory
        output_file = project_dir / "DISTILLED_TASKS.md"
        try:
            with open(output_file, "w", encoding="ascii", errors="ignore") as f:
                f.write(f"# Distilled Tasks: {project_name}\n\n{distilled_markdown}\n")
            log.info(f"[{project_name}] Successfully saved distilled tasks to {output_file.name}")
            return True
        except Exception as e:
            log.error(f"[{project_name}] Failed to save output file: {e}")
            
    return False

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Distill Raw Documents and Test Logs for a Single Project Directory.")
    parser.add_argument(
        "project_dir", 
        type=str, 
        help="Path to the specific project directory."
    )
    args = parser.parse_args()

    # Resolve the absolute path to bind the script strictly to the CLI input
    project_path = Path(args.project_dir).resolve()
    
    if not project_path.exists() or not project_path.is_dir():
        log.error(f"Fatal: Directory '{project_path}' does not exist or is not a directory.")
        return

    log.info("==================================================")
    log.info(f"STARTING PROJECT DISTILLATION")
    log.info(f"Target Project: {project_path.name}")
    log.info(f"Working Directory Bound To: {project_path}")
    log.info("==================================================\n")

    # Explicitly change the current working directory to the target to enforce local operation
    os.chdir(project_path)

    success = process_project_directory(project_path)

    log.info("\n==================================================")
    if success:
        log.info(f"DISTILLATION FINISHED SUCCESSFULLY")
        
        # --- AUTOMATED HANDOFF ---
        # Calculate the absolute path to the macrotask script relative to THIS script's location
        current_script_dir = Path(__file__).resolve().parent
        macrotask_script = (current_script_dir.parent / "2-startcore" / "1-distill-macrotask.py").resolve()
        
        distilled_file = project_path / "DISTILLED_TASKS.md"
        
        if macrotask_script.exists():
            log.info(f"Initiating seamless handoff to {macrotask_script.name}...")
            try:
                # Execute the macrotask script. 
                # Pass the generated markdown file as an argument and lock the CWD to the project path.
                subprocess.run(
                    ["python3", str(macrotask_script), str(distilled_file)], 
                    cwd=project_path, 
                    check=True
                )
                log.info("Handoff pipeline completed successfully.")
            except subprocess.CalledProcessError as e:
                log.error(f"Macrotask script failed with exit code: {e.returncode}")
            except Exception as e:
                log.error(f"Execution error during handoff: {e}")
        else:
            log.warning(f"Could not locate {macrotask_script}. Skipping automated handoff.")
            
    else:
        log.info(f"DISTILLATION FAILED OR SKIPPED")
    log.info("==================================================")

if __name__ == "__main__":
    main()
