import os
import argparse
import logging
import requests
from pathlib import Path

# ==============================================================================
# Configuration & Endpoints
# ==============================================================================

# Target the Orchestrator node for high-level distillation tasks
ORCHESTRATOR_ENDPOINT = os.getenv("ORCHESTRATOR_ENDPOINT", "http://192.168.2.134:8080/v1/chat/completions")

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
        "Read the provided raw documentation and extract ONLY a succinct, actionable "
        "list of explicit TO-DOs, architectural requirements, and implementation tasks. "
        "Output a clean, highly structured Markdown checklist. Do not include conversational filler."
    )
    
    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract actionable tasks from this {project_name} documentation:\n\n{raw_content}"}
        ],
        "temperature": 0.2, # Low temperature for analytical precision
        "max_tokens": 4096
    }

    try:
        response = requests.post(ORCHESTRATOR_ENDPOINT, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.error(f"[{project_name}] Orchestrator inference failed: {e}")
        return None

# ==============================================================================
# Core Operations
# ==============================================================================

def process_project_directory(project_dir: Path) -> bool:
    """Scans a single project directory, concatenates raw files, and generates a distilled task list."""
    project_name = project_dir.name
    
    # Look for raw text/markdown files within the project (adjust extensions as needed)
    raw_files = list(project_dir.rglob("*.txt")) + list(project_dir.rglob("*.md"))
    
    # Exclude existing distilled files or project state files to prevent infinite loops
    raw_files = [f for f in raw_files if "DISTILLED_TASKS" not in f.name and "project_state" not in f.name]

    if not raw_files:
        log.info(f"[{project_name}] No raw documentation found. Skipping.")
        return False

    log.info(f"[{project_name}] Found {len(raw_files)} raw files. Aggregating context...")
    
    aggregated_content = []
    for file_path in raw_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                aggregated_content.append(f"--- SOURCE: {file_path.name} ---\n{f.read()}")
        except Exception as e:
            log.warning(f"[{project_name}] Could not read {file_path.name}: {e}")

    full_text = "\n\n".join(aggregated_content)
    
    # Failsafe: Prevent blowing out the context window if a project folder contains massive logs
    char_limit = 40000 
    if len(full_text) > char_limit:
        log.warning(f"[{project_name}] Aggregated text exceeds {char_limit} characters. Truncating.")
        full_text = full_text[:char_limit] + "\n\n...[CONTENT TRUNCATED FOR CONTEXT LIMITS]..."

    distilled_markdown = extract_project_tasks(project_name, full_text)
    
    if distilled_markdown:
        output_file = project_dir / "DISTILLED_TASKS.md"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
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
    parser = argparse.ArgumentParser(description="Distill Raw Documents for a Single Project Directory.")
    parser.add_argument(
        "project_dir", 
        type=str, 
        help="Path to the specific project directory."
    )
    args = parser.parse_args()

    project_path = Path(args.project_dir).resolve()
    
    if not project_path.exists() or not project_path.is_dir():
        log.error(f"Fatal: Directory '{project_path}' does not exist or is not a directory.")
        return

    log.info("==================================================")
    log.info(f"STARTING PROJECT DISTILLATION")
    log.info(f"Target Project: {project_path.name}")
    log.info("==================================================\n")

    success = process_project_directory(project_path)

    log.info("\n==================================================")
    if success:
        log.info(f"DISTILLATION FINISHED SUCCESSFULLY")
    else:
        log.info(f"DISTILLATION FAILED OR SKIPPED")
    log.info("==================================================")

if __name__ == "__main__":
    main()
