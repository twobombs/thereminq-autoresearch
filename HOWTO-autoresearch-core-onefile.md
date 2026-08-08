# HOWTO: autoresearch-core-onefile.py

## Overview
`autoresearch-core-onefile.py` is the central engine of the ThereminQ Autoresearch project. It acts as an end-to-end agentic content generation pipeline. This script automates a complex, multi-stage workflow leveraging local LLMs to ingest raw inputs (either text prompts, files, or entire git repositories), break down complex tasks, orchestrate distributed map-reduce summarization and generation across an AI swarm, compile actionable task lists, execute automated unit tests, and synthesize a polished markdown report.

## Prerequisites
Before running the pipeline, ensure the ThereminQ local LLM swarm is online. The script expects several local model endpoints to be available:
- **Generation API (Phase 1):** Default `http://localhost:8081/v1`
- **Distiller API (Phase 2):** Default `http://localhost:8081/v1`
- **Orchestrator Nodes (Phase 3/4/6):** Default `http://localhost:8080/v1` and `http://localhost:8079/v1`
- **Stitcher Nodes (Phase 3):** Default `http://localhost:8070/v1` and `http://localhost:8071/v1`
- **Worker Nodes (Phase 3/5):** Default `http://localhost:8033/v1` and `http://localhost:8034/v1`

These endpoints are typically started using the `1-runinfra/start-zerg-swarm.sh` and related startup scripts.

## Command-Line Interface (CLI)

The script provides a robust CLI to handle different types of inputs and configurations.

### Core Input Arguments (Mutually Exclusive)
You must provide exactly one of the following input methods, or use the resume flag:
- `-p PROMPT`, `--prompt PROMPT`: Provide a direct text prompt to initiate the pipeline.
- `-f FILE`, `--file FILE`: Provide a path to a text file containing the prompt.
- `-g GIT`, `--git GIT`: Provide a Git repository URL (HTTPS or SSH) to clone, analyze, and process as the pipeline input.
- `-r`, `--resume`: Resume the pipeline from the furthest completed artifact in the target output directory. Useful if a previous run was interrupted.

### Git Intake Configurations (Used with `-g`)
- `--focus FOCUS`: Provide an optional analysis focus. The model will prioritize findings relevant to this focus when summarizing the repository.
- `--git-path GIT_PATH`: Specify a specific file or folder path within the cloned repository to restrict the ingest process to that specific subtree.

### Output Configurations
- `-d DIR`, `--dir DIR`: Base directory for storing all outputs. (Default: `run_data`)
- `-c CATEGORY`, `--category CATEGORY`: Category folder name within the base directory to organize runs logically. (Default: `projects`)
- `--iterate`: Enable iterative distillation for Phase 6.

### Example Commands
**Analyze a Git Repository:**
```bash
python autoresearch-core-onefile.py -g https://github.com/example/repo.git --focus "security vulnerabilities" -d outputs -c code_audits
```

**Generate Content from a File:**
```bash
python autoresearch-core-onefile.py -f my_prompt.txt
```

**Resume an Interrupted Run:**
```bash
python autoresearch-core-onefile.py -r -c my_existing_project
```

## Pipeline Architecture (The 7 Phases)

The script operates through a sequence of distributed, agentic phases:

### Phase 0: Git Repository Intake (If `-g` is used)
Clones the target git repository to a temporary directory. It recursively scans and ingests source code, filtering out binary and non-essential files. If the repository is larger than the context window, it triggers a parallelized map-reduce process across worker nodes to generate a compressed, dense technical summary of the codebase.

### Phase 1: Raw Content Generation (If `-p` or `-f` is used)
If a prompt is provided instead of a git repository, a generative model is used to draft a comprehensive, detailed markdown document based on the user's instructions.

### Phase 2: Fluff-to-Action Technical Distillation
Ingests the raw content (from Phase 0 or 1) and distills it. A Lead Engineer persona agent strips away fluff and extracts a succinct, highly structured Markdown list of explicit TO-DOs, architectural requirements, and implementation tasks.

### Phase 3: Distributed Orchestrator Cluster (Map-Reduce)
The core of the engine. The orchestrator breaks down the distilled query into atomic, independent sub-tasks (Decomposition). These sub-tasks are pushed to a parallel queue where multiple worker nodes independently execute them, generating code, configs, or documentation snippets. The orchestrator then merges these micro-reports back into a single cohesive document via sequential chunk synthesis and a rolling master stitch using a dedicated stitcher cluster via parallel tree reduction.

### Phase 4: Distributed Parallel Post-Processing
Acts as an executive editor. The massive document generated in Phase 3 is split into chunks. Orchestrator nodes perform semantic deduplication, boundary smoothing, and header unification to clean up the merged document, while strictly protecting generated code blocks.

### Phase 5: Automatic Unittests
Extracts all embedded code blocks from the Phase 4 document. The worker swarm rapidly generates succinct unit tests for Python, C/C++, and Bash artifacts. It then executes these tests in isolated subprocesses and compiles a telemetry report (JSON/CSV) of successes and failures.

### Phase 6: Todo Project Distillation
Reads the finalized documentation alongside the test execution telemetry from Phase 5. It performs a final distillation pass to produce an actionable `DISTILLED_TASKS.md` list, automatically elevating failed unit tests into high-priority actionable tasks embedded with the failing source code. If the `--iterate` flag is provided, it enables iterative distillation to append to an existing `DISTILLED_TASKS.md` file based on new telemetry.

## Output Structure

All outputs are saved to: `{DIR}/{CATEGORY}/run_{TIMESTAMP}_{UUID}/`

Key artifacts generated:
- `*_distilled.md`: The output from Phase 2.
- `tasks/`: Directory containing the atomic task chunks (Phase 3).
- `artifacts/`: Extracted code blocks and files generated by the workers.
- `FINAL_SYNTHESIS.md`: The raw stitched master document (Phase 3).
- `POLISHED_SYNTHESIS.md`: The final edited, deduplicated, and smoothed document (Phase 4).
- `tests/`: Generated unit test files (Phase 5).
- `reports/execution_report.json`: Telemetry of the automated test runs (Phase 5).
- `DISTILLED_TASKS.md`: The final actionable task list, including test failures (Phase 6).
