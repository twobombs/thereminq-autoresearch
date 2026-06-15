<img width="1016" height="443" alt="Screenshot from 2026-01-01 15-36-30" src="https://github.com/user-attachments/assets/b5c8d613-5dac-484a-baef-0032dfd8e484" />


# ThereminQ Autoresearch

A collection of utilities and autonomous workflows for the ThereminQ project. This repo acts as the nexus for AI-driven automation, bridging project management, deep research, workspace synchronization, and generative reporting.



### VDI Environment

To run the VDI environment, you can use the following command:

```bash
docker run --gpus all --device=/dev/kfd --device=/dev/dri:/dev/dri -p 6080:6080 -d twobombs/thereminq-tensors:jupyter
```

It is a good safety measure to also run the Inference engines with tooling enabled inside a container because 
- preinstalled tooling and guardrails provided by the Docker engine


<img width="2720" height="2480" alt="thereminq_autoresearch_architecture" src="https://github.com/user-attachments/assets/c6e71290-7cba-4c90-9524-c192f65346b4" />


## Scripts

### Build Infrastructure (`0-build/`)

*   **`0-build/build_llama.sh`**
    **Functional Description:** Shell script to clone and build `llama.cpp` from source with Vulkan support enabled, copying the resulting build to two separate directories (`llama.cpp-embedded` and `llama.orch`).

*   **`0-build/fetch_llamas.sh`**
    **Functional Description:** Shell script to download required GGUF model files (Qwen 27B/9B, Orchestrator 8B, Nomic embed text) from Hugging Face into a local directory.

### Run Infrastructure (`1-runinfra/`)

*   **`1-runinfra/launch27B.sh`**
    **Functional Description:** Bash script to launch a single local llama-server instance serving the Qwen3.6-27B model on port 8033 using Vulkan backend.

*   **`1-runinfra/launch9B.sh`**
    **Functional Description:** Bash script to launch a single local llama-server instance serving the Qwen3.5-9B model on port 8033.

*   **`1-runinfra/launch_nomid.sh`**
    **Functional Description:** Bash script to launch a local embedding server using the Nomic embed text model on port 8034.

*   **`1-runinfra/start-orchestrator.sh`**
    **Functional Description:** Bash script to start the local orchestrator node using the nvidia_Orchestrator-8B model on port 8080.

*   **`1-runinfra/start-zerg-swarm.sh`**
    **Functional Description:** Comprehensive bash script to ignite a local 6-node agentic swarm using Qwen3.5-9B, mapping instances to physical NUMA nodes and Vulkan devices across ports 8030-8035 with auto-restart and isolated shader caching.

### Core Orchestration (`2-startcore/`)

*   **`2-startcore/0-generate-macrotask.py`**
    **Functional Description:** Python script utilizing a local LLM to generate raw markdown content or documents from user prompts.

*   **`2-startcore/1-distill-macrotask.py`**
    **Functional Description:** Python script that reads technical documents and leverages an orchestrator LLM to distill them into actionable markdown lists of tasks and requirements.

*   **`2-startcore/2-full-agentic-workflow.py`**
    **Functional Description:** Unified LLM orchestrator engine orchestrating multi-phase processing: breaking down complex queries, distributing tasks to parallel worker nodes, and synthesizing the final cohesive output.

*   **`2-startcore/3-post_process_synthesis.py`**
    **Functional Description:** Distributed text-processing script designed to semantically deduplicate, smooth section transitions, and consolidate large documents while preserving critical code artifacts.

*   **`2-startcore/macrotask-example-prompt.txt`**
    **Functional Description:** Example structured text prompt used to test complex interdisciplinary synthesis tasks using the local LLM pipeline.

### Agile Engine (`3-agilengine/`)

*   **`3-agilengine/1-Automatic-Unittests.py`**
    **Functional Description:** Python script that parses markdown for code blocks and leverages an LLM swarm to concurrently generate and execute unit tests, aggregating the results.

*   **`3-agilengine/2-Todo-Project-Distill.py`**
    **Functional Description:** Python script that scans a project directory, concatenates relevant raw logs/code, and distills actionable tasks via the local Orchestrator node, producing `DISTILLED_TASKS.md`.

*   **`3-agilengine/4-daily-Agentic-Agile-report.py`**
    **Functional Description:** Deep-scan agentic control loop that ingests project files, distributes task extraction across an LLM swarm, merges them into a JSON state, and synthesizes a daily agile markdown report.

### MCP Servers (`4-MCPs/`)

*   **`4-MCPs/Agentic-local-MCP.py`**
    **Functional Description:** FastMCP server acting as a bridge to securely expose local read-only project state, wiki indexes, and orchestrator query tools to external MCP-compatible clients.

*   **`4-MCPs/Atlassian-ingress-MCP.py`**
    **Functional Description:** FastMCP server serving as a local bridge to Jira and Confluence, providing tools to fetch active sprints, create Jira tickets from local state, and publish synthesis reports to Confluence.

### Visualization (`5-viz/`)

*   **`5-viz/a1111-status-visualizer.py`**
    **Functional Description:** Python script querying a local LLM to translate project state into a Stable Diffusion prompt, then leveraging a local Automatic1111 API to generate an abstract visual dashboard snapshot.

### Misc Utilities (`9-misc/`)

*   **`9-misc/deep-local-research.py`**
    **Functional Description:** Autonomous web research script utilizing DuckDuckGo scraping, local reasoning models to draft content, and orchestrator models to verify and output formatted PDF reports.

*   **`9-misc/git-compare-and-merge.py`**
    **Functional Description:** AI-driven conflict resolver script that uses local AI models to intelligently read and resolve standard git merge conflicts by replacing conflict markers with merged code.

*   **`9-misc/local-discord-bot.py`**
    **Functional Description:** Simple asynchronous Python script running a Discord bot that listens for mentions and replies conversationally using a local llama-server LLM.

## Cohesive Swarm Workflow (Startup Sequence)

To create a fully cohesive local AI ecosystem, start the scripts in the following logical sequence:

1. **Infrastructure:** Execute `1-runinfra/start-zerg-swarm.sh` to ignite the underlying LLM swarm, ensuring all local API endpoints (e.g., ports 8030-8035) are online and ready to accept requests.
2. **Orchestration & Automated Pipeline:** Launch `2-startcore/2-full-agentic-workflow.py` to establish the master routing and task breakdown capabilities across the swarm. This now acts as an automated pipeline cascading into:
    * `2-startcore/3-post_process_synthesis.py` for text processing,
    * `3-agilengine/1-Automatic-Unittests.py` for automated test generation and execution,
    * `3-agilengine/2-Todo-Project-Distill.py` to generate tasks from tests and logs,
    * `2-startcore/1-distill-macrotask.py` to convert those tasks into actionable items.
3. **Project State & Knowledge:** Run `3-agilengine/4-daily-Agentic-Agile-report.py` to ingest new context and update the agile state (`project_state.json`).
4. **Integrations & Interfaces:** Start bridge services like `4-MCPs/Agentic-local-MCP.py` and `4-MCPs/Atlassian-ingress-MCP.py` to expose local state to external tools, and run `9-misc/local-discord-bot.py` to provide a conversational interface.
5. **On-Demand Utilities:** Use scripts like `9-misc/deep-local-research.py`, `9-misc/git-compare-and-merge.py`, or `5-viz/a1111-status-visualizer.py` as needed for specific tasks, leveraging the established infrastructure.

## Visualisation of Architecture

<img width="1024" height="559" alt="59dd6e06-67a8-4e34-963a-2a0bcfbb1f92" src="https://github.com/user-attachments/assets/eb4a275a-c59d-4fcd-8b73-6be2f3d45e80" />
