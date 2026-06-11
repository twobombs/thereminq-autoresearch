<img width="1016" height="443" alt="Screenshot from 2026-01-01 15-36-30" src="https://github.com/user-attachments/assets/b5c8d613-5dac-484a-baef-0032dfd8e484" />


# ThereminQ Autoresearch Tool Cupboard

A collection of auxiliary utilities and autonomous workflows for the ThereminQ Holoqubed project. This directory acts as the nexus for AI-driven automation, bridging project management, deep research, workspace synchronization, and generative reporting.

## Scripts

### Build & Infrastructure (`0-build/`, `1-runinfra/`)

*   **`0-build/build_llama.sh`**
    **Functional Description:** Automates the fetching, compilation, and setup of the `llama.cpp` inference engine.
    **Internal Workings:** Installs required build tools, clones the `llama.cpp` repository, configures it with Vulkan support via CMake, builds it, and creates multiple distinct directory copies (`llama.cpp-embedded`, `llama.orch`) to support isolated node execution in the swarm.

*   **`0-build/fetch_llamas.sh`**
    **Functional Description:** A simple fetch script to download the requisite GGUF models from Hugging Face.
    **Internal Workings:** Uses `wget` to pull down specific pre-quantized reasoning, orchestration, and embedding models (e.g., Qwen, Nemotron, Nomic).

*   **`1-runinfra/launch27B.sh`**
    **Functional Description:** A targeted launch script for a large 27B parameter LLM.
    **Internal Workings:** Launches `llama-server` bound to specific NUMA nodes utilizing `numactl` and running on Vulkan2, specifically configured with multi-target-prediction (draft-mtp) and speculative decoding.

*   **`1-runinfra/launch9B.sh`**
    **Functional Description:** A targeted launch script for a lighter 9B parameter orchestrator LLM.
    **Internal Workings:** Boots `llama-server` configured for Vulkan0 and pinned to NUMA node 0, handling orchestration tasks on port 8080.

*   **`1-runinfra/launch_nomid.sh`**
    **Functional Description:** A script to initialize the dense text embedding node.
    **Internal Workings:** Runs `llama-server` in embedding mode using the `nomic-embed-text` model on Vulkan0 and port 8034 to support RAG and similarity search operations.

*   **`1-runinfra/start-orchestrator.sh`**
    **Functional Description:** A bash script that launches the orchestrator node.
    **Internal Workings:** Executes `llama-server` configured with specific parameters for the Orchestrator model to handle complex queries efficiently.

*   **`1-runinfra/start-zerg-swarm.sh`**
    **Functional Description:** A bash script that orchestrates the initialization of a local 6-node LLM swarm.
    **Internal Workings:** It launches multiple `llama-server` instances in the background, utilizing `numactl` to strictly pin each process to specific NUMA nodes. This optimizes memory affinity and PCIe bus utilization while mapping the instances to a predetermined array of local API ports for the swarm topology.

### Core Orchestration (`2-startcore/`)

*   **`2-startcore/distill-macrotask.py`**
    **Functional Description:** A script that uses an LLM to distill actionable tasks from dense or fluffy technical documents.
    **Internal Workings:** Reads a specified text document and sends its content to an orchestrator-level LLM with a system prompt instructing it to extract only a clean, actionable markdown list of to-dos and requirements. It saves this distilled list alongside the original file.

*   **`2-startcore/full-agentic-workflow.py`**
    **Functional Description:** A Unified Local LLM Orchestrator Engine that handles hyper-granular decomposition, parallel dispatch across worker nodes, artifact harvesting, and final synthesis.
    **Internal Workings:** It implements a multi-phase architecture: 1) Uses an Orchestrator model to break a complex query into atomic pieces. 2) Saves these pieces to disk in a timestamped run directory. 3) Utilizing Python's `concurrent.futures`, it dispatches tasks across multiple parallel worker endpoints, explicitly extracting and saving generated file artifacts. 4) Synthesizes all worker outputs and artifacts into a cohesive final document using the Orchestrator model.

*   **`2-startcore/generate-macrotask.py`**
    **Functional Description:** A utility script to stream the generation of large markdown documents or raw content using a local LLM based on a direct prompt or an input file.
    **Internal Workings:** It calls the local LLM endpoint with a system prompt optimized for expert technical writing. It streams the response to the console in real-time and ultimately saves the output as a distinct markdown file in a categorized `raw/` subdirectory with a safely generated timestamped filename.

*   **`2-startcore/macrotask-example-prompt.txt`**
    **Functional Description:** A comprehensive, highly structured LLM prompt designed to elicit a deep, interdisciplinary synthesis combining agile project management, quantum mechanical computational principles, and llm wiki methodics.
    **Internal Workings:** This text file provides a standardized, complex input that can be fed into `generate-macrotask.py` to test the agentic pipeline's ability to handle demanding, multi-faceted constraints and structure generation.

### Agile Engine (`3-agilengine/`)

*   **`3-agilengine/AgenticAgile.py`**
    **Functional Description:** An Agentic Project Orchestration pipeline that manages a "Living Wiki" by continuously ingesting raw transcripts and extracting actionable project state.
    **Internal Workings:** It utilizes a local OpenAI-compatible API endpoint with multithreaded chunk processing (`concurrent.futures.ThreadPoolExecutor`) to parse raw logs (e.g., mock Slack logs). It extracts tasks, architectural decisions, and blockers, saving the state to `project_state.json`. It ultimately synthesizes an Automated Daily Agile Dashboard summarizing progress, risks, and velocity adjustments.

*   **`3-agilengine/Atlassian-suite.py`**
    **Functional Description:** A FastMCP server script acting as an ingress bridge to Atlassian tools, integrating the local agentic workflow with Jira and Confluence.
    **Internal Workings:** It authenticates with Atlassian APIs using HTTP Basic Auth. It exposes Jira active sprints as a read-only resource by fetching incomplete issues from the specified Jira project. It provides agentic tools to sync local tasks from `project_state.json` to automatically create Jira tickets, and tools to publish the `DAILY_SYNTHESIS.md` generated by `AgenticAgile.py` directly to a Confluence page.

### Visualization (`5-viz/`)

*   **`5-viz/a1111-status-visualizer.py`**
    **Functional Description:** An intermediary visual output generator that creates abstract dashboard snapshots representing the project's state.
    **Internal Workings:** It takes raw analytical output from the Orchestrator or project state data and queries a local LLM to translate it into a highly descriptive Stable Diffusion visual prompt (e.g., a glowing UI if active, a red/glitchy UI if blocked). It then sends this formulated prompt to a local Automatic1111 API (`/sdapi/v1/txt2img`) to generate an image and saves the resulting base64 payload as a PNG in the wiki assets directory.

### Misc Utilities (`9-misc/`)

*   **`9-misc/deep-local-research.py`**
    **Functional Description:** An autonomous local LLM web research script that conducts deep web scraping to synthesize complex technical reports. It filters out unwanted formats (e.g., video/image sites like YouTube and TikTok) and outputs a formatted PDF report using `fpdf`.
    **Internal Workings:** It employs a multi-step analysis workflow powered by local LLM endpoints (an Orchestrator model and a Reasoning model).
    1. **Tool Planning:** The Orchestrator plans the search strategy and invokes a deep web scraper (`perform_web_search`) utilizing DuckDuckGo (`ddgs`) with `concurrent.futures.ThreadPoolExecutor` for parallelized fetching.
    2. **Reasoning:** A large reasoning model (e.g., qwen-3.5-35b) drafts a live response from the scraped facts using streaming inference to manage context sizes safely.
    3. **Verification:** The Orchestrator edits and polishes the draft into a final PDF report.

*   **`9-misc/git-compare-and-merge.py`**
    **Functional Description:** An AI-Driven Git Merge Conflict Resolver that detects merge conflicts within files and delegates intelligent resolution to local AI models.
    **Internal Workings:** It reads conflicted files, identifies git conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`), and sends the conflicting blocks to a Reasoner model to intelligently draft a resolution based on the surrounding context. It then passes the resolved file to an Orchestrator model to verify code integrity and ensure all conflict markers are fully removed before automatically writing the clean, resolved file back to disk.

*   **`9-misc/local-discord-bot.py`**
    **Functional Description:** A lightweight integration bridging a local Discord bot to a local LLM server to provide a conversational AI assistant within a Discord server.
    **Internal Workings:** It uses the `discord.py` library to listen for direct mentions or specific commands. To ensure non-blocking interactions and keep the Discord bot responsive, it wraps requests to the local LLM server (e.g., via port 8033) in `asyncio.to_thread`. It seamlessly streams the AI's generated responses back into the Discord channel.

*   **`9-misc/mcp-workspace-bridge.py`**
    **Functional Description:** A Model Context Protocol (MCP) server script acting as a local workspace bridge, securely exposing local project files and tools to external clients.
    **Internal Workings:** Built on `FastMCP`, it exposes read-only resources (such as Agile project state, Wiki index, and Daily Synthesis reports) and actionable tools (like document ingestion and local orchestrator querying) over standard input/output (stdio). This allows external LLM clients supporting the MCP standard to securely inspect and interact with the local workspace.

## Cohesive Swarm Workflow (Startup Sequence)

To create a fully cohesive local AI ecosystem, start the scripts in the following logical sequence:

1. **Infrastructure:** Execute `1-runinfra/start-zerg-swarm.sh` to ignite the underlying LLM swarm, ensuring all local API endpoints (e.g., ports 8030-8035) are online and ready to accept requests.
2. **Orchestration:** Launch `2-startcore/full-agentic-workflow.py` to establish the master routing and task breakdown capabilities across the swarm.
3. **Project State & Knowledge:** Run `3-agilengine/AgenticAgile.py` to ingest new context and update the agile state (`project_state.json`).
4. **Integrations & Interfaces:** Start bridge services like `9-misc/mcp-workspace-bridge.py` and `3-agilengine/Atlassian-suite.py` to expose local state to external tools, and run `9-misc/local-discord-bot.py` to provide a conversational interface.
5. **On-Demand Utilities:** Use scripts like `9-misc/deep-local-research.py`, `9-misc/git-compare-and-merge.py`, or `5-viz/a1111-status-visualizer.py` as needed for specific tasks, leveraging the established infrastructure.

## Architecture Visuals

<img width="2816" height="1536" alt="Gemini_Generated_Image_3cnxrm3cnxrm3cnx" src="https://github.com/user-attachments/assets/100a743b-a893-42b5-8e37-3dbc221ed72f" />
<img width="2816" height="1536" alt="Gemini_Generated_Image_j6xa7dj6xa7dj6xa" src="https://github.com/user-attachments/assets/db95e0fa-c8bb-4e96-9aac-9c9224aa1ed2" />
<img width="2816" height="1536" alt="gemini_generated_image_j0jsqnj0jsqnj0js" src="https://github.com/user-attachments/assets/79406291-6eba-4621-aebd-387f852a714a" />
<img width="2528" height="1696" alt="gemini_generated_image_x8xdflx8xdflx8xd" src="https://github.com/user-attachments/assets/62637080-7f76-4b5f-b701-04d35eceb793" />
