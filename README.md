# ThereminQ-Autoresearch: A Multi-Tier Local Agentic Research Pipeline

**Authors:** ThereminQ Ecosystem (Attributed to Repository Maintainer / twobombs)

## Abstract
`ThereminQ-Autoresearch` presents a fully local, multi-tier agentic architecture for automated code repository ingestion, problem decomposition, and self-improving technical distillation. Designed to run offline using localized quantized LLMs (e.g., `Qwen`, `Gemma`), the framework implements an advanced Map-Reduce orchestration pipeline that parses vast codebases, delegates tasks across parallel worker nodes, and consolidates findings into actionable technical specifications. Furthermore, it incorporates an experimental Recursive Self-Improvement (RSI) workflow inspired by Dream-RSI to dynamically evolve the exploration policies of multi-agent searches, minimizing human-in-the-loop dependencies in complex software engineering workflows.

---

## 1. Introduction
The advent of Large Language Models (LLMs) has catalyzed the development of autonomous agents capable of performing complex software engineering tasks. However, existing cloud-based frameworks often face limitations regarding context window constraints, data privacy, and latency. `ThereminQ-Autoresearch` addresses these challenges by providing a robust, entirely local orchestration pipeline tailored for monolithic code evaluation and project distillation.

The pipeline automates a rigorous 7-phase research lifecycle: repository ingestion, summarization, map-reduce decomposition of massive queries, parallel worker execution, semantic deduplication, automatic unit test generation, and final project distillation. The framework ensures context safety and high throughput through a rigid hierarchical structure of specialized "Apex," "Worker," and "Stitcher" nodes, mitigating the common context-collapse issues found in naive long-context deployments.

---

## 2. System Architecture (Methodology)

The system relies on localized execution of quantized GGUF models via high-throughput backends like `llama.cpp`. The architecture is split across two primary orchestration files located at the root of the repository.

### 2.1 Core Agentic Pipeline (`autoresearch-core-onefile.py`)
This script acts as the primary monolithic engine orchestrating the 7-phase Map-Reduce research lifecycle:
- **Phase 0 & 1 (Ingress & Generation):** Fetches and clones remote Git repositories locally, evaluating code files while respecting token limitations. An "Apex" node handles high-level conceptual generation.
- **Phase 2 (Distillation):** Extracts actionable tasks and explicit architectural requirements from raw, fluffy technical documents.
- **Phase 3 (Map-Reduce & Decomposition):** Shatters complex objectives into mutually exclusive, atomic tasks. These atomic components are distributed to parallel "Worker" nodes. Instead of traditional tree-reduction (which often truncates context), chunks are preserved individually.
- **Phase 4 (Per-Chunk Post-Processing):** Performs semantic deduplication across outputs, protecting vital artifacts (like code blocks) through placeholder injection during rewrites.
- **Phase 5 (Automated Unit Testing):** Automatically parses generated artifacts and requests unit tests from the worker pool. It executes the unit tests in an isolated, sandboxed Python virtual environment and records telemetry.
- **Phase 6 (Final Project Distillation):** Synthesizes agent reconciliation reports and test execution logs into a final, highly-structured Markdown file of actionable TO-DOs.

### 2.2 Recursive Self-Improvement Workflow (`autoresearch-rsi-onefile.py`)
This script augments the core pipeline by implementing Recursive Self-Improvement at the *exploration* layer.
- **Dream-RSI Implementation:** The pipeline drives agentic search through decision rounds, allowing the policy to dynamically shape parallel execution batches.
- **Offline Dreaming:** The policy-development agent (running on the Apex node) analyzes past execution traces and replay scores to mutate the policy code itself. Version iterations are evaluated against recorded discovery trees at zero additional agent-call cost.
- **Strict Scope Isolation:** Unlike traditional conversational agents, workers strictly obey assigned scopes. If a dependency on another agent's output is required, the agent documents the hand-off rather than hallucinating new context.
- **Fixed Evaluation:** Every agentic attempt is scored by a fixed heuristic combining deliverable quality, log depth, and unit test pass rates.

---

## 3. Related Work

The ThereminQ framework builds heavily upon foundational concepts established in recent literature on multi-agent collaboration and recursive agentic pipelines:

- **MetaGPT (Hong et al., arXiv:2308.00352v7):** Introduces a multi-agent framework that enforces standard operating procedures (SOPs) to streamline complex software development. `ThereminQ` similarly utilizes rigid roles and structured hand-offs, minimizing hallucination during complex task decomposition.
- **AutoGen (Wu et al., arXiv:2308.08155v2):** A framework enabling next-generation LLM applications via conversational agents. While AutoGen focuses heavily on chat-based collaboration, `ThereminQ` replaces conversational iteration with a strict, asynchronous Map-Reduce orchestration and mechanical reconciliation to prevent "scope creep" across agents.
- **Dream-RSI (Zheng et al., arXiv:2609.14858v1):** Demonstrates Recursive Self-Improvement through evolving environments. The `autoresearch-rsi-onefile.py` script directly implements this concept, maintaining a strict boundary between the exploration policy (which determines agent search paths) and the underlying domain tasks.

---

## 4. Installation & Usage

### 4.1 Prerequisites
- Python 3.10+
- `git`
- Local LLM inference server (e.g., `llama-server`) running specific GGUF models.

### 4.2 Configuration
Ensure your environment variables correctly point to your local LLM instances. Key configurable endpoints include:
```bash
export OPENAI_API_BASE="http://localhost:9931/v1"
export WORKER_ENDPOINTS="http://localhost:8030/v1,http://localhost:8031/v1"
export LLM_MODEL="Qwen3.8-Flash-Next-UD-IQ4_XS"
```

### 4.3 Execution
**Run Core Pipeline:**
Process a local repository or direct prompt:
```bash
python3 autoresearch-core-onefile.py -g https://github.com/user/target-repo.git
```

**Run RSI Agentic Workflow:**
Execute the Recursive Self-Improvement workflow over $N$ rounds:
```bash
python3 autoresearch-rsi-onefile.py -p "Develop a fully concurrent web crawler" -n 3
```
To run the offline "dreaming" process over an existing history:
```bash
python3 autoresearch-rsi-onefile.py -r --dream-only
```

---

## 5. Conclusion & Future Work
`ThereminQ-Autoresearch` validates the efficacy of deploying fully localized, hierarchical multi-agent architectures for rigorous codebase ingestion and software engineering workflows. By implementing chunk-preserving Map-Reduce strategies alongside Dream-RSI driven exploration policies, the system prevents context collapse and continuously evolves its task resolution capabilities.

Future work will focus on integrating more granular Model Context Protocol (MCP) integrations and enhancing the local Virtual Desktop Infrastructure (VDI) agents for multi-modal visual verifications of generated software front-ends.

---

## References
1. Hong, S., et al. (2023). MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework. *arXiv preprint arXiv:2308.00352v7*. URL: [http://arxiv.org/abs/2308.00352v7](http://arxiv.org/abs/2308.00352v7)
2. Wu, Q., et al. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. *arXiv preprint arXiv:2308.08155v2*. URL: [http://arxiv.org/abs/2308.08155v2](http://arxiv.org/abs/2308.08155v2)
3. Zheng, T., et al. (2026). Dream-RSI: Recursive Self-Improvement through Evolving Worlds. *arXiv preprint arXiv:2609.14858v1*. URL: [http://arxiv.org/abs/2609.14858v1](http://arxiv.org/abs/2609.14858v1)
