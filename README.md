# ThereminQ-Autoresearch

**A fully local, multi-tier agentic research pipeline**

`ThereminQ-Autoresearch` ingests a prompt, a document or a whole Git repository. It decomposes the work into atomic assignments and runs them on a swarm of locally hosted, quantized LLMs (GGUF via `llama.cpp`, Vulkan backend). The output is a tested, distilled list of actionable tasks. Nothing leaves the machine: every tier is an OpenAI-compatible endpoint you host yourself.

The repository contains two self-contained pipeline scripts at its root. It also holds the infrastructure, helper tools and MCP servers around them, in numbered folders.

| Script | Architecture | Status |
|---|---|---|
| `autoresearch-rsi-onefile.py` | Two tiers (Apex + Workers), mechanical consolidation, recursive self-improvement of the exploration policy (Dream-RSI) | Current direction |
| `autoresearch-core-onefile.py` | Three tiers (Apex + Stitchers + Workers), model-driven map-reduce with per-chunk polishing | Original pipeline |

Detailed documentation lives in [`HOWTO-autoresearch-rsi-onefile.md`](HOWTO-autoresearch-rsi-onefile.md) (also as [PDF](HOWTO-autoresearch-rsi-onefile.pdf)) and [`HOWTO-autoresearch-core-onefile.md`](HOWTO-autoresearch-core-onefile.md). [`methodology_report.md`](methodology_report.md) is a generated write-up of the original three-tier design.

---

## Repository layout

| Path | Contents |
|---|---|
| `autoresearch-rsi-onefile.py` | RSI pipeline (see below) |
| `autoresearch-core-onefile.py` | Core three-tier pipeline (see below) |
| `docker-compose.yaml` | Example container stack: VDI workspace, orchestrator node and two 9B worker nodes (`twobombs/thereminq-tensors:jupyter`) |
| `0-build/` | `build_llamas.sh` builds `llama.cpp` with Vulkan and makes per-role copies. `fetch_llamas.sh` downloads the GGUF models. `GITS-SAC.sh` starts an in-container apex model on port 9931 |
| `1-runinfra/` | Launch scripts for the inference tiers: the full six-GPU swarm (`start-zerg-all.sh`), the worker swarm, VRAM-aware tier launcher (`launch-build-tiers.sh`), single-model launchers (2B/4B/9B/27B, Qwen-Next, GLM) |
| `2-startcore/` | The earlier split-file version of the core pipeline: generate → distill → agentic workflow → post-processing |
| `3-agilengine/` | Standalone unit-test generation, TODO distillation and a daily agentic "agile" report |
| `4-MCPs/` | MCP servers: local agile-state MCP, Atlassian/Jira ingress MCP, and `pyQrack-mcp.py` (PyQrack as an agent-drivable quantum simulator) |
| `5-viz/` | Turns orchestrator status output into a Stable Diffusion (A1111) prompt and image |
| `7-workdir/`, `8-workdir/` | Working directories; `8-workdir/testprompt.txt` is a sample prompt |
| `9-misc/` | Deep local research script, git compare-and-merge helper, local Discord bot |

---

## 1. RSI pipeline (`autoresearch-rsi-onefile.py`)

This script implements recursive self-improvement at the **exploration layer**, following Dream-RSI (Zheng et al., 2026). The models, the evaluator and the execution interfaces stay fixed. Only the exploration policy, the code that decides which agent assignments to run next, is improved from round to round.

### Tiers

| Tier | Default endpoint | Used for |
|---|---|---|
| **Apex** | `http://localhost:9931/v1` | Phase 1 generation, Phase 2 distillation, Phase 3 partitioning, offline policy development ("dreaming"), Phase 6 distillation. Single slot, strictly serial |
| **Workers** | `http://localhost:8030/v1` … `:8035/v1` | Every agent assignment, Phase 0 repository map-reduce, evaluator test generation |

There is **no stitcher tier**. No model merges the outputs of multiple agents. Consolidation is mechanical: a filesystem walk plus content hashing, written to `RECONCILE.md`.

### Phases

- **Phase 0 – Git intake** (`-g`, workers). Clones the repository and map-reduces its files into an intake document within size caps.
- **Phase 1 – Generation** (`-p`/`-f`, apex). Writes the raw document from the prompt.
- **Phase 2 – Distillation** (apex). Turns the raw document into actionable tasks and explicit requirements.
- **Phase 3 – Partition and rounds.** The apex splits the work into disjoint assignments (`comms/roster.json`). Each round:
  - **Online rollout:** the current policy selects batches of assignments for the workers.
  - **Scoring:** a fixed evaluator scores every attempt once.
  - **Dreaming:** new policy versions are written offline and replayed over the recorded discovery trees, at zero agent-call cost. The best version becomes the next round's policy.
- **Phase 5 – Test telemetry.** Per-round and cumulative execution reports (`reports/`).
- **Phase 6 – Project distillation** (apex). Combines intake, test telemetry and reconciliation into `DISTILLED_TASKS.md`.

**Scope isolation.** Every agent writes only into its own working directory. Cross-agent writes are quarantined as violations. Agents see each other through a shared comms map and record hand-offs there, instead of inventing another agent's output.

**Fixed evaluator.** Each attempt's score is final once assigned:

```
score = 0.6 × heuristic + 0.4 × unit-test pass rate
```

- **Heuristic:** rewards status, deliverables on disk, log depth and novelty, and penalizes ownership violations and truncation.
- **Pass rate:** comes from tests generated and executed for the attempt's deliverables.
- **Weights:** the mix is set by `EVAL_HEURISTIC_MIX` and the individual heuristic weights by `EVAL_W_*`.

**Test sandbox.** Phase 5 tests run in a per-run virtual environment (`tests/.venv`, `--system-site-packages`). Packages are installed from binary wheels only, with resource limits applied.

---

## 2. Core pipeline (`autoresearch-core-onefile.py`)

This is the original three-tier pipeline. It uses a dedicated stitcher tier for decomposition and editorial passes.

### Tiers

| Tier | Default endpoint | Default model |
|---|---|---|
| **Apex** (generation, distillation) | `http://localhost:8081/v1` | `Qwen3.6-27B-UD-IQ3_XXS.gguf` |
| **Stitchers** | `http://localhost:8070/v1`, `:8071/v1` | `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` |
| **Workers** | `http://localhost:8033/v1`, `:8034/v1` | `Qwen3.5-9B-IQ4_XS.gguf` |

### Phases

- **Phase 0 – Git intake / Phase 1 – Generation.** Same inputs as the RSI script.
- **Phase 2 – Distillation.** Raw document → actionable tasks.
- **Phase 3 – Map-reduce.** Decomposes the query into atomic tasks for the workers. Results are kept as individual chunks rather than tree-reduced, so no context is truncated.
- **Phase 4 – Per-chunk polish.**
  - Semantic deduplication by the stitchers, then two executive editor passes.
  - Code blocks are protected by placeholder injection (`[[PROTECTED_CODE_BLOCK###]]`) and recovered if a rewrite drops them.
  - Skip the whole phase with `--no-polish`, or only the executive passes with `--no-executive`.
- **Phase 5 – Automated unit tests.** Harvests artifacts from every chunk, requests tests from the workers and executes them. Records results in `execution_report.json`.
- **Phase 6 – Project distillation.** Writes `DISTILLED_TASKS.md`.

> **Note:** Unlike the RSI script, the core pipeline's Phase 5 installs model-generated `requirements` into the **host** Python (`pip install --break-system-packages`) rather than a virtual environment. Run it inside a container.

---

## 3. Requirements

- Linux (POSIX process groups and `ulimit` are used by the test runner and policy sandbox)
- Python ≥ 3.9 with `openai` and `requests`; `pytest` for Python tests
- `git` on `PATH` (for `-g`)
- `gcc`/`g++` and `bash` (Phase 5 C/C++ and shell tests)
- One or more `llama-server` instances (or any OpenAI-compatible server) serving GGUF models. See `0-build/` and `1-runinfra/`.

```bash
python3 -m pip install openai requests pytest
```

### Building the inference stack

```bash
cd 0-build
./build_llamas.sh     # llama.cpp with -DGGML_VULKAN=1, plus per-role copies
./fetch_llamas.sh     # apex, worker and alternative GGUF models
```

Then start the tiers with the scripts in `1-runinfra/`. Their ports must match the endpoints the pipeline script expects (next section).

---

## 4. Configuration

All endpoints and models can be overridden through environment variables. The two scripts have **different defaults**.

| Variable | RSI default | Core default |
|---|---|---|
| `OPENAI_API_BASE` (apex) | `http://localhost:9931/v1` | `http://localhost:8081/v1` |
| `LLM_MODEL` (apex) | `Qwen3.8-Flash-Next-UD-IQ4_XS` | `Qwen3.6-27B-UD-IQ3_XXS.gguf` |
| `DISTILLER_URL` / `DISTILLER_MODEL` | same as apex | same as apex |
| `WORKER_ENDPOINTS` | `:8030` … `:8035` | *not read — edit `WORKER_ENDPOINTS` in the source* (`:8033`, `:8034`) |
| `WORKER_MODEL` | `Qwen3.8-9B-Q4_K_M.gguf` | `Qwen3.5-9B-IQ4_XS.gguf` |
| `STITCHER_ENDPOINTS` / `STITCHER_MODEL` | — | `:8070`, `:8071` / `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` |

**Context budgets.** Every internal character budget is derived from the server geometry. Keep these variables in sync with the `-c` and `-np` flags of your `llama-server` processes:

- `APEX_SERVER_CTX` / `APEX_SERVER_NP`
- `WORKER_SERVER_CTX` / `WORKER_SERVER_NP`
- `STITCH_SERVER_CTX` / `STITCH_SERVER_NP` (core only)

Both scripts verify these against the servers at startup.

**Private Git hosts.** Phase 0 rejects git URLs that resolve to private or metadata-service addresses. For the RSI script, set `GIT_ALLOW_PRIVATE_HOSTS=1` to allow a LAN host such as a self-hosted Gitea.

Example for the RSI script:

```bash
export OPENAI_API_BASE="http://localhost:9931/v1"
export LLM_MODEL="Qwen3.8-Flash-Next-UD-IQ4_XS"
export WORKER_ENDPOINTS="http://localhost:8030/v1,http://localhost:8031/v1"
```

The full list of variables (evaluator weights, RSI budgets, sandbox limits) is in the HOWTO documents.

---

## 5. Usage

Both scripts share these arguments:

| Argument | Meaning |
|---|---|
| `-p PROMPT` \| `-f FILE` \| `-g GIT_URL` | Input (mutually exclusive) |
| `--focus TEXT` | Analysis focus for git intake (with `-g`) |
| `--git-path PATH` | Restrict intake to a file or folder in the repository (with `-g`) |
| `-d DIR` | Output base directory (default `run_data`) |
| `-c CATEGORY` | Category folder (default `projects`) |
| `-r` | Resume the most recent run in `DIR/CATEGORY` |
| `--iterate` | Re-run / refine Phase 6 on an existing `DISTILLED_TASKS.md` |

RSI-only arguments:

| Argument | Meaning |
|---|---|
| `-n ROUNDS` | Recursive rounds (default 1, max 12) |
| `--budget N` | Agent calls per round (default scales with the roster) |
| `--no-dream` | Skip policy improvement: fixed-exploration control |
| `--dream-only` | Run no agents; dream over the recorded trees and write the next policy (requires `-r`) |
| `--semantic-guidance` | Inject directional guidance into agent prompts (off by default) |

Core-only arguments:

| Argument | Meaning |
|---|---|
| `--no-polish` | Skip Phase 4 and test the raw synthesis chunks |
| `--no-executive` | Keep per-segment dedup but skip the executive editor passes |

Examples:

```bash
# Core pipeline over a repository
python3 autoresearch-core-onefile.py -g https://github.com/user/target-repo.git

# RSI pipeline, three recursive rounds
python3 autoresearch-rsi-onefile.py -p "Develop a fully concurrent web crawler" -n 3

# Offline dreaming over the most recent RSI run (only the apex tier needs to be up)
python3 autoresearch-rsi-onefile.py -r --dream-only
```

Each run writes to `DIR/CATEGORY/run_<timestamp>_<id>/`, which contains:

- the raw and distilled documents
- `RUN_MANIFEST.md`
- `reports/execution_report.json`
- `DISTILLED_TASKS.md`

RSI runs also contain:

- `work/` (per-agent deliverables)
- `comms/` (roster and comms map)
- `trees/` (discovery trees)
- `policy/` (policy versions)
- `dream/` (replay scores)

---

## 6. Containers (`docker-compose.yaml`)

The compose file defines four services on the `twobombs/thereminq-tensors:jupyter` image:

- a VDI workspace (noVNC on port 6080)
- an orchestrator node (`nvidia_Orchestrator-8B`, port 8080)
- two Qwen3.5-9B worker nodes (ports 8034 and 8035)

The LLM services build `llama.cpp` and fetch models on first start, then run `llama-server` on the Vulkan device.

Its ports do not match either script's defaults. When running a pipeline against this stack, set `OPENAI_API_BASE` and `WORKER_ENDPOINTS` (RSI) accordingly, or edit the core script's worker list.

---

## 7. Related work

- **MetaGPT** (Hong et al., 2023): multi-agent collaboration under standard operating procedures. ThereminQ likewise uses fixed roles and structured hand-offs.
- **AutoGen** (Wu et al., 2023): conversational multi-agent applications. ThereminQ replaces conversational iteration with asynchronous, scope-isolated assignments and mechanical reconciliation.
- **Dream-RSI** (Zheng et al., 2026): recursive self-improvement of the exploration policy by replaying recorded discovery histories. `autoresearch-rsi-onefile.py` implements this method. Its off-policy support probes are an extension that is not part of the paper.

## References

1. Hong, S., et al. (2023). MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework. *arXiv:2308.00352*. <http://arxiv.org/abs/2308.00352>
2. Wu, Q., et al. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. *arXiv:2308.08155*. <http://arxiv.org/abs/2308.08155>
3. Zheng, T., et al. (2026). Dream-RSI: Recursive Self-Improvement through Evolving Worlds. *arXiv:2609.14858*. <http://arxiv.org/abs/2609.14858>

## License

See [LICENSE](LICENSE).
