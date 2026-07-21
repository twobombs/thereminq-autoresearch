# ThereminQ-Autoresearch: A Hierarchical Multi-Agent Framework for Distributed, Context-Aware Automated Software Engineering

**Abstract:**
As large language models (LLMs) are increasingly integrated into automated software engineering, they face critical bottlenecks regarding context window degradation, catastrophic forgetting, and serial execution inefficiencies. We introduce `thereminq-autoresearch`, a distributed, multi-tier orchestration pipeline explicitly designed to maximize computational throughput on localized, self-hosted hardware. 

By implementing a hierarchical "pyramid" topology—segmenting agents into Apex (architectural synthesis), Orchestrator (map-reduce routing), and Worker (atomic execution) tiers—the framework overcomes single-LLM limitations. Operating across a strictly defined seven-phase pipeline, `thereminq-autoresearch` introduces robust methodologies for technical distillation, map-reduce chunking, rolling master stitching, and telemetry-driven autonomous testing. This report formalizes the architecture of the framework and grounds its approach in contemporary multi-agent and task decomposition literature.

note: this document has been generated

---

## 1. Introduction

The advent of conversational agents and large language models (LLMs) has catalyzed significant advancements in automated code generation, repository analysis, and technical documentation. However, current single-model workflows struggle when applied to entire repositories. Two primary limitations arise:
1. **Context Degradation:** As input length increases, LLMs suffer from the "Lost in the Middle" phenomenon, wherein they fail to retrieve or act upon information located in the center of long contexts (Liu et al., 2023).
2. **Serial Execution Bottlenecks:** Linearly processing massive codebases results in significant latency and underutilization of available local compute, hindering iterative research and development loops.

To address these challenges, we propose `thereminq-autoresearch`, a comprehensive Python-based orchestration framework designed for local hardware environments. By eschewing a monolithic approach in favor of a distributed multi-agent swarm, `thereminq-autoresearch` leverages specialized, distinct LLMs arranged in a pyramid topology. This system distributes heavy-hitting cognitive tasks to larger models, while dispatching high-volume, atomic operations to smaller, faster local models. The result is a high-throughput, context-aware pipeline capable of autonomous software research, generation, and validation.

---

## 2. Related Work

The architecture of `thereminq-autoresearch` synthesizes and advances multiple recent paradigms in the fields of autonomous agents, distributed prompt execution, and self-reflective code generation.

**Multi-Agent Collaborative Frameworks**
Recent literature has demonstrated that multi-agent collaboration significantly outperforms single-agent systems in complex software tasks. *MetaGPT* (Hong et al., 2023) established the efficacy of assigning distinct standard operating procedures (SOPs) and roles (e.g., product manager, engineer) to LLMs to streamline the development lifecycle. Similarly, *ChatDev* (Qian et al., 2023) utilizes a communicative agent approach, mimicking a virtual software company to mitigate hallucinations through cross-examination. For task routing, *HuggingGPT* (Shen et al., 2023) proved that an orchestrator LLM could effectively act as a controller to route tasks to specialized worker models. `thereminq-autoresearch` builds upon these foundations by codifying a rigid three-tier hierarchy tailored explicitly for hardware-constrained local swarms.

**Task Decomposition & Map-Reduce for LLMs**
To bypass context window limitations, task decomposition is essential. *Plan-and-Solve Prompting* (Wang et al., 2023) demonstrated that forcing an LLM to explicitly map out sub-tasks prior to execution dramatically improves zero-shot accuracy. In handling infinite or exceedingly large contexts, Map-Reduce strategies are frequently employed, where documents are chunked, processed in parallel (Map), and systematically synthesized (Reduce). `thereminq-autoresearch` codifies this explicitly in Phase 3 of its pipeline via its unified distributed orchestrator cluster.

**LLM-Driven Automated Code Generation and Testing**
Automated code generation is heavily reliant on iterative feedback loops. *Reflexion* (Shinn et al., 2023) introduced verbal reinforcement learning, wherein agents write code, execute tests, and use the resulting error logs as context to self-correct in subsequent iterations. This telemetry-driven feedback loop is the direct inspiration for Phase 5 (Automatic Unittests) in our framework, where worker models continuously refine generated functions against isolated runtime telemetry.

---

## 3. Methodology

The `thereminq-autoresearch` framework is driven by two core architectural pillars: the **Pyramid Topology** and the **Seven-Phase Execution Pipeline**.

### 3.1 The Pyramid Topology
To optimize local GPU/CPU compute tensors, the pipeline routes payloads through a tri-layered LLM hierarchy:
1. **Apex Layer (Architect/Director):** Utilizes the highest-parameter models available in the local swarm (e.g. Qwen-27B). This layer handles macro-level cognitive tasks such as interpreting the initial user prompt, validating final software architectures, and executing the final "master stitch" of disparate code modules.
2. **Orchestrator Layer (Auditors/Dispatchers):** Mid-tier models that act as routing nodes (e.g. specialized 8B Orchestrators). They manage the map-reduce control loops, perform semantic deduplication on generated chunks, and resolve boundary conflicts between isolated worker outputs.
3. **Worker Layer (Task Execution):** Highly quantized, fast, and lightweight models (e.g. Qwen-9B). These models operate in high-concurrency parallel environments, generating code snippets, translating logic, and running automated unit tests on atomic chunks.

### 3.2 The Seven-Phase Pipeline
The core script operationalizes the topology through a rigorous, sequential state machine spanning seven phases.

* **Phase 1: Git Repository Intake & Context Chunking**
  The repository is ingested and converted into an Abstract Syntax Tree (AST) representation. The framework calculates token density and fragments the codebase into discrete, context-safe chunks (typically bounded to 2,048 or 4,096 tokens).

* **Phase 2: Raw Content Generation**
  Worker models generate initial textual and code-based responses for each assigned chunk in a stateless, highly parallelized manner.

* **Phase 3: Fluff-to-Action Technical Distillation**
  Raw LLM outputs inherently contain conversational filler ("fluff"). Orchestrator models parse the Phase 1 outputs, stripping verbose boilerplate and distilling the content into dense, actionable pseudocode and logic steps.

* **Phase 4: Unified Distributed Orchestrator Cluster (Map-Reduce & Master Stitch)**
  * **Map-Reduce Execution:** The orchestrator dispatches the distilled chunks back to the Worker layer to translate into formal code.
  * **Atomic Task Decomposition:** Complex files are broken down into isolated functions or classes to prevent multi-objective drift.
  * **Rolling Master Stitch:** As Workers return compiled atomic functions, the Apex model sequentially integrates ("stitches") them into a contiguous master file. A rolling context window ensures the Apex model maintains semantic awareness of previously stitched functions, resolving namespace and dependency overlaps dynamically.

* **Phase 5: Distributed Parallel Post-Processing**
  The stitched master files are audited by Orchestrator models for semantic deduplication (removing overlapping logic generated by parallel workers) and header unification (ensuring consistent imports and logging frameworks).

* **Phase 6: Automatic Unittests & Telemetry Feedback**
  Drawing directly from the *Reflexion* methodology, this phase involves:
  1. **Generation:** Worker models write unit tests (e.g., `pytest` scripts) for the unified code.
  2. **Isolated Execution:** Tests are executed in a sandboxed runtime.
  3. **Telemetry Feedback:** Stack traces and test telemetry are piped back to the Worker models. If a test fails, the agent self-corrects the source code or test and re-executes until a passing state or maximum retry threshold is reached.

* **Phase 7: Project Task Distillation**
  Finally, the Apex model reviews the successfully compiled and tested artifacts, generating an agile-style Markdown report. This output synthesizes completed milestones, identifies unresolved technical debt, and outlines subsequent macro-tasks for the next orchestration loop.

---

## 4. Conclusion

The `thereminq-autoresearch` framework represents a robust, highly pragmatic approach to autonomous software engineering on self-hosted hardware. By constraining the unpredictable nature of LLMs within a rigid seven-phase pipeline and a pyramidal multi-agent topology, the framework effectively circumvents context degradation and serial processing bottlenecks. The explicit integration of Map-Reduce chunking, rolling master stitching, and Reflexion-style telemetry feedback ensures that generated code is not only contextually coherent but syntactically and logically robust.

---

## 5. References

1. Hong, S., Zhuge, M., Chen, J., Zheng, X., Cheng, Y., Zhang, C., ... & Wu, B. (2023). **MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework**. *arXiv preprint arXiv:2308.00352*. Available at: [https://arxiv.org/abs/2308.00352](https://arxiv.org/abs/2308.00352)
2. Qian, C., Cong, X., Yang, C., Chen, W., Su, Y., Xu, J., ... & Sun, M. (2023). **Communicative Agents for Software Development**. *arXiv preprint arXiv:2307.07924*. Available at: [https://arxiv.org/abs/2307.07924](https://arxiv.org/abs/2307.07924)
3. Shen, Y., Song, H., Su, J., Pan, T., Zhao, D., ... & Luo, J. (2023). **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face**. *arXiv preprint arXiv:2303.13761*. Available at: [https://arxiv.org/abs/2303.13761](https://arxiv.org/abs/2303.13761)
4. Wang, L., Xu, W., Lan, Y., Hu, Z., Lan, Y., Lee, R. K. W., & Lim, E. P. (2023). **Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models**. *arXiv preprint arXiv:2305.04091*. Available at: [https://arxiv.org/abs/2305.04091](https://arxiv.org/abs/2305.04091)
5. Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). **Reflexion: Language Agents with Verbal Reinforcement Learning**. *arXiv preprint arXiv:2303.11366*. Available at: [https://arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)
6. Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). **Lost in the Middle: How Language Models Use Long Contexts**. *arXiv preprint arXiv:2307.03172*. Available at: [https://arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172)



<img width="2816" height="1536" alt="Gemini_Generated_Image_qn4hgqqn4hgqqn4h" src="https://github.com/user-attachments/assets/9b0d020e-b5b3-42c9-bc32-aca01af8ac16" />

