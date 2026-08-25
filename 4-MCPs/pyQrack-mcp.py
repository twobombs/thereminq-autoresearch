#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mcp[cli]>=2.1",
#     "pydantic>=2.6",
#     "pyqrack",
#     "numpy",
# ]
# ///
"""
qrack-mcp — a single-file MCP server exposing PyQrack as an agent-drivable simulator.

RUN
    uv run qrack_mcp.py                    # stdio, for a local MCP client
    uv run qrack_mcp.py --selftest         # no client needed; validates the stack
    uv run qrack_mcp.py --print-config     # emits the client config block
    uv run qrack_mcp.py --transport streamable-http --port 8848

    (without uv: pip install "mcp[cli]" pydantic pyqrack numpy && python qrack_mcp.py)

CONFIG — every policy knob is an env var, so deployment is the command line alone:
    QRACK_MCP_MAX_QUBITS          per-session cap            (default 32)
    QRACK_MCP_TOTAL_QUBITS        summed live budget         (default 64)
    QRACK_MCP_MAX_SESSIONS        concurrent simulators      (default 8)
    QRACK_MCP_TTL                 idle session reap, seconds (default 900)
    QRACK_MCP_DENSE_LIMIT         qubit ceiling for probes   (default 20)
    QRACK_MCP_WORKERS             job pool threads           (default 4)

DESIGN NOTES
    * Agents send a validated gate IR, never Python. There is no eval path.
    * Every PyQrack call is confined to the ADAPTER section — one place to fix
      when the library moves. Verify `adjs`/`adjt`/`force_m` against your build.
    * No tool ever returns a state vector. Reductions only; the context window
      is a harder constraint than the GPU.
    * numpy is optional: without it, entanglement_probe degrades to unavailable
      and everything else still runs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, model_validator

# SDK shim: the server class was renamed in the 2.x MCP SDK. Support both so
# this file runs against whatever is on the host.
try:
    from mcp.server.mcpserver import MCPServer as _MCPServerClass  # SDK >= 2.0
except ImportError:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP as _MCPServerClass  # SDK 1.x

# ===========================================================================
# CONFIG
# ===========================================================================


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


MAX_QUBITS_PER_SESSION = _env_int("QRACK_MCP_MAX_QUBITS", 32)
MAX_TOTAL_QUBITS = _env_int("QRACK_MCP_TOTAL_QUBITS", 64)
MAX_SESSIONS = _env_int("QRACK_MCP_MAX_SESSIONS", 8)
SESSION_TTL_SECONDS = _env_int("QRACK_MCP_TTL", 900)
DENSE_PROBE_QUBIT_LIMIT = _env_int("QRACK_MCP_DENSE_LIMIT", 20)
POOL_WORKERS = _env_int("QRACK_MCP_WORKERS", 4)
MAX_RETURNED_ROWS = 64

# ===========================================================================
# CIRCUIT IR — agents describe circuits; they never execute code
# ===========================================================================

OpName = Literal[
    "h", "x", "y", "z", "s", "sdg", "t", "tdg",
    "rx", "ry", "rz", "u",
    "cx", "cy", "cz", "ccx", "swap", "iswap",
    "reset",
]

# op -> (n_targets, n_controls, n_params)
_ARITY: dict[str, tuple[int, int, int]] = {
    "h": (1, 0, 0), "x": (1, 0, 0), "y": (1, 0, 0), "z": (1, 0, 0),
    "s": (1, 0, 0), "sdg": (1, 0, 0), "t": (1, 0, 0), "tdg": (1, 0, 0),
    "rx": (1, 0, 1), "ry": (1, 0, 1), "rz": (1, 0, 1), "u": (1, 0, 3),
    "cx": (1, 1, 0), "cy": (1, 1, 0), "cz": (1, 1, 0), "ccx": (1, 2, 0),
    "swap": (2, 0, 0), "iswap": (2, 0, 0), "reset": (1, 0, 0),
}

CLIFFORD_OPS = {"h", "x", "y", "z", "s", "sdg", "cx", "cy", "cz", "swap"}


class Gate(BaseModel):
    """One instruction. Arity is enforced here so a malformed agent emission
    fails as a readable validation error, not a PyQrack segfault."""
    op: OpName
    targets: list[int] = Field(min_length=1, max_length=2)
    controls: list[int] = Field(default_factory=list, max_length=2)
    params: list[float] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def _check_arity(self) -> "Gate":
        nt, nc, npar = _ARITY[self.op]
        got = (len(self.targets), len(self.controls), len(self.params))
        if got != (nt, nc, npar):
            raise ValueError(
                f"{self.op} takes {nt} target(s), {nc} control(s), {npar} param(s); "
                f"got {got[0]}/{got[1]}/{got[2]}"
            )
        if any(i < 0 for i in self.targets + self.controls):
            raise ValueError(f"{self.op}: qubit indices must be non-negative")
        if len(set(self.targets + self.controls)) != nt + nc:
            raise ValueError(f"{self.op}: qubit indices must be distinct")
        return self

    def width(self) -> int:
        return max(self.targets + self.controls) + 1


class BackendConfig(BaseModel):
    """QrackSimulator construction flags, named as in pyqrack 2.18.

    This knob set is what makes differential verification meaningful: the same
    circuit under different engines must agree, so disagreement is a bug.
    sdrp/ncrp are the approximation rounding parameters — set them and check
    fidelity() to see what the approximation cost you."""
    stabilizer_hybrid: bool = True
    binary_decision_tree: bool = False
    schmidt_decompose_multi: bool = False
    gpu: bool = True
    host_pointer: bool = False
    sparse: bool = False
    near_clifford_tableau_writer: bool = False
    sdrp: float | None = Field(default=None, ge=0.0, le=1.0,
                               description="Schmidt decomposition rounding parameter")
    ncrp: float | None = Field(default=None, ge=0.0, le=1.0,
                               description="near-Clifford rounding parameter")
    noise: float = Field(default=0.0, ge=0.0, le=1.0)

    def label(self) -> str:
        short = {"stabilizer_hybrid": "stab", "binary_decision_tree": "qbdd",
                 "schmidt_decompose_multi": "schmidt-multi", "gpu": "gpu",
                 "host_pointer": "host", "sparse": "sparse",
                 "near_clifford_tableau_writer": "nc-tableau"}
        on = [v for k, v in short.items() if getattr(self, k)]
        if self.sdrp is not None:
            on.append(f"sdrp={self.sdrp}")
        if self.ncrp is not None:
            on.append(f"ncrp={self.ncrp}")
        if self.noise:
            on.append(f"noise={self.noise}")
        return "+".join(on) or "dense"


# ===========================================================================
# ADAPTER — the only section that touches pyqrack
# ===========================================================================

try:
    from pyqrack import Pauli, QrackSimulator
    PYQRACK_AVAILABLE = True
except Exception as _e:  # noqa: BLE001 — import may fail on missing native libs
    QrackSimulator = Pauli = None  # type: ignore[assignment]
    PYQRACK_AVAILABLE = False
    PYQRACK_ERROR = str(_e)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False


@contextlib.contextmanager
def _native_stdout_to_stderr():
    """Qrack's native layer writes diagnostics ("No platforms found...") straight
    to fd 1. Under the stdio transport fd 1 IS the JSON-RPC channel, so an
    unguarded init corrupts the stream and the client drops the connection.
    Redirect at the file-descriptor level — Python-level redirection is not
    enough to catch a C++ printf."""
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def make_simulator(qubits: int, cfg: BackendConfig) -> Any:
    """Verified against pyqrack 2.18.3. If a bump breaks this, it breaks HERE
    and nowhere else in the file."""
    if not PYQRACK_AVAILABLE:
        raise RuntimeError(f"pyqrack unavailable on this host: {PYQRACK_ERROR}")
    with _native_stdout_to_stderr():
        sim = QrackSimulator(
            qubit_count=qubits,
            is_stabilizer_hybrid=cfg.stabilizer_hybrid,
            is_binary_decision_tree=cfg.binary_decision_tree,
            is_schmidt_decompose_multi=cfg.schmidt_decompose_multi,
            is_gpu=cfg.gpu,
            is_host_pointer=cfg.host_pointer,
            is_sparse=cfg.sparse,
            is_near_clifford_tableau_writer=cfg.near_clifford_tableau_writer,
            noise=cfg.noise,
        )
    if cfg.sdrp is not None:
        sim.set_sdrp(cfg.sdrp)
    if cfg.ncrp is not None:
        sim.set_ncrp(cfg.ncrp)
    return sim


PAULI_BASIS = {"I": Pauli.PauliI, "X": Pauli.PauliX,
               "Y": Pauli.PauliY, "Z": Pauli.PauliZ} if PYQRACK_AVAILABLE else {}


_APPLY: dict[str, Callable[[Any, Gate], None]] = {
    "h":     lambda s, g: s.h(g.targets[0]),
    "x":     lambda s, g: s.x(g.targets[0]),
    "y":     lambda s, g: s.y(g.targets[0]),
    "z":     lambda s, g: s.z(g.targets[0]),
    "s":     lambda s, g: s.s(g.targets[0]),
    "sdg":   lambda s, g: s.adjs(g.targets[0]),
    "t":     lambda s, g: s.t(g.targets[0]),
    "tdg":   lambda s, g: s.adjt(g.targets[0]),
    "rx":    lambda s, g: s.r(Pauli.PauliX, g.params[0], g.targets[0]),
    "ry":    lambda s, g: s.r(Pauli.PauliY, g.params[0], g.targets[0]),
    "rz":    lambda s, g: s.r(Pauli.PauliZ, g.params[0], g.targets[0]),
    "u":     lambda s, g: s.u(g.targets[0], *g.params),
    "cx":    lambda s, g: s.mcx(g.controls, g.targets[0]),
    "cy":    lambda s, g: s.mcy(g.controls, g.targets[0]),
    "cz":    lambda s, g: s.mcz(g.controls, g.targets[0]),
    "ccx":   lambda s, g: s.mcx(g.controls, g.targets[0]),
    "swap":  lambda s, g: s.swap(*g.targets),
    "iswap": lambda s, g: s.iswap(*g.targets),
}


def apply_gates(sim: Any, gates: list[Gate]) -> None:
    for g in gates:
        if g.op == "reset":
            # A true reset: measure, then correct. force_m() would instead
            # *impose* an outcome, which is a different operation entirely.
            if sim.m(g.targets[0]):
                sim.x(g.targets[0])
            continue
        _APPLY[g.op](sim, g)


def sample(sim: Any, qubits: list[int], shots: int) -> Counter:
    return Counter(sim.measure_shots(qubits, shots))


def dense_state(sim: Any, qubits: int) -> list[complex]:
    """Guarded. This never reaches the wire — callers reduce it to scalars."""
    if qubits > DENSE_PROBE_QUBIT_LIMIT:
        raise ValueError(
            f"dense probe refused above {DENSE_PROBE_QUBIT_LIMIT} qubits "
            f"(session has {qubits}); use measure() or compare_backends() instead"
        )
    return sim.out_ket()


# ===========================================================================
# SESSIONS — a live QrackSimulator is gigabytes and must outlive one call
# ===========================================================================


@dataclass
class Session:
    handle: str
    sim: Any
    qubits: int
    backend: BackendConfig
    created: float
    last_used: float
    gate_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._guard = threading.Lock()

    def open(self, qubits: int, cfg: BackendConfig) -> Session:
        if qubits < 1 or qubits > MAX_QUBITS_PER_SESSION:
            raise ValueError(f"qubits must be 1..{MAX_QUBITS_PER_SESSION}, got {qubits}")
        with self._guard:
            self._reap_locked()
            if len(self._sessions) >= MAX_SESSIONS:
                self._evict_lru_locked()
            live = sum(s.qubits for s in self._sessions.values())
            if live + qubits > MAX_TOTAL_QUBITS:
                raise ValueError(
                    f"qubit budget exhausted: {live} live + {qubits} requested > "
                    f"{MAX_TOTAL_QUBITS}. Close a session first."
                )
            now = time.time()
            s = Session(uuid.uuid4().hex[:12], make_simulator(qubits, cfg),
                        qubits, cfg, now, now)
            self._sessions[s.handle] = s
            return s

    def get(self, handle: str) -> Session:
        with self._guard:
            s = self._sessions.get(handle)
            if s is None:
                raise KeyError(f"no such session '{handle}' (expired or already closed)")
            s.last_used = time.time()
            return s

    def close(self, handle: str) -> bool:
        with self._guard:
            return self._sessions.pop(handle, None) is not None

    def list(self) -> list[dict[str, Any]]:
        with self._guard:
            now = time.time()
            return [
                {"handle": s.handle, "qubits": s.qubits, "backend": s.backend.label(),
                 "gates": s.gate_count, "idle_s": round(now - s.last_used, 1)}
                for s in self._sessions.values()
            ]

    def _reap_locked(self) -> None:
        cutoff = time.time() - SESSION_TTL_SECONDS
        for h in [h for h, s in self._sessions.items() if s.last_used < cutoff]:
            del self._sessions[h]

    def _evict_lru_locked(self) -> None:
        victim = min(self._sessions.values(), key=lambda s: s.last_used)
        del self._sessions[victim.handle]


REGISTRY = SessionRegistry()

# ===========================================================================
# JOBS — MCP calls time out long before an annealing sweep finishes
# ===========================================================================

POOL = ThreadPoolExecutor(max_workers=POOL_WORKERS, thread_name_prefix="qrack")
JOBS: dict[str, Future] = {}


def submit(fn: Callable[[], Any]) -> str:
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = POOL.submit(fn)
    return job_id


# ===========================================================================
# SERVER
# ===========================================================================

server = _MCPServerClass(
    "qrack",
    instructions=(
        "GPU-accelerated quantum circuit simulation via PyQrack. Call estimate() "
        "before opening anything wide, always close_session() when done, and treat "
        "a 'suspicious' flag from compare_backends() as a simulator bug to bisect."
    ),
)


@server.tool()
def open_session(qubits: int, backend: BackendConfig | None = None) -> dict:
    """Allocate a persistent simulator and return a handle. ALWAYS call
    close_session when finished — allocations are large and the qubit budget
    is shared across all live sessions."""
    s = REGISTRY.open(qubits, backend or BackendConfig())
    return {"handle": s.handle, "qubits": s.qubits, "backend": s.backend.label()}


@server.tool()
def close_session(handle: str) -> dict:
    """Release a simulator and free its memory."""
    return {"closed": handle, "existed": REGISTRY.close(handle)}


@server.tool()
def apply(handle: str, gates: list[Gate]) -> dict:
    """Apply gates in order to a session. Returns counts and timing only —
    never state."""
    s = REGISTRY.get(handle)
    for g in gates:
        if g.width() > s.qubits:
            raise ValueError(
                f"gate {g.op} touches qubit {g.width() - 1}; session has {s.qubits}"
            )
    t0 = time.perf_counter()
    with s.lock:
        apply_gates(s.sim, gates)
        s.gate_count += len(gates)
    return {"applied": len(gates), "total_gates": s.gate_count,
            "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 2)}


@server.tool()
def measure(handle: str, qubits: list[int] | None = None,
            shots: int = 1024, top_k: int = 16) -> dict:
    """Sample the register. Returns the top_k outcomes plus the residual tail
    mass, so a wide distribution cannot flood the context window."""
    s = REGISTRY.get(handle)
    qs = list(range(s.qubits)) if qubits is None else qubits
    if any(q >= s.qubits for q in qs):
        raise ValueError(f"qubit index out of range for a {s.qubits}-qubit session")
    with s.lock:
        hist = sample(s.sim, qs, shots)
    top = hist.most_common(min(top_k, MAX_RETURNED_ROWS))
    shown = sum(c for _, c in top)
    return {
        "shots": shots,
        "distinct_outcomes": len(hist),
        "top": [{"bitstring": format(b, f"0{len(qs)}b"), "count": c} for b, c in top],
        "tail_mass": round((shots - shown) / shots, 4),
    }


@server.tool()
def entanglement_probe(handle: str, cut_after_qubit: int) -> dict:
    """Von Neumann entropy and Schmidt rank across a bipartition. Small
    registers only — reduces the dense state locally and returns two numbers."""
    if not NUMPY_AVAILABLE:
        raise RuntimeError("numpy is required for entanglement_probe")
    s = REGISTRY.get(handle)
    if not 0 <= cut_after_qubit < s.qubits - 1:
        raise ValueError(f"cut_after_qubit must be 0..{s.qubits - 2}")
    with s.lock:
        ket = dense_state(s.sim, s.qubits)
    left = cut_after_qubit + 1
    psi = np.asarray(ket, dtype=complex).reshape(2 ** left, 2 ** (s.qubits - left))
    spectrum = np.linalg.svd(psi, compute_uv=False) ** 2
    p = spectrum[spectrum > 1e-12]
    return {"cut_after_qubit": cut_after_qubit,
            "entropy_bits": round(float(-(p * np.log2(p)).sum()), 6),
            "schmidt_rank": int(len(p))}


@server.tool()
def expectation(handle: str, pauli_string: str, qubits: list[int] | None = None) -> dict:
    """Pauli expectation value, e.g. pauli_string='ZZI'. Cheap, exact, and
    width-independent in output size — prefer this over sampling when you want
    an observable rather than a distribution."""
    s = REGISTRY.get(handle)
    basis = pauli_string.upper().strip()
    if any(c not in "IXYZ" for c in basis):
        raise ValueError("pauli_string must contain only I, X, Y, Z")
    qs = list(range(len(basis))) if qubits is None else qubits
    if len(qs) != len(basis):
        raise ValueError(f"pauli_string has {len(basis)} terms but {len(qs)} qubits given")
    if any(q >= s.qubits for q in qs):
        raise ValueError(f"qubit index out of range for a {s.qubits}-qubit session")
    with s.lock:
        val = s.sim.pauli_expectation(qs, [PAULI_BASIS[c] for c in basis])
    return {"pauli_string": basis, "qubits": qs, "expectation": round(float(val), 8)}


@server.tool()
def fidelity(handle: str) -> dict:
    """Accumulated unitary fidelity for this session. Meaningful when sdrp/ncrp
    approximation is enabled: it is the running estimate of what the rounding
    has cost. 1.0 means nothing was approximated away."""
    s = REGISTRY.get(handle)
    with s.lock:
        f = float(s.sim.get_unitary_fidelity())
    return {"handle": handle, "backend": s.backend.label(),
            "unitary_fidelity": round(f, 8), "gates_applied": s.gate_count}


@server.tool()
def estimate(qubits: int, gates: list[Gate],
             backend: BackendConfig | None = None) -> dict:
    """Pre-flight cost estimate. Allocates nothing — call this before
    open_session on anything wide."""
    cfg = backend or BackendConfig()
    non_clifford = sum(1 for g in gates if g.op not in CLIFFORD_OPS)
    two_qubit = sum(1 for g in gates if len(g.targets) + len(g.controls) > 1)
    dense_bytes = (2 ** qubits) * 16
    return {
        "qubits": qubits,
        "gate_count": len(gates),
        "two_qubit_gates": two_qubit,
        "non_clifford_gates": non_clifford,
        "worst_case_state_bytes": dense_bytes,
        "worst_case_gib": round(dense_bytes / 2 ** 30, 4),
        "fits_budget": qubits <= MAX_QUBITS_PER_SESSION,
        "recommended_backend": cfg.label(),
        "note": (
            "The dense figure is the ceiling, not the expectation: "
            "Clifford-dominant circuits stay near-free under stabilizer_hybrid, "
            "and Schmidt decomposition avoids the ceiling entirely while the "
            "circuit stays separable. Two-qubit gate count is the better proxy "
            "for where that separability breaks."
        ),
    }


@server.tool()
def compare_backends(qubits: int, gates: list[Gate],
                     backends: list[BackendConfig], shots: int = 4096) -> dict:
    """Differential verification. Runs one circuit across several Qrack
    configurations and reports pairwise total-variation distance against the
    sampling noise floor. Divergence flagged 'suspicious' is a simulator bug,
    not physics — bisect the circuit to find a minimal repro."""
    if not 2 <= len(backends) <= 4:
        raise ValueError("supply between 2 and 4 backend configurations")
    if qubits > MAX_QUBITS_PER_SESSION:
        raise ValueError(f"qubits exceeds cap of {MAX_QUBITS_PER_SESSION}")

    dists: list[tuple[str, dict[int, float]]] = []
    for cfg in backends:
        sim = make_simulator(qubits, cfg)
        try:
            apply_gates(sim, gates)
            hist = sample(sim, list(range(qubits)), shots)
        finally:
            del sim
        dists.append((cfg.label(), {k: v / shots for k, v in hist.items()}))

    noise_floor = 3.0 / math.sqrt(shots)
    pairs = []
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            (a_label, a), (b_label, b) = dists[i], dists[j]
            keys = set(a) | set(b)
            tvd = 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)
            worst = max(keys, key=lambda k: abs(a.get(k, 0.0) - b.get(k, 0.0)),
                        default=None)
            pairs.append({
                "a": a_label, "b": b_label,
                "tvd": round(tvd, 5),
                "suspicious": tvd > noise_floor,
                "worst_bitstring": None if worst is None else format(worst, f"0{qubits}b"),
            })
    return {"shots": shots, "sampling_noise_floor": round(noise_floor, 5),
            "pairs": pairs}


@server.tool()
def submit_run(qubits: int, gates: list[Gate], shots: int = 1024,
               backend: BackendConfig | None = None) -> dict:
    """Fire-and-forget execution for long circuits. Returns a job id; poll it
    with job_status. Use this instead of apply/measure when the run may exceed
    the client's tool timeout."""
    cfg = backend or BackendConfig()

    def work() -> dict:
        sim = make_simulator(qubits, cfg)
        t0 = time.perf_counter()
        apply_gates(sim, gates)
        hist = sample(sim, list(range(qubits)), shots)
        return {
            "wall_s": round(time.perf_counter() - t0, 3),
            "distinct_outcomes": len(hist),
            "top": [{"bitstring": format(b, f"0{qubits}b"), "count": c}
                    for b, c in hist.most_common(MAX_RETURNED_ROWS)],
        }

    return {"job_id": submit(work), "state": "submitted"}


@server.tool()
def job_status(job_id: str) -> dict:
    """Poll a submitted run. The result is returned once, then the job is
    dropped from the store."""
    fut = JOBS.get(job_id)
    if fut is None:
        raise KeyError(f"no such job '{job_id}' (unknown, or result already collected)")
    if not fut.done():
        return {"job_id": job_id, "state": "running"}
    JOBS.pop(job_id, None)
    if fut.exception() is not None:
        return {"job_id": job_id, "state": "failed", "error": str(fut.exception())}
    return {"job_id": job_id, "state": "done", "result": fut.result()}


@server.resource("qrack://sessions")
def live_sessions() -> str:
    """Live simulator inventory — lets the agent see what it is holding open."""
    rows = REGISTRY.list()
    return json.dumps({
        "sessions": rows,
        "qubit_budget": {"used": sum(r["qubits"] for r in rows),
                         "total": MAX_TOTAL_QUBITS},
        "session_slots": {"used": len(rows), "total": MAX_SESSIONS},
    }, indent=2)


@server.resource("qrack://capabilities")
def capabilities() -> str:
    """Backend availability and the policy limits currently in force."""
    return json.dumps({
        "pyqrack": PYQRACK_AVAILABLE or f"unavailable: {PYQRACK_ERROR}",
        "numpy": NUMPY_AVAILABLE,
        "supported_ops": sorted(_ARITY),
        "limits": {
            "max_qubits_per_session": MAX_QUBITS_PER_SESSION,
            "max_total_qubits": MAX_TOTAL_QUBITS,
            "max_sessions": MAX_SESSIONS,
            "session_ttl_seconds": SESSION_TTL_SECONDS,
            "dense_probe_qubit_limit": DENSE_PROBE_QUBIT_LIMIT,
        },
    }, indent=2)


# ===========================================================================
# CLI
# ===========================================================================

CLIENT_CONFIG = {
    "mcpServers": {
        "qrack": {
            "command": "uv",
            "args": ["run", os.path.abspath(__file__)],
            "env": {"QRACK_MCP_MAX_QUBITS": "32", "QRACK_MCP_TOTAL_QUBITS": "64"},
        }
    }
}


def selftest() -> int:
    """Exercise the stack end to end without an MCP client."""
    ok = True

    def check(label: str, fn: Callable[[], Any]) -> Any:
        nonlocal ok
        try:
            out = fn()
            print(f"  ok    {label}: {out}")
            return out
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  FAIL  {label}: {type(e).__name__}: {e}")
            return None

    print(f"pyqrack: {'available' if PYQRACK_AVAILABLE else 'MISSING'}   "
          f"numpy: {'available' if NUMPY_AVAILABLE else 'MISSING'}")

    print("\n[1] IR validation")
    try:
        Gate(op="ccx", targets=[0], controls=[1])
        print("  FAIL  arity check did not reject a malformed ccx")
        ok = False
    except Exception:
        print("  ok    malformed ccx rejected")
    check("well-formed ccx", lambda: Gate(op="ccx", targets=[2], controls=[0, 1]).op)

    print("\n[2] estimate (no allocation)")
    bell = [Gate(op="h", targets=[0]), Gate(op="cx", targets=[1], controls=[0])]
    check("2-qubit bell", lambda: estimate(2, bell)["worst_case_state_bytes"])
    check("30-qubit ceiling", lambda: estimate(30, bell)["worst_case_gib"])

    if not PYQRACK_AVAILABLE:
        print("\n[3-5] skipped — pyqrack not importable here")
        return 0 if ok else 1

    print("\n[3] session lifecycle")
    h = check("open", lambda: open_session(2)["handle"])
    if h:
        check("apply", lambda: apply(h, bell)["total_gates"])
        check("measure", lambda: measure(h, shots=512)["distinct_outcomes"])
        if NUMPY_AVAILABLE:
            check("entropy (expect ~1.0)",
                  lambda: entanglement_probe(h, 0)["entropy_bits"])
        check("expectation ZZ (expect ~1.0)",
              lambda: expectation(h, "ZZ")["expectation"])
        check("fidelity", lambda: fidelity(h)["unitary_fidelity"])
        check("close", lambda: close_session(h)["existed"])

    print("\n[4] differential verification")
    check("stab vs qbdd", lambda: compare_backends(
        3,
        [Gate(op="h", targets=[0]), Gate(op="cx", targets=[1], controls=[0]),
         Gate(op="t", targets=[2]), Gate(op="cx", targets=[2], controls=[1])],
        [BackendConfig(),
         BackendConfig(stabilizer_hybrid=False, binary_decision_tree=True),
         BackendConfig(gpu=False, sdrp=0.5)],
        shots=2048,
    )["pairs"])

    print("\n[5] async jobs")
    jid = check("submit", lambda: submit_run(3, bell, shots=256)["job_id"])
    if jid:
        for _ in range(50):
            st = job_status(jid)
            if st["state"] != "running":
                print(f"  ok    job {st['state']}")
                break
            time.sleep(0.1)

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="PyQrack MCP server (single file)")
    ap.add_argument("--transport", choices=["stdio", "sse", "streamable-http"],
                    default="stdio")
    ap.add_argument("--host", default="127.0.0.1", help="network transports only")
    ap.add_argument("--port", type=int, default=8848, help="network transports only")
    ap.add_argument("--selftest", action="store_true",
                    help="run built-in checks and exit")
    ap.add_argument("--print-config", action="store_true",
                    help="emit an MCP client config block and exit")
    args = ap.parse_args()

    if args.print_config:
        print(json.dumps(CLIENT_CONFIG, indent=2))
        return 0
    if args.selftest:
        return selftest()

    if not PYQRACK_AVAILABLE:
        # stderr only — stdout is the stdio transport and must stay clean
        print(f"warning: pyqrack unavailable ({PYQRACK_ERROR}); "
              f"simulation tools will error on call", file=sys.stderr)

    if args.transport == "stdio":
        server.run()
    else:
        server.run(transport=args.transport, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
