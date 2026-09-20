#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastapi>=0.110",
#   "uvicorn[standard]>=0.29",
#   "httpx[socks]>=0.27",
#   "pyyaml>=6.0",
# ]
# ///
r"""
gemini-openai-gateway
=====================

OpenAI-compatible API (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`) that routes to the Google Gemini API, with configurable upstream endpoint, auth style, outbound proxy and inbound path routing.

## Quick start

```bash
pip install "fastapi>=0.110" "uvicorn[standard]>=0.29" "httpx[socks]>=0.27" "pyyaml>=6.0"
# or, with uv (reads the inline metadata above):  uv run openai2gemini.py
export GEMINI_API_KEY=...
python openai2gemini.py --print-config > config.yaml   # optional; env vars alone also work
python openai2gemini.py -c config.yaml
```

By default the gateway listens on 127.0.0.1 only. To bind a public interface (including `0.0.0.0` inside a container), set `client_api_keys` or `passthrough_client_key`; otherwise it refuses to start unless `allow_public_without_auth: true` is set explicitly.

Point any OpenAI client at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9931/v1", api_key="sk-local-change-me")
client.chat.completions.create(model="gemini-3.8-flash", messages=[{"role": "user", "content": "hi"}])
```

## Upstream modes

| mode | what happens | use when |
|---|---|---|
| `native` (default) | Translates to `generateContent` / `streamGenerateContent?alt=sse` / `batchEmbedContents` | You need thinking control, thought-signature handling for Gemini 3 tool calls, schema sanitising, safety settings, native tools, or a custom/Vertex-shaped endpoint |
| `openai` | Forwards the body to `<base>/v1beta/openai/...`, rewriting only model name and auth | You want minimal translation and Google's own compat layer is enough |

## What `native` mode translates

- **Messages**: `system`/`developer` -> `systemInstruction`; `assistant` -> `model`; `tool` results -> `functionResponse` (parallel results merged into one turn, names recovered from `tool_call_id`).
- **Content parts**: text, `image_url` (data URLs inline; http(s) fetched through the outbound proxy and inlined; `gs://` and Gemini file URIs as `fileData`), `input_audio`, `file`.
- **Tools**: OpenAI function tools -> `functionDeclarations`. JSON Schema is reduced to Gemini's subset (`$ref`/`$defs` inlined with recursive refs cut off, `allOf` merged, `oneOf`->`anyOf`, `["x","null"]`->`nullable`, unsupported keys/formats dropped, empty-object params omitted). Set `schema_mode: jsonschema` to send the raw schema as `parametersJsonSchema` instead. `tool_choice` -> `functionCallingConfig` (AUTO/NONE/ANY + allowed names).
- **Thought signatures**: signatures on returned `functionCall` parts are cached against the generated `tool_call` id and re-attached when the client sends the history back. For Gemini 3 history without a signature (e.g. from another model), `fallback_thought_signature` is inserted.
- **Generation params**: temperature, top_p, top_k, max_tokens / max_completion_tokens, stop, n, penalties, seed, `response_format` (json_object / json_schema).
- **Model routing**: `force_model` (default `gemini-3.8-flash`) overrides every chat request after `model_aliases` are applied; responses report the model that actually served the request. Set `force_model: ""` to honour client model names. Embeddings are never forced.
- **Model quirks**: `model_quirks` holds per-model-prefix rules (longest prefix wins per key): generation params to drop (Gemini 3.8 Flash rejects temperature/top_p/top_k/penalties/candidateCount), a thinking-level table, an output-token cap, stripping a trailing prefilled assistant turn, and sending `id` on `functionCall`/`functionResponse` parts.
- **Reasoning**: `reasoning_effort` -> `thinkingLevel` for models matching `thinking_level_prefixes`, otherwise `thinkingBudget` from `reasoning_budgets`, raised to the per-model floor in `reasoning_budget_min` (gemini-2.5-pro cannot turn thinking off). With `include_thoughts` (or per-request `include_reasoning: true`) thought summaries come back as `reasoning_content`.
- **Responses**: finish reasons mapped (`MAX_TOKENS`->`length`, safety family->`content_filter`, calls->`tool_calls`), usage incl. reasoning and cached tokens, `stream_options.include_usage` honoured.
- **Escape hatch**: anything Gemini-specific can go in a `gemini` object in the request body (`extra_body={"gemini": {...}}` in the OpenAI SDK): `tools` (e.g. `[{"googleSearch": {}}]`), `toolConfig`, `generationConfig`, `thinkingConfig`, `safetySettings`, `cachedContent`; for embeddings, `taskType`.

## Routing and proxy settings

**Outbound (gateway -> Gemini)**

- `gemini_base_url` + `native_path_template` / `models_path_template` / `gemini_openai_path` - point at Google, a Vertex-shaped endpoint, an internal API gateway, or a mock.
- `auth_mode` (`header` / `bearer` / `query` / `none`), `auth_header`, `extra_upstream_headers` (e.g. a `Host` override when `gemini_base_url` is an IP or internal load balancer). `none` sends no credentials in either upstream mode.
- `outbound_proxy`, or per-scheme `proxy_http` / `proxy_https`; `http://`, `https://` and `socks5://` are supported, with credentials in the URL (redacted in logs).
- `no_proxy`: hosts or httpx patterns (`*.corp.lan`) that bypass the proxy.
- `trust_env: true` reads the standard `HTTP(S)_PROXY` / `ALL_PROXY` / `NO_PROXY` environment variables through the same transports as explicit proxy settings (TLS settings and limits apply; note that httpx only applies `connect_retries` to direct connections, not to proxied ones).
- `verify_tls: /path/ca.pem` for TLS-intercepting corporate proxies.
- Upstream redirects are not followed, so the API key is never forwarded to another host.

**Inbound (client -> gateway)**

- `route_prefixes`: expose the API at several paths, e.g. `["/v1", "/openai/v1", "/"]`.
- `root_path`: set when a reverse proxy strips a prefix.
- `forwarded_allow_ips`: which proxies may set `X-Forwarded-*`.
- Streaming responses send `X-Accel-Buffering: no`. With nginx, also set `proxy_buffering off;` and a long `proxy_read_timeout`.

```nginx
location /gemini/ {
    proxy_pass http://127.0.0.1:9931/;   # strips /gemini -> set root_path: /gemini
    proxy_http_version 1.1;
    proxy_buffering off;
    proxy_read_timeout 600s;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

## Environment variables

Every config key maps to `GW_<KEY>`. Lists can be comma-separated or JSON, and dicts are JSON. Examples:

```bash
GW_OUTBOUND_PROXY=socks5://127.0.0.1:1080
GW_NO_PROXY=localhost,*.lan
GW_ROUTE_PREFIXES=/v1,/openai/v1
GW_MODEL_ALIASES='{"gpt-4o":"gemini-2.5-pro"}'
GW_CLIENT_API_KEYS=sk-a,sk-b
```

## Limits and notes

- The thought-signature cache is in-memory. If you run several replicas, use sticky sessions or rely on the fallback signature.
- Embeddings use `batchEmbedContents` (Gemini API). Vertex's `:predict` embedding shape is not translated, so use `upstream_mode: openai` or a separate instance for that.
- `fetch_remote_media` makes the gateway download URLs supplied by clients. Downloads are streamed with a hard size cap, redirects are followed manually (max 5) and, with `media_block_private_hosts`, every hop must resolve to public addresses only. DNS rebinding can still defeat a resolve-then-connect check, so if untrusted clients can reach the gateway, also restrict egress at the network level or disable the feature.
- Gemini returns no token counts for embeddings, so `usage` there is 0.
- `GET /health` reports mode, upstream and whether a key is configured.
- **Rate limits and overload**: on an upstream 429 or 503 (`retry_statuses`) the gateway waits the delay Gemini asks for (`RetryInfo.retryDelay`, then `Retry-After`, then the "retry in Ns" text). If none is given, it uses the fixed per-status delay in `retry_status_delays` (503 "high demand": 10s), otherwise exponential backoff from `retry_default_delay`. Then it retries, up to `retry_max_attempts` times. A delay longer than `retry_max_wait` (e.g. a daily quota) is not waited out: the 429 goes straight back to the client with a `Retry-After` header. Your client's own request timeout must exceed the wait, or it will give up first. Set `retry_max_attempts: 0` to disable.

CLI:
    python openai2gemini.py [-c config.yaml]      run the server
    python openai2gemini.py --print-config        print the annotated example config
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hmac
import ipaddress
import json
import logging
import mimetypes
import os
import re
import socket
import struct
import time
import urllib.parse
import urllib.request
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any

import httpx
import yaml
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

# =========================================================================== #
# SETTINGS - every hardcoded host, port, URL, path, limit and default lives   #
# here. DEFAULTS and the --print-config example below are built from these.   #
# At runtime, config file / GW_* environment variables override them.        #
# =========================================================================== #

# -- Inbound: where the gateway listens --------------------------------------
LISTEN_HOST = "127.0.0.1"                  # loopback only; see ALLOW_PUBLIC_WITHOUT_AUTH
LISTEN_PORT = 9931
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
ROUTE_PREFIXES = ["/v1"]                   # every prefix gets the full OpenAI surface
ROOT_PATH = ""                             # set when a reverse proxy strips a path prefix
FORWARDED_ALLOW_IPS = "127.0.0.1"          # peers allowed to set X-Forwarded-* headers
HEALTH_PATH = "/health"
LOOPBACK_HOSTNAMES = {"localhost"}         # names (besides loopback IPs) treated as local
ALLOW_PUBLIC_WITHOUT_AUTH = False

# -- Upstream: Gemini API endpoints and auth ----------------------------------
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
GEMINI_API_VERSION = "v1beta"
GEMINI_NATIVE_PATH_TEMPLATE = "/{version}/models/{model}:{method}"
GEMINI_MODELS_PATH_TEMPLATE = "/{version}/models"
GEMINI_OPENAI_PATH = "/v1beta/openai"
GEMINI_AUTH_MODE = "header"                # header | bearer | query | none
GEMINI_AUTH_HEADER = "x-goog-api-key"
GEMINI_FILES_HOST = "generativelanguage.googleapis.com"  # its /files/ URIs pass through as fileData

# -- Environment variable names -----------------------------------------------
ENV_PREFIX = "GW_"                         # config key foo_bar -> GW_FOO_BAR
CONFIG_PATH_ENV = "GW_CONFIG"
API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")   # checked in order

# -- Outbound network ---------------------------------------------------------
TIMEOUT_CONNECT = 10.0                     # seconds
TIMEOUT_READ = 300.0                       # seconds (also used for write)
CONNECT_RETRIES = 2                        # direct connections only (httpx limitation)
MAX_CONNECTIONS = 100
KEEPALIVE_FRACTION = 5                     # keep-alive pool = MAX_CONNECTIONS // this

# -- Models -------------------------------------------------------------------
DEFAULT_MODEL = "gemini-3.5-flash-lite"
FORCE_MODEL = "gemini-3.5-flash-lite"          # every chat request uses this model ("" = honour the request)
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
EXAMPLE_MODEL_ALIASES = {                  # shown in --print-config; runtime default is none
    "gpt-4o": "gemini-3.1-pro",
    "gpt-4o-mini": "gemini-3.5-flash",
    "text-embedding-3-small": "gemini-embedding-001",
}

# -- Reasoning / thinking -----------------------------------------------------
REASONING_BUDGETS = {"none": 0, "minimal": 512, "low": 1024, "medium": 8192, "high": 24576}
REASONING_BUDGET_MIN = {"gemini-2.5-pro": 128}          # per-model-prefix floor
THINKING_LEVEL_PREFIXES = ["gemini-3"]
THINKING_LEVELS = {"none": "low", "minimal": "low", "low": "low", "medium": "high", "high": "high"}

# -- Per-model quirks ---------------------------------------------------------
# Keyed by model-name prefix. For each key, the longest matching prefix wins.
#   drop_generation_params: generationConfig keys removed before sending
#   thinking_levels:        overrides THINKING_LEVELS for these models
#   max_output_tokens:      clamp for maxOutputTokens
#   strip_prefill:          drop a trailing assistant (model) turn without tool calls
#   function_call_ids:      send `id` on functionCall / functionResponse parts
MODEL_QUIRKS = {
    "gemini-3": {
        "drop_generation_params": ["candidateCount"],
    },
    "gemini-3.8-flash": {
        "drop_generation_params": ["temperature", "topP", "topK", "candidateCount",
                                   "presencePenalty", "frequencyPenalty"],
        "thinking_levels": {"none": "low", "minimal": "low", "low": "low",
                            "medium": "medium", "high": "high"},   # no "minimal" on 3.8 Flash
        "max_output_tokens": 65536,
        "strip_prefill": True,
        "function_call_ids": True,
    },
}
THOUGHT_SIGNATURE_PREFIXES = ["gemini-3"]
FALLBACK_THOUGHT_SIGNATURE = "skip_thought_signature_validator"
SIGNATURE_CACHE_SIZE = 20000               # tool_call ids remembered for signature round-trips

# -- Remote media -------------------------------------------------------------
FETCH_REMOTE_MEDIA = True
MAX_REMOTE_MEDIA_BYTES = 20 * 1024 * 1024
MEDIA_BLOCK_PRIVATE_HOSTS = True
MAX_MEDIA_REDIRECTS = 5
FALLBACK_IMAGE_MIME = "image/jpeg"
FALLBACK_FILE_MIME = "application/pdf"

# -- Protocol limits ----------------------------------------------------------
EMBED_BATCH_SIZE = 100                     # Gemini batchEmbedContents limit
MAX_STOP_SEQUENCES = 5                     # Gemini stopSequences limit
MODELS_PAGE_SIZE = 1000
MODELS_MAX_PAGES = 10
SCHEMA_MAX_DEPTH = 40                      # recursion cap for JSON Schema sanitising
ERROR_BODY_MAX_CHARS = 4000                # upstream error text passed back to clients
DEBUG_PAYLOAD_MAX_CHARS = 4000             # payload preview in DEBUG logs

# -- Upstream rate limiting / retries ----------------------------------------
RETRY_STATUSES = [429, 503]                # upstream statuses that are waited out and retried
RETRY_STATUS_DELAYS = {"503": 10.0}         # fixed wait per status when Gemini gives no delay
RETRY_MAX_ATTEMPTS = 5                     # retries per upstream call (0 = off)
RETRY_MAX_WAIT = 120.0                     # seconds; a longer requested delay is returned to the client instead
RETRY_DEFAULT_DELAY = 5.0                  # seconds, doubled per attempt when Gemini gives no delay
RETRY_PADDING = 0.5                        # seconds added to the requested delay

# =========================================================================== #

log = logging.getLogger("gemini-openai-gateway")


def _j(v: Any) -> str:
    """Render a value as JSON (valid YAML flow syntax) for the example config."""
    return json.dumps(v)


# Annotated example configuration (print with --print-config), built from SETTINGS.
EXAMPLE_CONFIG = f"""# gemini-openai-gateway configuration
# Every key can also be set via environment variable {ENV_PREFIX}<KEY_UPPERCASE>
# (lists: comma-separated or JSON; dicts: JSON). {API_KEY_ENV_VARS[0]} is also honoured.

# -- Inbound: what OpenAI clients talk to --------------------------------------
# Loopback by default. Binding anything else (e.g. 0.0.0.0 in a container)
# requires client_api_keys or passthrough_client_key, or the explicit override
# allow_public_without_auth: true.
listen_host: {LISTEN_HOST}
listen_port: {LISTEN_PORT}
log_level: {LOG_LEVEL}
allow_public_without_auth: {_j(ALLOW_PUBLIC_WITHOUT_AUTH)}

# Mount the OpenAI surface at several prefixes, e.g. for clients that hardcode
# "/openai/v1" or expect no "/v1" at all ("/" = root).
route_prefixes: {_j(ROUTE_PREFIXES)}

# Behind a reverse proxy that strips a path prefix (nginx location /gemini/ ->
# proxy_pass http://gw:{LISTEN_PORT}/;) set root_path so generated URLs stay correct.
root_path: {_j(ROOT_PATH)}

# Which peers may set X-Forwarded-For/Proto (comma list, or "*").
forwarded_allow_ips: {_j(FORWARDED_ALLOW_IPS)}

cors_origins: []

# Keys your clients must send as "Authorization: Bearer <key>". Empty = open.
client_api_keys: []
#  - sk-local-change-me

# If true and client_api_keys is empty, every client must send its own Gemini
# API key as the Bearer token (multi-tenant). The gateway's own key is never
# used in this mode.
passthrough_client_key: false

# -- Upstream: Gemini ---------------------------------------------------------
# native: translate to generateContent/streamGenerateContent (recommended)
# openai: forward to Gemini's own OpenAI-compatible endpoint
upstream_mode: native

gemini_base_url: {GEMINI_BASE_URL}
gemini_api_version: {GEMINI_API_VERSION}
native_path_template: {_j(GEMINI_NATIVE_PATH_TEMPLATE)}
models_path_template: {_j(GEMINI_MODELS_PATH_TEMPLATE)}
gemini_openai_path: {GEMINI_OPENAI_PATH}
gemini_api_key: ""            # prefer the {API_KEY_ENV_VARS[0]} env var

# How the key is sent upstream: header | bearer | query | none
# (query-mode keys are redacted from logs)
auth_mode: {GEMINI_AUTH_MODE}
auth_header: {GEMINI_AUTH_HEADER}
extra_upstream_headers: {{}}
#  Host: generativelanguage.googleapis.com   # when base_url points at an IP / internal LB

# Example - route through a Vertex-AI-shaped gateway with bearer tokens:
# gemini_base_url: https://europe-west4-aiplatform.googleapis.com
# gemini_api_version: v1
# native_path_template: "/{{version}}/projects/MY_PROJECT/locations/europe-west4/publishers/google/models/{{model}}:{{method}}"
# auth_mode: bearer

# -- Outbound network / proxy reroute -----------------------------------------
outbound_proxy: ""            # http://user:pass@proxy:3128  or  socks5://127.0.0.1:1080
proxy_http: ""                # per-scheme overrides
proxy_https: ""
no_proxy: []                  # ["localhost", "*.internal.lan", "10.0.0.5"]
trust_env: false              # true = use HTTP(S)_PROXY / ALL_PROXY / NO_PROXY env vars
verify_tls: true              # or a path to a CA bundle (for TLS-intercepting proxies)
timeout_connect: {TIMEOUT_CONNECT}
timeout_read: {TIMEOUT_READ}
connect_retries: {CONNECT_RETRIES}
max_connections: {MAX_CONNECTIONS}

# -- Models -------------------------------------------------------------------
default_model: {DEFAULT_MODEL}
default_embedding_model: {DEFAULT_EMBEDDING_MODEL}
force_model: {_j(FORCE_MODEL)}
model_aliases: {_j(EXAMPLE_MODEL_ALIASES)}

# -- Translation behaviour (native mode) --------------------------------------
schema_mode: sanitize         # sanitize -> Schema subset | jsonschema -> parametersJsonSchema
include_thoughts: false       # thought summaries returned as message.reasoning_content
reasoning_budgets: {_j(REASONING_BUDGETS)}
# Per-model minimum thinking budget (longest matching prefix wins).
reasoning_budget_min: {_j(REASONING_BUDGET_MIN)}
thinking_level_prefixes: {_j(THINKING_LEVEL_PREFIXES)}
thinking_levels: {_j(THINKING_LEVELS)}
# Per-model-prefix overrides (longest prefix wins per key); see MODEL_QUIRKS in the script.
model_quirks: {_j(MODEL_QUIRKS)}
thought_signature_prefixes: {_j(THOUGHT_SIGNATURE_PREFIXES)}
fallback_thought_signature: {FALLBACK_THOUGHT_SIGNATURE}
safety_settings: []
#  - {{category: HARM_CATEGORY_DANGEROUS_CONTENT, threshold: BLOCK_ONLY_HIGH}}

# -- Upstream rate limits ----------------------------------------------------
# On these statuses, wait the delay Gemini asks for and retry. A requested delay
# above retry_max_wait (e.g. an exhausted daily quota) is returned immediately.
retry_statuses: {_j(RETRY_STATUSES)}
# Fixed wait (seconds) per status when the response carries no delay; statuses
# not listed back off exponentially from retry_default_delay.
retry_status_delays: {_j(RETRY_STATUS_DELAYS)}
retry_max_attempts: {RETRY_MAX_ATTEMPTS}   # 0 = never retry
retry_max_wait: {RETRY_MAX_WAIT}
retry_default_delay: {RETRY_DEFAULT_DELAY}

# -- Remote media (http(s) image_url) -----------------------------------------
fetch_remote_media: {_j(FETCH_REMOTE_MEDIA)}
max_remote_media_bytes: {MAX_REMOTE_MEDIA_BYTES}
media_block_private_hosts: {_j(MEDIA_BLOCK_PRIVATE_HOSTS)}   # refuse URLs resolving to loopback/private/link-local
"""

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

DEFAULTS: dict[str, Any] = {
    # inbound (what clients talk to)
    "listen_host": LISTEN_HOST,
    "listen_port": LISTEN_PORT,
    "log_level": LOG_LEVEL,
    "allow_public_without_auth": ALLOW_PUBLIC_WITHOUT_AUTH,
    "root_path": ROOT_PATH,
    "route_prefixes": list(ROUTE_PREFIXES),
    "forwarded_allow_ips": FORWARDED_ALLOW_IPS,
    "cors_origins": [],
    "client_api_keys": [],              # empty = no client auth
    "passthrough_client_key": False,    # clients must bring their own Gemini key (only when client_api_keys is empty)

    # upstream (Gemini)
    "upstream_mode": "native",          # native | openai
    "gemini_base_url": GEMINI_BASE_URL,
    "gemini_api_version": GEMINI_API_VERSION,
    "native_path_template": GEMINI_NATIVE_PATH_TEMPLATE,
    "models_path_template": GEMINI_MODELS_PATH_TEMPLATE,
    "gemini_openai_path": GEMINI_OPENAI_PATH,
    "gemini_api_key": "",
    "auth_mode": GEMINI_AUTH_MODE,
    "auth_header": GEMINI_AUTH_HEADER,
    "extra_upstream_headers": {},

    # outbound proxy / network
    "outbound_proxy": "",               # applies to http and https unless overridden
    "proxy_http": "",
    "proxy_https": "",
    "no_proxy": [],                     # hosts or httpx patterns, e.g. "internal.lan", "*.corp"
    "trust_env": False,                 # read HTTP(S)_PROXY / ALL_PROXY / NO_PROXY env vars
    "verify_tls": True,                 # true | false | /path/to/ca-bundle.pem
    "timeout_connect": TIMEOUT_CONNECT,
    "timeout_read": TIMEOUT_READ,
    "connect_retries": CONNECT_RETRIES,
    "max_connections": MAX_CONNECTIONS,

    # models
    "default_model": DEFAULT_MODEL,
    "default_embedding_model": DEFAULT_EMBEDDING_MODEL,
    "force_model": FORCE_MODEL,         # if set, every chat request uses this model (after aliases)
    "model_aliases": {},                # e.g. {"gpt-4o": "gemini-2.5-pro"}

    # translation behaviour
    "schema_mode": "sanitize",          # sanitize (-> parameters) | jsonschema (-> parametersJsonSchema)
    "include_thoughts": False,          # return thought summaries as reasoning_content
    "reasoning_budgets": dict(REASONING_BUDGETS),
    "reasoning_budget_min": dict(REASONING_BUDGET_MIN),
    "thinking_level_prefixes": list(THINKING_LEVEL_PREFIXES),
    "thinking_levels": dict(THINKING_LEVELS),
    "model_quirks": json.loads(json.dumps(MODEL_QUIRKS)),
    "thought_signature_prefixes": list(THOUGHT_SIGNATURE_PREFIXES),
    "fallback_thought_signature": FALLBACK_THOUGHT_SIGNATURE,
    "safety_settings": [],

    # remote media
    "fetch_remote_media": FETCH_REMOTE_MEDIA,
    "max_remote_media_bytes": MAX_REMOTE_MEDIA_BYTES,
    "media_block_private_hosts": MEDIA_BLOCK_PRIVATE_HOSTS,

    # upstream rate limits
    "retry_statuses": list(RETRY_STATUSES),
    "retry_status_delays": dict(RETRY_STATUS_DELAYS),
    "retry_max_attempts": RETRY_MAX_ATTEMPTS,
    "retry_max_wait": RETRY_MAX_WAIT,
    "retry_default_delay": RETRY_DEFAULT_DELAY,
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class Config(dict):
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e


def _coerce(key: str, raw: str, default: Any) -> Any:
    if key == "verify_tls":
        low = raw.strip().lower()
        return True if low in _TRUE else False if low in _FALSE else raw
    if isinstance(default, bool):
        return raw.strip().lower() in _TRUE
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, list):
        raw = raw.strip()
        if raw.startswith("["):
            return json.loads(raw)
        return [x.strip() for x in raw.split(",") if x.strip()]
    if isinstance(default, dict):
        return json.loads(raw)
    return raw


def load_config(path: str | None = None) -> Config:
    cfg = dict(DEFAULTS)
    path = path or os.environ.get(CONFIG_PATH_ENV)
    if path and not os.path.exists(path):
        log.warning("config file %s not found, using defaults + environment", path)
        path = None
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for k in data:
            if k not in DEFAULTS:
                log.warning("unknown config key ignored: %s", k)
        cfg.update({k: v for k, v in data.items() if k in DEFAULTS})
    for k, d in DEFAULTS.items():
        env = os.environ.get(ENV_PREFIX + k.upper())
        if env is not None:
            cfg[k] = _coerce(k, env, d)
    if not cfg["gemini_api_key"]:
        cfg["gemini_api_key"] = next((os.environ[v] for v in API_KEY_ENV_VARS if os.environ.get(v)), "")

    prefixes = []
    for p in cfg["route_prefixes"] or ROUTE_PREFIXES:
        p = "/" + p.strip("/") if p.strip("/") else ""
        if p not in prefixes:
            prefixes.append(p)
    cfg["route_prefixes"] = prefixes
    cfg["root_path"] = ("/" + cfg["root_path"].strip("/")) if cfg["root_path"].strip("/") else ""
    cfg["gemini_base_url"] = cfg["gemini_base_url"].rstrip("/")
    cfg["gemini_openai_path"] = "/" + cfg["gemini_openai_path"].strip("/")
    cfg["client_api_keys"] = [str(k) for k in cfg["client_api_keys"] or []]
    if cfg["upstream_mode"] not in ("native", "openai"):
        raise ValueError("upstream_mode must be 'native' or 'openai'")
    if cfg["auth_mode"] not in ("header", "bearer", "query", "none"):
        raise ValueError("auth_mode must be header | bearer | query | none")
    return Config(cfg)


def _is_loopback_host(host: str) -> bool:
    if host.strip().lower() in LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def check_bind_safety(cfg: Config) -> None:
    """Refuse to expose an unauthenticated gateway (and the server's Gemini key) beyond loopback."""
    if _is_loopback_host(cfg.listen_host):
        return
    if cfg.client_api_keys or cfg.passthrough_client_key or cfg.allow_public_without_auth:
        return
    raise SystemExit(
        f"refusing to listen on {cfg.listen_host!r} without client authentication: anyone who can "
        "reach this port could spend your Gemini key and use fetch_remote_media against your "
        "network. Set client_api_keys (or passthrough_client_key), bind a loopback address, or set "
        "allow_public_without_auth: true if you really mean it.")


class _RedactKeyFilter(logging.Filter):
    """Mask `key=<secret>` query parameters in any log record (httpx logs full request URLs)."""
    _re = re.compile(r"([?&]key=)[^&\s\"'>]+")

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if "key=" in msg:
            record.msg, record.args = self._re.sub(r"\1***", msg), None
        return True


def _env_proxy_settings() -> tuple[str, str, list[str]]:
    env = urllib.request.getproxies()  # reads HTTP(S)_PROXY / ALL_PROXY / NO_PROXY, any case
    p_all = env.get("all", "")
    no = [h.strip() for h in env.get("no", "").split(",") if h.strip()]
    return env.get("http") or p_all, env.get("https") or p_all, no


def _env_no_proxy_pattern(entry: str) -> str | None:
    """Translate a curl-style NO_PROXY entry into an httpx mount pattern."""
    h = entry.strip().lower()
    if "/" in h:
        log.warning("NO_PROXY entry %r (CIDR) is not supported; ignored", entry)
        return None
    if h.startswith("."):
        return f"all://*{h}"
    if h in LOOPBACK_HOSTNAMES:
        return f"all://{h}"
    try:
        ipaddress.ip_address(h.strip("[]"))
        return f"all://{h}"
    except ValueError:
        return f"all://*{h}"  # conventional NO_PROXY semantics: the domain and its subdomains


def build_http_client(cfg: Config) -> httpx.AsyncClient:
    timeout = httpx.Timeout(connect=cfg.timeout_connect, read=cfg.timeout_read,
                            write=cfg.timeout_read, pool=cfg.timeout_connect)
    limits = httpx.Limits(max_connections=cfg.max_connections,
                          max_keepalive_connections=max(1, cfg.max_connections // KEEPALIVE_FRACTION))
    p_http = cfg.proxy_http or cfg.outbound_proxy
    p_https = cfg.proxy_https or cfg.outbound_proxy
    no_patterns = [h if "://" in h else f"all://{h}" for h in cfg.no_proxy]

    if cfg.trust_env and not (p_http or p_https):
        # Read the proxy environment ourselves so our transports (TLS settings,
        # limits, and retries for direct/no_proxy hosts) are used here too.
        # httpx does not pass `retries` to proxy pools, in any mode.
        p_http, p_https, env_no = _env_proxy_settings()
        if "*" in env_no:
            p_http = p_https = ""
        no_patterns += [p for p in map(_env_no_proxy_pattern, env_no) if p and p != "all://*"]

    def transport(proxy: str | None = None) -> httpx.AsyncHTTPTransport:
        return httpx.AsyncHTTPTransport(proxy=proxy or None, verify=cfg.verify_tls,
                                        retries=cfg.connect_retries, limits=limits)

    mounts: dict[str, httpx.AsyncBaseTransport | None] = {}
    if p_http:
        mounts["http://"] = transport(p_http)
    if p_https:
        mounts["https://"] = transport(p_https)
    if mounts:
        for pattern in no_patterns:
            mounts[pattern] = None  # None -> fall back to the direct transport
        log.info("outbound proxy: http=%s https=%s no_proxy=%s",
                 _redact_url(p_http), _redact_url(p_https), no_patterns)
    # Redirects are never followed automatically: httpx strips Authorization on a
    # cross-origin redirect but would forward x-goog-api-key. Media fetches follow
    # redirects manually, re-checking each hop.
    return httpx.AsyncClient(transport=transport(), mounts=mounts, timeout=timeout,
                             trust_env=False, follow_redirects=False)


def _redact_url(u: str) -> str:
    if not u:
        return "-"
    p = urllib.parse.urlsplit(u)
    if p.password:
        netloc = f"{p.username}:***@{p.hostname}" + (f":{p.port}" if p.port else "")
        return urllib.parse.urlunsplit((p.scheme, netloc, p.path, p.query, p.fragment))
    return u


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

class LRU:
    """Tiny LRU used to round-trip Gemini thought signatures via tool_call ids."""

    def __init__(self, size: int = SIGNATURE_CACHE_SIZE):
        self.size, self.d = size, OrderedDict()

    def put(self, k: str, v: str) -> None:
        self.d[k] = v
        self.d.move_to_end(k)
        while len(self.d) > self.size:
            self.d.popitem(last=False)

    def get(self, k: str | None) -> str | None:
        if k is None or k not in self.d:
            return None
        self.d.move_to_end(k)
        return self.d[k]


SIGNATURES = LRU()

_ERR_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
              404: "not_found_error", 429: "rate_limit_error"}

NO_BUFFER_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


def oai_error(status: int, message: str, etype: str | None = None, code: Any = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {
        "message": message, "type": etype or _ERR_TYPES.get(status, "api_error"),
        "param": None, "code": code}})


_RETRY_TEXT = re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)


def retry_delay(body: bytes, headers: httpx.Headers | dict | None = None) -> float | None:
    """Seconds Gemini asks us to wait: RetryInfo.retryDelay, then Retry-After, then message text."""
    try:
        j = json.loads(body)
        j = j[0] if isinstance(j, list) and j else j
        for d in (j.get("error") or {}).get("details") or []:
            if str(d.get("@type", "")).endswith("google.rpc.RetryInfo") and d.get("retryDelay"):
                return float(str(d["retryDelay"]).rstrip("s"))
    except Exception:
        pass
    ra = (headers or {}).get("retry-after")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    m = _RETRY_TEXT.search(body.decode("utf-8", "replace"))
    return float(m.group(1)) if m else None


async def send_upstream(cfg: Config, http: httpx.AsyncClient, method: str, url: str, *,
                        stream: bool = False, **kw: Any) -> httpx.Response:
    """Send one upstream request, waiting out rate limits (see retry_* settings).
    Returns the final response; on a non-retried error its body has already been read."""
    attempt = 0
    while True:
        resp = await http.send(http.build_request(method, url, **kw), stream=stream)
        if resp.status_code not in cfg.retry_statuses or attempt >= cfg.retry_max_attempts:
            return resp
        body = await resp.aread()
        asked = retry_delay(body, resp.headers)
        fixed = (cfg.retry_status_delays or {}).get(str(resp.status_code))
        if asked is not None:
            delay = asked
        elif fixed is not None:
            delay = float(fixed)
        else:
            delay = cfg.retry_default_delay * (2 ** attempt)
        if delay > cfg.retry_max_wait:
            log.warning("upstream %s: requested wait %.0fs exceeds retry_max_wait (%.0fs); returning it",
                        resp.status_code, delay, cfg.retry_max_wait)
            return resp
        await resp.aclose()
        attempt += 1
        log.warning("upstream %s (%s); waiting %.1fs, then retry %d/%d", resp.status_code,
                    "rate limit" if resp.status_code == 429 else "unavailable",
                    delay, attempt, cfg.retry_max_attempts)
        await asyncio.sleep(delay + RETRY_PADDING)


def upstream_error(status: int, body: bytes) -> JSONResponse:
    resp = _upstream_error(status, body)
    if status in (429, 503):
        delay = retry_delay(body)
        if delay is None:
            delay = RETRY_STATUS_DELAYS.get(str(status))
        if delay is not None:
            resp.headers["Retry-After"] = str(max(1, int(delay + 0.999)))
    return resp


def _upstream_error(status: int, body: bytes) -> JSONResponse:
    msg, code = body.decode("utf-8", "replace")[:ERROR_BODY_MAX_CHARS], None
    try:
        j = json.loads(body)
        j = j[0] if isinstance(j, list) and j else j
        err = j.get("error", {}) if isinstance(j, dict) else {}
        msg = err.get("message") or msg
        code = err.get("status") or err.get("code")
    except Exception:
        pass
    return oai_error(status, f"Gemini upstream error: {msg}", code=code)


def resolve_model(cfg: Config, requested: str | None, embedding: bool = False) -> str:
    m = requested or (cfg.default_embedding_model if embedding else cfg.default_model)
    m = cfg.model_aliases.get(m, m)
    if cfg.force_model and not embedding and m != cfg.force_model:
        log.debug("force_model: %s -> %s", m, cfg.force_model)
        m = cfg.force_model
    return m.removeprefix("models/")


def native_url(cfg: Config, model: str, method: str) -> str:
    return cfg.gemini_base_url + cfg.native_path_template.format(
        version=cfg.gemini_api_version, model=model, method=method)


def upstream_auth(cfg: Config, key: str | None, openai_style: bool = False) -> tuple[dict, dict]:
    headers, params = dict(cfg.extra_upstream_headers), {}
    if not key or cfg.auth_mode == "none":
        return headers, params
    if openai_style or cfg.auth_mode == "bearer":
        headers["Authorization"] = f"Bearer {key}"
    elif cfg.auth_mode == "header":
        headers[cfg.auth_header] = key
    elif cfg.auth_mode == "query":
        params["key"] = key
    return headers, params


def _key_matches(token: str, keys: list[str]) -> bool:
    tb, ok = token.encode(), False
    for k in keys:  # no early exit: compare against every key in constant time
        ok |= hmac.compare_digest(tb, k.encode())
    return ok


def authorize(cfg: Config, request: Request) -> tuple[str | None, JSONResponse | None]:
    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else request.headers.get("x-api-key")
    if cfg.client_api_keys:
        if token and _key_matches(token, cfg.client_api_keys):
            return cfg.gemini_api_key, None
        return None, oai_error(401, "Invalid or missing API key for this gateway.")
    if cfg.passthrough_client_key:
        if token:
            return token, None
        return None, oai_error(401, "Missing API key: send your own Gemini API key as the Bearer token.")
    return cfg.gemini_api_key, None


def resolve_upstream_key(cfg: Config, request: Request) -> tuple[str | None, JSONResponse | None]:
    """authorize(), then refuse to send a keyless request upstream (Google answers those
    with a confusing 403 about 'unregistered callers')."""
    key, err = authorize(cfg, request)
    if err:
        return None, err
    if not key and cfg.auth_mode != "none":
        return None, oai_error(
            503, "Gateway has no Gemini API key configured: set " + " or ".join(API_KEY_ENV_VARS)
            + " (or gemini_api_key in the config file) before starting the gateway, "
            "or use auth_mode: none for an upstream that needs no key.",
            "api_error", "gateway_key_missing")
    return key, None


def _starts(model: str, prefixes: list[str]) -> bool:
    return any(model.startswith(p) for p in prefixes)


def _prefix_lookup(model: str, table: dict[str, Any]) -> Any:
    best: tuple[str, Any] | None = None
    for p, v in (table or {}).items():
        if model.startswith(p) and (best is None or len(p) > len(best[0])):
            best = (p, v)
    return best[1] if best else None


def quirk(cfg: Config, model: str, key: str, default: Any = None) -> Any:
    """Longest model-prefix match in model_quirks that defines `key`."""
    best: tuple[str, Any] | None = None
    for p, q in (cfg.model_quirks or {}).items():
        if model.startswith(p) and isinstance(q, dict) and key in q and (best is None or len(p) > len(best[0])):
            best = (p, q[key])
    return best[1] if best else default


# --------------------------------------------------------------------------- #
# OpenAI -> Gemini translation                                                #
# --------------------------------------------------------------------------- #

def _parse_data_url(url: str) -> tuple[str, str]:
    header, _, data = url.partition(",")
    mime = header[5:].split(";")[0] or "application/octet-stream"
    if ";base64" in header:
        return mime, data
    return mime, base64.b64encode(urllib.parse.unquote_to_bytes(data)).decode()


def _proxy_configured(cfg: Config) -> bool:
    return bool(cfg.outbound_proxy or cfg.proxy_http or cfg.proxy_https or cfg.trust_env)


async def _check_public_host(cfg: Config, url: str) -> None:
    host = urllib.parse.urlsplit(url).hostname
    if not host:
        raise ValueError(f"media URL has no host: {url[:64]}")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        if _proxy_configured(cfg):
            return  # only the proxy can resolve it; the proxy's own egress rules apply
        raise ValueError(f"could not resolve media host {host}: {e}") from e
    for info in infos:
        ip = ipaddress.ip_address(str(info[4][0]).split("%")[0])
        if not ip.is_global:
            raise ValueError(f"media host {host} resolves to non-public address {ip}; refused")


async def fetch_media(cfg: Config, url: str, http: httpx.AsyncClient) -> tuple[bytes, str]:
    """GET a client-supplied URL with a streamed size cap and per-hop host checks."""
    for _ in range(MAX_MEDIA_REDIRECTS + 1):
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"unsupported media URL scheme: {url[:64]}")
        if cfg.media_block_private_hosts:
            await _check_public_host(cfg, url)
        try:
            async with http.stream("GET", url) as r:
                if r.is_redirect and "location" in r.headers:
                    url = urllib.parse.urljoin(str(r.url), r.headers["location"])
                    continue
                r.raise_for_status()
                declared = r.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > cfg.max_remote_media_bytes:
                    raise ValueError(f"media at {url} exceeds max_remote_media_bytes")
                buf = bytearray()
                async for chunk in r.aiter_bytes():
                    buf += chunk
                    if len(buf) > cfg.max_remote_media_bytes:
                        raise ValueError(f"media at {url} exceeds max_remote_media_bytes")
                return bytes(buf), r.headers.get("content-type", "").split(";")[0].strip()
        except httpx.HTTPError as e:
            raise ValueError(f"could not fetch media {url}: {e}") from e
    raise ValueError(f"too many redirects fetching media (max {MAX_MEDIA_REDIRECTS})")


async def url_to_part(cfg: Config, url: str, http: httpx.AsyncClient) -> dict:
    guess = mimetypes.guess_type(url.split("?")[0])[0]
    if url.startswith("data:"):
        mime, data = _parse_data_url(url)
        return {"inlineData": {"mimeType": mime, "data": data}}
    if url.startswith("gs://") or (urllib.parse.urlsplit(url).hostname == GEMINI_FILES_HOST
                                   and "/files/" in urllib.parse.urlsplit(url).path):
        return {"fileData": {"fileUri": url, "mimeType": guess or "application/octet-stream"}}
    if url.startswith(("http://", "https://")):
        if not cfg.fetch_remote_media:
            return {"fileData": {"fileUri": url, "mimeType": guess or FALLBACK_IMAGE_MIME}}
        content, ctype = await fetch_media(cfg, url, http)
        mime = ctype or guess or FALLBACK_IMAGE_MIME
        return {"inlineData": {"mimeType": mime, "data": base64.b64encode(content).decode()}}
    raise ValueError(f"unsupported media URL: {url[:64]}")


async def convert_content(cfg: Config, content: Any, http: httpx.AsyncClient) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    parts: list[dict] = []
    for p in content:
        if isinstance(p, str):
            parts.append({"text": p})
            continue
        t = p.get("type")
        if t in ("text", "input_text"):
            if p.get("text"):
                parts.append({"text": p["text"]})
        elif t == "image_url":
            iu = p.get("image_url")
            parts.append(await url_to_part(cfg, iu["url"] if isinstance(iu, dict) else iu, http))
        elif t == "input_audio":
            a = p.get("input_audio", {})
            parts.append({"inlineData": {"mimeType": f"audio/{a.get('format', 'wav')}", "data": a.get("data", "")}})
        elif t == "file":
            f = p.get("file", {})
            if f.get("file_data"):
                fd = f["file_data"]
                if fd.startswith("data:"):
                    mime, data = _parse_data_url(fd)
                else:
                    mime = mimetypes.guess_type(f.get("filename", ""))[0] or FALLBACK_FILE_MIME
                    data = fd
                parts.append({"inlineData": {"mimeType": mime, "data": data}})
            elif f.get("file_id"):
                parts.append({"fileData": {"fileUri": f["file_id"],
                                           "mimeType": mimetypes.guess_type(f.get("filename", ""))[0] or FALLBACK_FILE_MIME}})
        else:
            log.warning("ignoring unsupported content part type: %s", t)
    return parts


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def _parse_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        v = json.loads(raw or "{}")
        return v if isinstance(v, dict) else {"value": v}
    except (TypeError, json.JSONDecodeError):
        return {"_raw": raw}


async def build_contents(cfg: Config, messages: list[dict], model: str,
                         http: httpx.AsyncClient) -> tuple[list[dict], list[dict]]:
    system_parts: list[dict] = []
    contents: list[dict] = []
    id_to_name: dict[str, str] = {}
    send_ids = bool(quirk(cfg, model, "function_call_ids", False))

    for m in messages:
        role = m.get("role")
        if role in ("system", "developer"):
            system_parts += await convert_content(cfg, m.get("content"), http)
            continue
        if role == "user":
            g_role, parts = "user", await convert_content(cfg, m.get("content"), http)
        elif role == "assistant":
            g_role, parts = "model", await convert_content(cfg, m.get("content"), http)
            calls = list(m.get("tool_calls") or [])
            if m.get("function_call"):  # legacy
                calls.append({"id": None, "function": m["function_call"]})
            for tc in calls:
                fn = tc.get("function", {})
                if tc.get("id"):
                    id_to_name[tc["id"]] = fn.get("name")
                part = {"functionCall": {"name": fn.get("name"), "args": _parse_args(fn.get("arguments"))}}
                if send_ids and tc.get("id"):
                    part["functionCall"]["id"] = tc["id"]
                sig = SIGNATURES.get(tc.get("id"))
                if sig:
                    part["thoughtSignature"] = sig
                parts.append(part)
        elif role in ("tool", "function"):
            name = m.get("name") or id_to_name.get(m.get("tool_call_id")) or "unknown_function"
            raw = _text_of(m.get("content"))
            try:
                resp = json.loads(raw)
                if not isinstance(resp, dict):
                    resp = {"result": resp}
            except (TypeError, json.JSONDecodeError):
                resp = {"result": raw}
            fr: dict[str, Any] = {"name": name, "response": resp}
            if send_ids and m.get("tool_call_id"):
                fr["id"] = m["tool_call_id"]
            g_role, parts = "user", [{"functionResponse": fr}]
        else:
            log.warning("ignoring message with unknown role: %s", role)
            continue

        if not parts:
            continue
        # Gemini wants parallel function responses in a single turn, and tolerates
        # merged consecutive same-role turns better than repeated ones.
        if contents and contents[-1]["role"] == g_role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": g_role, "parts": parts})

    # Gemini 3 requires a thought signature on the first functionCall of each step.
    # History produced elsewhere (other models, other gateways) won't have one.
    if cfg.fallback_thought_signature and _starts(model, cfg.thought_signature_prefixes):
        for c in contents:
            if c["role"] != "model":
                continue
            fcs = [p for p in c["parts"] if "functionCall" in p]
            if fcs and not any("thoughtSignature" in p for p in fcs):
                fcs[0]["thoughtSignature"] = cfg.fallback_thought_signature

    # Some models reject a prefilled (trailing) model turn. Tool-call turns are never
    # trailing in valid history, but keep them just in case.
    if (contents and contents[-1]["role"] == "model" and quirk(cfg, model, "strip_prefill", False)
            and not any("functionCall" in p for p in contents[-1]["parts"])):
        log.info("dropping trailing assistant prefill: not supported by %s", model)
        contents.pop()

    if not contents and system_parts:  # system-only prompt
        contents, system_parts = [{"role": "user", "parts": system_parts}], []
    return system_parts, contents


_SCHEMA_KEYS = {"type", "format", "title", "description", "nullable", "enum", "maxItems", "minItems",
                "properties", "required", "minProperties", "maxProperties", "minLength", "maxLength",
                "pattern", "anyOf", "propertyOrdering", "items", "minimum", "maximum"}
_FORMATS = {"STRING": {"enum", "date-time"}, "INTEGER": {"int32", "int64"}, "NUMBER": {"float", "double"}}


def sanitize_schema(s: Any, defs: dict | None = None, depth: int = 0,
                    _stack: frozenset[str] = frozenset()) -> dict:
    """Reduce an arbitrary JSON Schema to the OpenAPI subset Gemini's `Schema` accepts.

    `_stack` holds the $ref names currently being expanded on this path. A ref that
    points back into the stack is recursive; it is cut to an open OBJECT instead of
    being re-inlined, which would otherwise grow exponentially with branching refs.
    """
    if not isinstance(s, dict) or depth > SCHEMA_MAX_DEPTH:
        return {}
    defs = {**(defs or {}), **(s.get("$defs") or {}), **(s.get("definitions") or {})}

    def rec(sub: Any, stack: frozenset[str] = _stack) -> dict:
        return sanitize_schema(sub, defs, depth + 1, stack)

    if "$ref" in s:
        name = str(s["$ref"]).split("/")[-1]
        if name in _stack:
            out = {"type": "OBJECT"}
            if s.get("description"):
                out["description"] = s["description"]
            return out
        target = defs.get(name, {})
        return rec({**target, **{k: v for k, v in s.items() if k != "$ref"}}, _stack | {name})
    if isinstance(s.get("allOf"), list):
        merged = {k: v for k, v in s.items() if k != "allOf"}
        for sub in s["allOf"]:
            sub = rec(sub) if "$ref" in (sub or {}) else sub or {}
            for k, v in sub.items():
                if k == "properties":
                    merged.setdefault("properties", {}).update(v)
                elif k == "required":
                    merged["required"] = list(dict.fromkeys((merged.get("required") or []) + v))
                else:
                    merged.setdefault(k, v)
        s = merged
    if "oneOf" in s:
        s = {**{k: v for k, v in s.items() if k != "oneOf"}, "anyOf": s["oneOf"]}
    if "const" in s:
        s = {**{k: v for k, v in s.items() if k != "const"}, "enum": [s["const"]]}

    out: dict[str, Any] = {}
    t = s.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        if "null" in t:
            out["nullable"] = True
        if len(non_null) == 1:
            t = non_null[0]
        else:
            if non_null:
                out["anyOf"] = [{"type": str(x).upper()} for x in non_null]
            t = None
    if t:
        out["type"] = str(t).upper()

    for k, v in s.items():
        if k == "type" or k not in _SCHEMA_KEYS:
            continue
        if k == "properties" and isinstance(v, dict):
            props = {pk: rec(pv) for pk, pv in v.items()}
            if props:
                out["properties"] = props
        elif k == "items":
            out["items"] = rec(v[0] if isinstance(v, list) and v else v)
        elif k == "anyOf" and isinstance(v, list):
            subs = []
            for sub in v:
                if isinstance(sub, dict) and sub.get("type") == "null":
                    out["nullable"] = True
                    continue
                subs.append(rec(sub))
            if len(subs) == 1:
                for sk, sv in subs[0].items():
                    out.setdefault(sk, sv)
            elif subs:
                out["anyOf"] = subs
        elif k == "enum" and isinstance(v, list):
            if None in v:
                out["nullable"] = True
            vals = [x for x in v if x is not None]
            if all(isinstance(x, str) for x in vals):
                out["enum"] = vals
                out.setdefault("type", "STRING")
            else:
                out["description"] = (s.get("description", "") + f" Allowed values: {vals}").strip()
        elif k == "format":
            if v in _FORMATS.get(str(s.get("type", "")).upper(), set()):
                out["format"] = v
        elif k == "description":
            out.setdefault("description", v)
        else:
            out[k] = v

    if out.get("type") == "ARRAY" and "items" not in out:
        out["items"] = {"type": "STRING"}
    if "required" in out:
        props = out.get("properties") or {}
        out["required"] = [r for r in out["required"] if r in props]
        if not out["required"]:
            del out["required"]
    if out.get("type") == "OBJECT" and not out.get("properties"):
        out.pop("properties", None)
    return out


def _function_decl(cfg: Config, f: dict) -> dict:
    d: dict[str, Any] = {"name": f["name"]}
    if f.get("description"):
        d["description"] = f["description"]
    params = f.get("parameters")
    if params:
        if cfg.schema_mode == "jsonschema":
            d["parametersJsonSchema"] = params
        else:
            s = sanitize_schema(params)
            if s.get("properties"):  # Gemini rejects OBJECT with empty properties
                d["parameters"] = s
    return d


def convert_tools(cfg: Config, tools: list | None, functions: list | None) -> list[dict]:
    decls = [_function_decl(cfg, t.get("function", {})) for t in (tools or []) if t.get("type") == "function"]
    decls += [_function_decl(cfg, f) for f in (functions or [])]
    return [{"functionDeclarations": decls}] if decls else []


def convert_tool_choice(tc: Any) -> dict | None:
    if tc is None:
        return None
    if isinstance(tc, str):
        mode = {"none": "NONE", "auto": "AUTO", "required": "ANY"}.get(tc)
        return {"functionCallingConfig": {"mode": mode}} if mode else None
    if isinstance(tc, dict):
        name = (tc.get("function") or {}).get("name") or tc.get("name")
        if name:
            return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}}
    return None


async def build_native_payload(cfg: Config, body: dict, model: str, http: httpx.AsyncClient) -> dict:
    system_parts, contents = await build_contents(cfg, body.get("messages") or [], model, http)
    payload: dict[str, Any] = {"contents": contents}
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}

    tools = convert_tools(cfg, body.get("tools"), body.get("functions"))
    if tools:
        payload["tools"] = tools
        tc = convert_tool_choice(body.get("tool_choice") or body.get("function_call"))
        if tc:
            payload["toolConfig"] = tc

    gc: dict[str, Any] = {}
    for src, dst in (("temperature", "temperature"), ("top_p", "topP"), ("n", "candidateCount"),
                     ("presence_penalty", "presencePenalty"), ("frequency_penalty", "frequencyPenalty"),
                     ("seed", "seed"), ("top_k", "topK")):
        if body.get(src) is not None:
            gc[dst] = body[src]
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    if max_tokens:
        gc["maxOutputTokens"] = max_tokens
    stop = body.get("stop")
    if stop:
        gc["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)[:MAX_STOP_SEQUENCES]

    rf = body.get("response_format") or {}
    if rf.get("type") == "json_object":
        gc["responseMimeType"] = "application/json"
    elif rf.get("type") == "json_schema":
        gc["responseMimeType"] = "application/json"
        schema = (rf.get("json_schema") or {}).get("schema")
        if schema:
            if cfg.schema_mode == "jsonschema":
                gc["responseJsonSchema"] = schema
            else:
                gc["responseSchema"] = sanitize_schema(schema)

    thinking: dict[str, Any] = {}
    effort = body.get("reasoning_effort") or (body.get("reasoning") or {}).get("effort")
    if effort:
        e = str(effort).lower()
        if _starts(model, cfg.thinking_level_prefixes):
            levels = quirk(cfg, model, "thinking_levels") or cfg.thinking_levels
            if e in levels:
                thinking["thinkingLevel"] = levels[e]
        elif e in cfg.reasoning_budgets:
            budget = cfg.reasoning_budgets[e]
            floor = _prefix_lookup(model, cfg.reasoning_budget_min)
            if floor is not None and budget < floor:
                budget = floor
            thinking["thinkingBudget"] = budget
    if body.get("include_reasoning", cfg.include_thoughts):
        thinking["includeThoughts"] = True
    if thinking:
        gc["thinkingConfig"] = thinking

    dropped = [k for k in quirk(cfg, model, "drop_generation_params", []) if gc.pop(k, None) is not None]
    if dropped:
        log.debug("dropped generation params unsupported by %s: %s", model, dropped)
    cap = quirk(cfg, model, "max_output_tokens")
    if cap and gc.get("maxOutputTokens", 0) > cap:
        gc["maxOutputTokens"] = cap

    # Escape hatch: {"gemini": {...}} in the request body (OpenAI SDKs: extra_body={"gemini": {...}})
    # Applied after quirks on purpose: anything set here is sent as-is.
    extra = body.get("gemini") or {}
    if extra.get("tools"):
        payload.setdefault("tools", []).extend(extra["tools"])
    if extra.get("toolConfig"):
        payload["toolConfig"] = extra["toolConfig"]
    if extra.get("generationConfig"):
        gc.update(extra["generationConfig"])
    if extra.get("thinkingConfig"):
        gc["thinkingConfig"] = {**gc.get("thinkingConfig", {}), **extra["thinkingConfig"]}
    if extra.get("cachedContent"):
        payload["cachedContent"] = extra["cachedContent"]
    safety = extra.get("safetySettings") or cfg.safety_settings
    if safety:
        payload["safetySettings"] = safety
    if gc:
        payload["generationConfig"] = gc
    return payload


# --------------------------------------------------------------------------- #
# Gemini -> OpenAI translation                                                #
# --------------------------------------------------------------------------- #

_FINISH = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter",
           "RECITATION": "content_filter", "BLOCKLIST": "content_filter",
           "PROHIBITED_CONTENT": "content_filter", "SPII": "content_filter",
           "IMAGE_SAFETY": "content_filter", "LANGUAGE": "content_filter",
           "MALFORMED_FUNCTION_CALL": "stop", "OTHER": "stop"}


def map_finish(reason: str | None, has_tools: bool) -> str:
    if has_tools and reason in (None, "STOP"):
        return "tool_calls"
    return _FINISH.get(reason or "STOP", "stop")


def convert_parts(parts: list[dict]) -> tuple[str, str, list[dict]]:
    text, reasoning, calls = [], [], []
    for p in parts:
        if "functionCall" in p:
            fc = p["functionCall"]
            tid = fc.get("id") or "call_" + uuid.uuid4().hex[:24]
            if p.get("thoughtSignature"):
                SIGNATURES.put(tid, p["thoughtSignature"])
            calls.append({"id": tid, "type": "function", "function": {
                "name": fc.get("name"), "arguments": json.dumps(fc.get("args") or {}, ensure_ascii=True)}})
        elif "text" in p:
            (reasoning if p.get("thought") else text).append(p["text"])
        elif "inlineData" in p:
            d = p["inlineData"]
            text.append(f"![generated](data:{d.get('mimeType')};base64,{d.get('data')})")
        elif "executableCode" in p:
            ec = p["executableCode"]
            text.append(f"\n```{(ec.get('language') or 'python').lower()}\n{ec.get('code', '')}\n```\n")
        elif "codeExecutionResult" in p:
            text.append(f"\n```\n{p['codeExecutionResult'].get('output', '')}\n```\n")
    return "".join(text), "".join(reasoning), calls


def convert_usage(um: dict) -> dict:
    p = um.get("promptTokenCount", 0)
    c = um.get("candidatesTokenCount", 0)
    t = um.get("thoughtsTokenCount", 0)
    return {"prompt_tokens": p, "completion_tokens": c + t,
            "total_tokens": um.get("totalTokenCount", p + c + t),
            "prompt_tokens_details": {"cached_tokens": um.get("cachedContentTokenCount", 0)},
            "completion_tokens_details": {"reasoning_tokens": t}}


def native_to_openai(data: dict, model_label: str) -> dict:
    choices = []
    for i, cand in enumerate(data.get("candidates") or []):
        text, reasoning, calls = convert_parts((cand.get("content") or {}).get("parts") or [])
        msg: dict[str, Any] = {"role": "assistant", "content": text if (text or not calls) else None}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if calls:
            msg["tool_calls"] = calls
        choices.append({"index": cand.get("index", i), "message": msg,
                        "finish_reason": map_finish(cand.get("finishReason"), bool(calls)), "logprobs": None})
    if not choices:
        blocked = (data.get("promptFeedback") or {}).get("blockReason")
        choices = [{"index": 0, "message": {"role": "assistant", "content": ""},
                    "finish_reason": "content_filter" if blocked else "stop", "logprobs": None}]
    return {"id": "chatcmpl-" + (data.get("responseId") or uuid.uuid4().hex), "object": "chat.completion",
            "created": int(time.time()), "model": model_label, "system_fingerprint": None,
            "choices": choices, "usage": convert_usage(data.get("usageMetadata") or {})}


async def sse_native(resp: httpx.Response, model_label: str, include_usage: bool):
    cid, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
    state: dict[int, dict] = {}
    usage_md: dict = {}

    def chunk(choices: list, usage: dict | None = None) -> str:
        c = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_label,
             "system_fingerprint": None, "choices": choices}
        if usage is not None:
            c["usage"] = usage
        return f"data: {json.dumps(c, ensure_ascii=True)}\n\n"

    try:
        async for line in resp.aiter_lines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("unparseable upstream SSE line: %s", raw[:200])
                continue
            if data.get("error"):
                yield f"data: {json.dumps({'error': data['error']})}\n\n"
                break
            if data.get("usageMetadata"):
                usage_md = data["usageMetadata"]
            if not data.get("candidates") and (data.get("promptFeedback") or {}).get("blockReason"):
                st0 = state.setdefault(0, {"role": False, "tools": 0, "done": False})
                if not st0["done"]:
                    delta0 = {} if st0["role"] else {"role": "assistant", "content": ""}
                    st0["role"] = st0["done"] = True
                    yield chunk([{"index": 0, "delta": delta0,
                                  "finish_reason": "content_filter", "logprobs": None}])
            for i, cand in enumerate(data.get("candidates") or []):
                idx = cand.get("index", i)
                st = state.setdefault(idx, {"role": False, "tools": 0, "done": False})
                text, reasoning, calls = convert_parts((cand.get("content") or {}).get("parts") or [])
                delta: dict[str, Any] = {}
                if not st["role"]:
                    delta["role"], st["role"] = "assistant", True
                    delta["content"] = ""
                if text:
                    delta["content"] = text
                if reasoning:
                    delta["reasoning_content"] = reasoning
                if calls:
                    delta["tool_calls"] = [{"index": st["tools"] + k, **c} for k, c in enumerate(calls)]
                    st["tools"] += len(calls)
                if delta:
                    yield chunk([{"index": idx, "delta": delta, "finish_reason": None, "logprobs": None}])
                if cand.get("finishReason") and not st["done"]:
                    st["done"] = True
                    yield chunk([{"index": idx, "delta": {},
                                  "finish_reason": map_finish(cand["finishReason"], st["tools"] > 0),
                                  "logprobs": None}])
        if not state:
            state[0] = {"role": False, "tools": 0, "done": False}
            yield chunk([{"index": 0, "delta": {"role": "assistant", "content": ""},
                          "finish_reason": None, "logprobs": None}])
        for idx, st in state.items():
            if not st["done"]:
                yield chunk([{"index": idx, "delta": {}, "finish_reason": map_finish(None, st["tools"] > 0),
                              "logprobs": None}])
        if include_usage:
            yield chunk([], convert_usage(usage_md))
        yield "data: [DONE]\n\n"
    except httpx.HTTPError as e:
        yield f"data: {json.dumps({'error': {'message': f'upstream stream error: {e!r}', 'type': 'api_error'}})}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        await resp.aclose()


# --------------------------------------------------------------------------- #
# App                                                                         #
# --------------------------------------------------------------------------- #

def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http = build_http_client(cfg)
        log.info("upstream=%s mode=%s prefixes=%s root_path=%r", cfg.gemini_base_url,
                 cfg.upstream_mode, cfg.route_prefixes, cfg.root_path)
        if not cfg.gemini_api_key and not cfg.passthrough_client_key and cfg.auth_mode != "none":
            log.warning("no Gemini API key configured (%s unset): upstream requests will be refused "
                        "until one is set and the gateway is restarted", " / ".join(API_KEY_ENV_VARS))
        yield
        await app.state.http.aclose()

    app = FastAPI(title="gemini-openai-gateway", lifespan=lifespan, root_path=cfg.root_path,
                  docs_url=None, redoc_url=None)
    app.state.cfg = cfg
    if cfg.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cfg.cors_origins, allow_methods=["*"],
                           allow_headers=["*"], allow_credentials=False)

    router = APIRouter()

    async def _json(request: Request) -> tuple[dict | None, JSONResponse | None]:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError
            return body, None
        except Exception:
            return None, oai_error(400, "Request body must be a JSON object.")

    async def passthrough(request: Request, key: str | None, subpath: str,
                          body: dict | None = None, method: str = "POST", embedding: bool = False):
        http: httpx.AsyncClient = request.app.state.http
        url = f"{cfg.gemini_base_url}{cfg.gemini_openai_path}/{subpath}"
        headers, params = upstream_auth(cfg, key, openai_style=True)
        if body is not None:
            body = {**body, "model": resolve_model(cfg, body.get("model"), embedding)}
        try:
            resp = await send_upstream(cfg, http, method, url, stream=True,
                                       json=body, headers=headers, params=params)
        except httpx.HTTPError as e:
            return oai_error(502, f"upstream request failed: {e!r}", "api_error")
        if resp.status_code >= 400:
            content = await resp.aread()
            await resp.aclose()
            return upstream_error(resp.status_code, content)
        ctype = resp.headers.get("content-type", "application/json")
        if "text/event-stream" in ctype:
            async def gen():
                try:
                    async for b in resp.aiter_bytes():
                        yield b
                finally:
                    await resp.aclose()
            return StreamingResponse(gen(), media_type="text/event-stream", headers=NO_BUFFER_HEADERS)
        content = await resp.aread()
        await resp.aclose()
        return Response(content, status_code=resp.status_code, media_type=ctype)

    @router.post("/chat/completions")
    async def chat_completions(request: Request):
        key, err = resolve_upstream_key(cfg, request)
        if err:
            return err
        body, err = await _json(request)
        if err:
            return err
        if cfg.upstream_mode == "openai":
            return await passthrough(request, key, "chat/completions", body)
        if not body.get("messages"):
            return oai_error(400, "'messages' is required.")

        http: httpx.AsyncClient = request.app.state.http
        requested = body.get("model") or cfg.default_model
        model = resolve_model(cfg, requested)
        label = model if cfg.force_model else requested  # never claim a model we didn't use
        try:
            payload = await build_native_payload(cfg, body, model, http)
        except ValueError as e:
            return oai_error(400, str(e))

        stream = bool(body.get("stream"))
        url = native_url(cfg, model, "streamGenerateContent" if stream else "generateContent")
        headers, params = upstream_auth(cfg, key)
        if log.isEnabledFor(logging.DEBUG):
            log.debug("-> %s %s", url, json.dumps(payload)[:DEBUG_PAYLOAD_MAX_CHARS])
        try:
            if stream:
                params["alt"] = "sse"
                resp = await send_upstream(cfg, http, "POST", url, stream=True,
                                           json=payload, headers=headers, params=params)
                if resp.status_code >= 400:
                    content = await resp.aread()
                    await resp.aclose()
                    return upstream_error(resp.status_code, content)
                include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
                return StreamingResponse(sse_native(resp, label, include_usage),
                                         media_type="text/event-stream", headers=NO_BUFFER_HEADERS)
            resp = await send_upstream(cfg, http, "POST", url, json=payload, headers=headers, params=params)
        except httpx.HTTPError as e:
            return oai_error(502, f"upstream request failed: {e!r}", "api_error")
        if resp.status_code >= 400:
            return upstream_error(resp.status_code, resp.content)
        return JSONResponse(native_to_openai(resp.json(), label))

    @router.post("/embeddings")
    async def embeddings(request: Request):
        key, err = resolve_upstream_key(cfg, request)
        if err:
            return err
        body, err = await _json(request)
        if err:
            return err
        if cfg.upstream_mode == "openai":
            return await passthrough(request, key, "embeddings", body, embedding=True)

        inputs = body.get("input")
        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, list) or not all(isinstance(x, str) for x in inputs) or not inputs:
            return oai_error(400, "'input' must be a string or a list of strings (token arrays unsupported).")
        requested = body.get("model") or cfg.default_embedding_model
        model = resolve_model(cfg, requested, embedding=True)
        extra = body.get("gemini") or {}
        http: httpx.AsyncClient = request.app.state.http
        headers, params = upstream_auth(cfg, key)
        url = native_url(cfg, model, "batchEmbedContents")

        vectors: list[list[float]] = []
        for start in range(0, len(inputs), EMBED_BATCH_SIZE):
            reqs = []
            for text in inputs[start:start + EMBED_BATCH_SIZE]:
                r: dict[str, Any] = {"model": f"models/{model}", "content": {"parts": [{"text": text}]}}
                if body.get("dimensions"):
                    r["outputDimensionality"] = body["dimensions"]
                if extra.get("taskType"):
                    r["taskType"] = extra["taskType"]
                reqs.append(r)
            try:
                resp = await send_upstream(cfg, http, "POST", url, json={"requests": reqs},
                                           headers=headers, params=params)
            except httpx.HTTPError as e:
                return oai_error(502, f"upstream request failed: {e!r}", "api_error")
            if resp.status_code >= 400:
                return upstream_error(resp.status_code, resp.content)
            vectors += [e.get("values", []) for e in resp.json().get("embeddings", [])]

        as_b64 = body.get("encoding_format") == "base64"
        data = [{"object": "embedding", "index": i,
                 "embedding": base64.b64encode(struct.pack(f"<{len(v)}f", *v)).decode() if as_b64 else v}
                for i, v in enumerate(vectors)]
        return {"object": "list", "data": data, "model": requested,
                "usage": {"prompt_tokens": 0, "total_tokens": 0}}

    @router.get("/models")
    async def list_models(request: Request):
        key, err = resolve_upstream_key(cfg, request)
        if err:
            return err
        if cfg.upstream_mode == "openai":
            return await passthrough(request, key, "models", method="GET")
        http: httpx.AsyncClient = request.app.state.http
        headers, params = upstream_auth(cfg, key)
        url = cfg.gemini_base_url + cfg.models_path_template.format(version=cfg.gemini_api_version)
        ids: list[str] = []
        page = None
        try:
            for _ in range(MODELS_MAX_PAGES):
                q = {**params, "pageSize": MODELS_PAGE_SIZE, **({"pageToken": page} if page else {})}
                resp = await send_upstream(cfg, http, "GET", url, headers=headers, params=q)
                if resp.status_code >= 400:
                    log.warning("model listing failed: %s %s", resp.status_code, resp.text[:300])
                    break
                j = resp.json()
                ids += [m["name"].split("/", 1)[-1] for m in j.get("models", []) if m.get("name")]
                page = j.get("nextPageToken")
                if not page:
                    break
        except httpx.HTTPError as e:
            log.warning("model listing failed: %r", e)
        data = [{"id": i, "object": "model", "created": 0, "owned_by": "google"} for i in dict.fromkeys(ids)]
        for alias in cfg.model_aliases:
            data.append({"id": alias, "object": "model", "created": 0, "owned_by": "gateway-alias"})
        if not data:
            data.append({"id": cfg.default_model, "object": "model", "created": 0, "owned_by": "google"})
        return {"object": "list", "data": data}

    @router.get("/models/{model_id:path}")
    async def get_model(model_id: str, request: Request):
        _, err = authorize(cfg, request)
        if err:
            return err
        return {"id": model_id, "object": "model", "created": 0,
                "owned_by": "gateway-alias" if model_id in cfg.model_aliases else "google"}

    for prefix in cfg.route_prefixes:
        app.include_router(router, prefix=prefix)

    @app.get(HEALTH_PATH)
    async def health():
        return {"status": "ok", "mode": cfg.upstream_mode, "upstream": cfg.gemini_base_url,
                "prefixes": cfg.route_prefixes, "key_configured": bool(cfg.gemini_api_key)}

    return app


def setup_logging(level: str) -> None:
    logging.basicConfig(level=level.upper(), format=LOG_FORMAT)
    # httpx logs every request URL at INFO; keep it quiet and redact any key that slips through.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    redact = _RedactKeyFilter()
    for h in logging.getLogger().handlers:
        h.addFilter(redact)


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenAI-compatible gateway for the Gemini API")
    ap.add_argument("-c", "--config", help="YAML config file (or set GW_CONFIG)")
    ap.add_argument("--print-config", action="store_true", help="print the annotated example config and exit")
    args = ap.parse_args()
    if args.print_config:
        print(EXAMPLE_CONFIG, end="")
        return
    cfg = load_config(args.config)
    setup_logging(cfg.log_level)
    check_bind_safety(cfg)

    import uvicorn
    uvicorn.run(create_app(cfg), host=cfg.listen_host, port=cfg.listen_port,
                proxy_headers=True, forwarded_allow_ips=cfg.forwarded_allow_ips,
                log_level=cfg.log_level.lower())


if __name__ == "__main__":
    main()
