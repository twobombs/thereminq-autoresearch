import os
import io
import sys
import json
import re
import ipaddress
import openai
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from fpdf import FPDF
from urllib.parse import urlparse

try:
    from pypdf import PdfReader  # optional: pip install pypdf
except ImportError:
    PdfReader = None

# --- 1. CONFIGURATION ---

def secure_local_endpoint(url: str):
    """Enforces that scripts using 'api_key=not-needed' don't silently leak to the cloud.
    Parses the host as an IP so '192.168.evil.com' can't pass a string prefix check."""
    host = urlparse(url).hostname
    if host == "localhost":
        return url
    try:
        ip = ipaddress.ip_address(host or "")
    except ValueError:
        raise ValueError(f"SECURITY HALT: non-IP or missing host in endpoint: {url}")
    if not (ip.is_loopback or ip.is_private or ip.is_unspecified):
        raise ValueError(f"SECURITY HALT: Attempted to bind unauthenticated client to external endpoint: {url}")
    return url

ORCHESTRATOR_URL = secure_local_endpoint(os.getenv("ORCHESTRATOR_URL", "http://localhost:8033/v1"))
REASONING_URL = secure_local_endpoint(os.getenv("REASONING_URL", "http://localhost:9931/v1"))

ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "nemotron-orchestrator-8b")
REASONING_MODEL = os.getenv("REASONING_MODEL", "qwen-3.5-35b")

MAX_SEARCH_RESULTS = 4
CHARS_PER_PAGE = 15000
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "60000"))  # total cap across all searches
REASONER_MAX_TOKENS = int(os.getenv("REASONER_MAX_TOKENS", "4096"))
REASONER_THINKING = os.getenv("REASONER_THINKING", "0") == "1"   # 1 = allow thinking tokens
EDITOR_MODE = os.getenv("EDITOR_MODE", "format")                 # "format" or "verify"
CLIENT_TIMEOUT = (10.0, 600.0)

UNICODE_FONT = os.getenv("PDF_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
UNICODE_FONT_BOLD = os.getenv("PDF_FONT_BOLD", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

# Client Setup
orch_client = openai.OpenAI(base_url=ORCHESTRATOR_URL, api_key="not-needed", timeout=CLIENT_TIMEOUT)
reason_client = openai.OpenAI(base_url=REASONING_URL, api_key="not-needed", timeout=CLIENT_TIMEOUT)

# --- 2. DEEP WEB SCRAPING TOOL (WITH SMART FILTERING) ---
BANNED_DOMAINS = {"youtube.com", "youtu.be", "vimeo.com", "tiktok.com", "instagram.com"}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def is_banned(url):
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in BANNED_DOMAINS)


def normalize_url(url):
    """Prefer arXiv abstract pages over raw PDFs when no PDF reader is available."""
    if PdfReader is None and "arxiv.org/pdf/" in url:
        return url.replace("/pdf/", "/abs/").removesuffix(".pdf")
    return url


def extract_pdf_text(data):
    reader = PdfReader(io.BytesIO(data))
    pages = []
    total = 0
    for page in reader.pages:
        t = page.extract_text() or ""
        pages.append(t)
        total += len(t)
        if total > CHARS_PER_PAGE:
            break
    return " ".join(pages)


def extract_html_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.extract()
    return soup.get_text(separator=" ", strip=True)


def fetch_page(url, snippet):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
    except requests.exceptions.RequestException:
        return f"Connection error. Snippet fallback: {snippet}"

    if resp.status_code != 200:
        return f"Failed to load page (HTTP {resp.status_code}). Snippet fallback: {snippet}"

    ctype = resp.headers.get("Content-Type", "").lower()
    try:
        if "application/pdf" in ctype or url.lower().endswith(".pdf"):
            if PdfReader is None:
                return f"PDF source (pypdf not installed). Snippet fallback: {snippet}"
            text = extract_pdf_text(resp.content)
        elif "html" in ctype or "text" in ctype or not ctype:
            text = extract_html_text(resp.text)
        else:
            return f"Unsupported content type ({ctype}). Snippet fallback: {snippet}"
    except Exception as e:
        return f"Parse error ({e}). Snippet fallback: {snippet}"

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return f"Empty page. Snippet fallback: {snippet}"
    if len(text) > CHARS_PER_PAGE:
        text = text[:CHARS_PER_PAGE] + "... [CONTENT TRUNCATED FOR LENGTH]"
    return text


def perform_web_search(query):
    print(f"[Orchestrator] Searching Web for: {query}")
    try:
        results = DDGS().text(query, max_results=10)
        if not results:
            return "No results found."

        parts = []
        scraped = 0
        for res in results:
            if scraped >= MAX_SEARCH_RESULTS:
                break

            url = res.get("href", "")
            title = res.get("title", "Unknown Title")
            snippet = res.get("body", "")

            if not url or is_banned(url):
                print(f"   [Skipping Video/Empty Link]: {url}")
                continue

            url = normalize_url(url)
            print(f"   [Deep Reading]: {url}")
            text = fetch_page(url, snippet)

            parts.append(f"\n\n--- Source: {title} ---\nURL: {url}\nCONTENT:\n{text}\n")
            scraped += 1

        return "".join(parts)

    except Exception as e:
        return f"Search Error: {str(e)}"

# --- 3. PDF GENERATION ---
def sanitize_filename(query):
    clean_name = re.sub(r"[^\w\s-]", "", query).strip().replace(" ", "_")[:50]
    return (clean_name or "research_report") + ".pdf"


def save_to_pdf(query, content):
    filename = sanitize_filename(query)
    print(f"\n[System] Saving output to {filename}...")

    pdf = FPDF()
    pdf.add_page()

    unicode_ok = os.path.exists(UNICODE_FONT) and os.path.exists(UNICODE_FONT_BOLD)
    if unicode_ok:
        pdf.add_font("DejaVu", fname=UNICODE_FONT)
        pdf.add_font("DejaVu", style="B", fname=UNICODE_FONT_BOLD)
        family = "DejaVu"
    else:
        print("   [Warning] Unicode font not found; falling back to latin-1 (Greek letters and math symbols become '?').")
        family = "helvetica"
        query = query.encode("latin-1", "replace").decode("latin-1")
        content = content.encode("latin-1", "replace").decode("latin-1")

    pdf.set_font(family, style="B", size=14)
    pdf.multi_cell(0, 10, text=f"Query: {query}")
    pdf.ln(5)

    pdf.set_font(family, size=11)
    # markdown=True renders **bold**; needs the bold font variant registered above
    pdf.multi_cell(0, 6, text=content, markdown=unicode_ok)

    pdf.output(filename)
    return filename

# --- 4. THE WORKFLOW ENGINE ---
def stream_completion(client, label, **kwargs):
    """Streams a chat completion, printing answer tokens to stdout and any
    reasoning tokens to stderr. Returns the answer text only."""
    answer_parts = []
    try:
        stream = client.chat.completions.create(stream=True, **kwargs)
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                sys.stderr.write(reasoning)
                sys.stderr.flush()
            if getattr(delta, "content", None):
                sys.stdout.write(delta.content)
                sys.stdout.flush()
                answer_parts.append(delta.content)
    except Exception as e:
        print(f"\n[{label} Error]: {str(e)}")
    return "".join(answer_parts)


def gather_context(user_query):
    tools = [{
        "type": "function",
        "function": {
            "name": "perform_web_search",
            "description": "Scrape in-depth research articles from the internet.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    }]

    try:
        response = orch_client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[{"role": "user", "content": user_query}],
            tools=tools
        )
    except Exception as e:
        print(f"[Orchestrator Planning Error]: {e}. Falling back to searching the raw query.")
        return perform_web_search(user_query)

    msg = response.choices[0].message
    if not msg.tool_calls:
        print("[Orchestrator] No search required.")
        return ""

    context = ""
    for tool_call in msg.tool_calls:
        try:
            args = json.loads(tool_call.function.arguments or "{}")
            query = args.get("query") or user_query
        except (json.JSONDecodeError, AttributeError):
            print("   [Warning] Malformed tool call; using the user query instead.")
            query = user_query

        remaining = MAX_CONTEXT_CHARS - len(context)
        if remaining <= 0:
            print("   [Stop] Context budget reached; skipping further searches.")
            break
        result = perform_web_search(query)
        if len(result) > remaining:
            result = result[:remaining] + "\n... [CONTEXT BUDGET REACHED]"
        context += result

    return context


def run_orchestration_loop(user_query):
    print("[Orchestrator] Planning strategy...")
    context = gather_context(user_query)

    print(f"\n[Reasoner] Processing deep logic (Live Draft, thinking={'on' if REASONER_THINKING else 'off'}):\n" + "-" * 50)
    draft = stream_completion(
        reason_client, "Reasoner",
        model=REASONING_MODEL,
        messages=[
            {"role": "system", "content": "Analyze provided facts to answer the user's query. Synthesize complex research. Be precise. Do not repeat the context. Conclude your answer and stop."},
            {"role": "user", "content": f"FACTS:\n{context}\n\nUSER TASK: {user_query}"}
        ],
        max_tokens=REASONER_MAX_TOKENS,
        extra_body={"chat_template_kwargs": {"enable_thinking": REASONER_THINKING}},
    )
    print("\n" + "-" * 50)

    draft_empty = not draft.strip()
    if draft_empty:
        print("[Warning] The reasoning model returned an empty response.")
        print("[Fallback] The orchestrator will now attempt to write the report from scratch.")

    if draft_empty:
        task = """TASK: The draft is empty. Write the final report for the user based ONLY on the facts above.
    If the facts are insufficient, say so plainly instead of inventing content."""
    elif EDITOR_MODE == "verify":
        task = """TASK: You are the final-stage Editor. Review the Draft Answer against the Facts.
    - Only change a claim if it directly contradicts the facts; do not remove reasoning just because the facts don't mention it.
    - Otherwise output the draft as-is, polished for the user."""
    else:  # "format"
        task = """TASK: You are the final-stage copy editor. Keep the draft's content, claims and reasoning intact.
    Fix formatting, structure, grammar and clarity only. Do not add, remove or change technical claims."""

    editor_prompt = f"""
    FACTS FROM WEB:
    {context}

    DRAFT ANSWER:
    {draft}

    {task}
    CRITICAL RULE: DO NOT include meta-commentary, audit notes, or explain your grading process.
    Output ONLY the final, polished report intended for the user.
    """

    print(f"\n[Orchestrator] Final edit (mode={EDITOR_MODE}, Live Edit):\n" + "=" * 50)
    verification = stream_completion(
        orch_client, "Orchestrator",
        model=ORCHESTRATOR_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert editor. Output only the final document text without any meta-commentary."},
            {"role": "user", "content": editor_prompt}
        ],
    )
    print("\n" + "=" * 50)

    if verification.strip():
        return verification
    if not draft_empty:
        print("[Warning] Editor returned nothing; saving the unedited draft.")
        return draft
    return "No report could be generated: both the reasoning model and the editor returned empty output."

# --- 5. CLI EXECUTION ---
if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
        print(f"[System] Custom CLI query: '{user_prompt}'")
    else:
        user_prompt = "research on the negative energy spike when teleporting a qubit according to ER=EPR and devise a strategy to execute the thesis on a quantum computer that shows that ER=EPR is an example of quantum gravity"
        print("[System] Default mode: Deep Research Test...")

    try:
        final_result = run_orchestration_loop(user_prompt)
        saved_file = save_to_pdf(user_prompt, final_result)
        print(f"[Success] Report generated: {os.path.abspath(saved_file)}")
    except Exception as e:
        print(f"\n[Error] The workflow failed: {str(e)}")
