"""
ollama_batch.py

Drop-in replacement for claude_batch.py that generates with a local Ollama model instead of the paid
Claude Message Batches API. It has the same functions (build_request, run, dry_run, json_list_schema),
so the generators pick one with --backend (see llm_backend.py).

Flow: build requests -> run them one by one against Ollama's /api/chat with the JSON schema as the
`format` (structured output) -> collect {custom_id: items}. Finished results are saved to the state
file as they come in, so re-running a generator with the same state file only does the missing ones.

Setup: `ollama pull qwen2.5:3b-instruct`. OLLAMA_HOST overrides the server address and
OLLAMA_WORKERS the number of parallel requests (the server must allow them: OLLAMA_NUM_PARALLEL).
Small local models make mistakes, so keep the validation in the generators and check a sample by hand.
"""

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from claude_batch import json_list_schema  # noqa: F401  (re-exported: generators build schemas with it)

DEFAULT_MODEL = "qwen2.5:3b-instruct"
DEFAULT_EFFORT = "low"  # accepted for symmetry with claude_batch; Ollama has no such setting
DEFAULT_MAX_TOKENS = 4000
TEMPERATURE = 0.8
TIMEOUT_SECONDS = 900  # the first request also loads the model
RETRIES = 2
SAVE_EVERY = 10
WORKERS = int(os.environ.get("OLLAMA_WORKERS", "1"))


def _url():
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    return (host if "://" in host else f"http://{host}").rstrip("/") + "/api/chat"


def build_request(custom_id, prompt, schema, model=DEFAULT_MODEL, effort=DEFAULT_EFFORT,
                  max_tokens=DEFAULT_MAX_TOKENS, system=None):
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "messages": messages,
            "format": schema,
            "stream": False,
            "options": {"temperature": TEMPERATURE, "num_predict": max_tokens},
        },
    }


def _load_results(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("results", {})
    return {}


def _save_results(path, results):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f)
    os.replace(tmp, path)


def _complete(request, field):
    """Items for one request, or None when the model failed, was cut off or broke the schema."""
    body = json.dumps(request["params"]).encode()
    for attempt in range(RETRIES + 1):
        try:
            req = urllib.request.Request(_url(), data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
                data = json.load(resp)
            if data.get("done_reason") == "length":
                return None  # truncated JSON; retrying gives the same
            return json.loads(data["message"]["content"])[field]
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(5 * (attempt + 1))  # server busy or still starting
        except (json.JSONDecodeError, KeyError, TypeError):
            continue  # a different sample may follow the schema
    return None


def run(requests, state_path, field):
    """Run the requests not yet in the state file; returns {custom_id: items} for all of them."""
    results = _load_results(state_path)
    todo = [r for r in requests if r["custom_id"] not in results]
    wanted = {r["custom_id"] for r in requests}
    print(f"{len(requests) - len(todo)} of {len(requests)} requests already done, {len(todo)} to run "
          f"on {todo[0]['params']['model'] if todo else '-'} with {WORKERS} worker(s)")
    failed, started = 0, time.time()
    try:
        with ThreadPoolExecutor(max_workers=max(1, WORKERS)) as pool:
            futures = {pool.submit(_complete, r, field): r["custom_id"] for r in todo}
            for i, fut in enumerate(as_completed(futures), start=1):
                items = fut.result()
                if items is None:
                    failed += 1
                else:
                    results[futures[fut]] = items
                if i % SAVE_EVERY == 0 or i == len(todo):
                    _save_results(state_path, results)
                    per = (time.time() - started) / i
                    print(f"  {i}/{len(todo)} done, {failed} failed, about {per * (len(todo) - i) / 60:.0f} min left")
    finally:
        _save_results(state_path, results)
    if failed:
        print(f"{failed} requests produced no usable output (server error, truncated or off-schema)")
    return {cid: items for cid, items in results.items() if cid in wanted}


def dry_run(requests, show=1):
    """Print example prompts and request count without calling the server."""
    print(f"{len(requests)} requests would be sent. Example prompt(s):")
    for req in requests[:show]:
        print("-" * 60)
        print(req["params"]["messages"][-1]["content"])
