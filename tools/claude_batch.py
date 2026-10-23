"""
claude_batch.py

Shared helpers for generating synthetic data with the Claude Message Batches API (50% of the
standard price, results within 24h). Used by generate_tinytech.py and generate_git_sft_extras.py.

Flow: build requests -> submit (batch ids are saved to a state file) -> wait -> collect JSON results.
Re-running a generator with the same state file resumes instead of submitting (and paying) again.
Authentication: ANTHROPIC_API_KEY, or a profile from `ant auth login`.
"""

import json
import os
import time

# The anthropic package is imported inside the functions that need it, so this module (and
# json_list_schema) can be used without it, e.g. by the Ollama backend.

DEFAULT_MODEL = "claude-opus-5-5"
# Writing simple paragraphs needs little reasoning; low effort keeps thinking tokens (and cost) down.
DEFAULT_EFFORT = "low"
DEFAULT_MAX_TOKENS = 8000
MAX_REQUESTS_PER_BATCH = 100_000
POLL_SECONDS = 60


def json_list_schema(field, item_schema):
    """Output schema {field: [item, ...]} for structured outputs."""
    return {
        "type": "object",
        "properties": {field: {"type": "array", "items": item_schema}},
        "required": [field],
        "additionalProperties": False,
    }


def build_request(custom_id, prompt, schema, model=DEFAULT_MODEL, effort=DEFAULT_EFFORT,
                  max_tokens=DEFAULT_MAX_TOKENS, system=None):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    params = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
    )
    if system:
        params["system"] = system
    return Request(custom_id=custom_id, params=MessageCreateParamsNonStreaming(**params))


def _load_state(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_state(path, state):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def submit(client, requests, state_path):
    """Submit requests (unless the state file already holds batch ids) and return the batch ids."""
    state = _load_state(state_path)
    if state.get("batch_ids"):
        print(f"Resuming batches from {state_path}: {state['batch_ids']}")
        return state["batch_ids"]
    ids = []
    for start in range(0, len(requests), MAX_REQUESTS_PER_BATCH):
        batch = client.messages.batches.create(requests=requests[start:start + MAX_REQUESTS_PER_BATCH])
        ids.append(batch.id)
        print(f"Submitted batch {batch.id} ({min(MAX_REQUESTS_PER_BATCH, len(requests) - start)} requests)")
        _save_state(state_path, {"batch_ids": ids})
    return ids


def wait(client, batch_ids):
    for batch_id in batch_ids:
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                c = batch.request_counts
                print(f"Batch {batch_id} ended: {c.succeeded} succeeded, {c.errored} errored, {c.expired} expired")
                break
            print(f"Batch {batch_id}: {batch.processing_status}, {batch.request_counts.processing} processing")
            time.sleep(POLL_SECONDS)


def collect(client, batch_ids, field):
    """Yield (custom_id, items) for every succeeded request; results arrive in any order."""
    failed = 0
    for batch_id in batch_ids:
        for result in client.messages.batches.results(batch_id):
            if result.result.type != "succeeded":
                failed += 1
                continue
            msg = result.result.message
            if msg.stop_reason in ("refusal", "max_tokens"):
                failed += 1
                continue
            text = next((b.text for b in msg.content if b.type == "text"), "")
            try:
                yield result.custom_id, json.loads(text)[field]
            except (json.JSONDecodeError, KeyError):
                failed += 1
    if failed:
        print(f"{failed} requests produced no usable output (errored, refused, truncated or expired)")


def run(requests, state_path, field):
    """Submit or resume, wait, and return {custom_id: items}."""
    import anthropic

    client = anthropic.Anthropic()
    batch_ids = submit(client, requests, state_path)
    wait(client, batch_ids)
    return dict(collect(client, batch_ids, field))


def dry_run(requests, show=1):
    """Print example prompts and request count without calling the API."""
    print(f"{len(requests)} requests would be submitted. Example prompt(s):")
    for req in requests[:show]:
        print("-" * 60)
        print(req["params"]["messages"][0]["content"])
