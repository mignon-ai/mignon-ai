"""
llm_backend.py

Lets the data generators (generate_tinytech.py, generate_git_sft_extras.py) run on either backend:
  --backend claude   the paid Claude Message Batches API (claude_batch.py), the default
  --backend ollama   a local Ollama model (ollama_batch.py), free but slower and less accurate
"""

import claude_batch
import ollama_batch

BACKENDS = {"claude": claude_batch, "ollama": ollama_batch}


def add_arguments(ap):
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="claude")
    ap.add_argument("--model", default=None, help="model name (default: the backend's default)")
    ap.add_argument("--effort", default=None, help="reasoning effort (Claude only)")


def resolve(args):
    """(backend module, model, effort) for the parsed arguments."""
    backend = BACKENDS[args.backend]
    return backend, args.model or backend.DEFAULT_MODEL, args.effort or backend.DEFAULT_EFFORT
