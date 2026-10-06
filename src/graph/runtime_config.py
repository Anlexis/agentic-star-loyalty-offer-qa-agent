"""AgentCore Platform v1.0"""

# RET-C2-087 - the runtime configuration file and the retrieval block inside it.
#
# Two files sit in config/ and they are not interchangeable:
#
#   config/agent.yaml   the static manifest - identity, entry-point class,
#                       trust level, and the compile-time `requires` gates.
#                       Every key sits at ROOT level; there is no `agent:`
#                       block and no `config:` block inside it.
#   config/config.yaml  the runtime parameters - `max_retry`, `timeout_s`, and
#                       this template's `retrieval` block. This is the file the
#                       registry loads and hands to the graph as
#                       `Graph(config=...)`.
#
# Reading runtime values out of the manifest is therefore a silent no-op: the
# lookup succeeds, returns nothing, and the code falls back to its defaults, so
# a declared value simply never takes effect while every test stays green. The
# loader below exists so exactly one module knows which file holds what.

from pathlib import Path
from typing import Any, Dict

from src.schemas.caller_contract import (
    SCORE_THRESHOLD_MAX,
    SCORE_THRESHOLD_MIN,
    TOP_K_MAX,
    TOP_K_MIN,
    CallerFieldError,
    finite_in_range,
)

# src/graph/runtime_config.py -> parents[2] = repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_CONFIG_PATH = _REPO_ROOT / "config" / "config.yaml"
MANIFEST_PATH = _REPO_ROOT / "config" / "agent.yaml"

# Last-resort retrieval values, used only when config/config.yaml is unreadable
# in an exotic deployment layout. They mirror the shipped file so behaviour does
# not change shape between the two paths.
DEFAULT_RETRIEVAL: Dict[str, Any] = {
    "top_k": 4,
    "score_threshold": 0.25,
    "kb_path": "config/kb/loyalty_kb.json",
}


def load_runtime_config() -> Dict[str, Any]:
    """Return the parsed config/config.yaml mapping, or {} if it is unreadable."""
    try:
        import yaml

        loaded = yaml.safe_load(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def load_manifest() -> Dict[str, Any]:
    """Return the parsed config/agent.yaml mapping, or {} if it is unreadable."""
    try:
        import yaml

        loaded = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def validate_retrieval_block(block: Any) -> Dict[str, Any]:
    """Validate an operator-declared `retrieval` block; raise on a bad value.

    Applied to the DECLARED configuration, not to caller input, so the failure
    mode is a startup error naming the key rather than a request rejection - a
    misconfigured relevance floor should stop the deployment, not quietly
    answer every question differently.
    """
    if not isinstance(block, dict):
        raise CallerFieldError("retrieval", "must be a mapping")

    validated = dict(DEFAULT_RETRIEVAL)
    validated.update(block)

    validated["top_k"] = int(
        finite_in_range(
            validated.get("top_k"),
            field="retrieval.top_k",
            minimum=TOP_K_MIN,
            maximum=TOP_K_MAX,
            integer=True,
        )
    )
    validated["score_threshold"] = finite_in_range(
        validated.get("score_threshold"),
        field="retrieval.score_threshold",
        minimum=SCORE_THRESHOLD_MIN,
        maximum=SCORE_THRESHOLD_MAX,
    )
    kb_path = validated.get("kb_path")
    if not isinstance(kb_path, str) or not kb_path.strip():
        raise CallerFieldError("retrieval.kb_path", "must be a non-empty path")
    validated["kb_path"] = kb_path.strip()
    return validated


def resolve_retrieval(graph_config: Any = None) -> Dict[str, Any]:
    """Resolve the effective retrieval block, validated.

    Preference order: the config the graph was constructed with (what the
    registry passes in production) -> config/config.yaml on disk (the same file,
    read directly, which is what a bare `Agent()` in a test gets) -> the
    defaults above.
    """
    block: Any = None
    if isinstance(graph_config, dict):
        candidate = graph_config.get("retrieval")
        if isinstance(candidate, dict) and candidate:
            block = candidate
    if block is None:
        candidate = load_runtime_config().get("retrieval")
        block = candidate if isinstance(candidate, dict) and candidate else {}
    return validate_retrieval_block(block)
