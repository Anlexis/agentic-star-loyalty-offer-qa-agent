"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for the agent.
# Entry points are adapters only - no business logic here.
# For platform-level routing, the gateway calls agent.invoke() directly.

import json
import os
import secrets
from typing import Any, Dict
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets import factory as secrets_factory
from src.graph.graph import LoyaltyOfferQAAgent
from src.graph.runtime_config import load_manifest, load_runtime_config

app = FastAPI(title="Agent")

# The runtime parameters live in config/config.yaml, NOT in the manifest.
# Constructing the agent without them would leave max_retry, timeout_s and the
# retrieval block at their built-in defaults while config/config.yaml declared
# something else - a mismatch nothing would report.
_RUNTIME_CONFIG = load_runtime_config()
_MANIFEST = load_manifest()

# The secrets namespace is the manifest's `namespace` value - not the template
# id. Reading it from the manifest keeps the two from drifting apart.
_SECRETS_NAMESPACE = str(_MANIFEST.get("namespace") or "ret")
_AGENT_NAME = str(_MANIFEST.get("name") or "LoyaltyOfferQAAgent")

agent = LoyaltyOfferQAAgent(config=_RUNTIME_CONFIG)
agent.compile()
agent.provision_secrets(secrets_factory(namespace=_SECRETS_NAMESPACE, agent_name=_AGENT_NAME))

# Adapter-level structural limits. They sit in front of the domain validation
# so an oversized body is refused before anything parses it; the per-field
# bounds still apply afterwards (src/schemas/caller_contract.py).
MAX_INPUT_CHARS = 16_000
MAX_CONTEXT_BYTES = 8_192
MAX_CONTEXT_FIELDS = 16


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""
    # Optional per-request retrieval overrides. Supported fields and their
    # bounds are defined by the caller-data contract; anything else is refused.
    input_context: Dict[str, Any] = Field(default_factory=dict)


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> Dict[str, Any]:
    trust = getattr(request.state, "trust_level", TrustLevel.ANONYMOUS)
    # Standalone caller auth: when INVOKE_AUTH_TOKEN is set on the server
    # environment, callers that no upstream middleware vouched for (still
    # ANONYMOUS) must present it as a Bearer token and run at
    # VERIFIED_EXTERNAL. Middleware-established trust is never demoted.
    # This adapter is the entry-point auth boundary - a deployment-level
    # caller credential, not an agent secret, so ctx.secrets does not apply
    # (no InvocationContext exists before auth).
    expected = os.environ.get("INVOKE_AUTH_TOKEN")
    if expected and trust is TrustLevel.ANONYMOUS:
        supplied = request.headers.get("authorization", "")
        # Compare bytes: compare_digest raises TypeError on non-ASCII str input
        # (headers decode as latin-1), which would 500 instead of the generic 401.
        if not secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode()):
            # Generic body on purpose - do not leak whether the token was absent,
            # malformed, or wrong.
            raise HTTPException(status_code=401, detail="Token is invalid or expired.")
        trust = TrustLevel.VERIFIED_EXTERNAL

    if len(req.input) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=413, detail="input is too large.")
    if len(req.input_context) > MAX_CONTEXT_FIELDS:
        raise HTTPException(status_code=413, detail="input_context has too many fields.")
    try:
        context_bytes = len(json.dumps(req.input_context).encode("utf-8"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="input_context is not serialisable.") from None
    if context_bytes > MAX_CONTEXT_BYTES:
        raise HTTPException(status_code=413, detail="input_context is too large.")

    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        result: Dict[str, Any] = agent.invoke(req.input, ctx=ctx, input_context=req.input_context)
        return result


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "agent": _AGENT_NAME}
