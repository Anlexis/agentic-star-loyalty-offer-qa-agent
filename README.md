# Loyalty Offer Q&A Agent

AI agent for answering loyalty programme and offer questions, built with Agentic Star.

> **Category**: Cat 2 (domain-specific retrieval-augmented pipeline)
> **Industry**: Retail
> **Template ID**: RET-C2-087

## Overview

Answers questions about a retail loyalty programme. Given a natural-language question from a
member — or from a service agent acting on their behalf — it retrieves from a seeded loyalty
knowledge base covering programme rules, the offer catalogue and the redemption catalogue, then
returns a grounded, citation-backed answer about how points are earned and expire, what a member
can redeem them for, and which promotions are currently running.

The agent is deterministic and makes no network call on the request path. Every claim it makes
about an offer is grounded in the retrieved knowledge base rather than asserted on its own, which
matters because a promotional claim that the source material does not support is a
consumer-protection problem, not merely an inaccuracy.

It answers questions about policy and catalogue only, and never reports an individual member's
account balance. That is the boundary it is built around rather than a missing feature: there is
no account or point-of-sale integration anywhere in the pipeline, so a question about one member's
own points is refused rather than guessed at.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode: if the platform is unreachable or the installed SDK does not match, start-up fails rather
than bringing up a partially working agent. This is intentional — a half-running agent is worse
than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
docs/         design and operational documentation
```

See `docs/` for the design and the test specification.

## Customising

1. Adjust `config/` for your own environment and policies.
2. Replace the knowledge sources and sample data with your own.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
