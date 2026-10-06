# RET-C2-087 — Unit Tests: manifest / runtime-config consistency
#
# Two files, two jobs, and mixing them up is a silent failure rather than a
# loud one:
#
#   config/agent.yaml   the static manifest — identity, entry-point class,
#                       trust level, compile-time `requires`. Flat: every key
#                       at root level.
#   config/config.yaml  the runtime parameters — max_retry, timeout_s, and the
#                       retrieval block the domain nodes actually run on.
#
# A reader pointed at the wrong file does not raise; it gets nothing back and
# falls through to its defaults, so the declared value never takes effect while
# every test stays green. These tests pin both files against the code that
# consumes them, including an end-to-end check that a declared retrieval value
# reaches the inner graph.
#
# Deterministic — no LLM, no network.

import json
import pathlib

import yaml

from framework.schemas.trust_level import TrustLevel

from src.graph.graph import LoyaltyOfferQAAgent, LoyaltyOfferQAGraphNode
from src.graph.runtime_config import DEFAULT_RETRIEVAL
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import from_json

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = yaml.safe_load((_ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))
_RUNTIME = yaml.safe_load((_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))


class TestManifestIdentity:
    def test_cfg_01_manifest_is_flat(self):
        # The registry reads every key at ROOT level; a nested `agent:` block
        # would be read as nothing at all.
        assert "agent" not in _MANIFEST
        assert "config" not in _MANIFEST
        assert _MANIFEST["id"] == "RET-C2-087"

    def test_cfg_02_declared_class_is_the_graph_class(self):
        assert _MANIFEST["class"] == "src.graph.graph.LoyaltyOfferQAAgent"
        assert _MANIFEST["class"].endswith(LoyaltyOfferQAAgent.__name__)
        assert _MANIFEST["name"] == LoyaltyOfferQAAgent().name

    def test_cfg_03_category_and_industry(self):
        assert _MANIFEST["category"] == "Cat 2"
        assert _MANIFEST["industry"] == "RET"
        assert _MANIFEST["base_type"] == "RAGAgent"

    def test_namespace_is_declared(self):
        # src/api/server.py provisions secrets under this namespace; reading it
        # from the manifest is what keeps the two from drifting apart.
        assert _MANIFEST["namespace"] == "ret"

    def test_generation_mode_matches_the_implementation(self):
        # This template invokes no model — a declared "llm" mode would be a
        # claim the code does not back up.
        assert _MANIFEST["generation_mode"] == "deterministic"


class TestManifestSecurity:
    def test_cfg_04_required_trust_level_matches_outer_gate_nodes(self):
        declared = TrustLevel(_MANIFEST["required_trust_level"])
        assert declared is TrustLevel.VERIFIED_EXTERNAL
        assert PreProcessNode.required_trust_level is declared
        assert PostProcessNode.required_trust_level is declared

    def test_requires_block_is_empty_and_that_matches_the_code(self):
        # Declaring a secret or an extra that is not provisioned makes the agent
        # fail at compile time. This template requests none and constructs no
        # model client, so both lists are empty by derivation, not by default.
        assert _MANIFEST["requires"]["secrets"] == []
        assert _MANIFEST["requires"]["extras"] == []
        src_text = " ".join(
            p.read_text(encoding="utf-8") for p in (_ROOT / "src").rglob("*.py") if "examples" not in p.parts
        )
        assert "ctx.secrets.require(" not in src_text

    def test_cfg_05_max_retry_within_framework_ceiling(self):
        max_retry = _RUNTIME["max_retry"]
        assert isinstance(max_retry, int)
        assert 0 <= max_retry < 10  # AgentBaseGraph MAX_RETRY_CEILING

    def test_timeout_uses_the_runtime_key_name(self):
        assert "timeout_s" in _RUNTIME
        assert "timeout_seconds" not in _RUNTIME

    def test_hitl_is_not_enabled(self):
        # This template declares no human-in-the-loop handling at all.
        assert (_RUNTIME.get("hitl") or {}).get("enabled", False) is False
        assert LoyaltyOfferQAGraphNode.propagate_hitl is False


class TestRetrievalBlock:
    def test_cfg_06_retrieval_block_matches_the_module_defaults(self):
        # The fallback defaults mirror the shipped file, so behaviour does not
        # change shape between the two paths.
        retrieval = _RUNTIME["retrieval"]
        assert retrieval["top_k"] == DEFAULT_RETRIEVAL["top_k"]
        assert retrieval["score_threshold"] == DEFAULT_RETRIEVAL["score_threshold"]
        assert retrieval["kb_path"] == DEFAULT_RETRIEVAL["kb_path"]
        assert (_ROOT / retrieval["kb_path"]).is_file()

    def test_retrieval_is_not_declared_in_the_manifest(self):
        # Runtime tuning lives in config/config.yaml. A copy in the manifest
        # would be read by nobody.
        assert "retrieval" not in _MANIFEST

    def test_cfg_07_parent_config_forwards_the_runtime_retrieval_block(self):
        cfg = LoyaltyOfferQAGraphNode()._parent_config()
        forwarded = cfg["configurable"]["retrieval"]
        assert forwarded, "_parent_config() must never forward an empty retrieval block"
        assert forwarded["top_k"] == _RUNTIME["retrieval"]["top_k"]
        assert forwarded["score_threshold"] == _RUNTIME["retrieval"]["score_threshold"]
        assert forwarded["kb_path"] == _RUNTIME["retrieval"]["kb_path"]

    def test_a_declared_value_reaches_the_inner_graph(self):
        # The end-to-end check that matters: a value declared in the runtime
        # config, handed to the agent the way the registry hands it over,
        # arrives in the state field the domain nodes read.
        declared = {"top_k": 7, "score_threshold": 0.5, "kb_path": "config/kb/loyalty_kb.json"}
        agent = LoyaltyOfferQAAgent(config={"retrieval": declared})
        seeded = from_json(agent._extra_initial_state()["retrieval_config"], {})
        assert seeded["top_k"] == 7
        assert seeded["score_threshold"] == 0.5


class TestSeededKnowledgeBase:
    def _entries(self):
        path = _ROOT / _RUNTIME["retrieval"]["kb_path"]
        return json.loads(path.read_text(encoding="utf-8"))

    def test_cfg_08_kb_is_a_well_formed_entry_list(self):
        entries = self._entries()
        assert isinstance(entries, list)
        assert len(entries) >= 5, "seeded KB must carry a usable corpus"
        for entry in entries:
            assert set(entry.keys()) == {"id", "title", "category", "source", "tags", "content"}
            assert entry["id"] and entry["title"] and entry["content"]

    def test_kb_ids_are_unique(self):
        ids = [e["id"] for e in self._entries()]
        assert len(ids) == len(set(ids))

    def test_kb_categories_are_the_documented_three(self):
        categories = {e["category"] for e in self._entries()}
        assert categories <= {"program_rules", "offer_catalog", "redemption_catalog"}
