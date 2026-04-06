"""Ollama integration tests.

Requires a running Ollama server with llama3.2:1b pulled:

    ollama pull llama3.2:1b

Run with:

    pytest tests/test_ollama.py -v
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = "llama3.2:1b"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    """Return True if Ollama server is reachable."""
    try:
        urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return True
    except Exception:
        return False


def _model_available() -> bool:
    """Return True if MODEL is pulled in Ollama."""
    try:
        resp = urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        data = json.loads(resp.read())
        names = [m.get("name", "") for m in data.get("models", [])]
        return any(MODEL in n for n in names)
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama server not available at " + OLLAMA_BASE_URL,
)

requires_model = pytest.mark.skipif(
    not _model_available(),
    reason=f"Model {MODEL} not pulled in Ollama",
)


# ---------------------------------------------------------------------------
# 1. Provider resolution
# ---------------------------------------------------------------------------

class TestOllamaProviderResolution:
    """local/<model> resolves to the Ollama backend and can generate text."""

    @requires_ollama
    @requires_model
    def test_llm_generate_returns_nonempty_string(self):
        """LLM('llama3.2:1b', provider='ollama') generates a response."""
        from aevyra_reflex.agent import LLM

        llm = LLM(model=MODEL, provider="ollama", base_url=OLLAMA_BASE_URL)
        response = llm.generate("Reply with the single word: hello", temperature=0.0)
        assert isinstance(response, str)
        assert len(response.strip()) > 0

    @requires_ollama
    @requires_model
    def test_local_prefix_resolves_to_ollama(self):
        """'local/llama3.2:1b' shorthand resolves correctly."""
        from aevyra_reflex.optimizer import PromptOptimizer, OptimizerConfig

        opt = PromptOptimizer(config=OptimizerConfig())
        opt.add_provider("local", MODEL)
        # Provider list should have one entry with provider_name='local'
        assert len(opt._providers) == 1
        assert opt._providers[0]["provider_name"] == "local"
        assert opt._providers[0]["model"] == MODEL

    @requires_ollama
    @requires_model
    def test_tokens_used_populated_after_generate(self):
        """tokens_used is updated after a generate call."""
        from aevyra_reflex.agent import LLM

        llm = LLM(model=MODEL, provider="ollama", base_url=OLLAMA_BASE_URL)
        before = llm.tokens_used
        llm.generate("Say one word.", temperature=0.0)
        assert llm.tokens_used > before


# ---------------------------------------------------------------------------
# 2. Parallel worker detection
# ---------------------------------------------------------------------------

class TestOllamaParallelDetection:
    """_check_parallel_config() adjusts max_workers based on OLLAMA_NUM_PARALLEL."""

    def _make_ollama_optimizer(self, strategy="structural", max_workers=4):
        from aevyra_reflex.optimizer import PromptOptimizer, OptimizerConfig
        opt = PromptOptimizer(config=OptimizerConfig(
            strategy=strategy,
            max_workers=max_workers,
        ))
        opt._providers = [{"provider_name": "local", "model": MODEL, "label": MODEL}]
        opt._metrics = []
        return opt

    def test_no_env_var_falls_back_to_sequential(self, monkeypatch):
        """Without OLLAMA_NUM_PARALLEL, max_workers is capped to 1."""
        monkeypatch.delenv("OLLAMA_NUM_PARALLEL", raising=False)
        opt = self._make_ollama_optimizer(strategy="structural", max_workers=4)
        opt._check_parallel_config()
        assert opt.config.max_workers == 1

    def test_env_var_set_caps_workers(self, monkeypatch):
        """OLLAMA_NUM_PARALLEL=2 caps max_workers to 2 when it was higher."""
        monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "2")
        opt = self._make_ollama_optimizer(strategy="structural", max_workers=4)
        opt._check_parallel_config()
        assert opt.config.max_workers == 2

    def test_env_var_set_does_not_raise_workers(self, monkeypatch):
        """OLLAMA_NUM_PARALLEL=4 does not raise max_workers if already lower."""
        monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "4")
        opt = self._make_ollama_optimizer(strategy="structural", max_workers=2)
        opt._check_parallel_config()
        assert opt.config.max_workers == 2

    def test_iterative_strategy_skips_parallel_check(self, monkeypatch):
        """Iterative strategy doesn't need parallelism — no adjustment."""
        monkeypatch.delenv("OLLAMA_NUM_PARALLEL", raising=False)
        opt = self._make_ollama_optimizer(strategy="iterative", max_workers=4)
        opt._check_parallel_config()
        # Iterative is sequential by design — max_workers unchanged
        assert opt.config.max_workers == 4

    def test_non_ollama_provider_skips_check(self, monkeypatch):
        """Non-Ollama providers are not affected."""
        monkeypatch.delenv("OLLAMA_NUM_PARALLEL", raising=False)
        from aevyra_reflex.optimizer import PromptOptimizer, OptimizerConfig
        opt = PromptOptimizer(config=OptimizerConfig(strategy="structural", max_workers=4))
        opt._providers = [{"provider_name": "openai", "model": "gpt-4o-mini", "label": "test"}]
        opt._metrics = []
        opt._check_parallel_config()
        assert opt.config.max_workers == 4


# ---------------------------------------------------------------------------
# 3. End-to-end mini run
# ---------------------------------------------------------------------------

class TestOllamaEndToEnd:
    """A real optimizer.run() using llama3.2:1b as the target model."""

    @requires_ollama
    @requires_model
    def test_mini_run_completes(self, tmp_path, monkeypatch):
        """Full baseline → optimize → verify loop with a 3-sample dataset."""
        from unittest.mock import MagicMock, patch

        from aevyra_reflex.optimizer import PromptOptimizer, OptimizerConfig
        from aevyra_reflex.run_store import RunStore

        # Tiny in-memory dataset: 3 sentiment classification samples
        try:
            from aevyra_verdict import Dataset
            from aevyra_verdict.dataset import Conversation, Message
            from aevyra_verdict import ExactMatch
        except ImportError:
            pytest.skip("aevyra_verdict not installed")

        conversations = [
            Conversation(
                messages=[Message(role="user", content="The movie was great!")],
                ideal="positive",
            ),
            Conversation(
                messages=[Message(role="user", content="I hated every minute.")],
                ideal="negative",
            ),
            Conversation(
                messages=[Message(role="user", content="It was okay, nothing special.")],
                ideal="neutral",
            ),
        ]
        dataset = Dataset(conversations=conversations)

        store = RunStore(root=tmp_path / ".reflex")

        config = OptimizerConfig(
            strategy="iterative",
            max_iterations=1,
            score_threshold=0.99,   # won't converge — just run 1 iter
            reasoning_model="llama3.2:1b",
            reasoning_provider="ollama",
            reasoning_base_url=OLLAMA_BASE_URL,
        )

        opt = (
            PromptOptimizer(config=config)
            .set_dataset(dataset)
            .add_provider("local", MODEL, base_url=OLLAMA_BASE_URL)
            .add_metric(ExactMatch())
        )

        result = opt.run(
            "Classify the sentiment as positive, negative, or neutral. "
            "Reply with one word only.",
            run_store=store,
        )

        assert result.best_prompt is not None
        assert result.baseline is not None
        assert result.final is not None
        assert len(result.iterations) >= 1
        assert 0.0 <= result.best_score <= 1.0

    @requires_ollama
    @requires_model
    def test_mini_run_persists_to_store(self, tmp_path):
        """Run is saved to the run store with baseline and at least 1 iteration."""
        from unittest.mock import patch

        from aevyra_reflex.optimizer import PromptOptimizer, OptimizerConfig
        from aevyra_reflex.run_store import RunStore

        try:
            from aevyra_verdict import Dataset, ExactMatch
            from aevyra_verdict.dataset import Conversation, Message
        except ImportError:
            pytest.skip("aevyra_verdict not installed")

        conversations = [
            Conversation(
                messages=[Message(role="user", content="Summarise: The cat sat on the mat.")],
                ideal="cat mat",
            ),
        ]
        dataset = Dataset(conversations=conversations)
        store = RunStore(root=tmp_path / ".reflex")

        config = OptimizerConfig(
            strategy="iterative",
            max_iterations=1,
            score_threshold=0.99,
            reasoning_model="llama3.2:1b",
            reasoning_provider="ollama",
            reasoning_base_url=OLLAMA_BASE_URL,
        )

        opt = (
            PromptOptimizer(config=config)
            .set_dataset(dataset)
            .add_provider("local", MODEL, base_url=OLLAMA_BASE_URL)
            .add_metric(ExactMatch())
        )

        opt.run("You are a helpful assistant.", run_store=store)

        run = store.get_latest_run()
        assert run is not None
        assert run.load_baseline() is not None
        iters = run.load_iterations()
        assert len(iters) >= 1
