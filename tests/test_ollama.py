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

    @requires_ollama
    @requires_model
    def test_score_does_not_regress(self, tmp_path):
        """best_score should be >= baseline after optimization.

        If the model produces identical scores throughout, the test is skipped
        rather than failed — a flat score line isn't a regression.
        """
        from aevyra_reflex.optimizer import PromptOptimizer, OptimizerConfig

        try:
            from aevyra_verdict import Dataset, RougeScore
            from aevyra_verdict.dataset import Conversation, Message
        except ImportError:
            pytest.skip("aevyra_verdict not installed")

        conversations = [
            Conversation(
                messages=[Message(role="user", content="Summarise in one sentence: The cat sat on the mat and looked out the window.")],
                ideal="A cat sat on the mat and gazed outside.",
            ),
            Conversation(
                messages=[Message(role="user", content="Summarise in one sentence: The dog ran through the park chasing a red ball.")],
                ideal="A dog chased a ball through the park.",
            ),
            Conversation(
                messages=[Message(role="user", content="Summarise in one sentence: The chef prepared a delicious meal using fresh ingredients from the garden.")],
                ideal="A chef cooked a fresh meal from garden ingredients.",
            ),
            Conversation(
                messages=[Message(role="user", content="Summarise in one sentence: The students studied late into the night before their final exams.")],
                ideal="Students studied late at night before exams.",
            ),
            Conversation(
                messages=[Message(role="user", content="Summarise in one sentence: The rain fell heavily on the city streets washing away the dust.")],
                ideal="Heavy rain washed the dust from city streets.",
            ),
            Conversation(
                messages=[Message(role="user", content="Summarise in one sentence: The engineer fixed the broken pipeline before the water pressure dropped too low.")],
                ideal="An engineer repaired the pipeline before pressure dropped.",
            ),
        ]
        dataset = Dataset(conversations=conversations)

        config = OptimizerConfig(
            strategy="iterative",
            max_iterations=1,
            score_threshold=0.99,   # run all iters
            reasoning_model="llama3.2:1b",
            reasoning_provider="ollama",
            reasoning_base_url=OLLAMA_BASE_URL,
        )

        result = (
            PromptOptimizer(config=config)
            .set_dataset(dataset)
            .add_provider("local", MODEL, base_url=OLLAMA_BASE_URL)
            .add_metric(RougeScore())
            .run("Summarise the sentence concisely.")
        )

        baseline = result.baseline.mean_score
        best = result.best_score

        if baseline == best and all(it.score == baseline for it in result.iterations):
            pytest.skip(
                f"Score flat across all iterations ({baseline:.3f}) — "
                "model may be deterministic on this dataset; skipping regression check."
            )

        assert best >= baseline, (
            f"Score regressed: baseline={baseline:.3f}, best={best:.3f}"
        )


# ---------------------------------------------------------------------------
# 4. Auto strategy end-to-end
# ---------------------------------------------------------------------------

class TestOllamaAutoStrategy:
    """Full optimizer.run() with strategy='auto' using a live Ollama instance."""

    @requires_ollama
    @requires_model
    def test_auto_strategy_completes(self, tmp_path):
        """auto strategy runs baseline → phases → verify without crashing."""
        from aevyra_reflex.optimizer import PromptOptimizer, OptimizerConfig
        from aevyra_reflex.run_store import RunStore

        try:
            from aevyra_verdict import Dataset, ExactMatch
            from aevyra_verdict.dataset import Conversation, Message
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
            strategy="auto",
            max_iterations=2,          # keep it short
            score_threshold=0.99,      # won't converge — just run iterations
            val_ratio=0.0,             # tiny dataset, skip val split
            reasoning_model="llama3.2:1b",
            reasoning_provider="ollama",
            reasoning_base_url=OLLAMA_BASE_URL,
        )

        result = (
            PromptOptimizer(config=config)
            .set_dataset(dataset)
            .add_provider("local", MODEL, base_url=OLLAMA_BASE_URL)
            .add_metric(ExactMatch())
            .run(
                "Classify the sentiment as positive, negative, or neutral. Reply with one word only.",
                run_store=store,
            )
        )

        assert result.best_prompt is not None
        assert result.baseline is not None
        assert result.final is not None
        assert len(result.iterations) >= 1
        assert 0.0 <= result.best_score <= 1.0

    @requires_ollama
    @requires_model
    def test_auto_strategy_persists_strategy_state(self, tmp_path):
        """auto strategy saves strategy_state to checkpoint so resume is possible."""
        from aevyra_reflex.optimizer import PromptOptimizer, OptimizerConfig
        from aevyra_reflex.run_store import RunStore

        try:
            from aevyra_verdict import Dataset, ExactMatch
            from aevyra_verdict.dataset import Conversation, Message
        except ImportError:
            pytest.skip("aevyra_verdict not installed")

        conversations = [
            Conversation(
                messages=[Message(role="user", content="Great product!")],
                ideal="positive",
            ),
            Conversation(
                messages=[Message(role="user", content="Terrible experience.")],
                ideal="negative",
            ),
        ]
        dataset = Dataset(conversations=conversations)
        store = RunStore(root=tmp_path / ".reflex")

        config = OptimizerConfig(
            strategy="auto",
            max_iterations=1,
            score_threshold=0.99,
            val_ratio=0.0,
            reasoning_model="llama3.2:1b",
            reasoning_provider="ollama",
            reasoning_base_url=OLLAMA_BASE_URL,
        )

        (
            PromptOptimizer(config=config)
            .set_dataset(dataset)
            .add_provider("local", MODEL, base_url=OLLAMA_BASE_URL)
            .add_metric(ExactMatch())
            .run(
                "Classify sentiment. Reply with one word: positive, negative, or neutral.",
                run_store=store,
            )
        )

        run = store.get_latest_run()
        assert run is not None
        checkpoint = run.load_checkpoint()
        # After at least 1 iteration, strategy_state should be populated
        assert checkpoint is not None
        assert isinstance(checkpoint.strategy_state, dict)
        assert len(checkpoint.strategy_state) > 0, (
            "strategy_state should be saved to checkpoint by auto strategy"
        )
