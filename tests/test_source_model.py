"""Tests for the --source-model / source_model migration feature.

Covers:
- SOURCE_MODEL_CONTEXT template formatting
- LLM.source_model stored on construction
- Context prepended to every generate() call when source_model is set
- Context NOT prepended when source_model is None
- OptimizerConfig.source_model field persisted
- LLM construction from optimizer passes source_model through
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 1. Prompt template
# ---------------------------------------------------------------------------

class TestSourceModelContextTemplate:
    """SOURCE_MODEL_CONTEXT renders correctly."""

    def test_formats_source_model(self):
        from aevyra_reflex.prompts import SOURCE_MODEL_CONTEXT

        rendered = SOURCE_MODEL_CONTEXT.format(source_model="gpt-4o")
        assert "gpt-4o" in rendered

    def test_contains_migration_guidance(self):
        from aevyra_reflex.prompts import SOURCE_MODEL_CONTEXT

        rendered = SOURCE_MODEL_CONTEXT.format(source_model="claude-sonnet")
        # Should mention model-family idioms (XML tags, Markdown, etc.)
        assert "XML" in rendered or "Markdown" in rendered

    def test_ends_with_separator(self):
        """The context block ends with a separator so it doesn't bleed into the prompt."""
        from aevyra_reflex.prompts import SOURCE_MODEL_CONTEXT

        rendered = SOURCE_MODEL_CONTEXT.format(source_model="llama3")
        # Should end with the horizontal rule separator
        assert rendered.rstrip().endswith("---")


# ---------------------------------------------------------------------------
# 2. LLM class
# ---------------------------------------------------------------------------

class TestLLMSourceModel:
    """LLM stores source_model and injects it into generate()."""

    def _make_llm(self, source_model=None):
        """Build an LLM with a mock backend."""
        with patch("aevyra_reflex.agent._resolve_agent_backend") as mock_resolve:
            mock_backend = MagicMock()
            mock_backend.tokens_used = 0
            mock_backend.generate.return_value = "mocked response"
            mock_resolve.return_value = mock_backend
            from aevyra_reflex.agent import LLM
            llm = LLM(
                model="mock-model",
                provider="openai",
                source_model=source_model,
            )
        return llm, mock_backend

    def test_source_model_stored(self):
        llm, _ = self._make_llm(source_model="claude-sonnet")
        assert llm.source_model == "claude-sonnet"

    def test_source_model_defaults_to_none(self):
        llm, _ = self._make_llm()
        assert llm.source_model is None

    def test_context_prepended_when_source_model_set(self):
        """When source_model is set, generate() prepends the context block."""
        llm, backend = self._make_llm(source_model="gpt-4o")

        llm.generate("Improve this prompt.")

        # The actual prompt sent to the backend should contain migration context
        actual_prompt = backend.generate.call_args[0][0]
        assert "gpt-4o" in actual_prompt
        assert "Migration context" in actual_prompt
        assert "Improve this prompt." in actual_prompt

    def test_context_not_prepended_when_source_model_none(self):
        """When source_model is None, generate() passes the prompt unchanged."""
        llm, backend = self._make_llm(source_model=None)

        original_prompt = "Improve this prompt."
        llm.generate(original_prompt)

        actual_prompt = backend.generate.call_args[0][0]
        assert actual_prompt == original_prompt

    def test_original_prompt_preserved_at_end(self):
        """The user's prompt is preserved (not replaced) when context is prepended."""
        llm, backend = self._make_llm(source_model="llama3")
        user_prompt = "Some task-specific prompt text."
        llm.generate(user_prompt)

        actual_prompt = backend.generate.call_args[0][0]
        # User prompt should appear after the context block
        assert actual_prompt.endswith(user_prompt)

    def test_temperature_passed_through(self):
        llm, backend = self._make_llm(source_model="gpt-4o")
        llm.generate("hello", temperature=0.0)
        backend.generate.assert_called_once()
        _, kwargs = backend.generate.call_args
        assert kwargs.get("temperature") == 0.0

    def test_tokens_accumulated(self):
        llm, backend = self._make_llm(source_model="claude-3")
        backend.tokens_used = 0

        # Simulate tokens accumulating across calls
        def fake_generate(prompt, *, temperature=1.0):
            backend.tokens_used += 100
            return "ok"

        backend.generate.side_effect = fake_generate
        llm.generate("first call")
        assert llm.tokens_used == 100
        llm.generate("second call")
        assert llm.tokens_used == 200


# ---------------------------------------------------------------------------
# 3. OptimizerConfig
# ---------------------------------------------------------------------------

class TestOptimizerConfigSourceModel:
    """source_model field on OptimizerConfig."""

    def test_default_is_none(self):
        from aevyra_reflex.optimizer import OptimizerConfig

        config = OptimizerConfig()
        assert config.source_model is None

    def test_can_be_set(self):
        from aevyra_reflex.optimizer import OptimizerConfig

        config = OptimizerConfig(source_model="gpt-4o-mini")
        assert config.source_model == "gpt-4o-mini"

    def test_included_in_asdict(self):
        """source_model is serialised when the optimizer saves config."""
        from dataclasses import asdict
        from aevyra_reflex.optimizer import OptimizerConfig

        config = OptimizerConfig(source_model="claude-haiku")
        d = asdict(config)
        assert d["source_model"] == "claude-haiku"


# ---------------------------------------------------------------------------
# 4. Optimizer wires source_model into LLM
# ---------------------------------------------------------------------------

class TestOptimizerPassesSourceModel:
    """PromptOptimizer passes source_model from config to the LLM constructor."""

    def test_source_model_passed_to_llm(self):
        """When OptimizerConfig.source_model is set, the LLM is built with it."""
        from aevyra_reflex.optimizer import OptimizerConfig, PromptOptimizer

        config = OptimizerConfig(
            strategy="iterative",
            source_model="gpt-4o",
        )
        opt = PromptOptimizer(config=config)

        captured_kwargs: dict = {}

        class _CaptureLLM:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)
                self.tokens_used = 0
                self.model = kwargs.get("model", "")

        strategy_mock = MagicMock()
        strategy_mock.run.return_value = MagicMock(
            iterations=[],
            best_prompt="p",
            best_score=0.9,
        )

        try:
            from aevyra_verdict import Dataset
            from aevyra_verdict.dataset import Conversation, Message
        except ImportError:
            import unittest
            raise unittest.SkipTest("aevyra_verdict not installed")

        dataset = Dataset(conversations=[
            Conversation(
                messages=[Message(role="user", content="hi")],
                ideal="hello",
            )
        ])
        opt.set_dataset(dataset)
        opt._providers = [{"provider_name": "mock", "model": "mock-model", "label": "mock"}]
        opt._metrics = [MagicMock()]

        with patch("aevyra_reflex.optimizer.LLM", _CaptureLLM), \
             patch("aevyra_reflex.strategies.get_strategy", return_value=lambda: strategy_mock):
            try:
                opt.run("Initial prompt.")
            except Exception:
                pass  # We only care that LLM was constructed correctly

        assert captured_kwargs.get("source_model") == "gpt-4o"
