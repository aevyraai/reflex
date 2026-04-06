# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for agent backend resolution and multi-backend support."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from aevyra_reflex.agent import (
    LLM,
    Agent,  # backward compat alias
    _AnthropicBackend,
    _OllamaBackend,
    _OpenAIBackend,
    _resolve_agent_backend,
)


class TestResolveAgentBackend(unittest.TestCase):
    """Test backend resolution from provider/model/base_url hints."""

    # --- Explicit provider ---

    def test_explicit_anthropic(self):
        backend = _resolve_agent_backend("claude-sonnet-4-20250514", provider="anthropic")
        self.assertIsInstance(backend, _AnthropicBackend)
        self.assertEqual(backend.model, "claude-sonnet-4-20250514")

    def test_explicit_ollama(self):
        backend = _resolve_agent_backend("llama3.1:70b", provider="ollama")
        self.assertIsInstance(backend, _OllamaBackend)
        self.assertEqual(backend.model, "llama3.1:70b")
        self.assertEqual(backend.base_url, "http://localhost:11434")

    def test_explicit_ollama_custom_url(self):
        backend = _resolve_agent_backend(
            "llama3.1", provider="ollama", base_url="http://gpu-box:11434"
        )
        self.assertIsInstance(backend, _OllamaBackend)
        self.assertEqual(backend.base_url, "http://gpu-box:11434")

    def test_explicit_openai(self):
        backend = _resolve_agent_backend(
            "gpt-4o", provider="openai", api_key="sk-test"
        )
        self.assertIsInstance(backend, _OpenAIBackend)
        self.assertEqual(backend.model, "gpt-4o")

    def test_explicit_openai_with_base_url(self):
        backend = _resolve_agent_backend(
            "my-model", provider="openai",
            base_url="http://localhost:8000/v1", api_key="dummy"
        )
        self.assertIsInstance(backend, _OpenAIBackend)
        self.assertEqual(backend._base_url, "http://localhost:8000/v1")

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "or-key-123"})
    def test_explicit_alias_openrouter(self):
        backend = _resolve_agent_backend(
            "meta-llama/llama-3.1-70b-instruct", provider="openrouter"
        )
        self.assertIsInstance(backend, _OpenAIBackend)
        self.assertIn("openrouter.ai", backend._base_url)

    @patch.dict("os.environ", {"TOGETHER_API_KEY": "tog-key"})
    def test_explicit_alias_together(self):
        backend = _resolve_agent_backend(
            "meta-llama/Llama-3.1-8B-Instruct", provider="together"
        )
        self.assertIsInstance(backend, _OpenAIBackend)
        self.assertIn("together.xyz", backend._base_url)

    # --- Model name heuristics ---

    def test_heuristic_claude_model(self):
        backend = _resolve_agent_backend("claude-sonnet-4-20250514")
        self.assertIsInstance(backend, _AnthropicBackend)

    def test_heuristic_claude_opus(self):
        backend = _resolve_agent_backend("claude-opus-4-20250514")
        self.assertIsInstance(backend, _AnthropicBackend)

    def test_heuristic_bare_model_name_is_ollama(self):
        """No slash in model name + no base_url → Ollama."""
        backend = _resolve_agent_backend("llama3.1:8b")
        self.assertIsInstance(backend, _OllamaBackend)

    def test_heuristic_bare_model_no_tag(self):
        backend = _resolve_agent_backend("mistral")
        self.assertIsInstance(backend, _OllamaBackend)

    def test_heuristic_slashed_model_is_openai(self):
        """Slashed model with no provider hint → OpenAI-compatible."""
        backend = _resolve_agent_backend("meta-llama/llama-3.1-70b-instruct")
        self.assertIsInstance(backend, _OpenAIBackend)

    # --- base_url heuristics ---

    def test_localhost_11434_is_ollama(self):
        backend = _resolve_agent_backend(
            "my-model", base_url="http://localhost:11434"
        )
        self.assertIsInstance(backend, _OllamaBackend)

    def test_127_0_0_1_11434_is_ollama(self):
        backend = _resolve_agent_backend(
            "my-model", base_url="http://127.0.0.1:11434"
        )
        self.assertIsInstance(backend, _OllamaBackend)

    def test_other_base_url_is_openai(self):
        backend = _resolve_agent_backend(
            "my-model", base_url="http://localhost:8000/v1"
        )
        self.assertIsInstance(backend, _OpenAIBackend)

    # --- max_tokens propagation ---

    def test_max_tokens_propagated(self):
        backend = _resolve_agent_backend("claude-sonnet-4-20250514", max_tokens=8192)
        self.assertEqual(backend.max_tokens, 8192)

    def test_max_tokens_ollama(self):
        backend = _resolve_agent_backend("llama3.1", max_tokens=2048)
        self.assertIsInstance(backend, _OllamaBackend)
        self.assertEqual(backend.max_tokens, 2048)


class TestLLMInit(unittest.TestCase):
    """Test LLM class initialization with different backends."""

    @patch("aevyra_reflex.agent._resolve_agent_backend")
    def test_default_uses_claude(self, mock_resolve):
        mock_resolve.return_value = MagicMock()
        llm = LLM()
        mock_resolve.assert_called_once_with(
            "claude-sonnet-4-20250514", 4096,
            provider=None, api_key=None, base_url=None,
        )

    @patch("aevyra_reflex.agent._resolve_agent_backend")
    def test_with_provider(self, mock_resolve):
        mock_resolve.return_value = MagicMock()
        llm = LLM(model="llama3.1", provider="ollama")
        mock_resolve.assert_called_once_with(
            "llama3.1", 4096,
            provider="ollama", api_key=None, base_url=None,
        )

    @patch("aevyra_reflex.agent._resolve_agent_backend")
    def test_with_all_kwargs(self, mock_resolve):
        mock_resolve.return_value = MagicMock()
        llm = LLM(
            model="gpt-4o", max_tokens=8192,
            provider="openai", api_key="sk-test",
            base_url="http://my-endpoint/v1",
        )
        mock_resolve.assert_called_once_with(
            "gpt-4o", 8192,
            provider="openai", api_key="sk-test",
            base_url="http://my-endpoint/v1",
        )

    @patch("aevyra_reflex.agent._resolve_agent_backend")
    def test_generate_delegates_to_backend(self, mock_resolve):
        mock_backend = MagicMock()
        mock_backend.generate.return_value = "Hello!"
        mock_resolve.return_value = mock_backend

        llm = LLM()
        result = llm.generate("test prompt", temperature=0.5)

        mock_backend.generate.assert_called_once_with("test prompt", temperature=0.5)
        self.assertEqual(result, "Hello!")

    @patch("aevyra_reflex.agent._resolve_agent_backend")
    def test_agent_alias_is_llm(self, mock_resolve):
        """Agent is a backward-compat alias for LLM."""
        mock_resolve.return_value = MagicMock()
        self.assertIs(Agent, LLM)
        agent = Agent()
        self.assertIsInstance(agent, LLM)


class TestOptimizerConfigReasoningFields(unittest.TestCase):
    """Test that OptimizerConfig passes reasoning model fields through."""

    def test_defaults_to_none(self):
        from aevyra_reflex.optimizer import OptimizerConfig
        config = OptimizerConfig()
        self.assertIsNone(config.reasoning_provider)
        self.assertIsNone(config.reasoning_api_key)
        self.assertIsNone(config.reasoning_base_url)

    def test_can_set_reasoning_fields(self):
        from aevyra_reflex.optimizer import OptimizerConfig
        config = OptimizerConfig(
            reasoning_model="llama3.1:70b",
            reasoning_provider="ollama",
            reasoning_base_url="http://gpu-box:11434",
        )
        self.assertEqual(config.reasoning_model, "llama3.1:70b")
        self.assertEqual(config.reasoning_provider, "ollama")
        self.assertEqual(config.reasoning_base_url, "http://gpu-box:11434")
        self.assertIsNone(config.reasoning_api_key)


class TestAnthropicBackend(unittest.TestCase):
    """Test _AnthropicBackend."""

    def test_generate_calls_messages_create(self):
        backend = _AnthropicBackend("claude-sonnet-4-20250514", 4096)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="result")]
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        result = backend.generate("hello", temperature=0.5)

        mock_client.messages.create.assert_called_once_with(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            temperature=0.5,
            messages=[{"role": "user", "content": "hello"}],
        )
        self.assertEqual(result, "result")


class TestOpenAIBackend(unittest.TestCase):
    """Test _OpenAIBackend."""

    def test_generate_calls_chat_completions(self):
        backend = _OpenAIBackend("gpt-4o", 4096, api_key="sk-test")
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "result"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        backend._client = mock_client

        result = backend.generate("hello", temperature=0.7)

        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            max_tokens=4096,
            temperature=0.7,
            messages=[{"role": "user", "content": "hello"}],
        )
        self.assertEqual(result, "result")


class TestOllamaBackend(unittest.TestCase):
    """Test _OllamaBackend."""

    def test_default_base_url(self):
        backend = _OllamaBackend("llama3.1", 4096)
        self.assertEqual(backend.base_url, "http://localhost:11434")

    def test_custom_base_url_strips_trailing_slash(self):
        backend = _OllamaBackend("llama3.1", 4096, base_url="http://gpu:11434/")
        self.assertEqual(backend.base_url, "http://gpu:11434")

    @patch("urllib.request.urlopen")
    def test_generate_calls_ollama_api(self, mock_urlopen):
        import json

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "hello back"}).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        backend = _OllamaBackend("llama3.1", 4096)
        result = backend.generate("hello", temperature=0.5)

        self.assertEqual(result, "hello back")
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        self.assertIn("/api/generate", req.full_url)

    @patch("urllib.request.urlopen")
    def test_generate_connection_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        backend = _OllamaBackend("llama3.1", 4096)
        with self.assertRaises(ConnectionError) as ctx:
            backend.generate("hello")
        self.assertIn("ollama serve", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
