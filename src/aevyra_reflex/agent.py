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

"""LLM agent for prompt optimization tasks.

Handles all LLM calls that the optimization strategies need: summarizing
datasets, diagnosing failures, proposing revised prompts, generating
candidate prompts, mutating champions, and judging pairwise duels.

Supports multiple backends:
    - Anthropic (Claude) — default, best results
    - OpenAI-compatible APIs — OpenAI, OpenRouter, Together, etc.
    - Local models — Ollama, vLLM, or any OpenAI-compatible local server
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent backends
# ---------------------------------------------------------------------------

class _AnthropicBackend:
    """Calls Claude via the Anthropic SDK."""

    def __init__(self, model: str, max_tokens: int, api_key: str | None = None):
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError(
                    "The Anthropic reasoning backend requires the anthropic package. "
                    "Install it with: pip install aevyra-reflex  "
                    "(anthropic is included by default)"
                )
            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = Anthropic(**kwargs)
        return self._client

    def generate(self, prompt: str, *, temperature: float = 1.0) -> str:
        from anthropic import OverloadedError, RateLimitError

        delays = [5, 10, 20, 40, 60]
        for attempt, delay in enumerate(delays + [None]):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except (OverloadedError, RateLimitError) as exc:
                if delay is None:
                    raise
                logger.warning(
                    "Reasoning model %s (attempt %d/%d): %s — retrying in %ds",
                    self.model, attempt + 1, len(delays), exc.__class__.__name__, delay,
                )
                time.sleep(delay)

        if hasattr(response, "usage") and response.usage:
            self.tokens_used = getattr(self, "tokens_used", 0) + (
                response.usage.input_tokens + response.usage.output_tokens
            )
        return response.content[0].text


class _OpenAIBackend:
    """Calls any OpenAI-compatible API (OpenAI, OpenRouter, Together, Ollama, vLLM)."""

    def __init__(
        self,
        model: str,
        max_tokens: int,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._base_url = base_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "Using an OpenAI-compatible reasoning model requires the openai package. "
                    "Install it with: pip install aevyra-reflex[openai]"
                )
            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def generate(self, prompt: str, *, temperature: float = 1.0) -> str:
        from openai import APIStatusError

        delays = [5, 10, 20, 40, 60]
        for attempt, delay in enumerate(delays + [None]):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except APIStatusError as exc:
                if exc.status_code not in (429, 529) or delay is None:
                    raise
                logger.warning(
                    "Reasoning model %s (attempt %d/%d): HTTP %d — retrying in %ds",
                    self.model, attempt + 1, len(delays), exc.status_code, delay,
                )
                time.sleep(delay)

        if response.usage:
            self.tokens_used = getattr(self, "tokens_used", 0) + (
                response.usage.prompt_tokens + response.usage.completion_tokens
            )
        return response.choices[0].message.content or ""


class _OllamaBackend:
    """Calls Ollama directly via its REST API (no SDK required)."""

    def __init__(self, model: str, max_tokens: int, base_url: str = "http://localhost:11434"):
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")
        self._session = None

    def _get_session(self):
        return None  # we use urllib directly

    def generate(self, prompt: str, *, temperature: float = 1.0) -> str:
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": self.max_tokens,
            },
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
                self.tokens_used = getattr(self, "tokens_used", 0) + (
                    data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                )
                return data.get("response", "")
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                f"Is 'ollama serve' running? Error: {e}"
            ) from e


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

def _resolve_agent_backend(
    model: str,
    max_tokens: int = 4096,
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> _AnthropicBackend | _OpenAIBackend | _OllamaBackend:
    """Pick the right backend based on provider/model hints.

    Resolution order:
        1. Explicit provider kwarg: "anthropic", "openai", "ollama", or any
           alias in PROVIDER_ALIASES (e.g. "gemini", "openrouter", "groq")
        2. Model name heuristics:
           - "claude-*"         → Anthropic
           - "gemini-*"         → OpenAI-compat against Google's v1beta endpoint
           - "ollama/*" or no / → Ollama  (if base_url looks local or default)
           - everything else    → OpenAI-compatible
        3. base_url heuristics:
           - contains "localhost" or "127.0.0.1" + port 11434 → Ollama
           - any other base_url                                → OpenAI-compatible
    """
    import os
    from aevyra_reflex.optimizer import PROVIDER_ALIASES

    # Explicit provider
    if provider:
        p = provider.lower()
        if p == "anthropic":
            return _AnthropicBackend(model, max_tokens, api_key=api_key)
        if p == "ollama":
            url = base_url or "http://localhost:11434"
            return _OllamaBackend(model, max_tokens, base_url=url)
        # Check if it's a known alias
        if p in PROVIDER_ALIASES:
            alias = PROVIDER_ALIASES[p]
            resolved_url = base_url or alias["base_url"]
            resolved_key = api_key or os.environ.get(alias["env_key"]) or os.environ.get("OPENAI_API_KEY")
            return _OpenAIBackend(model, max_tokens, api_key=resolved_key, base_url=resolved_url)
        # Generic openai-compatible
        return _OpenAIBackend(model, max_tokens, api_key=api_key, base_url=base_url)

    # Heuristics from model name
    if model.startswith("claude"):
        return _AnthropicBackend(model, max_tokens, api_key=api_key)

    if model.startswith("gemini"):
        alias = PROVIDER_ALIASES["gemini"]
        resolved_key = api_key or os.environ.get(alias["env_key"])
        return _OpenAIBackend(model, max_tokens, api_key=resolved_key, base_url=alias["base_url"])

    # Check base_url for Ollama
    if base_url and ("localhost:11434" in base_url or "127.0.0.1:11434" in base_url):
        return _OllamaBackend(model, max_tokens, base_url=base_url)

    # No slash in model name + no base_url → likely Ollama (e.g. "llama3.1:8b")
    if "/" not in model and not base_url:
        return _OllamaBackend(model, max_tokens)

    # Default: OpenAI-compatible
    return _OpenAIBackend(model, max_tokens, api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# Public Agent class
# ---------------------------------------------------------------------------

class LLM:
    """LLM backend for prompt optimization tasks.

    This is the reasoning model that reflex (the agent) uses to analyze
    failures, propose rewrites, and recommend strategies. It is not itself
    an agent — reflex is the agent; this is one of its tools.

    Works with Claude (default), OpenAI-compatible APIs, or local models.

    Examples::

        # Claude (default)
        llm = LLM(model="claude-sonnet-4-20250514")

        # Gemini — via Google's OpenAI-compatible endpoint (GOOGLE_API_KEY)
        llm = LLM(model="gemini-2.0-flash")
        llm = LLM(model="gemini-2.5-pro", provider="gemini")

        # Local Ollama model
        llm = LLM(model="llama3.1:70b", provider="ollama")

        # OpenRouter
        llm = LLM(
            model="meta-llama/llama-3.1-70b-instruct",
            provider="openrouter",
        )

        # Any OpenAI-compatible endpoint
        llm = LLM(
            model="my-model",
            provider="openai",
            base_url="http://localhost:8000/v1",
        )
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        source_model: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.provider = provider
        self.source_model = source_model  # original model this prompt was written for
        self.tokens_used: int = 0  # cumulative reasoning tokens across all calls
        self._backend = _resolve_agent_backend(
            model, max_tokens,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
        )
        backend_name = type(self._backend).__name__
        logger.info(f"Reasoning model: {model} (backend: {backend_name})")

    def generate(self, prompt: str, *, temperature: float = 1.0) -> str:
        """Send a prompt to the agent and return the text response.

        If ``source_model`` was set at construction time, a migration context
        block is prepended to every prompt so the reasoning model knows which
        model family the prompt originated from.
        """
        if self.source_model:
            from aevyra_reflex.prompts import SOURCE_MODEL_CONTEXT
            prompt = SOURCE_MODEL_CONTEXT.format(source_model=self.source_model) + prompt
        before = getattr(self._backend, "tokens_used", 0)
        result = self._backend.generate(prompt, temperature=temperature)
        self.tokens_used += getattr(self._backend, "tokens_used", 0) - before
        return result

    def summarize_dataset(self, examples: list[str]) -> str:
        """Generate a concise summary of an eval dataset from sample inputs."""
        from aevyra_reflex.prompts import DATASET_SUMMARY_PROMPT

        formatted = "\n\n".join(f"- {ex}" for ex in examples)
        prompt = DATASET_SUMMARY_PROMPT.format(examples=formatted)
        return self.generate(prompt, temperature=0.3)

    def diagnose_and_revise(
        self,
        system_prompt: str,
        failing_samples: list[dict[str, Any]],
        *,
        label_free: bool = False,
    ) -> tuple[str, str, str]:
        """Analyze failures and propose a revised system prompt.

        Args:
            system_prompt: The current system prompt.
            failing_samples: List of dicts with keys: input, response, ideal, score.
            label_free: If True, injects a context block telling the reasoning model
                that scores come from an LLM judge (not reference comparison) and to
                focus on quality/instruction-following failures rather than correctness.

        Returns:
            (revised_prompt, reasoning, change_summary) — the new prompt, the
            agent's full analysis, and a one-liner summarising what changed.
        """
        from aevyra_reflex.prompts import DIAGNOSE_FAILURES_PROMPT, LABEL_FREE_EVAL_CONTEXT

        samples_text = _format_failing_samples(failing_samples)
        eval_context = LABEL_FREE_EVAL_CONTEXT if label_free else ""
        prompt = DIAGNOSE_FAILURES_PROMPT.format(
            system_prompt=system_prompt,
            failing_samples=samples_text,
            eval_context=eval_context,
        )
        response = self.generate(prompt, temperature=0.7)
        return (
            _extract_revised_prompt(response),
            response,
            _extract_change_summary(response),
        )

    def refine(
        self,
        system_prompt: str,
        iteration: int,
        score_trajectory: list[float],
        mean_score: float,
        target_score: float,
        failing_samples: list[dict[str, Any]],
        previous_reasoning: str,
        rewrite_log: list[dict[str, Any]] | None = None,
        *,
        label_free: bool = False,
    ) -> tuple[str, str, str]:
        """Refine a prompt based on ongoing iteration results.

        Args:
            rewrite_log: List of dicts with keys: iteration, score, delta,
                change_summary — the causal history of what was tried and what worked.
            label_free: If True, injects a context block telling the reasoning model
                that scores come from an LLM judge (not reference comparison) and to
                focus on quality/instruction-following failures rather than correctness.

        Returns:
            (revised_prompt, reasoning, change_summary)
        """
        from aevyra_reflex.prompts import REFINE_PROMPT, LABEL_FREE_EVAL_CONTEXT

        improved = len(score_trajectory) >= 2 and score_trajectory[-1] > score_trajectory[-2]
        trajectory_str = " → ".join(f"{s:.3f}" for s in score_trajectory)
        samples_text = _format_failing_samples(failing_samples)
        history_text = _format_rewrite_log(rewrite_log or [])
        eval_context = LABEL_FREE_EVAL_CONTEXT if label_free else ""

        prompt = REFINE_PROMPT.format(
            system_prompt=system_prompt,
            iteration=iteration,
            score_trajectory=trajectory_str,
            rewrite_history=history_text,
            mean_score=mean_score,
            target_score=target_score,
            failing_samples=samples_text,
            previous_reasoning=previous_reasoning,
            improved_text="improved" if improved else "not improved",
            eval_context=eval_context,
        )
        response = self.generate(prompt, temperature=0.7)
        return (
            _extract_revised_prompt(response),
            response,
            _extract_change_summary(response),
        )

    def generate_candidate(
        self,
        dataset_summary: str,
        sample_inputs: list[str],
        base_instruction: str,
        tip: str,
    ) -> str:
        """Generate a candidate system prompt for the PDO strategy."""
        from aevyra_reflex.prompts import GENERATE_CANDIDATE_PROMPT

        inputs_text = "\n".join(f"- {s}" for s in sample_inputs)
        prompt = GENERATE_CANDIDATE_PROMPT.format(
            dataset_summary=dataset_summary,
            sample_inputs=inputs_text,
            base_instruction=base_instruction or "(none provided)",
            tip=tip,
        )
        return self.generate(prompt, temperature=1.0).strip()

    def mutate_champion(
        self,
        champion_prompt: str,
        mutation_tip: str,
        sample_inputs: list[str],
    ) -> str:
        """Create a mutation of the current best-performing prompt."""
        from aevyra_reflex.prompts import MUTATE_CHAMPION_PROMPT

        inputs_text = "\n".join(f"- {s}" for s in sample_inputs[:3])
        prompt = MUTATE_CHAMPION_PROMPT.format(
            champion_prompt=champion_prompt,
            mutation_tip=mutation_tip,
            sample_inputs=inputs_text,
        )
        return self.generate(prompt, temperature=1.0).strip()

    # ------------------------------------------------------------------
    # Auto strategy methods
    # ------------------------------------------------------------------

    def recommend_axis(
        self,
        current_prompt: str,
        dataset_sample: list[str],
        phase_history: list[dict[str, Any]],
        axes_available: list[str],
        axes_used: list[str],
    ) -> str:
        """Recommend which optimization axis to apply next.

        Returns one of: 'structural', 'iterative', 'fewshot', 'pdo'.
        """
        from aevyra_reflex.prompts import RECOMMEND_AXIS_PROMPT

        sample_text = "\n".join(f"- {s}" for s in dataset_sample)
        history_text = _format_phase_history(phase_history)

        prompt = RECOMMEND_AXIS_PROMPT.format(
            current_prompt=current_prompt,
            sample_inputs=sample_text,
            phase_history=history_text or "(first phase — no history yet)",
            axes_used=", ".join(axes_used) if axes_used else "(none yet)",
            axes_available=", ".join(axes_available) if axes_available else "(all used — pick best to repeat)",
        )
        response = self.generate(prompt, temperature=0.3)

        # Parse the axis from JSON response
        try:
            data = json.loads(response)
            axis = data.get("axis", "").strip().lower()
            if axis in ("structural", "iterative", "fewshot", "pdo"):
                logger.info(f"  Agent recommends '{axis}': {data.get('reasoning', '')}")
                return axis
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: try to find axis name in response
        response_lower = response.lower()
        for candidate in ("structural", "iterative", "fewshot", "pdo"):
            if candidate in response_lower:
                logger.info(f"  Agent recommends '{candidate}' (extracted from text)")
                return candidate

        # Default fallback
        if axes_available:
            return axes_available[0]
        return "iterative"

    # ------------------------------------------------------------------
    # Few-shot strategy methods
    # ------------------------------------------------------------------

    def select_fewshot_examples(
        self,
        base_instruction: str,
        candidate_exemplars: list[dict[str, str]],
        max_examples: int = 5,
        selection_strategy: str = "diverse",
    ) -> tuple[list[dict[str, str]], str]:
        """Select the best few-shot examples from a pool of candidates.

        Returns:
            (selected_examples, refined_instruction)
        """
        from aevyra_reflex.prompts import SELECT_FEWSHOT_PROMPT

        candidates_text = _format_candidates(candidate_exemplars)
        prompt = SELECT_FEWSHOT_PROMPT.format(
            base_instruction=base_instruction,
            candidates=candidates_text,
            max_examples=max_examples,
            selection_strategy=selection_strategy,
        )
        response = self.generate(prompt, temperature=0.7)
        return _parse_fewshot_response(response, candidate_exemplars)

    def refine_fewshot(
        self,
        current_instruction: str,
        current_examples: list[dict[str, str]],
        candidate_exemplars: list[dict[str, str]],
        failing_samples: list[dict[str, Any]],
        max_examples: int = 5,
        score_trajectory: list[float] | None = None,
    ) -> tuple[list[dict[str, str]], str]:
        """Refine the few-shot example selection based on failures.

        Returns:
            (selected_examples, refined_instruction)
        """
        from aevyra_reflex.prompts import REFINE_FEWSHOT_PROMPT

        current_ex_text = _format_current_examples(current_examples)
        candidates_text = _format_candidates(candidate_exemplars)
        samples_text = _format_failing_samples(failing_samples)
        trajectory_str = " → ".join(f"{s:.3f}" for s in (score_trajectory or []))

        prompt = REFINE_FEWSHOT_PROMPT.format(
            current_instruction=current_instruction,
            current_examples=current_ex_text,
            score_trajectory=trajectory_str or "(first iteration)",
            failing_samples=samples_text,
            candidates=candidates_text,
            max_examples=max_examples,
        )
        response = self.generate(prompt, temperature=0.7)
        return _parse_fewshot_response(response, candidate_exemplars)

    # ------------------------------------------------------------------
    # Structural strategy methods
    # ------------------------------------------------------------------

    def analyze_prompt_structure(
        self,
        system_prompt: str,
        failing_samples: list[dict[str, Any]],
        structural_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Analyze the structural weaknesses of a system prompt."""
        from aevyra_reflex.prompts import ANALYZE_STRUCTURE_PROMPT

        samples_text = _format_failing_samples(failing_samples)
        history_text = _format_structural_history(structural_history or [])
        prompt = ANALYZE_STRUCTURE_PROMPT.format(
            system_prompt=system_prompt,
            failing_samples=samples_text,
            structural_history=history_text,
        )
        return self.generate(prompt, temperature=0.5)

    def restructure_prompt(
        self,
        current_prompt: str,
        transform_instruction: str,
        analysis: str,
    ) -> str:
        """Restructure a prompt using a specific structural transformation."""
        from aevyra_reflex.prompts import RESTRUCTURE_PROMPT

        prompt = RESTRUCTURE_PROMPT.format(
            current_prompt=current_prompt,
            transform_instruction=transform_instruction,
            analysis=analysis,
        )
        return self.generate(prompt, temperature=0.7).strip()

    def freeform_restructure(
        self,
        current_prompt: str,
        failing_samples: list[dict[str, Any]],
        analysis: str,
        score_trajectory: list[float] | None = None,
        structural_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate a free-form structural improvement guided by failure analysis."""
        from aevyra_reflex.prompts import FREEFORM_RESTRUCTURE_PROMPT

        samples_text = _format_failing_samples(failing_samples)
        trajectory_str = " → ".join(f"{s:.3f}" for s in (score_trajectory or []))
        history_text = _format_structural_history(structural_history or [])

        prompt = FREEFORM_RESTRUCTURE_PROMPT.format(
            current_prompt=current_prompt,
            score_trajectory=trajectory_str or "(first iteration)",
            structural_history=history_text,
            analysis=analysis,
            failing_samples=samples_text,
        )
        return self.generate(prompt, temperature=0.8).strip()

    # ------------------------------------------------------------------
    # PDO strategy methods
    # ------------------------------------------------------------------

    def judge_pairwise(
        self,
        question: str,
        response_a: str,
        response_b: str,
        ideal: str | None = None,
    ) -> str:
        """Judge which of two responses is better. Returns 'A' or 'B'.

        When ``ideal`` is provided the judge uses reference-comparison criteria
        (correctness 50 %).  When it is ``None`` (label-free tasks) the judge
        switches to quality / instruction-following / conciseness criteria so
        the comparison is not biased toward a non-existent reference answer.
        """
        from aevyra_reflex.prompts import (
            PAIRWISE_JUDGE_PROMPT,
            PAIRWISE_CRITERIA_WITH_IDEAL,
            PAIRWISE_CRITERIA_LABEL_FREE,
        )

        ideal_section = f"## Reference answer\n{ideal}" if ideal else ""
        criteria_section = PAIRWISE_CRITERIA_WITH_IDEAL if ideal else PAIRWISE_CRITERIA_LABEL_FREE
        prompt = PAIRWISE_JUDGE_PROMPT.format(
            question=question,
            response_a=response_a,
            response_b=response_b,
            ideal_section=ideal_section,
            criteria_section=criteria_section,
        )
        response = self.generate(prompt, temperature=0.0)
        return _extract_winner(response)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_failing_samples(samples: list[dict[str, Any]], max_samples: int = 10) -> str:
    """Format failing samples for inclusion in a prompt."""
    lines = []
    for i, s in enumerate(samples[:max_samples]):
        lines.append(f"### Sample {i + 1} (score: {s.get('score', 'N/A')})")
        lines.append(f"**Input:** {s.get('input', 'N/A')}")
        lines.append(f"**Model response:** {s.get('response', 'N/A')}")
        if s.get("ideal"):
            lines.append(f"**Reference answer:** {s['ideal']}")
        lines.append("")
    if len(samples) > max_samples:
        lines.append(f"... and {len(samples) - max_samples} more failing samples")
    return "\n".join(lines)


def _extract_change_summary(response: str) -> str:
    """Extract the one-line change summary from the agent's response.

    Looks for a '## Summary of change' section and returns the first non-empty
    line after it. Falls back to an empty string if not found.
    """
    marker = "## Summary of change"
    idx = response.lower().find(marker.lower())
    if idx == -1:
        return ""
    text = response[idx + len(marker):]
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line.strip("[]").strip()
    return ""


def _format_rewrite_log(rewrite_log: list[dict]) -> str:
    """Format the causal rewrite history for inclusion in a prompt.

    Each entry should have: iteration, score, delta, change_summary.
    """
    if not rewrite_log:
        return "(no previous rewrites — this is the first refinement)"
    lines = []
    for entry in rewrite_log:
        iteration = entry.get("iteration", "?")
        score = entry.get("score", 0.0)
        delta = entry.get("delta", 0.0)
        summary = entry.get("change_summary") or "(no summary)"
        sign = "+" if delta >= 0 else ""
        effect = "✓ helped" if delta > 0.005 else ("✗ no effect" if abs(delta) <= 0.005 else "✗ hurt")
        lines.append(
            f"Iter {iteration} (score: {score:.4f}, Δ{sign}{delta:.4f} — {effect}): {summary}"
        )
    return "\n".join(lines)


def _extract_revised_prompt(response: str) -> str:
    """Extract the revised prompt from the agent's response.

    Looks for a '## Revised prompt' section header, then takes everything after it.
    Falls back to the full response if the header isn't found.
    """
    marker = "## Revised prompt"
    idx = response.lower().find(marker.lower())
    if idx != -1:
        text = response[idx + len(marker):]
        # Strip any leading newlines or whitespace
        return text.strip()
    # Fallback: look for content after the last markdown header
    lines = response.strip().split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("#"):
            return "\n".join(lines[i + 1:]).strip()
    return response.strip()


def _extract_winner(response: str) -> str:
    """Extract 'A' or 'B' from a judge response."""
    # Try JSON parse first
    try:
        data = json.loads(response)
        winner = data.get("winner", "").strip().upper()
        if winner in ("A", "B"):
            return winner
    except (json.JSONDecodeError, AttributeError):
        pass

    # Try to find JSON in the response
    match = re.search(r'\{[^}]*"winner"\s*:\s*"([AB])"[^}]*\}', response)
    if match:
        return match.group(1).upper()

    # Last resort: look for standalone A or B
    response_upper = response.upper()
    if '"A"' in response_upper or "WINNER: A" in response_upper:
        return "A"
    if '"B"' in response_upper or "WINNER: B" in response_upper:
        return "B"

    logger.warning("Could not extract winner from judge response, defaulting to A")
    return "A"


def _format_candidates(candidates: list[dict[str, str]], max_show: int = 20) -> str:
    """Format candidate exemplars for inclusion in a prompt."""
    lines = []
    for i, c in enumerate(candidates[:max_show]):
        score_str = f" (score: {c['score']:.3f})" if "score" in c else ""
        lines.append(f"### Candidate {i + 1}{score_str}")
        lines.append(f"**Input:** {c['input']}")
        lines.append(f"**Output:** {c['output']}")
        lines.append("")
    if len(candidates) > max_show:
        lines.append(f"... and {len(candidates) - max_show} more candidates")
    return "\n".join(lines)


def _format_current_examples(examples: list[dict[str, str]]) -> str:
    """Format the currently selected few-shot examples."""
    if not examples:
        return "(none selected yet)"
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"{i}. Input: {ex['input']}")
        lines.append(f"   Output: {ex['output']}")
    return "\n".join(lines)


def _format_structural_history(history: list[dict[str, Any]]) -> str:
    """Format structural experiment history for context, with outcome labels."""
    if not history:
        return "(no previous experiments — this is the first round)"
    lines = []
    for h in history:
        score_before = h.get("score_before", 0.0)
        score_after = h.get("score_after", 0.0)
        delta = score_after - score_before
        sign = "+" if delta >= 0 else ""
        effect = "✓ helped" if delta > 0.005 else ("✗ no effect" if abs(delta) <= 0.005 else "✗ hurt")
        best = h.get("best_transform", "?")
        transforms = ", ".join(h.get("transforms_tried", []))
        lines.append(
            f"Iter {h['iteration']} (score: {score_after:.4f}, Δ{sign}{delta:.4f} — {effect}): "
            f"winner={best}  tried=[{transforms}]"
        )
    return "\n".join(lines)


def _format_phase_history(history: list[dict[str, Any]]) -> str:
    """Format auto-strategy phase history for the agent."""
    if not history:
        return ""
    lines = []
    for h in history:
        imp = h.get("improvement", 0)
        sign = "+" if imp >= 0 else ""
        lines.append(
            f"Phase {h['phase']} ({h['axis']}): "
            f"{h.get('iterations_used', '?')} iterations, "
            f"score {h.get('score_before', 0):.3f} → {h.get('score_after', 0):.3f} "
            f"({sign}{imp:.3f}), converged={h.get('converged', False)}"
        )
    return "\n".join(lines)


def _parse_fewshot_response(
    response: str,
    candidate_exemplars: list[dict[str, str]],
) -> tuple[list[dict[str, str]], str]:
    """Parse the agent's fewshot selection response.

    Extracts the instruction and selected examples. Falls back to the
    top candidates if parsing fails.
    """
    # Extract instruction
    instruction = ""
    inst_marker = "## Instruction"
    inst_idx = response.find(inst_marker)
    examples_marker = "## Selected examples"
    ex_idx = response.find(examples_marker)

    if inst_idx != -1 and ex_idx != -1:
        instruction = response[inst_idx + len(inst_marker):ex_idx].strip()
    elif inst_idx != -1:
        instruction = response[inst_idx + len(inst_marker):].strip()

    # Extract examples
    examples: list[dict[str, str]] = []
    if ex_idx != -1:
        ex_text = response[ex_idx + len(examples_marker):]
        # Parse EXAMPLE_N_INPUT / EXAMPLE_N_OUTPUT pairs
        import re
        input_pattern = re.compile(r"EXAMPLE_\d+_INPUT:\s*(.+?)(?=\nEXAMPLE_\d+_OUTPUT:)", re.DOTALL)
        output_pattern = re.compile(r"EXAMPLE_\d+_OUTPUT:\s*(.+?)(?=\n\nEXAMPLE_|\n\n##|\Z)", re.DOTALL)

        inputs = input_pattern.findall(ex_text)
        outputs = output_pattern.findall(ex_text)

        for inp, out in zip(inputs, outputs):
            examples.append({"input": inp.strip(), "output": out.strip()})

    # Fallback: if parsing failed, take top candidates
    if not examples and candidate_exemplars:
        examples = [
            {"input": c["input"], "output": c["output"]}
            for c in candidate_exemplars[:5]
        ]

    if not instruction:
        instruction = "(instruction not extracted)"

    return examples, instruction


# Backward compatibility alias
Agent = LLM
