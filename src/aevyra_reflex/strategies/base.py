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

"""Abstract base class for optimization strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from aevyra_reflex.agent import Agent
from aevyra_reflex.result import OptimizationResult


class Strategy(ABC):
    """Base class for prompt optimization strategies.

    A strategy receives the eval infrastructure (verdict Dataset, providers,
    metrics) plus a Claude agent, and runs an optimization loop that returns
    an OptimizationResult.
    """

    @abstractmethod
    def run(
        self,
        *,
        initial_prompt: str,
        dataset: Any,  # aevyra_verdict.Dataset
        providers: list[dict[str, Any]],
        metrics: list[Any],
        agent: Agent,
        config: Any,  # OptimizerConfig
        on_iteration: Any | None = None,
    ) -> OptimizationResult:
        """Run the optimization loop.

        Args:
            initial_prompt: The starting system prompt.
            dataset: A verdict Dataset instance.
            providers: List of provider specs (dicts with provider_name, model, etc.).
            metrics: List of verdict Metric instances.
            agent: The Claude agent for prompt operations.
            config: OptimizerConfig with strategy parameters.
            on_iteration: Optional callback(iteration_record) for progress reporting.

        Returns:
            OptimizationResult with the best prompt and full history.
        """
        ...
