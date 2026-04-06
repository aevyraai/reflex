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

"""Optimization strategies for reflex."""

from aevyra_reflex.strategies.auto import AutoStrategy
from aevyra_reflex.strategies.base import Strategy
from aevyra_reflex.strategies.fewshot import FewShotStrategy
from aevyra_reflex.strategies.iterative import IterativeStrategy
from aevyra_reflex.strategies.pdo import PDOStrategy
from aevyra_reflex.strategies.structural import StructuralStrategy

_REGISTRY: dict[str, type[Strategy]] = {
    "auto": AutoStrategy,
    "iterative": IterativeStrategy,
    "pdo": PDOStrategy,
    "fewshot": FewShotStrategy,
    "structural": StructuralStrategy,
}


def get_strategy(name: str) -> type[Strategy]:
    """Look up a strategy by name.

    Raises ValueError if the name isn't registered. Use
    ``register_strategy()`` to add custom strategies.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown strategy {name!r}. Available: {available}")
    return _REGISTRY[name]


def register_strategy(name: str, cls: type[Strategy]) -> None:
    """Register a custom strategy so it can be used by name.

    Example::

        from aevyra_reflex.strategies import Strategy, register_strategy

        class MonteCarloStrategy(Strategy):
            def run(self, *, initial_prompt, dataset, providers, metrics,
                    agent, config, on_iteration=None):
                # ... your optimization loop ...
                return OptimizationResult(...)

        register_strategy("montecarlo", MonteCarloStrategy)

    Then use it:

        # Python API
        config = OptimizerConfig(strategy="montecarlo")

        # CLI
        aevyra-reflex optimize data.jsonl prompt.md -m local/llama3.1 -s montecarlo

    Args:
        name: Short name for the strategy (used in CLI ``-s`` flag).
        cls: A class that inherits from ``Strategy``.

    Raises:
        TypeError: If *cls* doesn't inherit from ``Strategy``.
    """
    if not (isinstance(cls, type) and issubclass(cls, Strategy)):
        raise TypeError(
            f"Strategy class must inherit from Strategy, got {cls!r}"
        )
    _REGISTRY[name] = cls


def list_strategies() -> list[str]:
    """Return the names of all registered strategies."""
    return sorted(_REGISTRY)


__all__ = [
    "Strategy",
    "AutoStrategy",
    "IterativeStrategy",
    "PDOStrategy",
    "FewShotStrategy",
    "StructuralStrategy",
    "get_strategy",
    "register_strategy",
    "list_strategies",
]
