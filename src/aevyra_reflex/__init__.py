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

"""aevyra-reflex — agentic prompt optimization."""

from aevyra_reflex.agent import LLM
from aevyra_reflex.callbacks import MLflowCallback, WandbCallback
from aevyra_reflex.optimizer import (
    OptimizerConfig,
    PromptOptimizer,
    parse_verdict_results,
)
from aevyra_reflex.result import OptimizationResult
from aevyra_reflex.run_store import RunStore
from aevyra_reflex.strategies import Strategy, register_strategy
from aevyra_reflex.trace import AgentTrace, TraceNode

__version__ = "0.2.0"

__all__ = [
    "AgentTrace",
    "LLM",
    "MLflowCallback",
    "TraceNode",
    "WandbCallback",
    "OptimizerConfig",
    "OptimizationResult",
    "PromptOptimizer",
    "RunStore",
    "Strategy",
    "parse_verdict_results",
    "register_strategy",
]
