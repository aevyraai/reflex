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

"""PDO (Prompt Duel Optimizer) strategy.

Maintains a pool of candidate system prompts and uses dueling bandits with
Thompson sampling to converge on the best one. Inspired by Meta's prompt-ops
PDO paper (arXiv:2510.13907), rebuilt on top of verdict.

Each round:
  1. Thompson sampling selects a pair of prompts to duel
  2. Both prompts are evaluated on a sample of the dataset via verdict
  3. An LLM judge (or metric comparison) picks the winner
  4. Win matrix is updated; rankings are recalculated
  5. Periodically, the top prompts are mutated to explore new candidates
"""

from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from aevyra_reflex.agent import Agent
from aevyra_reflex.prompts import GENERATION_TIPS, MUTATION_TIPS
from aevyra_reflex.result import IterationRecord, OptimizationResult
from aevyra_reflex.strategies.base import Strategy

logger = logging.getLogger(__name__)


class PDOStrategy(Strategy):
    """Dueling bandit prompt optimization with Thompson sampling."""

    def run(
        self,
        *,
        initial_prompt: str,
        dataset: Any,
        providers: list[dict[str, Any]],
        metrics: list[Any],
        agent: Agent,
        config: Any,
        on_iteration: Any | None = None,
    ) -> OptimizationResult:
        pdo_config = _PDOConfig.from_optimizer_config(config)

        # Summarize the dataset so the agent understands the task
        sample_inputs = _get_sample_inputs(dataset, n=5)
        dataset_summary = agent.summarize_dataset(sample_inputs)
        logger.info(f"Dataset summary: {dataset_summary[:200]}...")

        # Generate initial prompt pool
        logger.info(
            f"Generating initial pool of {pdo_config.initial_pool_size} prompts "
            f"({config.max_workers or 4} workers in parallel)"
        )
        t0 = time.time()
        pool = _initialize_pool(
            agent=agent,
            base_prompt=initial_prompt,
            dataset_summary=dataset_summary,
            sample_inputs=sample_inputs,
            pool_size=pdo_config.initial_pool_size,
            max_workers=config.max_workers or 4,
        )
        logger.info(
            f"Initialized pool with {len(pool)} prompts in {time.time() - t0:.1f}s"
        )

        # Win matrix: W[i,j] = number of times prompt i beat prompt j
        K = len(pool)
        W = np.zeros((K, K), dtype=int)

        iterations: list[IterationRecord] = []
        best_idx = 0
        best_score = 0.0

        for round_num in range(1, pdo_config.total_rounds + 1):
            logger.info(f"Round {round_num}/{pdo_config.total_rounds} (pool size: {len(pool)})")

            # Select all duel pairs for this round upfront (independent draws)
            duel_pairs = []
            for _ in range(pdo_config.duels_per_round):
                i, j = _thompson_sample_pair(
                    K=len(pool),
                    W=W,
                    alpha=pdo_config.thompson_alpha,
                    t=round_num,
                )
                duel_pairs.append((i, j))

            # Snapshot tokens before this round's agent calls
            reasoning_before = agent.tokens_used

            # Run duels in parallel
            def _run_one_duel(pair: tuple[int, int]) -> tuple[int, int, str]:
                i, j = pair
                winner = _run_duel(
                    prompt_a=pool[i],
                    prompt_b=pool[j],
                    dataset=dataset,
                    providers=providers,
                    metrics=metrics,
                    agent=agent,
                    config=config,
                    num_samples=pdo_config.samples_per_duel,
                )
                return i, j, winner

            max_duel_workers = min(len(duel_pairs), config.max_workers or 4)
            logger.info(
                f"  Running {len(duel_pairs)} duels "
                f"({max_duel_workers} workers in parallel)"
            )
            t0 = time.time()

            with ThreadPoolExecutor(max_workers=max_duel_workers) as duel_pool:
                for i, j, winner in duel_pool.map(_run_one_duel, duel_pairs):
                    if winner == "A":
                        W[i, j] += 1
                        logger.info(f"    Duel: prompt {i} beat prompt {j}")
                    else:
                        W[j, i] += 1
                        logger.info(f"    Duel: prompt {j} beat prompt {i}")

            duel_elapsed = time.time() - t0
            logger.info(
                f"  {len(duel_pairs)} duels completed in {duel_elapsed:.1f}s "
                f"({duel_elapsed / len(duel_pairs):.1f}s/duel)"
            )

            # Compute rankings
            rankings = _copeland_ranking(W)
            current_best_idx = rankings[0]

            # Record iteration
            record = IterationRecord(
                iteration=round_num,
                system_prompt=pool[current_best_idx],
                score=_win_rate(W, current_best_idx),
                reasoning=f"Pool size: {len(pool)}, champion index: {current_best_idx}",
            )
            iterations.append(record)
            if on_iteration:
                on_iteration(record)

            if record.score > best_score:
                best_score = record.score
                best_idx = current_best_idx

            logger.info(f"  Champion win rate: {record.score:.3f}")

            # Mutate top prompts periodically to explore (in parallel)
            if round_num % pdo_config.mutation_frequency == 0:
                top_indices = rankings[:pdo_config.num_top_to_mutate]
                tip_keys = list(MUTATION_TIPS.keys())

                def _mutate(idx: int) -> str:
                    tip = MUTATION_TIPS[random.choice(tip_keys)]
                    return agent.mutate_champion(
                        champion_prompt=pool[idx],
                        mutation_tip=tip,
                        sample_inputs=sample_inputs,
                    )

                with ThreadPoolExecutor(max_workers=min(len(top_indices), config.max_workers or 4)) as mut_pool:
                    mutants = list(mut_pool.map(_mutate, top_indices))

                for mutant in mutants:
                    pool.append(mutant)
                    new_K = len(pool)
                    new_W = np.zeros((new_K, new_K), dtype=int)
                    new_W[:W.shape[0], :W.shape[1]] = W
                    W = new_W

                logger.info(f"  Mutated {len(top_indices)} prompts → pool size: {len(pool)}")

            # Capture reasoning tokens consumed this round (dueling judges + mutations)
            record.reasoning_tokens = agent.tokens_used - reasoning_before

            # Prune worst performers if pool gets too large
            if len(pool) > pdo_config.max_pool_size:
                pool, W = _prune_pool(
                    pool, W, rankings,
                    keep=pdo_config.max_pool_size,
                )
                # Remap best_idx
                if best_idx >= len(pool):
                    best_idx = 0
                    best_score = _win_rate(W, 0)
                logger.info(f"  Pruned → pool size: {len(pool)}")

        # Final: return the best prompt found
        final_rankings = _copeland_ranking(W)
        final_best = final_rankings[0]

        return OptimizationResult(
            best_prompt=pool[final_best],
            best_score=_win_rate(W, final_best),
            iterations=iterations,
            converged=best_score >= config.score_threshold,
        )


# ---------------------------------------------------------------------------
# PDO internals
# ---------------------------------------------------------------------------

class _PDOConfig:
    """PDO-specific parameters extracted from OptimizerConfig.extra_kwargs."""

    def __init__(
        self,
        total_rounds: int = 50,
        duels_per_round: int = 3,
        samples_per_duel: int = 10,
        initial_pool_size: int = 6,
        thompson_alpha: float = 1.2,
        mutation_frequency: int = 5,
        num_top_to_mutate: int = 2,
        max_pool_size: int = 20,
    ):
        self.total_rounds = total_rounds
        self.duels_per_round = duels_per_round
        self.samples_per_duel = samples_per_duel
        self.initial_pool_size = initial_pool_size
        self.thompson_alpha = thompson_alpha
        self.mutation_frequency = mutation_frequency
        self.num_top_to_mutate = num_top_to_mutate
        self.max_pool_size = max_pool_size

    @classmethod
    def from_optimizer_config(cls, config: Any) -> _PDOConfig:
        kw = config.extra_kwargs
        return cls(
            total_rounds=kw.get("total_rounds", config.max_iterations),
            duels_per_round=kw.get("duels_per_round", 3),
            samples_per_duel=kw.get("samples_per_duel", 10),
            initial_pool_size=kw.get("initial_pool_size", 6),
            thompson_alpha=kw.get("thompson_alpha", 1.2),
            mutation_frequency=kw.get("mutation_frequency", 5),
            num_top_to_mutate=kw.get("num_top_to_mutate", 2),
            max_pool_size=kw.get("max_pool_size", 20),
        )


def _get_sample_inputs(dataset: Any, n: int = 5) -> list[str]:
    """Extract sample user inputs from a verdict Dataset."""
    samples = []
    for convo in dataset.conversations[:n]:
        msg = convo.last_user_message
        if msg:
            samples.append(msg[:500])  # Truncate long inputs
    return samples


def _initialize_pool(
    *,
    agent: Agent,
    base_prompt: str,
    dataset_summary: str,
    sample_inputs: list[str],
    pool_size: int,
    max_workers: int = 4,
) -> list[str]:
    """Generate the initial pool of candidate prompts (in parallel)."""
    pool = [base_prompt] if base_prompt.strip() else []
    tip_keys = list(GENERATION_TIPS.keys())
    num_to_generate = pool_size - len(pool)

    def _gen_candidate(i: int) -> str:
        tip_key = tip_keys[i % len(tip_keys)]
        tip = GENERATION_TIPS[tip_key]
        return agent.generate_candidate(
            dataset_summary=dataset_summary,
            sample_inputs=sample_inputs,
            base_instruction=base_prompt,
            tip=tip,
        )

    workers = min(num_to_generate, max_workers)
    with ThreadPoolExecutor(max_workers=workers) as gen_pool:
        candidates = list(gen_pool.map(_gen_candidate, range(num_to_generate)))

    pool.extend(candidates)
    return pool


def _thompson_sample_pair(
    K: int,
    W: np.ndarray,
    alpha: float,
    t: int,
) -> tuple[int, int]:
    """Double Thompson Sampling for Copeland dueling bandits.

    Adapted from prompt-ops PDO. Samples pairwise win probabilities from
    Beta posteriors, computes Copeland scores, and selects two prompts to duel.
    """
    rng = np.random.default_rng()
    N = W + W.T

    if N.sum() == 0:
        pair = rng.choice(K, size=2, replace=False)
        return int(pair[0]), int(pair[1])

    # Thompson-sample pairwise win probabilities
    theta = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            th = rng.beta(W[i, j] + 1, W[j, i] + 1)
            theta[i, j] = th
            theta[j, i] = 1.0 - th

    # Copeland scores from sampled probabilities
    copeland = np.sum(theta > 0.5, axis=1)

    # First arm: highest sampled Copeland score
    max_score = copeland.max()
    first_candidates = np.flatnonzero(copeland == max_score)
    first = int(rng.choice(first_candidates))

    # Second arm: Thompson draw conditioned on first
    theta2 = np.full(K, 0.5)
    for k in range(K):
        if k != first:
            theta2[k] = rng.beta(W[k, first] + 1, W[first, k] + 1)

    candidates = [k for k in range(K) if k != first]
    best_val = max(theta2[k] for k in candidates)
    second_choices = [k for k in candidates if abs(theta2[k] - best_val) < 1e-12]
    second = int(rng.choice(second_choices))

    return first, second


def _run_duel(
    *,
    prompt_a: str,
    prompt_b: str,
    dataset: Any,
    providers: list[dict[str, Any]],
    metrics: list[Any],
    agent: Agent,
    config: Any,
    num_samples: int,
) -> str:
    """Run a head-to-head duel between two prompts. Returns 'A' or 'B'.

    Samples a subset of the dataset, runs both prompts through the first
    provider, and uses the agent as a pairwise judge on each sample. The
    prompt that wins the majority of comparisons wins the duel.
    """
    from aevyra_verdict import EvalRunner
    from aevyra_verdict.dataset import Conversation, Dataset, Message
    from aevyra_verdict.runner import RunConfig

    # Sample a subset of the dataset
    convos = dataset.conversations
    sample_indices = random.sample(range(len(convos)), min(num_samples, len(convos)))
    sampled_convos = [convos[i] for i in sample_indices]

    # Use only the first provider for duels (speed over thoroughness)
    provider_spec = providers[0]

    run_config = RunConfig(
        temperature=config.eval_temperature,
        max_tokens=config.max_tokens,
    )

    def _eval_prompt(prompt: str) -> list[str]:
        """Run a prompt on sampled conversations and return responses."""
        injected = []
        for convo in sampled_convos:
            messages = list(convo.messages)
            if messages and messages[0].role == "system":
                messages[0] = Message(role="system", content=prompt)
            else:
                messages.insert(0, Message(role="system", content=prompt))
            injected.append(Conversation(messages=messages, ideal=convo.ideal, metadata=convo.metadata))

        ds = Dataset(conversations=injected)
        runner = EvalRunner(config=run_config)
        runner.add_provider(
            provider_spec["provider_name"],
            provider_spec["model"],
            label=provider_spec.get("label"),
            api_key=provider_spec.get("api_key"),
            base_url=provider_spec.get("base_url"),
        )
        # Add a dummy metric if none — we just need completions
        if metrics:
            for m in metrics:
                runner.add_metric(m)
        else:
            from aevyra_verdict import RougeScore
            runner.add_metric(RougeScore())

        results = runner.run(ds, show_progress=False)

        # Extract response texts
        responses = []
        for label, model_result in results.model_results.items():
            for comp in model_result.completions:
                responses.append(comp.text if comp else "")
            break  # Only first model
        return responses

    max_workers = getattr(config, "max_workers", 4) or 4

    # Run both prompts in parallel
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(2, max_workers)) as pool:
        future_a = pool.submit(_eval_prompt, prompt_a)
        future_b = pool.submit(_eval_prompt, prompt_b)
        responses_a = future_a.result()
        responses_b = future_b.result()
    logger.debug(f"    Duel evals: {time.time() - t0:.1f}s (2 prompts, {len(responses_a)} samples each)")

    # Pairwise judging — all comparisons in parallel
    wins_a = 0
    wins_b = 0

    # Pre-compute randomized orders so judging threads are pure
    judge_args = []
    for idx, (ra, rb) in enumerate(zip(responses_a, responses_b)):
        question = sampled_convos[idx].last_user_message or ""
        ideal = sampled_convos[idx].ideal
        swap = random.random() < 0.5
        judge_args.append((idx, question, ra, rb, ideal, swap))

    def _judge_one(args: tuple) -> tuple[int, int]:
        """Returns (delta_a, delta_b) — one of (1,0) or (0,1)."""
        idx, question, ra, rb, ideal, swap = args
        if not swap:
            winner = agent.judge_pairwise(question, ra, rb, ideal)
            return (1, 0) if winner == "A" else (0, 1)
        else:
            winner = agent.judge_pairwise(question, rb, ra, ideal)
            return (0, 1) if winner == "A" else (1, 0)

    max_judge_workers = min(len(judge_args), max_workers)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_judge_workers) as pool:
        for da, db in pool.map(_judge_one, judge_args):
            wins_a += da
            wins_b += db
    logger.debug(
        f"    Judging: {time.time() - t0:.1f}s "
        f"({len(judge_args)} comparisons, {max_judge_workers} workers)"
    )

    return "A" if wins_a >= wins_b else "B"


def _copeland_ranking(W: np.ndarray) -> list[int]:
    """Rank prompts by Copeland score (wins - losses) with win-rate tiebreaker."""
    K = W.shape[0]
    N = W + W.T
    scores = np.zeros(K)
    winrates = np.zeros(K)

    for i in range(K):
        wins = losses = 0
        wr_sum = 0.0
        count = 0
        for j in range(K):
            if i == j:
                continue
            if N[i, j] > 0:
                if W[i, j] > W[j, i]:
                    wins += 1
                elif W[i, j] < W[j, i]:
                    losses += 1
                wr_sum += W[i, j] / N[i, j]
                count += 1
        scores[i] = wins - losses
        winrates[i] = wr_sum / count if count > 0 else 0.0

    # Sort descending by Copeland score, then by win rate
    order = np.lexsort((-winrates, -scores))
    return order.tolist()


def _win_rate(W: np.ndarray, idx: int) -> float:
    """Compute the overall win rate for a prompt."""
    total_wins = W[idx].sum()
    total_games = W[idx].sum() + W[:, idx].sum()
    return total_wins / total_games if total_games > 0 else 0.0


def _prune_pool(
    pool: list[str],
    W: np.ndarray,
    rankings: list[int],
    keep: int,
) -> tuple[list[str], np.ndarray]:
    """Remove the worst-ranked prompts from the pool."""
    keep_indices = rankings[:keep]
    keep_indices.sort()  # Preserve relative order

    new_pool = [pool[i] for i in keep_indices]
    new_W = W[np.ix_(keep_indices, keep_indices)]
    return new_pool, new_W
