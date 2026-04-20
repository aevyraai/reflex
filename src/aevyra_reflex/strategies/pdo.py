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
        resume_state: dict | None = None,
        update_strategy_state: Any | None = None,
        eval_fn: Any | None = None,
    ) -> OptimizationResult:
        if eval_fn is not None:
            raise NotImplementedError(
                "Pipeline mode (set_pipeline) is not yet supported with the 'pdo' strategy. "
                "Use strategy='iterative', 'structural', or 'auto' with set_pipeline()."
            )

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

        # Adaptive ranking state — one Beta alpha per method.
        # method_alphas[k] starts at 1 (uniform Dirichlet prior) and grows
        # each time method k's predicted champion wins its next-round duels.
        ranking_method = pdo_config.ranking_method
        method_alphas = np.ones(len(RANKING_METHODS))
        # Per-method champion predicted at the END of each round, used to
        # score method accuracy after the NEXT round's duels.
        prev_method_champions: dict[str, int] = {}

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

            # Collect (winner_idx, loser_idx) for this round — used to score
            # each method's previous-round champion prediction.
            round_outcomes: list[tuple[int, int]] = []  # (winner, loser)

            with ThreadPoolExecutor(max_workers=max_duel_workers) as duel_pool:
                for i, j, winner in duel_pool.map(_run_one_duel, duel_pairs):
                    if winner == "A":
                        W[i, j] += 1
                        round_outcomes.append((i, j))
                        logger.info(f"    Duel: prompt {i} beat prompt {j}")
                    else:
                        W[j, i] += 1
                        round_outcomes.append((j, i))
                        logger.info(f"    Duel: prompt {j} beat prompt {i}")

            duel_elapsed = time.time() - t0
            logger.info(
                f"  {len(duel_pairs)} duels completed in {duel_elapsed:.1f}s "
                f"({duel_elapsed / len(duel_pairs):.1f}s/duel)"
            )

            # --- Adaptive ranking: update method_alphas ---
            # For each method, check if its champion from the previous round
            # won more duels than it lost in this round.  A win increments the
            # method's alpha (shifting Dirichlet weight toward it).
            if ranking_method == "auto" and prev_method_champions:
                for k, method in enumerate(RANKING_METHODS):
                    champ = prev_method_champions.get(method)
                    if champ is None or champ >= len(pool):
                        continue
                    wins = sum(1 for w, _ in round_outcomes if w == champ)
                    losses = sum(1 for _, loser in round_outcomes if loser == champ)
                    if wins + losses > 0 and wins >= losses:
                        method_alphas[k] += 1.0

                # Log dominant method (highest alpha weight)
                dominant_idx = int(np.argmax(method_alphas))
                total_alpha = method_alphas.sum()
                weights_pct = [f"{m}={method_alphas[k]/total_alpha:.0%}"
                               for k, m in enumerate(RANKING_METHODS)]
                logger.info(
                    f"  Ranking weights: {', '.join(weights_pct)} "
                    f"(dominant: {RANKING_METHODS[dominant_idx]})"
                )

            # Compute rankings with the chosen method
            rankings = _rank(W, ranking_method, method_alphas)
            current_best_idx = rankings[0]

            # Store each method's champion prediction for the next round
            if ranking_method == "auto":
                prev_method_champions = {
                    m: int(np.argmax(_scores_for_method(W, m)))
                    for m in RANKING_METHODS
                }

            # Record iteration
            record = IterationRecord(
                iteration=round_num,
                system_prompt=pool[current_best_idx],
                score=_win_rate(W, current_best_idx),
                reasoning=(
                    f"Pool size: {len(pool)}, champion index: {current_best_idx}, "
                    f"ranking: {ranking_method}"
                ),
            )
            iterations.append(record)
            # NOTE: on_iteration is called AFTER reasoning_tokens is set below

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

            # Fire callback now that reasoning_tokens is populated
            if on_iteration:
                on_iteration(record)

            # Prune worst performers if pool gets too large
            if len(pool) > pdo_config.max_pool_size:
                pool, W = _prune_pool(
                    pool, W, rankings,
                    keep=pdo_config.max_pool_size,
                )
                # Remap best_idx after pruning
                if best_idx >= len(pool):
                    best_idx = 0
                    best_score = _win_rate(W, 0)
                logger.info(f"  Pruned → pool size: {len(pool)}")

        # Final: return the best prompt found
        final_rankings = _rank(W, ranking_method, method_alphas)
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
        ranking_method: str = "auto",
    ):
        self.total_rounds = total_rounds
        self.duels_per_round = duels_per_round
        self.samples_per_duel = samples_per_duel
        self.initial_pool_size = initial_pool_size
        self.thompson_alpha = thompson_alpha
        self.mutation_frequency = mutation_frequency
        self.num_top_to_mutate = num_top_to_mutate
        self.max_pool_size = max_pool_size
        self.ranking_method = ranking_method

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
            ranking_method=kw.get("ranking_method", "auto"),
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
        max_workers=config.max_workers,
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
        for m in metrics:
            runner.add_metric(m)

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


# ---------------------------------------------------------------------------
# Ranking methods
# ---------------------------------------------------------------------------

#: All individual ranking methods available for PDO.
#: Order matters — indices are used in method_alphas for the adaptive fusion.
RANKING_METHODS = ("copeland", "borda", "elo", "avg_winrate")


def _copeland_scores(W: np.ndarray) -> np.ndarray:
    """Copeland score = (opponents beaten) - (opponents lost to).

    Win-rate against each opponent is used as a fractional tiebreaker, added
    at a tiny scale so it only affects tied Copeland scores.
    """
    K = W.shape[0]
    N = W + W.T
    scores = np.zeros(K)
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
        wr = wr_sum / count if count > 0 else 0.0
        scores[i] = (wins - losses) + wr * 0.01  # tiebreak
    return scores


def _borda_scores(W: np.ndarray) -> np.ndarray:
    """Borda score = mean win rate across all opponents with recorded matches.

    Unlike Copeland, Borda rewards consistent performance rather than just
    beating the majority — a prompt that wins 60 % of matches against every
    opponent scores higher than one that dominates two opponents but loses
    badly to others.
    """
    K = W.shape[0]
    N = W + W.T
    scores = np.zeros(K)
    for i in range(K):
        total = 0.0
        count = 0
        for j in range(K):
            if i != j and N[i, j] > 0:
                total += W[i, j] / N[i, j]
                count += 1
        scores[i] = total / count if count > 0 else 0.0
    return scores


def _elo_scores(
    W: np.ndarray,
    k_factor: float = 32.0,
    initial: float = 1000.0,
    passes: int = 3,
) -> np.ndarray:
    """Elo ratings estimated from the win matrix.

    Runs ``passes`` rounds over all pairs to let ratings converge. Each pair
    (i, j) with N[i,j] > 0 contributes one "match" with fractional outcome
    W[i,j] / N[i,j].
    """
    K = W.shape[0]
    ratings = np.full(K, initial, dtype=float)
    for _ in range(passes):
        for i in range(K):
            for j in range(i + 1, K):
                n_ij = int(W[i, j] + W[j, i])
                if n_ij == 0:
                    continue
                actual_i = W[i, j] / n_ij
                expected_i = 1.0 / (1.0 + 10.0 ** ((ratings[j] - ratings[i]) / 400.0))
                delta = k_factor * (actual_i - expected_i)
                ratings[i] += delta
                ratings[j] -= delta
    return ratings


def _avg_winrate_scores(W: np.ndarray) -> np.ndarray:
    """Overall win rate: total wins / total games played."""
    K = W.shape[0]
    scores = np.zeros(K)
    for i in range(K):
        total_wins = float(W[i].sum())
        total_games = total_wins + float(W[:, i].sum())
        scores[i] = total_wins / total_games if total_games > 0 else 0.0
    return scores


def _scores_for_method(W: np.ndarray, method: str) -> np.ndarray:
    """Return raw scores (higher = better) for a named ranking method."""
    if method == "copeland":
        return _copeland_scores(W)
    if method == "borda":
        return _borda_scores(W)
    if method == "elo":
        return _elo_scores(W)
    if method == "avg_winrate":
        return _avg_winrate_scores(W)
    raise ValueError(f"Unknown ranking method: {method!r}")


def _normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1]. Returns zeros if all equal."""
    mn, mx = scores.min(), scores.max()
    if mx > mn:
        return (scores - mn) / (mx - mn)
    return np.zeros_like(scores)


def _fused_ranking(W: np.ndarray, weights: np.ndarray) -> list[int]:
    """Weighted fusion of all ranking methods.

    Normalizes each method's scores to [0, 1] then combines with the given
    weights. ``weights`` must have the same length as ``RANKING_METHODS``.
    """
    normed = np.stack([_normalize(_scores_for_method(W, m)) for m in RANKING_METHODS])
    fused = weights @ normed  # shape (K,)
    return np.argsort(-fused).tolist()


def _rank(
    W: np.ndarray,
    method: str,
    method_alphas: np.ndarray | None = None,
) -> list[int]:
    """Rank prompts using the specified method.

    Args:
        W: Win matrix — W[i, j] = times prompt i beat prompt j.
        method: One of ``RANKING_METHODS``, ``"fused"``, or ``"auto"``.
        method_alphas: Beta posterior alpha parameters, one per entry in
            ``RANKING_METHODS``. Only used when ``method="auto"``.  When
            ``None``, uniform weights are used for fused/auto.

    Returns:
        Indices sorted best-first.
    """
    if method in RANKING_METHODS:
        return np.argsort(-_scores_for_method(W, method)).tolist()

    rng = np.random.default_rng()
    n_methods = len(RANKING_METHODS)

    if method == "fused":
        weights = np.ones(n_methods) / n_methods
    elif method == "auto":
        alphas = method_alphas if method_alphas is not None else np.ones(n_methods)
        weights = rng.dirichlet(alphas)
    else:
        raise ValueError(
            f"Unknown ranking_method {method!r}. "
            f"Choose from {list(RANKING_METHODS) + ['fused', 'auto']}."
        )

    return _fused_ranking(W, weights)


# Keep the original function for backward compatibility.
def _copeland_ranking(W: np.ndarray) -> list[int]:
    """Rank prompts by Copeland score (wins - losses) with win-rate tiebreaker."""
    return np.argsort(-_copeland_scores(W)).tolist()


def _win_rate(W: np.ndarray, idx: int) -> float:
    """Compute the overall win rate for a prompt."""
    total_wins = float(W[idx].sum())
    total_games = total_wins + float(W[:, idx].sum())
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
