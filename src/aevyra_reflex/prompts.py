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

"""Prompt templates used by reflex's optimization strategies.

Two families of templates:
  1. Iterative strategy — diagnose failures and propose revisions
  2. PDO strategy — generate candidate prompts, mutate champions, judge duels
"""

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

DATASET_SUMMARY_PROMPT = """\
You are a data analyst. Write a concise summary of this eval dataset based on \
the sample inputs below. Focus on:
- Input structure (plain text, Q&A, JSON, etc.)
- Common patterns, terminology, or domain
- Typical input length
- Whether reference answers are present and what they look like

Do NOT attempt to answer the questions. Keep the summary to 3-5 sentences.

## Sample inputs

{examples}
"""

# ---------------------------------------------------------------------------
# Iterative strategy
# ---------------------------------------------------------------------------

DIAGNOSE_FAILURES_PROMPT = """\
You are an expert prompt engineer analyzing why a system prompt is \
underperforming on certain eval samples.

## Current system prompt
{system_prompt}

## Task description
The model using this system prompt is being evaluated on a dataset. Below are \
samples where it scored poorly. For each sample you can see the input, the \
model's response, the reference answer (if available), and the score.

## Failing samples
{failing_samples}

## Your task
1. Identify the 2-3 most common failure patterns (e.g., wrong format, missing \
details, hallucinations, too verbose, ignoring instructions).
2. For each pattern, explain WHY the current prompt causes it.
3. Propose a revised system prompt that addresses these failures.

The revised prompt should:
- Keep what's working well in the original
- Be specific about the changes and why they help
- Stay concise — longer prompts aren't always better

Return your response in this exact format:

## Failure patterns
1. [pattern]: [explanation]
2. [pattern]: [explanation]
3. [pattern]: [explanation]

## Revised prompt
[Your complete revised system prompt here — the full text, not a diff]
"""

REFINE_PROMPT = """\
You are an expert prompt engineer. You've been iterating on a system prompt \
to improve eval scores. Here's where things stand:

## Current system prompt (iteration {iteration})
{system_prompt}

## Score trajectory
{score_trajectory}

## Current eval results
Mean score: {mean_score:.4f} (target: {target_score:.4f})

## Failing samples from this iteration
{failing_samples}

## Previous failure patterns you identified
{previous_reasoning}

## Your task
The score has {improved_text} since last iteration. \
Analyze why, then propose a revised prompt. Be surgical — don't change what's \
working. Focus on the remaining failure modes.

## Revised prompt
[Your complete revised system prompt — the full text, not a diff]
"""

# ---------------------------------------------------------------------------
# PDO strategy — prompt generation
# ---------------------------------------------------------------------------

GENERATE_CANDIDATE_PROMPT = """\
You are an expert prompt engineer. Generate a high-quality system prompt for \
the task described below.

## Dataset summary
{dataset_summary}

## Sample inputs (do NOT answer them)
{sample_inputs}

## Prompt engineering tip
{tip}

## Base instruction (starting point)
{base_instruction}

## Output
Return exactly one system prompt. No explanation, no JSON wrapping — just the \
prompt text itself.
"""

MUTATE_CHAMPION_PROMPT = """\
You are an expert prompt engineer specializing in prompt optimization. Your \
task is to create a mutation of the current best-performing system prompt.

## Current best prompt
{champion_prompt}

## Mutation strategy
{mutation_tip}

## Sample inputs (for context)
{sample_inputs}

## Output
Return exactly one mutated system prompt. No explanation — just the prompt text.
"""

# ---------------------------------------------------------------------------
# PDO strategy — pairwise judging
# ---------------------------------------------------------------------------

PAIRWISE_JUDGE_PROMPT = """\
You are an impartial judge evaluating two competing responses to the same task.

## Task input
{question}

## Response A
{response_a}

## Response B
{response_b}

{ideal_section}

## Evaluation criteria
1. **Correctness (50%)** — Does the answer match the expected output / factual reality?
2. **Quality (30%)** — Is the reasoning clear, complete, and well-structured?
3. **Instruction following (20%)** — Does the response follow the format and constraints?

## Instructions
- Compare both responses carefully against the criteria
- Select the winner: "A" or "B"
- If truly equal, pick the one that is more concise and direct

Return ONLY this JSON, nothing else:
{{"reasoning": "Your justification in ~50 words", "winner": "A or B"}}
"""

# ---------------------------------------------------------------------------
# Tips for prompt generation diversity
# ---------------------------------------------------------------------------

GENERATION_TIPS = {
    "persona": "Give the model a specific expert persona relevant to the task.",
    "step_by_step": "Include explicit step-by-step reasoning instructions.",
    "format": "Be very precise about the expected output format.",
    "examples": "Include 1-2 brief examples showing the expected input→output pattern.",
    "constraints": "Add specific constraints: what to avoid, length limits, edge cases.",
    "simple": "Keep the instruction clear, direct, and minimal.",
    "context": "Provide rich context about the domain and what makes a good answer.",
    "chain_of_thought": "Ask the model to think through the problem before answering.",
}

# ---------------------------------------------------------------------------
# Auto strategy — axis recommendation
# ---------------------------------------------------------------------------

RECOMMEND_AXIS_PROMPT = """\
You are an expert prompt optimization strategist. You're running an automated \
multi-phase optimization pipeline. After each phase, you decide which \
optimization axis to apply next.

## Available axes
- **structural**: Reorganize the prompt's structure, formatting, section order, \
and information hierarchy. Best when the prompt is disorganized, has poor \
formatting, or buries important instructions.
- **iterative**: Diagnose specific failure patterns and revise the wording. \
Best when the prompt's structure is fine but the instructions are unclear, \
incomplete, or cause specific error patterns.
- **fewshot**: Add or refine few-shot examples in the prompt. Best when the \
model understands the task but struggles with the expected format or misses \
edge cases that examples would clarify.
- **pdo**: Tournament-style search over prompt variants using pairwise judging. \
Best as a final polish step when the prompt is close to good but needs \
fine-tuning, or when you've exhausted other approaches.

## Current prompt
{current_prompt}

## Sample inputs from the dataset
{sample_inputs}

## Optimization history
{phase_history}

## Axes already used (avoid repeating unless strongly justified)
{axes_used}

## Axes still available
{axes_available}

## Your task
Based on the current prompt quality and the optimization history, recommend \
the single best axis to try next. Consider:
1. What type of weakness is most limiting performance right now?
2. Which axis addresses that weakness most directly?
3. Avoid repeating an axis that already gave diminishing returns.

Return ONLY a JSON object:
{{"axis": "structural|iterative|fewshot|pdo", "reasoning": "1-2 sentence justification"}}
"""

MUTATION_TIPS = {
    "expand": "Keep the core instruction but add clarifying guidance or edge case handling.",
    "compress": "Make the prompt more concise while preserving the key instructions.",
    "rephrase": "Reword the prompt significantly while keeping the same intent.",
    "few_shot": "Add 1-2 concrete examples to demonstrate the expected behavior.",
    "emphasis": "Strengthen the most important instructions with clearer emphasis.",
    "structure": "Reorganize the prompt with better structure (sections, numbered steps).",
}

# ---------------------------------------------------------------------------
# Few-shot strategy
# ---------------------------------------------------------------------------

SELECT_FEWSHOT_PROMPT = """\
You are an expert prompt engineer specializing in few-shot example selection.

## Task instruction (the base system prompt)
{base_instruction}

## Candidate exemplars
Below are input-output pairs from the dataset, sorted by quality. Your job is \
to select the {max_examples} most useful ones to include as few-shot examples \
in the system prompt.

{candidates}

## Selection strategy: {selection_strategy}

## Selection criteria
- **Diversity**: Pick examples that cover different types/patterns of inputs
- **Clarity**: Prefer examples where the output is unambiguous and well-formed
- **Edge cases**: Include at least one example that demonstrates tricky or \
non-obvious behavior
- **Brevity**: Prefer shorter examples that still convey the pattern clearly

## Output format
Return your response in this exact format:

## Instruction
[The base instruction, optionally refined to work better with examples. \
Keep changes minimal — the instruction should reference the examples naturally.]

## Selected examples
EXAMPLE_1_INPUT: [exact input text]
EXAMPLE_1_OUTPUT: [exact output text]

EXAMPLE_2_INPUT: [exact input text]
EXAMPLE_2_OUTPUT: [exact output text]

(continue for all selected examples)
"""

REFINE_FEWSHOT_PROMPT = """\
You are an expert prompt engineer optimizing few-shot examples in a system \
prompt. Here's the current state:

## Current instruction
{current_instruction}

## Current examples in the prompt
{current_examples}

## Score trajectory
{score_trajectory}

## Failing samples (where the model still gets it wrong)
{failing_samples}

## Candidate exemplars available (sorted by quality)
{candidates}

## Your task
The model is still failing on certain types of inputs. Analyze the failure \
patterns and decide:
1. Should any current examples be swapped out?
2. Are there candidates that better cover the failure modes?
3. Should the instruction be tweaked to work better with the examples?

Select up to {max_examples} examples total.

## Output format
Return your response in this exact format:

## Reasoning
[Brief analysis of why certain examples should be swapped]

## Instruction
[The refined instruction text]

## Selected examples
EXAMPLE_1_INPUT: [exact input text]
EXAMPLE_1_OUTPUT: [exact output text]

EXAMPLE_2_INPUT: [exact input text]
EXAMPLE_2_OUTPUT: [exact output text]

(continue for all selected examples)
"""

# ---------------------------------------------------------------------------
# Structural strategy
# ---------------------------------------------------------------------------

ANALYZE_STRUCTURE_PROMPT = """\
You are an expert in prompt structure and formatting. Analyze the structural \
weaknesses of this system prompt.

## Current system prompt
{system_prompt}

## Failing samples
{failing_samples}

## Previous structural experiments
{structural_history}

## Your analysis
Focus on STRUCTURAL issues, not content issues:
- Is information organized logically?
- Are constraints placed where the model will attend to them?
- Is the formatting (markdown, plain text, XML) optimal for this model?
- Are sections clearly delineated or is everything blurred together?
- Is there a clear hierarchy of importance?
- Are instructions ordered from most to least important?

Return a concise analysis (3-5 sentences) identifying the top structural \
weakness and what type of reorganization would help most.
"""

RESTRUCTURE_PROMPT = """\
You are an expert prompt engineer. Restructure the following system prompt \
using the specified transformation while preserving ALL semantic content.

## Current prompt
{current_prompt}

## Structural transformation
{transform_instruction}

## Structural analysis
{analysis}

## Rules
- Preserve every instruction, constraint, and piece of information
- Do NOT add new content or remove existing content
- ONLY change the structure, formatting, and organization
- The result must be a complete, self-contained system prompt

## Output
Return the restructured prompt. No explanation — just the prompt text.
"""

FREEFORM_RESTRUCTURE_PROMPT = """\
You are an expert prompt engineer specializing in prompt structure. You've been \
experimenting with different structural organizations of a system prompt.

## Current prompt
{current_prompt}

## Score trajectory
{score_trajectory}

## Structural analysis
{analysis}

## Failing samples
{failing_samples}

## Your task
Based on the failures and your structural analysis, create the best possible \
reorganization of this prompt. You may:
- Reorder sections for better attention allocation
- Change formatting (markdown, XML tags, plain text, numbered lists)
- Add structural markers (headers, delimiters, tags)
- Regroup related instructions
- Adjust information hierarchy

Do NOT add new semantic content. Only restructure what's already there.

## Output
Return the restructured prompt. No explanation — just the prompt text.
"""
