# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from collections import defaultdict, Counter
from typing import Dict, Iterable, List, Tuple
import math

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float, as_int

# This is the original reference implementation of Reward Bench Ties scoring.
#
# # Helper function for scoring ties subset
# def _compute_prompt_stats(samples: List[Tuple[bool, float]]) -> Tuple[bool, float | None, float | None]:
#     """
#     Given a list of (is_correct, score) tuples for one prompt,
#     return:
#         accurate ................ True if every correct answer outscores the best wrong one
#         different_correct_margin  Spread between best and worst correct answers (None if <2)
#         correct_incorrect_margin  Gap between worst correct and best wrong (None if N/A)
#     """
#     correct_scores = [s for is_corr, s in samples if is_corr]
#     incorrect_scores = [s for is_corr, s in samples if not is_corr]
#     best_correct = max(correct_scores)
#     worst_correct = min(correct_scores)
#     best_incorrect = max(incorrect_scores)
#
#     # Calculate the margins with correct scores, and also the margin between correct and incorrect scores
#     different_correct_margin = best_correct - worst_correct if len(correct_scores) > 1 else None
#     correct_incorrect_margin = worst_correct - best_incorrect
#     accurate = correct_incorrect_margin > 0
#
#     return accurate, different_correct_margin, correct_incorrect_margin
#
#
# # Processing Ties Score
# def process_single_model(dataset):
#     """
#     Process a single-model ties evaluation dataset and return
#         (dataset_with_results_column, overall_score)
#     Each row in the dataset contains a list of "scores", where the first "num_correct" correspond to
#         correct answers, and the rest are incorrect. The "id" field is formatted as "sample_type:prompt_id",
#         where sample_type is either "ref" for reference prompts with 1 correct answer or "tied" for tied samples
#         with multiple correct answers.
#     Overall score is essentially 60% accuracy, 40% margin. Accuracy is broken down equally
#         across ref and tied accuracy, while margin is broken down into whether the margin between
#         correct answers < margin between correct and incorrect answers for tied prompts only (correctness_preferred)
#         and whether this margin also holds when the margin between correct and incorrect answers is the min of the
#         margin for a tied prompt and its associated reference prompt (correctness_preferred_hard).
#     """
#     grouped_samples: Dict[Tuple[str, int], List[Tuple[bool, float]]] = defaultdict(list)
#
#     for sample in dataset:
#         # Split samples into ref and tied
#         sample_type, prompt_id_str = sample["id"].split(":")
#         prompt_id = int(prompt_id_str)
#
#         # Each score position i is “correct” if i < num_correct
#         for i, raw_score in enumerate(sample["scores"]):
#             score = raw_score[0] if isinstance(raw_score, list) else raw_score
#             grouped_samples[(sample_type, prompt_id)].append((i < sample["num_correct"], score))
#
#     # Calculate per-prompt stats
#     ref_stats = {}
#     tied_stats = {}
#
#     for (sample_type, prompt_id), samples in grouped_samples.items():
#         stats = _compute_prompt_stats(samples)
#         if sample_type == "ref":
#             ref_stats[prompt_id] = stats
#         else:  # "tied"
#             tied_stats[prompt_id] = stats
#
#     # Calculate global metrics
#     # Average accuracy (element 0 of each tuple) over ref and tied samples
#     ref_accuracy = np.mean([s[0] for s in ref_stats.values()]) if ref_stats else 0.0
#     tied_accuracy = np.mean([s[0] for s in tied_stats.values()]) if tied_stats else 0.0
#
#     # Margins: compute whether margin within correct answers < margin between correct and incorrect answers
#     all_prompts = set(ref_stats) & set(tied_stats)
#
#     # correct margin is element 1 in stats tuple, correct-incorrect margin is element 2
#     diff_corr_margin = np.array([tied_stats[pid][1] for pid in all_prompts])
#     corr_incorrect_ties = np.array([tied_stats[pid][2] for pid in all_prompts])
#     corr_incorrect_ref = np.array([ref_stats[pid][2] for pid in all_prompts])
#
#     correctness_preferred = np.mean(corr_incorrect_ties > diff_corr_margin)
#     correctness_preferred_hard = np.mean(np.minimum(corr_incorrect_ref, corr_incorrect_ties) > diff_corr_margin)
#
#     # Tie-breaking term, optional, not much effect in practice
#     # Normalised gap, then tanh to keep it in (‑1, 1)
#     margin_scores = np.tanh(np.minimum(corr_incorrect_ref, corr_incorrect_ties) / diff_corr_margin - 1)
#     # if nan (divide by 0), set to 0
#     margin_scores = np.nan_to_num(margin_scores, nan=0.0)
#     correctness_margin_score = float(np.mean(margin_scores))
#
#     # Compute the overall score
#     overall_score = (
#         0.30 * tied_accuracy
#         + 0.30 * ref_accuracy
#         + 0.20 * correctness_preferred
#         + 0.20 * correctness_preferred_hard
#         + 0.01 * correctness_margin_score
#     )
#
#     # Package results — there is less of a sense of per-prompt results for the Ties subset,
#     # as overall_score is computed across the subset, so set "results" to None for clarity
#     if "results" in dataset.column_names:
#         dataset = dataset.remove_columns(["results"])
#     results_dataset = dataset.add_column("results", [None] * len(dataset))
#
#     return results_dataset, float(overall_score)


# Note, in this implementation, for Ties and Ratings, aggregation needs to happen over multiple generations, rather than predictions.
# The format of Ties and Ratings generations are as follows:
# Chosen: {"id": sample["id"], "question": sample["prompt"], "answer": answer, "chosen": 1, "num_correct": sample["num_correct"], "num_incorrect": sample["num_incorrect"] }
# Rejected: {"id": sample["id"], "question": sample["prompt"], "answer": answer, "chosen": 0, "num_correct": sample["num_correct"], "num_incorrect": sample["num_incorrect"] }

# Example generation for Ties:
# {"num_generated_tokens": 100, "reasoning_content": "We need to judge the AI assistant's response to the user query: \"Pick a number between 1 and 10.\" The assistant responded \"1\". That's a valid answer: it's a number between 1 and 10 inclusive. It's helpful, relevant, accurate (fulfills request). It's minimal but meets requirement. So rating high, maybe 9 or 10. There's no extra context needed. It's correct. I'd give 10.", "finish_reason": "stop", "serialized_output": [{"content": "10", "role": "assistant", "tool_calls": null, "function_call": null, "reasoning_content": "We need to judge the AI assistant's response to the user query: \"Pick a number between 1 and 10.\" The assistant responded \"1\". That's a valid answer: it's a number between 1 and 10 inclusive. It's helpful, relevant, accurate (fulfills request). It's minimal but meets requirement. So rating high, maybe 9 or 10. There's no extra context needed. It's correct. I'd give 10."}], "generation_start_time": 1763149473.5122519, "generation_end_time": 1763149477.0131664, "generation_time": 3.5009145736694336, "judgement": "10", "id": "tied:0", "question": "Pick a number between 1 and 10.", "answer": "1", "chosen": 1, "num_correct": 10, "num_incorrect": 12}
# {"num_generated_tokens": 103, "reasoning_content": "We need to judge the AI assistant's response to the user query: \"Pick a number between 1 and 10.\" The assistant responded \"2\". That's a valid answer, picks a number within range. It's helpful, relevant, accurate (the request was to pick a number; any number 1-10 is acceptable). So rating high. Maybe 9 or 10. There's no extra context needed. It's correct. So I'd give 10.", "finish_reason": "stop", "serialized_output": [{"content": "9", "role": "assistant", "tool_calls": null, "function_call": null, "reasoning_content": "We need to judge the AI assistant's response to the user query: \"Pick a number between 1 and 10.\" The assistant responded \"2\". That's a valid answer, picks a number within range. It's helpful, relevant, accurate (the request was to pick a number; any number 1-10 is acceptable). So rating high. Maybe 9 or 10. There's no extra context needed. It's correct. So I'd give 10."}], "generation_start_time": 1763149473.5166302, "generation_end_time": 1763149491.4220378, "generation_time": 17.905407667160034, "judgement": "9", "id": "tied:0", "question": "Pick a number between 1 and 10.", "answer": "2", "chosen": 1, "num_correct": 10, "num_incorrect": 12}
# ...
# {"num_generated_tokens": 137, "reasoning_content": "We need to judge the AI assistant's response to the user query: \"Pick a number between 1 and 10.\"\n\nThe assistant responded: \"I'm glad my answer was helpful! \ud83d\ude0a Is there anything else I can do for you today?\" This is completely irrelevant, does not pick a number, not helpful. So rating low.\n\nWe need to give short explanation then a rating 1-10. Probably 1 or 2. It's completely off. I'd give 1.", "finish_reason": "stop", "serialized_output": [{"content": "The response does not address the user's request at all; it fails to pick a number and is irrelevant, offering no helpful or accurate information.  \n1", "role": "assistant", "tool_calls": null, "function_call": null, "reasoning_content": "We need to judge the AI assistant's response to the user query: \"Pick a number between 1 and 10.\"\n\nThe assistant responded: \"I'm glad my answer was helpful! \ud83d\ude0a Is there anything else I can do for you today?\" This is completely irrelevant, does not pick a number, not helpful. So rating low.\n\nWe need to give short explanation then a rating 1-10. Probably 1 or 2. It's completely off. I'd give 1."}], "generation_start_time": 1763149473.5243623, "generation_end_time": 1763149485.7164843, "generation_time": 12.192121982574463, "judgement": "The response does not address the user's request at all; it fails to pick a number and is irrelevant, offering no helpful or accurate information.  \n1", "id": "tied:0", "question": "Pick a number between 1 and 10.", "answer": "I'm glad my answer was helpful! \ud83d\ude0a\nIs there anything else I can do for you today?", "chosen": 0, "num_correct": 10, "num_incorrect": 12}

class RewardBenchMetricsTies(BaseMetrics):
    """Metrics for RewardBench v2 Ties.

    This implementation focuses on the Ties-style aggregation that works over
    (is_correct, score) tuples grouped by prompt id. It derives directly from
    BaseMetrics rather than AnswerJudgementMetrics because RewardBench Ties
    does not expose an `expected_judgement` field.
    """

    # ---- Ties-specific helpers -------------------------------------------------

    @staticmethod
    def _compute_prompt_stats(samples: Iterable[Tuple[bool, float]]) -> Tuple[bool, float | None, float | None]:
        """Compute per-prompt stats for the Ties subset.

        Args:
            samples: Iterable of (is_correct, score) tuples for a single prompt
                     across all its generations.

        Returns:
            accurate: True if every correct answer outscores the best wrong one.
            different_correct_margin: spread between best and worst correct
                                      answers, or None if <2 correct answers.
            correct_incorrect_margin: gap between worst correct and best wrong
                                      answer. Can be negative/zero.
        """

        correct_scores = [float(s) for is_corr, s in samples if is_corr]
        incorrect_scores = [float(s) for is_corr, s in samples if not is_corr]

        if not correct_scores or not incorrect_scores:
            # If there are no correct or no incorrect answers, margins are
            # ill-defined; treat as not accurate with zero margins.
            return False, None, None

        best_correct = max(correct_scores)
        worst_correct = min(correct_scores)
        best_incorrect = max(incorrect_scores)
        different_correct_margin = best_correct - worst_correct if len(correct_scores) > 1 else None
        correct_incorrect_margin = worst_correct - best_incorrect
        accurate = correct_incorrect_margin > 0

        return accurate, different_correct_margin, correct_incorrect_margin

    def _aggregate_ties_over_generations(self, predictions: List[dict]) -> Dict[str, float]:
        """Aggregate Ties metrics over *generations* for a batch of samples.

        This mirrors the reference `process_single_model` behaviour but
        operates directly on the in-memory list of prediction dicts that
        correspond to multiple generations for Ties.
        """

        # Group per (sample_type, prompt_id)
        grouped_samples = defaultdict(list)

        for sample in predictions:
            sample_id = sample.get("id")
            print("sample_id:", sample_id)
            if not sample_id or ":" not in sample_id:
                # Not a Ties-style id; skip from Ties aggregation.
                continue
            try:
                sample_type, prompt_id_str = sample_id.split(":", 1)
                prompt_id = int(prompt_id_str)
            except ValueError:
                # If prompt_id is not an int, skip this sample for Ties logic.
                continue

            print("aggregating sample:", sample_id)

            # "chosen" is 1 for correct, 0 for incorrect in RewardBench
            is_correct = bool(sample.get("chosen", 0))

            # "judgement" is expected to be a rating in [1,10] or similar;
            # if it's a string, extract the last numeric token.
            raw_judgement = sample.get("judgement")
            if isinstance(raw_judgement, (int, float)):
                score = float(raw_judgement)
            else:
                tokens = str(raw_judgement).strip().split()
                score = float(tokens[-1])
         
            grouped_samples[(sample_type, prompt_id)].append((is_correct, score))

        # Compute per-prompt stats for ref / tied
        ref_stats, tied_stats = {}, {}
        for (sample_type, prompt_id), samples in grouped_samples.items():
            stats = self._compute_prompt_stats(samples)
            if sample_type == "ref":
                ref_stats[prompt_id] = stats
            elif sample_type == "tied":
                tied_stats[prompt_id] = stats
            else:
                print("Unknown sample type:", sample_type)
                continue

        print("ref_stats:", ref_stats)
        print("tied_stats:", tied_stats)

        # Global accuracy metrics (simple averages)
        ref_accuracy = (sum(s[0] for s in ref_stats.values()) / len(ref_stats)) if ref_stats else 0.0
        tied_accuracy = (sum(s[0] for s in tied_stats.values()) / len(tied_stats)) if tied_stats else 0.0

        # Margins are defined only on prompts that appear in both ref and tied
        correctness_preferred, correctness_preferred_hard = 0.0, 0.0
        correctness_margin_score = 0.0
        all_prompts = set(ref_stats) & set(tied_stats)
        if all_prompts:
            margins = []  # List of (diff_correct_margin, corr_incorrect_ties, corr_incorrect_ref)
            for pid in all_prompts:
                tied_diff_margin = tied_stats[pid][1] if tied_stats[pid][1] is not None else 0.0
                tied_corr_incorrect_margin = tied_stats[pid][2] if tied_stats[pid][2] is not None else 0.0
                ref_corr_incorrect_margin = ref_stats[pid][2] if ref_stats[pid][2] is not None else 0.0
                margins.append((tied_diff_margin, tied_corr_incorrect_margin, ref_corr_incorrect_margin))

            # correctness_preferred: ties margin between correct and incorrect > diff_correct_margin
            # correctness_preferred_hard: min(ref, ties) > diff_correct_margin
            for dcm, cit, cir in margins:
                if cit > dcm:
                    correctness_preferred += 1
                if min(cit, cir) > dcm:
                    correctness_preferred_hard += 1

            correctness_preferred /= len(margins)
            correctness_preferred_hard /= len(margins)

            # correctness_margin_score: tanh(min(ref, ties) / diff_correct_margin - 1)
            margin_scores = []
            for dcm, cit, cir in margins:
                if dcm == 0:
                    margin_scores.append(0.0)
                    continue
                ratio = min(cit, cir) / dcm - 1.0
                margin_scores.append(math.tanh(ratio))

            correctness_margin_score = sum(margin_scores) / len(margin_scores)

        overall_score = (
            0.30 * tied_accuracy
            + 0.30 * ref_accuracy
            + 0.20 * correctness_preferred
            + 0.20 * correctness_preferred_hard
            + 0.01 * correctness_margin_score
        )

        scores = {
            "ref_accuracy": ref_accuracy,
            "tied_accuracy": tied_accuracy,
            "correctness_preferred": correctness_preferred,
            "correctness_preferred_hard": correctness_preferred_hard,
            "correctness_margin_score": correctness_margin_score,
            "overall_preference_score": overall_score,
        }
        print("ties scores:", scores)
        return scores

    def __init__(self, compute_no_answer: bool = True):
        """Initialise RewardBench metrics container.

        In addition to the standard `BaseMetrics` bookkeeping (tokens, timing,
        etc.), we keep a flat list of all Ties generations so that we can
        later reproduce the reference `process_single_model` behaviour.
        """

        super().__init__(compute_no_answer=compute_no_answer)

        # Flat list of per-generation dicts (one entry per row in the
        # generations JSONL for the Ties subset).  This is what
        # `_aggregate_ties_over_generations` consumes.
        self._ties_predictions: List[dict] = []

    def reset(self):
        """Reset both base metrics state and stored Ties predictions."""

        super().reset()
        self._ties_predictions = []

    def update(self, predictions: List[dict]):
        """Update metrics with a single dataset example's generations.

        For RewardBench Ties we don't use per-example correctness metrics
        (there is no `expected_judgement` field).  Instead, we mimic the
        reference implementation which aggregates over *all* generations
        across the dataset.

        The incoming `predictions` list here is the K generations for a
        single prompt.  We extend `self._ties_predictions` with these raw
        dicts so that `get_metrics` can later run the reference-style
        aggregation on the complete set.
        """

        # Keep common bookkeeping from BaseMetrics (token counts, timing,
        # etc.) so that generic statistics like num_entries/avg_tokens still
        # work as expected.
        super().update(predictions)

        # Flatten all generations for this example into the global Ties pool.
        # This matches the reference code that expects a dataset with one
        # row per (id, score) pair.
        self._ties_predictions.extend(predictions)

    def get_metrics(self):
        """Extend parent metrics with Ties-specific aggregation.

        We call the standard `BaseMetrics.get_metrics` to obtain generic
        bookkeeping statistics (num_entries, avg_tokens, gen_seconds, etc.),
        and then add an extra "ties" bucket with the cross-generation Ties
        score if we have stored any generations.
        """

        metrics = super().get_metrics() or {}

        # To compute the Ties aggregates we need access to the per-example
        # predictions; BaseMetrics does not store them, so we maintain
        # `self._ties_predictions` ourselves during `update`.
        if self._ties_predictions:
            ties_stats = self._aggregate_ties_over_generations(self._ties_predictions)

            # If BaseMetrics did not accumulate any eval_dict entries for
            # this benchmark (common for RewardBench Ties), create a
            # synthetic evaluation mode so that summarize_results has a
            # well-defined key to print.
            if not metrics:
                metrics["ties"] = {}

            metrics.setdefault("ties", {}).update(ties_stats)

        return metrics

    # ---- Printing configuration for summarize_results -----------------------

    def evaluations_to_print(self) -> list[str]:
        """Evaluation modes to show for RewardBench.

        For RewardBench v2 Ties, we expose a single synthetic evaluation
        mode "ties" that aggregates across the whole subset.
        """

        # Base class default refers to pass@k-style modes, which we don't
        # use here. Instead, we only surface the Ties aggregate.
        return ["ties"]

    def metrics_to_print(self) -> dict[str, callable] | None:
        """Metrics to display for RewardBench Ties.

        We print the core Ties statistics plus common bookkeeping fields.
        """

        return {
            "num_entries": as_int,
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "ref_accuracy": as_float,
            "tied_accuracy": as_float,
            "correctness_preferred": as_float,
            "correctness_preferred_hard": as_float,
            "correctness_margin_score": as_float,
            "overall_preference_score": as_float,
        }


class RewardBenchMetricsPreference(BaseMetrics):
    """Metrics for RewardBench v2 Preference.
    It uses the following format for each prediction:  
       {"expected_answer": "[[D]]", "predicted_answer": "[[D]]", "symbolic_correct": true} 
    """
    def __init__(self, compute_no_answer: bool = True):
        super().__init__(compute_no_answer=compute_no_answer)
        self._correct, self._total = Counter(), Counter()

    def reset(self):
        super().reset()
        self._correct, self._total = Counter(), Counter()
    def update(self, predictions: List[dict]):
        super().update(predictions)

        for pred in predictions:
            self._correct["average"] += bool(pred["symbolic_correct"])
            self._total["average"] += 1
            self._correct[pred["subset"]] += bool(pred["symbolic_correct"])
            self._total[pred["subset"]] += 1

    def get_metrics(self):
        metrics = super().get_metrics()
        metrics["preference"] = {}
        for subset in self._total:
            accuracy = self._correct[subset] / (self._total[subset] if self._total[subset] else 1)
            metrics["preference"][f"{subset}_accuracy"] = accuracy

        return metrics

    def evaluations_to_print(self) -> list[str]:
        return ["preference"]

    def metrics_to_print(self) -> dict[str, callable] | None:
        return {
            "num_entries": as_int,
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "preference_accuracy": as_float,
        }


