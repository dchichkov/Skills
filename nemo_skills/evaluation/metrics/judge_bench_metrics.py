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

"""
Metrics for JudgeBench evaluation.

Supports two modes:
1. Simple mode: Single judgment per sample, just computes accuracy
2. Two-pass mode: Two judgments per sample (normal and position-swapped),
   computes accuracy with position bias detection

Reference: https://github.com/ScalerLab/JudgeBench
"""

from collections import defaultdict
from typing import List

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float, as_int, as_percentage


def flip_judgment(decision: str) -> str:
    """Flip A and B in a judgment decision.
    
    When we swap the order of answers A and B, the judgment needs to be flipped:
    - If judge said "A" wins (when A and B were swapped), it means original B wins
    - So we flip "A" -> "B" and "B" -> "A"
    """
    if decision == "A":
        return "B"
    elif decision == "B":
        return "A"
    elif decision == "A>B":
        return "B>A"
    elif decision == "B>A":
        return "A>B"
    return decision  # tie or None stays the same


class JudgeBenchMetrics(BaseMetrics):
    """Metrics for JudgeBench preference evaluation.
    
    Handles two evaluation modes:
    
    1. Simple mode (single judgment):
       - Uses `predicted_winner` and `expected_winner` fields
       - Computes simple accuracy
       
    2. Two-pass mode (with position swap):
       - Uses `predicted_winner` (normal order) and `predicted_winner_swapped` (swapped order)
       - Computes:
         - accuracy: voting-based accuracy (both agree with label = correct)
         - all_correct: both judgments match label
         - all_incorrect: neither judgment matches label
         - consistency: both judgments agree (after flipping the swapped one)
    
    Expected data format:
    - `expected_winner`: "A" or "B" (ground truth)
    - `predicted_winner`: "A", "B", or "tie" (model's judgment in normal order)
    - `predicted_winner_swapped`: "A", "B", or "tie" (model's judgment in swapped order, optional)
    """
    
    def __init__(self, compute_no_answer: bool = True):
        super().__init__(compute_no_answer=compute_no_answer)
        self._predictions: List[dict] = []
    
    def reset(self):
        super().reset()
        self._predictions = []
    
    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Get correctness scores for a single prediction."""
        expected = prediction.get("expected_winner")
        predicted = prediction.get("predicted_winner")
        predicted_swapped = prediction.get("predicted_winner_swapped")
        
        scores = {}
        
        # Simple accuracy (single judgment)
        if predicted is not None and expected is not None:
            scores["correct"] = (predicted == expected)
        else:
            scores["correct"] = False
        
        # Also support symbolic_correct if already computed by evaluator
        if "symbolic_correct" in prediction:
            scores["symbolic_correct"] = prediction["symbolic_correct"]
        
        return scores
    
    def update(self, predictions: List[dict]):
        """Update metrics with predictions for a single sample."""
        super().update(predictions)
        
        # Store predictions for aggregation
        self._predictions.extend(predictions)
        
        # Compute pass@k style metrics using _get_score_dict
        predicted_answers = [pred.get("predicted_winner") for pred in predictions]
        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)
    
    def get_metrics(self):
        """Compute final metrics including JudgeBench-specific ones."""
        # Get base metrics (pass@k, majority@k, etc.)
        metrics = super().get_metrics()
        
        # Compute JudgeBench-specific metrics
        judge_bench_stats = self._compute_judge_bench_metrics()
        
        # Add to the main metrics dict
        if "pass@1" in metrics:
            metrics["pass@1"].update(judge_bench_stats)
        else:
            metrics["judge_bench"] = judge_bench_stats
            metrics["judge_bench"]["num_entries"] = self.total
        
        return metrics
    
    def _compute_judge_bench_metrics(self) -> dict:
        """Compute JudgeBench-specific metrics.
        
        Returns dict with:
        - accuracy: Simple accuracy (single pass) or voting accuracy (two pass)
        - consistency: How often both judgments agree (two pass only)
        - all_correct: Both judgments correct (two pass only)
        - all_incorrect: Both judgments incorrect (two pass only)
        """
        if not self._predictions:
            return {}
        
        # Check if we have two-pass data
        has_swapped = any("predicted_winner_swapped" in p for p in self._predictions)
        
        # Group predictions by sample ID or use each prediction as a sample
        # For now, treat each prediction independently
        
        if not has_swapped:
            # Simple mode: just compute accuracy
            return self._compute_simple_metrics()
        else:
            # Two-pass mode: compute full JudgeBench metrics
            return self._compute_two_pass_metrics()
    
    def _compute_simple_metrics(self) -> dict:
        """Compute metrics for simple (single-pass) evaluation."""
        n_correct = 0
        n_total = 0
        n_null = 0
        
        for pred in self._predictions:
            expected = pred.get("expected_winner")
            predicted = pred.get("predicted_winner")
            
            if predicted is None:
                n_null += 1
                n_total += 1
                continue
            
            n_total += 1
            if predicted == expected:
                n_correct += 1
        
        accuracy = 100.0 * n_correct / n_total if n_total > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "null_rate": 100.0 * n_null / n_total if n_total > 0 else 0.0,
        }
    
    def _compute_two_pass_metrics(self) -> dict:
        """Compute metrics for two-pass (position swap) evaluation.
        
        Following the reference implementation:
        - decision1: judgment in normal order
        - decision2: judgment in swapped order, then flipped back
        
        Voting mechanism:
        - Both agree with label: correct (+1)
        - Both disagree with label: incorrect (-1)
        - Otherwise: tie (0)
        """
        n_all_correct = 0
        n_all_incorrect = 0
        n_some_correct = 0
        
        n_correct = 0  # Voting-based correct
        n_incorrect = 0  # Voting-based incorrect
        n_tie = 0  # Voting-based tie
        
        n_nulls = 0
        n_inconsistent = 0
        n_total = 0
        
        for pred in self._predictions:
            label = pred.get("expected_winner")
            decision1 = pred.get("predicted_winner")
            decision2_raw = pred.get("predicted_winner_swapped")
            
            # Flip the swapped judgment back to normal orientation
            decision2 = flip_judgment(decision2_raw) if decision2_raw else None
            
            n_total += 1
            
            if decision1 is None or decision2 is None:
                n_nulls += 1
                continue
            
            # Consistency metrics
            if decision1 == label and decision2 == label:
                n_all_correct += 1
            elif decision1 != label and decision2 != label:
                n_all_incorrect += 1
            else:
                n_some_correct += 1
            
            # Position consistency (do both judgments agree after flipping?)
            if decision1 != decision2:
                n_inconsistent += 1
            
            # Voting-based correctness
            counter = 0
            flipped_label = flip_judgment(label)
            
            for decision in [decision1, decision2]:
                if decision == label:
                    counter += 1
                elif decision == flipped_label:
                    counter -= 1
            
            if counter > 0:
                n_correct += 1
            elif counter < 0:
                n_incorrect += 1
            else:
                n_tie += 1
        
        # Compute percentages
        n_valid = n_total - n_nulls
        
        return {
            "accuracy": 100.0 * n_correct / n_valid if n_valid > 0 else 0.0,
            "all_correct": 100.0 * n_all_correct / n_valid if n_valid > 0 else 0.0,
            "all_incorrect": 100.0 * n_all_incorrect / n_valid if n_valid > 0 else 0.0,
            "some_correct": 100.0 * n_some_correct / n_valid if n_valid > 0 else 0.0,
            "consistency": 100.0 * (n_valid - n_inconsistent) / n_valid if n_valid > 0 else 0.0,
            "null_rate": 100.0 * n_nulls / n_total if n_total > 0 else 0.0,
            "tie_rate": 100.0 * n_tie / n_valid if n_valid > 0 else 0.0,
        }
    
    def evaluations_to_print(self) -> list[str]:
        """Evaluation modes to show."""
        # Check if we have two-pass data
        has_swapped = any("predicted_winner_swapped" in p for p in self._predictions)
        
        if has_swapped:
            return ["pass@1", "judge_bench"]
        else:
            return [f"pass@1[avg-of-{self.max_k}]", f"pass@{self.max_k}"]
    
    def metrics_to_print(self) -> dict[str, callable] | None:
        """Metrics to display."""
        return {
            "num_entries": as_int,
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "correct": as_percentage,
            "symbolic_correct": as_percentage,
            "accuracy": as_percentage,
            "all_correct": as_percentage,
            "all_incorrect": as_percentage,
            "some_correct": as_percentage,
            "consistency": as_percentage,
            "null_rate": as_percentage,
            "tie_rate": as_percentage,
        }
