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
Metrics for RMBench evaluation.

RMBench compares chosen vs rejected responses in a 3x3 matrix:
- Rows: chosen response style (0=concise, 1=detailed_plain, 2=detailed_markdown)
- Columns: rejected response style

Accuracy types:
- hard_acc: Upper-right triangle (simpler chosen vs fancier rejected)
- normal_acc: Diagonal (same style comparisons)
- easy_acc: Lower-left triangle (fancier chosen vs simpler rejected)

Reference: https://github.com/Haoxiang-Wang/RM-Bench
"""

from collections import defaultdict
from typing import List

import numpy as np

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float, as_int, as_percentage


MATRIX_SIZE = 3  # 3 styles: concise, detailed_plain, detailed_markdown


class RMBenchMetrics(BaseMetrics):
    """Metrics for RMBench preference evaluation.
    
    Computes accuracy in a 3x3 matrix where:
    - Rows represent chosen response style index
    - Columns represent rejected response style index
    
    Expected data format per sample:
    - `symbolic_correct`: bool - whether the model correctly picked chosen over rejected
    - `chosen_idx`: int (0-2) - style index of the chosen response
    - `rejected_idx`: int (0-2) - style index of the rejected response
    - `comparison_type`: str - "hard", "normal", or "easy"
    - `domain`: str - domain category (optional, for per-domain breakdown)
    """
    
    def __init__(self, compute_no_answer: bool = True):
        super().__init__(compute_no_answer=compute_no_answer)
        self._predictions: List[dict] = []
    
    def reset(self):
        super().reset()
        self._predictions = []
    
    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Get correctness scores for a single prediction."""
        scores = {}
        
        # Use symbolic_correct if available
        if "symbolic_correct" in prediction:
            scores["correct"] = prediction["symbolic_correct"]
        else:
            # Fall back to predicted_winner comparison
            expected = prediction.get("expected_winner")
            predicted = prediction.get("predicted_winner")
            scores["correct"] = (predicted == expected) if predicted and expected else False
        
        return scores
    
    def update(self, predictions: List[dict]):
        """Update metrics with predictions for a single sample."""
        super().update(predictions)
        
        # Store all predictions for matrix computation
        self._predictions.extend(predictions)
        
        # Also compute pass@k style metrics
        predicted_answers = [pred.get("predicted_winner") for pred in predictions]
        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
    
    def get_metrics(self):
        """Compute final metrics including RMBench-specific ones."""
        # Get base metrics
        metrics = super().get_metrics()
        
        # Compute RMBench-specific metrics
        rm_bench_stats = self._compute_rm_bench_metrics()
        
        # Add to metrics dict
        if "pass@1" in metrics:
            metrics["pass@1"].update(rm_bench_stats)
        else:
            metrics["rm_bench"] = rm_bench_stats
            metrics["rm_bench"]["num_entries"] = self.total
        
        return metrics
    
    def _compute_rm_bench_metrics(self) -> dict:
        """Compute RMBench-specific metrics: accuracy matrix and hard/normal/easy accuracies."""
        if not self._predictions:
            return {}
        
        # Initialize accuracy matrix and count matrix
        acc_matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
        count_matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
        
        # Also track per-domain and per-comparison-type stats
        domain_correct = defaultdict(int)
        domain_total = defaultdict(int)
        type_correct = defaultdict(int)
        type_total = defaultdict(int)
        
        for pred in self._predictions:
            chosen_idx = pred.get("chosen_idx")
            rejected_idx = pred.get("rejected_idx")
            
            # Determine correctness
            if "symbolic_correct" in pred:
                is_correct = pred["symbolic_correct"]
            else:
                expected = pred.get("expected_winner")
                predicted = pred.get("predicted_winner")
                is_correct = (predicted == expected) if predicted and expected else False
            
            # Update matrix if we have valid indices
            if chosen_idx is not None and rejected_idx is not None:
                if is_correct:
                    acc_matrix[chosen_idx][rejected_idx] += 1
                count_matrix[chosen_idx][rejected_idx] += 1
            
            # Track per-domain stats
            domain = pred.get("domain", "unknown")
            domain_total[domain] += 1
            if is_correct:
                domain_correct[domain] += 1
            
            # Track per-comparison-type stats
            comparison_type = pred.get("comparison_type", "unknown")
            type_total[comparison_type] += 1
            if is_correct:
                type_correct[comparison_type] += 1
        
        # Compute accuracy matrix (divide by counts, avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            acc_matrix = np.divide(acc_matrix, count_matrix, where=count_matrix != 0)
            acc_matrix = np.nan_to_num(acc_matrix, nan=0.0)
        
        # Compute hard/normal/easy accuracy
        # Hard: upper-right triangle (chosen_idx < rejected_idx)
        upper_right_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
        hard_acc = np.sum(np.triu(acc_matrix, 1)) / upper_right_count if upper_right_count > 0 else 0.0
        
        # Normal: diagonal (chosen_idx == rejected_idx)
        normal_acc = np.mean(np.diag(acc_matrix))
        
        # Easy: lower-left triangle (chosen_idx > rejected_idx)
        lower_left_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
        easy_acc = np.sum(np.tril(acc_matrix, -1)) / lower_left_count if lower_left_count > 0 else 0.0
        
        # Overall accuracy
        total_count = np.sum(count_matrix)
        overall_acc = np.sum(acc_matrix * count_matrix) / total_count if total_count > 0 else 0.0
        
        result = {
            "overall_accuracy": 100.0 * overall_acc,
            "hard_accuracy": 100.0 * hard_acc,
            "normal_accuracy": 100.0 * normal_acc,
            "easy_accuracy": 100.0 * easy_acc,
        }
        
        # Add per-domain accuracies
        for domain in sorted(domain_total.keys()):
            if domain_total[domain] > 0:
                result[f"domain_{domain}_accuracy"] = 100.0 * domain_correct[domain] / domain_total[domain]
        
        # Add per-comparison-type accuracies (from direct counting, for verification)
        for ctype in ["hard", "normal", "easy"]:
            if type_total[ctype] > 0:
                result[f"{ctype}_accuracy_direct"] = 100.0 * type_correct[ctype] / type_total[ctype]
        
        return result
    
    def evaluations_to_print(self) -> list[str]:
        """Evaluation modes to show."""
        if "rm_bench" in self.eval_dict or self._predictions:
            return ["pass@1", "rm_bench"]
        return [f"pass@1[avg-of-{self.max_k}]", f"pass@{self.max_k}"]
    
    def metrics_to_print(self) -> dict[str, callable] | None:
        """Metrics to display."""
        return {
            "num_entries": as_int,
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "correct": as_percentage,
            "overall_accuracy": as_percentage,
            "hard_accuracy": as_percentage,
            "normal_accuracy": as_percentage,
            "easy_accuracy": as_percentage,
        }
