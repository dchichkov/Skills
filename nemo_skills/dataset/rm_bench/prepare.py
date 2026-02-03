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
Prepare RMBench dataset for evaluation.

RMBench format:
{
    "id": unique identifier,
    "prompt": the prompt given to the model,
    "chosen": [resp_concise, resp_detailed_plain, resp_detailed_markdown],
    "rejected": [resp_concise, resp_detailed_plain, resp_detailed_markdown],
    "domain": "chat, code, math, safety-refuse, safety-response"
}

RMBench evaluation compares ALL combinations of chosen vs rejected (3x3 = 9 per sample).
The accuracy matrix has:
- Rows: chosen response style index (0=concise, 1=detailed_plain, 2=detailed_markdown)
- Columns: rejected response style index

Metrics:
- hard_acc: Upper-right triangle (simpler chosen vs fancier rejected)
- normal_acc: Diagonal (same style comparisons)
- easy_acc: Lower-left triangle (fancier chosen vs simpler rejected)

We create all 9 comparison pairs per sample with random position shuffling.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

# Style indices and names (order matters for hard/normal/easy computation)
STYLES = ["concise", "detailed_plain", "detailed_markdown"]


if __name__ == "__main__":
    random.seed(42)

    dataset = load_dataset("Haoxiang-Wang/RM-Bench", split='test')
    print(f"Loaded dataset with {len(dataset)} samples")

    output_path = Path(__file__).parent / "preference" / "test.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    with open(output_path, "w") as fout:
        for idx, sample in enumerate(dataset):
            # Create all 9 comparison pairs (chosen[i] vs rejected[j])
            for chosen_idx in range(len(STYLES)):
                for rejected_idx in range(len(STYLES)):
                    chosen_response = sample["chosen"][chosen_idx]
                    rejected_response = sample["rejected"][rejected_idx]

                    # Determine comparison type based on matrix position
                    if chosen_idx < rejected_idx:
                        comparison_type = "hard"  # Upper-right: simpler chosen vs fancier rejected
                    elif chosen_idx == rejected_idx:
                        comparison_type = "normal"  # Diagonal: same style
                    else:
                        comparison_type = "easy"  # Lower-left: fancier chosen vs simpler rejected

                    # Randomly assign to A or B to avoid position bias
                    if random.random() < 0.5:
                        answer_a = chosen_response
                        answer_b = rejected_response
                        expected_answer = "[[A]]"  # A is chosen (better)
                    else:
                        answer_a = rejected_response
                        answer_b = chosen_response
                        expected_answer = "[[B]]"  # B is chosen (better)

                    prepared = {
                        "id": f"{sample['id']}_{chosen_idx}_{rejected_idx}",
                        "question": sample["prompt"],
                        "answer_a": answer_a,
                        "answer_b": answer_b,
                        "expected_answer": expected_answer,
                        "original_id": sample["id"],
                        "domain": sample["domain"],
                        "chosen_style": STYLES[chosen_idx],
                        "rejected_style": STYLES[rejected_idx],
                        "chosen_idx": chosen_idx,
                        "rejected_idx": rejected_idx,
                        "comparison_type": comparison_type,
                    }

                    fout.write(json.dumps(prepared) + "\n")
                    total_samples += 1

    print(f"Prepared dataset with {total_samples} samples ({len(dataset)} original × 9 comparisons)")
