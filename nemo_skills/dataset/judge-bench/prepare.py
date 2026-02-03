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
import json
from pathlib import Path
from datasets import load_dataset


if __name__ == "__main__":
    dataset = load_dataset("ScalerLab/JudgeBench", split='gpt')
    print(f"Prepared dataset with {len(dataset)} samples")

    output_path = Path(__file__).parent / "preference" / "test.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as fout:
        for sample in dataset:
            # Convert label like "B>A" to expected_answer like "[[B]]"
            label = sample["label"]

            if label == "A>B":
                expected_answer = "[[A]]"
            elif label == "B>A":
                expected_answer = "[[B]]"
            else:
                raise ValueError(f"Invalid label: {label}")

            prepared = {
                "id": sample["pair_id"],
                "question": sample["question"],
                "answer_a": sample["response_A"],
                "answer_b": sample["response_B"],
                "expected_answer": expected_answer,
            }

            fout.write(json.dumps(prepared) + "\n")
