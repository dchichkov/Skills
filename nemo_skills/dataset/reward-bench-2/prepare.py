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
from datasets import load_dataset, concatenate_datasets
import numpy as np



if __name__ == "__main__":

    dataset = load_dataset("allenai/reward-bench-2", split='test')
    # select some samples from Ties
    #dataset = dataset.filter(lambda x: x["subset"] == "Ties")

    # select some samples from Ties and NonTies
    #dataset = concatenate_datasets([dataset.filter(lambda x: x["subset"] == "Ties").select(range(30)),
    #                                dataset.filter(lambda x: x["subset"] != "Ties").select(range(10))])
    print(f"Prepared dataset with {len(dataset)} samples")

    # dumping the data as test.jsonl, note that the shuffling logic is not ideal, but it matches the one in reward-bench-2
    np.random.seed(42)
    output_path = Path(__file__).parent / "preference" / "test.jsonl"
    ties_path = Path(__file__).parent  / "ties" / "test.jsonl"
    ratings_path = Path(__file__).parent  / "ratings" / "test.jsonl"
    with open(output_path, "w") as test, open(ties_path, "w") as ties, open(ratings_path, "w") as ratings:
        for sample in dataset:
            for answer in sample["chosen"]:
                prepared = {
                    "id": sample["id"],
                    "subset": sample["subset"],
                    "question": sample["prompt"],
                    "answer": answer,
                    "chosen": 1,
                    "num_correct": sample["num_correct"],
                    "num_incorrect": sample["num_incorrect"],
                }
                ratings.write(json.dumps(prepared) + "\n")
                if sample["subset"] == "Ties":
                    ties.write(json.dumps(prepared) + "\n")

            for answer in sample["rejected"]:
                prepared = {
                    "id": sample["id"],
                    "subset": sample["subset"],
                    "question": sample["prompt"],
                    "answer": answer,
                    "chosen": 0,
                    "num_correct": sample["num_correct"],
                    "num_incorrect": sample["num_incorrect"],
                }
                ratings.write(json.dumps(prepared) + "\n")
                if sample["subset"] == "Ties":
                    ties.write(json.dumps(prepared) + "\n")

            if sample["subset"] != "Ties":
                assert len(sample["chosen"]) == 1
                answer_a = sample["chosen"][0]
                answer_b,answer_c, answer_d  = sample["rejected"][:3]

                # shuffle, this uses the same logic as run_generative_v2.py
                chosen, shuffle_option = "[[A]]", np.random.randint(0,4)
                if shuffle_option == 1:   answer_a, answer_b, chosen = answer_b, answer_a, "[[B]]"
                elif shuffle_option == 2: answer_a, answer_c, chosen = answer_c, answer_a, "[[C]]"
                elif shuffle_option == 3: answer_a, answer_d, chosen = answer_d, answer_a, "[[D]]"

                prepared = {
                    "id": sample["id"],
                    "subset": sample["subset"],
                    "question": sample["prompt"],
                    "answer_a": answer_a,
                    "answer_b": answer_b,
                    "answer_c": answer_c,
                    "answer_d": answer_d,
                    "expected_answer": chosen
                }

                test.write(json.dumps(prepared) + "\n")

