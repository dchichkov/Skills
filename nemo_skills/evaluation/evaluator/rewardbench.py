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
import logging
import re

from tqdm import tqdm

from nemo_skills.evaluation.evaluator.base import BaseEvaluatorConfig
from nemo_skills.evaluation.math_grader import extract_answer
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))


def eval_rewardbench(cfg):
    eval_config = BaseEvaluatorConfig(**cfg)

    def extract_letter(text):
        # return prediction, if it contains [[A]] or similar boxed format
        extract_answer = re.search(r"^(\[\[\s*[A-Z]\s*\]\])$", text)
        if extract_answer:
            return extract_answer.group(1)        
        return None

    jsonl_file = eval_config.input_file
    with open(jsonl_file, "rt", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]
    with open(jsonl_file, "wt", encoding="utf-8") as fout:
        for sample in tqdm(data):
            # Per-sample values override config defaults for backward compatibility
  
            sample["predicted_answer"] = extract_letter(sample["generation"])            
            sample["symbolic_correct"] = sample["predicted_answer"] == sample["expected_answer"]
            fout.write(json.dumps(sample) + "\n")
