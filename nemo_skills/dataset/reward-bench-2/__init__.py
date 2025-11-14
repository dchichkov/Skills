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




IS_BENCHMARK_GROUP = True
METRICS_TYPE = "reward-bench-2"

SCORE_MODULE = "nemo_skills.dataset.reward-bench-2.score"

BENCHMARKS = {
    "reward-bench-2.preference": {
        "GENERATION_ARGS": "++prompt_config=judge/reward-bench/reward-bench-2.preference ++generation_key=generation ++eval_type=rewardbench",
    },

    # Ties only split, included in the .ratings evaluation
    #
    #"reward-bench-2.ties": {
    #    "GENERATION_ARGS": "++prompt_config=judge/reward-bench/reward-bench-2.ties ++generation_key=judgement",
    #},

    "reward-bench-2.ratings": {
        "GENERATION_ARGS": "++prompt_config=judge/reward-bench/reward-bench-2.ratings ++generation_key=judgement",
    },

}