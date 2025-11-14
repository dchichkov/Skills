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

# Scoring based on: https://artificialanalysis.ai/methodology/intelligence-benchmarking#intelligence-index-evaluation-suite-overview


# Overall Ties score formula
# ref_accuracy = mean of accurate over all reference prompts.
# tied_accuracy = mean of accurate over all tied prompts.
# For prompts present in both ref and tied:
# diff_corr_margin = different_correct_margin on tied
# corr_incorrect_ties = correct_incorrect_margin on tied
# corr_incorrect_ref = correct_incorrect_margin on ref
#
# Then compute:
# correctness_preferred = mean(corr_incorrect_ties > diff_corr_margin)
# correctness_preferred_hard = mean(min(corr_incorrect_ref, corr_incorrect_ties) > diff_corr_margin)
# correctness_margin_score = mean(tanh(min(corr_incorrect_ref, corr_incorrect_ties) / diff_corr_margin − 1)), with NaNs treated as 0 when diff_corr_margin is 0.
# Final weighted overall score:
#    overall = 0.30*tied_accuracy + 0.30*ref_accuracy + 0.20*correctness_preferred +
#              0.20*correctness_preferred_hard + 0.01*correctness_margin_score*tied_accuracy
#
# Only prompt IDs that appear in both ref and tied contribute to the margin-based terms.



def compute_score(metrics: dict):
    """Compute overall RewardBench v2 score from individual benchmark metrics."""
    print(metrics)

    overall_score = 0.0
    return {
        "overall_score": overall_score,
        }

    mmlu_pro = metrics["mmlu-pro"]["pass@1"]["symbolic_correct"]
    hle = metrics["hle"]["pass@1"]["judge_correct"]
    gpqa = metrics["gpqa"]["pass@1"]["symbolic_correct"]

    aime25 = metrics["aime24"]["pass@1[avg-of-10]"]["symbolic_correct"]

    scicode = metrics["scicode"]["pass@1[avg-of-3]"]["subtask_accuracy"]
    livecodebench = metrics["livecodebench"]["pass@1[avg-of-3]"]["accuracy"]

    ifbench = metrics["ifbench"]["pass@1[avg-of-5]"]["average_score"]

    aalcr = metrics["aalcr"]["pass@1[avg-of-3]"]["judge_correct"]

    math_score = aime25
    code_score = (scicode + livecodebench) / 2

    overall_score = (mmlu_pro + hle + gpqa + aime25 + scicode + livecodebench + ifbench + aalcr) / 8
    return {
        "overall_score": overall_score,
        "math_score": math_score,
        "code_score": code_score,
        "mmlu_pro": mmlu_pro,
        "hle": hle,
        "gpqa": gpqa,
        "aime25": aime25,
        "scicode": scicode,
        "livecodebench": livecodebench,
        "ifbench": ifbench,
        "aalcr": aalcr,
    }
