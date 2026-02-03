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
Evaluator for JudgeBench preference evaluation.

Handles multiple output formats from different judge prompts:
- Arena Hard style: [[A>B]], [[B>A]], [[A>>B]], [[B>>A]], [[A=B]]
- Skywork style: [[A]], [[B]]
- AutoJ style: "Response 1", "Response 2", "Tie"
- Vanilla style: "Output (a)", "Output (b)"
- Prometheus style: [RESULT] A, [RESULT] B
- JudgeLM style: score-based (first score > second = A wins)
- PandaLM style: "1", "2", "Tie"
"""

import json
import logging
import re

from tqdm import tqdm

from nemo_skills.evaluation.evaluator.base import BaseEvaluatorConfig
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class JudgeBenchEvaluatorConfig(BaseEvaluatorConfig):
    # If specified, only use this extraction method. Otherwise try all.
    # Options: "arena_hard", "skywork", "autoj", "vanilla", "prometheus", "judgelm", "pandalm", "auto"
    extraction_mode: str = "auto"


def extract_winner(text: str, mode: str = "auto") -> str | None:
    """
    Extract the winner (A or B) from judge model output.
    
    Tries multiple extraction patterns based on common judge prompt formats.
    Returns "A", "B", "tie", or None if extraction fails.
    
    Args:
        text: The model's generation/output text
        mode: Extraction mode - "auto" tries all patterns, or specify a specific one
        
    Returns:
        "A", "B", "tie", or None
    """
    if not text:
        return None
    
    text = text.strip()
    
    extractors = {
        "arena_hard": _extract_arena_hard,
        "skywork": _extract_skywork,
        "autoj": _extract_autoj,
        "vanilla": _extract_vanilla,
        "prometheus": _extract_prometheus,
        "judgelm": _extract_judgelm,
        "pandalm": _extract_pandalm,
    }
    
    if mode != "auto" and mode in extractors:
        return extractors[mode](text)
    
    # Auto mode: try each extractor in order of specificity
    # More specific patterns first (arena_hard with comparison operators)
    # then simpler patterns (skywork with just [[A]])
    for extractor_name in ["arena_hard", "prometheus", "autoj", "vanilla", "pandalm", "skywork", "judgelm"]:
        result = extractors[extractor_name](text)
        if result is not None:
            LOG.debug(f"Extracted winner '{result}' using {extractor_name} extractor")
            return result
    
    return None


def _extract_arena_hard(text: str) -> str | None:
    """
    Arena Hard style: [[A>B]], [[A>>B]], [[B>A]], [[B>>A]], [[A=B]]
    
    The first letter in the comparison is the winner if > or >>.
    """
    # Match patterns like [[A>B]], [[A>>B]], [[B>A]], [[B>>A]]
    match = re.search(r'\[\[\s*([AB])\s*>+\s*([AB])\s*\]\]', text, re.IGNORECASE)
    if match:
        winner = match.group(1).upper()
        return winner
    
    # Match tie pattern [[A=B]]
    match = re.search(r'\[\[\s*[AB]\s*=\s*[AB]\s*\]\]', text, re.IGNORECASE)
    if match:
        return "tie"
    
    return None


def _extract_skywork(text: str) -> str | None:
    """
    Skywork style: [[A]] or [[B]]
    
    Simple boxed single letter format.
    """
    match = re.search(r'\[\[\s*([AB])\s*\]\]', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def _extract_autoj(text: str) -> str | None:
    """
    AutoJ style: "Response 1" (=A), "Response 2" (=B), or "Tie"
    
    Looks for patterns like "final decision is Response 1" or just "Response 1" at the end.
    """
    # Look for "Response 1" or "Response 2" pattern
    # More specific: "decision is Response X"
    match = re.search(r'(?:decision\s+is\s+)?Response\s*1\b', text, re.IGNORECASE)
    if match:
        return "A"
    
    match = re.search(r'(?:decision\s+is\s+)?Response\s*2\b', text, re.IGNORECASE)
    if match:
        return "B"
    
    # Check for tie
    if re.search(r'\bTie\b', text, re.IGNORECASE):
        return "tie"
    
    return None


def _extract_vanilla(text: str) -> str | None:
    """
    Vanilla style: "Output (a)" or "Output (b)"
    """
    # Look for "Output (a)" or "Output (b)"
    match = re.search(r'Output\s*\(\s*([ab])\s*\)', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def _extract_prometheus(text: str) -> str | None:
    """
    Prometheus style: [RESULT] A or [RESULT] B
    """
    match = re.search(r'\[RESULT\]\s*([AB])\b', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def _extract_judgelm(text: str) -> str | None:
    """
    JudgeLM style: Two scores separated by space, e.g., "8 7"
    
    First score is for Assistant 1 (A), second for Assistant 2 (B).
    Higher score wins.
    """
    # Look for two numbers (scores) at the beginning of the response
    match = re.search(r'^[\s]*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)', text)
    if match:
        score_a = float(match.group(1))
        score_b = float(match.group(2))
        if score_a > score_b:
            return "A"
        elif score_b > score_a:
            return "B"
        else:
            return "tie"
    return None


def _extract_pandalm(text: str) -> str | None:
    """
    PandaLM style: "1" (=A), "2" (=B), or "Tie"
    
    Usually at the end of the response.
    """
    # Check for Tie first
    if re.search(r'\bTie\b', text, re.IGNORECASE):
        return "tie"
    
    # Look for standalone 1 or 2 (usually the final verdict)
    # Be careful not to match numbers in other contexts
    lines = text.strip().split('\n')
    last_line = lines[-1].strip() if lines else ""
    
    if last_line == "1":
        return "A"
    elif last_line == "2":
        return "B"
    
    return None


def normalize_expected_answer(expected: str) -> str | None:
    """
    Normalize expected_answer to "A" or "B" format.
    
    Handles various formats:
    - "[[A]]", "[[B]]" -> "A", "B"
    - "A", "B" -> "A", "B"
    - "[[A>B]]" -> "A"
    - "[[B>A]]" -> "B"
    """
    if not expected:
        return None
    
    expected = expected.strip()
    
    # Handle [[A>B]] or [[B>A]] format
    match = re.search(r'\[\[\s*([AB])\s*>+\s*[AB]\s*\]\]', expected, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Handle [[A]] or [[B]] format
    match = re.search(r'\[\[\s*([AB])\s*\]\]', expected, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Handle plain A or B
    if expected.upper() in ("A", "B"):
        return expected.upper()
    
    return None


def eval_judge_bench(cfg):
    """
    Evaluate JudgeBench predictions.
    
    For each sample:
    1. Extract the predicted winner from model generation
    2. Normalize both predicted and expected answers to "A"/"B" format
    3. Compare and set symbolic_correct
    """
    eval_config = JudgeBenchEvaluatorConfig(**cfg)
    
    jsonl_file = eval_config.input_file
    with open(jsonl_file, "rt", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]
    
    extraction_stats = {"A": 0, "B": 0, "tie": 0, "failed": 0}
    
    with open(jsonl_file, "wt", encoding="utf-8") as fout:
        for sample in tqdm(data, desc="Evaluating JudgeBench"):
            generation = sample.get("generation", "")
            expected = sample.get("expected_answer", "")
            
            # Extract predicted winner
            predicted = extract_winner(generation, mode=eval_config.extraction_mode)
            
            # Track stats
            if predicted in ("A", "B", "tie"):
                extraction_stats[predicted] += 1
            else:
                extraction_stats["failed"] += 1
            
            # Normalize expected answer
            expected_normalized = normalize_expected_answer(expected)
            
            # Format predicted answer to match expected format for display
            if predicted == "A":
                predicted_formatted = "[[A]]"
            elif predicted == "B":
                predicted_formatted = "[[B]]"
            elif predicted == "tie":
                predicted_formatted = "[[tie]]"
            else:
                predicted_formatted = None
            
            sample["predicted_answer"] = predicted_formatted
            sample["predicted_winner"] = predicted  # Normalized form for debugging
            sample["expected_winner"] = expected_normalized  # Normalized form for debugging
            
            # Compare normalized forms
            sample["symbolic_correct"] = (predicted == expected_normalized) if predicted and expected_normalized else False
            
            fout.write(json.dumps(sample) + "\n")
    
    # Log extraction statistics
    total = sum(extraction_stats.values())
    LOG.info(
        f"Extraction stats: A={extraction_stats['A']}, B={extraction_stats['B']}, "
        f"tie={extraction_stats['tie']}, failed={extraction_stats['failed']} (total={total})"
    )
