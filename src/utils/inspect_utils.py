from inspect_ai.log import EvalLog
from typing import Dict, Any, List
import numpy as np
import random
import math

def extract_scores_from_log(log: EvalLog) -> Dict[str, Any]:
    """Extract scores and metrics from the evaluation log.

    Args:
        log: The evaluation log from Inspect

    Returns:
        Dictionary containing extracted results including metadata
    """
    if log.results is None:
        # Eval failed before producing results - extract error info
        error_msg = f"Eval failed: status={log.status}"
        if log.error:
            error_msg += f", error={log.error.message}"
        raise RuntimeError(error_msg)

    results = {
        "model": log.eval.model,
        "total_samples": log.results.total_samples,
        "completed_samples": log.results.completed_samples
    }

    # Include metadata if present
    if log.eval.metadata:
        results["metadata"] = log.eval.metadata

    for score in log.results.scores:
        score_dict = {}
        for metric_name, metric_value in score.metrics.items():
            score_dict[metric_name] = metric_value.value
        score_dict["scorer"] = score.scorer
        results[score.name] = score_dict

    return results

def group_by_sample_epoch(log: EvalLog, scorer: str, metric: str = 'accuracy') -> Dict[str, List[EvalLog]]:
    """Group EvalLog by sample and epoch.

    Args:
        log: The evaluation log from Inspect
        scorer: Name of the scorer to use (e.g., 'hle_scorer', 'gpqa_scorer')
        metric: Name of the metric within the scorer (default: 'accuracy')

    Returns:
        Dictionary mapping sample ID to list of scores
    """
    result = {}
    for sample in log.samples:
        if sample.id not in result:
            result[sample.id] = []
        score = 1 if sample.scores[scorer].value == 'C' else 0
        result[sample.id].append(score)

    return result

def compute_bootstrap_over_epochs(log: EvalLog, scorer: Dict[str, str], n_bootstrap: int = 1000) -> Dict[str, float]:
    """Compute bootstrap over epochs.

    Args:
        log: The evaluation log from Inspect
        scorer: Dict with 'scorer' and optional 'metric' keys (e.g., {'scorer': 'hle_scorer', 'metric': 'accuracy'})
                If 'metric' is not provided, defaults to 'accuracy'
        n_bootstrap: Number of bootstrap samples

    Returns:
        Dictionary with accuracy, stderr, scorer, and epochs
    """
    scorer_name = scorer['scorer']
    metric_name = scorer.get('metric', 'accuracy')

    grouped = group_by_sample_epoch(log, scorer_name, metric_name)
    scores_grouped = list(grouped.values())
    n_epochs = len(scores_grouped[0])
    bootstraps = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        samples = [random.choice(scores) for scores in scores_grouped]
        bootstraps[i] = np.mean(samples)

    results = {
        "accuracy": np.mean(bootstraps),
        "stderr": np.std(bootstraps),
        "scorer": "manual_bootstrap",
        "epochs": n_epochs
    }
    return results

def compute_pass_at_k(log: EvalLog, scorer: Dict[str, str], n_bootstrap: int = 1000) -> Dict[str, Dict[str, float]]:
    """Compute pass@k for k=1 to k=n_epochs.

    For each k, randomly select k answers per question and check if at least 1 is correct.
    Uses bootstrap to compute stderr, except for k=epochs where stderr=0.

    Args:
        log: The evaluation log from Inspect
        scorer: Dict with 'scorer' and optional 'metric' keys (e.g., {'scorer': 'hle_scorer', 'metric': 'accuracy'})
                If 'metric' is not provided, defaults to 'accuracy'
        n_bootstrap: Number of bootstrap samples (default 1000)

    Returns:
        Dictionary mapping k (as string) to accuracy and stderr
    """
    scorer_name = scorer['scorer']
    metric_name = scorer.get('metric', 'accuracy')

    grouped = group_by_sample_epoch(log, scorer_name, metric_name)
    scores_grouped = list(grouped.values())
    n_epochs = len(scores_grouped[0])

    result = {}

    for k in range(1, n_epochs + 1):
        if k == n_epochs:
            accuracy = np.mean([max(scores) for scores in scores_grouped])
            stderr = 0.0
        else:
            bootstraps = np.zeros(n_bootstrap)

            for i in range(n_bootstrap):
                samples = []
                for scores in scores_grouped:
                    selected = random.sample(scores, k)
                    samples.append(1 if any(selected) else 0)
                bootstraps[i] = np.mean(samples)

            accuracy = np.mean(bootstraps)
            stderr = np.std(bootstraps)

        result[str(k)] = {
            "accuracy": accuracy,
            "stderr": stderr
        }

    return result


def compute_accuracy_stderr_from_correctness(
    per_sample_epoch_correct: Dict[str, List[int]],
) -> Dict[str, float]:
    """Compute overall accuracy and stderr from stored per-sample epoch correctness.

    Args:
        per_sample_epoch_correct: sample_id -> list of 0/1 correctness values (length=epochs)

    Returns:
        Dict with keys: accuracy, stderr
    """
    flat: list[float] = []
    for arr in per_sample_epoch_correct.values():
        for v in arr:
            flat.append(float(v))

    if not flat:
        raise ValueError("No correctness data provided (empty checkpoint?)")

    x = np.asarray(flat, dtype=np.float64)
    acc = float(np.mean(x))
    stderr = float(np.std(x, ddof=0) / math.sqrt(x.size))
    return {"accuracy": acc, "stderr": stderr}


def compute_bootstrap_over_epochs_from_correctness(
    per_sample_epoch_correct: Dict[str, List[int]],
    n_bootstrap: int = 1000,
) -> Dict[str, float]:
    """Compute the same manual bootstrap metric, without needing an EvalLog."""
    scores_grouped = list(per_sample_epoch_correct.values())
    if not scores_grouped:
        raise ValueError("No correctness data provided (empty checkpoint?)")
    n_epochs = len(scores_grouped[0])
    if any(len(s) != n_epochs for s in scores_grouped):
        raise ValueError("Inconsistent epoch lengths in correctness data")

    bootstraps = np.zeros(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        samples = [random.choice(scores) for scores in scores_grouped]
        bootstraps[i] = np.mean(samples)

    return {
        "accuracy": float(np.mean(bootstraps)),
        "stderr": float(np.std(bootstraps)),
        "scorer": "manual_bootstrap",
        "epochs": int(n_epochs),
    }


def compute_pass_at_k_from_correctness(
    per_sample_epoch_correct: Dict[str, List[int]],
    n_bootstrap: int = 1000,
) -> Dict[str, Dict[str, float]]:
    """Compute pass@k for k=1..epochs, without needing an EvalLog."""
    scores_grouped = list(per_sample_epoch_correct.values())
    if not scores_grouped:
        raise ValueError("No correctness data provided (empty checkpoint?)")
    n_epochs = len(scores_grouped[0])
    if any(len(s) != n_epochs for s in scores_grouped):
        raise ValueError("Inconsistent epoch lengths in correctness data")

    result: Dict[str, Dict[str, float]] = {}

    for k in range(1, n_epochs + 1):
        if k == n_epochs:
            accuracy = float(np.mean([max(scores) for scores in scores_grouped]))
            stderr = 0.0
        else:
            bootstraps = np.zeros(n_bootstrap, dtype=np.float64)
            for i in range(n_bootstrap):
                samples = []
                for scores in scores_grouped:
                    selected = random.sample(scores, k)
                    samples.append(1 if any(selected) else 0)
                bootstraps[i] = np.mean(samples)
            accuracy = float(np.mean(bootstraps))
            stderr = float(np.std(bootstraps))

        result[str(k)] = {"accuracy": accuracy, "stderr": stderr}

    return result