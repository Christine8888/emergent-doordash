from inspect_ai.log import EvalLog
from typing import Dict, Any, List
import numpy as np
import random

def extract_scores_from_log(log: EvalLog) -> Dict[str, Any]:
    """Extract scores and metrics from the evaluation log.
    
    Args:
        log: The evaluation log from Inspect
        
    Returns:
        Dictionary containing extracted results
    """
    results = {
        "model": log.eval.model,
        "total_samples": log.results.total_samples,
        "completed_samples": log.results.completed_samples
    }
    
    for score in log.results.scores:
        score_dict = {}
        for metric_name, metric_value in score.metrics.items():
            score_dict[metric_name] = metric_value.value
        score_dict["scorer"] = score.scorer
        results[score.name] = score_dict
    
    return results

def group_by_sample_epoch(log: EvalLog, scorer: str) -> Dict[str, List[EvalLog]]:
    """Group EvalLog by sample and epoch."""
    result = {}
    for sample in log.samples:
        if sample.id not in result:
            result[sample.id] = []
        score = 1 if sample.scores[scorer].value == 'C' else 0
        result[sample.id].append(score)
    
    return result

def compute_bootstrap_over_epochs(log: EvalLog, scorer: str, n_bootstrap: int = 1000) -> Dict[str, float]:
    """Compute bootstrap over epochs."""

    grouped = group_by_sample_epoch(log, scorer)
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

def compute_pass_at_k(log: EvalLog, scorer: str, n_bootstrap: int = 1000) -> Dict[str, Dict[str, float]]:
    """Compute pass@k for k=1 to k=n_epochs.

    For each k, randomly select k answers per question and check if at least 1 is correct.
    Uses bootstrap to compute stderr, except for k=epochs where stderr=0.

    Args:
        log: The evaluation log from Inspect
        scorer: Name of the scorer to use
        n_bootstrap: Number of bootstrap samples (default 1000)

    Returns:
        Dictionary mapping k (as string) to accuracy and stderr
    """
    grouped = group_by_sample_epoch(log, scorer)
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