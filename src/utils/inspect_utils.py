from inspect_ai.log import EvalLog
from typing import Dict, Any

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

def compute_bootstrap_over_epochs(log: EvalLog) -> Dict[str, Any]:
    """Compute bootstrap over epochs."""
    return log.results.bootstrap