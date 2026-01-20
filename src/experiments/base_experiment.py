"""Base experiment class for all experiments."""

import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from inspect_ai import eval
from inspect_ai.dataset import Sample
from utils.eval_utils import get_valid_problem_ids
from utils.inspect_utils import extract_scores_from_log, compute_bootstrap_over_epochs, compute_pass_at_k
from utils.setup import setup_logging
from experiments.runner import setup_vllm_env
import json

logger = setup_logging()


class Experiment(ABC):
    """Base class for experiments.

    Subclasses must define:
    - name: Experiment name (e.g., "cot_intext")
    - eval_name: Eval dataset name (e.g., "gpqa")
    - data_path: Path to hint data JSONL file
    - build_task(): Method to construct Inspect task

    Example:
        class MyExperiment(Experiment):
            name = "my_exp"
            eval_name = "gpqa"
            data_path = "data/hints.jsonl"

            def build_task(self, hint_fraction, sample_ids):
                # Build and return Inspect task
                return my_task(sample_ids=sample_ids, solver=my_solver)
    """

    # Subclasses must define these
    name: str = NotImplemented
    eval_name: str = NotImplemented
    data_path: str = NotImplemented

    def __init__(
        self,
        model_name: str,
        vllm_port: int,
        timeout: int = 600,
        max_connections: int = 32,
    ):
        """Initialize experiment.

        Args:
            model_name: Name of model being evaluated
            vllm_port: Port where vLLM server is running
            timeout: Timeout for eval tasks
            max_connections: Max concurrent connections
        """
        self.model_name = model_name
        self.vllm_port = vllm_port
        self.timeout = timeout
        self.max_connections = max_connections

        setup_vllm_env(vllm_port)

    @abstractmethod
    def build_task(self, hint_fraction: float, sample_ids: set[str]):
        """Build the Inspect task for this experiment.

        Args:
            hint_fraction: Fraction of hint to provide
            sample_ids: Set of sample IDs to evaluate on

        Returns:
            Inspect Task object
        """
        pass

    @classmethod
    def get_output_filename(
        cls,
        results_dir: str,
        model_name: str,
        fewshot: int,
        hint_fraction: float,
    ) -> str:
        """Get output filename for this configuration.

        Args:
            results_dir: Results directory
            model_name: Model name
            fewshot: Number of fewshot examples
            hint_fraction: Hint fraction

        Returns:
            Full path to output file
        """
        output_dir = Path(results_dir) / cls.eval_name / cls.name / f"{fewshot}shot" / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{cls.eval_name}_{cls.name}_{fewshot}shot_{hint_fraction}.json"
        return str(output_dir / filename)

    def run(
        self,
        hint_fraction: float,
        fewshot: int,
        epochs: int,
        results_dir: str,
        limit: Optional[int] = None,
    ) -> dict:
        """Run the experiment.

        Args:
            hint_fraction: Fraction of hint to provide
            fewshot: Number of fewshot examples
            epochs: Number of epochs
            results_dir: Directory to save results
            limit: Optional limit on number of samples

        Returns:
            Dictionary with results and metadata
        """
        # Get output filename
        output_file = self.get_output_filename(
            results_dir=results_dir,
            model_name=self.model_name,
            fewshot=fewshot,
            hint_fraction=hint_fraction,
        )

        # Check if output already exists
        if os.path.exists(output_file):
            logger.info(f"Output already exists: {output_file}")
            return {"filename": output_file, "status": "skipped"}

        valid_samples = get_valid_problem_ids([self.data_path])
        if valid_samples is None:
            raise ValueError(f"Failed to load sample IDs from {self.data_path}")

        sample_ids = set(valid_samples.keys())
        logger.info(f"Running {self.name} on {len(sample_ids)} samples")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Fewshot: {fewshot}")
        logger.info(f"  Hint fraction: {hint_fraction}")
        logger.info(f"  Epochs: {epochs}")

        # Build task
        task = self.build_task(
            hint_fraction=hint_fraction,
            sample_ids=sample_ids if not limit else list(sample_ids)[:limit]
        )

        # Run evaluation - Inspect logs go in same dir as results JSON
        output_dir = Path(output_file).parent

        eval_log = eval(
            task,
            model=f"vllm/{self.model_name}",
            log_dir=str(output_dir),
            epochs=epochs,
            limit=limit,
            max_connections=self.max_connections,
            display="plain",
            fail_on_error=False,
            retry_on_error=10,
            metadata={
                "timeout": self.timeout,
                "hint_fraction": hint_fraction,
                "fewshot": fewshot,
                "data_path": self.data_path,
                "solver_name": self.name,
            }
        )

        # Extract results
        results = extract_scores_from_log(eval_log[0])

        # Compute bootstrap metrics if multiple epochs
        if epochs > 1:
            scorer_name = f"{self.eval_name}_scorer"
            bootstrap_metric = {'scorer': scorer_name, 'metric': 'accuracy'}
            results["manual_bootstrap"] = compute_bootstrap_over_epochs(eval_log[0], bootstrap_metric)
            results["pass_at_k"] = compute_pass_at_k(eval_log[0], bootstrap_metric)

        # Save results
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_file}")
        return {
            "filename": output_file,
            "status": "completed",
            "results": results
        }
