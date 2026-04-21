from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.hinted_accuracy import (
    EXPECTED_FRACTIONS,
    LUKE_SUPPORTED_FRACTIONERS,
    discover_fractioners,
    discover_models_for_benchmark,
    load_results_with_ci_for_combo,
)


@dataclass
class PCAResult:
    model_names: list[str]
    feature_names: list[str]
    matrix: np.ndarray
    components: np.ndarray
    scores: np.ndarray
    explained_variance_ratio: np.ndarray
    feature_means: np.ndarray
    feature_stds: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def _feature_sort_key(name: str) -> tuple[str, float]:
    if "@" not in name:
        return name, 0.0
    feature_group, feature_value = name.split("@", 1)
    return feature_group, float(feature_value)


def _orient_components(
    *,
    matrix: np.ndarray,
    components: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if scores.shape[1] == 0:
        return components, scores
    pc1 = scores[:, 0]
    row_means = matrix.mean(axis=1)
    correlation = float(np.corrcoef(pc1, row_means)[0, 1])
    if np.isnan(correlation):
        correlation = 1.0
    if correlation < 0:
        return -components, -scores
    return components, scores


def run_pca(
    *,
    model_names: list[str],
    feature_names: list[str],
    matrix: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> PCAResult:
    matrix_array = np.asarray(matrix, dtype=float)
    if matrix_array.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={matrix_array.shape}")
    if matrix_array.shape[0] != len(model_names):
        raise ValueError(
            "Number of model names does not match matrix rows: "
            f"{len(model_names)} vs {matrix_array.shape[0]}"
        )
    if matrix_array.shape[1] != len(feature_names):
        raise ValueError(
            "Number of feature names does not match matrix columns: "
            f"{len(feature_names)} vs {matrix_array.shape[1]}"
        )
    if matrix_array.size == 0:
        raise ValueError("Cannot run PCA on an empty matrix.")

    feature_means = matrix_array.mean(axis=0)
    feature_stds = matrix_array.std(axis=0)
    safe_stds = np.where(feature_stds <= 0, 1.0, feature_stds)
    z = (matrix_array - feature_means[None, :]) / safe_stds[None, :]

    _, singular_values, vt = np.linalg.svd(z, full_matrices=False)
    scores = z @ vt.T
    components, scores = _orient_components(
        matrix=matrix_array,
        components=np.asarray(vt, dtype=float),
        scores=np.asarray(scores, dtype=float),
    )

    explained_variance = (singular_values**2) / max(z.shape[0] - 1, 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()
    return PCAResult(
        model_names=list(model_names),
        feature_names=list(feature_names),
        matrix=matrix_array,
        components=np.asarray(components, dtype=float),
        scores=np.asarray(scores, dtype=float),
        explained_variance_ratio=np.asarray(explained_variance_ratio, dtype=float),
        feature_means=np.asarray(feature_means, dtype=float),
        feature_stds=np.asarray(feature_stds, dtype=float),
        metadata={} if metadata is None else dict(metadata),
    )


def build_component_score_map(
    result: PCAResult,
    *,
    component_idx: int,
) -> dict[str, float]:
    if component_idx < 0 or component_idx >= result.scores.shape[1]:
        raise IndexError(
            f"component_idx={component_idx} out of range for scores shape {result.scores.shape}"
        )
    return {
        model_name: float(result.scores[idx, component_idx])
        for idx, model_name in enumerate(result.model_names)
    }


def format_component_equation(
    result: PCAResult,
    *,
    component_idx: int,
    precision: int = 3,
    feature_formatter: Callable[[str], str] | None = None,
) -> str:
    if component_idx < 0 or component_idx >= result.components.shape[0]:
        raise IndexError(
            f"component_idx={component_idx} out of range for components shape {result.components.shape}"
        )
    formatter = feature_formatter or (lambda feature_name: f"z({feature_name})")
    terms = [
        f"{float(weight):+.{precision}f}·{formatter(feature_name)}"
        for feature_name, weight in zip(
            result.feature_names,
            result.components[component_idx],
            strict=True,
        )
    ]
    return f"PC{component_idx + 1} = " + " ".join(terms)


def _print_component_loadings(
    *,
    component_idx: int,
    result: PCAResult,
) -> None:
    if component_idx >= result.components.shape[0]:
        return

    print(f"PC{component_idx + 1} loadings")
    for feature_name, weight in sorted(
        zip(result.feature_names, result.components[component_idx], strict=True),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    ):
        print(f"{feature_name}: {float(weight):+.4f}")
    print("")


def _print_feature_summary(
    *,
    result: PCAResult,
    max_components: int,
) -> None:
    means = result.feature_means
    stds = result.feature_stds
    component_vectors = [
        result.components[idx]
        if result.components.shape[0] > idx
        else np.zeros(len(result.feature_names), dtype=float)
        for idx in range(max_components)
    ]

    loading_headers = "  ".join(
        f"pc{component_idx + 1}_loading" for component_idx in range(max_components)
    )
    delta_headers = "  ".join(
        f"pc{component_idx + 1}_delta_acc" for component_idx in range(max_components)
    )
    print("Feature summary")
    print(f"feature  mean_acc  std_acc  {loading_headers}  {delta_headers}")
    print(
        "-------  --------  -------  "
        + "  ".join("-----------" for _ in range(max_components))
        + "  "
        + "  ".join("-------------" for _ in range(max_components))
    )

    for idx, feature_name in enumerate(result.feature_names):
        deltas = [float(vector[idx]) * float(stds[idx]) for vector in component_vectors]
        loading_text = "  ".join(f"{float(vector[idx]):+.4f}" for vector in component_vectors)
        delta_text = "  ".join(f"{float(delta):+.4f}" for delta in deltas)
        print(
            f"{feature_name}  "
            f"{float(means[idx]):.4f}  "
            f"{float(stds[idx]):.4f}  "
            f"{loading_text}  "
            f"{delta_text}"
        )
    print("")


def _print_ranked_component_scores(
    *,
    result: PCAResult,
    component_idx: int,
) -> None:
    if component_idx >= result.scores.shape[1]:
        return

    print(f"PC{component_idx + 1} ranking")
    for rank, (model_idx, model) in enumerate(
        sorted(
            enumerate(result.model_names),
            key=lambda item: float(result.scores[item[0], component_idx]),
            reverse=True,
        ),
        start=1,
    ):
        print(f"{rank:>2}. {model}: {float(result.scores[model_idx, component_idx]):+.4f}")
    print("")


def print_pca_report(
    *,
    result: PCAResult,
    summary_lines: list[str] | None = None,
    max_components: int = 4,
) -> None:
    if summary_lines:
        print("PCA summary")
        for line in summary_lines:
            print(line)
        print("")

    print("Explained variance ratio")
    cumulative_explained_variance = np.cumsum(result.explained_variance_ratio)
    for idx, value in enumerate(result.explained_variance_ratio[: max_components + 1], start=1):
        print(
            f"PC{idx}: {float(value):.4f} "
            f"(cumulative={float(cumulative_explained_variance[idx - 1]):.4f})"
        )
    print("")

    for component_idx in range(max_components):
        _print_component_loadings(
            component_idx=component_idx,
            result=result,
        )

    _print_feature_summary(
        result=result,
        max_components=max_components,
    )

    for component_idx in range(max_components):
        _print_ranked_component_scores(
            result=result,
            component_idx=component_idx,
        )

    print("Model scores")
    for idx, model in sorted(
        enumerate(result.model_names),
        key=lambda item: float(result.scores[item[0], 0]),
    ):
        component_values = " ".join(
            f"pc{component_idx + 1}={float(result.scores[idx, component_idx]):+.4f}"
            for component_idx in range(min(max_components, result.scores.shape[1]))
        )
        print(f"{model}: {component_values}")


def _discover_fractioners_for_hinted_pca(
    *,
    benchmark: str,
    hint_type: str,
    fractioner: str | None,
) -> list[str]:
    if fractioner is not None:
        return [fractioner]

    fractioners: set[str] = set()
    for model in discover_models_for_benchmark(benchmark):
        fractioners.update(
            discover_fractioners(
                benchmark=benchmark,
                model=model,
                hint_type=hint_type,
            )
        )

    if benchmark == "aime2025_2026" and hint_type == "answer_not_revealed":
        fractioners.update(LUKE_SUPPORTED_FRACTIONERS)

    return sorted(fractioners)


def build_hinted_pca_result(
    *,
    benchmark: str,
    hint_type: str,
    fractioner: str | None,
    hint_fractions: list[float] | None = None,
) -> PCAResult:
    selected_hint_fractions = (
        list(EXPECTED_FRACTIONS)
        if hint_fractions is None
        else [float(value) for value in hint_fractions]
    )
    rows_by_model: dict[str, dict[str, float]] = {}
    fractioners = _discover_fractioners_for_hinted_pca(
        benchmark=benchmark,
        hint_type=hint_type,
        fractioner=fractioner,
    )

    for current_fractioner in fractioners:
        payload = load_results_with_ci_for_combo(
            benchmark=benchmark,
            hint_type=hint_type,
            fractioner=current_fractioner,
        )
        for model, fraction_map in payload.items():
            missing = [
                hint_fraction
                for hint_fraction in selected_hint_fractions
                if float(hint_fraction) not in fraction_map
            ]
            if missing:
                continue

            feature_row = rows_by_model.setdefault(str(model), {})
            for hint_fraction in selected_hint_fractions:
                feature_name = f"{current_fractioner}@{hint_fraction:.1f}"
                feature_row[feature_name] = float(
                    fraction_map[float(hint_fraction)]["accuracy"]
                )

    if not rows_by_model:
        raise ValueError("No usable hinted-accuracy rows found.")

    shared_features = sorted(
        set.intersection(*(set(row.keys()) for row in rows_by_model.values())),
        key=_feature_sort_key,
    )
    if not shared_features:
        raise ValueError("No shared hint-accuracy features found across models.")

    model_names = sorted(rows_by_model)
    matrix = np.asarray(
        [
            [rows_by_model[model_name][feature_name] for feature_name in shared_features]
            for model_name in model_names
        ],
        dtype=float,
    )
    shared_fractioners = sorted(
        {feature_name.split("@", 1)[0] for feature_name in shared_features}
    )
    return run_pca(
        model_names=model_names,
        feature_names=shared_features,
        matrix=matrix,
        metadata={
            "benchmark": benchmark,
            "hint_type": hint_type,
            "requested_fractioner": fractioner,
            "shared_fractioners": shared_fractioners,
            "hint_fractions": list(selected_hint_fractions),
        },
    )


def build_baseline_benchmark_pca_result(
    *,
    scores_df: pd.DataFrame,
    benchmark_order: list[str],
    canonicalize_model_name: Callable[[str], str] | None = None,
) -> PCAResult:
    df = scores_df[scores_df["benchmark"].isin(benchmark_order)].copy()
    if canonicalize_model_name is not None:
        df["model"] = df["model"].map(lambda value: canonicalize_model_name(str(value)))
    else:
        df["model"] = df["model"].map(lambda value: str(value))

    pivot = df.pivot(index="model", columns="benchmark", values="score")
    pivot = pivot.reindex(columns=benchmark_order)
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.empty:
        raise ValueError("No models have complete benchmark coverage for baseline PCA.")

    model_names = [str(model_name) for model_name in pivot.index.tolist()]
    feature_names = [str(benchmark_name) for benchmark_name in benchmark_order]
    matrix = pivot.to_numpy(dtype=float)
    return run_pca(
        model_names=model_names,
        feature_names=feature_names,
        matrix=matrix,
        metadata={
            "benchmarks": list(benchmark_order),
        },
    )
