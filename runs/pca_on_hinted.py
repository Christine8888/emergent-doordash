from __future__ import annotations

import argparse

import numpy as np

from src.hinted_accuracy import (
    EXPECTED_FRACTIONS,
    LUKE_SUPPORTED_FRACTIONERS,
    discover_fractioners,
    discover_models_for_benchmark,
    load_results_with_ci_for_combo,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PCA on the same hinted-accuracy results used by the plot interface."
    )
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--hint-type", type=str, required=True)
    parser.add_argument(
        "--fractioner",
        type=str,
        default=None,
        help="Optional specific fractioner. If omitted, use the fractioners discovered per model.",
    )
    parser.add_argument(
        "--hint-fractions",
        type=float,
        nargs="+",
        default=None,
        help="Optional hint fractions to include, e.g. --hint-fractions 0.0 0.5 1.0",
    )
    return parser.parse_args()


def _feature_sort_key(name: str) -> tuple[str, float]:
    fractioner, hint_fraction = name.split("@", 1)
    return fractioner, float(hint_fraction)


def _discover_fractioners_for_pca(
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


def _load_feature_rows(
    *,
    benchmark: str,
    hint_type: str,
    fractioner: str | None,
    hint_fractions: list[float],
) -> dict[str, dict[str, float]]:
    rows_by_model: dict[str, dict[str, float]] = {}
    fractioners = _discover_fractioners_for_pca(
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
                for hint_fraction in hint_fractions
                if float(hint_fraction) not in fraction_map
            ]
            if missing:
                continue

            feature_row = rows_by_model.setdefault(str(model), {})
            for hint_fraction in hint_fractions:
                feature_name = f"{current_fractioner}@{hint_fraction:.1f}"
                feature_row[feature_name] = float(
                    fraction_map[float(hint_fraction)]["accuracy"]
                )

    return rows_by_model


def _run_pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std = np.where(std <= 0, 1.0, std)
    z = (matrix - mean) / std

    _, singular_values, vt = np.linalg.svd(z, full_matrices=False)
    scores = z @ vt.T
    explained_variance = (singular_values**2) / max(z.shape[0] - 1, 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()

    # Keep PC1 increasing with overall accuracy so the printed ordering is intuitive.
    if scores.shape[1] > 0:
        pc1 = scores[:, 0]
        row_means = matrix.mean(axis=1)
        correlation = float(np.corrcoef(pc1, row_means)[0, 1])
        if not np.isnan(correlation) and correlation < 0:
            vt = -vt
            scores = -scores

    return explained_variance_ratio, scores, vt


def _print_component_loadings(
    *,
    component_idx: int,
    shared_features: list[str],
    components: np.ndarray,
) -> None:
    if component_idx >= components.shape[0]:
        return

    print(f"PC{component_idx + 1} loadings")
    for feature_name, weight in sorted(
        zip(shared_features, components[component_idx], strict=True),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    ):
        print(f"{feature_name}: {float(weight):+.4f}")
    print("")


def _print_feature_summary(
    *,
    shared_features: list[str],
    matrix: np.ndarray,
    components: np.ndarray,
) -> None:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    component_vectors = [
        components[idx] if components.shape[0] > idx else np.zeros(len(shared_features), dtype=float)
        for idx in range(4)
    ]

    print("Feature summary")
    print(
        "feature  mean_acc  std_acc  "
        "pc1_loading  pc2_loading  pc3_loading  pc4_loading  "
        "pc1_delta_acc  pc2_delta_acc  pc3_delta_acc  pc4_delta_acc"
    )
    print(
        "-------  --------  -------  "
        "-----------  -----------  -----------  -----------  "
        "-------------  -------------  -------------  -------------"
    )
    for idx, feature_name in enumerate(shared_features):
        deltas = [float(vector[idx]) * float(stds[idx]) for vector in component_vectors]
        print(
            f"{feature_name}  "
            f"{float(means[idx]):.4f}  "
            f"{float(stds[idx]):.4f}  "
            f"{float(component_vectors[0][idx]):+.4f}  "
            f"{float(component_vectors[1][idx]):+.4f}  "
            f"{float(component_vectors[2][idx]):+.4f}  "
            f"{float(component_vectors[3][idx]):+.4f}  "
            f"{deltas[0]:+.4f}  "
            f"{deltas[1]:+.4f}  "
            f"{deltas[2]:+.4f}  "
            f"{deltas[3]:+.4f}"
        )
    print("")


def _print_ranked_component_scores(
    *,
    models: list[str],
    scores: np.ndarray,
    component_idx: int,
) -> None:
    if component_idx >= scores.shape[1]:
        return

    print(f"PC{component_idx + 1} ranking")
    for rank, (model_idx, model) in enumerate(
        sorted(
            enumerate(models),
            key=lambda item: float(scores[item[0], component_idx]),
            reverse=True,
        ),
        start=1,
    ):
        print(f"{rank:>2}. {model}: {float(scores[model_idx, component_idx]):+.4f}")
    print("")


def main() -> None:
    args = _parse_args()
    hint_fractions = (
        sorted({float(fraction) for fraction in args.hint_fractions})
        if args.hint_fractions is not None
        else list(EXPECTED_FRACTIONS)
    )
    invalid_fractions = [
        fraction for fraction in hint_fractions if fraction not in EXPECTED_FRACTIONS
    ]
    if invalid_fractions:
        raise ValueError(
            f"Unsupported hint fractions: {invalid_fractions}. "
            f"Expected subset of {EXPECTED_FRACTIONS}."
        )

    rows_by_model = _load_feature_rows(
        benchmark=args.benchmark,
        hint_type=args.hint_type,
        fractioner=args.fractioner,
        hint_fractions=hint_fractions,
    )
    if not rows_by_model:
        raise ValueError("No usable hinted-accuracy rows found.")

    shared_features = sorted(
        set.intersection(*(set(row.keys()) for row in rows_by_model.values())),
        key=_feature_sort_key,
    )
    if not shared_features:
        raise ValueError("No shared hint-accuracy features found across models.")
    shared_fractioners = sorted({feature_name.split("@", 1)[0] for feature_name in shared_features})

    models = sorted(rows_by_model)
    matrix = np.asarray(
        [
            [rows_by_model[model][feature_name] for feature_name in shared_features]
            for model in models
        ],
        dtype=float,
    )

    explained_variance_ratio, scores, components = _run_pca(matrix)

    print("PCA summary")
    print(f"benchmark: {args.benchmark}")
    print(f"hint_type: {args.hint_type}")
    print(f"fractioner: {args.fractioner or 'all_shared_fractioners'}")
    print(f"hint_fractions: {', '.join(f'{fraction:.1f}' for fraction in hint_fractions)}")
    print(f"n_models: {len(models)}")
    print(f"n_features: {len(shared_features)}")
    print(f"shared_fractioners: {', '.join(shared_fractioners)}")
    print("")

    print("Explained variance ratio")
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    for idx, value in enumerate(explained_variance_ratio[:5], start=1):
        print(
            f"PC{idx}: {float(value):.4f} "
            f"(cumulative={float(cumulative_explained_variance[idx - 1]):.4f})"
        )
    print("")

    _print_component_loadings(
        component_idx=0,
        shared_features=shared_features,
        components=components,
    )
    _print_component_loadings(
        component_idx=1,
        shared_features=shared_features,
        components=components,
    )
    _print_component_loadings(
        component_idx=2,
        shared_features=shared_features,
        components=components,
    )
    _print_component_loadings(
        component_idx=3,
        shared_features=shared_features,
        components=components,
    )
    _print_feature_summary(
        shared_features=shared_features,
        matrix=matrix,
        components=components,
    )
    _print_ranked_component_scores(
        models=models,
        scores=scores,
        component_idx=0,
    )
    _print_ranked_component_scores(
        models=models,
        scores=scores,
        component_idx=1,
    )
    _print_ranked_component_scores(
        models=models,
        scores=scores,
        component_idx=2,
    )
    _print_ranked_component_scores(
        models=models,
        scores=scores,
        component_idx=3,
    )

    print("Model scores")
    for idx, model in sorted(
        enumerate(models),
        key=lambda item: float(scores[item[0], 0]),
    ):
        if scores.shape[1] > 1:
            print(
                f"{model}: pc1={float(scores[idx, 0]):+.4f} "
                f"pc2={float(scores[idx, 1]):+.4f}"
            )
        else:
            print(f"{model}: pc1={float(scores[idx, 0]):+.4f}")


if __name__ == "__main__":
    # python -m runs.pca_on_hinted --benchmark aime2025_2026 --hint-type answer_not_revealed --fractioner mask_word --hint-fractions 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    main()
