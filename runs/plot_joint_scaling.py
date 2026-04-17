from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runs.fit_eci import EVAL_TO_ECI, load_baseline_scores
from src.hinted_accuracy import load_results_with_ci_for_combo


PLOTS_ROOT = Path("plots/joint_scaling_plots")
PC_BENCHMARK_ORDER = [EVAL_TO_ECI[eval_name] for eval_name in EVAL_TO_ECI]
JOINT_LOWER_ASYMPTOTE = 0.0
OLD_GPQA_ROOT = Path(
    "/nlp/scr/suzeva/projects/emergent-doordash/christine_experiments/20251113/results/"
    "gpqa/solution_intext_masked/0shot"
)
MODELS_TO_USE: list[str] | None = [
    "google/gemma-3-27b-it",
    "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "google/gemma-3-12b-it",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot accuracy vs capability, one curve per hint fraction.")
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--hint-type", type=str, required=True)
    parser.add_argument("--fractioner", type=str, required=True)
    parser.add_argument("--eci-file", type=str, required=True)
    parser.add_argument("--num-holdout-models", type=int, default=0)
    parser.add_argument(
        "--include-cross",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include the capability-by-hint interaction term in the joint fit.",
    )
    return parser.parse_args()


def _load_eci_map(path: Path) -> dict[str, float]:
    import csv

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "model" not in reader.fieldnames or "eci_our_fit" not in reader.fieldnames:
            raise ValueError(f"Expected columns 'model' and 'eci_our_fit' in {path}")

        out: dict[str, float] = {}
        for row in reader:
            model = str(row.get("model", "")).strip()
            eci_raw = row.get("eci_our_fit")
            if not model or eci_raw in (None, ""):
                continue
            out[_canonicalize_old_gpqa_model_name(model)] = float(eci_raw)
    return out


def _eci_benchmark_label(path: Path) -> str:
    stem = path.stem
    prefix = "eci_model_capabilities__simple__"
    if not stem.startswith(prefix):
        return "unknown"
    encoded = stem[len(prefix) :]
    if not encoded:
        return "unknown"
    benchmark_names: list[str] = []
    for part in encoded.split("--"):
        name = part.split("__", 1)[0].strip()
        if name and name not in benchmark_names:
            benchmark_names.append(name)
    return ", ".join(benchmark_names) if benchmark_names else "unknown"


def _format_title_text(lines: list[str], *, width: int = 72) -> str:
    wrapped_lines: list[str] = []
    for line in lines:
        stripped = str(line).strip()
        if not stripped:
            continue
        wrapped = textwrap.wrap(
            stripped,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped_lines.extend(wrapped if wrapped else [stripped])
    return "\n".join(wrapped_lines)


def _is_old_gpqa_combo(*, benchmark: str, hint_type: str, fractioner: str) -> bool:
    return (
        benchmark == "gpqa"
        and hint_type == "answer_not_revealed"
        and fractioner == "mask_word"
    )


def _validate_old_gpqa_combo(*, benchmark: str, hint_type: str, fractioner: str) -> None:
    if benchmark != "gpqa":
        return
    if _is_old_gpqa_combo(
        benchmark=benchmark,
        hint_type=hint_type,
        fractioner=fractioner,
    ):
        return
    raise ValueError(
        "GPQA is only supported via the old results directory for "
        "hint_type=answer_not_revealed and fractioner=mask_word."
    )


def _parse_old_gpqa_fraction_from_filename(name: str) -> float:
    match = re.match(r"^gpqa_solution_intext_masked_0shot_(.+)\.json$", name)
    if not match:
        raise ValueError(f"Unexpected GPQA filename: {name}")
    return float(match.group(1))


def _canonicalize_old_gpqa_model_name(model: str) -> str:
    if "/" in model:
        return model
    if model.startswith("Qwen"):
        return f"Qwen/{model}"
    if model.startswith("Llama-"):
        return f"meta-llama/{model}"
    if model.startswith("gemma-"):
        return f"google/{model}"
    return model


def _discover_old_gpqa_models() -> list[str]:
    if not OLD_GPQA_ROOT.exists():
        raise FileNotFoundError(f"Missing OLD_GPQA_ROOT: {OLD_GPQA_ROOT}")
    return sorted(path.name for path in OLD_GPQA_ROOT.iterdir() if path.is_dir())


def _load_old_gpqa_results_with_ci() -> dict[str, dict[float, dict[str, float]]]:
    out: dict[str, dict[float, dict[str, float]]] = {}
    for old_model in _discover_old_gpqa_models():
        model = _canonicalize_old_gpqa_model_name(old_model)
        model_dir = OLD_GPQA_ROOT / old_model
        fraction_map: dict[float, dict[str, float]] = {}
        for path in sorted(model_dir.glob("gpqa_solution_intext_masked_0shot_*.json")):
            try:
                hint_fraction = _parse_old_gpqa_fraction_from_filename(path.name)
            except ValueError:
                continue

            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            metric = payload.get("manual_bootstrap")
            if not isinstance(metric, dict):
                print(
                    f"[plot_accuracy_vs_eci_by_hint][WARN] missing manual_bootstrap "
                    f"model={old_model} canonical_model={model} path={path}"
                )
                continue

            accuracy = metric.get("accuracy")
            stderr = metric.get("stderr")
            if not isinstance(accuracy, (int, float)) or not isinstance(stderr, (int, float)):
                print(
                    f"[plot_accuracy_vs_eci_by_hint][WARN] invalid manual_bootstrap metric "
                    f"model={old_model} canonical_model={model} path={path}"
                )
                continue

            ci_low = max(0.0, float(accuracy) - 1.96 * float(stderr))
            ci_high = min(1.0, float(accuracy) + 1.96 * float(stderr))
            fraction_map[float(hint_fraction)] = {
                "accuracy": float(accuracy),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }

        if fraction_map:
            out[model] = dict(sorted(fraction_map.items()))
    return out


def _load_combo_results(
    *,
    benchmark: str,
    hint_type: str,
    fractioner: str,
) -> tuple[dict[str, dict[float, dict[str, float]]], list[str]]:
    _validate_old_gpqa_combo(
        benchmark=benchmark,
        hint_type=hint_type,
        fractioner=fractioner,
    )
    if _is_old_gpqa_combo(
        benchmark=benchmark,
        hint_type=hint_type,
        fractioner=fractioner,
    ):
        combo_results = _load_old_gpqa_results_with_ci()
        return combo_results, sorted(combo_results.keys())

    combo_results = load_results_with_ci_for_combo(
        benchmark=benchmark,
        hint_type=hint_type,
        fractioner=fractioner,
    )
    canonical_combo_results = {
        _canonicalize_old_gpqa_model_name(str(model)): stats
        for model, stats in combo_results.items()
    }
    return canonical_combo_results, sorted(canonical_combo_results.keys())


def _resolve_models_to_use(
    *,
    available_models: list[str],
    benchmark: str,
) -> list[str]:
    canonical_available_models = [
        _canonicalize_old_gpqa_model_name(str(model)) for model in available_models
    ]
    if MODELS_TO_USE is None:
        return canonical_available_models

    canonical_models_to_use = [
        _canonicalize_old_gpqa_model_name(str(model)) for model in MODELS_TO_USE
    ]
    missing_models = sorted(set(canonical_models_to_use) - set(canonical_available_models))
    if missing_models:
        raise ValueError(
            f"Configured MODELS_TO_USE missing for benchmark={benchmark}: {missing_models}. "
            f"Available models: {sorted(canonical_available_models)}"
        )
    return canonical_models_to_use


def _compute_pca_from_baselines(scores_df: Any) -> dict[str, Any]:
    df = scores_df[scores_df["benchmark"].isin(PC_BENCHMARK_ORDER)].copy()
    df["model"] = df["model"].map(lambda value: _canonicalize_old_gpqa_model_name(str(value)))
    pivot = df.pivot(index="model", columns="benchmark", values="score")
    pivot = pivot.reindex(columns=PC_BENCHMARK_ORDER)
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.empty:
        raise ValueError("No models have complete benchmark coverage for PC capability.")

    matrix = pivot.to_numpy(dtype=float)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std = np.where(std <= 0, 1.0, std)
    z = (matrix - mean) / std

    _, singular_values, vt = np.linalg.svd(z, full_matrices=False)
    pc1 = vt[0]
    scores = z @ pc1

    correlation = np.corrcoef(scores, z.mean(axis=1))[0, 1]
    if np.isnan(correlation):
        correlation = 1.0
    if correlation < 0:
        vt = -vt
        scores = -scores

    explained_variance = (singular_values**2) / max(z.shape[0] - 1, 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()
    pc1_terms = []
    for benchmark, weight in zip(PC_BENCHMARK_ORDER, pc1):
        sign = "+" if float(weight) >= 0 else "-"
        pc1_terms.append(f"{sign}{abs(float(weight)):.3f}·z({benchmark})")
    pc1_equation = "PC1 = " + " ".join(pc1_terms)

    return {
        "capability_map": {
        str(model): float(score)
        for model, score in zip(pivot.index.tolist(), scores)
        },
        "benchmarks": list(PC_BENCHMARK_ORDER),
        "components": np.asarray(vt, dtype=float),
        "explained_variance_ratio": np.asarray(explained_variance_ratio, dtype=float),
        "benchmark_label": ", ".join(eval_name for eval_name in EVAL_TO_ECI),
        "equation": pc1_equation,
    }


def _plot_pca_component_weights(
    *,
    components: np.ndarray,
    benchmarks: list[str],
    output_path: Path,
) -> None:
    n_components = min(5, components.shape[0])
    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(benchmarks)), 1.0 + 0.8 * n_components))
    image = ax.imshow(components[:n_components], aspect="auto", cmap="coolwarm")
    ax.set_xticks(np.arange(len(benchmarks)))
    ax.set_xticklabels(benchmarks, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_components))
    ax.set_yticklabels([f"PC-{i}" for i in range(1, n_components + 1)])
    ax.set_title("Principal component weights")
    fig.colorbar(image, ax=ax, shrink=0.9, label="weight")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_pca_explained_variance(
    *,
    explained_variance_ratio: np.ndarray,
    output_path: Path,
) -> None:
    n_components = min(10, explained_variance_ratio.shape[0])
    x = np.arange(1, n_components + 1)
    y = explained_variance_ratio[:n_components]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, y, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Explained variance by component")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _sigmoid_curve(x: np.ndarray, lower: float, slope: float, bias: float) -> np.ndarray:
    return lower + (1.0 - lower) * (1.0 / (1.0 + np.exp(-(slope * x + bias))))


def _fit_sigmoid(xs: list[float], ys: list[float]) -> tuple[np.ndarray, np.ndarray] | None:
    if len(xs) < 4:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.allclose(y, y[0]):
        return None

    try:
        from scipy.optimize import curve_fit

        lower0 = float(np.clip(np.min(y) - 0.02, 0.0, 0.95))
        y_mid = 0.5 * (float(np.min(y)) + float(np.max(y)))
        mid_idx = int(np.argmin(np.abs(y - y_mid)))
        x_mid = float(x[mid_idx])
        slope0 = 0.2
        bias0 = -slope0 * x_mid

        params, _ = curve_fit(
            _sigmoid_curve,
            x,
            y,
            p0=[lower0, slope0, bias0],
            bounds=([0.0, 1e-6, -200.0], [0.99, 10.0, 200.0]),
            maxfev=20000,
        )
    except Exception:
        return None

    x_fit = np.linspace(float(np.min(x)) - 2.0, float(np.max(x)) + 2.0, 200, dtype=float)
    y_fit = _sigmoid_curve(x_fit, *params)
    return x_fit, y_fit


def _print_accuracy_summary_table(
    rows: list[dict[str, Any]],
    hint_fractions: list[float],
    *,
    capability_label: str,
) -> None:
    model_to_capability = {
        str(row["model"]): float(row["capability"])
        for row in rows
    }
    score_map: dict[tuple[str, float], float] = {}
    for row in rows:
        score_map[(str(row["model"]), float(row["hint_fraction"]))] = float(row["accuracy"])

    models = sorted(model_to_capability.keys(), key=lambda model: model_to_capability[model])
    model_width = max(len("Model"), max(len(model) for model in models))
    capability_width = max(len(capability_label), 10)
    frac_headers = [f"h={hint_fraction:.1f}" for hint_fraction in hint_fractions]
    frac_width = max(7, max(len(header) for header in frac_headers))

    header = [
        "Model".ljust(model_width),
        capability_label.rjust(capability_width),
        *[header.rjust(frac_width) for header in frac_headers],
    ]
    separator = [
        "-" * model_width,
        "-" * capability_width,
        *["-" * frac_width for _ in frac_headers],
    ]

    print("\nAccuracy summary by model and hint fraction:")
    print("  " + " ".join(header))
    print("  " + " ".join(separator))
    for model in models:
        row = [
            model.ljust(model_width),
            f"{model_to_capability[model]:.3f}".rjust(capability_width),
        ]
        for hint_fraction in hint_fractions:
            value = score_map.get((model, float(hint_fraction)))
            row.append("--".rjust(frac_width) if value is None else f"{value:.4f}".rjust(frac_width))
        print("  " + " ".join(row))


def _add_model_name_axis(
    ax: plt.Axes,
    *,
    rows: list[dict[str, Any]],
    label: str,
) -> None:
    plotted_models = sorted(
        {
            (str(row["model"]), float(row["capability"]))
            for row in rows
        },
        key=lambda item: item[1],
    )
    top_ax = ax.secondary_xaxis("top")
    top_ax.set_xticks([capability for _, capability in plotted_models])
    top_ax.set_xticklabels([model for model, _ in plotted_models], rotation=60, ha="left", fontsize=8)
    top_ax.set_xlabel(label, fontsize=11)


def _plot_capability_view(
    *,
    rows: list[dict[str, Any]],
    benchmark: str,
    hint_type: str,
    fractioner: str,
    capability_method: str,
    capability_label: str,
    capability_benchmark_label: str,
    capability_equation: str | None,
    output_dir: Path,
) -> Path:
    hint_fractions = sorted({float(row["hint_fraction"]) for row in rows})
    _print_accuracy_summary_table(rows, hint_fractions, capability_label=capability_label)

    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.cm.viridis
    colors = {h: cmap(i / max(len(hint_fractions) - 1, 1)) for i, h in enumerate(hint_fractions)}

    for hint_fraction in hint_fractions:
        series_rows = sorted(
            [row for row in rows if float(row["hint_fraction"]) == hint_fraction],
            key=lambda row: float(row["capability"]),
        )
        xs = [float(row["capability"]) for row in series_rows]
        ys = [float(row["accuracy"]) for row in series_rows]
        color = colors[hint_fraction]

        ax.scatter(xs, ys, color=color, alpha=0.85, s=45, label=f"h={hint_fraction:.2f}")

        fit = _fit_sigmoid(xs, ys)
        if fit is not None:
            x_fit, y_fit = fit
            ax.plot(x_fit, y_fit, "-", color=color, alpha=0.7, linewidth=2)

    _add_model_name_axis(
        ax,
        rows=rows,
        label="Model",
    )
    ax.set_xlabel(capability_label, fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    title_lines = [
        f"Accuracy vs {capability_label} by Hint Fraction",
        f"benchmark={benchmark} hint_type={hint_type} fractioner={fractioner}",
        f"{capability_method}_benchmarks={capability_benchmark_label}",
    ]
    if capability_equation:
        title_lines.append(capability_equation)
    ax.set_title(_format_title_text(title_lines), fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    output_path = output_dir / f"accuracy_vs_{capability_method}_by_hint.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _format_hint_fraction_for_filename(hint_fraction: float) -> str:
    return f"{hint_fraction:.2f}".replace(".", "p")


def _plot_capability_view_per_hint_with_error_bars(
    *,
    rows: list[dict[str, Any]],
    benchmark: str,
    hint_type: str,
    fractioner: str,
    capability_method: str,
    capability_label: str,
    capability_benchmark_label: str,
    capability_equation: str | None,
    output_dir: Path,
 ) -> Path:
    hint_fractions = sorted({float(row["hint_fraction"]) for row in rows})
    n_panels = len(hint_fractions)
    ncols = min(3, max(1, n_panels))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    for idx, hint_fraction in enumerate(hint_fractions):
        ax = axes[idx // ncols][idx % ncols]
        series_rows = sorted(
            [row for row in rows if float(row["hint_fraction"]) == hint_fraction],
            key=lambda row: float(row["capability"]),
        )
        if not series_rows:
            continue

        xs = np.asarray([float(row["capability"]) for row in series_rows], dtype=float)
        ys = np.asarray([float(row["accuracy"]) for row in series_rows], dtype=float)
        yerr = np.asarray(
            [
                [
                    max(0.0, float(row["accuracy"]) - float(row["ci_low"])),
                    max(0.0, float(row["ci_high"]) - float(row["accuracy"])),
                ]
                for row in series_rows
            ],
            dtype=float,
        ).T

        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            fmt="o",
            color="#1f77b4",
            ecolor="#1f77b4",
            elinewidth=1.5,
            capsize=4,
            alpha=0.85,
            markersize=6,
        )

        fit = _fit_sigmoid(xs.tolist(), ys.tolist())
        if fit is not None:
            x_fit, y_fit = fit
            ax.plot(x_fit, y_fit, "-", color="#ff7f0e", alpha=0.8, linewidth=2)

        ax.set_xlabel(capability_label, fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"hint_fraction={hint_fraction:.2f}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    title_lines = [
        f"Accuracy vs {capability_label} with Error Bars by Hint Fraction",
        f"benchmark={benchmark} hint_type={hint_type} fractioner={fractioner}",
        f"{capability_method}_benchmarks={capability_benchmark_label}",
    ]
    if capability_equation:
        title_lines.append(capability_equation)
    fig.suptitle(_format_title_text(title_lines), fontsize=14)
    output_path = output_dir / f"accuracy_vs_{capability_method}_by_hint_subplots_with_error_bars.png"
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hint_identity(hint_fraction: float) -> float:
    return float(hint_fraction)


def _joint_sigmoid(z: np.ndarray | float, *, lower: float | None) -> np.ndarray | float:
    lower_value = 0.0 if lower is None else float(lower)
    sigmoid = 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))
    return lower_value + (1.0 - lower_value) * sigmoid


def _clip_accuracy_for_logit(y: np.ndarray, *, lower: float | None) -> np.ndarray:
    lower_value = 0.0 if lower is None else float(lower)
    scaled = (np.asarray(y, dtype=float) - lower_value) / max(1.0 - lower_value, 1e-8)
    return np.clip(scaled, 1e-4, 1.0 - 1e-4)


def _fit_linear_init(
    *,
    capability: np.ndarray,
    hint: np.ndarray,
    accuracy: np.ndarray,
    include_cross: bool,
    lower: float | None,
) -> np.ndarray:
    target = np.log(_clip_accuracy_for_logit(accuracy, lower=lower) / (1.0 - _clip_accuracy_for_logit(accuracy, lower=lower)))
    if include_cross:
        design = np.column_stack([capability, hint, capability * hint, np.ones_like(capability)])
    else:
        design = np.column_stack([capability, hint, np.ones_like(capability)])
    params, *_ = np.linalg.lstsq(design, target, rcond=None)
    return np.asarray(params, dtype=float)


def _fit_joint_sigmoid_model(
    *,
    df: pd.DataFrame,
    fit_models: set[str],
    include_cross: bool,
    lower: float | None,
) -> dict[str, Any]:
    from scipy.optimize import minimize

    train_df = df[df["model"].isin(fit_models)].copy()
    if train_df.empty:
        raise ValueError("No training rows available for the joint fit.")

    capability = train_df["eci"].to_numpy(dtype=float)
    hint = train_df["hint_fraction"].to_numpy(dtype=float)
    accuracy = train_df["accuracy"].to_numpy(dtype=float)
    init = _fit_linear_init(
        capability=capability,
        hint=hint,
        accuracy=accuracy,
        include_cross=include_cross,
        lower=lower,
    )

    def predict_from_params(params: np.ndarray, x_capability: np.ndarray, x_hint: np.ndarray) -> np.ndarray:
        if include_cross:
            alpha, beta, gamma, delta = params
            z = alpha * x_capability + beta * x_hint + gamma * x_capability * x_hint + delta
        else:
            alpha, beta, delta = params
            z = alpha * x_capability + beta * x_hint + delta
        return np.asarray(_joint_sigmoid(z, lower=lower), dtype=float)

    def objective(params: np.ndarray) -> float:
        predictions = predict_from_params(params, capability, hint)
        residuals = predictions - accuracy
        return float(np.mean(residuals * residuals))

    result = minimize(
        objective,
        init,
        method="L-BFGS-B",
        options={"maxiter": 2000},
    )
    params = np.asarray(result.x, dtype=float)

    def predict(single_capability: float, single_hint: float) -> float:
        prediction = predict_from_params(
            params,
            np.asarray([single_capability], dtype=float),
            np.asarray([single_hint], dtype=float),
        )[0]
        return float(prediction)

    return {
        "params": params,
        "include_cross": bool(include_cross),
        "lower": lower,
        "predict": predict,
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_fun": float(result.fun),
        "optimizer_nit": int(getattr(result, "nit", -1)),
    }


def _fit_1d_sigmoid(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    lower: float | None,
) -> dict[str, Any] | None:
    from scipy.optimize import curve_fit

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 3 or np.allclose(y, y[0]):
        return None

    target = np.log(_clip_accuracy_for_logit(y, lower=lower) / (1.0 - _clip_accuracy_for_logit(y, lower=lower)))
    slope0, bias0 = np.linalg.lstsq(
        np.column_stack([x, np.ones_like(x)]),
        target,
        rcond=None,
    )[0]
    if not np.isfinite(slope0):
        slope0 = 0.1
    if not np.isfinite(bias0):
        bias0 = -np.mean(x) * slope0

    lower_value = 0.0 if lower is None else float(lower)

    def fn(x_input: np.ndarray, slope: float, bias: float) -> np.ndarray:
        return lower_value + (1.0 - lower_value) * (1.0 / (1.0 + np.exp(-(slope * x_input + bias))))

    try:
        params, _ = curve_fit(
            fn,
            x,
            y,
            p0=[float(slope0), float(bias0)],
            bounds=([-20.0, -500.0], [20.0, 500.0]),
            maxfev=20000,
        )
    except Exception:
        return None

    slope, bias = [float(value) for value in params]
    if abs(slope) <= 1e-8:
        return None

    def predict(x_input: float) -> float:
        return float(fn(np.asarray([x_input], dtype=float), slope, bias)[0])

    return {
        "params": [slope, bias],
        "midpoint": float(-bias / slope),
        "predict": predict,
    }


def _fit_individual_sigmoids_by_hint(
    *,
    df: pd.DataFrame,
    fit_models: set[str] | None,
    lower: float | None,
) -> dict[float, dict[str, Any]]:
    results: dict[float, dict[str, Any]] = {}
    hint_fractions = sorted(df["hint_fraction"].unique().tolist())
    for hint_fraction in hint_fractions:
        hint_df = df[df["hint_fraction"] == hint_fraction]
        if fit_models is not None:
            hint_df = hint_df[hint_df["model"].isin(fit_models)]
        fit = _fit_1d_sigmoid(
            xs=hint_df["eci"].to_numpy(dtype=float),
            ys=hint_df["accuracy"].to_numpy(dtype=float),
            lower=lower,
        )
        if fit is not None:
            results[float(hint_fraction)] = fit
    return results


def _fit_individual_sigmoids_by_model(
    *,
    df: pd.DataFrame,
    fit_models: set[str] | None,
    lower: float | None,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    model_names = sorted(set(df["model"].tolist()) if fit_models is None else fit_models)
    for model in model_names:
        model_df = df[df["model"] == model]
        fit = _fit_1d_sigmoid(
            xs=model_df["hint_fraction"].to_numpy(dtype=float),
            ys=model_df["accuracy"].to_numpy(dtype=float),
            lower=lower,
        )
        if fit is not None:
            results[str(model)] = fit
    return results


def _compute_rms_joint(
    *,
    joint_result: dict[str, Any],
    df: pd.DataFrame,
    models: set[str] | None,
) -> float:
    eval_df = df if models is None else df[df["model"].isin(models)]
    if eval_df.empty:
        return float("nan")
    predictions = np.asarray(
        [
            joint_result["predict"](float(row["eci"]), float(row["hint_fraction"]))
            for _, row in eval_df.iterrows()
        ],
        dtype=float,
    )
    actual = eval_df["accuracy"].to_numpy(dtype=float)
    return float(np.sqrt(np.mean((actual - predictions) ** 2)))


def _compute_rms_individual_by_hint(
    *,
    individual_by_hint: dict[float, dict[str, Any]],
    df: pd.DataFrame,
    models: set[str] | None,
) -> float:
    eval_df = df if models is None else df[df["model"].isin(models)]
    if eval_df.empty:
        return float("nan")
    predictions: list[float] = []
    actual: list[float] = []
    for _, row in eval_df.iterrows():
        hint_fraction = float(row["hint_fraction"])
        fit = individual_by_hint.get(hint_fraction)
        if fit is None:
            continue
        predictions.append(float(fit["predict"](float(row["eci"]))))
        actual.append(float(row["accuracy"]))
    if not predictions:
        return float("nan")
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predictions)) ** 2)))


def _compute_midpoint_errors(
    *,
    joint_result: dict[str, Any],
    individual_fits: dict[float, dict[str, Any]],
    hint_fractions: list[float],
) -> dict[float, float]:
    params = np.asarray(joint_result["params"], dtype=float)
    errors: dict[float, float] = {}
    for hint_fraction in hint_fractions:
        fit = individual_fits.get(float(hint_fraction))
        if fit is None:
            continue
        if joint_result["include_cross"]:
            alpha, beta, gamma, delta = params
            denom = alpha + gamma * float(hint_fraction)
            if abs(denom) <= 1e-8:
                continue
            midpoint = (-beta * float(hint_fraction) - delta) / denom
        else:
            alpha, beta, delta = params
            if abs(alpha) <= 1e-8:
                continue
            midpoint = (-beta * float(hint_fraction) - delta) / alpha
        errors[float(hint_fraction)] = abs(float(midpoint) - float(fit["midpoint"]))
    return errors


def _format_joint_equation(joint_result: dict[str, Any]) -> str:
    params = np.asarray(joint_result["params"], dtype=float)
    lower = joint_result.get("lower")
    if joint_result["include_cross"]:
        alpha, beta, gamma, delta = params
        expr = f"σ({alpha:.3f}·C + {beta:.3f}·h + {gamma:.3f}·C·h + {delta:.2f})"
    else:
        alpha, beta, delta = params
        expr = f"σ({alpha:.3f}·C + {beta:.3f}·h + {delta:.2f})"
    if lower is not None:
        lower_value = float(lower)
        return f"{lower_value:.2f} + {1.0 - lower_value:.2f}·{expr}"
    return expr


def _build_joint_scaling_df(
    *,
    base_rows: list[dict[str, Any]],
    eci_map: dict[str, float],
    train_models: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in base_rows:
        model = str(row["model"])
        if model not in eci_map:
            continue
        rows.append(
            {
                "model": model,
                "hint_fraction": float(row["hint_fraction"]),
                "accuracy": float(row["accuracy"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "eci": float(eci_map[model]),
                "split": "train" if model in train_models else "test",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No rows available for ECI joint scaling after merging ECI values.")
    return df


def _plot_joint_accuracy_vs_eci_by_hint(
    *,
    df: pd.DataFrame,
    joint_result: dict[str, Any],
    label: str,
    joint_equation: str,
    output_dir: Path,
    filename_stem: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 7))
    hint_fractions = sorted(df["hint_fraction"].unique().tolist())
    colors = {h: plt.cm.viridis(i / max(len(hint_fractions) - 1, 1)) for i, h in enumerate(hint_fractions)}
    eci_range = np.linspace(float(df["eci"].min()) - 5.0, float(df["eci"].max()) + 5.0, 120)

    for hint_fraction in hint_fractions:
        hint_df = df[df["hint_fraction"] == hint_fraction].sort_values("eci")
        train_df = hint_df[hint_df["split"] == "train"]
        test_df = hint_df[hint_df["split"] == "test"]
        color = colors[float(hint_fraction)]

        ax.scatter(
            train_df["eci"],
            train_df["accuracy"],
            color=color,
            alpha=0.8,
            s=60,
            marker="o",
            label=f"h={float(hint_fraction):.2f}",
        )
        if not test_df.empty:
            ax.scatter(
                test_df["eci"],
                test_df["accuracy"],
                color=color,
                alpha=0.8,
                s=60,
                marker="s",
                edgecolors="black",
            )

        y_fit = [joint_result["predict"](float(eci), float(hint_fraction)) for eci in eci_range]
        ax.plot(eci_range, y_fit, "-", color=color, alpha=0.5, linewidth=2)

    ax.set_xlabel("eci", fontsize=12)
    ax.set_ylabel("accuracy", fontsize=12)
    ax.set_title(_format_title_text([label, joint_equation]), fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = output_dir / f"{filename_stem}.png"
    _save_figure(fig, output_path)
    return output_path


def _plot_joint_individual_fits_by_hint(
    *,
    df: pd.DataFrame,
    joint_result: dict[str, Any],
    individual_by_hint_all: dict[float, dict[str, Any]],
    individual_by_hint_train: dict[float, dict[str, Any]],
    label: str,
    joint_equation: str,
    output_dir: Path,
    filename_stem: str,
) -> Path:
    hint_fractions = sorted(df["hint_fraction"].unique().tolist())
    colors = {h: plt.cm.viridis(i / max(len(hint_fractions) - 1, 1)) for i, h in enumerate(hint_fractions)}
    eci_range = np.linspace(float(df["eci"].min()) - 5.0, float(df["eci"].max()) + 5.0, 120)

    n_cols = 7
    n_rows = max(1, int(np.ceil(len(hint_fractions) / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, hint_fraction in enumerate(hint_fractions):
        ax = axes_flat[idx]
        hint_df = df[df["hint_fraction"] == hint_fraction].sort_values("eci")
        train_df = hint_df[hint_df["split"] == "train"]
        test_df = hint_df[hint_df["split"] == "test"]
        color = colors[float(hint_fraction)]

        ax.scatter(train_df["eci"], train_df["accuracy"], color=color, alpha=0.8, s=40)
        if not test_df.empty:
            ax.scatter(
                test_df["eci"],
                test_df["accuracy"],
                color=color,
                alpha=0.8,
                s=40,
                marker="s",
                edgecolors="black",
            )

        y_joint = [joint_result["predict"](float(eci), float(hint_fraction)) for eci in eci_range]
        ax.plot(eci_range, y_joint, "--", color="gray", linewidth=2, label="joint (train)")

        train_fit = individual_by_hint_train.get(float(hint_fraction))
        if train_fit is not None:
            ax.plot(
                eci_range,
                [train_fit["predict"](float(eci)) for eci in eci_range],
                "-",
                color="orange",
                linewidth=2,
                label="indiv (train)",
            )

        all_fit = individual_by_hint_all.get(float(hint_fraction))
        if all_fit is not None:
            ax.plot(
                eci_range,
                [all_fit["predict"](float(eci)) for eci in eci_range],
                "-",
                color=color,
                linewidth=2,
                label="indiv (all)",
            )
            ax.axvline(float(all_fit["midpoint"]), color=color, linestyle=":", alpha=0.5)

        ax.set_title(f"h = {float(hint_fraction):.2f}", fontsize=11)
        ax.set_xlabel("eci")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        if idx == 0:
            ax.legend(fontsize=6)

    for idx in range(len(hint_fractions), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        _format_title_text([f"{label} - Individual fits per hint", f"Joint: {joint_equation}"]),
        fontsize=12,
    )
    fig.tight_layout()

    output_path = output_dir / f"{filename_stem}.png"
    _save_figure(fig, output_path)
    return output_path


def _plot_joint_accuracy_vs_hint_by_model(
    *,
    df: pd.DataFrame,
    eci_map: dict[str, float],
    joint_result: dict[str, Any],
    individual_by_model: dict[str, dict[str, Any]],
    label: str,
    joint_equation: str,
    output_dir: Path,
    filename_stem: str,
) -> Path:
    models_sorted = sorted(df["model"].unique().tolist(), key=lambda model: float(eci_map.get(str(model), 0.0)))
    n_cols = 4
    n_rows = max(1, int(np.ceil(len(models_sorted) / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    hint_range = np.linspace(0.0, 1.0, 120)
    hint_fractions = sorted(df["hint_fraction"].unique().tolist())
    colors = {model: plt.cm.coolwarm(i / max(len(models_sorted) - 1, 1)) for i, model in enumerate(models_sorted)}

    for idx, model in enumerate(models_sorted):
        ax = axes_flat[idx]
        model_df = df[df["model"] == model].sort_values("hint_fraction")
        eci = float(eci_map[str(model)])

        ax.scatter(model_df["hint_fraction"], model_df["accuracy"], color=colors[model], alpha=0.8, s=40)
        ax.plot(
            hint_range,
            [joint_result["predict"](eci, float(hint_fraction)) for hint_fraction in hint_range],
            "--",
            color="gray",
            linewidth=2,
            label="joint fit",
        )

        individual_fit = individual_by_model.get(str(model))
        if individual_fit is not None:
            ax.plot(
                hint_range,
                [individual_fit["predict"](float(hint_fraction)) for hint_fraction in hint_range],
                "-",
                color=colors[model],
                linewidth=2,
                label="individual fit",
            )

        ax.set_title(f"{model}\neci={eci:.1f}", fontsize=8)
        ax.set_xlabel("hint fraction")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(hint_fractions)
        ax.set_xticklabels([f"{float(h):.2f}" for h in hint_fractions], rotation=45, ha="right", fontsize=7)
        if idx == 0:
            ax.legend(fontsize=8)

    for idx in range(len(models_sorted), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        _format_title_text([f"{label} - Accuracy vs Hint per model", f"Joint: {joint_equation}"]),
        fontsize=12,
    )
    fig.tight_layout()

    output_path = output_dir / f"{filename_stem}.png"
    _save_figure(fig, output_path)
    return output_path


def _plot_h0_fits_by_model_sweep(
    *,
    df: pd.DataFrame,
    models_sorted_by_eci: list[str],
    include_cross: bool,
    lower_asymptote: float | None,
    label: str,
    output_dir: Path,
    filename_stem: str,
) -> Path:
    sweep_hint_fraction = 0.0
    df_hint0 = df[df["hint_fraction"] == sweep_hint_fraction].copy()
    if df_hint0.empty:
        raise ValueError("No rows found for hint_fraction = 0.0 when building sweep fit panels.")

    individual_by_hint_all = _fit_individual_sigmoids_by_hint(
        df=df,
        fit_models=None,
        lower=lower_asymptote,
    )
    all_fit_h0 = individual_by_hint_all.get(sweep_hint_fraction)

    eci_range = np.linspace(float(df_hint0["eci"].min()) - 5.0, float(df_hint0["eci"].max()) + 5.0, 120)
    n_panels = len(models_sorted_by_eci) - 1
    n_cols = 5
    n_rows = max(1, int(np.ceil(n_panels / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.2 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, n_models in enumerate(range(2, len(models_sorted_by_eci) + 1)):
        ax = axes_flat[idx]
        train_models = set(models_sorted_by_eci[:n_models])
        test_models = set(models_sorted_by_eci[n_models:])

        df_panel = df_hint0.copy()
        df_panel["split"] = df_panel["model"].map(lambda model: "train" if model in train_models else "test")
        train_df = df_panel[df_panel["split"] == "train"].sort_values("eci")
        test_df = df_panel[df_panel["split"] == "test"].sort_values("eci")

        joint_result = _fit_joint_sigmoid_model(
            df=df,
            fit_models=train_models,
            include_cross=include_cross,
            lower=lower_asymptote,
        )
        individual_by_hint_train = _fit_individual_sigmoids_by_hint(
            df=df,
            fit_models=train_models,
            lower=lower_asymptote,
        )
        train_fit_h0 = individual_by_hint_train.get(sweep_hint_fraction)

        ax.scatter(
            train_df["eci"],
            train_df["accuracy"],
            color="#1f77b4",
            alpha=0.8,
            s=36,
            marker="o",
            label="train data",
        )
        if not test_df.empty:
            ax.scatter(
                test_df["eci"],
                test_df["accuracy"],
                color="#1f77b4",
                alpha=0.8,
                s=36,
                marker="s",
                edgecolors="black",
                label="test data",
            )

        ax.plot(
            eci_range,
            [joint_result["predict"](float(eci), sweep_hint_fraction) for eci in eci_range],
            "--",
            color="gray",
            linewidth=2,
            label="joint (train)",
        )

        if train_fit_h0 is not None:
            ax.plot(
                eci_range,
                [train_fit_h0["predict"](float(eci)) for eci in eci_range],
                "-",
                color="orange",
                linewidth=2,
                label="indiv (train)",
            )

        if all_fit_h0 is not None:
            ax.plot(
                eci_range,
                [all_fit_h0["predict"](float(eci)) for eci in eci_range],
                "-",
                color="black",
                linewidth=2,
                alpha=0.9,
                label="indiv (all)",
            )
            ax.axvline(float(all_fit_h0["midpoint"]), color="black", linestyle=":", alpha=0.4)

        ax.set_title(f"n_train={n_models}, n_test={len(test_models)}", fontsize=10)
        ax.set_xlabel("eci")
        ax.set_ylabel("accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        _format_title_text([f"{label} - h = 0 fits across model sweep"], width=80),
        fontsize=12,
    )
    fig.tight_layout()

    output_path = output_dir / f"{filename_stem}.png"
    _save_figure(fig, output_path)
    return output_path


def _run_joint_model_sweep(
    *,
    df: pd.DataFrame,
    models_sorted_by_eci: list[str],
    include_cross: bool,
    lower_asymptote: float | None,
) -> pd.DataFrame:
    sweep_hint_fraction = 0.0
    individual_by_hint_all = _fit_individual_sigmoids_by_hint(
        df=df,
        fit_models=None,
        lower=lower_asymptote,
    )
    df_hint0 = df[df["hint_fraction"] == sweep_hint_fraction].copy()

    rows: list[dict[str, float]] = []
    for n_models in range(2, len(models_sorted_by_eci) + 1):
        train_models = set(models_sorted_by_eci[:n_models])
        test_models = set(models_sorted_by_eci[n_models:])

        joint_result = _fit_joint_sigmoid_model(
            df=df,
            fit_models=train_models,
            include_cross=include_cross,
            lower=lower_asymptote,
        )
        individual_by_hint_train = _fit_individual_sigmoids_by_hint(
            df=df,
            fit_models=train_models,
            lower=lower_asymptote,
        )

        midpoint_errors_joint = _compute_midpoint_errors(
            joint_result=joint_result,
            individual_fits=individual_by_hint_all,
            hint_fractions=[sweep_hint_fraction],
        )
        midpoint_errors_individual: dict[float, float] = {}
        for hint_fraction in [sweep_hint_fraction]:
            all_fit = individual_by_hint_all.get(float(hint_fraction))
            train_fit = individual_by_hint_train.get(float(hint_fraction))
            if all_fit is None or train_fit is None:
                continue
            midpoint_errors_individual[float(hint_fraction)] = abs(
                float(train_fit["midpoint"]) - float(all_fit["midpoint"])
            )

        row: dict[str, float] = {
            "n_models": float(n_models),
            "rms_h0_test": _compute_rms_joint(
                joint_result=joint_result,
                df=df_hint0,
                models=test_models,
            ) if test_models else float("nan"),
            "rms_indiv_h0_test": _compute_rms_individual_by_hint(
                individual_by_hint=individual_by_hint_train,
                df=df_hint0,
                models=test_models,
            ) if test_models else float("nan"),
            "rms_indiv_allfit_h0_test": _compute_rms_individual_by_hint(
                individual_by_hint=individual_by_hint_all,
                df=df_hint0,
                models=test_models,
            ) if test_models else float("nan"),
        }
        row["delta_rms_h0_test"] = row["rms_h0_test"] - row["rms_indiv_h0_test"]
        for hint_fraction in [sweep_hint_fraction]:
            key = f"midpoint_joint_h_{float(hint_fraction):.1f}"
            indiv_key = f"midpoint_indiv_h_{float(hint_fraction):.1f}"
            delta_key = f"delta_midpoint_h_{float(hint_fraction):.1f}"
            row[key] = float(midpoint_errors_joint.get(float(hint_fraction), float("nan")))
            row[indiv_key] = float(midpoint_errors_individual.get(float(hint_fraction), float("nan")))
            row[delta_key] = float(
                midpoint_errors_joint.get(float(hint_fraction), float("nan"))
                - midpoint_errors_individual.get(float(hint_fraction), float("nan"))
            )
        rows.append(row)

    sweep_df = pd.DataFrame(rows)
    sweep_df["n_models"] = sweep_df["n_models"].astype(int)
    return sweep_df


def _plot_joint_model_sweep(
    *,
    sweep_df: pd.DataFrame,
    label: str,
    output_dir: Path,
    filename_stem: str,
) -> Path:
    sweep_hint_fraction = 0.0
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(
        sweep_df["n_models"],
        sweep_df["rms_h0_test"],
        "o-",
        color="red",
        label="joint",
    )
    axes[0, 0].plot(
        sweep_df["n_models"],
        sweep_df["rms_indiv_h0_test"],
        "x--",
        color="red",
        alpha=0.9,
        label="individual (train fit)",
    )
    axes[0, 0].plot(
        sweep_df["n_models"],
        sweep_df["rms_indiv_allfit_h0_test"],
        "^-",
        color="black",
        alpha=0.85,
        label="individual (all fit)",
    )
    axes[0, 0].set_xlabel("number of train models")
    axes[0, 0].set_ylabel("rms")
    axes[0, 0].set_title("test models only, hint = 0")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(
        sweep_df["n_models"],
        sweep_df["delta_rms_h0_test"],
        "o-",
        color="red",
        label="joint - individual",
    )
    axes[0, 1].axhline(0.0, color="black", linestyle="--", alpha=0.5)
    axes[0, 1].set_xlabel("number of train models")
    axes[0, 1].set_ylabel("delta RMS (joint - individual)")
    axes[0, 1].set_title("test models only, hint = 0\n(negative = joint wins)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    midpoint_joint_col = f"midpoint_joint_h_{sweep_hint_fraction:.1f}"
    midpoint_indiv_col = f"midpoint_indiv_h_{sweep_hint_fraction:.1f}"
    if midpoint_joint_col in sweep_df.columns:
        axes[1, 0].plot(
            sweep_df["n_models"],
            sweep_df[midpoint_joint_col],
            "o-",
            color="#1f77b4",
            label="joint",
        )
    if midpoint_indiv_col in sweep_df.columns:
        axes[1, 0].plot(
            sweep_df["n_models"],
            sweep_df[midpoint_indiv_col],
            "x--",
            color="#1f77b4",
            alpha=0.9,
            label="individual",
        )
    axes[1, 0].set_xlabel("number of train models")
    axes[1, 0].set_ylabel("midpoint error (eci units)")
    axes[1, 0].set_title("midpoint error, hint = 0")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    delta_midpoint_col = f"delta_midpoint_h_{sweep_hint_fraction:.1f}"
    if delta_midpoint_col in sweep_df.columns:
        axes[1, 1].plot(
            sweep_df["n_models"],
            sweep_df[delta_midpoint_col],
            "o-",
            color="#1f77b4",
            label="joint - individual",
        )
    axes[1, 1].axhline(0.0, color="black", linestyle="--", alpha=0.5)
    axes[1, 1].set_xlabel("number of train models")
    axes[1, 1].set_ylabel("delta midpoint error (eci units)")
    axes[1, 1].set_title("delta midpoint error, hint = 0\n(negative = joint wins)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"{label} (fitting joint scaling)", fontsize=12)
    fig.tight_layout()

    output_path = output_dir / f"{filename_stem}.png"
    _save_figure(fig, output_path)
    return output_path


def _run_eci_joint_scaling(
    *,
    df: pd.DataFrame,
    models: list[str],
    output_dir: Path,
    label: str,
    include_cross: bool,
    lower_asymptote: float | None,
    num_holdout_models: int,
) -> dict[str, Any]:
    if num_holdout_models < 0:
        raise ValueError(f"num_holdout_models must be >= 0, got {num_holdout_models}")
    if num_holdout_models > len(models):
        raise ValueError(
            f"num_holdout_models ({num_holdout_models}) cannot exceed number of models ({len(models)})"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    models_sorted_by_eci = sorted(models, key=lambda model: float(df[df["model"] == model]["eci"].iloc[0]))
    holdout_models = set(models_sorted_by_eci[-num_holdout_models:]) if num_holdout_models > 0 else set()
    train_models = set(models_sorted_by_eci[:-num_holdout_models]) if num_holdout_models > 0 else set(models_sorted_by_eci)
    filename_suffix = f"__n_test_{len(holdout_models)}"

    df = df.copy()
    df["split"] = df["model"].map(lambda model: "train" if model in train_models else "test")

    joint_result = _fit_joint_sigmoid_model(
        df=df,
        fit_models=train_models,
        include_cross=include_cross,
        lower=lower_asymptote,
    )
    joint_equation = _format_joint_equation(joint_result)

    individual_by_hint_all = _fit_individual_sigmoids_by_hint(
        df=df,
        fit_models=None,
        lower=lower_asymptote,
    )
    individual_by_hint_train = _fit_individual_sigmoids_by_hint(
        df=df,
        fit_models=train_models,
        lower=lower_asymptote,
    )
    individual_by_model = _fit_individual_sigmoids_by_model(
        df=df,
        fit_models=None,
        lower=lower_asymptote,
    )

    plot_paths = {
        "accuracy_vs_eci_by_hint": str(
            _plot_joint_accuracy_vs_eci_by_hint(
                df=df,
                joint_result=joint_result,
                label=label,
                joint_equation=joint_equation,
                output_dir=output_dir,
                filename_stem=f"accuracy_vs_eci_by_hint{filename_suffix}",
            )
        ),
        "individual_fits_by_hint": str(
            _plot_joint_individual_fits_by_hint(
                df=df,
                joint_result=joint_result,
                individual_by_hint_all=individual_by_hint_all,
                individual_by_hint_train=individual_by_hint_train,
                label=label,
                joint_equation=joint_equation,
                output_dir=output_dir,
                filename_stem=f"individual_fits_by_hint{filename_suffix}",
            )
        ),
        "accuracy_vs_hint_by_model": str(
            _plot_joint_accuracy_vs_hint_by_model(
                df=df,
                eci_map={str(model): float(eci) for model, eci in zip(df["model"], df["eci"])},
                joint_result=joint_result,
                individual_by_model=individual_by_model,
                label=label,
                joint_equation=joint_equation,
                output_dir=output_dir,
                filename_stem=f"accuracy_vs_hint_by_model{filename_suffix}",
            )
        ),
        "h0_fits_by_model_sweep": str(
            _plot_h0_fits_by_model_sweep(
                df=df,
                models_sorted_by_eci=models_sorted_by_eci,
                include_cross=include_cross,
                lower_asymptote=lower_asymptote,
                label=label,
                output_dir=output_dir,
                filename_stem="h0_fits_by_model_sweep",
            )
        ),
    }

    sweep_df = _run_joint_model_sweep(
        df=df,
        models_sorted_by_eci=models_sorted_by_eci,
        include_cross=include_cross,
        lower_asymptote=lower_asymptote,
    )
    plot_paths["model_sweep"] = str(
        _plot_joint_model_sweep(
            sweep_df=sweep_df,
            label=label,
            output_dir=output_dir,
            filename_stem="model_sweep",
        )
    )

    metrics = {
        "joint_equation": joint_equation,
        "joint_params": [float(value) for value in np.asarray(joint_result["params"], dtype=float)],
        "include_cross": bool(include_cross),
        "lower_asymptote": lower_asymptote,
        "optimizer_success": bool(joint_result["optimizer_success"]),
        "optimizer_status": int(joint_result["optimizer_status"]),
        "optimizer_message": str(joint_result["optimizer_message"]),
        "n_train_models": int(len(train_models)),
        "n_test_models": int(len(holdout_models)),
        "rms_train": _compute_rms_joint(joint_result=joint_result, df=df, models=train_models),
        "rms_test": _compute_rms_joint(joint_result=joint_result, df=df, models=holdout_models) if holdout_models else float("nan"),
        "rms_all": _compute_rms_joint(joint_result=joint_result, df=df, models=None),
        "rms_indiv_train": _compute_rms_individual_by_hint(individual_by_hint=individual_by_hint_train, df=df, models=train_models),
        "rms_indiv_test": _compute_rms_individual_by_hint(individual_by_hint=individual_by_hint_train, df=df, models=holdout_models) if holdout_models else float("nan"),
        "rms_indiv_all": _compute_rms_individual_by_hint(individual_by_hint=individual_by_hint_train, df=df, models=None),
        "train_models": models_sorted_by_eci[: len(train_models)],
        "holdout_models": models_sorted_by_eci[len(train_models) :],
        "plot_paths": plot_paths,
    }
    metrics["delta_rms_train"] = float(metrics["rms_train"]) - float(metrics["rms_indiv_train"])
    metrics["delta_rms_test"] = float(metrics["rms_test"]) - float(metrics["rms_indiv_test"])
    metrics["delta_rms_all"] = float(metrics["rms_all"]) - float(metrics["rms_indiv_all"])
    midpoint_errors_all = _compute_midpoint_errors(
        joint_result=joint_result,
        individual_fits=individual_by_hint_all,
        hint_fractions=sorted(df["hint_fraction"].unique().tolist()),
    )
    metrics["mean_midpoint_error_all"] = (
        float(np.mean(list(midpoint_errors_all.values()))) if midpoint_errors_all else float("nan")
    )

    _write_json(output_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    args = _parse_args()
    if not 0.0 <= float(JOINT_LOWER_ASYMPTOTE) < 1.0:
        raise ValueError(
            f"JOINT_LOWER_ASYMPTOTE must be in [0, 1), got {JOINT_LOWER_ASYMPTOTE}"
        )

    scores_df = load_baseline_scores()
    eci_path = Path(args.eci_file)
    eci_map = _load_eci_map(eci_path)
    eci_benchmark_label = _eci_benchmark_label(eci_path)
    pca_result = _compute_pca_from_baselines(scores_df)
    pc1_map = pca_result["capability_map"]
    pc_benchmark_label = pca_result["benchmark_label"]
    pc1_equation = pca_result["equation"]

    combo_results, models = _load_combo_results(
        benchmark=args.benchmark,
        hint_type=args.hint_type,
        fractioner=args.fractioner,
    )
    if _is_old_gpqa_combo(
        benchmark=args.benchmark,
        hint_type=args.hint_type,
        fractioner=args.fractioner,
    ):
        print(f"[plot_accuracy_vs_eci_by_hint] using_old_gpqa_root={OLD_GPQA_ROOT}")

    models = _resolve_models_to_use(
        available_models=models,
        benchmark=args.benchmark,
    )
    rows: list[dict[str, Any]] = []
    print(
        f"[plot_accuracy_vs_eci_by_hint] selected_models={len(models)} "
        f"models={models}"
    )
    base_rows: list[dict[str, Any]] = []
    for model in models:
        if model not in combo_results:
            raise ValueError(
                f"Configured model missing combo results for benchmark={args.benchmark}: "
                f"model={model}"
            )
        for hint_fraction, stats in sorted(combo_results[model].items()):
            base_rows.append(
                {
                    "model": model,
                    "fractioner": args.fractioner,
                    "hint_fraction": float(hint_fraction),
                    "accuracy": float(stats["accuracy"]),
                    "ci_low": float(stats["ci_low"]),
                    "ci_high": float(stats["ci_high"]),
                }
            )

    if not base_rows:
        raise ValueError("No usable rows found after combining hinted accuracy with capability data.")

    output_dir = PLOTS_ROOT / f"{args.benchmark}__{args.hint_type}__{args.fractioner}"
    output_dir.mkdir(parents=True, exist_ok=True)
    joint_output_dir = output_dir / "joint_scaling_eci"
    _plot_pca_component_weights(
        components=pca_result["components"],
        benchmarks=pca_result["benchmarks"],
        output_path=output_dir / "pca_component_weights.png",
    )
    _plot_pca_explained_variance(
        explained_variance_ratio=pca_result["explained_variance_ratio"],
        output_path=output_dir / "pca_explained_variance.png",
    )
    views = [
        ("eci", "ECI", eci_map, eci_benchmark_label, None),
        ("pc1", "PC1", pc1_map, pc_benchmark_label, pc1_equation),
    ]
    for (
        capability_method,
        capability_label,
        capability_map,
        capability_benchmark_label,
        capability_equation,
    ) in views:
        rows = []
        for row in base_rows:
            model = str(row["model"])
            if model not in capability_map:
                continue
            rows.append(
                {
                    **row,
                    "capability": float(capability_map[model]),
                }
            )
        if not rows:
            print(
                f"[plot_accuracy_vs_eci_by_hint][WARN] no rows for capability_method={capability_method}"
            )
            continue

        missing_models = sorted({str(row["model"]) for row in base_rows} - set(capability_map.keys()))
        if missing_models:
            raise ValueError(
                f"Configured models missing {capability_method} capability values: {missing_models}"
            )

        output_path = _plot_capability_view(
            rows=rows,
            benchmark=args.benchmark,
            hint_type=args.hint_type,
            fractioner=args.fractioner,
            capability_method=capability_method,
            capability_label=capability_label,
            capability_benchmark_label=capability_benchmark_label,
            capability_equation=capability_equation,
            output_dir=output_dir,
        )
        print(f"[plot_accuracy_vs_eci_by_hint] {output_path}")
        per_hint_output_path = _plot_capability_view_per_hint_with_error_bars(
            rows=rows,
            benchmark=args.benchmark,
            hint_type=args.hint_type,
            fractioner=args.fractioner,
            capability_method=capability_method,
            capability_label=capability_label,
            capability_benchmark_label=capability_benchmark_label,
            capability_equation=capability_equation,
            output_dir=output_dir,
        )
        print(f"[plot_accuracy_vs_eci_by_hint] {per_hint_output_path}")

    joint_df = _build_joint_scaling_df(
        base_rows=base_rows,
        eci_map=eci_map,
        train_models=set(),
    )
    joint_metrics = _run_eci_joint_scaling(
        df=joint_df,
        models=models,
        output_dir=joint_output_dir,
        label=f"{args.benchmark} {args.fractioner} (ECI joint scaling)",
        include_cross=bool(args.include_cross),
        lower_asymptote=float(JOINT_LOWER_ASYMPTOTE),
        num_holdout_models=int(args.num_holdout_models),
    )
    print(f"[plot_accuracy_vs_eci_by_hint] joint_scaling_output_dir={joint_output_dir}")
    for name, path in sorted(joint_metrics["plot_paths"].items()):
        print(f"[plot_accuracy_vs_eci_by_hint] joint_plot[{name}]={path}")


if __name__ == "__main__":
    # python -m runs.plot_joint_scaling --benchmark aime2025_2026 --hint-type answer_not_revealed --include-cross --fractioner mask_word --num-holdout-models 4 --eci-file data/eci_model_capabilities__simple__arc_challenge--bbh__prompt_type_answer_only--hellaswag__split_validation--math__levels_5__fewshot_0--mmlu_5_shot__language_en_us__cot_true--piqa--winogrande__dataset_name_winogrande_xl__fewshot_5.csv


    # python -m runs.plot_joint_scaling --benchmark gpqa --hint-type answer_not_revealed --fractioner mask_word
    main()
