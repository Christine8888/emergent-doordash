from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def format_title_text(lines: list[str], *, width: int = 72) -> str:
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


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _x_padding_from_df(df: pd.DataFrame, x_field: str) -> tuple[float, float, float]:
    x_min = float(df[x_field].min())
    x_max = float(df[x_field].max())
    x_span = max(x_max - x_min, 1e-6)
    x_padding = max(0.2, 0.1 * x_span)
    return x_min, x_max, x_padding


def plot_pca_component_weights(
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
    save_figure(fig, output_path)


def plot_pca_explained_variance(
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
    save_figure(fig, output_path)


def _print_accuracy_summary_table(
    rows: list[dict[str, Any]],
    hint_fractions: list[float],
    *,
    x_key: str,
    x_label: str,
) -> None:
    model_to_x = {
        str(row["model"]): float(row[x_key])
        for row in rows
    }
    score_map: dict[tuple[str, float], float] = {}
    for row in rows:
        score_map[(str(row["model"]), float(row["hint_fraction"]))] = float(row["accuracy"])

    models = sorted(model_to_x.keys(), key=lambda model: model_to_x[model])
    model_width = max(len("Model"), max(len(model) for model in models))
    x_width = max(len(x_label), 10)
    frac_headers = [f"h={hint_fraction:.1f}" for hint_fraction in hint_fractions]
    frac_width = max(7, max(len(header) for header in frac_headers))

    header = [
        "Model".ljust(model_width),
        x_label.rjust(x_width),
        *[header.rjust(frac_width) for header in frac_headers],
    ]
    separator = [
        "-" * model_width,
        "-" * x_width,
        *["-" * frac_width for _ in frac_headers],
    ]

    print("\nAccuracy summary by model and hint fraction:")
    print("  " + " ".join(header))
    print("  " + " ".join(separator))
    for model in models:
        row = [
            model.ljust(model_width),
            f"{model_to_x[model]:.3f}".rjust(x_width),
        ]
        for hint_fraction in hint_fractions:
            value = score_map.get((model, float(hint_fraction)))
            row.append("--".rjust(frac_width) if value is None else f"{value:.4f}".rjust(frac_width))
        print("  " + " ".join(row))


def _add_model_name_axis(
    ax: plt.Axes,
    *,
    rows: list[dict[str, Any]],
    x_key: str,
    label: str,
) -> None:
    plotted_models = sorted(
        {
            (str(row["model"]), float(row[x_key]))
            for row in rows
        },
        key=lambda item: item[1],
    )
    top_ax = ax.secondary_xaxis("top")
    top_ax.set_xticks([x_value for _, x_value in plotted_models])
    top_ax.set_xticklabels([model for model, _ in plotted_models], rotation=60, ha="left", fontsize=8)
    top_ax.set_xlabel(label, fontsize=11)


def plot_accuracy_vs_x_by_hint(
    *,
    rows: list[dict[str, Any]],
    benchmark: str,
    hint_type: str,
    fractioner: str,
    x_method: str,
    x_label: str,
    x_benchmark_label: str,
    x_equation: str | None,
    output_dir: Path,
    fit_series_fn: Callable[[list[float], list[float]], tuple[np.ndarray, np.ndarray] | None],
    x_key: str = "x_value",
) -> Path:
    hint_fractions = sorted({float(row["hint_fraction"]) for row in rows})
    _print_accuracy_summary_table(rows, hint_fractions, x_key=x_key, x_label=x_label)

    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.cm.viridis
    colors = {h: cmap(i / max(len(hint_fractions) - 1, 1)) for i, h in enumerate(hint_fractions)}

    for hint_fraction in hint_fractions:
        series_rows = sorted(
            [row for row in rows if float(row["hint_fraction"]) == hint_fraction],
            key=lambda row: float(row[x_key]),
        )
        xs = [float(row[x_key]) for row in series_rows]
        ys = [float(row["accuracy"]) for row in series_rows]
        color = colors[hint_fraction]

        ax.scatter(xs, ys, color=color, alpha=0.85, s=45, label=f"h={hint_fraction:.2f}")

        fit = fit_series_fn(xs, ys)
        if fit is not None:
            x_fit, y_fit = fit
            ax.plot(x_fit, y_fit, "-", color=color, alpha=0.7, linewidth=2)

    _add_model_name_axis(
        ax,
        rows=rows,
        x_key=x_key,
        label="Model",
    )
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    title_lines = [
        f"Accuracy vs {x_label} by Hint Fraction",
        f"benchmark={benchmark} hint_type={hint_type} fractioner={fractioner}",
        f"{x_method}_benchmarks={x_benchmark_label}",
    ]
    if x_equation:
        title_lines.append(x_equation)
    ax.set_title(format_title_text(title_lines), fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    output_path = output_dir / f"accuracy_vs_{x_method}_by_hint.png"
    fig.tight_layout()
    save_figure(fig, output_path)
    return output_path


def plot_accuracy_vs_x_by_hint_by_family(
    *,
    rows: list[dict[str, Any]],
    benchmark: str,
    hint_type: str,
    fractioner: str,
    x_method: str,
    x_label: str,
    x_benchmark_label: str,
    x_equation: str | None,
    output_dir: Path,
    fit_series_fn: Callable[[list[float], list[float]], tuple[np.ndarray, np.ndarray] | None],
    x_key: str = "x_value",
) -> Path:
    hint_fractions = sorted({float(row["hint_fraction"]) for row in rows})
    family_names = sorted({str(row["model_family"]) for row in rows})
    if not family_names:
        raise ValueError("No model families found for family-faceted plot.")

    n_panels = len(family_names)
    ncols = min(2, max(1, n_panels))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(8 * ncols, 5.5 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes_flat = np.atleast_1d(axes).flatten()

    cmap = plt.cm.viridis
    colors = {h: cmap(i / max(len(hint_fractions) - 1, 1)) for i, h in enumerate(hint_fractions)}

    for panel_idx, family_name in enumerate(family_names):
        ax = axes_flat[panel_idx]
        family_rows = [row for row in rows if str(row["model_family"]) == family_name]
        for hint_fraction in hint_fractions:
            series_rows = sorted(
                [row for row in family_rows if float(row["hint_fraction"]) == hint_fraction],
                key=lambda row: float(row[x_key]),
            )
            if not series_rows:
                continue

            xs = [float(row[x_key]) for row in series_rows]
            ys = [float(row["accuracy"]) for row in series_rows]
            color = colors[hint_fraction]

            ax.scatter(xs, ys, color=color, alpha=0.85, s=45)

            fit = fit_series_fn(xs, ys)
            if fit is not None:
                x_fit, y_fit = fit
                ax.plot(x_fit, y_fit, "-", color=color, alpha=0.7, linewidth=2)

        _add_model_name_axis(
            ax,
            rows=family_rows,
            x_key=x_key,
            label="Model",
        )
        ax.set_title(family_name, fontsize=12)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    for panel_idx in range(n_panels, len(axes_flat)):
        axes_flat[panel_idx].set_visible(False)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=colors[hint_fraction],
            label=f"h={hint_fraction:.2f}",
            markersize=6,
        )
        for hint_fraction in hint_fractions
    ]
    title_lines = [
        f"Accuracy vs {x_label} by Hint Fraction and Model Family",
        f"benchmark={benchmark} hint_type={hint_type} fractioner={fractioner}",
        f"{x_method}_benchmarks={x_benchmark_label}",
    ]
    if x_equation:
        title_lines.append(x_equation)
    fig.suptitle(format_title_text(title_lines), fontsize=14)
    fig.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 0.9, 0.95))

    output_path = output_dir / f"accuracy_vs_{x_method}_by_hint_by_family.png"
    save_figure(fig, output_path)
    return output_path


def plot_accuracy_vs_x_by_hint_subplots_with_error_bars(
    *,
    rows: list[dict[str, Any]],
    benchmark: str,
    hint_type: str,
    fractioner: str,
    x_method: str,
    x_label: str,
    x_benchmark_label: str,
    x_equation: str | None,
    output_dir: Path,
    fit_series_fn: Callable[[list[float], list[float]], tuple[np.ndarray, np.ndarray] | None],
    x_key: str = "x_value",
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
            key=lambda row: float(row[x_key]),
        )
        if not series_rows:
            continue

        xs = np.asarray([float(row[x_key]) for row in series_rows], dtype=float)
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

        fit = fit_series_fn(xs.tolist(), ys.tolist())
        if fit is not None:
            x_fit, y_fit = fit
            ax.plot(x_fit, y_fit, "-", color="#ff7f0e", alpha=0.8, linewidth=2)

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"hint_fraction={hint_fraction:.2f}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    title_lines = [
        f"Accuracy vs {x_label} with Error Bars by Hint Fraction",
        f"benchmark={benchmark} hint_type={hint_type} fractioner={fractioner}",
        f"{x_method}_benchmarks={x_benchmark_label}",
    ]
    if x_equation:
        title_lines.append(x_equation)
    fig.suptitle(format_title_text(title_lines), fontsize=14)
    output_path = output_dir / f"accuracy_vs_{x_method}_by_hint_subplots_with_error_bars.png"
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output_path)
    return output_path


def plot_joint_accuracy_vs_x_by_hint(
    *,
    df: pd.DataFrame,
    x_field: str,
    x_label: str,
    joint_predict_fn: Callable[[float, float], float],
    label: str,
    joint_equation: str,
    output_dir: Path,
    filename_stem: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 7))
    hint_fractions = sorted(df["hint_fraction"].unique().tolist())
    colors = {h: plt.cm.viridis(i / max(len(hint_fractions) - 1, 1)) for i, h in enumerate(hint_fractions)}
    x_min, x_max, x_padding = _x_padding_from_df(df, x_field)
    x_range = np.linspace(x_min - x_padding, x_max + x_padding, 120)

    for hint_fraction in hint_fractions:
        hint_df = df[df["hint_fraction"] == hint_fraction].sort_values(x_field)
        train_df = hint_df[hint_df["split"] == "train"]
        test_df = hint_df[hint_df["split"] == "test"]
        color = colors[float(hint_fraction)]

        ax.scatter(
            train_df[x_field],
            train_df["accuracy"],
            color=color,
            alpha=0.8,
            s=60,
            marker="o",
            label=f"h={float(hint_fraction):.2f}",
        )
        if not test_df.empty:
            ax.scatter(
                test_df[x_field],
                test_df["accuracy"],
                color=color,
                alpha=0.8,
                s=60,
                marker="s",
                edgecolors="black",
            )

        y_fit = [joint_predict_fn(float(x_value), float(hint_fraction)) for x_value in x_range]
        ax.plot(x_range, y_fit, "-", color=color, alpha=0.5, linewidth=2)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("accuracy", fontsize=12)
    ax.set_title(format_title_text([label, joint_equation]), fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = output_dir / f"{filename_stem}.png"
    save_figure(fig, output_path)
    return output_path


def plot_joint_individual_fits_by_hint(
    *,
    df: pd.DataFrame,
    x_field: str,
    x_label: str,
    joint_predict_fn: Callable[[float, float], float],
    individual_by_hint_all: dict[float, dict[str, Any]],
    individual_by_hint_train: dict[float, dict[str, Any]],
    label: str,
    joint_equation: str,
    output_dir: Path,
    filename_stem: str,
) -> Path:
    hint_fractions = sorted(df["hint_fraction"].unique().tolist())
    colors = {h: plt.cm.viridis(i / max(len(hint_fractions) - 1, 1)) for i, h in enumerate(hint_fractions)}
    x_min, x_max, x_padding = _x_padding_from_df(df, x_field)
    x_range = np.linspace(x_min - x_padding, x_max + x_padding, 120)

    n_cols = 7
    n_rows = max(1, int(np.ceil(len(hint_fractions) / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, hint_fraction in enumerate(hint_fractions):
        ax = axes_flat[idx]
        hint_df = df[df["hint_fraction"] == hint_fraction].sort_values(x_field)
        train_df = hint_df[hint_df["split"] == "train"]
        test_df = hint_df[hint_df["split"] == "test"]
        color = colors[float(hint_fraction)]

        ax.scatter(train_df[x_field], train_df["accuracy"], color=color, alpha=0.8, s=40)
        if not test_df.empty:
            ax.scatter(
                test_df[x_field],
                test_df["accuracy"],
                color=color,
                alpha=0.8,
                s=40,
                marker="s",
                edgecolors="black",
            )

        y_joint = [joint_predict_fn(float(x_value), float(hint_fraction)) for x_value in x_range]
        ax.plot(x_range, y_joint, "--", color="gray", linewidth=2, label="joint (train)")

        train_fit = individual_by_hint_train.get(float(hint_fraction))
        if train_fit is not None:
            ax.plot(
                x_range,
                [train_fit["predict"](float(x_value)) for x_value in x_range],
                "-",
                color="orange",
                linewidth=2,
                label="indiv (train)",
            )

        all_fit = individual_by_hint_all.get(float(hint_fraction))
        if all_fit is not None:
            ax.plot(
                x_range,
                [all_fit["predict"](float(x_value)) for x_value in x_range],
                "-",
                color=color,
                linewidth=2,
                label="indiv (all)",
            )
            ax.axvline(float(all_fit["midpoint"]), color=color, linestyle=":", alpha=0.5)

        ax.set_title(f"h = {float(hint_fraction):.2f}", fontsize=11)
        ax.set_xlabel(x_label)
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        if idx == 0:
            ax.legend(fontsize=6)

    for idx in range(len(hint_fractions), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        format_title_text([f"{label} - Individual fits per hint", f"Joint: {joint_equation}"]),
        fontsize=12,
    )
    fig.tight_layout()

    output_path = output_dir / f"{filename_stem}.png"
    save_figure(fig, output_path)
    return output_path


def plot_joint_accuracy_vs_hint_by_model(
    *,
    df: pd.DataFrame,
    model_to_x: dict[str, float],
    x_label: str,
    joint_predict_fn: Callable[[float, float], float],
    individual_by_model: dict[str, dict[str, Any]],
    label: str,
    joint_equation: str,
    output_dir: Path,
    filename_stem: str,
) -> Path:
    models_sorted = sorted(df["model"].unique().tolist(), key=lambda model: float(model_to_x.get(str(model), 0.0)))
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
        x_value = float(model_to_x[str(model)])

        ax.scatter(model_df["hint_fraction"], model_df["accuracy"], color=colors[model], alpha=0.8, s=40)
        ax.plot(
            hint_range,
            [joint_predict_fn(x_value, float(hint_fraction)) for hint_fraction in hint_range],
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

        ax.set_title(f"{model}\n{x_label}={x_value:.1f}", fontsize=8)
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
        format_title_text([f"{label} - Accuracy vs Hint per model", f"Joint: {joint_equation}"]),
        fontsize=12,
    )
    fig.tight_layout()

    output_path = output_dir / f"{filename_stem}.png"
    save_figure(fig, output_path)
    return output_path


def plot_h0_fits_by_model_sweep(
    *,
    panels: list[dict[str, Any]],
    x_label: str,
    label: str,
    output_dir: Path,
    filename_stem: str,
) -> Path:
    n_panels = len(panels)
    n_cols = 5
    n_rows = max(1, int(np.ceil(n_panels / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.2 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, panel in enumerate(panels):
        ax = axes_flat[idx]
        train_df = panel["train_df"]
        test_df = panel["test_df"]
        x_range = np.asarray(panel["x_range"], dtype=float)
        predict_joint = panel["predict_joint"]
        predict_train = panel.get("predict_train")
        predict_all = panel.get("predict_all")
        midpoint_all = panel.get("midpoint_all")
        n_train = int(panel["n_train"])
        n_test = int(panel["n_test"])

        ax.scatter(
            train_df["x"],
            train_df["accuracy"],
            color="#1f77b4",
            alpha=0.8,
            s=36,
            marker="o",
            label="train data",
        )
        if not test_df.empty:
            ax.scatter(
                test_df["x"],
                test_df["accuracy"],
                color="#1f77b4",
                alpha=0.8,
                s=36,
                marker="s",
                edgecolors="black",
                label="test data",
            )

        ax.plot(
            x_range,
            [predict_joint(float(x_value)) for x_value in x_range],
            "--",
            color="gray",
            linewidth=2,
            label="joint (train)",
        )

        if predict_train is not None:
            ax.plot(
                x_range,
                [predict_train(float(x_value)) for x_value in x_range],
                "-",
                color="orange",
                linewidth=2,
                label="indiv (train)",
            )

        if predict_all is not None:
            ax.plot(
                x_range,
                [predict_all(float(x_value)) for x_value in x_range],
                "-",
                color="black",
                linewidth=2,
                alpha=0.9,
                label="indiv (all)",
            )
            if midpoint_all is not None:
                ax.axvline(float(midpoint_all), color="black", linestyle=":", alpha=0.4)

        ax.set_title(f"n_train={n_train}, n_test={n_test}", fontsize=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel("accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        format_title_text([f"{label} - h = 0 fits across model sweep"], width=80),
        fontsize=12,
    )
    fig.tight_layout()

    output_path = output_dir / f"{filename_stem}.png"
    save_figure(fig, output_path)
    return output_path


def plot_joint_model_sweep(
    *,
    sweep_df: pd.DataFrame,
    x_label: str,
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
    axes[1, 0].set_ylabel(f"midpoint error ({x_label} units)")
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
    axes[1, 1].set_ylabel(f"delta midpoint error ({x_label} units)")
    axes[1, 1].set_title("delta midpoint error, hint = 0\n(negative = joint wins)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"{label} (fitting joint scaling)", fontsize=12)
    fig.tight_layout()

    output_path = output_dir / f"{filename_stem}.png"
    save_figure(fig, output_path)
    return output_path


def plot_joint_x_axis_model_sweep_comparison(
    *,
    comparison_df: pd.DataFrame,
    label: str,
    output_dir: Path,
    filename_stem: str,
) -> Path:
    if comparison_df.empty:
        raise ValueError("comparison_df must not be empty for model sweep comparison plotting.")

    plot_df = comparison_df.sort_values(["sort_index", "n_models"]).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    for _, method_df in plot_df.groupby("x_axis_name", sort=False):
        method_df = method_df.sort_values("n_models").reset_index(drop=True)
        line_label = str(method_df["comparison_label"].iloc[0])
        axes[0].plot(
            method_df["n_models"],
            method_df["rms_h0_test"],
            "o-",
            linewidth=2,
            label=line_label,
        )
        axes[1].plot(
            method_df["n_models"],
            method_df["delta_rms_h0_test"],
            "o-",
            linewidth=2,
            label=line_label,
        )

    axes[0].set_xlabel("number of train models")
    axes[0].set_ylabel("rms")
    axes[0].set_title("test models only, hint = 0")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("number of train models")
    axes[1].set_ylabel("delta RMS (joint - individual)")
    axes[1].set_title("test models only, hint = 0\n(negative = joint wins)")
    axes[1].axhline(0.0, color="black", linestyle="--", alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False)

    fig.suptitle(
        format_title_text([f"{label} - model sweep across x-axis methods"], width=90),
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))

    output_path = output_dir / f"{filename_stem}.png"
    save_figure(fig, output_path)
    return output_path


def plot_joint_x_axis_delta_rms_comparison(
    *,
    comparison_df: pd.DataFrame,
    label: str,
    output_dir: Path,
    filename_stem: str,
) -> Path:
    if comparison_df.empty:
        raise ValueError("comparison_df must not be empty for delta RMS comparison plotting.")

    x_positions = np.arange(len(comparison_df), dtype=float)
    x_labels = comparison_df["comparison_label"].astype(str).tolist()
    metric_specs = [
        ("delta_rms_train", "train", "#1f77b4"),
        ("delta_rms_test", "test", "#d62728"),
        ("delta_rms_all", "all", "#2ca02c"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(max(12, 0.8 * len(comparison_df)), 12), sharex=True)
    for ax, (metric_name, split_label, color) in zip(axes, metric_specs, strict=True):
        ax.bar(x_positions, comparison_df[metric_name].to_numpy(dtype=float), color=color, alpha=0.85)
        ax.axhline(0.0, color="black", linestyle="--", alpha=0.5)
        ax.set_ylabel("delta RMS")
        ax.set_title(f"{split_label} split (joint - individual train fit)")
        ax.grid(True, axis="y", alpha=0.3)

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(x_labels, rotation=45, ha="right")
    axes[-1].set_xlabel("x-axis method")

    fig.suptitle(
        format_title_text([f"{label} - delta RMS across x-axis methods"], width=90),
        fontsize=12,
    )
    fig.tight_layout()

    output_path = output_dir / f"{filename_stem}.png"
    save_figure(fig, output_path)
    return output_path


def plot_joint_x_axis_absolute_rms_comparison(
    *,
    comparison_df: pd.DataFrame,
    label: str,
    output_dir: Path,
    filename_stem: str,
) -> Path:
    if comparison_df.empty:
        raise ValueError("comparison_df must not be empty for absolute RMS comparison plotting.")

    x_positions = np.arange(len(comparison_df), dtype=float)
    x_labels = comparison_df["comparison_label"].astype(str).tolist()
    metric_specs = [
        ("rms_train", "rms_indiv_train", "train"),
        ("rms_test", "rms_indiv_test", "test"),
        ("rms_all", "rms_indiv_all", "all"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(max(12, 0.8 * len(comparison_df)), 12), sharex=True)
    bar_width = 0.38
    for ax, (joint_metric, indiv_metric, split_label) in zip(axes, metric_specs, strict=True):
        ax.bar(
            x_positions - bar_width / 2.0,
            comparison_df[joint_metric].to_numpy(dtype=float),
            width=bar_width,
            color="#d62728",
            alpha=0.85,
            label="joint",
        )
        ax.bar(
            x_positions + bar_width / 2.0,
            comparison_df[indiv_metric].to_numpy(dtype=float),
            width=bar_width,
            color="#1f77b4",
            alpha=0.85,
            label="individual (train fit)",
        )
        ax.set_ylabel("RMS")
        ax.set_title(f"{split_label} split")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(x_labels, rotation=45, ha="right")
    axes[-1].set_xlabel("x-axis method")

    fig.suptitle(
        format_title_text([f"{label} - absolute RMS across x-axis methods"], width=90),
        fontsize=12,
    )
    fig.tight_layout()

    output_path = output_dir / f"{filename_stem}.png"
    save_figure(fig, output_path)
    return output_path


def plot_joint_x_axis_delta_rms_family(
    *,
    comparison_df: pd.DataFrame,
    label: str,
    output_dir: Path,
    filename_stem: str,
) -> Path:
    if comparison_df.empty:
        raise ValueError("comparison_df must not be empty for family delta RMS plotting.")

    plot_df = comparison_df.sort_values("hint_fraction").reset_index(drop=True)
    hint_fractions = plot_df["hint_fraction"].to_numpy(dtype=float)
    metric_specs = [
        ("delta_rms_train", "train", "#1f77b4"),
        ("delta_rms_test", "test", "#d62728"),
        ("delta_rms_all", "all", "#2ca02c"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for ax, (metric_name, split_label, color) in zip(axes, metric_specs, strict=True):
        ax.plot(
            hint_fractions,
            plot_df[metric_name].to_numpy(dtype=float),
            "o-",
            color=color,
            linewidth=2,
        )
        ax.axhline(0.0, color="black", linestyle="--", alpha=0.5)
        ax.set_ylabel("delta RMS")
        ax.set_title(f"{split_label} split (joint - individual train fit)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("hint fraction")
    axes[-1].set_xticks(hint_fractions)

    fig.suptitle(
        format_title_text([f"{label} - delta RMS for hinted accuracy logit family"], width=90),
        fontsize=12,
    )
    fig.tight_layout()

    output_path = output_dir / f"{filename_stem}.png"
    save_figure(fig, output_path)
    return output_path


def plot_joint_x_axis_absolute_rms_family(
    *,
    comparison_df: pd.DataFrame,
    label: str,
    output_dir: Path,
    filename_stem: str,
) -> Path:
    if comparison_df.empty:
        raise ValueError("comparison_df must not be empty for family absolute RMS plotting.")

    plot_df = comparison_df.sort_values("hint_fraction").reset_index(drop=True)
    hint_fractions = plot_df["hint_fraction"].to_numpy(dtype=float)
    metric_specs = [
        ("rms_train", "rms_indiv_train", "train"),
        ("rms_test", "rms_indiv_test", "test"),
        ("rms_all", "rms_indiv_all", "all"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for ax, (joint_metric, indiv_metric, split_label) in zip(axes, metric_specs, strict=True):
        ax.plot(
            hint_fractions,
            plot_df[joint_metric].to_numpy(dtype=float),
            "o-",
            color="#d62728",
            linewidth=2,
            label="joint",
        )
        ax.plot(
            hint_fractions,
            plot_df[indiv_metric].to_numpy(dtype=float),
            "x--",
            color="#1f77b4",
            linewidth=2,
            alpha=0.9,
            label="individual (train fit)",
        )
        ax.set_ylabel("RMS")
        ax.set_title(f"{split_label} split")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("hint fraction")
    axes[-1].set_xticks(hint_fractions)

    fig.suptitle(
        format_title_text([f"{label} - absolute RMS for hinted accuracy logit family"], width=90),
        fontsize=12,
    )
    fig.tight_layout()

    output_path = output_dir / f"{filename_stem}.png"
    save_figure(fig, output_path)
    return output_path
