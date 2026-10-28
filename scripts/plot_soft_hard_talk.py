"""Scientific figures from real immutable soft/hard checkpoints; no generated illustrations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"soft": "#E69F00", "hard": "#0072B2"}
STYLES = {"soft": "-.", "hard": "--"}
MARKERS = {19: "o", 73: "s", 1234: "^"}
LABELS = {"soft": "Soft", "hard": "Hard"}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 18,
        "axes.titlesize": 19,
        "axes.labelsize": 18,
        "legend.fontsize": 15,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 140,
        "axes.grid": True,
        "grid.alpha": 0.2,
    }
)


def save(fig, path):
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(path.with_suffix("." + suffix), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def table(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def positive_scale(ax, values):
    positive = np.asarray(values)
    positive = positive[positive > 0]
    if len(positive):
        ax.set_yscale("symlog", linthresh=min(positive.min() / 10, 1e-8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    marker = json.loads((args.bundle / "bundle.json").read_text())
    for name, sha in marker["sha256"].items():
        assert hashlib.sha256((args.bundle / name).read_bytes()).hexdigest() == sha
    evidence = json.loads((args.bundle / "evidence.json").read_text())
    trials = evidence["trials"]
    seeds = sorted({v["seed"] for v in trials})
    n = len(seeds)
    indexed = {(v["seed"], v["method"]): v for v in trials}

    def final(row):
        return row["snapshots"][-1]["metrics"]

    root = args.output
    fields = {}
    sections = {}
    for index, row in enumerate(trials):
        directory = args.assets / f"{index:02d}-{row['seed']}-{row['method']}"
        fields[(row["seed"], row["method"])] = dict(np.load(directory / "final-fields.npz"))
        sections[(row["seed"], row["method"])] = dict(np.load(directory / "sections.npz"))
    snapshots = []
    temporal = []
    bc = []
    cost_rows = []
    final_rows = []
    pairs = []
    for row in trials:
        common = dict(seed=row["seed"], method=row["method"])
        for snap in row["snapshots"]:
            m = snap["metrics"]
            snapshots.append(
                dict(
                    **common,
                    stage=snap["stage"],
                    seconds=snap["seconds"],
                    **{k: v for k, v in m.items() if not isinstance(v, (list, dict))},
                )
            )
            for value in m["temporal"]:
                temporal.append(dict(**common, training_seconds=snap["seconds"], **value))
            for face, value in m["boundaries"].items():
                bc.append(dict(**common, training_seconds=snap["seconds"], face=face, **value))
        final_rows.append(
            dict(
                **common,
                training_seconds=row["seconds"],
                checkpoint_seconds=row["snapshots"][-1]["seconds"],
                overshoot=row["overshoot_seconds"],
                adam_updates=row["adam_updates"],
                lbfgs_steps=row["lbfgs_steps"],
                closures=row["closures"],
                **{k: v for k, v in final(row).items() if not isinstance(v, (list, dict))},
            )
        )
        for operation, value in row["costs"].items():
            cost_rows.append(
                dict(
                    **common,
                    operation=operation,
                    **{k: v for k, v in value.items() if not isinstance(v, (list, dict))},
                )
            )
    for seed in seeds:
        soft, hard = final(indexed[seed, "soft"]), final(indexed[seed, "hard"])
        pairs.append(
            dict(
                seed=seed,
                soft=soft["relative_l2"],
                hard=hard["relative_l2"],
                hard_over_soft=hard["relative_l2"] / soft["relative_l2"],
                hard_wins=hard["relative_l2"] < soft["relative_l2"],
            )
        )
    for name, rows in [
        ("checkpoints", snapshots),
        ("physical-time", temporal),
        ("boundary-operators", bc),
        ("costs", cost_rows),
        ("final", final_rows),
        ("pairs", pairs),
    ]:
        table(root / (name + ".csv"), rows)
    # Seed 19 and final physical time were chosen before training.
    data = fields[19, "soft"]
    x = data["x"]
    time = data["t"]
    reference = data["reference"][:, :, -1]
    maps = [
        reference,
        fields[19, "soft"]["prediction"][:, :, -1],
        fields[19, "hard"]["prediction"][:, :, -1],
    ]
    vmin = min(v.min() for v in maps)
    vmax = max(v.max() for v in maps)
    titles = ["Эталон", "Soft", "Hard"]
    names = ["reference", "soft", "hard"]

    def heatmap(ax, value, title, vmin, vmax, cmap):
        ax.grid(False)
        image = ax.imshow(
            value.T,
            origin="lower",
            extent=[0, 1, 0, 1],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set(xlabel="x", ylabel="y", title=title)
        return image

    for name, title, value in zip(names, titles, maps):
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        image = heatmap(ax, value, title + "; t=0,5", vmin, vmax, "RdBu_r")
        fig.colorbar(image, ax=ax, label="Отклонение температуры u")
        if name != "reference":
            fig.suptitle(f"seed 19; обучение {indexed[19, name]['seconds']:.2f} с", fontsize=16)
        save(fig, root / f"01-temperature-{name}")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), layout="constrained")
    axes.flat[3].axis("off")
    for ax, value, title in zip(axes.flat, maps, titles):
        image = heatmap(ax, value, title, vmin, vmax, "RdBu_r")
    fig.colorbar(image, ax=list(axes.flat[:3]), label="u", shrink=0.8)
    fig.suptitle(
        "Температура: seed 19, t=0,5\n"
        + "; ".join(f"{LABELS[m]}: {indexed[19, m]['seconds']:.2f} с" for m in ("soft", "hard")),
        fontsize=17,
    )
    save(fig, root / "01-temperature-joint")
    errors = [np.abs(fields[19, m]["error"][:, :, -1]) for m in ("soft", "hard")]
    emax = max(v.max() for v in errors)
    for method, value in zip(("soft", "hard"), errors):
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        image = heatmap(ax, value, LABELS[method] + ": |u−u*|", 0, emax, "magma")
        fig.colorbar(image, ax=ax, label="Абсолютная ошибка")
        m = final(indexed[19, method])
        rmse = m["temporal"][-1]["rmse"]
        fig.suptitle(f"seed 19; t=0,5; max={value.max():.2e}; RMSE={rmse:.2e}", fontsize=15)
        save(fig, root / f"02-error-{method}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 7.5), layout="constrained")
    for ax, value, title in zip(axes, errors, ("Soft", "Hard")):
        image = heatmap(ax, value, title, 0, emax, "magma")
    fig.colorbar(image, ax=list(axes), label="|u−u*|", shrink=0.7)
    fig.suptitle("Общая линейная шкала; seed 19, t=0,5")
    save(fig, root / "02-error-joint")
    # Physical sections, including signed errors without division by a vanishing reference.
    for error_mode in (False, True):
        fig_all, axes = plt.subplots(1, 3, figsize=(10, 7.5), layout="constrained")
        for it, instant in enumerate((0.1, 0.25, 0.5)):
            truth = np.exp(-0.2 * np.pi**2 * instant) * np.cos(np.pi * x)
            for compact, ax in ((False, None), (True, axes[it])):
                if not compact:
                    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
                ax.plot(x, truth * 0 if error_mode else truth, "k-", label="Эталон", linewidth=2)
                for method in ("soft", "hard"):
                    value = sections[19, method]["prediction"][it]
                    ax.plot(
                        x,
                        value - truth if error_mode else value,
                        color=COLORS[method],
                        linestyle=STYLES[method],
                        label=LABELS[method],
                        linewidth=2,
                    )
                ax.set(
                    xlabel="x", ylabel="u−u*" if error_mode else "u", title=f"t={instant:g}; y=0"
                )
                if not compact:
                    ax.legend()
                    save(fig, root / f"03-section{'-error' if error_mode else ''}-{it + 1}")
        handles, labels = axes[0].get_legend_handles_labels()
        fig_all.legend(handles, labels, loc="outside lower center", ncol=3)
        fig_all.suptitle("Ошибка сечений" if error_mode else "Пространственные сечения; seed 19")
        save(fig_all, root / ("03-section-errors-joint" if error_mode else "03-sections-joint"))
    # Every observed seed is visible; medians use the exact same physical-time nodes.
    for metric, label in [
        ("relative_l2", "Относительная L2-ошибка"),
        ("rmse", "Пространственная RMSE"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        for method in ("soft", "hard"):
            values = []
            for seed in seeds:
                series = final(indexed[seed, method])["temporal"]
                v = np.array([s[metric] for s in series])
                values.append(v)
                ax.plot(
                    time,
                    v,
                    color=COLORS[method],
                    linestyle=STYLES[method],
                    alpha=0.45,
                    linewidth=1.2,
                    label=f"{LABELS[method]}, seed {seed}",
                )
            ax.plot(
                time,
                np.median(values, axis=0),
                color=COLORS[method],
                linestyle=STYLES[method],
                linewidth=3,
                label=f"{LABELS[method]}, медиана {n}",
            )
        ax.set_yscale("log")
        ax.set(xlabel="Физическое время t", ylabel=label, title="Финальные модели")
        ax.legend(fontsize=12, ncol=2)
        save(fig, root / f"04-physical-time-{metric}")
    # No interpolation or fictitious common training-time grid.
    for metric, ylabel, prefix in [
        ("relative_l2", "Общая относительная L2-ошибка", "05-error-training-time"),
        ("pde_rmse", "Независимая PDE RMS", "09-pde-training-time"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        for row in trials:
            ss = row["snapshots"]
            xx = [s["seconds"] for s in ss]
            yy = [s["metrics"][metric] for s in ss]
            method = row["method"]
            ax.plot(
                xx,
                yy,
                color=COLORS[method],
                linestyle=STYLES[method],
                marker=MARKERS[row["seed"]],
                label=f"{LABELS[method]}, seed {row['seed']}",
                linewidth=1.6,
            )
            transition = next(s for s in ss if s["stage"] == "adam_end")
            ax.scatter(
                [transition["seconds"]],
                [transition["metrics"][metric]],
                facecolors="none",
                edgecolors="black",
                s=130,
                zorder=5,
            )
        ax.axvline(60, color="gray", linestyle=":", label="План перехода: 60 с")
        ax.set_yscale("log")
        ax.set(
            xlabel="Фактическое время обучения, с",
            ylabel=ylabel,
            title="○ — последнее Adam-состояние",
        )
        ax.legend(fontsize=13, ncol=2)
        save(fig, root / prefix)
    for metric, title in [
        ("relative_l2", "Общая ошибка"),
        ("final_relative_l2", "Ошибка при t=0,5"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        for i, seed in enumerate(seeds):
            vals = [final(indexed[seed, m])[metric] for m in ("soft", "hard")]
            ax.plot([i - 0.12, i + 0.12], vals, color="gray", linewidth=1)
            for offset, method, val in zip((-0.12, 0.12), ("soft", "hard"), vals):
                ax.scatter(
                    i + offset,
                    val,
                    color=COLORS[method],
                    marker=MARKERS[seed],
                    s=90,
                    label=LABELS[method] if i == 0 else None,
                    zorder=3,
                )
            ax.annotate(
                f"H/S={vals[1] / vals[0]:.3g}",
                (i, np.sqrt(vals[0] * vals[1])),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=13,
            )
        for method in ("soft", "hard"):
            ax.axhline(
                np.median([final(indexed[s, method])[metric] for s in seeds]),
                color=COLORS[method],
                linestyle=STYLES[method],
                label=f"Медиана {LABELS[method]}",
            )
        ax.set_yscale("log")
        ax.set_xticks(range(n), [str(s) for s in seeds])
        ax.set(xlabel="Seed", ylabel="Относительная L2-ошибка", title=title)
        ax.legend()
        save(fig, root / "06-paired-{}".format(metric))
    bc_keys = ["ic", "x_lower", "y_lower", "y_upper", "x_upper"]
    bc_titles = ["Начальное условие", "Neumann: x=0", "Neumann: y=0", "Neumann: y=1", "Robin: x=1"]
    for key, title in zip(bc_keys, bc_titles):
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        all_values = []
        for method, offset in [("soft", -0.1), ("hard", 0.1)]:
            vals = [
                final(indexed[s, method])["ic_max"]
                if key == "ic"
                else final(indexed[s, method])["boundaries"][key]["max"]
                for s in seeds
            ]
            all_values.extend(vals)
            ax.plot(
                np.arange(n) + offset,
                vals,
                color=COLORS[method],
                linestyle=STYLES[method],
                marker="o",
                label=LABELS[method],
            )
        positive_scale(ax, all_values)
        ax.set_xticks(range(n), [str(s) for s in seeds])
        ax.set(xlabel="Seed", ylabel="Максимальный дефект (symlog)", title=title)
        ax.legend()
        save(fig, root / f"07-constraint-{key}")
    operations = ["loss_gradient", "adam_update", "lbfgs_closure"]
    opnames = ["Функционал + градиент", "Обновление Adam", "Замыкание L-BFGS"]
    fig, ax = plt.subplots(figsize=(10, 7.5), layout="constrained")
    ratios = []
    for i, op in enumerate(operations):
        medians = {}
        for method, offset in [("soft", -0.13), ("hard", 0.13)]:
            vals = []
            for seed in seeds:
                c = indexed[seed, method]["costs"][op]
                v = c["median_ms"]
                vals.append(v)
                ax.errorbar(
                    i + offset,
                    v,
                    yerr=[[max(0, v - c["min_ms"])], [max(0, c["max_ms"] - v)]],
                    fmt=MARKERS[seed],
                    color=COLORS[method],
                    capsize=3,
                    alpha=0.65,
                    label=f"{LABELS[method]}, {seed}" if i == 0 else None,
                )
            medians[method] = np.median(vals)
            ax.plot(
                [i + offset - 0.06, i + offset + 0.06],
                [medians[method]] * 2,
                color=COLORS[method],
                linewidth=4,
            )
        ratios.append(medians["hard"] / medians["soft"])
    ax.set_yscale("log")
    ax.set_xticks(range(3), opnames)
    ax.set(ylabel="Миллисекунды (log)", title="Цена операции: медианы и полный диапазон")
    ax.legend(fontsize=14, ncol=2)
    fig.text(
        0.5,
        0.01,
        "Hard/Soft медиан: " + ", ".join(f"{v:.2f}" for v in ratios),
        ha="center",
        fontsize=15,
    )
    save(fig, root / "08-operation-cost")
    for key, title in [
        ("adam_updates", "Обновления Adam"),
        ("lbfgs_steps", "Внешние вызовы L-BFGS"),
        ("closures", "Замыкания L-BFGS"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        for method, offset in [("soft", -0.16), ("hard", 0.16)]:
            ax.bar(
                np.arange(n) + offset,
                [indexed[s, method][key] for s in seeds],
                width=0.3,
                color=COLORS[method],
                label=LABELS[method],
            )
        ax.set_xticks(range(n), [str(s) for s in seeds])
        ax.set(xlabel="Seed", ylabel="Количество", title=title)
        ax.legend()
        save(fig, root / f"08-work-{key}")
    sizes = {
        suffix: sum(p.stat().st_size for p in root.glob("*." + suffix))
        for suffix in ("png", "pdf", "svg")
    }
    summary = dict(
        planned_seeds=[19, 73, 1234],
        observed_seeds=seeds,
        missing_seeds=sorted({19, 73, 1234} - set(seeds)),
        complete=len(trials) == 6,
        pairs=pairs,
        trials=final_rows,
        costs=cost_rows,
        plot_bytes=sizes,
        training_time_alignment="individual accepted checkpoints only; no interpolation",
        cost_errorbars="full repeat range, not confidence intervals",
        statistical_significance_claimed=False,
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    text = [
        "# Частичные результаты soft/hard" if len(trials) < 6 else "# Результаты soft/hard",
        "",
        f"Завершено {len(trials)}/6 запусков, {n}/3 полных пар. Не выполнены seed: {summary['missing_seeds']}.",
        "Предел пользователя — 20 минут; незавершённая матрица не выдаётся за n=3.",
        "",
        "| Seed | Soft Erel | Hard Erel | Hard/Soft |",
        "|---|---:|---:|---:|",
    ]
    text.extend(
        f"| {p['seed']} | {p['soft']:.6g} | {p['hard']:.6g} | {p['hard_over_soft']:.4g} |"
        for p in pairs
    )
    text += [
        "",
        "| Seed | Метод | Время, с | Adam | L-BFGS | Замыкания | Erel после Adam | Erel финал | PDE RMS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in trials:
        a = next(s for s in row["snapshots"] if s["stage"] == "adam_end")
        f = final(row)
        text.append(
            f"| {row['seed']} | {LABELS[row['method']]} | {row['seconds']:.3f} | {row['adam_updates']} | {row['lbfgs_steps']} | {row['closures']} | {a['metrics']['relative_l2']:.6g} | {f['relative_l2']:.6g} | {f['pde_rmse']:.6g} |"
        )
    text += [
        "",
        "| Seed | Операция | Soft, мс | Hard, мс | Hard/Soft |",
        "|---|---|---:|---:|---:|",
    ]
    for seed in seeds:
        for op, title in zip(operations, opnames):
            soft = indexed[seed, "soft"]["costs"][op]["median_ms"]
            hard = indexed[seed, "hard"]["costs"][op]["median_ms"]
            text.append(f"| {seed} | {title} | {soft:.4g} | {hard:.4g} | {hard / soft:.4g} |")
    text += [
        "",
        "Снижение ошибки после Adam описывает продолжение обучения; причинный эффект L-BFGS без отдельной абляции не установлен.",
    ]
    text += [
        "",
        "Кривые сохраняют фактические секунды; число внешних L-BFGS-вызовов не равно числу замыканий.",
        "Медианы и диапазоны повторений не являются доверительными интервалами. Непрерывная ошибка не сертифицирована.",
        "Время включает штатные сохранения; строгое равенство 180 секунд не заявляется. Презентация не создавалась.",
        "",
        f"Все PNG: {sizes['png'] / 1024**2:.2f} MiB; векторные PDF/SVG сохранены отдельно.",
    ]
    (root / "README.md").write_text("\n".join(text) + "\n")
    print(json.dumps(dict(trials=len(trials), seeds=seeds, plot_bytes=sizes, output=str(root))))


if __name__ == "__main__":
    main()
