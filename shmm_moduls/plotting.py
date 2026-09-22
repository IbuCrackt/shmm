"""plotting.py - Diagramme aus RunResult-Listen (Kapitel 4, "Metriken und
Wiederholungen").

Unabhängig vom Training: braucht nur eine Liste von RunResult (z.B. aus
training.load_all_results) und kann daher jederzeit neu ausgeführt werden,
auch während oder nach einem abgebrochenen Lauf (siehe regenerate_plots.py).
"""
import csv
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def _color_for_index(i: int, n: int):
    cmap = plt.get_cmap("viridis")
    return cmap(i / max(n - 1, 1))


def plot_all_models_overview(results: list, metric: str, out_path: Path) -> None:
    """Überblicks-Plot: alle Modelle als Subplots in einer Datei."""
    exp_names = list(dict.fromkeys(r.exp_name for r in results))
    configs = list(dict.fromkeys(r.hierarchy_config for r in results if r.hierarchy_config))

    n = len(exp_names)
    if n == 0:
        return

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, exp_name in enumerate(exp_names):
        ax = axes[idx // ncols][idx % ncols]
        runs = [r for r in results if r.exp_name == exp_name]
        for i, run in enumerate(runs):
            if run.error is not None or metric not in run.history:
                continue
            ax.plot(run.history[metric], label=f"lr={run.learning_rate:g}",
                    color=_color_for_index(i, len(runs)), linewidth=2)
        ax.set_title(exp_name, fontsize=11, fontweight='bold')
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(metric.capitalize(), fontsize=10)
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    config_label = configs[0] if len(configs) == 1 else "Alle Configs"
    fig.suptitle(f"Überblick: Alle Modelle - {metric.upper()} [{config_label}]", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Überblicks-Plot gespeichert: {out_path}")


def plot_training_curves_per_model(results: list, metric: str, out_dir: Path) -> None:
    """Pro Modell (und ggf. pro hierarchy_config) ein separater Plot mit
    Lernraten als Kurven."""
    exp_names = list(dict.fromkeys(r.exp_name for r in results))
    configs = list(dict.fromkeys(r.hierarchy_config for r in results if r.hierarchy_config))
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = [(out_dir / c, c) for c in configs] if len(configs) > 1 else [(out_dir, None)]
    for target_dir, config in groups:
        target_dir.mkdir(parents=True, exist_ok=True)
        for exp_name in exp_names:
            runs = [r for r in results if r.exp_name == exp_name and (config is None or r.hierarchy_config == config)]
            if not any(metric in r.history for r in runs if r.error is None):
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            for i, run in enumerate(runs):
                if run.error is not None or metric not in run.history:
                    continue
                ax.plot(run.history[metric], label=f"lr={run.learning_rate:g}",
                        color=_color_for_index(i, len(runs)), linewidth=2)

            title = f"{exp_name}{f' [{config}]' if config else ''} - {metric.upper()}"
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.legend(fontsize=11, loc='best')
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(target_dir / f"{exp_name}_{metric}.png", dpi=150)
            plt.close(fig)
            print(f"  ✓ {target_dir.name}/{exp_name}_{metric}.png")


def plot_final_comparison(results: list, metric: str, out_path: Path) -> None:
    """Balkendiagramm: finaler Metrik-Wert pro Modell und Lernrate."""
    configs = list(dict.fromkeys(r.hierarchy_config for r in results if r.hierarchy_config))

    if len(configs) > 1:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for config in configs:
            config_results = [r for r in results if r.hierarchy_config == config]
            _plot_final_comparison_single(config_results, metric,
                                           out_path.parent / f"final_{metric}_comparison_{config}.png", config)
    else:
        _plot_final_comparison_single(results, metric, out_path, None)


def _plot_final_comparison_single(results: list, metric: str, out_path: Path, config_label: str | None = None) -> None:
    exp_names = list(dict.fromkeys(r.exp_name for r in results))
    lrs = list(dict.fromkeys(r.learning_rate for r in results))

    fig, ax = plt.subplots(figsize=(max(10, 1.5 * len(exp_names)), 7))
    bar_width = 0.8 / max(len(lrs), 1)
    x_base = range(len(exp_names))

    for i, lr in enumerate(lrs):
        values = []
        for exp_name in exp_names:
            match = [r for r in results if r.exp_name == exp_name and r.learning_rate == lr]
            values.append(match[0].history[metric][-1] if match and match[0].error is None and metric in match[0].history else float("nan"))
        offsets = [x + i * bar_width for x in x_base]
        ax.bar(offsets, values, width=bar_width, label=f"lr={lr:g}", color=_color_for_index(i, len(lrs)))

    tick_positions = [x + bar_width * (len(lrs) - 1) / 2 for x in x_base]
    ax.set_xticks(list(tick_positions))
    ax.set_xticklabels(exp_names, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel(f"Finaler {metric.upper()}", fontsize=12)
    title = f"Vergleich finaler {metric.upper()} pro Modell" + (f" [{config_label}]" if config_label else "")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Plot gespeichert: {out_path}")


def generate_plots_for_config(results: list, config_dir: Path, metrics=("loss", "accuracy")) -> None:
    """Erstellt das komplette Plot-Set (Overview, Pro-Modell-Kurven,
    finaler Vergleich) für EINE Gruppe von Ergebnissen (typischerweise
    eine hierarchy_config)."""
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        plot_all_models_overview(results, metric, config_dir / f"overview_{metric}.png")

    plots_subdir = config_dir / "plots"
    for metric in metrics:
        plot_training_curves_per_model(results, metric, plots_subdir / f"{metric}_curves")

    for metric in metrics:
        plot_final_comparison(results, metric, config_dir / f"final_{metric}_comparison.png")


def generate_all_plots(results: list, results_root: Path, metrics=("loss", "accuracy")) -> None:
    """Gruppiert `results` nach hierarchy_config und erzeugt für jede
    Gruppe das komplette Plot-Set. Kann sowohl am Ende eines
    Trainingslaufs als auch komplett unabhängig davon aufgerufen werden
    (siehe regenerate_plots.py)."""
    results_root = Path(results_root)
    configs = list(dict.fromkeys(r.hierarchy_config for r in results)) or [""]

    for config_name in configs:
        config_results = [r for r in results if r.hierarchy_config == config_name]
        if not config_results:
            continue
        target_dir = results_root if config_name == "" else results_root / config_name
        print(f"\n  📈 {config_name or '(Standard)'}:")
        generate_plots_for_config(config_results, target_dir, metrics=metrics)


_ALPHA_RE = re.compile(r"alpha([0-9]*\.?[0-9]+)")


def _extract_alpha(hierarchy_config: str) -> float | None:
    """Extrahiert den alpha-Wert aus dem `hierarchy_config`-String (z.B.
    "hmm0042_seed42_alpha0.734_M4" -> 0.734); None, falls kein
    alpha-Anteil vorhanden ist (z.B. bei der Klassifikationsaufgabe)."""
    if not hierarchy_config:
        return None
    match = _ALPHA_RE.search(hierarchy_config)
    return float(match.group(1)) if match else None


def plot_alpha_sweep(
    results: list, exp_names: list[str], out_path: Path, metric: str = "loss",
    learning_rate: float | None = None, alpha_extractor=_extract_alpha,
    scatter: bool = True, connect_sorted: bool = False,
) -> None:
    """
    Stellt den finalen Metrik-Wert mehrerer Modelle als Funktion des
    Mischungsparameters alpha dar - analog zu Lafferty, McCallum & Pereira
    (2001), Abschnitt 5.2 ("mixed-order sources"): dort wird über viele
    zufällig erzeugte Datensätze mit zufällig gezogenem alpha der
    Testfehler mehrerer Modelle gegen alpha aufgetragen. Erwartet
    RunResults, deren `hierarchy_config` den alpha-Wert enthält (siehe
    run_token_level.py); Läufe ohne erkennbares alpha werden ignoriert.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    any_points = False
    for i, exp_name in enumerate(exp_names):
        exp_results = [r for r in results if r.exp_name == exp_name and r.error is None]
        if learning_rate is not None:
            exp_results = [r for r in exp_results if r.learning_rate == learning_rate]

        by_config: dict[str, list] = {}
        for r in exp_results:
            by_config.setdefault(r.hierarchy_config, []).append(r)

        points = []
        for config, rs in by_config.items():
            alpha = alpha_extractor(config)
            if alpha is None:
                continue
            values = [r.history[metric][-1] for r in rs if metric in r.history]
            if values:
                points.append((alpha, min(values)))

        if not points:
            print(f"  ⚠ plot_alpha_sweep: keine Punkte für '{exp_name}' (metric={metric})")
            continue

        points.sort(key=lambda p: p[0])
        alphas, values = zip(*points)
        color = _color_for_index(i, max(len(exp_names), 2))
        if scatter:
            ax.scatter(alphas, values, s=18, alpha=0.6, color=color, label=exp_name)
        if connect_sorted:
            ax.plot(alphas, values, color=color, alpha=0.5, linewidth=1)
        any_points = True

    if not any_points:
        plt.close(fig)
        print("  ⚠ plot_alpha_sweep: keine Daten zum Plotten gefunden.")
        return

    n_datasets = len(set(r.hierarchy_config for r in results
                          if r.hierarchy_config and alpha_extractor(r.hierarchy_config) is not None))
    ax.set_xlabel(r"$\alpha$ (Mischparameter Order-1 $\rightarrow$ Order-2)", fontsize=12)
    ax.set_ylabel(f"finaler {metric}", fontsize=12)
    ax.set_title(f"{metric} vs. $\\alpha$ über {n_datasets} zufällige Datensätze\n"
                 f"(analog zu Lafferty, McCallum & Pereira 2001, Abschnitt 5.2)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Plot gespeichert: {out_path}")


def save_raw_results(results: list, out_path: Path) -> None:
    """Speichert alle Felder aller RunResults als eine JSON-Datei (bequemer
    Gesamt-Export, z.B. für externe Auswertung)."""
    import json
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"  ✓ Rohdaten gespeichert: {out_path}")


# --- Mittelwert/Streuung über die Wiederholungen (Abschnitt 4.4) -------
#
# Abschnitt 4.4 ("Metriken und Wiederholungen") beschreibt, dass der
# Modellvergleich NICHT auf einem einzelnen Lauf, sondern auf der
# Verteilung der Gütewerte über die unabhängig gezogenen Wiederholungen
# derselben Parameterkombination beruht (nach Reimers & Gurevych 2017).
# Die Funktionen unten setzen genau das um: Läufe mit identischer
# "Einstellung" (siehe group_fn) werden zusammengefasst und deren
# Mittelwert/Standardabweichung berechnet/geplottet.

def group_by_classification_config(hierarchy_config: str) -> str:
    """Gruppen-Schlüssel für die Klassifikationsaufgabe: entfernt das
    "_runNNNN"-Suffix aus dem hierarchy_config (siehe
    run_classification.py), sodass alle CLASS_NUM_RUNS Wiederholungen
    derselben Parameterkombination (D, K, L, C, M) in eine Gruppe fallen."""
    return re.sub(r"_run\d+$", "", hierarchy_config)


def group_all_token_level(hierarchy_config: str) -> str:
    """Gruppen-Schlüssel für die Zustandsdekodierung: wirft ALLE Läufe in
    eine einzige Gruppe. K, out_degree und M sind über alle
    TOKENLEVEL_NUM_HMMS Durchläufe konstant (siehe params.py) - nur das
    konkrete HMM und alpha variieren zufällig pro Durchlauf (Def. 3.1) -
    daher bilden laut Abschnitt 4.4 ALLE Durchläufe gemeinsam die
    Verteilung, über die gemittelt wird (siehe daneben plot_alpha_sweep
    für die Betrachtung ALS Funktion von alpha statt gemittelt darüber)."""
    return "alle_durchläufe"


def aggregate_final_metric(
    results: list, metric: str = "accuracy", group_fn=group_by_classification_config,
) -> dict[tuple[str, float, str], dict]:
    """
    Gruppiert `results` nach (exp_name, learning_rate, group_fn(hierarchy_config))
    und berechnet Mittelwert/Standardabweichung/Anzahl des jeweils LETZTEN
    (finalen) Werts von `metric` je Gruppe.

    Returns: {(exp_name, learning_rate, group): {"mean", "std", "n"}}
    """
    buckets: dict[tuple[str, float, str], list] = defaultdict(list)
    for r in results:
        if r.error is not None or metric not in r.history or not r.history[metric]:
            continue
        key = (r.exp_name, r.learning_rate, group_fn(r.hierarchy_config))
        buckets[key].append(r.history[metric][-1])

    return {
        key: {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}
        for key, values in buckets.items()
    }


def save_aggregate_table(
    results: list, out_path: Path, metrics=("loss", "accuracy"), group_fn=group_by_classification_config,
) -> None:
    """Schreibt eine CSV-Tabelle mit Mittelwert/Standardabweichung/Anzahl
    je (Gruppe, Modell, Lernrate, Metrik) - die Rohdaten für die
    Ergebnistabellen der Arbeit."""
    rows = []
    for metric in metrics:
        for (exp_name, lr, group), stats in aggregate_final_metric(results, metric=metric, group_fn=group_fn).items():
            rows.append({"group": group, "exp_name": exp_name, "learning_rate": lr, "metric": metric, **stats})
    rows.sort(key=lambda r: (r["group"], r["metric"], r["exp_name"]))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "exp_name", "learning_rate", "metric", "mean", "std", "n"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓ Aggregat-Tabelle gespeichert: {out_path}")


def select_best_learning_rates(
    results: list, group: str, metric: str = "loss", group_fn=group_by_classification_config,
) -> dict[str, float]:
    """Für jedes Modell (innerhalb `group`) die Lernrate mit dem besten
    mittleren `metric`-Wert (für 'loss' kleinster, sonst größter Mittelwert) -
    "das beste der vier Ergebnisse" aus Abschnitt 4.2. Gibt {exp_name: lr}
    zurück."""
    agg = aggregate_final_metric(results, metric=metric, group_fn=group_fn)
    better = (lambda a, b: a < b) if metric == "loss" else (lambda a, b: a > b)
    best: dict[str, tuple[float, float]] = {}  # exp_name -> (lr, mean)
    for (exp_name, lr, g), stats in agg.items():
        if g != group:
            continue
        cur = best.get(exp_name)
        if cur is None or better(stats["mean"], cur[1]):
            best[exp_name] = (lr, stats["mean"])
    return {exp_name: lr for exp_name, (lr, _mean) in best.items()}


def plot_metric_distribution(
    results: list, metric: str, out_path: Path, group: str, group_fn=group_by_classification_config,
) -> None:
    """
    Balkendiagramm mit Fehlerbalken: Mittelwert ± Standardabweichung von
    `metric` (letzte Epoche) je Modell über ALLE Wiederholungen von
    `group` hinweg - "das average Diagramm über alle Durchläufe derselben
    Einstellung" (Abschnitt 4.4). Je Modell wird die Lernrate mit dem
    besten mittleren Loss gezeigt (select_best_learning_rates).
    """
    best_lr = select_best_learning_rates(results, group, metric="loss", group_fn=group_fn)
    agg = aggregate_final_metric(results, metric=metric, group_fn=group_fn)

    names, means, stds, ns = [], [], [], []
    for exp_name, lr in best_lr.items():
        stats = agg.get((exp_name, lr, group))
        if stats is None:
            continue
        names.append(exp_name)
        means.append(stats["mean"])
        stds.append(stats["std"])
        ns.append(stats["n"])

    if not names:
        print(f"  ⚠ plot_metric_distribution: keine Daten für metric={metric}, group={group}")
        return

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(names)), 7))
    ax.bar(range(len(names)), means, yerr=stds, capsize=4, color=_color_for_index(2, 5))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel(f"{metric} (Mittelwert ± Std., letzte Epoche)", fontsize=12)
    ax.set_title(f"{metric} über {max(ns)} Wiederholungen [{group}]", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Plot gespeichert: {out_path}")


def plot_average_training_curve(
    results: list, exp_name: str, learning_rate: float, metric: str, out_path: Path,
    group: str, group_fn=group_by_classification_config,
) -> None:
    """
    Mittlere Trainingskurve (± Std. als Band) für EIN Modell/EINE
    Lernrate über alle Wiederholungen von `group` hinweg, epochenweise
    gemittelt. Da Läufe wegen des Konvergenzkriteriums (Abschnitt 4.2)
    unterschiedlich lange trainiert werden, wird pro Epoche nur über die
    Läufe gemittelt, die diese Epoche noch erreicht haben (kürzere Läufe
    tragen nur zu den ersten Epochen bei; np.nanmean/np.nanstd ignorieren
    die dann fehlenden Werte).
    """
    curves = [
        r.history[metric] for r in results
        if r.exp_name == exp_name and r.learning_rate == learning_rate and r.error is None
        and group_fn(r.hierarchy_config) == group and r.history.get(metric)
    ]
    if not curves:
        print(f"  ⚠ plot_average_training_curve: keine Daten für {exp_name} (lr={learning_rate:g}, metric={metric})")
        return

    max_len = max(len(c) for c in curves)
    padded = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        padded[i, :len(c)] = c
    mean = np.nanmean(padded, axis=0)
    std = np.nanstd(padded, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = np.arange(max_len)
    ax.plot(epochs, mean, color="steelblue", linewidth=2, label=f"Mittelwert (n={len(curves)})")
    ax.fill_between(epochs, mean - std, mean + std, color="steelblue", alpha=0.25, label="± 1 Std.")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f"{exp_name} (lr={learning_rate:g}) - {metric.upper()} über {len(curves)} Wiederholungen [{group}]",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Plot gespeichert: {out_path}")


# --- Trainingsdauer/Epochenzahl (Rechenaufwand-Vergleich) --------------

def aggregate_timing(
    results: list, group_fn=group_by_classification_config, by_learning_rate: bool = True,
) -> dict:
    """
    Mittelwert/Standardabweichung/Anzahl von train_seconds, epochs_trained
    und der daraus abgeleiteten Sekunden pro Epoche, gruppiert wie
    aggregate_final_metric. Läuft NICHT über evaluate_true_model-Ergebnisse
    (epochs_trained=0, kein Training) - diese werden automatisch
    ausgeschlossen. Ergänzt die Gütewert-Auswertung in Abschnitt 4.4 um
    den tatsächlichen Rechenaufwand je Modellklasse.

    by_learning_rate=True: Schlüssel (exp_name, learning_rate, group).
    by_learning_rate=False: Schlüssel (exp_name, group) - über alle
        Lernraten hinweg zusammengefasst (für den reinen
        Architektur-Kostenvergleich, unabhängig davon, welche Lernrate
        beim jeweiligen Lauf verwendet wurde).
    """
    buckets: dict[tuple, list] = defaultdict(list)
    for r in results:
        if r.error is not None or r.epochs_trained <= 0:
            continue
        key = ((r.exp_name, r.learning_rate, group_fn(r.hierarchy_config)) if by_learning_rate
               else (r.exp_name, group_fn(r.hierarchy_config)))
        buckets[key].append((r.train_seconds, r.epochs_trained, r.train_seconds / r.epochs_trained))

    out = {}
    for key, rows in buckets.items():
        seconds, epochs, per_epoch = (np.asarray(x, dtype=float) for x in zip(*rows))
        out[key] = {
            "train_seconds_mean": float(seconds.mean()), "train_seconds_std": float(seconds.std()),
            "epochs_trained_mean": float(epochs.mean()), "epochs_trained_std": float(epochs.std()),
            "seconds_per_epoch_mean": float(per_epoch.mean()), "seconds_per_epoch_std": float(per_epoch.std()),
            "n": len(rows),
        }
    return out


def save_timing_table(results: list, out_path: Path, group_fn=group_by_classification_config) -> None:
    """CSV mit Trainingsdauer, Epochenzahl und Sekunden/Epoche - Mittelwert/
    Std./n je (Gruppe, Modell, Lernrate)."""
    rows = []
    for (exp_name, lr, group), stats in aggregate_timing(results, group_fn=group_fn, by_learning_rate=True).items():
        rows.append({"group": group, "exp_name": exp_name, "learning_rate": lr, **stats})
    rows.sort(key=lambda r: (r["group"], r["exp_name"], r["learning_rate"]))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["group", "exp_name", "learning_rate", "train_seconds_mean", "train_seconds_std",
                  "epochs_trained_mean", "epochs_trained_std", "seconds_per_epoch_mean", "seconds_per_epoch_std", "n"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓ Zeit-Tabelle gespeichert: {out_path}")


_TIMING_LABELS = {
    "train_seconds": "Trainingsdauer (s)",
    "epochs_trained": "Anzahl Epochen",
    "seconds_per_epoch": "Sekunden/Epoche",
}


def plot_timing_distribution(
    results: list, out_path: Path, group: str, group_fn=group_by_classification_config,
    metric_key: str = "train_seconds",
) -> None:
    """Balkendiagramm: Mittelwert ± Std. von train_seconds/epochs_trained/
    seconds_per_epoch je Modell, über ALLE Lernraten und Wiederholungen
    von `group` hinweg (Rechenaufwand ist nicht Teil der in Abschnitt 4.4
    berichteten Gütemetriken, aber nützlich für den Vergleich der
    Modellklassen untereinander)."""
    agg = aggregate_timing(results, group_fn=group_fn, by_learning_rate=False)
    rows = [(exp_name, stats) for (exp_name, g), stats in agg.items() if g == group]
    rows.sort(key=lambda kv: kv[1][f"{metric_key}_mean"])
    if not rows:
        print(f"  ⚠ plot_timing_distribution: keine Daten für group={group}")
        return

    names = [name for name, _ in rows]
    means = [stats[f"{metric_key}_mean"] for _, stats in rows]
    stds = [stats[f"{metric_key}_std"] for _, stats in rows]

    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(names)), 7))
    ax.bar(range(len(names)), means, yerr=stds, capsize=4, color=_color_for_index(3, 5))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=11)
    label = _TIMING_LABELS[metric_key]
    ax.set_ylabel(f"{label} (Mittelwert ± Std.)", fontsize=12)
    ax.set_title(f"{label} über alle Lernraten/Wiederholungen [{group}]", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Plot gespeichert: {out_path}")


def generate_aggregate_report(
    results: list, out_dir: Path, group_fn=group_by_classification_config,
    metrics=("loss", "accuracy"),
) -> None:
    """
    Erzeugt den vollständigen Aggregat-Bericht über alle Wiederholungen
    einer Gruppe hinweg: eine CSV-Tabelle (Mittelwert/Std./n je Modell,
    Lernrate, Metrik) sowie je Gruppe ein Verteilungs-Balkendiagramm und
    eine gemittelte Trainingskurve pro Modell (jeweils bei der Lernrate
    mit dem besten mittleren Loss). Fasst save_aggregate_table,
    plot_metric_distribution und plot_average_training_curve zu einem
    Aufruf zusammen - der übliche Weg, das "Diagramm über alle N
    Wiederholungen derselben Einstellung" zu bekommen. Enthält zusätzlich
    eine Auswertung der Trainingsdauer (save_timing_table/
    plot_timing_distribution): Trainingszeit, Epochenzahl und Sekunden/
    Epoche je Modell, ebenfalls mit Mittelwert/Std. über alle
    Wiederholungen.
    """
    out_dir = Path(out_dir)
    save_aggregate_table(results, out_dir / "aggregate.csv", metrics=metrics, group_fn=group_fn)
    save_timing_table(results, out_dir / "timing.csv", group_fn=group_fn)

    groups = sorted({group_fn(r.hierarchy_config) for r in results if r.hierarchy_config or True})
    for group in groups:
        group_dir = out_dir / "aggregate" / group if group != "alle_durchläufe" else out_dir / "aggregate"
        for metric in metrics:
            plot_metric_distribution(results, metric, group_dir / f"{metric}_distribution.png", group=group, group_fn=group_fn)

        for metric_key in ("train_seconds", "epochs_trained", "seconds_per_epoch"):
            plot_timing_distribution(results, group_dir / f"{metric_key}_distribution.png", group=group, group_fn=group_fn, metric_key=metric_key)

        best_lr = select_best_learning_rates(results, group, metric="loss", group_fn=group_fn)
        for exp_name, lr in best_lr.items():
            for metric in metrics:
                plot_average_training_curve(
                    results, exp_name, lr, metric,
                    group_dir / "average_curves" / f"{exp_name}_{metric}.png",
                    group=group, group_fn=group_fn,
                )
