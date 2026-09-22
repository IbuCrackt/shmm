"""regenerate_plots.py - Erstellt/aktualisiert Diagramme (inkl. Mittelwert/
Std. über alle Wiederholungen) aus zwischengespeicherten Trainings-
ergebnissen, unabhängig vom Training selbst (liest nur die JSON-Dateien
unter <results_dir>/partial_results/).

Nützlich um Diagramme zu sehen, während ein Training noch läuft, oder um
sie aus einem abgebrochenen/abgestürzten Lauf neu zu erzeugen, ohne neu zu
trainieren.

Direkt aufrufbar, kein Argumentparser:

    from regenerate_plots import regenerate_plots
    from rescrf.plotting import group_all_token_level, group_by_classification_config
    regenerate_plots(params.TOKENLEVEL_RESULTS_PATH, alpha_sweep=params.TOKENLEVEL_ALPHA_SWEEP_MODELS,
                      group_fn=group_all_token_level)
    regenerate_plots(params.CLASS_RESULTS_PATH, group_fn=group_by_classification_config)
"""
from pathlib import Path

from shmm_moduls.training import load_all_results
from shmm_moduls.plotting import (
    generate_all_plots, save_raw_results, plot_alpha_sweep,
    generate_aggregate_report, group_by_classification_config,
)


def regenerate_plots(
    results_dir: Path, metrics: tuple[str, ...] = ("loss", "accuracy"),
    alpha_sweep: list[str] | None = None, group_fn=group_by_classification_config,
) -> None:
    results_dir = Path(results_dir)
    partial_dir = results_dir / "partial_results"
    if not partial_dir.exists():
        print(f"Kein Verzeichnis mit Zwischenergebnissen gefunden: {partial_dir}")
        return

    results = load_all_results(partial_dir)
    if not results:
        print(f"Keine gespeicherten Ergebnisse in {partial_dir} gefunden.")
        return

    ok = [r for r in results if r.error is None]
    print(f"✓ {len(results)} Ergebnisse gefunden ({len(ok)} erfolgreich)")

    save_raw_results(results, results_dir / "results.json")
    if not ok:
        return

    generate_all_plots(ok, results_dir, metrics=metrics)

    if alpha_sweep:
        for metric in metrics:
            plot_alpha_sweep(ok, exp_names=alpha_sweep, out_path=results_dir / f"alpha_sweep_{metric}.png", metric=metric)

    generate_aggregate_report(ok, results_dir, group_fn=group_fn, metrics=metrics)

    print(f"\n✓ Diagramme aktualisiert unter: {results_dir.resolve()}")


if __name__ == "__main__":
    import params as P
    from shmm_moduls.plotting import group_all_token_level

    regenerate_plots(P.TOKENLEVEL_RESULTS_PATH, alpha_sweep=P.TOKENLEVEL_ALPHA_SWEEP_MODELS, group_fn=group_all_token_level)
    regenerate_plots(P.CLASS_RESULTS_PATH, group_fn=group_by_classification_config)
