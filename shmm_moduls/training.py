"""training.py - Gemeinsame Trainings- und Auswertungslogik für beide
Aufgaben (Zustandsdekodierung, Sequenzklassifikation; siehe Kapitel 4 der
Bachelorarbeit, "Trainings- und Auswertungsverfahren").

Enthält:
    - RunResult: ein einzelnes, vollständig serialisierbares Trainings-
      /Auswertungsergebnis.
    - ConvergenceStopping / BestValueStopping / BestOfHeadsAdapter: die in
      Abschnitt 4.2 ("Trainingsverfahren") und 4.3 ("Von der versteckten
      Repräsentation zur Vorhersage") beschriebenen Stopp- und
      Auswahlmechanismen.
    - Absturzsicherheit: jedes RunResult wird sofort nach Abschluss als
      eigene JSON-Datei gespeichert (save_single_result); result_exists/
      load_single_result/cached_run erlauben es, einen abgebrochenen Lauf
      fortzusetzen, ohne bereits fertige Kombinationen erneut zu berechnen.
    - run_single_training: trainiert ein einzelnes Modell für genau eine
      der beiden Aufgaben ("token_level" oder "classification").
    - evaluate_true_model: wertet das datengenerierende HMM selbst aus
      (Abschnitt 4.1, "Referenzauswertung des datengenerierenden Modells"),
      ausschließlich per .evaluate(), niemals .fit().
"""
import json
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import tensorflow as tf
from hidten import HMMMode

from .model import get_model


@dataclass
class RunResult:
    exp_name: str
    learning_rate: float
    history: dict = field(default_factory=dict)
    train_seconds: float = 0.0
    error: str | None = None
    hierarchy_config: str = ""
    task: str = ""
    epochs_trained: int = 0
    converged: bool | None = None  # None = nicht anwendbar (feste Epochenzahl)
    best_loss: float | None = None
    best_epoch: int | None = None
    restored_best_weights: bool = False
    best_head: str | None = None  # nur bei task="classification": "head_mean" oder "head_last"


class ConvergenceStopping(tf.keras.callbacks.Callback):
    """
    Stoppt das Training, wenn sich der überwachte Wert (z.B. 'loss')
    zwischen zwei aufeinanderfolgenden Epochen für `patience` Epochen in
    Folge um weniger als `min_delta` verändert hat.

    Unterschied zu tf.keras.callbacks.EarlyStopping: EarlyStopping
    vergleicht jede Epoche mit dem BESTEN bisherigen Wert.
    ConvergenceStopping vergleicht jede Epoche nur mit der UNMITTELBAR
    VORHERIGEN Epoche - genau die "Änderung zwischen zwei Epochen" aus
    Abschnitt 4.2.
    """

    def __init__(self, min_delta: float, patience: int, monitor: str = "loss"):
        super().__init__()
        self.min_delta = min_delta
        self.patience = patience
        self.monitor = monitor
        self.prev_value: float | None = None
        self.wait = 0
        self.stopped_epoch: int | None = None
        self.converged = False

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.prev_value is not None:
            delta = abs(self.prev_value - current)
            if delta < self.min_delta:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.converged = True
                    self.model.stop_training = True
            else:
                self.wait = 0

        self.prev_value = current


class BestValueStopping(tf.keras.callbacks.Callback):
    """
    Verfolgt den bisher NIEDRIGSTEN gesehenen Wert von `monitor` (Standard:
    'loss') über den gesamten Trainingsverlauf. Wird dieser Bestwert für
    `patience` Epochen in Folge nicht mehr unterschritten, wird das
    Training gestoppt. In JEDEM Fall - egal ob dieser Callback selbst
    gestoppt hat, ConvergenceStopping gestoppt hat, oder max_epochs
    erreicht wurde - werden am Ende des Trainings die Gewichte auf den
    Zustand der besten Epoche zurückgesetzt (Abschnitt 4.2).
    """

    def __init__(self, patience: int, monitor: str = "loss", min_delta: float = 0.0):
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch: int | None = None
        self.restored = False

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.best_value is None or current < self.best_value - self.min_delta:
            self.best_value = current
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            self.restored = True


class BestOfHeadsAdapter(tf.keras.callbacks.Callback):
    """
    Für die duale Klassifizierungs-Kopf-Architektur (Mean-Pooling- und
    Last-Position-Kopf, Abschnitt 4.3 "Von der versteckten Repräsentation
    zur Vorhersage"): überschreibt logs['loss'] (und, falls vorhanden,
    logs['accuracy']) jede Epoche mit dem Wert des Kopfs mit dem
    niedrigeren Loss, damit alle nachfolgenden Callbacks (Convergence-
    Stopping, BestValueStopping, ModelCheckpoint) und das history-Objekt
    auf Basis des jeweils besseren Kopfs entscheiden bzw. diesen unter den
    STANDARD-Keys 'loss'/'accuracy' aufzeichnen. Ohne die 'accuracy'-
    Konsolidierung existiert für Klassifikationsläufe NUR
    'head_mean_accuracy'/'head_last_accuracy', aber kein 'accuracy' -
    jeder Plot, der (wie plot_all_models_overview/plot_final_comparison)
    nach 'accuracy' sucht, bliebe dann für alle Klassifikationsmodelle
    leer. Die einzelnen Kopf-Losses/Metriken bleiben zusätzlich unter
    ihren eigenen Keys erhalten.

    WICHTIG: muss als ERSTER Callback laufen, damit alle nachfolgenden
    bereits den angepassten Wert sehen.
    """

    def __init__(self, head_names: tuple[str, ...]):
        super().__init__()
        self.head_names = head_names
        self.best_head_per_epoch: list[str] = []

    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return
        available = [(h, logs[f"{h}_loss"]) for h in self.head_names if f"{h}_loss" in logs]
        if not available:
            return
        best_head, best_loss = min(available, key=lambda kv: kv[1])
        logs["loss"] = best_loss
        acc_key = f"{best_head}_accuracy"
        if acc_key in logs:
            logs["accuracy"] = logs[acc_key]
        self.best_head_per_epoch.append(best_head)


# --- Absturzsicherheit / Resumability -------------------------------

def _sanitize_filename_part(s: str) -> str:
    return str(s).replace("/", "_").replace(" ", "_")


def _result_path(results_dir: Path, hierarchy_config: str, exp_name: str, learning_rate: float) -> Path:
    """Deterministischer Dateipfad für ein einzelnes RunResult - dieselbe
    Konvention wird von save/load/result_exists verwendet."""
    results_dir = Path(results_dir)
    config_part = _sanitize_filename_part(hierarchy_config or "default")
    exp_part = _sanitize_filename_part(exp_name)
    lr_part = f"lr_{learning_rate:g}"
    return results_dir / f"{config_part}__{exp_part}__{lr_part}.json"


def result_exists(results_dir: Path, hierarchy_config: str, exp_name: str, learning_rate: float) -> bool:
    """Prüft, ob für die gegebene Kombination bereits ein ERFOLGREICHES
    Ergebnis gespeichert ist.

    Ein zuvor gespeichertes Ergebnis MIT gesetztem `error`-Feld zählt
    bewusst NICHT als "vorhanden" - es wird beim nächsten Aufruf
    automatisch erneut versucht. Sonst würde ein Fehler durch einen
    (inzwischen behobenen) Bug für immer als "fertig" zwischengespeichert
    bleiben und beim Fortsetzen eines abgebrochenen Laufs nie neu
    berechnet.
    """
    path = _result_path(results_dir, hierarchy_config, exp_name, learning_rate)
    if not path.exists():
        return False
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("error") is None


def load_single_result(
    results_dir: Path, hierarchy_config: str, exp_name: str, learning_rate: float
) -> "RunResult | None":
    path = _result_path(results_dir, hierarchy_config, exp_name, learning_rate)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return _dict_to_result(data)


def _dict_to_result(data: dict) -> RunResult:
    """Wandelt ein aus JSON geladenes dict robust in ein RunResult um."""
    data.setdefault("task", "")
    data.setdefault("epochs_trained", 0)
    data.setdefault("converged", None)
    data.setdefault("best_loss", None)
    data.setdefault("best_epoch", None)
    data.setdefault("restored_best_weights", False)
    data.setdefault("best_head", None)
    return RunResult(**data)


def save_single_result(result: RunResult, results_dir: Path) -> Path:
    """Speichert ein einzelnes RunResult sofort als eigene JSON-Datei -
    jeder Lauf bekommt eine eigene Datei, damit bei einem Absturz nichts
    durch eine halb geschriebene gemeinsame Datei korrumpiert werden kann
    und bereits fertige Läufe immer erhalten bleiben."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = _result_path(results_dir, result.hierarchy_config, result.exp_name, result.learning_rate)
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    return path


def load_all_results(results_dir: Path) -> list[RunResult]:
    """Lädt alle einzeln gespeicherten RunResult-JSON-Dateien aus einem Verzeichnis."""
    results_dir = Path(results_dir)
    return [_dict_to_result(json.load(open(fpath))) for fpath in sorted(results_dir.glob("*.json"))]


def all_cached(results_dir: Path, hierarchy_config: str, expected_pairs: list[tuple[str, float]]) -> bool:
    """True, wenn für ALLE (exp_name, lr)-Paare bereits ein erfolgreiches
    Ergebnis vorliegt - erlaubt es, die (teure) Datengenerierung für einen
    ganzen Durchlauf zu überspringen, wenn er schon vollständig vorliegt."""
    return all(result_exists(results_dir, hierarchy_config, name, lr) for name, lr in expected_pairs)


def load_cached(results_dir: Path, hierarchy_config: str, expected_pairs: list[tuple[str, float]]) -> list[RunResult]:
    return [load_single_result(results_dir, hierarchy_config, name, lr) for name, lr in expected_pairs]


def cached_run(
    results_dir: Path, hierarchy_config: str, exp_name: str, learning_rate: float,
    train_fn: Callable[[], RunResult],
) -> RunResult:
    """Lädt ein zwischengespeichertes RunResult, falls vorhanden und
    erfolgreich (siehe result_exists), sonst führt `train_fn()` aus (z.B.
    run_single_training oder evaluate_true_model) und gibt dessen Ergebnis
    zurück. Zentrale Stelle für die in Abschnitt 4.4 beschriebene
    Absturzsicherheit - vermeidet, dass beide Trainingsskripte denselben
    "vorhanden? laden : trainieren"-Code duplizieren."""
    if result_exists(results_dir, hierarchy_config, exp_name, learning_rate):
        return load_single_result(results_dir, hierarchy_config, exp_name, learning_rate)
    return train_fn()


def print_summary(results: list[RunResult]) -> None:
    """Knappe Abschluss-Ausgabe: wie viele Läufe erfolgreich/fehlgeschlagen
    sind. Wird von beiden Trainingsskripten am Ende aufgerufen."""
    failed = [r for r in results if r.error is not None]
    ok = len(results) - len(failed)
    print(f"\n✓ {ok}/{len(results)} Läufe erfolgreich"
          + (f", {len(failed)} fehlgeschlagen" if failed else ""))


# --- Referenzauswertung des datengenerierenden Modells (Abschnitt 4.1) --

def evaluate_true_model(
    hmm,
    dataset: tf.data.Dataset,
    T: int,
    steps: int,
    exp_name: str = "true_generator",
    hierarchy_config: str = "",
    results_dir: Path | None = None,
    num_symbols: int = 2,
) -> RunResult:
    """
    Wertet das DATENGENERIERENDE HMM auf demselben Dataset aus, das auch
    für das Training der übrigen Modelle verwendet wird - als fixe
    Referenz ("was ist mit Kenntnis der wahren Dynamik erreichbar",
    Abschnitt 4.1). Es wird ausschließlich `model.evaluate(...)`
    aufgerufen, niemals `model.fit(...)`: `.evaluate()` führt per
    Definition keine Gradienten-Updates aus; zusätzlich wird
    `model.trainable = False` gesetzt, um dies defensiv abzusichern. Das
    wahre HMM lernt hier also nie.

    `hmm(x, mode=HMMMode.POSTERIOR)` liefert für eine One-Hot-kodierte
    Emissionssequenz `x` der Form [Batch, T, num_symbols] eine
    Zustands-Posterior-Verteilung der Form [Batch, T, heads, num_states]
    (zusätzliche "heads"-Dimension an Achse 2, wie in HMMBlock.call und
    data.create_data). Da hier nicht mehrere Köpfe kombiniert werden
    müssen, wird die heads-Dimension einfach an Index 0 herausgeschnitten.
    """
    print(f"\n=== Referenzauswertung '{exp_name}' (wahres Modell, KEIN Training) "
          f"{f'| {hierarchy_config}' if hierarchy_config else ''} ===")

    try:
        input_layer = tf.keras.Input(shape=(T, num_symbols))
        outputs = hmm(input_layer, mode=HMMMode.POSTERIOR)
        outputs = outputs[:, :, 0, :]
        eval_model = tf.keras.Model(inputs=input_layer, outputs=outputs)
        eval_model.trainable = False  # defensiv: darf nie lernen

        eval_model.compile(
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

        start = time.time()
        eval_out = eval_model.evaluate(dataset, steps=steps, verbose=0, return_dict=True)
        elapsed = time.time() - start

        history = {k: [v] for k, v in eval_out.items()}
        print(f"  -> Referenz-loss={eval_out.get('loss'):.4f}"
              + (f", accuracy={eval_out['accuracy']:.4f}" if "accuracy" in eval_out else ""))

        result = RunResult(
            exp_name=exp_name, learning_rate=0.0, history=history, train_seconds=elapsed,
            hierarchy_config=hierarchy_config, task="token_level", epochs_trained=0,
        )
    except Exception as exc:
        print(f"  -> FEHLER bei der Referenzauswertung: {exc}")
        result = RunResult(
            exp_name=exp_name, learning_rate=0.0, error=str(exc),
            hierarchy_config=hierarchy_config, task="token_level",
        )

    if results_dir is not None:
        path = save_single_result(result, results_dir)
        print(f"  -> Ergebnis gespeichert: {path}")

    return result


# --- Training eines einzelnen Modells (Abschnitt 4.2/4.3) --------------

def run_single_training(
    exp_name: str,
    model_config: dict,
    learning_rate: float,
    T: int,
    weight_decay: float,
    dataset: tf.data.Dataset,
    save_checkpoints: bool,
    convergence_threshold: float,
    convergence_patience: int = 5,
    best_value_patience: int | None = None,
    max_epochs: int = 100,
    steps_per_epoch: int = 100,
    checkpoint_path=None,
    hierarchy_config: str = "",
    task: str = "token_level",  # "token_level" oder "classification" (Kapitel 3.1 / 3.2)
    results_dir: Path | None = None,
    num_symbols: int = 2,
) -> RunResult:
    """
    Trainiert ein einzelnes Modell bis zur Konvergenz (Abschnitt 4.2):
    Training läuft bis zu `max_epochs`, bricht aber ab, sobald sich der
    Loss für `convergence_patience` Epochen in Folge um weniger als
    `convergence_threshold` verändert hat. Parallel dazu läuft immer
    BestValueStopping: bricht zusätzlich ab, wenn `best_value_patience`
    Epochen kein neues Minimum mehr erreicht wurde. Am Ende werden IMMER
    die Gewichte der besten Epoche wiederhergestellt.

    task="token_level": Vorhersage pro Position, direkt die Modellausgabe.
    task="classification": zwei parallele Pooling-Köpfe (Mean/Last) auf
        der versteckten Repräsentation vor dem Unembed-Layer des
        Backbones, gemeinsam trainiert (Abschnitt 4.3); RunResult.best_head
        vermerkt, welcher Kopf am Ende gewann.

    results_dir: wenn gesetzt, wird das Ergebnis SOFORT nach Abschluss
        (auch im Fehlerfall) als eigene JSON-Datei dort gespeichert.
    num_symbols: Größe des Emissionsalphabets (One-Hot-Tiefe von
        `dataset`'s x-Komponente) - muss zur tatsächlichen Kodierung der
        Daten passen.
    """
    print(f"\n=== Training '{exp_name}' | lr={learning_rate:g}" +
          (f" | {hierarchy_config}" if hierarchy_config else "") + " ===")

    def _finish(result: RunResult) -> RunResult:
        if results_dir is not None:
            path = save_single_result(result, results_dir)
            print(f"  -> Ergebnis gespeichert: {path}")
        return result

    config = deepcopy(model_config)
    for optional_key in ("mlp", "rnn", "transformer", "mlp_only"):
        config.setdefault(optional_key, None)

    classification_head_names: tuple[str, ...] | None = None

    try:
        base_model = get_model(T=T, learning_rate=learning_rate, weight_decay=weight_decay,
                                num_symbols=num_symbols, **config)

        if task == "token_level":
            model = base_model

        elif task == "classification":
            # base_model.unembed wird hier NIE aufgerufen (return_hidden=True
            # liefert die Repräsentation davor, siehe Abschnitt 4.3). Ohne
            # trainable=False bleiben seine Gewichte trotzdem als trainierbare
            # Variablen des äußeren Modells registriert (Keras verfolgt alle
            # Sublayer-Gewichte, unabhängig davon, ob sie im konkreten
            # Forward-Pass benutzt werden) - der Optimizer versucht dann,
            # Gradienten für nie benutzte Variablen zu berechnen und Keras
            # warnt "Gradients do not exist for variables [...] when
            # minimizing the loss". Da dieser Layer für die
            # Klassifikationsaufgabe ohnehin nicht gebraucht wird, wird er
            # explizit von der Optimierung ausgenommen.
            base_model.unembed.trainable = False
            input_layer = tf.keras.Input(shape=(T, num_symbols))
            hidden = base_model(input_layer, return_hidden=True)  # [B, T, latent/d_model]

            mean_pooled = tf.keras.layers.GlobalAveragePooling1D()(hidden)
            last_pooled = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(hidden)

            num_classes = config.get("output")
            head_mean_logits = tf.keras.layers.Dense(num_classes, name="head_mean")(mean_pooled)
            head_last_logits = tf.keras.layers.Dense(num_classes, name="head_last")(last_pooled)

            model = tf.keras.Model(inputs=input_layer, outputs=[head_mean_logits, head_last_logits])
            model.compile(
                optimizer=tf.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
                loss={
                    "head_mean": tf.losses.CategoricalCrossentropy(from_logits=True),
                    "head_last": tf.losses.CategoricalCrossentropy(from_logits=True),
                },
                metrics={"head_mean": ["accuracy"], "head_last": ["accuracy"]},
            )
            classification_head_names = ("head_mean", "head_last")

        else:
            raise ValueError(f"Unbekannter task: {task} (erwartet: 'token_level' oder 'classification')")

    except Exception as exc:
        print(f"  -> FEHLER beim Erstellen des Modells: {exc}")
        return _finish(RunResult(exp_name, learning_rate, error=str(exc),
                                  hierarchy_config=hierarchy_config, task=task))

    callbacks = []
    if classification_head_names is not None:
        head_a, head_b = classification_head_names
        dataset = dataset.map(lambda x, y: (x, {head_a: y, head_b: y}))
        # muss als ERSTER Callback laufen (siehe BestOfHeadsAdapter-Docstring)
        head_adapter_cb = BestOfHeadsAdapter(head_names=classification_head_names)
        callbacks.append(head_adapter_cb)
    else:
        head_adapter_cb = None

    if save_checkpoints and checkpoint_path:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        config_suffix = f"_{hierarchy_config}" if hierarchy_config else ""
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path / f"lr_{learning_rate:g}{config_suffix}.weights.h5"),
            monitor="loss", verbose=0, save_best_only=False, save_weights_only=True,
        ))

    convergence_cb = ConvergenceStopping(min_delta=convergence_threshold, patience=convergence_patience, monitor="loss")
    callbacks.append(convergence_cb)
    resolved_best_patience = best_value_patience if best_value_patience is not None else convergence_patience
    best_value_cb = BestValueStopping(patience=resolved_best_patience, monitor="loss")
    callbacks.append(best_value_cb)

    start = time.time()
    try:
        history = model.fit(dataset, epochs=max_epochs, steps_per_epoch=steps_per_epoch,
                             callbacks=callbacks, verbose=2)
    except Exception as exc:
        print(f"  -> FEHLER beim Training: {exc}")
        return _finish(RunResult(exp_name, learning_rate, error=str(exc),
                                  hierarchy_config=hierarchy_config, task=task))

    elapsed = time.time() - start
    epochs_trained = len(history.history.get("loss", []))

    best_head = None
    if head_adapter_cb is not None and head_adapter_cb.best_head_per_epoch:
        idx = best_value_cb.best_epoch
        best_head = (head_adapter_cb.best_head_per_epoch[idx]
                     if idx is not None and idx < len(head_adapter_cb.best_head_per_epoch)
                     else head_adapter_cb.best_head_per_epoch[-1])

    status = (" (Konvergenz)" if convergence_cb.converged else
              " (kein neues Minimum mehr)" if best_value_cb.stopped_epoch is not None else
              " (max_epochs erreicht)")
    print(f"  -> fertig in {elapsed:.1f}s nach {epochs_trained} Epochen{status}, "
          f"bester loss={best_value_cb.best_value:.4f} (Epoche {best_value_cb.best_epoch})")
    if best_head is not None:
        print(f"  -> besserer Kopf: {best_head}")

    return _finish(RunResult(
        exp_name=exp_name, learning_rate=learning_rate, history=history.history, train_seconds=elapsed,
        hierarchy_config=hierarchy_config, task=task, epochs_trained=epochs_trained,
        converged=convergence_cb.converged, best_loss=best_value_cb.best_value,
        best_epoch=best_value_cb.best_epoch, restored_best_weights=best_value_cb.restored,
        best_head=best_head,
    ))
