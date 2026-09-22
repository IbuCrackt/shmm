from __future__ import annotations

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

Template = Union[np.ndarray, List["Template"]]

BASE_ATTAMPTS = 50


def _get_primitive_pattern_test(L: int, rng: np.random.Generator) -> np.ndarray:
    """Liefert eines von 6 einfachen 0/1-Grundmustern der Länge L (Level 0)."""
    reps = L // 2 + 1
    templates = [
        np.ones(L, dtype=int),
        np.zeros(L, dtype=int),
        np.tile([0, 1], reps)[:L],
        np.concatenate([np.zeros(L // 2, dtype=int), np.ones(L - L // 2, dtype=int)]),
        np.concatenate([np.ones(L // 2, dtype=int), np.zeros(L - L // 2, dtype=int)]),
    ]
    idx = rng.integers(len(templates))
    return templates[idx].copy()

def _get_primitive_pattern(L: int, rng: np.random.Generator, alphabet_size: int = 2) -> np.ndarray:
    return rng.integers(0, alphabet_size, size=L)


def _sample_noise_length(L: int, alpha: float, rng: np.random.Generator) -> int:
    """Länge ~ Beta(alpha, alpha), linear auf [0.5*L, 1.5*L] skaliert."""
    frac = rng.beta(alpha, alpha)
    length = 0.5 * L + frac * L
    return max(1, int(round(length)))


def _create_noise(
    L: int, alpha: float, rng: np.random.Generator, alphabet_size: int = 2
) -> np.ndarray:
    """Zufällige Noise-Sequenz (Symbole aus {0, ..., alphabet_size-1}) mit
    Beta-verteilter Länge."""
    length = _sample_noise_length(L, alpha, rng)
    return rng.integers(0, alphabet_size, size=length)


def _random_symbols(n: int, rng: np.random.Generator, alphabet_size: int = 2) -> np.ndarray:
    """Exakt n zufällige Symbole aus {0, ..., alphabet_size-1}."""
    if n <= 0:
        return np.empty(0, dtype=int)
    return rng.integers(0, alphabet_size, size=n)


def _template_signature(template: Template) -> tuple:
    """Eindeutige, hashbare Signatur eines Templates (zum Vergleich auf Gleichheit)."""
    if isinstance(template, np.ndarray):
        return ("leaf", template.tobytes())
    return ("node", tuple(_template_signature(c) for c in template))


def _build_template(
    level: int,
    L: int,
    K: int,
    alpha: float,
    rng: np.random.Generator,
    check_uniqueness: bool = True,
    max_retries: int = BASE_ATTAMPTS,
    alphabet_size: int = 2,
) -> Template:
    """
    Baut rekursiv die *feste* Struktur eines Musters von `level` bis 0 auf.

    level == 0: ein einzelnes Grundmuster (np.ndarray) der Länge L, mit
        Symbolen aus {0, ..., alphabet_size-1}.
    level > 0: eine Liste von K unterschiedlichen Sub-Templates aus
        level - 1. Noise wird hier NICHT gespeichert, da sie bei
        jeder Instanziierung neu gewürfelt wird.
    """
    if level == 0:
        return _get_primitive_pattern(L, rng, alphabet_size=alphabet_size)

    children: List[Template] = []
    seen: set = set()
    stale_attempts = 0
    while len(children) < K:
        child = _build_template(
            level - 1, L, K, alpha, rng, check_uniqueness, max_retries, alphabet_size
        )
        sig = _template_signature(child)
        if check_uniqueness and sig in seen:
            stale_attempts += 1
            if stale_attempts <= max_retries:
                continue
            warnings.warn(
                f"Konnte auf Level {level} keine {K} unterschiedlichen Muster finden "
                f"(K vermutlich zu groß relativ zur verfügbaren Vielfalt oder einfach nur pech) "
                f"akzeptiere Duplikat."
            )
        seen.add(sig)
        children.append(child)
        stale_attempts = 0
    return children


def _instantiate(
    template: Template,
    L: int,
    alpha: float,
    rng: np.random.Generator,
    head_noise: Optional[bool],
    tail_noise: Optional[bool],
    alphabet_size: int = 2,
) -> np.ndarray:
    """
    Erzeugt aus einem Template eine konkrete Sequenz über dem
    `alphabet_size`-wertigen Alphabet. Die Struktur (welches Submuster wo
    steht) kommt aus dem Template, alle Noise-Segmente werden hier
    gewürfelt (mit erwartete Länge L).
    """
    if isinstance(template, np.ndarray):
        return template.copy()

    use_head = False if head_noise is None else head_noise
    use_tail = (rng.random() < 0.5) if tail_noise is None else tail_noise

    parts: List[np.ndarray] = []
    if use_head:
        parts.append(_create_noise(L, alpha, rng, alphabet_size=alphabet_size))

    for i, child in enumerate(template):
        parts.append(_instantiate(child, L, alpha, rng, head_noise, tail_noise, alphabet_size))
        if i < len(template) - 1:
            parts.append(_create_noise(L, alpha, rng, alphabet_size=alphabet_size))

    if use_tail:
        parts.append(_create_noise(L, alpha, rng, alphabet_size=alphabet_size))

    return np.concatenate(parts)

def create_hierarchical_data(
    N: int,
    L: int,
    classes: int,
    K: int,
    D: int,
    alpha: float = 3.0,
    head_noise: Optional[bool] = False,
    tail_noise: Optional[bool] = None,
    check_uniqueness: bool = True,
    seed: Optional[int] = None,
    alphabet_size: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Erstellt hierarchische Daten (siehe Paper).

    Aufbau:
        - Level 0: primitive Muster der Länge L über einem
          `alphabet_size`-wertigen Alphabet (Standard 4, NICHT binär -
          analog zum synthetischen Vergleichsexperiment bei Lafferty,
          McCallum & Pereira 2001, Abschnitt 5.2).
        - Level d (1 <= d <= D): K unterschiedliche Muster aus Level d-1,
          jeweils getrennt durch Noise. Die Noise-Länge ist
          Beta(alpha, alpha)-verteilt mit Erwartungswert L (skaliert auf
          [0.5*L, 1.5*L]), unabhängig vom Level. Jede komponierte Sequenz
          endet zusätzlich zufällig (50/50) mit Noise oder fest,
          falls head_noise/tail_noise gesetzt sind.
        - Level D: das Klassenmuster. Pro Klasse wird EIN Template
          erzeugt, die Noise wird bei jedem der Samples neu
          gewürfelt.

    Da sich pro Kompositionsschritt K Muster + Noise dazwischen/außen
    aneinanderreihen, ist die resultierende Gesamtlänge einer Level-D-
    Sequenz im Erwartungswert deutlich größer als L (wächst etwa mit einer
    Potenz von D) und variiert zusätzlich zufällig von Sample zu
    Sample. Daher werden alle N Sequenzen rechtsseitig mit zufälligen
    Symbolen auf die über alle Samples maximal aufgetretene Länge
    aufgefüllt.

    Args:
        N: Anzahl der Samples insgesamt.
        L: Länge der primitiven Muster (Level 0) und Erwartungswert der
            Noise-Länge auf jedem Level.
        classes: Anzahl der Klassen.
        K: Anzahl unterschiedlicher Submuster pro Level.
        D: Anzahl der Level (D=0 -> nur Level-0-Grundmuster als Klassen).
        alpha: Parameter der Beta-Verteilung für die Noise-Länge.
        head_noise / tail_noise: True/False erzwingt Noise am Anfang/Ende
            jeder komponierten Sequenz, None = zufällig 50/50.
        check_uniqueness: prüft, ob die K Submuster je Level (und die
            Klassen-Templates) paarweise unterschiedlich sind.
        seed: Random Seed.
        alphabet_size: Größe des Emissionsalphabets (Anzahl möglicher
            Symbolwerte in den erzeugten `emissions`). Standard 4 statt
            binär. Der Aufrufer MUSS denselben Wert konsistent für die
            One-Hot-Kodierung der Daten verwenden (siehe `num_symbols`
            in `training.py` bzw. `tf.one_hot(..., depth=alphabet_size)`
            in `auto_train_classification_simple.py`).

    Returns:
        emissions: [N, max_len] array mit Werten aus {0, ...,
            alphabet_size-1} (max_len = Länge der längsten
            instanziierten Sequenz; kürzere werden aufgefüllt).
        states: [N, classes] one-hot Array, welche Klasse aktiv ist.
    """
    if classes < 1 or K < 1 or D < 0 or L < 1 or N < 1:
        raise ValueError("N, L, classes, K müssen >= 1 sein und D >= 0.")
    if alphabet_size < 2:
        raise ValueError("alphabet_size muss >= 2 sein.")

    rng = np.random.default_rng(seed)

    class_templates: List[Template] = []
    class_signatures: set = set()
    for _ in range(classes):
        attempts = 0
        while True:
            template = _build_template(
                D, L, K, alpha, rng, check_uniqueness, alphabet_size=alphabet_size
            )
            sig = _template_signature(template)
            if not check_uniqueness or sig not in class_signatures or attempts > BASE_ATTAMPTS:
                class_signatures.add(sig)
                class_templates.append(template)
                if attempts > BASE_ATTAMPTS:
                    warnings.warn(
                        f"Konnte für die Klassen keine {classes} unterschiedlichen Muster finden "
                        f"(classes vermutlich zu groß relativ zur verfügbaren Vielfalt oder einfach nur pech) "
                        f"akzeptiere Duplikat."
                    )
                break
            attempts += 1

    base = N // classes
    counts = [base] * classes
    for i in range(N - base * classes):
        counts[i % classes] += 1

    sequences: List[np.ndarray] = []
    labels: List[int] = []
    for c, template in enumerate(class_templates):
        for _ in range(counts[c]):
            seq = _instantiate(
                template, L, alpha, rng, head_noise, tail_noise, alphabet_size=alphabet_size
            )
            sequences.append(seq)
            labels.append(c)

    order = rng.permutation(len(sequences))

    max_len = max(len(s) for s in sequences)
    emissions = np.zeros((N, max_len), dtype=int)
    states = np.zeros((N, classes), dtype=int)
    for out_idx, src_idx in enumerate(order):
        seq = sequences[src_idx]
        c = labels[src_idx]
        if len(seq) < max_len:
            pad = _random_symbols(max_len - len(seq), rng, alphabet_size=alphabet_size)
            seq = np.concatenate([seq, pad])
        emissions[out_idx] = seq
        states[out_idx, c] = 1

    return emissions, states


if __name__ == "__main__":
    emissions, states = create_hierarchical_data(
        N=20, L=10, classes=2, K=3, D=2, alpha=3.0, seed=42
    )
    print("emissions shape:", emissions.shape)
    print("states shape:", states.shape)
    print("Klassenverteilung:", states.sum(axis=0))
    print(emissions[0])
    print(states[0])