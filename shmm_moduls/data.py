from pathlib import Path

import numpy as np
import tensorflow as tf

from .util import create_HMM_large, create_HMM_XOR_COMBINED_IF, create_random_HMM_order_mix
from .hierarchical_data import create_hierarchical_data


def load_dataset(
    emissions: str | Path, states: str | Path, num_symbols: int = 2
) -> tf.data.Dataset:
    """Args:
        num_symbols: Größe des Emissionsalphabets (One-Hot-Tiefe). Muss zur
            tatsächlichen Kodierung der gespeicherten Emissionen passen.
            Standard 2 (binär) für Rückwärtskompatibilität mit `create_data`
            (feste Beispiel-HMMs, weiterhin binär).
    """
    x = np.load(emissions)
    y = np.load(states)
    x = np.eye(num_symbols)[x]

    df = tf.data.Dataset.from_tensor_slices((x, y))
    df = df.repeat()
    df = df.shuffle(100)
    return df


def create_data(
    N: int,
    T: int,
    alpha: float | None = None,
    return_model: bool = False,
):
    """Create two tensors `(states, emissions)` that serve as a dataset
    to train a model on.

    Args:
        N: Anzahl Sequenzen (Batchgröße der Stichprobe).
        T: Sequenzlänge.
        alpha: Wenn None, wird `create_HMM_large()` (fest, Order-1)
            verwendet. Andernfalls `create_HMM_XOR_COMBINED_IF(alpha)`
            (fest, Order-1/2-Mischung mit vorgegebenem alpha).
        return_model: Wenn True, wird zusätzlich das verwendete
            `TFHMM`-Objekt zurückgegeben - z.B. um es anschließend
            (ohne Training!) auf denselben Daten auszuwerten und als
            Referenz ("was ist mit dem wahren Modell erreichbar")
            neben den trainierten Modellen zu führen.

    Returns:
        (states, emissions) wie bisher, oder bei `return_model=True`
        (states, emissions, hmm).
    """
    if alpha is None:
        hmm = create_HMM_large()
    else:
        hmm = create_HMM_XOR_COMBINED_IF(alpha=alpha)

    states, (emissions,) = hmm.sample(B=N, T=T)
    states = tf.argmax(states[:, :, 0, :], -1)
    emissions = tf.argmax(emissions[:, :, 0, :], -1)

    if return_model:
        return states, emissions, hmm
    return states, emissions


def create_random_data(
    N: int,
    T: int,
    K: int = 3,
    out_degree: int = 2,
    M: int = 4,
    alpha: float | None = None,
    alpha_range: tuple[float, float] = (0.0, 1.0),
    seed: int | None = None,
    return_model: bool = True,
):
    """Wie `create_data`, aber mit einem bei jedem Aufruf frisch
    gezogenen, zufälligen, dünn besetzten, alpha-gemischten HMM
    (siehe `util.create_random_HMM_order_mix`) statt einem der beiden
    fest verdrahteten Beispiel-HMMs.

    Für ~1000 unterschiedliche Experimente einfach 1000 verschiedene
    `seed`-Werte durchlaufen (siehe `auto_train_simple.py`).

    Args:
        N, T: wie bei `create_data`.
        K: Größe des zugrunde liegenden Symbolalphabets (Zustandsraum
            hat K^2 Elemente).
        out_degree: Sparsity-Regler (erlaubte Folgezustände pro
            Zustand).
        M: Größe des Emissionsalphabets (Standard 4, NICHT binär -
            siehe `util.create_random_HMM_order_mix`). Der Aufrufer
            muss denselben Wert für die One-Hot-Kodierung der
            zurückgegebenen `emissions` und beim Modellaufbau
            (`num_symbols` in `training.py`) verwenden.
        alpha: Fester Mischungsparameter, oder None für zufällige Wahl
            aus `alpha_range`.
        alpha_range: Bereich für zufälliges alpha, falls `alpha=None`.
        seed: Seed für sowohl die HMM-Struktur/Parameter als auch,
            sofern `alpha=None`, die Ziehung von alpha selbst.
        return_model: Wie bei `create_data`; hier standardmäßig True,
            da das HMM bei zufälliger Erzeugung i.d.R. sofort für die
            Referenzauswertung gebraucht wird.

    Returns:
        (states, emissions, alpha_used) oder, falls `return_model`
        zusätzlich gesetzt bleibt (Standard), (states, emissions, hmm,
        alpha_used).
    """
    hmm, alpha_used = create_random_HMM_order_mix(
        K=K, out_degree=out_degree, M=M, alpha=alpha, alpha_range=alpha_range, seed=seed,
    )

    states, (emissions,) = hmm.sample(B=N, T=T)
    states = tf.argmax(states[:, :, 0, :], -1)
    emissions = tf.argmax(emissions[:, :, 0, :], -1)

    if return_model:
        return states, emissions, hmm, alpha_used
    return states, emissions, alpha_used