import numpy as np
from hidten.tf import TFHMM

"""
A, B, C
1 - AB
2 - AC
3 - BA

5 - BC
6 - CA
7 - CB
"""


def create_HMM_large(restrict: bool = True, initialize: bool = True) -> TFHMM:
    """Returns a hidten HMM with set intializations for transitioner and
    emitter.
    """
    hmm = TFHMM(states=9)

    if restrict:
        hmm.transitioner.allow = [
            (0, 1),
            (1, 3), (1, 5),
            (2, 6), (2, 7),
            (3, 1), (3, 2),
            (5, 6), (5, 7),
            (6, 1), (6, 2),
            (7, 3), (7, 5),
        ]
        hmm.transitioner.allow_start = [0]

    if initialize:
        hmm.transitioner.initializer = np.array([
            1.0,
            0.7, 0.3,
            0.2, 0.8,
            0.1, 0.9,
            1.0, 0.0,
            0.5, 0.5,
            1.0, 0.0,
        ])

        hmm.emitter[0].initializer = [
            1.0, 0.0,
            0.25, 0.75,
            0.1, 0.9,
            0.75, 0.25,
            0.5, 0.5,
            0.25, 0.75,
            0.9, 0.1,
            0.75, 0.25,
            0.5, 0.5,
        ]

    hmm.build((2, ))

    return hmm


def create_HMM_XOR_COMBINED_IF(
    restrict: bool = True,
    initialize: bool = True,
    alpha: float = 0.0
) -> TFHMM:
    """Returns a second-order HMM represented as a 9-state first-order HMM.

    State i represents a pair (s_{t-1}, s_t)
    """

    a = alpha
    b = 1.0 - alpha

    hmm = TFHMM(states=9)

    if restrict:
        hmm.transitioner.allow = [
            (0, 1),
            (1, 4), (1, 5),
            (2, 7), (2, 8),
            (4, 4), (4, 5),
            (5, 7), (5, 8),
            (7, 4), (7, 5),
            (8, 7), (8, 8),
        ]

        hmm.transitioner.allow_start = [0]

    if initialize:
        hmm.transitioner.initializer = np.array([
            1.0,
            0.5, 0.5,
            0.5, 0.5,
            0.5, 0.5,
            0.5, 0.5,
            0.5, 0.5,
            0.5, 0.5,
        ])

        hmm.emitter[0].initializer = [
            b, a,  # (0,0)
            a, b,  # (0,1)
            a, b,  # (0,2)
            a, b,  # (1,0)
            b, a,  # (1,1)
            a, b,  # (1,2)
            a, b,  # (2,0)
            a, b,  # (2,1)
            b, a,  # (2,2)
        ]

    hmm.build((2,))

    return hmm


def create_random_HMM_order_mix(
    K: int = 3,
    out_degree: int = 2,
    M: int = 4,
    alpha: float | None = None,
    alpha_range: tuple[float, float] = (0.0, 1.0),
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    restrict: bool = True,
    initialize: bool = True,
) -> tuple[TFHMM, float]:
    """Zufälliges, dünn besetztes HMM, das stufenlos zwischen einem
    Order-1- und einem Order-2-Prozess über einem K-wertigen Alphabet
    interpoliert - eine randomisierte Verallgemeinerung von
    `create_HMM_XOR_COMBINED_IF` (siehe Bachelorarbeit, Def. 3.1
    "Zufälliges gemischtes HMM").

    Konstruktion (Zustand = Paar der letzten beiden "wahren" Symbole,
    analog zu `create_HMM_XOR_COMBINED_IF`):
        - N = K^2 Zustände, Zustand `sid(prev, curr) = prev*K + curr`
          repräsentiert das Symbolpaar (s_{t-1}, s_t).
        - Von jedem Zustand (prev, curr) sind nur `out_degree` von K
          möglichen Folgezuständen (curr, next) erlaubt (zufällig
          gewählt, mit Dirichlet-verteilten Übergangswahrscheinlichkeiten).
          Diese Sparsity ist an Lafferty, McCallum & Pereira (2001,
          Abschnitt 5.2) angelehnt, die ihre Übergangs-/Emissionstabellen
          aus demselben Grund dünn besetzen: um den Bayes-Fehler des
          resultierenden Modells zu begrenzen ("In order to limit the
          size of the Bayes error rate for the resulting models, the
          conditional probability tables p_alpha are constrained to be
          sparse") - NICHT, wie in einer früheren Fassung dieses
          Kommentars fälschlich behauptet, um Label-Bias zu vermeiden
          (das ist ein separates Experiment in Abschnitt 5.1 desselben
          Papers).
        - Die Emission (M mögliche Symbole - standardmäßig NICHT binär,
          analog zum synthetischen Vergleichsexperiment bei Lafferty et
          al. 2001, Abschnitt 5.2) mischt zwei Zielverteilungen konvex
          mit `alpha`:
              e(prev, curr) = (1-alpha) * e1(curr) + alpha * e2(prev, curr)
          wobei `e1(curr)` NUR von `curr` abhängt (bei alpha=0 ist der
          beobachtbare Prozess dadurch trotz K^2 Zuständen ein reiner
          Order-1-Prozess) und `e2(prev, curr)` echt vom vollen Paar
          abhängt (bei alpha=1 ein echter Order-2-Prozess). Beides sind
          bei jedem Aufruf zufällig gezogene Dirichlet-Verteilungen über
          den M Emissionssymbolen.

    Args:
        K: Größe des zugrunde liegenden "wahren" Symbolalphabets
            (Zustandsraum hat K^2 Elemente).
        out_degree: Anzahl erlaubter Folgezustände pro Zustand
            (Sparsity-Regler). Wird auf `min(out_degree, K)` begrenzt.
        M: Größe des Emissionsalphabets (Anzahl beobachtbarer Symbole).
            Standard 4 statt binär - der Aufrufer MUSS denselben Wert
            konsistent für die One-Hot-Kodierung der erzeugten Daten
            verwenden (siehe `data.create_random_data` als `M`, und
            `num_symbols` in `training.py`).
        alpha: Mischungsparameter in [0, 1]. Wenn None, wird alpha
            zufällig gleichverteilt aus `alpha_range` gezogen (siehe
            Rückgabewert).
        alpha_range: Bereich, aus dem alpha gezogen wird, falls `alpha`
            nicht explizit gesetzt ist.
        rng: Optionaler `numpy.random.Generator`. Hat Vorrang vor `seed`.
        seed: Wird nur verwendet, wenn `rng` nicht gesetzt ist.
        restrict / initialize: wie bei den übrigen Funktionen in diesem
            Modul.

    Returns:
        (hmm, alpha_used): das gebaute TFHMM sowie das tatsächlich
        verwendete alpha (nützlich, wenn `alpha=None` war und der Wert
        zufällig gezogen wurde - z.B. zum Loggen/Tagging des
        Experiments).
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    if alpha is None:
        lo, hi = alpha_range
        alpha = float(rng.uniform(lo, hi))
    a = float(alpha)

    N = K * K
    deg = min(out_degree, K)

    def sid(prev: int, curr: int) -> int:
        return prev * K + curr

    hmm = TFHMM(states=N)

    if restrict or initialize:
        allow: list[tuple[int, int]] = []
        trans_probs: list[float] = []
        for prev in range(K):
            for curr in range(K):
                src = sid(prev, curr)
                next_syms = rng.choice(K, size=deg, replace=False)
                probs = rng.dirichlet(np.ones(deg))
                for nxt, p in zip(next_syms, probs):
                    allow.append((src, sid(curr, int(nxt))))
                    trans_probs.append(float(p))

    if restrict:
        hmm.transitioner.allow = allow
        # Start in einem beliebigen (aber festen) Paar-Zustand, analog
        # zu den bestehenden Beispielen (allow_start = [0]).
        hmm.transitioner.allow_start = [sid(0, 0)]

    if initialize:
        hmm.transitioner.initializer = np.array(trans_probs)

        # Order-1-Zielverteilung: hängt nur von `curr` ab (K Stück).
        order1_target = rng.dirichlet(np.ones(M), size=K)          # [K, M]
        # Order-2-Zielverteilung: hängt vom vollen Paar ab (K*K Stück).
        order2_target = rng.dirichlet(np.ones(M), size=(K, K))     # [K, K, M]

        emit = np.zeros((N, M))
        for prev in range(K):
            for curr in range(K):
                emit[sid(prev, curr)] = (
                    (1.0 - a) * order1_target[curr] + a * order2_target[prev, curr]
                )
        hmm.emitter[0].initializer = emit.reshape(-1).tolist()

    hmm.build((M,))

    return hmm, a
