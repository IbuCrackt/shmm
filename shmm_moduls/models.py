"""models.py - Modell-Konfigurationen für den Modellvergleich (Kapitel 4
der Bachelorarbeit: HMM erster/höherer Ordnung, mehrschichtige HMM-Stapel,
RNN/LSTM-Varianten mit und ohne HMM, Transformer, MLP-Baseline).

Jede Config in MODELS wird an `rescrf.get_model(...)` durchgereicht. Das
Feld "output" (Anzahl Ausgabeklassen bzw. -zustände) hängt vom jeweiligen
Durchlauf ab - Anzahl HMM-Zustände (Zustandsdekodierung) bzw. Anzahl
Klassen (Klassifikation), siehe Kapitel 3 - und wird deshalb NICHT hier
festgelegt, sondern von run_token_level.py / run_classification.py vor
jedem Training in eine Kopie der jeweiligen Config eingesetzt.
"""
from typing import Literal

BASE_LATENT = 32
BASE_STATES = 5


def _hmm(embed=16, states=BASE_STATES, order=1, heads=3, embed_ignore_order=False, latent=None):
    cfg = {
        "embed": embed,
        "states": states,
        "order": order,
        "heads": heads,
        "embedIgnoreOrder": embed_ignore_order,
    }
    if latent is not None:
        cfg["latent"] = latent
    return cfg


def _mlp(units=9):
    return {"units": units}


def _rnn(type="lstm", units=32, bidirectional=False):
    return {"type": type, "units": units, "bidirectional": bidirectional}


def _mlp_only(units=32, layers=1, latent=BASE_LATENT, activation_hidden="relu"):
    """Config für reines MLP-Baseline-Modell (kein HMM/RNN/Transformer)."""
    return {"layers": layers, "latent": latent, "units": units, "activation_hidden": activation_hidden}


def _transformer(d_model=64, num_heads=4, num_layers=2, dff=256, dropout=0.1):
    return {"d_model": d_model, "num_heads": num_heads, "num_layers": num_layers, "dff": dff, "dropout": dropout}


MODELS = {
    "mlp_baseline": {
        "mlp_only": _mlp_only(units=32, layers=1),
    },
    "single_hmm_order1": {
        "layers": 1, "latent": BASE_LATENT, "hmm": _hmm(order=1),
    },
    "single_hmm_order1_mlp": {
        "layers": 1, "latent": BASE_LATENT, "hmm": _hmm(order=1), "mlp": _mlp(units=9),
    },
    "single_hmm_order2": {
        "layers": 1, "latent": BASE_LATENT, "hmm": _hmm(order=2),
    },
    "single_hmm_order3": {
        "layers": 1, "latent": BASE_LATENT, "hmm": _hmm(order=3),
    },
    "multilayer_hmm_2x_order1": {
        "layers": 2, "latent": BASE_LATENT, "hmm": _hmm(order=1),
    },
    "multilayer_hmm_3x_order1": {
        "layers": 3, "latent": BASE_LATENT, "hmm": _hmm(order=1),
    },
    "multilayer_hmm_2x_order1_mlp": {
        "layers": 2, "latent": BASE_LATENT, "hmm": _hmm(order=1), "mlp": _mlp(units=9),
    },
    "multilayer_hmm_mixed_order": {
        "layers": 2, "latent": BASE_LATENT,
        "hmm": [_hmm(order=1, latent=BASE_LATENT), _hmm(order=2, latent=BASE_LATENT)],
    },
    "lstm_only": {
        "layers": 0, "latent": BASE_LATENT, "rnn": _rnn("lstm", units=16), "hmm": _hmm(),
    },
    "gru_only": {
        "layers": 0, "latent": BASE_LATENT, "rnn": _rnn("gru", units=16), "hmm": _hmm(),
    },
    "rnn_only": {
        "layers": 0, "latent": BASE_LATENT, "rnn": _rnn("rnn", units=16), "hmm": _hmm(),
    },
    "lstm_hmm_order1": {
        "layers": 1, "latent": BASE_LATENT, "rnn": _rnn("lstm", units=16), "hmm": _hmm(order=1),
    },
    "lstm_hmm_order2": {
        "layers": 1, "latent": BASE_LATENT, "rnn": _rnn("lstm", units=16), "hmm": _hmm(order=2),
    },
    "gru_hmm_order1": {
        "layers": 1, "latent": BASE_LATENT, "rnn": _rnn("gru", units=16), "hmm": _hmm(order=1),
    },
    "bilstm_hmm_order1": {
        "layers": 1, "latent": BASE_LATENT,
        "rnn": _rnn("lstm", units=16, bidirectional=True), "hmm": _hmm(order=1),
    },
    "lstm_multilayer_hmm_2x": {
        "layers": 2, "latent": BASE_LATENT, "rnn": _rnn("lstm", units=16), "hmm": _hmm(order=1),
    },
    "lstm_multilayer_hmm_mlp": {
        "layers": 2, "latent": BASE_LATENT, "rnn": _rnn("lstm", units=16),
        "hmm": _hmm(order=1), "mlp": _mlp(units=9),
    },
    "transformer_small": {
        "transformer": _transformer(d_model=32, num_heads=2, num_layers=2),
    },
    "transformer_base": {
        "transformer": _transformer(d_model=64, num_heads=4, num_layers=2),
    },
    "transformer_large": {
        "transformer": _transformer(d_model=128, num_heads=8, num_layers=4),
    },
}

# Benannte Teilmengen von MODELS. "base"/"base_class" sind die in
# params.ALL_MODELS=False verwendete, reduzierte Modellbatterie für
# Zustandsdekodierung bzw. Klassifikation; "full" ist die vollständige
# Batterie (params.ALL_MODELS=True). Es gibt bewusst keine dritte,
# kleinere "test"-Teilmenge mehr - params.TEST steuert stattdessen die
# Datengröße/Wiederholungszahl, nicht die Modellauswahl (siehe params.py).
EXPERIMENTS = {
    "base": [
        "single_hmm_order1",
        "single_hmm_order2",
        "multilayer_hmm_2x_order1",
        "multilayer_hmm_2x_order1_mlp",
        "bilstm_hmm_order1",
        "transformer_small",
    ],
    "base_class": [
        "mlp_baseline",
        "single_hmm_order1",
        "single_hmm_order2",
        "multilayer_hmm_2x_order1",
        "multilayer_hmm_2x_order1_mlp",
        "bilstm_hmm_order1",
        "transformer_small",
    ],
    "full": list(MODELS),
}


def get_experiments(experiment: Literal["base", "base_class", "full"]) -> dict:
    """Gibt die Modell-Configs für einen der drei Sätze aus EXPERIMENTS zurück."""
    return {name: MODELS[name] for name in EXPERIMENTS[experiment]}
