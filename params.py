"""params.py - Alle einstellbaren Parameter an einem Ort.

Ändere hier, nicht in run_token_level.py / run_classification.py.

TEST = True:  kleine, schnelle Werte zum lokalen Ausprobieren (Sekunden
    bis wenige Minuten - kein HPC, kein ibutils nötig).
TEST = False: die tatsächlich für die Bachelorarbeit verwendeten Werte
    (siehe Kapitel 3 "Daten" und Kapitel 4 "Trainings- und
    Auswertungsverfahren"). Ein voller Lauf ist für den HPC gedacht und
    dauert - je nach Cluster - Stunden bis Tage.
"""
from pathlib import Path

TEST = True

# False: reduzierte Modellbatterie (models.EXPERIMENTS["base"/"base_class"]).
# True: alle Modelle (models.EXPERIMENTS["full"]).
ALL_MODELS = False

# --- Trainingsverfahren, gemeinsam für beide Aufgaben (Abschnitt 4.2) ---
WEIGHT_DECAY = 1e-2
CONVERGENCE_THRESHOLD = 1e-4
CONVERGENCE_PATIENCE = 5
BEST_VALUE_PATIENCE = 5
MAX_EPOCHS = 5 if TEST else 100
STEPS_PER_EPOCH = 10 if TEST else 100
# zwei feste Lernraten, siehe Abschnitt 4.2 ("Trainingsverfahren")
LEARNING_RATES = [1e-1] if TEST else [1e-2, 1e-3]
SAVE_CHECKPOINTS = False

# --- Zustandsdekodierung: zufällige gemischte HMM (Abschnitt 3.1) ------
TOKENLEVEL_T = 20 if TEST else 100                # Sequenzlänge
TOKENLEVEL_BATCH_SIZE = 8 if TEST else 64         # = N: Sequenzen pro HMM UND Trainings-Batchgröße
TOKENLEVEL_NUM_HMMS = 2 if TEST else 1000         # Anzahl unabhängiger Durchläufe
TOKENLEVEL_HMM_K = 3 if TEST else 5               # Basisalphabet, Zustandsraum = K^2
TOKENLEVEL_HMM_OUT_DEGREE = 2                     # Sparsity (vgl. Lafferty et al. 2001, Abschn. 5.2)
TOKENLEVEL_HMM_M = 26                              # Emissionsalphabet (NICHT binär, Def. 3.1)
TOKENLEVEL_HMM_ALPHA_RANGE = (0.0, 1.0)
TOKENLEVEL_HMM_SEED_START = 20260921
TOKENLEVEL_RESULTS_PATH = Path("./results_token_level")
# Modelle, die im alpha-Sweep-Diagramm gegenübergestellt werden
# (Loss/Accuracy vs. alpha, analog zu Lafferty et al. 2001, Abschn. 5.2)
TOKENLEVEL_ALPHA_SWEEP_MODELS = [
    "true_generator", "single_hmm_order1", "single_hmm_order2",
    "multilayer_hmm_2x_order1", "bilstm_hmm_order1", "transformer_small",
]

# --- Sequenzklassifikation: hierarchisch komponierte Muster (Abschnitt 3.2) --
CLASS_N_SAMPLES = 50 if TEST else 8192             # N (Def. 3.2 "Konkrete Parametrisierung")
CLASS_BATCH_SIZE = 8 if TEST else 64
CLASS_HIERARCHY_LEVELS = [1] if TEST else [1, 2, 3]   # D
CLASS_PATTERNS_PER_LEVEL = [3]                         # K
CLASS_BASE_PATTERN_LENGTH = [4] if TEST else [8]       # L
CLASS_NUM_CLASSES = [2] if TEST else [4]             # C
CLASS_ALPHABET_SIZE = 2 if TEST else 4                  # M (NICHT binär)
CLASS_NOISE_ALPHA = 3.0                                  # Beta(alpha, alpha) für Rauschlänge
CLASS_NUM_RUNS = 2 if TEST else 1000                     # Wiederholungen je Parameterkombination
CLASS_RUN_SEED_START = 20260921
CLASS_RESULTS_PATH = Path("./results_classification")
