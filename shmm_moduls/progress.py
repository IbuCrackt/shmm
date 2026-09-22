"""progress.py - Dünner Wrapper um ibutils.LoadBar.

ibutils ist nur auf dem HPC verfügbar (siehe Modul-Docstring von
run_token_level.py/run_classification.py). Außerhalb davon (z.B. lokale
Tests) bleiben beide Funktionen No-Ops, damit dieselben Skripte auch ohne
ibutils lauffähig sind.

Vorher war dieser Code identisch in auto_train_simple.py und
auto_train_classification_simple.py dupliziert - jetzt eine gemeinsame
Stelle.
"""
try:
    import ibutils
except ImportError:
    ibutils = None


def make_load_bar(maximum: int, title: str):
    if ibutils is None:
        return None
    bar = ibutils.LoadBar(maximum, auto_time_estimation=True)
    bar.set_title(title)
    return bar


def tick(load_bar, amount: int = 1) -> None:
    if load_bar is not None:
        load_bar.add_progress(amount)
