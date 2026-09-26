"""ControlRoom qualification benchmark: a versioned synthetic corpus, gold cases, an HTTP-driven
runner and a frozen quality gate. See ``examples/controlroom_qualification/README.md``."""

from .corpus import DATASET_VERSION, build_corpus, corpus_manifest, write_corpus
from .gate import evaluate_gate, load_gate
from .gold import CASES, Case, Scope
from .runner import run_qualification

__all__ = [
    "CASES",
    "Case",
    "DATASET_VERSION",
    "Scope",
    "build_corpus",
    "corpus_manifest",
    "evaluate_gate",
    "load_gate",
    "run_qualification",
    "write_corpus",
]
