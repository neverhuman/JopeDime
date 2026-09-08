"""Learned fitness corrector: calibrate GP proxy fitness to honest-CV fitness.

Study package. The deployable surface is `scorer.py` (numpy-only, JSON
artifact, selector_reranker.py pattern); everything else is harvest/eval
machinery that runs against frozen copies of production artifacts.
"""

SCHEMA_VERSION = "fitness_corrector_v1"
REPORT_SCHEMA = "jope-prime-fitness-corrector-report-v1"
STUDY_SEED = 20260714
