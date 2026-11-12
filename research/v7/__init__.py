"""Scan4Disease V7: cumulative on V1-V6. Data gate first (`research/data/`), then declared builds (`builds.py`), each judged
by pre-registered rules (`promotion.py`, `stats.py`) on development rows only. No module here reads a test, reserved or
HIBA label; the one external read (`external_read.py`) is sealed, receipted and runs once.
"""
