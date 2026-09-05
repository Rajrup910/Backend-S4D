"""A process-wide lock on the test split, so "no test read" is enforced, not promised.

Hard Rule 2 says the test set is touched once per experiment. Until session 6 that was
kept by reading the code: each runner fitted on val and then read test exactly once, at
the end, and the discipline lived in the reviewer's head.

That stops being sufficient here. Session 4 (S4) adds an OOF-fitted variant of four
runners and a comparison driver that invokes each of them twice in one process, and the
frozen-analysis-plan work (S9) requires that *nothing* reads test until the single
pre-registered test pass. A driver that accidentally triggered a test read would not
fail loudly -- it would quietly produce a number, and every downstream selection made
afterwards would be contaminated in a way no artifact records.

So the lock is mechanical. `block_test_reads()` arms it; the two functions that can
actually pull test data off disk -- `research.ensembling.data.load_split_matrix` and
`research.selective.features.load_features` -- call `check_split()` before doing so and
raise `TestSplitLocked` if the lock is armed. It is deliberately a hard error rather than
a warning: a warning in a 4-runner batch scrolls past.

    from research import testguard
    testguard.block_test_reads("S4 fit-only mode")
    ...                                   # any test read now raises
    with testguard.test_unlocked("S9 pre-registered single pass"):
        test = load_split_matrix("test")  # the one sanctioned read

The lock is per-process and defaults to *off*, so every existing script keeps its current
behaviour with no changes and no import of this module.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

TEST_SPLIT = "test"

_blocked: bool = False
_reason: str = ""


class TestSplitLocked(RuntimeError):
    """Raised when code attempts to read the test split while the lock is armed."""


def block_test_reads(reason: str) -> None:
    """Arm the lock. `reason` is echoed in the exception so the caller is identifiable."""
    global _blocked, _reason
    _blocked = True
    _reason = reason


def allow_test_reads() -> None:
    """Disarm the lock. Call sites should prefer `test_unlocked` so it cannot be left off."""
    global _blocked, _reason
    _blocked = False
    _reason = ""


def is_blocked() -> bool:
    return _blocked


def reason() -> str:
    return _reason


@contextmanager
def test_unlocked(purpose: str) -> Iterator[None]:
    """Temporarily lift the lock for one sanctioned read, restoring it afterwards."""
    global _blocked, _reason
    previous_blocked, previous_reason = _blocked, _reason
    _blocked, _reason = False, ""
    try:
        yield
    finally:
        _blocked, _reason = previous_blocked, previous_reason
    del purpose  # documentation only; kept in the signature so call sites must state one


def check_split(split: str, source: str) -> None:
    """Raise if `split` is test and the lock is armed. Cheap enough to call unconditionally."""
    if _blocked and split == TEST_SPLIT:
        raise TestSplitLocked(
            f"{source} attempted to read the '{TEST_SPLIT}' split while test reads are "
            f"locked ({_reason}). Under Hard Rule 2 the test split is read once, in the "
            f"pre-registered pass. If this read is that pass, wrap it in "
            f"`research.testguard.test_unlocked(...)`."
        )
