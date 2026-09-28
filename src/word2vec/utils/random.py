"""Word2Vec's pseudo-random number generator.

Matches the next_random update used in word2vec.c.
"""

from __future__ import annotations

_MASK = (1 << 64) - 1
_MULTIPLIER = 25214903917
_INCREMENT = 11


class Word2VecRandom:
    """Reproduces Word2Vec's next_random state for one thread."""

    def __init__(self, thread_id: int = 0) -> None:
        self._state = thread_id & _MASK

    def next(self) -> int:
        """Advance the state and return the next random value."""
        
        self._state = (self._state * _MULTIPLIER + _INCREMENT) & _MASK
        
        return self._state