"""One center word's context ids.

Matches the window loop in word2vec.c. For each center, Word2Vec draws
one reduced radius and uses the words inside that radius. The caller
does this while visiting the sentence, instead of storing every pair first.
"""

from __future__ import annotations
from word2vec.utils.random import Word2VecRandom

class ContextGenerator:
    def __init__(self, window_size: int, thread_id: int = 0):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        self.window_size = window_size
        self._rng = Word2VecRandom(thread_id)

    def context_ids(self, sentence: list[int], position: int) -> list[int]:
        """Draw one radius and return the context ids around ``position``."""
        window = self.window_size
        length = len(sentence)
        next_random = self._rng.next()
        
        radius = window - (next_random % window)
        start = max(0, position - radius)
        stop = min(length, position + radius + 1)
        
        context_ids = []

        for index in range(start, stop):
            if index == position:
                continue
            context_ids.append(sentence[index])

        return context_ids