"""Context pairs for one sentence at a time.

Matches the window loop in word2vec.c.

For each center word, Word2Vec generates a pseudo-random value,
uses it to choose a reduced window radius, and produces
(center, context) pairs from that window.

Sentence boundaries are preserved: pairs never cross into
the next sentence.
"""

from __future__ import annotations

from word2vec.utils.random import Word2VecRandom


class ContextGenerator:
    def __init__(self, window_size: int, thread_id: int = 0) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        self.window_size = window_size
        self._rng = Word2VecRandom(thread_id)

    def pairs(self, sentences: list[list[int]]) -> list[tuple[int, int]]:
        """Return (center, context) IDs, sentence by sentence, left to right."""
        window = self.window_size
        pairs: list[tuple[int, int]] = []

        for sentence in sentences:
            length = len(sentence)

            for position, center in enumerate(sentence):
                next_random = self._rng.next()
                radius = window - (next_random % window)

                start = max(0, position - radius)
                stop = min(length, position + radius + 1)

                for context_index in range(start, stop):
                    if context_index == position:
                        continue

                    pairs.append((center, sentence[context_index]))

        return pairs