"""Word to integer-id vocabulary.

IDs follow the original word2vec sort: higher frequency first. Ties keep
first-seen order so the mapping is stable. Words below ``min_count`` are
omitted entirely. ``word2vec.c`` does the same in ``SortVocab`` and never
introduces an ``<UNK>`` token; ``SearchVocab`` returns -1 for them.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class Vocabulary:
    def __init__(self) -> None:
        self.word_to_id: dict[str, int] = {}
        self.id_to_word: list[str] = []
        self.word_counts: dict[str, int] = {}

    @property
    def size(self) -> int:
        return len(self.id_to_word)

    def __len__(self) -> int:
        return self.size

    def add_word(self, word: str) -> int:
        """Add ``word`` or increment its count. Returns its id."""
        word_id = self.word_to_id.get(word)
        if word_id is None:
            word_id = len(self.id_to_word)
            self.word_to_id[word] = word_id
            self.id_to_word.append(word)
            self.word_counts[word] = 0
        self.word_counts[word] += 1
        return word_id

    def build(self, tokens: list[str], min_count: int) -> Vocabulary:
        """Count tokens, drop words below ``min_count``, and assign ids.

        Replaces any vocabulary already stored on this instance.
        """
        counts = Counter(tokens)
        kept = [(word, count) for word, count in counts.items() if count >= min_count]
        kept.sort(key=lambda item: -item[1])

        self.word_to_id = {}
        self.id_to_word = []
        self.word_counts = {}
        for word, count in kept:
            word_id = len(self.id_to_word)
            self.word_to_id[word] = word_id
            self.id_to_word.append(word)
            self.word_counts[word] = count
        return self

    def get_id(self, word: str) -> int | None:
        """Return the id for ``word``, or None if it is not in the vocabulary."""
        return self.word_to_id.get(word)

    def get_word(self, word_id: int) -> str:
        return self.id_to_word[word_id]

    def save(self, path: str | Path) -> None:
        payload = {
            "id_to_word": self.id_to_word,
            "word_counts": self.word_counts,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Vocabulary:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vocabulary = cls()
        vocabulary.id_to_word = list(data["id_to_word"])
        vocabulary.word_counts = {word: int(count) for word, count in data["word_counts"].items()}
        vocabulary.word_to_id = {word: index for index, word in enumerate(vocabulary.id_to_word)}
        if set(vocabulary.word_counts) != set(vocabulary.id_to_word):
            raise ValueError("word_counts keys do not match id_to_word")
        return vocabulary
