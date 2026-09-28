"""In-memory corpus of sentences as integer ids.

Tokenization follows ``word2vec.c`` ``ReadWord``: split on whitespace, keep
case, and leave punctuation attached to the token. A newline is a sentence
boundary (the C code emits ``</s>`` there). Boundaries are the sentence
lists themselves, not a ``</s>`` id.

Words removed by ``min_count`` are dropped from each sentence. ``SortVocab``
discards them, ``SearchVocab`` returns -1, and ``TrainModelThread`` skips
that read with ``continue``. They are not replaced by ``<UNK>``.
"""

from __future__ import annotations

from pathlib import Path

from word2vec.vocabulary import Vocabulary


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences on newlines."""
    return text.splitlines()


def tokenize(sentence: str) -> list[str]:
    """Split one sentence on whitespace."""
    return sentence.split()


class Corpus:
    def __init__(self, sentences: list[list[int]], vocabulary: Vocabulary) -> None:
        self.sentences = sentences
        self.vocabulary = vocabulary

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        min_count: int = 1,
        vocabulary: Vocabulary | None = None,
    ) -> Corpus:
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_text(text, min_count=min_count, vocabulary=vocabulary)

    @classmethod
    def from_text(
        cls,
        text: str,
        min_count: int = 1,
        vocabulary: Vocabulary | None = None,
    ) -> Corpus:
        tokenized = [tokenize(sentence) for sentence in split_sentences(text)]
        
        if vocabulary is None:
            flat = [token for sentence in tokenized for token in sentence]
            vocabulary = Vocabulary().build(flat, min_count)
        
        id_sentences: list[list[int]] = []
        
        for sentence in tokenized:
            ids: list[int] = []
            
            for token in sentence:
                word_id = vocabulary.get_id(token)
                if word_id is None:
                    continue
                ids.append(word_id)
            
            id_sentences.append(ids)
        
        return cls(id_sentences, vocabulary)

    def statistics(self, top_n: int = 10) -> dict:
        """Summarize the integer corpus.

        ``num_tokens`` counts tokens that remain after rare words are dropped.
        Frequencies are the original counts of words that stayed in the vocabulary.
        """
        ranked = sorted(
            self.vocabulary.word_counts.items(),
            key=lambda item: (-item[1], self.vocabulary.word_to_id[item[0]]),
        )
        return {
            "num_sentences": len(self.sentences),
            "num_tokens": sum(len(sentence) for sentence in self.sentences),
            "vocabulary_size": len(self.vocabulary),
            "most_frequent_words": ranked[:top_n],
        }
