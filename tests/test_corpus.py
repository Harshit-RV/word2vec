from collections import Counter
from pathlib import Path

import pytest

from word2vec.corpus import Corpus, split_sentences, tokenize

ROOT = Path(__file__).resolve().parents[1]

TEXTS = [
    "",
    "cat\n",
    "The cat sat on the mat.\n",
    "The cat sat on the mat.\nThe cat sat on the rug.\nA zebra appears once!\n",
    "Cat\ncat\nCAT\n",
    "a b c\n\nd e\n",
    "\n".join(f"word{i % 7} extra{i} end." for i in range(40)) + "\n",
]


def raw_corpus_paths() -> list[Path]:
    paths: list[Path] = []
    for folder in (ROOT / "data" / "raw", ROOT / "src" / "data" / "raw"):
        if folder.is_dir():
            paths.extend(sorted(folder.glob("*.txt")))
    return paths


def token_counts(text: str) -> Counter:
    return Counter(token for line in text.splitlines() for token in line.split())


def first_seen(tokens: list[str]) -> dict[str, int]:
    index: dict[str, int] = {}
    for position, token in enumerate(tokens):
        index.setdefault(token, position)
    return index


def assert_corpus(text: str, corpus: Corpus, min_count: int) -> None:
    lines = text.splitlines()
    counts = token_counts(text)
    kept_counts = {word: count for word, count in counts.items() if count >= min_count}
    dropped = {word for word, count in counts.items() if count < min_count}

    assert split_sentences(text) == lines
    assert len(corpus.sentences) == len(lines)
    assert corpus.vocabulary.word_counts == kept_counts
    assert corpus.vocabulary.size == len(kept_counts) == len(corpus.vocabulary)
    assert "<UNK>" not in corpus.vocabulary.word_to_id
    assert set(corpus.vocabulary.id_to_word) == set(kept_counts)

    for word in dropped:
        assert corpus.vocabulary.get_id(word) is None

    seen = first_seen([token for line in lines for token in line.split()])
    ordered = corpus.vocabulary.id_to_word
    for earlier, later in zip(ordered, ordered[1:]):
        assert kept_counts[earlier] >= kept_counts[later]
        if kept_counts[earlier] == kept_counts[later]:
            assert seen[earlier] < seen[later]

    for line, word_ids in zip(lines, corpus.sentences):
        tokens = tokenize(line)
        assert tokens == line.split()
        expected = [token for token in tokens if counts[token] >= min_count]
        assert [corpus.vocabulary.get_word(word_id) for word_id in word_ids] == expected
        assert word_ids == [corpus.vocabulary.get_id(token) for token in expected]

    variants: dict[str, set[str]] = {}
    for token in counts:
        variants.setdefault(token.lower(), set()).add(token)
    for forms in variants.values():
        present = [form for form in forms if form in kept_counts]
        assert len({corpus.vocabulary.get_id(form) for form in present}) == len(present)

    stats = corpus.statistics()
    ranked = sorted(kept_counts.items(), key=lambda item: (-item[1], corpus.vocabulary.word_to_id[item[0]]))
    assert stats["num_sentences"] == len(lines)
    assert stats["num_tokens"] == sum(len(sentence) for sentence in corpus.sentences)
    assert stats["num_tokens"] == sum(kept_counts.values())
    assert stats["vocabulary_size"] == len(kept_counts)
    assert stats["most_frequent_words"] == ranked[:10]


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("min_count", [1, 2, 100])
def test_any_text_and_length(text: str, min_count: int, tmp_path: Path):
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")

    from_text = Corpus.from_text(text, min_count=min_count)
    from_file = Corpus.from_file(path, min_count=min_count)

    assert_corpus(text, from_text, min_count)
    assert from_file.sentences == from_text.sentences
    assert from_file.vocabulary.word_to_id == from_text.vocabulary.word_to_id
    assert from_file.vocabulary.word_counts == from_text.vocabulary.word_counts


@pytest.mark.parametrize("path", raw_corpus_paths())
@pytest.mark.parametrize("min_count", [1, 2])
def test_raw_corpus_file(path: Path, min_count: int):
    text = path.read_text(encoding="utf-8")
    assert_corpus(text, Corpus.from_file(path, min_count=min_count), min_count)
