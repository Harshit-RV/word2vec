import json
from collections import Counter
from pathlib import Path

import pytest

from word2vec.vocabulary import Vocabulary

TOKEN_LISTS = [
    [],
    ["only"],
    ["the", "cat", "sat", "the", "cat"],
    ["b", "a", "b", "a", "c", "c", "c"],
    ["The", "the", "THE", "cat.", "cat."],
    ["word"] * 20 + ["rare"] * 2 + ["once"],
    [f"w{i % 5}" for i in range(30)],
]


def kept_words(tokens: list[str], min_count: int) -> list[tuple[str, int]]:
    counts = Counter(tokens)
    kept = [(word, count) for word, count in counts.items() if count >= min_count]
    kept.sort(key=lambda item: -item[1])
    return kept


@pytest.mark.parametrize("tokens", TOKEN_LISTS)
@pytest.mark.parametrize("min_count", [1, 2, 100])
def test_build_for_any_token_list(tokens: list[str], min_count: int):
    vocabulary = Vocabulary().build(list(tokens), min_count)
    kept = kept_words(tokens, min_count)
    counts = Counter(tokens)

    assert vocabulary.word_counts == dict(kept)
    assert vocabulary.id_to_word == [word for word, _ in kept]
    assert vocabulary.size == len(kept) == len(vocabulary)
    assert "<UNK>" not in vocabulary.word_to_id

    for index, (word, count) in enumerate(kept):
        assert vocabulary.get_id(word) == index
        assert vocabulary.get_word(index) == word
        assert vocabulary.word_counts[word] == count

    for word, count in counts.items():
        if count < min_count:
            assert vocabulary.get_id(word) is None


@pytest.mark.parametrize("tokens", TOKEN_LISTS)
def test_add_word_follows_first_seen_order(tokens: list[str]):
    vocabulary = Vocabulary()
    for token in tokens:
        vocabulary.add_word(token)

    assert vocabulary.word_counts == dict(Counter(tokens))
    assert vocabulary.id_to_word == list(dict.fromkeys(tokens))
    assert vocabulary.size == len(set(tokens)) == len(vocabulary)
    for word in vocabulary.id_to_word:
        assert vocabulary.get_word(vocabulary.get_id(word)) == word


@pytest.mark.parametrize("tokens", TOKEN_LISTS)
@pytest.mark.parametrize("min_count", [1, 2])
def test_save_and_load_round_trip(tokens: list[str], min_count: int, tmp_path: Path):
    vocabulary = Vocabulary().build(list(tokens), min_count)
    path = tmp_path / "vocab.json"
    vocabulary.save(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["id_to_word"] == vocabulary.id_to_word
    assert payload["word_counts"] == vocabulary.word_counts

    loaded = Vocabulary.load(path)
    assert loaded.word_to_id == vocabulary.word_to_id
    assert loaded.id_to_word == vocabulary.id_to_word
    assert loaded.word_counts == vocabulary.word_counts
    assert loaded.size == vocabulary.size
