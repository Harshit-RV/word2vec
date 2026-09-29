import pytest

from word2vec.training.context import ContextGenerator

SENTENCE = [[10, 20, 30, 40, 50, 60, 70]]


def collect_pairs(generator: ContextGenerator, sentences: list[list[int]]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for sentence in sentences:
        for position, center in enumerate(sentence):
            for context_id in generator.context_ids(sentence, position):
                pairs.append((center, context_id))
    return pairs


def original_pairs(sentences: list[list[int]], window: int, thread_id: int) -> list[tuple[int, int]]:
    """Direct transcription of the window loop in word2vec.c."""
    state = thread_id & ((1 << 64) - 1)
    pairs: list[tuple[int, int]] = []
    for sentence in sentences:
        length = len(sentence)
        for position, center in enumerate(sentence):
            state = (state * 25214903917 + 11) & ((1 << 64) - 1)
            b = state % window
            for a in range(b, window * 2 + 1 - b):
                if a == window:
                    continue
                context_index = position - window + a
                if context_index < 0 or context_index >= length:
                    continue
                pairs.append((center, sentence[context_index]))
    return pairs


def test_window_size_one_on_a_seven_word_sentence():
    pairs = collect_pairs(ContextGenerator(window_size=1, thread_id=42), SENTENCE)
    assert pairs == [
        (10, 20),
        (20, 10),
        (20, 30),
        (30, 20),
        (30, 40),
        (40, 30),
        (40, 50),
        (50, 40),
        (50, 60),
        (60, 50),
        (60, 70),
        (70, 60),
    ]


def test_window_size_two_matches_original_loop():
    pairs = collect_pairs(ContextGenerator(window_size=2, thread_id=42), SENTENCE)
    assert pairs == original_pairs(SENTENCE, window=2, thread_id=42)
    for center, context in pairs:
        distance = abs(SENTENCE[0].index(center) - SENTENCE[0].index(context))
        assert 1 <= distance <= 2


def test_edges_and_window_larger_than_sentence():
    sentence = [[1, 2, 3]]
    pairs = collect_pairs(ContextGenerator(window_size=10, thread_id=1), sentence)
    assert pairs == original_pairs(sentence, window=10, thread_id=1)
    assert all(context in sentence[0] for _, context in pairs)
    assert (1, 2) in pairs
    assert (3, 2) in pairs


def test_one_word_sentence_has_no_pairs():
    assert collect_pairs(ContextGenerator(window_size=5, thread_id=0), [[9]]) == []


def test_multiple_sentences_do_not_cross_boundaries():
    sentences = [[1, 2, 3], [], [4, 5]]
    pairs = collect_pairs(ContextGenerator(window_size=2, thread_id=0), sentences)
    assert pairs == original_pairs(sentences, window=2, thread_id=0)
    centers = {center for center, _ in pairs}
    assert centers == {1, 2, 3, 4, 5}
    assert (3, 4) not in pairs
    assert (4, 3) not in pairs
    assert (2, 4) not in pairs


def test_duplicate_pairs_only_when_the_sentence_repeats_a_word():
    unique = collect_pairs(ContextGenerator(window_size=1, thread_id=0), [[1, 2, 3]])
    assert len(unique) == len(set(unique))
    repeated = collect_pairs(ContextGenerator(window_size=1, thread_id=0), [[8, 8]])
    assert repeated == [(8, 8), (8, 8)]


def test_dynamic_window_changes_how_many_contexts_a_center_gets():
    window = 4
    sentence = [list(range(12))]
    first = collect_pairs(ContextGenerator(window_size=window, thread_id=42), sentence)
    second = collect_pairs(ContextGenerator(window_size=window, thread_id=42), sentence)
    other = collect_pairs(ContextGenerator(window_size=window, thread_id=7), sentence)
    assert first == second
    assert first != other

    interior = {4, 5, 6}
    counts = {word: sum(center == word for center, _ in first) for word in interior}
    assert len(set(counts.values())) > 1
    for count in counts.values():
        assert count % 2 == 0
        assert 2 <= count <= window * 2


def test_same_generator_draws_a_new_window_each_pass():
    generator = ContextGenerator(window_size=4, thread_id=42)
    sentence = [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
    assert collect_pairs(generator, sentence) != collect_pairs(generator, sentence)


def test_default_thread_id_is_zero():
    sentence = [[0, 1, 2, 3, 4, 5]]
    assert collect_pairs(ContextGenerator(window_size=3), sentence) == collect_pairs(
        ContextGenerator(window_size=3, thread_id=0), sentence
    )


@pytest.mark.parametrize("window", [1, 2, 5])
@pytest.mark.parametrize("thread_id", [0, 1, 42])
def test_matches_word2vec_c_loop(window: int, thread_id: int):
    sentences = [
        [10, 20, 30, 40, 50, 60, 70],
        [1],
        [4, 4, 5],
        [],
        list(range(12)),
    ]
    assert collect_pairs(ContextGenerator(window, thread_id=thread_id), sentences) == original_pairs(
        sentences, window, thread_id
    )
