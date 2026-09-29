import torch

from word2vec.skipgram import Skipgram
from word2vec.softmax import Softmax
from word2vec.training.context import ContextGenerator


def test_objective_averages_log_probability_over_word_positions():
    input_embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        requires_grad=True,
    )
    output_embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        requires_grad=True,
    )
    sentences = [[0, 1, 2]]

    objective = Skipgram(input_embeddings, output_embeddings).skipgram_objective(
        sentences,
        ContextGenerator(window_size=1, thread_id=0),
    )

    softmax = Softmax(input_embeddings, output_embeddings)
    expected = (
        torch.log(softmax.standard_softmax(1, 0))
        + torch.log(softmax.standard_softmax(0, 1))
        + torch.log(softmax.standard_softmax(2, 1))
        + torch.log(softmax.standard_softmax(1, 2))
    )
    assert torch.allclose(objective, expected / 3)
    assert objective.requires_grad

    objective.backward()

    assert input_embeddings.grad is not None
    assert output_embeddings.grad is not None
