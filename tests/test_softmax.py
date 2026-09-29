import math

import torch

from word2vec.softmax import Softmax


def test_softmax_is_the_probability_over_the_vocabulary():
    input_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    output_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    softmax = Softmax(input_embeddings, output_embeddings)

    probability = softmax.standard_softmax(output_id=2, input_id=0)

    dots = [1.0, 0.0, 0.5]
    shift = max(dots)
    weights = [math.exp(value - shift) for value in dots]
    expected = weights[2] / sum(weights)
    assert math.isclose(probability.item(), expected, rel_tol=1e-6)

def test_softmax_matches_pytorch():
    input_embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    output_embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    )

    softmax = Softmax(input_embeddings, output_embeddings)

    probability = softmax.standard_softmax(
        output_id=2,
        input_id=0,
    )

    input_vector = input_embeddings[0]
    scores = output_embeddings @ input_vector
    expected = torch.softmax(scores, dim=0)[2]

    assert torch.allclose(probability, expected)