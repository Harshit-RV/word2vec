import torch

from word2vec.initialization import initialize_embeddings


def test_initialize_embeddings():
    input_embeddings, output_embeddings = initialize_embeddings(100, 50)
    bound = 0.5 / 50

    assert input_embeddings.shape == (100, 50)
    assert output_embeddings.shape == (100, 50)
    assert torch.all(input_embeddings >= -bound)
    assert torch.all(input_embeddings < bound)
    assert torch.all(output_embeddings == 0)
    assert input_embeddings.requires_grad
    assert output_embeddings.requires_grad
    assert input_embeddings.grad is None
    assert output_embeddings.grad is None
