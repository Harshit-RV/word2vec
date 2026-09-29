import torch

def initialize_embeddings(
    vocab_size: int,
    embedding_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  # Copies from the original C implementation: (next_random / (real)0x7fffffff - 0.5) / layer1_size
  input_embeddings = ((torch.rand(vocab_size, embedding_dim)  - 0.5) / embedding_dim).requires_grad_()
  output_embeddings = torch.zeros(vocab_size, embedding_dim, requires_grad=True)

  print(input_embeddings)
  print(output_embeddings)

  return (input_embeddings, output_embeddings)

initialize_embeddings(20, 100)