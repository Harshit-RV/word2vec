import torch

class Softmax:
    def __init__(
        self,
        input_embeddings: torch.Tensor,
        output_embeddings: torch.Tensor,
    ):
        self.input_embeddings = input_embeddings
        self.output_embeddings = output_embeddings

    def standard_softmax(self, output_id: int, input_id: int) -> torch.Tensor:
        """Probability of output_id given input_id."""
        v_o = self.output_embeddings[output_id]
        v_i = self.input_embeddings[input_id]

        numerator = torch.exp(torch.dot(v_o, v_i))
        denominator = torch.zeros((), dtype=v_i.dtype, device=v_i.device)

        for word_id in range(self.output_embeddings.shape[0]):
            v_w = self.output_embeddings[word_id]
            denominator = denominator + torch.exp(torch.dot(v_w, v_i))

        return numerator / denominator