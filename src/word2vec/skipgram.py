import torch
from word2vec.softmax import Softmax
from word2vec.training.context import ContextGenerator

class Skipgram:
    def __init__(
        self,
        input_embeddings: torch.Tensor,
        output_embeddings: torch.Tensor,
    ):
        self.input_embeddings = input_embeddings
        self.output_embeddings = output_embeddings

    def skipgram_objective(
        self,
        sentences: list[list[int]],
        context_generator: ContextGenerator,
    ) -> torch.Tensor:
        softmax = Softmax(self.input_embeddings, self.output_embeddings)
        total = torch.zeros((), dtype=self.input_embeddings.dtype, device=self.input_embeddings.device)
        # T in the paper: one count per word position, not per context pair.
        word_count = sum(len(sentence) for sentence in sentences)

        for sentence in sentences:
            for position, center in enumerate(sentence):
                for context_id in context_generator.context_ids(sentence, position):
                    total += torch.log(softmax.standard_softmax(context_id, center))

        if word_count == 0:
            return total
        return total / word_count
