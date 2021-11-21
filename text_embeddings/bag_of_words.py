from typing import List
from text_embeddings.base import BaseEmbedingModel
import numpy as np

class BagOfWords(BaseEmbedingModel):
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def call(self, sequence):
        if type(sequence) == List:            
            return [self.call(x) for x in sequence]

        counts = np.bincount(sequence, minlength=self.vocab_size)
        return counts