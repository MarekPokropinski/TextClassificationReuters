from typing import List
from text_embeddings.base import BaseEmbedingModel
import numpy as np

def build_idf(sequences, vocab_size):
    counts = np.zeros(vocab_size)
    sequence_counts = np.empty(vocab_size)
    for sequence in sequences:
        sequence_counts[:] = 0
        sequence_counts[sequence] = 1
        counts += sequence_counts
    
    N = len(sequences) + 1
    counts += 1 # avoid 0 in denominator
    return np.log(N/counts)

class TFIDF(BaseEmbedingModel):
    def __init__(self, vocab_size: int, idf: np.ndarray) -> None:
        self.vocab_size = vocab_size
        self.idf = idf

    def call(self, sequence):
        if type(sequence) == List:
            return [self.call(x) for x in sequence]

        counts = np.bincount(sequence, minlength=self.vocab_size)
        tf = counts / counts.sum()
        tfidf = tf * self.idf
        return tfidf
