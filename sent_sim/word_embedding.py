import h5py
import os
from sent_sim.utils import load_words


class WordEmbedding:

    def __init__(self, path):
        vocab_path = os.path.join(path, "vocab.txt")
        self.emb_path = os.path.join(path, "word.embedding.h5")
        self._words = load_words(vocab_path)
        self._word_emb = None
        self._live_data = None

    @property
    def word_emb(self):
        if self._word_emb is None:
            self._live_data = h5py.File(self.emb_path, 'r')
            self._word_emb = self._live_data["emb"][:]
        return self._word_emb

    @property
    def words(self):
        return self._words

    def __len__(self):
        return len(self.words)
