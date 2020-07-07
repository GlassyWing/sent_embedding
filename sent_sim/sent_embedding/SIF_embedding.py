from abc import ABC
import numpy as np
from sklearn.decomposition import TruncatedSVD

from sent_sim.sim_metrics import cos_sim
from sent_sim.word_embedding import WordEmbedding
from .sent_embedding import SentEmbedding
from ..word_weight import WordWeight


def compute_pc(X, npc=1):
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_


def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


class SIFEmbedding(SentEmbedding, ABC):

    def __init__(self, seg, word_weight: WordWeight, word_embedding: WordEmbedding,
                 sim_metric=cos_sim):
        self.seg = seg
        self.word_weight = word_weight
        self.word_embedding = word_embedding
        self.ind2weight = None
        self.sim_metric = sim_metric

    def lookup_idx(self, word):
        w = word.lower()
        if len(w) > 1 and w[0] == '#':
            w = w.replace("#", "")
        if w in self.word_embedding.words:
            return self.word_embedding.words[w]
        elif "unknown" in self.word_embedding.words:
            return self.word_embedding.words["unknown"]
        else:
            return len(self.word_embedding) - 1

    def get_seq(self, sentence):
        tokens = self.seg.cut(sentence)
        X1 = [self.lookup_idx(word) for word in tokens]
        return X1

    def _prepare_data(self, seqs):
        lengths = [len(seq) for seq in seqs]
        max_len = np.max(lengths)

        x = (np.ones(shape=(len(seqs), max_len)) * -1).astype(np.int32)
        x_mask = np.zeros_like(x)

        for idx, s in enumerate(seqs):
            x[idx, :lengths[idx]] = s
            x_mask[idx, :lengths[idx]] = 1.

        return x, x_mask

    def sentences2idx(self, sentences):

        seqs = []
        for s in sentences:
            seqs.append(self.get_seq(s))

        return self._prepare_data(seqs)

    def seqs2weight(self, seqs, mask):

        return (self.word_weight.weight4ind[seqs.flatten()] * mask.flatten()).reshape(seqs.shape)

    def get_weighted_average(self, x, w):
        """
        :param x: shape = (n, l)
        :param w: shape = (n, l)
        """
        n, l = x.shape
        # (n, l, d)
        word_emb = self.word_embedding.word_emb[x.flatten(), :].reshape(n, l, -1)

        # (n, d)
        emb = np.einsum("nl, nld -> nd", w, word_emb) / np.expand_dims(np.count_nonzero(w, axis=1), axis=1)

        return emb

    def sent_embedding(self, sentences):
        x, x_mask = self.sentences2idx(sentences)
        w = self.seqs2weight(x, x_mask)
        emb = self.get_weighted_average(x, w)
        emb = remove_pc(emb)
        return emb

    def sent_sim(self, queries, values=None):
        if values is None:
            sentences = queries
        else:
            sentences = queries + values

        emb = self.sent_embedding(sentences)
        if values is None:
            sim_matrix = self.sim_metric(emb, emb)
        else:
            l_q = len(queries)
            sim_matrix = self.sim_metric(emb[:l_q], emb[l_q:])

        return sim_matrix
