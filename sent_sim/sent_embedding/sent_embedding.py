from abc import abstractmethod, ABCMeta


class SentEmbedding(metaclass=ABCMeta):

    @abstractmethod
    def sent_embedding(self, sentences):
        pass

    @abstractmethod
    def sent_sim(self, queries, values=None):
        pass
