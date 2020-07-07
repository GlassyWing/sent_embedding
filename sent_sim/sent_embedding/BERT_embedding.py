from concurrent.futures import ThreadPoolExecutor

from transformers import BertTokenizer, BertModel, PretrainedConfig
import torch

from .sent_embedding import SentEmbedding


class BertEmbedding(SentEmbedding):

    def __init__(self, vocab_path, config_path, weights_path):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)
        self.config = PretrainedConfig.from_json_file(config_path)
        self.model = BertModel.from_pretrained(weights_path,
                                               config=self.config)

    def _sent_embedding(self, sent):
        sent_idx = self.tokenizer.encode(sent,
                                         add_special_tokens=False,
                                         pad_to_max_length=False,
                                         truncation=True,
                                         max_length=512)
        sent_tensor = torch.tensor([sent_idx])
        with torch.no_grad():
            last_hidden_states, pool_outputs, all_hidden_states = self.model(sent_tensor, output_hidden_states=True)
        sent_emb = all_hidden_states[-2][0]
        sent_emb = sent_emb.mean(dim=0)
        return sent_emb

    def sent_embedding(self, sentences):
        """

        :param sentences: a list contains n sentence
        :return: Vector. shape = (n, d)
        """

        with ThreadPoolExecutor() as executor:
            sent_embs = []
            for sent_emb in executor.map(self._sent_embedding, sentences):
                sent_embs.append(sent_emb)

        sent_embs = torch.stack(sent_embs, dim=0)
        sent_embs = sent_embs / torch.norm(sent_embs, dim=-1, keepdim=True)

        return sent_embs

    def sent_sim(self, queries, values=None):
        queries = self.sent_embedding(queries)
        values = queries if values is None else self.sent_embedding(values)

        # (L_q, L_v)
        sim_matrix = torch.mm(queries, values.t())
        return sim_matrix
