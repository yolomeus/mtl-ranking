from abc import ABC, abstractmethod
from typing import List

from transformers import BertTokenizer


class Preprocessor(ABC):
    @abstractmethod
    def preprocess(self, batch_dict: List[dict]):
        """

        :param batch_dict: list of data samples, each represented as dict.
        :return: inputs to feed to the model.
        """

    def __call__(self, *args, **kwargs):
        return self.preprocess(*args, **kwargs)


class BERT(Preprocessor):
    """Tokenizes query and doc pairs to produce BERT input tensors.
    """

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def preprocess(self, batch):
        queries, docs = zip(*[x['x'] for x in batch])
        tokenized = self.tokenizer(queries, docs, padding=True, truncation=True, return_tensors='pt')
        return tokenized
