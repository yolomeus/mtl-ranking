from abc import abstractmethod, ABC

import torch
from transformers import BertModel

from model.body.base import MTLBody


class BERT(MTLBody, ABC):

    def __init__(self, model_name, num_layers, out_dim):
        super().__init__(out_dim)
        self.model_name = model_name
        self.bert = self._build_bert(num_layers)

    @abstractmethod
    def _build_bert(self, num_layers):
        """Initialize and return the bert model.

        :return: the bert model.
        """

    def forward(self, inputs):
        outputs = self.bert.forward(**inputs, output_hidden_states=True)
        return outputs.hidden_states


class BERTHuggingFace(BERT):
    """A Pre-Trained BERT from the huggingface transformers library that returns hidden states of all layers, including
    the initial matrix embeddings (layer 0).
    """

    def __init__(self, model_name, num_layers, out_dim):
        """

        :param model_name: huggingface model name or path to model checkpoint file
        """
        super().__init__(model_name, num_layers, out_dim)

    def _build_bert(self, num_layers):
        bert = BertModel.from_pretrained(self.model_name, num_hidden_layers=num_layers)
        return bert


class BERTBaseFromLocalCheckpoint(BERT):
    def __init__(self, ckpt_path, model_name, num_layers, out_dim):
        super().__init__(model_name, num_layers, out_dim)
        self.ckpt_path = ckpt_path

    def _build_bert(self, num_layers):
        state_dict = torch.load(self.ckpt_path)
        bert = BertModel.from_pretrained('bert-base-uncased', state_dict=state_dict, num_hidden_layers=num_layers)
        return bert
