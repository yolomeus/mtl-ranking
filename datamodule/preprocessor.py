from abc import ABC, abstractmethod
from typing import List

import spacy as spacy
from transformers import BertTokenizer, BertTokenizerFast


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

    def __init__(self, use_fast_tokenizer):
        if use_fast_tokenizer:
            self.bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        else:
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def preprocess(self, batch):
        queries, docs = zip(*[x['x'] for x in batch])
        tokenized = self.bert_tokenizer(queries, docs, padding=True, truncation=True, return_tensors='pt')
        return tokenized


class BERTRetokenization(BERT):
    """A preprocessor that expects pre-computed meta based on SpaCy tokenization. BERT tokenization is applied
    and the meta are recomputed accordingly.
    """

    def __init__(self, use_fast_tokenizer, spacy_version, force_spacy_dl):
        super().__init__(use_fast_tokenizer)
        self.force_spacy_dl = force_spacy_dl
        self.spacy_tokenizer = self._get_spacy_tokenizer(spacy_version)

    def preprocess(self, batch):

        queries, passages = zip(*[(x['x']) for x in batch])
        tokenized = self.bert_tokenizer(queries, passages, padding=True, truncation=True, return_tensors='pt')

        spans1 = []
        spans2 = []
        for x in batch:
            if 'span2' in x:
                spans1.append(x['span1'])
                spans2.append(x['span2'])
            else:
                spans1.append(x['span1'])

        if len(spans2) > 0:
            assert len(spans1) == len(spans2)
            new_spans1, new_spans2 = [list(map(lambda x: self._retokenize_span(*x), zip(queries, passages, s)))
                                      for s in [spans1, spans2]]

            return tokenized, (new_spans1, new_spans2)

        new_spans = list(map(lambda x: self._retokenize_span(*x), zip(queries, passages, spans1)))

        return tokenized, new_spans

    def _retokenize_span(self, query, passage, span):
        """Given a list of original tokens and meta, recompute new meta for the wordpiece bert_tokenizer.
        """

        vanilla_tokens = [token.text_with_ws for token in self.spacy_tokenizer(query + ' ' + passage)]

        # everything left to the span
        a, b = span
        text_left_span = ''.join(vanilla_tokens[:a])
        text_original_span = ''.join(vanilla_tokens[a:b])

        left_retokenized = self.bert_tokenizer(text_left_span,
                                               add_special_tokens=False)['input_ids']

        span_retokenized = self.bert_tokenizer(text_original_span,
                                               add_special_tokens=False)['input_ids']

        # for some special characters the tokenizer returns an empty list instead of the unk token
        if len(span_retokenized) == 0:
            span_retokenized = [self.bert_tokenizer.unk_token_id]

        sep_pos = len(self.bert_tokenizer(query, add_special_tokens=False)['input_ids'])

        a, b = left_retokenized, span_retokenized
        # shift by 1 for [cls]
        new_span = (1 + len(a), 1 + len(a) + len(b))
        # shift another for [sep] if span is within passage, i.e. right to sep
        if new_span[0] > sep_pos:
            new_span = (new_span[0] + 1, new_span[1] + 1)

        return new_span

    def _get_spacy_tokenizer(self, version):
        if not spacy.util.is_package('en_core_web_sm') or self.force_spacy_dl:
            spacy.cli.download('en_core_web_sm-' + version, direct=True)

        import en_core_web_sm
        return en_core_web_sm.load().tokenizer
