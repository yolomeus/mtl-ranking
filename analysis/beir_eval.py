import logging
import os
import pathlib
from typing import List, Tuple

import torch
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.reranking import Rerank
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from hydra.utils import instantiate
from tqdm import tqdm
from transformers import BertTokenizerFast


class BeirMTLFromCheckpoint:
    def __init__(self, ckpt_path, device='cuda:0'):
        self.device = device
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.model = self.load_mtl_model(ckpt_path).to(self.device)

    def predict(self, sentences: List[Tuple[str, str]], batch_size: int) -> List[float]:

        scores = []
        for i in tqdm(range(0, len(sentences), batch_size), total=len(sentences) // batch_size):
            batch = sentences[i:i + batch_size]
            queries, docs = zip(*batch)
            x = self.tokenizer(queries, docs, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                preds = self.model({'trec2019': x.to(self.device)}, meta={'trec2019': None})['trec2019']
                scores.append(preds)

        scores = torch.cat(scores, 0)
        scores = scores.softmax(1)[:, -1]
        return scores.cpu().numpy()

    @staticmethod
    def load_mtl_model(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model = instantiate(checkpoint['hyper_parameters']['model'])

        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k in state_dict:
            new_k = k.lstrip('model.')
            new_state_dict[new_k] = state_dict[k]
        del state_dict

        model.load_state_dict(new_state_dict)
        return model


def main():
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    dataset = "scifact"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    sent_bert = DenseRetrievalExactSearch(SentenceBERT("msmarco-distilbert-base-v3"), batch_size=16)
    retriever = EvaluateRetrieval(sent_bert, score_function="dot")
    retrieve_results = retriever.retrieve(corpus, queries)

    # re-ranking
    model = BeirMTLFromCheckpoint('../data/model_checkpoints/mtl/baseline/lr=2e-6/09-11-15/'
                                  'checkpoints/epoch-001-val_trec_map_trec2019-0.3443.ckpt')
    reranker = Rerank(model, batch_size=16)
    rerank_results = reranker.rerank(corpus, queries, retrieve_results, top_k=100)

    EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values)


if __name__ == '__main__':
    main()
