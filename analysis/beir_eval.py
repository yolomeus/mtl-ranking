import logging
import os
import pathlib
from argparse import ArgumentParser
from os.path import exists
from typing import List, Tuple, Dict

import numpy as np
import torch
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.reranking import Rerank
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from beir.retrieval.search.lexical import BM25Search
from hydra.utils import instantiate
from tqdm import tqdm
from transformers import BertTokenizerFast


class BeirMTLFromCheckpoint:
    """Wrapper around mode.MultiTaskModel so it can be used for beir evaluation.
    """

    def __init__(self, ckpt_path, device='cuda:0', output_name='trec2019'):
        """
        :param ckpt_path: path to the pl checkpoint of a MultiTaskModel
        :param device: the device to run inference on.
        """

        self.device = device
        self.output_name = output_name

        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.model = self.load_mtl_model(ckpt_path).to(self.device)

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


class BeirMTLRerank(BeirMTLFromCheckpoint):

    def predict(self, sentences: List[Tuple[str, str]], batch_size: int) -> List[float]:
        scores = []
        for i in tqdm(range(0, len(sentences), batch_size), total=len(sentences) // batch_size):
            batch = sentences[i:i + batch_size]
            queries, docs = zip(*batch)
            x = self.tokenizer(queries, docs, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                preds = self.model({self.output_name: x.to(self.device)},
                                   meta={self.output_name: None})[self.output_name]
                scores.append(preds)

        scores = torch.cat(scores, 0)
        scores = scores.softmax(1)[:, -1]
        return scores.cpu().numpy()


class BeirMTLDense(BeirMTLFromCheckpoint):

    def __init__(self, ckpt_path, layer_to_use):
        super().__init__(ckpt_path)
        self.layer_to_use = layer_to_use

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        return self.encode_texts(queries, batch_size)

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        corpus = [(x['title'] + self.tokenizer.sep_token + x['text']) for x in corpus]
        return self.encode_texts(corpus, batch_size)

    def encode_texts(self, texts, batch_size):
        texts = [x.strip() for x in texts]
        encoded = []
        for i in tqdm(range(0, len(texts), batch_size), total=len(texts) // batch_size):
            batch = texts[i:i + batch_size]
            x = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                embeds = self.model.body(x.to(self.device))[self.layer_to_use]
                encoded.append(embeds.mean(1))

        encoded = torch.cat(encoded, dim=0).cpu().numpy()
        return encoded


def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])


def get_dataset(base_url, name):
    url = base_url + f'{name}.zip'
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = os.path.join(out_dir, name)
    if not exists(data_path):
        util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels


def get_bm25_results(corpus, queries, dataset_name, hostname, init_index):
    retrieval_model = BM25Search(index_name=dataset_name,
                                 hostname=hostname,
                                 initialize=init_index)
    retriever = EvaluateRetrieval(retrieval_model, score_function="dot")
    retrieve_results = retriever.retrieve(corpus, queries)
    return retrieve_results


def evaluate_reranking(model_ckpt, batch_size, cutoffs, top_k, qrels, corpus, queries, dataset_name, es_hostname,
                       skip_create_index):
    results = get_bm25_results(corpus, queries, dataset_name, es_hostname, skip_create_index)

    # re-ranking
    model = BeirMTLRerank(model_ckpt)
    reranker = Rerank(model, batch_size=batch_size)
    rerank_results = reranker.rerank(corpus, queries, results, top_k=top_k)

    results = EvaluateRetrieval.evaluate(qrels, rerank_results, cutoffs)
    return results


def evaluate_dense(model_ckpt, batch_size, cutoffs, qrels, corpus, queries, layer_to_use):
    model = DenseRetrievalExactSearch(BeirMTLDense(model_ckpt, layer_to_use), batch_size=batch_size)
    retriever = EvaluateRetrieval(model, score_function='cos_sim')
    retrieved = retriever.retrieve(corpus, queries)
    results = retriever.evaluate(qrels, retrieved, cutoffs)
    return results


def main(args):
    setup_logging()

    base_url = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/'
    ds_to_results = {}
    for dataset_name in args.datasets:
        corpus, queries, qrels = get_dataset(base_url,
                                             dataset_name)

        if args.retrieval_type == 'rerank':
            results = evaluate_reranking(args.model_ckpt, args.batch_size, args.cutoffs, args.top_k, qrels, corpus,
                                         queries,
                                         dataset_name, args.es_hostname, args.skip_create_index)
        elif args.retrieval_type == 'dense':
            results = evaluate_dense(args.model_ckpt, args.batch_size, args.cutoffs, qrels, corpus, queries,
                                     args.layer_to_use)
        else:
            raise NotImplementedError

        ds_to_results[dataset_name] = results

    log_dir = f'logs/{args.retrieval_type}/top_{args.top_k}/bs_{args.batch_size}/'
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, args.model_name + '.csv'), 'w') as fp:
        # create header
        header = ['dataset']
        for x in list(ds_to_results.values())[0]:
            header.extend(x.keys())

        fp.write(','.join(map(str, header)) + '\n')

        # write metric for each ds
        for ds_name in ds_to_results:
            row = [ds_name]
            for item in ds_to_results[ds_name]:
                row.extend(item.values())

            fp.write(','.join(map(str, row)) + '\n')


if __name__ == '__main__':
    supported_datasets = ['trec-covid', 'nfcorpus', 'nq', 'hotpotqa', 'fiqa', 'arguana', 'webis-touche2020', 'quora',
                          'dbpedia-entity', 'scidocs', 'fever', 'climate-fever', 'scifact']

    ap = ArgumentParser()

    ap.add_argument('model_ckpt', type=str, help='path to model checkpoint')
    ap.add_argument('model_name', type=str, help='model name used for output file.')
    ap.add_argument('--retrieval_type', type=str, default='rerank', choices=['dense', 'rerank'])
    ap.add_argument('--layer_to_use', type=int, default=12, help='which BERT layer to get embeddings from. Only '
                                                                 'applies to dense retrieval')

    ap.add_argument('--batch_size', type=int, default=32, help='batch size for predicting relevance scores')
    ap.add_argument('--cutoffs', nargs='+', default=[1, 10, 20, 50, 100],
                    help='cutoff thresholds for ranking metrics.')
    ap.add_argument('--top_k', type=int, default=100, help='re-rank the top k docs retrieved by bm25')
    ap.add_argument('--datasets', nargs='+', default=supported_datasets,
                    help='cutoff thresholds for ranking metrics.')
    ap.add_argument('--skip_create_index', action='store_false',
                    help='whether to create bm25 elastic search indices '
                         'or use existing ones for all specified datasets.')
    ap.add_argument('--es_hostname', type=str, default='localhost', help='hostname of the elastic search instance to '
                                                                         'be used')
    main(ap.parse_args())
