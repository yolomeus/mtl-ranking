import logging
import os
import pathlib
from argparse import ArgumentParser
from os.path import exists
from typing import List, Tuple

import torch
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.reranking import Rerank
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search
from hydra.utils import instantiate
from tqdm import tqdm
from transformers import BertTokenizerFast


class BeirMTLFromCheckpoint:
    """Wrapper around mode.MultiTaskModel so it can be used for beir evaluation.
    """

    def __init__(self, ckpt_path, device='cuda:0'):
        """
        :param ckpt_path: path to the pl checkpoint of a MultiTaskModel
        :param device: the device to run inference on.
        """

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


def main(args):
    setup_logging()

    base_url = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/'
    ds_to_results = {}
    for dataset_name in args.datasets:
        corpus, queries, qrels = get_dataset(base_url,
                                             dataset_name)

        results = get_bm25_results(corpus, queries, dataset_name, args.es_hostname, args.skip_create_index)

        # re-ranking
        model = BeirMTLFromCheckpoint(args.model_ckpt)
        reranker = Rerank(model, batch_size=args.batch_size)
        rerank_results = reranker.rerank(corpus, queries, results, top_k=args.top_k)

        results = EvaluateRetrieval.evaluate(qrels, rerank_results, args.cutoffs)
        ds_to_results[dataset_name] = results

    log_dir = f'logs/top_{args.top_k}/bs_{args.batch_size}/'
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
