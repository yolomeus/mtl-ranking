import glob
import json
import pickle
import re
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
import pytrec_eval
from pingouin import tost
from scipy.stats import ttest_rel
from tqdm import tqdm


def load_pickle(file: str):
    with open(file, 'rb') as fp:
        x = pickle.load(fp)

    query2doc2rel = defaultdict(dict)
    for q_id, doc_id, rel in zip(x['q_ids'], x['doc_ids'], x['predictions']):
        query2doc2rel[q_id][doc_id] = rel

    return query2doc2rel


def load_csv(file: str):
    x = pd.read_csv(file)

    query2doc2rel = defaultdict(dict)
    for _, (q_id, doc_id, pred, _) in x.iterrows():
        q_id, doc_id = str(int(q_id)), str(int(doc_id))
        query2doc2rel[q_id][doc_id] = pred

    return query2doc2rel


def load_qrels(qrels_file: str):
    qrels = defaultdict(dict)
    with open(qrels_file, 'r', encoding='utf8') as fp:
        lines = fp.readlines()

    lines = map(lambda x: x.strip().split(), lines)
    for q_id, _, doc_id, rel in lines:
        qrels[q_id][doc_id] = int(rel)

    return qrels


def get_load_fn(file):
    if file.endswith('.pkl'):
        return load_pickle
    elif file.endswith('.csv'):
        return load_csv
    elif 'qrels' in file and file.endswith('.txt'):
        return load_qrels
    raise NotImplementedError


def test_predictions(file1: str, file2: str, qrels: str, metric_names: List[str],
                     test_fn: Callable):
    load1 = get_load_fn(file1)
    load2 = get_load_fn(file2)
    load_qr = get_load_fn(qrels)

    x1 = load1(file1)
    x2 = load2(file2)
    qrels = load_qr(qrels)
    row = []
    for metric_name in metric_names:
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {metric_name})
        result1 = evaluator.evaluate(x1)
        result2 = evaluator.evaluate(x2)

        q_ids = list(set(result2.keys()) & set(result1.keys()))

        scores_1 = [result1[i][metric_name] for i in q_ids]
        scores_2 = [result2[i][metric_name] for i in q_ids]

        pval = test_fn(scores_1, scores_2)
        row.append({'metric': metric_name, 'pval': pval})

    return row


def get_prediction_paths(base_path, filename):
    """
    :param base_path: directory to search in
    :param filename: name of each prediction file
    :return: list of paths to prediction files matching `filename`
    """
    path_pattern = Path(base_path) / "*/*/predictions/" / filename
    x = glob.glob(str(path_pattern))
    return x


def get_other_name(other_file):
    x = str(Path(other_file).as_uri())
    name = re.search(r'.*/(.*/.*)/predictions', x).group(1)
    name = name.replace('/', '_')
    return name


def test_baseline_against_all(baseline_file, others_files, qrels_file, metric_names, test_fn):
    results = {}
    for other_file in tqdm(others_files, total=len(others_files)):
        other_name = get_other_name(other_file)
        result = test_predictions(baseline_file, other_file, qrels_file, metric_names, test_fn)
        results[other_name] = result

    return results


def dump_json(file, x):
    with open(file, 'w') as fp:
        json.dump(x, fp)


# significance test functions
def ttest_one_sided(a, b):
    return ttest_rel(a, b, alternative='less').pvalue


def tost_5_percent(a, b):
    bound = np.mean([a, b]) * 0.05
    return tost(a, b, bound=bound, paired=True).loc['TOST', 'pval']


def tost_2_percent(a, b):
    bound = np.mean([a, b]) * 0.02
    return tost(a, b, bound=bound, paired=True).loc['TOST', 'pval']


def main():
    ap = ArgumentParser()
    ap.add_argument('base_dir', type=str,
                    help='base directory to prediction folders, i.e. "./" for "./<task>/<layer>/predictions"')
    ap.add_argument('out_dir', type=str, help='output directory for test results')
    ap.add_argument('qrels_file', type=str, help='qrels.txt following the trec format, e.g. "2019qrels-pass.txt"')
    ap.add_argument('--pred_files', type=str, default='trec2019.csv',
                    help='Shared name for prediction files for a given dataset. All prediction files are expected to '
                         'be named like this.')
    ap.add_argument('--metrics', nargs='+', default=['map', 'recip_rank', 'ndcg_cut_10', 'ndcg_cut_20', 'P_10', 'P_20'])

    args = ap.parse_args()

    file_name = args.pred_files
    baseline_file = f'{args.base_dir}/baseline/predictions/' + file_name
    others_files = get_prediction_paths(args.base_dir, file_name)

    ttest_res = test_baseline_against_all(baseline_file, others_files, args.qrels_file, args.metrics, ttest_one_sided)
    tost_5_res = test_baseline_against_all(baseline_file, others_files, args.qrels_file, args.metrics, tost_5_percent)
    tost_2_res = test_baseline_against_all(baseline_file, others_files, args.qrels_file, args.metrics, tost_5_percent)

    name = file_name.split('.')[0]
    dump_json(f'./significance/ttest_{name}.json', ttest_res)
    dump_json(f'./significance/tost_5_percent_{name}.json', tost_5_res)
    dump_json(f'./significance/tost_2_percent_{name}.json', tost_2_res)


if __name__ == '__main__':
    main()
