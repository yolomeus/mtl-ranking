import pickle
from argparse import ArgumentParser
from collections import defaultdict
from os.path import split, join
from pathlib import Path

import numpy as np
import pandas as pd
import pytrec_eval
from scipy.stats import ttest_rel
from scipy.stats.stats import Ttest_relResult


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


def export_results(results, file_names, export_dir):
    file_names = map(lambda x: x.split('.')[0], file_names)
    with open(join(export_dir, '-'.join(['ttest', *file_names]) + '.csv'), 'w') as fp:
        results = map(lambda x: ','.join(map(str, x)) + '\n', results)
        fp.writelines(results)


def test_predictions(file1, file2, qrels, metric_names):
    load1 = get_load_fn(file1)
    load2 = get_load_fn(file2)
    load_qr = get_load_fn(qrels)

    x1 = load1(file1)
    x2 = load2(file2)
    qrels = load_qr(qrels)

    file1_name = split(file1)[-1]
    file2_name = split(file2)[-1]

    results = [('metric', file1_name, file2_name, 'statistic', 'pvalue')]

    for metric_name in metric_names:
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {metric_name})
        result1 = evaluator.evaluate(x1)
        result2 = evaluator.evaluate(x2)

        q_ids = list(set(result2.keys()) & set(result1.keys()))

        scores_1 = [result1[i][metric_name] for i in q_ids]
        scores_2 = [result2[i][metric_name] for i in q_ids]

        avg_score1 = np.mean(scores_1)
        avg_score2 = np.mean(scores_2)

        ttest_result: Ttest_relResult = ttest_rel(scores_2, scores_1)

        results.append((metric_name, avg_score1, avg_score2, ttest_result.statistic, ttest_result.pvalue))

        print(f'{metric_name}:',
              '\nfile1:', avg_score1, '\nfile2', avg_score2,
              '\n')
        print(ttest_result, '\n')
        print('-' * 100)

    export_results(results, [file1_name, file2_name], Path(file1).parent)


def main():
    ap = ArgumentParser()
    ap.add_argument('pred_file1', type=str, help='file containing predictions, either as pkl or csv.')
    ap.add_argument('pred_file2', type=str, help='file containing predictions, either as pkl or csv.')
    ap.add_argument('qrels_file', type=str, help='qrels.txt following the trec format.')
    ap.add_argument('--metrics', nargs='+', default=['recip_rank', 'map'])

    args = ap.parse_args()

    test_predictions(args.pred_file1, args.pred_file2, args.qrels_file, args.metrics)


if __name__ == '__main__':
    main()
