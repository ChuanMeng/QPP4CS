import sys
sys.path.append('./')

from pyserini.index import IndexReader
from pyserini.search.lucene import LuceneSearcher
import json
import string
from collections import Counter
import re
import numpy as np
import argparse
import math
from evaluation_QPP import evaluation
import pytrec_eval
from evaluate import load
import os
from scipy.stats import pearsonr

def RM1(qtokens, pid_list, score_list, index_reader, k, mu=1000):
    V =[]
    doc_len = np.zeros(k)

    for idx_p, pid in enumerate(pid_list[:k]):
        V+=index_reader.get_document_vector(pid).keys()
        doc_len[idx_p] = sum(index_reader.get_document_vector(pid).values())

    V = list(set(V))
    mat = np.zeros([k, len(V)])

    for idx_p, pid in enumerate(pid_list[:k]):
        for token in index_reader.get_document_vector(pid).keys():
            mat[idx_p, V.index(token)] = index_reader.get_document_vector(pid)[token]

    _p_w_q = np.dot(np.array([score_list[:k] / doc_len , ]), mat) # [1, V] become a probability distribution
    p_w_q = np.asarray(_p_w_q/ sum(score_list[:k])).squeeze() # normalisation [V]
    rm1 = np.sort(np.array(list(zip(V, p_w_q)), dtype=[('tokens', np.object), ('token_scores', np.float32)]), order='token_scores')[::-1] # [V]
    return rm1

def CLARITY(rm1, index_reader, term_num=100):

    rm1_cut = rm1[:term_num] # [term num]
    p_w_q = rm1_cut['token_scores'] / rm1_cut['token_scores'].sum() # make sure it is a probability distribution after sampling
    p_t_D = np.array([[index_reader.get_term_counts(token, analyzer=None)[1] for token in rm1_cut['tokens']], ]) / index_reader.stats()['total_terms'] # [1, term num]

    return np.log(p_w_q / p_t_D).dot(p_w_q)[0]


def WIG(qtokens, score_list, k):
    corpus_score = np.mean(score_list)
    wig_norm = (np.mean(score_list[:k]) - corpus_score)/ np.sqrt(len(qtokens))
    wig_no_norm = np.mean(score_list[:k]) / np.sqrt(len(qtokens))

    return wig_norm, wig_no_norm


def NQC(score_list, k):
    corpus_score = np.mean(score_list)
    nqc_norm = np.std(score_list[:k]) / corpus_score
    nqc_no_norm = np.std(score_list[:k])

    return nqc_norm, nqc_no_norm

def SIGMA_MAX(score_list):
    max_std=0
    scores=[]

    for idx, score in enumerate(score_list):
        scores.append(score)
        if np.std(scores)>max_std:
            max_std = np.std(scores)

    return max_std, len(scores)

def SIGMA_X(qtokens, score_list, x):

    top_score = score_list[0]
    scores = []

    for idx, score in enumerate(score_list):
        if score>=(top_score*x):
            scores.append(score)

    return np.std(scores)/np.sqrt(len(qtokens)), len(scores)

def SMV(score_list, k):
    corpus_score = np.mean(score_list)
    mu = np.mean(score_list[:k])
    smv_norm = np.mean(np.array(score_list[:k])*abs(np.log(score_list[:k]/mu)))/corpus_score
    smv_no_norm = np.mean(np.array(score_list[:k])*abs(np.log(score_list[:k]/mu)))

    return smv_norm, smv_no_norm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_clarity", type=int, default=100)
    parser.add_argument("--term_num", type=int, default=100)
    parser.add_argument("--mu", type=int, default=1000)
    parser.add_argument("--k_wig", type=int, default=5)
    parser.add_argument("--k_nqc", type=int, default=100)
    parser.add_argument("--k_smv", type=int, default=100)
    parser.add_argument("--x", type=float, default=0.5)

    parser.add_argument("--query_path", type=str, default='')
    parser.add_argument("--run_path", type=str, default='')
    parser.add_argument("--index_path", type=str, default='')
    parser.add_argument("--qrels_path", type=str)
    parser.add_argument("--output_path", type=str)

    args = parser.parse_args()

    dataset_class = args.query_path.split("/")[-3]
    dataset_name = args.query_path.split("/")[-1].split(".")[0]
    query_type = "-".join(args.query_path.split("/")[-1].split(".")[1].split("-")[1:])
    retriever = "-".join(args.run_path.split("/")[-1].split(".")[1].split("-")[1:])

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Load index
    searcher = LuceneSearcher(args.index_path)
    index_reader = IndexReader(args.index_path)

    # Load run file
    with open(args.run_path, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)

    # load qrel file
    with open(args.qrels_path, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    # load query file
    query = {}
    query_reader = open(args.query_path, 'r').readlines()

    for idx, line in enumerate(query_reader):
        qid, qtext = line.split('\t')

        if qid not in qrel:
            continue

        query[qid] = qtext

    predicted_performance = {}
    count=0
    for qid, qtext in query.items():
        count += 1
        if count == 1 or count % 10 == 0:
            print(f"{count}/{len(query)}")


        predicted_performance[qid] = {}
        qtokens = index_reader.analyze(qtext)

        pid_list = [pid for (pid, score) in sorted(run[qid].items(), key=lambda x: x[1], reverse=True)]
        score_list = [score for (pid, score) in sorted(run[qid].items(), key=lambda x: x[1], reverse=True)]

        assert  len(pid_list)==len(score_list)==1000

        rm1 = RM1(qtokens, pid_list, score_list, index_reader, args.k_clarity, mu=args.mu)
        predicted_performance[qid][f"clarity-score-k{args.k_clarity}"]= CLARITY(rm1, index_reader, term_num=args.term_num)
        del rm1

        predicted_performance[qid][f"wig-norm-k{args.k_wig}"],  predicted_performance[qid][f"wig-no-norm-{args.k_wig}"] = WIG(qtokens, score_list, args.k_wig)

        predicted_performance[qid][f"nqc-norm-k{args.k_nqc}"],  predicted_performance[qid][f"nqc-no-norm-{args.k_nqc}"] = NQC(score_list, args.k_nqc)

        predicted_performance[qid][f"smv-norm-k{args.k_smv}"],  predicted_performance[qid][f"smv-no-norm-{args.k_smv}"] = SMV(score_list, args.k_smv)

        predicted_performance[qid][f"sigma_x{args.x}"], actual_k = SIGMA_X(qtokens, score_list, args.x)

        predicted_performance[qid][f"sigma_max"], actual_k = SIGMA_MAX(score_list)


    name_list = []
    for qid, v in predicted_performance.items():
        name_list = list(v.keys())
        break

    for name in name_list:
        output_path_ = f"{args.output_path}/{dataset_name}.{retriever}.{query_type}-{name}"
        print(f"{name} on the {dataset_name} dataset")
        print(f"Write predicted performance into the file {output_path_}")

        with open(output_path_, 'w') as pp_w:
            for qid, v in predicted_performance.items():
                pp_w.write(qid + '\t' + str(v[name]) + '\n')

