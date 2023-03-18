import sys
sys.path.append('./')

from pyserini.index import IndexReader
from pyserini.search.lucene import LuceneSearcher
import json
import string
from nltk.tokenize import word_tokenize
from collections import Counter
import re
import numpy as np
import argparse
import math
from evaluation_QPP import evaluation
import pytrec_eval
from evaluate import load
import os
from utils import data_split
from scipy.stats import pearsonr

def IDF(term, index_reader):
    df, cf = index_reader.get_term_counts(term, analyzer=None)

    if df == 0:
        return 0.0
    else:
        return math.log2(index_reader.stats()['documents']/df)

def SCQ(term, index_reader):
    df, cf = index_reader.get_term_counts(term, analyzer=None)

    if cf == 0:
        return 0.0
    else:
        part_A =  1 + math.log2(cf)
        part_B = IDF(term, index_reader)

    return part_A * part_B

def avg_max_sum_std_IDF(qtokens, index_reader):
    v=[]
    for t in qtokens:
        v.append(IDF(t, index_reader))
    return [np.mean(v), max(v), sum(v), np.std(v)]

def avg_max_sum_SCQ(qtokens, index_reader):
    scq=[]
    for t in qtokens:
        scq.append(SCQ(t, index_reader))
    return [np.mean(scq), max(scq), sum(scq)]


def ictf(term, index_reader):
    df, cf = index_reader.get_term_counts(term, analyzer=None)
    if cf==0:
        return 0.0
    else:
        return math.log2(index_reader.stats()['total_terms']/cf)

def avgICTF(qtokens, index_reader):
    v=[]
    for t in qtokens:
        v.append(ictf(t, index_reader))
    return np.mean(v)

def SCS_1(qtokens, index_reader):

    part_A = math.log2(1/len(qtokens))
    part_B = avgICTF(qtokens, index_reader)

    return part_A + part_B

def SCS_2(qtokens, index_reader):
    v=[]
    qtf=Counter(qtokens)
    ql=len(qtokens)

    for t in qtokens:
        pml=qtf[t]/ql
        df, cf = index_reader.get_term_counts(t, analyzer=None)
        pcoll = cf / index_reader.stats()['total_terms']

        if pcoll==0:
            v.append(0.0)
        else:
            v.append(pml*math.log2(pml/pcoll))

    return sum(v)

def QS(qtokens, qtoken2did):

    q2did_set=set()
    for t in qtokens:
        q2did_set = q2did_set.union(set(qtoken2did[t]))

    n_Q = len(q2did_set)
    N = index_reader.stats()['documents']

    return -math.log2(n_Q/N)


def VAR(t, index_reader):
    # one query token, multiple docs containing it
    postings_list = index_reader.get_postings_list(t, analyzer=None)

    if postings_list == None:
        return 0.0, 0.0
    else:
        tf_array = np.array([posting.tf for posting in postings_list])
        tf_idf_array = np.log2(1 + tf_array) * IDF(t, index_reader)

        return np.var(tf_idf_array), np.std(tf_idf_array)

def avg_max_sum_VAR(qtokens, qtoken2x):
    v=[]
    for t in qtokens:
        v.append(qtoken2x[t])

    return [np.mean(v), max(v), sum(v)]


def t2did(t, index_reader):
    postings_list = index_reader.get_postings_list(t, analyzer=None)

    if postings_list == None:
        return []
    else:
        return [posting.docid for posting in postings_list]

def PMI(t_i, t_j, index_reader, qtoken2did):

    titj_doc_count = len(set(qtoken2did[t_i]).intersection(set(qtoken2did[t_j])))
    ti_doc_count = len(qtoken2did[t_i])
    tj_doc_count = len(qtoken2did[t_j])

    if titj_doc_count>0:
        part_A = titj_doc_count/index_reader.stats()['documents']
        part_B = (ti_doc_count/index_reader.stats()['documents'])*(tj_doc_count/index_reader.stats()['documents'])

        return math.log2(part_A/part_B)
    else:
        return 0.0

def avg_max_sum_PMI(qtokens, index_reader, qtoken2did):
    pair=[]
    pair_num =0

    if len(qtokens)==0:
        return [0.0, 0.0, 0.0]
    else:
        for i in range(0, len(qtokens)):
            for j in range(i + 1, len(qtokens)):
                pair_num += 1
                pair.append(PMI(qtokens[i], qtokens[j], index_reader, qtoken2did))

        assert len(pair) == pair_num

        return [np.mean(pair), max(pair), sum(pair)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str) # precomputation, baselines, ppl, PPL-QPP
    parser.add_argument("--index_path", type=str)
    parser.add_argument("--query_path", type=str)
    parser.add_argument("--query_path_2", type=str, default=None)
    parser.add_argument("--query_path_3", type=str, default=None)
    parser.add_argument("--query_path_4", type=str, default=None)
    parser.add_argument("--query_path_5", type=str, default=None)
    parser.add_argument("--query_path_6", type=str, default=None)
    parser.add_argument("--qrels_path", type=str)
    parser.add_argument("--actual_performance_path", type=str)
    parser.add_argument("--target_metric", type=str)
    parser.add_argument("--LM", type=str)
    parser.add_argument("--qpp_names", nargs='+')
    parser.add_argument("--alpha", type=float)
    args = parser.parse_args()

    output_path = "./output/pre-retrieval"
    dataset_class = args.query_path.split("/")[-3]
    dataset_name = args.query_path.split("/")[-1].split(".")[0]
    query_type= "-".join(args.query_path.split("/")[-1].split(".")[1].split("-")[1:])


    searcher = LuceneSearcher(args.index_path)
    index_reader = IndexReader(args.index_path)

    query = {}
    qtoken_set = set()
    query_reader = open(args.query_path, 'r').readlines()

    for line in query_reader:
        qid, qtext = line.split('\t')
        query[qid] = qtext

        qtokens = index_reader.analyze(qtext)
        for qtoken in qtokens:
            qtoken_set.add(qtoken)


    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if args.mode=="precomputation":
        others=[args.query_path_2, args.query_path_3, args.query_path_4, args.query_path_5]
        for other in others:
            if other is not None:
                query_reader_more = open(other, 'r').readlines()
                for line in query_reader_more:
                    qid, qtext = line.split('\t')

                    qtokens = index_reader.analyze(qtext)
                    for qtoken in qtokens:
                        qtoken_set.add(qtoken)

        print(f"# unique query tokens {len(qtoken_set)}")

        qtoken2var = {}
        qtoken2std = {}
        qtoken2did = {}

        count=0
        for qtoken in qtoken_set:
            qtoken2var[qtoken], qtoken2std[qtoken] =VAR(qtoken, index_reader)
            qtoken2did[qtoken] = t2did(qtoken, index_reader)

            count += 1
            if count % 50 == 0 or count == 1:
                print(f"{count}/{len(qtoken_set)}")


        f = open(f'{output_path}/{dataset_class}.qtoken2var.json', 'w')
        f.write(json.dumps(qtoken2var))
        f.close()

        f = open(f'{output_path}/{dataset_class}.qtoken2std.json', 'w')
        f.write(json.dumps(qtoken2std))
        f.close()

        f = open(f'{output_path}/{dataset_class}.qtoken2did.json', 'w')
        f.write(json.dumps(qtoken2did))
        f.close()
        print("Save successfully")

    elif args.mode=="baselines":

        with open(args.qrels_path, 'r') as f_qrel:
            qrel = pytrec_eval.parse_qrel(f_qrel)

        qtoken2var = json.load(open(f'{output_path}/{dataset_class}.qtoken2var.json'))
        qtoken2std = json.load(open(f'{output_path}/{dataset_class}.qtoken2std.json'))
        qtoken2did = json.load(open(f'{output_path}/{dataset_class}.qtoken2did.json'))

        predicted_performance = {}

        count = 0

        for qid, qtext in query.items():
            if qid not in qrel:
                continue

            count += 1
            if count == 1 or count % 10 == 0:
                print(f"{count}/{len(query)}")

            predicted_performance[qid] = {}
            qtokens = index_reader.analyze(qtext)

            # QS
            predicted_performance[qid]["QS"] = QS(qtokens, qtoken2did)

            # PMI
            predicted_performance[qid]["PMI-avg"], predicted_performance[qid]["PMI-max"], predicted_performance[qid]["PMI-sum"] = avg_max_sum_PMI(qtokens, index_reader,qtoken2did)

            # ql
            predicted_performance[qid]["ql"] = len(qtokens)

            # VAR (var)
            predicted_performance[qid]["VAR-var-avg"], predicted_performance[qid]["VAR-var-max"], predicted_performance[qid]["VAR-var-sum"] = avg_max_sum_VAR(qtokens,qtoken2var)

            # VAR (std)
            predicted_performance[qid]["VAR-std-avg"], predicted_performance[qid]["VAR-std-max"], predicted_performance[qid]["VAR-std-sum"] = avg_max_sum_VAR(qtokens,qtoken2std)

            # IDF
            predicted_performance[qid]["IDF-avg"], predicted_performance[qid]["IDF-max"], predicted_performance[qid]["IDF-sum"], predicted_performance[qid]["IDF-std"] = avg_max_sum_std_IDF(qtokens, index_reader)

            # SCQ
            predicted_performance[qid]["SCQ-avg"], predicted_performance[qid]["SCQ-max"], predicted_performance[qid]["SCQ-sum"] = avg_max_sum_SCQ(qtokens, index_reader)

            # avgICTF
            predicted_performance[qid]["avgICTF"] = avgICTF(qtokens, index_reader)

            # SCS
            predicted_performance[qid]["SCS-1"] = SCS_1(qtokens, index_reader)
            predicted_performance[qid]["SCS-2"] = SCS_2(qtokens, index_reader)


    elif args.mode=="ppl":
        perplexity = load("perplexity", module_type="metric")
        results = perplexity.compute(predictions=list(query.values()), model_id=args.LM)

        predicted_performance = {}

        for index, qid in enumerate(query.keys()):
            predicted_performance[qid] = {}
            predicted_performance[qid][f"ppl-{args.LM}"] = results['perplexities'][index]

    elif args.mode=="PPL-QPP":
        with open(args.qrels_path, 'r') as f_qrel:
            qrel = pytrec_eval.parse_qrel(f_qrel)

        for qpp_name in args.qpp_names:
            qpp_bank = {}
            with open(f"{output_path}/{dataset_name}.{query_type}-{qpp_name}", 'r') as r:
                for line in r:
                    qid, pp_value = line.rstrip().split()
                    if qid in qrel:
                        qpp_bank[qid] = float(pp_value)

            ppl_bank = {}
            with open(f"{output_path}/{dataset_name}.{query_type}-ppl-{args.LM}", 'r') as r:
                for line in r:
                    qid, pp_value = line.rstrip().split()
                    if qid in qrel:
                        ppl_bank[qid] = 1 / np.log(float(pp_value))

            ppl_min = min(ppl_bank.values())
            ppl_max = max(ppl_bank.values())

            qpp_min = min(qpp_bank.values())
            qpp_max = max(qpp_bank.values())

            ppl_avg = np.mean(list(ppl_bank.values()))
            ppl_std = np.std(list(ppl_bank.values()))

            qpp_avg = np.mean(list(qpp_bank.values()))
            qpp_std = np.std(list(qpp_bank.values()))

            predicted_performance = {}

            for qid, qtext in query.items():
                if qid not in qrel:
                    continue

                predicted_performance[qid] = {}

                min_max_ppl = (ppl_bank[qid] - ppl_min) / (ppl_max - ppl_min)
                min_max_qpp = (qpp_bank[qid] - qpp_min) / (qpp_max - qpp_min)

                z_ppl = (ppl_bank[qid] - ppl_avg) / ppl_std
                z_qpp = (qpp_bank[qid] - qpp_avg) / qpp_std

                predicted_performance[qid][f"{qpp_name}-ppl-{args.LM}-{args.alpha}_mm"] = args.alpha * min_max_ppl + ((1.00 - args.alpha) * min_max_qpp)
                predicted_performance[qid][f"{qpp_name}-ppl-{args.LM}-{args.alpha}_z"] = args.alpha * z_ppl + ((1.00 - args.alpha) * z_qpp)


    else:
        raise NotImplemented

    if args.mode!="precomputation":
        name_list = []
        for qid, v in predicted_performance.items():
            name_list = list(v.keys())
            break

        for name in name_list:
            output_path_ = f"{output_path}/{dataset_name}.{query_type}-{name}"
            print(f"{name} on the {dataset_name} dataset")
            print(f"Write predicted performance into the file {output_path_}")

            with open(output_path_, 'w') as pp_w:
                for qid, v in predicted_performance.items():
                    pp_w.write(qid + '\t' + str(v[name]) + '\n')




