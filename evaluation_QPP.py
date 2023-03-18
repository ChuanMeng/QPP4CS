import argparse
import json
import os
import numpy as np
from scipy.stats import pearsonr,spearmanr,kendalltau
from collections import defaultdict
import glob

def evaluation(ap_path=None, pp_path=None, target_metric="ndcg@3"):

    ap={}
    with open(ap_path, 'r') as r:
        ap_bank = json.loads(r.read())

    for qid in ap_bank.keys():
        ap[qid]=float(ap_bank[qid][target_metric])

    pp={}
    with open(pp_path, 'r') as r:
        for line in r:
            qid, pp_value = line.rstrip().split()
            pp[qid]=float(pp_value)

    ap_list = []
    pp_list = []

    for qid in ap.keys():
        ap_list.append(ap[qid])
        pp_list.append(pp[qid])


    pearson_coefficient, pearson_pvalue = pearsonr(ap_list, pp_list)
    kendall_coefficient, kendall_pvalue = kendalltau(ap_list, pp_list)
    spearman_coefficient, spearman_pvalue = spearmanr(ap_list, pp_list)

    result_dict = {"Pearson": round(pearson_coefficient, 3), "Kendall": round(kendall_coefficient, 3),
            "Spearman": round(spearman_coefficient, 3), "len_ap": len(ap), "len_pp": len(pp),
            "P_pvalue": pearson_pvalue, "K_pvalue": kendall_pvalue,
            "S_pvalue": spearman_pvalue}

    print(result_dict)

    return result_dict


def evaluation_glob(ap_path=None, pattern=None, target_metrics=["ndcg@3", "ndcg@100", "recall@100", "map@100"]):
    for target_metric in target_metrics:
        for pp_path in sorted(glob.glob(pattern)):
            name = pp_path.split("/")[-1]
            dataset = pp_path.split("/")[-1].split(".")[0]
            output_path ="/".join(pattern.split("/")[:-1])

            result_dict = evaluation(ap_path=ap_path, pp_path=pp_path, target_metric=target_metric)

            with open(f"{output_path}/result.{dataset}", 'a+', encoding='utf-8') as w:
                name_=f"{name}-{target_metric}:"
                w.write(f"{name_.ljust(55, ' ')} {str(result_dict)}{os.linesep}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--ap_path", type=str)
    parser.add_argument("--pp_path", type=str)
    parser.add_argument("--pattern", type=str)
    parser.add_argument("--target_metric", type=str)
    parser.add_argument("--target_metrics", nargs='+')
    args = parser.parse_args()


    if args.pattern is not None:
        evaluation_glob(args.ap_path, args.pattern, target_metrics=args.target_metrics)
    else:
        result = evaluation(args.ap_path, args.pp_path, target_metric=args.target_metric)


