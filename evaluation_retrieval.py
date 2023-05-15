import argparse
import os
import sys
import pytrec_eval
import json


def evaluation(args):
    mapping = {"ndcg_cut_3": "ndcg@3",
               "ndcg_cut_10": "ndcg@10",
               "ndcg_cut_100": "ndcg@100",
               "ndcg_cut_1000": "ndcg@1000",
               "mrr_5": "mrr@5",
               "mrr_10": "mrr@10",
               "mrr_100": "mrr@100",
               "map_cut_10": "map@10",
               "map_cut_100": "map@100",
               "map_cut_1000": "map@1000",
               "recall_5": "recall@5",
               "recall_100": "recall@100",
               "recall_1000": 'recall@1000'}

    with open(args.run, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)

    with open(args.qrel, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    print("len(list(run))",len(list(run)))
    print("len(list(qrel))",len(list(qrel)))


    run_5 = {}
    run_10 = {}
    run_100 = {}


    for qid,did_score in run.items():
        sorted_did_score = [(did, score) for did, score in sorted(did_score.items(), key=lambda item: item[1], reverse=True)]
        run_5[qid]= dict(sorted_did_score[0:5])
        run_10[qid] = dict(sorted_did_score[0:10])
        run_100[qid] = dict(sorted_did_score[0:100])


    evaluator_ndcg = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg_cut_3','ndcg_cut_10','ndcg_cut_100','ndcg_cut_1000'})
    results_ndcg = evaluator_ndcg.evaluate(run)


    results = {}
    for qid, _ in results_ndcg.items():
        results[qid]={}
        for measure, score in results_ndcg[qid].items():
            results[qid][mapping[measure]]=score


    if "cast-20.qrels" in args.qrel:
        # use relevance scale ≥ 2 as positive for MRR on CAsT-20
        for q_id, pid_rel in qrel.items():
            for p_id, rel in pid_rel.items():
                if int(rel) == 0:
                    qrel[q_id][p_id] = 0
                elif int(rel) == 1:
                    qrel[q_id][p_id] = 0
                elif int(rel) >= 2:
                    qrel[q_id][p_id] = 1
                else:
                    raise Exception

    evaluator_general = pytrec_eval.RelevanceEvaluator(qrel, {'map_cut_10','map_cut_100', 'map_cut_1000','recall_5', 'recall_100','recall_1000'})
    results_general = evaluator_general.evaluate(run)

    for qid, _ in results.items():
        for measure, score in results_general[qid].items():
            results[qid][mapping[measure]] = score


    evaluator_rr = pytrec_eval.RelevanceEvaluator(qrel, {'recip_rank'})
    results_rr_5 = evaluator_rr.evaluate(run_5)
    results_rr_10 = evaluator_rr.evaluate(run_10)
    results_rr_100 = evaluator_rr.evaluate(run_100)

    for qid, _ in results.items():
        results[qid][mapping["mrr_5"]] = results_rr_5[qid]['recip_rank']
        results[qid][mapping["mrr_10"]] = results_rr_10[qid]['recip_rank']
        results[qid][mapping["mrr_100"]] = results_rr_100[qid]['recip_rank']


    for measure in mapping.values():
        overall = pytrec_eval.compute_aggregated_measure(measure, [result[measure] for result in results.values()])
        print('{}: {:.4f}'.format(measure, overall))


    run_name = args.run.split("/")[-1]
    dataset_name, pure_run_name, tail = run_name.split(".")

    if "cast" in dataset_name:
        output_path = f'datasets/cast-19-20/actual_performance/{dataset_name}.actual-performance-{pure_run_name}.json'
        if not os.path.exists(f"datasets/cast-19-20/actual_performance/"):
            os.makedirs(f"datasets/cast-19-20/actual_performance/")
    elif "or-quac" in dataset_name:
        output_path = f'datasets/or-quac/actual_performance/{dataset_name}.actual-performance-{pure_run_name}.json'
        if not os.path.exists(f"datasets/or-quac/actual_performance/"):
            os.makedirs(f"datasets/or-quac/actual_performance/")
    else:
        NotImplementedError



    f = open(output_path, 'w')
    f.write(json.dumps(results))
    f.close()

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--run',type=str,required=True)
    parser.add_argument('--qrel',type=str,required=True)

    args = parser.parse_args()
    evaluation(args)
