from torch.utils.data import Dataset
import numpy as np
import torch
import random
#from transformers import BertTokenizer
from pyserini.search.lucene import LuceneSearcher
import pytrec_eval
import json
import os
from utils import data_split

class Dataset(Dataset):
    def __init__(self, args, fold_ids):
        super(Dataset, self).__init__()

        self.args = args
        self.input = {}
        self.fold_ids = fold_ids  # load which fold?
        self.searcher = LuceneSearcher(args.index_path)

        data_path =  args.checkpoint_path_+ f"data.{self.args.setup}.pkl"

        if fold_ids is not None and args.cross_validate:
            fold_ids = [str(id) for id in fold_ids]
            fold_ids_text = "-".join(fold_ids)
            data_path = args.checkpoint_path_ + f"data.{self.args.setup}-{fold_ids_text}.pkl"

        if os.path.exists(data_path):
            self.input = torch.load(data_path)
        else:
            self.load()
            torch.save(self.input, data_path)

    def load(self):
        query = {}
        query_reader = open(self.args.query_path, 'r').readlines()
        for line in query_reader:
            qid, qtext = line.split('\t')
            query[qid] = qtext

        with open(self.args.qrels_path, 'r') as f_qrels:
            qrels = pytrec_eval.parse_qrel(f_qrels)

        with open(self.args.actual_performance_path, 'r') as ap_r:
            ap_bank = json.loads(ap_r.read())

        with open(self.args.run_path, 'r') as f_run:
            run = pytrec_eval.parse_run(f_run)

        num_q = len(list(run))
        cur_q =0
        #num_skip=0

        num_q_real = 0

        if self.args.cross_validate:
            qid_in_folds=[]
            foldid2qid=data_split(self.args.dataset_name)
            for fold_id in self.fold_ids:
                qid_in_folds+=foldid2qid[fold_id]

            print(f"fold_ids: {self.fold_ids}")
            print(f"qid_in_folds: {qid_in_folds}")

        for qid in run.keys():
            cur_q += 1
            #print(f"qid:{cur_q}/{num_q}")

            if qid not in qrels and self.args.mode=="training":
                #print(f"skip {qid}")
                continue

            if self.args.cross_validate:
                if qid not in qid_in_folds:
                    #print(f"skip {qid}")
                    continue

            num_q_real += 1

            self.input[qid]={}
            self.input[qid]["qtext"]=query[qid]

            run[qid] = sorted(run[qid].items(), key=lambda x: x[1], reverse=True)
            first_pid = run[qid][0][0]
            self.input[qid]["doc_text"] = json.loads(self.searcher.doc(first_pid).raw())['contents']

            if self.args.mode=="training":
                self.input[qid]["ap"] = float(ap_bank[qid][self.args.target_metric])

        print(f"process {num_q_real} out of {num_q} queries.")

    def __getitem__(self, index):
        return None

    def __len__(self):
        return None


