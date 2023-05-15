from torch.utils.data import Dataset
import numpy as np
import torch
import random
from transformers import BertTokenizer
from pyserini.search.lucene import LuceneSearcher
import pytrec_eval
import more_itertools
import json
import os
from utils import data_split

class Dataset(Dataset):
    def __init__(self, args, fold_ids):
        super(Dataset, self).__init__()

        self.args = args
        self.fold_ids = fold_ids
        self.input = []
        self.tokeniser = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
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


        with open(self.args.run_path, 'r') as f_run:
            run = pytrec_eval.parse_run(f_run)

        num_q = len(list(run))
        cur_q =0

        num_q_real=0

        if self.args.cross_validate:
            qid_in_folds=[]
            foldid2qid=data_split(self.args.dataset_name)
            for fold_id in self.fold_ids:
                qid_in_folds+=foldid2qid[fold_id]

            print(f"fold_ids: {self.fold_ids}")
            print(f"qid_in_folds: {qid_in_folds}")

        for qid in run.keys():
            cur_q+=1
            #print(f"qid:{cur_q}/{num_q}")

            if qid not in qrels and self.args.mode=="training":
                #print(f"skip {qid}")
                continue

            if self.args.cross_validate:
                if qid not in qid_in_folds:
                    #print(f"skip {key}")
                    continue

            num_q_real+=1

            pid_list = [pid for (pid, score) in sorted(run[qid].items(), key=lambda x: x[1], reverse=True)[:100]]
            #pid_list = [pid for pid in run[qid].keys()][:100]
            #print("[pid_list]",pid_list)

            assert len(pid_list)==100

            windows = list(more_itertools.windowed(pid_list, n=4, step=4))
            assert len(windows)==25
            #print("[windows]",windows,"\n")
            win_count = 0

            for window in windows:  # 250
                # one window
                #print("[current window]",window)
                if win_count < 25:  # 25
                    assert len(window) ==4
                    pos_list = [win_count * 4 + i for i in range(0, 4)]
                    #print("[pos_list]",pos_list)
                    pid_rel = [pid_judged for (pid_judged, rel) in qrels[qid].items()  if pid_judged in window and int(rel)>0]  # [relevant document ids]
                    #print("[qrels[qid]]", qrels[qid])
                    #print("[pid_rel]",pid_rel)
                    numrel = len(pid_rel)
                    #print("[numrel]",numrel)
                    win_count += 1
                    query_passage = self.tokeniser([query[qid] for _ in window], [json.loads(self.searcher.doc(pid).raw())['contents'] for pid in window], padding=True, truncation='only_second', return_tensors='pt')

                    self.input.append([qid, window, query_passage["input_ids"], query_passage["attention_mask"], query_passage["token_type_ids"], torch.tensor([numrel]), torch.tensor(pos_list), win_count])

                elif win_count >=25:
                    raise Exception("win_count should smaller than 25")

        print(f"process {num_q_real} out of {num_q} queries.")

    def __getitem__(self, index):
        qid, window, input_ids, attention_mask, token_type_ids, numrel, pos_list, win_count = self.input[index]
        return [qid, window, input_ids, attention_mask, token_type_ids, numrel, pos_list, win_count]

    def __len__(self):
        return len(self.input)

def collate_fn(data):
    qid, window, input_ids, attention_mask, token_type_ids, numrel, pos_list, win_count = data

    return {'qid':qid,
            'window': window,
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'token_type_ids':token_type_ids,
            'numrel': numrel,
            'pos_list': pos_list,
            'win_count': win_count,
            }


