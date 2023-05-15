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
from scipy import stats
from utils import data_split

class Dataset(Dataset):
    def __init__(self, args, fold_ids):
        super(Dataset, self).__init__()

        self.args = args
        self.fold_ids = fold_ids
        self.input = []
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
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

            if qid not in qrels:
                #print(f"skip {qid}")
                continue

            if self.args.cross_validate:
                if qid not in qid_in_folds:
                    #print(f"skip {key}")
                    continue

            num_q_real+=1

            pid_k = [pid for (pid, score) in sorted(run[qid].items(), key=lambda x: x[1], reverse=True)[:self.args.k]]
            score_k = [score for (pid, score) in sorted(run[qid].items(), key=lambda x: x[1], reverse=True)[:self.args.k]]

            score_zscore = torch.from_numpy(stats.zscore(score_k)) # k
            score_std = torch.zeros(self.args.k-1) # k-1

            for i in range(0, self.args.k - 1):
                #print(i)
                #print(score_k[0:i + 2])
                score_std[i] = torch.tensor(np.std(score_k[0:i + 2]))

            #print(score_std)
            rs=torch.cat([score_zscore, score_std]) # 2k-1
            #print(rs)
            #print(qid, pid_k)
            q = self.tokenizer(query[qid], padding='max_length', truncation=True, return_tensors='pt', max_length=512)
            qa = self.tokenizer([query[qid] for _ in pid_k], [json.loads(self.searcher.doc(pid).raw())['contents'] for pid in pid_k], padding='max_length', truncation='only_second', return_tensors='pt', max_length=512)
            #print(self.tokenizer.batch_decode(q["input_ids"]))
            #for ids in qa["input_ids"]:
            #print(self.tokenizer.batch_decode(qa["input_ids"]))

            #print(q["input_ids"].size()) # [1, 512]
            #print(qa["input_ids"].size()) # [5, 512]
            self.input.append([qid, rs, q["input_ids"], q["attention_mask"], q["token_type_ids"], qa["input_ids"], qa["attention_mask"], qa["token_type_ids"], torch.tensor(float(ap_bank[qid][self.args.target_metric]))])


        print(f"process {num_q_real} out of {num_q} queries.")

    def __getitem__(self, index):
        qid, rs, q_input_ids, q_attention_mask, q_token_type_ids, qa_input_ids, qa_attention_mask, qa_token_type_ids, ap = self.input[index]
        return [qid, rs, q_input_ids, q_attention_mask, q_token_type_ids, qa_input_ids, qa_attention_mask, qa_token_type_ids, ap]

    def __len__(self):
        return len(self.input)

def collate_fn(data):
    qid, rs, q_input_ids, q_attention_mask, q_token_type_ids, qa_input_ids, qa_attention_mask, qa_token_type_ids, ap = zip(*data)


    return {'qid': qid, #  tuple(batch)
            'RS': torch.stack(rs).to(torch.float32),  # [batch * (2k-1)]
            'q_input_ids': torch.cat(q_input_ids,0), # [batch, 512]
            'q_attention_mask':torch.cat(q_attention_mask,0),  # [batch, 512]
            'q_token_type_ids':torch.cat(q_token_type_ids,0),  # [batch, 512]
            'qa_input_ids': torch.stack(qa_input_ids),  # [batch, k, 512]
            'qa_attention_mask': torch.stack(qa_attention_mask),  # [batch, k, 512]
            'qa_token_type_ids': torch.stack(qa_token_type_ids),  # [batch, k, 512]
            'ap': torch.tensor(ap), # [batch]
            }


