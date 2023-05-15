from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pytorch_transformers import BertForTokenClassification, BertTokenizer
import spacy
import torch
import json
import argparse
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from spacy.lang.en import English
from tqdm import tqdm

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.split(' ')
        tokens = []
        labels = []
        valid = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            for m in range(len(token)):
                if m == 0:
                    valid.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        # mask out labels for current turn.
        while len(input_ids) < max_seq_length:
            input_ids.append(tokenizer.pad_token_id)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(valid) == max_seq_length
        return torch.tensor([input_ids], dtype=torch.long), torch.tensor([input_mask], dtype=torch.long), torch.tensor(
            [valid], dtype=torch.long), torch.tensor([segment_ids], dtype=torch.long)


class Ner(BertForTokenClassification):
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,attention_mask_label=None, device='cuda'):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=device)

        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            # attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class QuReTeCRewriter:
    def __init__(self, model_path):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = Ner.from_pretrained(model_path).to(device).eval()
        self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
        self.nlp = English()
        self.history = []
        self.history_query = []



    def rewrite(self, query: str, response: Optional[str] = None, response_num: Optional[int] = 0):

        self.history_query += [query]
        self.history += [query]

        if len(self.history_query) == 1:
            if response:
                self.history += [response]
            return query

        if response_num != 0:
            input_list = self.history_query[:-response_num] + self.history[-2 * response_num:]
        else:
            input_list = self.history_query


        qtext = " ".join([tok.text for tok in self.nlp(input_list[-1])])
        htext = " ".join([tok.text for tok in self.nlp(" ".join(input_list[:-1]))])

        src_text = "{} [SEP] {}".format(htext, qtext)

        input_id, mask, valid, tt_ids = convert_examples_to_features([src_text], 300, self.tokenizer)
        tokens = self.tokenizer.convert_ids_to_tokens(input_id.reshape(-1).numpy())
        logits = self.model(input_id.to(self.device), tt_ids.to(self.device), attention_mask=mask.to(self.device), valid_ids=valid.to(self.device), device= self.device)
        pred = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        post_processed_tokens = []

        for t, v in zip(tokens, valid.reshape(-1).numpy().tolist()):
            if t == "[SEP]":
                break
            if v == 1:
                post_processed_tokens.append(t)
            else:
                post_processed_tokens[-1] += t.replace("##", '')
        tokens_to_add = [t for t, p in zip(post_processed_tokens, pred.reshape(-1).cpu().numpy()) if p == 2]

        if response:
            self.history += [response]

        return query + " " + " ".join(set(tokens_to_add)) if len(tokens_to_add)>0 else query

    def reset_history(self):
        self.history = []
        self.history_query = []



class T5QueryRewriter:
    def __init__(self, max_length=64, num_beams=10, early_stopping=True):

        self.nlp = English()

        self.history = []
        self.history_query = []

        self.max_length = max_length
        self.num_beams =num_beams
        self.early_stopping = early_stopping

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained("castorini/t5-base-canard", truncation_side="left")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("castorini/t5-base-canard")
        self.model.to(device).eval()

    def rewrite(self, query: str, response: Optional[str] = None, response_num: Optional[int] = 0) -> str:
        self.history_query += [query]
        self.history += [query]

        if response_num != 0:
            src_text = " ||| ".join(self.history_query[:-response_num] + self.history[-2 * response_num:])
        else:
            src_text = " ||| ".join(self.history_query)

        src_text = " ".join([tok.text for tok in self.nlp(src_text)])
        input_ids = self.tokenizer(src_text, return_tensors="pt", add_special_tokens=True, truncation=True).input_ids.to(self.device)

        output_ids = self.model.generate(
            input_ids,
            max_length=self.max_length,
            num_beams=self.num_beams,
            early_stopping=self.early_stopping,
        )

        rewrite_text = self.tokenizer.decode(
            output_ids[0],
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True,
        )

        if response:
            self.history += [response]
            
        return rewrite_text

    def reset_history(self):
        self.history = []
        self.history_query = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="or-quac")
    parser.add_argument("--rewriter", type=str, default="QuReTeC")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--response_num", type=int)

    args = parser.parse_args()

    if args.rewriter=="T5":
        rewriter = T5QueryRewriter()
    elif args.rewriter =="QuReTeC":
        rewriter = QuReTeCRewriter(args.model_path)
    else:
        raise NotImplementedError

    if args.dataset == "cast-19":
        with open("datasets/cast-19-20/raw/evaluation_topics_v1.0.json", "r") as fin:
            raw_data = json.load(fin)

        out_automatic_queries = open(f"datasets/cast-19-20/queries/{args.dataset}.queries-{args.rewriter}-Q.tsv", "w")

        for conversations in tqdm(raw_data):
            topic_number = str(conversations['number'])
            for turn in conversations['turn']:
                query_number, raw_utterance = str(turn['number']), turn['raw_utterance']
                #print("{}_{}".format(topic_number, query_number))
                automatic_query= rewriter.rewrite(raw_utterance, None, 0)
                out_automatic_queries.write("{}_{}\t{}\n".format(topic_number, query_number, automatic_query))
            rewriter.reset_history()
        out_automatic_queries.close()

    elif args.dataset=="cast-20":
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher("datasets/cast-19-20/index")

        suffix = "QA" if args.response_num > 0 else "Q"

        with open("datasets/cast-19-20/raw/2020_automatic_evaluation_topics_v1.0.json", "r") as fin:
            raw_data = json.load(fin)
            out_automatic_queries = open(f"datasets/cast-19-20/queries/{args.dataset}.queries-{args.rewriter}-{suffix}.tsv", "w")

            for conversations in tqdm(raw_data):
                topic_number = str(conversations['number'])
                for turn in conversations['turn']:
                    query_number, raw_utterance, automatic_canonical_result_id = str(turn['number']), turn['raw_utterance'], turn['automatic_canonical_result_id']
                    #print("{}_{}".format(topic_number, query_number))
                    doc = searcher.doc(automatic_canonical_result_id)
                    json_doc = json.loads(doc.raw())
                    response = json_doc['contents']
                    automatic_query  = rewriter.rewrite(raw_utterance, response, args.response_num)
                    out_automatic_queries.write("{}_{}\t{}\n".format(topic_number, query_number, automatic_query))

                rewriter.reset_history()
            out_automatic_queries.close()

    elif "or-quac" in args.dataset:
        suffix="QA" if args.response_num>0 else "Q"
        subset=args.dataset.split("-")[-1]
        print(f"load {args.dataset}")
        with open(f"datasets/or-quac/raw/{subset}.txt", "r") as f, open(f"datasets/or-quac/queries/{args.dataset}.queries-{args.rewriter}-{suffix}.tsv", "w") as h:
            responses = []
            last_dialog_id = None

            for line in tqdm(f):
                obj = json.loads(line)
                qid = obj['qid']
                raw_utterance = obj["question"]
                dialog_id = qid[:qid.rfind('#')]
                response = obj["answer"]["text"]

                if dialog_id != last_dialog_id:
                    last_dialog_id = dialog_id
                    rewriter.reset_history()

                #print(f"{qid}")
                automatic_query = rewriter.rewrite(raw_utterance, response, args.response_num)
                h.write(f"{qid}\t{automatic_query}\n")