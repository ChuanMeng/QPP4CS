import sys
sys.path.append('./')

from supervisedQPP.BERTQPP.dataset import Dataset
from utils import replicability
from evaluation_QPP import evaluation

import json
import torch
import argparse
import os
import time
from transformers import get_constant_schedule
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader
import math

def training(args):

    for epoch_id in range(1,args.epoch_num+1):
        print("**************************************")
        print(f"train up to {epoch_id} epoch")
        replicability(seed=args.random_seed)

        if not args.cross_validate:
            dataset = Dataset(args, None)

            input_dict = dataset.input
            input_examples=[]

            for qid in input_dict:
                qtext = input_dict[qid]["qtext"]
                firstdoctext = input_dict[qid]["doc_text"]
                ap = input_dict[qid]["ap"]
                input_examples.append(InputExample(texts=[qtext, firstdoctext], label=ap))

            dataloader = DataLoader(input_examples, shuffle=True, batch_size=args.batch_size)

            if args.warm_up_path is None:
                model = CrossEncoder('bert-base-uncased', num_labels=1)
            else:
                print(f"warm up from {args.warm_up_path}")
                model = CrossEncoder(args.warm_up_path, num_labels=1)

            # Train the model
            model.fit(train_dataloader=dataloader,
                    epochs=epoch_id,
                    scheduler="constantlr",
                    optimizer_params={'lr': args.lr},
                    max_grad_norm= args.clip,
                    weight_decay=0.0,
                    output_path=args.checkpoint_path_+str(epoch_id))

            model.save(args.checkpoint_path_+str(epoch_id))
        else:
            for fold_id_inference in range(1, args.fold_num + 1):

                fold_ids_training = [j for j in range(1, args.fold_num + 1) if j != fold_id_inference]
                print(f"Training the model {fold_id_inference} with folds {fold_ids_training}.")

                dataset = Dataset(args, fold_ids_training)

                input_dict = dataset.input
                input_examples = []

                for qid in input_dict:
                    qtext = input_dict[qid]["qtext"]
                    firstdoctext = input_dict[qid]["doc_text"]
                    ap = input_dict[qid]["ap"]
                    input_examples.append(InputExample(texts=[qtext, firstdoctext], label=ap))

                dataloader = DataLoader(input_examples, shuffle=True, batch_size=args.batch_size)

                if args.warm_up_path is None:
                    model = CrossEncoder('bert-base-uncased', num_labels=1)
                else:
                    print(f"warm up from {args.warm_up_path}")
                    model = CrossEncoder(args.warm_up_path, num_labels=1)

                model.fit(train_dataloader=dataloader,
                        epochs=epoch_id,
                        scheduler="constantlr",
                        optimizer_params={'lr': args.lr},
                        max_grad_norm=args.clip,
                        weight_decay=0.0,
                        output_path=args.checkpoint_path_ + str(epoch_id)+"."+str(fold_id_inference))

                model.save(args.checkpoint_path_ + str(epoch_id)+"."+str(fold_id_inference))


def inference(args):

    for epoch_id in range(1,args.epoch_num+1):
        print("**************************************")
        print(f"infer epoch {epoch_id}")
        replicability(seed=args.random_seed)

        if not args.cross_validate:
            dataset = Dataset(args, None)
            input_dict = dataset.input

            sentences = []
            qids = []

            for qid in input_dict:
                sentences.append([input_dict[qid]["qtext"], input_dict[qid]["doc_text"]])
                qids.append(qid)

            model = CrossEncoder(args.checkpoint_path_ + str(epoch_id), num_labels=1)
            pp = model.predict(sentences)

            with open(args.output_path_ + "-" + str(epoch_id), 'w') as pp_w:
                for index, qid in enumerate(qids):
                    pp_w.write(qid + '\t' + str(pp[index]) + '\n')

        else:
            key_list_global = []
            output_list_global = []

            for fold_id_inference in range(1, args.fold_num + 1):
                print(f"infer {args.setup} for folder {fold_id_inference}")
                dataset = Dataset(args, [fold_id_inference])
                input_dict = dataset.input

                sentences = []
                qids = []

                for qid in input_dict:
                    sentences.append([input_dict[qid]["qtext"], input_dict[qid]["doc_text"]])
                    qids.append(qid)

                model = CrossEncoder(args.checkpoint_path_ + str(epoch_id) + "." + str(fold_id_inference), num_labels=1)
                pp = model.predict(sentences)

                for index, qid in enumerate(qids):
                    key_list_global.append(qid)
                    output_list_global.append(pp[index])

            with open(args.output_path_ + "-" + str(epoch_id), 'w') as pp_w:
                for index, qid in enumerate(key_list_global):
                    pp_w.write(qid + '\t' + str(output_list_global[index]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default='BERTQPP')
    parser.add_argument("--mode", type=str, default='training')
    parser.add_argument("--cross_validate", action='store_true')
    parser.add_argument("--fold_num", type=int, default=5)
    parser.add_argument("--target_metric", type=str, default="ndcg@3")

    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--warm_up_path", type=str, default=None)
    parser.add_argument("--query_path", type=str, default='')
    parser.add_argument("--index_path", type=str, default='')
    parser.add_argument("--qrels_path", type=str, default='')
    parser.add_argument("--run_path", type=str, default='')
    parser.add_argument("--actual_performance_path", type=str, default='')

    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--epoch_num", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--clip", type=float, default=1.)

    parser.add_argument("--interval", type=int, default=10)

    args = parser.parse_args()

    print("torch_version:{}".format(torch.__version__))
    print("CUDA_version:{}".format(torch.version.cuda))
    print("cudnn_version:{}".format(torch.backends.cudnn.version()))

    args.dataset_class = args.query_path.split("/")[-3]
    args.dataset_name = args.query_path.split("/")[-1].split(".")[0]
    args.query_type = "-".join(args.query_path.split("/")[-1].split(".")[1].split("-")[1:])
    args.retriever = "-".join(args.run_path.split("/")[-1].split(".")[1].split("-")[1:])

    if args.warm_up_path is not None:
        args.setup = f"{args.dataset_name}.{args.retriever}.{args.query_type}-{args.name}-{args.target_metric}-warm-up"
    else:
        args.setup = f"{args.dataset_name}.{args.retriever}.{args.query_type}-{args.name}-{args.target_metric}"

    if args.checkpoint_name is not None:
        args.checkpoint_path_ = f"{args.checkpoint_path}/{args.checkpoint_name}/"
    else:
        args.checkpoint_path_ = f"{args.checkpoint_path}/{args.setup}/"

    args.output_path_ = f"{args.output_path}/{args.setup}"

    if not os.path.exists(args.checkpoint_path_):
        os.makedirs(args.checkpoint_path_)

    if args.mode == 'inference':
        inference(args)
    elif args.mode == 'training':
        training(args)
    else:
        Exception("no ther mode")