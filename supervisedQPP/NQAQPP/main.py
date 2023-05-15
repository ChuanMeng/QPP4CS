import sys
sys.path.append('./')
import json
import torch
from supervisedQPP.NQAQPP.dataset import Dataset, collate_fn
from supervisedQPP.NQAQPP.model import NQAQPP
from supervisedQPP.NQAQPP.driver import Driver
from utils import replicability
import argparse
import os
import time
from transformers import get_constant_schedule

def training(args):

    last_epoch = 0

    with open(args.checkpoint_path_ + "record.json", 'w', encoding='utf-8') as w:
        json.dump({"trained": [],"inferred":[]}, w)

    if not args.cross_validate:
        model = NQAQPP(args)

        if args.warm_up_path is not None:
            print(f"warm up from {args.warm_up_path}")
            model.load_state_dict(torch.load(args.warm_up_path))

        model_optimizer = torch.optim.Adam(model.parameters(), args.lr)
        driver = Driver(args, model, None, None)
        model_optimizer.zero_grad()

        dataset = Dataset(args, None)

        for i in range(last_epoch+1, args.epoch_num+1):
            driver.training(dataset, collate_fn, i, model_optimizer)
            driver.serialize(args.checkpoint_path_,i)
    else:
        for fold_id_inference in range(1, args.fold_num + 1):
            fold_ids_training = [j for j in range(1, args.fold_num + 1) if j != fold_id_inference]
            print(f"Training the model {fold_id_inference} with folds {fold_ids_training}.")

            model = NQAQPP(args)

            if args.warm_up_path is not None:
                print(f"warm up from {args.warm_up_path}")
                model.load_state_dict(torch.load(args.warm_up_path))

            model_optimizer = torch.optim.Adam(model.parameters(), args.lr)
            driver = Driver(args, model, fold_id_inference, fold_ids_training)
            model_optimizer.zero_grad()  # Clears the gradients of all optimized torch.Tensor s.

            dataset = Dataset(args, fold_ids_training)

            for epoch_id in range(last_epoch + 1, args.epoch_num + 1):
                driver.training(dataset, collate_fn, epoch_id, model_optimizer)
                driver.serialize(args.checkpoint_path_, epoch_id)


def inference(args):

    model = NQAQPP(args)

    for epoch_id in range(1,args.epoch_num+1):
        print("**************************************")
        print(f"infer epoch {epoch_id}")

        if not args.cross_validate:
            dataset = Dataset(args, None)
            checkpoint_name = args.checkpoint_path_ + str(epoch_id) + '.pkl'
            model.load_state_dict(torch.load(checkpoint_name))
            driver = Driver(args, model, None, None)
            driver.inference(dataset, collate_fn, epoch_id)
        else:
            key_list_global=[]
            output_list_global=[]

            for fold_id in range(1, args.fold_num+1):
                fold_id_inference = fold_id
                dataset = Dataset(args, [fold_id_inference])

                checkpoint_name = args.checkpoint_path_ + str(epoch_id) +'.'+str(fold_id_inference)+'.pkl'
                model.load_state_dict(torch.load(checkpoint_name))
                driver = Driver(args, model, fold_id_inference, None)

                key_list, output_list = driver.inference_cv(dataset, collate_fn, epoch_id)
                key_list_global+=key_list
                output_list_global+=output_list

            driver.merge(key_list_global, output_list_global, epoch_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='NQAQPP')
    parser.add_argument("--mode", type=str, default='training')
    parser.add_argument("--cross_validate", action='store_true')
    parser.add_argument("--fold_num", type=int, default=5)
    parser.add_argument("--target_metric", type=str, default='ndcg@3')

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
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--lr", type=float, default= 2e-5)
    parser.add_argument("--batch_size", type=int, default=4) #  32, 64, 128
    parser.add_argument("--clip", type=float, default=1.)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--d", type=int, default=100) # [10, 20, 50, 100]
    parser.add_argument("--l", type=int, default=768)

    parser.add_argument("--interval", type=int, default=10)

    args = parser.parse_args()

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

    replicability(seed=args.random_seed)

    print("torch_version:{}".format(torch.__version__))
    print("CUDA_version:{}".format(torch.version.cuda))
    print("cudnn_version:{}".format(torch.backends.cudnn.version()))

    if args.mode == 'inference':
        inference(args)
    elif args.mode == 'training':
        training(args)
    else:
        Exception("no ther mode")