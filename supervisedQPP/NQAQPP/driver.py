import sys
sys.path.append('./')
import json
import os
from collections import defaultdict
import torch
from utils import rounder
import csv
from evaluation_QPP import evaluation

class Driver(object):
    def __init__(self, args, model, fold_id_inference, fold_ids_training):
        super(Driver, self).__init__()
        self.args = args
        self.fold_id_inference = fold_id_inference
        self.fold_ids_training = fold_ids_training

        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model


    def training(self, dataset, collate_fn, epoch, optimizer):
        self.model.train()

        data_loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=self.args.batch_size, shuffle=True)

        step = 0
        loss_display = 0

        for j, data in enumerate(data_loader, 0):
            if torch.cuda.is_available():
                data_cuda = dict()
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data_cuda[key] = value.cuda()
                    else:
                        data_cuda[key] = value
                data = data_cuda


            loss= self.model(data)
            loss_display += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % self.args.interval == 0:
                if self.args.cross_validate:
                    print(f'Cross-validation training {self.args.setup} with the folds {self.fold_ids_training}')
                else:
                    print(f'Training: {self.args.setup}')

                print(f'Epoch:{epoch}, Step:{step}, Loss:{loss_display/self.args.interval}, LR:{self.args.lr}')

                loss_display = 0


    def serialize(self, checkpoint_path, epoch_id):

        if self.args.cross_validate:
            torch.save(self.model.state_dict(), os.path.join(checkpoint_path, '.'.join([str(epoch_id), str(self.fold_id_inference), 'pkl'])))
            print("Saved the model (for the fold id {}) trained on epoch {} ".format(self.fold_id_inference,epoch_id))
        else:
            torch.save(self.model.state_dict(), os.path.join(checkpoint_path, '.'.join([str(epoch_id), 'pkl'])))
            print("Saved the model trained on epoch {} ".format(epoch_id))

        with open(checkpoint_path + "record.json", 'r', encoding='utf-8') as r:
            record = json.load(r)

            if self.args.cross_validate:
                record["trained"].append(str(epoch_id)+"."+str(self.fold_id_inference))
            else:
                record["trained"].append(str(epoch_id))

        with open(checkpoint_path + "record.json", 'w', encoding='utf-8') as w:
            json.dump(record, w)

    def inference(self, dataset, collate_fn, epoch):
        self.model.eval()

        with torch.no_grad():
            data_loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=self.args.batch_size, shuffle=False)

            pp_w = open(self.args.output_path_ + "-" + str(epoch), 'w')
            for index, data in enumerate(data_loader, 0):
                if (index+1) % self.args.interval ==0 or (index+1)==1:
                    print("{}: doing {} / total {} in epoch {}".format(self.args.setup, index+1, len(data_loader), str(epoch)))

                if torch.cuda.is_available():
                    data_cuda = dict()
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            data_cuda[key] = value.cuda()
                        else:
                            data_cuda[key] = value
                    data = data_cuda

                pp = self.model(data).tolist()

                for index, qid in enumerate(data["qid"]):
                    pp_w.write(qid + '\t' + str(pp[index]) + '\n')

            pp_w.close()


    def inference_cv(self, dataset, collate_fn, epoch):
        self.model.eval()

        key_list=[]
        output_list=[]

        with torch.no_grad():
            data_loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=self.args.batch_size, shuffle=False)

            for index, data in enumerate(data_loader, 0):
                if (index+1) % self.args.interval ==0 or (index+1)==1:
                    print("{}: doing {} / total {} in epoch {} for folder {}".format(self.args.setup, index+1, len(data_loader), str(epoch), str(self.fold_id_inference)))

                if torch.cuda.is_available():
                    data_cuda = dict()
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            data_cuda[key] = value.cuda()
                        else:
                            data_cuda[key] = value
                    data = data_cuda

                pp = self.model(data).tolist()

                for index, qid in enumerate(data["qid"]):
                    key_list.append(qid)
                    output_list.append(pp[index])

            return key_list, output_list


    def merge(self, key_list, output_list, epoch):
        self.model.eval()

        with torch.no_grad():

            with open(self.args.output_path_ + "-" + str(epoch), 'w') as pp_w:
                for index, qid in enumerate(key_list):
                    pp_w.write(qid + '\t' + str(output_list[index]) + '\n')
