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

        data_loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=None, batch_sampler=None, shuffle=False)

        step = 0
        count = 0
        loss_display = 0
        qid_list=[]
        for j, data in enumerate(data_loader, 0):
            if torch.cuda.is_available():
                data_cuda = dict()
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data_cuda[key] = value.cuda()
                    else:
                        data_cuda[key] = value
                data = data_cuda

            count += 1
            qid_list.append(data['qid'])

            loss= self.model(data)
            loss = loss/25
            loss_display += loss.item()
            loss.backward()

            if count == 25:
                assert len(set(qid_list)) == 1
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                if self.args.cross_validate:
                    print(f'Cross-validation training {self.args.setup} with the folds {self.fold_ids_training}')
                else:
                    print(f'Training: {self.args.setup}')

                print(f'Epoch:{epoch}, Step:{step}, Loss:{loss_display}, LR:{self.args.lr}')

                count = 0
                loss_display = 0
                qid_list=[]



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
            data_loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=None, batch_sampler=None, shuffle=False)

            w_chunk = open(self.args.checkpoint_path_ + f"chunk.{self.args.setup}-{str(epoch)}", 'a')
            w_chunk_detail = open(self.args.checkpoint_path_+ f"chunk_detailed.{self.args.setup}-{str(epoch)}", 'a')

            chunk = ''
            chunk_detailed = ''

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

                reldocs = self.model(data).item()

                c = 0
                for _ in data["window"]:
                    chunk_detailed += data["qid"] + '\t' + str(data["window"][c]) + '\t' + str(c + 1)  # chunk
                    if c == 0:
                        chunk_detailed += '\t' + str(data['win_count']) + '\t' + str(reldocs) + '\n'
                    else:
                        chunk_detailed += '\n'
                    c += 1

                chunk += data["qid"] + '\t' + str(data['win_count']) + '\t' + str(reldocs) + '\n'  # overall

                if index % 100 == 0:
                    w_chunk.write(chunk)
                    w_chunk_detail.write(chunk_detailed)
                    chunk = ''
                    chunk_detailed = ''

            w_chunk.write(chunk)
            w_chunk_detail.write(chunk_detailed)
            w_chunk.close()
            w_chunk_detail.close()

            # write the final predicted performance file
            pp = {}

            chunk = csv.reader(open(self.args.checkpoint_path_ + f"chunk.{self.args.setup}-{str(epoch)}",'r'), delimiter='\t')
            pp_w = open(self.args.output_path_ + "-"+str(epoch), 'a')

            qid = ''
            weighted = 0


            for line in chunk:
                if qid == '' or line[0] == qid:
                    qid = line[0]
                    weighted += float(1 / float(line[1])) * float(line[2])
                else:
                    pp[qid] = weighted

                    weighted = 0
                    qid = line[0]
                    weighted += float(1 / float(line[1])) * float(line[2])

            pp[qid] = weighted

            # min-max normalize
            norm = ''
            for qid, one_pp in pp.items():
                if max(pp.values()) - min(pp.values())==0:
                    one_pp_norm=0
                else:
                    one_pp_norm = round((float(one_pp) - min(pp.values())) /(max(pp.values()) - min(pp.values())), 5)

                norm += qid + '\t' + str(one_pp_norm) + '\n'

            pp_w.write(norm)
            pp_w.close()


    def inference_cv(self, dataset, collate_fn, epoch):
        self.model.eval()

        key_list=[]
        output_list=[]

        with torch.no_grad():
            data_loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, batch_size=None, batch_sampler=None, shuffle=False)

            w_chunk = open(self.args.checkpoint_path_ + str(epoch)+"."+str(self.fold_id_inference)+"-" +"chunk", 'a')
            w_chunk_detail = open(self.args.checkpoint_path_+ str(epoch)+"."+ str(self.fold_id_inference) +"-"+"chunk_detailed", 'a')

            chunk = ''
            chunk_detailed = ''

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

                reldocs = self.model(data).item()

                c = 0
                for _ in data["window"]:
                    chunk_detailed += data["qid"] + '\t' + str(data["window"][c]) + '\t' + str(c + 1)  # chunk
                    if c == 0:
                        chunk_detailed += '\t' + str(data['win_count']) + '\t' + str(reldocs) + '\n'
                    else:
                        chunk_detailed += '\n'
                    c += 1

                chunk += data["qid"] + '\t' + str(data['win_count']) + '\t' + str(reldocs) + '\n'  # overall

                if index % 100 == 0:
                    w_chunk.write(chunk)
                    w_chunk_detail.write(chunk_detailed)
                    chunk = ''
                    chunk_detailed = ''

            w_chunk.write(chunk)
            w_chunk_detail.write(chunk_detailed)
            w_chunk.close()
            w_chunk_detail.close()

            chunk = csv.reader(open(self.args.checkpoint_path_ + str(epoch)+"."+str(self.fold_id_inference)+"-" +"chunk", 'r'), delimiter='\t')

            pp = {}

            qid = ''
            weighted = 0

            for line in chunk:
                if qid == '' or line[0] == qid:
                    qid = line[0]
                    weighted += float(1 / float(line[1])) * float(line[2])
                else:
                    pp[qid] = weighted
                    weighted = 0
                    qid = line[0]
                    weighted += float(1 / float(line[1])) * float(line[2])

            pp[qid] = weighted

            for qid, score in pp.items():
                key_list.append(qid)
                output_list.append(score)

            return key_list, output_list


    def merge(self, key_list, output_list, epoch):
        self.model.eval()

        with torch.no_grad():

            pp_w = open(self.args.output_path_ + "-" + str(epoch), 'a')
            norm = ''
            for idx, qid in enumerate(key_list):
                if max(output_list) - min(output_list)==0:
                    one_pp_norm=0
                else:
                    one_pp_norm = round((float(output_list[idx]) - min(output_list)) /(max(output_list) - min(output_list)), 5)
                norm += qid + '\t' + str(one_pp_norm) + '\n'

            pp_w.write(norm)
            pp_w.close()
