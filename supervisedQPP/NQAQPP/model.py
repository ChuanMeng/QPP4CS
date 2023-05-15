import torch.nn.functional as F
import torch
from transformers import BertModel
import numpy as np


class NQAQPP(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        nn_dropout = torch.nn.Dropout(self.args.dropout)
        nn_relu = torch.nn.ReLU()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.sigmoid = torch.nn.Sigmoid()

        self.ffn1 = torch.nn.Sequential(
            torch.nn.Linear(2 * self.args.k - 1, self.args.d),
            nn_relu,
            nn_dropout,
            torch.nn.Linear(self.args.d, self.args.d),
        )

        self.ffn2 = torch.nn.Sequential(
            torch.nn.Linear(self.args.l, self.args.d),
            nn_relu,
            nn_dropout,
            torch.nn.Linear(self.args.d, self.args.d),
        )

        self.ffn3_1 = torch.nn.Sequential(
            torch.nn.Linear(self.args.l, self.args.d),
            nn_relu,
            nn_dropout,
            torch.nn.Linear(self.args.d, self.args.d),
        )

        self.ffn3_2 = torch.nn.Sequential(
            torch.nn.Linear(self.args.k * self.args.d, self.args.d),
            nn_relu,
            nn_dropout,
            torch.nn.Linear(self.args.d, self.args.d),
        )

        self.ffn4 = torch.nn.Sequential(
            torch.nn.Linear(self.args.d + (2 * self.args.k - 1) + self.args.d + self.args.l + self.args.d + self.args.d, self.args.d),
            nn_relu,
            nn_dropout,
            torch.nn.Linear(self.args.d, 1),
        )

    def forward(self, data):
        # data['RS'] [batch, 2k-1]

        assert data["qa_input_ids"].size()[1]==self.args.k
        assert data["qa_input_ids"].size()[2] == 512
        batch_size =  data["qa_input_ids"].size()[0]

        ffn1_out = self.ffn1(data['RS']) # [batch, d]

        q_res = self.bert(data["q_input_ids"], data["q_attention_mask"], data["q_token_type_ids"]).last_hidden_state[:, 0]

        ffn2_out = self.ffn2(q_res) # [batch, d]

        qa_res = self.bert(data["qa_input_ids"].reshape(batch_size * self.args.k, 512),
                  data["qa_attention_mask"].reshape(batch_size * self.args.k, 512),
                  data["qa_token_type_ids"].reshape(batch_size * self.args.k, 512)).last_hidden_state[:, 0] # [batch*k, l]

        ffn3_1_out = self.ffn3_1(qa_res)  # [batch*k, d]
        ffn3_1_out=ffn3_1_out.reshape(batch_size, self.args.k, self.args.d)
        ffn3_1_out=ffn3_1_out.reshape(batch_size, self.args.k*self.args.d) # [batch, k*d]

        ffm3_2_out = self.ffn3_2(ffn3_1_out) # [batch, d]

        pp = self.ffn4(torch.cat([ffn1_out, data['RS'], ffn2_out, q_res, ffm3_2_out, ffn1_out * ffn2_out * ffm3_2_out], dim=-1))
        # [batch, 1]

        if self.args.mode=="training":
            #print(pp.view(-1))
            #print(data["ap"].view(-1))
            #print("======")
            return self.loss(pp.view(-1), data["ap"].view(-1))

        elif self.args.mode=="inference":
            return self.sigmoid(pp.view(-1)) # [batch]


