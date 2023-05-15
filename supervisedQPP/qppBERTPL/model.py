import torch.nn.functional as F
import torch
from transformers import BertModel
import numpy as np

class qppBERTPL(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.emb_dim = 768
        self.max_pos = 1000
        self.position_enc = torch.nn.Embedding(self.max_pos, self.emb_dim, padding_idx=0)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # bidirectional – If True, becomes a bidirectional LSTM. Default: False
        self.lstm = torch.nn.LSTM(input_size=self.emb_dim, hidden_size=self.bert.config.hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.2)

        self.utility = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, 100), # 768-->100
            torch.nn.Linear(100, 5), # 100-->5 (4+1)
            torch.nn.LogSoftmax(dim=1) # [Batch, 5] values in the range [-inf, 0)
        )

    def forward(self, data):
        # pos_list   [0,1,2,3]
        # input_ids   List of token ids to be fed to a model.  [batch, 512]
        # attention_mask   List of indices specifying which tokens should be attended to by the model  [batch, 512]
        # token_type_ids    List of token type ids to be fed to a model (when return_token_type_ids=True or if “token_type_ids” is in self.model_input_names).  [batch, 512]
        # .last_hidden_state   Sequence of hidden-states at the output of the last layer of the model.
        res = self.bert(data["input_ids"], data["attention_mask"], data["token_type_ids"]).last_hidden_state  # [BATCH, LEN, DIM]  [batch, 512, 768]
        res = res[:, 0]  # get CLS token rep [BATCH, DIM]  [batch, 768]

        res = res + self.position_enc(data["pos_list"])  # [BATCH, DIM]
        res = res.unsqueeze(1)  # [BATCH, 1, DIM]
        # res [L,N,H_in]
        # lstm_output  (L,N,H)
        # recent_hidden (h_n, c_n)
        # h_n  (1, 1, 768), c_n  (1, 1, 768)
        lstm_output, recent_hidden = self.lstm(res)  # [BATCH, DIM]
        # recent_hidden[0].squeeze(1)  [1, 768]
        predicted_numrel = self.utility(recent_hidden[0].squeeze(1)) # [1,5]

        if self.args.mode=="training":
            return F.nll_loss(predicted_numrel, data['numrel'])

        elif self.args.mode=="inference":
            return torch.argmax(predicted_numrel) # [1]


