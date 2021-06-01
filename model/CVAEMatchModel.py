import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel, AlbertModel
from transformers import BertPreTrainedModel
from model.attention import AttentionMerge
from parser1 import args

# CVAE encoder - BiGRU
class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, seq_len):
        super(GRUEncoder, self).__init__()
        # GRU编码层，双向GRU，2*hidden
        # input_shape:[batch_size, seq_len, input_size]
        # output_shape:[batch_size, seq_len, 2*hidden_size], hidden_n
        self.GRU_Layer = nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bias=True,
                                batch_first=True,
                                dropout=dropout,
                                bidirectional=True)
        # 均值方差计算模块
        self.fc_mean = nn.Linear(seq_len * 2 * hidden_size, seq_len * hidden_size)
        # 合并后面并把表示减半
        # [batch_size, seq_len, 2 * hidden_size] ==> [batch_size, (seq_len * 2 * hidden_size)//2]
        self.fc_mean_act = nn.ReLU()
        self.fc_logvar = nn.Linear(seq_len * 2 * hidden_size, seq_len * hidden_size)
        self.fc_logvar_act = nn.ReLU()

    def forward(self, inputs):
        outputs, _ = self.GRU_Layer(inputs)
        # 输出均值方差[[batch_size, (seq_len * 2 * hidden_size)//2], [batch_size, (seq_len * 2 * hidden_size)//2]]

        # 改变形状放入linear
        outputs = outputs.view(outputs.size(0), -1)
        return self.fc_mean_act(self.fc_mean(outputs)), self.fc_logvar_act(self.fc_logvar(outputs))


# CVAE Decoder - BiGRU
class GRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, seq_len):
        # ！！此处传入的input_size和hidden_size互相交换了，input_size是256， hidden_size是768
        super(GRUDecoder, self).__init__()
        # input: [batch_size, (seq_len*hidden_size*2)//2]
        # output: [batch_size, seq_len, 768]
        self.seq_len = seq_len
        self.input_size = input_size

        self.fc_expand = nn.Linear((seq_len*input_size*2)//2, seq_len*input_size*2)
        self.GRU_Layer = nn.GRU(input_size=input_size*2,
                                hidden_size=hidden_size//2,
                                num_layers=num_layers,
                                bias=True,
                                batch_first=True,
                                dropout=dropout,
                                bidirectional=True)

    def forward(self, inputs):
        outputs = self.fc_expand(inputs) # [batch, (seq_len*256*2)//2](latent) ==>[batch, seq*256]
        outputs = outputs.view((-1, self.seq_len, self.input_size*2)) # [batch, seq_len, 512]
        outputs, _ = self.GRU_Layer(outputs)
        return outputs


# CVAE Module
class CVaeModel(nn.Module):
    # output:mean, logvar, lantent_z, recons_x
    # 重参数化
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def __init__(self, input_size, hidden_size, num_layers, dropout, seq_len):
        super(CVaeModel, self).__init__()
        # input_shape: [batch_size, seq_len, input_size]
        # output(mean&logvar): [[batch_size, seq_len, hidden_size//2], [batch_size, seq_len, hidden_size//2]]
        self.encoder_module = GRUEncoder(input_size=input_size,
                                         hidden_size=hidden_size,
                                         num_layers=num_layers,
                                         dropout=dropout,
                                         seq_len=seq_len)
        self.decoder_module = GRUDecoder(input_size=hidden_size,
                                         hidden_size=input_size,
                                         num_layers=num_layers,
                                         dropout=dropout,
                                         seq_len=seq_len)

    def forward(self, representation):
        mean, logvar = self.encoder_module(representation)
        #print('均值方差')
        print(mean.shape, logvar.shape)
        latent_z = self.reparameterize(mean, logvar)
        #print('latent_shape:', latent_z.shape)
        output = self.decoder_module(latent_z)
        #print('output_shape', output.shape)
        return mean, logvar, latent_z, output
        

# CVAE MatchModel
class CVaeBertMatchModel(BertPreTrainedModel):
    def __init__(self, config):
        super(CVaeBertMatchModel, self).__init__(config)
        self.bert = BertModel(config)
        # cvae返回(latent_z, output) output就是重构的x:[batch,seq,768]
        # lantent_z = [batch, seq*hidden]
        self.input_size = config.hidden_size
        self.dropout = config.hidden_dropout_prob
        self.max_len = args.max_length
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.cvae_module = CVaeModel(input_size=self.input_size,
                                     hidden_size=self.hidden_size,
                                     num_layers=self.num_layers,
                                     dropout=self.dropout,
                                     seq_len=self.max_len)
        # 加一个FFN
        # self.linear1 = nn.Linear(seq_len*hidden_size, seq_len*hidden_size*2)
        # self.linear2 = nn.Linear(seq_len*hidden_size*2, seq_len*hidden_size)
        self.linear3 = nn.Linear(self.max_len * self.hidden_size, 1)
        self.reconstruction_loss_func = nn.MSELoss()
        self.task_loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):

        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        last_hidden_state = outputs[0]

        mean, logvar, latent_z, recons_x = self.cvae_module(representation=last_hidden_state)
        logits = self.linear3(latent_z)
        if labels is not None:
            task_loss = self.task_loss_func(logits, labels)
            recons_loss = self.reconstruction_loss_func(recons_x, last_hidden_state)
            KLD_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            loss = 0.5 * task_loss + 0.5 * (recons_loss + KLD_loss)
            return loss, logits
        else:
            return logits






