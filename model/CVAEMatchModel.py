import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel, AlbertModel
from transformers import BertPreTrainedModel
from attention import AttentionMerge


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
        #均值方差计算模块
        self.fc_mean = nn.Linear(seq_len * 2 * hidden_size, (seq_len * 2 * hidden_size)//2)
        # 合并后面并把表示减半
        # [batch_size, seq_len, 2 * hidden_size] ==> [batch_size, (seq_len * 2 * hidden_size)//2]
        self.fc_mean_act = nn.ReLU()
        self.fc_logvar = nn.Linear(seq_len * 2 * hidden_size, (seq_len * 2 * hidden_size)//2)
        self.fc_logvar_act = nn.ReLU()

    def forward(self, inputs):
        outputs, _ = self.GRU_Layer(inputs)
        # 输出均值方差[[batch_size, (seq_len * 2 * hidden_size)//2], [batch_size, (seq_len * 2 * hidden_size)//2]]

        # 改变形状放入linear
        outputs = outputs.view(outputs.size(0), -1)
        return self.fc_mean_act(self.fc_mean(outputs)), self.fc_logvar_act(self.fc_logvar(outputs))


class GRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRUDecoder, self).__init__()
        self.GRU_Layer = nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bias=True,
                                batch_first=True,
                                dropout=dropout,
                                bidirectional=True)


class CVaeModel(nn.Module):

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

    def forward(self, representation):
        mean, logvar = self.encoder_module(representation)
        print('均值方差')
        print(mean.shape, logvar.shape)
        latent_z = self.reparameterize(mean, logvar)
        return latent_z
        



