import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel, AlbertModel
from transformers import BertPreTrainedModel
# from attention import AttentionMerge
from parser1 import args


# CVAE encoder - BiGRU
class GRUEncoder(nn.Module):
    def __init__(self, input_size, num_layers, dropout):
        super(GRUEncoder, self).__init__()
        # GRU编码层，双向GRU，2*hidden
        # input_shape:[batch_size, seq_len, input_size]
        # output_shape:[batch_size, seq_len, input_size]
        self.GRU_Layer = nn.GRU(input_size=input_size,
                                hidden_size=input_size//2,
                                num_layers=num_layers,
                                bias=True,
                                batch_first=True,
                                dropout=dropout,
                                bidirectional=True)
        # 均值方差计算模块
        # [B, S, h] ==> [B, S, h//2]
        self.fc_mean = nn.Linear(input_size, input_size//2)
        self.fc_mean_act = nn.ReLU()
        self.fc_logvar = nn.Linear(input_size, input_size//2)
        self.fc_logvar_act = nn.ReLU()

    def forward(self, inputs):
        encoder_outputs, _ = self.GRU_Layer(inputs)

        # 输出均值方差[[batch_size, seq_len, input_size//2], [batch_size, seq_len, hidden//2]]

        # 改变形状放入linear
        # print(outputs.shape)
        #outputs = outputs.reshape((outputs.size(0), -1))
        return self.fc_mean_act(self.fc_mean(encoder_outputs)), self.fc_logvar_act(self.fc_logvar(encoder_outputs)), encoder_outputs


# CVAE Decoder - BiGRU
class GRUDecoder(nn.Module):
    def __init__(self, input_size, num_layers, dropout):
        super(GRUDecoder, self).__init__()

        # [B, S, h//2] ==> [B, S, h//2]
        self.fc_expand = nn.Linear(input_size//2, input_size)
        self.GRU_Layer = nn.GRU(input_size=input_size,
                                hidden_size=input_size//2,
                                num_layers=num_layers,
                                bias=True,
                                batch_first=True,
                                dropout=dropout,
                                bidirectional=True)

    def forward(self, inputs):
        outputs = self.fc_expand(inputs)  # [batch, (seq_len*256*2)//2](latent) ==>[batch, seq*256]
        #outputs = outputs.reshape((-1, self.seq_len, self.input_size * 2))  # [batch, seq_len, 512]
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

    def __init__(self, input_size, num_layers, dropout):
        super(CVaeModel, self).__init__()
        # input_shape: [batch_size, seq_len, input_size]
        # output(mean&logvar): [[batch_size, seq_len, hidden_size//2], [batch_size, seq_len, hidden_size//2]]
        self.encoder_module = GRUEncoder(input_size=input_size,
                                         num_layers=num_layers,
                                         dropout=dropout)
        self.decoder_module = GRUDecoder(input_size=input_size,
                                         num_layers=num_layers,
                                         dropout=dropout)

    def forward(self, representation):
        mean, logvar, encoder_outputs = self.encoder_module(representation)
        # print('均值方差')
        latent_z = self.reparameterize(mean, logvar)
        # print('latent_shape:', latent_z.shape)
        recons_x = self.decoder_module(latent_z)
        # print('output_shape', output.shape)
        return mean, logvar, latent_z, recons_x, encoder_outputs


# CVAE MatchModel
class CVaeBertMatchModel(BertPreTrainedModel):
    def __init__(self, config):
        super(CVaeBertMatchModel, self).__init__(config)
        self.bert = BertModel(config)
        # cvae返回(latent_z, output) output就是重构的x:[batch,seq,768]
        # lantent_z = [batch, seq*hidden]
        self.input_size = config.hidden_size
        self.dropout = config.hidden_dropout_prob
        self.GRU_Layer = nn.GRU(input_size=self.input_size,
                                hidden_size=self.input_size // 2,
                                num_layers=args.num_layers,
                                bias=True,
                                batch_first=True,
                                dropout=self.dropout,
                                bidirectional=True)
        self.linear = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):

        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        last_hidden_state = outputs[0]
        encoder_outputs, _ = self.GRU_Layer(last_hidden_state)
        cls = encoder_outputs[:, 0, :]
        logits = self.linear(cls)
        if labels is not None:
            loss = self.task_loss_func(logits, labels.float())
            return loss, logits
        else:
            return logits






