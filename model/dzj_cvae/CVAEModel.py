import torch
import torch.nn as nn
from sklearn.preprocessing import LabelBinarizer
from torch.nn import functional as F
from .Attention import AttentionInArgs
from .SelfAttention import SelfAttention
from .GATModel import GAT 

import random
import numpy as np 
# random.seed(123)
# np.random.seed(123)
# torch.manual_seed(123)


# In[4]
class TextRNN(nn.Module):
    def __init__(self, 
                 input_size = 256, 
                 hidden_size = 128,
                 output_size = 768,
                 n_layers = 2,
                 dropout =  0.4,  # 0.5, 0.4
                 args = None
                 ):
        super(TextRNN, self).__init__()
        self.rnn = nn.LSTM(input_size = input_size, 
                           hidden_size = hidden_size, 
                           num_layers = n_layers, 
                           bidirectional = True, 
                           batch_first = True, 
                           dropout = dropout)

        self.fc = nn.Linear(hidden_size*2, output_size)     # 双向RNN，且arg1, arg2
        self.dropout = nn.Dropout(dropout) 
        
        # self.reset_weigths()

    # def reset_weigths(self):
    #     """reset weights
    #     """
    #     for weight in self.parameters():
    #         nn.init.xavier_normal_(weight)

    def forward(self, input_ids):
        # [8, 80, 300] 
        arg_out = self.dropout(input_ids)
        # out: [batch, seq_len, hidden_dim * 2]
        # hideen: [batch, num_layers * 2, hidden_dim]
        # cell/c: [batch, num_layers * 2, hidden_dim]
        arg_out, (_, _) = self.rnn(arg_out)                                

        out = self.fc(arg_out)                                         # [8, 2]   

        return out


class TextCNN(nn.Module):   
    # output_size为输出类别（2个类别，0和1）,三种kernel，size分别是3,4，5，每种kernel有100个
    def __init__(self, input_size=768, output_size=768, filter_num=128, kernel_lst=(2, 4), dropout=0.5):
        super(TextCNN, self).__init__()

        self.conv01 = nn.Conv2d(1, filter_num, (kernel_lst[0], input_size))
        self.fc1 = nn.Linear(filter_num*2 - kernel_lst[0] + 1, output_size)
        self.conv02 = nn.Conv2d(1, filter_num, (kernel_lst[1], input_size))
        self.fc2 = nn.Linear(filter_num*2 - kernel_lst[1] + 1, output_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
       # [128, 20, 200] (batch, seq_len, embedding_dim)
       # [128, 1, 20, 200] 即 (batch, channel_num, seq_len, embedding_dim)
       # [8, 256, 768]
        x = x.unsqueeze(1)         
        out1 = self.conv01(x)    # ([8, 128, 255, 1])
        out1 = out1.view(out1.shape[0], out1.shape[1], -1)          
        out1 = self.fc1(out1)

        out2 = self.conv02(x)    # ([8, 128, 253, 1])
        out2 = out2.view(out2.shape[0], out2.shape[1], -1)   
        out2 = self.fc2(out2)    

        out = torch.cat([out1, out2], dim=1)          # [128, 256, 1, 1]

        out = self.dropout(out)

        return out


# In[1]
class CVAEModel(nn.Module):
    def __init__(self, 
                input_size=256,
                hidden_size=256,
                output_size=768
                ):
        super(CVAEModel, self).__init__()

        # 加RNN
        # [8, 256, 768]
        self.rnn01 = TextRNN(input_size=input_size, hidden_size = 256, output_size = 768)
        self.rnn02 = TextRNN(input_size=input_size, hidden_size = 256, output_size = 768)
        # 768 -> 256
        self.fc11 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc12 = nn.Linear(hidden_size, hidden_size // 2)

        self.fc21 = nn.Linear(hidden_size // 2, hidden_size) 
        self.fc22 = nn.Linear(hidden_size, hidden_size) 

        self.layernorm = nn.LayerNorm(hidden_size)
        self.lb = LabelBinarizer()

        # 加attention
        self.attention = AttentionInArgs(input_size=768,
                                         embedding_dim=256,
                                         output_size=768
                                         )
        self.self_attention = SelfAttention(input_size=768,
                                         embedding_dim=256,
                                         output_size=768
                                         )
        # torch.Size([8, 256, 768]) torch.Size([8, 256, 256])
        self.gat = GAT(in_features=768, n_dim=256, n_class=768)

        # cnn
        self.cnn01 = TextCNN(input_size, output_size)
        self.cnn11 = TextCNN(input_size, output_size, kernel_lst=(6, 8))
        self.cnn02 = TextCNN(input_size, output_size)

        # self.pos = nn.Parameter(torch.randn(1, 256, 768)) 
        # self.neg = nn.Parameter(torch.zeros(1, 256, 768))

        # self.pos = torch.randn(1, 256, 768) + 1
        # self.neg = torch.FloatTensor(1, 256, 768) 
        
        # self.pos = torch.FloatTensor(1, 256, 768) + 1
        # self.neg = torch.FloatTensor(1, 256, 768)

        self.com = torch.FloatTensor(1, 256, 768) + 0
        self.con = torch.FloatTensor(1, 256, 768) + 1
        self.exp = torch.FloatTensor(1, 256, 768) + 2
        self.tem = torch.FloatTensor(1, 256, 768) + 3

        # self.cons = [self.neg, self.pos]
        self.cons = [self.com, self.con, self.exp, self.tem]
    
    # 将标签进行one-hot编码
    def to_categrical(self, y, device=None):
        # print(self.com, self.con, self.exp, self.tem)
        y_n = y.cpu().detach()
        # self.lb.fit(list(range(0, 4)))
        # y_one_hot = self.lb.transform(y_n)
        # y_one_hot = torch.FloatTensor(y_one_hot).to(device)
        
        y_one_hot = self.cons[y_n[0]]
        for y in y_n[1:]:
            y_one_hot = torch.cat([y_one_hot, self.cons[y]], dim=0)
        y_one_hot = y_one_hot.to(device)
        return y_one_hot
    
    def to_categrical_neg(self, y, device=None):
        y_n = y.cpu().detach()
        # self.lb.fit(list(range(0, 4)))
        # y_one_hot = 1 - self.lb.transform(y_n)
        # y_one_hot = torch.FloatTensor(y_one_hot).to(device)

        # idx = random.randint(0, 1)
        y_one_hot = self.cons[1 - y_n[0]]
        for y in y_n[1:]:
            # idx = random.randint(0, 1)
            y_one_hot = torch.cat([y_one_hot, self.cons[1 - y]], dim=0)
        y_one_hot = y_one_hot.to(device)
        return y_one_hot

    def _add_attention(self, x):
        # 加attention
        arg_len = x.shape[1] // 2
        # X: [8, 128, 768], [8, 128, 768] --> [8, 256, 768]
        # X: [8, 82, 768], [8, 82, 768] --> [8, 164, 768]
        arg1, arg2 = x[:, :arg_len, :], x[:, arg_len:, :]
        # logging.info(str(arg1) + ' ' + str(arg1.shape))
        # logging.info(str(arg2) + ' ' + str(arg2.shape))
        # 目标: [8, 256, 256]
        out = self.attention(arg1, arg2)
        # logging.info('adj: ' + str(adj[0]) + ' ' + str(adj.shape))
        # 只是做个 mm
        # out = self.gat(x, out)
        return out

    def encode(self, x, y=None, Training=False, device=None):
        if Training:
            # 加RNN
            con = x
            y_c = self.to_categrical(y, device)
            # y_c = y_c.unsqueeze(1)
            # # 输入样本和标签y的one-hot向量连接
            con = con + y_c   
            
            # 加RNN
            out = self.rnn01(con)
            # 加CNN
            out = self.cnn01(con)
            
            
            # print(out.shape)
            
            # 加CNN2
            # out = self.cnn11(out)

            # 加attention
            # out = self._add_attention(x)

            return F.relu(self.fc11(out)), F.relu(self.fc12(out))
            # return F.relu(self.fc11(x + out)), F.relu(self.fc12(x + out))
        else:
            # 只用RNN
            out = self.rnn01(x)
            # 加CNN
            out = self.cnn01(x)

            # 加CNN2
            # out = self.cnn11(out)
                        
            # 加attention
            # out = self._add_attention(x)

            return F.relu(out)
            # return F.relu(x + out)
        
    # 再参数化
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z, y=None, Training=False, device=None):   
        # con = z
        # y_c = self.to_categrical(y, device)
        # y_c = y_c.unsqueeze(1)
        # con = con + y_c
        # out = self.fc21(con)
        
        con = z
        y_c = self.to_categrical(y, device)
        con = self.fc21(con)
        out = con + y_c


        out = self.rnn02(out)  # 在这修改

        return F.relu(out)
        

    @classmethod
    def loss_function(cls, recon_x, x, mu, logvar):
        bz = x.shape[0]
        # recon_x, x = recon_x.view(bz, -1), x.view(-1)
        # print(recon_x.shape, x.shape)
        BCE = nn.MSELoss()(recon_x, x)
        # BCE = nn.CrossEntropyLoss(recon_x, x)

        # BCE = torch.mean(-torch.log(recon_x.div(recon_x + x)))
        # print(BCE.shape, BCE)

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def forward(self, x, y=None, Training=False, device=None):   
        # 训练 CVAE
        if Training:
            # Encode 
            mu, logvar = self.encode(x, y, Training, device)
            # 再参数化
            z = self.reparameterize(mu, logvar)
            # Decode
            out = self.decode(z, y, Training, device)
            
            return out, mu, logvar
        else:
            out = self.encode(x)
            return out

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
    
    def forward(self, x, y, temp=None):
        if temp is None:
            temp = self.temp
        return self.cos(x, y) / temp        