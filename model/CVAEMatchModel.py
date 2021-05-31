import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel, AlbertModel
from transformers import BertPreTrainedModel
from attention import AttentionMerge

class CVAEencoder(nn.Module):
    def __init__(self, input_size, ):
        pass

class CVAEdecoder(nn.Module):
    def __init__(self):
        pass


class CVAEModel(nn.Module):
    def __init__(self, input_size, latent_size, output_size):
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.encoder = CVAEencoder()
        self.decoder = CVAEdecoder()




class BertCVAEModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
