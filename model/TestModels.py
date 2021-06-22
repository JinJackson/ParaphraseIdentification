from transformers import BertModel, BertPreTrainedModel, BertForMaskedLM, BertTokenizer
from transformers.modeling_bert import BertOnlyMLMHead
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from attention import attention

class TestModel(BertPreTrainedModel):
    def __init__(self, config):
        super(TestModel, self).__init__(config)
        self.bert = BertModel(config)
        self.input_size = config.hidden_size
        self.GRU_Layer = nn.GRU(input_size=self.input_size,
                                hidden_size=self.input_size//2,
                                num_layers=2,
                                bias=True,
                                batch_first=True,
                                dropout=config.hidden_dropout_prob,
                                bidirectional=True)
        self.cls = BertOnlyMLMHead(config)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_outputs, cls = outputs[:2]
        gru_outputs, _ = self.GRU_Layer(sequence_outputs)
        prediction_scores = self.cls(gru_outputs)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        output = (prediction_scores,) + outputs[2:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output


model = TestModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = 'I love New [MASK] city.'
labels = tokenizer("I love New York city.", return_tensors="pt")["input_ids"]

inputs = tokenizer(text=text, return_tensors='pt')
outputs = model(**inputs, labels=labels)

print(len(outputs))
print(outputs[0].shape, outputs[1].shape)
print(outputs[0])
print(outputs[1])

text1 = '今天天气怎么样？'
text2 = '天气不太好，是雨天'

model = BertModel.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

inputs = tokenizer(text=text1, text_pair=text2, return_tensors='pt')
print(inputs)
outputs = model(**inputs)

sequence_outputs, cls = outputs[:2]

outputs, p_attn = attention(query=sequence_outputs, key=sequence_outputs, value=sequence_outputs)

print(outputs.shape)
print(p_attn.shape)

print(p_attn)