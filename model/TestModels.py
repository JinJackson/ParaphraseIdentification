from transformers import BertModel, BertPreTrainedModel, BertForMaskedLM, BertTokenizer
from transformers.modeling_bert import BertOnlyMLMHead
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

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
                                dropout=dropout,
                                bidirectional=True)
        self.cls = BertOnlyMLMHead(config)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_outputs, cls = outputs[:2]
        prediction_scores = self.cls(sequence_outputs)
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

