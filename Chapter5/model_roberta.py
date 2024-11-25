import torch
import torch.nn as nn
from transformers import RobertaModel


class RoBERTaClass(nn.Module):
    def __init__(self, MODEL_NAME_OR_PATH, n_classes: int):
        super(RoBERTaClass, self).__init__()
        self.roberta = RobertaModel.from_pretrained(MODEL_NAME_OR_PATH)
        self.classify_linear = nn.Linear(1024, n_classes)  # 隐藏层大小为 1024
        self.dropout = nn.Dropout(0.3)

    def forward(self, ids, mask, special_tokens_mask):
        last_hidden_state, _ = self.roberta(ids, attention_mask=mask, return_dict=False)
        masked_state = last_hidden_state * special_tokens_mask.unsqueeze(-1).float()
        final_output = torch.sum(masked_state, dim=1)
        output_2 = self.dropout(final_output)
        output = self.classify_linear(output_2)
        return output
