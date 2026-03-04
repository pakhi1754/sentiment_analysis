import torch.nn as nn
from transformers import RobertaModel
from config import MODEL_NAME, DROPOUT


class RobertaMultiTaskRegression(nn.Module):
    """
    RoBERTa model with three heads for multi-task regression

      sentiment_head  — predicts normalized sentiment score (0-1)
      toxicity_head   — predicts normalized hate_speech_score (0-1)
      aux_head        — predicts insult/humiliate/dehumanize/violence (0-1 each)
    """

    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(MODEL_NAME)
        h = self.roberta.config.hidden_size   # 768 for roberta-base

        self.sentiment_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(h, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.toxicity_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(h, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.aux_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(h, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        cls = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return (
            self.sentiment_head(cls).squeeze(1),   # (batch,)
            self.toxicity_head(cls).squeeze(1),    # (batch,)
            self.aux_head(cls)                      # (batch, 4)
        )


def freeze_bottom_layers(model, n_layers=6):
    """
    Function for freezing bottom n transformer layers for stable learning of task heads.
    """
    for name, param in model.roberta.named_parameters():
        if any(f"layer.{i}." in name for i in range(n_layers)):
            param.requires_grad = False
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    total  = sum(1 for p in model.parameters())
    print(f"Frozen {frozen}/{total} parameter groups")


def unfreeze_all_layers(model):
    """
    Function for unfreezing all layers for full fine-tuning.
    """
    for param in model.roberta.parameters():
        param.requires_grad = True