import torch
import json
from transformers import RobertaTokenizer
from model import RobertaMultiTaskRegression
from config import SAVE_DIR, device

# Load config
with open(f"{SAVE_DIR}/config.json") as f:
    config = json.load(f)

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained(SAVE_DIR)

# Load model
model = RobertaMultiTaskRegression()
model.load_state_dict(torch.load(f"{SAVE_DIR}/best_roberta_multi_task.pt", map_location=device))
model.to(device)
model.eval()

def predict(text: str):
    encoding = tokenizer(
        text,
        max_length=config["max_len"],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        sent_pred, tox_pred, aux_pred = model(input_ids, attention_mask)

    return {
        "sentiment_score": float(sent_pred.cpu().numpy()[0]),
        "toxicity_score": float(tox_pred.cpu().numpy()[0]),
        "aux_labels": aux_pred.cpu().numpy()[0].tolist()
    }

if __name__ == "__main__":
    print(predict("I love this product!"))