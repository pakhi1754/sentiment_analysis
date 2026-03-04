import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from config import MODEL_NAME, MAX_LEN, BATCH_SIZE

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

# Custom dataset
class HateSpeechDataset(Dataset):
    def __init__(self, df, max_len=MAX_LEN):
        self.texts            = df["text"].tolist()
        self.sentiment_scores = df["sentiment"].values.astype(np.float32)
        self.toxicity_scores  = df["toxicity_score"].values.astype(np.float32)
        self.aux_labels       = df[["insult", "humiliate", "dehumanize", "violence"]].values.astype(np.float32)
        self.max_len          = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids":       enc["input_ids"].squeeze(0),
            "attention_mask":  enc["attention_mask"].squeeze(0),
            "sentiment_score": torch.tensor(self.sentiment_scores[idx], dtype=torch.float),
            "toxicity_score":  torch.tensor(self.toxicity_scores[idx],  dtype=torch.float),
            "aux_labels":      torch.tensor(self.aux_labels[idx],       dtype=torch.float)
        }


def make_dataloaders(train_df, val_df, test_df):
    """
    Function for creating dataloaders.
    param train_df: training dataframe
    param val_df: validation dataframe
    param test_df: test dataframe
    returns: dataloaders for train, test and validation sets
    """
    train_loader = DataLoader(HateSpeechDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(HateSpeechDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(HateSpeechDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader