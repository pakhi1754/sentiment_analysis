from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch

def load_data():
    ds = load_dataset("syedkhalid076/Sentiment-Analysis")
    return ds

#================================TOKENIZING==============================================

def tokenize_data(ds):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def tokenize_function(examples):
         return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    
    tokenized_ds = ds.map(tokenize_function, batched=True)
    tokenized_ds = tokenized_ds.rename_column("label", "labels") # Required by HuggingFace Trainer
    tokenized_ds.set_format("torch")

    return tokenized_ds, tokenizer

