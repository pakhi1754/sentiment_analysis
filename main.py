import numpy as np
import torch
from sklearn.metrics import f1_score
from config import device, RANDOM_SEED, SAVE_DIR
from data_preprocessing import load_raw_data, aggregate_annotations, normalize_and_label, split_data
from dataset import make_dataloaders
from model import RobertaMultiTaskRegression
from baseline import run_tfidf_baseline
from train import train
from evaluate import evaluate, plot_confusion_matrices
import numpy as np

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"Using device: {device}")

# Loading and processing data
df = load_raw_data()
agg_df   = aggregate_annotations(df)
agg_df   = normalize_and_label(agg_df)
train_df, val_df, test_df = split_data(agg_df)

# TF-IDF baseline
tfidf_sent_f1, tfidf_tox_f1 = run_tfidf_baseline(train_df, test_df)

# Making dataloaders
train_loader, val_loader, test_loader = make_dataloaders(train_df, val_df, test_df)

# Training
print("\n" + "="*55)
print(" TRAINING: RoBERTa Dual-Head + Auxiliary Supervision")
print("="*55 + "\n")
model = RobertaMultiTaskRegression().to(device)
train(model, train_loader, val_loader)

print("\n" + "="*55)
print(" FINAL TEST EVALUATION (best checkpoint)")
print("="*55)

model.load_state_dict(torch.load(f"{SAVE_DIR}/best_roberta_multi_task.pt", map_location=device))
result = evaluate(model, test_loader, split_name="Test")
_, test_sent_f1, test_tox_f1, sent_preds, tox_preds, sent_true, tox_true = result

plot_confusion_matrices(sent_true, sent_preds, tox_true, tox_preds)

print("\n" + "="*70)
print(" FINAL RESULTS COMPARISON")
print("="*70)
print(f"{'Model':<45} | {'Sent F1':>7} | {'Tox F1':>7}")
print("-"*70)

print(f"{'TF-IDF + Logistic Regression':<45} | {tfidf_sent_f1:>7.4f} | {tfidf_tox_f1:>7.4f}")
print(f"{'RoBERTa Dual-Head + Auxiliary':<45} | {test_sent_f1:>7.4f} | {test_tox_f1:>7.4f}")
print("-"*70)
print(f"\nImprovement over TF-IDF baseline:")
print(f"  Sentiment: {test_sent_f1 - tfidf_sent_f1:+.4f}")
print(f"  Toxicity:  {test_tox_f1  - tfidf_tox_f1:+.4f}")
print("="*70)