import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay
from torch.amp import autocast
from config import device, SENT_THRESHOLD, TOX_THRESHOLD_LOW, TOX_THRESHOLD_HIGH, PLOT_PATH
from loss import combined_loss


def evaluate(model, loader, split_name="Val"):
    model.eval()
    sent_preds_all, sent_true_scores, sent_true_labels = [], [], []
    tox_preds_all,  tox_true_scores,  tox_true_labels  = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids   = batch["input_ids"].to(device)
            attn_mask   = batch["attention_mask"].to(device)
            sent_target = batch["sentiment_score"].to(device)
            tox_target  = batch["toxicity_score"].to(device)
            aux_target  = batch["aux_labels"].to(device)

            with autocast(device_type=device.type):
                sent_pred, tox_pred, aux_pred = model(input_ids, attn_mask)
                loss, _, _, _ = combined_loss(sent_pred, tox_pred, aux_pred, sent_target, tox_target, aux_target)

            total_loss += loss.item()
            sent_preds_all.extend(sent_pred.cpu().numpy())
            tox_preds_all.extend(tox_pred.cpu().numpy())
            sent_true_scores.extend(sent_target.cpu().numpy())
            tox_true_scores.extend(tox_target.cpu().numpy())

    sent_preds_all   = np.array(sent_preds_all)
    tox_preds_all    = np.array(tox_preds_all)
    sent_true_scores = np.array(sent_true_scores)
    tox_true_scores = np.array(tox_true_scores)
    
    sent_true_labels.extend((sent_true_scores >= SENT_THRESHOLD).astype(int))
    tox_true_labels.extend(
        np.where(tox_true_scores < TOX_THRESHOLD_LOW, 0,
        np.where(tox_true_scores< TOX_THRESHOLD_HIGH, 1, 2)))
    sent_class_preds = (sent_preds_all >= SENT_THRESHOLD).astype(int)
    tox_class_preds  = np.where(tox_preds_all < TOX_THRESHOLD_LOW,  0, np.where(tox_preds_all < TOX_THRESHOLD_HIGH, 1, 2))

    avg_loss = total_loss / len(loader)
    sent_mae = np.mean(np.abs(sent_preds_all - sent_true_scores))
    tox_mae  = np.mean(np.abs(tox_preds_all  - tox_true_scores))

    print(f"\n{'='*55}")
    print(f" {split_name} | Loss: {avg_loss:.4f} | Sent MAE: {sent_mae:.4f} | Tox MAE: {tox_mae:.4f}")
    print(f"{'='*55}")
    print(f"  Sent pred — min: {sent_preds_all.min():.3f} | max: {sent_preds_all.max():.3f} | mean: {sent_preds_all.mean():.3f}")
    print(f"  Tox pred  — min: {tox_preds_all.min():.3f} | max: {tox_preds_all.max():.3f} | mean: {tox_preds_all.mean():.3f}")

    tox_counts  = np.bincount(tox_class_preds,  minlength=3)
    sent_counts = np.bincount(sent_class_preds, minlength=2)
    print(f"  Sent counts — neg: {sent_counts[0]} | pos: {sent_counts[1]}")
    print(f"  Tox counts  — counter: {tox_counts[0]} | neutral: {tox_counts[1]} | hate: {tox_counts[2]}")

    sent_f1  = f1_score(sent_true_labels, sent_class_preds, average="macro")
    print(f"\n[Sentiment] Macro F1: {sent_f1:.4f}")
    print(classification_report(sent_true_labels, sent_class_preds, target_names=["negative", "positive"], digits=4))

    tox_f1  = f1_score(tox_true_labels, tox_class_preds, average="macro")
    print(f"\n[Toxicity] Macro F1: {tox_f1:.4f}")
    print(classification_report(tox_true_labels, tox_class_preds, target_names=["counter", "neutral", "hate"], digits=4))

    return avg_loss, sent_f1, tox_f1, sent_class_preds, tox_class_preds, sent_true_labels, tox_true_labels


def plot_confusion_matrices(sent_true, sent_pred, tox_true, tox_pred, save_path=PLOT_PATH):
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    ConfusionMatrixDisplay.from_predictions(
        sent_true, sent_pred,
        display_labels=["negative", "positive"],
        ax=axes[0], colorbar=False
    )
    axes[0].set_title("Sentiment")

    ConfusionMatrixDisplay.from_predictions(
        tox_true, tox_pred,
        display_labels=["counter", "neutral", "hate"],
        ax=axes[1], colorbar=False
    )
    axes[1].set_title("Toxicity")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")