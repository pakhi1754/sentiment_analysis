# Multi-Task RoBERTa for Sentiment & Toxicity Classification

Multi-task learning model that jointly predicts **sentiment** and **toxicity** from social media text using a single shared RoBERTa backbone with separate regression heads.

## Key Design Choices

**Regression** — both tasks train on continuous scores. This preserves annotation uncertainty from multi-annotator disagreement and avoids information loss from premature binning. Labels are only applied at inference via thresholds.

**Multi-task learning** — sentiment and toxicity share a RoBERTa backbone, allowing gradient signal from both tasks to improve shared representations.

**Auxiliary supervision** — a third head simultaneously predicts sub-components (insult, humiliation, dehumanization, violence), forcing the backbone to learn fine-grained toxicity signal.

## Results

| Model | Sentiment F1 | Toxicity F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 0.779 | 0.616 |
| **RoBERTa** | **0.844** | **0.691** |

The toxicity result is notable: the model is essentially at the limit of what is learnable from this data, with remaining error attributable to inherent annotator disagreement rather than model limitations.

## Dataset

[UCBerkeley DLab Measuring Hate Speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) — 39,565 social media posts with 135,556 annotations from 7,912 annotators across Reddit, Twitter, and YouTube.

## File Structure
```
├── config.py              # All hyperparameters and constants
├── data_preprocessing.py  # Loading, normalization, splitting
├── dataset.py             # PyTorch Dataset and DataLoader creation
├── model.py               # RobertaMultiTaskRegression architecture
├── loss.py                # Huber + MSE combined loss
├── train.py               # Training loop with early stopping
├── evaluate.py            # Evaluation, confusion matrices
├── baseline.py            # TF-IDF + Logistic Regression baseline
└── main.py                # Entry point — runs full pipeline
```