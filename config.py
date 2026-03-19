import torch

# Reproducibility
RANDOM_SEED = 42

# Model
MODEL_NAME = "roberta-base"
DROPOUT    = 0.1
MAX_LEN    = 128

# Training
BATCH_SIZE     = 64
EPOCHS         = 10
PATIENCE       = 3
UNFREEZE_EPOCH = 2

# Learning rates
BACKBONE_LR          = 1e-5   # conservative — preserves pretrained weights
BACKBONE_LR_UNFROZEN = 5e-6   # reduced further after unfreezing all layers
HEAD_LR              = 3e-5   # faster learning for randomly initialized heads
WEIGHT_DECAY         = 0.01
WARMUP_FRACTION      = 0.2

# Loss weights
ALPHA       = 0.4   # sentiment
BETA        = 0.6   # toxicity (higher — harder task)
AUX_WEIGHT  = 0.2   # sub-component auxiliary task
HUBER_DELTA = 0.1   # Huber loss transition point

# Normalization constants from dataset statistics
TOX_MIN,  TOX_MAX  = -8.34, 6.30

# Thresholds — computed analytically from original bin boundaries:
SENT_THRESHOLD     = 0.5
TOX_THRESHOLD_LOW  = round((-1.5 - TOX_MIN) / (TOX_MAX - TOX_MIN), 3)  # 0.467
TOX_THRESHOLD_HIGH = round(( 1.0 - TOX_MIN) / (TOX_MAX - TOX_MIN), 3)  # 0.638

# Paths
SAVE_DIR  = "./model_artifacts"
PLOT_PATH = "confusion_matrices.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")