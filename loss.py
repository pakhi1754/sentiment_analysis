import torch.nn as nn
from config import ALPHA, BETA, AUX_WEIGHT, HUBER_DELTA

sent_criterion = nn.HuberLoss(delta=HUBER_DELTA)
tox_criterion  = nn.HuberLoss(delta=HUBER_DELTA)
aux_criterion  = nn.MSELoss()   # sub-components are smoother, MSE is fine


def combined_loss(sent_pred, tox_pred, aux_pred, sent_target, tox_target, aux_target, alpha=ALPHA, beta=BETA, aux_weight=AUX_WEIGHT):
    """
    Function for weighted combination of three regression losses:
      - Sentiment Huber loss    (weight: alpha)
      - Toxicity Huber loss     (weight: beta)
      - Auxiliary MSE loss      (weight: aux_weight)
    """
    l_sent = sent_criterion(sent_pred, sent_target)
    l_tox  = tox_criterion(tox_pred,   tox_target)
    l_aux  = aux_criterion(aux_pred,   aux_target)
    total  = alpha * l_sent + beta * l_tox + aux_weight * l_aux
    return total, l_sent, l_tox, l_aux