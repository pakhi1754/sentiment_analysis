import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup, RobertaTokenizer
from model import freeze_bottom_layers, unfreeze_all_layers
from loss import combined_loss
from evaluate import evaluate
from config import device, EPOCHS, PATIENCE, UNFREEZE_EPOCH, BACKBONE_LR, BACKBONE_LR_UNFROZEN, HEAD_LR, WEIGHT_DECAY, SAVE_DIR, MODEL_NAME, MAX_LEN
import json

def train(model, train_loader, val_loader):
    """
    Function for training the model with layer freezing and early stoppiong.
    param model: the multi-task regression model
    param train_loader: Dataloader for training set
    param test_loader: Dataloader for validation set
    """
    # Saving tokenizer and config
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(SAVE_DIR)

    config = {
        "model_name": MODEL_NAME,
        "max_len": MAX_LEN,
        "num_sentiment_outputs": 1,
        "num_toxicity_outputs": 1,
        "num_aux_labels": 4
    }

    with open(f"{SAVE_DIR}/config.json", "w") as f:
        json.dump(config, f)
        
    freeze_bottom_layers(model) # freeze bottom 6 layers

    optimizer = torch.optim.AdamW([
        {"params": model.roberta.parameters(),        "lr": BACKBONE_LR},
        {"params": model.sentiment_head.parameters(), "lr": HEAD_LR},
        {"params": model.toxicity_head.parameters(),  "lr": HEAD_LR},
        {"params": model.aux_head.parameters(),       "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    scaler      = GradScaler()
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.2), num_training_steps=total_steps)

    best_avg_f1         = 0.0
    epochs_no_improve   = 0

    for epoch in range(1, EPOCHS + 1):

        # Unfreeze all layers at UNFREEZE_EPOCH with reduced backbone LR
        if epoch == UNFREEZE_EPOCH:
            unfreeze_all_layers(model)
            for pg in optimizer.param_groups:
                if pg["params"] == list(model.roberta.parameters()):
                    pg["lr"] = BACKBONE_LR_UNFROZEN
            print(f"  [Epoch {epoch}] All layers unfrozen, backbone LR -> {BACKBONE_LR_UNFROZEN}\n")

        model.train()
        t_loss = t_sent = t_tox = t_aux = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids   = batch["input_ids"].to(device)
            attn_mask   = batch["attention_mask"].to(device)
            sent_target = batch["sentiment_score"].to(device)
            tox_target  = batch["toxicity_score"].to(device)
            aux_target  = batch["aux_labels"].to(device)

            optimizer.zero_grad()
            with autocast(device_type=device.type):
                sent_pred, tox_pred, aux_pred = model(input_ids, attn_mask)
                loss, l_sent, l_tox, l_aux = combined_loss(sent_pred, tox_pred, aux_pred, sent_target, tox_target, aux_target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            t_loss += loss.item(); t_sent += l_sent.item()
            t_tox  += l_tox.item(); t_aux  += l_aux.item()

            if step % 100 == 0:
                print(f"  Epoch {epoch}/{EPOCHS} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f} | Sent: {l_sent.item():.4f} | Tox: {l_tox.item():.4f} | Aux: {l_aux.item():.4f}")

        n = len(train_loader)
        print(f"\nEpoch {epoch}/{EPOCHS} | Loss: {t_loss/n:.4f} | Sent: {t_sent/n:.4f} | Tox: {t_tox/n:.4f} | Aux: {t_aux/n:.4f}")

        result = evaluate(model, val_loader, split_name=f"Epoch {epoch} Val")
        _, sent_f1, tox_f1, _, _, _, _ = result
        avg_f1 = (sent_f1 + tox_f1) / 2

        if avg_f1 > best_avg_f1:
            best_avg_f1         = avg_f1
            epochs_no_improve   = 0
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_roberta_multi_task.pt")
            print(f"  -> Best saved (avg F1: {best_avg_f1:.4f})\n")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{PATIENCE})\n")
            if epochs_no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break