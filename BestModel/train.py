

import math
import pandas as pd
import numpy as np
import torch
import random
import re
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME   = "roberta-base"
MAX_LENGTH   = 128
BATCH_SIZE   = 16
GRAD_ACCUM   = 2           # effective batch = 32
EPOCHS       = 8
LR           = 2e-5        # standard roberta-base fine-tuning LR
WARMUP_RATIO = 0.06        # ~6% warmup (standard)
WEIGHT_DECAY = 0.01
SEED         = 42
PATIENCE     = 3

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"
print(f"Using device: {device}")
if use_amp:
    scaler = torch.cuda.amp.GradScaler()
    print("AMP (fp16) enabled.")

# ── Helpers ────────────────────────────────────────────────────────────────────
def clean_text(t: str) -> str:
    return re.sub(r'\s+', ' ', str(t)).strip()

def build_input(keyword, text) -> str:
    """Prepend keyword as context: 'homeless: [text]'"""
    kw  = str(keyword).strip() if pd.notna(keyword) else ""
    txt = clean_text(text)
    return f"{kw}: {txt}" if kw else txt

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(
    "dontpatronizeme_pcl.tsv",
    sep="\t", header=None,
    names=["par_id","art_id","keyword","country","text","label"],
    skip_blank_lines=True, encoding="utf-8", encoding_errors="replace",
)
df["par_id"]       = pd.to_numeric(df["par_id"], errors="coerce")
df                 = df.dropna(subset=["par_id"]).copy()
df["par_id"]       = df["par_id"].astype(int)
df["text"]         = df["text"].fillna("").astype(str)
df["label"]        = pd.to_numeric(df["label"], errors="coerce")
df                 = df.dropna(subset=["label"]).copy()
df["label"]        = df["label"].astype(int)
df["binary_label"] = (df["label"] >= 2).astype(int)
df["input_text"]   = df.apply(lambda r: build_input(r["keyword"], r["text"]), axis=1)

# ── Official Train/Dev Split ───────────────────────────────────────────────────
train_ids = pd.read_csv("train_semeval_parids-labels.csv")
dev_ids   = pd.read_csv("dev_semeval_parids-labels.csv")
train_ids["par_id"] = train_ids["par_id"].astype(int)
dev_ids["par_id"]   = dev_ids["par_id"].astype(int)

cols     = ["par_id","input_text","binary_label"]
train_df = train_ids[["par_id"]].merge(df[cols], on="par_id", how="left")
dev_df   = dev_ids[["par_id"]].merge(df[cols], on="par_id", how="left")

for frame, name in [(train_df,"train"), (dev_df,"dev")]:
    assert frame["input_text"].notna().all(),   f"Missing texts in {name}"
    assert frame["binary_label"].notna().all(), f"Missing labels in {name}"
assert set(train_df["par_id"]).isdisjoint(set(dev_df["par_id"])), "Train/dev overlap!"

train_df["binary_label"] = train_df["binary_label"].astype(int)
dev_df["binary_label"]   = dev_df["binary_label"].astype(int)

n_pos = int(train_df["binary_label"].sum())
n_neg = len(train_df) - n_pos
pos_weight = n_neg / n_pos   # ≈ 9.55

print(f"Train: {len(train_df)} | PCL={n_pos} ({n_pos/len(train_df)*100:.1f}%) | pos_weight={pos_weight:.2f}")
print(f"Dev:   {len(dev_df)}   | PCL={int(dev_df['binary_label'].sum())}")

# ── Dataset ────────────────────────────────────────────────────────────────────
class PCLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = texts, labels
        self.tok, self.max_len  = tokenizer, max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i], max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[i], dtype=torch.long),
        }

# ── Tokenizer & Model ──────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model     = model.to(device)

# ── DataLoaders — standard shuffle, NO WeightedRandomSampler ──────────────────
train_ds = PCLDataset(train_df["input_text"].tolist(), train_df["binary_label"].tolist(), tokenizer, MAX_LENGTH)
dev_ds   = PCLDataset(dev_df["input_text"].tolist(),   dev_df["binary_label"].tolist(),   tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False)

# ── Loss: pure class-weighted CE, no oversampling ─────────────────────────────
# pos_weight = n_neg/n_pos is the mathematically correct weight for balanced loss
class_w = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_w)
print(f"Loss weight: [No-PCL=1.0, PCL={pos_weight:.2f}]")

# ── Optimizer ─────────────────────────────────────────────────────────────────
no_decay   = ["bias", "LayerNorm.weight"]
opt_params = [
    {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     "weight_decay": WEIGHT_DECAY},
    {"params": [p for n,p in model.named_parameters() if     any(nd in n for nd in no_decay)],
     "weight_decay": 0.0},
]
optimizer = AdamW(opt_params, lr=LR)

total_steps  = math.ceil(len(train_loader) / GRAD_ACCUM) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
print(f"Total steps: {total_steps} | Warmup: {warmup_steps}")

# ── Training Function ──────────────────────────────────────────────────────────
def train_epoch(model, loader):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    last_step = -1

    for step, batch in enumerate(loader):
        last_step = step
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl  = batch["label"].to(device)

        if use_amp:
            with torch.cuda.amp.autocast():
                out  = model(input_ids=ids, attention_mask=mask)
                loss = loss_fn(out.logits, lbl) / GRAD_ACCUM
            scaler.scale(loss).backward()
        else:
            out  = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, lbl) / GRAD_ACCUM
            loss.backward()

        total_loss += loss.item() * GRAD_ACCUM

        if (step + 1) % GRAD_ACCUM == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # flush leftover gradients from last partial accumulation batch
    if (last_step + 1) % GRAD_ACCUM != 0:
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(loader)

# ── Evaluation ────────────────────────────────────────────────────────────────
def get_probs(model, loader):
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl  = batch["label"].to(device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    out = model(input_ids=ids, attention_mask=mask)
            else:
                out = model(input_ids=ids, attention_mask=mask)
            probs_all.extend(torch.softmax(out.logits, 1)[:,1].cpu().numpy())
            labels_all.extend(lbl.cpu().numpy())
    return np.array(probs_all), np.array(labels_all)

def find_best_threshold(probs, labels):
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(labels, (probs >= t).astype(int), pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_f1, best_t

# ── Training Loop ─────────────────────────────────────────────────────────────
best_f1_val  = 0.0
best_thr     = 0.5
best_ckpt    = "best_model.pt"
patience_cnt = 0

for epoch in range(EPOCHS):
    loss = train_epoch(model, train_loader)
    probs, labels = get_probs(model, dev_loader)
    f1, thr = find_best_threshold(probs, labels)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Dev F1: {f1:.4f} @ thr={thr:.2f}")

    if f1 > best_f1_val:
        best_f1_val, best_thr = f1, thr
        torch.save(model.state_dict(), best_ckpt)
        print(f"  ✓ Saved (F1={best_f1_val:.4f}, thr={best_thr:.2f})")
        patience_cnt = 0
    else:
        patience_cnt += 1
        print(f"  No improvement. Patience: {patience_cnt}/{PATIENCE}")
        if patience_cnt >= PATIENCE:
            print("Early stopping triggered.")
            break

# ── Final Evaluation ───────────────────────────────────────────────────────────
print("\nLoading best model for final evaluation...")
model.load_state_dict(torch.load(best_ckpt, map_location=device))
model.to(device)

final_probs, final_labels = get_probs(model, dev_loader)
final_preds = (final_probs >= best_thr).astype(int)
final_f1    = f1_score(final_labels, final_preds, pos_label=1)

print(f"\nBest threshold: {best_thr:.2f} | Final Dev F1: {final_f1:.4f}")
print("\nFinal Classification Report:")
print(classification_report(final_labels, final_preds, target_names=["No PCL","PCL"]))

# ── Save Outputs ───────────────────────────────────────────────────────────────
with open("best_threshold.txt","w") as f:
    f.write(f"{best_thr:.4f}\n")

np.savetxt("dev.txt", final_preds.astype(int), fmt="%d")
pd.DataFrame({"par_id": dev_df["par_id"], "pred": final_preds}).to_csv(
    "dev_predictions_with_ids.csv", index=False
)
print(f"\nSaved: dev.txt, best_threshold.txt, dev_predictions_with_ids.csv")
print(f"Best Dev F1: {final_f1:.4f}")