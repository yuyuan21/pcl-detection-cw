
import pandas as pd
import numpy as np
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

MODEL_NAME = "roberta-base"
MAX_LENGTH = 128
BATCH_SIZE = 16

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"

def clean_text(t):
    return re.sub(r'\s+', ' ', str(t)).strip()

def build_input(keyword, text):
    kw  = str(keyword).strip() if pd.notna(keyword) else ""
    txt = clean_text(text)
    return f"{kw}: {txt}" if kw else txt

# ── Load test set ──────────────────────────────────────────────────────────────
test_df = pd.read_csv(
    "task4_test.tsv",
    sep="\t", header=None,
    names=["par_id","art_id","keyword","country","text"],
    skip_blank_lines=True, encoding="utf-8", encoding_errors="replace",
)
test_df["par_id"]     = test_df["par_id"].fillna("").astype(str).str.strip()
test_df               = test_df[test_df["par_id"] != ""].copy()
test_df["text"]       = test_df["text"].fillna("").astype(str)
test_df["input_text"] = test_df.apply(lambda r: build_input(r["keyword"], r["text"]), axis=1)
print(f"Test set: {len(test_df)} samples")

class PCLDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts, self.tok, self.max_len = texts, tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], max_length=self.max_len,
                       padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model     = model.to(device)
model.eval()

with open("best_threshold.txt") as f:
    threshold = float(f.read().strip())
print(f"Using threshold: {threshold:.4f}")

test_ds     = PCLDataset(test_df["input_text"].tolist(), tokenizer, MAX_LENGTH)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

all_probs = []
with torch.no_grad():
    for batch in test_loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        if use_amp:
            with torch.cuda.amp.autocast():
                out = model(input_ids=ids, attention_mask=mask)
        else:
            out = model(input_ids=ids, attention_mask=mask)
        all_probs.extend(torch.softmax(out.logits, 1)[:,1].cpu().numpy())

preds = (np.array(all_probs) >= threshold).astype(int)
assert len(preds) == len(test_df), f"Prediction count mismatch: {len(preds)} vs {len(test_df)}"
np.savetxt("test.txt", preds, fmt="%d")
print(f"Saved test.txt — {len(preds)} predictions | PCL={preds.sum()} ({preds.mean()*100:.1f}%)")
