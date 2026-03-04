# PCL Detection Coursework 

This repository contains my submission for the NLP coursework: binary classification of Patronising and Condescending Language (PCL).

**Leaderboard name:** Cath_detector1213  

## BestModel 
All required files are in `BestModel/`:

- `BestModel/train.py` — training script (RoBERTa-based)
- `BestModel/generate_test.py` — generate predictions for dev/test
- `BestModel/dev.txt` — dev predictions (one label per line: 0/1)
- `BestModel/test.txt` — test predictions (one label per line: 0/1)
- `BestModel/link_to_model_ckpt.txt` — link to the best checkpoint

> Note: The model checkpoint is provided via a downloadable link https://drive.google.com/file/d/1yiqjtkjqDPAm16ivrs_NtdbO7xH_3wtx/view?usp=sharing.

## Method summary
Compared to the RoBERTa-base baseline, the final system uses:
1. **Keyword-augmented input**: prepend the provided keyword to the paragraph (e.g., `keyword: text`).
2. **Class-weighted cross-entropy** to address strong class imbalance.
3. **Threshold calibration** on the official dev set to maximise positive-class F1.

## How to run (optional)
From the repository root:
```bash
python BestModel/train.py
python BestModel/generate_test.py