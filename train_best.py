"""
BertCNNOnly modelini augmented_dataset.csv üzerinde eğitir ve best_model.pt kaydeder.

Kullanım:
    python train_best.py
    python train_best.py --csv dataset.csv --epochs 10 --batch 16
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import pandas as pd
from transformers import AutoTokenizer

from bil import LABEL_COLS, TEXT_COL, MAX_LEN, DEVICE
from bench import BertCNNOnly

OFFLINE = os.environ.get("HF_OFFLINE", "0") == "1"


class ReqDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts  = texts
        self.labels = labels.astype(np.float32)
        self.tok    = tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            str(self.texts[idx]),
            truncation=True, padding='max_length',
            max_length=MAX_LEN, return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.float),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    default="augmented_dataset.csv")
    parser.add_argument("--epochs", type=int,   default=5)
    parser.add_argument("--batch",  type=int,   default=8)
    parser.add_argument("--lr",     type=float, default=1e-5)
    parser.add_argument("--out",    default="best_model.pt")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        fallback = "dataset.csv"
        print(f"UYARI: {args.csv} bulunamadi, {fallback} kullaniliyor.")
        args.csv = fallback

    df     = pd.read_csv(args.csv)
    texts  = df[TEXT_COL].fillna("").astype(str).tolist()
    labels = (df[LABEL_COLS]
              .apply(pd.to_numeric, errors='coerce')
              .fillna(0.0).clip(0, 1).astype(int)
              .values.astype(np.float32))

    bert_name = os.environ.get(
        "BERT_MODEL",
        "./BERTürk" if os.path.isdir("./BERTürk") else "dbmdz/bert-base-turkish-cased"
    )

    print(f"Veri seti : {args.csv}  ({len(texts)} satir)")
    print(f"BERT model: {bert_name}")
    print(f"Ayarlar   : {args.epochs} epoch | batch={args.batch} | lr={args.lr}")
    print(f"Device    : {DEVICE}")

    tok    = AutoTokenizer.from_pretrained(bert_name, local_files_only=OFFLINE)
    loader = DataLoader(ReqDataset(texts, labels, tok),
                        batch_size=args.batch, shuffle=True)

    model     = BertCNNOnly(bert_name, num_labels=len(LABEL_COLS)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn   = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    for ep in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        loop = tqdm(loader, desc=f"Epoch {ep+1}/{args.epochs}")
        for batch in loop:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labs = batch['labels'].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(ids, mask), labs)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg = epoch_loss / len(loader)
        print(f"  Epoch {ep+1} ortalama loss: {avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), args.out)
            print(f"  --> {args.out} kaydedildi (loss={best_loss:.4f})")

    print(f"\nEgitim tamamlandi. En iyi model: {args.out}  (loss={best_loss:.4f})")


if __name__ == "__main__":
    main()
