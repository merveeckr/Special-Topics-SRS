"""
5-Fold Cross-Validation
TF-IDF+LogReg, ST+LogReg ve BERTürk+BiLSTM+CNN için 5-fold CV uygular.
Her fold için F1-macro, Hamming, Accuracy raporlar; mean ± std hesaplar.

Kullanım:
    python cross_val.py                         # tüm modeller
    python cross_val.py --skip_bert             # sadece TF-IDF ve ST (hızlı)
    python cross_val.py --csv dataset.csv --folds 5
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, hamming_loss, accuracy_score

from transformers import AutoTokenizer

from bil import BertBiLSTMCNN, LABEL_COLS, TEXT_COL, MAX_LEN, DEVICE, THRESH

OFFLINE = os.environ.get("HF_OFFLINE", "0") == "1"


# ── Dataset ──────────────────────────────────────────────────────────────────

class SimpleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts  = texts
        self.labels = labels.astype(np.float32)
        self.tok    = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(str(self.texts[idx]), truncation=True,
                       padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.float),
        }


# ── Metrik yardımcısı ────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    return {
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'hamming':  hamming_loss(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
    }


# ── Fold fonksiyonları ───────────────────────────────────────────────────────

def fold_tfidf(X_tr, y_tr, X_val, y_val):
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_tr_v  = vec.fit_transform(X_tr)
    X_val_v = vec.transform(X_val)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, n_jobs=-1))
    clf.fit(X_tr_v, y_tr)
    preds = (clf.predict_proba(X_val_v) >= 0.5).astype(int)
    return compute_metrics(y_val, preds)


def fold_st(X_tr, y_tr, X_val, y_val, st_model_name):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(st_model_name)
    emb_tr  = st.encode(X_tr,  batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    emb_val = st.encode(X_val, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, n_jobs=-1))
    clf.fit(emb_tr, y_tr)
    preds = (clf.predict_proba(emb_val) >= 0.5).astype(int)
    return compute_metrics(y_val, preds)


def fold_bert(X_tr, y_tr, X_val, y_val, model_name, epochs, batch_size=8, lr=1e-5):
    from tqdm import tqdm
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=OFFLINE)
    tr_ds  = SimpleDataset(X_tr,  y_tr,  tok)
    val_ds = SimpleDataset(X_val, y_val, tok)
    tr_loader  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model     = BertBiLSTMCNN(bert_model_name=model_name, num_labels=len(LABEL_COLS)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn   = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        model.train()
        loop = tqdm(tr_loader, desc=f"    BERT ep {ep+1}/{epochs}", leave=False)
        for batch in loop:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labs = batch['labels'].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(ids, mask), labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loop.set_postfix(loss=f"{loss.item():.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            preds  = (torch.sigmoid(logits).cpu().numpy() >= THRESH).astype(int)
            all_preds.append(preds)
            all_labels.append(batch['labels'].numpy())

    return compute_metrics(np.vstack(all_labels), np.vstack(all_preds))


# ── Ana CV döngüsü ────────────────────────────────────────────────────────────

def run_cv(texts, labels, name, fold_fn, n_folds=5, **kwargs):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    print(f"\n{'='*60}")
    print(f"Model: {name}  ({n_folds}-fold CV)")
    print(f"{'='*60}")

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(texts)):
        X_tr  = [texts[i] for i in tr_idx]
        X_val = [texts[i] for i in val_idx]
        y_tr  = labels[tr_idx]
        y_val = labels[val_idx]

        t0 = time.perf_counter()
        metrics = fold_fn(X_tr, y_tr, X_val, y_val, **kwargs)
        elapsed = time.perf_counter() - t0

        fold_results.append(metrics)
        print(f"  Fold {fold_idx+1}/{n_folds}  "
              f"F1={metrics['f1_macro']:.4f}  "
              f"Hamming={metrics['hamming']:.4f}  "
              f"Acc={metrics['accuracy']:.4f}  "
              f"({elapsed:.1f}s)")

    # İstatistikler
    f1s   = [r['f1_macro'] for r in fold_results]
    hams  = [r['hamming']  for r in fold_results]
    accs  = [r['accuracy'] for r in fold_results]

    summary = {
        'name':      name,
        'f1_mean':   np.mean(f1s),   'f1_std':  np.std(f1s),
        'ham_mean':  np.mean(hams),  'ham_std': np.std(hams),
        'acc_mean':  np.mean(accs),  'acc_std': np.std(accs),
    }
    print(f"\n  ── ÖZET ──")
    print(f"  F1-macro  : {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"  Hamming   : {summary['ham_mean']:.4f} ± {summary['ham_std']:.4f}")
    print(f"  Accuracy  : {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}")
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       default="dataset.csv")
    parser.add_argument("--folds",     type=int,  default=5)
    parser.add_argument("--epochs",    type=int,  default=5)
    parser.add_argument("--skip_bert", action="store_true", help="BERT'i atla (hızlı çalıştır)")
    parser.add_argument("--skip_st",   action="store_true", help="ST modellerini atla")
    args = parser.parse_args()

    df     = pd.read_csv(args.csv)
    texts  = df[TEXT_COL].fillna("").astype(str).tolist()
    labels = df[LABEL_COLS].apply(pd.to_numeric, errors='coerce').fillna(0.0).clip(0,1).astype(int).values.astype(float)

    print(f"\nVeri seti: {len(texts)} satır | {args.folds}-fold CV | BERT epochs: {args.epochs}")
    print(f"DEVICE: {DEVICE}")

    summaries = []

    # 1. TF-IDF + LogReg
    s = run_cv(texts, labels, "TF-IDF(1-2gram) + LogReg",
               fold_fn=fold_tfidf, n_folds=args.folds)
    summaries.append(s)

    # 2. ST + LogReg
    if not args.skip_st:
        default_bert = "./BERTürk" if os.path.isdir("./BERTürk") else "dbmdz/bert-base-turkish-cased"
        st_name = os.environ.get("ST_MODEL", "intfloat/multilingual-e5-small")
        if OFFLINE and not os.path.isdir(st_name):
            print(f"\nOFFLINE: ST modeli ({st_name}) atlanıyor.")
        else:
            s = run_cv(texts, labels, f"ST + LogReg ({st_name})",
                       fold_fn=fold_st, n_folds=args.folds, st_model_name=st_name)
            summaries.append(s)

    # 3. BERTürk + BiLSTM + CNN
    if not args.skip_bert:
        bert_name = os.environ.get("BERT_MODEL",
                    "./BERTürk" if os.path.isdir("./BERTürk") else "dbmdz/bert-base-turkish-cased")
        s = run_cv(texts, labels, f"BERTürk+BiLSTM+CNN (ep={args.epochs})",
                   fold_fn=fold_bert, n_folds=args.folds,
                   model_name=bert_name, epochs=args.epochs)
        summaries.append(s)

    # ── Final karşılaştırma tablosu ──────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print(f"{'':50s} {'F1-macro':>18} {'Hamming':>18} {'Accuracy':>12}")
    print(f"{'Model':<50} {'mean ± std':>18} {'mean ± std':>18} {'mean ± std':>12}")
    print("-"*80)
    for s in summaries:
        f1_str  = f"{s['f1_mean']:.4f} ± {s['f1_std']:.4f}"
        ham_str = f"{s['ham_mean']:.4f} ± {s['ham_std']:.4f}"
        acc_str = f"{s['acc_mean']:.4f} ± {s['acc_std']:.4f}"
        print(f"{s['name']:<50} {f1_str:>18} {ham_str:>18} {acc_str:>12}")
    print("="*80)

    # CSV olarak kaydet
    out_rows = []
    for s in summaries:
        out_rows.append({
            'Model':       s['name'],
            'F1_mean':     round(s['f1_mean'],  4),
            'F1_std':      round(s['f1_std'],   4),
            'Hamming_mean':round(s['ham_mean'], 4),
            'Hamming_std': round(s['ham_std'],  4),
            'Acc_mean':    round(s['acc_mean'], 4),
            'Acc_std':     round(s['acc_std'],  4),
        })
    pd.DataFrame(out_rows).to_csv("cv_results.csv", index=False)
    print("\nSonuçlar cv_results.csv olarak kaydedildi.")


if __name__ == "__main__":
    main()
