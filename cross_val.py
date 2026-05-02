"""
5-Fold Cross-Validation
TF-IDF+LogReg, BERTürk+MLP, BERTürk+CNN, BERTürk+BiLSTM+CNN için 5-fold CV.

Kullanım:
    python cross_val.py                                        # tüm modeller, augmented_dataset.csv
    python cross_val.py --csv dataset.csv                      # orijinal veri
    python cross_val.py --csv augmented_dataset.csv --folds 5  # augmented
    python cross_val.py --skip_bert                            # sadece TF-IDF (hızlı)
    python cross_val.py --models tfidf,cnn                     # seçili modeller
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

from bil import LABEL_COLS, TEXT_COL, MAX_LEN, DEVICE, THRESH
from bench import BertMLP, BertCNNOnly, BertBiLSTMCNN

OFFLINE = os.environ.get("HF_OFFLINE", "0") == "1"


# ── Dataset ───────────────────────────────────────────────────────────────────

class SimpleDataset(Dataset):
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


# ── Metrik ───────────────────────────────────────────────────────────────────

def metrics(y_true, y_pred):
    return {
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'hamming':  hamming_loss(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
    }


# ── Fold fonksiyonları ────────────────────────────────────────────────────────

def fold_tfidf(X_tr, y_tr, X_val, y_val, **_):
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, n_jobs=-1))
    clf.fit(vec.fit_transform(X_tr), y_tr)
    preds = (clf.predict_proba(vec.transform(X_val)) >= 0.5).astype(int)
    return metrics(y_val, preds)


def fold_bert_arch(X_tr, y_tr, X_val, y_val, model_class, model_name,
                   epochs=5, batch_size=8, lr=1e-5, **_):
    """Herhangi bir BERT tabanlı mimariyi eğitip değerlendirir."""
    from tqdm import tqdm
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=OFFLINE)
    tr_loader  = DataLoader(SimpleDataset(X_tr,  y_tr,  tok), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SimpleDataset(X_val, y_val, tok), batch_size=batch_size, shuffle=False)

    model     = model_class(model_name, num_labels=len(LABEL_COLS)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn   = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        model.train()
        loop = tqdm(tr_loader, desc=f"    ep {ep+1}/{epochs}", leave=False)
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
            loop.set_postfix(loss=f"{loss.item():.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            preds  = (torch.sigmoid(logits).cpu().numpy() >= THRESH).astype(int)
            all_preds.append(preds)
            all_labels.append(batch['labels'].numpy())

    # GPU belleğini serbest bırak
    del model
    torch.cuda.empty_cache()

    return metrics(np.vstack(all_labels), np.vstack(all_preds))


# ── CV döngüsü ────────────────────────────────────────────────────────────────

def run_cv(texts, labels, name, fold_fn, n_folds=5, **kwargs):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    print(f"\n{'='*62}")
    print(f"  {name}  |  {n_folds}-fold CV")
    print(f"{'='*62}")

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(texts)):
        X_tr  = [texts[i] for i in tr_idx]
        X_val = [texts[i] for i in val_idx]
        y_tr  = labels[tr_idx]
        y_val = labels[val_idx]

        t0 = time.perf_counter()
        m  = fold_fn(X_tr, y_tr, X_val, y_val, **kwargs)
        elapsed = time.perf_counter() - t0

        fold_results.append(m)
        print(f"  Fold {fold_idx+1}/{n_folds}  "
              f"F1={m['f1_macro']:.4f}  Ham={m['hamming']:.4f}  "
              f"Acc={m['accuracy']:.4f}  ({elapsed:.0f}s)")

    f1s  = [r['f1_macro'] for r in fold_results]
    hams = [r['hamming']  for r in fold_results]
    accs = [r['accuracy'] for r in fold_results]

    summary = {
        'name':      name,
        'f1_mean':   np.mean(f1s),  'f1_std':  np.std(f1s),
        'ham_mean':  np.mean(hams), 'ham_std': np.std(hams),
        'acc_mean':  np.mean(accs), 'acc_std': np.std(accs),
    }
    print(f"\n  F1-macro : {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"  Hamming  : {summary['ham_mean']:.4f} ± {summary['ham_std']:.4f}")
    print(f"  Accuracy : {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}")
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       default="augmented_dataset.csv",
                        help="CSV dosyası (varsayılan: augmented_dataset.csv)")
    parser.add_argument("--folds",     type=int, default=5)
    parser.add_argument("--epochs",    type=int, default=5)
    parser.add_argument("--batch",     type=int, default=8)
    parser.add_argument("--lr",        type=float, default=1e-5)
    parser.add_argument("--skip_bert", action="store_true",
                        help="Tüm BERT modellerini atla (sadece TF-IDF)")
    parser.add_argument("--models",    default="all",
                        help="Virgülle ayrılmış model listesi: tfidf,mlp,cnn,bilstm_cnn  (varsayılan: all)")
    args = parser.parse_args()

    # CSV'nin varlığını kontrol et
    if not os.path.isfile(args.csv):
        fallback = "dataset.csv"
        print(f"UYARI: {args.csv} bulunamadı, {fallback} kullanılıyor.")
        args.csv = fallback

    df     = pd.read_csv(args.csv)
    texts  = df[TEXT_COL].fillna("").astype(str).tolist()
    labels = (df[LABEL_COLS]
              .apply(pd.to_numeric, errors='coerce')
              .fillna(0.0).clip(0, 1).astype(int)
              .values.astype(float))

    bert_name = os.environ.get(
        "BERT_MODEL",
        "./BERTürk" if os.path.isdir("./BERTürk") else "dbmdz/bert-base-turkish-cased"
    )

    enabled = {m.strip().lower() for m in args.models.split(",")} if args.models != "all" else None

    def want(key):
        if args.skip_bert and key != "tfidf":
            return False
        return enabled is None or key in enabled

    print(f"\nVeri seti : {args.csv}  ({len(texts)} satır)")
    print(f"BERT model: {bert_name}")
    print(f"Ayarlar   : {args.folds}-fold | {args.epochs} epoch | batch={args.batch} | lr={args.lr}")
    print(f"Device    : {DEVICE}")

    summaries = []
    shared = dict(epochs=args.epochs, batch_size=args.batch, lr=args.lr, model_name=bert_name)

    # 1. TF-IDF + LogReg
    if want("tfidf"):
        summaries.append(run_cv(texts, labels, "TF-IDF(1-2gram) + LogReg",
                                fold_tfidf, n_folds=args.folds))

    # 2. BERTürk + MLP
    if want("mlp"):
        summaries.append(run_cv(texts, labels, "BERTürk + MLP",
                                fold_bert_arch, n_folds=args.folds,
                                model_class=BertMLP, **shared))

    # 3. BERTürk + CNN  ← ana hedef
    if want("cnn"):
        summaries.append(run_cv(texts, labels, "BERTürk + CNN",
                                fold_bert_arch, n_folds=args.folds,
                                model_class=BertCNNOnly, **shared))

    # 4. BERTürk + BiLSTM + CNN  (karşılaştırma referansı)
    if want("bilstm_cnn"):
        summaries.append(run_cv(texts, labels, "BERTürk + BiLSTM + CNN",
                                fold_bert_arch, n_folds=args.folds,
                                model_class=BertBiLSTMCNN, **shared))

    if not summaries:
        print("Hiçbir model seçilmedi.")
        return

    # ── Final tablo ───────────────────────────────────────────────────────────
    print(f"\n\n{'='*82}")
    print(f"{'Model':<35} {'F1-macro (mean±std)':>20} {'Hamming (mean±std)':>20} {'Acc (mean±std)':>14}")
    print("-" * 82)
    best_acc = max(s['acc_mean'] for s in summaries)
    for s in summaries:
        marker = " *" if s['acc_mean'] == best_acc else ""
        print(f"{s['name']:<35}"
              f"  {s['f1_mean']:.4f} ± {s['f1_std']:.4f}"
              f"  {s['ham_mean']:.4f} ± {s['ham_std']:.4f}"
              f"  {s['acc_mean']:.4f} ± {s['acc_std']:.4f}{marker}")
    print("=" * 82)
    print("* = en yüksek ortalama accuracy")

    # ── CSV kaydet ────────────────────────────────────────────────────────────
    out_rows = [{
        'Model':        s['name'],
        'F1_mean':      round(s['f1_mean'],  4),
        'F1_std':       round(s['f1_std'],   4),
        'Hamming_mean': round(s['ham_mean'], 4),
        'Hamming_std':  round(s['ham_std'],  4),
        'Acc_mean':     round(s['acc_mean'], 4),
        'Acc_std':      round(s['acc_std'],  4),
        'Dataset':      args.csv,
        'Epochs':       args.epochs,
        'Folds':        args.folds,
    } for s in summaries]
    pd.DataFrame(out_rows).to_csv("cv_results.csv", index=False)
    print("\nSonuclar cv_results.csv olarak kaydedildi.")


if __name__ == "__main__":
    main()
