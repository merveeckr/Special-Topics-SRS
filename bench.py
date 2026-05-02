import os
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, hamming_loss, accuracy_score

from transformers import AutoTokenizer

from sentence_transformers import SentenceTransformer

from transformers import AutoModel

from bil import (
    BertBiLSTMCNN,
    LABEL_COLS,
    TEXT_COL,
    MAX_LEN,
    DEVICE,
    THRESH,
)


# ── Ablasyon mimarileri ────────────────────────────────────────────────────────

class BertMLP(nn.Module):
    """Sadece BERT [CLS] çıktısı → MLP."""
    def __init__(self, bert_model_name, dropout=0.3, num_labels=9):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)


class BertBiLSTMOnly(nn.Module):
    """BERT → BiLSTM → mean-pool → MLP (CNN yok)."""
    def __init__(self, bert_model_name, lstm_hidden=256, dropout=0.3, num_labels=9):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        hidden = self.bert.config.hidden_size
        self.lstm = nn.LSTM(
            input_size=hidden, hidden_size=lstm_hidden,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        lstm_out, _ = self.lstm(out.last_hidden_state)
        # attention_mask ile geçerli token'ların ortalaması
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (lstm_out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.classifier(pooled)


class BertCNNOnly(nn.Module):
    """BERT → Multi-channel CNN → max-pool → concat → MLP (BiLSTM yok)."""
    def __init__(self, bert_model_name, cnn_out=128, kernel_sizes=(2, 3, 4), dropout=0.3, num_labels=9):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        hidden = self.bert.config.hidden_size
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden, cnn_out, k) for k in kernel_sizes
        ])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out * len(kernel_sizes), 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        x = out.last_hidden_state.permute(0, 2, 1)  # (B, H, L)
        cat = torch.cat([self.pool(conv(x)).squeeze(-1) for conv in self.convs], dim=1)
        return self.classifier(cat)
# Offline/SSL settings (corporate environments)
OFFLINE = os.environ.get("HF_OFFLINE", "0") == "1"
try:
    import certifi  # type: ignore
    ca = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", ca)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
except Exception:
    pass



@dataclass
class BenchmarkResult:
    name: str
    train_seconds: float
    infer_seconds_per_sample: float
    val_f1_macro: float
    val_hamming: float
    val_accuracy: float


def load_dataset(csv_path: str, test_size: float = 0.15, seed: int = 42):
    df = pd.read_csv(csv_path)
    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    # Clean labels: coerce -> fillna 0 -> clip to {0,1} -> int
    lab = df[LABEL_COLS].apply(pd.to_numeric, errors='coerce').fillna(0.0).clip(0.0, 1.0).astype(int)
    labels = lab.values.astype(float)
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, shuffle=True
    )
    return X_train, X_val, y_train, y_val


class SimpleTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }
        return item


def run_bert_bilstm_cnn(csv_path: str, model_name: str, epochs: int = 1, batch_size: int = 8, lr: float = 1e-5) -> BenchmarkResult:
    X_train, X_val, y_train, y_val = load_dataset(csv_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=OFFLINE)

    train_ds = SimpleTextDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_ds = SimpleTextDataset(X_val, y_val, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = BertBiLSTMCNN(bert_model_name=model_name, num_labels=len(LABEL_COLS)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    from tqdm import tqdm
    t0 = time.perf_counter()
    model.train()
    for ep in range(epochs):
        loop = tqdm(train_loader, desc=f"BERT epoch {ep+1}/{epochs}", leave=True)
        for batch in loop:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loop.set_postfix(loss=f"{loss.item():.4f}")
    train_seconds = time.perf_counter() - t0

    # Eval
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        # warmup
        for batch in val_loader:
            model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            break
        t1 = time.perf_counter()
        n_samples = 0
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].cpu().numpy()
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= THRESH).astype(int)
            all_preds.append(preds)
            all_labels.append(labels)
            n_samples += input_ids.size(0)
        t2 = time.perf_counter()
    infer_seconds_per_sample = (t2 - t1) / max(1, n_samples)

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    ham = hamming_loss(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    return BenchmarkResult(
        name=f"BERT+BiLSTM+CNN ({model_name})",
        train_seconds=train_seconds,
        infer_seconds_per_sample=infer_seconds_per_sample,
        val_f1_macro=f1m,
        val_hamming=ham,
        val_accuracy=acc,
    )


def run_tfidf_logreg(csv_path: str, threshold: float = 0.5) -> BenchmarkResult:
    from sklearn.feature_extraction.text import TfidfVectorizer
    X_train, X_val, y_train, y_val = load_dataset(csv_path)

    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    emb_train = vectorizer.fit_transform(X_train)
    emb_val = vectorizer.transform(X_val)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, n_jobs=-1))
    clf.fit(emb_train, y_train)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    probs = clf.predict_proba(emb_val)
    preds = (probs >= threshold).astype(int)
    t3 = time.perf_counter()

    f1m = f1_score(y_val, preds, average='macro', zero_division=0)
    ham = hamming_loss(y_val, preds)
    acc = accuracy_score(y_val, preds)

    return BenchmarkResult(
        name="TF-IDF(1-2gram)+LogReg",
        train_seconds=t1 - t0,
        infer_seconds_per_sample=(t3 - t2) / max(1, len(X_val)),
        val_f1_macro=f1m,
        val_hamming=ham,
        val_accuracy=acc,
    )


def run_dropout_ablation(csv_path: str, model_name: str, dropout_rates: List[float] = None, epochs: int = 1, batch_size: int = 8) -> List[BenchmarkResult]:
    """BertBiLSTMCNN'i farklı dropout oranlarıyla çalıştırarak ablation study yapar."""
    if dropout_rates is None:
        dropout_rates = [0.1, 0.3, 0.5]

    X_train, X_val, y_train, y_val = load_dataset(csv_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=OFFLINE)
    train_ds = SimpleTextDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_ds = SimpleTextDataset(X_val, y_val, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    results = []
    for dropout in dropout_rates:
        print(f"  dropout={dropout} ...")
        model = BertBiLSTMCNN(bert_model_name=model_name, num_labels=len(LABEL_COLS), dropout=dropout).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        loss_fn = nn.BCEWithLogitsLoss()

        from tqdm import tqdm
        t0 = time.perf_counter()
        model.train()
        for ep in range(epochs):
            loop = tqdm(train_loader, desc=f"  Ablation dropout={dropout} epoch {ep+1}/{epochs}", leave=True)
            for batch in loop:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loop.set_postfix(loss=f"{loss.item():.4f}")
        train_seconds = time.perf_counter() - t0

        model.eval()
        all_preds, all_labels = [], []
        n_samples = 0
        with torch.no_grad():
            t1 = time.perf_counter()
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                logits = model(input_ids, attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= THRESH).astype(int)
                all_preds.append(preds)
                all_labels.append(batch['labels'].cpu().numpy())
                n_samples += input_ids.size(0)
            t2 = time.perf_counter()

        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_labels)
        results.append(BenchmarkResult(
            name=f"BERT+BiLSTM+CNN (dropout={dropout})",
            train_seconds=train_seconds,
            infer_seconds_per_sample=(t2 - t1) / max(1, n_samples),
            val_f1_macro=f1_score(y_true, y_pred, average='macro', zero_division=0),
            val_hamming=hamming_loss(y_true, y_pred),
            val_accuracy=accuracy_score(y_true, y_pred),
        ))
    return results


def _train_eval_model(model, train_loader, val_loader, epochs, label, lr=1e-5):
    """Genel eğit-değerlendir yardımcı fonksiyon."""
    from tqdm import tqdm
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    t0 = time.perf_counter()
    model.train()
    for ep in range(epochs):
        loop = tqdm(train_loader, desc=f"  {label} epoch {ep+1}/{epochs}", leave=True)
        for batch in loop:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loop.set_postfix(loss=f"{loss.item():.4f}")
    train_seconds = time.perf_counter() - t0

    model.eval()
    all_preds, all_labels_list = [], []
    n_samples = 0
    with torch.no_grad():
        t1 = time.perf_counter()
        for batch in val_loader:
            logits = model(batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append((probs >= THRESH).astype(int))
            all_labels_list.append(batch['labels'].cpu().numpy())
            n_samples += batch['input_ids'].size(0)
        t2 = time.perf_counter()

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels_list)
    return BenchmarkResult(
        name=label,
        train_seconds=train_seconds,
        infer_seconds_per_sample=(t2 - t1) / max(1, n_samples),
        val_f1_macro=f1_score(y_true, y_pred, average='macro', zero_division=0),
        val_hamming=hamming_loss(y_true, y_pred),
        val_accuracy=accuracy_score(y_true, y_pred),
    )


def run_arch_ablation(csv_path: str, model_name: str, epochs: int = 5, batch_size: int = 8) -> List[BenchmarkResult]:
    """
    Mimari ablasyon: 4 varyantı aynı koşullarda eğitir ve karşılaştırır.
      1. BERTürk + MLP          (ne LSTM ne CNN)
      2. BERTürk + BiLSTM       (sadece LSTM, CNN yok)
      3. BERTürk + CNN           (sadece CNN, LSTM yok)
      4. BERTürk + BiLSTM + CNN  (tam model)
    """
    X_train, X_val, y_train, y_val = load_dataset(csv_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=OFFLINE)
    train_ds = SimpleTextDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_ds   = SimpleTextDataset(X_val,   y_val,   tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    variants = [
        ("BERTürk + MLP",          BertMLP(model_name,          num_labels=len(LABEL_COLS))),
        ("BERTürk + BiLSTM",       BertBiLSTMOnly(model_name,   num_labels=len(LABEL_COLS))),
        ("BERTürk + CNN",          BertCNNOnly(model_name,       num_labels=len(LABEL_COLS))),
        ("BERTürk + BiLSTM + CNN", BertBiLSTMCNN(model_name,    num_labels=len(LABEL_COLS))),
    ]

    results = []
    for label, model in variants:
        print(f"\n  [{label}] eğitiliyor ({epochs} epoch)...")
        model = model.to(DEVICE)
        try:
            res = _train_eval_model(model, train_loader, val_loader, epochs, label)
            results.append(res)
            print(f"  → F1={res.val_f1_macro:.4f}  Hamming={res.val_hamming:.4f}  Acc={res.val_accuracy:.4f}")
        except Exception as e:
            print(f"  HATA [{label}]: {e}")
    return results


def run_st_embedding_logreg(csv_path: str, st_model_name: str, threshold: float = 0.5) -> BenchmarkResult:
    X_train, X_val, y_train, y_val = load_dataset(csv_path)
    # If OFFLINE, st_model_name must be a local directory
    if OFFLINE and not os.path.isdir(st_model_name):
        raise RuntimeError(f"OFFLINE=1: SentenceTransformer path not found: {st_model_name}")
    st_model = SentenceTransformer(st_model_name)

    # Encode
    t0 = time.perf_counter()
    emb_train = st_model.encode(X_train, batch_size=64, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    emb_val = st_model.encode(X_val, batch_size=64, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    t1 = time.perf_counter()

    # Train One-vs-Rest Logistic Regression
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, n_jobs=-1))
    clf.fit(emb_train, y_train)
    t2 = time.perf_counter()

    # Inference
    t3 = time.perf_counter()
    probs = clf.predict_proba(emb_val)
    preds = (probs >= threshold).astype(int)
    t4 = time.perf_counter()

    f1m = f1_score(y_val, preds, average='macro', zero_division=0)
    ham = hamming_loss(y_val, preds)
    acc = accuracy_score(y_val, preds)

    train_seconds = (t1 - t0) + (t2 - t1)
    infer_seconds_per_sample = (t4 - t3) / max(1, len(X_val))

    return BenchmarkResult(
        name=f"ST+LogReg ({st_model_name})",
        train_seconds=train_seconds,
        infer_seconds_per_sample=infer_seconds_per_sample,
        val_f1_macro=f1m,
        val_hamming=ham,
        val_accuracy=acc,
    )


def main():
    default_csv = "dataset.csv"
    csv_path = os.environ.get("CSV_PATH", default_csv)

    # Varsayılan BERT: önce yerel BERTürk klasörünü dene
    default_bert = "./BERTürk" if os.path.isdir("./BERTürk") else "dbmdz/bert-base-turkish-cased"
    bert_model = os.environ.get("BERT_MODEL", default_bert)

    # ST modelleri: ST_MODELS env değişkeni verilmişse onu kullan,
    # internet yoksa (OFFLINE=1 veya ST_MODELS="") boş liste ile atla
    st_env = os.environ.get("ST_MODELS", "")
    if OFFLINE:
        candidates_st = []
        print("OFFLINE mod: Sentence-Transformers atlanıyor.")
    elif st_env.strip() == "SKIP":
        candidates_st = []
        print("ST_MODELS=SKIP: Sentence-Transformers atlanıyor.")
    elif st_env.strip():
        sep = ";" if ";" in st_env else ","
        candidates_st = [s.strip() for s in st_env.split(sep) if s.strip()]
    else:
        candidates_st = [
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "intfloat/multilingual-e5-small",
            "intfloat/multilingual-e5-base",
            "sentence-transformers/LaBSE",
        ]

    results: List[BenchmarkResult] = []

    print("Running TF-IDF(1-2gram) + LogisticRegression ...")
    try:
        res_tfidf = run_tfidf_logreg(csv_path)
        results.append(res_tfidf)
    except Exception as e:
        print("TF-IDF+LogReg failed:", e)

    print("\nRunning Sentence-Transformers + LogisticRegression ...")
    for st_name in candidates_st:
        try:
            res_st = run_st_embedding_logreg(csv_path, st_name)
            results.append(res_st)
        except Exception as e:
            print(f"{st_name} failed:", e)

    print("\nRunning BERT+BiLSTM+CNN fine-tune (1 epoch) ...")
    try:
        res_bert = run_bert_bilstm_cnn(csv_path, bert_model, epochs=int(os.environ.get("EPOCHS", "1")))
        results.append(res_bert)
    except Exception as e:
        print("BERT+BiLSTM+CNN failed:", e)

    print("\nRunning Dropout Ablation Study (dropout ∈ {0.1, 0.3, 0.5}) ...")
    try:
        ablation_results = run_dropout_ablation(csv_path, bert_model, epochs=int(os.environ.get("EPOCHS", "1")))
        results.extend(ablation_results)
    except Exception as e:
        print("Dropout ablation failed:", e)

    arch_results: List[BenchmarkResult] = []
    if os.environ.get("ARCH_ABLATION", "1") != "0":
        print("\nRunning Architecture Ablation (MLP / BiLSTM / CNN / BiLSTM+CNN) ...")
        try:
            arch_results = run_arch_ablation(
                csv_path, bert_model,
                epochs=int(os.environ.get("EPOCHS", "5")),
            )
        except Exception as e:
            print("Architecture ablation failed:", e)

    if results or arch_results:
        print("\n=== Benchmark Results ===")
        print(f"{'Model':<45} {'Train(s)':>9} {'Infer(ms)':>10} {'F1-macro':>9} {'Hamming':>8} {'Acc':>7}")
        print("-" * 95)
        for r in results:
            print(f"{r.name:<45} {r.train_seconds:>9.2f} {r.infer_seconds_per_sample*1000:>10.2f} {r.val_f1_macro:>9.4f} {r.val_hamming:>8.4f} {r.val_accuracy:>7.4f}")

        print("\n=== Dropout Ablation Summary ===")
        ablation = [r for r in results if "dropout=" in r.name]
        if ablation:
            print(f"{'Dropout':>10} {'F1-macro':>10} {'Hamming':>10} {'Acc':>8}")
            print("-" * 44)
            for r in ablation:
                d = r.name.split("dropout=")[1].rstrip(")")
                print(f"{d:>10} {r.val_f1_macro:>10.4f} {r.val_hamming:>10.4f} {r.val_accuracy:>8.4f}")

    if arch_results:
        print("\n=== Architecture Ablation Summary ===")
        print(f"{'Mimari':<30} {'F1-macro':>10} {'Hamming':>10} {'Acc':>8}")
        print("-" * 62)
        best_f1 = max(r.val_f1_macro for r in arch_results)
        for r in arch_results:
            marker = " ◄ EN İYİ" if r.val_f1_macro == best_f1 else ""
            print(f"{r.name:<30} {r.val_f1_macro:>10.4f} {r.val_hamming:>10.4f} {r.val_accuracy:>8.4f}{marker}")
    else:
        print("No results produced.")


if __name__ == "__main__":
    main()

