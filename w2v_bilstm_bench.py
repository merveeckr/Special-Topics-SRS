"""
Word2Vec + BiLSTM Baseline
BERT+BiLSTM+CNN ile karşılaştırma amacıyla statik Word2Vec embedding + BiLSTM mimarisi.
Gösterir: contextual (BERT) vs statik (Word2Vec) embedding farkı.

Kullanım:
    python w2v_bilstm_bench.py --csv yeni.csv
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss, accuracy_score

from bil import TEXT_COL, LABEL_COLS, DEVICE, THRESH


# ── Word2Vec helpers ────────────────────────────────────────────────────────

def tokenize_turkish(text: str):
    """Basit boşluk/noktalama tabanlı tokenizer."""
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def train_word2vec(sentences, vector_size=128, window=5, min_count=1, epochs=10):
    from gensim.models import Word2Vec
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs,
        seed=42,
    )
    return model


def text_to_indices(text, vocab, unk_idx=0):
    return [vocab.get(w, unk_idx) for w in tokenize_turkish(text)]


# ── Dataset ─────────────────────────────────────────────────────────────────

class W2VDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=64):
        self.seqs = [text_to_indices(t, vocab) for t in texts]
        self.labels = labels.astype(np.float32)
        self.max_len = max_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx][:self.max_len]
        pad = [0] * (self.max_len - len(seq))
        seq = seq + pad
        return {
            'input': torch.tensor(seq, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float),
        }


# ── Model ────────────────────────────────────────────────────────────────────

class W2VBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, lstm_hidden=128, num_labels=9,
                 dropout=0.3, pretrained_weights=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        if pretrained_weights is not None:
            # pretrained_weights: (vocab_size+1, embed_dim) numpy array
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weights))

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_labels),
        )

    def forward(self, x):
        emb = self.embedding(x)           # (B, T, E)
        out, _ = self.lstm(emb)           # (B, T, 2*H)
        pooled = out.max(dim=1).values    # (B, 2*H)
        return self.classifier(pooled)


# ── Main ─────────────────────────────────────────────────────────────────────

def run(csv_path: str, epochs: int = 5, batch_size: int = 16, lr: float = 1e-3,
        embed_dim: int = 128, lstm_hidden: int = 128, max_len: int = 64):

    df = pd.read_csv(csv_path)
    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    labels = df[LABEL_COLS].apply(pd.to_numeric, errors='coerce').fillna(0.0).clip(0.0, 1.0).astype(int).values.astype(float)

    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.15, random_state=42)

    # 1. Word2Vec eğitimi
    print("Word2Vec eğitiliyor ...")
    t0 = time.perf_counter()
    tokenized = [tokenize_turkish(t) for t in X_train]
    w2v = train_word2vec(tokenized, vector_size=embed_dim)
    vocab = {word: idx + 1 for idx, word in enumerate(w2v.wv.index_to_key)}  # 0 = PAD

    # Pretrained embedding matrisi
    embed_matrix = np.zeros((len(vocab) + 1, embed_dim), dtype=np.float32)
    for word, idx in vocab.items():
        embed_matrix[idx] = w2v.wv[word]

    # 2. Dataset & DataLoader
    train_ds = W2VDataset(X_train, y_train, vocab, max_len=max_len)
    val_ds = W2VDataset(X_val, y_val, vocab, max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. Model
    model = W2VBiLSTM(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        lstm_hidden=lstm_hidden,
        num_labels=len(LABEL_COLS),
        pretrained_weights=embed_matrix,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # 4. Eğitim
    best_f1, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            x = batch['input'].to(DEVICE)
            y = batch['labels'].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        # Val
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch['input'].to(DEVICE))
                preds = (torch.sigmoid(logits).cpu().numpy() >= THRESH).astype(int)
                all_preds.append(preds)
                all_labels.append(batch['labels'].numpy())
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_labels)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        print(f"  Epoch {epoch+1}/{epochs} — loss: {epoch_loss/len(train_ds):.4f} | F1-macro: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    train_seconds = time.perf_counter() - t0

    # 5. Final inference timing
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_labels = [], []
    n_samples = 0
    with torch.no_grad():
        t1 = time.perf_counter()
        for batch in val_loader:
            logits = model(batch['input'].to(DEVICE))
            preds = (torch.sigmoid(logits).cpu().numpy() >= THRESH).astype(int)
            all_preds.append(preds)
            all_labels.append(batch['labels'].numpy())
            n_samples += batch['input'].size(0)
        t2 = time.perf_counter()

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    ham = hamming_loss(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    infer_ms = (t2 - t1) / max(1, n_samples) * 1000

    print("\n=== Word2Vec + BiLSTM Results ===")
    print(f"  Train time  : {train_seconds:.2f}s  (W2V eğitimi + model eğitimi)")
    print(f"  Infer       : {infer_ms:.2f} ms/sample")
    print(f"  F1-macro    : {f1m:.4f}")
    print(f"  Hamming     : {ham:.4f}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Vocab size  : {len(vocab)}")

    return {"name": "Word2Vec+BiLSTM", "train_s": train_seconds, "infer_ms": infer_ms,
            "f1_macro": f1m, "hamming": ham, "accuracy": acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="dataset.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--lstm_hidden", type=int, default=128)
    args = parser.parse_args()

    run(
        csv_path=args.csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embed_dim=args.embed_dim,
        lstm_hidden=args.lstm_hidden,
    )
