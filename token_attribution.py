"""
Token-Level Quality Attribution (Sequence Labeling Proxy)
Her gereksinim cümlesinde hangi kelimenin hangi kalite boyutunu ihlal ettiğini
gradient saliency ile tespit eder.

Yöntem: Integrated Gradients (basitleştirilmiş) — model hangi token'a bakarak
        bir etiketi 0 tahmin etti?

Çıktı:
  - token_attribution_results.csv  : her token için skor tablosu
  - token_attribution_plots/       : HTML highlight dosyaları (renk kodlu)

Kullanım:
    python token_attribution.py --csv dataset.csv --n 20 --threshold 0.3
    python token_attribution.py --model_path best_model.pt --n 50
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from transformers import AutoTokenizer

from bil import BertBiLSTMCNN, LABEL_COLS, TEXT_COL, MAX_LEN, DEVICE

OFFLINE = os.environ.get("HF_OFFLINE", "0") == "1"


# ── Model yükleme ─────────────────────────────────────────────────────────────

def load_model(model_name, model_path=None):
    model = BertBiLSTMCNN(bert_model_name=model_name, num_labels=len(LABEL_COLS)).to(DEVICE)
    if model_path and os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Ağırlıklar yüklendi: {model_path}")
    else:
        print("best_model.pt bulunamadı — pretrained BERT ağırlıkları kullanılıyor.")
    model.eval()
    return model


# ── Gradient saliency ─────────────────────────────────────────────────────────

def get_token_saliency(model, tokenizer, text: str):
    """
    Her token için saliency skoru döndürür.
    Yöntem: register_full_backward_hook — retain_grad() gerektirmez.
    Dönüş: tokens (list[str]), saliency (array shape [seq_len, n_labels])
    """
    enc = tokenizer(
        text, truncation=True, padding='max_length',
        max_length=MAX_LEN, return_tensors='pt'
    )
    input_ids      = enc['input_ids'].to(DEVICE)
    attention_mask = enc['attention_mask'].to(DEVICE)
    seq_len        = int(attention_mask[0].sum().item())

    emb_fwd  = {}   # forward: embedding değerleri
    emb_grad = {}   # backward: embedding gradyanları

    def fwd_hook(module, inp, out):
        emb_fwd['val'] = out.detach().clone()          # değerleri kaydet

    def bwd_hook(module, grad_in, grad_out):  # module/grad_in unused by design
        emb_grad['val'] = grad_out[0].detach().clone() # gradyanları yakala

    h_fwd = model.bert.embeddings.word_embeddings.register_forward_hook(fwd_hook)
    h_bwd = model.bert.embeddings.word_embeddings.register_full_backward_hook(bwd_hook)

    # Olasılıklar için gradient'siz forward
    with torch.no_grad():
        logits_det = model(input_ids, attention_mask)
    probs = torch.sigmoid(logits_det)[0].cpu().numpy()

    saliency = np.zeros((seq_len, len(LABEL_COLS)), dtype=np.float32)

    for label_idx in range(len(LABEL_COLS)):
        model.zero_grad()
        emb_grad.clear()

        model.train()
        with torch.enable_grad():
            logits = model(input_ids, attention_mask)
            logits[0, label_idx].backward()

        emb  = emb_fwd.get('val')
        grad = emb_grad.get('val')
        if emb is not None and grad is not None:
            score = (grad[0, :seq_len] * emb[0, :seq_len]).norm(dim=-1).cpu().numpy()
            if score.max() > 0:
                score = score / score.max()
            saliency[:, label_idx] = score

    h_fwd.remove()
    h_bwd.remove()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())[:seq_len]
    preds  = (probs >= 0.5).astype(int)

    return tokens, saliency, preds, probs


# ── HTML görselleştirme ───────────────────────────────────────────────────────

def _heat_color(score: float) -> str:
    """0→beyaz, 1→kırmızı arası renk."""
    r = 255
    g = int(255 * (1 - score))
    b = int(255 * (1 - score))
    return f"rgb({r},{g},{b})"


def save_html(tokens, saliency, preds, probs, text, req_idx, label, label_idx, out_dir):
    """Belirli bir label için token highlight HTML dosyası."""
    scores = saliency[:, label_idx]
    html_parts = [
        "<html><head><meta charset='utf-8'>"
        "<style>body{font-family:Calibri,sans-serif;margin:30px;}"
        ".token{display:inline-block;padding:2px 4px;margin:2px;border-radius:3px;font-size:16px;}"
        ".info{color:#555;font-size:13px;margin-bottom:8px;}"
        "</style></head><body>",
        f"<h3>Gereksinim #{req_idx + 1} — <b>{label}</b> "
        f"(Tahmin: {'✓ Mevcut' if preds[label_idx] == 1 else '✗ Eksik'}, "
        f"Olasılık: {probs[label_idx]:.3f})</h3>",
        f"<p class='info'>{text}</p><hr/><p>",
    ]
    for tok, sc in zip(tokens, scores):
        if tok in ('[PAD]', '[CLS]', '[SEP]'):
            continue
        word = tok.replace('##', '')
        color = _heat_color(float(sc))
        html_parts.append(
            f"<span class='token' style='background:{color};' title='saliency={sc:.3f}'>{word}</span>"
        )
    html_parts.append("</p></body></html>")

    fname = os.path.join(out_dir, f"req{req_idx+1}_{label}.html")
    with open(fname, 'w', encoding='utf-8') as f:
        f.write("".join(html_parts))
    return fname


# ── CSV özeti ────────────────────────────────────────────────────────────────

def build_summary_row(req_idx, text, tokens, saliency, preds, probs, top_k=3):
    """Tahmin edilen eksik etiket başına en salienty top-k token'ı döndürür."""
    rows = []
    for label_idx, label in enumerate(LABEL_COLS):
        if preds[label_idx] == 1:
            continue  # sadece eksik etiketler
        scores = saliency[:, label_idx]
        # [CLS],[SEP],[PAD] hariç tut
        valid_idx = [i for i, t in enumerate(tokens) if t not in ('[PAD]', '[CLS]', '[SEP]')]
        valid_scores = [(scores[i], tokens[i].replace('##', '')) for i in valid_idx]
        valid_scores.sort(reverse=True)
        top_tokens = [t for _, t in valid_scores[:top_k]]
        rows.append({
            'req_idx':       req_idx + 1,
            'requirement':   text[:120],
            'missing_label': label,
            'prob':          round(float(probs[label_idx]), 4),
            'top_tokens':    ', '.join(top_tokens),
            'saliency_scores': ', '.join(f"{s:.3f}" for s, _ in valid_scores[:top_k]),
        })
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        default="dataset.csv")
    parser.add_argument("--n",          type=int, default=20, help="Analiz edilecek satır sayısı")
    parser.add_argument("--model_path", default="best_model.pt")
    parser.add_argument("--out_dir",    default="token_attribution_plots")
    parser.add_argument("--top_k",      type=int, default=3, help="Her etiket için top-k token")
    parser.add_argument("--html",       action="store_true", help="HTML dosyaları üret")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bert_name = os.environ.get("BERT_MODEL",
                "./BERTürk" if os.path.isdir("./BERTürk") else "dbmdz/bert-base-turkish-cased")

    print(f"Model yükleniyor: {bert_name}")
    tokenizer = AutoTokenizer.from_pretrained(bert_name, local_files_only=OFFLINE)
    model     = load_model(bert_name, args.model_path)

    df    = pd.read_csv(args.csv)
    texts = df[TEXT_COL].fillna("").astype(str).tolist()[:args.n]

    all_rows = []
    print(f"\n{args.n} gereksinim için token attribution hesaplanıyor...\n")

    for i, text in enumerate(texts):
        print(f"  [{i+1}/{args.n}] {text[:70]}...")
        try:
            tokens, saliency, preds, probs = get_token_saliency(model, tokenizer, text)
        except Exception as e:
            print(f"    HATA: {e}")
            continue

        missing = [LABEL_COLS[j] for j, p in enumerate(preds) if p == 0]
        if not missing:
            print(f"    → Tüm etiketler mevcut, atlıyorum.")
            continue

        print(f"    Eksik etiketler: {missing}")

        # CSV satırları
        rows = build_summary_row(i, text, tokens, saliency, preds, probs, top_k=args.top_k)
        all_rows.extend(rows)

        # HTML görseller
        if args.html:
            for label_idx, label in enumerate(LABEL_COLS):
                if preds[label_idx] == 0:
                    fname = save_html(tokens, saliency, preds, probs,
                                      text, i, label, label_idx, args.out_dir)
                    print(f"    → {fname}")

    # Sonuçları kaydet
    if all_rows:
        out_csv = "token_attribution_results.csv"
        pd.DataFrame(all_rows).to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"\nSonuçlar kaydedildi: {out_csv}")
        print(f"Toplam {len(all_rows)} eksik etiket-gereksinim çifti analiz edildi.\n")

        # Örnek çıktı
        print("=== Örnek Sonuçlar ===")
        sample = pd.DataFrame(all_rows).head(10)
        for _, row in sample.iterrows():
            print(f"  Req #{row['req_idx']} | {row['missing_label']:<15} "
                  f"| İhlaline yol açan tokenlar: [{row['top_tokens']}]")
    else:
        print("Eksik etiketli gereksinim bulunamadı.")

    print("\n--- Etiket bazında token ihlal sıklığı ---")
    if all_rows:
        df_res = pd.DataFrame(all_rows)
        freq = df_res['missing_label'].value_counts()
        for label, count in freq.items():
            print(f"  {label:<15}: {count} gereksinimde eksik")


if __name__ == "__main__":
    main()
