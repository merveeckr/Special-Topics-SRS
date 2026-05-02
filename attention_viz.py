"""
BERT Attention Heatmap Visualizer
Seçili gereksinim cümleleri için BERT'ün attention ağırlıklarını görselleştirir.
Kullanım:
    python attention_viz.py --csv yeni.csv --n 5 --layer 11 --head 0
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from transformers import AutoTokenizer, AutoModel

from bil import MODEL_NAME, TEXT_COL, MAX_LEN, DEVICE, LABEL_COLS, THRESH, BertBiLSTMCNN


def get_attention_weights(text: str, tokenizer, bert_model, device, max_len=MAX_LEN):
    enc = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt',
    )
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)

    with torch.no_grad():
        outputs = bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
    # attentions: tuple of (batch, heads, seq, seq), one per layer
    attentions = outputs.attentions  # len = num_layers

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    # Kırp [PAD] token'larını
    seq_len = int(attention_mask[0].sum().item())
    tokens = tokens[:seq_len]
    attn_layers = [a[0, :, :seq_len, :seq_len].cpu().numpy() for a in attentions]
    return tokens, attn_layers


def plot_attention_heatmap(tokens, attn_matrix, title: str, save_path: str):
    fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.5), max(5, len(tokens) * 0.4)))
    im = ax.imshow(attn_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=attn_matrix.max())
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("Anahtar (Key) Token'lar")
    ax.set_ylabel("Sorgu (Query) Token'lar")
    ax.set_title(title, fontsize=10, pad=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Kaydedildi: {save_path}")


def plot_mean_attention_bar(tokens, mean_attn_vec, title: str, save_path: str):
    """Her token'ın aldığı ortalama attention miktarını çubuk grafik olarak gösterir."""
    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.6), 3))
    colors = plt.cm.Blues(mean_attn_vec / (mean_attn_vec.max() + 1e-9))
    ax.bar(range(len(tokens)), mean_attn_vec, color=colors)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Ortalama Attention")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Kaydedildi: {save_path}")


def plot_all_heads_grid(tokens, attn_heads, layer_idx: int, save_path: str):
    """Bir katmanın tüm head'lerini grid olarak çizer."""
    n_heads = attn_heads.shape[0]
    cols = 4
    rows = (n_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    for h in range(n_heads):
        im = axes[h].imshow(attn_heads[h], cmap='Blues', aspect='auto')
        axes[h].set_title(f"Head {h}", fontsize=8)
        axes[h].axis('off')
    for h in range(n_heads, len(axes)):
        axes[h].axis('off')
    fig.suptitle(f"Layer {layer_idx} — Tüm Attention Head'leri\nTokenlar: {' '.join(tokens[:20])}", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Kaydedildi: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="BERT Attention Heatmap")
    parser.add_argument("--csv", default="dataset.csv")
    parser.add_argument("--model", default=None, help="BERT model path (varsayılan: bil.MODEL_NAME)")
    parser.add_argument("--n", type=int, default=3, help="Görselleştirilecek gereksinim sayısı")
    parser.add_argument("--layer", type=int, default=-1, help="Hangi BERT katmanı (-1=son katman)")
    parser.add_argument("--head", type=int, default=0, help="Hangi attention head")
    parser.add_argument("--out_dir", default="attention_plots", help="Çıktı klasörü")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bert_name = args.model or MODEL_NAME

    import pandas as pd
    df = pd.read_csv(args.csv)
    texts = df[TEXT_COL].fillna("").astype(str).tolist()[:args.n]

    print(f"Model yükleniyor: {bert_name}")
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    # Sadece BERT backbone — attention ağırlıkları için
    bert_model = AutoModel.from_pretrained(bert_name).to(DEVICE)
    bert_model.eval()

    for i, text in enumerate(texts):
        print(f"\nGereksinim {i+1}: {text[:80]}...")
        tokens, attn_layers = get_attention_weights(text, tokenizer, bert_model, DEVICE)

        layer_idx = args.layer if args.layer >= 0 else len(attn_layers) - 1
        layer_attn = attn_layers[layer_idx]  # (heads, seq, seq)
        head_attn = layer_attn[args.head]    # (seq, seq)
        mean_attn = layer_attn.mean(axis=0)  # (seq, seq) — tüm head'lerin ortalaması

        # 1. Seçili head'in heatmap'i
        plot_attention_heatmap(
            tokens, head_attn,
            title=f"Gereksinim {i+1} | Layer {layer_idx} Head {args.head}\n\"{text[:60]}...\"",
            save_path=os.path.join(args.out_dir, f"req{i+1}_layer{layer_idx}_head{args.head}.png")
        )

        # 2. Tüm head'lerin ortalaması — hangi tokenlara odaklanıldığı
        mean_per_token = mean_attn.mean(axis=0)  # her token'a gelen ortalama attention
        plot_mean_attention_bar(
            tokens, mean_per_token,
            title=f"Gereksinim {i+1} | Layer {layer_idx} — Ortalama Token Attention\n\"{text[:60]}...\"",
            save_path=os.path.join(args.out_dir, f"req{i+1}_layer{layer_idx}_mean_bar.png")
        )

        # 3. Son katmanın tüm head'leri grid
        plot_all_heads_grid(
            tokens, layer_attn, layer_idx,
            save_path=os.path.join(args.out_dir, f"req{i+1}_layer{layer_idx}_all_heads.png")
        )

    print(f"\nTüm görseller '{args.out_dir}/' klasörüne kaydedildi.")
    print("Rapor için önerilen görseller:")
    print(f"  - req*_layer{layer_idx}_head{args.head}.png   → seçili head heatmap")
    print(f"  - req*_layer{layer_idx}_mean_bar.png          → token importance bar")
    print(f"  - req*_layer{layer_idx}_all_heads.png         → tüm head'ler grid")


if __name__ == "__main__":
    main()
