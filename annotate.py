"""
Dataset Annotation Helper
Etiketlenmemiş veya eksik etiketli gereksinimleri hızlıca etiketlemek için
terminal tabanlı araç.

Kullanım:
    python annotate.py --input dataset.csv --output dataset_extended.csv
    python annotate.py --input raw_requirements.csv --output dataset.csv --start 0
    python annotate.py --input dataset.csv --output dataset.csv --resume   # kaldığı yerden devam

Kontroller:
    1 → etiket mevcut (kalite iyi)
    0 → etiket eksik  (kalite sorunu)
    Enter (boş) → önceki değeri koru
    s → bu satırı atla
    q → kaydet ve çık
    ? → mevcut satırı tekrar göster
"""

import os
import argparse
import pandas as pd
import sys

from bil import LABEL_COLS, TEXT_COL

SAVE_EVERY = 10  # her 10 satırda otomatik kaydet


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def show_requirement(idx, total, text, current_labels):
    print(f"\n{'='*70}")
    print(f"Gereksinim [{idx+1}/{total}]")
    print(f"{'='*70}")
    print(f"\n  {text}\n")
    print(f"  Mevcut etiketler:")
    for j, label in enumerate(LABEL_COLS):
        val = current_labels[j]
        marker = "✓" if val == 1 else "✗" if val == 0 else "?"
        print(f"    {j+1:2}. {label:<15} [{marker}]")
    print()


def get_input_for_label(label: str, current_val) -> int | None:
    """Kullanıcıdan etiket değeri alır. None = atla (değiştirme)."""
    curr_str = str(int(current_val)) if current_val in (0, 1) else "?"
    while True:
        raw = input(f"  {label:<15} (şu an: {curr_str}) [0/1/Enter=değiştirme/s=satırı atla/q=çık]: ").strip().lower()
        if raw == "":
            return current_val  # değişiklik yok
        if raw == "q":
            return "QUIT"
        if raw == "s":
            return "SKIP"
        if raw in ("0", "1"):
            return int(raw)
        print("  → Geçersiz giriş. 0, 1, Enter, s veya q girin.")


def annotate(df: pd.DataFrame, start_idx: int = 0) -> pd.DataFrame:
    total = len(df)
    idx   = start_idx

    print(f"\nEtiketleme başlıyor. Satır {idx+1}/{total}'den devam ediliyor.")
    print("Kontroller: 0=eksik, 1=mevcut, Enter=değiştirme, s=satır atla, q=kaydet&çık\n")

    while idx < total:
        row  = df.iloc[idx]
        text = str(row[TEXT_COL])
        current_labels = [row[label] if label in df.columns else 0 for label in LABEL_COLS]

        show_requirement(idx, total, text, current_labels)

        new_labels = list(current_labels)
        skip_row   = False

        for j, label in enumerate(LABEL_COLS):
            result = get_input_for_label(label, current_labels[j])
            if result == "QUIT":
                print(f"\n  Kaydedilip çıkılıyor... (satır {idx+1} tamamlanmadı)")
                return df
            if result == "SKIP":
                skip_row = True
                break
            new_labels[j] = result

        if not skip_row:
            for j, label in enumerate(LABEL_COLS):
                df.at[idx, label] = new_labels[j]
            df.at[idx, '_annotated'] = 1
            print(f"  ✓ Satır {idx+1} kaydedildi.")
        else:
            print(f"  → Satır {idx+1} atlandı.")

        idx += 1

        # Otomatik kaydet
        if idx % SAVE_EVERY == 0:
            print(f"\n  [Otomatik kayıt: {idx}/{total}]")

    print(f"\n{'='*70}")
    print(f"Etiketleme tamamlandı! Toplam {idx - start_idx} satır işlendi.")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="dataset.csv",
                        help="Giriş CSV dosyası")
    parser.add_argument("--output", default="dataset_extended.csv",
                        help="Çıkış CSV dosyası")
    parser.add_argument("--start",  type=int, default=None,
                        help="Başlangıç satırı (varsayılan: otomatik)")
    parser.add_argument("--resume", action="store_true",
                        help="Daha önce kaldığı yerden devam et (_annotated sütununa göre)")
    parser.add_argument("--stats",  action="store_true",
                        help="Etiketleme istatistiklerini göster ve çık")
    args = parser.parse_args()

    # Giriş dosyasını yükle
    if not os.path.isfile(args.input):
        print(f"HATA: {args.input} bulunamadı.")
        sys.exit(1)

    df = pd.read_csv(args.input)

    # Gerekli sütunları ekle
    if TEXT_COL not in df.columns:
        print(f"HATA: '{TEXT_COL}' sütunu bulunamadı. Mevcut sütunlar: {list(df.columns)}")
        sys.exit(1)

    for label in LABEL_COLS:
        if label not in df.columns:
            df[label] = 0

    if '_annotated' not in df.columns:
        df['_annotated'] = 0

    # İstatistik modu
    if args.stats:
        total      = len(df)
        annotated  = int(df['_annotated'].sum())
        remaining  = total - annotated
        print(f"\n=== Etiketleme İstatistikleri ({args.input}) ===")
        print(f"  Toplam satır    : {total}")
        print(f"  Etiketlenen     : {annotated}  ({100*annotated/total:.1f}%)")
        print(f"  Kalan           : {remaining}  ({100*remaining/total:.1f}%)")
        print(f"\n  Etiket dağılımı (etiketlenmiş satırlar):")
        ann_df = df[df['_annotated'] == 1]
        if len(ann_df) > 0:
            for label in LABEL_COLS:
                pos = int(ann_df[label].sum())
                neg = len(ann_df) - pos
                print(f"    {label:<15}: {pos} mevcut ({100*pos/len(ann_df):.1f}%), "
                      f"{neg} eksik ({100*neg/len(ann_df):.1f}%)")
        return

    # Başlangıç satırını belirle
    if args.resume or args.start is None:
        # İlk etiketlenmemiş satırdan başla
        unannotated = df[df['_annotated'] != 1]
        start_idx = int(unannotated.index[0]) if len(unannotated) > 0 else len(df)
        if args.resume:
            print(f"Devam ediliyor: satır {start_idx+1}'den başlanıyor.")
    else:
        start_idx = args.start

    if start_idx >= len(df):
        print("Tüm satırlar zaten etiketlenmiş!")
        return

    # Etiketleme
    try:
        df = annotate(df, start_idx=start_idx)
    except KeyboardInterrupt:
        print("\n\n  Ctrl+C algılandı — kaydediliyor...")

    # Kaydet
    df.to_csv(args.output, index=False, encoding='utf-8-sig')
    annotated = int(df['_annotated'].sum())
    print(f"\nKaydedildi: {args.output}")
    print(f"Etiketlenen: {annotated}/{len(df)} satır")


if __name__ == "__main__":
    main()
