"""
EDA: Easy Data Augmentation for Turkish Requirements Dataset
Referans: Wei & Zou, EMNLP 2019

Operasyonlar:
  SR  - Synonym Replacement  (Türkçe eş anlamlı kelime sözlüğü)
  RS  - Random Swap           (iki rastgele kelimeyi yer değiştir)
  RD  - Random Deletion       (rastgele kelime sil)
  RI  - Random Insertion      (sözlükten rastgele kelime ekle)

Strateji — sınıf dengesizliğine duyarlı artırma:
  Nadir negatifler (Appropriate/Correct/Feasible/Necessary, ≤6 örnek) → 8x
  Orta  negatifler (Singular/Conforming/Complete, ≤143 örnek)         → 4x
  Unambiguous (229 negatif)                                            → 3x
  Verifiable  (479 negatif)                                            → 2x
  Tamamen pozitif satırlar                                             → 1x

Kullanım:
  python eda_augment.py
  python eda_augment.py --input dataset.csv --output augmented_dataset.csv --seed 42
"""

import os
import re
import random
import argparse
import pandas as pd
import numpy as np

LABEL_COLS = ['Appropriate', 'Complete', 'Conforming', 'Correct', 'Feasible',
              'Necessary', 'Singular', 'Unambiguous', 'Verifiable']
TEXT_COL   = 'Requirement'

# ── Türkçe yazılım gereksinimi eş anlamlı sözlüğü ────────────────────────────
# Gereksinim metinlerinde sık geçen kelimeler + eş anlamlıları
SYNONYMS: dict[str, list[str]] = {
    # Eylemler
    "göster":          ["görüntüle", "listele", "sun"],
    "görüntüle":       ["göster", "listele", "sun"],
    "listele":         ["göster", "görüntüle", "sırala"],
    "kaydet":          ["sakla", "depola", "yaz"],
    "sakla":           ["kaydet", "depola", "koru"],
    "depola":          ["kaydet", "sakla", "arşivle"],
    "sil":             ["kaldır", "çıkar", "temizle"],
    "kaldır":          ["sil", "çıkar", "temizle"],
    "güncelle":        ["yenile", "değiştir", "düzenle"],
    "yenile":          ["güncelle", "değiştir", "düzenle"],
    "gönder":          ["ilet", "aktar", "transfer et"],
    "ilet":            ["gönder", "aktar", "yönlendir"],
    "oluştur":         ["yarat", "üret", "hazırla"],
    "yarat":           ["oluştur", "üret", "hazırla"],
    "doğrula":         ["onayla", "denetle", "kontrol et"],
    "onayla":          ["doğrula", "denetle", "onayla"],
    "denetle":         ["doğrula", "kontrol et", "incele"],
    "kontrol":         ["denetim", "doğrulama", "inceleme"],
    "ara":             ["sorgula", "bul", "filtrele"],
    "sorgula":         ["ara", "bul", "sorgula"],
    "raporla":         ["sun", "çıkar", "hazırla"],
    "yönet":           ["idare et", "kontrol et", "denetle"],
    "erişim":          ["giriş", "bağlantı", "ulaşım"],
    "giriş":           ["erişim", "bağlantı", "oturum açma"],
    "çalıştır":        ["başlat", "uygula", "işlet"],
    "başlat":          ["çalıştır", "aktive et", "aç"],
    "durdur":          ["sonlandır", "kapat", "iptal et"],
    "kapat":           ["durdur", "sonlandır", "bitir"],
    "hesapla":         ["belirle", "tespit et", "bul"],
    "belirle":         ["hesapla", "tanımla", "tespit et"],
    "tanımla":         ["belirle", "açıkla", "spesifik et"],
    "sağla":           ["sun", "ver", "temin et"],
    "sun":             ["sağla", "ver", "göster"],
    "bildir":          ["uyar", "haber ver", "raporla"],
    "uyar":            ["bildir", "ikaz et", "haber ver"],
    "izle":            ["takip et", "denetle", "gözlemle"],
    "takip":           ["izleme", "denetim", "kontrol"],
    "şifrele":         ["koru", "güvence altına al"],
    "filtrele":        ["ara", "sorgula", "sınırla"],
    "sınırla":         ["kısıtla", "filtrele", "engelle"],
    # İsimler
    "sistem":          ["uygulama", "yazılım", "platform"],
    "uygulama":        ["sistem", "yazılım", "program"],
    "yazılım":         ["uygulama", "sistem", "program"],
    "kullanıcı":       ["operatör", "kişi", "çalışan"],
    "operatör":        ["kullanıcı", "yetkili", "personel"],
    "veri":            ["bilgi", "kayıt", "içerik"],
    "bilgi":           ["veri", "kayıt", "içerik"],
    "kayıt":           ["veri", "bilgi", "log"],
    "rapor":           ["belge", "çıktı", "özet"],
    "belge":           ["rapor", "doküman", "dosya"],
    "dosya":           ["belge", "doküman", "kayıt"],
    "ekran":           ["arayüz", "panel", "görünüm"],
    "arayüz":          ["ekran", "panel", "kullanıcı arabirimi"],
    "panel":           ["ekran", "arayüz", "kontrol paneli"],
    "hata":            ["arıza", "sorun", "istisna"],
    "arıza":           ["hata", "sorun", "bozukluk"],
    "uyarı":           ["bildirim", "mesaj", "ikaz"],
    "bildirim":        ["uyarı", "mesaj", "ihbar"],
    "mesaj":           ["bildirim", "uyarı", "iletişim"],
    "yetki":           ["izin", "erişim hakkı", "ayrıcalık"],
    "izin":            ["yetki", "onay", "ruhsat"],
    "şifre":           ["parola", "kimlik bilgisi", "kod"],
    "parola":          ["şifre", "kimlik bilgisi"],
    "performans":      ["hız", "verim", "etkinlik"],
    "hız":             ["performans", "sürat", "yanıt süresi"],
    "güvenlik":        ["koruma", "emniyet", "savunma"],
    "koruma":          ["güvenlik", "emniyet", "güvence"],
    "veritabanı":      ["veri tabanı", "depo", "ambar"],
    "bağlantı":        ["erişim", "giriş", "link"],
    "süre":            ["zaman", "periyot", "interval"],
    "zaman":           ["süre", "vakit", "periyot"],
    "limit":           ["sınır", "kısıt", "eşik"],
    "sınır":           ["limit", "kısıt", "üst sınır"],
    "format":          ["biçim", "yapı", "şablon"],
    "biçim":           ["format", "yapı", "düzen"],
    "liste":           ["tablo", "dizin", "katalog"],
    "tablo":           ["liste", "çizelge", "matris"],
    # Modal fiil ekleri (kısmi eşleştirme için kısa formlar)
    "melidir":         ["malıdır", "gerekir"],
    "malıdır":         ["melidir", "gerekir"],
    "meli":            ["malı", "gerekli"],
    "bilir":           ["ebilir", "yapabilir"],
}


def _split_words(text: str) -> list[str]:
    return text.split()


def _join_words(words: list[str]) -> str:
    return " ".join(words)


def synonym_replacement(text: str, n: int = 1) -> str:
    words = _split_words(text)
    candidates = [(i, w) for i, w in enumerate(words) if w.lower().rstrip(".,;:!?") in SYNONYMS]
    random.shuffle(candidates)
    replaced = words[:]
    for i, w in candidates[:n]:
        clean = w.lower().rstrip(".,;:!?")
        punct = w[len(clean):]
        syn = random.choice(SYNONYMS[clean])
        replaced[i] = syn + punct
    return _join_words(replaced)


def random_insertion(text: str, n: int = 1) -> str:
    words = _split_words(text)
    all_syns = [s for syns in SYNONYMS.values() for s in syns]
    for _ in range(n):
        insert_word = random.choice(all_syns)
        pos = random.randint(0, len(words))
        words.insert(pos, insert_word)
    return _join_words(words)


def random_swap(text: str, n: int = 1) -> str:
    words = _split_words(text)
    if len(words) < 2:
        return text
    for _ in range(n):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return _join_words(words)


def random_deletion(text: str, p: float = 0.1) -> str:
    words = _split_words(text)
    if len(words) == 1:
        return text
    result = [w for w in words if random.random() > p]
    return _join_words(result) if result else words[0]


def eda(text: str, alpha: float = 0.1, n_aug: int = 4) -> list[str]:
    """
    Bir metinden n_aug adet artırılmış versiyon üret.
    alpha: her operasyonda değiştirilecek kelime oranı
    """
    n_ops = max(1, int(alpha * len(_split_words(text))))
    augmented = []
    ops = [synonym_replacement, random_insertion, random_swap,
           lambda t, _n: random_deletion(t, p=alpha)]
    for i in range(n_aug):
        op = ops[i % len(ops)]
        try:
            aug = op(text, n_ops)
        except Exception:
            aug = text
        if aug.strip() and aug != text:
            augmented.append(aug)
    if not augmented:
        augmented = [text]
    return augmented[:n_aug]


# ── Artırma miktarını sınıf bazında belirle ───────────────────────────────────

def get_augmentation_factor(row: pd.Series) -> int:
    """Her satır için artırma çarpanını döndür (orijinal hariç)."""
    labels = pd.to_numeric(pd.Series({c: row[c] for c in LABEL_COLS}), errors='coerce').fillna(0).values

    # Nadir negatifler: Appropriate/Correct/Feasible/Necessary (≤6 negatif)
    rare_neg_cols = ['Appropriate', 'Correct', 'Feasible', 'Necessary']
    if any(labels[LABEL_COLS.index(c)] == 0 for c in rare_neg_cols):
        return 8

    # Orta nadir: Complete, Conforming (≤55 negatif)
    mid_neg_cols = ['Complete', 'Conforming']
    if any(labels[LABEL_COLS.index(c)] == 0 for c in mid_neg_cols):
        return 4

    # Singular
    if labels[LABEL_COLS.index('Singular')] == 0:
        return 4

    # Unambiguous
    if labels[LABEL_COLS.index('Unambiguous')] == 0:
        return 3

    # Verifiable
    if labels[LABEL_COLS.index('Verifiable')] == 0:
        return 2

    # Tamamen pozitif
    return 1


def augment_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    original_rows = df.to_dict('records')
    augmented_rows = []

    stats = {'total_orig': len(df), 'augmented': 0, 'skipped': 0}

    for row in original_rows:
        text = str(row[TEXT_COL])
        n_aug = get_augmentation_factor(pd.Series(row))
        if n_aug == 0:
            continue
        aug_texts = eda(text, alpha=0.15, n_aug=n_aug)
        for aug_text in aug_texts:
            new_row = row.copy()
            new_row[TEXT_COL] = aug_text
            augmented_rows.append(new_row)
            stats['augmented'] += 1

    augmented_df = pd.DataFrame(augmented_rows)
    combined = pd.concat([df, augmented_df], ignore_index=True)
    stats['total_final'] = len(combined)
    return combined, stats


def print_stats(df_orig: pd.DataFrame, df_aug: pd.DataFrame) -> None:
    print("\n=== Artırma Sonrası Etiket Dağılımı ===")
    print(f"{'Etiket':<15} {'Önce neg':>10} {'Sonra neg':>10} {'Artış':>8}")
    print("-" * 47)
    labels_orig = df_orig[LABEL_COLS].apply(pd.to_numeric, errors='coerce').fillna(0)
    labels_aug  = df_aug[LABEL_COLS].apply(pd.to_numeric, errors='coerce').fillna(0)
    for col in LABEL_COLS:
        n_before = int((labels_orig[col] == 0).sum())
        n_after  = int((labels_aug[col] == 0).sum())
        mult = f"{n_after/max(1,n_before):.1f}x" if n_before > 0 else "—"
        print(f"  {col:<13} {n_before:>10} {n_after:>10} {mult:>8}")
    print(f"\nToplam satir: {len(df_orig):,} -> {len(df_aug):,} "
          f"(+{len(df_aug)-len(df_orig):,} yeni ornek)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="dataset.csv")
    parser.add_argument("--output", default="augmented_dataset.csv")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Orijinal veri seti: {len(df)} satır")

    df_aug, stats = augment_dataset(df, seed=args.seed)
    df_aug.to_csv(args.output, index=False, encoding='utf-8-sig')

    print_stats(df, df_aug)
    print(f"\nArtırılmış veri seti kaydedildi: {args.output}")


if __name__ == "__main__":
    main()
