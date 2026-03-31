"""
classifier.py
-------------
Kural tabanlı cinsiyet sınıflandırıcısı.
F0 (temel frekans) değerine göre Erkek / Kadın / Çocuk tahmini yapar.

Literatür eşikleri:
  Erkek  : 85  – 165 Hz
  Kadın  : 165 – 255 Hz
  Çocuk  : 255 – 500 Hz

Veri setinize göre calibrate_thresholds() ile ince ayar yapabilirsiniz.
"""

# ──────────────────────────────────────────────
# SINIF ETİKETLERİ
# ──────────────────────────────────────────────
LABEL_MAP = {
    "E": "Erkek",
    "K": "Kadın",
    "C": "Çocuk",
}

LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


# ──────────────────────────────────────────────
# VARSAYILAN EŞİKLER
# ──────────────────────────────────────────────
DEFAULT_THRESHOLDS = {
    "erkek_max" : 165,   # F0 < bu değer  → Erkek
    "kadin_max" : 255,   # F0 < bu değer  → Kadın (erkek_max ile arasında)
    # F0 >= kadin_max → Çocuk
}


# ──────────────────────────────────────────────
# SINIFLANDIRICI
# ──────────────────────────────────────────────

def classify(mean_f0, thresholds=None):
    """
    mean_f0    : ses dosyasının ortalama F0 değeri (Hz)
    thresholds : özel eşik değerleri (None → varsayılan kullanılır)

    Döndürür: ('E'|'K'|'C', 'Erkek'|'Kadın'|'Çocuk')
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    erkek_max = thresholds["erkek_max"]
    kadin_max = thresholds["kadin_max"]

    if mean_f0 <= 0:
        return "?", "Bilinmiyor"

    if mean_f0 < erkek_max:
        return "E", "Erkek"
    elif mean_f0 < kadin_max:
        return "K", "Kadın"
    else:
        return "C", "Çocuk"


# ──────────────────────────────────────────────
# DATASET ÜZERİNDE TOPLU DEĞERLENDİRME
# ──────────────────────────────────────────────

def evaluate(results_df, thresholds=None):
    """
    results_df : her satırda mean_f0 ve Cinsiyet (gerçek etiket) olan DataFrame
    Döndürür:
        accuracy    : genel doğruluk (0-1)
        predictions : tahmin listesi
        conf_matrix : 3x3 karışıklık matrisi (dict of dict)
    """
    import pandas as pd

    labels  = ["E", "K", "C"]
    correct = 0
    total   = 0
    predictions = []

    # Karışıklık matrisi başlat
    conf_matrix = {true: {pred: 0 for pred in labels} for true in labels}

    for _, row in results_df.iterrows():
        true_label = row["Cinsiyet"]
        mean_f0    = row.get("mean_f0", 0)

        pred_code, pred_name = classify(mean_f0, thresholds)

        predictions.append(pred_code)

        if true_label in labels and pred_code in labels:
            conf_matrix[true_label][pred_code] += 1
            if true_label == pred_code:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, predictions, conf_matrix


# ──────────────────────────────────────────────
# OTOMATİK EŞİK KALİBRASYONU (OPSİYONEL)
# ──────────────────────────────────────────────

def calibrate_thresholds(results_df):
    """
    Veri setindeki F0 ortalamalarına bakarak otomatik eşik önerir.
    results_df : mean_f0 ve Cinsiyet sütunları olan DataFrame
    """
    import numpy as np

    groups = {}
    for label in ["E", "K", "C"]:
        subset = results_df[results_df["Cinsiyet"] == label]["mean_f0"]
        if len(subset) > 0:
            groups[label] = {
                "mean": float(np.mean(subset)),
                "std" : float(np.std(subset)),
                "max" : float(np.max(subset)),
                "min" : float(np.min(subset)),
            }

    # Erkek-Kadın sınırı: ikisinin orta noktası
    erkek_max = 165
    kadin_max = 255

    if "E" in groups and "K" in groups:
        erkek_max = (groups["E"]["mean"] + groups["K"]["mean"]) / 2

    if "K" in groups and "C" in groups:
        kadin_max = (groups["K"]["mean"] + groups["C"]["mean"]) / 2

    return {
        "erkek_max": round(erkek_max, 1),
        "kadin_max": round(kadin_max, 1),
    }, groups
