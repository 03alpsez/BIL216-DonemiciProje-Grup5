# 🎙️ Ses Analizi ve Cinsiyet Sınıflandırma
**Grup 05 | 2025-2026 Bahar Dönemi Dönemiçi Projesi**

---

## 📁 Proje Yapısı

```
proje/
├── app.py              → Streamlit arayüzü (ana dosya)
├── audio_analysis.py   → ZCR, Enerji, Otokorelasyon, F0 fonksiyonları
├── classifier.py       → Kural tabanlı sınıflandırıcı
├── data_loader.py      → Excel okuma ve dataset yönetimi
├── requirements.txt    → Gerekli kütüphaneler
└── README.md           → Bu dosya
```

---

## ⚙️ Kurulum

### 1. Python kurulu olduğunu doğrula (3.9 veya üzeri)
```bash
python --version
```

### 2. Sanal ortam oluştur (önerilen)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 3. Kütüphaneleri yükle
```bash
pip install -r requirements.txt
```

### 4. Uygulamayı çalıştır
```bash
streamlit run app.py
```
ÖNEMLİ******
cd "C:\Users\Alperen Sezer\Desktop\Donemici_Proje"
.\venv\Scripts\activate
streamlit run app.py

Tarayıcıda otomatik olarak `http://localhost:8501` açılır.

---

## 🖥️ Kullanım

### Sekme 1 – Tekli Ses Testi
1. Bir `.wav` dosyası yükle
2. Otomatik olarak tahmin gelir: **Erkek / Kadın / Çocuk**
3. Otokorelasyon, FFT, Enerji ve ZCR grafikleri görüntülenir

### Sekme 2 – Dataset Analizi
1. Grubunuzun `.xlsx` metadata dosyasını yükle
2. Tüm `.wav` dosyalarını seç (çoklu seçim)
3. **Analizi Başlat** butonuna bas
4. Accuracy, Confusion Matrix ve istatistiksel tablo görüntülenir

---

## 🔧 Eşik Ayarları (Sol Panel)
- **Erkek/Kadın sınırı:** Varsayılan 165 Hz
- **Kadın/Çocuk sınırı:** Varsayılan 255 Hz
- Sliderla değiştirerek kendi veri setinize göre ince ayar yapabilirsiniz

---

## 📊 Algoritma Özeti

```
1. Ses yükle (22050 Hz, mono)
2. 25ms pencereler + 10ms hop
3. Her pencere → Enerji + ZCR hesapla
4. Voiced tespiti: Enerji yüksek VE ZCR düşük
5. Voiced pencerelerde Otokorelasyon → F0
6. Ortalama F0 → Kural tabanlı sınıf kararı
```

### F0 Eşikleri (literatür)
| Sınıf | F0 Aralığı |
|-------|-----------|
| Erkek | < 165 Hz  |
| Kadın | 165–255 Hz|
| Çocuk | > 255 Hz  |

---

## 📦 Kullanılan Kütüphaneler
- **librosa** – ses yükleme
- **numpy** – sayısal hesaplamalar
- **pandas** – veri işleme
- **matplotlib** – grafikler
- **streamlit** – web arayüzü
- **openpyxl** – Excel okuma
