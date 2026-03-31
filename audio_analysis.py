"""
audio_analysis.py
-----------------
Ses sinyalinden özellik çıkarma fonksiyonları:
  - Kısa Süreli Enerji (STE)
  - Sıfır Geçiş Oranı (ZCR)
  - Voiced bölge tespiti
  - Otokorelasyon ile F0 tespiti
"""

import numpy as np
import librosa


# ──────────────────────────────────────────────
# 1. SES YÜKLEME
# ──────────────────────────────────────────────

def load_audio(filepath, sr=22050):
    """
    Ses dosyasını yükler ve normalize eder.
    filepath : .wav dosyasının tam yolu
    sr       : örnekleme hızı (Hz)
    """
    audio, sr = librosa.load(filepath, sr=sr, mono=True)
    return audio, sr


# ──────────────────────────────────────────────
# 2. PENCERELENMİŞ ANALİZ
# ──────────────────────────────────────────────

def get_frames(audio, sr, frame_ms=25, hop_ms=10):
    """
    Sinyali örtüşen kısa pencerelere böler.
    frame_ms : pencere süresi (ms)
    hop_ms   : atlama adımı (ms)
    Döndürür: frames (2D array), frame_length, hop_length
    """
    frame_length = int(sr * frame_ms / 1000)
    hop_length   = int(sr * hop_ms  / 1000)

    # librosa.util.frame : (frame_length, num_frames)
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    return frames, frame_length, hop_length


# ──────────────────────────────────────────────
# 3. KISA SÜRELİ ENERJİ (STE)
# ──────────────────────────────────────────────

def compute_energy(frames):
    """
    Her pencere için kısa süreli enerji hesaplar.
    E[n] = sum(x[n]^2)
    """
    energy = np.sum(frames ** 2, axis=0)
    return energy


# ──────────────────────────────────────────────
# 4. SIFIR GEÇİŞ ORANI (ZCR)
# ──────────────────────────────────────────────

def compute_zcr(frames):
    """
    Her pencere için sıfır geçiş oranı hesaplar.
    ZCR[n] = (1/2N) * sum(|sign(x[n]) - sign(x[n-1])|)
    """
    signs      = np.sign(frames)
    sign_diff  = np.diff(signs, axis=0)
    zcr        = np.sum(np.abs(sign_diff), axis=0) / (2 * frames.shape[0])
    return zcr


# ──────────────────────────────────────────────
# 5. VOICED BÖLGE TESPİTİ
# ──────────────────────────────────────────────

def detect_voiced_frames(energy, zcr,
                          energy_threshold_ratio=0.05,
                          zcr_threshold=0.15):
    """
    Sesli (voiced) pencereleri tespit eder.
    - Enerji eşiğin üzerinde  → sesli aday
    - ZCR eşiğin altında      → voiced (periyodik sinyal)

    energy_threshold_ratio : max enerjinin yüzdesi
    zcr_threshold          : maksimum ZCR değeri
    """
    energy_threshold = energy_threshold_ratio * np.max(energy)
    voiced_mask      = (energy > energy_threshold) & (zcr < zcr_threshold)
    return voiced_mask


# ──────────────────────────────────────────────
# 6. OTOKORELASYONla F0 TESPİTİ (TEK PENCERE)
# ──────────────────────────────────────────────

def autocorrelation_f0(frame, sr, f0_min=50, f0_max=500):
    """
    Tek bir pencerede otokorelasyon yöntemiyle F0 hesaplar.

    R(τ) = Σ x[n] * x[n-τ]

    τ_min ve τ_max : F0 aralığına karşılık gelen lag değerleri
    """
    n        = len(frame)
    lag_min  = int(sr / f0_max)   # yüksek frekans → küçük lag
    lag_max  = int(sr / f0_min)   # düşük frekans  → büyük lag

    lag_max  = min(lag_max, n - 1)

    if lag_min >= lag_max:
        return 0.0

    # Otokorelasyon hesabı
    autocorr = np.correlate(frame, frame, mode='full')
    autocorr = autocorr[n - 1:]   # sadece pozitif lag'ler

    # İlgili lag aralığında en yüksek tepe noktasını bul
    segment  = autocorr[lag_min:lag_max]

    if len(segment) == 0 or np.max(segment) == 0:
        return 0.0

    peak_lag = np.argmax(segment) + lag_min

    if peak_lag == 0:
        return 0.0

    f0 = sr / peak_lag
    return f0


def get_autocorr_array(frame):
    """
    Grafik çizmek için otokorelasyon dizisini döndürür.
    """
    n        = len(frame)
    autocorr = np.correlate(frame, frame, mode='full')
    autocorr = autocorr[n - 1:]
    # Normalize et
    if autocorr[0] != 0:
        autocorr = autocorr / autocorr[0]
    return autocorr


# ──────────────────────────────────────────────
# 7. FFT SPEKTRUMU (KARŞILAŞTIRMA İÇİN)
# ──────────────────────────────────────────────

def compute_fft(frame, sr):
    """
    Pencere için magnitude spektrumu ve frekans eksenini döndürür.
    """
    N        = len(frame)
    windowed = frame * np.hanning(N)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs    = np.fft.rfftfreq(N, d=1.0 / sr)
    return freqs, spectrum


# ──────────────────────────────────────────────
# 8. ANA ÖZNİTELİK ÇIKARIMI (TÜM DOSYA)
# ──────────────────────────────────────────────

def extract_features(filepath, sr=22050):
    """
    Bir ses dosyasından tüm özellikleri çıkarır.

    Döndürür (dict):
        mean_f0    : ortalama temel frekans (Hz)
        std_f0     : F0 standart sapması
        mean_zcr   : ortalama ZCR
        mean_energy: ortalama enerji
        voiced_ratio: sesli pencere oranı
        f0_values  : tüm voiced pencere F0 değerleri (grafik için)
        energy_arr : enerji dizisi (grafik için)
        zcr_arr    : ZCR dizisi (grafik için)
        sample_frame: örnek pencere (otokorr/FFT grafiği için)
        sr         : örnekleme hızı
    """
    audio, sr = load_audio(filepath, sr=sr)

    frames, frame_length, hop_length = get_frames(audio, sr)

    energy = compute_energy(frames)
    zcr    = compute_zcr(frames)

    voiced_mask = detect_voiced_frames(energy, zcr)

    # Voiced pencerelerde F0 hesapla
    f0_values = []
    for i, is_voiced in enumerate(voiced_mask):
        if is_voiced:
            f0 = autocorrelation_f0(frames[:, i], sr)
            if f0 > 0:
                f0_values.append(f0)

    f0_values = np.array(f0_values)

    mean_f0  = float(np.mean(f0_values))  if len(f0_values) > 0 else 0.0
    std_f0   = float(np.std(f0_values))   if len(f0_values) > 0 else 0.0
    mean_zcr = float(np.mean(zcr))
    mean_energy = float(np.mean(energy))
    voiced_ratio = float(np.sum(voiced_mask) / len(voiced_mask))

    # Grafik için orta pencereyi al
    mid = frames.shape[1] // 2
    sample_frame = frames[:, mid]

    return {
        "mean_f0"     : mean_f0,
        "std_f0"      : std_f0,
        "mean_zcr"    : mean_zcr,
        "mean_energy" : mean_energy,
        "voiced_ratio": voiced_ratio,
        "f0_values"   : f0_values,
        "energy_arr"  : energy,
        "zcr_arr"     : zcr,
        "voiced_mask" : voiced_mask,
        "sample_frame": sample_frame,
        "audio"       : audio,
        "sr"          : sr,
        "hop_length"  : hop_length,
    }
