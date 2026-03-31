"""
app.py
------
Streamlit tabanlı Ses Analizi ve Cinsiyet Sınıflandırma Arayüzü
Grup 05 – 2025-2026 Bahar Dönemi Dönemiçi Projesi
"""

import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from audio_analysis import extract_features, get_autocorr_array, compute_fft
from classifier    import classify, evaluate, calibrate_thresholds, DEFAULT_THRESHOLDS, LABEL_MAP
from data_loader   import load_single_group, load_all_metadata, check_files, dataset_summary


# ══════════════════════════════════════════════
# SAYFA AYARLARI
# ══════════════════════════════════════════════

st.set_page_config(
    page_title  = "Ses Analizi – Grup 05",
    page_icon   = "🎙️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ══════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Genel Arka Plan ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #070B14 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Material Icons — asla ezme ── */
.material-symbols-rounded,
.material-symbols-outlined,
.material-icons,
span.notranslate {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0D1220 !important;
    border-right: 1px solid #1E2940 !important;
    min-width: 260px !important;
}
[data-testid="stSidebarContent"] {
    padding: 1rem 1.2rem !important;
}

/* ── Tüm metinler — ikon span'ları hariç ── */
p, li, label {
    color: #B8C4D8 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
div:not(.material-symbols-rounded):not(.material-icons) {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
h1, h2, h3, h4 {
    color: #E8EFF8 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
}

/* ── Slider değerleri ── */
[data-testid="stSlider"] {
    padding-bottom: 4px !important;
}
[data-testid="stTickBarMin"],
[data-testid="stTickBarMax"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #4A6080 !important;
    white-space: nowrap !important;
}

/* ── Tab bar ── */
[data-testid="stTabs"] button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #6B7FA3 !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00E5CC !important;
    border-bottom: 2px solid #00E5CC !important;
    background: #0D1829 !important;
}

/* ── Metrik kartları ── */
[data-testid="stMetric"] {
    background: #0D1829 !important;
    border: 1px solid #1E2F4A !important;
    border-radius: 12px !important;
    padding: 16px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stMetric"]:hover { border-color: #00E5CC !important; }
[data-testid="stMetricLabel"] {
    color: #6B7FA3 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stMetricValue"] {
    color: #E8EFF8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.5rem !important;
}
[data-testid="stMetricDelta"] { color: #00E5CC !important; }

/* ── Butonlar ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #00C9AF, #006BFF) !important;
    color: #fff !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    letter-spacing: 0.03em !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 20px #00C9AF33 !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Download butonu ── */
[data-testid="stDownloadButton"] > button {
    background: #0D1829 !important;
    border: 1px solid #00E5CC !important;
    color: #00E5CC !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0D1829 !important;
    border: 1.5px dashed #1E3050 !important;
    border-radius: 14px !important;
    padding: 8px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: #00E5CC !important; }

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #00C9AF, #006BFF) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0D1829 !important;
    border: 1px solid #1E2F4A !important;
    border-radius: 12px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #1E2F4A !important;
}

/* ── Alert kutuları ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 4px !important;
}

/* ── Divider ── */
hr { border-color: #1E2940 !important; }

/* ── Code ── */
code {
    font-family: 'JetBrains Mono', monospace !important;
    color: #00E5CC !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div style="
    padding: 36px 0 24px 0;
    border-bottom: 1px solid #1E2940;
    margin-bottom: 28px;
">
    <div style="display:flex; align-items:center; gap:16px;">
        <div style="
            width: 52px; height: 52px;
            background: linear-gradient(135deg, #00C9AF22, #006BFF22);
            border: 1px solid #00C9AF55;
            border-radius: 14px;
            display: flex; align-items: center; justify-content: center;
            font-size: 26px;
        ">🎙️</div>
        <div>
            <h1 style="
                margin: 0;
                font-size: 1.65rem;
                font-weight: 800;
                letter-spacing: -0.01em;
                font-family: 'Plus Jakarta Sans', sans-serif;
                background: linear-gradient(90deg, #E8EFF8 30%, #00E5CC);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            ">Ses İşareti Analizi ve Cinsiyet Sınıflandırma</h1>
            <p style="margin:4px 0 0 0; color:#4A6080; font-size:0.8rem; letter-spacing:0.05em; text-transform:uppercase; font-family:'Plus Jakarta Sans', sans-serif;">
                Grup 05 &nbsp;·&nbsp; 2025–2026 Bahar Dönemi &nbsp;·&nbsp; Dönemiçi Proje
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SIDEBAR – EŞİK AYARLARI
# ══════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding: 20px 0 8px 0;">
        <p style="font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase;
                  color:#3A5070; margin:0 0 6px 0; font-family:'Plus Jakarta Sans',sans-serif;">Kontrol Paneli</p>
        <h2 style="font-size:1.05rem; font-weight:700; color:#E8EFF8;
                   margin:0; font-family:'Plus Jakarta Sans',sans-serif;">Eşik Değerleri</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#111D30; border:1px solid #1E3050; border-radius:10px;
                padding:14px 16px; margin:12px 0; font-size:0.82rem; line-height:1.8;
                font-family:'Plus Jakarta Sans',sans-serif;">
        <span style="color:#00E5CC; font-weight:700;">F0 (Hz)</span>
        <span style="color:#6B7FA3;"> değerine göre karar:</span><br>
        <span style="color:#5B8EFF;">▸ Erkek</span>
        <span style="color:#4A6080;"> — F0 &lt; Erkek Eşiği</span><br>
        <span style="color:#FF6B8A;">▸ Kadın</span>
        <span style="color:#4A6080;"> — arası</span><br>
        <span style="color:#00E5A0;">▸ Çocuk</span>
        <span style="color:#4A6080;"> — F0 ≥ Kadın Eşiği</span>
    </div>
    """, unsafe_allow_html=True)

    erkek_max = st.slider(
        "Erkek / Kadın Sınırı (Hz)",
        min_value=100, max_value=220,
        value=DEFAULT_THRESHOLDS["erkek_max"],
        step=5,
    )
    kadin_max = st.slider(
        "Kadın / Çocuk Sınırı (Hz)",
        min_value=200, max_value=400,
        value=DEFAULT_THRESHOLDS["kadin_max"],
        step=5,
    )

    thresholds = {"erkek_max": erkek_max, "kadin_max": kadin_max}

    st.markdown("<hr style='border-color:#1E2940; margin:20px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <p style="font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase;
              color:#3A5070; margin:0 0 10px 0; font-family:'Plus Jakarta Sans',sans-serif;">Renk Kodu</p>
    <div style="display:flex; flex-direction:column; gap:8px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:10px;height:10px;border-radius:50%;background:#5B8EFF;flex-shrink:0;"></div>
            <span style="color:#B8C4D8; font-size:0.85rem; font-family:'Plus Jakarta Sans',sans-serif;">Erkek</span>
        </div>
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:10px;height:10px;border-radius:50%;background:#FF6B8A;flex-shrink:0;"></div>
            <span style="color:#B8C4D8; font-size:0.85rem; font-family:'Plus Jakarta Sans',sans-serif;">Kadın</span>
        </div>
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:10px;height:10px;border-radius:50%;background:#00E5A0;flex-shrink:0;"></div>
            <span style="color:#B8C4D8; font-size:0.85rem; font-family:'Plus Jakarta Sans',sans-serif;">Çocuk</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SEKMELER
# ══════════════════════════════════════════════

tab1, tab2 = st.tabs(["🎵 Tekli Ses Testi", "📊 Dataset Analizi"])


# ──────────────────────────────────────────────
# SEKME 1 – TEKLİ SES TESTİ
# ──────────────────────────────────────────────

with tab1:

    st.markdown("""
    <div style="padding:20px 0 16px 0;">
        <h2 style="font-size:1.2rem; font-weight:800; color:#E8EFF8; margin:0 0 4px 0;">
            Tek Ses Dosyası Analizi
        </h2>
        <p style="color:#4A6080; font-size:0.82rem; margin:0;">
            Bir .wav dosyası yükleyin — Otokorelasyon ile F0 hesaplanır, sınıf tahmini yapılır.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Bir .wav dosyası yükleyin",
        type=["wav"],
        help="Analiz edilecek ses dosyasını seçin.",
    )

    if uploaded is not None:

        # Geçici dosyaya yaz (librosa disk üzerinden okur)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Ses analizi yapılıyor..."):
            try:
                feats = extract_features(tmp_path)
            except Exception as e:
                st.error(f"Analiz hatası: {e}")
                os.unlink(tmp_path)
                st.stop()

        os.unlink(tmp_path)

        mean_f0 = feats["mean_f0"]
        pred_code, pred_name = classify(mean_f0, thresholds)

        # ── Tahmin Sonucu ──
        color_map = {"E": "#5B8EFF", "K": "#FF6B8A", "C": "#00E5A0", "?": "#6B7FA3"}
        icon_map  = {"E": "♂", "K": "♀", "C": "◉", "?": "?"}
        color     = color_map.get(pred_code, "#6B7FA3")
        icon      = icon_map.get(pred_code, "?")

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {color}12, {color}06);
            border: 1px solid {color}40;
            border-left: 4px solid {color};
            padding: 20px 28px;
            border-radius: 14px;
            margin: 20px 0 24px 0;
            display: flex;
            align-items: center;
            gap: 20px;
        ">
            <div style="
                width: 56px; height: 56px; flex-shrink:0;
                background: {color}20;
                border: 1.5px solid {color}50;
                border-radius: 14px;
                display: flex; align-items: center; justify-content: center;
                font-size: 1.8rem; color: {color};
            ">{icon}</div>
            <div>
                <p style="margin:0; font-size:0.72rem; letter-spacing:0.1em;
                          text-transform:uppercase; color:{color}99;">Tahmin Sonucu</p>
                <h2 style="margin:2px 0 4px 0; font-size:1.8rem; font-weight:800;
                           color:{color}; letter-spacing:-0.02em;">{pred_name}</h2>
                <p style="margin:0; color:#4A6080; font-size:0.85rem; font-family:'JetBrains Mono',monospace;">
                    Ort. F0 = <span style="color:{color}; font-weight:600;">{mean_f0:.1f} Hz</span>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Özellik Metrikleri ──
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ortalama F0",    f"{feats['mean_f0']:.1f} Hz")
        c2.metric("F0 Std Sapma",   f"{feats['std_f0']:.1f} Hz")
        c3.metric("Ortalama ZCR",   f"{feats['mean_zcr']:.4f}")
        c4.metric("Sesli Oran",     f"%{feats['voiced_ratio']*100:.1f}")

        st.markdown("<hr style='border-color:#1E2940; margin:8px 0 20px 0;'>", unsafe_allow_html=True)

        # ── Grafikler ──
        st.markdown("""
        <p style="font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase;
                  color:#3A5070; margin:0 0 14px 0;">📈 Sinyal Grafikleri</p>
        """, unsafe_allow_html=True)

        col_l, col_r = st.columns(2)

        # Sol: Otokorelasyon vs FFT
        with col_l:
            st.markdown("<p style='color:#6B7FA3; font-size:0.82rem; font-weight:600; margin-bottom:8px;'>Otokorelasyon & FFT Karşılaştırması</p>", unsafe_allow_html=True)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5))
            fig.patch.set_facecolor("#0A0F1E")
            for ax in (ax1, ax2):
                ax.set_facecolor("#0D1829")
                ax.tick_params(colors="#6B7FA3", labelsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#1E2F4A")

            frame   = feats["sample_frame"]
            sr      = feats["sr"]

            # Otokorelasyon
            autocorr = get_autocorr_array(frame)
            lags     = np.arange(len(autocorr))
            ax1.plot(lags[:len(autocorr)//2], autocorr[:len(autocorr)//2],
                     color="#5B8EFF", linewidth=1.2)
            ax1.set_title("Otokorelasyon R(τ)", color="#B8C4D8", fontsize=9, pad=6)
            ax1.set_xlabel("Lag (örnek)", color="#4A6080", fontsize=7)
            ax1.set_ylabel("R(τ)", color="#4A6080", fontsize=7)

            # FFT
            freqs, spectrum = compute_fft(frame, sr)
            mask = freqs < 800
            ax2.plot(freqs[mask], spectrum[mask], color="#FF6B8A", linewidth=1.2)
            ax2.set_title("FFT Büyüklük Spektrumu |X(f)|", color="#B8C4D8", fontsize=9, pad=6)
            ax2.set_xlabel("Frekans (Hz)", color="#4A6080", fontsize=7)
            ax2.set_ylabel("|X(f)|", color="#4A6080", fontsize=7)

            if mean_f0 > 0:
                ax2.axvline(mean_f0, color="#00E5A0", linestyle="--",
                            linewidth=1.5, label=f"F0={mean_f0:.0f} Hz")
                ax2.legend(fontsize=7, facecolor="#0D1829", labelcolor="#B8C4D8", edgecolor="#1E2F4A")

            fig.tight_layout(pad=1.5)
            st.pyplot(fig)
            plt.close(fig)

        # Sağ: Enerji & ZCR zaman serisi
        with col_r:
            st.markdown("<p style='color:#6B7FA3; font-size:0.82rem; font-weight:600; margin-bottom:8px;'>Enerji ve ZCR Zaman Serisi</p>", unsafe_allow_html=True)
            fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(6, 5))
            fig2.patch.set_facecolor("#0A0F1E")
            for ax in (ax3, ax4):
                ax.set_facecolor("#0D1829")
                ax.tick_params(colors="#6B7FA3", labelsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#1E2F4A")

            energy_arr  = feats["energy_arr"]
            zcr_arr     = feats["zcr_arr"]
            voiced_mask = feats["voiced_mask"]
            hop         = feats["hop_length"]
            times       = np.arange(len(energy_arr)) * hop / sr

            ax3.plot(times, energy_arr, color="#5B8EFF", linewidth=1.2)
            ax3.fill_between(times, energy_arr,
                             where=voiced_mask[:len(times)],
                             alpha=0.25, color="#00E5A0", label="Voiced")
            ax3.set_title("Kısa Süreli Enerji", color="#B8C4D8", fontsize=9, pad=6)
            ax3.set_xlabel("Zaman (s)", color="#4A6080", fontsize=7)
            ax3.legend(fontsize=7, facecolor="#0D1829", labelcolor="#B8C4D8", edgecolor="#1E2F4A")

            ax4.plot(times, zcr_arr[:len(times)], color="#FFB347", linewidth=1.2)
            ax4.axhline(0.15, color="#FF6B8A", linestyle="--",
                        linewidth=1, label="Eşik=0.15")
            ax4.set_title("Sıfır Geçiş Oranı (ZCR)", color="#B8C4D8", fontsize=9, pad=6)
            ax4.set_xlabel("Zaman (s)", color="#4A6080", fontsize=7)
            ax4.legend(fontsize=7, facecolor="#0D1829", labelcolor="#B8C4D8", edgecolor="#1E2F4A")

            fig2.tight_layout(pad=1.5)
            st.pyplot(fig2)
            plt.close(fig2)

        # F0 Dağılım Histogramı
        if len(feats["f0_values"]) > 0:
            st.markdown("<p style='color:#6B7FA3; font-size:0.82rem; font-weight:600; margin:16px 0 8px 0;'>F0 Dağılımı (Voiced Pencereler)</p>", unsafe_allow_html=True)
            fig3, ax5 = plt.subplots(figsize=(8, 2.5))
            fig3.patch.set_facecolor("#0A0F1E")
            ax5.set_facecolor("#0D1829")
            ax5.tick_params(colors="#6B7FA3", labelsize=7)
            for spine in ax5.spines.values():
                spine.set_edgecolor("#1E2F4A")

            ax5.hist(feats["f0_values"], bins=30, color="#5B8EFF",
                     edgecolor="#0A0F1E", alpha=0.85)
            ax5.axvline(mean_f0, color="#00E5A0", linewidth=2,
                        linestyle="--", label=f"Ort. F0={mean_f0:.0f} Hz")
            ax5.axvline(erkek_max, color="#6B7FA3", linewidth=1,
                        linestyle=":", label=f"E/K={erkek_max} Hz")
            ax5.axvline(kadin_max, color="#6B7FA3", linewidth=1,
                        linestyle="-.", label=f"K/Ç={kadin_max} Hz")
            ax5.set_xlabel("F0 (Hz)", color="#4A6080", fontsize=8)
            ax5.set_ylabel("Pencere sayısı", color="#4A6080", fontsize=8)
            ax5.set_title("F0 Histogramı", color="#B8C4D8", fontsize=9)
            ax5.legend(fontsize=7, facecolor="#0D1829", labelcolor="#B8C4D8", edgecolor="#1E2F4A")
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)


# ──────────────────────────────────────────────
# SEKME 2 – DATASET ANALİZİ
# ──────────────────────────────────────────────

with tab2:

    st.markdown("""
    <div style="padding:20px 0 16px 0;">
        <h2 style="font-size:1.2rem; font-weight:800; color:#E8EFF8; margin:0 0 4px 0;">
            Tüm Dataset Üzerinde Analiz
        </h2>
        <p style="color:#4A6080; font-size:0.82rem; margin:0;">
            Excel metadata ve ses dosyalarını yükleyin — toplu analiz, Confusion Matrix ve istatistiksel özet.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#0D1829; border:1px solid #1E3050; border-radius:12px;
                padding:16px 20px; margin-bottom:20px; font-size:0.82rem; line-height:1.8;">
        <span style="color:#00E5CC; font-weight:700; font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase;">
            📁 Klasör Yapısı
        </span><br>
        <code style="color:#B8C4D8; font-size:0.78rem; white-space:pre;">Dataset/
  Grup_05/
    Grup_05_MetaVeri.xlsx
    G05_D01_C_09_Notr_C1.wav
    ...</code>
    </div>
    """, unsafe_allow_html=True)

    col_up1, col_up2 = st.columns(2)

    with col_up1:
        excel_upload = st.file_uploader(
            "📋 Metadata Excel'ini yükleyin (.xlsx)",
            type=["xlsx"],
            key="excel_upload",
        )

    with col_up2:
        wav_uploads = st.file_uploader(
            "🎵 Ses dosyalarını yükleyin (.wav) — çoklu seçim",
            type=["wav"],
            accept_multiple_files=True,
            key="wav_upload",
        )

    if excel_upload and wav_uploads:

        with st.spinner("Dosyalar hazırlanıyor..."):

            # Geçici klasör oluştur
            tmp_dir = tempfile.mkdtemp()

            # Excel kaydet
            excel_path = os.path.join(tmp_dir, excel_upload.name)
            with open(excel_path, "wb") as f:
                f.write(excel_upload.read())

            # WAV dosyalarını kaydet
            wav_saved = {}
            for wf in wav_uploads:
                wp = os.path.join(tmp_dir, wf.name)
                with open(wp, "wb") as f:
                    f.write(wf.read())
                wav_saved[wf.name] = wp

        df_meta = load_single_group(excel_path, wav_dir=tmp_dir)

        # Dosya varlık kontrolü
        n_ok, missing = check_files(df_meta)
        st.success(f"✅ {n_ok} / {len(df_meta)} ses dosyası eşleşti.")
        if missing:
            with st.expander(f"⚠️ {len(missing)} eksik dosya"):
                st.write(missing)

        # Analiz butonu
        if st.button("🚀 Analizi Başlat", type="primary"):

            results = []
            progress = st.progress(0)
            status   = st.empty()

            valid_rows = df_meta[df_meta["wav_path"].apply(os.path.exists)]

            for i, (_, row) in enumerate(valid_rows.iterrows()):
                status.text(f"Analiz ediliyor: {row['Dosya_Adi']} ({i+1}/{len(valid_rows)})")
                try:
                    feats = extract_features(row["wav_path"])
                    pred_code, pred_name = classify(feats["mean_f0"], thresholds)
                    results.append({
                        "Dosya_Adi"   : row["Dosya_Adi"],
                        "Cinsiyet"    : row["Cinsiyet"],
                        "Yas"         : row["Yas"],
                        "Duygu"       : row.get("Duygu", "-"),
                        "mean_f0"     : round(feats["mean_f0"], 1),
                        "std_f0"      : round(feats["std_f0"],  1),
                        "mean_zcr"    : round(feats["mean_zcr"], 4),
                        "voiced_ratio": round(feats["voiced_ratio"], 3),
                        "Tahmin"      : pred_code,
                        "Dogru_mu"    : "✅" if pred_code == row["Cinsiyet"] else "❌",
                    })
                except Exception as e:
                    results.append({
                        "Dosya_Adi": row["Dosya_Adi"],
                        "Cinsiyet" : row["Cinsiyet"],
                        "Tahmin"   : "?",
                        "Dogru_mu" : "⚠️",
                        "mean_f0"  : 0,
                        "Hata"     : str(e),
                    })
                progress.progress((i + 1) / len(valid_rows))

            status.empty()
            progress.empty()

            res_df = pd.DataFrame(results)
            st.session_state["results_df"] = res_df

        # ── Sonuçları Göster ──
        if "results_df" in st.session_state:
            res_df = st.session_state["results_df"]

            # Otomatik eşik kalibrasyonu
            cal_thresh, groups = calibrate_thresholds(res_df)

            with st.expander("💡 Otomatik Eşik Önerisi (veriye göre)"):
                st.write(f"Önerilen Erkek/Kadın sınırı: **{cal_thresh['erkek_max']} Hz**")
                st.write(f"Önerilen Kadın/Çocuk sınırı: **{cal_thresh['kadin_max']} Hz**")
                st.caption("Sol paneldeki sliderları bu değerlere ayarlayabilirsiniz.")

            # Accuracy & Confusion Matrix
            accuracy, preds, conf = evaluate(res_df, thresholds)

            st.markdown("<hr style='border-color:#1E2940; margin:20px 0;'>", unsafe_allow_html=True)
            st.markdown("""
            <p style="font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase;
                      color:#3A5070; margin:0 0 14px 0;">📊 Sınıflandırma Başarısı</p>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Genel Doğruluk", f"%{accuracy*100:.1f}")

            for col_widget, label in zip([m2, m3, m4], ["E", "K", "C"]):
                subset = res_df[res_df["Cinsiyet"] == label]
                if len(subset) > 0:
                    sub_acc = (subset["Tahmin"] == subset["Cinsiyet"]).mean()
                    col_widget.metric(
                        f"{LABEL_MAP[label]} Doğruluk",
                        f"%{sub_acc*100:.1f}",
                        delta=f"{len(subset)} kayıt",
                    )

            # Confusion Matrix
            st.markdown("""
            <p style="font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase;
                      color:#3A5070; margin:20px 0 10px 0;">🔲 Karışıklık Matrisi (Confusion Matrix)</p>
            """, unsafe_allow_html=True)
            labels = ["E", "K", "C"]
            label_names = ["Erkek", "Kadın", "Çocuk"]

            conf_data = [[conf.get(t, {}).get(p, 0) for p in labels] for t in labels]
            conf_display = pd.DataFrame(
                conf_data,
                index=[f"Gerçek: {n}" for n in label_names],
                columns=[f"Tahmin: {n}" for n in label_names],
            )

            # Renklendirme: köşegen yeşil, diğerleri kırmızı tonu
            def color_matrix(val):
                if val == 0:
                    return "color: #555"
                return ""

            st.dataframe(
                conf_display.style
                    .background_gradient(
                        cmap=__import__("matplotlib.colors", fromlist=["LinearSegmentedColormap"])
                            .LinearSegmentedColormap.from_list("custom", ["#fdb912", "#a90432"]),
                        axis=None
                    )
                    .format("{:d}"),
                use_container_width=True,
            )

            # ── İstatistiksel Tablo (Rapor için) ──
            st.markdown("<hr style='border-color:#1E2940; margin:20px 0;'>", unsafe_allow_html=True)
            st.markdown("""
            <p style="font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase;
                      color:#3A5070; margin:0 0 10px 0;">📋 İstatistiksel Özet Tablo (Rapor İçin)</p>
            """, unsafe_allow_html=True)

            stat_rows = []
            for label, name in LABEL_MAP.items():
                subset = res_df[res_df["Cinsiyet"] == label]
                if len(subset) > 0:
                    acc = (subset["Tahmin"] == subset["Cinsiyet"]).mean()
                    stat_rows.append({
                        "Sınıf"         : name,
                        "Örnek Sayısı"  : len(subset),
                        "Ort. F0 (Hz)"  : round(subset["mean_f0"].mean(), 1),
                        "Std. Sapma"    : round(subset["mean_f0"].std(), 1),
                        "Başarı (%)"    : f"%{acc*100:.0f}",
                    })

            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

            # ── F0 Dağılım Kutu Grafiği ──
            st.markdown("""
            <p style="font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase;
                      color:#3A5070; margin:16px 0 10px 0;">📦 Sınıfa Göre F0 Dağılımı</p>
            """, unsafe_allow_html=True)

            color_map_box = {"E": "#5B8EFF", "K": "#FF6B8A", "C": "#00E5A0"}
            fig4, ax = plt.subplots(figsize=(8, 4))
            fig4.patch.set_facecolor("#0A0F1E")
            ax.set_facecolor("#0D1829")
            ax.tick_params(colors="#6B7FA3", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1E2F4A")

            data_to_plot = []
            tick_labels  = []
            colors_box   = []
            for label, name in LABEL_MAP.items():
                subset = res_df[res_df["Cinsiyet"] == label]["mean_f0"]
                if len(subset) > 0:
                    data_to_plot.append(subset.values)
                    tick_labels.append(name)
                    colors_box.append(color_map_box[label])

            bps = ax.boxplot(data_to_plot, patch_artist=True,
                             medianprops=dict(color="white", linewidth=2))
            for patch, color in zip(bps["boxes"], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.axhline(erkek_max, color="#6B7FA3", linewidth=1,
                       linestyle=":", label=f"E/K sınırı={erkek_max} Hz")
            ax.axhline(kadin_max, color="#8A9AB8", linewidth=1,
                       linestyle="-.", label=f"K/Ç sınırı={kadin_max} Hz")

            ax.set_xticklabels(tick_labels, color="#B8C4D8")
            ax.set_ylabel("F0 (Hz)", color="#4A6080", fontsize=8)
            ax.set_title("Sınıf Bazlı F0 Dağılımı", color="#B8C4D8", fontsize=10)
            ax.legend(fontsize=7, facecolor="#0D1829", labelcolor="#B8C4D8", edgecolor="#1E2F4A")
            fig4.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

            # ── Tüm Sonuç Tablosu ──
            st.markdown("<hr style='border-color:#1E2940; margin:20px 0;'>", unsafe_allow_html=True)
            st.markdown("""
            <p style="font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase;
                      color:#3A5070; margin:0 0 10px 0;">📝 Tüm Sonuçlar</p>
            """, unsafe_allow_html=True)
            st.dataframe(res_df, use_container_width=True, hide_index=True)

            # Excel indirme
            import io
            excel_buffer = io.BytesIO()
            res_df.to_excel(excel_buffer, index=False, sheet_name="Sonuclar")
            excel_buffer.seek(0)
            st.download_button(
                "⬇️ Sonuçları Excel olarak indir",
                data      = excel_buffer,
                file_name = "Grup05_Sonuclar.xlsx",
                mime      = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    elif excel_upload is None and wav_uploads:
        st.warning("Lütfen metadata Excel dosyasını da yükleyin.")
    elif excel_upload and not wav_uploads:
        st.warning("Lütfen .wav ses dosyalarını da yükleyin.")
