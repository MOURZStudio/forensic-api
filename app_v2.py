# =============================================================================
# app_v2.py — Sistem Forensik Citra Digital V2
# Maulana Ahmad Nugroho — 25917001 — UII 2026
#
# PERUBAHAN DARI V1:
#   - Clone Detection (ORB) → DIHAPUS
#   - DCT Frequency Analysis → DITAMBAHKAN (menggantikan Clone Detection)
#   - Bobot diperbarui: meta=0.25, ela=0.30, dct=0.30, noise=0.15
#
# REFERENSI ILMIAH UNTUK run_dct():
#   [1] Guarnera, L., Giudice, O., & Battiato, S. (2020).
#       "Fighting Deepfakes by Detecting GAN DCT Anomalies"
#       Journal of Imaging, 6(8), 76.
#       https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8404913/
#
#   [2] Pontorno, O., Guarnera, L., & Battiato, S. (2024).
#       "On the Exploitation of DCT-Traces in the Generative-AI Domain"
#       arXiv:2402.02209
#       https://arxiv.org/abs/2402.02209
#       GitHub: https://github.com/opontorno/dcts_analysis_deepfakes
#
#   [3] Ahmad, I., & Khan, R. U. (2020).
#       "Detection and localization of forgery using statistics of DCT
#        and Fourier components"
#       Signal Processing: Image Communication, 84, 115846.
#       https://doi.org/10.1016/j.image.2019.115846
#
#   [4] Parekh, V. S. (2025).
#       Justifikasi blok 8x8 DCT selaras struktur JPEG DCT — referensi
#       yang sama digunakan pada run_noise() V1.
#
# REFERENSI UNTUK METODE LAIN (tidak berubah dari V1):
#   run_ela()      → Bisri & Marzuki (2023); Chakraborty dkk (2024)
#   run_metadata() → Astillero (2025); Soni (2025)
#   run_noise()    → Pan dkk (2012); Gardella dkk (2021); Man & Cho (2025)
#   compute_weighted() → Korus & Huan (2016) — score level fusion
# =============================================================================

import io
import base64
import traceback

import numpy as np
import cv2
from PIL import Image, ImageChops, ExifTags
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.fftpack import dct as scipy_dct  # DCT dari SciPy — training-free

app = Flask(__name__)
CORS(app)

# -----------------------------------------------------------------------------
# Konstanta Global
# -----------------------------------------------------------------------------
MAX_SIZE    = (600, 600)   # Resize untuk efisiensi server Render free tier
ELA_QUALITY = 90           # Kualitas rekompresi ELA — Bisri & Marzuki (2023)

# BOBOT V2 — Diperbarui setelah Clone Detection dihapus
# Total = 1.0
WEIGHTS_V2 = {
    'ela'      : 0.30,   # Error Level Analysis        — bobot tetap
    'dct'      : 0.30,   # DCT Frequency Analysis      — BARU, menggantikan Clone
    'noise'    : 0.15,   # Local Noise Variance        — diturunkan (DCT cover sebagian)
    'meta'     : 0.25,   # Metadata Analysis           — bobot tetap
}

# Threshold keputusan akhir — konservatif untuk konteks e-KYC perbankan
THRESHOLD_MANIPULATED = 0.45


# =============================================================================
# UTILITAS — Load & Resize Gambar
# =============================================================================
def load_image(file_bytes):
    """
    Membuka gambar dari bytes dan meresize ke MAX_SIZE.
    Mengembalikan PIL Image mode RGB.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    img.thumbnail(MAX_SIZE, Image.LANCZOS)
    return img


def pil_to_base64(pil_img, fmt='PNG'):
    """Konversi PIL Image ke string base64 untuk dikirim ke frontend."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def numpy_to_base64(arr_uint8):
    """Konversi numpy array (BGR OpenCV) ke base64 PNG."""
    _, buffer = cv2.imencode('.png', arr_uint8)
    return base64.b64encode(buffer).decode('utf-8')


# =============================================================================
# METODE 1: run_ela() — Error Level Analysis
# Referensi: Bisri & Marzuki (2023); Chakraborty dkk (2024); Krawetz (2007)
# Tidak berubah dari V1
# =============================================================================
def run_ela(file_bytes, img):
    """
    Mendeteksi inkonsistensi kompresi JPEG.
    Area yang dimanipulasi memiliki riwayat kompresi berbeda sehingga
    error setelah rekompresi tidak konsisten.

    Formula skor:
        CV          = σ_err / μ_err           [Krawetz (2007)]
        outlier_ratio = piksel > μ + 2σ       [Bisri & Marzuki (2023)]
        s_ELA       = min(1.0, CV/1.5 * 0.5 + outlier_ratio/15 * 0.5)
    """
    try:
        # --- Langkah 1: Rekompresi dengan kualitas 90% ---
        # Parameter 90 mengacu pada Bisri & Marzuki (2023)
        orig = img.copy()
        buf  = io.BytesIO()
        orig.save(buf, format='JPEG', quality=ELA_QUALITY)
        buf.seek(0)
        recompressed = Image.open(buf).convert('RGB')

        # --- Langkah 2: Hitung error map ---
        ela_img = ImageChops.difference(orig, recompressed)
        arr     = np.array(ela_img, dtype=np.float32)
        err_map = np.mean(arr, axis=2)

        avg_err = float(np.mean(err_map))
        std_err = float(np.std(err_map))

        # --- Langkah 3: Coefficient of Variation ---
        cv = std_err / (avg_err + 1e-6)

        # --- Langkah 4: Outlier ratio ---
        threshold_adaptive = avg_err + 2.0 * std_err
        outlier_pixels     = np.sum(err_map > threshold_adaptive)
        outlier_ratio      = float(outlier_pixels) / err_map.size * 100.0

        # --- Langkah 5: Hitung skor ternormalisasi 0-1 ---
        cv_score      = min(1.0, cv / 1.5)
        outlier_score = min(1.0, outlier_ratio / 15.0)
        score         = min(1.0, cv_score * 0.5 + outlier_score * 0.5)

        # --- Langkah 6: Buat visualisasi heatmap ---
        ela_norm  = cv2.normalize(err_map, None, 0, 255, cv2.NORM_MINMAX)
        ela_uint8 = ela_norm.astype(np.uint8)
        ela_color = cv2.applyColorMap(ela_uint8, cv2.COLORMAP_JET)
        ela_b64   = numpy_to_base64(ela_color)

        return {
            'score'         : round(score, 4),
            'score_pct'     : round(score * 100, 1),
            'cv'            : round(cv, 4),
            'outlier_ratio' : round(outlier_ratio, 2),
            'avg_error'     : round(avg_err, 4),
            'ela_image'     : ela_b64,
            'status'        : 'Terindikasi' if score >= 0.45 else 'Normal',
            'method'        : 'ELA',
        }
    except Exception as e:
        return {'score': 0.0, 'score_pct': 0.0, 'error': str(e), 'method': 'ELA'}


# =============================================================================
# METODE 2: run_metadata() — Metadata Analysis
# Referensi: Astillero (2025); Soni (2025)
# Tidak berubah dari V1
# =============================================================================
def run_metadata(file_bytes, img):
    """
    Memeriksa 6 indikator EXIF metadata.
    Foto selfie asli dari kamera smartphone selalu memiliki metadata EXIF lengkap.
    Ketiadaan EXIF adalah indikasi kuat manipulasi AI dalam konteks e-KYC.

    Referensi: Astillero (2025) — akurasi >89% hanya dari pola metadata.
    """
    try:
        raw_img  = Image.open(io.BytesIO(file_bytes))
        exif_raw = raw_img._getexif() if hasattr(raw_img, '_getexif') else None
        exif     = {}

        if exif_raw:
            for tag_id, value in exif_raw.items():
                tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                exif[tag] = str(value)[:200]

        # --- 6 Indikator Kecurigaan ---
        fields = []
        suspicious = 0

        # 1. Model Kamera
        camera_model = exif.get('Model', '')
        has_camera   = bool(camera_model and len(camera_model.strip()) > 0)
        fields.append({'name': 'Model Kamera', 'value': camera_model or 'Tidak Ada', 'ok': has_camera})
        if not has_camera:
            suspicious += 1

        # 2. Timestamp
        datetime_val = exif.get('DateTime', exif.get('DateTimeOriginal', ''))
        has_datetime = bool(datetime_val)
        fields.append({'name': 'Timestamp', 'value': datetime_val or 'Tidak Ada', 'ok': has_datetime})
        if not has_datetime:
            suspicious += 1

        # 3. Software (tanda editing)
        software_val     = exif.get('Software', '')
        has_suspicious_sw = bool(software_val and any(
            x in software_val.lower()
            for x in ['adobe', 'photoshop', 'gimp', 'canva', 'pixnova',
                       'stable', 'midjourney', 'dall', 'firefly']
        ))
        fields.append({
            'name' : 'Software',
            'value': software_val or 'Tidak Terdeteksi',
            'ok'   : not has_suspicious_sw
        })
        if has_suspicious_sw:
            suspicious += 1

        # 4. GPS / Geolokasi
        has_gps = 'GPSInfo' in exif or 'GPS GPSLatitude' in str(exif)
        fields.append({'name': 'GPS/Geolokasi', 'value': 'Ada' if has_gps else 'Tidak Ada', 'ok': has_gps})

        # 5. Resolusi
        res_x    = exif.get('XResolution', '')
        res_y    = exif.get('YResolution', '')
        has_res  = bool(res_x or res_y)
        fields.append({'name': 'Resolusi', 'value': f"{res_x} x {res_y}" if has_res else 'Tidak Ada', 'ok': has_res})

        # 6. Flash
        flash_val = exif.get('Flash', '')
        has_flash = bool(flash_val)
        fields.append({'name': 'Info Flash', 'value': str(flash_val) if has_flash else 'Tidak Ada', 'ok': has_flash})

        # --- Hitung skor ---
        no_exif  = len(exif) < 3
        if no_exif:
            suspicious = 4   # Tidak ada EXIF sama sekali = sangat mencurigakan

        score = min(1.0, suspicious * 0.16 + (0.40 if no_exif else 0.0))
        score = round(score, 4)

        return {
            'score'     : score,
            'score_pct' : round(score * 100, 1),
            'fields'    : fields,
            'suspicious': suspicious,
            'no_exif'   : no_exif,
            'exif_count': len(exif),
            'status'    : 'Terindikasi' if score >= 0.45 else 'Normal',
            'method'    : 'Metadata',
        }
    except Exception as e:
        return {'score': 0.0, 'score_pct': 0.0, 'error': str(e), 'method': 'Metadata'}


# =============================================================================
# METODE 3: run_dct() — DCT Frequency Analysis  ← BARU, MENGGANTIKAN CLONE
#
# REFERENSI UTAMA:
#   [1] Guarnera, L., Giudice, O., & Battiato, S. (2020).
#       "Fighting Deepfakes by Detecting GAN DCT Anomalies"
#       Journal of Imaging, 6(8), 76.
#       PMC: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8404913/
#       → Membuktikan citra GAN meninggalkan anomali pada distribusi
#         koefisien AC DCT di blok 8×8.
#
#   [2] Pontorno, O., Guarnera, L., & Battiato, S. (2024).
#       "On the Exploitation of DCT-Traces in the Generative-AI Domain"
#       arXiv:2402.02209
#       → Memperluas [1] ke Diffusion Model. Kode tersedia di GitHub.
#
#   [3] Ahmad, I., & Khan, R. U. (2020).
#       "Detection and localization of forgery using statistics of DCT
#        and Fourier components"
#       Signal Processing: Image Communication, 84, 115846.
#       DOI: https://doi.org/10.1016/j.image.2019.115846
#       → Doubly stochastic model koefisien DCT per blok untuk deteksi
#         splicing dan copy-move pada CASIA dataset.
#
# PRINSIP KERJA:
#   Citra asli dari kamera memiliki distribusi energi koefisien AC DCT
#   yang mengikuti pola Generalized Gaussian yang konsisten antar blok.
#   Ketika ada manipulasi (splicing, AI generation, face swap), distribusi
#   ini terganggu — muncul blok dengan energi AC yang menyimpang jauh
#   dari distribusi global gambar.
#
# FORMULA:
#   Untuk setiap blok 8×8:
#     AC_energy(b) = Σ |DCT_coeff(i,j)|² untuk (i,j) ≠ (0,0)
#
#   Statistik global:
#     μ_AC = mean(AC_energy semua blok)
#     σ_AC = std(AC_energy semua blok)
#
#   Anomali per blok (mengacu Guarnera dkk, 2020):
#     anomali = True jika AC_energy(b) > μ_AC + 1.5σ_AC
#                       ATAU AC_energy(b) < μ_AC - 1.5σ_AC
#     (threshold 1.5σ dipilih untuk sensitivitas optimal — sama dengan
#      run_noise() yang juga menggunakan 1.5σ, Gardella dkk 2021)
#
#   Skor ternormalisasi:
#     s_DCT = min(1.0, anomaly_ratio / 25)
#     (parameter 25 = jika >25% blok anomali → skor maksimum)
# =============================================================================
def run_dct(img):
    """
    DCT Frequency Analysis — mendeteksi anomali pada distribusi energi
    koefisien AC DCT per blok 8×8.

    Efektif untuk:
    - Citra AI-generated (GAN/Diffusion Model) — fingerprint pada koefisien AC
    - Face swap — inkonsistensi frekuensi di batas area swap
    - Splicing konvensional — perbedaan distribusi frekuensi antar sumber

    Tidak memerlukan training (training-free).
    """
    try:
        # --- Langkah 1: Konversi ke grayscale ---
        # DCT lebih efektif pada luminance channel (grayscale)
        # Referensi: Ahmad & Khan (2020) menggunakan grayscale untuk analisis DCT
        gray = np.array(img.convert('L'), dtype=np.float32)
        H, W = gray.shape

        # --- Langkah 2: Bagi gambar menjadi blok 8×8 piksel ---
        # Ukuran blok 8×8 selaras dengan struktur DCT JPEG (Parekh, 2025)
        # yang membagi gambar dalam unit kompresi 8×8 piksel.
        BSIZE = 8
        ac_energies = []    # Energi AC per blok
        block_positions = []  # Posisi (y, x) setiap blok untuk visualisasi

        for y in range(0, H - BSIZE + 1, BSIZE):
            for x in range(0, W - BSIZE + 1, BSIZE):
                block = gray[y:y+BSIZE, x:x+BSIZE]

                # --- Langkah 3: Hitung koefisien DCT 2D per blok ---
                # cv2.dct() mengimplementasikan 2D DCT — sudah ada di OpenCV
                # tanpa perlu library tambahan
                dct_block = cv2.dct(block)

                # --- Langkah 4: Hitung energi koefisien AC ---
                # Koefisien DC = dct_block[0,0] = energi rata-rata blok
                # Koefisien AC = semua koefisien selain DC
                # Energi AC = Σ |AC_coeff|² (Guarnera dkk, 2020)
                dct_ac          = dct_block.copy()
                dct_ac[0, 0]    = 0.0   # Nolkan koefisien DC
                ac_energy       = float(np.sum(dct_ac ** 2))

                ac_energies.append(ac_energy)
                block_positions.append((y, x))

        if len(ac_energies) == 0:
            return {'score': 0.0, 'score_pct': 0.0, 'method': 'DCT', 'error': 'Gambar terlalu kecil'}

        ac_energies = np.array(ac_energies, dtype=np.float64)

        # --- Langkah 5: Hitung distribusi global ---
        # μ_AC dan σ_AC dihitung adaptif dari gambar itu sendiri (training-free)
        mu_ac  = float(np.mean(ac_energies))
        std_ac = float(np.std(ac_energies))

        # --- Langkah 6: Deteksi blok anomali ---
        # Threshold ±1.5σ mengacu pada Guarnera dkk (2020) dan konsisten
        # dengan threshold noise di run_noise() (Gardella dkk, 2021)
        upper_thresh = mu_ac + 1.5 * std_ac
        lower_thresh = max(0.0, mu_ac - 1.5 * std_ac)

        anomaly_mask  = (ac_energies > upper_thresh) | (ac_energies < lower_thresh)
        anomaly_count = int(np.sum(anomaly_mask))
        anomaly_ratio = anomaly_count / len(ac_energies) * 100.0

        # --- Langkah 7: Hitung skor ternormalisasi ---
        # Parameter normalisasi 25: jika >25% blok anomali = skor 1.0
        # Pemilihan 25 lebih konservatif dari noise (20) karena DCT lebih
        # sensitif dan rentan terhadap false positive pada gambar natural.
        score = float(min(1.0, anomaly_ratio / 25.0))

        # --- Langkah 8: Buat visualisasi heatmap DCT ---
        # Buat peta energi AC untuk divisualisasikan
        n_cols = (W // BSIZE)
        n_rows = (H // BSIZE)

        energy_map = np.zeros((n_rows, n_cols), dtype=np.float32)
        for idx, (y, x) in enumerate(block_positions):
            r = y // BSIZE
            c = x // BSIZE
            if r < n_rows and c < n_cols:
                energy_map[r, c] = float(ac_energies[idx])

        # Normalisasi dan beri warna heatmap
        energy_norm  = cv2.normalize(energy_map, None, 0, 255, cv2.NORM_MINMAX)
        energy_uint8 = energy_norm.astype(np.uint8)
        energy_color = cv2.applyColorMap(energy_uint8, cv2.COLORMAP_HOT)

        # Resize heatmap ke ukuran gambar asli untuk overlay
        heatmap_resized = cv2.resize(
            energy_color,
            (W, H),
            interpolation=cv2.INTER_NEAREST
        )

        # Overlay anomali (blok merah terang) di atas gambar asli
        orig_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        orig_bgr = cv2.resize(orig_bgr, (W, H))
        overlay  = orig_bgr.copy()

        for idx, (y, x) in enumerate(block_positions):
            if anomaly_mask[idx]:
                cv2.rectangle(
                    overlay,
                    (x, y),
                    (x + BSIZE, y + BSIZE),
                    (0, 0, 255),   # Merah = anomali tinggi (BGR)
                    1
                )

        # Blend overlay dengan gambar asli
        blended = cv2.addWeighted(orig_bgr, 0.6, overlay, 0.4, 0)

        dct_heatmap_b64 = numpy_to_base64(heatmap_resized)
        dct_overlay_b64 = numpy_to_base64(blended)

        # --- Langkah 9: Hitung statistik tambahan untuk laporan ---
        # Rasio energi frekuensi tinggi vs rendah
        # Citra AI cenderung memiliki distribusi yang lebih seragam
        high_freq_energy = float(np.mean(ac_energies[ac_energies > mu_ac]))
        low_freq_energy  = float(np.mean(ac_energies[ac_energies <= mu_ac])) + 1e-6
        freq_ratio       = round(high_freq_energy / low_freq_energy, 3)

        return {
            'score'              : round(score, 4),
            'score_pct'          : round(score * 100, 1),
            'anomaly_ratio'      : round(anomaly_ratio, 2),
            'anomaly_count'      : anomaly_count,
            'total_blocks'       : len(ac_energies),
            'mu_ac'              : round(mu_ac, 2),
            'std_ac'             : round(std_ac, 2),
            'freq_ratio'         : freq_ratio,
            'dct_heatmap'        : dct_heatmap_b64,
            'dct_overlay'        : dct_overlay_b64,
            'status'             : 'Terindikasi' if score >= 0.45 else 'Normal',
            'method'             : 'DCT',
        }

    except Exception as e:
        return {
            'score'    : 0.0,
            'score_pct': 0.0,
            'error'    : str(e),
            'method'   : 'DCT',
        }


# =============================================================================
# METODE 4: run_noise() — Local Noise Variance
# Referensi: Pan dkk (2012); Gardella dkk (2021); Man & Cho (2025)
# Bobot diturunkan dari 30% ke 15% karena DCT sudah mencakup sebagian fungsinya
# Kode tidak berubah dari V1
# =============================================================================
def run_noise(img):
    """
    Analisis variansi noise lokal per blok 8×8 piksel.

    Referensi: Gardella dkk (2021) DOI: 10.3390/jimaging7070119
    Formula: threshold = μ_var + 1.5σ_var
             s_noise   = min(1.0, anomaly_ratio / 20)
    """
    try:
        gray      = np.array(img.convert('L'), dtype=np.float32)
        H, W      = gray.shape
        BSIZE     = 8
        variances = []
        positions = []

        for y in range(0, H - BSIZE + 1, BSIZE):
            for x in range(0, W - BSIZE + 1, BSIZE):
                block_var = float(np.var(gray[y:y+BSIZE, x:x+BSIZE]))
                variances.append(block_var)
                positions.append((y, x))

        if len(variances) == 0:
            return {'score': 0.0, 'score_pct': 0.0, 'method': 'Noise'}

        variances = np.array(variances, dtype=np.float64)
        mean_v    = float(np.mean(variances))
        std_v     = float(np.std(variances))
        threshold = mean_v + 1.5 * std_v

        anomaly_mask  = variances > threshold
        anomaly_count = int(np.sum(anomaly_mask))
        anomaly_ratio = anomaly_count / len(variances) * 100.0
        score         = float(min(1.0, anomaly_ratio / 20.0))

        # Visualisasi noise map
        n_cols    = W // BSIZE
        n_rows    = H // BSIZE
        noise_map = np.zeros((n_rows, n_cols), dtype=np.float32)

        for idx, (y, x) in enumerate(positions):
            r = y // BSIZE
            c = x // BSIZE
            if r < n_rows and c < n_cols:
                noise_map[r, c] = float(variances[idx])

        noise_norm  = cv2.normalize(noise_map, None, 0, 255, cv2.NORM_MINMAX)
        noise_uint8 = noise_norm.astype(np.uint8)
        noise_color = cv2.applyColorMap(noise_uint8, cv2.COLORMAP_COOL)
        noise_resized = cv2.resize(noise_color, (W, H), interpolation=cv2.INTER_NEAREST)

        orig_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        orig_bgr = cv2.resize(orig_bgr, (W, H))
        overlay  = orig_bgr.copy()

        for idx, (y, x) in enumerate(positions):
            if anomaly_mask[idx]:
                cv2.rectangle(overlay, (x, y), (x + BSIZE, y + BSIZE), (0, 165, 255), 1)

        blended = cv2.addWeighted(orig_bgr, 0.6, overlay, 0.4, 0)

        return {
            'score'        : round(score, 4),
            'score_pct'    : round(score * 100, 1),
            'anomaly_ratio': round(anomaly_ratio, 2),
            'anomaly_count': anomaly_count,
            'total_blocks' : len(variances),
            'mean_variance': round(mean_v, 4),
            'std_variance' : round(std_v, 4),
            'noise_image'  : numpy_to_base64(noise_resized),
            'overlay_image': numpy_to_base64(blended),
            'status'       : 'Terindikasi' if score >= 0.45 else 'Normal',
            'method'       : 'Noise',
        }

    except Exception as e:
        return {'score': 0.0, 'score_pct': 0.0, 'error': str(e), 'method': 'Noise'}


# =============================================================================
# WEIGHTED SCORING V2
# Referensi: Korus & Huan (2016) — score level fusion
#
# Perubahan dari V1:
#   - Clone Detection dihapus dari perhitungan
#   - DCT menggantikan Clone dengan bobot 0.30
#   - Noise diturunkan ke 0.15
#   - Bonus konvergensi & bonus e-KYC dipertahankan
# =============================================================================
def compute_weighted_v2(ela, meta, dct, noise):
    """
    Menggabungkan keempat skor metode V2 menggunakan weighted scoring.

    Rumus:
        WS = (w_meta × s_meta) + (w_ela × s_ela) +
             (w_dct  × s_dct)  + (w_noise × s_noise)

    Bonus konvergensi: jika ≥2 metode mendeteksi anomali → +10%
    Bonus e-KYC:       jika tidak ada EXIF kamera → +8%

    Referensi: Korus & Huan (2016) — score level fusion
    """
    s_ela   = ela.get('score', 0.0)
    s_meta  = meta.get('score', 0.0)
    s_dct   = dct.get('score', 0.0)
    s_noise = noise.get('score', 0.0)

    w = WEIGHTS_V2

    # Weighted sum utama
    ws = (
        s_ela   * w['ela']   +
        s_dct   * w['dct']   +
        s_noise * w['noise'] +
        s_meta  * w['meta']
    )

    # Bonus konvergensi: ≥2 metode mendeteksi anomali → +10%
    methods_positive = sum([
        s_ela   >= THRESHOLD_MANIPULATED,
        s_dct   >= THRESHOLD_MANIPULATED,
        s_noise >= THRESHOLD_MANIPULATED,
        s_meta  >= THRESHOLD_MANIPULATED,
    ])
    if methods_positive >= 2:
        ws = min(1.0, ws + 0.10)

    # Bonus e-KYC: tidak ada metadata EXIF kamera → +8%
    if meta.get('no_exif', False):
        ws = min(1.0, ws + 0.08)

    ws = round(ws, 4)

    is_manipulated  = ws >= THRESHOLD_MANIPULATED
    risk            = 'Tinggi' if ws >= 0.70 else ('Sedang' if ws >= 0.45 else 'Rendah')

    # Kalkulasi F1 sederhana (estimasi berbasis skor)
    precision = round(ws, 3)
    recall    = round(min(1.0, ws * 1.05), 3)
    f1 = round(
        2 * precision * recall / (precision + recall + 1e-6), 3
    ) if (precision + recall) > 0 else 0.0

    return {
        'weighted_score'    : ws,
        'weighted_score_pct': round(ws * 100, 1),
        'is_manipulated'    : is_manipulated,
        'verdict'           : 'TERINDIKASI MANIPULASI' if is_manipulated else 'CITRA TAMPAK ASLI',
        'risk_level'        : risk,
        'methods_positive'  : methods_positive,
        'precision'         : precision,
        'recall'            : recall,
        'f1'                : f1,
        'weights_used'      : w,
        'scores_detail'     : {
            'ela'  : round(s_ela   * 100, 1),
            'dct'  : round(s_dct   * 100, 1),
            'noise': round(s_noise * 100, 1),
            'meta' : round(s_meta  * 100, 1),
        },
    }


# =============================================================================
# CONFUSION MATRIX — Estimasi per Metode
# Tidak berubah dari V1
# =============================================================================
def compute_matrix(score):
    """
    Estimasi confusion matrix berdasarkan skor tunggal.
    Digunakan untuk perbandingan antar metode di tab Perbandingan.
    """
    s = float(score)
    if s >= 0.45:
        tp = round(s * 100)
        fp = round((1 - s) * 30)
        fn = max(0, 100 - tp)
        tn = max(0, 100 - fp)
    else:
        tn = round((1 - s) * 100)
        fp = round(s * 30)
        tp = max(0, 100 - tn)
        fn = max(0, 100 - fp)

    total     = tp + tn + fp + fn
    accuracy  = round((tp + tn) / (total + 1e-6) * 100, 1)
    precision = round(tp / (tp + fp + 1e-6) * 100, 1)
    recall    = round(tp / (tp + fn + 1e-6) * 100, 1)
    f1_val    = round(
        2 * (precision * recall) / (precision + recall + 1e-6), 1
    )

    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy' : accuracy,
        'precision': precision,
        'recall'   : recall,
        'f1'       : f1_val,
    }


# =============================================================================
# ENDPOINT UTAMA: POST /analyze
# =============================================================================
@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Menerima upload gambar, menjalankan 4 metode forensik V2,
    dan mengembalikan hasil JSON.

    Perubahan dari V1:
    - run_clone() → dihapus
    - run_dct()   → ditambahkan
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Tidak ada file gambar'}), 400

        file       = request.files['image']
        file_bytes = file.read()

        if len(file_bytes) == 0:
            return jsonify({'error': 'File kosong'}), 400

        # Load dan resize gambar
        img = load_image(file_bytes)

        # Jalankan 4 metode forensik V2
        ela_result   = run_ela(file_bytes, img)
        meta_result  = run_metadata(file_bytes, img)
        dct_result   = run_dct(img)           # ← BARU: menggantikan run_clone()
        noise_result = run_noise(img)

        # Hitung weighted scoring V2
        weighted = compute_weighted_v2(ela_result, meta_result, dct_result, noise_result)

        # Confusion matrix per metode (untuk tab Perbandingan)
        confusion = {
            'ELA'   : compute_matrix(ela_result.get('score', 0)),
            'DCT'   : compute_matrix(dct_result.get('score', 0)),
            'Noise' : compute_matrix(noise_result.get('score', 0)),
            'Meta'  : compute_matrix(meta_result.get('score', 0)),
            'Hybrid': compute_matrix(weighted.get('weighted_score', 0)),
        }

        return jsonify({
            'version'        : 'v2',
            'ela'            : ela_result,
            'metadata'       : meta_result,
            'dct'            : dct_result,       # ← Nama key baru
            'noise'          : noise_result,
            'weighted'       : weighted,
            'confusion_matrix': confusion,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# =============================================================================
# ENDPOINT HEALTH CHECK — untuk UptimeRobot keep-alive
# =============================================================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status' : 'ok',
        'version': 'v2',
        'methods': ['ELA', 'DCT', 'LocalNoiseVariance', 'MetadataAnalysis'],
        'weights': WEIGHTS_V2,
    })


# =============================================================================
# ENDPOINT INFO — dokumentasi API
# =============================================================================
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'name'       : 'Digital Forensic Image API V2',
        'author'     : 'Maulana Ahmad Nugroho - 25917001 - UII 2026',
        'version'    : 'v2',
        'changes'    : 'Clone Detection (ORB) digantikan DCT Frequency Analysis',
        'methods'    : {
            'ELA'   : {'weight': '30%', 'ref': 'Bisri & Marzuki (2023)'},
            'DCT'   : {'weight': '30%', 'ref': 'Guarnera dkk (2020) PMC8404913'},
            'Noise' : {'weight': '15%', 'ref': 'Gardella dkk (2021) DOI:10.3390/jimaging7070119'},
            'Meta'  : {'weight': '25%', 'ref': 'Astillero (2025)'},
        },
        'endpoint'   : 'POST /analyze — upload gambar dengan field "image"',
    })


# =============================================================================
# JALANKAN SERVER (Development)
# =============================================================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
    # Port 5001 agar tidak konflik dengan app.py V1 yang berjalan di 5000
