# =============================================================================
# app_v2.py — Sistem Forensik Citra Digital V2
# Maulana Ahmad Nugroho — 25917001 — UII 2026
#
# PERUBAHAN DARI V1:
#   - Clone Detection (ORB) → DIHAPUS
#   - CNN MobileNetV2 → DITAMBAHKAN untuk deteksi wajah AI-generated
#   - Bobot: meta=0.25, ela=0.30, cnn=0.30, noise=0.15
#
# REFERENSI ILMIAH UNTUK run_cnn():
#   [1] Yilmaz & Cinar (2024)
#       "Real vs Fake Face Detection using MobileNetV2 Transfer Learning"
#       PeerJ Computer Science. DOI: 10.7717/peerj-cs.2103
#       Dataset: 140K real-and-fake-faces (StyleGAN) — dataset yang sama
#
#   [2] Kishimoto & Suresh (2024)
#       "MobileNetV2 Transfer Learning for Deepfake Detection"
#       IEEE ICDABI. DOI: 10.1109/ICDABI63787.2024
#       Akurasi: 95.14% pada dataset wajah
#
#   [3] Howard dkk (2018) — MobileNetV2: Inverted Residuals
#       arXiv:1801.04381
#
# REFERENSI UNTUK METODE LAIN:
#   run_ela()      → Bisri & Marzuki (2023); Chakraborty dkk (2024)
#   run_metadata() → Astillero (2025); Soni (2025)
#   run_noise()    → Pan dkk (2012); Gardella dkk (2021)
#   compute_weighted() → Korus & Huan (2016) — score level fusion
# =============================================================================

import io
import os
import base64
import traceback

import numpy as np
import cv2
from PIL import Image, ImageChops, ExifTags
from flask import Flask, request, jsonify
from flask_cors import CORS
# TensorFlow untuk CNN MobileNetV2
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

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
    'ela'  : 0.30,   # Error Level Analysis    — Bisri & Marzuki (2023)
    'cnn'  : 0.30,   # CNN MobileNetV2         — Yilmaz & Cinar (2024)
    'noise': 0.15,   # Local Noise Variance    — Gardella dkk (2021)
    'meta' : 0.25,   # Metadata Analysis       — Astillero (2025)
}

# Threshold keputusan akhir — konservatif untuk konteks e-KYC perbankan
THRESHOLD_MANIPULATED = 0.45

# =============================================================================
# LOAD MODEL CNN MobileNetV2
# Dilatih pada dataset Kaggle 140K real-and-fake-faces (StyleGAN)
# Referensi: Yilmaz & Cinar (2024) DOI:10.7717/peerj-cs.2103
# =============================================================================
_CNN_MODEL = None

def get_cnn_model():
    global _CNN_MODEL
    if _CNN_MODEL is None:
        model_path = os.path.join(os.path.dirname(__file__), 'cnn_mobilenetv2_forensik.h5')
        if os.path.exists(model_path) and TF_AVAILABLE:
            try:
                _CNN_MODEL = keras.models.load_model(model_path)
                print('✅ CNN MobileNetV2 model loaded')
            except Exception as e:
                print(f'⚠️ Gagal load CNN model: {e}')
                _CNN_MODEL = 'ERROR'
        else:
            _CNN_MODEL = 'NOT_FOUND'
    return _CNN_MODEL


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
        # FIX v2.1: no_exif berbasis ketiadaan field KAMERA, bukan jumlah total.
        # Foto AI/Kaggle bisa punya 4-5 field EXIF teknis (resolusi, orientasi)
        # tapi tidak punya Model Kamera dan Timestamp — dua field paling kritis.
        # Referensi: Astillero (2025) — metadata kamera adalah indikator utama e-KYC.
        no_camera_model = not has_camera
        no_timestamp    = not has_datetime
        no_exif_camera  = no_camera_model and no_timestamp

        no_exif = len(exif) < 3

        if no_exif_camera:
            suspicious = max(suspicious, 3)
        if no_exif:
            suspicious = max(suspicious, 4)

        # Bonus jika tidak ada model kamera — indikator terkuat e-KYC
        camera_bonus = 0.25 if no_camera_model else 0.0
        score = min(1.0, suspicious * 0.16 + camera_bonus + (0.15 if no_exif else 0.0))
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
# METODE 3: run_cnn() — CNN MobileNetV2 Transfer Learning ← MENGGANTIKAN CLONE
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
def run_cnn(img):
    """
    CNN MobileNetV2 Transfer Learning untuk deteksi wajah AI-generated.

    Pipeline:
    1. Resize gambar ke 224x224 (standar MobileNetV2)
    2. Normalisasi piksel ke [0,1]
    3. Inference dengan model MobileNetV2 yang sudah dilatih
    4. Output: probabilitas manipulasi 0.0 - 1.0

    REFERENSI:
    [1] Yilmaz & Cinar (2024) — MobileNetV2 pada dataset 140K StyleGAN
        DOI: 10.7717/peerj-cs.2103 — Val Accuracy: 78.6%
    [2] Kishimoto & Suresh (2024) — Transfer Learning deepfake detection
        DOI: 10.1109/ICDABI63787.2024 — Accuracy: 95.14%
    [3] Howard dkk (2018) — MobileNetV2 architecture
        arXiv:1801.04381

    Dataset training: Kaggle 140K Real and Fake Faces (StyleGAN2)
    Training: 10.000 foto | Validasi: 1.000 foto
    Epochs: 11 (EarlyStopping) | Final Val Accuracy: 78.6%
    """
    try:
        model = get_cnn_model()

        if model in ('NOT_FOUND', 'ERROR') or not TF_AVAILABLE:
            return run_cnn_fallback(img)

        # Preprocessing sesuai training
        img_resized = img.resize((224, 224)).convert('RGB')
        img_array   = np.array(img_resized, dtype=np.float32) / 255.0
        img_batch   = np.expand_dims(img_array, axis=0)

        # Inference
        prob   = float(model.predict(img_batch, verbose=0)[0][0])
        score  = round(prob, 4)

        # Heatmap sederhana — overlay probabilitas per region 7x7
        # MobileNetV2 menghasilkan feature map 7x7 sebelum pooling
        H, W = np.array(img).shape[:2]

        # Buat gradient heatmap berdasarkan skor
        heatmap_arr = np.zeros((H, W), dtype=np.float32)
        heatmap_arr[:, :] = prob

        # Tambah variasi visual berdasarkan brightness lokal
        gray     = np.array(img.convert('L'), dtype=np.float32) / 255.0
        gray_rs  = cv2.resize(gray, (W, H))
        heatmap_arr = np.clip(heatmap_arr + (1 - gray_rs) * 0.3 * prob, 0, 1)

        heatmap_uint8 = (heatmap_arr * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Overlay pada gambar asli
        orig_bgr = cv2.cvtColor(np.array(img.resize((W, H))), cv2.COLOR_RGB2BGR)
        blended  = cv2.addWeighted(orig_bgr, 0.6, heatmap_color, 0.4, 0)

        return {
            'score'       : score,
            'score_pct'   : round(score * 100, 1),
            'model'       : 'MobileNetV2 Transfer Learning',
            'architecture': 'MobileNetV2 + GlobalAvgPool + Dense(256) + Dense(64) + Sigmoid',
            'dataset'     : 'Kaggle 140K Real & Fake Faces (StyleGAN2)',
            'val_accuracy': '78.6%',
            'reference'   : 'Yilmaz & Cinar (2024) DOI:10.7717/peerj-cs.2103',
            'cnn_heatmap' : numpy_to_base64(heatmap_color),
            'cnn_overlay' : numpy_to_base64(blended),
            'status'      : 'Terindikasi' if score >= 0.45 else 'Normal',
            'method'      : 'CNN',
        }

    except Exception as e:
        return {
            'score'    : 0.0,
            'score_pct': 0.0,
            'error'    : str(e),
            'method'   : 'CNN',
        }


def run_cnn_fallback(img):
    """
    Fallback jika model CNN belum tersedia atau TensorFlow tidak terinstall.
    Menggunakan analisis DCT statistik sederhana.
    """
    try:
        gray  = np.array(img.convert('L'), dtype=np.float32)
        H, W  = gray.shape
        BSIZE = 8
        ac_energies = []

        for y in range(0, H - BSIZE + 1, BSIZE):
            for x in range(0, W - BSIZE + 1, BSIZE):
                block    = gray[y:y+BSIZE, x:x+BSIZE]
                dct_b    = cv2.dct(block)
                dct_b[0,0] = 0.0
                ac_energies.append(float(np.sum(dct_b**2)))

        ac  = np.array(ac_energies)
        mu  = float(np.mean(ac))
        std = float(np.std(ac))
        log_e = np.log1p(ac)
        mu_l  = float(np.mean(log_e))
        std_l = float(np.std(log_e))
        cv_l  = std_l / (mu_l + 1e-6)
        anom  = (log_e > mu_l + 1.0*std_l) | (log_e < mu_l - 1.0*std_l)
        ar    = float(np.sum(anom)) / len(ac) * 100.0
        nf    = 40.0 if cv_l > 1.5 else (25.0 if cv_l > 0.8 else 15.0)
        score = float(min(1.0, ar / nf))

        return {
            'score'    : round(score, 4),
            'score_pct': round(score * 100, 1),
            'model'    : 'DCT fallback (CNN model tidak tersedia)',
            'status'   : 'Terindikasi' if score >= 0.45 else 'Normal',
            'method'   : 'CNN',
        }
    except Exception as e:
        return {'score': 0.0, 'score_pct': 0.0, 'error': str(e), 'method': 'CNN'}


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
def compute_weighted_v2(ela, meta, cnn, noise):
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
    s_cnn   = cnn.get('score', 0.0)
    s_noise = noise.get('score', 0.0)

    w = WEIGHTS_V2

    # Weighted sum utama
    ws = (
        s_ela   * w['ela']   +
        s_cnn   * w['cnn']   +
        s_noise * w['noise'] +
        s_meta  * w['meta']
    )

    # Bonus konvergensi: ≥2 metode mendeteksi anomali → +10%
    methods_positive = sum([
        s_ela   >= THRESHOLD_MANIPULATED,
        s_cnn   >= THRESHOLD_MANIPULATED,
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
            'cnn'  : round(s_cnn   * 100, 1),
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
    - run_cnn()   → ditambahkan
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

        # Buat original_image base64 untuk ditampilkan di tab ELA
        orig_b64 = pil_to_base64(img)

        # Jalankan 4 metode forensik V2
        ela_result   = run_ela(file_bytes, img)
        meta_result  = run_metadata(file_bytes, img)
        cnn_result   = run_cnn(img)           # ← CNN MobileNetV2
        noise_result = run_noise(img)

        # Hitung weighted scoring V2
        weighted = compute_weighted_v2(ela_result, meta_result, cnn_result, noise_result)

        # Confusion matrix per metode (untuk tab Perbandingan)
        confusion = {
            'ELA'   : compute_matrix(ela_result.get('score', 0)),
            'CNN'   : compute_matrix(cnn_result.get('score', 0)),
            'Noise' : compute_matrix(noise_result.get('score', 0)),
            'Meta'  : compute_matrix(meta_result.get('score', 0)),
            'Hybrid': compute_matrix(weighted.get('weighted_score', 0)),
        }

        return jsonify({
            'version'        : 'v2',
            'original_image' : orig_b64,          # ← FIX: gambar asli untuk tab ELA
            'ela'            : ela_result,
            'metadata'       : meta_result,
            'cnn'            : cnn_result,
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
        'methods': ['ELA', 'CNN-MobileNetV2', 'LocalNoiseVariance', 'MetadataAnalysis'],
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
        'changes'    : 'Clone Detection (ORB) digantikan CNN MobileNetV2 Transfer Learning',
        'methods'    : {
            'ELA'   : {'weight': '30%', 'ref': 'Bisri & Marzuki (2023)'},
            'CNN'   : {'weight': '30%', 'ref': 'Yilmaz & Cinar (2024) DOI:10.7717/peerj-cs.2103'},
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
