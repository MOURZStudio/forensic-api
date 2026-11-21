# app_publikasi.py — Deteksi Manipulasi Citra Wajah pada Layanan Digital Onboarding Perbankan
# Maulana Ahmad Nugroho — 25917001 — UII 2026
#
# Versi PUBLIKASI (terpisah dari versi tesis app_v2.py).
# Hanya 2 metode, sesuai naskah publikasi:
#   1. Analisis metadata EXIF  -> S_m
#   2. CNN MobileNetV2         -> S_c
# Weighted scoring:  S = W_m x S_m + W_c x S_c,  W_m = 0,40  W_c = 0,60
# Keputusan:         S > T  -> manipulasi,  selain itu -> asli,  T = 0,50
#
# Model CNN dijalankan dalam format TFLite (hasil konversi langsung dari
# cnn_mobilenetv2_forensik.h5, bobot float32 tidak diubah — lihat convert_tflite.py)
# agar muat di RAM 512 MB Render free tier. Kode ini juga dipakai oleh
# evaluate_publikasi.py untuk uji batch, sehingga hasil web dan hasil batch identik.

import io
import os
import threading
import traceback

import base64
import re

import cv2
import numpy as np
from PIL import Image, ImageOps, ExifTags
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    from ai_edge_litert.interpreter import Interpreter
    LITERT_AVAILABLE = True
except ImportError:  # pragma: no cover
    LITERT_AVAILABLE = False

VERSION = 'publikasi-1.4'

# -----------------------------------------------------------------------------
# Parameter (bagian 2.2, 2.3 dan 2.7 naskah)
# -----------------------------------------------------------------------------
MAX_SIZE = (600, 600)      # praproses: maksimum 600 x 600 piksel
CNN_SIZE = (224, 224)      # masukan MobileNetV2
W_M = 0.40                 # bobot metadata
W_C = 0.60                 # bobot CNN
T = 0.50                   # ambang keputusan

# Indikator 2 (tag software):
#   - kode firmware / versi perangkat (mis. 'S901EXXU2AVF1', '26.3.1')  -> normal
#   - penanda aplikasi kamera bawaan (mis. 'HDR+ 1.0', 'MediaTek Camera') -> normal
#   - nama software / aplikasi (mis. 'Adobe Photoshop', 'Polarr Photo Editor') -> mencurigakan
EDITING_SOFTWARE = ['adobe', 'photoshop', 'lightroom', 'gimp', 'canva', 'picsart',
                    'snapseed', 'facetune', 'faceapp', 'meitu', 'pixnova', 'remini',
                    'polarr', 'vsco', 'pixlr', 'fotor', 'beautyplus', 'youcam',
                    'stable', 'midjourney', 'dall', 'firefly', 'editor', 'edit']
CAMERA_FIRMWARE_HINTS = ['camera', 'hdr+', 'android', 'ios', 'miui', 'hyperos',
                         'oneui', 'one ui', 'harmonyos', 'coloros', 'funtouch']


# Indikator 1: pola nama model per merek. Model yang tidak sesuai pola merek
# (mis. 'iPhone I', merek Apple dengan kode Samsung) dianggap TIDAK VALID dan
# diperlakukan sama dengan model kamera tidak ditemukan.
MODEL_PATTERNS = {
    'apple':   r'^(iphone|ipad)\s*(\d{1,2}|se|x|xr|xs)\b',
    'samsung': r'^(sm-|gt-|sch-|sgh-|shv-|galaxy\s)[a-z0-9]',
    'google':  r'^pixel\s*(\d|fold|tablet)',
    'xiaomi':  r'^((redmi|poco|mi|xiaomi)\b|m?\d{4,}[a-z0-9]*$|[a-z]?\d{3,}[a-z0-9]+$)',
    'redmi':   r'^((redmi|poco)\b|m?\d{4,}[a-z0-9]*$)',
    'poco':    r'^((poco)\b|m?\d{4,}[a-z0-9]*$)',
    'oppo':    r'^(cph\d{4}|pa[a-z]m\d{2}|oppo\s|reno|find)',
    'realme':  r'^(rmx\d{4}|realme\s)',
    'vivo':    r'^(v\d{4}|vivo\s|i\d{4}|x\d{2,})',
    'huawei':  r'^([a-z]{3}-[a-z]{1,2}\d{2}|huawei\s|p\d{2}|mate|nova)',
    'honor':   r'^([a-z]{3}-[a-z]{1,2}\d{2}|honor\s)',
    'oneplus': r'^(oneplus|[a-z]{2}\d{4}|cph\d{4})',
    'infinix': r'^(infinix\s|x\d{3,})',
    'tecno':   r'^(tecno\s|[a-z]{2}\d{1,2}[a-z]?$)',
}


def validate_camera(make, model):
    """Kembalikan (valid: bool, keterangan: str) untuk pasangan merek-model."""
    if not model:
        return False, 'Tidak ditemukan'
    mk = (make or '').strip().lower()
    md = model.strip().lower()
    for brand, pat in MODEL_PATTERNS.items():
        if brand in mk:
            md2 = md[len(brand):].strip() if md.startswith(brand) else md
            if re.search(pat, md2) or re.search(pat, md):
                return True, 'Sesuai pola perangkat'
            return False, 'Tidak sesuai pola model ' + brand.capitalize()
    # merek lain / tidak dikenal: model wajar bila memuat huruf dan angka
    if re.search(r'[a-z]', md) and re.search(r'\d', md) and len(md) >= 3:
        return True, 'Merek di luar daftar pola'
    return False, 'Format model tidak wajar'


def classify_software(value):
    """Kembalikan (mencurigakan: bool, kategori: str) untuk nilai tag Software."""
    v = (value or '').strip()
    if not v:
        return False, 'Tidak ada tag software'
    low = v.lower()
    if any(k in low for k in EDITING_SOFTWARE):
        return True, 'Nama software editing'
    if any(k in low for k in CAMERA_FIRMWARE_HINTS):
        return False, 'Aplikasi kamera bawaan'
    tokens = v.replace('(', ' ').replace(')', ' ').split()
    if tokens and all(any(ch.isdigit() for ch in t) for t in tokens):
        return False, 'Kode firmware/versi perangkat'
    return True, 'Nama software/aplikasi'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'cnn_mobilenetv2_forensik.tflite')
FACE_MODEL_PATH = os.path.join(BASE_DIR, 'face_detection_yunet_2023mar.onnx')

# Deteksi & crop wajah untuk jalur CNN — menyamakan masukan dengan data latih
# (FFHQ/StyleGAN: wajah terpotong rapat di tengah bingkai).
FACE_DETECT_MAX = 1280     # sisi terpanjang citra saat deteksi wajah
FACE_SCORE_MIN = 0.60      # ambang keyakinan detektor YuNet
FACE_MARGIN = 2.0          # sisi crop persegi = 2,0 x sisi terpanjang kotak wajah

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
CORS(app)

# -----------------------------------------------------------------------------
# Model CNN (dimuat sekali)
# -----------------------------------------------------------------------------
_interp = None
_interp_lock = threading.Lock()
_model_error = None


def get_interpreter():
    global _interp, _model_error
    if _interp is None and _model_error is None:
        try:
            if not LITERT_AVAILABLE:
                raise RuntimeError('ai-edge-litert tidak terpasang')
            it = Interpreter(model_path=MODEL_PATH, num_threads=1)
            it.allocate_tensors()
            _interp = it
        except Exception as e:  # pragma: no cover
            _model_error = str(e)
    return _interp


# -----------------------------------------------------------------------------
# Praproses (2.2)
# -----------------------------------------------------------------------------
def load_oriented(file_bytes):
    """Buka citra, koreksi orientasi sesuai tag EXIF Orientation, konversi ke RGB."""
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)
    return img.convert('RGB')


def load_image(file_bytes):
    """Konversi ke RGB lalu perkecil hingga maks 600 x 600 (rasio dipertahankan)."""
    img = load_oriented(file_bytes)
    img.thumbnail(MAX_SIZE, Image.LANCZOS)
    return img


_face_det = None


def detect_face_crop(img):
    """
    Deteksi wajah (YuNet, OpenCV) dan kembalikan crop persegi di sekitar wajah
    dengan sisi = FACE_MARGIN x sisi terpanjang kotak wajah.
    Jika tidak ada wajah terdeteksi, kembalikan (None, info).
    """
    global _face_det
    W, H = img.size
    scale = min(1.0, FACE_DETECT_MAX / max(W, H))
    small = img.resize((max(1, round(W * scale)), max(1, round(H * scale))), Image.BILINEAR) \
        if scale < 1.0 else img
    bgr = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR)
    if _face_det is None:
        _face_det = cv2.FaceDetectorYN.create(FACE_MODEL_PATH, '', (320, 320), FACE_SCORE_MIN)
    _face_det.setInputSize((bgr.shape[1], bgr.shape[0]))
    _, faces = _face_det.detect(bgr)
    if faces is None or len(faces) == 0:
        return None, {'detected': False, 'count': 0}

    # pilih wajah dengan luas kotak terbesar
    best = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = [float(v) / scale for v in best[:4]]
    conf = float(best[14])
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * FACE_MARGIN
    side = min(side, W, H)
    left = int(round(min(max(cx - side / 2, 0), W - side)))
    top = int(round(min(max(cy - side / 2, 0), H - side)))
    side = int(round(side))
    crop = img.crop((left, top, left + side, top + side))
    return crop, {
        'detected': True,
        'count': int(len(faces)),
        'confidence': round(conf, 4),
        'box': [int(round(x)), int(round(y)), int(round(w)), int(round(h))],
        'crop': [left, top, side, side],
        'face_ratio_pct': round(100.0 * w * h / (W * H), 2),
    }


# -----------------------------------------------------------------------------
# Jalur 1 — Analisis metadata EXIF (2.3)
# -----------------------------------------------------------------------------
def read_exif(file_bytes):
    """Metadata dibaca dari berkas asli sebelum citra diubah."""
    exif = {}
    try:
        raw = Image.open(io.BytesIO(file_bytes))
        exif_raw = raw._getexif() if hasattr(raw, '_getexif') else None
        if exif_raw:
            for tag_id, value in exif_raw.items():
                tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                exif[tag] = str(value)[:200]
    except Exception:
        pass
    return exif


def run_metadata(file_bytes):
    """
    S_m = min(1; 0,16 x c + 0,25 x a + 0,15 x b)

    c : jumlah indikator mencurigakan (model kamera tidak ada, tag software
        editing, timestamp tidak ada) -> 0..3.
        Jika model kamera DAN timestamp tidak ada -> c minimal 3.
        Jika EXIF praktis kosong (< 3 tag terbaca)  -> c = 4.
    a : 1 jika model kamera tidak ditemukan atau tidak valid, selain itu 0.
    b : 1 jika tag EXIF yang terbaca < 3, selain itu 0.
    GPS, resolusi dan flash = informasi tambahan (tidak menambah c).
    """
    exif = read_exif(file_bytes)

    camera_make = exif.get('Make', '').strip()
    camera_model = exif.get('Model', '').strip()
    camera_valid, camera_note = validate_camera(camera_make, camera_model)
    has_camera = camera_valid   # model tidak valid diperlakukan sama dengan tidak ditemukan

    software = exif.get('Software', '').strip()
    edit_sw, sw_kind = classify_software(software)

    timestamp = exif.get('DateTimeOriginal') or exif.get('DateTime') or ''
    has_timestamp = bool(timestamp)

    has_gps = 'GPSInfo' in exif
    res_x, res_y = exif.get('XResolution', ''), exif.get('YResolution', '')
    has_res = bool(res_x or res_y)
    flash = exif.get('Flash', '')
    has_flash = bool(flash)

    c = int(not has_camera) + int(edit_sw) + int(not has_timestamp)
    exif_count = len(exif)
    b = 1 if exif_count < 3 else 0
    if not has_camera and not has_timestamp:
        c = max(c, 3)
    if b:
        c = max(c, 4)
    a = 1 if not has_camera else 0

    s_m = min(1.0, 0.16 * c + 0.25 * a + 0.15 * b)

    indicators = [
        {'no': 1, 'name': 'Merek dan model kamera',
         'value': ((f'{camera_make} {camera_model}'.strip() + f' ({camera_note})')
                   if camera_model else 'Tidak ditemukan'),
         'suspicious': not has_camera, 'role': 'Menambah nilai c'},
        {'no': 2, 'name': 'Tag software',
         'value': (f'{software} ({sw_kind})' if software else 'Tidak ada'),
         'suspicious': edit_sw, 'role': 'Menambah nilai c'},
        {'no': 3, 'name': 'Timestamp',
         'value': timestamp or 'Tidak ditemukan',
         'suspicious': not has_timestamp, 'role': 'Menambah nilai c'},
        {'no': 4, 'name': 'GPS/geolokasi',
         'value': 'Ada' if has_gps else 'Tidak ditemukan',
         'suspicious': not has_gps, 'role': 'Informasi tambahan'},
        {'no': 5, 'name': 'Resolusi',
         'value': f'{res_x} x {res_y}' if has_res else 'Tidak ditemukan',
         'suspicious': not has_res, 'role': 'Informasi tambahan'},
        {'no': 6, 'name': 'Info flash kamera',
         'value': flash if has_flash else 'Tidak ditemukan',
         'suspicious': not has_flash, 'role': 'Informasi tambahan'},
    ]

    return {
        'score': s_m,
        'score_pct': round(s_m * 100, 2),
        'c': c, 'a': a, 'b': b,
        'exif_count': exif_count,
        'formula': f'min(1; 0,16 x {c} + 0,25 x {a} + 0,15 x {b})',
        'indicators': indicators,
        'decision': 'Manipulasi' if s_m > T else 'Asli',
    }


# -----------------------------------------------------------------------------
# Jalur 2 — CNN MobileNetV2 (2.4)
# -----------------------------------------------------------------------------
def run_cnn(img_full, img_fallback):
    """
    Jalur CNN: deteksi & crop wajah -> 224 x 224 -> x/255 -> S_c = sigmoid(...)
    img_full     : citra RGB orientasi benar, resolusi asli (untuk deteksi & crop)
    img_fallback : citra hasil praproses 600 x 600 (dipakai bila wajah tidak terdeteksi)
    """
    it = get_interpreter()
    if it is None:
        raise RuntimeError(f'Model CNN tidak dapat dimuat: {_model_error}')
    crop, face = detect_face_crop(img_full)
    src = crop if crop is not None else img_fallback
    cnn_in = src.resize(CNN_SIZE, Image.BICUBIC).convert('RGB')
    buf = io.BytesIO()
    cnn_in.save(buf, format='JPEG', quality=90)
    x = np.asarray(cnn_in, dtype=np.float32) / 255.0
    x = np.expand_dims(x, 0)
    with _interp_lock:
        inp = it.get_input_details()[0]['index']
        out = it.get_output_details()[0]['index']
        it.set_tensor(inp, x)
        it.invoke()
        p_real = float(it.get_tensor(out)[0][0])
    # Model dilatih dengan folder Kaggle 'fake/' dan 'real/'; Keras mengurutkan kelas
    # secara alfabetis (fake = 0, real = 1), sehingga neuron sigmoid memberi peluang
    # citra ASLI. Skor manipulasi S_c = 1 - p_real. (Diverifikasi pada contoh FFHQ
    # asli vs StyleGAN: keluaran mentah rata-rata 0,89 untuk asli, 0,49-0,71 untuk palsu.)
    s_c = 1.0 - p_real
    return {
        'score': s_c,
        'score_pct': round(s_c * 100, 2),
        'model': 'MobileNetV2 (ImageNet, beku) + GAP + Dense256/Drop0,5 + Dense64/Drop0,3 + Sigmoid',
        'raw_sigmoid_p_real': p_real,
        'face': face,
        'input': 'crop wajah' if crop is not None else 'citra utuh (wajah tidak terdeteksi)',
        'cnn_input_image': base64.b64encode(buf.getvalue()).decode('ascii'),
        'decision': 'Manipulasi' if s_c > T else 'Asli',
    }


# -----------------------------------------------------------------------------
# Weighted scoring (2.7)
# -----------------------------------------------------------------------------
def weighted(s_m, s_c):
    s = W_M * s_m + W_C * s_c
    return {
        'score': s,
        'score_pct': round(s * 100, 2),
        'formula': f'0,40 x {s_m:.4f} + 0,60 x {s_c:.4f}',
        'threshold': T,
        'decision': 'Manipulasi' if s > T else 'Asli',
    }


def analyze_bytes(file_bytes):
    """Satu pipeline untuk web dan uji batch."""
    meta = run_metadata(file_bytes)
    img_full = load_oriented(file_bytes)
    img = img_full.copy()
    img.thumbnail(MAX_SIZE, Image.LANCZOS)
    cnn = run_cnn(img_full, img)
    final = weighted(meta['score'], cnn['score'])
    return {
        'version': VERSION,
        'metadata': meta,
        'cnn': cnn,
        'weighted': final,
        'scenarios': {
            'metadata_tunggal': meta['decision'],
            'cnn_tunggal': cnn['decision'],
            'gabungan': final['decision'],
        },
    }


# -----------------------------------------------------------------------------
# Endpoint
# -----------------------------------------------------------------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Tidak ada file gambar (field "image")'}), 400
        file_bytes = request.files['image'].read()
        if not file_bytes:
            return jsonify({'error': 'File kosong'}), 400
        result = analyze_bytes(file_bytes)
        result['filename'] = request.files['image'].filename
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    loaded = get_interpreter() is not None
    return jsonify({
        'status': 'ok' if loaded else 'model_error',
        'version': VERSION,
        'cnn_loaded': loaded,
        'cnn_error': _model_error,
        'methods': ['MetadataEXIF', 'CNN-MobileNetV2'],
        'cnn_preprocess': 'koreksi orientasi EXIF + deteksi & crop wajah (YuNet) + 224x224 / 255',
        'weights': {'metadata': W_M, 'cnn': W_C},
        'threshold': T,
    }), (200 if loaded else 503)


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'name': 'Deteksi Manipulasi Citra Wajah — versi publikasi',
        'author': 'Maulana Ahmad Nugroho - 25917001 - UII 2026',
        'version': VERSION,
        'formula': 'S = 0,40 x S_m + 0,60 x S_c ; S > 0,50 -> manipulasi',
        'endpoint': 'POST /analyze (multipart, field "image")',
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
