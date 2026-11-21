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

import numpy as np
from PIL import Image, ExifTags
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    from ai_edge_litert.interpreter import Interpreter
    LITERT_AVAILABLE = True
except ImportError:  # pragma: no cover
    LITERT_AVAILABLE = False

VERSION = 'publikasi-1.0'

# -----------------------------------------------------------------------------
# Parameter (bagian 2.2, 2.3 dan 2.7 naskah)
# -----------------------------------------------------------------------------
MAX_SIZE = (600, 600)      # praproses: maksimum 600 x 600 piksel
CNN_SIZE = (224, 224)      # masukan MobileNetV2
W_M = 0.40                 # bobot metadata
W_C = 0.60                 # bobot CNN
T = 0.50                   # ambang keputusan

EDITING_SOFTWARE = ['adobe', 'photoshop', 'lightroom', 'gimp', 'canva', 'picsart',
                    'snapseed', 'facetune', 'faceapp', 'meitu', 'pixnova', 'remini',
                    'stable', 'midjourney', 'dall', 'firefly']

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'cnn_mobilenetv2_forensik.tflite')

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
def load_image(file_bytes):
    """Konversi ke RGB lalu perkecil hingga maks 600 x 600 (rasio dipertahankan)."""
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    img.thumbnail(MAX_SIZE, Image.LANCZOS)
    return img


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
    a : 1 jika model kamera tidak ditemukan, selain itu 0.
    b : 1 jika tag EXIF yang terbaca < 3, selain itu 0.
    GPS, resolusi dan flash = informasi tambahan (tidak menambah c).
    """
    exif = read_exif(file_bytes)

    camera_make = exif.get('Make', '').strip()
    camera_model = exif.get('Model', '').strip()
    has_camera = bool(camera_model)

    software = exif.get('Software', '').strip()
    edit_sw = bool(software) and any(k in software.lower() for k in EDITING_SOFTWARE)

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
         'value': (f'{camera_make} {camera_model}'.strip() or 'Tidak ditemukan'),
         'suspicious': not has_camera, 'role': 'Menambah nilai c'},
        {'no': 2, 'name': 'Tag software',
         'value': software or 'Tidak ada',
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
def run_cnn(img):
    """x' = citra RGB 224 x 224 / 255  ->  S_c = sigmoid(...)"""
    it = get_interpreter()
    if it is None:
        raise RuntimeError(f'Model CNN tidak dapat dimuat: {_model_error}')
    x = np.asarray(img.resize(CNN_SIZE, Image.BICUBIC).convert('RGB'),
                   dtype=np.float32) / 255.0
    x = np.expand_dims(x, 0)
    with _interp_lock:
        inp = it.get_input_details()[0]['index']
        out = it.get_output_details()[0]['index']
        it.set_tensor(inp, x)
        it.invoke()
        s_c = float(it.get_tensor(out)[0][0])
    return {
        'score': s_c,
        'score_pct': round(s_c * 100, 2),
        'model': 'MobileNetV2 (ImageNet, beku) + GAP + Dense256/Drop0,5 + Dense64/Drop0,3 + Sigmoid',
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
    img = load_image(file_bytes)
    cnn = run_cnn(img)
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
