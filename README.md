# Digital Forensic Image API

API backend untuk sistem deteksi manipulasi citra digital berbasis pendekatan hybrid:
- Metadata Analysis
- Error Level Analysis (ELA)
- Clone Detection (ORB approximation)
- Local Noise Variance

## Endpoint

### POST /analyze
Upload citra untuk dianalisis.

**Request:** multipart/form-data dengan field `image`

**Response:** JSON berisi skor per metode, heatmap base64, weighted score, confusion matrix estimasi

### GET /health
Cek status API.

## Deploy ke Railway

1. Push repo ini ke GitHub
2. Connect ke Railway via GitHub
3. Railway otomatis detect Python dan deploy

## Teknologi
- Flask + Gunicorn
- Pillow (PIL)
- NumPy

---

## Versi Publikasi (`app_publikasi.py`)

Terpisah dari versi tesis (`app_v2.py` tidak diubah). Hanya 2 metode sesuai naskah publikasi:

- **S_m** — analisis metadata EXIF: `min(1; 0,16c + 0,25a + 0,15b)`
- **S_c** — CNN MobileNetV2 (model `.tflite`, konversi float32 dari `.h5`, lihat `convert_tflite.py`)
- **S = 0,40 S_m + 0,60 S_c**, keputusan **S > 0,50 → manipulasi**

Deploy Render (service terpisah):

- Build Command: `pip install -r requirements_publikasi.txt`
- Start Command: `gunicorn app_publikasi:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1`
- Environment: `PYTHON_VERSION=3.11.9`

Uji batch data uji (kode yang sama dengan API):

```
python evaluate_publikasi.py data_uji/ hasil/
```
