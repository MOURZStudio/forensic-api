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
