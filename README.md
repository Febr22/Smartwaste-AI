# Smartwaste-AI: Klasifikasi Jenis Sampah (TrashNet)

Tugas Besar Deep Learning. Proyek ini membandingkan **CNN baseline (dari nol)** dengan **EfficientNetB0 (transfer learning + fine-tuning)** untuk mengenali 6 jenis sampah dari foto: *cardboard, glass, metal, paper, plastic, trash*. Model usulan diintegrasikan ke prototipe sistem **FastAPI + React** yang dapat dijalankan dengan Docker.

## Masalah bisnis

Pemilahan sampah di TPS3R, bank sampah, dan fasilitas daur ulang (MRF) masih banyak dilakukan manual: lambat, tidak konsisten, dan berisiko bagi petugas. Sampah daur ulang yang terkontaminasi jenis lain juga turun nilai jualnya. Model klasifikasi citra dapat menjadi dasar asisten pemilahan berbasis kamera atau aplikasi web.

## Hasil utama

Data uji yang sama, rerata ± simpangan baku dari 3 seed (42, 123, 2024).

| Model | Acc. | F1 (macro) | Param (M) | Inferensi CPU (ms) |
|---|---|---|---|---|
| Baseline CNN | 0,7018 ± 0,0053 | 0,6895 ± 0,0077 | 1,24 | 43,86 |
| **EfficientNetB0 (usulan)** | **0,9024 ± 0,0070** | **0,8875 ± 0,0089** | 4,06 | 52,00 |

Studi ablasi (F1 macro): tanpa fine-tuning 0,8251; tanpa augmentasi 0,8908; usulan penuh 0,8875. Detail lengkap ada di `tables/` dan `figures/`.

## Struktur repositori

```
Smartwaste-AI/
├── Smartwaste_AI.ipynb      # notebook lengkap: data, EDA, pelatihan, evaluasi, simpan model
├── models/                  # model final (.h5 dan .keras), class_names.json, model_info.json
├── runs/                    # riwayat, metrik, dan probabilitas prediksi tiap run
├── tables/                  # tabel hasil untuk laporan (CSV)
├── figures/                 # EDA, kurva pelatihan, confusion matrix, Grad-CAM
├── backend/                 # API FastAPI (model serving)
├── frontend/                # antarmuka React (Vite)
├── docker-compose.yml
└── README.md
```

Bobot per-run (`runs/*/core.weights.h5`, sekitar 200 MB) tidak disertakan karena ukurannya. Model final tetap lengkap di `models/`.

## 1. Notebook

Buka `Smartwaste_AI.ipynb` di Google Colab (GPU T4). Seluruh eksperimen (9 run) memakan waktu sekitar 2–3 jam. Dataset diunduh otomatis dari Hugging Face ([garythung/trashnet](https://huggingface.co/datasets/garythung/trashnet)), seed tetap, dan run yang sudah selesai dilewati saat dijalankan ulang (*resume*).

## 2. Menjalankan prototipe

### Cara A: Docker (disarankan)

Prasyarat: Docker Desktop terpasang dan **sedang berjalan** (ikon paus di system tray berstatus *Engine running*).

```bash
# jalankan dari root repo (folder yang berisi docker-compose.yml)
docker compose up --build
```

Build pertama memakan 5–15 menit karena mengunduh TensorFlow dan paket npm. Setelah log menampilkan `model dimuat` dan `Uvicorn running`:

| Layanan | Alamat |
|---|---|
| Aplikasi web | http://localhost:5173 |
| API | http://localhost:8000 |
| Dokumentasi API (Swagger) | http://localhost:8000/docs |

Perintah berguna:

```bash
docker compose up -d             # jalan di latar belakang
docker compose logs -f backend   # lihat log backend
docker compose down              # hentikan
```

### Cara B: Manual (tanpa Docker)

Backend:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate           # Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Frontend (terminal lain):

```bash
cd frontend
copy .env.example .env           # Linux/macOS: cp .env.example .env
npm install
npm run dev
```

Gunakan `python -m pip` dan `python -m uvicorn` (bukan `pip` / `uvicorn` langsung) supaya aman di Windows, terutama bila path proyek mengandung spasi.

## API

| Endpoint | Fungsi |
|---|---|
| `GET /health` | status dan model yang termuat |
| `GET /models` | daftar model dan model default |
| `POST /predict?model=efficientnetb0` | unggah gambar (form field `file`), mengembalikan kelas, probabilitas, dan waktu inferensi |

Contoh:

```bash
curl -X POST "http://localhost:8000/predict?model=efficientnetb0" -F "file=@contoh.jpg"
```

Respons:

```json
{
  "model": "efficientnetb0",
  "prediction": "plastic",
  "confidence": 0.93,
  "probabilities": { "plastic": 0.93, "glass": 0.04, "metal": 0.02, "...": 0.01 },
  "inference_ms": 52.1
}
```

Validasi: hanya JPG/PNG/WEBP, maksimal 8 MB. Mengembalikan 415 (tipe salah), 413 (terlalu besar), 400 (bukan gambar valid atau model tidak dikenal).

## Catatan teknis penting

- **Muat `.h5` dengan `load_h5_model`.** Lapisan `Normalization` milik EfficientNet tidak menghitung ulang mean dan variance saat `.h5` dimuat, sehingga prediksi salah kecuali `finalize_state()` dipanggil sebelum predict pertama. Ini sudah diterapkan di `backend/inference.py`. Format `.keras` tidak membutuhkan langkah ini dan dipakai sebagai cadangan.
- **Pra-pemrosesan:** RGB, `exif_transpose`, resize 224×224 bicubic, float32 dengan rentang 0–255 dan **tanpa dibagi 255** (normalisasi sudah ada di dalam model).
- **Urutan kelas** mengikuti `models/class_names.json`.
- **Versi library:** model dilatih dan diuji dengan TensorFlow 2.21.0 dan Keras 3.13.2 (lihat `models/model_info.json`). Samakan versinya bila `.h5` gagal dimuat.

## Troubleshooting (Windows)

| Gejala | Penyebab dan solusi |
|---|---|
| Notifikasi *"Part of this app has been blocked ... uvicorn.exe / _pywrap_tf2.pyd"* | Smart App Control memblokir berkas native yang tidak bertanda tangan. Gunakan **Docker** (Cara A), karena kontainer Linux tidak terkena aturan ini. |
| `unknown command ".../pip.exe"` atau `unexpected extra argument` | Path proyek mengandung spasi sehingga launcher venv rusak. Gunakan `python -m pip` dan `python -m uvicorn`. |
| `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified` | Docker Desktop belum berjalan. Buka Docker Desktop, tunggu sampai *Engine running*, lalu ulangi. |
| `no configuration file provided: not found` | Perintah `docker compose` dijalankan di folder yang salah. Jalankan dari root repo. |
| Web menampilkan *"Tidak dapat terhubung ke API"* | Backend belum selesai memuat model. Cek http://localhost:8000/health lalu muat ulang halaman. |

## Dataset dan lisensi

TrashNet oleh Gary Thung dan Mindy Yang, dimuat dari Hugging Face. Dataset tidak disalin ke repositori ini; notebook mengunduhnya otomatis. Periksa halaman dataset untuk ketentuan lisensi.
