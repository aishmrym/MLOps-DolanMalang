# MLOps-DolanMalang

Sistem prediksi kelayakan aktivitas luar ruang (Outdoor Comfort Index) untuk lokasi wisata Malang Raya, dibangun sebagai proyek mata kuliah Machine Learning Operation (MLOps).

## Tujuan Proyek

Prakiraan cuaca publik untuk area Malang Raya kurang akurat di lokasi pegunungan karena resolusi grid model global (9-25 km) tidak menangkap perbedaan elevasi yang tajam dalam radius yang sempit (446 mdpl di Kayutangan sampai 2.177 mdpl di Bromo). Proyek ini membangun model regresi yang memprediksi Outdoor Comfort Index per lokasi, per jam, untuk horizon H+1 sampai H+7, dengan mengoreksi bias prakiraan mentah pada tiap mikroklimat.

Sistem dirancang sebagai pipeline MLOps penuh: pengambilan data terjadwal, pelabelan otomatis (label muncul sendiri setelah waktu prakiraan terlewati), retraining berbasis pemicu (jadwal, performa, drift distribusi), dan monitoring produksi. Detail masalah, strategi continual learning, dan kriteria keberhasilan ada di dokumen LK-01 dan rancangan pipeline data ada di LK-03 pada folder `docs/`.

## Struktur Direktori

```
MLOps-DolanMalang/
├── .devcontainer/
│   └── devcontainer.json      # konfigurasi environment Codespaces
├── .github/
│   └── workflows/              # GitHub Actions (data ingestion, CI)
├── config/
│   ├── locations.yaml           # daftar lokasi + koordinat + elevasi
│   └── model_config.yaml       # hyperparameter & lead time
├── data/
│   ├── raw/                    # snapshot mentah hasil ingestion (append-only)
│   ├── interim/                # data yang sudah dibersihkan (preprocess.py)
│   ├── processed/               # data siap latih (fitur + label) — LK-05
│   └── external/                # referensi statis (kalender libur, dsb.)
├── models/                      # model terlatih (tracked via DVC, bukan Git langsung)
├── notebooks/                   # eksperimen & EDA, penamaan: 01-nama-eksperimen.ipynb
├── src/
│   ├── ingestion/
│   │   └── ingest_data.py       # menarik data dari Open-Meteo (LK-04)
│   ├── features/
│   │   └── preprocess.py        # membersihkan data mentah (LK-04)
│   ├── training/                # training & evaluasi model
│   └── serving/                 # REST API untuk inference
├── tests/                        # unit test untuk src/
├── docs/                          # dokumen proyek (LK-01, LK-03, dsb.)
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

Struktur ini mengikuti pola Cookiecutter Data Science: data mentah dan data olahan dipisah, kode produksi (`src/`) dipisah dari eksplorasi (`notebooks/`), dan konfigurasi dipisah dari kode agar mudah diubah tanpa menyentuh logika.

> Catatan: instruksi LK-04 menyebut folder `src/data` sebagai lokasi skrip ingestion. Repositori ini memakai `src/ingestion/` dan `src/features/` secara konsisten dengan rancangan struktur yang sudah ditetapkan sejak LK-02/LK-03 — fungsinya sama persis, hanya penamaan folder yang berbeda.

## Cara Menjalankan (GitHub Codespaces)

1. Buka repositori ini di GitHub, klik tombol **Code > Codespaces > Create codespace on main**.
2. Tunggu Codespaces membangun environment (image Python 3.11 + ekstensi VS Code untuk Python, Jupyter, dan Git). Proses ini otomatis menjalankan `pip install -r requirements.txt` lewat `postCreateCommand`, jadi tidak perlu instalasi manual.
3. Setelah environment siap, jalankan notebook di `notebooks/` atau script di `src/` langsung dari VS Code di browser.

Untuk menjalankan secara lokal (opsional, di luar Codespaces):

```bash
git clone https://github.com/<username>/MLOps-DolanMalang.git
cd MLOps-DolanMalang
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Menjalankan Data Ingestion & Preprocessing (LK-04)

### 1. Menarik data mentah

```bash
python src/ingestion/ingest_data.py
```

Script ini akan:
- Membaca daftar lokasi dari `config/locations.yaml` (8 titik di Malang Raya dengan elevasi 5–2.177 mdpl).
- Memecah permintaan API per 3 lokasi per panggilan (menghindari rate limit HTTP 429, sesuai temuan pada LK-01).
- Menarik tiga sumber: Forecast API + Air Quality API (digabung jadi satu snapshot fitur), dan Previous Runs API (dipakai untuk cek bias per lead time).
- Retry otomatis dengan exponential backoff untuk error jaringan/rate limit; kegagalan satu chunk tidak menggagalkan seluruh proses.
- Validasi skema (kolom wajib ada, rasio null wajar) sebelum file diterima.
- Menyimpan hasil ke `data/raw/snapshot_YYYYMMDDThhmm.csv` dan `data/raw/bias_previous_runs_YYYYMMDDThhmm.csv` — setiap eksekusi membuat file baru, tidak pernah menimpa yang lama.

Opsi yang tersedia:

```bash
python src/ingestion/ingest_data.py --chunk-size 2 --sleep 2.0   # lebih lambat, lebih aman dari rate limit
python src/ingestion/ingest_data.py --skip-previous-runs         # lebih cepat saat debugging
```

### 2. Membersihkan data (preprocessing)

```bash
python src/features/preprocess.py
```

Secara default script ini otomatis memakai snapshot forecast terbaru di `data/raw/`. Bisa juga menunjuk file spesifik:

```bash
python src/features/preprocess.py --input data/raw/snapshot_20260929T2300.csv
```

Langkah pembersihan yang dijalankan (urut):
1. Normalisasi timestamp ke WIB.
2. Deduplikasi baris `(location_id, time)` yang tumpang tindih antar pemanggilan API.
3. Validasi rentang nilai wajar per variabel (mis. kelembapan harus 0–100%) — nilai di luar rentang diubah jadi kosong, bukan dihapus barisnya.
4. Interpolasi gap kosong pendek (≤ 2 jam) per lokasi; gap lebih panjang sengaja dibiarkan kosong dan ditandai lewat kolom `is_imputed`, akan ditangani lebih lanjut di LK-05.
5. Menambahkan fitur kalender: `is_weekend` dan `is_holiday` (libur nasional Indonesia, lewat paket `holidays`).

Hasil disimpan ke `data/interim/cleaned_YYYYMMDDThhmm.csv`.

### 3. Menjadwalkan ulang (persiapan LK-05)

Kedua script ini ditulis agar bisa dipanggil berulang tanpa efek samping merusak (setiap run menambah file baru, bukan menimpa), sehingga siap dipasang sebagai job terjadwal (GitHub Actions cron) pada tahap continual learning berikutnya.

## Branching Strategy

Proyek ini menggunakan **GitHub Flow**:

- `main` selalu dalam kondisi yang bisa dijalankan (stabil).
- Setiap eksperimen atau fitur baru dikerjakan di branch terpisah, contoh: `feat/initial-eda`, `feat/data-ingestion`.
- Branch di-push ke GitHub, lalu dibuka Pull Request ke `main`.
- Merge ke `main` hanya dilakukan setelah perubahan divalidasi (review sendiri/dosen, dan notebook/script sudah dijalankan tanpa error).

## Status

Proyek sudah melewati tahap inisiasi (LK-01, LK-02), perancangan pipeline data (LK-03), dan implementasi ingestion + preprocessing (LK-04). Detail lengkap ada di `docs/`.
