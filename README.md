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

## Versioning Data dengan DVC (LK-05)

Data di `data/raw/` dilacak DVC, bukan Git. Git hanya menyimpan file penunjuk kecil `data/raw.dvc` (berisi hash `md5` isi folder), sedangkan data aslinya disimpan di cache/remote DVC. Repositori tetap ringan dan tiap versi data tetap bisa dipulihkan.

### Alur menambah versi data
1. `python src/ingestion/ingest_data.py` mengambil data terbaru. Setiap run membuat file baru bertimestamp di `data/raw/`.
2. `dvc status` menunjukkan `data/raw` berubah.
3. `dvc add data/raw` menghitung ulang hash dan memperbarui `data/raw.dvc`.
4. `git add data/raw.dvc && git commit -m "Update dataset"` lalu `git tag v2-data` untuk menandai versi.
5. `dvc diff v1-data v2-data` membandingkan dua versi dataset.

### Mengembalikan versi data lama
`git checkout v1-data` lalu `dvc checkout` memulihkan isi `data/raw/` persis seperti versi 1.

### Mengapa versioning data krusial untuk reproduksibilitas model
Model hanya bisa direproduksi jika kode, parameter, dan data yang dipakai saat training diketahui persis. Pada sistem continual learning, data bertambah setiap hari, sehingga model hari ini dan model bulan lalu dilatih dari data berbeda. Tanpa versioning, kita tidak bisa menjawab model dilatih dengan data yang mana, dan tidak bisa menelusuri apakah penurunan akurasi berasal dari kode atau dari data yang berubah (data drift). Dengan DVC, setiap commit Git terikat ke hash dataset tertentu, sehingga eksperimen dapat diulang dan hasil dua versi data dapat dibandingkan secara adil.

## Status

Proyek sudah melewati tahap inisiasi (LK-01, LK-02), perancangan pipeline data (LK-03), dan implementasi ingestion + preprocessing (LK-04). Detail lengkap ada di `docs/`.


## Eksperimen ML dengan MLflow (LK-06)

Model memprediksi **Outdoor Comfort Index** (0-100) 24 jam ke depan per lokasi dari kondisi saat ini. Label dihitung otomatis oleh `src/features/comfort_index.py`, lalu digeser ke depan, sehingga model memprediksi masa depan dan bukan menghitung ulang rumus yang sama. Split train/test berbasis waktu (tanpa kebocoran), dan setiap run dibandingkan dengan **baseline naif** (indeks saat ini dipakai sebagai prediksi).

### Menjalankan eksperimen
```bash
pip install -r requirements.txt
bash scripts/run_experiments.sh                 # 5 run dengan hyperparameter berbeda
python src/training/train.py --model ridge --alpha 1.0 --run-version v1   # satu run
```

### Membuka MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Di Codespaces, buka tab **Ports** lalu klik alamat port 5000. Jika muncul `Invalid Host header`, jalankan:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000 \
    --allowed-hosts "*" --cors-allowed-origins "*"
```

### Memilih dan mendaftarkan model terbaik
```bash
python src/training/select_best.py
```
Skrip menampilkan tabel perbandingan seluruh run, mendaftarkan run dengan `test_mae` terendah ke Model Registry (alias `champion`), dan menulis `models/best_model_metadata.json`.

### Yang dicatat di setiap run
- **Parameter:** model, hyperparameter, horizon, seed, ukuran train/test, batas split.
- **Metrik:** MAE/RMSE/R2 train dan test, MAE baseline, skill score, MAE per lokasi.
- **Artefak:** model (dengan signature), plot fitur penting, plot aktual vs prediksi, `test_predictions.csv`, `requirements.txt`.
- **Tag reproduksibilitas:** `dataset_version` (hash md5 dari `data/raw.dvc`) dan `git_commit`, sehingga tiap run bisa ditelusuri ke versi data dan kode yang tepat.

### Reproduksibilitas
Seed tetap (`--seed 42`), split berbasis waktu yang deterministik, data versi DVC, dan lingkungan Codespaces yang sama. Menjalankan ulang perintah yang sama pada versi data yang sama menghasilkan metrik yang sama.
Proyek sudah melewati tahap inisiasi (LK-01, LK-02), perancangan pipeline data (LK-03), implementasi ingestion + preprocessing (LK-04), dan Manajemen Data Versioning Menggunakan DVC. Detail lengkap ada di `docs/`.

## Model Registry dan Versioning (LK-07)

Model terdaftar di MLflow Model Registry dengan nama `dolan-malang-comfort-index`.

**Versi aktif untuk inferensi (alias `production`): v1**, yaitu run `gradient_boosting_v4_20261101`.

### Perbandingan versi

| Versi | Run | Hyperparameter utama | Test MAE | Test RMSE | Test R² | Gap train-test (MAE) | Skill vs baseline | Alias |
|---|---|---|---|---|---|---|---|---|
| v1 | gradient_boosting_v4_20261101 | n_estimators=150, max_depth=3, learning_rate=0.05 | 9.727 | 12.647 | 0.2945 | 3.740 | 18.5% | champion, production |
| v2 | gradient_boosting_v6_20261101 | n_estimators=250, max_depth=4, learning_rate=0.03 | 9.594 | 12.613 | 0.2983 | 4.732 | 19.6% | staging |

Baseline naif (indeks dari prakiraan mentah) memiliki test MAE 11.929. Kedua versi dilatih pada 4 snapshot data yang sama dengan split berbasis waktu (cutoff 30 Oktober 2026).

### Alasan memilih v1

v2 memang menurunkan test MAE dari 9.727 menjadi 9.594 (turun 0.133 atau sekitar 1.4%), tetapi gap train-test naik dari 3.740 menjadi 4.732 (naik sekitar 26%). Peningkatan akurasi yang sangat kecil dibarengi tanda overfitting yang lebih besar, sehingga v1 dipilih sebagai versi production karena lebih stabil terhadap data baru. v2 dipertahankan pada alias `staging` sebagai kandidat yang akan dievaluasi ulang ketika data bertambah.

### Alur transisi

1. Run terbaik hasil `select_best.py` didaftarkan sebagai **v1** (alias `champion`).
2. Varian dengan parameter berbeda dilatih dan didaftarkan sebagai **v2**.
3. v1 dipromosikan `staging` lalu `production`; v2 dipromosikan ke `staging`.

MLflow menandai stage (None/Staging/Production) sebagai deprecated sejak versi 2.9 dan menggantinya dengan alias. Karena itu alur ini memakai alias `staging` dan `production`, dan stage legacy tetap diubah paralel supaya transisi terlihat pada UI.

### Memuat model aktif

```python
import mlflow

model = mlflow.pyfunc.load_model("models:/dolan-malang-comfort-index@production")
```

Uji kesiapan inferensi (memuat model dari registry, memprediksi 8 lokasi dan satu record, mengukur latensi):

```bash
python src/serving/verify_inference.py
```

Hasil uji terakhir: model termuat dalam 3.19 detik dan latensi rata-rata 3.5 ms per prediksi. MAE batch 7.33 hanya bukti bahwa model dapat dipanggil, bukan ukuran akurasi. Akurasi resminya adalah test MAE pada tabel di atas.

### Silsilah data (data lineage)

Versi model ditautkan ke versi data dan kode lewat DVC dan Git:

- `models/registry_lineage.yaml` mencatat versi model, run, hash dataset DVC, dan commit Git.
- `models/production.dvc` menunjuk artefak model yang sedang berstatus production.
- `data/raw.dvc` menyimpan hash dataset yang dipakai untuk melatih model. Hash dataset v1: `942f1feba20abaaae240802876e41de7.dir`, commit Git `268978f`.

### Perintah yang dipakai

```bash
python src/training/registry_workflow.py status      # lihat versi, stage, dan alias
python src/training/registry_workflow.py compare     # bandingkan antar versi
python src/training/registry_workflow.py register --run-name NAMA_RUN
python src/training/registry_workflow.py promote 1 production
python src/training/registry_workflow.py lineage     # tulis models/registry_lineage.yaml
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Catatan reproduksibilitas: `mlflow.db` tidak ikut Git. Di lingkungan baru, jalankan `bash scripts/run_experiments.sh` lalu `python src/training/select_best.py` terlebih dahulu agar v1 terdaftar.