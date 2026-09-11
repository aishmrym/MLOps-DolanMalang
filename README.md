# MLOps-DolanMalang

Sistem prediksi kelayakan aktivitas luar ruang (Outdoor Comfort Index) untuk lokasi wisata Malang Raya, dibangun sebagai proyek mata kuliah Machine Learning Operation (MLOps).

## Tujuan Proyek

Prakiraan cuaca publik untuk area Malang Raya kurang akurat di lokasi pegunungan karena resolusi grid model global (9-25 km) tidak menangkap perbedaan elevasi yang tajam dalam radius yang sempit (446 mdpl di Kayutangan sampai 2.177 mdpl di Bromo). Proyek ini membangun model regresi yang memprediksi Outdoor Comfort Index per lokasi, per jam, untuk horizon H+1 sampai H+7, dengan mengoreksi bias prakiraan mentah pada tiap mikroklimat.

Sistem dirancang sebagai pipeline MLOps penuh: pengambilan data terjadwal, pelabelan otomatis (label muncul sendiri setelah waktu prakiraan terlewati), retraining berbasis pemicu (jadwal, performa, drift distribusi), dan monitoring produksi. Detail masalah, strategi continual learning, dan kriteria keberhasilan ada di dokumen LK-01 pada folder `docs/`.

## Struktur Direktori

```
MLOps-DolanMalang/
├── .devcontainer/
│   └── devcontainer.json      # konfigurasi environment Codespaces
├── .github/
│   └── workflows/              # GitHub Actions (data ingestion, CI)
├── config/
│   ├── locations.yaml          # daftar lokasi + koordinat + elevasi
│   └── model_config.yaml       # hyperparameter & lead time
├── data/
│   ├── raw/                    # snapshot mentah hasil ingestion (tidak diubah)
│   ├── interim/                # data yang sudah divalidasi/dibersihkan
│   ├── processed/               # data siap latih (fitur + label)
│   └── external/                # referensi statis (kalender libur, dsb.)
├── models/                      # model terlatih (tracked via DVC, bukan Git langsung)
├── notebooks/                   # eksperimen & EDA, penamaan: 01-nama-eksperimen.ipynb
├── src/
│   ├── ingestion/                # fetch data dari Open-Meteo API
│   ├── features/                 # feature engineering & perhitungan Outdoor Comfort Index
│   ├── training/                  # training & evaluasi model
│   └── serving/                   # REST API untuk inference
├── tests/                        # unit test untuk src/
├── docs/                          # dokumen proyek (LK-01, dsb.)
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

Struktur ini mengikuti pola Cookiecutter Data Science: data mentah dan data olahan dipisah, kode produksi (`src/`) dipisah dari eksplorasi (`notebooks/`), dan konfigurasi dipisah dari kode agar mudah diubah tanpa menyentuh logika.

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

## Branching Strategy

Proyek ini menggunakan **GitHub Flow**:

- `main` selalu dalam kondisi yang bisa dijalankan (stabil).
- Setiap eksperimen atau fitur baru dikerjakan di branch terpisah, contoh: `feat/initial-eda`, `feat/ingestion-pipeline`.
- Branch di-push ke GitHub, lalu dibuka Pull Request ke `main`.
- Merge ke `main` hanya dilakukan setelah perubahan divalidasi (review sendiri/dosen, dan notebook/script sudah dijalankan tanpa error).

## Status

Proyek masih dalam tahap inisiasi (LK-01 dan LK-02). Sumber data, strategi continual learning, dan kriteria keberhasilan lengkap ada di `docs/LK01-Aisha_Maryam-245150201111040.pdf`.