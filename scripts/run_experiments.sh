#!/usr/bin/env bash
# Menjalankan 5 eksperimen dengan hyperparameter yang berbeda secara bermakna.
# Jalankan dari mana saja: skrip ini otomatis pindah ke root repositori.
set -euo pipefail
cd "$(dirname "$0")/.."

# v1: baseline linear (regularisasi sedang)
python src/training/train.py --model ridge --alpha 1.0 --run-version v1

# v2: random forest dangkal (kapasitas rendah, risiko underfit)
python src/training/train.py --model random_forest \
    --n-estimators 50 --max-depth 3 --min-samples-leaf 5 --run-version v2

# v3: random forest dalam (kapasitas tinggi, risiko overfit)
python src/training/train.py --model random_forest \
    --n-estimators 300 --max-depth 12 --min-samples-leaf 1 --run-version v3

# v4: gradient boosting konservatif (learning rate kecil)
python src/training/train.py --model gradient_boosting \
    --n-estimators 150 --learning-rate 0.05 --max-depth 3 --run-version v4

# v5: gradient boosting agresif (learning rate besar, pohon dalam)
python src/training/train.py --model gradient_boosting \
    --n-estimators 400 --learning-rate 0.1 --max-depth 6 --run-version v5

echo "Selesai. Buka MLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db"
