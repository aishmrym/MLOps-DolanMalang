set -euo pipefail
cd "$(dirname "$0")/.."

RUN_NAME="${RUN_NAME:-ci-local}"

python src/training/train.py \
    --model gradient_boosting \
    --n-estimators 150 \
    --learning-rate 0.05 \
    --max-depth 3 \
    --min-samples-leaf 1 \
    --horizon-hours 24 \
    --seed 42 \
    --run-name "${RUN_NAME}" \
    ${TRACKING_URI:+--tracking-uri "${TRACKING_URI}"}