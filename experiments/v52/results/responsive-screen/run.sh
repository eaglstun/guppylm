#!/bin/zsh
set -e
cd /private/tmp/guppylm-dev-apply
AQ=/Users/eeaglstun/Documents/dev/guppylm-exp-aquarium
for mode in uniform weighted; do
  extra=()
  if [[ "$mode" == weighted ]]; then
    extra=(--source-share responsive=.02)
  fi
  uv run --offline --python 3.10 --with torch==2.14.1 --with tokenizers==0.23.3 --with numpy==2.2.6 \
    python -m tools.aquarium.v52.train \
    --train runs/v52/contract-responsive-v1/train_A.json \
    --development runs/v52/contract-responsive-v1/development.json \
    --historical runs/v52/contract-responsive-v1/historical.json \
    --data "$AQ/data_aq" --init runs/v52/full-v3-talkie-uppercase-v2-vocabulary-A/best.pt \
    --out runs/v52/responsive-screen/$mode --steps 600 --warmup 60 --lr .00005 \
    --eval-every 300 --seed 42 --device mps "${extra[@]}" \
    > runs/v52/responsive-screen/$mode.log 2>&1
  echo "DONE $mode"
done
