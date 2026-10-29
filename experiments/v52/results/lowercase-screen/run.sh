#!/bin/zsh
set -eu
cd /private/tmp/guppylm-dev-apply
ROOT=runs/v52/lowercase-screen
DATA=runs/v52/contract-lowercase-v1-reviewed
py() { uv run --offline --python 3.10 --with torch==2.14.1 --with tokenizers==0.23.3 --with numpy==2.2.6 python -u "$@"; }
py -m tools.aquarium.v52.train --train "$DATA/train_A.json" --development "$DATA/development.json" --historical "$DATA/historical.json" --data "$DATA/data" --init runs/v52/full-v3-talkie-uppercase-v2-vocabulary-A/best.pt --out "$ROOT/model" --steps 600 --warmup 60 --lr .00005 --eval-every 300 --seed 42 --device mps > "$ROOT/train.log" 2>&1
echo TRAINED
py -m experiments.v52.results.lowercase-screen.conversations > "$ROOT/conversations.log" 2>&1
echo REVIEW_READY
