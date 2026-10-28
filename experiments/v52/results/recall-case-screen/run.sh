#!/bin/zsh
set -eu
cd /private/tmp/guppylm-dev-apply
ROOT=runs/v52/recall-case-screen
AQ=/Users/eeaglstun/Documents/dev/guppylm-exp-aquarium
py() { uv run --offline --python 3.10 --with torch==2.14.1 --with tokenizers==0.23.3 --with numpy==2.2.6 python -u "$@"; }
for arm in recall paired; do
 py -m tools.aquarium.v52.train --train "$ROOT/$arm/train_A.json" --development "$ROOT/$arm/development.json" --historical "$ROOT/$arm/historical.json" --data "$AQ/data_aq" --init runs/v52/full-responsive-v2-A/best.pt --out "$ROOT/$arm/model" --steps 600 --warmup 60 --lr .00005 --eval-every 300 --source-share recall_screen=.05 --seed 42 --device mps > "$ROOT/$arm/train.log" 2>&1
 echo "TRAINED $arm"
done
for arm in starting recall paired; do
 if [[ "$arm" == starting ]]; then CK=runs/v52/full-responsive-v2-A/best.pt; else CK="$ROOT/$arm/model/final.pt"; fi
 py -m experiments.v52.results.recall-case-screen.evaluate --checkpoint "$CK" --out "$ROOT/$arm-validation.json" > "$ROOT/$arm-validation.log" 2>&1
 echo "VALIDATED $arm"
done
for arm in recall paired; do
 py -m tools.aquarium.v52.evaluate --contract "$ROOT/$arm/contract.json" --checkpoint "$ROOT/$arm/model/final.pt" --tokenizer "$AQ/data_aq/tokenizer.json" --out "$ROOT/$arm-development.json" --device cpu --threads 2 > "$ROOT/$arm-development.log" 2>&1
 echo "DEVELOPMENT $arm"
done
