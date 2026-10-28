#!/bin/zsh
set -eu
cd /private/tmp/guppylm-dev-apply
exec uv run --offline --python 3.10 --with torch==2.14.1 --with tokenizers==0.23.3 --with numpy==2.2.6 python -u -m tools.aquarium.v52.train \
 --train runs/v52/contract-responsive-v2-reviewed/train_A.json \
 --development runs/v52/contract-responsive-v2-reviewed/development.json \
 --historical runs/v52/contract-responsive-v2-reviewed/historical.json \
 --data /Users/eeaglstun/Documents/dev/guppylm-exp-aquarium/data_aq \
 --init /Users/eeaglstun/Documents/dev/guppylm-exp-aquarium/runs/aq-td/best.pt \
 --out runs/v52/full-responsive-v2-A --steps 15000 --eval-every 1000 --seed 42 --device mps
