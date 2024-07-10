#!/usr/bin/bash

INPATH=$1
OUTPATH=$2

for d in $INPATH/*/; do
    echo "Processing $d..."
    python wav2midi/dataset.py $d ./data
done
