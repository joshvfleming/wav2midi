#!/usr/bin/bash

INPATH=$1
OUTPATH=$2

for d in $INPATH/*/; do
    echo "Processing $d..."
    python process_dataset.py $d $OUTPATH
done
