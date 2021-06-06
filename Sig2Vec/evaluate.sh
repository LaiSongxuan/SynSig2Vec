#!/usr/bin/env sh
for seed in $(seq 111 111 555)
do
    echo $seed
    python evaluate_stylus.py --epoch End --seed $seed
    # python evaluate_finger.py --epoch End --seed $seed
done

