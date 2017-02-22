#!/bin/bash

for system in arcstd archybrid;
do
    for noise in none singleton word;
    do
        for oracle in static dynamic;
        do
            for l2 in 1e-5 1e-6;
            do
                nohup ./bin/trans_parser --dynet-mem 1024 \
                    --dynet-seed 1234 \
                    --train \
                    --architecture d15 \
                    --system ${system} \
                    -T ./data/PTB_train_auto.conll \
                    -d ./data/PTB_development_auto.conll \
                    -w ./data/sskip.100.vectors.ptb_filtered \
                    --lambda ${l2} \
                    --noisify_method ${noise} \
                    --optimizer_enable_eta_decay true \
                    --optimizer_enable_clipping true \
                    --external_eval ./script/eval_ex_enpunt.py \
                    --max_iter 30 > ptb_sd.${system}.${noise}.${oracle}.${l2}.log &
            done
        done
    done
done
