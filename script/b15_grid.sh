#!/bin/bash

for system in arcstd archybrid;
do
    for oracle in static dynamic;
    do
        for l2 in 1e-5 1e-6;
        do
            nohup ./bin/trans_parser --dynet-mem 1024 \
                --dynet-seed 1234 \
                --train \
                --architecture b15 \
                --system ${system} \
                -T ./data/PTB_train_auto.conll \
                -d ./data/PTB_development_auto.conll \
                -w ./data/sskip.100.vectors.ptb_filtered \
                --lambda ${l2} \
                --optimizer_enable_eta_decay true \
                --optimizer_enable_clipping true \
                --external_eval ./script/eval_ex_enpunt.py \
                --supervised_oracle ${oracle} \
                --max_iter 30 > ptb_sd.b15.${system}.${oracle}.${l2}.log &
        done
    done
done
