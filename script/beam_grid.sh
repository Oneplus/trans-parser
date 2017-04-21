#!/bin/bash
l2=1e-6
b=8
for system in arcstd archybrid;
do
    nohup ./bin/trans_parser --dynet-mem 4096 \
        --dynet-seed 1234 \
        --train \
        --architecture d15 \
        --system ${system} \
        -T ./data/PTB_train_auto.conll \
        -d ./data/PTB_development_auto.conll \
        -w ./data/sskip.100.vectors.ptb_filtered \
        --lambda ${l2} \
        --noisify_method singleton \
        --optimizer_enable_eta_decay true \
        --optimizer_enable_clipping true \
        --external_eval ./script/eval_ex_enpunt.py \
        --supervised_objective structure \
        --beam_size ${b} \
        --supervised_pretrain_iter 5 \
        --evaluate_skips 5 \
        --max_iter 50 > ptb_sd.d15.${system}.singleton.static.${l2}.b${b}.log &
done
