for system in arcstd archybrid;
do
    for l2 in 1e-5 1e-6;
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
            --beam_size 8 \
            --max_iter 100 > ptb_sd.d15.${system}.singleton.static.${l2}.b8.log &
    done
done
