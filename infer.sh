export CUDA_VISIBLE_DEVICES=0
CUDA_LAUNCH_BLOCKING=1 python train_qe.py \
        --ddp-backend=no_c10d \
        --save-dir checkpoints_test \
        --restore-estimator-file checkpoint_3_550.pt \
        ../fairseq-qe/ccmt19_qe/data4/ \
        -s en \
        -t zh \
        --ter hter \
        --max-update 3000 \
        --task translation_qe \
        --method hMoElp \
        --mean-pool-gating-network \
        --num-experts 5 \
        --arch transformerdualdecoder_wmt_en_de \
        --optimizer adam \
        --clip-norm 0.0 \
        --lr-scheduler inverse_sqrt \
        --warmup-init-lr 1e-07 \
        --warmup-updates 400 \
        --lr 0.0002 \
        --min-lr 1e-09 \
        --max-tokens 1024 \
        --save-interval-updates 1 \
        --restore-file ../../fairseq-qe/checkpoints/dual/checkpoint15.pt \
        --reset-optimizer \
        --reset-dataloader \
        --raw-text \
        --share-all-embeddings \
        --maximize-best-checkpoint-metric \
        --keep-interval-updates 2 \
        --estimator-xml-dim 10260 \
        --estimator-transformer-dim 10260 \
        --share-estimator 1 \
        --estimator-xml-only 0 \
        --estimator-hidden-dim 512 \
        --valid-subset test \
        --xml-model-path ../fairseq-qe/xml_model/mlm_tlm_xnli15_1024.pth \
        --loss-combine=0.5  \
        --xml-tgt-only=0 \
        --topk-time-step=3 \
        --seed 1234 \
        --evaluate 1 \
        --num-workers 0