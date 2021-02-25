# Verdi
Code for paper "Verdi: Quality Estimation and Error Detection for Bilingual Corpora" published in WWW'21


## Train Predictor 
```shell
python train.py \
     "train_data_path" \
     --ddp-backend=no_c10d \
     -s en \
     -t zh \
     --max-update 300000 \
     --task translation_moe_dual \
     --method hMoElp \
     --mean-pool-gating-network \
     --num-experts 5 \
     --arch transformerdualdecoder_wmt_en_de \
     --optimizer adam \
     --clip-norm 0.0 \
     --lr-scheduler inverse_sqrt \
     --warmup-init-lr 1e-07 \
     --warmup-updates 4000 \
     --lr 0.0007 \
     --min-lr 1e-09 \
     --max-tokens 4096 \
     --save-dir "checkpoint_dir" \
     --share-all-embeddings \
     --log-interval 200
```

## Quality Estimation Commands

Refer to [train.sh](train.sh) and [infer.sh](infer.sh) for sentence-level training and inference.

Refer to [train_word.sh](train_word.sh) and [infer_word.sh](infer_word.sh) for word-level training and inference.
