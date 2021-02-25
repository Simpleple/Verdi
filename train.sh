rm checkpoints_sent/checkpoint_*
python train_qe.py \
	"qe_data_dir" \
	--ddp-backend=no_c10d \
	--save-dir checkpoints_sent \
	--restore-estimator-file checkpoint_best.pt \
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
	--restore-file "NMT_predictor_checkpoint" \
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
	--valid-subset dev \
	--xml-model-path "XLM_checkpoint" \
	--loss-combine=0.5  \
	--xml-tgt-only=1 \
	--topk-time-step=3 \
        --seed 1113
