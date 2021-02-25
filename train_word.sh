ckpt_dir=checkpoints_word
rm ${ckpt_dir}/checkpoint_*
export CUDA_VISIBLE_DEVICES=0
python train_qe_word.py \
	--ddp-backend=no_c10d \
	--save-dir=${ckpt_dir} \
	--restore-estimator-file checkpoint_best.pt \
	../fairseq-qe/ccmt19_qe/data4 \
	-s en \
	-t zh \
	--ter hter \
	--max-update 3000 \
	--task translation_qe_word \
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
	--valid-subset dev \
	--xml-model-path ../fairseq-qe/xml_model/mlm_tlm_xnli15_1024.pth \
	--loss-combine=0.8 \
	--xml-tgt-only=1 \
	--topk-time-step=3 \
        --seed=1112 