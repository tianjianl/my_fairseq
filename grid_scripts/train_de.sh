#!/bin/bash

#$ -N de_en_translation
#$ -j y -o iwslt14_de_en_weight_decay_0_new.log
#$ -M tli104@jhu.edu
#$ -m e
#$ -l ram_free=20G,mem_free=20G,gpu=1,hostname=c0*|c1[0123456789]
#$ -wd /home/tli104/my_fairseq
# Submit to GPU queue
#$ -q g.q

# Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
source /home/gqin2/scripts/acquire-gpu -n 1

conda activate wd
mkdir -p /export/c11/tli104/checkpoints_de/new

fairseq-train data/data-bin-de \
	        --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
		--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
		--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
		--dropout 0.3  --weight-decay 0.0 \
		--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
		--max-tokens 4096 \
		--eval-bleu \
		--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
		--eval-bleu-detok moses \
		--eval-bleu-remove-bpe \
		--eval-bleu-print-samples \
		--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
		--max-epoch 60 \
		--no-epoch-checkpoints \
		--save-dir /export/c11/tli104/checkpoints_de/new

