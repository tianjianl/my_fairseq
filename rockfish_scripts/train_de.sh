#!/bin/bash
#SBATCH -A danielk_gpu
#SBATCH --partition a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="iwslt14-de-en"
#SBATCH --output="iwslt14_de_en_log.txt"
#SBATCH --mem=20G
#SBATCH --mail-user=tli104@jhu.edu

module load anaconda
conda activate fairseqenv

rm -r /scratch4/cs601/tli104/iwslt/de
mkdir -p /scratch4/cs601/tli104/iwslt/de

srun fairseq-train data/data-bin-de \
	        --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
		--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
		--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
		--dropout 0.3  --weight-decay 0.0001 \
		--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
		--max-tokens 4096 \
		--eval-bleu \
		--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
		--eval-bleu-detok moses \
		--eval-bleu-remove-bpe \
		--eval-bleu-print-samples \
		--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
		--max-update 50000 \
		--no-epoch-checkpoints \
		--save-dir /scratch4/cs601/tli104/iwslt/de

