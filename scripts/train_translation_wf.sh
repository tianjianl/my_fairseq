#!/bin/bash
#SBATCH -A danielk80_gpu
#SBATCH --partition ica100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name="iwslt"
#SBATCH --output="iwslt14_%j.txt"
#SBATCH --mem=20G
#SBATCH --mail-user=tli104@jhu.edu
#SBATCH --mail-type=ALL

module load anaconda
conda activate fairseqenv


echo "Direction $1 to en"
lang=$1

for num in 4000 8000 12000
do 
	rm -r /scratch4/cs601/tli104/wf_$num/checkpoints_$1
	mkdir -p /scratch4/cs601/tli104/wf_$num/checkpoints_$1
done

fairseq-train data/data-bin-$1 \
	        --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
		--task translation_intra_distillation \
		--weighted-freezing \
		--start-freezing 4000 \
		--smooth-scores \
		--importance-metric "magnitude" \
		--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
		--lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
		--dropout 0.3  --weight-decay 0.0001 --attention-dropout 0.1 \
		--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
		--max-tokens 4096 \
		--eval-bleu \
		--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
		--eval-bleu-detok moses \
		--eval-bleu-remove-bpe \
		--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
		--max-update 20000 \
		--no-epoch-checkpoints \
		--save-dir /scratch4/cs601/tli104/wf_4000/checkpoints_$1


fairseq-train data/data-bin-$1 \
	        --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
		--task translation_intra_distillation \
		--weighted-freezing \
		--start-freezing 6000 \
		--smooth-scores \
		--importance-metric "magnitude" \
		--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
		--lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
		--dropout 0.3  --weight-decay 0.0001 --attention-dropout 0.1 \
		--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
		--max-tokens 4096 \
		--eval-bleu \
		--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
		--eval-bleu-detok moses \
		--eval-bleu-remove-bpe \
		--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
		--max-update 20000 \
		--no-epoch-checkpoints \
		--save-dir /scratch4/cs601/tli104/wf_6000/checkpoints_$1

fairseq-train data/data-bin-$1 \
	        --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
		--task translation_intra_distillation \
		--weighted-freezing \
		--start-freezing 8000 \
		--smooth-scores \
		--importance-metric "magnitude" \
		--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
		--lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
		--dropout 0.3  --weight-decay 0.0001 --attention-dropout 0.1 \
		--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
		--max-tokens 4096 \
		--eval-bleu \
		--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
		--eval-bleu-detok moses \
		--eval-bleu-remove-bpe \
		--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
		--max-update 20000 \
		--no-epoch-checkpoints \
		--save-dir /scratch4/cs601/tli104/wf_8000/checkpoints_$1
exit 1

mkdir -p /root/lp/checkpoints_$1
fairseq-train data/data-bin-$1 \
	        --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
		--task translation_intra_distillation \
		--weighted-freezing \
		--importance-metric "loss-perserving" \
		--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
		--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
		--dropout 0.3  --weight-decay 0.0001 --attention-dropout 0.1 \
		--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
		--max-tokens 4096 \
		--eval-bleu \
		--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
		--eval-bleu-detok moses \
		--eval-bleu-remove-bpe \
		--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
		--max-epoch 20 \
		--no-epoch-checkpoints \
		--save-dir /root/lp/checkpoints_$1

mkdir -p /root/fisher/checkpoints_$1
fairseq-train data/data-bin-$1 \
	        --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
		--task translation_intra_distillation \
		--weighted-freezing \
		--importance-metric "fisher" \
		--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
		--lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
		--dropout 0.3  --weight-decay 0.0001 --attention-dropout 0.1 \
		--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
		--max-tokens 4096 \
		--eval-bleu \
		--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
		--eval-bleu-detok moses \
		--eval-bleu-remove-bpe \
		--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
		--max-epoch 20 \
		--no-epoch-checkpoints \
		--save-dir /root/fisher/checkpoints_$1

