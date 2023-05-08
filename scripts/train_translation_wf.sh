echo "Direction $1 to en"
lang=$1
mkdir -p /root/wf/checkpoints_$1
fairseq-train data/data-bin-$1 \
	        --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
		--task translation_intra_distillation \
		--weighted-freezing \
		--importance-metric "magnitude" \
		--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
		--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
		--dropout 0.3  --weight-decay 0.0001 --attention-dropout 0.1 \
		--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
		--max-tokens 4096 \
		--eval-bleu \
		--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
		--eval-bleu-detok moses \
		--eval-bleu-remove-bpe \
		--eval-bleu-print-samples \
		--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
		--max-epoch 20 \
		--no-epoch-checkpoints \
		--save-dir /root/wf/checkpoints_$1

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
		--eval-bleu-print-samples \
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
		--lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
		--dropout 0.3  --weight-decay 0.0001 --attention-dropout 0.1 \
		--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
		--max-tokens 4096 \
		--eval-bleu \
		--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
		--eval-bleu-detok moses \
		--eval-bleu-remove-bpe \
		--eval-bleu-print-samples \
		--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
		--max-epoch 20 \
		--no-epoch-checkpoints \
		--save-dir /root/fisher/checkpoints_$1

