# Gated Adapters for Multi-Domain Neural Machine Translation

This project provides modified fairseq codebase required to train the model introduced in the paper ["Gated Adapters for Multi-Domain Neural Machine Translation"](https://ebooks.iospress.nl/doi/10.3233/FAIA230404).

## Citation
```
@inproceedings{klimaszewski23,
  author       = {Klimaszewski, Mateusz and Belligoli, Zeno and Kumar, Satendra and Stergiadis, Emmanouil},
  title        = {Gated Adapters for Multi-Domain Neural Machine Translation},
  booktitle    = {{ECAI} 2023 - 26th European Conference on Artificial Intelligence},
  series       = {Frontiers in Artificial Intelligence and Applications},
  volume       = {372},
  pages        = {1264--1271},
  publisher    = {{IOS} Press},
  year         = {2023},
  url          = {https://doi.org/10.3233/FAIA230404},
  doi          = {10.3233/FAIA230404},
}
```

## Installation

```bash
conda create -n gated python=3.8.5
conda activate gated
conda install -y pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=10.2 -c pytorch
pip install -e .
python3 -m pip install tensorboardX tensorboard sacrebleu==2.2.0 transformers==4.21.1
```

## Data pre-processing
The fairseq binarisation directory should contain additional domain tags files:

- `train.domain.$SRC-$TGT`
- `valid.domain.$SRC-$TGT`
- `test.domain.$SRC-$TGT`

Each line of a file should contain space separated values that represent a probability of a sentence belonging to a specific-domain.

By default, the setup expects 6 values (see `--adapter-names` config for default value and the order of domains).

### Teacher training
To estimate the above probabilities, we trained a `roberta-large` using model trained via [transformers library](https://github.com/huggingface/transformers) [run_glue.py script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py).

```bash
python run_glue.py \
  --model_name_or_path roberta-large \
  --train_file train.classifier.json \
  --validation_file valid.classifier.json \
  --output_dir classifier_output \
  --learning_rate 1e-6 \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --save_steps 250 \
  --pad_to_max_length False \
  --warmup_steps 100 \
  --weight_decay 0.01 \
  --logging_steps 250 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --seed 1234 \
  --data_seed 1234 \
  --dataloader_num_workers 8 \
  --metric_for_best_model "eval_accuracy" \
  --load_best_model_at_end True

```

## Training

### Gated Adapters model

The training scripts expects:
- binarised data in `bin` directory with additional domain tags files (see `Data pre-processing` section)
- a starting checkpoint trained on the mixture of data (denoted as `./models/mix/checkpoint_best.pt`)

```bash
fairseq-train \
    "bin" \
    --tensorboard-logdir "tensorboard_logs" \
    --finetune-from-model "./models/mix/checkpoint_best.pt" \
    --ddp-backend no_c10d \
    --log-file "logs" \
    --seed 7272 \
    --bottleneck 2 \
    --save-dir "output" \
    --arch transformer_gated_adapters_mean \
    --load-domains \
    --share-all-embeddings \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --dropout 0.3 \
    --max-epoch 60 \
    --patience 4 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy_with_gates \
    --gate-loss-type "kl_div" \
    --gate-loss-scaling 0.5 \
    --label-smoothing 0.1 \
    --update-freq 4 \
    --max-tokens 1792 \
    --all-gather-list-size 20000 \
    --num-workers 16 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok space \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric
```

## Inference
The inference step requires fairseq binarisation of the data stored in a `bin` directory (however, without the domain weights' files).
```bash
fairseq-generate "bin" \
--path "checkpoint.pt" \
--batch-size 64 \
--beam 5 \
--max-len-a 1.2 \
--max-len-b 10 \
--remove-bpe sentencepiece \
--results-path "output"
```
