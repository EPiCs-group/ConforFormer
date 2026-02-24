#!/bin/bash

# start of script
unimol_path=path/to/unimol/source
data_path=path/to/your/data
save_dir=path/for/saving/results

n_gpu=1
MASTER_PORT=10086

lr=5e-4
warmup_init_lr=1e-6
lr_patience=3
lr_threshold=0.25

wd=1e-4
batch_size=128
update_freq=1

contrastive_loss=2
contrast_temperature=0.07

masked_token_loss=1
masked_coord_loss=5
masked_dist_loss=10
x_norm_loss=0.01
delta_pair_repr_norm_loss=0.01
mask_prob=0.15
only_polar=0
noise_type="uniform"
noise=1.0
seed=0
warmup_steps=5000
max_steps=2000000

# model architectures
num_layers=15
attn_heads=64
ffn_dropout=0.1
attn_dropout=0.1
embedding_dropout=0.1
embed_dim=512
fnn_hidden_dim=2048
g_kernel_channels=128
activation_fn=gelu
mode=train
dropout=0.1

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
torchrun --standalone --nnodes=1 --nproc_per_node=$n_gpu $(which unicore-train) $data_path  --user-dir $unimol_path \
       --train-subset train --train-db-type lmdb --valid-subset valid --valid-db-type lmdb --dict-name dict_omol_full.txt \
       --num-workers 12 --ddp-backend=c10d \
       --task unimol_contrast_head --loss unimol_contrast_head --arch unimol_contrast  \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
       --lr-scheduler reduce_lr_on_plateau --lr $lr --warmup-updates $warmup_steps --warmup-init-lr $warmup_init_lr --lr-patience $lr_patience --lr-threshold $lr_threshold --lr-shrink 0.5 \
       --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
       --max-update $max_steps --log-interval 1 --log-format simple \
       --save-interval-updates 5000 --validate-interval-updates 5000 --keep-interval-updates 2 --no-epoch-checkpoints  \
       --masked-token-loss $masked_token_loss --masked-coord-loss $masked_coord_loss --masked-dist-loss $masked_dist_loss \
       --x-norm-loss $x_norm_loss --delta-pair-repr-norm-loss $delta_pair_repr_norm_loss \
       --contrastive-loss $contrastive_loss --contrast-temperature $contrast_temperature \
       --mask-prob $mask_prob --noise-type $noise_type --noise $noise --batch-size $batch_size --batch-size-valid $batch_size --batch-size-valid $batch_size \
       --save-dir $save_dir  --only-polar $only_polar \
       --encoder-layers $num_layers --encoder-embed-dim $embed_dim --encoder-ffn-embed-dim $fnn_hidden_dim --encoder-attention-heads $attn_heads \
       --activation-fn $activation_fn --pooler-activation-fn $activation_fn \
       --emb-dropout $dropout --dropout $dropout --attention-dropout $dropout \
       --mode $mode
