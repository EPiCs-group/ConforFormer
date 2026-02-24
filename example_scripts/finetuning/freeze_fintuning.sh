#!/bin/bash

# start of script
n_gpu=1
unimol_dir=path/to/unimol/source
data_path=path/to/your/data
start_weight_path=path/to/model/weights
collect_results_in=path/for/output

dict_name=path_to_your_dict
only_polar=0
conf_size=11
seed=42

task_names=("bace" "bbbp" "clintox" "esol" "freesolv" "hiv" "lipo" "muv" "qm7dft" "qm8dft" "qm9dft" "sider" "tox21" "toxcast")


# task_names=("esol" "freesolv" "lipo" "qm7dft" "qm8dft" "qm9dft") # only non classification tasks

for task_name in "${task_names[@]}"; do
    if [ "$task_name" == "bace" ]; then
        task_num=2
        lr=1e-4
        batch_size=64
        update_freq=1 # total 64
        epoch=60
        dropout=0.1
        warmup=0.06

        metric="valid_agg_auc"
        loss_func=finetune_cross_entropy
        max_best=true

    elif [ "$task_name" == "bbbp" ]; then
        task_num=2
        lr=4e-4
        batch_size=128
        update_freq=1 # total 128
        epoch=40
        dropout=0.0
        warmup=0.06

        metric="valid_agg_auc"
        loss_func=finetune_cross_entropy
        max_best=true
        
    elif [ "$task_name" == "clintox" ]; then
        task_num=2
        lr=5e-5
        batch_size=64
        update_freq=4 # total 256
        epoch=100
        dropout=0.5
        warmup=0.1

        metric="valid_agg_auc"
        loss_func=multi_task_BCE
        max_best=true

    elif [ "$task_name" == "esol" ]; then
        task_num=1
        lr=5e-4
        batch_size=256
        update_freq=1 # total 256
        epoch=100
        dropout=0.2
        warmup=0.06
        
        metric="valid_agg_rmse"
        loss_func=finetune_mse
	max_best=false
        
    elif [ "$task_name" == "freesolv" ]; then
        task_num=1
        lr=8e-5
        batch_size=64
        update_freq=1 # total 64
        epoch=60
        dropout=0.2
        warmup=0.1
        
        metric="valid_agg_rmse"
        loss_func=finetune_mse
	max_best=false

    elif [ "$task_name" == "hiv" ]; then
        task_num=2
        lr=5e-5
        batch_size=64
        update_freq=4 # total 256
        epoch=5
        dropout=0.2
        warmup=0.1
        
        metric="valid_agg_auc"
        loss_func=finetune_cross_entropy
        max_best=true
        
    elif [ "$task_name" == "lipo" ]; then
        task_num=1
        lr=1e-4
        batch_size=32
        update_freq=1 # total 32
        epoch=80
        dropout=0.1
        warmup=0.06

        metric="valid_agg_rmse"
        loss_func=finetune_mse
	max_best=false

    elif [ "$task_name" == "muv" ]; then
        task_num=17
        lr=2e-5
        batch_size=128
        update_freq=1 # total 128
        epoch=40
        dropout=0.0
        warmup=0.0

        metric="valid_agg_auc"
        loss_func=multi_task_BCE
        max_best=true
        
    elif [ "$task_name" == "pcba" ]; then
        task_num=128
        lr=1e-4
        batch_size=32
        update_freq=4 # total 128
        epoch=40
        dropout=0.1
        warmup=0.06

        metric="valid_agg_auc"
        loss_func=multi_task_BCE
        max_best=true
        
    elif [ "$task_name" == "qm7dft" ]; then
        task_num=1
        lr=3e-4
        batch_size=32
        update_freq=1 # total 32
        epoch=100
        dropout=0.0
        warmup=0.06
        
        metric="valid_agg_mae"
        loss_func=finetune_smooth_mae
	max_best=false

    elif [ "$task_name" == "qm8dft" ]; then
        task_num=12
        lr=1e-4
        batch_size=32
        update_freq=1 # total 32
        epoch=40
        dropout=0.0
        warmup=0.06

        metric="valid_agg_mae"
        loss_func=finetune_smooth_mae
	max_best=false
        
    elif [ "$task_name" == "qm9dft" ]; then
        task_num=3
        lr=1e-4
        batch_size=128
        update_freq=1 # total 128
        epoch=40
        dropout=0.0
        warmup=0.06

        metric="valid_agg_mae"
        loss_func=finetune_smooth_mae
	max_best=false
        
    elif [ "$task_name" == "sider" ]; then
        task_num=27
        lr=5e-4
        batch_size=32
        update_freq=1 # total 32
        epoch=80
        dropout=0.0
        warmup=0.1

        metric="valid_agg_auc"
        loss_func=multi_task_BCE
        max_best=true
        
    elif [ "$task_name" == "tox21" ]; then
        task_num=12
        lr=1e-4
        batch_size=128
        update_freq=1 # total 128
        epoch=80
        dropout=0.1
        warmup=0.06

        metric="valid_agg_auc"
        loss_func=multi_task_BCE
        max_best=true
        
    else # This is for toxcast
        task_num=617
        lr=1e-4
        batch_size=64
        update_freq=1 # total 64
        epoch=80
        dropout=0.1
        warmup=0.06

        metric="valid_agg_auc"
        loss_func=multi_task_BCE
        max_best=true
    fi
    
    if [ "$max_best" = true ]; then
        maximize_flag="--maximize-best-checkpoint-metric"
    else
        maximize_flag=""
    fi
    
    echo "Now running $task_name"
    save_dir="${collect_results_in}/${task_name}"
    torchrun --standalone --nnodes=1 --nproc_per_node=$n_gpu $(which unicore-train) $data_path --user-dir $unimol_dir \
             --train-subset train --valid-subset valid,test --task-name $task_name \
             --conf-size $conf_size \
             --num-workers 8 --ddp-backend=c10d \
             --dict-name $dict_name \
             --task mol_finetune --loss $loss_func --arch unimol_base \
             --finetune-from-model $start_weight_path \
             --classification-head-name $task_name --num-classes $task_num \
             --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 \
             --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --pooler-dropout $dropout \
             --update-freq $update_freq --batch-size $batch_size --seed $seed\
             --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb_$task_name \
             --validate-interval 1 --save-interval 1 --keep-last-epochs 3 \
             --log-interval 10 --log-format simple \
             --save-dir $save_dir  --only-polar $only_polar \
             --best-checkpoint-metric $metric --patience 20 \
             --keep-interval-updates 1 \
             $maximize_flag --freeze-layers \
             # the additional flags --unfreeze-last-layer or --unfreeze-last-3-layers can also be added.
             
done

# end of script

