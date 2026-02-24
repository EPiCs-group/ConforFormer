#!/bin/bash


# directories
unimol_dir=./path/to/unimol/source
data_path=./path/to/contrastive/benchmark
data_subset=./name/of/contrastive/benchmark
dict_name=name_of_dictionary

task_name=unimol_contrast
results_path=results
weight_path=./path/to/your/weights
only_polar=0

output_db_name=name_of_output_db

# scripts get_sims.py and get_embed.py are interchangeable with this execution
python get_sims.py --user-dir $unimol_dir $data_path --valid-subset $data_subset --valid-db-type lmdb \
       --path $weight_path --results-path $results_path --dict-name=$dict_name \
       --num-workers 12 --ddp-backend=c10d --batch-size 1 \
       --arch contrast --task $task_name \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --only-polar $only_polar --mode "infer" \
       --db-name $output_db_name

conda deactivate