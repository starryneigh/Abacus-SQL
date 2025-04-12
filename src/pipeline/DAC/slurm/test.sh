#!/bin/bash
#SBATCH -J test
#SBATCH -o ./slurm/test.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:a100-sxm4-80gb:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 6:00:00
#SBATCH --mem 256g

source /home/dzrwang/.bashrc
conda activate llm_table

export VLLM_WORKER_MULTIPROC_METHOD=spawn

PART=dev
SHOT=5
TRAIN_DATA="--train_data_file ./result/Spider/train.json --train_schema_file ./dataset/Spider/tables.json --train_database_path ./dataset/Spider/database"
for MODEL in Llama3-chat; do
    for SCALE in 8b; do
        for DATASET in Spider Bird KaggleDBQA; do
            for TEMPERATURE in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
                if [[ "$MODEL" != "gpt-3.5-turbo" || "$SCALE" != "-" ]] && [ ! -d "./model/$MODEL/$SCALE" ]; then
                    continue
                fi
                if [ "$MODEL" == "gpt-3.5-turbo" ]; then
                    MODEL_NAME_OR_PATH=gpt-3.5-turbo
                else
                    MODEL_NAME_OR_PATH=./model/$MODEL/$SCALE
                fi

                if [ ! -f "./result/$DATASET/$MODEL/$SCALE/analyze/temperature/$PART.$TEMPERATURE.json" ]; then
                    python3 debug.py \
                        --llm_name_or_path $MODEL_NAME_OR_PATH \
                        --config_file ./config/$MODEL.json \
                        --data_file ./result/$DATASET/$MODEL/$SCALE/generate/$PART.json \
                        --schema_file ./dataset/$DATASET/tables.json \
                        --database_path ./dataset/$DATASET/database \
                        --dump_file ./result/$DATASET/$MODEL/$SCALE/analyze/temperature/$PART.$TEMPERATURE.json \
                        --temperature $TEMPERATURE
                fi
            done
        done
    done
done
