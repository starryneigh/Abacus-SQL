#!/bin/bash
#SBATCH -J generate
#SBATCH -o ./DAC/logs/generate.out
#SBATCH -e ./DAC/logs/generate.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00
#SBATCH --mem 64G

# export VLLM_WORKER_MULTIPROC_METHOD=spawn

PART=test
SHOT=5
TRAIN_DATA="--train_data_file ./DAC/result/Spider/train.json --train_schema_file ./dataset/Spider/tables.json --train_database_path ./dataset/Spider/database"
for MODEL in Qwen2.5-Coder; do
    for SCALE in 7b; do
        for DATASET in Spider; do
            if [[ "$MODEL" != "gpt-3.5-turbo" || "$SCALE" != "-" ]] && [ ! -d "./model/$MODEL/$SCALE" ]; then
                continue
            fi
            if [ "$MODEL" == "gpt-3.5-turbo" ]; then
                MODEL_NAME_OR_PATH=gpt-3.5-turbo
            else
                MODEL_NAME_OR_PATH=./model/$MODEL/$SCALE
            fi

            mkdir -p ./DAC/result/$DATASET/$MODEL/$SCALE/initialize
            if [ ! -f "./DAC/result/$DATASET/$MODEL/$SCALE/initialize/$PART.json" ]; then
                python3 ./DAC/generate.py \
                    --llm_name_or_path $MODEL_NAME_OR_PATH \
                    --config_file ./config/$MODEL.json \
                    $TRAIN_DATA \
                    --dev_data_file ./DAC/result/$DATASET/$PART.json \
                    --dev_schema_file ./dataset/$DATASET/tables.json \
                    --dev_database_path ./dataset/$DATASET/database \
                    --dump_file ./DAC/result/$DATASET/$MODEL/$SCALE/initialize/$PART.json \
                    --shot $SHOT
            fi

            mkdir -p ./DAC/result/$DATASET/$MODEL/$SCALE/align
            if [ ! -f "./DAC/result/$DATASET/$MODEL/$SCALE/align/$PART.json" ]; then
                python3 ./DAC/align.py \
                    --llm_name_or_path $MODEL_NAME_OR_PATH \
                    --config_file ./config/$MODEL.json \
                    $TRAIN_DATA \
                    --dev_data_file ./DAC/result/$DATASET/$MODEL/$SCALE/initialize/$PART.json \
                    --dev_schema_file ./dataset/$DATASET/tables.json \
                    --dev_database_path ./dataset/$DATASET/database \
                    --dump_file ./DAC/result/$DATASET/$MODEL/$SCALE/align/$PART.json \
                    --shot $SHOT
            fi

            mkdir -p ./DAC/result/$DATASET/$MODEL/$SCALE/hallucinate
            if [ ! -f "./DAC/result/$DATASET/$MODEL/$SCALE/hallucinate/$PART.json" ]; then
                python3 ./DAC/hallucinate.py \
                    --llm_name_or_path $MODEL_NAME_OR_PATH \
                    --config_file ./config/$MODEL.json \
                    $TRAIN_DATA \
                    --dev_data_file ./DAC/result/$DATASET/$MODEL/$SCALE/align/$PART.json \
                    --dev_schema_file ./dataset/$DATASET/tables.json \
                    --dev_database_path ./dataset/$DATASET/database \
                    --dump_file ./DAC/result/$DATASET/$MODEL/$SCALE/hallucinate/$PART.json \
                    --shot $SHOT
            fi

            mkdir -p ./DAC/result/$DATASET/$MODEL/$SCALE/generate
            if [ ! -f "./DAC/result/$DATASET/$MODEL/$SCALE/generate/$PART.json" ]; then
                python3 ./DAC/generate.py \
                    --llm_name_or_path $MODEL_NAME_OR_PATH \
                    --config_file ./config/$MODEL.json \
                    $TRAIN_DATA \
                    --dev_data_file ./DAC/result/$DATASET/$MODEL/$SCALE/hallucinate/$PART.json \
                    --dev_schema_file ./dataset/$DATASET/tables.json \
                    --dev_database_path ./dataset/$DATASET/database \
                    --dump_file ./DAC/result/$DATASET/$MODEL/$SCALE/generate/$PART.json \
                    --shot $SHOT \
                    --aligned
            fi

            mkdir -p ./DAC/result/$DATASET/$MODEL/$SCALE/debug
            if [ ! -f "./DAC/result/$DATASET/$MODEL/$SCALE/debug/$PART.json" ]; then
                python3 ./DAC/debug.py \
                    --llm_name_or_path $MODEL_NAME_OR_PATH \
                    --config_file ./config/$MODEL.json \
                    --data_file ./DAC/result/$DATASET/$MODEL/$SCALE/generate/$PART.json \
                    --schema_file ./dataset/$DATASET/tables.json \
                    --database_path ./dataset/$DATASET/database \
                    --dump_file ./DAC/result/$DATASET/$MODEL/$SCALE/debug/$PART.json \
                    --temperature 0.8
            fi
        done
    done
done
