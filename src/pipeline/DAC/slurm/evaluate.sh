#!/bin/bash
#SBATCH -J dzrwang_evaluate
#SBATCH -o ./slurm/evaluate.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:geforce_rtx_2080_ti:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 6:00:00

source /home/dzrwang/.bashrc
conda activate llm_table
set -e

# for MODEL in Llama3-chat Deepseek-Coder-chat gpt-3.5-turbo; do
#         for SCALE in - 6.7b 8b 33b 70b; do
#                 for PART in initialize debug; do
#                         for DATASET in Spider Bird KaggleDBQA; do
#                                 if [ ! -e "./result/$DATASET/$MODEL/$SCALE/$PART/dev.sql" ]; then
#                                         continue
#                                 fi
#                                 if [ -f "./result/$DATASET/$MODEL/$SCALE/$PART/dev.eval.json" ]; then
#                                         continue
#                                 fi
#                                 echo "Evaluating ./result/$DATASET/$MODEL/$SCALE/$PART/dev.sql"

#                                 if [ "$DATASET" = "Bird" ]; then
#                                         python3 -u ./evaluate/src/Bird/evaluation.py \
#                                                 --db_root_path ./dataset/$DATASET/database/ \
#                                                 --predicted_sql_path ./result/$DATASET/$MODEL/$SCALE/$PART/dev.sql \
#                                                 --data_mode dev \
#                                                 --ground_truth_path ./dataset/$DATASET/dev.sql \
#                                                 --diff_json_path ./result/$DATASET/$MODEL/$SCALE/$PART/dev.json \
#                                                 --dump_file ./result/$DATASET/$MODEL/$SCALE/$PART/dev.eval.json
#                                 elif [ "$DATASET" = "Spider" ] || [ "$DATASET" = "KaggleDBQA" ]; then
#                                         python3 ./evaluate/src/Spider/evaluation.py \
#                                                 --gold ./dataset/$DATASET/dev.sql \
#                                                 --pred ./result/$DATASET/$MODEL/$SCALE/$PART/dev.sql \
#                                                 --db ./dataset/$DATASET/database/ \
#                                                 --etype exec \
#                                                 --plug_value \
#                                                 --evaluate_results_file ./result/$DATASET/$MODEL/$SCALE/$PART/dev.eval.json
#                                 else
#                                         echo "Error: unknown dataset $DATASET"
#                                 fi
#                         done
#                 done
#         done
# done

MODEL=Llama3-chat
PART=analyze/temperature
SCALE=8b
for DATASET in Spider Bird KaggleDBQA; do
        for TEMPERATURE in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
                if [ ! -e "./result/$DATASET/$MODEL/$SCALE/$PART/dev.$TEMPERATURE.sql" ]; then
                        echo "File not found: ./result/$DATASET/$MODEL/$SCALE/$PART/dev.$TEMPERATURE.sql"
                        continue
                fi
                if [ -f "./result/$DATASET/$MODEL/$SCALE/$PART/dev.$TEMPERATURE.eval.json" ]; then
                        echo "File already exists: ./result/$DATASET/$MODEL/$SCALE/$PART/dev.$TEMPERATURE.eval.json"
                        continue
                fi
                echo "Evaluating ./result/$DATASET/$MODEL/$SCALE/$PART/dev.$TEMPERATURE.sql"

                if [ "$DATASET" = "Bird" ]; then
                        python3 -u ./evaluate/src/Bird/evaluation.py \
                                --db_root_path ./dataset/$DATASET/database/ \
                                --predicted_sql_path ./result/$DATASET/$MODEL/$SCALE/$PART/dev.$TEMPERATURE.sql \
                                --data_mode dev \
                                --ground_truth_path ./dataset/$DATASET/dev.sql \
                                --diff_json_path ./result/$DATASET/$MODEL/$SCALE/$PART/dev.$TEMPERATURE.json \
                                --dump_file ./result/$DATASET/$MODEL/$SCALE/$PART/dev.$TEMPERATURE.eval.json
                elif [ "$DATASET" = "Spider" ] || [ "$DATASET" = "KaggleDBQA" ]; then
                        python3 ./evaluate/src/Spider/evaluation.py \
                                --gold ./dataset/$DATASET/dev.sql \
                                --pred ./result/$DATASET/$MODEL/$SCALE/$PART/dev.$TEMPERATURE.sql \
                                --db ./dataset/$DATASET/database/ \
                                --etype exec \
                                --plug_value \
                                --evaluate_results_file ./result/$DATASET/$MODEL/$SCALE/$PART/dev.$TEMPERATURE.eval.json
                else
                        echo "Error: unknown dataset $DATASET"
                fi
        done
done
