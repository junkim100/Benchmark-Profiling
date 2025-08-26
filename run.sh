# Benchmark Profiling pipeline: preprocess → train → extract → damage → evaluate
# Author: Dongjun Kim
VERSION=v1.1
EVAL_ONLY=false

# Process flags first
while getopts "e" opt; do
    case $opt in
        e)  # Fixed from 'eval' to 'e'
            EVAL_ONLY=true
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

# Shift processed options out
shift $((OPTIND - 1))

# Now handle output directory
SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
OUTPUT_DIR=${1:-$SCRIPT_DIR/outputs}
if [ -z $OUTPUT_DIR ]; then
    OUTPUT_DIR=$SCRIPT_DIR/outputs
fi
# Remove trailing slash if present
OUTPUT_DIR=$(echo $OUTPUT_DIR | sed 's:/*$::')
BASE_CHECKPOINT_PATH=$OUTPUT_DIR/$VERSION
BASE_DATASET_PATH=$SCRIPT_DIR/data_preprocess/datasets

touch $BASE_CHECKPOINT_PATH/failed.log
touch $BASE_DATASET_PATH/failed.log

# Function for enhanced logging
log() {
    local message=$1

    # Use ANSI escape codes for color (e.g., green)
    echo "===================================================================================================="
    echo -e "\033[1;32m[run.sh] $(date '+%Y-%m-%d %H:%M:%S')\n$message\033[0m"
    echo "===================================================================================================="

}

cat <<EOF

====================================================================================================

██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗███╗   ███╗ █████╗ ██████╗ ██╗  ██╗
██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║████╗ ████║██╔══██╗██╔══██╗██║ ██╔╝
██████╔╝█████╗  ██╔██╗ ██║██║     ███████║██╔████╔██║███████║██████╔╝█████╔╝
██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║██║╚██╔╝██║██╔══██║██╔══██╗██╔═██╗
██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║██║ ╚═╝ ██║██║  ██║██║  ██║██║  ██╗
╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝

██████╗ ██████╗  ██████╗ ███████╗██╗██╗     ██╗███╗   ██╗ ██████╗
██╔══██╗██╔══██╗██╔═══██╗██╔════╝██║██║     ██║████╗  ██║██╔════╝
██████╔╝██████╔╝██║   ██║█████╗  ██║██║     ██║██╔██╗ ██║██║  ███╗
██╔═══╝ ██╔══██╗██║   ██║██╔══╝  ██║██║     ██║██║╚██╗██║██║   ██║
██║     ██║  ██║╚██████╔╝██║     ██║███████╗██║██║ ╚████║╚██████╔╝
╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝


Author: Dongjun Kim
Version: $VERSION
Script directory: $SCRIPT_DIR
Output directory: $OUTPUT_DIR
Evaluation Only Mode: $EVAL_ONLY

====================================================================================================

EOF
sleep 5

##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
############################################################################### Hyperparameter and dataset configurations ################################################################################
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################

CONFIG_FILE=config.yml

# Read config YAML using grep and sed to extract values directly
echo "Reading config file: $CONFIG_FILE"

# Extract k_values directly from config.yml
K_VALUES=($(grep -A 10 'k_values:' $CONFIG_FILE | grep -v '#' | grep -o '[0-9]\+\.[0-9]\+' | head -n 10))

CONFIG=$(python3 -c "
import yaml
import json
with open('$CONFIG_FILE', 'r') as file:
    config = yaml.safe_load(file)
print(json.dumps(config))
")

# Extract components from the JSON configuration using jq
declare -A MODEL_DIR_NAMES
while IFS=$'\t' read -r MODEL TOKENIZER; do
    MODEL_DIR_NAMES[$MODEL]=$TOKENIZER
done < <(echo $CONFIG | jq -r '.models[] | "\(.name)\t\(.tokenizer)"')

SETTINGS=$(echo $CONFIG | jq -c '.settings')
DATASETS=$(echo $CONFIG | jq -c '.datasets')
EVALS=$(echo $CONFIG | jq -c '.evals')

CUDA_VISIBLE_DEVICES=$(echo $SETTINGS | jq -r '.cuda_visible_devices')
TOTAL_CARDS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
TRAIN_BATCH_SIZE=$(echo $SETTINGS | jq -r '.train_batch_size')
GRADIENT_ACCUMULATION_STEPS=$(echo $SETTINGS | jq -r '.gradient_accumulation_steps')
EVAL_BATCH_SIZE=$(echo $SETTINGS | jq -r '.eval_batch_size')
FRONT_REMOVAL=$(echo $SETTINGS | jq -r '.front_removal')
BACK_REMOVAL=$(echo $SETTINGS | jq -r '.back_removal')


BENCHMARKS=($(echo $EVALS | jq -r '.benchmarks[]'))
NUM_FEWSHOTS=($(echo $EVALS | jq -r '.num_fewshot[]'))

# Export CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false



for MODEL in ${!MODEL_DIR_NAMES[@]}; do
    TOKENIZER_NAME=${MODEL_DIR_NAMES[$MODEL]}
    TOKENIZER_PATH=$MODEL

    echo $DATASETS | jq -c '.[]' | while IFS= read -r dataset_entry; do
        DATASET=$(echo $dataset_entry | jq -r '.name')
        DATASET=$(echo $DATASET | sed 's:/*$::') # remove trailing slash
        HF_PATH=$(echo $dataset_entry | jq -r '.path')
        KEY=$(echo $dataset_entry | jq -r '.key')
        SUBSET=$(echo $dataset_entry | jq -r '.subset')

##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
######################################################################################### Dataset Preprocessing ##########################################################################################
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
        if [ $EVAL_ONLY != true ]; then
            DATASET_PATH=${BASE_DATASET_PATH}/${DATASET}
            DATASET_ORIGINAL_PATH=$DATASET_PATH/original
            DATASET_OUTPUT_PATH=$DATASET_PATH/${TOKENIZER_NAME}

            mkdir -p $DATASET_ORIGINAL_PATH
            mkdir -p $DATASET_OUTPUT_PATH

            # Check if .jsonl files exist in dataset_path
            jsonl_files=$(find $DATASET_ORIGINAL_PATH -type f -name "*.jsonl") &> /dev/null

            if [ -z "$jsonl_files" ]; then
                log "Dataset Preprocess\n\tNo original .jsonl files found in $DATASET_ORIGINAL_PATH\n\tDownloading original dataset..."

                # Run download script and capture output
                if [ $SUBSET = "null" ]; then
                    cd $SCRIPT_DIR/data_preprocess && \
                        python3 download_dataset.py \
                        --dataset_path $HF_PATH \
                        --output_dir $DATASET_ORIGINAL_PATH \
                        &> /dev/null
                else
                    cd $SCRIPT_DIR/data_preprocess && \
                        python3 download_dataset.py \
                        --dataset_path $HF_PATH \
                        --subset $SUBSET \
                        --output_dir $DATASET_ORIGINAL_PATH \
                        &> /dev/null
                fi
            else
                log "Dataset Preprocess\n\tOriginal .jsonl files already exist in $DATASET_ORIGINAL_PATH\n\tSkipping dataset download."
            fi

            SPLITS=$(find $DATASET_ORIGINAL_PATH -type f -name "*.jsonl" | rev | cut -d'/' -f1 | rev | cut -d'.' -f1 | tr '\n' ',')
            SPLITS=${SPLITS::-1}

            # Use IFS to split the string into an array
            IFS=',' read -ra SPLITS_ARRAY <<< $SPLITS

            # Loop through each element in the array
            for SPLIT in ${SPLITS_ARRAY[@]}; do
                ORIGINAL_DATA_PATH=$DATASET_ORIGINAL_PATH/$SPLIT.jsonl

                if [[ $KEY == *","* ]]; then
                    SAVE_KEY=$(echo $KEY | sed "s/,/_/g")
                else
                    SAVE_KEY=$KEY
                fi

                preprocessed_files=$(find $DATASET_OUTPUT_PATH -type f -name ${SPLIT}_${SAVE_KEY}.bin) &> /dev/null

                if [ -z $preprocessed_files ]; then
                    log "Dataset Preprocess\n\tDataset: $DATASET\n\tSplit: $SPLIT\n\tSave path: $DATASET_OUTPUT_PATH"
                    cd $SCRIPT_DIR/data_preprocess && \
                    python3 preprocess.py  write \
                        --file_path $ORIGINAL_DATA_PATH \
                        --key $KEY \
                        --save_prefix ${SPLIT}_${SAVE_KEY} \
                        --save_path $DATASET_OUTPUT_PATH \
                        --task $DATASET \
                        --do_keep_newlines \
                        --do_split_sentences \
                        --seq_length 512 \
                        --tokenizer_path $TOKENIZER_PATH \
                        --num_per_doc -1 \
                        --num_workers 16 \
                        --apply_chat_template \
                        &> $DATASET_OUTPUT_PATH/${SPLIT}_${SAVE_KEY}_write.log

                    if [ $? -eq 0 ]; then
                        log "Reading Dataset\n\tDataset: $DATASET\n\tSplit: $SPLIT"
                        (cd $SCRIPT_DIR/data_preprocess && \
                        python3 preprocess.py read \
                            --read_path_prefix=$DATASET_OUTPUT_PATH/${SPLIT}_${SAVE_KEY} \
                            --seq_length 512 \
                            --tokenizer_path $TOKENIZER_PATH) \
                            &> $DATASET_OUTPUT_PATH/${SPLIT}_${SAVE_KEY}_read.log
                        log "${SPLIT}_${SAVE_KEY} Preprocessing successful"
                    else
                        log "Error: Data Preprocessing failed"
                        echo "Failed Data Preprocessing: $MODEL $DATASET" >> "$BASE_DATASET_PATH/failed.log"
                        exit 1
                    fi
                else
                    log "Dataset Preprocess\n\tPreprocessed files already exist in $DATASET_OUTPUT_PATH\n\tSkipping dataset preprocessing."
                fi
            done
        fi

##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
####################################################################### Gradient Accumulation, Region extraction and Region Damage #######################################################################
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################

        CHECKPOINT_PATH=${BASE_CHECKPOINT_PATH}/${DATASET}/${TOKENIZER_NAME}
        TRAIN_OUTPUT_PATH=$CHECKPOINT_PATH/train
        EXTRACT_OUTPUT_PATH=$CHECKPOINT_PATH/extract
        DAMAGE_OUTPUT_PATH=$CHECKPOINT_PATH/damage

        EVAL_OUTPUT_PATH=$CHECKPOINT_PATH/eval

        mkdir -p $TRAIN_OUTPUT_PATH
        mkdir -p $EXTRACT_OUTPUT_PATH
        mkdir -p $DAMAGE_OUTPUT_PATH
        mkdir -p $EVAL_OUTPUT_PATH



    ################################################################################# Step 1: Gradient accumulation ##################################################################################
        if [ $EVAL_ONLY != true ]; then
            log "Gradient Accumulation\n\tModel: $MODEL\n\tDataset: $DATASET\n\tSave path: $TRAIN_OUTPUT_PATH"
            touch $TRAIN_OUTPUT_PATH/training.log

            cd $SCRIPT_DIR/training/step1_supervised_finetuning && \
            deepspeed accumulate_grad.py \
            --pretrain_train_data_path $DATASET_OUTPUT_PATH/train_$SAVE_KEY \
            --model_name_or_path $MODEL \
            --output_dir $TRAIN_OUTPUT_PATH \
            --total_cards $TOTAL_CARDS \
            --per_device_train_batch_size $TRAIN_BATCH_SIZE \
            --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
            --full_checkpoint \
            --zero_stage 2 \
            --max_seq_len 1024 \
            --learning_rate 5e-5 \
            --weight_decay 0.001 \
            --seed 1234 \
            --deepspeed \
            &> $TRAIN_OUTPUT_PATH/training.log

            if [ $? -eq 0 ]; then
                log "Training successful"
            else
                log "Error: Training failed"
                echo "Failed Training: $MODEL $DATASET" >> "$BASE_CHECKPOINT_PATH/failed.log"
                continue
            fi
        fi

        for K in ${K_VALUES[@]}; do
            ################################################################################# Step 2: Region extraction ##################################################################################
            if [ $EVAL_ONLY != true ]; then
                log "Extracting Region\n\tModel: $MODEL\n\tDataset: $DATASET\n\tK: $K\n\tSave path: $EXTRACT_OUTPUT_PATH"
                touch $EXTRACT_OUTPUT_PATH/extraction${K}.log

                cd $SCRIPT_DIR/extract_region && \
                python3 extract_region.py \
                generate_masks \
                --input_dir $TRAIN_OUTPUT_PATH \
                --output_dir $EXTRACT_OUTPUT_PATH \
                --k $K \
                --front_removal $FRONT_REMOVAL \
                --back_removal $BACK_REMOVAL \
                &> $EXTRACT_OUTPUT_PATH/extraction${K}.log

                if [ $? -eq 0 ]; then
                    log "Region extraction successful"
                else
                    log "Error: Region extraction failed"
                    echo "Failed Region Extraction: $MODEL $DATASET $K" >> "$BASE_CHECKPOINT_PATH/failed.log"
                    continue
                fi
            fi

            ################################################################################### Step 3: Region damage ####################################################################################
            if [ $EVAL_ONLY != true ]; then
                log "Damaging Region\n\tModel: $MODEL\n\tDataset: $DATASET\n\tK: $K\n\tSave path: $DAMAGE_OUTPUT_PATH"
                touch $DAMAGE_OUTPUT_PATH/damage${K}.log

                cd $SCRIPT_DIR/damage_region && \
                python3 damage_model.py \
                --checkpoints_dir $EXTRACT_OUTPUT_PATH \
                --original_model $MODEL \
                --output_dir $DAMAGE_OUTPUT_PATH \
                --k $K \
                &> $DAMAGE_OUTPUT_PATH/damage${K}.log

                if [ $? -eq 0 ]; then
                    log "Damage region successful"
                else
                    log "Error: Damage region failed"
                    echo "Failed Damage Region: $MODEL $DATASET $K" >> "$BASE_CHECKPOINT_PATH/failed.log"
                    continue
                fi
            fi


        done

##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
############################################################################################### Evaluation ###############################################################################################
##########################################################################################################################################################################################################
##########################################################################################################################################################################################################
        log "Running Evaluation"

        # Add this trap at the beginning of your script
        cleanup() {
            echo "Interrupt received. Cleaning up and terminating all processes..."
            # Kill all child processes
            pkill -P $$
            # Close file descriptor if open
            [[ -e /proc/$$/fd/3 ]] && exec 3>&-
            exit
        }

        # Set traps for various signals
        trap cleanup SIGINT SIGTERM

        IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"

        # Create a FIFO for GPU management (create only once outside the K loop)
        FIFO_FILE=$(mktemp -u)
        mkfifo $FIFO_FILE
        exec 3<>$FIFO_FILE
        rm $FIFO_FILE

        # Function to run a task on a specific GPU
        run_task() {
            local gpu_id=$1
            local task_type=$2  # "original", "top", "bottom"
            local benchmark=$3
            local num_fewshot=$4
            local k=$5
            local checkpoint_name=$6  # only for damaged models
            local checkpoint=$7       # only for damaged models

            # Set paths based on task type
            if [ "$task_type" == "original" ]; then
                local SAVE_PATH="$BASE_CHECKPOINT_PATH/original_eval/$TOKENIZER_NAME"
                local MODEL_PATH="$MODEL"
                local EVAL_FILE_PREFIX="${benchmark}"
                local REGION="original"
            else
                # Ensure no double slashes in paths
                local checkpoint_clean=$(echo $checkpoint | sed 's:/*$::')
                local MODEL_PATH="${checkpoint_clean}/${task_type}${k}"
                local SAVE_PATH="${EVAL_OUTPUT_PATH}/${checkpoint_name}/${task_type}${k}"
                local EVAL_FILE_PREFIX="${benchmark}${k}"
                local REGION="${task_type,,}"
            fi

            mkdir -p $SAVE_PATH
            touch $SAVE_PATH/${EVAL_FILE_PREFIX}.log

            max_attempts=3
            attempt=0

            while true; do
                gpu_memory_utilization=0.8

                cd $SCRIPT_DIR && \
                echo y | CUDA_VISIBLE_DEVICES=$gpu_id lm_eval --model vllm \
                --model_args pretrained=$MODEL_PATH,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=$gpu_memory_utilization,data_parallel_size=1,trust_remote_code=True \
                --tasks $benchmark \
                --batch_size 2 \
                --confirm_run_unsafe_code \
                &> $SAVE_PATH/${EVAL_FILE_PREFIX}.log

                if [ $? -eq 0 ]; then
                    # log "${task_type} evaluation successful for benchmark: $benchmark on GPU $gpu_id"
                    break
                elif [ "$attempt" -ge "$max_attempts" ]; then
                    log "Error: ${task_type} evaluation failed for $benchmark after $max_attempts tries on GPU $gpu_id."
                    echo "Failed ${task_type} Evaluation: $MODEL $DATASET ${task_type} $k $benchmark" >> "$BASE_CHECKPOINT_PATH/failed.log"
                    break
                fi

                attempt=$((attempt + 1))
            done

            # Release this GPU for next task
            echo $gpu_id >&3
        }

        # First, handle original model evaluations (only once, not per K)
        ORIGINAL_SAVE_PATH="$BASE_CHECKPOINT_PATH/original_eval/$TOKENIZER_NAME"
        mkdir -p $ORIGINAL_SAVE_PATH

        # Initialize the original tasks array
        declare -a ORIGINAL_TASKS=()

        for i in ${!BENCHMARKS[@]}; do
            BENCHMARK=${BENCHMARKS[$i]}
            NUM_FEWSHOT=${NUM_FEWSHOTS[$i]}

            # Check if original evaluation already exists
            if [ ! -f "$ORIGINAL_SAVE_PATH/${BENCHMARK}.log" ]; then
                ORIGINAL_TASKS+=("original $BENCHMARK $NUM_FEWSHOT 0 - -")  # K value doesn't matter for original
            else
                log "Original Evaluation already exists for benchmark: $BENCHMARK\n\tSkipping original evaluation."
            fi
        done

        # Run original evaluations first if there are any
        if [ ${#ORIGINAL_TASKS[@]} -gt 0 ]; then
            # Initialize the semaphore with available GPUs
            for gpu_id in "${GPU_ARRAY[@]}"; do
                echo $gpu_id >&3
            done

            for ((i=0; i<${#ORIGINAL_TASKS[@]}; i++)); do
                # Read a GPU ID from the semaphore
                read gpu_id <&3

                # Parse the current task
                task="${ORIGINAL_TASKS[$i]}"
                read task_type benchmark num_fewshot k checkpoint_name checkpoint <<< "$task"

                # Run the task in the background
                log "Starting original evaluation for benchmark $benchmark on GPU $gpu_id ($(($i+1))/${#ORIGINAL_TASKS[@]})"
                run_task $gpu_id $task_type $benchmark $num_fewshot $k $checkpoint_name $checkpoint &
            done

            # Wait for all original evaluations to complete
            wait
        fi

        # Initialize a single task queue for all K values
        declare -a ALL_TASKS=()

        # Build task queue for all K values
        for K in ${K_VALUES[@]}; do
            for checkpoint in $(ls -d $DAMAGE_OUTPUT_PATH/*); do
                checkpoint_name=$(basename $checkpoint)
                if [[ $checkpoint_name == *"damage"* ]]; then
                    continue
                fi

                for i in ${!BENCHMARKS[@]}; do
                    BENCHMARK=${BENCHMARKS[$i]}
                    NUM_FEWSHOT=${NUM_FEWSHOTS[$i]}

                    # Add each evaluation type to the queue with K value
                    ALL_TASKS+=("top $BENCHMARK $NUM_FEWSHOT $K $checkpoint_name $checkpoint")
                    ALL_TASKS+=("bottom $BENCHMARK $NUM_FEWSHOT $K $checkpoint_name $checkpoint")

                done
            done


        done

        # Initialize the semaphore with available GPUs (only once)
        for gpu_id in "${GPU_ARRAY[@]}"; do
            echo $gpu_id >&3
        done

        # Process all tasks at once
        TOTAL_TASKS=${#ALL_TASKS[@]}
        for ((i=0; i<TOTAL_TASKS; i++)); do
            # Read a GPU ID from the semaphore
            read gpu_id <&3

            # Parse the current task
            task="${ALL_TASKS[$i]}"
            read task_type benchmark num_fewshot k checkpoint_name checkpoint <<< "$task"

            # Run the task in the background
            log "Starting ${task_type} evaluation for benchmark $benchmark with K=$k on GPU $gpu_id ($(($i+1))/$TOTAL_TASKS)"
            run_task $gpu_id $task_type $benchmark $num_fewshot $k $checkpoint_name $checkpoint &
        done

        # Wait for all tasks to complete
        wait
        log "All evaluations completed"


        # Close the FIFO (only once at the end of everything)
        exec 3>&-

    done
done