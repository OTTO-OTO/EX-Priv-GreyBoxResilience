# EX-Priv

Welcome to the repository for "Enhancing the Resilience of LLMs against Grey-box Extractions." This code is adapted from the Llama-recipe repository(https://github.com/meta-llama/llama-recipes).

## Installation

```bash
pip install -r requirements.txt
```

## About EX-Priv

Within the EX-Priv folder, you will find our training code and the specific algorithm for EX-Priv. Given the multitude of models, we only demonstrate the practical application of EX-Priv in the Llama2-7B here, along with the specific training code for the 7 models used in our experiments.

## Running the Experiments

1. **Running the EX-Priv Algorithm**: If you want to run the EX-Priv algorithm, you can use the provided bash script:

    ```bash
    bash runEX_Priv-Llama.sh
    ```

2. **Replicating Our Training Experiments**: To replicate our experiments on training models, you can execute the following command:

    ```bash
    torchrun --nnodes 1 --nproc_per_node 4 --master_port=25001 SFT-Llama2.py \
        --enable_fsdp \
        --model_name llama \
        --project_name Llama-Sft-On-MixDataset \
        --group_name Layer0 \
        --model_path meta-llama/Llama-2-7b-chat-hf \
        --dist_checkpoint_root_folder path_to_save_checkpoints \
        --dist_checkpoint_root_load_folder path_to_save_checkpoints \
        --dist_checkpoint_folder exact_folder_name_of_your_checkpoints \
        --target_layer 0 \
        --lr 2e-5 \
        --weight_decay 0.1 \
        --dataset gen_dataset \
        --data_path path_to_your_training_data/validation_data.json \
        --eval_file_path path_to_your_validation_data/validation_data.json \
        --batch_size_training 32 \
        --val_batch_size 32 \
        --num_epochs 5 \
        --pure_bf16 \
        --low_cpu_fsdp False \
        --save_model True \
        --save_optimizer \
        --load_from_ckpt False \
        --scheduler Cosine \
        --record_loss True \
        --data_per_GPU 400 \ 
        --fsdp_cpu_offload False \
        --gamma 1 \
        --wipe_layer 1 \
        --seed 68 \
        --model_type llama \
        --loss_store_path mmlu-alpacamix-loss-record \
        --eval_step 40'
    ```

    Alternatively, you can use the `ModelTraining.py` file we provide. This file contains the execution code for the Llama2-7B model. 

    ```bash
    python ModelTraining.py
    ```

    If you wish to extend this to other models, simply change the target python file name in the command line:

    ```bash
    torchrun --nnodes 1 --nproc_per_node 4 --master_port=25001 SFT-Mistral.py \...
    ```

3. **Additional Configuration Settings**: For more parameter settings, you can find them in the `EX-Priv/training_utils/configs/training.py`.
