# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    project_name = ''
    group_name = ''
    model_name: str="llama70B"
    model_path: str="/gemini/data-1"
    enable_fsdp: bool=True
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=4
    gradient_accumulation_steps: int=1
    num_epochs: int=3
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "alpaca_dataset"
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "PATH/to/save/PEFT/model"  # save peft model
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="model_ckpt" # will be used if using FSDP
    dist_checkpoint_root_load_folder: str="model_ckpt"
    dist_checkpoint_folder: str="SFT" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    load_from_ckpt: bool=False
    scheduler: str='StepLR'
    data_per_GPU: int = 1000
    T_max: int=1000
    eta_min: float=0
    record_loss: bool=False
    mode: int=0  # 0:普通训练， 1：右移一位(/2)---loss是否会清零, 2: 右移完了再左移一位(/2*2)---查看精度损失, 3: 使用原函数进行eval
    shift_bit: int=1 # 移动位数 2^shift_bit
    wipe_layer: int=0 # wipe的layer
    layer_start_point: int=0 # 从什么地方开始移位 
    target_layer: int = 0 # 一点点数，最大是8
    wipe_all: bool = False
    lambda_step: int = 1


    
    
    
