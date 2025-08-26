"""
This script is designed for fine-tuning transformers models on causal language modeling tasks using DeepSpeed
for efficient distributed training. It supports various configurations for model optimization, data handling,
and training strategies.

Key Features:
-------------
1. **Distributed Training**:
    - Utilizes DeepSpeed for distributed training across multiple GPUs.
    - Supports ZeRO Offload techniques to optimize memory usage.

2. **Data Handling**:
    - Custom dataset class (`MyDataset`) for handling preprocessed binary data.
    - Data collator (`MyDataCollatorForSupervisedDataset`) to pad sequences and create attention masks.

3. **Model Configuration**:
    - Loads models from Hugging Face's Transformers library.
    - Configures models with gradient checkpointing and dropout settings.

4. **Training Configuration**:
    - Supports gradient accumulation and learning rate scheduling.
    - Allows configuration of batch sizes, sequence lengths, and optimization parameters.

File Structure:
---------------
1. **Imports**:
   - Organized into standard library, third-party libraries, and local modules for clarity.
   - Environment setup for PyTorch CUDA memory allocation.

2. **Utility Functions**:
   - `set_first_false_to_true`: Adjusts attention masks by setting the first occurrence of False to True in each row.

3. **Classes**:
   - `MyDataCollatorForSupervisedDataset`: Collates examples by padding sequences to a fixed length.
   - `MyDataset`: Custom dataset class for loading preprocessed data efficiently.

4. **Main Functionality**:
   - `main()`: Orchestrates the training process by setting up devices, loading data, configuring models, and running training loops.

5. **Helper Functions**:
   - Includes functions for displaying sample batches, setting up samplers, and training epochs.

Usage:
------
To run the script, use command-line arguments to specify model paths, data paths, and training configurations.
Example command:
    python accumulate_grad.py --model_name_or_path <model_path> --pretrain_train_data_path <train_data> --num_train_epochs 3

Notes:
------
- Ensure that DeepSpeed is installed and configured correctly for distributed training.
- The script assumes that input data is preprocessed into binary format with appropriate indexing files (.idx, .bin).
- Customizable through command-line arguments for various training parameters and configurations.
"""

import os
import sys
import math
import json
from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import Dict, Sequence, List

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.utils import safe_get_full_grad

# Add parent directory to sys.path for importing local modules
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

# Local application/library specific imports
from utils.utils import (
    print_rank_0,
    to_device,
    set_random_seed,
    get_all_reduce_mean,
    get_optimizer_grouped_parameters,
    load_hf_tokenizer,
)
from utils.ds_utils import get_train_ds_config
from utils.model.model_utils import create_hf_model

# Environment setup for PyTorch CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Constants
IGNORE_INDEX = -100


def set_first_false_to_true(mask_tensor):
    """
    Sets the first occurrence of False in each row of a mask tensor to True.

    Args:
        mask_tensor (torch.Tensor): A boolean tensor where each row is processed independently.

    Returns:
        torch.Tensor: The modified tensor with the first False set to True in each row.
    """
    # Find the position of the first False in each row
    first_false_indices = (~mask_tensor).cumsum(dim=1) == 1

    # Set the first False in each row to True
    mask_tensor[first_false_indices] = True
    return mask_tensor


@dataclass
class MyDataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning by padding sequences to a fixed length."""

    pad_token_id: int
    max_seq_len: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        Pads input sequences and labels to the maximum sequence length.

        Args:
            instances (Sequence[Dict]): A list of dictionaries containing 'input_ids' and 'labels'.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with padded input_ids, labels, and attention_mask.
        """
        # Extract input_ids and labels from instances
        input_ids, labels = (
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )

        # Pad sequences to the longest sequence in the batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        # Ensure all sequences are padded to max_seq_len
        if input_ids.size(1) < self.max_seq_len:
            padding_size = self.max_seq_len - input_ids.size(1)
            padding = torch.full(
                (input_ids.size(0), padding_size),
                self.pad_token_id,
                dtype=input_ids.dtype,
            )
            input_ids = torch.cat((input_ids, padding), dim=1)

        if labels.size(1) < self.max_seq_len:
            padding_size = self.max_seq_len - labels.size(1)
            padding = torch.full(
                (labels.size(0), padding_size), IGNORE_INDEX, dtype=labels.dtype
            )
            labels = torch.cat((labels, padding), dim=1)

        # Create attention mask where True indicates a valid token position
        attention_mask = input_ids.ne(self.pad_token_id)

        # Adjust attention mask to ensure the first False is set to True (right-padding)
        attention_mask = set_first_false_to_true(attention_mask)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class MyDataset(Dataset):
    """Custom dataset class for handling preprocessed binary data."""

    def __init__(self, data_prefix: str, seq_length: int, pad_id: int):
        """
        Initializes the dataset with paths to binary data files.

        Args:
            data_prefix (str): Path prefix for the dataset files (without suffix).
            seq_length (int): Maximum sequence length for samples.
            pad_id (int): Padding token ID used in sequences.

        Notes:
            - The data_prefix should be a complete path without file extensions.
              Suffixes (.idx, .bin, .dis) are automatically appended as needed.
              Example: '/path/to/data' will map to '/path/to/data.idx', etc.
        """
        super(MyDataset, self).__init__()

        self.idx_file_path = f"{data_prefix}.idx"
        self.bin_file_path = f"{data_prefix}.bin"
        self.dis_file_path = f"{data_prefix}.dis"

        self.seq_length = seq_length
        self.pad_id = pad_id

        self.index_start_pos = None  # Starting position of each sample
        self.index_length = None  # Length of each sample

        self._load_index()
        self._load_bin()
        self._load_dis()

        # Verify data integrity after loading
        self._check()

    def _check(self):
        """Verify that the loaded data is consistent and correct."""
        assert self.index_length[-1] + self.index_start_pos[-1] == len(
            self.bin_buffer
        ), "Data validation error!"

    def _load_index(self):
        """Load index information from file."""

        file_size = os.stat(self.idx_file_path).st_size

        # Ensure the file size is a multiple of 12 bytes (8B start pos + 4B length)
        assert file_size % 12 == 0, "File size must be a multiple of 12 bytes."
        self.total_sample = file_size // 12

        with open(self.idx_file_path, "rb") as f:
            # Read start positions (8 bytes each)
            self.index_start_pos = np.frombuffer(
                f.read(self.total_sample * 8), dtype=np.uint64
            ).tolist()

            # Read lengths (4 bytes each)
            self.index_length = np.frombuffer(
                f.read(self.total_sample * 4), dtype=np.uint32
            ).tolist()

    def _load_bin(self):
        """Load binary data using memory mapping."""

        self.bin_buffer = np.memmap(self.bin_file_path, dtype=np.uint32, mode="r")

    def _load_dis(self):
        """Load distribution metadata if available."""

        self.distributed = torch.load(self.dis_file_path)

        if len(self.distributed) != 0:
            assert sum(self.distributed) == self.total_sample

    def __len__(self):
        return self.total_sample

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing 'input_ids' and 'labels'.

            Notes:
                - Uses dynamic length based on the sample size.
                - Truncates samples that exceed seq_length.
                - No padding is applied at this stage; handled by collator.
        """

        start_idx = self.index_start_pos[idx]
        length = min(self.index_length[idx], self.seq_length)

        data = torch.as_tensor(
            self.bin_buffer[start_idx : start_idx + length].tolist(), dtype=torch.long
        )

        labels = data.clone()  # Labels are identical to input IDs initially

        return {"input_ids": data, "labels": labels}


def _make_supervised_data_module(args, train_data_prefix, pad_id=0) -> Dict:
    """
    Initializes datasets and data collators for supervised training.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing configuration settings.
        train_data_prefix (str): Path prefix for the training dataset files.
        pad_id (int): ID of the padding token used in tokenized sequences.

    Returns:
        Dict: A dictionary containing train_dataset and train_data_collator.
    """
    # Initialize datasets
    train_dataset = MyDataset(
        data_prefix=train_data_prefix, seq_length=args.max_seq_len, pad_id=pad_id
    )

    # Initialize data collators
    train_data_collator = MyDataCollatorForSupervisedDataset(
        pad_token_id=pad_id, max_seq_len=args.max_seq_len
    )
    return {
        "train_dataset": train_dataset,
        "train_data_collator": train_data_collator,
    }


def parse_args():
    """
    Parses command-line arguments for fine-tuning a transformers model on a causal language modeling task.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )

    # Data paths
    parser.add_argument(
        "--pretrain_train_data_path",
        type=str,
        default="default_train_path/train",
        help="Path to the training data.",
    )

    # Model configuration
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to new pretrained model or model identifier from huggingface.co/models.",
    )

    # Training configuration
    parser.add_argument(
        "--total_cards", type=int, help="Total cards for distributed training on GPUs."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )

    # Sequence length and learning parameters
    parser.add_argument(
        "--max_seq_len", type=int, default=512, help="The maximum sequence length."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    # Regularization and optimization
    parser.add_argument(
        "--weight_decay", type=float, default=0.001, help="Weight decay to use."
    )

    # Epochs and gradient accumulation
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    # Save checkpoint only once at the end of training
    parser.add_argument(
        "--full_checkpoint",
        action="store_true",
        help="Save checkpoint only once at the end of training.",
    )

    # Learning rate scheduler
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.",
    )

    # Output and seed
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="A seed for reproducible training."
    )

    # Distributed training
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training on GPUs.",
    )

    # Additional features and configurations
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for model.",
    )
    parser.add_argument(
        "--disable_dropout",
        action="store_true",
        help="Disable the dropout of the model.",
    )

    # DeepSpeed features
    parser.add_argument(
        "--offload", action="store_true", help="Enable ZeRO Offload techniques."
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model (and clones).",
    )

    # Add DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


def main():
    """
    Main function to fine-tune a transformers model on a causal language modeling task using DeepSpeed for distributed training.
    """
    # Parse command-line arguments
    args = parse_args()

    # Set the device for training
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()

    # Get global rank for distributed training
    args.global_rank = torch.distributed.get_rank()

    # Configure DeepSpeed settings
    ds_config = get_train_ds_config(offload=args.offload, stage=args.zero_stage)
    ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    ds_config["train_batch_size"] = (
        args.per_device_train_batch_size
        * torch.distributed.get_world_size()
        * args.gradient_accumulation_steps
    )

    # Set random seed for reproducibility
    set_random_seed(args.seed)

    # Synchronize processes
    torch.distributed.barrier()

    # Load tokenizer and configure padding settings
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Create model using the specified configuration
    model = create_hf_model(
        AutoModelForCausalLM,
        args.model_name_or_path,
        tokenizer,
        ds_config=None,
        disable_dropout=args.disable_dropout,
    )

    # Initialize datasets and data collators
    data_module = _make_supervised_data_module(
        args,
        train_data_prefix=args.pretrain_train_data_path,
        pad_id=tokenizer.eos_token_id,
    )
    train_dataset = data_module["train_dataset"]
    train_data_collator = data_module["train_data_collator"]

    # if train dataset has more than 500k samples, we will only use the first 500k samples
    if len(train_dataset) > 500000:
        train_dataset = torch.utils.data.Subset(train_dataset, range(500000))

    def display_sample_batches(dataset, data_collator, tokenizer, global_rank):
        """Displays sample batches from the dataset for verification."""
        sample_batch = data_collator([dataset[i] for i in range(5)])
        print_rank_0("\nSample Batch:", global_rank)
        print_rank_0(sample_batch, global_rank)
        for i in range(2):
            print_rank_0("\nSample Text:", global_rank)
            print_rank_0(tokenizer.decode(dataset[i]["input_ids"]), global_rank)

    def setup_samplers(train_dataset, local_rank):
        """Sets up samplers based on whether distributed training is used."""
        if local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)
        return train_sampler

    # Display sample batches for verification
    display_sample_batches(
        train_dataset, train_data_collator, tokenizer, args.global_rank
    )
    print_rank_0(f"len_train_sample: {len(train_dataset)}", args.global_rank)

    # Set up data samplers based on distributed settings
    train_sampler = setup_samplers(train_dataset, args.local_rank)

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=train_data_collator,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
    )

    # Prepare optimizer and scheduler for training
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay
    )
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(
        optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95)
    )
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    # Initialize DeepSpeed with model and optimizer
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Training loop setup
    batch_size = args.total_cards * args.per_device_train_batch_size

    # Checkpointing setup
    if args.full_checkpoint:
        checkpoint_interval = len(train_dataset)
    else:
        # Determine checkpoint interval based on dataset size and batch size
        if len(train_dataset) >= 100000:
            checkpoint_interval = 100000
        elif len(train_dataset) >= 10000:
            checkpoint_interval = 10000
        elif len(train_dataset) >= 1000:
            checkpoint_interval = 1000
        elif len(train_dataset) >= 100:
            checkpoint_interval = 100
        else:
            raise ValueError(
                "Training data is too small. Set full_checkpoint to True for a single checkpoint."
            )
    # Determine when to save checkpoints
    save_samples = list(
        range(checkpoint_interval, len(train_dataset) + 1, checkpoint_interval)
    )
    save_steps = [math.ceil(samples / batch_size) for samples in save_samples]
    save_dict = dict(zip(save_steps, save_samples))

    print_rank_0(
        f"\n============================================================ Dataset Info ============================================================",
        args.global_rank,
    )
    print_rank_0(
        f"Dataset length: {len(train_dataset)}, Batch size: {batch_size}, Checkpoint interval: {checkpoint_interval}",
        args.global_rank,
    )
    print_rank_0(f"Save Dict: {save_dict}", args.global_rank)
    if hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()
    print_rank_0(
        "\n============================================================ Running training ============================================================",
        args.global_rank,
    )

    if hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()

    def train_one_epoch(epoch, model, train_dataloader):
        """Trains the model for one epoch and handles checkpointing.

        Args:
            epoch (int): Current epoch number
            model: The model to train
            train_dataloader: DataLoader for training data
            optimizer: The optimizer
            args: Training arguments and configuration
        """
        total_steps = len(train_dataloader)
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {total_steps}",
            args.global_rank,
        )

        model.train()

        def _save_checkpoint(step):
            """Helper function to save model checkpoints."""
            if args.full_checkpoint:
                checkpoint_name = "full"
            else:
                checkpoint_name = save_dict[step + 1]
            save_dir = os.path.join(args.output_dir, f"checkpoint_{checkpoint_name}")
            os.makedirs(save_dir, exist_ok=True)

            for name, param in model.named_parameters():
                # Get gradient and compute product
                gradient = safe_get_full_grad(param)
                grad_param_product = torch.mul(gradient, param)

                # Save tensor
                clean_name = name.replace("module.", "")
                save_path = os.path.join(save_dir, f"{clean_name}.pt")
                torch.save(grad_param_product.bfloat16(), save_path)

            # Clear GPU memory
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()

        for step, batch in tqdm(
            enumerate(train_dataloader), total=total_steps, unit="batch"
        ):
            # Training step
            batch = to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            model.backward(loss)

            # Logging
            if step in save_steps:
                print_rank_0(
                    f"Epoch {epoch+1}/{args.num_train_epochs}, "
                    f"Step {step+1}/{total_steps}, "
                    f"Loss {loss.item():.4f}",
                    args.global_rank,
                )

            # Checkpoint saving
            if (step + 1) in save_steps:
                # Save checkpoint
                _save_checkpoint(step)

                # Early stopping condition
                if save_dict[step + 1] == save_samples[-1]:
                    print_rank_0(
                        f"\n============================================================ Reached {save_samples[-1]} steps, stopping training ============================================================",
                        args.global_rank,
                    )
                    return True

        return False

    for epoch in range(args.num_train_epochs):
        train_one_epoch(epoch, model, train_dataloader)


if __name__ == "__main__":
    main()
