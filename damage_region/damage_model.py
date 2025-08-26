#!/usr/bin/env python3
"""
Benchmark Profiling: Mechanistic Diagnosis of LLM Benchmarks
Apply damage to critical parameter regions in language models.

Copyright (c) 2025 Dongjun Kim
Licensed under the Apache License, Version 2.0
"""

import os
import torch
import fire
from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_masks(model, mask_dir):
    """
    Apply masks to the model's parameters.
    Parameters with a value of 1 in the mask are set to 0 in the model.
    Handles shape mismatches by adjusting the mask as needed.
    """
    for name, param in model.named_parameters():
        # Create a standardized way to translate parameter names to valid filenames
        clean_name = name.replace(".", "_") + ".pt"
        mask_path = os.path.join(mask_dir, clean_name)

        if os.path.exists(mask_path):
            # Load mask
            mask = torch.load(mask_path, map_location=param.device)

            # Check if mask shape matches parameter shape
            if mask.shape != param.shape:
                print(
                    f"Shape mismatch for parameter '{name}': "
                    f"mask shape {mask.shape}, parameter shape {param.shape}"
                )

                # Handle mismatched shapes
                if len(mask.shape) == len(param.shape):  # Same number of dimensions
                    if mask.shape[0] < param.shape[0]:  # Mask has fewer rows
                        print(f"Padding mask for '{name}'...")
                        padding = (0, 0) * (len(mask.shape) - 1) + (
                            0,
                            param.shape[0] - mask.shape[0],
                        )
                        mask = torch.nn.functional.pad(mask, padding)
                    elif mask.shape[0] > param.shape[0]:  # Mask has more rows
                        print(f"Truncating mask for '{name}'...")
                        mask = mask[: param.shape[0], ...]

                else:
                    raise ValueError(
                        f"Cannot handle mismatched dimensions for '{name}': "
                        f"mask shape {mask.shape}, parameter shape {param.shape}"
                    )

            # Zero out parameters where mask is True
            param.data[mask] = 0.0
        else:
            print(f"Warning: Mask not found for parameter: {name}")


def damage_model(checkpoints_dir, output_dir, original_model, k):
    """
    Load and modify model for each checkpoint directory by applying top, bottom, and random masks.

    Args:
        checkpoints_dir: Directory containing checkpoint folders with top, bottom, and random masks.
        output_dir: Directory to save damaged models.
        original_model: Path to the original model.
        k: Percentage of parameters to damage (e.g., 0.01 for 1%).
    """
    # Verify directories exist and are correct
    assert os.path.exists(
        checkpoints_dir
    ), f"Checkpoints directory does not exist: {checkpoints_dir}"

    k_formatted = "{:.10f}".format(k).rstrip("0").rstrip(".")

    for checkpoint in sorted(os.listdir(checkpoints_dir)):
        checkpoint_dir = os.path.join(checkpoints_dir, checkpoint)

        if not os.path.isdir(checkpoint_dir):
            continue

        print(f"Processing checkpoint: {checkpoint}")

        # Define the mask directories for top, bottom, and randomm masks
        top_mask_dir = os.path.join(checkpoint_dir, f"top{k_formatted}")
        bottom_mask_dir = os.path.join(checkpoint_dir, f"bottom{k_formatted}")

        # Check if directories exist
        if not os.path.isdir(top_mask_dir):
            print(f"Warning: Top mask directory {top_mask_dir} does not exist.")
            continue
        if not os.path.isdir(bottom_mask_dir):
            print(f"Warning: Bottom mask directory {bottom_mask_dir} does not exist.")
            continue

        # Load model and tokenizer for each checkpoint
        print(f"Loading model from {original_model}...")
        model = AutoModelForCausalLM.from_pretrained(original_model)
        tokenizer = AutoTokenizer.from_pretrained(original_model)

        # Create and save top damaged model
        print("Applying top masks...")
        apply_masks(model, top_mask_dir)
        top_output_dir = os.path.join(output_dir, checkpoint, f"top{k_formatted}")
        os.makedirs(top_output_dir, exist_ok=True)
        model.save_pretrained(top_output_dir)
        tokenizer.save_pretrained(top_output_dir)
        print(f"Top {k_formatted}% damaged model saved to {top_output_dir}.")

        # Reload model to apply bottom masks
        print("Reloading model for bottom damage application...")
        model = AutoModelForCausalLM.from_pretrained(original_model)

        # Create and save bottom damaged model
        print("Applying bottom masks...")
        apply_masks(model, bottom_mask_dir)
        bottom_output_dir = os.path.join(output_dir, checkpoint, f"bottom{k_formatted}")
        os.makedirs(bottom_output_dir, exist_ok=True)
        model.save_pretrained(bottom_output_dir)
        tokenizer.save_pretrained(bottom_output_dir)
        print(f"Bottom {k_formatted}% damaged model saved to {bottom_output_dir}.")


if __name__ == "__main__":
    fire.Fire(damage_model)
