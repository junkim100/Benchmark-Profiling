#!/usr/bin/env python3
"""
Benchmark Profiling: Mechanistic Diagnosis of LLM Benchmarks
Extract critical parameter regions from fine-tuned models (MLP layers only).

Copyright (c) 2025 Dongjun Kim
Licensed under the Apache License, Version 2.0
"""

import os
import torch
import fire
from tqdm import tqdm
import csv
import math
import re
from save_selected_params import save_selected_params


def parse_stats_file(file_path):
    """
    Visualization stripped. This function previously parsed CSV stats for plotting.
    """
    return []


def format_k_value(k_value):
    """
    Visualization stripped. Kept for backward compatibility.
    """
    try:
        k_value_float = float(k_value)
        return f"{k_value_float:.10f}".rstrip("0").rstrip(".")
    except ValueError:
        return str(k_value)


def visualize(data, k_value, input_dir, front_removal, back_removal):
    """
    Visualization stripped. No-op.
    """
    print("[extract_region] Visualization disabled.")
    return


def collect_csv_and_visualize(input_dir, front_removal=2, back_removal=2):
    """
    Visualization stripped. No-op.
    """
    print("[extract_region] CSV visualization disabled.")
    return


@torch.no_grad()
def dynamic_batched_top_k(all_values, k_fraction, device=torch.device("cuda:0")):
    """
    Computes top, bottom, and random selections for a given fraction (k_fraction).
    - all_values: A 1D tensor containing the values from which we pick.
    - k_fraction: The fraction of parameters (0.0 < k_fraction < 1.0) to pick for top
                  and bottom. Also used for random picks.
    - device: Torch device for all operations.
    Returns:
      (top_indices_full, bottom_indices_full, random_indices_full)
      where each is a 1D torch.LongTensor of selected indices in all_values.
    """
    total_params = all_values.size(0)
    # Decide batch size dynamically
    base_batch_size = 1_000_000
    adapted_batch_size = max(
        base_batch_size, int(math.sqrt(total_params / (k_fraction + 1e-9)))
    )
    print(f"Using batch size: {adapted_batch_size} for k_fraction={k_fraction:.10f}")

    # Indices accumulators
    top_indices_full = torch.tensor([], dtype=torch.long, device=device)
    bottom_indices_full = torch.tensor([], dtype=torch.long, device=device)
    # random_indices_full = torch.tensor([], dtype=torch.long, device=device)

    # Process parameters in dynamic batches
    for start_idx in range(0, total_params, adapted_batch_size):
        end_idx = min(start_idx + adapted_batch_size, total_params)
        batch_values = all_values[start_idx:end_idx]

        # How many elements from this batch do we select?
        num_elements_in_batch = batch_values.size(0)
        num_k_elements = max(1, math.ceil(k_fraction * num_elements_in_batch))
        print(
            f"Batch start: {start_idx}, "
            f"Selection: {num_k_elements}/{num_elements_in_batch} (k_fraction={k_fraction:.10f})"
        )

        # Top k (largest is True)
        top_indices = (
            torch.topk(batch_values, num_k_elements, largest=True).indices + start_idx
        ).to(device)
        # Bottom k (largest=False -> smallest)
        bottom_indices = (
            torch.topk(batch_values, num_k_elements, largest=False).indices + start_idx
        ).to(device)
        # Random k
        # random_indices_local = torch.randperm(num_elements_in_batch, device=device)[
        #     :num_k_elements
        # ]
        # random_indices_local = (random_indices_local + start_idx).to(device)

        # Accumulate
        top_indices_full = torch.cat((top_indices_full, top_indices))
        bottom_indices_full = torch.cat((bottom_indices_full, bottom_indices))
        # random_indices_full = torch.cat((random_indices_full, random_indices_local))

    # return top_indices_full, bottom_indices_full, random_indices_full
    return top_indices_full, bottom_indices_full


@torch.no_grad()
def generate_masks(input_dir, output_dir, k, front_removal=2, back_removal=2):
    """
    Main function to generate top/bottom/random masks from MLP layers only, using absolute values.
    - input_dir: Directory containing checkpoint subfolders with .pt parameter files.
    - output_dir: Directory to save masks and CSV statistics.
    - k: Fraction of the total model parameters to pick. However, we only pick from
         MLP layers. The exact count is k * total_parameters (of the entire model),
         but allocated solely among MLP parameters. Non-MLP parameters receive
         zero-filled masks.
    - We also use the absolute values of the MLP parameters for top/bottom selection.
    Example usage:
      python script.py generate_masks --input_dir=... --output_dir=... --k=0.05
    """
    device = torch.device("cuda:0")
    # Walk the input_dir for checkpoints
    for checkpoint_dir in tqdm(
        sorted(os.listdir(input_dir)), desc="Processing checkpoints", unit="checkpoint"
    ):
        checkpoint_path = os.path.join(input_dir, checkpoint_dir)
        if not os.path.isdir(checkpoint_path):
            continue

        # Prepare lists for MLP and non-MLP data
        mlp_values_list = []
        mlp_files = []
        mlp_shapes = []
        nonmlp_files = []
        nonmlp_shapes = []

        # Keep track of total params for the entire checkpoint
        total_params_count = 0

        # -------------------
        # 1) Collect .pt files; separate MLP vs Non-MLP; track shapes
        # -------------------
        for root, _, files in os.walk(checkpoint_path):
            for file in tqdm(
                files, desc=f"Processing files in {checkpoint_dir}", unit="file"
            ):
                if file.endswith(".pt"):
                    file_path = os.path.join(root, file)
                    tensor = torch.load(file_path, map_location=device)
                    numel = tensor.numel()
                    total_params_count += numel

                    # Check if this belongs to an MLP layer
                    # Extract layer number from the file path
                    match = re.search(r"model\.layers\.(\d+)\.", file_path)
                    if match:
                        layer_idx = int(match.group(1))
                    else:
                        # If no valid layer index can be extracted, skip this file
                        continue

                    def get_total_layers(checkpoint_path):
                        layer_indices = set()
                        for root, _, files in os.walk(checkpoint_path):
                            for file in files:
                                if file.endswith(".pt"):
                                    file_path = os.path.join(root, file)
                                    match = re.search(
                                        r"model\.layers\.(\d+)\.", file_path
                                    )
                                    if match:
                                        layer_idx = int(match.group(1))
                                        layer_indices.add(layer_idx)
                        return max(layer_indices) + 1 if layer_indices else 0

                    total_layers = get_total_layers(checkpoint_path)

                    # Determine if this is a valid MLP layer
                    if "mlp" in file_path.lower() and (
                        front_removal <= layer_idx < total_layers - back_removal
                    ):
                        # Use the absolute values of MLP parameters
                        abs_tensor = tensor.abs()
                        mlp_values_list.append(abs_tensor.view(-1))
                        mlp_files.append(file)
                        mlp_shapes.append(tensor.shape)
                    else:
                        # Non-MLP parameter or MLP parameter from unwanted layers
                        nonmlp_files.append(file)
                        nonmlp_shapes.append(tensor.shape)

        if not mlp_values_list:
            print(f"No MLP layers found in checkpoint: {checkpoint_dir}. Skipping.")
            continue

        # -------------------
        # 2) Convert MLP lists to single large 1D tensor of absolute values
        # -------------------
        mlp_values = torch.cat(mlp_values_list).to(device)

        # Number to select across the entire model (k fraction).
        num_to_select = int(math.floor(k * total_params_count))

        # If we don't have enough MLP params to fulfill num_to_select, skip
        mlp_total = mlp_values.size(0)
        if mlp_total < num_to_select:
            print(
                f"MLP layers have {mlp_total} params, but want to select {num_to_select}. "
                f"Skipping {checkpoint_dir}."
            )
            continue

        # Fraction of the MLP subset needed to reach total of num_to_select
        fraction_of_mlp = num_to_select / mlp_total

        # 3) Use dynamic_batched_top_k on the MLP parameters (ABSOLUTE VALUES)
        # top_indices_mlp, bottom_indices_mlp, random_indices_mlp = dynamic_batched_top_k(
        #     mlp_values, fraction_of_mlp, device=device
        # )
        top_indices_mlp, bottom_indices_mlp = dynamic_batched_top_k(
            mlp_values, fraction_of_mlp, device=device
        )

        # -------------------
        # 4) Prepare to write CSV: One per checkpoint
        # -------------------
        csv_dir = os.path.join(output_dir, checkpoint_dir)
        os.makedirs(csv_dir, exist_ok=True)

        # Format k_value for the CSV filename
        k_formatted = "{:.10f}".format(k).rstrip("0").rstrip(".")
        stats_csv_path = os.path.join(csv_dir, f"layer_statistics_{k_formatted}.csv")
        with open(stats_csv_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Layer", "Total Parameters", "Top", "Bottom"])

            # We'll create maps for the final masks
            top_masks = {}
            bottom_masks = {}
            # random_masks = {}

            # -------------------------------
            # 5) Reconstruct the selection indices for MLP files
            # -------------------------------
            # Sort the mlp files by layer number
            def extract_layer_number(filename):
                match = re.search(r"model\.layers\.(\d+)\.", filename)
                return int(match.group(1)) if match else float("inf")

            # Create list of tuples with (layer_number, filename, shape) to maintain correspondence
            file_info = [
                (extract_layer_number(f), f, s) for f, s in zip(mlp_files, mlp_shapes)
            ]

            # Sort the list by layer number
            file_info.sort(key=lambda x: x[0])

            # Unpack the sorted lists back into mlp_files and mlp_shapes
            mlp_files = [x[1] for x in file_info]
            mlp_shapes = [x[2] for x in file_info]

            offset = 0
            for idx, file in enumerate(
                tqdm(
                    mlp_files,
                    desc=f"Creating MLP masks in {checkpoint_dir}",
                    unit="file",
                )
            ):
                shape = mlp_shapes[idx]
                numel = shape.numel()

                # Among the combined MLP array, local indices for this file are those
                # that fall in [offset, offset+numel).
                local_top_indices = (
                    top_indices_mlp[
                        (top_indices_mlp >= offset) & (top_indices_mlp < offset + numel)
                    ]
                    - offset
                )
                local_bottom_indices = (
                    bottom_indices_mlp[
                        (bottom_indices_mlp >= offset)
                        & (bottom_indices_mlp < offset + numel)
                    ]
                    - offset
                )
                # local_random_indices = (
                #     random_indices_mlp[
                #         (random_indices_mlp >= offset)
                #         & (random_indices_mlp < offset + numel)
                #     ]
                #     - offset
                # )

                # Build empty masks for this file
                top_mask = torch.zeros(numel, dtype=torch.bool, device=device)
                bottom_mask = torch.zeros(numel, dtype=torch.bool, device=device)
                # random_mask = torch.zeros(numel, dtype=torch.bool, device=device)

                # Mark selected positions
                top_mask[local_top_indices] = True
                bottom_mask[local_bottom_indices] = True
                # random_mask[local_random_indices] = True

                # Reshape back to original shape
                top_masks[file] = top_mask.view(shape)
                bottom_masks[file] = bottom_mask.view(shape)

                # Save the selected parameter indices and layer name
                params_dir = os.path.join(output_dir, checkpoint_dir, "selected_params")
                os.makedirs(params_dir, exist_ok=True)
                save_selected_params(local_top_indices, file, params_dir, k)

                # Write a row summarizing the selection for this file
                top_count = local_top_indices.numel()
                bottom_count = local_bottom_indices.numel()
                writer.writerow(
                    [
                        f"Layer: {file}",
                        f"Total Parameters: {numel}",
                        f"Top: {top_count} ({(top_count / numel) * 100:.2f}%)",
                        f"Bottom: {bottom_count} ({(bottom_count / numel) * 100:.2f}%)",
                    ]
                )

                offset += numel

            # -------------------------------
            # 6) Create zero-filled masks for non-MLP files
            # because we're not selecting from them
            # -------------------------------
            for idx, file in enumerate(
                tqdm(
                    nonmlp_files,
                    desc=f"Creating Non-MLP masks in {checkpoint_dir}",
                    unit="file",
                )
            ):
                shape = nonmlp_shapes[idx]
                numel = shape.numel()

                # All-zero masks
                top_masks[file] = torch.zeros(
                    numel, dtype=torch.bool, device=device
                ).view(shape)
                bottom_masks[file] = torch.zeros(
                    numel, dtype=torch.bool, device=device
                ).view(shape)
                # random_masks[file] = torch.zeros(
                #     numel, dtype=torch.bool, device=device
                # ).view(shape)

                # Write a row summarizing the selection for this file (all zero)
                writer.writerow(
                    [
                        f"Layer: {file}",
                        f"Total Parameters: {numel}",
                        f"Top: {0} (0.00%)",
                        f"Bottom: {0} (0.00%)",
                    ]
                )

        # Save the masks
        top_dir = os.path.join(output_dir, checkpoint_dir, f"top{k_formatted}")
        bottom_dir = os.path.join(output_dir, checkpoint_dir, f"bottom{k_formatted}")

        os.makedirs(top_dir, exist_ok=True)
        os.makedirs(bottom_dir, exist_ok=True)

        for file in tqdm(
            top_masks.keys(), desc=f"Saving masks in {checkpoint_dir}", unit="file"
        ):

            torch.save(
                top_masks[file],
                os.path.join(
                    top_dir, f"{file.replace('.pt', '').replace('.', '_')}.pt"
                ),
            )

            torch.save(
                bottom_masks[file],
                os.path.join(
                    bottom_dir, f"{file.replace('.pt', '').replace('.', '_')}.pt"
                ),
            )

        # Analysis stripped: previously analyzed and visualized selected parameters here.
        # This block has been intentionally removed to keep the extractor minimal.


def analyze_selected_parameters(input_dir, k=None, output_dir=None):
    """
    Analysis stripped. No-op.
    """
    print("[extract_region] Analysis disabled.")
    return


def process_analysis_results(df, k, output_dir, prefix):
    """
    Analysis stripped. No-op.
    """
    print("[extract_region] Analysis post-processing disabled.")
    return


if __name__ == "__main__":
    # We use Fire to provide a quick CLI to our generate_masks function.
    # Example usage:
    #   python script_name.py generate_masks --input_dir=... --output_dir=... --k=0.01
    fire.Fire({"generate_masks": generate_masks})
