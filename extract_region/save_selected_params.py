import os
import torch
import argparse
import json
import csv
from pathlib import Path


def save_selected_params(indices, layer_name, output_dir, k_value):
    """
    Save the selected parameter indices and layer name to a file.

    Args:
        indices: Tensor of selected parameter indices (flattened)
        layer_name: Name of the layer (PT file)
        output_dir: Directory to save the output file
        k_value: The k value used for selection
    """
    # Format k_value for consistent naming
    k_formatted = "{:.10f}".format(float(k_value)).rstrip("0").rstrip(".")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a clean layer name for the file
    clean_layer_name = layer_name.replace(".pt", "").replace(".", "_")

    # Convert indices to a list for JSON serialization
    indices_list = indices.cpu().tolist() if torch.is_tensor(indices) else indices

    # Create data structure to save
    data = {
        "layer_name": layer_name,
        "indices": indices_list,
        "k_value": k_value
    }

    # Save as JSON
    json_path = os.path.join(output_dir, f"selected_params_{clean_layer_name}_{k_formatted}.json")
    with open(json_path, 'w') as f:
        json.dump(data, f)

    # Also save as CSV for easier analysis
    csv_path = os.path.join(output_dir, f"selected_params_{k_formatted}.csv")

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow(["Layer", "K Value", "Index Count", "Indices"])

        # Write data row
        writer.writerow([
            layer_name,
            k_value,
            len(indices_list),
            str(indices_list[:10]) + "..." if len(indices_list) > 10 else str(indices_list)
        ])

    print(f"Saved selected parameters for {layer_name} with k={k_formatted} to {json_path}")
    print(f"Updated CSV summary at {csv_path}")

    return json_path, csv_path


def load_selected_params(file_path):
    """
    Load selected parameter indices and layer name from a file.

    Args:
        file_path: Path to the JSON file containing saved indices

    Returns:
        Dictionary with layer_name, indices, and k_value
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Convert indices back to tensor if needed
    if isinstance(data["indices"], list):
        data["indices"] = torch.tensor(data["indices"])

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save or load selected parameter indices")
    parser.add_argument("--action", choices=["save", "load"], required=True,
                        help="Whether to save or load indices")
    parser.add_argument("--indices", type=str, help="Comma-separated list of indices (for save)")
    parser.add_argument("--layer_name", type=str, help="Name of the layer (for save)")
    parser.add_argument("--output_dir", type=str, help="Output directory (for save)")
    parser.add_argument("--k_value", type=float, help="K value used for selection (for save)")
    parser.add_argument("--file_path", type=str, help="Path to the saved file (for load)")

    args = parser.parse_args()

    if args.action == "save":
        if not all([args.indices, args.layer_name, args.output_dir, args.k_value]):
            raise ValueError("For save action, indices, layer_name, output_dir, and k_value are required")

        # Convert string of indices to list
        indices = [int(idx) for idx in args.indices.split(",")]

        save_selected_params(indices, args.layer_name, args.output_dir, args.k_value)

    elif args.action == "load":
        if not args.file_path:
            raise ValueError("For load action, file_path is required")

        data = load_selected_params(args.file_path)
        print(f"Loaded data for layer: {data['layer_name']}")
        print(f"K value: {data['k_value']}")
        print(f"Number of indices: {len(data['indices'])}")
        print(f"First 10 indices: {data['indices'][:10]}")
