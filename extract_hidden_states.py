#!/usr/bin/env python3
"""
Extract hidden states from an LLM at specific layers and token positions.

This script processes a dataset of queries and extracts hidden states from
a specified transformer layer and token position, saving the results as a PyTorch tensor.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from position_spec import PositionSpec, resolve_position_spec


def load_dataset(dataset_path: str) -> List[str]:
    """
    Load queries from a JSON dataset file.

    Args:
        dataset_path: Path to JSON file containing list of query objects

    Returns:
        List of query strings
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON file to contain a list, got {type(data)}")

    queries = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not a dictionary: {item}")
        if "query" not in item:
            raise ValueError(f"Item {i} missing 'query' field: {item}")
        queries.append(item["query"])

    return queries


def extract_hidden_states_batched(
    queries: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: List[int],
    position_spec: PositionSpec,
    batch_size: int,
    device: str
) -> dict:
    """
    Extract hidden states from queries in batches for multiple layers.

    Args:
        queries: List of query strings
        model: The language model
        tokenizer: The tokenizer
        layers: List of layer indices to extract from
        position_spec: Position specification (int or pattern-based)
        batch_size: Number of queries to process at once
        device: Device to use for computation

    Returns:
        Dict mapping layer indices to tensors of shape (num_queries, hidden_dim)
    """
    # Initialize dict to store hidden states for each layer
    all_hidden_states = {layer: [] for layer in layers}

    # Process queries in batches
    for i in tqdm(range(0, len(queries), batch_size), desc="Processing batches"):
        batch_queries = queries[i:i + batch_size]

        # Tokenize with padding
        inputs = tokenizer(
            batch_queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048  # Reasonable max length
        )

        # Move to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        # Extract hidden states from all specified layers in one pass
        # hidden_states is a tuple of length (num_layers + 1)
        # Index 0 is embeddings, index 1 is layer 0, etc.
        
        # Extract hidden state at specified token position for each query in batch
        for batch_idx in range(input_ids.shape[0]):
            query_idx = i + batch_idx
            token_idx = resolve_position_spec(
                position_spec,
                input_ids[batch_idx],
                attention_mask[batch_idx],
                tokenizer,
                query_idx=query_idx
            )

            # Extract the hidden state at the specified position for each layer
            for layer in layers:
                layer_hidden_states = outputs.hidden_states[layer]  # Shape: (batch_size, seq_len, hidden_dim)
                hidden_state = layer_hidden_states[batch_idx, token_idx, :].cpu()
                all_hidden_states[layer].append(hidden_state)

    # Stack all hidden states into tensors for each layer
    return {layer: torch.stack(hidden_states) for layer, hidden_states in all_hidden_states.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Extract hidden states from an LLM at specific layers and token positions"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g., 'Qwen/Qwen2.5-7B-Instruct')"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="Layer index to extract from (0-indexed, includes embedding layer at 0). Cannot be used with --layers."
    )
    parser.add_argument(
        "--layers",
        type=str,
        help="Comma-separated list of layer indices to extract from (e.g., '1,5,10,15'). Cannot be used with --layer."
    )
    parser.add_argument(
        "--position",
        type=str,
        required=True,
        help="Token position: integer (e.g., '-1' for last token) or pattern string (e.g., '<tool_call>')"
    )
    parser.add_argument(
        "--token_offset",
        type=int,
        default=0,
        help="For pattern position: which token in the pattern (0=first, -1=last). Default: 0"
    )
    parser.add_argument(
        "--occurrence",
        type=int,
        default=0,
        help="For pattern position: which occurrence to use (0=first, -1=last). Default: 0"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing queries (default: 8)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="clusters",
        help="Output directory for saved tensors (default: clusters)"
    )

    args = parser.parse_args()

    # Validate layer arguments
    if args.layer is not None and args.layers is not None:
        raise ValueError("Cannot specify both --layer and --layers. Use one or the other.")
    if args.layer is None and args.layers is None:
        raise ValueError("Must specify either --layer or --layers.")
    
    # Parse layers
    if args.layer is not None:
        layers = [args.layer]
    else:
        try:
            layers = [int(x.strip()) for x in args.layers.split(',')]
        except ValueError:
            raise ValueError(f"Invalid --layers format: {args.layers}. Expected comma-separated integers.")

    # Create position specification from CLI args
    position_spec = PositionSpec.from_cli_args(
        args.position, args.token_offset, args.occurrence
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    print(f"Configuration: Layers={layers}, Position={position_spec.to_filename_safe()}, Batch Size={args.batch_size}")

    model_path = args.model
    if "-reason" in model_path:
        model_path = model_path.replace("-reason", "")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with hidden state extraction enabled
    print("Loading model (this may take a while for large models)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        output_hidden_states=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()

    # Determine device
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")

    # Validate layer indices
    num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
    for layer in layers:
        if layer >= num_layers or layer < 0:
            raise ValueError(
                f"Layer index {layer} out of range. Model has {num_layers} layers "
                f"(0=embeddings, 1-{num_layers-1}=transformer layers)"
            )

    # Get model_basename from args.model (same as format_input.py)
    model_basename = args.model.split("/")[-1]

    def extract_data_name_and_raw_data_name(filepath: str, model_basename: str):
        """
        Extract both data_name (from directory path) and raw_data_name (from filename).
        """
        # Extract data_name from directory path
        path_obj = Path(filepath)
        data_dir_parts = path_obj.parent.parts
        
        # Remove 'data' prefix if present
        if len(data_dir_parts) > 0 and data_dir_parts[0] == 'data':
            data_dir_parts = data_dir_parts[1:]
        
        # Join remaining parts with underscore
        if len(data_dir_parts) == 0:
            raise ValueError(f"Cannot extract data_name from path: {filepath}")
        data_name = "_".join(data_dir_parts)
        
        # Extract raw_data_name from filename 
        # Expected format: formatted-{raw_data_name}-{model_basename}.json
        filename = path_obj.stem  # Remove .json extension
        expected_prefix = "formatted-"
        expected_suffix = f"-{model_basename}"

        if filename.startswith(expected_prefix) and filename.endswith(expected_suffix):
            raw_data_name = filename[len(expected_prefix):-len(expected_suffix)]
            return data_name, raw_data_name
        raise ValueError(f"Invalid filename format: {filepath}. Expected: formatted-{{raw_data_name}}-{model_basename}.json")

    data_name, raw_data_name = extract_data_name_and_raw_data_name(args.dataset, model_basename)

    # Create hierarchical output directory: clusters/{model_basename}/{data_name}/{raw_data_name}/
    model_data_output_dir = os.path.join(args.output_dir, model_basename, data_name, raw_data_name)
    os.makedirs(model_data_output_dir, exist_ok=True)

    print(f"\nProcessing Dataset: {args.dataset}")
    queries = load_dataset(args.dataset)
    print(f"Loaded {len(queries)} queries")

    all_layer_hidden_states = extract_hidden_states_batched(
        queries, model, tokenizer, layers, position_spec, args.batch_size, device
    )

    # Save hidden states for each layer
    output_paths = []
    for layer, hidden_states in all_layer_hidden_states.items():
        output_path = os.path.join(
            model_data_output_dir,
            f"{raw_data_name}_L{layer}_{position_spec.to_filename_safe()}.pt"
        )
        torch.save(hidden_states, output_path)
        output_paths.append(output_path)
        print(f"Saved layer {layer} hidden states to: {output_path}")
        print(f"  Shape: {hidden_states.shape}")

    print(f"\n✓ Extraction complete! Saved {len(layers)} layer files.")
    print(f"Output files: {output_paths}")


if __name__ == "__main__":
    main()
