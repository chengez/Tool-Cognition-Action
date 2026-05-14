import json
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Inference'))

from inference.model_map import MODEL_TO_HANDLER
from eval_from_local import get_handler, load_data

def parse_args():
    parser = argparse.ArgumentParser(description="Format input history for LLM function calling.")
    parser.add_argument("--raw_data", type=str, default="data/example.json", help="Path to data JSON file.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model code (will be mapped to handler).")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save output JSON files.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    raw_data_filename = os.path.basename(args.raw_data)
    handler_name = MODEL_TO_HANDLER.get(args.model)
    if handler_name is None:
        raise ValueError(f"Unknown model code: {args.model}")
    handler = get_handler(handler_name, model_path=args.model, load_model=False)
    data = load_data(args.raw_data)
    formatted_data = []
    for sample in data:
        history = sample["history"]
        tools = sample.get("function", None)
        formatted_history = handler.format_input(history, tools=tools, use_time_stamp=False)
        formatted_data.append({
            "id": sample["id"],
            "query": formatted_history,
        })
    model_basename = args.model.split("/")[-1]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = f"{args.output_dir}/formatted-{raw_data_filename.split('.')[0]}-{model_basename}.json"
    with open(output_path, "w") as f:
        json.dump(formatted_data, f, indent=4)
    print(f"Formatted data saved to {output_path}")