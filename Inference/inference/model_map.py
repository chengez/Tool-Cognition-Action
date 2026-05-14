# Maps model code to handler module name (without .py)
MODEL_TO_HANDLER = {
    # API models
    "gpt-4.1-mini-2025-04-14-FC": "openai",
    "gpt-4.1-nano-2025-04-14-FC": "openai",
    "gpt-4.1-2025-04-14-FC": "openai",
    "gpt-4o-mini-2024-07-18-FC": "openai",
    "gpt-4o-2024-11-20-FC": "openai",
    "o3-2025-04-16-FC": "openai",
    "o4-mini-2025-04-16-FC": "openai",

    "deepseek-chat": "deepseek",

    # Local models
    "meta-llama/Llama-3.1-8B-Instruct": "llama3_1",
    "meta-llama/Llama-3.2-3B-Instruct": "llama3_2",
    "Qwen/Qwen3-0.6B": "qwen3",
    "Qwen/Qwen3-1.7B": "qwen3",
    "Qwen/Qwen3-4B": "qwen3",
    "Qwen/Qwen3-8B": "qwen3",
    "Qwen/Qwen3-14B": "qwen3",
    "Qwen/Qwen3-32B": "qwen3",
    "Qwen/Qwen3-0.6B-reason": "qwen3_reason",
    "Qwen/Qwen3-1.7B-reason": "qwen3_reason",
    "Qwen/Qwen3-4B-reason": "qwen3_reason",
    "Qwen/Qwen3-8B-reason": "qwen3_reason",
    "Qwen/Qwen3-14B-reason": "qwen3_reason",
    "Qwen/Qwen3-32B-reason": "qwen3_reason",
    "Qwen/Qwen2.5-7B-Instruct": "qwen2_5",
    "mistralai/Ministral-8B-Instruct-2410": "ministral",


    # saved models by DPO
    "saved_dpo_models/Meta-Llama-3.1-8B-Instruct":"llama3_1",
    "saved_dpo_models/Llama-3.2-3B-Instruct":"llama3_2",
    "saved_dpo_models/Ministral-8B-Instruct-2410":"ministral",
    "saved_dpo_models/Qwen3-8B":"qwen3",
    "saved_dpo_models/Qwen3-4B":"qwen3",
    "saved_dpo_models/Qwen2.5-7B-Instruct":"qwen2_5",
}

MODEL_TO_TOOLCALL_SIGNATURE = {
    "gpt-4.1-mini-2025-04-14-FC": "ChatCompletionMessageToolCall",
    "gpt-4.1-nano-2025-04-14-FC": "ChatCompletionMessageToolCall",
    "gpt-4.1-2025-04-14-FC": "ChatCompletionMessageToolCall",
    "gpt-4o-mini-2024-07-18-FC": "ChatCompletionMessageToolCall",
    "gpt-4o-2024-11-20-FC": "ChatCompletionMessageToolCall",
    "o3-2025-04-16-FC": "ChatCompletionMessageToolCall",
    "o4-mini-2025-04-16-FC": "ChatCompletionMessageToolCall",

    "deepseek-chat": "ChatCompletionMessageToolCall",


    "meta-llama/Llama-3.1-8B-Instruct": "\"name\"<AND>\"parameters\"",
    "meta-llama/Llama-3.2-3B-Instruct": "\"name\"<AND>\"parameters\"",
    "Qwen/Qwen3-0.6B": "<tool_call><AND></tool_call>",
    "Qwen/Qwen3-1.7B": "<tool_call><AND></tool_call>",
    "Qwen/Qwen3-4B": "<tool_call><AND></tool_call>",
    "Qwen/Qwen3-8B": "<tool_call><AND></tool_call>",
    "Qwen/Qwen3-14B": "<tool_call><AND></tool_call>",
    "Qwen/Qwen3-32B": "<tool_call><AND></tool_call>",
    "Qwen/Qwen3-0.6B-reason": "<tool_call><AND></tool_call>",
    "Qwen/Qwen3-1.7B-reason": "<tool_call><AND></tool_call>",
    "Qwen/Qwen3-4B-reason": "<tool_call><AND></tool_call>",
    "Qwen/Qwen3-8B-reason": "<tool_call><AND></tool_call>",
    "Qwen/Qwen3-14B-reason": "<tool_call><AND></tool_call>",
    "Qwen/Qwen3-32B-reason": "<tool_call><AND></tool_call>",
    "Qwen/Qwen2.5-7B-Instruct": "<tool_call><AND></tool_call>",
    "mistralai/Ministral-8B-Instruct-2410": "\"arguments\"<AND>\"name\"",


    "saved_dpo_models/Meta-Llama-3.1-8B-Instruct":"\"name\"<AND>\"parameters\"",
    "saved_dpo_models/Llama-3.2-3B-Instruct":"\"name\"<AND>\"parameters\"",
    "saved_dpo_models/Ministral-8B-Instruct-2410":"\"arguments\"<AND>\"name\"",
    "saved_dpo_models/Qwen3-8B":"<tool_call><AND></tool_call>",
    "saved_dpo_models/Qwen3-4B":"<tool_call><AND></tool_call>",
    "saved_dpo_models/Qwen2.5-7B-Instruct":"<tool_call><AND></tool_call>",
}


