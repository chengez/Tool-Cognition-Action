from inference.model_handler import Base_Handler
from jinja2 import Environment, FileSystemLoader
import os
import torch
from inference.sys_pmts import *
def raise_exception(message):
    raise Exception(message)
class Ministral_Handler(Base_Handler):
    SAMPLING = {"temperature": 0.0, "max_tokens": 2048}

    def __init__(self, model_path="mistralai/Ministral-8B-Instruct-2410", load_model=True, use_api=False,
                 base_url="http://localhost:8000/v1", api_key="EMPTY"):
        super().__init__("ministral")
        self.model_path = model_path
        self.use_api = use_api
        self.sampling = dict(self.SAMPLING)
        if use_api:
            from inference.local._vllm_api import make_client
            self.client = make_client(base_url=base_url, api_key=api_key)
        elif load_model:
            from vllm import LLM, SamplingParams
            self.llm = LLM(model=self.model_path, tensor_parallel_size=torch.cuda.device_count())
            self.sampling_params = SamplingParams(**self.SAMPLING)

    def format_input(self, history, tools=None, add_generation_prompt=True, time_elapsed_level=0, use_time_stamp=False, use_special_sys_prompt_naive=False, use_special_sys_prompt_rule=False, add_sys_start_msg=True, final_time_string=None):
        """
        Format the input history for Ministral using the Jinja template.
        Args:
            history: List of message dicts (role/content/tool_calls/etc)
            tools: List of tool/function definitions (from the 'function' field in data)
            add_generation_prompt: Whether to add assistant generation prompt (default True)
        Returns:
            Rendered prompt string
        """
        env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), '../../templates')),
                          trim_blocks=True, lstrip_blocks=True)
        template = env.get_template('ministral.jinja')
        messages = []
        for msg in history:
            m = {"role": msg["role"]}
            if use_time_stamp:
                m["time"] = msg['time']
            time_string = msg.get('time') if msg.get('time') and type(msg.get('time')) is str else (msg.get('time')[time_elapsed_level] if msg.get('time') else '')
            if msg.get("content") is not None and "tool_calls" not in msg:
                m["content"] = f"[{time_string}] " + msg["content"] if use_time_stamp else msg["content"]
                if m["role"] == "system" and use_special_sys_prompt_naive:
                    m["content"] = m["content"] + NAIVE
                elif m["role"] == "system" and use_special_sys_prompt_rule:
                    m["content"] = m["content"] + RULE
            if "tool_calls" in msg:
                m["tool_calls"] = msg["tool_calls"]
                m["content"] = None
            if msg["role"] == "tool" and "tool_call_id" in msg:
                m["tool_call_id"] = msg["tool_call_id"]
                if len(msg["tool_call_id"]) != 9:
                    raise ValueError(f"Tool call ID {msg['tool_call_id']} is not 9 characters long.")
            messages.append(m)
        rendered = template.render(
            bos_token="<s>",
            eos_token="</s>",
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            add_sys_start_msg=add_sys_start_msg,
            raise_exception=raise_exception,
            use_time_stamp=use_time_stamp,
            final_time_string=final_time_string,
        )
        return rendered

    def run_inference(self, formatted_inputs):
        """
        Run batch inference using vllm (in-process) or the vLLM OpenAI-compatible API.
        Args:
            formatted_inputs: List of formatted prompt strings
        Returns:
            List of output strings
        """
        if self.use_api:
            from inference.local._vllm_api import run_completions
            return run_completions(
                self.client, self.model_path, formatted_inputs,
                sampling=self.sampling,
            )
        outputs = self.llm.generate(formatted_inputs, self.sampling_params)
        return [output.outputs[0].text.strip() if output.outputs else "" for output in outputs]
