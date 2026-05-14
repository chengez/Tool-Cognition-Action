from inference.model_handler import Base_Handler
from jinja2 import Environment, FileSystemLoader
import os
import torch
from inference.sys_pmts import *
def raise_exception(message):
    raise Exception(message)

class Llama3_2_Handler(Base_Handler):
    SAMPLING = {"temperature": 0.0, "max_tokens": 4096}

    def __init__(self, model_path="meta-llama/Llama-3.2-3B-Instruct", load_model=True, use_api=False,
                 base_url="http://localhost:8000/v1", api_key="EMPTY"):
        super().__init__("llama3_2")
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

    def format_input(self, history, tools=None, tools_in_user_message=False, date_string="26 Jul 2024", add_generation_prompt=True, custom_tools=None, builtin_tools=None, time_elapsed_level=0, use_time_stamp=False, use_special_sys_prompt_naive=False, use_special_sys_prompt_rule=False, add_sys_start_msg=True, final_time_string=None):
        """
        Format the input history for Llama3.2 using the Jinja template.
        Args:
            history: List of message dicts (role/content/tool_calls/etc)
            tools: List of tool/function definitions (from the 'function' field in data)
            tools_in_user_message: Whether to include tools in user message (default False)
            date_string: Date string for prompt (default "26 Jul 2024")
            add_generation_prompt: Whether to add assistant generation prompt (default True)
            custom_tools: Custom tools (optional)
            builtin_tools: Builtin tools (optional)
        Returns:
            Rendered prompt string
        """
        env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), '../../templates')),
                          trim_blocks=True, lstrip_blocks=True)
        template = env.get_template('llama3_2.jinja')
        messages = []
        sys_date_str = history[0].get('time', '2024-07-26').split('T')[0]
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
            messages.append(m)

        rendered = template.render(
            bos_token="<|begin_of_text|>",
            eos_token="<|eot_id|>",
            messages=messages,
            tools=tools,
            tools_in_user_message=tools_in_user_message,
            date_string=sys_date_str,
            add_generation_prompt=add_generation_prompt,
            add_sys_start_msg=add_sys_start_msg,
            raise_exception=raise_exception,
            use_time_stamp=use_time_stamp,
            final_time_string=final_time_string
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
