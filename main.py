import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'itergen'))

from transformers import PreTrainedModel
from transformers.generation import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

#This function was removed in transformers 4.44 so I'm implementing it here instead
def _get_logits_warper(self, generation_config, device=None):
    warpers = LogitsProcessorList()
    if generation_config.temperature is not None and generation_config.temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(generation_config.temperature))
    if generation_config.top_k is not None and generation_config.top_k != 0:
        warpers.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=1))
    if generation_config.top_p is not None and generation_config.top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=1))
    return warpers

PreTrainedModel._get_logits_warper = _get_logits_warper

import json

from itergen.main import IterGen
from tools import TOOLS
from scenarios import SCENARIOS

MAX_RETRIES = 5
TOOL_NAMES = list(TOOLS.keys())

with open(os.path.join(os.path.dirname(__file__), 'grammar.txt')) as f:
    grammar = f.read()

iter_gen = IterGen(
    grammar=grammar,
    model_id="Qwen/Qwen3-1.7B",
    device='cuda',
    quantize=True,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

# Patch tokenizer to always disable Qwen3 thinking mode
_orig_apply_chat_template = iter_gen.tokenizer.apply_chat_template
def _apply_chat_template_no_think(conversation, **kwargs):
    kwargs['enable_thinking'] = False
    return _orig_apply_chat_template(conversation, **kwargs)
iter_gen.tokenizer.apply_chat_template = _apply_chat_template_no_think


def get_value_type(value):
    """Map a JSON-parsed value to the type string used in TOOLS."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return None


def check_signature(tool_name, args):
    """Return (ok, reason). Accepts integer where float is expected."""
    expected = TOOLS[tool_name]
    for param, expected_type in expected.items():
        if param not in args:
            return False, f"missing argument '{param}'"
        actual_type = get_value_type(args[param])
        if actual_type != expected_type:
            # integer is acceptable where float is expected
            if expected_type == "float" and actual_type == "integer":
                continue
            return False, f"'{param}' expected {expected_type}, got {actual_type}"
    return True, None


def format_tools():
    lines = []
    for name, params in TOOLS.items():
        param_str = ", ".join(f"{p}: {t}" for p, t in params.items())
        lines.append(f"  {name}({param_str})")
    return "\n".join(lines)


def build_prompt(scenario):
    return [
        {
            "role": "system",
            "content": (
                "You are a tool-calling assistant. "
                "Respond with a single JSON tool call in the format "
                "{\"name\": \"<tool_name>\", \"args\": {<arguments>}}. "
                "Use the exact parameter names as defined below.\n"
                f"Available tools:\n{format_tools()}"
            ),
        },
        {
            "role": "user",
            "content": scenario,
        },
    ]


print(f"\n{'='*60}")
for i, scenario in enumerate(SCENARIOS):
    print(f"\nScenario {i+1}: {scenario}")
    prompt = build_prompt(scenario)

    result = None
    retries = 0
    reason = None

    for attempt in range(MAX_RETRIES):
        iter_gen.start(prompt)

        # Generate tool name, record token count before and after
        tokens_before_tool_name = iter_gen.session_tokens.shape[1]
        iter_gen.forward(unit='tool_name', num=1)
        tokens_after_tool_name = iter_gen.session_tokens.shape[1]
        print(f"  [debug] tokens — before tool_name: {tokens_before_tool_name}, after tool_name: {tokens_after_tool_name}, tool_name tokens: {tokens_after_tool_name - tokens_before_tool_name}")

        tool_name_tokens = iter_gen.view('tool_name')
        if not tool_name_tokens or not tool_name_tokens[0]:
            reason = "no tool_name generated"
            retries += 1
            continue
        tool_name = tool_name_tokens[0][-1].strip('"')
        print(f"  [debug] tool_name: {tool_name!r}")

        if tool_name not in TOOLS:
            reason = f"unknown tool '{tool_name}'"
            retries += 1
            continue

        # Generate arglist, record token count after
        for arg_attempt in range(MAX_RETRIES):
            iter_gen.forward()
            tokens_after_args = iter_gen.session_tokens.shape[1]
            args_token_count = tokens_after_args - tokens_after_tool_name
            print(f"  [debug] tokens — after args: {tokens_after_args}, args tokens: {args_token_count}, will rewind: {args_token_count}")

            raw = iter_gen.structured_gen[0]
            print(f"  [debug] generated: {raw!r}")

            try:
                parsed = json.loads(raw)
                args = parsed.get("args", {})
            except json.JSONDecodeError:
                reason = "JSON parse error"
                iter_gen.backward(unit='token', num=args_token_count)
                retries += 1
                continue

            ok, reason = check_signature(tool_name, args)
            if not ok:
                iter_gen.backward(unit='token', num=args_token_count)
                retries += 1
                continue

            result = parsed
            break

        if result:
            break

    if result:
        print(f"  Tool:    {result['name']}")
        print(f"  Args:    {result['args']}")
        print(f"  Status:  PASS (retries: {retries})")
    else:
        print(f"  Status:  FAIL after {MAX_RETRIES} retries — {reason}")

print(f"\n{'='*60}")

