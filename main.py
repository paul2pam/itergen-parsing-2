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


def build_prompt(scenario):
    return [
        {
            "role": "system",
            "content": (
                "/no_think "
                "You are a tool-calling assistant. "
                "Respond with a single JSON tool call in the format "
                "{\"name\": \"<tool_name>\", \"args\": {<arguments>}}. "
                f"Available tools: {', '.join(TOOL_NAMES)}."
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
    iter_gen.start(prompt)
    print("Now we start generating attempts")

    result = None
    retries = 0
    reason = None

    for attempt in range(MAX_RETRIES):
        iter_gen.start(prompt)

        # Phase 1: generate and validate tool_name, clean restart on failure
        iter_gen.forward(unit='tool_name', num=1)
        print(f"  [debug] after tool_name forward: {iter_gen.structured_gen[0]!r}")
        tool_name_tokens = iter_gen.view('tool_name')
        if not tool_name_tokens or not tool_name_tokens[0]:
            reason = "no tool_name generated"
            retries += 1
            continue
        tool_name = tool_name_tokens[0][-1].strip('"')
        if tool_name not in TOOLS:
            reason = f"unknown tool '{tool_name}'"
            retries += 1
            continue

        # Phase 2: generate and validate arglist, backtracking arglist on failure
        arg_retries = 0
        for arg_attempt in range(MAX_RETRIES):
            iter_gen.forward()  # generates arglist + closing } to complete start
            raw = iter_gen.structured_gen[0]
            try:
                parsed = json.loads(raw)
                args = parsed.get("args", {})
            except json.JSONDecodeError:
                reason = "JSON parse error"
                iter_gen.backward(unit='arglist', num=1)
                arg_retries += 1
                continue
            ok, reason = check_signature(tool_name, args)
            if not ok:
                iter_gen.backward(unit='arglist', num=1)
                arg_retries += 1
                continue
            result = parsed
            retries += arg_retries
            break

        if result:
            break
        retries += arg_retries

    if result:
        print(f"  Tool:    {result['name']}")
        print(f"  Args:    {result['args']}")
        print(f"  Status:  PASS (retries: {retries})")
    else:
        print(f"  Status:  FAIL after {MAX_RETRIES} retries — {reason}")

print(f"\n{'='*60}")

