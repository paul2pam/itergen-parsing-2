import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'itergen'))

from transformers import PreTrainedModel
from transformers.generation import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

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

from itergen.main import IterGen
from tools import TOOLS

with open(os.path.join(os.path.dirname(__file__), 'grammar.txt')) as f:
    grammar = f.read()

iter_gen = IterGen(
    grammar=grammar,
    model_id="Qwen/Qwen3-1.7B",
    device='cuda',
    quantize=True,
)

