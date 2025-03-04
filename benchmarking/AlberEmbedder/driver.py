import json
import math
import os
import re
import time
import glob
from types import SimpleNamespace

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from PIL import Image, ImageOps
from transformers import AutoTokenizer

from AlbersEmbed import _generate
from AlbersEmbed import *


def _apply_chat_template(prompt, images, verbose, apply_chat_template=True):
    if apply_chat_template is False:
        print(f'*** Prompt ***\n{prompt}\n*** Images ***\n{images}\n*** Output ***') if verbose else None
        return prompt, images
    if images is not None:
        images = [i for i in images] if isinstance(images, list) else [images]
        img_prompt = '\n'.join([f'<|image_{i+1}|>' for i in range(len(images))]) + '\n'
    else:
        img_prompt = ''
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [f"<|user|>\n{img_prompt}{i.strip()}<|end|>\n<|assistant|>\n" for i in prompt]
    if verbose:
        prompt_str = "\n".join(map(str.strip, prompt)).strip()
        images_str = "\n".join(map(str, images)) if images else "None"
        print(f'*** Prompt ***\n{prompt_str}\n*** Images ***\n{images_str}\n*** Output ***')
    prompt = prompt[0] if len(prompt) == 1 else prompt
    return prompt, images



model_path = "/Users/sean/Documents/safeTensors/Phi-3-vision-128k-instruct"
with open(f"{model_path}/config.json", "r") as f:
    config = json.load(f)

model_config = SimpleNamespace(**config)

model = Phi3VForCausalLM(model_config)

model_weight = [(k, v.transpose(0, 2, 3, 1) if "patch_embedding.weight" in k else v) for wf in glob.glob(f"{model_path}/*.safetensors") for k, v in mx.load(wf).items()]
model.load_weights(model_weight)
mx.eval(model.parameters())
model.eval()

processor = Phi3VProcessor(model_path)

# def generate(prompt, images=None, preload=None, blind_model=False, quantize_model=False, quantize_cache=False, use_adapter=False, max_tokens=512, verbose=True, return_tps=False, early_stop=False, stream=True, apply_chat_template=True, enable_api=False):
prompt = "what is it? describe it in detail, be interpretive"
images= Image.open("/Users/sean/git/MLX/AlberEmbedder/artworks-q0Od83shXkoy3s2Y-NayCKQ-t500x500.jpg")
# images= None
preload=model, processor
blind_model=False
quantize_model=False
quantize_cache=False
use_adapter=False
max_tokens=512
verbose=True
return_tps=False
early_stop=False
stream=True
apply_chat_template=True
enable_api=False

A = _generate(*preload, *_apply_chat_template(prompt, images, verbose, apply_chat_template), max_tokens=max_tokens, verbose=verbose, return_tps=return_tps, early_stop=early_stop, stream=stream)

print()

print(A[0])
i = A[1]
print(len(i), type(i))

