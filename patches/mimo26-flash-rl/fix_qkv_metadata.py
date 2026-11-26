"""Correct MiMo-V2.6-Flash-RL quantization metadata from stdin to stdout."""

import json
import sys

config = json.load(sys.stdin)
quantization = config["quantization"]
if quantization["quant_algo"] != "MIXED_PRECISION":
    raise ValueError("Expected a MIXED_PRECISION checkpoint")

excluded = quantization["exclude_modules"]
if "*.self_attn.*" in excluded:
    index = excluded.index("*.self_attn.*")
    excluded[index] = "*.self_attn.o_proj"
elif "*.self_attn.o_proj" not in excluded:
    raise ValueError("Expected an attention exclusion for this checkpoint")

layers = quantization["quantized_layers"]
for layer_id in range(48):
    name = f"model.layers.{layer_id}.self_attn.qkv_proj"
    entry = layers.setdefault(name, {"quant_algo": "FP8_PB_WO"})
    if entry["quant_algo"] != "FP8_PB_WO":
        raise ValueError(f"Unexpected quantization for {name}: {entry}")

for name in (
    "model.decoder.self_attn.qkv_proj",
    "model.decoder.mlp.gate_up_proj",
    "model.decoder.mlp.down_proj",
):
    entry = layers.setdefault(name, {"quant_algo": "FP8_PB_WO"})
    if entry["quant_algo"] != "FP8_PB_WO":
        raise ValueError(f"Unexpected quantization for {name}: {entry}")

json.dump(config, sys.stdout, indent=2)
sys.stdout.write("\n")
