# MiMo-V2.6-Flash-RL (NVFP4) boot fix — layer-0 manifest inconsistency

## Symptom
Boot of r0b0tlab/MiMo-V2.6-Flash-RL-NVFP4 on 2x GB10 (TP2) crashes in
SGLang weight load:

    KeyError: "model.layers.0.mlp.gate_up_proj.weight_scale_inv"

## Root cause
The checkpoint manifest (hf_quant_config.json, quantization section) and the
weights disagree about **layer 0**:

- exclude_modules lists  model.layers.0.mlp.*   ("layer 0 MLP is NOT quantized")
- quantized_layers has   0 entries for layer 0   (first key is model.layers.1.mlp.*)
- BUT the safetensors DO contain  model.layers.0.mlp.{gate,up,down}_proj.weight_scale_inv
  (layer 0 dense MLP is genuinely block-FP8)

So SGLang registers layer 0 as unquantized (no weight_scale_inv param), then the
loader hits params_dict["...gate_up_proj.weight_scale_inv"] -> KeyError.

This is an export inconsistency in the r0b0tlab checkpoint, NOT a SGLang bug:
the load path is identical across all tested SGLang commits (582389ce .. be2079f).

## Fix (data, not code)
Remove model.layers.0.mlp.* from exclude_modules so layer 0 resolves to
block-FP8 (matching the actual weights) and the scales are loaded:

    exclude_modules:
      before: [..., "model.layers.0.mlp.*", ...]
      after:  [...]

See layer0_exclude_modules.patch (the exact one-line JSON change).

## Why not a code patch
A code patch (skip the missing scale) would load layer 0 as BF16 and discard
the block-FP8 scales that ARE in the checkpoint — i.e. it would run the model
differently from what its weights represent. Fixing the manifest loads the
model as it actually is.

## QKV metadata and unreadable responses

The checkpoint stores block-FP8 QKV weights and scales for all 48 layers.
Its manifest omits their quantization entries and excludes `*.self_attn.*`.
SGLang therefore loads the raw FP8 values as unquantized weights and skips the
scales. The model can start but produce unreadable responses.

`fix_qkv_metadata.py` adds the missing `FP8_PB_WO` entries and narrows the
exclusion to `*.self_attn.o_proj`, which is BF16. Apply it after the layer-0
manifest fix. It reads JSON from standard input and writes JSON to standard
output, so the input remains unchanged:

```bash
python3 patches/mimo26-flash-rl/fix_qkv_metadata.py \
  < /path/to/hf_quant_config.json > /path/to/hf_quant_config.corrected.json
```

Replace the model's config with the corrected file, sync it to the worker,
and restart the model. The fused TP4-to-TP2 loader in this branch dequantizes,
reorders, and requantizes the QKV weights and scales. With the corrected config,
the model returned `4` for `2 + 2` and a coherent explanation of the blue sky.

The same script adds block-FP8 metadata for the EAGLE MTP draft projections.
The draft checkpoint contains FP8 weights and scales for these layers. Without
the entries, the draft loader cannot load the scales. The draft's fused QKV
loader also reorders the TP4 weights and scales for TP2.
