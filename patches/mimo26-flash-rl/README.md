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
