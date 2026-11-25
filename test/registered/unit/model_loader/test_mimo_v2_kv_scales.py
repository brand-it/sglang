import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.models.mimo_v2 import MiMoV2Model
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestMiMoV2KVScales(unittest.TestCase):
    def test_scales_are_available_to_cache_and_attention(self):
        data = {
            "kv_cache": {
                "scaling_factor": {
                    "0": {"0": {"k_scale": 0.125, "v_scale": 0.25}, "1": 0.5},
                    "1": {"0": {"k_scale": 0.0625, "v_scale": 0.125}, "1": 0.25},
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scales.json"
            path.write_text(json.dumps(data))
            for rank in (0, 1):
                with self.subTest(rank=rank):
                    layers = [
                        SimpleNamespace(
                            self_attn=SimpleNamespace(
                                attn=SimpleNamespace(
                                    k_scale=None,
                                    v_scale=None,
                                    k_scale_float=None,
                                    v_scale_float=None,
                                )
                            )
                        )
                        for _ in range(2)
                    ]
                    MiMoV2Model.load_kv_cache_scales(
                        SimpleNamespace(layers=layers, _kv_cache_parallel_layout=(rank, 2)),
                        str(path),
                    )
                    for index, layer in enumerate(layers):
                        scale = data["kv_cache"]["scaling_factor"][str(rank)][str(index)]
                        key = scale["k_scale"] if isinstance(scale, dict) else scale
                        value = scale["v_scale"] if isinstance(scale, dict) else scale
                        attn = layer.self_attn.attn
                        self.assertEqual(attn.k_scale, key)
                        self.assertEqual(attn.v_scale, value)
                        self.assertEqual(attn.k_scale_float, key)
                        self.assertEqual(attn.v_scale_float, value)
                        self.assertEqual(0.5 * attn.k_scale_float, 0.5 * key)


if __name__ == "__main__":
    unittest.main()
