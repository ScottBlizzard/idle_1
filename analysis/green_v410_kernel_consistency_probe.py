"""Real-width synthetic kernel Taylor checks, isolated from formal outputs."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import numpy as np
from green_bridge_v400_compiled_mpfr import CompiledMPFRBackend
from green_bridge_v400_interval import Interval
from green_bridge_v400_interval_jet import affine_control_jet, constant_jet
from green_bridge_v400_mpfr_tensor_executor import _decode_jet
from green_v410_fixed_budget_certificate import certify_five_outputs, OUTPUT_KEYS

class Evaluator:
    def __init__(self, backend, width, scale):
        self.backend = backend
        rng = np.random.default_rng(719)
        self.base = rng.normal(size=width)
        self.direction = rng.normal(size=width) * scale
        self.gamma = np.ones(width, dtype=np.float32)
        self.beta = np.zeros(width, dtype=np.float32)

    def evaluate(self, domain):
        b = self.backend
        jets = [affine_control_jet(Interval.point(x, domain.precision_bits),
                Interval.point(d, domain.precision_bits), domain)
                for x, d in zip(self.base, self.direction)]
        with b.resident_jet_buffer(jets) as buf:
            with b.resident_layer_norm_jet2(buf, np.float32(1e-5), self.gamma, self.beta) as ln:
                result = _decode_jet(b.export_resident_jet_buffer(ln)['outputs'][0], domain.precision_bits)
        return {name: result for name in OUTPUT_KEYS}

if __name__ == '__main__':
    backend = CompiledMPFRBackend(Path(sys.argv[1]))
    p = 384
    point = lambda x: constant_jet(Interval.point(x, p))
    with backend.resident_jet_buffer([point(1000000)]) as q:
        with backend.resident_jet_buffer([point(0), point(1000000)]) as k:
            with backend.resident_jet_buffer([point(1), point(1)]) as v:
                with backend.resident_causal_attention_all_heads_jet2(q, k, v, 2, 1, 1, 0) as a:
                    print('SATURATED_ATTENTION_CONSTANT_ONE', backend.export_resident_jet_buffer(a), flush=True)
    for width in (3, 4, 768):
        for scale in (1e-3, 1e-9):
            certify_five_outputs(Evaluator(backend, width, scale), leaf_budget=4)
            print('PASS', width, scale, flush=True)
