"""GPU preflight for the pinned SM89 grouped kernels; no model weights needed."""
import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bench_io import json_bytes, write_new


def check(out, devices=None):
    import torch
    from scripts.eval_longvideobench_qwen36 import load_backend, routed_experts
    if not torch.cuda.is_available() or torch.__version__.split('+')[0] != '2.8.0' or torch.version.cuda != '12.8':
        raise RuntimeError('Require the pinned PyTorch 2.8.0 + CUDA 12.8 GPU environment')
    if Path(out).exists():
        raise FileExistsError(out)
    backend, provenance = load_backend()
    indices = list(range(torch.cuda.device_count())) if devices is None else [int(i) for i in devices.split(',')]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('Select at least one distinct visible CUDA device')
    results = []
    for index in indices:
        if torch.cuda.get_device_capability(index) != (8, 9):
            raise ValueError('This preflight targets SM89 only')
        with torch.cuda.device(index), torch.inference_mode():
            torch.manual_seed(42)
            device = f'cuda:{index}'
            counts = torch.tensor([3, 0, 5, 9], dtype=torch.long, device='cpu')
            a = torch.randn(17, 128, device=device, dtype=torch.bfloat16) / math.sqrt(128)
            b = torch.randn(4, 128, 64, device=device, dtype=torch.bfloat16)
            nn = backend.gmm(a, b, counts)
            nt = backend.gmm(a, b.transpose(1, 2).contiguous(), counts, trans_b=True)
            reference = torch.cat([part @ b[i] for i, part in enumerate(a.split(counts.tolist()))])
            torch.testing.assert_close(nn, reference, atol=.02, rtol=.02)
            torch.testing.assert_close(nt, reference, atol=.02, rtol=.02)
            experts = SimpleNamespace(num_experts=4,
                gate_up_proj=torch.randn(4, 128, 128, device=device, dtype=torch.bfloat16) / math.sqrt(128),
                down_proj=torch.randn(4, 128, 64, device=device, dtype=torch.bfloat16) / math.sqrt(64),
                act_fn=torch.nn.functional.silu)
            # Expert 2 receives no tokens, exercising zero-sized groups.
            routes = torch.tensor([[i % 2, 3] for i in range(17)], device=device, dtype=torch.int32)
            weights = torch.softmax(torch.randn(17, 2, device=device), dim=-1)
            actual = routed_experts(experts, a, routes, weights, backend)
            expected = torch.zeros_like(a, dtype=torch.float32)
            for expert in (0, 1, 3):
                tokens, slots = torch.where(routes == expert)
                gate, up = torch.nn.functional.linear(a[tokens], experts.gate_up_proj[expert]).chunk(2, dim=-1)
                output = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, experts.down_proj[expert])
                expected.index_add_(0, tokens, output.float() * weights[tokens, slots, None])
            torch.testing.assert_close(actual.float(), expected, atol=.02, rtol=.02)
            torch.cuda.synchronize(index)
            results.append(dict(device=device, name=torch.cuda.get_device_name(index),
                nn_nt_max_abs_error=float(max((nn - reference).abs().max(), (nt - reference).abs().max()).item()),
                routed_max_abs_error=float((actual.float() - expected).abs().max().item())))
    result = dict(schema='bench_runtime_check_v1', torch=torch.__version__, cuda=torch.version.cuda,
        atol=.02, rtol=.02, tests='small BF16 NN/NT + permutation/weighted routed-expert forward, including empty group',
        limitations='Not a long-context, full-model, or performance validation', devices=results, backend=provenance)
    write_new(out, json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--devices', help='Comma-separated visible CUDA indices; default all')
    print(json.dumps(check(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
