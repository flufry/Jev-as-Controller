from collections import Counter
import json
from types import SimpleNamespace

import pytest
import torch

from scripts import eval_longvideobench_qwen36 as evaluator


def config():
    return SimpleNamespace(model_type='qwen3_5_moe',
        architectures=['Qwen3_5MoeForConditionalGeneration'], tie_word_embeddings=False,
        text_config=SimpleNamespace(num_hidden_layers=40, num_experts=256,
                                    num_experts_per_tok=8, tie_word_embeddings=False))


@pytest.mark.parametrize('count', [4, 5, 6])
def test_complete_layers_and_visual_input_placement(count):
    plan = evaluator.device_plan(config(), count)
    layers = [plan[f'model.language_model.layers.{index}'] for index in range(40)]
    assert set(layers) == {f'cuda:{index}' for index in range(count)}
    assert layers == sorted(layers)
    sizes = Counter(layers)
    assert max(sizes.values())-min(sizes.values()) <= 1
    assert plan['model.visual'] == plan['model.language_model.embed_tokens'] == 'cuda:0'
    assert plan['lm_head'] == plan['model.language_model.norm'] == f'cuda:{count-1}'
    assert len(plan) == 45 and all(value.startswith('cuda:') for value in plan.values())


@pytest.mark.parametrize('change', [dict(model_type='qwen3_5'), dict(architectures=['OtherModel']),
    dict(quantization_config={}), dict(tie_word_embeddings=True)])
def test_wrong_or_quantized_architecture_rejected(change):
    model_config = config()
    vars(model_config).update(change)
    with pytest.raises(ValueError):
        evaluator.device_plan(model_config, 6)


class Backend:
    """CPU 分组参考；循环仅用于测试，不进入生产后端。"""
    def permute(self, hidden, indices, num_out, workspace, expanded):
        assert indices.dtype == torch.int32
        assert num_out == 0 and not workspace and expanded == indices.numel()
        order = torch.argsort(indices.reshape(-1), stable=True)
        tokens = torch.arange(len(hidden)).repeat_interleave(indices.shape[1])
        return hidden[tokens[order]], torch.argsort(order).reshape(indices.shape), []

    def gmm(self, hidden, weight, counts, trans_b):
        assert trans_b and counts.device.type == 'cpu' and counts.dtype == torch.int64
        chunks, offset = [], 0
        for expert, count in enumerate(counts.tolist()):
            chunks.append(hidden[offset:offset+count] @ weight[expert].T)
            offset += count
        assert offset == len(hidden)
        return torch.cat(chunks)

    def unpermute(self, hidden, inverse, weights, count, topk):
        assert weights.dtype == torch.float32 and inverse.shape == (count, topk)
        return (hidden[inverse].float()*weights[..., None]).sum(dim=1).to(hidden.dtype)


def experts():
    torch.manual_seed(42)
    return SimpleNamespace(num_experts=4, act_fn=torch.nn.functional.silu,
        gate_up_proj=torch.randn(4, 32, 16, dtype=torch.bfloat16)*.05,
        down_proj=torch.randn(4, 16, 16, dtype=torch.bfloat16)*.05)


def test_grouped_math_keeps_weights_routing_and_unused_experts():
    module = experts()
    hidden = torch.randn(7, 16, dtype=torch.bfloat16)
    indices = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1], [1, 0]])
    weights = torch.tensor([[.25, .75]]*7, dtype=torch.bfloat16)
    unchanged = [module.gate_up_proj.clone(), module.down_proj.clone(), indices.clone(), weights.clone()]
    result = evaluator.routed_experts(module, hidden, indices, weights, Backend())
    expected = torch.zeros_like(hidden)
    for token in range(len(hidden)):
        for slot in range(2):
            expert = indices[token, slot]
            gate, up = torch.nn.functional.linear(hidden[token], module.gate_up_proj[expert]).chunk(2)
            expected[token] += torch.nn.functional.linear(module.act_fn(gate)*up,
                module.down_proj[expert])*weights[token, slot]
    torch.testing.assert_close(result, expected, rtol=.03, atol=.0001)
    for actual, old in zip([module.gate_up_proj, module.down_proj, indices, weights], unchanged):
        assert torch.equal(actual, old)
    assert evaluator.routed_experts(module, hidden[:0], indices[:0], weights[:0], Backend()).shape == (0, 16)
    with pytest.raises(ValueError, match='BF16'):
        evaluator.routed_experts(module, hidden.float(), indices, weights, Backend())


@pytest.mark.parametrize('collection', [list, set])
def test_loader_dispatch_and_no_quantization_or_offload(monkeypatch, collection):
    seen = {}
    diagnostics = {key: collection() for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')}

    class Model:
        def eval(self):
            return self

        def requires_grad_(self, enabled):
            assert not enabled
            return self

    class Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            seen.update(path=path, **kwargs)
            return Model(), diagnostics

    monkeypatch.setattr(evaluator, 'validate_loaded', lambda *_: {'checked': True})
    monkeypatch.setattr(evaluator, 'install_backend', lambda *_: 40)
    args = SimpleNamespace(checkpoint='local/official', gpu_count=6, max_memory_gib=18.5)
    model, report = evaluator.load_model(args, config(), object(), model_class=Loader)
    assert seen['local_files_only'] is True and seen['dtype'] == torch.bfloat16
    assert seen['attn_implementation'] == 'flash_attention_2' and seen['experts_implementation'] == 'eager'
    assert seen['device_map'] == evaluator.device_plan(config(), 6)
    assert set(seen['max_memory']) == set(range(6))
    assert all(value == int(18.5*2**30) for value in seen['max_memory'].values())
    assert 'quantization_config' not in seen and 'offload_folder' not in seen
    assert report['patched_expert_layers'] == 40 and report['checked']
    assert json.loads(json.dumps(report))['loading_info'] == {key: [] for key in diagnostics}
    assert all(isinstance(value, collection) for value in diagnostics.values())


@pytest.mark.parametrize('key', ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'])
def test_nonempty_loading_sets_are_rejected_before_audit(monkeypatch, key):
    diagnostics = {name: set() for name in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')}
    diagnostics[key] = {'invalid_weight'}

    class Loader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object(), diagnostics

    monkeypatch.setattr(evaluator, 'validate_loaded', lambda *_: pytest.fail('不能审计不完整模型'))
    monkeypatch.setattr(evaluator, 'install_backend', lambda *_: pytest.fail('不能使用不完整模型'))
    args = SimpleNamespace(checkpoint='local/official', gpu_count=6, max_memory_gib=18.5)
    with pytest.raises(ValueError, match='模型加载不完整'):
        evaluator.load_model(args, config(), object(), model_class=Loader)


def test_install_keeps_official_router_and_shared_expert_unchanged():
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
    tiny = SimpleNamespace(hidden_size=16, num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=16, shared_expert_intermediate_size=16, hidden_act='silu',
        _experts_implementation='eager')
    block = Qwen3_5MoeSparseMoeBlock(tiny)
    router_forward = block.gate.forward.__func__
    shared_forward = block.shared_expert.forward.__func__
    parameters = {name: parameter for name, parameter in block.named_parameters()}
    model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=1)),
                            modules=block.modules)
    assert evaluator.install_backend(model, Backend()) == 1
    assert block.gate.forward.__func__ is router_forward
    assert block.shared_expert.forward.__func__ is shared_forward
    assert all(parameters[name] is parameter for name, parameter in block.named_parameters())
    with pytest.raises(RuntimeError, match='禁止 CPU'):
        block.experts(torch.empty(1, 16), torch.tensor([[0, 1]]), torch.tensor([[.5, .5]]))


@pytest.mark.parametrize('kind', ['quantized', 'map', 'dtype', 'cpu', 'budget'])
def test_actual_loaded_state_is_strictly_checked(kind):
    plan = evaluator.device_plan(config(), 4)
    parameters = [(f'p{gpu}', SimpleNamespace(device=torch.device(f'cuda:{gpu}'), dtype=torch.bfloat16,
                   numel=lambda: 10, element_size=lambda: 2)) for gpu in range(4)]
    model = SimpleNamespace(is_quantized=False, config=config(), hf_device_map=plan,
                            named_parameters=lambda: iter(parameters))
    budgets = {gpu: 100 for gpu in range(4)}
    if kind == 'quantized':
        model.is_quantized = True
    elif kind == 'map':
        model.hf_device_map = dict(plan, lm_head='cpu')
    elif kind in ('dtype', 'cpu'):
        setattr(parameters[0][1], 'dtype' if kind == 'dtype' else 'device',
                torch.float32 if kind == 'dtype' else torch.device('cpu'))
    else:
        budgets[0] = 1
    with pytest.raises(ValueError):
        evaluator.validate_loaded(model, plan, budgets)
