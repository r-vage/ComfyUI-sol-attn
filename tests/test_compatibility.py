"""CPU regressions for production wrappers; only external backends are mocked."""
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_wrappers():
    package = types.ModuleType('sol_cpu_tests')
    package.__path__ = [str(ROOT)]
    backend = types.ModuleType('sol_cpu_tests.sol_kernel')
    backend.sol_attn = Mock(side_effect=lambda q, k, v, **kw: v)
    with patch.dict(sys.modules, {'sol_cpu_tests': package, 'sol_cpu_tests.sol_kernel': backend,
                                 'comfy.ldm.minimax.model': None,
                                 'comfy.model_management': None, 'comfy.quant_ops': None}):
        nodes = importlib.import_module('sol_cpu_tests.nodes')
        minimax = importlib.import_module('sol_cpu_tests.minimax')
    return nodes, minimax


nodes, h3 = load_wrappers()


class CudaTensor(torch.Tensor):
    """CPU storage with a mocked device boundary; no CUDA/Triton arithmetic."""
    @property
    def device(self):
        return torch.device('cuda:0')


def activation(*shape):
    return torch.zeros(shape, dtype=torch.bfloat16).as_subclass(CudaTensor)


def layout(start=64, stop=128, signature=(4, 1, 2, 2, 2)):
    return types.SimpleNamespace(signature=signature, seq_len=stop,
                                 segments=[(0, start, 'text'), (start, stop, 'video')])


class GenericTests(unittest.TestCase):
    def setUp(self):
        self.q = activation(1, 1, 128, 128)
        self.fallback = Mock(return_value='dense')
        self.backend = patch.object(nodes, 'sol_attn', side_effect=lambda q, k, v, **kw: v).start()
        self.addCleanup(patch.stopall)
        patch.object(torch.cuda, 'get_device_capability', return_value=(8, 9)).start()
        self.run_attention = nodes._make_override(1.3, 1, True)

    def test_masks_and_original_fallback_arguments(self):
        mask = object()
        for args, kw in [((mask, None, True, False), {}), ((), {'mask': mask, 'skip_reshape': True})]:
            with self.subTest(args=args):
                self.assertEqual(self.run_attention(self.fallback, self.q, self.q, self.q, 1, *args, **kw), 'dense')
                self.fallback.assert_called_with(self.q, self.q, self.q, 1, *args, **kw)
        self.backend.assert_not_called()

    def test_prior_override_gets_untouched_arguments(self):
        prior = Mock(return_value='prior')
        run = nodes._make_override(1.3, 1, True, prior)
        mask = object()
        self.assertEqual(run(self.fallback, self.q, self.q, self.q, 1, mask, skip_reshape=True), 'prior')
        prior.assert_called_once_with(self.fallback, self.q, self.q, self.q, 1, mask, skip_reshape=True)

    def test_reshape_flags_positional_and_keyword(self):
        for positional in (True, False):
            for output_4d in (True, False):
                args = (None, None, True, output_4d) if positional else ()
                kw = {} if positional else {'skip_reshape': True, 'skip_output_reshape': output_4d}
                out = self.run_attention(self.fallback, self.q, self.q, self.q, 1, *args, **kw)
                self.assertEqual(tuple(out.shape), (1, 1, 128, 128) if output_4d else (1, 128, 128))
        self.assertEqual(self.run_attention(self.fallback, self.q, self.q, self.q, 1, None, None, False), 'dense')

    def test_precision_and_conflicts(self):
        for args, kw in [((None, torch.float32, True), {}), ((), {'skip_reshape': True, 'low_precision_attention': False})]:
            self.assertEqual(self.run_attention(self.fallback, self.q, self.q, self.q, 1, *args, **kw), 'dense')
        for args, kw in [((None,), {'mask': None}), ((None, None, True), {'skip_reshape': True}), ((None,)*5, {})]:
            with self.assertRaises(TypeError):
                self.run_attention(self.fallback, self.q, self.q, self.q, 1, *args, **kw)
        self.backend.assert_not_called()


class LayoutTests(unittest.TestCase):
    def call_model(self, options, callback, payload=None, video=None):
        video = torch.zeros(1, 1, 1, 2, 2) if video is None else video
        def original(x, timestep, context, transformer_options, **kw):
            return callback(transformer_options)
        return h3._make_span_injector(original)([video, torch.zeros(1, 1, 2)], None,
                                               torch.zeros(1, 4, 1), options, minimax_payload=payload)

    def test_old_and_new_constructors(self):
        calls = []
        def old(a, b, c, d, e, keyframes=None, refs=None, frame_count=None):
            calls.append((keyframes, refs, frame_count))
            return layout(signature=(a, b, c, d, e))
        def new(a, b, c, d, e, keyframes=None, refs=None):
            calls.append((keyframes, refs))
            return layout(signature=(a, b, c, d, e))
        for constructor in (old, new):
            with patch.object(h3, 'PackedLayout', constructor):
                self.assertEqual(self.call_model({}, lambda o: h3._conditioning_span(o, 128),
                                                 {'keyframes': ['k'], 'refs': ['r'], 'frame_count': 9}), (64, 128))
        self.assertEqual(calls, [(['k'], ['r'], 9), (['k'], ['r'])])

    def test_matching_payload_and_native_preference(self):
        with patch.object(h3, 'PackedLayout', side_effect=AssertionError('must reuse')):
            self.assertEqual(self.call_model({}, lambda o: h3._conditioning_span(o, 128), {'layout': layout()}), (64, 128))
            def modern(o):
                o['minimax_h3_layout'] = layout(80)
                return h3._conditioning_span(o, 128)
            self.assertEqual(self.call_model({}, modern, {'layout': layout()}), (80, 128))

        # A payload layout is usable even if importing its constructor failed.
        with patch.object(h3, 'PackedLayout', None):
            self.assertEqual(self.call_model({}, lambda o: h3._conditioning_span(o, 128),
                                             {'layout': layout()}), (64, 128))

    def test_legacy_failure_is_cached_without_growing_log_messages(self):
        def blocks(options):
            reasons = []
            for _ in range(4):
                with self.assertRaises(h3._Unsupported) as caught:
                    h3._conditioning_span(options, 128)
                reasons.append(str(caught.exception))
            self.assertEqual(len(set(reasons)), 1)
        with patch.object(h3, 'PackedLayout', side_effect=ValueError('broken layout')) as constructor:
            self.call_model({}, blocks)
            constructor.assert_called_once()

    def test_mismatch_rebuilt_and_state_cleared_between_calls(self):
        options = {'sol_h3_video_span': (8, 128), 'minimax_h3_layout': layout(8)}
        with patch.object(h3, 'PackedLayout', return_value=layout(72)) as constructor:
            self.assertEqual(self.call_model(options, lambda o: h3._conditioning_span(o, 128),
                                             {'layout': layout(signature=(0,)*5)}), (72, 128))
            constructor.assert_called_once()
        self.assertNotIn('sol_h3_video_span', options)
        with (patch.object(h3, 'PackedLayout', side_effect=ValueError('changed layout unavailable')),
              self.assertRaisesRegex(h3._Unsupported, 'changed layout unavailable')):
            self.call_model(options, lambda o: h3._conditioning_span(o, 128))
        self.assertNotIn('layout', options['sol_h3_layout_state'])
        with patch.object(h3, 'PackedLayout', return_value=layout(32)):
            self.assertEqual(self.call_model(options, lambda o: h3._conditioning_span(o, 128)), (32, 128))

    def test_missing_invalid_and_mismatched_layouts(self):
        invalid = [None, layout(stop=129), layout(start=-1), layout(start=129),
                   types.SimpleNamespace(seq_len=128, segments=[(0, 64, 'video'), (64, 128, 'audio')])]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(h3._Unsupported):
                h3._conditioning_span({'minimax_h3_layout': value}, 128)
        with self.assertRaisesRegex(h3._Unsupported, 'signature'):
            self.call_model({}, lambda o: (o.update(minimax_h3_layout=layout(signature=(0,)*5)),
                                           h3._conditioning_span(o, 128)))


class Attention:
    heads = 1
    head_dim = 128
    qkv_proj = staticmethod(lambda x: torch.cat([x, x, x], dim=-1))
    q_norm = k_norm = out_proj = staticmethod(lambda x: x)
    def forward(self, x, **kw):
        return x


class HandoffTests(unittest.TestCase):
    def setUp(self):
        patch.object(torch.cuda, 'get_device_capability', return_value=(8, 9)).start()
        self.backend = patch.object(h3, 'sol_attn', side_effect=lambda q, k, v, **kw: v).start()
        self.addCleanup(patch.stopall)

    def test_requested_protection_falls_back_before_handoff(self):
        for options in ({}, {'minimax_h3_layout': layout(stop=129)}, {'low_precision_attention': False}):
            x = activation(128, 128)
            handoff = [x]
            def dense(value, handoff=handoff, **kw):
                self.assertIs(value, handoff)
                self.assertEqual(len(value), 1)
                return value.pop()
            run = h3._make_sol_attention_forward(Attention(), dense, 1.3, 1, True, h3._SolLog())
            self.assertIs(run(handoff, transformer_options=options), x)
        self.backend.assert_not_called()

    def test_valid_protection_and_disabled_protection(self):
        for mode, options, expected in [('off', {}, (0, 0)), ('exact_kv', {'minimax_h3_layout': layout(65)}, (0, 2)),
                                        ('exact_kv_and_rows', {'minimax_h3_layout': layout(65)}, (0, 2))]:
            handoff = [activation(128, 128)]
            run = h3._make_sol_attention_forward(Attention(), Mock(side_effect=AssertionError), 1.3, 1, True,
                                                 h3._SolLog(), sink_conditioning=mode)
            out = run(handoff, transformer_options=options)
            self.assertEqual(handoff, [])
            self.assertEqual(out.shape, (128, 128))
            self.assertEqual(self.backend.call_args.kwargs['sink_blocks'], expected)
            self.assertEqual(self.backend.call_args.kwargs['sink_q'], expected if mode == 'exact_kv_and_rows' else (0, 0))


class FusionTests(unittest.TestCase):
    def wrapper(self, fallback, gate=None, mod=None):
        self.block = types.SimpleNamespace(adaln_proj=lambda t: [torch.zeros(2, 8)]*6,
                                          norm1=lambda x: x.clone(), norm2=lambda x: x.clone(),
                                          attn=Mock(return_value=activation(4, 8)), mlp=lambda x: x)
        self.log = h3._FusionLog()
        return h3._make_fused_h3_block_forward(self.block, fallback, h3._SegmentIndexCache(lambda *a: object()),
                                              self.log, mod or (lambda x, *a: x), gate or (lambda x, *a: x))

    def test_callback_fused_and_new_eager(self):
        cb = Mock(return_value=activation(4, 8))
        fallback = Mock(return_value='eager')
        run = self.wrapper(fallback)
        run(activation(4, 8), None, [(0, 4, 0)], None, attention=cb)
        cb.assert_called_once()
        self.block.attn.assert_not_called()
        self.assertEqual(run(torch.zeros(4, 8), None, [(0, 4, 0)], None, attention=cb), 'eager')
        self.assertIs(fallback.call_args.kwargs['attention'], cb)

    def test_legacy_fallback_and_explicit_callback_not_dropped(self):
        def legacy(x, t, segments, rope, transformer_options=None):
            return 'legacy'
        run = self.wrapper(legacy)
        self.assertEqual(run(torch.zeros(4, 8), None, [(0, 4, 0)], None), 'legacy')
        with self.assertRaises(TypeError):
            run(torch.zeros(4, 8), None, [(0, 4, 0)], None, attention=lambda x: x)

    def test_tensor_modulation_rows_eager_before_projection(self):
        eager = Mock(return_value='eager')
        run = self.wrapper(eager)
        self.block.adaln_proj = Mock(side_effect=AssertionError)
        cb = Mock()
        for row in (torch.zeros(4), torch.tensor(0)):
            segments = [(0, 4, row)]
            self.assertEqual(run(activation(4, 8), None, segments, None, attention=cb), 'eager')
            self.assertIs(eager.call_args.args[2], segments)
            self.assertIs(eager.call_args.kwargs['attention'], cb)
        self.assertIn('per-token modulation rows require eager execution', self.log.fallbacks)

    def test_error_after_residual_mutation_is_never_retried(self):
        eager = Mock()
        def gate(x, *args):
            x.add_(1)
            raise RuntimeError('after mutation')
        run = self.wrapper(eager, gate=gate)
        x = activation(4, 8)
        with self.assertRaisesRegex(RuntimeError, 'after mutation'):
            run(x, None, [(0, 4, 0)], None)
        self.assertTrue(torch.all(x == 1))
        eager.assert_not_called()


class Patcher:
    def __init__(self):
        self.model = types.SimpleNamespace(diffusion_model=type('MiniMaxH3Model', (), {
            'blocks': [types.SimpleNamespace(attn=Attention()) for _ in range(2)], '_forward': lambda *a, **kw: None})(),
            model_sampling=types.SimpleNamespace(percent_to_sigma=lambda p: 1-p))
        self.object_patches = {}
        self.object_patches_backup = {}
    def clone(self):
        clone = Patcher()
        clone.model = self.model
        clone.object_patches = self.object_patches.copy()
        clone.object_patches_backup = self.object_patches_backup.copy()
        return clone
    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        if name in self.object_patches_backup:
            return self.object_patches_backup[name]
        value = self.model
        for part in name.split('.'):
            value = value[int(part)] if part.isdigit() else getattr(value, part)
        return value
    def add_object_patch(self, name, value):
        self.object_patches[name] = value


class CompositionTests(unittest.TestCase):
    def apply_node(self, model, scheduled, dense=''):
        kw = {'model': model, 'enabled': True, 'min_tokens': 4096, 'strict': False, 'thresh_type': 'diag',
              'int8_qk': False, 'int8_pv': False, 'sink_conditioning': 'exact_kv', 'dense_blocks': dense}
        if scheduled:
            return h3.MiniMaxH3ScheduledSolAttentionPatch().patch(**kw, tau_start=1.3, tau_end=.8,
                                                                 curve='linear', dense_percent=.2)[0]
        return h3.MiniMaxH3MemoryEfficientSolAttentionPatch().patch(**kw, tau=1.3)[0]

    def test_all_patch_chains_restore_third_party_and_preserve_source(self):
        with patch.object(h3, '_plot_tau_schedule', return_value=None):
            for first in (False, True):
                for second in (False, True):
                    with self.subTest(first=first, second=second):
                        source = Patcher()
                        path = 'diffusion_model.blocks.0.attn.forward'
                        third_party = lambda *a, **kw: 'third party'
                        source.add_object_patch(path, third_party)
                        a = self.apply_node(source, first)
                        wrapper = a.get_model_object(path)
                        b = self.apply_node(a, second, '0')
                        self.assertIs(b.get_model_object(path), third_party)
                        self.assertIs(a.get_model_object(path), wrapper)
                        self.assertIs(source.get_model_object(path), third_party)
                        self.assertEqual(len(source.object_patches), 1)
                        self.assertTrue(hasattr(b.get_model_object('diffusion_model.blocks.1.attn.forward'), '_minimax_h3_sol_fallback'))

    def test_loaded_model_resolves_backup_and_keeps_unrelated_wrappers(self):
        source = Patcher()
        path = 'diffusion_model.blocks.0.attn.forward'
        backup = source.get_model_object(path)
        source.object_patches_backup[path] = backup
        source.model.diffusion_model.blocks[0].attn.forward = Mock(return_value='other loaded patch')
        a = self.apply_node(source, False)
        self.assertIs(a.get_model_object(path)._minimax_h3_sol_fallback, backup)
        untouched = self.apply_node(source, False, '0')
        self.assertNotIn(path, untouched.object_patches)
        self.assertIs(untouched.get_model_object(path), backup)


class DependencyTests(unittest.TestCase):
    def test_missing_backend_keeps_layout_and_feedforward_usable(self):
        package = types.ModuleType('sol_missing_backend')
        package.__path__ = [str(ROOT)]
        comfy_model = types.ModuleType('comfy.ldm.minimax.model')
        comfy_model.PackedLayout = layout
        with patch.dict(sys.modules, {'sol_missing_backend': package, 'sol_missing_backend.sol_kernel': None,
                                      'comfy.ldm.minimax.model': comfy_model,
                                      'comfy.model_management': None, 'comfy.quant_ops': None}):
            mod = importlib.import_module('sol_missing_backend.minimax')
        self.assertIsNone(mod.sol_attn)
        self.assertIsNotNone(mod.BACKEND_IMPORT_ERROR)
        self.assertIs(mod.PackedLayout, layout)
        x = torch.arange(32).reshape(8, 4).float()
        run = mod._make_chunked_forward(lambda x: x*2, 2, 1, mod._ChunkLog())
        torch.testing.assert_close(run(x), x*2)
        model = Patcher()
        diffusion = model.model.diffusion_model
        for block in diffusion.blocks:
            block.mlp = types.SimpleNamespace(forward=lambda value: value*2)
        diffusion.token_refiner = types.SimpleNamespace(blocks=[])
        patched = mod.MiniMaxH3ChunkFeedForward().patch(model, True, 2, 1)[0]
        torch.testing.assert_close(patched.get_model_object('diffusion_model.blocks.0.mlp.forward')(x), x*2)
        self.assertEqual(model.object_patches, {})
        with self.assertRaisesRegex(RuntimeError, 'failed to import:'):
            mod.MiniMaxH3MemoryEfficientSolAttentionPatch().patch(None, True, 1.3, 1, False, 'diag', False, False, 'off', '')


if __name__ == '__main__':
    unittest.main()
