"""Opt-in CPU checks against local ComfyUI Git tags (no downloads).

Set COMFYUI_ROOT to a Git checkout containing v0.30.1 through v0.37.0.
The real release layout/helper definitions execute with PyTorch on CPU;
only imports of ComfyUI runtime backends are excluded.
"""
import ast
import inspect
import json
import math
import os
import subprocess
import types
import unittest
from unittest.mock import patch

import torch
from test_compatibility import ROOT, h3, nodes

COMFY_ROOT = os.environ.get('COMFYUI_ROOT')


def source_at(tag, path):
    return subprocess.check_output(['git', '-C', COMFY_ROOT, 'show', f'{tag}:{path}'], text=True)


def definitions(source):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign))]
    namespace = {'torch': torch, 'nn': torch.nn, 'math': math}
    exec(compile(tree, '<ComfyUI release definitions>', 'exec'), namespace)  # noqa: S102 - trusted local release source
    return namespace


class WorkflowSchemaTests(unittest.TestCase):
    def test_original_workflow_interfaces(self):
        def workflow_contract(value):
            # Help text can evolve without changing serialized workflow inputs.
            if isinstance(value, dict):
                return {key: workflow_contract(item) for key, item in value.items() if key != 'tooltip'}
            if isinstance(value, list):
                return [workflow_contract(item) for item in value]
            return value

        fixture = json.loads((ROOT / 'tests/fixtures/node_interfaces.json').read_text())
        for module in (nodes, h3):
            for name, cls in module.NODE_CLASS_MAPPINGS.items():
                actual = {'inputs': cls.INPUT_TYPES(), 'outputs': cls.RETURN_TYPES,
                          'function': cls.FUNCTION, 'category': cls.CATEGORY,
                          'display_name': module.NODE_DISPLAY_NAME_MAPPINGS[name]}
                self.assertEqual(workflow_contract(json.loads(json.dumps(actual))), workflow_contract(fixture[name]))


@unittest.skipUnless(COMFY_ROOT, 'set COMFYUI_ROOT for local release interface checks')
class ReleaseInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        all_tags = subprocess.check_output(['git', '-C', COMFY_ROOT, 'tag', '-l'], text=True).splitlines()
        cls.tags = sorted((tag for tag in all_tags if tag.startswith('v') and
                          (0, 30, 1) <= tuple(map(int, tag[1:].split('.'))) <= (0, 37, 0)),
                         key=lambda tag: tuple(map(int, tag[1:].split('.'))))
        if not cls.tags:
            raise unittest.SkipTest('no supported release tags available locally')

    def test_all_available_release_layouts_and_block_callbacks(self):
        for tag in self.tags:
            with self.subTest(tag=tag):
                source = source_at(tag, 'comfy/ldm/minimax/model.py')
                ns = definitions(source)
                constructor = ns['PackedLayout']
                params = inspect.signature(constructor).parameters
                self.assertIn('refs', params)
                keyframes = [{'frame_index': 0, 'resolved_frame_index': 0,
                              'latent': torch.zeros(1, 24, 1, 4, 4)}]
                refs = [{'kind': 'image', 'latent_h': 4, 'latent_w': 4},
                        {'kind': 'audio', 'ref_audio_t': 2}]
                for payload in ({}, {'keyframes': keyframes, 'frame_count': 1}, {'refs': refs}):
                    expected = constructor(4, 1, 4, 4, 2, **{k: v for k, v in payload.items() if k in params})
                    def original(x, t, context, options, expected=expected, **kw):
                        return h3._conditioning_span(options, expected.seq_len)
                    with patch.object(h3, 'PackedLayout', constructor):
                        out = h3._make_span_injector(original)(
                            [torch.zeros(1, 24, 1, 3, 3), torch.zeros(1, 32, 2, 2)], None,
                            torch.zeros(1, 4, 8), {}, minimax_payload=payload)
                    self.assertEqual(out, next((a, b) for a, b, kind in expected.segments if kind == 'video'))

                # Execute the release's eager block forward as the captured
                # fallback, including newer tensor rows and attention override.
                block = types.SimpleNamespace(adaln_proj=lambda t: [torch.ones(2, 8)*.1]*6,
                                              norm1=lambda x: x.clone(), norm2=lambda x: x.clone(),
                                              attn=lambda x, **kw: x*.5, mlp=lambda x: x*.25)
                eager = types.MethodType(ns['DiTBlock'].forward, block)
                callback_supported = 'attention' in inspect.signature(eager).parameters
                self.assertEqual(callback_supported, tuple(map(int, tag[1:].split('.'))) >= (0, 35, 0))
                fused = h3._make_fused_h3_block_forward(block, eager, None, h3._FusionLog(), None, None)
                rows = [0]
                if '_mod_row' in ns:
                    rows.append(torch.tensor([0, 1, 0, 1]))
                for row in rows:
                    segments = [(0, 4, row)]
                    kw = {'attention': lambda x, **opts: x*.75} if callback_supported else {}
                    x = torch.ones(4, 8)
                    torch.testing.assert_close(fused(x.clone(), None, segments, None, **kw),
                                               eager(x.clone(), None, segments, None, **kw))
        print(f'Checked {len(self.tags)} local ComfyUI releases: {", ".join(self.tags)}')

    def test_attention_argument_order_in_each_release(self):
        expected = ['mask', 'attn_precision', 'skip_reshape', 'skip_output_reshape']
        for tag in self.tags:
            tree = ast.parse(source_at(tag, 'comfy/ldm/modules/attention.py'))
            fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'attention_pytorch')
            self.assertEqual([a.arg for a in fn.args.args][4:], expected, tag)


if __name__ == '__main__':
    unittest.main()
