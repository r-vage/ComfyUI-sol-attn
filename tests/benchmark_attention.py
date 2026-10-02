"""Reproducible attention microbenchmark using installed backends and real kernels.

Run twice in separate processes; retain the second run after Triton autotuning.
No model checkpoint, ComfyUI server, downloads, or dependency installation needed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
from torch.nn import functional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sol_kernel import sol_attn


def device_status():
    try:
        return subprocess.check_output([
            'nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,driver_version',
            '--format=csv,noheader',
        ], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def timed_call(fn):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    output = fn()
    stop.record()
    stop.synchronize()
    elapsed = start.elapsed_time(stop)
    del output
    return elapsed


def relative_l2(actual, expected):
    # Used only at the selected accuracy size, outside all timing/memory samples.
    actual, expected = actual.float(), expected.float()
    return float(torch.linalg.vector_norm(actual-expected) / torch.linalg.vector_norm(expected))


@torch.inference_mode()
def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')
    if args.iterations < 3 or args.warmup < 1 or args.rounds < 1:
        raise ValueError('Use at least 3 iterations, 1 warmup, and 1 round')
    if args.sink_tokens < 0 or any(t <= args.sink_tokens for t in args.tokens):
        raise ValueError('sink_tokens must be nonnegative and smaller than each sequence')
    sage_error = None
    try:
        from sageattention import sageattn
    except ImportError as exc:
        sageattn = None
        sage_error = str(exc)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((ROOT/'sol_kernel').glob('*.py'))}
    result = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
        'total_vram_mib': torch.cuda.get_device_properties(0).total_memory / 2**20,
        'python': platform.python_version(), 'platform': platform.platform(),
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'triton': triton.__version__,
        'sageattention': importlib.metadata.version('sageattention') if sageattn else None,
        'sage_import_error': sage_error, 'config': vars(args) | {'output': str(args.output)},
        'kernel_sha256': hashes, 'status_before': device_status(), 'measurements': [],
        'timing': 'CUDA events per call, round-wise shuffled methods, no CUDA graphs or cache flush; warmup excluded',
        'memory': 'peak torch CUDA allocated bytes above resident inputs; output included; reserved/non-PyTorch allocations excluded',
    }
    print(json.dumps({k: result[k] for k in ('gpu', 'torch', 'triton', 'sageattention', 'status_before')}), flush=True)
    for tokens in args.tokens:
        torch.manual_seed(args.seed)
        qkv = torch.randn((1, tokens, 3 * 56 * 128), device='cuda', dtype=torch.bfloat16)
        q, k, v = [part.view(1, tokens, 56, 128) for part in qkv.split(56*128, dim=-1)]
        sinks = (0, (args.sink_tokens+63)//64)
        methods = {
            'sdpa': lambda q=q, k=k, v=v: functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=0.0, is_causal=False).transpose(1, 2),
        }
        if sageattn:
            methods['sage'] = lambda q=q, k=k, v=v: sageattn(q, k, v, tensor_layout='NHD', is_causal=False)
        methods.update({
            'sol_bf16': lambda q=q, k=k, v=v, sinks=sinks: sol_attn(q, k, v, tau=args.tau, sink_blocks=sinks),
            'sol_int8_qk': lambda q=q, k=k, v=v, sinks=sinks: sol_attn(q, k, v, tau=args.tau, int8_qk=True, sink_blocks=sinks),
            'sol_int8_qk_pv': lambda q=q, k=k, v=v, sinks=sinks: sol_attn(q, k, v, tau=args.tau, int8_qk=True, int8_pv=True, sink_blocks=sinks),
        })
        row = {'tokens': tokens, 'input_shape': list(q.shape), 'input_strides': list(q.stride()),
               'resident_qkv_mib': qkv.numel()*qkv.element_size()/2**20,
               'status_before': device_status(), 'methods': {}}
        # Finish compilation and autotuning for every method before any timing.
        for name, fn in methods.items():
            print(f'{tokens}: warmup {name}', flush=True)
            try:
                for _ in range(args.warmup):
                    output = fn()
                    torch.cuda.synchronize()
                    if not bool(torch.isfinite(output).all()):
                        raise ValueError('nonfinite output')
                    del output
                row['methods'][name] = {'samples_ms': [], 'round_medians_ms': []}
            except (RuntimeError, ValueError) as exc:
                row['methods'][name] = {'error': f'{type(exc).__name__}: {exc}'}
                gc.collect()
                torch.cuda.empty_cache()
        randomizer = random.Random(args.seed+tokens)
        order = [name for name in methods if 'error' not in row['methods'][name]]
        for _ in range(args.rounds):
            randomizer.shuffle(order)
            for name in order:
                # A short untimed call removes idle/cache effects after switching methods.
                output = methods[name]()
                del output
                torch.cuda.synchronize()
                samples = [timed_call(methods[name]) for _ in range(args.iterations)]
                row['methods'][name]['samples_ms'].extend(samples)
                row['methods'][name]['round_medians_ms'].append(statistics.median(samples))
        for name in order:
            measurement = row['methods'][name]
            samples = measurement['samples_ms']
            measurement.update(median_ms=statistics.median(samples),
                               p10_ms=sorted(samples)[len(samples)//10],
                               p90_ms=sorted(samples)[len(samples)*9//10])
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            output = methods[name]()
            torch.cuda.synchronize()
            measurement['peak_working_mib'] = (torch.cuda.max_memory_allocated()-base)/2**20
            del output
            print(f'{tokens}: {name}: {measurement["median_ms"]:.3f} ms, '
                  f'{measurement["peak_working_mib"]:.1f} MiB', flush=True)
        if tokens == args.accuracy_tokens and 'error' not in row['methods']['sdpa']:
            reference = methods['sdpa']()
            bf16 = methods['sol_bf16']() if 'error' not in row['methods']['sol_bf16'] else None
            for name in order:
                output = methods[name]()
                row['methods'][name]['relative_l2_vs_sdpa'] = relative_l2(output, reference)
                if bf16 is not None:
                    row['methods'][name]['relative_l2_vs_sol_bf16'] = relative_l2(output, bf16)
                del output
            exact = sol_attn(q, k, v, tau=-100)
            row['all_exact_sol_relative_l2_vs_sdpa'] = relative_l2(exact, reference)
            del exact, reference, bf16
        row['status_after'] = device_status()
        result['measurements'].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        del methods, fn, q, k, v, qkv
        gc.collect()
        torch.cuda.empty_cache()
    result['status_after'] = device_status()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=int, nargs='+', default=[4096, 8192, 16384, 32768, 65536])
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--sink-tokens', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--accuracy-tokens', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=104)
    parser.add_argument('--output', type=Path, required=True)
    benchmark(parser.parse_args())
