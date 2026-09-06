"""CPU scaling benchmark for the torch LUT model.

    python -m lut_torch.bench_cpu                 # both experiments, default shape
    python -m lut_torch.bench_cpu --experiment grid --threads 1,2,4,8,16 --batches 1,8,32
    python -m lut_torch.bench_cpu --experiment concurrency --total-cores 16

Answers the two questions that actually decide how to spend a node:

**grid** -- how does one training process scale with ``torch.set_num_threads`` at a given
batch size?  The torch port has exactly one axis of CPU parallelism (ATen's intra-op
thread pool), and its efficiency depends on how much work each tensor op has to hand out.
Batch size is what sets that, so threads and batch have to be swept together.

**concurrency** -- given a fixed core budget, is it better to run one wide job or several
narrow ones?  Independent runs (seeds, ablations) are embarrassingly parallel and never
suffer the sublinear intra-op scaling, so past some point ``K`` processes of ``T`` threads
beat one process of ``K*T`` threads.  This measures where that crossover sits.

Timings exclude warm-up.  Report `seq/s` (sequences per second) rather than steps/s, since
steps are not comparable across batch sizes.
"""

import argparse
import json
import os
import time


# Shape of the current POS experiments; override on the command line.
DEFAULT_SHAPE = dict(
    context_size=32, embedding_dim=64, positional_dim=5, num_layers=6, num_heads=4,
    vocab_size_input=8000, vocab_size_output=18, n_t=64, n_c=6,
    attn_qk_dim=8, attn_qk_n_c=6, attn_qk_n_t=64, POS_task=True,
    weight_sharing=8, freeze_PE=True, qkpe_direct_sign=True,
)


def _time_steps(threads, batch, shape, target_seconds, min_steps, max_steps, seed=0,
                backend=None):
    """Time training steps in this process. Returns (seconds_per_step, steps)."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from .config import LutConfig
    from .kernels import set_kernel_backend, set_kernel_threads
    from .modules import LutModel

    torch.set_num_threads(threads)
    if backend is not None:
        set_kernel_backend(backend)
        if backend in ('auto', 'numba'):
            try:
                import numba
                numba.set_num_threads(threads)
                set_kernel_threads(threads)
            except ImportError:
                pass
    cfg = LutConfig(**shape)

    import random
    random.seed(seed)
    np.random.seed(seed)
    model = LutModel(cfg)
    optimizer = torch.optim.SGD([{'params': model.table_parameters()}], lr=0.01)

    rng = np.random.default_rng(seed)
    inputs = torch.from_numpy(
        rng.integers(0, cfg.vocab_size_input, (batch, cfg.context_size)))
    targets = torch.from_numpy(
        rng.integers(0, cfg.vocab_size_output, (batch, cfg.context_size)))

    def step():
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size_output),
                               targets.reshape(-1), reduction='sum') / batch
        loss.backward()
        optimizer.step()

    step()                                   # warm up allocator and thread pool
    start = time.perf_counter()
    step()
    one = time.perf_counter() - start

    steps = int(max(min_steps, min(max_steps, round(target_seconds / max(one, 1e-6)))))
    start = time.perf_counter()
    for _ in range(steps):
        step()
    return (time.perf_counter() - start) / steps, steps


def _worker(payload):
    """Concurrency-experiment child: pin threads before torch is ever imported."""
    threads, batch, shape, seconds, seed = payload
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    per_step, steps = _time_steps(threads, batch, shape, seconds, 3, 200, seed=seed)
    return per_step, steps


def run_grid(args, shape):
    print(f"\n{'=' * 78}\nGRID: one process, threads x batch"
          f"\n{'=' * 78}")
    print(f"{'batch':>6} {'threads':>8} {'ms/step':>10} {'seq/s':>9} "
          f"{'speedup':>8} {'efficiency':>11}")
    results = []
    for batch in args.batches:
        base = None
        for threads in args.threads:
            per_step, steps = _time_steps(threads, batch, shape, args.seconds,
                                          args.min_steps, args.max_steps)
            seq_s = batch / per_step
            if base is None:
                base = seq_s
            speedup = seq_s / base
            eff = speedup / threads
            print(f"{batch:>6} {threads:>8} {per_step * 1000:>10.1f} {seq_s:>9.2f} "
                  f"{speedup:>7.2f}x {eff * 100:>10.0f}%")
            results.append(dict(experiment='grid', batch=batch, threads=threads,
                                ms_per_step=per_step * 1000, seq_per_s=seq_s,
                                speedup=speedup, efficiency=eff, steps=steps))
        print()
    return results


def run_concurrency(args, shape):
    import multiprocessing as mp

    total = args.total_cores
    splits = [(k, total // k) for k in args.processes if total % k == 0]
    print(f"\n{'=' * 78}\nCONCURRENCY: {total} cores split K ways, batch {args.conc_batch}"
          f"\n{'=' * 78}")
    print(f"{'procs':>6} {'threads':>8} {'ms/step':>10} {'aggregate seq/s':>17} "
          f"{'vs 1 proc':>10}")

    ctx = mp.get_context('spawn')
    results = []
    base = None
    for procs, threads in splits:
        payload = [(threads, args.conc_batch, shape, args.seconds, s)
                   for s in range(procs)]
        start = time.perf_counter()
        with ctx.Pool(procs) as pool:
            per_proc = pool.map(_worker, payload)
        wall = time.perf_counter() - start

        # Aggregate throughput: every process ran its own steps concurrently.
        total_sequences = sum(steps * args.conc_batch for _, steps in per_proc)
        slowest = max(p for p, _ in per_proc)
        aggregate = procs * args.conc_batch / slowest
        if base is None:
            base = aggregate
        print(f"{procs:>6} {threads:>8} {slowest * 1000:>10.1f} {aggregate:>17.2f} "
              f"{aggregate / base:>9.2f}x")
        results.append(dict(experiment='concurrency', processes=procs, threads=threads,
                            batch=args.conc_batch, ms_per_step=slowest * 1000,
                            aggregate_seq_per_s=aggregate, vs_one_process=aggregate / base,
                            wall_seconds=wall, sequences=total_sequences))
    return results


def run_backends(args, shape):
    """torch path vs fused-Numba-kernel path, at the same threads x batch cells.

    Answers the question the kernel work (lut_torch/PLAN_numba_kernels.md) was done to
    answer: how much of the CPU wall-clock is framework dispatch overhead the kernels
    remove, versus real work that scales with threads either way.
    """
    from .kernels import numba_available
    if not numba_available():
        print("\nBACKENDS: numba not importable, skipping")
        return []

    print(f"\n{'=' * 78}\nBACKENDS: torch vs fused Numba kernels, closest selection"
          f"\n{'=' * 78}")
    print(f"{'batch':>6} {'threads':>8} {'torch ms':>10} {'numba ms':>10} "
          f"{'speedup':>8} {'torch seq/s':>12} {'numba seq/s':>12}")
    results = []
    for batch in args.batches:
        for threads in args.threads:
            t_ms, _ = _time_steps(threads, batch, shape, args.seconds,
                                  args.min_steps, args.max_steps, backend='torch')
            n_ms, _ = _time_steps(threads, batch, shape, args.seconds,
                                  args.min_steps, args.max_steps, backend='numba')
            speedup = t_ms / n_ms
            print(f"{batch:>6} {threads:>8} {t_ms * 1000:>10.1f} {n_ms * 1000:>10.1f} "
                  f"{speedup:>7.2f}x {batch / t_ms:>12.2f} {batch / n_ms:>12.2f}")
            results.append(dict(experiment='backends', batch=batch, threads=threads,
                                torch_ms_per_step=t_ms * 1000, numba_ms_per_step=n_ms * 1000,
                                speedup=speedup, torch_seq_per_s=batch / t_ms,
                                numba_seq_per_s=batch / n_ms))
        print()
    return results


def csv_ints(text):
    return [int(part) for part in text.split(',') if part.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiment', choices=['grid', 'concurrency', 'backends', 'both'],
                   default='both')
    p.add_argument('--threads', type=csv_ints, default=[1, 2, 4, 8, 16, 20])
    p.add_argument('--batches', type=csv_ints, default=[1, 8, 32])
    p.add_argument('--processes', type=csv_ints, default=[1, 2, 4, 8, 16])
    p.add_argument('--total-cores', type=int, default=16)
    p.add_argument('--conc-batch', type=int, default=8)
    p.add_argument('--seconds', type=float, default=4.0,
                   help='Target measured seconds per cell')
    p.add_argument('--min-steps', type=int, default=3)
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--new-MHA', dest='new_MHA', action='store_true')
    p.add_argument('--num-layers', type=int, default=None)
    p.add_argument('--json-out', type=str, default=None)
    args = p.parse_args(argv)

    shape = dict(DEFAULT_SHAPE)
    shape['new_MHA'] = args.new_MHA
    if args.num_layers is not None:
        shape['num_layers'] = args.num_layers

    import torch
    print(f"torch {torch.__version__} | host CPUs {os.cpu_count()} | "
          f"default threads {torch.get_num_threads()}")
    print("shape: " + ', '.join(f"{k}={v}" for k, v in sorted(shape.items())))

    results = []
    if args.experiment in ('grid', 'both'):
        results += run_grid(args, shape)
    if args.experiment in ('concurrency', 'both'):
        results += run_concurrency(args, shape)
    if args.experiment == 'backends':
        results += run_backends(args, shape)

    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == '__main__':
    main()
