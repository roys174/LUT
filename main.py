"""SNN transformer architecture in pure Python/NumPy."""
"""Based on Spiking Manifesto (Izhikevich 2025)"""
"""The code strives for simplicity, not efficiency."""
"""Eugene Izhikevich, October 2025"""
"""Python port by Claude, February 2026"""

import argparse
import hashlib
import io
import math
import os
import random
import sys
from pathlib import Path
import numpy as np
import tiktoken
from word_tokenizer import POS_LABELS, WordTokenizer
from multiprocessing.pool import ThreadPool
from tqdm import tqdm

try:
    from numba import njit, prange, get_num_threads, get_thread_id, set_num_threads
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    prange = range
    get_num_threads = lambda: 1
    get_thread_id = lambda: 0
    set_num_threads = lambda n: None
    def njit(*args, **kwargs):
        def decorator(f): return f
        return decorator


# Global learning rate (updated each step)
learning_rate = 0.0
UNK_TOKEN_ID = 0
IGNORE_INDEX = -1


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def softmax(x, size, temperature):
    max_val = np.max(x[:size])
    x[:size] = np.exp((x[:size] - max_val) / temperature)
    s = np.sum(x[:size])
    x[:size] /= s


def random_vector(size, scale):
    return scale * 2.0 * (np.random.rand(size).astype(np.float32) - 0.5)


def sample(probabilities, n):
    cumsum = np.cumsum(probabilities[:n])
    return min(int(np.searchsorted(cumsum, random.random())), n - 1)


def configure_cpu_parallelism(args, parser):
    if args.batch_size < 1:
        parser.error('--batch-size must be at least 1')
    if args.cpu_cores < 1:
        parser.error('--cpu-cores must be at least 1')
    if args.num_workers < 0:
        parser.error('--num-workers must be non-negative')
    if args.numba_threads < 0:
        parser.error('--numba-threads must be non-negative')
    if args.replay_workers < 0:
        parser.error('--replay-workers must be non-negative')

    cpu_cores = args.cpu_cores

    if args.batch_size == 1:
        batch_workers = 1
    elif args.num_workers == 0:
        batch_workers = min(args.batch_size, cpu_cores)
    else:
        batch_workers = min(args.batch_size, args.num_workers)

    replay_units = args.num_layers * (1 + args.num_heads) + (2 if args.factored_output else 1)
    if args.batch_size == 1:
        replay_workers = 1
    elif args.replay_workers == 0:
        replay_workers = min(replay_units, cpu_cores)
    else:
        replay_workers = min(replay_units, args.replay_workers)

    if HAS_NUMBA:
        if args.numba_threads == 0:
            requested_numba_threads = max(1, cpu_cores // max(1, batch_workers))
        else:
            requested_numba_threads = args.numba_threads
        try:
            set_num_threads(requested_numba_threads)
        except ValueError as exc:
            print(f"Warning: could not set Numba threads to {requested_numba_threads}: {exc}")
            print("         If Numba was capped by the environment, start with NUMBA_NUM_THREADS=<cores>.")
        actual_numba_threads = get_num_threads()
    else:
        requested_numba_threads = 1
        actual_numba_threads = 1

    args.num_workers = batch_workers
    args.replay_workers = replay_workers
    args.numba_threads = actual_numba_threads
    args.requested_numba_threads = requested_numba_threads


def fill_vector_with_random_integers(N, max_value):
    return np.array([random.randint(0, max_value - 1) for _ in range(N)], dtype=np.int32)


def fill_vector_with_random_integers_different_from_vector2(vector2, N, max_value):
    vector = np.zeros(N, dtype=np.int32)
    for i in range(N):
        while True:
            vector[i] = random.randint(0, max_value - 1)
            if vector[i] != vector2[i]:
                break
    return vector


def neurons_needed_for_comparisons(n_c):
    if n_c < 1:
        raise ValueError("n_c must be at least 1")
    n = 2
    while n * (n - 1) // 2 < n_c:
        n += 1
    return n


def build_no_overlap_comparison_groups(n_c, input_dim):
    neurons_per_table = neurons_needed_for_comparisons(n_c)
    if input_dim < neurons_per_table:
        raise ValueError(
            f"input_dim={input_dim} is too small for n_c={n_c}; "
            f"need at least {neurons_per_table} neurons")

    groups = []
    for start in range(0, input_dim, neurons_per_table):
        group = list(range(start, min(start + neurons_per_table, input_dim)))
        pad = 0
        while len(group) < neurons_per_table:
            group.append(pad % input_dim)
            pad += 1
        groups.append(group)
    return groups


def non_overlapping_table_count(input_dim, n_c):
    return len(build_no_overlap_comparison_groups(n_c, input_dim))


def _comparison_pairs_for_group(group, n_c):
    pairs = []
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            if group[i] == group[j]:
                continue
            pairs.append((group[i], group[j]))
            if len(pairs) == n_c:
                return pairs
    raise ValueError(
        f"group {group} cannot provide {n_c} distinct-neuron comparisons")


def _comparison_arrays_from_groups(groups, n_c):
    a_arr = np.zeros((len(groups), n_c), dtype=np.int32)
    b_arr = np.zeros((len(groups), n_c), dtype=np.int32)
    for i, group in enumerate(groups):
        pairs = _comparison_pairs_for_group(group, n_c)
        a_arr[i] = [a for a, _ in pairs]
        b_arr[i] = [b for _, b in pairs]
    return a_arr, b_arr


def build_no_overlap_comparison_arrays(n_c, input_dim, n_t=None):
    groups = build_no_overlap_comparison_groups(n_c, input_dim)
    if n_t is not None:
        groups = [groups[i % len(groups)] for i in range(n_t)]
    return _comparison_arrays_from_groups(groups, n_c)


def build_overlap_comparison_groups(n_t, n_c, input_dim):
    base_groups = build_no_overlap_comparison_groups(n_c, input_dim)
    if n_t <= len(base_groups):
        return base_groups[:n_t]
    if len(base_groups) < 2:
        raise ValueError(
            "overlap mode needs at least two base groups when requesting "
            "extra overlapping tables")

    neurons_per_table = len(base_groups[0])
    split = (neurons_per_table + 1) // 2
    groups = list(base_groups)
    offset = 0
    while len(groups) < n_t:
        left_idx = offset % len(base_groups)
        right_idx = (left_idx + 1 + offset // len(base_groups)) % len(base_groups)
        if right_idx == left_idx:
            right_idx = (right_idx + 1) % len(base_groups)
        groups.append(base_groups[left_idx][:split] + base_groups[right_idx][split:])
        offset += 1
    return groups


def build_overlap_comparison_arrays(n_t, n_c, input_dim):
    return _comparison_arrays_from_groups(
        build_overlap_comparison_groups(n_t, n_c, input_dim), n_c)


def build_comparison_arrays(mode, requested_n_t, n_c, input_dim):
    if mode == 'no-overlap':
        return build_no_overlap_comparison_arrays(n_c, input_dim)
    if mode == 'overlap':
        return build_overlap_comparison_arrays(requested_n_t, n_c, input_dim)
    if mode == 'random':
        a_arr = np.zeros((requested_n_t, n_c), dtype=np.int32)
        b_arr = np.zeros((requested_n_t, n_c), dtype=np.int32)
        for i in range(requested_n_t):
            a_arr[i] = fill_vector_with_random_integers(n_c, input_dim)
            b_arr[i] = fill_vector_with_random_integers_different_from_vector2(
                a_arr[i], n_c, input_dim)
        return a_arr, b_arr
    raise ValueError(f"unknown LUT comparison mode {mode!r}")


def configure_lut_comparison_mode(args, parser):
    if args.n_c < 1:
        parser.error('--n-c must be at least 1')
    if args.attn_qk_n_c < 1:
        parser.error('--attn_qk_n_c must be at least 1')

    if args.lut_comparison_mode == 'no-overlap':
        try:
            args.n_t = non_overlapping_table_count(args.embedding_dim, args.n_c)
            args.attn_qk_n_t = non_overlapping_table_count(
                args.embedding_dim, args.attn_qk_n_c)
        except ValueError as exc:
            parser.error(f"--lut-comparison-mode no-overlap: {exc}")
    elif args.lut_comparison_mode == 'overlap':
        try:
            build_overlap_comparison_groups(args.n_t, args.n_c, args.embedding_dim)
            build_overlap_comparison_groups(
                args.attn_qk_n_t, args.attn_qk_n_c, args.embedding_dim)
        except ValueError as exc:
            parser.error(f"--lut-comparison-mode overlap: {exc}")


def configure_weight_sharing(args, parser):
    weight_sharing = getattr(args, 'weight_sharing', None)
    if weight_sharing is None:
        args.final_n_t = args.n_t
        args.final_share_m = 1
        return
    if not getattr(args, 'new_attention', False):
        parser.error('--weight-sharing requires --new-attention')
    if weight_sharing < 1:
        parser.error('--weight-sharing must be at least 1')
    if weight_sharing > args.n_t:
        parser.error('--weight-sharing must be no greater than --n-t')
    if args.n_t % weight_sharing != 0:
        parser.error('--n-t must be divisible by --weight-sharing')
    args.final_n_t = weight_sharing
    args.final_share_m = args.n_t // weight_sharing


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class LUT:
    __slots__ = ['y_dim', 'S', 'a_arr', 'b_arr', 'bitmasks', 'trees']
    def __init__(self, n_t):
        self.y_dim = 0
        self.S = None
        self.a_arr = None
        self.b_arr = None
        self.bitmasks = None
        self.trees = np.arange(n_t, dtype=np.int32)


class LUTUpdateStats:
    """Counts row and j-bin usage for LUT S updates."""
    def __init__(self):
        self.counts = {}
        self.coord_counts = {}
        self.j_counts = {}
        self.j_component_counts = {}
        self.j_joint_counts = {}

    def ensure(self, name, lut):
        if name not in self.counts:
            self.counts[name] = np.zeros(lut.S.shape[:2], dtype=np.uint64)

    def ensure_coord(self, name, lut):
        if name not in self.coord_counts:
            self.coord_counts[name] = np.zeros((lut.S.shape[0], lut.y_dim), dtype=np.uint64)

    def ensure_j(self, name, n_tables, n_bins):
        if name not in self.j_counts:
            self.j_counts[name] = np.zeros((n_tables, n_bins), dtype=np.uint64)

    def ensure_j_component(self, name, component, n_tables, n_bins):
        key = (name, component)
        if key not in self.j_component_counts:
            self.j_component_counts[key] = np.zeros((n_tables, n_bins), dtype=np.uint64)

    def ensure_j_joint(self, name, row_component, col_component,
                       n_tables, n_row_bins, n_col_bins):
        key = (name, row_component, col_component)
        if key not in self.j_joint_counts:
            self.j_joint_counts[key] = np.zeros(
                (n_tables, n_row_bins, n_col_bins), dtype=np.uint64)

    @staticmethod
    def _add_2d_counts(counts, tables, values):
        flat = (np.asarray(tables, dtype=np.int64) * counts.shape[1]
                + np.asarray(values, dtype=np.int64)).ravel()
        if flat.size:
            counts.ravel()[:] += np.bincount(flat, minlength=counts.size).astype(np.uint64)

    def record_rows(self, name, lut, trees, rows):
        self.ensure(name, lut)
        self._add_2d_counts(self.counts[name], trees, rows)

    def record_coord_updates(self, name, lut, trees, updates):
        self.ensure_coord(name, lut)
        counts = self.coord_counts[name]
        tree_arr = np.asarray(trees, dtype=np.int64)
        update_arr = np.asarray(updates)
        y_dim = lut.y_dim
        if update_arr.shape[-1] < y_dim:
            raise ValueError(
                f"Coordinate update for {name} has {update_arr.shape[-1]} dims; expected at least {y_dim}")
        update_arr = update_arr[..., :y_dim]
        if update_arr.shape[:-1] != tree_arr.shape:
            update_arr = np.broadcast_to(update_arr, tree_arr.shape + (y_dim,))
        mask = update_arr != 0
        if not mask.size:
            return
        flat_trees = tree_arr.ravel()
        flat_mask = mask.reshape(flat_trees.size, y_dim)
        for tree in np.unique(flat_trees):
            counts[int(tree)] += flat_mask[flat_trees == tree].sum(axis=0).astype(np.uint64)

    def record_simple_coords(self, name, lut, grad):
        trees = np.broadcast_to(lut.trees, grad.shape[:1] + (lut.S.shape[0],))
        self.record_coord_updates(name, lut, trees, grad[:, np.newaxis, :lut.y_dim])

    def record_attention_coords(self, name, lut, y_grad, args):
        trees = np.asarray(lut.trees, dtype=np.int32)
        for pos in range(1, args.context_size):
            update_vec = y_grad[pos, :lut.y_dim]
            repeated = np.broadcast_to(update_vec, (pos, lut.S.shape[0], lut.y_dim))
            tree_rows = np.broadcast_to(trees, (pos, lut.S.shape[0]))
            self.record_coord_updates(name, lut, tree_rows, repeated)

    def record_attention_scalar_coords(self, name, lut, scalar_grad, args):
        trees = np.asarray(lut.trees, dtype=np.int32)
        for pos in range(1, args.context_size):
            values = scalar_grad[pos, :pos]
            repeated = np.broadcast_to(values[:, np.newaxis, np.newaxis],
                                       (pos, lut.S.shape[0], lut.y_dim))
            tree_rows = np.broadcast_to(trees, (pos, lut.S.shape[0]))
            self.record_coord_updates(name, lut, tree_rows, repeated)

    def record_new_attention_final_coords(self, name, lut, y_grad, args):
        trees = np.asarray(lut.trees, dtype=np.int32)
        n_logical = trees.shape[0]
        for tok_i in range(1, args.context_size):
            update_vec = y_grad[tok_i, :lut.y_dim]
            repeated = np.broadcast_to(update_vec, (tok_i, n_logical, lut.y_dim))
            tree_rows = np.broadcast_to(trees, (tok_i, n_logical))
            self.record_coord_updates(name, lut, tree_rows, repeated)

    def record_j_values(self, name, n_tables, n_bins, tables, j_values):
        self.ensure_j(name, n_tables, n_bins)
        self._add_2d_counts(self.j_counts[name], tables, j_values)

    def record_j_component(self, name, component, n_tables, n_bins, tables, j_values):
        self.ensure_j_component(name, component, n_tables, n_bins)
        self._add_2d_counts(self.j_component_counts[(name, component)], tables, j_values)

    def record_j_joint(self, name, row_component, col_component,
                       n_tables, n_row_bins, n_col_bins,
                       tables, row_values, col_values):
        self.ensure_j_joint(name, row_component, col_component,
                            n_tables, n_row_bins, n_col_bins)
        counts = self.j_joint_counts[(name, row_component, col_component)]
        flat = (
            np.asarray(tables, dtype=np.int64) * n_row_bins * n_col_bins
            + np.asarray(row_values, dtype=np.int64) * n_col_bins
            + np.asarray(col_values, dtype=np.int64)
        ).ravel()
        if flat.size:
            counts.ravel()[:] += np.bincount(flat, minlength=counts.size).astype(np.uint64)

    def record_simple(self, name, lut, j):
        trees = np.broadcast_to(lut.trees, j.shape)
        self.record_rows(name, lut, trees, j)
        self.record_j_values(name, lut.S.shape[0], lut.S.shape[1], trees, j)

    def record_attention_rows(self, name, lut, jV, jPE, args):
        CS = args.context_size
        n_tables = lut.S.shape[0]
        trees = np.asarray(lut.trees, dtype=np.int32)
        qk_bins = 1 << args.n_c
        pe_bins = 1 << args.positional_dim
        for pos in range(1, CS):
            jQ = jV[pos]
            jK = jV[:pos]
            pe_idx = pos - np.arange(pos)
            jPE_slice = jPE[pe_idx]
            j_bins = (jQ << args.shift_qk) | (jK << args.positional_dim) | jPE_slice
            tree_rows = np.broadcast_to(trees, j_bins.shape)
            self.record_rows(name, lut, tree_rows, j_bins)
            self.record_j_component(
                name, "q", n_tables, qk_bins, tree_rows,
                np.broadcast_to(jQ, j_bins.shape))
            self.record_j_component(name, "k", n_tables, qk_bins, tree_rows, jK)
            self.record_j_component(name, "pe", n_tables, pe_bins, tree_rows, jPE_slice)

    def record_new_attention_final(self, name, lut, final_v_j, qkpe_j, args):
        CS = args.context_size
        N_T = args.n_t
        pd = args.positional_dim
        n_tables = lut.S.shape[0]
        final_v_bins = 1 << args.n_c
        qkpe_bins = 1 << pd
        trees = np.asarray(lut.trees, dtype=np.int32)
        for tok_i in range(1, CS):
            rows = np.zeros((tok_i, N_T), dtype=np.int32)
            qkpe_rows = np.zeros((tok_i, N_T), dtype=np.int32)
            final_v_rows = np.zeros((tok_i, N_T), dtype=np.int32)
            for tok_j in range(tok_i):
                pair_idx = tok_i * (tok_i - 1) // 2 + tok_j
                qkpe_rows[tok_j] = qkpe_j[pair_idx]
                final_v_rows[tok_j] = final_v_j[tok_j]
                rows[tok_j] = (final_v_rows[tok_j] << pd) | qkpe_rows[tok_j]
            tree_rows = np.broadcast_to(trees, rows.shape)
            self.record_rows(name, lut, tree_rows, rows)
            self.record_j_component(
                name, "qkpe", n_tables, qkpe_bins, tree_rows, qkpe_rows)
            self.record_j_component(
                name, "final_v", n_tables, final_v_bins, tree_rows, final_v_rows)
            self.record_j_joint(
                name, "qkpe", "final_v", n_tables, qkpe_bins, final_v_bins,
                tree_rows, qkpe_rows, final_v_rows)

    @staticmethod
    def j_stats_path(path):
        return path[:-4] + "_j_stats.npz" if path.endswith(".npz") else path + "_j_stats.npz"

    @staticmethod
    def coord_stats_path(path):
        return path[:-4] + "_coord_stats.npz" if path.endswith(".npz") else path + "_coord_stats.npz"

    @staticmethod
    def _kl_to_uniform(counts):
        total = float(np.sum(counts))
        n = int(np.size(counts))
        if total <= 0.0 or n <= 0:
            return 0.0
        probs = np.asarray(counts, dtype=np.float64).ravel() / total
        probs = probs[probs > 0.0]
        return float(np.sum(probs * np.log2(probs * n)))

    @staticmethod
    def _write_coord_summary(f, name, counts):
        for table_idx in range(counts.shape[0]):
            coord_counts = counts[table_idx]
            total = int(coord_counts.sum())
            updated = int(np.count_nonzero(coord_counts))
            total_coords = int(coord_counts.size)
            kl_to_uniform = LUTUpdateStats._kl_to_uniform(coord_counts)
            f.write(
                f"{name},{table_idx},{total},{updated},{total_coords},"
                f"{float(coord_counts.mean()):.6f},"
                f"{float(coord_counts.var()):.6f},"
                f"{int(coord_counts.max()) if coord_counts.size else 0},"
                f"{total_coords - updated},{kl_to_uniform:.6f}\n"
            )

    @staticmethod
    def _write_distribution_summary(f, kind, name, component, counts):
        for table_idx in range(counts.shape[0]):
            table_counts = counts[table_idx]
            total = int(table_counts.sum())
            max_bin = int(np.argmax(table_counts)) if table_counts.size else -1
            f.write(
                f"{kind},{name},{component},{table_idx},{total},"
                f"{int(np.count_nonzero(table_counts))},{table_counts.size},"
                f"{float(table_counts.mean()):.6f},"
                f"{float(table_counts.var()):.6f},"
                f"{int(table_counts.max()) if table_counts.size else 0},{max_bin}\n"
            )

    @staticmethod
    def _write_distribution_top(f, kind, name, component, counts, top_n=20):
        for table_idx in range(counts.shape[0]):
            table_counts = counts[table_idx]
            total = int(table_counts.sum())
            if table_counts.size == 0:
                continue
            order = np.argsort(table_counts)[::-1][:min(top_n, table_counts.size)]
            for rank, bin_idx in enumerate(order, start=1):
                count = int(table_counts[bin_idx])
                if count == 0:
                    break
                share = float(count / total) if total else 0.0
                f.write(
                    f"top,{kind},{name},{component},{table_idx},{rank},"
                    f"{int(bin_idx)},,{count},{share:.8f}\n"
                )
            zero_bins = np.flatnonzero(table_counts == 0)[:top_n]
            for rank, bin_idx in enumerate(zero_bins, start=1):
                f.write(
                    f"zero,{kind},{name},{component},{table_idx},{rank},"
                    f"{int(bin_idx)},,0,0.00000000\n"
                )
            scarce_bins = np.flatnonzero(table_counts > 0)
            if scarce_bins.size:
                scarce_order = scarce_bins[np.argsort(table_counts[scarce_bins])[:top_n]]
                for rank, bin_idx in enumerate(scarce_order, start=1):
                    count = int(table_counts[bin_idx])
                    share = float(count / total) if total else 0.0
                    f.write(
                        f"scarce_nonzero,{kind},{name},{component},{table_idx},{rank},"
                        f"{int(bin_idx)},,{count},{share:.8f}\n"
                    )

    @staticmethod
    def _write_joint_summary(f, name, row_component, col_component, counts):
        for table_idx in range(counts.shape[0]):
            table_counts = counts[table_idx]
            flat = table_counts.ravel()
            total = int(flat.sum())
            max_flat = int(np.argmax(flat)) if flat.size else 0
            max_row, max_col = np.unravel_index(max_flat, table_counts.shape)
            f.write(
                f"joint,{name},{row_component}_x_{col_component},{table_idx},{total},"
                f"{int(np.count_nonzero(flat))},{flat.size},"
                f"{float(flat.mean()):.6f},"
                f"{float(flat.var()):.6f},"
                f"{int(flat.max()) if flat.size else 0},{int(max_row)}:{int(max_col)}\n"
            )

    @staticmethod
    def _write_joint_top(f, name, row_component, col_component, counts, top_n=20):
        component = f"{row_component}_x_{col_component}"
        for table_idx in range(counts.shape[0]):
            table_counts = counts[table_idx]
            flat = table_counts.ravel()
            total = int(flat.sum())
            if flat.size == 0:
                continue
            order = np.argsort(flat)[::-1][:min(top_n, flat.size)]
            for rank, flat_idx in enumerate(order, start=1):
                count = int(flat[flat_idx])
                if count == 0:
                    break
                row_idx, col_idx = np.unravel_index(int(flat_idx), table_counts.shape)
                share = float(count / total) if total else 0.0
                f.write(
                    f"top,joint,{name},{component},{table_idx},{rank},"
                    f"{int(row_idx)},{int(col_idx)},{count},{share:.8f}\n"
                )
            zero_flat = np.flatnonzero(flat == 0)[:top_n]
            for rank, flat_idx in enumerate(zero_flat, start=1):
                row_idx, col_idx = np.unravel_index(int(flat_idx), table_counts.shape)
                f.write(
                    f"zero,joint,{name},{component},{table_idx},{rank},"
                    f"{int(row_idx)},{int(col_idx)},0,0.00000000\n"
                )

    def save(self, path):
        arrays = {name: counts for name, counts in self.counts.items()}
        np.savez_compressed(path, **arrays)
        summary_path = path[:-4] + "_summary.csv" if path.endswith(".npz") else path + "_summary.csv"
        hist_path = path[:-4] + "_histogram.csv" if path.endswith(".npz") else path + "_histogram.csv"
        coord_stats_path = self.coord_stats_path(path)
        coord_summary_path = path[:-4] + "_coord_stats_summary.csv" if path.endswith(".npz") else path + "_coord_stats_summary.csv"
        j_stats_path = self.j_stats_path(path)
        j_summary_path = path[:-4] + "_j_stats_summary.csv" if path.endswith(".npz") else path + "_j_stats_summary.csv"
        j_top_path = path[:-4] + "_j_stats_top.csv" if path.endswith(".npz") else path + "_j_stats_top.csv"
        with open(summary_path, "w") as f:
            f.write("lut,table,total_updates,updated_rows,total_rows,mean_updates_per_row,var_updates_per_row,max_updates_on_row\n")
            for name in sorted(self.counts):
                counts = self.counts[name]
                for table_idx in range(counts.shape[0]):
                    row_counts = counts[table_idx]
                    f.write(
                        f"{name},{table_idx},{int(row_counts.sum())},"
                        f"{int(np.count_nonzero(row_counts))},{row_counts.size},"
                        f"{float(row_counts.mean()):.6f},"
                        f"{float(row_counts.var()):.6f},{int(row_counts.max())}\n"
                    )
        with open(hist_path, "w") as f:
            f.write("lut,table,updates_per_row,num_rows\n")
            for name in sorted(self.counts):
                counts = self.counts[name]
                for table_idx in range(counts.shape[0]):
                    hist = np.bincount(counts[table_idx].astype(np.int64))
                    for updates_per_row in range(hist.size - 1, -1, -1):
                        num_rows = int(hist[updates_per_row])
                        if num_rows:
                            f.write(f"{name},{table_idx},{updates_per_row},{num_rows}\n")

        coord_arrays = {name: counts for name, counts in self.coord_counts.items()}
        np.savez_compressed(coord_stats_path, **coord_arrays)
        with open(coord_summary_path, "w") as f:
            f.write("lut,table,total_coordinate_updates,updated_coordinates,total_coordinates,mean_updates_per_coordinate,var_updates_per_coordinate,max_updates_on_coordinate,zero_updated_coordinates,kl_to_uniform_bits\n")
            for name in sorted(self.coord_counts):
                self._write_coord_summary(f, name, self.coord_counts[name])

        j_arrays = {}
        for name, counts in self.j_counts.items():
            j_arrays[f"simple__{name}"] = counts
        for (name, component), counts in self.j_component_counts.items():
            j_arrays[f"component__{name}__{component}"] = counts
        for (name, row_component, col_component), counts in self.j_joint_counts.items():
            j_arrays[f"joint__{name}__{row_component}__{col_component}"] = counts
        np.savez_compressed(j_stats_path, **j_arrays)
        with open(j_summary_path, "w") as f:
            f.write("kind,lut,component,table,total_observations,observed_bins,total_bins,mean_count_per_bin,var_count_per_bin,max_count,max_bin\n")
            for name in sorted(self.j_counts):
                self._write_distribution_summary(f, "simple", name, "j", self.j_counts[name])
            for key in sorted(self.j_component_counts):
                name, component = key
                self._write_distribution_summary(
                    f, "component", name, component, self.j_component_counts[key])
            for key in sorted(self.j_joint_counts):
                name, row_component, col_component = key
                self._write_joint_summary(
                    f, name, row_component, col_component, self.j_joint_counts[key])
        with open(j_top_path, "w") as f:
            f.write("record_type,kind,lut,component,table,rank,bin,bin2,count,share\n")
            for name in sorted(self.j_counts):
                self._write_distribution_top(f, "simple", name, "j", self.j_counts[name])
            for key in sorted(self.j_component_counts):
                name, component = key
                self._write_distribution_top(
                    f, "component", name, component, self.j_component_counts[key])
            for key in sorted(self.j_joint_counts):
                name, row_component, col_component = key
                self._write_joint_top(
                    f, name, row_component, col_component, self.j_joint_counts[key])
        return summary_path, hist_path, coord_stats_path, coord_summary_path, j_stats_path, j_summary_path, j_top_path


def parse_lut_update_snapshot_steps(text):
    steps = set()
    for part in (text or "").split(','):
        part = part.strip()
        if not part:
            continue
        try:
            step = int(part)
        except ValueError as exc:
            raise ValueError(
                f"Bad --lut-update-stats-snapshot-steps value {text!r}; "
                "expected comma-separated positive integers") from exc
        if step < 1:
            raise ValueError(
                f"Bad --lut-update-stats-snapshot-steps value {text!r}; "
                "steps must be 1-based positive integers")
        steps.add(step)
    return steps


def lut_update_snapshot_path(path, step):
    base = Path(path)
    suffix = base.suffix or '.npz'
    if suffix == '.npz':
        return str(base.with_name(f"{base.stem}_step_{step}{suffix}"))
    return str(base.with_name(f"{base.name}_step_{step}{suffix}"))


def save_lut_update_stats(update_stats, base_path, step=None):
    (summary_path, hist_path, coord_stats_path, coord_summary_path,
     j_stats_path, j_summary_path, j_top_path) = update_stats.save(base_path)
    saved = {
        'stats_path': base_path,
        'summary_path': summary_path,
        'hist_path': hist_path,
        'coord_stats_path': coord_stats_path,
        'coord_summary_path': coord_summary_path,
        'j_stats_path': j_stats_path,
        'j_summary_path': j_summary_path,
        'j_top_path': j_top_path,
    }
    if step is not None:
        snapshot_path = lut_update_snapshot_path(base_path, step)
        (snapshot_summary, snapshot_hist, snapshot_coord_stats, snapshot_coord_summary,
         snapshot_j_stats, snapshot_j_summary, snapshot_j_top) = update_stats.save(snapshot_path)
        saved.update({
            'snapshot_path': snapshot_path,
            'snapshot_summary_path': snapshot_summary,
            'snapshot_hist_path': snapshot_hist,
            'snapshot_coord_stats_path': snapshot_coord_stats,
            'snapshot_coord_summary_path': snapshot_coord_summary,
            'snapshot_j_stats_path': snapshot_j_stats,
            'snapshot_j_summary_path': snapshot_j_summary,
            'snapshot_j_top_path': snapshot_j_top,
        })
    return saved


class NewAttentionHead:
    __slots__ = ['V', 'Final', 'Score', 'Value', 'Positional_encoding',
                 '_v_j', '_v_r_min', '_v_u_min',
                 'Q', '_q_j', '_q_r_min', '_q_u_min',
                 'K', '_k_j', '_k_r_min', '_k_u_min',
                 '_q_embed', '_k_embed', '_v_embed', '_qk_embed',
                 '_q_grad', '_k_grad', '_v_grad', '_pe_grad',
                 '_qkpe_j', '_qkpe_r_min', '_qkpe_u_min',
                 '_qk_mean', '_qk_inv_std',
                 '_final_v_j', '_final_v_r_min', '_final_v_u_min',
                 '_qkpe_a_arr', '_qkpe_b_arr', '_qkpe_bitmasks',
                 '_val_j', '_val_r_min', '_val_u_min',
                 '_pe_j', '_pe_r_min', '_pe_u_min',
                 '_attn_w', '_v_vecs']
    def __init__(self, args):
        N_T = args.n_t
        Q_N_T = args.attn_qk_n_t
        CS = args.context_size
        POSITIONAL_DIM = args.positional_dim
        ED = args.embedding_dim
        QD = args.attn_qk_dim
        soft = getattr(args, 'soft_attention', False)
        self.Positional_encoding = np.zeros((CS, QD), dtype=np.float32)
        n_pairs = CS * (CS - 1) // 2
        self._v_j = np.zeros((n_pairs, N_T), dtype=np.int32)
        self._v_r_min = np.zeros((n_pairs, N_T), dtype=np.int32)
        self._v_u_min = np.zeros((n_pairs, N_T), dtype=np.float32)
        self._q_embed = np.zeros((CS, QD), dtype=np.float32)
        self._k_embed = np.zeros((CS, QD), dtype=np.float32)
        self._v_embed = np.zeros((CS, ED), dtype=np.float32)
        self._qk_embed = np.zeros((n_pairs, QD), dtype=np.float32)
        self._q_grad = np.zeros((CS, QD), dtype=np.float32)
        self._k_grad = np.zeros((CS, QD), dtype=np.float32)
        self._v_grad = np.zeros((CS, ED), dtype=np.float32)
        self._pe_grad = np.zeros((CS, QD), dtype=np.float32)
        self._q_j = np.zeros((CS, Q_N_T), dtype=np.int32)
        self._q_r_min = np.zeros((CS, Q_N_T), dtype=np.int32)
        self._q_u_min = np.zeros((CS, Q_N_T), dtype=np.float32)
        self._k_j = np.zeros((CS, Q_N_T), dtype=np.int32)
        self._k_r_min = np.zeros((CS, Q_N_T), dtype=np.int32)
        self._k_u_min = np.zeros((CS, Q_N_T), dtype=np.float32)
        self._val_j = np.zeros((CS, N_T), dtype=np.int32)
        self._val_r_min = np.zeros((CS, N_T), dtype=np.int32)
        self._val_u_min = np.zeros((CS, N_T), dtype=np.float32)
        self._final_v_j = np.zeros((CS, N_T), dtype=np.int32)
        self._final_v_r_min = np.zeros((CS, N_T), dtype=np.int32)
        self._final_v_u_min = np.zeros((CS, N_T), dtype=np.float32)
        self._qkpe_j = np.zeros((n_pairs, N_T), dtype=np.int32)
        self._qkpe_r_min = np.zeros((n_pairs, N_T), dtype=np.int32)
        self._qkpe_u_min = np.zeros((n_pairs, N_T), dtype=np.float32)
        self._qk_mean = np.zeros(n_pairs, dtype=np.float32)
        self._qk_inv_std = np.ones(n_pairs, dtype=np.float32)
        self._qkpe_a_arr = np.zeros((N_T, POSITIONAL_DIM), dtype=np.int32)
        self._qkpe_b_arr = np.zeros((N_T, POSITIONAL_DIM), dtype=np.int32)
        self._qkpe_bitmasks = (1 << np.arange(POSITIONAL_DIM)).astype(np.int32)
        self._pe_j = np.zeros((CS, N_T), dtype=np.int32)
        self._pe_r_min = np.zeros((CS, N_T), dtype=np.int32)
        self._pe_u_min = np.zeros((CS, N_T), dtype=np.float32)
        if soft:
            self.Q = None
            self.K = None
            self.V = None
            self.Final = None
            self.Score = LUT(N_T)
            self.Value = LUT(N_T)
            self._attn_w = np.zeros((CS, CS), dtype=np.float32)
            self._v_vecs = np.zeros((CS, ED), dtype=np.float32)
        else:
            self.Q = LUT(Q_N_T)
            self.K = LUT(Q_N_T)
            self.V = LUT(N_T)
            self.Final = LUT(N_T)
            self.Score = None
            self.Value = None
            self._attn_w = None
            self._v_vecs = None



class AttentionHead:
    __slots__ = ['V', 'Score', 'Value', 'Positional_encoding',
                 '_v_j', '_v_r_min', '_v_u_min',
                 '_val_j', '_val_r_min', '_val_u_min',
                 '_pe_j', '_pe_r_min', '_pe_u_min',
                 '_attn_w', '_v_vecs']
    def __init__(self, args):
        N_T = args.n_t
        CS = args.context_size
        POSITIONAL_DIM = args.positional_dim
        ED = args.embedding_dim
        soft = getattr(args, 'soft_attention', False)
        self.Positional_encoding = np.zeros((CS, N_T, POSITIONAL_DIM), dtype=np.float32)
        self._v_j = np.zeros((CS, N_T), dtype=np.int32)
        self._v_r_min = np.zeros((CS, N_T), dtype=np.int32)
        self._v_u_min = np.zeros((CS, N_T), dtype=np.float32)
        self._pe_j = np.zeros((CS, N_T), dtype=np.int32)
        self._pe_r_min = np.zeros((CS, N_T), dtype=np.int32)
        self._pe_u_min = np.zeros((CS, N_T), dtype=np.float32)
        if soft:
            self.V = None
            self.Score = LUT(N_T)
            self.Value = LUT(N_T)
            self._val_j     = np.zeros((CS, N_T), dtype=np.int32)
            self._val_r_min = np.zeros((CS, N_T), dtype=np.int32)
            self._val_u_min = np.zeros((CS, N_T), dtype=np.float32)
            self._attn_w    = np.zeros((CS, CS),  dtype=np.float32)
            self._v_vecs    = np.zeros((CS, ED),  dtype=np.float32)
        else:
            self.V = LUT(N_T)
            self.Score = None
            self.Value = None
            self._val_j     = None
            self._val_r_min = None
            self._val_u_min = None
            self._attn_w    = None
            self._v_vecs    = None


class StandardOutputHead:
    __slots__ = ['unembedder', 'output', 'tokens', '_j', '_r_min', '_u_min']
    def __init__(self, args):
        N_T = args.n_t
        CS = args.context_size
        VOCAB_SIZE = args.vocab_size_output
        self.unembedder = LUT(N_T)
        self.output = np.zeros((CS, VOCAB_SIZE), dtype=np.float32)
        self.tokens = None  # set by load_targets
        self._j = np.zeros((CS, N_T), dtype=np.int32)
        self._r_min = np.zeros((CS, N_T), dtype=np.int32)
        self._u_min = np.zeros((CS, N_T), dtype=np.float32)

    def build(self, n_c, args):
        build_LUT(self.unembedder, n_c, args.vocab_size_output, args)

    def load_targets(self, tokens, context_size):
        self.tokens = tokens

    def forward(self, z, args):
        lut = self.unembedder
        cache_index_batch(lut, z, self._j, self._r_min, self._u_min)
        if HAS_NUMBA and lut.S.dtype == np.float32:
            _lut_forward_kernel(lut.S, lut.trees, self._j, self.output,
                                args.context_size, args.n_t, lut.y_dim)
        else:
            self.output[:] = 0.0
            self.output[:, :lut.y_dim] += _f32(lut.S[lut.trees, self._j]).sum(axis=1)

    def backward(self, x_grad, args, skip_s_update=False, update_stats=None, stats_name='output'):
        global learning_rate
        lut = self.unembedder
        if update_stats is not None and not skip_s_update:
            update_stats.record_simple(stats_name, lut, self._j)
            update_stats.record_simple_coords(stats_name, lut, self.output)
        if HAS_NUMBA and lut.S.dtype == np.float32:
            if skip_s_update:
                _lut_backward_kernel_noupdate(lut.S, lut.trees, self._j, self._r_min, self._u_min,
                                              self.output, x_grad, lut.a_arr, lut.b_arr,
                                              lut.y_dim, args.context_size, args.n_t)
            else:
                _lut_backward_kernel(lut.S, lut.trees, self._j, self._r_min, self._u_min,
                                      self.output, x_grad, lut.a_arr, lut.b_arr,
                                      lut.y_dim, args.context_size, args.n_t,
                                      np.float32(learning_rate))
        else:
            trees = lut.trees
            for pos in range(args.context_size):
                j_bins = self._j[pos]
                jbar_bins = j_bins ^ (1 << self._r_min[pos])
                S_j = _f32(lut.S[trees, j_bins])
                S_jbar = _f32(lut.S[trees, jbar_bins])
                gi = ((S_jbar - S_j) * self.output[pos, :lut.y_dim]).sum(axis=1)
                sign_u = np.where(self._u_min[pos] > 0, 1.0, -1.0)
                v = gi * (-0.5 * sign_u / (1 + np.abs(self._u_min[pos]))**2)
                n = x_grad.shape[1]
                idx_a = lut.a_arr[trees, self._r_min[pos]]
                idx_b = lut.b_arr[trees, self._r_min[pos]]
                x_grad[pos] += np.bincount(idx_a, weights=v, minlength=n)
                x_grad[pos] -= np.bincount(idx_b, weights=v, minlength=n)
                if not skip_s_update:
                    lut.S[trees, j_bins] -= learning_rate * self.output[pos, :lut.y_dim]

    def compute_gradients(self, args):
        CS = args.context_size
        targets = self.tokens[:CS] if args.POS_task else self.tokens[1:CS + 1]
        valid = targets != IGNORE_INDEX
        if HAS_NUMBA and valid.all():
            _softmax_cross_entropy_kernel(self.output, targets,
                                          CS, args.vocab_size_output)
            return

        x = self.output
        x -= x.max(axis=1, keepdims=True)
        np.exp(x, out=x)
        x /= x.sum(axis=1, keepdims=True)
        x[~valid] = 0.0
        if valid.any():
            rows = np.nonzero(valid)[0]
            x[rows, targets[valid]] -= 1.0

    def sample_token(self, args):
        VOCAB_SIZE = args.vocab_size_output
        softmax(self.output[args.context_size - 1], VOCAB_SIZE, args.temperature)
        return sample(self.output[args.context_size - 1], VOCAB_SIZE)

    def training_loss(self, args):
        CS = args.context_size
        x = self.output[:CS]
        targets = self.tokens[:CS] if args.POS_task else self.tokens[1:CS + 1]
        valid = targets != IGNORE_INDEX
        if not valid.any():
            return 0.0
        x_valid = x[valid]
        targets_valid = targets[valid]
        x_max = x_valid.max(axis=1)
        log_sum_exp = x_max + np.log(np.exp(x_valid - x_max[:, None]).sum(axis=1))
        return float((log_sum_exp - x_valid[np.arange(len(targets_valid)), targets_valid]).mean())

    def validation_loss(self, args):
        if args.POS_task:
            return self.validation_loss_all_positions(args)
        last = args.context_size - 1
        target = int(self.tokens[args.context_size])
        if target == IGNORE_INDEX:
            return None
        softmax(self.output[last], args.vocab_size_output, 1.0)
        prob = max(self.output[last][target], 1e-30)
        return -math.log(prob)

    def validation_accuracy(self, args):
        if args.POS_task:
            return self.validation_accuracy_all_positions(args)
        last = args.context_size - 1
        target = int(self.tokens[args.context_size])
        if target == IGNORE_INDEX:
            return None
        pred = int(np.argmax(self.output[last]))
        return 1.0 if pred == target else 0.0

    def validation_loss_all_positions(self, args):
        CS = args.context_size
        targets = self.tokens[:CS]
        valid = targets != IGNORE_INDEX
        if not valid.any():
            return None
        x = self.output[:CS][valid]
        targets = targets[valid]
        x_max = x.max(axis=1)
        log_sum_exp = x_max + np.log(np.exp(x - x_max[:, None]).sum(axis=1))
        losses = log_sum_exp - x[np.arange(len(targets)), targets]
        return float(losses.sum()), int(len(targets))

    def validation_accuracy_all_positions(self, args):
        CS = args.context_size
        targets = self.tokens[:CS]
        valid = targets != IGNORE_INDEX
        if not valid.any():
            return None
        preds = np.argmax(self.output[:CS], axis=1)
        correct = int((preds[valid] == targets[valid]).sum())
        return correct, int(valid.sum())


class FactoredOutputHead:
    __slots__ = ['unembedder_hi', 'output_hi',
                 'unembedder_lo', 'output_lo',
                 'tokens_hi', 'tokens_lo',
                 '_hi_j', '_hi_r_min', '_hi_u_min',
                 '_lo_j', '_lo_r_min', '_lo_u_min']
    def __init__(self, args):
        N_T = args.n_t
        CS = args.context_size
        VOCAB_HI = args.vocab_hi
        self.unembedder_hi = LUT(N_T)
        self.output_hi = np.zeros((CS, VOCAB_HI), dtype=np.float32)
        self.unembedder_lo = LUT(N_T)
        self.output_lo = np.zeros((CS, 256), dtype=np.float32)
        self.tokens_hi = np.zeros(CS + 1, dtype=np.int32)
        self.tokens_lo = np.zeros(CS + 1, dtype=np.int32)
        self._hi_j = np.zeros((CS, N_T), dtype=np.int32)
        self._hi_r_min = np.zeros((CS, N_T), dtype=np.int32)
        self._hi_u_min = np.zeros((CS, N_T), dtype=np.float32)
        self._lo_j = np.zeros((CS, N_T), dtype=np.int32)
        self._lo_r_min = np.zeros((CS, N_T), dtype=np.int32)
        self._lo_u_min = np.zeros((CS, N_T), dtype=np.float32)

    def build(self, n_c, args):
        build_LUT(self.unembedder_hi, n_c, args.vocab_hi, args)
        build_LUT(self.unembedder_lo, n_c, 256, args)

    def load_targets(self, tokens, context_size):
        t = tokens[:context_size + 1]
        self.tokens_hi[:context_size + 1] = t // 256
        self.tokens_lo[:context_size + 1] = t % 256

    def forward(self, z, args):
        CS = args.context_size
        NT = args.n_t
        cache_index_batch(self.unembedder_hi, z, self._hi_j, self._hi_r_min, self._hi_u_min)
        cache_index_batch(self.unembedder_lo, z, self._lo_j, self._lo_r_min, self._lo_u_min)
        if HAS_NUMBA and self.unembedder_hi.S.dtype == np.float32:
            _lut_forward_kernel(self.unembedder_hi.S, self.unembedder_hi.trees, self._hi_j,
                                self.output_hi, CS, NT, self.unembedder_hi.y_dim)
            _lut_forward_kernel(self.unembedder_lo.S, self.unembedder_lo.trees, self._lo_j,
                                self.output_lo, CS, NT, self.unembedder_lo.y_dim)
        else:
            self.output_hi[:] = 0.0
            self.output_lo[:] = 0.0
            self.output_hi[:, :self.unembedder_hi.y_dim] += _f32(self.unembedder_hi.S[self.unembedder_hi.trees, self._hi_j]).sum(axis=1)
            self.output_lo[:, :self.unembedder_lo.y_dim] += _f32(self.unembedder_lo.S[self.unembedder_lo.trees, self._lo_j]).sum(axis=1)

    def backward(self, x_grad, args, skip_s_update=False, update_stats=None, stats_prefix='output'):
        global learning_rate
        for stats_suffix, lut, j, r_min, u_min, output in [
            ('hi', self.unembedder_hi, self._hi_j, self._hi_r_min, self._hi_u_min, self.output_hi),
            ('lo', self.unembedder_lo, self._lo_j, self._lo_r_min, self._lo_u_min, self.output_lo),
        ]:
            if update_stats is not None and not skip_s_update:
                update_stats.record_simple(f'{stats_prefix}_{stats_suffix}', lut, j)
                update_stats.record_simple_coords(f'{stats_prefix}_{stats_suffix}', lut, output)
            if HAS_NUMBA and lut.S.dtype == np.float32:
                if skip_s_update:
                    _lut_backward_kernel_noupdate(lut.S, lut.trees, j, r_min, u_min,
                                                  output, x_grad, lut.a_arr, lut.b_arr,
                                                  lut.y_dim, args.context_size, args.n_t)
                else:
                    _lut_backward_kernel(lut.S, lut.trees, j, r_min, u_min,
                                          output, x_grad, lut.a_arr, lut.b_arr,
                                          lut.y_dim, args.context_size, args.n_t,
                                          np.float32(learning_rate))
            else:
                trees = lut.trees
                for pos in range(args.context_size):
                    j_bins = j[pos]
                    jbar_bins = j_bins ^ (1 << r_min[pos])
                    S_j = _f32(lut.S[trees, j_bins])
                    S_jbar = _f32(lut.S[trees, jbar_bins])
                    gi = ((S_jbar - S_j) * output[pos, :lut.y_dim]).sum(axis=1)
                    sign_u = np.where(u_min[pos] > 0, 1.0, -1.0)
                    v = gi * (-0.5 * sign_u / (1 + np.abs(u_min[pos]))**2)
                    n = x_grad.shape[1]
                    idx_a = lut.a_arr[trees, r_min[pos]]
                    idx_b = lut.b_arr[trees, r_min[pos]]
                    x_grad[pos] += np.bincount(idx_a, weights=v, minlength=n)
                    x_grad[pos] -= np.bincount(idx_b, weights=v, minlength=n)
                    if not skip_s_update:
                        lut.S[trees, j_bins] -= learning_rate * output[pos, :lut.y_dim]

    def compute_gradients(self, args):
        VOCAB_HI = args.vocab_hi
        for pos in range(args.context_size):
            softmax(self.output_hi[pos], VOCAB_HI, 1.0)
            self.output_hi[pos][self.tokens_hi[pos + 1]] -= 1.0
            softmax(self.output_lo[pos], 256, 1.0)
            self.output_lo[pos][self.tokens_lo[pos + 1]] -= 1.0

    def sample_token(self, args):
        VOCAB_SIZE = args.vocab_size_output
        VOCAB_HI = args.vocab_hi
        softmax(self.output_hi[args.context_size - 1], VOCAB_HI, args.temperature)
        hi = sample(self.output_hi[args.context_size - 1], VOCAB_HI)
        softmax(self.output_lo[args.context_size - 1], 256, args.temperature)
        lo = sample(self.output_lo[args.context_size - 1], 256)
        token_id = hi * 256 + lo
        if token_id >= VOCAB_SIZE:
            token_id = hi * 256
        return token_id

    def training_loss(self, args):
        """Mean cross-entropy over all context positions (hi + lo components)."""
        CS = args.context_size
        loss = 0.0
        for x, targets in [(self.output_hi[:CS], self.tokens_hi[1:CS + 1]),
                            (self.output_lo[:CS], self.tokens_lo[1:CS + 1])]:
            x_max = x.max(axis=1)
            log_sum_exp = x_max + np.log(np.exp(x - x_max[:, None]).sum(axis=1))
            loss += float((log_sum_exp - x[np.arange(CS), targets]).mean())
        return loss

    def validation_loss(self, args):
        last = args.context_size - 1
        softmax(self.output_hi[last], args.vocab_hi, 1.0)
        softmax(self.output_lo[last], 256, 1.0)
        target_hi = self.tokens_hi[args.context_size]
        target_lo = self.tokens_lo[args.context_size]
        prob_hi = max(self.output_hi[last][target_hi], 1e-30)
        prob_lo = max(self.output_lo[last][target_lo], 1e-30)
        return -math.log(prob_hi) + -math.log(prob_lo)

    def validation_accuracy(self, args):
        last = args.context_size - 1
        pred_hi = int(np.argmax(self.output_hi[last]))
        pred_lo = int(np.argmax(self.output_lo[last]))
        target_hi = int(self.tokens_hi[args.context_size])
        target_lo = int(self.tokens_lo[args.context_size])
        return 1.0 if pred_hi == target_hi and pred_lo == target_lo else 0.0


class Model:
    __slots__ = ['Token_embedder', 'tokens', 'z', '_x_buf', 'FFN',
                 '_ffn_j', '_ffn_r_min', '_ffn_u_min', 'head', 'output_head',
                 'ln_attn_gamma', 'ln_attn_beta', 'ln_ffn_gamma', 'ln_ffn_beta',
                 '_ln_attn_xhat', '_ln_attn_rstd', '_ln_ffn_xhat', '_ln_ffn_rstd',
                 '_ln_ffn_buf', '_drop_attn_mask', '_drop_ffn_mask']
    def __init__(self, args):
        VOCAB_SIZE = args.vocab_size_input
        EMBEDDING_DIM = args.embedding_dim
        NUM_LAYERS = args.num_layers
        NUM_HEADS = args.num_heads
        N_T = args.n_t
        CS = args.context_size
        new_attention = args.new_attention

        self.Token_embedder = np.zeros((VOCAB_SIZE, EMBEDDING_DIM), dtype=np.float32)
        self.tokens = np.zeros(CS + 1, dtype=np.int32)
        self.z = np.zeros((CS, EMBEDDING_DIM), dtype=np.float32)
        self._x_buf = np.zeros_like(self.z)

        self.FFN = [LUT(N_T) for _ in range(NUM_LAYERS)]
        self._ffn_j = [np.zeros((CS, N_T), dtype=np.int32) for _ in range(NUM_LAYERS)]
        self._ffn_r_min = [np.zeros((CS, N_T), dtype=np.int32) for _ in range(NUM_LAYERS)]
        self._ffn_u_min = [np.zeros((CS, N_T), dtype=np.float32) for _ in range(NUM_LAYERS)]

        if new_attention:
            self.head = [[NewAttentionHead(args) for _ in range(NUM_HEADS)] for _ in range(NUM_LAYERS)]
        else:
            self.head = [[AttentionHead(args) for _ in range(NUM_HEADS)] for _ in range(NUM_LAYERS)]

        if args.factored_output:
            self.output_head = FactoredOutputHead(args)
        else:
            self.output_head = StandardOutputHead(args)

        # LayerNorm parameters (NL x ED)
        self.ln_attn_gamma = np.ones((NUM_LAYERS, EMBEDDING_DIM), dtype=np.float32)
        self.ln_attn_beta  = np.zeros((NUM_LAYERS, EMBEDDING_DIM), dtype=np.float32)
        self.ln_ffn_gamma  = np.ones((NUM_LAYERS, EMBEDDING_DIM), dtype=np.float32)
        self.ln_ffn_beta   = np.zeros((NUM_LAYERS, EMBEDDING_DIM), dtype=np.float32)
        # LN caches
        self._ln_attn_xhat = [np.zeros((CS, EMBEDDING_DIM), np.float32) for _ in range(NUM_LAYERS)]
        self._ln_attn_rstd = [np.zeros((CS, 1), np.float32) for _ in range(NUM_LAYERS)]
        self._ln_ffn_xhat  = [np.zeros((CS, EMBEDDING_DIM), np.float32) for _ in range(NUM_LAYERS)]
        self._ln_ffn_rstd  = [np.zeros((CS, 1), np.float32) for _ in range(NUM_LAYERS)]
        self._ln_ffn_buf   = np.zeros((CS, EMBEDDING_DIM), dtype=np.float32)
        # Dropout masks
        self._drop_attn_mask = [None] * NUM_LAYERS
        self._drop_ffn_mask  = [None] * NUM_LAYERS


class TrainingData:
    __slots__ = ['data', 'pos_data', 'length',
                 'val_data', 'val_pos_data', 'val_length',
                 'testing_input_data']
    def __init__(self):
        self.data = None
        self.pos_data = None
        self.length = 0
        self.val_data = None
        self.val_pos_data = None
        self.val_length = 0
        self.testing_input_data = None


class SparseGradAccumulator:
    """Lightweight per-batch-element gradient storage (~1 MB vs ~3 GB for dense).

    Instead of allocating full-size S gradient buffers, we save only the small
    y_grad arrays during backward.  The actual S updates are replayed at apply
    time using the saved y_grads together with the cached index arrays that
    already live on each BatchElement.
    """
    __slots__ = ['ffn_y_grads', 'attn_y_grads', 'pe_grads',
                 'attn_score_grads', 'attn_val_grads',
                 'new_attn_q_grads', 'new_attn_k_grads', 'new_attn_v_grads',
                 'ln_attn_dgamma', 'ln_attn_dbeta', 'ln_ffn_dgamma', 'ln_ffn_dbeta']

    def __init__(self, args):
        NL = args.num_layers
        NH = args.num_heads
        CS = args.context_size
        NT = args.n_t
        PD = args.positional_dim
        ED = args.embedding_dim
        QD = getattr(args, 'attn_qk_dim', ED)
        self.ffn_y_grads = [None] * NL      # filled during backward
        self.attn_y_grads = [None] * NL     # filled during backward
        if getattr(args, 'new_attention', False):
            self.pe_grads = [[np.zeros((CS, QD), dtype=np.float32)
                              for _ in range(NH)] for _ in range(NL)]
            self.new_attn_q_grads = [[np.zeros((CS, QD), dtype=np.float32)
                                      for _ in range(NH)] for _ in range(NL)]
            self.new_attn_k_grads = [[np.zeros((CS, QD), dtype=np.float32)
                                      for _ in range(NH)] for _ in range(NL)]
            self.new_attn_v_grads = [[np.zeros((CS, ED), dtype=np.float32)
                                      for _ in range(NH)] for _ in range(NL)]
            self.attn_score_grads = None
            self.attn_val_grads = None
        else:
            self.pe_grads = [[np.zeros((CS, NT, PD), dtype=np.float32)
                              for _ in range(NH)] for _ in range(NL)]
            self.new_attn_q_grads = None
            self.new_attn_k_grads = None
            self.new_attn_v_grads = None
            if getattr(args, 'soft_attention', False):
                self.attn_score_grads = [[np.zeros((CS, CS), dtype=np.float32)
                                          for _ in range(NH)] for _ in range(NL)]
                self.attn_val_grads   = [[np.zeros((CS, ED), dtype=np.float32)
                                          for _ in range(NH)] for _ in range(NL)]
            else:
                self.attn_score_grads = None
                self.attn_val_grads   = None
        self.ln_attn_dgamma = np.zeros((NL, ED), dtype=np.float32)
        self.ln_attn_dbeta  = np.zeros((NL, ED), dtype=np.float32)
        self.ln_ffn_dgamma  = np.zeros((NL, ED), dtype=np.float32)
        self.ln_ffn_dbeta   = np.zeros((NL, ED), dtype=np.float32)

    def zero(self):
        for i in range(len(self.ffn_y_grads)):
            self.ffn_y_grads[i] = None
        for i in range(len(self.attn_y_grads)):
            self.attn_y_grads[i] = None
        if self.attn_score_grads is not None:
            for layer in self.attn_score_grads:
                for g in layer:
                    g[:] = 0
            for layer in self.attn_val_grads:
                for g in layer:
                    g[:] = 0
        if self.new_attn_q_grads is not None:
            for grads in (self.new_attn_q_grads, self.new_attn_k_grads, self.new_attn_v_grads):
                for layer in grads:
                    for g in layer:
                        g[:] = 0
        for layer in self.pe_grads:
            for g in layer:
                g[:] = 0
        self.ln_attn_dgamma[:] = 0
        self.ln_attn_dbeta[:] = 0
        self.ln_ffn_dgamma[:] = 0
        self.ln_ffn_dbeta[:] = 0


def _share_lut(dst, src):
    """Make dst LUT share src's S table and parameters."""
    dst.S = src.S
    dst.y_dim = src.y_dim
    dst.a_arr = src.a_arr
    dst.b_arr = src.b_arr
    dst.bitmasks = src.bitmasks
    dst.trees = src.trees


class BatchElement:
    """Per-sequence scratch space that duck-types as Model, sharing S tables."""
    __slots__ = ['Token_embedder', 'tokens', 'z', '_x_buf', 'FFN',
                 '_ffn_j', '_ffn_r_min', '_ffn_u_min', 'head', 'output_head',
                 'grad_accum',
                 'ln_attn_gamma', 'ln_attn_beta', 'ln_ffn_gamma', 'ln_ffn_beta',
                 '_ln_attn_xhat', '_ln_attn_rstd', '_ln_ffn_xhat', '_ln_ffn_rstd',
                 '_ln_ffn_buf', '_drop_attn_mask', '_drop_ffn_mask']

    def __init__(self, m, args):
        CS = args.context_size
        ED = args.embedding_dim
        NT = args.n_t
        NL = args.num_layers
        NH = args.num_heads

        # Shared (read-only during batch processing)
        self.Token_embedder = m.Token_embedder
        self.FFN = m.FFN  # shared LUT objects (S tables, a/b arrays)
        # Shared LN parameters
        self.ln_attn_gamma = m.ln_attn_gamma
        self.ln_attn_beta  = m.ln_attn_beta
        self.ln_ffn_gamma  = m.ln_ffn_gamma
        self.ln_ffn_beta   = m.ln_ffn_beta

        # Per-element scratch
        self.tokens = np.zeros(CS + 1, dtype=np.int32)
        self.z = np.zeros((CS, ED), dtype=np.float32)
        self._x_buf = np.zeros_like(self.z)
        self._ffn_j = [np.zeros((CS, NT), dtype=np.int32) for _ in range(NL)]
        self._ffn_r_min = [np.zeros((CS, NT), dtype=np.int32) for _ in range(NL)]
        self._ffn_u_min = [np.zeros((CS, NT), dtype=np.float32) for _ in range(NL)]

        # Per-element attention heads (own scratch, shared LUTs and PE)
        if getattr(args, 'new_attention', False):
            self.head = [[NewAttentionHead(args) for _ in range(NH)] for _ in range(NL)]
        else:
            self.head = [[AttentionHead(args) for _ in range(NH)] for _ in range(NL)]
        for l in range(NL):
            for h in range(NH):
                if getattr(args, 'new_attention', False):
                    _share_lut(self.head[l][h].Q, m.head[l][h].Q)
                    _share_lut(self.head[l][h].K, m.head[l][h].K)
                    _share_lut(self.head[l][h].V, m.head[l][h].V)
                    _share_lut(self.head[l][h].Final, m.head[l][h].Final)
                    self.head[l][h]._qkpe_a_arr = m.head[l][h]._qkpe_a_arr
                    self.head[l][h]._qkpe_b_arr = m.head[l][h]._qkpe_b_arr
                    self.head[l][h]._qkpe_bitmasks = m.head[l][h]._qkpe_bitmasks
                elif getattr(args, 'soft_attention', False):
                    _share_lut(self.head[l][h].Score, m.head[l][h].Score)
                    _share_lut(self.head[l][h].Value, m.head[l][h].Value)
                else:
                    _share_lut(self.head[l][h].V, m.head[l][h].V)
                self.head[l][h].Positional_encoding = m.head[l][h].Positional_encoding

        # Per-element output head (own output buffers, shared LUT)
        if args.factored_output:
            self.output_head = FactoredOutputHead(args)
            _share_lut(self.output_head.unembedder_hi, m.output_head.unembedder_hi)
            _share_lut(self.output_head.unembedder_lo, m.output_head.unembedder_lo)
        else:
            self.output_head = StandardOutputHead(args)
            _share_lut(self.output_head.unembedder, m.output_head.unembedder)

        # Per-element LN caches and dropout masks
        self._ln_attn_xhat = [np.zeros((CS, ED), np.float32) for _ in range(NL)]
        self._ln_attn_rstd = [np.zeros((CS, 1), np.float32) for _ in range(NL)]
        self._ln_ffn_xhat  = [np.zeros((CS, ED), np.float32) for _ in range(NL)]
        self._ln_ffn_rstd  = [np.zeros((CS, 1), np.float32) for _ in range(NL)]
        self._ln_ffn_buf   = np.zeros((CS, ED), dtype=np.float32)
        self._drop_attn_mask = [None] * NL
        self._drop_ffn_mask  = [None] * NL

        # Sparse gradient accumulators (saves y_grads, not full S buffers)
        self.grad_accum = SparseGradAccumulator(args)


# ---------------------------------------------------------------------------
# LUT operations
# ---------------------------------------------------------------------------

def build_LUT_custom(lut, total_n_c, y_dim, args, n_t=None, n_c=None,
                     input_dim=None, s_n_t=None, trees=None):
    requested_n_t = args.n_t if n_t is None else n_t
    N_C = args.n_c if n_c is None else n_c
    INPUT_DIM = args.embedding_dim if input_dim is None else input_dim
    mode = getattr(args, 'lut_comparison_mode', 'random')
    s_dtype = np.float16 if getattr(args, 'fp16', False) else np.float32

    a_arr, b_arr = build_comparison_arrays(mode, requested_n_t, N_C, INPUT_DIM)

    N_T = a_arr.shape[0]
    S_N_T = N_T if s_n_t is None else s_n_t
    lut.y_dim = y_dim
    if trees is None:
        lut.trees = np.arange(N_T, dtype=np.int32)
    else:
        trees = np.asarray(trees, dtype=np.int32)
        if trees.shape != (N_T,):
            raise ValueError(
                f"trees mapping has shape {trees.shape}; expected ({N_T},)")
        if trees.size and (trees.min() < 0 or trees.max() >= S_N_T):
            raise ValueError(
                f"trees mapping must reference physical trees in [0, {S_N_T})")
        lut.trees = trees
    num_bins = 1 << total_n_c
    lut.S = np.zeros((S_N_T, num_bins, y_dim), dtype=s_dtype)
    lut.a_arr = a_arr
    lut.b_arr = b_arr
    lut.bitmasks = (1 << np.arange(N_C)).astype(np.int32)


def build_LUT(lut, total_n_c, y_dim, args):
    build_LUT_custom(lut, total_n_c, y_dim, args)


def cache_index_batch(lut, x_batch, j_out, r_min_out, u_min_out):
    """Batch cache_index across all positions. x_batch: (CS, dim), outputs: (CS, N_T)."""
    if HAS_NUMBA and x_batch.dtype == np.float32:
        _cache_index_kernel(x_batch, lut.a_arr, lut.b_arr, lut.bitmasks,
                            j_out, r_min_out, u_min_out,
                            x_batch.shape[0], lut.a_arr.shape[0], lut.a_arr.shape[1])
    else:
        u = x_batch[:, lut.a_arr] - x_batch[:, lut.b_arr]      # (CS, N_T, N_C)
        j_out[:] = ((u > 0).astype(np.int32) * lut.bitmasks).sum(axis=2)
        abs_u = np.abs(u)
        r_min_out[:] = abs_u.argmin(axis=2)
        cs_idx = np.arange(x_batch.shape[0])[:, np.newaxis]
        tree_idx = np.arange(lut.a_arr.shape[0])
        u_min_out[:] = u[cs_idx, tree_idx, r_min_out]


def cache_PE_index_batch(u_batch, j_out, r_min_out, u_min_out, args):
    """Batch cache_PE_index across all positions. u_batch: (CS, N_T, PD), outputs: (CS, N_T)."""
    if HAS_NUMBA and u_batch.dtype == np.float32:
        _cache_pe_index_kernel(u_batch, args.pe_bitmasks,
                               j_out, r_min_out, u_min_out,
                               u_batch.shape[0], u_batch.shape[1], u_batch.shape[2])
    else:
        j_out[:] = ((u_batch > 0).astype(np.int32) * args.pe_bitmasks).sum(axis=2)
        abs_u = np.abs(u_batch)
        r_min_out[:] = abs_u.argmin(axis=2)
        cs_idx = np.arange(u_batch.shape[0])[:, np.newaxis]
        tree_idx = np.arange(u_batch.shape[1])
        u_min_out[:] = u_batch[cs_idx, tree_idx, r_min_out]


def _f32(arr):
    """Cast to float32 if needed (no-op if already float32)."""
    return arr if arr.dtype == np.float32 else arr.astype(np.float32)


def _normalize_vector_to_target(vec, target_std, eps):
    mean = np.float32(vec.mean())
    centered = vec - mean
    inv_std = np.float32(1.0) / np.sqrt(np.float32((centered * centered).mean()) + eps)
    return np.float32(target_std) * centered * inv_std


def _normalize_vector_backward(vec, grad, target_std, eps):
    mean = np.float32(vec.mean())
    centered = vec - mean
    inv_std = np.float32(1.0) / np.sqrt(np.float32((centered * centered).mean()) + eps)
    xhat = centered * inv_std
    grad_mean = np.float32(grad.mean())
    grad_xhat_mean = np.float32((grad * xhat).mean())
    return np.float32(target_std) * inv_std * (grad - grad_mean - xhat * grad_xhat_mean)


def layernorm_forward(x, gamma, beta, eps=1e-5):
    """x: (CS, ED). Returns (y, xhat, rstd)."""
    mean = x.mean(axis=1, keepdims=True)
    rstd = 1.0 / np.sqrt(x.var(axis=1, keepdims=True) + eps)
    xhat = (x - mean) * rstd
    return xhat * gamma + beta, xhat, rstd


def layernorm_backward(dy, xhat, rstd, gamma):
    """Returns (dx, dgamma, dbeta)."""
    N = dy.shape[1]
    dgamma = (dy * xhat).sum(axis=0)
    dbeta  = dy.sum(axis=0)
    dx = (gamma * rstd / N) * (
        N * dy
        - dy.sum(axis=1, keepdims=True)
        - xhat * (dy * xhat).sum(axis=1, keepdims=True)
    )
    return dx, dgamma, dbeta


def CONCATENATE_vec(Q, P, PE, args):
    """Vectorized CONCATENATE for arrays of indices."""
    return (Q << args.shift_qk) | (P << args.positional_dim) | PE

def CONCATENATE_QK_PE(QK, PE, args):
    """Vectorized CONCATENATE for arrays of indices."""
    return (QK << args.positional_dim) | PE


def record_attention_update_rows(update_stats, name, lut, jV, jPE, args):
    update_stats.record_attention_rows(name, lut, jV, jPE, args)


# ---------------------------------------------------------------------------
# Numba JIT kernels (fused loops — no temporary allocations)
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _attention_forward_kernel(S, all_jV, all_jPE, y, shift_qk, pd, CS, N_T, y_dim):
    """Fused attention forward: parallel over independent output cells."""
    for work in prange((CS - 1) * y_dim):
        pos = work // y_dim + 1
        d = work - (pos - 1) * y_dim
        acc = np.float32(0.0)
        for pos1 in range(pos):
            pe_pos = pos - pos1
            for t in range(N_T):
                j = (all_jV[pos, t] << shift_qk) | (all_jV[pos1, t] << pd) | all_jPE[pe_pos, t]
                acc += S[t, j, d]
        y[pos, d] += acc

@njit(parallel=True, cache=True)
def _new_attention_forward_kernel(
    QS, KS, VS, FS,
    q_j, k_j, val_j,
    q_embed, k_embed, v_embed,
    qkpe_j, qkpe_r_min, qkpe_u_min,
    qk_mean_arr, qk_inv_std_arr,
    final_v_j, final_v_r_min, final_v_u_min,
    y,
    PE,
    q_trees, k_trees, v_trees, final_trees,
    final_v_a_arr, final_v_b_arr, final_v_bitmasks,
    qkpe_a_arr, qkpe_b_arr, qkpe_bitmasks,
    pd, CS, N_T, N_C, Q_N_T, QD, y_dim, qkpe_direct_sign,
    normalize_qk, qk_norm_target_std, qk_norm_eps
):
    """Fused forward for Q(x_i)+K(x_j)+PE and V(x_j) into Final LUT."""
    n_pairs = CS * (CS - 1) // 2

    # Phase 1: token-side Q and K embeddings.
    for work in prange(CS * QD):
        pos = work // QD
        d = work - pos * QD
        q_acc = np.float32(0.0)
        k_acc = np.float32(0.0)
        for t in range(Q_N_T):
            q_acc += QS[q_trees[t], q_j[pos, t], d]
            k_acc += KS[k_trees[t], k_j[pos, t], d]
        q_embed[pos, d] = q_acc
        k_embed[pos, d] = k_acc

    # Phase 2: token-side V embeddings.
    for work in prange(CS * y_dim):
        pos = work // y_dim
        d = work - pos * y_dim
        v_acc = np.float32(0.0)
        for t in range(N_T):
            v_acc += VS[v_trees[t], val_j[pos, t], d]
        v_embed[pos, d] = v_acc

    # Phase 3: final value-side row bits from V(x_j).
    for work in prange(CS * N_T):
        pos = work // N_T
        t = work - pos * N_T
        j = np.int32(0)
        min_abs_u = np.float32(1e38)
        r_min = np.int32(0)
        u_min_val = np.float32(0.0)

        for c in range(N_C):
            u_val = v_embed[pos, final_v_a_arr[t, c]] - v_embed[pos, final_v_b_arr[t, c]]
            if u_val > np.float32(0.0):
                j |= final_v_bitmasks[c]
            au = abs(u_val)
            if au < min_abs_u:
                min_abs_u = au
                r_min = np.int32(c)
                u_min_val = u_val

        final_v_j[pos, t] = j
        final_v_r_min[pos, t] = r_min
        final_v_u_min[pos, t] = u_min_val

    # Phase 4: final qkpe-side row bits from Q(x_i)+K(x_j)+PE(i-j).
    if normalize_qk:
        active_dim = pd if qkpe_direct_sign else QD
        for pair in prange(n_pairs):
            tok_i = int((np.sqrt(8.0 * pair + 1.0) + 1.0) * 0.5)
            tok_j = pair - (tok_i * (tok_i - 1)) // 2
            pe_pos = tok_i - tok_j
            qk_mean_val = np.float32(0.0)

            for d in range(active_dim):
                qk_mean_val += q_embed[tok_i, d] + k_embed[tok_j, d]
            qk_mean_val /= np.float32(active_dim)

            qk_var = np.float32(0.0)
            for d in range(active_dim):
                qk_centered = q_embed[tok_i, d] + k_embed[tok_j, d] - qk_mean_val
                qk_var += qk_centered * qk_centered
            qk_var /= np.float32(active_dim)

            qk_mean_arr[pair] = qk_mean_val
            qk_inv_std_arr[pair] = np.float32(1.0) / np.sqrt(qk_var + qk_norm_eps)

    for work in prange(n_pairs * N_T):
        pair = work // N_T
        t = work - pair * N_T
        tok_i = int((np.sqrt(8.0 * pair + 1.0) + 1.0) * 0.5)
        tok_j = pair - (tok_i * (tok_i - 1)) // 2
        pe_pos = tok_i - tok_j
        j = np.int32(0)
        min_abs_u = np.float32(1e38)
        r_min = np.int32(0)
        u_min_val = np.float32(0.0)

        if normalize_qk:
            qk_mean_val = qk_mean_arr[pair]
            qk_inv_std_val = qk_inv_std_arr[pair]

        for c in range(pd):
            if qkpe_direct_sign:
                if normalize_qk:
                    qk_val = qk_norm_target_std * (
                        q_embed[tok_i, c] + k_embed[tok_j, c] - qk_mean_val
                    ) * qk_inv_std_val
                    u_val = qk_val + PE[pe_pos, c]
                else:
                    u_val = q_embed[tok_i, c] + k_embed[tok_j, c] + PE[pe_pos, c]
            else:
                a = qkpe_a_arr[t, c]
                b = qkpe_b_arr[t, c]
                if normalize_qk:
                    ua = (
                        qk_norm_target_std * (
                            q_embed[tok_i, a] + k_embed[tok_j, a] - qk_mean_val
                        ) * qk_inv_std_val
                        + PE[pe_pos, a]
                    )
                    ub = (
                        qk_norm_target_std * (
                            q_embed[tok_i, b] + k_embed[tok_j, b] - qk_mean_val
                        ) * qk_inv_std_val
                        + PE[pe_pos, b]
                    )
                else:
                    ua = q_embed[tok_i, a] + k_embed[tok_j, a] + PE[pe_pos, a]
                    ub = q_embed[tok_i, b] + k_embed[tok_j, b] + PE[pe_pos, b]
                u_val = ua - ub
            if u_val > np.float32(0.0):
                j |= qkpe_bitmasks[c]
            au = abs(u_val)
            if au < min_abs_u:
                min_abs_u = au
                r_min = np.int32(c)
                u_min_val = u_val

        qkpe_j[pair, t] = j
        qkpe_r_min[pair, t] = r_min
        qkpe_u_min[pair, t] = u_min_val

    # Phase 5: accumulate final LUT output into each y cell.
    for work in prange((CS - 1) * y_dim):
        tok_i = work // y_dim + 1
        d = work - (tok_i - 1) * y_dim
        acc = np.float32(0.0)

        for tok_j in range(tok_i):
            pair = tok_i * (tok_i - 1) // 2 + tok_j
            for t in range(N_T):
                j = (final_v_j[tok_j, t] << pd) | qkpe_j[pair, t]
                acc += FS[final_trees[t], j, d]

        y[tok_i, d] += acc


@njit(parallel=True, cache=True)
def _new_attention_pe_only_forward_kernel(
    VS, FS,
    val_j,
    q_embed, k_embed, v_embed,
    qkpe_j, qkpe_r_min, qkpe_u_min,
    final_v_j, final_v_r_min, final_v_u_min,
    y,
    PE,
    v_trees, final_trees,
    final_v_a_arr, final_v_b_arr, final_v_bitmasks,
    qkpe_a_arr, qkpe_b_arr, qkpe_bitmasks,
    pd, CS, N_T, N_C, QD, y_dim, qkpe_direct_sign
):
    """Fused PE-only forward: PE(i-j) and V(x_j) into Final LUT."""
    n_pairs = CS * (CS - 1) // 2

    # Keep Q/K scratch explicitly zero so callers can assert the ablation.
    for work in prange(CS * QD):
        pos = work // QD
        d = work - pos * QD
        q_embed[pos, d] = np.float32(0.0)
        k_embed[pos, d] = np.float32(0.0)

    # Token-side V embeddings.
    for work in prange(CS * y_dim):
        pos = work // y_dim
        d = work - pos * y_dim
        v_acc = np.float32(0.0)
        for t in range(N_T):
            v_acc += VS[v_trees[t], val_j[pos, t], d]
        v_embed[pos, d] = v_acc

    # Final value-side row bits from V(x_j).
    for work in prange(CS * N_T):
        pos = work // N_T
        t = work - pos * N_T
        j = np.int32(0)
        min_abs_u = np.float32(1e38)
        r_min = np.int32(0)
        u_min_val = np.float32(0.0)

        for c in range(N_C):
            u_val = v_embed[pos, final_v_a_arr[t, c]] - v_embed[pos, final_v_b_arr[t, c]]
            if u_val > np.float32(0.0):
                j |= final_v_bitmasks[c]
            au = abs(u_val)
            if au < min_abs_u:
                min_abs_u = au
                r_min = np.int32(c)
                u_min_val = u_val

        final_v_j[pos, t] = j
        final_v_r_min[pos, t] = r_min
        final_v_u_min[pos, t] = u_min_val

    # Final qkpe-side row bits from PE(i-j) only.
    for work in prange(n_pairs * N_T):
        pair = work // N_T
        t = work - pair * N_T
        tok_i = int((np.sqrt(8.0 * pair + 1.0) + 1.0) * 0.5)
        tok_j = pair - (tok_i * (tok_i - 1)) // 2
        pe_pos = tok_i - tok_j
        j = np.int32(0)
        min_abs_u = np.float32(1e38)
        r_min = np.int32(0)
        u_min_val = np.float32(0.0)

        for c in range(pd):
            if qkpe_direct_sign:
                u_val = PE[pe_pos, c]
            else:
                a = qkpe_a_arr[t, c]
                b = qkpe_b_arr[t, c]
                u_val = PE[pe_pos, a] - PE[pe_pos, b]
            if u_val > np.float32(0.0):
                j |= qkpe_bitmasks[c]
            au = abs(u_val)
            if au < min_abs_u:
                min_abs_u = au
                r_min = np.int32(c)
                u_min_val = u_val

        qkpe_j[pair, t] = j
        qkpe_r_min[pair, t] = r_min
        qkpe_u_min[pair, t] = u_min_val

    # Accumulate final LUT output into each y cell.
    for work in prange((CS - 1) * y_dim):
        tok_i = work // y_dim + 1
        d = work - (tok_i - 1) * y_dim
        acc = np.float32(0.0)

        for tok_j in range(tok_i):
            pair = tok_i * (tok_i - 1) // 2 + tok_j
            for t in range(N_T):
                j = (final_v_j[tok_j, t] << pd) | qkpe_j[pair, t]
                acc += FS[final_trees[t], j, d]

        y[tok_i, d] += acc

@njit(parallel=True)
def _new_attention_final_backward_kernel(
    FS,
    final_v_j, final_v_r_min, final_v_u_min,
    qkpe_j, qkpe_r_min, qkpe_u_min,
    y_grad,
    q_embed, k_embed, PE,
    qk_mean_arr, qk_inv_std_arr,
    q_grad, k_grad, v_grad, pe_grad,
    final_trees,
    final_v_a_arr, final_v_b_arr,
    qkpe_a_arr, qkpe_b_arr,
    pd, CS, N_T, y_dim, q_dim, lr, skip_s_update, pe_only, qkpe_direct_sign,
    freeze_pe, normalize_qk, qk_norm_target_std, qk_norm_eps
):
    """Backward through the new attention Final LUT into Q/K/V/PE embeddings."""
    n_pairs = CS * (CS - 1) // 2
    n_threads = get_num_threads()
    q_grad_thr = np.zeros((n_threads, CS, q_dim), dtype=np.float32)
    k_grad_thr = np.zeros((n_threads, CS, q_dim), dtype=np.float32)
    v_grad_thr = np.zeros((n_threads, CS, y_dim), dtype=np.float32)
    pe_grad_thr = np.zeros((n_threads, CS, q_dim), dtype=np.float32)

    # Phase 1: compute embedding-space gradients from fragile final bits.
    for work in prange(n_pairs * N_T):
        tid = get_thread_id()
        pair = work // N_T
        t = work - pair * N_T
        tok_i = int((np.sqrt(8.0 * pair + 1.0) + 1.0) * 0.5)
        tok_j = pair - (tok_i * (tok_i - 1)) // 2
        pe_pos = tok_i - tok_j
        tree = final_trees[t]
        j_bin = (final_v_j[tok_j, t] << pd) | qkpe_j[pair, t]

        r_v = final_v_r_min[tok_j, t]
        jbar_v = j_bin ^ (1 << (r_v + pd))
        gi_v = np.float32(0.0)
        for d in range(y_dim):
            gi_v += (FS[tree, jbar_v, d] - FS[tree, j_bin, d]) * y_grad[tok_i, d]
        u_v = final_v_u_min[tok_j, t]
        sign_v = np.float32(1.0) if u_v > np.float32(0.0) else np.float32(-1.0)
        delta_v = gi_v * (np.float32(-0.5) * sign_v / (np.float32(1.0) + abs(u_v)) ** 2)
        a_v = final_v_a_arr[t, r_v]
        b_v = final_v_b_arr[t, r_v]
        v_grad_thr[tid, tok_j, a_v] += delta_v
        v_grad_thr[tid, tok_j, b_v] -= delta_v

        r_qkpe = qkpe_r_min[pair, t]
        jbar_qkpe = j_bin ^ (1 << r_qkpe)
        gi_qkpe = np.float32(0.0)
        for d in range(y_dim):
            gi_qkpe += (FS[tree, jbar_qkpe, d] - FS[tree, j_bin, d]) * y_grad[tok_i, d]
        u_qkpe = qkpe_u_min[pair, t]
        sign_qkpe = np.float32(1.0) if u_qkpe > np.float32(0.0) else np.float32(-1.0)
        delta_qkpe = gi_qkpe * (np.float32(-0.5) * sign_qkpe / (np.float32(1.0) + abs(u_qkpe)) ** 2)
        if normalize_qk:
            active_dim = pd if qkpe_direct_sign else q_dim
            if qkpe_direct_sign:
                a_qkpe = r_qkpe
                b_qkpe = np.int32(-1)
            else:
                a_qkpe = qkpe_a_arr[t, r_qkpe]
                b_qkpe = qkpe_b_arr[t, r_qkpe]

            if not pe_only:
                qk_mean_val = qk_mean_arr[pair]
                qk_inv_std_val = qk_inv_std_arr[pair]

                qk_xhat_a = (q_embed[tok_i, a_qkpe] + k_embed[tok_j, a_qkpe] - qk_mean_val) * qk_inv_std_val
                if qkpe_direct_sign:
                    qk_grad_mean = delta_qkpe / np.float32(active_dim)
                    qk_grad_xhat_mean = delta_qkpe * qk_xhat_a / np.float32(active_dim)
                else:
                    qk_xhat_b = (q_embed[tok_i, b_qkpe] + k_embed[tok_j, b_qkpe] - qk_mean_val) * qk_inv_std_val
                    qk_grad_mean = np.float32(0.0)
                    qk_grad_xhat_mean = delta_qkpe * (qk_xhat_a - qk_xhat_b) / np.float32(active_dim)

                for d_qk in range(active_dim):
                    upstream = np.float32(0.0)
                    if d_qk == a_qkpe:
                        upstream += delta_qkpe
                    if (not qkpe_direct_sign) and d_qk == b_qkpe:
                        upstream -= delta_qkpe
                    qk_xhat = (q_embed[tok_i, d_qk] + k_embed[tok_j, d_qk] - qk_mean_val) * qk_inv_std_val
                    qk_delta = (
                        qk_norm_target_std * qk_inv_std_val
                        * (upstream - qk_grad_mean - qk_xhat * qk_grad_xhat_mean)
                    )
                    q_grad_thr[tid, tok_i, d_qk] += qk_delta
                    k_grad_thr[tid, tok_j, d_qk] += qk_delta

            if not freeze_pe:
                pe_grad_thr[tid, pe_pos, a_qkpe] += delta_qkpe
                if not qkpe_direct_sign:
                    pe_grad_thr[tid, pe_pos, b_qkpe] -= delta_qkpe
        elif qkpe_direct_sign:
            if not pe_only:
                q_grad_thr[tid, tok_i, r_qkpe] += delta_qkpe
                k_grad_thr[tid, tok_j, r_qkpe] += delta_qkpe
            if not freeze_pe:
                pe_grad_thr[tid, pe_pos, r_qkpe] += delta_qkpe
        else:
            a_qkpe = qkpe_a_arr[t, r_qkpe]
            b_qkpe = qkpe_b_arr[t, r_qkpe]
            if not pe_only:
                q_grad_thr[tid, tok_i, a_qkpe] += delta_qkpe
                q_grad_thr[tid, tok_i, b_qkpe] -= delta_qkpe
                k_grad_thr[tid, tok_j, a_qkpe] += delta_qkpe
                k_grad_thr[tid, tok_j, b_qkpe] -= delta_qkpe
            if not freeze_pe:
                pe_grad_thr[tid, pe_pos, a_qkpe] += delta_qkpe
                pe_grad_thr[tid, pe_pos, b_qkpe] -= delta_qkpe

    # Phase 2: reduce thread-local gradients.
    for work in prange(CS * q_dim):
        pos = work // q_dim
        d = work - pos * q_dim
        q_total = np.float32(0.0)
        k_total = np.float32(0.0)
        pe_total = np.float32(0.0)
        for thr in range(n_threads):
            q_total += q_grad_thr[thr, pos, d]
            k_total += k_grad_thr[thr, pos, d]
            pe_total += pe_grad_thr[thr, pos, d]
        q_grad[pos, d] += q_total
        k_grad[pos, d] += k_total
        pe_grad[pos, d] += pe_total

    for work in prange(CS * y_dim):
        pos = work // y_dim
        d = work - pos * y_dim
        total = np.float32(0.0)
        for thr in range(n_threads):
            total += v_grad_thr[thr, pos, d]
        v_grad[pos, d] += total

    # Phase 3: update Final.S after all gradients read the original rows.
    # Parallelize over physical tree and output dim so logical slots sharing a
    # tree cannot race on the same S row.
    if not skip_s_update:
        final_n_t = FS.shape[0]
        for work in prange(final_n_t * y_dim):
            tree = work // y_dim
            d = work - tree * y_dim
            for t in range(N_T):
                if final_trees[t] == tree:
                    for tok_i in range(1, CS):
                        for tok_j in range(tok_i):
                            pair = tok_i * (tok_i - 1) // 2 + tok_j
                            j_bin = (final_v_j[tok_j, t] << pd) | qkpe_j[pair, t]
                            FS[tree, j_bin, d] -= lr * y_grad[tok_i, d]

@njit(parallel=True, cache=True)
def _new_attention_final_s_replay_kernel(FS, final_trees, final_v_j, qkpe_j,
                                         y_grad, pd, CS, N_T, y_dim, scale):
    """Replay Final.S updates for new attention from cached row halves."""
    final_n_t = FS.shape[0]
    for work in prange(final_n_t * y_dim):
        tree = work // y_dim
        d = work - tree * y_dim
        for t in range(N_T):
            if final_trees[t] == tree:
                for tok_i in range(1, CS):
                    for tok_j in range(tok_i):
                        pair = tok_i * (tok_i - 1) // 2 + tok_j
                        j_bin = (final_v_j[tok_j, t] << pd) | qkpe_j[pair, t]
                        FS[tree, j_bin, d] -= scale * y_grad[tok_i, d]


@njit(parallel=True)
def _attention_backward_kernel(S, all_jV, all_rV, all_uV, all_jPE, all_rPE, all_uPE,
                                y_grad, x_grad, pos_grad, a_arr, b_arr,
                                shift_qk, pd, CS, N_T, y_dim, lr, freeze_pe):
    """Fused attention backward with parallel gradient computation.

    Three-phase approach:
      Phase 1 (parallel over causal-pair, tree): each thread accumulates all
        x_grad and pe_grad contributions into its own buffer.
      Phase 2: reduce per-thread buffers into x_grad and pos_grad.
      Phase 3: update S tables in parallel over tree rows.
    """
    ED = x_grad.shape[1]
    n_threads = get_num_threads()
    x_grad_thr = np.zeros((n_threads, CS, ED), dtype=np.float32)
    pe_grad_thr = np.zeros((n_threads, CS, N_T, pd), dtype=np.float32)
    n_pairs = CS * (CS - 1) // 2

    # Phase 1: parallel gradient computation over all causal pairs and trees.
    for work in prange(n_pairs * N_T):
        tid = get_thread_id()
        pair = work // N_T
        t = work - pair * N_T
        pos = int((np.sqrt(8.0 * pair + 1.0) + 1.0) * 0.5)
        pos1 = pair - (pos * (pos - 1)) // 2
        pe_pos = pos - pos1

        jQ = all_jV[pos, t]
        jK = all_jV[pos1, t]
        j_bin = (jQ << shift_qk) | (jK << pd) | all_jPE[pe_pos, t]

        abs_uQ = abs(all_uV[pos, t])
        abs_uK = abs(all_uV[pos1, t])
        q_wins = abs_uQ < abs_uK

        if q_wins:
            flip_bit = all_rV[pos, t] + shift_qk
        else:
            flip_bit = all_rV[pos1, t] + pd
        jbar = j_bin ^ (1 << flip_bit)

        gi = np.float32(0.0)
        for d in range(y_dim):
            gi += (S[t, jbar, d] - S[t, j_bin, d]) * y_grad[pos, d]

        if q_wins:
            u_min = all_uV[pos, t]
            r = all_rV[pos, t]
            target_pos = pos
        else:
            u_min = all_uV[pos1, t]
            r = all_rV[pos1, t]
            target_pos = pos1

        sign_u = np.float32(1.0) if u_min > np.float32(0.0) else np.float32(-1.0)
        abs_u_min = abs(u_min)
        v = gi * (np.float32(-0.5) * sign_u / (np.float32(1.0) + abs_u_min) ** 2)

        idx_a = a_arr[t, r]
        idx_b = b_arr[t, r]
        x_grad_thr[tid, target_pos, idx_a] += v
        x_grad_thr[tid, target_pos, idx_b] -= v

        abs_uPE = abs(all_uPE[pe_pos, t])
        if not freeze_pe and abs_uPE < abs_uQ and abs_uPE < abs_uK:
            rPE = all_rPE[pe_pos, t]
            jbarPE = j_bin ^ (1 << rPE)
            giPE = np.float32(0.0)
            for d in range(y_dim):
                giPE += (S[t, jbarPE, d] - S[t, j_bin, d]) * y_grad[pos, d]
            u_pe = all_uPE[pe_pos, t]
            sign_pe = np.float32(1.0) if u_pe > np.float32(0.0) else np.float32(-1.0)
            deltaPE = giPE * (np.float32(-0.5) * sign_pe / (np.float32(1.0) + abs(u_pe)) ** 2)
            pe_grad_thr[tid, pe_pos, t, rPE] += deltaPE

    # Phase 2: reduce per-thread buffers
    for target in prange(CS):
        for d in range(ED):
            total = np.float32(0.0)
            for thr in range(n_threads):
                total += x_grad_thr[thr, target, d]
            x_grad[target, d] += total

    for pe_pos in prange(CS):
        for t in range(N_T):
            for r in range(pd):
                total = np.float32(0.0)
                for thr in range(n_threads):
                    total += pe_grad_thr[thr, pe_pos, t, r]
                pos_grad[pe_pos, t, r] += total

    # Phase 3: S update. Parallel over tree rows; collisions are only within a
    # tree row and are kept sequential.
    for t in prange(N_T):
        for pos in range(1, CS):
            for pos1 in range(pos):
                j_bin = (all_jV[pos, t] << shift_qk) | (all_jV[pos1, t] << pd) | all_jPE[pos - pos1, t]
                for d in range(y_dim):
                    S[t, j_bin, d] -= lr * y_grad[pos, d]

@njit(parallel=True)
def _attention_backward_kernel_noupdate(S, all_jV, all_rV, all_uV, all_jPE, all_rPE, all_uPE,
                                         y_grad, x_grad, pos_grad, a_arr, b_arr,
                                         shift_qk, pd, CS, N_T, y_dim, freeze_pe):
    """Like _attention_backward_kernel but without Phase 3 (no S update)."""
    ED = x_grad.shape[1]
    n_threads = get_num_threads()
    x_grad_thr = np.zeros((n_threads, CS, ED), dtype=np.float32)
    pe_grad_thr = np.zeros((n_threads, CS, N_T, pd), dtype=np.float32)
    n_pairs = CS * (CS - 1) // 2

    # Phase 1: parallel gradient computation over all causal pairs and trees.
    for work in prange(n_pairs * N_T):
        tid = get_thread_id()
        pair = work // N_T
        t = work - pair * N_T
        pos = int((np.sqrt(8.0 * pair + 1.0) + 1.0) * 0.5)
        pos1 = pair - (pos * (pos - 1)) // 2
        pe_pos = pos - pos1

        jQ = all_jV[pos, t]
        jK = all_jV[pos1, t]
        j_bin = (jQ << shift_qk) | (jK << pd) | all_jPE[pe_pos, t]

        abs_uQ = abs(all_uV[pos, t])
        abs_uK = abs(all_uV[pos1, t])
        q_wins = abs_uQ < abs_uK

        if q_wins:
            flip_bit = all_rV[pos, t] + shift_qk
        else:
            flip_bit = all_rV[pos1, t] + pd
        jbar = j_bin ^ (1 << flip_bit)

        gi = np.float32(0.0)
        for d in range(y_dim):
            gi += (S[t, jbar, d] - S[t, j_bin, d]) * y_grad[pos, d]

        if q_wins:
            u_min = all_uV[pos, t]
            r = all_rV[pos, t]
            target_pos = pos
        else:
            u_min = all_uV[pos1, t]
            r = all_rV[pos1, t]
            target_pos = pos1

        sign_u = np.float32(1.0) if u_min > np.float32(0.0) else np.float32(-1.0)
        abs_u_min = abs(u_min)
        v = gi * (np.float32(-0.5) * sign_u / (np.float32(1.0) + abs_u_min) ** 2)

        idx_a = a_arr[t, r]
        idx_b = b_arr[t, r]
        x_grad_thr[tid, target_pos, idx_a] += v
        x_grad_thr[tid, target_pos, idx_b] -= v

        abs_uPE = abs(all_uPE[pe_pos, t])
        if not freeze_pe and abs_uPE < abs_uQ and abs_uPE < abs_uK:
            rPE = all_rPE[pe_pos, t]
            jbarPE = j_bin ^ (1 << rPE)
            giPE = np.float32(0.0)
            for d in range(y_dim):
                giPE += (S[t, jbarPE, d] - S[t, j_bin, d]) * y_grad[pos, d]
            u_pe = all_uPE[pe_pos, t]
            sign_pe = np.float32(1.0) if u_pe > np.float32(0.0) else np.float32(-1.0)
            deltaPE = giPE * (np.float32(-0.5) * sign_pe / (np.float32(1.0) + abs(u_pe)) ** 2)
            pe_grad_thr[tid, pe_pos, t, rPE] += deltaPE

    # Phase 2: reduce per-thread buffers
    for target in prange(CS):
        for d in range(ED):
            total = np.float32(0.0)
            for thr in range(n_threads):
                total += x_grad_thr[thr, target, d]
            x_grad[target, d] += total

    for pe_pos in prange(CS):
        for t in range(N_T):
            for r in range(pd):
                total = np.float32(0.0)
                for thr in range(n_threads):
                    total += pe_grad_thr[thr, pe_pos, t, r]
                pos_grad[pe_pos, t, r] += total

@njit(parallel=True, cache=True)
def _cache_index_kernel(x, a_arr, b_arr, bitmasks, j_out, r_min_out, u_min_out, CS, N_T, N_C):
    """Parallel cache_index: computes j, r_min, u_min for all (pos, tree) pairs."""
    for work in prange(CS * N_T):
        pos = work // N_T
        t = work - pos * N_T
        j = np.int32(0)
        min_abs_u = np.float32(1e38)
        r_min = np.int32(0)
        u_min_val = np.float32(0.0)
        for c in range(N_C):
            u_val = x[pos, a_arr[t, c]] - x[pos, b_arr[t, c]]
            if u_val > np.float32(0.0):
                j |= bitmasks[c]
            au = abs(u_val)
            if au < min_abs_u:
                min_abs_u = au
                r_min = np.int32(c)
                u_min_val = u_val
        j_out[pos, t] = j
        r_min_out[pos, t] = r_min
        u_min_out[pos, t] = u_min_val

@njit(parallel=True, cache=True)
def _cache_pe_index_kernel(u_batch, pe_bitmasks, j_out, r_min_out, u_min_out, CS, N_T, PD):
    """Parallel cache_PE_index: computes j, r_min, u_min from PE tensor."""
    for work in prange(CS * N_T):
        pos = work // N_T
        t = work - pos * N_T
        j = np.int32(0)
        min_abs_u = np.float32(1e38)
        r_min = np.int32(0)
        u_min_val = np.float32(0.0)
        for p in range(PD):
            u_val = u_batch[pos, t, p]
            if u_val > np.float32(0.0):
                j |= pe_bitmasks[p]
            au = abs(u_val)
            if au < min_abs_u:
                min_abs_u = au
                r_min = np.int32(p)
                u_min_val = u_val
        j_out[pos, t] = j
        r_min_out[pos, t] = r_min
        u_min_out[pos, t] = u_min_val

@njit(parallel=True, cache=True)
def _lut_forward_add_kernel(S, trees, j, output, CS, N_T, y_dim):
    """Fused LUT forward that adds to output (no zeroing). Used for FFN residual."""
    for work in prange(CS * y_dim):
        pos = work // y_dim
        d = work - pos * y_dim
        acc = np.float32(0.0)
        for t in range(N_T):
            acc += S[trees[t], j[pos, t], d]
        output[pos, d] += acc

@njit(parallel=True, cache=True)
def _lut_forward_kernel(S, trees, j, output, CS, N_T, y_dim):
    """Fused LUT forward: parallel over independent output cells."""
    for work in prange(CS * y_dim):
        pos = work // y_dim
        d = work - pos * y_dim
        acc = np.float32(0.0)
        for t in range(N_T):
            acc += S[trees[t], j[pos, t], d]
        output[pos, d] = acc

@njit(parallel=True)
def _lut_backward_kernel(S, trees, j, r_min, u_min, grad, x_grad, a_arr, b_arr,
                          y_dim, CS, N_T, lr):
    """Fused backward for simple LUT (FFN, output head). grad: (CS, >=y_dim).

    Phase 1 (parallel): compute x_grad using original S into thread-local rows.
    Phase 2: update S in parallel over tree rows; repeated bins within a row
    are still applied sequentially.
    """
    ED = x_grad.shape[1]
    n_threads = get_num_threads()
    x_grad_thr = np.zeros((n_threads, CS, ED), dtype=np.float32)

    # Phase 1: parallel x_grad computation
    for work in prange(CS * N_T):
        tid = get_thread_id()
        pos = work // N_T
        t = work - pos * N_T
        tree = trees[t]
        j_bin = j[pos, t]
        r = r_min[pos, t]
        jbar_bin = j_bin ^ (1 << r)
        gi = np.float32(0.0)
        for d in range(y_dim):
            gi += (S[tree, jbar_bin, d] - S[tree, j_bin, d]) * grad[pos, d]
        u = u_min[pos, t]
        sign_u = np.float32(1.0) if u > np.float32(0.0) else np.float32(-1.0)
        v = gi * (np.float32(-0.5) * sign_u / (np.float32(1.0) + abs(u)) ** 2)
        x_grad_thr[tid, pos, a_arr[t, r]] += v
        x_grad_thr[tid, pos, b_arr[t, r]] -= v

    for pos in prange(CS):
        for d in range(ED):
            total = np.float32(0.0)
            for thr in range(n_threads):
                total += x_grad_thr[thr, pos, d]
            x_grad[pos, d] += total

    # Phase 2: S update
    for t in prange(N_T):
        tree = trees[t]
        for pos in range(CS):
            j_bin = j[pos, t]
            for d in range(y_dim):
                S[tree, j_bin, d] -= lr * grad[pos, d]

@njit(parallel=True)
def _lut_backward_kernel_noupdate(S, trees, j, r_min, u_min, grad, x_grad, a_arr, b_arr,
                                   y_dim, CS, N_T):
    """Like _lut_backward_kernel but without the S update."""
    ED = x_grad.shape[1]
    n_threads = get_num_threads()
    x_grad_thr = np.zeros((n_threads, CS, ED), dtype=np.float32)

    for work in prange(CS * N_T):
        tid = get_thread_id()
        pos = work // N_T
        t = work - pos * N_T
        tree = trees[t]
        j_bin = j[pos, t]
        r = r_min[pos, t]
        jbar_bin = j_bin ^ (1 << r)
        gi = np.float32(0.0)
        for d in range(y_dim):
            gi += (S[tree, jbar_bin, d] - S[tree, j_bin, d]) * grad[pos, d]
        u = u_min[pos, t]
        sign_u = np.float32(1.0) if u > np.float32(0.0) else np.float32(-1.0)
        v = gi * (np.float32(-0.5) * sign_u / (np.float32(1.0) + abs(u)) ** 2)
        x_grad_thr[tid, pos, a_arr[t, r]] += v
        x_grad_thr[tid, pos, b_arr[t, r]] -= v

    for pos in prange(CS):
        for d in range(ED):
            total = np.float32(0.0)
            for thr in range(n_threads):
                total += x_grad_thr[thr, pos, d]
            x_grad[pos, d] += total

@njit(parallel=True, cache=True)
def _lut_s_replay_kernel(S, trees, j, y_grad, y_dim, CS, N_T, scale):
    """Replay LUT S updates from saved y_grad and cached j indices."""
    for t in prange(N_T):
        tree = trees[t]
        for pos in range(CS):
            j_bin = j[pos, t]
            for d in range(y_dim):
                S[tree, j_bin, d] -= scale * y_grad[pos, d]

@njit(parallel=True, cache=True)
def _attention_s_replay_kernel(S, all_jV, all_jPE, y_grad, shift_qk, pd, CS, N_T, y_dim, scale):
    """Replay attention S updates from saved y_grad and cached j indices."""
    for t in prange(N_T):
        for pos in range(1, CS):
            for pos1 in range(pos):
                j_bin = (all_jV[pos, t] << shift_qk) | (all_jV[pos1, t] << pd) | all_jPE[pos - pos1, t]
                for d in range(y_dim):
                    S[t, j_bin, d] -= scale * y_grad[pos, d]

@njit(parallel=True, cache=True)
def _softmax_cross_entropy_kernel(output, target_indices, CS, VOCAB):
    """Parallel softmax + subtract one-hot over all context positions."""
    for pos in prange(CS):
        max_val = output[pos, 0]
        for v in range(1, VOCAB):
            if output[pos, v] > max_val:
                max_val = output[pos, v]
        s = np.float32(0.0)
        for v in range(VOCAB):
            output[pos, v] = np.exp(output[pos, v] - max_val)
            s += output[pos, v]
        for v in range(VOCAB):
            output[pos, v] /= s
        output[pos, target_indices[pos]] -= np.float32(1.0)


# ---------------------------------------------------------------------------
# Soft attention Numba kernels
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _soft_attn_forward_kernel(ScoreS, ValueS, jV, val_j, jPE, y, attn_w, v_vecs,
                               shift_qk, pd, CS, N_T, ED):
    """Fused soft attention forward: raw scores, softmax, value vectors, weighted sum."""
    # Phase 1: raw scores + softmax (parallel over pos >= 1)
    for pos in prange(1, CS):
        for pos1 in range(pos):
            pe_pos = pos - pos1
            s = np.float32(0.0)
            for t in range(N_T):
                j_bin = (jV[pos, t] << shift_qk) | (jV[pos1, t] << pd) | jPE[pe_pos, t]
                s += ScoreS[t, j_bin, 0]
            attn_w[pos, pos1] = s
        max_s = attn_w[pos, 0]
        for pos1 in range(1, pos):
            if attn_w[pos, pos1] > max_s:
                max_s = attn_w[pos, pos1]
        s_sum = np.float32(0.0)
        for pos1 in range(pos):
            val = np.exp(attn_w[pos, pos1] - max_s)
            attn_w[pos, pos1] = val
            s_sum += val
        for pos1 in range(pos):
            attn_w[pos, pos1] /= s_sum

    # Phase 2: value vectors v_vecs[pos] = sum_t ValueS[t, val_j[pos,t], :] (parallel over pos)
    for pos in prange(CS):
        for d in range(ED):
            v_vecs[pos, d] = np.float32(0.0)
        for t in range(N_T):
            j_val = val_j[pos, t]
            for d in range(ED):
                v_vecs[pos, d] += ValueS[t, j_val, d]

    # Phase 3: weighted sum into y (parallel over pos >= 1)
    for pos in prange(1, CS):
        for pos1 in range(pos):
            w = attn_w[pos, pos1]
            for d in range(ED):
                y[pos, d] += w * v_vecs[pos1, d]


@njit(parallel=True)
def _soft_attn_backward_value_kernel(ValueS, val_j, val_r_min, val_u_min,
                                      attn_w, y_grad, val_grad, x_grad,
                                      val_a_arr, val_b_arr, CS, N_T, ED, lr, skip_s_update):
    """Soft attention value backward: val_grad, STE x_grad update, optional Value.S update."""
    n_threads = get_num_threads()
    x_grad_thr = np.zeros((n_threads, CS, ED), dtype=np.float32)

    # Phase 1: val_grad[pos1] = sum_{pos>pos1} attn_w[pos,pos1] * y_grad[pos] (parallel over pos1)
    for pos1 in prange(CS):
        for d in range(ED):
            val_grad[pos1, d] = np.float32(0.0)
        for pos in range(pos1 + 1, CS):
            w = attn_w[pos, pos1]
            for d in range(ED):
                val_grad[pos1, d] += w * y_grad[pos, d]

    # Phase 2: Value STE — compute x_grad contribution (parallel over pos1)
    for pos1 in prange(CS):
        tid = get_thread_id()
        for t in range(N_T):
            j_val = val_j[pos1, t]
            r = val_r_min[pos1, t]
            jbar = j_val ^ (1 << r)
            gi = np.float32(0.0)
            for d in range(ED):
                gi += (ValueS[t, jbar, d] - ValueS[t, j_val, d]) * val_grad[pos1, d]
            u = val_u_min[pos1, t]
            sign_u = np.float32(1.0) if u > np.float32(0.0) else np.float32(-1.0)
            v = gi * (np.float32(-0.5) * sign_u / (np.float32(1.0) + abs(u)) ** 2)
            x_grad_thr[tid, pos1, val_a_arr[t, r]] += v
            x_grad_thr[tid, pos1, val_b_arr[t, r]] -= v

    # Phase 3: reduce x_grad_thr into x_grad (parallel over pos)
    for pos in prange(CS):
        for d in range(ED):
            total = np.float32(0.0)
            for thr in range(n_threads):
                total += x_grad_thr[thr, pos, d]
            x_grad[pos, d] += total

    # Phase 4: Value.S update (sequential — collisions possible across pos1)
    if not skip_s_update:
        for pos1 in range(CS):
            for t in range(N_T):
                j_val = val_j[pos1, t]
                for d in range(ED):
                    ValueS[t, j_val, d] -= lr * val_grad[pos1, d]


@njit(parallel=True)
def _soft_attn_backward_score_kernel(ScoreS, jV, rV, uV, jPE, rPE, uPE,
                                      attn_w, v_vecs, y_grad, score_grad, x_grad, pos_grad,
                                      score_a_arr, score_b_arr,
                                      shift_qk, pd, CS, N_T, ED, lr, skip_s_update, freeze_pe):
    """Soft attention score backward: score_grad, STE x_grad + pe_grad, optional Score.S update."""
    n_threads = get_num_threads()
    x_grad_thr = np.zeros((n_threads, CS, ED), dtype=np.float32)
    pe_grad_thr = np.zeros((n_threads, CS, N_T, pd), dtype=np.float32)

    # Phase 1: compute score_grad (parallel over pos >= 1)
    # score_grad[pos, pos1] = attn_w[pos,pos1] * (a[pos,pos1] - baseline[pos])
    # Uses score_grad as temp for affinities, then overwrites with score_grad.
    for pos in prange(1, CS):
        baseline = np.float32(0.0)
        for pos1 in range(pos):
            a = np.float32(0.0)
            for d in range(ED):
                a += y_grad[pos, d] * v_vecs[pos1, d]
            score_grad[pos, pos1] = a
            baseline += attn_w[pos, pos1] * a
        for pos1 in range(pos):
            score_grad[pos, pos1] = attn_w[pos, pos1] * (score_grad[pos, pos1] - baseline)

    # Phase 2: Score STE — x_grad and pe_grad contributions (parallel over pos >= 1)
    for pos in prange(1, CS):
        tid = get_thread_id()
        for pos1 in range(pos):
            sg = score_grad[pos, pos1]
            pe_pos = pos - pos1
            for t in range(N_T):
                jQ_t  = jV[pos, t];  jK_t = jV[pos1, t];  jPE_t = jPE[pe_pos, t]
                j_bin = (jQ_t << shift_qk) | (jK_t << pd) | jPE_t
                abs_uQ  = abs(uV[pos, t])
                abs_uK  = abs(uV[pos1, t])
                abs_uPE = abs(uPE[pe_pos, t])
                q_wins = abs_uQ < abs_uK
                if q_wins:
                    flip_bit = rV[pos, t] + shift_qk
                    u_min = uV[pos, t];  r = rV[pos, t]
                else:
                    flip_bit = rV[pos1, t] + pd
                    u_min = uV[pos1, t];  r = rV[pos1, t]
                jbar = j_bin ^ (1 << flip_bit)
                gi = (ScoreS[t, jbar, 0] - ScoreS[t, j_bin, 0]) * sg
                sign_u = np.float32(1.0) if u_min > np.float32(0.0) else np.float32(-1.0)
                v = gi * (np.float32(-0.5) * sign_u / (np.float32(1.0) + abs(u_min)) ** 2)
                idx_a = score_a_arr[t, r];  idx_b = score_b_arr[t, r]
                if q_wins:
                    x_grad[pos, idx_a] += v   # unique per prange pos — no race
                    x_grad[pos, idx_b] -= v
                else:
                    x_grad_thr[tid, pos1, idx_a] += v
                    x_grad_thr[tid, pos1, idx_b] -= v
                # PE branch
                if not freeze_pe and abs_uPE < abs_uQ and abs_uPE < abs_uK:
                    rPE_t = rPE[pe_pos, t]
                    jbarPE = j_bin ^ (1 << rPE_t)
                    giPE = (ScoreS[t, jbarPE, 0] - ScoreS[t, j_bin, 0]) * sg
                    u_pe = uPE[pe_pos, t]
                    sign_pe = np.float32(1.0) if u_pe > np.float32(0.0) else np.float32(-1.0)
                    deltaPE = giPE * (np.float32(-0.5) * sign_pe / (np.float32(1.0) + abs(u_pe)) ** 2)
                    pe_grad_thr[tid, pe_pos, t, rPE_t] += deltaPE

    # Phase 3: reduce thread-local buffers (parallel)
    for target in prange(CS):
        for d in range(ED):
            total = np.float32(0.0)
            for thr in range(n_threads):
                total += x_grad_thr[thr, target, d]
            x_grad[target, d] += total

    for pe_pos in prange(CS):
        for t in range(N_T):
            for r in range(pd):
                total = np.float32(0.0)
                for thr in range(n_threads):
                    total += pe_grad_thr[thr, pe_pos, t, r]
                pos_grad[pe_pos, t, r] += total

    # Phase 4: Score.S update (sequential — collisions possible across (pos,pos1) pairs)
    if not skip_s_update:
        for pos in range(1, CS):
            for pos1 in range(pos):
                sg = score_grad[pos, pos1]
                pe_pos = pos - pos1
                for t in range(N_T):
                    j_bin = (jV[pos, t] << shift_qk) | (jV[pos1, t] << pd) | jPE[pe_pos, t]
                    ScoreS[t, j_bin, 0] -= lr * sg


@njit(cache=True)
def _soft_attn_score_s_replay_kernel(ScoreS, jV, jPE, score_grad, shift_qk, pd, CS, N_T, scale):
    """Replay Score.S updates from saved score_grad and cached j indices."""
    for pos in range(1, CS):
        for pos1 in range(pos):
            sg = score_grad[pos, pos1]
            pe_pos = pos - pos1
            for t in range(N_T):
                j_bin = (jV[pos, t] << shift_qk) | (jV[pos1, t] << pd) | jPE[pe_pos, t]
                ScoreS[t, j_bin, 0] -= scale * sg


@njit(cache=True)
def _soft_attn_value_s_replay_kernel(ValueS, val_j, val_grad, CS, N_T, ED, scale):
    """Replay Value.S updates from saved val_grad and cached j indices."""
    for pos1 in range(CS):
        for t in range(N_T):
            j_val = val_j[pos1, t]
            for d in range(ED):
                ValueS[t, j_val, d] -= scale * val_grad[pos1, d]


# ---------------------------------------------------------------------------
# Model operations
# ---------------------------------------------------------------------------

def initialize_positional_encoding(shape, args):
    """Initialize frozen PE with binary row signs and random magnitudes."""
    scale = np.float32(getattr(args, 'PE_init_scale', 1.0))
    pe = (scale * np.random.rand(*shape)).astype(np.float32)

    signed_dims = min(int(getattr(args, 'positional_dim', shape[-1])), shape[-1])
    if signed_dims > 0:
        row_bins = (np.arange(shape[0], dtype=np.int64) % (1 << signed_dims))
        bit_positions = (signed_dims - 1 - np.arange(signed_dims, dtype=np.int64))
        signs = np.where(
            ((row_bins[:, np.newaxis] >> bit_positions[np.newaxis, :]) & 1) != 0,
            np.float32(1.0),
            np.float32(-1.0),
        ).astype(np.float32)
        sign_shape = (shape[0],) + (1,) * (len(shape) - 2) + (signed_dims,)
        pe[..., :signed_dims] *= signs.reshape(sign_shape)
    return pe


def build_Model(m, args):
    VOCAB_SIZE_INPUT = args.vocab_size_input
    EMBEDDING_DIM = args.embedding_dim
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads
    N_C = args.n_c
    POSITIONAL_DIM = args.positional_dim
    freeze_pe = getattr(args, 'freeze_PE', False)
    new_attention = getattr(args, 'new_attention', False)
    
    N_T = args.n_t

    m.Token_embedder[:] = random_vector(VOCAB_SIZE_INPUT * EMBEDDING_DIM, 1.0).reshape(VOCAB_SIZE_INPUT, EMBEDDING_DIM)

    frozen_new_pe = None
    frozen_table_pe = None
    if freeze_pe:
        if new_attention:
            frozen_new_pe = initialize_positional_encoding(
                (args.context_size, args.attn_qk_dim), args)
        else:
            frozen_table_pe = initialize_positional_encoding(
                (args.context_size, POSITIONAL_DIM), args)

    for l in range(NUM_LAYERS):
        build_LUT(m.FFN[l], N_C, EMBEDDING_DIM, args)
        for h in range(NUM_HEADS):
            if new_attention:
                head = m.head[l][h]
                if freeze_pe:
                    head.Positional_encoding[:] = frozen_new_pe
                else:
                    head.Positional_encoding[:] = random_vector(
                        args.context_size * args.attn_qk_dim, 1.0
                    ).reshape(args.context_size, args.attn_qk_dim)
                build_LUT_custom(head.Q, args.attn_qk_n_c, args.attn_qk_dim, args,
                                 n_t=args.attn_qk_n_t, n_c=args.attn_qk_n_c,
                                 input_dim=EMBEDDING_DIM)
                build_LUT_custom(head.K, args.attn_qk_n_c, args.attn_qk_dim, args,
                                 n_t=args.attn_qk_n_t, n_c=args.attn_qk_n_c,
                                 input_dim=EMBEDDING_DIM)
                build_LUT_custom(head.V, N_C, EMBEDDING_DIM, args,
                                 n_t=N_T, n_c=N_C, input_dim=EMBEDDING_DIM)
                final_n_t = getattr(args, 'final_n_t', N_T)
                final_share_m = getattr(args, 'final_share_m', 1)
                final_trees = np.repeat(np.arange(final_n_t, dtype=np.int32), final_share_m)
                build_LUT_custom(head.Final, N_C + POSITIONAL_DIM, EMBEDDING_DIM, args,
                                 n_t=N_T, n_c=N_C, input_dim=EMBEDDING_DIM,
                                 s_n_t=final_n_t, trees=final_trees)
                qkpe_a, qkpe_b = build_comparison_arrays(
                    getattr(args, 'lut_comparison_mode', 'random'), N_T,
                    POSITIONAL_DIM, args.attn_qk_dim)
                if qkpe_a.shape[0] != N_T:
                    qkpe_a, qkpe_b = build_no_overlap_comparison_arrays(
                        POSITIONAL_DIM, args.attn_qk_dim, n_t=N_T)
                head._qkpe_a_arr[:] = qkpe_a
                head._qkpe_b_arr[:] = qkpe_b
            else:
                if freeze_pe:
                    m.head[l][h].Positional_encoding[:] = frozen_table_pe[:, np.newaxis, :]
                else:
                    m.head[l][h].Positional_encoding[:] = random_vector(
                        args.context_size * N_T * POSITIONAL_DIM, 1.0
                    ).reshape(args.context_size, N_T, POSITIONAL_DIM)
                if getattr(args, 'soft_attention', False):
                    build_LUT(m.head[l][h].Score, N_C + N_C + POSITIONAL_DIM, 1, args)
                    build_LUT(m.head[l][h].Value, N_C, EMBEDDING_DIM, args)
                else:
                    build_LUT(m.head[l][h].V, N_C + N_C + POSITIONAL_DIM, EMBEDDING_DIM, args)

    m.output_head.build(N_C, args)


def _lut_save(d, prefix, lut):
    d[f'{prefix}_S']   = lut.S
    d[f'{prefix}_a']   = lut.a_arr
    d[f'{prefix}_b']   = lut.b_arr

def _load_array_checked(d, key, target):
    saved = d[key]
    if saved.shape != target.shape:
        raise ValueError(
            f"Checkpoint array {key} has shape {saved.shape}; expected {target.shape}. "
            "Check that --n-t, --n-c, dimensions, and --weight-sharing match the checkpoint.")
    target[:] = saved


def _lut_load(d, prefix, lut):
    _load_array_checked(d, f'{prefix}_S', lut.S)
    _load_array_checked(d, f'{prefix}_a', lut.a_arr)
    _load_array_checked(d, f'{prefix}_b', lut.b_arr)

def save_model(m, path, args):
    """Save all learned weights to a .npz file."""
    d = {'token_embedder': m.Token_embedder}
    for l in range(args.num_layers):
        _lut_save(d, f'ffn_{l}', m.FFN[l])
        for h in range(args.num_heads):
            d[f'pe_{l}_{h}'] = m.head[l][h].Positional_encoding
            if getattr(args, 'new_attention', False):
                _lut_save(d, f'head_{l}_{h}_Q', m.head[l][h].Q)
                _lut_save(d, f'head_{l}_{h}_K', m.head[l][h].K)
                _lut_save(d, f'head_{l}_{h}_V', m.head[l][h].V)
                _lut_save(d, f'head_{l}_{h}_Final', m.head[l][h].Final)
                d[f'head_{l}_{h}_qkpe_a'] = m.head[l][h]._qkpe_a_arr
                d[f'head_{l}_{h}_qkpe_b'] = m.head[l][h]._qkpe_b_arr
            elif getattr(args, 'soft_attention', False):
                _lut_save(d, f'head_{l}_{h}_Score', m.head[l][h].Score)
                _lut_save(d, f'head_{l}_{h}_Value', m.head[l][h].Value)
            else:
                _lut_save(d, f'head_{l}_{h}_V', m.head[l][h].V)
    if args.factored_output:
        _lut_save(d, 'output_hi', m.output_head.unembedder_hi)
        _lut_save(d, 'output_lo', m.output_head.unembedder_lo)
    else:
        _lut_save(d, 'output', m.output_head.unembedder)
    if getattr(args, 'layernorm', False):
        d['ln_attn_gamma'] = m.ln_attn_gamma
        d['ln_attn_beta']  = m.ln_attn_beta
        d['ln_ffn_gamma']  = m.ln_ffn_gamma
        d['ln_ffn_beta']   = m.ln_ffn_beta
    np.savez_compressed(path, **d)


def model_save_display_path(path):
    """Return the actual .npz filename np.savez_compressed will create."""
    path = str(path)
    return path if path.endswith(".npz") else path + ".npz"


def checkpoint_save_path(path, step):
    """Return a step-tagged checkpoint path derived from --save-model."""
    path = str(path)
    if path.endswith(".npz"):
        return path[:-4] + f"_step_{step}.npz"
    return path + f"_step_{step}"


def save_training_checkpoint(m, args, step):
    """Save a training checkpoint and return the displayed filename."""
    checkpoint_path = (
        checkpoint_save_path(args.save_model, step)
        if args.save_multiple_checkpoints else args.save_model
    )
    save_model(m, checkpoint_path, args)
    return model_save_display_path(checkpoint_path)


def load_model(m, path, args):
    """Load weights saved by save_model into an already-built model."""
    d = np.load(path)
    m.Token_embedder[:] = d['token_embedder']
    for l in range(args.num_layers):
        _lut_load(d, f'ffn_{l}', m.FFN[l])
        for h in range(args.num_heads):
            m.head[l][h].Positional_encoding[:] = d[f'pe_{l}_{h}']
            if getattr(args, 'new_attention', False):
                _lut_load(d, f'head_{l}_{h}_Q', m.head[l][h].Q)
                _lut_load(d, f'head_{l}_{h}_K', m.head[l][h].K)
                _lut_load(d, f'head_{l}_{h}_V', m.head[l][h].V)
                _lut_load(d, f'head_{l}_{h}_Final', m.head[l][h].Final)
                m.head[l][h]._qkpe_a_arr[:] = d[f'head_{l}_{h}_qkpe_a']
                m.head[l][h]._qkpe_b_arr[:] = d[f'head_{l}_{h}_qkpe_b']
            elif getattr(args, 'soft_attention', False):
                _lut_load(d, f'head_{l}_{h}_Score', m.head[l][h].Score)
                _lut_load(d, f'head_{l}_{h}_Value', m.head[l][h].Value)
            else:
                _lut_load(d, f'head_{l}_{h}_V', m.head[l][h].V)
    if args.factored_output:
        _lut_load(d, 'output_hi', m.output_head.unembedder_hi)
        _lut_load(d, 'output_lo', m.output_head.unembedder_lo)
    else:
        _lut_load(d, 'output', m.output_head.unembedder)
    if getattr(args, 'layernorm', False) and 'ln_attn_gamma' in d:
        m.ln_attn_gamma[:] = d['ln_attn_gamma']
        m.ln_attn_beta[:]  = d['ln_attn_beta']
        m.ln_ffn_gamma[:]  = d['ln_ffn_gamma']
        m.ln_ffn_beta[:]   = d['ln_ffn_beta']
    d.close()
    print(f"Loaded model weights from {path}")


def new_attention_forward(head, x, y, args):
    CS = args.context_size
    lutQ = head.Q
    lutK = head.K
    lutV = head.V
    lutFinal = head.Final
    treesQ = lutQ.trees
    treesK = lutK.trees
    treesV = lutV.trees
    treesFinal = lutFinal.trees
    y_dim = lutFinal.y_dim
    q_dim = args.attn_qk_dim
    n_pairs = CS * (CS - 1) // 2
    pe_only = getattr(args, 'PE_only', False)
    qkpe_direct_sign = getattr(args, 'qkpe_direct_sign', False)
    normalize_qk = getattr(args, 'normalize_qk', False) and not pe_only
    qk_norm_target_std = np.float32(getattr(args, 'qk_norm_target_std', 1.0 / math.sqrt(3.0)))
    qk_norm_eps = np.float32(getattr(args, 'qk_norm_eps', 1e-6))

    if not pe_only:
        cache_index_batch(lutQ, x, head._q_j, head._q_r_min, head._q_u_min)
        cache_index_batch(lutK, x, head._k_j, head._k_r_min, head._k_u_min)
    cache_index_batch(lutV, x, head._val_j, head._val_r_min, head._val_u_min)

    if (HAS_NUMBA and lutFinal.S.dtype == np.float32
            and lutQ.S.dtype == np.float32
            and lutK.S.dtype == np.float32
            and lutV.S.dtype == np.float32):
        if pe_only:
            _new_attention_pe_only_forward_kernel(
                lutV.S, lutFinal.S,
                head._val_j,
                head._q_embed, head._k_embed, head._v_embed,
                head._qkpe_j, head._qkpe_r_min, head._qkpe_u_min,
                head._final_v_j, head._final_v_r_min, head._final_v_u_min,
                y,
                head.Positional_encoding,
                treesV, treesFinal,
                lutFinal.a_arr, lutFinal.b_arr, lutFinal.bitmasks,
                head._qkpe_a_arr, head._qkpe_b_arr, head._qkpe_bitmasks,
                args.positional_dim, CS, args.n_t, args.n_c, q_dim, y_dim,
                qkpe_direct_sign
            )
        else:
            _new_attention_forward_kernel(
                lutQ.S, lutK.S, lutV.S, lutFinal.S,
                head._q_j, head._k_j, head._val_j,
                head._q_embed, head._k_embed, head._v_embed,
                head._qkpe_j, head._qkpe_r_min, head._qkpe_u_min,
                head._qk_mean, head._qk_inv_std,
                head._final_v_j, head._final_v_r_min, head._final_v_u_min,
                y,
                head.Positional_encoding,
                treesQ, treesK, treesV, treesFinal,
                lutFinal.a_arr, lutFinal.b_arr, lutFinal.bitmasks,
                head._qkpe_a_arr, head._qkpe_b_arr, head._qkpe_bitmasks,
                args.positional_dim, CS, args.n_t, args.n_c,
                args.attn_qk_n_t, q_dim, y_dim, qkpe_direct_sign,
                normalize_qk, qk_norm_target_std, qk_norm_eps
            )
    else:
        if pe_only:
            head._q_embed.fill(0.0)
            head._k_embed.fill(0.0)
            for pos in range(CS):
                head._v_embed[pos] = _f32(lutV.S[treesV, head._val_j[pos]]).sum(axis=0)
        else:
            for pos in range(CS):
                head._q_embed[pos] = _f32(lutQ.S[treesQ, head._q_j[pos]]).sum(axis=0)
                head._k_embed[pos] = _f32(lutK.S[treesK, head._k_j[pos]]).sum(axis=0)
                head._v_embed[pos] = _f32(lutV.S[treesV, head._val_j[pos]]).sum(axis=0)

        cache_index_batch(lutFinal, head._v_embed,
                          head._final_v_j, head._final_v_r_min, head._final_v_u_min)

        active_dim = args.positional_dim if qkpe_direct_sign else q_dim
        for pair in range(n_pairs):
            tok_i = int((np.sqrt(8.0 * pair + 1.0) + 1.0) * 0.5)
            tok_j = pair - (tok_i * (tok_i - 1)) // 2
            pe_pos = tok_i - tok_j
            if normalize_qk:
                qk_vec = _normalize_vector_to_target(
                    head._q_embed[tok_i, :active_dim] + head._k_embed[tok_j, :active_dim],
                    qk_norm_target_std, qk_norm_eps)
            for t in range(args.n_t):
                j = np.int32(0)
                min_abs_u = np.float32(1e38)
                r_min = np.int32(0)
                u_min_val = np.float32(0.0)
                for c in range(args.positional_dim):
                    if qkpe_direct_sign:
                        if normalize_qk:
                            u_val = qk_vec[c] + head.Positional_encoding[pe_pos, c]
                        elif pe_only:
                            u_val = head.Positional_encoding[pe_pos, c]
                        else:
                            u_val = (head._q_embed[tok_i, c] + head._k_embed[tok_j, c]
                                     + head.Positional_encoding[pe_pos, c])
                    else:
                        a = head._qkpe_a_arr[t, c]
                        b = head._qkpe_b_arr[t, c]
                        if normalize_qk:
                            ua = qk_vec[a] + head.Positional_encoding[pe_pos, a]
                            ub = qk_vec[b] + head.Positional_encoding[pe_pos, b]
                        elif pe_only:
                            ua = head.Positional_encoding[pe_pos, a]
                            ub = head.Positional_encoding[pe_pos, b]
                        else:
                            ua = (head._q_embed[tok_i, a] + head._k_embed[tok_j, a]
                                  + head.Positional_encoding[pe_pos, a])
                            ub = (head._q_embed[tok_i, b] + head._k_embed[tok_j, b]
                                  + head.Positional_encoding[pe_pos, b])
                        u_val = ua - ub
                    if u_val > np.float32(0.0):
                        j |= head._qkpe_bitmasks[c]
                    au = abs(u_val)
                    if au < min_abs_u:
                        min_abs_u = au
                        r_min = np.int32(c)
                        u_min_val = u_val
                head._qkpe_j[pair, t] = j
                head._qkpe_r_min[pair, t] = r_min
                head._qkpe_u_min[pair, t] = u_min_val

        for tok_i in range(1, CS):
            for tok_j in range(tok_i):
                pair_idx = tok_i * (tok_i - 1) // 2 + tok_j
                jFinal = (head._final_v_j[tok_j] << args.positional_dim) | head._qkpe_j[pair_idx]
                y[tok_i, :y_dim] += _f32(lutFinal.S[treesFinal, jFinal]).sum(axis=0)

def attention_forward(head, x, y, args):
    CS = args.context_size

    if getattr(args, 'soft_attention', False):
        shift_qk = args.shift_qk
        pd = args.positional_dim
        ED = args.embedding_dim

        # Two independent cache calls (Score: Q+K indexing; Value: K-only indexing)
        cache_index_batch(head.Score, x, head._v_j, head._v_r_min, head._v_u_min)
        cache_index_batch(head.Value, x, head._val_j, head._val_r_min, head._val_u_min)
        cache_PE_index_batch(head.Positional_encoding, head._pe_j, head._pe_r_min, head._pe_u_min, args)

        if HAS_NUMBA and head.Score.S.dtype == np.float32:
            _soft_attn_forward_kernel(head.Score.S, head.Value.S,
                                       head._v_j, head._val_j, head._pe_j,
                                       y, head._attn_w, head._v_vecs,
                                       shift_qk, pd, CS, args.n_t, ED)
        else:
            # Value vectors: v_vecs[pos] = sum_t Value.S[t, val_j[pos, t], :]
            val_trees = head.Value.trees
            v_vecs = _f32(head.Value.S[val_trees, head._val_j]).sum(axis=1)  # (CS, ED)
            head._v_vecs[:] = v_vecs

            # Raw scores for all causal (pos, pos1) pairs
            score_trees = head.Score.trees
            raw_scores = np.full((CS, CS), -1e9, dtype=np.float32)
            for pos in range(1, CS):
                jQ  = head._v_j[pos]                              # (N_T,)
                jK  = head._v_j[:pos]                             # (pos, N_T)
                jPE = head._pe_j[pos - np.arange(pos)]            # (pos, N_T)
                j_bins = (jQ << shift_qk) | (jK << pd) | jPE     # (pos, N_T)
                raw_scores[pos, :pos] = _f32(head.Score.S[score_trees, j_bins, 0]).sum(axis=1)

            # Softmax + weighted sum
            for pos in range(1, CS):
                s = raw_scores[pos, :pos].copy()
                s -= s.max()
                np.exp(s, out=s)
                s /= s.sum()
                head._attn_w[pos, :pos] = s
                y[pos, :ED] += (s[:, np.newaxis] * v_vecs[:pos]).sum(axis=0)
        return

    lut = head.V
    trees = lut.trees
    y_dim = lut.y_dim

    cache_index_batch(lut, x, head._v_j, head._v_r_min, head._v_u_min)
    cache_PE_index_batch(head.Positional_encoding, head._pe_j, head._pe_r_min, head._pe_u_min, args)

    all_jV = head._v_j    # (CS, N_T) — zero-copy
    all_jPE = head._pe_j  # (CS, N_T) — zero-copy

    if HAS_NUMBA and lut.S.dtype == np.float32:
        _attention_forward_kernel(lut.S, all_jV, all_jPE, y,
                                   args.shift_qk, args.positional_dim,
                                   CS, args.n_t, y_dim)
    else:
        for pos in range(1, CS):
            P = pos
            jQ = all_jV[pos]
            jK = all_jV[:P]
            jPE = all_jPE[pos - np.arange(P)]
            j_bins = CONCATENATE_vec(jQ, jK, jPE, args)
            y[pos, :y_dim] += _f32(lut.S[trees, j_bins]).sum(axis=1).sum(axis=0)


def _lut_backward_python(lut, j, r_min, u_min, grad, x_grad, y_dim, CS, N_T, skip_s_update):
    """Python fallback for simple LUT backward with optional S update."""
    global learning_rate
    ED = x_grad.shape[1]
    trees = lut.trees
    for pos in range(CS):
        for t in range(N_T):
            tree = trees[t]
            j_bin = int(j[pos, t])
            r = int(r_min[pos, t])
            jbar = j_bin ^ (1 << r)
            gi = ((_f32(lut.S[tree, jbar, :y_dim]) - _f32(lut.S[tree, j_bin, :y_dim]))
                  * grad[pos, :y_dim]).sum()
            u = float(u_min[pos, t])
            sign_u = 1.0 if u > 0.0 else -1.0
            v = gi * (-0.5 * sign_u / (1.0 + abs(u)) ** 2)
            x_grad[pos, lut.a_arr[t, r]] += v
            x_grad[pos, lut.b_arr[t, r]] -= v
            if not skip_s_update:
                lut.S[tree, j_bin, :y_dim] -= learning_rate * grad[pos, :y_dim]


def new_attention_backward(head, x_grad, y_grad, args, skip_s_update=False,
                           pe_grad_out=None, q_grad_out=None, k_grad_out=None,
                           v_grad_out=None, update_stats=None,
                           stats_prefix='attention'):
    global learning_rate
    CS = args.context_size
    N_T = args.n_t
    Q_N_T = args.attn_qk_n_t
    POSITIONAL_DIM = args.positional_dim
    QD = args.attn_qk_dim
    ED = args.embedding_dim
    pe_only = getattr(args, 'PE_only', False)
    freeze_pe = getattr(args, 'freeze_PE', False)
    qkpe_direct_sign = getattr(args, 'qkpe_direct_sign', False)
    normalize_qk = getattr(args, 'normalize_qk', False) and not pe_only
    qk_norm_target_std = np.float32(getattr(args, 'qk_norm_target_std', 1.0 / math.sqrt(3.0)))
    qk_norm_eps = np.float32(getattr(args, 'qk_norm_eps', 1e-6))

    q_grad = q_grad_out if q_grad_out is not None else head._q_grad
    k_grad = k_grad_out if k_grad_out is not None else head._k_grad
    v_grad = v_grad_out if v_grad_out is not None else head._v_grad
    pe_grad = pe_grad_out if pe_grad_out is not None else head._pe_grad
    q_grad.fill(0.0)
    k_grad.fill(0.0)
    v_grad.fill(0.0)
    pe_grad.fill(0.0)

    lutQ = head.Q
    lutK = head.K
    lutV = head.V
    lutFinal = head.Final
    treesFinal = lutFinal.trees
    y_dim = lutFinal.y_dim

    if update_stats is not None and not skip_s_update:
        if not pe_only:
            update_stats.record_simple(f'{stats_prefix}_Q', lutQ, head._q_j)
            update_stats.record_simple(f'{stats_prefix}_K', lutK, head._k_j)
        update_stats.record_simple(f'{stats_prefix}_V', lutV, head._val_j)
        update_stats.record_new_attention_final(
            f'{stats_prefix}_Final', lutFinal, head._final_v_j, head._qkpe_j, args)
        update_stats.record_new_attention_final_coords(
            f'{stats_prefix}_Final', lutFinal, y_grad, args)

    if (HAS_NUMBA and lutFinal.S.dtype == np.float32
            and lutQ.S.dtype == np.float32
            and lutK.S.dtype == np.float32
            and lutV.S.dtype == np.float32):
        _new_attention_final_backward_kernel(
            lutFinal.S,
            head._final_v_j, head._final_v_r_min, head._final_v_u_min,
            head._qkpe_j, head._qkpe_r_min, head._qkpe_u_min,
            y_grad,
            head._q_embed, head._k_embed, head.Positional_encoding,
            head._qk_mean, head._qk_inv_std,
            q_grad, k_grad, v_grad, pe_grad,
            treesFinal,
            lutFinal.a_arr, lutFinal.b_arr,
            head._qkpe_a_arr, head._qkpe_b_arr,
            POSITIONAL_DIM, CS, N_T, y_dim, QD,
            np.float32(learning_rate), skip_s_update, pe_only, qkpe_direct_sign,
            freeze_pe, normalize_qk, qk_norm_target_std, qk_norm_eps)
    else:
        for tok_i in range(1, CS):
            y_g = y_grad[tok_i, :y_dim]
            for tok_j in range(tok_i):
                pair_idx = tok_i * (tok_i - 1) // 2 + tok_j
                pe_pos = tok_i - tok_j
                for t in range(N_T):
                    tree = treesFinal[t]
                    j_bin = (int(head._final_v_j[tok_j, t]) << POSITIONAL_DIM) | int(head._qkpe_j[pair_idx, t])
                    S_j = _f32(lutFinal.S[tree, j_bin])

                    r_v = int(head._final_v_r_min[tok_j, t])
                    jbar_v = j_bin ^ (1 << (r_v + POSITIONAL_DIM))
                    gi_v = ((_f32(lutFinal.S[tree, jbar_v]) - S_j) * y_g).sum()
                    u_v = float(head._final_v_u_min[tok_j, t])
                    sign_v = 1.0 if u_v > 0.0 else -1.0
                    delta_v = gi_v * (-0.5 * sign_v / (1.0 + abs(u_v)) ** 2)
                    a_v = int(lutFinal.a_arr[t, r_v])
                    b_v = int(lutFinal.b_arr[t, r_v])
                    v_grad[tok_j, a_v] += delta_v
                    v_grad[tok_j, b_v] -= delta_v

                    r_qkpe = int(head._qkpe_r_min[pair_idx, t])
                    jbar_qkpe = j_bin ^ (1 << r_qkpe)
                    gi_qkpe = ((_f32(lutFinal.S[tree, jbar_qkpe]) - S_j) * y_g).sum()
                    u_qkpe = float(head._qkpe_u_min[pair_idx, t])
                    sign_qkpe = 1.0 if u_qkpe > 0.0 else -1.0
                    delta_qkpe = gi_qkpe * (-0.5 * sign_qkpe / (1.0 + abs(u_qkpe)) ** 2)
                    if normalize_qk:
                        active_dim = POSITIONAL_DIM if qkpe_direct_sign else QD
                        upstream = np.zeros(active_dim, dtype=np.float32)
                        if qkpe_direct_sign:
                            upstream[r_qkpe] = delta_qkpe
                        else:
                            a_qkpe = int(head._qkpe_a_arr[t, r_qkpe])
                            b_qkpe = int(head._qkpe_b_arr[t, r_qkpe])
                            upstream[a_qkpe] += delta_qkpe
                            upstream[b_qkpe] -= delta_qkpe

                        qk_raw = (head._q_embed[tok_i, :active_dim]
                                  + head._k_embed[tok_j, :active_dim])
                        qk_delta = _normalize_vector_backward(
                            qk_raw, upstream, qk_norm_target_std, qk_norm_eps)
                        q_grad[tok_i, :active_dim] += qk_delta
                        k_grad[tok_j, :active_dim] += qk_delta
                        if not freeze_pe:
                            pe_grad[pe_pos, :active_dim] += upstream
                    elif qkpe_direct_sign:
                        if not pe_only:
                            q_grad[tok_i, r_qkpe] += delta_qkpe
                            k_grad[tok_j, r_qkpe] += delta_qkpe
                        if not freeze_pe:
                            pe_grad[pe_pos, r_qkpe] += delta_qkpe
                    else:
                        a_qkpe = int(head._qkpe_a_arr[t, r_qkpe])
                        b_qkpe = int(head._qkpe_b_arr[t, r_qkpe])
                        if not pe_only:
                            q_grad[tok_i, a_qkpe] += delta_qkpe
                            q_grad[tok_i, b_qkpe] -= delta_qkpe
                            k_grad[tok_j, a_qkpe] += delta_qkpe
                            k_grad[tok_j, b_qkpe] -= delta_qkpe
                        if not freeze_pe:
                            pe_grad[pe_pos, a_qkpe] += delta_qkpe
                            pe_grad[pe_pos, b_qkpe] -= delta_qkpe

        if not skip_s_update:
            for t in range(N_T):
                tree = treesFinal[t]
                for tok_i in range(1, CS):
                    y_g = y_grad[tok_i, :y_dim]
                    for tok_j in range(tok_i):
                        pair_idx = tok_i * (tok_i - 1) // 2 + tok_j
                        j_bin = (int(head._final_v_j[tok_j, t]) << POSITIONAL_DIM) | int(head._qkpe_j[pair_idx, t])
                        lutFinal.S[tree, j_bin] -= learning_rate * y_g

    if pe_grad_out is None and not freeze_pe:
        head.Positional_encoding -= learning_rate * pe_grad

    if update_stats is not None and not skip_s_update:
        if not pe_only:
            update_stats.record_simple_coords(f'{stats_prefix}_Q', lutQ, q_grad)
            update_stats.record_simple_coords(f'{stats_prefix}_K', lutK, k_grad)
        update_stats.record_simple_coords(f'{stats_prefix}_V', lutV, v_grad)

    if not pe_only:
        if HAS_NUMBA and lutQ.S.dtype == np.float32:
            if skip_s_update:
                _lut_backward_kernel_noupdate(lutQ.S, lutQ.trees, head._q_j, head._q_r_min, head._q_u_min,
                                              q_grad, x_grad, lutQ.a_arr, lutQ.b_arr,
                                              QD, CS, Q_N_T)
            else:
                _lut_backward_kernel(lutQ.S, lutQ.trees, head._q_j, head._q_r_min, head._q_u_min,
                                     q_grad, x_grad, lutQ.a_arr, lutQ.b_arr,
                                     QD, CS, Q_N_T, np.float32(learning_rate))
        else:
            _lut_backward_python(lutQ, head._q_j, head._q_r_min, head._q_u_min,
                                 q_grad, x_grad, QD, CS, Q_N_T, skip_s_update)

        if HAS_NUMBA and lutK.S.dtype == np.float32:
            if skip_s_update:
                _lut_backward_kernel_noupdate(lutK.S, lutK.trees, head._k_j, head._k_r_min, head._k_u_min,
                                              k_grad, x_grad, lutK.a_arr, lutK.b_arr,
                                              QD, CS, Q_N_T)
            else:
                _lut_backward_kernel(lutK.S, lutK.trees, head._k_j, head._k_r_min, head._k_u_min,
                                     k_grad, x_grad, lutK.a_arr, lutK.b_arr,
                                     QD, CS, Q_N_T, np.float32(learning_rate))
        else:
            _lut_backward_python(lutK, head._k_j, head._k_r_min, head._k_u_min,
                                 k_grad, x_grad, QD, CS, Q_N_T, skip_s_update)

    if HAS_NUMBA and lutV.S.dtype == np.float32:
        if skip_s_update:
            _lut_backward_kernel_noupdate(lutV.S, lutV.trees, head._val_j, head._val_r_min, head._val_u_min,
                                          v_grad, x_grad, lutV.a_arr, lutV.b_arr,
                                          ED, CS, N_T)
        else:
            _lut_backward_kernel(lutV.S, lutV.trees, head._val_j, head._val_r_min, head._val_u_min,
                                 v_grad, x_grad, lutV.a_arr, lutV.b_arr,
                                 ED, CS, N_T, np.float32(learning_rate))
    else:
        _lut_backward_python(lutV, head._val_j, head._val_r_min, head._val_u_min,
                             v_grad, x_grad, ED, CS, N_T, skip_s_update)


def attention_backward(head, x_grad, y_grad, args, skip_s_update=False, pe_grad_out=None,
                       score_grad_out=None, val_grad_out=None, update_stats=None,
                       stats_prefix='attention'):
    global learning_rate
    CS = args.context_size
    N_T = args.n_t
    POSITIONAL_DIM = args.positional_dim
    freeze_pe = getattr(args, 'freeze_PE', False)

    if getattr(args, 'soft_attention', False):
        shift_qk = args.shift_qk
        pd = POSITIONAL_DIM
        ED = x_grad.shape[1]
        val_lut = head.Value
        score_lut = head.Score
        if update_stats is not None and not skip_s_update:
            update_stats.record_simple(f'{stats_prefix}_Value', val_lut, head._val_j)
            record_attention_update_rows(update_stats, f'{stats_prefix}_Score',
                                         score_lut, head._v_j, head._pe_j, args)

        if HAS_NUMBA and score_lut.S.dtype == np.float32:
            val_grad   = val_grad_out   if val_grad_out   is not None else np.zeros((CS, ED),              dtype=np.float32)
            score_grad = score_grad_out if score_grad_out is not None else np.zeros((CS, CS),              dtype=np.float32)
            pos_grad   = pe_grad_out    if pe_grad_out    is not None else np.zeros((CS, N_T, POSITIONAL_DIM), dtype=np.float32)

            _soft_attn_backward_value_kernel(
                val_lut.S, head._val_j, head._val_r_min, head._val_u_min,
                head._attn_w, y_grad, val_grad, x_grad,
                val_lut.a_arr, val_lut.b_arr,
                CS, N_T, ED, np.float32(learning_rate), skip_s_update)

            _soft_attn_backward_score_kernel(
                score_lut.S, head._v_j, head._v_r_min, head._v_u_min,
                head._pe_j, head._pe_r_min, head._pe_u_min,
                head._attn_w, head._v_vecs, y_grad, score_grad, x_grad, pos_grad,
                score_lut.a_arr, score_lut.b_arr,
                shift_qk, pd, CS, N_T, ED, np.float32(learning_rate), skip_s_update,
                freeze_pe)

            if update_stats is not None and not skip_s_update:
                update_stats.record_simple_coords(f'{stats_prefix}_Value', val_lut, val_grad)
                update_stats.record_attention_scalar_coords(
                    f'{stats_prefix}_Score', score_lut, score_grad, args)

            if pe_grad_out is None and not freeze_pe:
                head.Positional_encoding -= learning_rate * pos_grad
        else:
            val_trees = val_lut.trees
            score_trees = score_lut.trees

            # --- Value path ---
            val_grad = (_f32(head._attn_w)[:, :, np.newaxis] * y_grad[:, np.newaxis, :]).sum(axis=0)

            jbar_val = head._val_j ^ (1 << head._val_r_min)
            S_j_val    = _f32(val_lut.S[val_trees, head._val_j])
            S_jbar_val = _f32(val_lut.S[val_trees, jbar_val])
            gi_val = ((S_jbar_val - S_j_val) * val_grad[:, np.newaxis, :]).sum(axis=2)
            sign_u_val = np.where(head._val_u_min > 0, 1.0, -1.0)
            v_val = gi_val * (-0.5 * sign_u_val / (1 + np.abs(head._val_u_min))**2)

            pos_idx = np.arange(CS)[:, np.newaxis]
            idx_a_val = val_lut.a_arr[val_trees, head._val_r_min]
            idx_b_val = val_lut.b_arr[val_trees, head._val_r_min]
            flat_a_val = (pos_idx * ED + idx_a_val).ravel()
            flat_b_val = (pos_idx * ED + idx_b_val).ravel()
            flat_size = x_grad.size
            x_grad.ravel()[:] += np.bincount(flat_a_val, weights=v_val.ravel(), minlength=flat_size)
            x_grad.ravel()[:] -= np.bincount(flat_b_val, weights=v_val.ravel(), minlength=flat_size)

            if not skip_s_update:
                for pos1 in range(CS):
                    val_lut.S[val_trees, head._val_j[pos1]] -= learning_rate * val_grad[pos1]

            if val_grad_out is not None:
                val_grad_out[:] = val_grad

            # --- Score path ---
            a = y_grad @ head._v_vecs.T
            baseline = (head._attn_w * a).sum(axis=1, keepdims=True)
            score_grad = head._attn_w * (a - baseline)

            pos_grad = np.zeros((CS, N_T, POSITIONAL_DIM), dtype=np.float32)

            for pos in range(1, CS):
                for pos1 in range(pos):
                    sg = score_grad[pos, pos1]
                    pe_pos = pos - pos1
                    for t in range(N_T):
                        jQ_t  = int(head._v_j[pos, t])
                        jK_t  = int(head._v_j[pos1, t])
                        jPE_t = int(head._pe_j[pe_pos, t])
                        j_bin = (jQ_t << shift_qk) | (jK_t << pd) | jPE_t

                        abs_uQ  = abs(float(head._v_u_min[pos, t]))
                        abs_uK  = abs(float(head._v_u_min[pos1, t]))
                        abs_uPE = abs(float(head._pe_u_min[pe_pos, t]))

                        q_wins = abs_uQ < abs_uK
                        if q_wins:
                            flip_bit = int(head._v_r_min[pos, t]) + shift_qk
                            u_min = float(head._v_u_min[pos, t])
                            r = int(head._v_r_min[pos, t])
                        else:
                            flip_bit = int(head._v_r_min[pos1, t]) + pd
                            u_min = float(head._v_u_min[pos1, t])
                            r = int(head._v_r_min[pos1, t])

                        jbar = j_bin ^ (1 << flip_bit)
                        gi = (float(score_lut.S[t, jbar, 0]) - float(score_lut.S[t, j_bin, 0])) * sg
                        sign_u = 1.0 if u_min > 0 else -1.0
                        v = gi * (-0.5 * sign_u / (1.0 + abs(u_min))**2)

                        if q_wins:
                            x_grad[pos, score_lut.a_arr[t, r]] += v
                            x_grad[pos, score_lut.b_arr[t, r]] -= v
                        else:
                            x_grad[pos1, score_lut.a_arr[t, r]] += v
                            x_grad[pos1, score_lut.b_arr[t, r]] -= v

                        if not freeze_pe and abs_uPE < abs_uQ and abs_uPE < abs_uK:
                            rPE = int(head._pe_r_min[pe_pos, t])
                            jbarPE = j_bin ^ (1 << rPE)
                            giPE = (float(score_lut.S[t, jbarPE, 0]) - float(score_lut.S[t, j_bin, 0])) * sg
                            u_pe = float(head._pe_u_min[pe_pos, t])
                            sign_pe = 1.0 if u_pe > 0 else -1.0
                            deltaPE = giPE * (-0.5 * sign_pe / (1.0 + abs(u_pe))**2)
                            pos_grad[pe_pos, t, rPE] += deltaPE

                        if not skip_s_update:
                            score_lut.S[t, j_bin, 0] -= learning_rate * sg

            if score_grad_out is not None:
                score_grad_out[:] = score_grad

            if update_stats is not None and not skip_s_update:
                update_stats.record_simple_coords(f'{stats_prefix}_Value', val_lut, val_grad)
                update_stats.record_attention_scalar_coords(
                    f'{stats_prefix}_Score', score_lut, score_grad, args)

            if pe_grad_out is not None:
                pe_grad_out += pos_grad
            elif not freeze_pe:
                head.Positional_encoding -= learning_rate * pos_grad
        return

    lut = head.V
    trees = lut.trees
    y_dim = lut.y_dim
    if update_stats is not None and not skip_s_update:
        record_attention_update_rows(update_stats, stats_prefix, lut,
                                     head._v_j, head._pe_j, args)
        update_stats.record_attention_coords(stats_prefix, lut, y_grad, args)

    pos_grad = np.zeros((CS, N_T, POSITIONAL_DIM), dtype=np.float32)

    # Cache arrays already computed in attention_forward — zero-copy references
    all_jV = head._v_j        # (CS, N_T)
    all_rV = head._v_r_min    # (CS, N_T)
    all_uV = head._v_u_min    # (CS, N_T)
    all_jPE = head._pe_j      # (CS, N_T)
    all_rPE = head._pe_r_min  # (CS, N_T)
    all_uPE = head._pe_u_min  # (CS, N_T)

    if HAS_NUMBA and lut.S.dtype == np.float32:
        if skip_s_update:
            _attention_backward_kernel_noupdate(lut.S, all_jV, all_rV, all_uV,
                                                all_jPE, all_rPE, all_uPE,
                                                y_grad, x_grad, pos_grad,
                                                lut.a_arr, lut.b_arr,
                                                args.shift_qk, POSITIONAL_DIM,
                                                CS, N_T, y_dim, freeze_pe)
        else:
            _attention_backward_kernel(lut.S, all_jV, all_rV, all_uV,
                                        all_jPE, all_rPE, all_uPE,
                                        y_grad, x_grad, pos_grad,
                                        lut.a_arr, lut.b_arr,
                                        args.shift_qk, POSITIONAL_DIM,
                                        CS, N_T, y_dim,
                                        np.float32(learning_rate), freeze_pe)
    else:
        ED = x_grad.shape[1]
        flat_size = x_grad.size
        abs_all_uV = np.abs(all_uV)
        abs_all_uPE = np.abs(all_uPE)
        trees_b_full = np.broadcast_to(trees, (CS, N_T))
        pos1_grid = np.broadcast_to(np.arange(CS)[:, np.newaxis], (CS, N_T))

        for pos in range(1, CS):
            P = pos
            y_g = y_grad[pos, :y_dim]

            jQ = all_jV[pos]
            rQ = all_rV[pos]
            abs_uQ = abs_all_uV[pos]

            jK = all_jV[:P]
            rK = all_rV[:P]
            abs_uK = abs_all_uV[:P]

            pe_idx = pos - np.arange(P)
            jPE = all_jPE[pe_idx]
            rPE = all_rPE[pe_idx]

            j_bins = CONCATENATE_vec(jQ, jK, jPE, args)
            S_j = _f32(lut.S[trees, j_bins])

            q_mask = abs_uQ < abs_uK
            flip_bit = np.where(q_mask, rQ + args.shift_qk, rK + POSITIONAL_DIM)
            jbar_bins = j_bins ^ (1 << flip_bit)
            S_jbar = _f32(lut.S[trees, jbar_bins])

            gi = ((S_jbar - S_j) * y_g).sum(axis=2)
            uQ_vals = all_uV[pos]
            uK_vals = all_uV[:P]
            u_min_qk = np.where(q_mask, uQ_vals, uK_vals)
            abs_u_min_qk = np.where(q_mask, abs_uQ, abs_uK)
            sign_u = np.where(u_min_qk > 0, 1.0, -1.0)
            v = gi * (-0.5 * sign_u / (1 + abs_u_min_qk)**2)
            r_min_qk = np.where(q_mask, rQ, rK)

            trees_b = trees_b_full[:P]

            if q_mask.any():
                q_t = trees_b[q_mask]
                q_r = r_min_qk[q_mask]
                q_v = v[q_mask]
                x_grad[pos] += np.bincount(lut.a_arr[q_t, q_r], weights=q_v, minlength=ED)
                x_grad[pos] -= np.bincount(lut.b_arr[q_t, q_r], weights=q_v, minlength=ED)

            k_mask = ~q_mask
            if k_mask.any():
                k_t = trees_b[k_mask]
                k_r = r_min_qk[k_mask]
                k_v = v[k_mask]
                k_pos1 = pos1_grid[:P][k_mask]
                flat_a = k_pos1 * ED + lut.a_arr[k_t, k_r]
                flat_b = k_pos1 * ED + lut.b_arr[k_t, k_r]
                x_grad.ravel()[:] += np.bincount(flat_a, weights=k_v, minlength=flat_size)
                x_grad.ravel()[:] -= np.bincount(flat_b, weights=k_v, minlength=flat_size)

            abs_uPE = abs_all_uPE[pe_idx]
            pe_mask = (abs_uPE < abs_uQ) & (abs_uPE < abs_uK)
            if freeze_pe:
                pe_mask = np.zeros_like(pe_mask, dtype=bool)
            if pe_mask.any():
                jbarPE_bins = j_bins ^ (1 << rPE)
                S_jbarPE = _f32(lut.S[trees, jbarPE_bins])
                giPE = ((S_jbarPE - S_j) * y_g).sum(axis=2)
                uPE_vals = all_uPE[pe_idx]
                u_pe = uPE_vals[pe_mask]
                sign_pe = np.where(u_pe > 0, 1.0, -1.0)
                deltaPE = giPE[pe_mask] * (-0.5 * sign_pe / (1 + np.abs(u_pe))**2)
                pe_t = trees_b[pe_mask]
                pe_r = rPE[pe_mask]
                pe_pair = pos1_grid[:P][pe_mask]
                np.add.at(pos_grad, (pe_idx[pe_pair], pe_t, pe_r), deltaPE)

            if not skip_s_update:
                np.subtract.at(lut.S, (trees_b.ravel(), j_bins.ravel()),
                                np.float32(learning_rate) * y_g)

    if pe_grad_out is not None:
        pe_grad_out += pos_grad
    elif not freeze_pe:
        head.Positional_encoding -= learning_rate * pos_grad


def model_forward(m, args, training=True):
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads
    use_ln = getattr(args, 'layernorm', False)
    dr     = getattr(args, 'dropout', 0.0)

    for l in range(NUM_LAYERS):
        # Attention (pre-norm)
        if use_ln:
            m._x_buf[:], m._ln_attn_xhat[l][:], m._ln_attn_rstd[l][:] = \
                layernorm_forward(m.z, m.ln_attn_gamma[l], m.ln_attn_beta[l])
        else:
            np.copyto(m._x_buf, m.z)

        z_pre_attn = m.z.copy() if (dr > 0.0 and training) else None
        for h in range(NUM_HEADS):
            if getattr(args, 'new_attention', False):
                new_attention_forward(m.head[l][h], m._x_buf, m.z, args)
            else:
                attention_forward(m.head[l][h], m._x_buf, m.z, args)
        if dr > 0.0 and training:
            mask = (np.random.rand(*m.z.shape) >= dr).astype(np.float32) / (1.0 - dr)
            m._drop_attn_mask[l] = mask
            m.z[:] = z_pre_attn + (m.z - z_pre_attn) * mask

        # FFN (pre-norm)
        lut = m.FFN[l]
        if use_ln:
            m._ln_ffn_buf[:], m._ln_ffn_xhat[l][:], m._ln_ffn_rstd[l][:] = \
                layernorm_forward(m.z, m.ln_ffn_gamma[l], m.ln_ffn_beta[l])
            ffn_input = m._ln_ffn_buf
        else:
            ffn_input = m.z

        z_pre_ffn = m.z.copy() if (dr > 0.0 and training) else None
        cache_index_batch(lut, ffn_input, m._ffn_j[l], m._ffn_r_min[l], m._ffn_u_min[l])
        if HAS_NUMBA and lut.S.dtype == np.float32:
            _lut_forward_add_kernel(lut.S, lut.trees, m._ffn_j[l], m.z,
                                    args.context_size, args.n_t, lut.y_dim)
        else:
            m.z[:, :lut.y_dim] += _f32(lut.S[lut.trees, m._ffn_j[l]]).sum(axis=1)
        if dr > 0.0 and training:
            mask = (np.random.rand(*m.z.shape) >= dr).astype(np.float32) / (1.0 - dr)
            m._drop_ffn_mask[l] = mask
            m.z[:] = z_pre_ffn + (m.z - z_pre_ffn) * mask

    m.output_head.forward(m.z, args)


def model_backward(m, args, grad_accum=None, update_stats=None):
    global learning_rate
    EMBEDDING_DIM = args.embedding_dim
    NUM_LAYERS = args.num_layers
    NUM_HEADS = args.num_heads
    skip_s = grad_accum is not None
    use_ln = getattr(args, 'layernorm', False)
    dr     = getattr(args, 'dropout', 0.0)

    x_grad = np.zeros((args.context_size, EMBEDDING_DIM), dtype=np.float32)

    if args.factored_output:
        m.output_head.backward(x_grad, args, skip_s_update=skip_s,
                               update_stats=update_stats, stats_prefix='output')
    else:
        m.output_head.backward(x_grad, args, skip_s_update=skip_s,
                               update_stats=update_stats, stats_name='output')

    for l in range(NUM_LAYERS - 1, -1, -1):
        # FFN backward
        y_grad = x_grad.copy()  # don't zero-out x_grad, but add to it (resnet connections)
        if dr > 0.0 and m._drop_ffn_mask[l] is not None:
            y_grad *= m._drop_ffn_mask[l]

        lut = m.FFN[l]
        trees = lut.trees
        CS = args.context_size
        y_dim = lut.y_dim
        j = m._ffn_j[l]
        r_min = m._ffn_r_min[l]
        u_min = m._ffn_u_min[l]

        x_before_ffn = x_grad.copy() if use_ln else None

        # FFN backward uses batch-read semantics: read S for ALL positions
        # before any updates. This can't be fused into a per-position kernel
        # because positions share bins (2^n_c is small).
        jbar = j ^ (1 << r_min)
        S_j = _f32(lut.S[trees, j])
        S_jbar = _f32(lut.S[trees, jbar])
        gi = ((S_jbar - S_j) * y_grad[:, np.newaxis, :y_dim]).sum(axis=2)
        sign_u = np.where(u_min > 0, 1.0, -1.0)
        v = gi * (-0.5 * sign_u / (1 + np.abs(u_min))**2)

        ED = EMBEDDING_DIM
        pos_idx = np.arange(CS)[:, np.newaxis]
        idx_a = lut.a_arr[trees, r_min]
        idx_b = lut.b_arr[trees, r_min]
        flat_a = (pos_idx * ED + idx_a).ravel()
        flat_b = (pos_idx * ED + idx_b).ravel()
        v_flat = v.ravel()
        flat_size = x_grad.size
        x_grad.ravel()[:] += np.bincount(flat_a, weights=v_flat, minlength=flat_size)
        x_grad.ravel()[:] -= np.bincount(flat_b, weights=v_flat, minlength=flat_size)

        if use_ln:
            ln_ffn_dx = x_grad - x_before_ffn
            dx, dgamma, dbeta = layernorm_backward(
                ln_ffn_dx, m._ln_ffn_xhat[l], m._ln_ffn_rstd[l], m.ln_ffn_gamma[l])
            x_grad[:] = x_before_ffn + dx
            if grad_accum is not None:
                grad_accum.ln_ffn_dgamma[l] += dgamma
                grad_accum.ln_ffn_dbeta[l]  += dbeta
            else:
                m.ln_ffn_gamma[l] -= learning_rate * dgamma
                m.ln_ffn_beta[l]  -= learning_rate * dbeta

        if grad_accum is not None:
            # Save y_grad for sparse replay at apply time
            grad_accum.ffn_y_grads[l] = y_grad[:, :y_dim].copy()
        else:
            if update_stats is not None:
                update_stats.record_simple(f'ffn_{l}', lut, j)
                update_stats.record_simple_coords(f'ffn_{l}', lut, y_grad)
            if HAS_NUMBA and lut.S.dtype == np.float32:
                _lut_s_replay_kernel(lut.S, trees, j, y_grad, y_dim, CS, args.n_t,
                                     np.float32(learning_rate))
            else:
                for pos in range(CS):
                    lut.S[trees, j[pos]] -= learning_rate * y_grad[pos, :y_dim]

        # Attention backward
        y_grad = x_grad.copy()  # don't zero-out x_grad, but add to it (resnet connections)
        if dr > 0.0 and m._drop_attn_mask[l] is not None:
            y_grad *= m._drop_attn_mask[l]

        x_before_attn = x_grad.copy() if use_ln else None

        soft = getattr(args, 'soft_attention', False)
        new_attn = getattr(args, 'new_attention', False)
        if grad_accum is not None and not soft:
            # Save y_grad once per layer (shared across heads) for hard/new attention replay
            grad_accum.attn_y_grads[l] = y_grad.copy()
        for h in range(NUM_HEADS):
            if new_attn:
                if grad_accum is not None:
                    new_attention_backward(
                        m.head[l][h], x_grad, y_grad, args,
                        skip_s_update=True,
                        pe_grad_out=grad_accum.pe_grads[l][h],
                        q_grad_out=grad_accum.new_attn_q_grads[l][h],
                        k_grad_out=grad_accum.new_attn_k_grads[l][h],
                        v_grad_out=grad_accum.new_attn_v_grads[l][h])
                else:
                    new_attention_backward(m.head[l][h], x_grad, y_grad, args,
                                           update_stats=update_stats,
                                           stats_prefix=f'head_{l}_{h}')
            elif grad_accum is not None:
                attention_backward(m.head[l][h], x_grad, y_grad, args,
                                   skip_s_update=True,
                                   pe_grad_out=grad_accum.pe_grads[l][h],
                                   score_grad_out=grad_accum.attn_score_grads[l][h] if soft else None,
                                   val_grad_out=grad_accum.attn_val_grads[l][h] if soft else None)
            else:
                stats_prefix = f'head_{l}_{h}' if soft else f'head_{l}_{h}_V'
                attention_backward(m.head[l][h], x_grad, y_grad, args,
                                   update_stats=update_stats,
                                   stats_prefix=stats_prefix)

        if use_ln:
            ln_attn_dx = x_grad - x_before_attn
            dx, dgamma, dbeta = layernorm_backward(
                ln_attn_dx, m._ln_attn_xhat[l], m._ln_attn_rstd[l], m.ln_attn_gamma[l])
            x_grad[:] = x_before_attn + dx
            if grad_accum is not None:
                grad_accum.ln_attn_dgamma[l] += dgamma
                grad_accum.ln_attn_dbeta[l]  += dbeta
            else:
                m.ln_attn_gamma[l] -= learning_rate * dgamma
                m.ln_attn_beta[l]  -= learning_rate * dbeta

    # no need to compute gradients for the embedder; just update the synaptic values
    # (disabled in the C version too)
    # for pos in range(args.context_size):
    #     for k in range(EMBEDDING_DIM):
    #         m.Token_embedder[m.tokens[pos]][k] -= learning_rate * x_grad[pos][k]


def model_training_step(m, args, update_stats=None):
    model_forward(m, args, training=True)
    loss = m.output_head.training_loss(args)
    m.output_head.compute_gradients(args)
    model_backward(m, args, update_stats=update_stats)
    return loss


def _batch_element_step(be, args):
    """Forward + compute_gradients + backward with gradient accumulation for one batch element."""
    model_forward(be, args, training=True)
    loss = be.output_head.training_loss(args)
    be.output_head.compute_gradients(args)
    model_backward(be, args, grad_accum=be.grad_accum)
    return loss


def _replay_ffn_task(S, trees, j_list, ygrad_list, y_dim, CS, N_T, scale, use_numba):
    """Replay FFN S updates for one layer across all batch elements."""
    for j, y_grad in zip(j_list, ygrad_list):
        if use_numba and S.dtype == np.float32:
            _lut_s_replay_kernel(S, trees, j, y_grad, y_dim, CS, N_T, scale)
        else:
            for pos in range(CS):
                S[trees, j[pos]] -= scale * y_grad[pos]


def _replay_attn_task(S, jV_list, jPE_list, ygrad_list, PE, pe_grad_list,
                      shift_qk, pd, CS, N_T, y_dim, scale, use_numba,
                      freeze_pe=False):
    """Replay attention S + PE updates for one head across all batch elements."""
    for jV, jPE, y_grad in zip(jV_list, jPE_list, ygrad_list):
        if use_numba and S.dtype == np.float32:
            _attention_s_replay_kernel(S, jV, jPE, y_grad,
                                       shift_qk, pd, CS, N_T, y_dim, scale)
        else:
            trees = np.arange(N_T, dtype=np.int32)
            trees_b_full = np.broadcast_to(trees, (CS, N_T))
            for pos in range(1, CS):
                y_g = y_grad[pos, :y_dim]
                jQ = jV[pos]
                jK = jV[:pos]
                pe_idx = pos - np.arange(pos)
                jPE_slice = jPE[pe_idx]
                j_bins = (jQ << shift_qk) | (jK << pd) | jPE_slice
                trees_b = trees_b_full[:pos]
                np.subtract.at(S, (trees_b.ravel(), j_bins.ravel()), scale * y_g)
    if not freeze_pe:
        for pe_grad in pe_grad_list:
            PE -= scale * pe_grad


def _replay_new_attn_task(head, q_j_list, k_j_list, val_j_list,
                          final_v_j_list, qkpe_j_list, ygrad_list,
                          q_grad_list, k_grad_list, v_grad_list,
                          pe_grad_list, CS, N_T, Q_N_T, scale, use_numba,
                          pe_only=False, freeze_pe=False):
    """Replay new-attention Q/K/V/Final S updates and vector PE updates."""
    q_lut = head.Q
    k_lut = head.K
    v_lut = head.V
    final_lut = head.Final
    y_dim = final_lut.y_dim

    for q_j, k_j, val_j, final_v_j, qkpe_j, y_grad, q_grad, k_grad, v_grad in zip(
            q_j_list, k_j_list, val_j_list, final_v_j_list, qkpe_j_list,
            ygrad_list, q_grad_list, k_grad_list, v_grad_list):
        if not pe_only:
            if use_numba and q_lut.S.dtype == np.float32:
                _lut_s_replay_kernel(q_lut.S, q_lut.trees, q_j, q_grad,
                                     q_lut.y_dim, CS, Q_N_T, scale)
            else:
                for pos in range(CS):
                    q_lut.S[q_lut.trees, q_j[pos]] -= scale * q_grad[pos, :q_lut.y_dim]

            if use_numba and k_lut.S.dtype == np.float32:
                _lut_s_replay_kernel(k_lut.S, k_lut.trees, k_j, k_grad,
                                     k_lut.y_dim, CS, Q_N_T, scale)
            else:
                for pos in range(CS):
                    k_lut.S[k_lut.trees, k_j[pos]] -= scale * k_grad[pos, :k_lut.y_dim]

        if use_numba and v_lut.S.dtype == np.float32:
            _lut_s_replay_kernel(v_lut.S, v_lut.trees, val_j, v_grad,
                                 v_lut.y_dim, CS, N_T, scale)
        else:
            for pos in range(CS):
                v_lut.S[v_lut.trees, val_j[pos]] -= scale * v_grad[pos, :v_lut.y_dim]

        if use_numba and final_lut.S.dtype == np.float32:
            _new_attention_final_s_replay_kernel(final_lut.S, final_lut.trees,
                                                 final_v_j, qkpe_j, y_grad,
                                                 head._qkpe_bitmasks.shape[0],
                                                 CS, N_T, y_dim, scale)
        else:
            for t in range(N_T):
                tree = final_lut.trees[t]
                for tok_i in range(1, CS):
                    y_g = y_grad[tok_i, :y_dim]
                    for tok_j in range(tok_i):
                        pair = tok_i * (tok_i - 1) // 2 + tok_j
                        j_bin = (int(final_v_j[tok_j, t]) << head._qkpe_bitmasks.shape[0]) | int(qkpe_j[pair, t])
                        final_lut.S[tree, j_bin] -= scale * y_g

    if not freeze_pe:
        for pe_grad in pe_grad_list:
            head.Positional_encoding -= scale * pe_grad


def _replay_output_task(S, trees, j_list, output_list, y_dim, CS, N_T, scale, use_numba):
    """Replay output head S updates for one LUT across all batch elements."""
    for j, output in zip(j_list, output_list):
        if use_numba and S.dtype == np.float32:
            _lut_s_replay_kernel(S, trees, j, output, y_dim, CS, N_T, scale)
        else:
            for pos in range(CS):
                S[trees, j[pos]] -= scale * output[pos, :y_dim]


def _replay_soft_attn_task(ScoreS, ValueS, score_trees, val_trees,
                            jV_list, jPE_list, val_j_list, score_grad_list, val_grad_list,
                            PE, pe_grad_list, shift_qk, pd, CS, N_T, scale, use_numba=False,
                            freeze_pe=False):
    """Replay soft attention Score + Value + PE updates across all batch elements."""
    ED = ValueS.shape[2]
    if use_numba:
        for jV, jPE, val_j, sg, vg in zip(jV_list, jPE_list, val_j_list, score_grad_list, val_grad_list):
            _soft_attn_value_s_replay_kernel(ValueS, val_j, vg, CS, N_T, ED, scale)
            _soft_attn_score_s_replay_kernel(ScoreS, jV, jPE, sg, shift_qk, pd, CS, N_T, scale)
    else:
        for jV, jPE, val_j, sg, vg in zip(jV_list, jPE_list, val_j_list, score_grad_list, val_grad_list):
            # Value LUT update: each tree gets the full val_grad
            for pos1 in range(CS):
                ValueS[val_trees, val_j[pos1]] -= scale * vg[pos1]
            # Score LUT update (sequential to handle repeated j_bin collisions)
            for pos in range(1, CS):
                for pos1 in range(pos):
                    pe_pos = pos - pos1
                    scalar_sg = sg[pos, pos1]
                    for t in range(N_T):
                        j_bin = (int(jV[pos, t]) << shift_qk) | (int(jV[pos1, t]) << pd) | int(jPE[pe_pos, t])
                        ScoreS[t, j_bin, 0] -= scale * scalar_sg
    if not freeze_pe:
        for pe_grad in pe_grad_list:
            PE -= scale * pe_grad


def apply_averaged_gradients(m, batch_elements, lr, args, pool=None, update_stats=None):
    """Replay sparse S updates using saved y_grads and cached indices from batch elements.

    When pool is provided, replays are parallelized across independent S tables
    (6 FFN + 24 attention + 1-2 output = ~32 work units).
    """
    bs = len(batch_elements)
    scale = np.float32(lr / bs)
    CS = args.context_size
    N_T = args.n_t
    use_numba = HAS_NUMBA
    freeze_pe = getattr(args, 'freeze_PE', False)

    # Build work items for all independent S tables
    work = []

    for l in range(args.num_layers):
        # FFN: one task per layer
        lut = m.FFN[l]
        j_list = [be._ffn_j[l] for be in batch_elements]
        ygrad_list = [be.grad_accum.ffn_y_grads[l] for be in batch_elements]
        if update_stats is not None:
            for j, ygrad in zip(j_list, ygrad_list):
                update_stats.record_simple(f'ffn_{l}', lut, j)
                update_stats.record_simple_coords(f'ffn_{l}', lut, ygrad)
        work.append(lambda _l=lut, _j=j_list, _y=ygrad_list:
                    _replay_ffn_task(_l.S, _l.trees, _j, _y, _l.y_dim,
                                     CS, N_T, scale, use_numba))

        # Attention: one task per (layer, head)
        for h in range(args.num_heads):
            head = m.head[l][h]
            pe_grad_list = [be.grad_accum.pe_grads[l][h] for be in batch_elements]
            if getattr(args, 'new_attention', False):
                q_j_list = [be.head[l][h]._q_j for be in batch_elements]
                k_j_list = [be.head[l][h]._k_j for be in batch_elements]
                val_j_list = [be.head[l][h]._val_j for be in batch_elements]
                final_v_j_list = [be.head[l][h]._final_v_j for be in batch_elements]
                qkpe_j_list = [be.head[l][h]._qkpe_j for be in batch_elements]
                ygrad_list = [be.grad_accum.attn_y_grads[l] for be in batch_elements]
                q_grad_list = [be.grad_accum.new_attn_q_grads[l][h] for be in batch_elements]
                k_grad_list = [be.grad_accum.new_attn_k_grads[l][h] for be in batch_elements]
                v_grad_list = [be.grad_accum.new_attn_v_grads[l][h] for be in batch_elements]
                if update_stats is not None:
                    for q_j, k_j, val_j, final_v_j, qkpe_j, ygrad, q_grad, k_grad, v_grad in zip(
                            q_j_list, k_j_list, val_j_list, final_v_j_list, qkpe_j_list,
                            ygrad_list, q_grad_list, k_grad_list, v_grad_list):
                        if not getattr(args, 'PE_only', False):
                            update_stats.record_simple(f'head_{l}_{h}_Q', head.Q, q_j)
                            update_stats.record_simple_coords(f'head_{l}_{h}_Q', head.Q, q_grad)
                            update_stats.record_simple(f'head_{l}_{h}_K', head.K, k_j)
                            update_stats.record_simple_coords(f'head_{l}_{h}_K', head.K, k_grad)
                        update_stats.record_simple(f'head_{l}_{h}_V', head.V, val_j)
                        update_stats.record_simple_coords(f'head_{l}_{h}_V', head.V, v_grad)
                        update_stats.record_new_attention_final(
                            f'head_{l}_{h}_Final', head.Final, final_v_j, qkpe_j, args)
                        update_stats.record_new_attention_final_coords(
                            f'head_{l}_{h}_Final', head.Final, ygrad, args)
                work.append(lambda _head=head, _qj=q_j_list, _kj=k_j_list, _vj=val_j_list,
                                   _fvj=final_v_j_list, _qkpj=qkpe_j_list, _y=ygrad_list,
                                   _qg=q_grad_list, _kg=k_grad_list, _vg=v_grad_list,
                                   _pg=pe_grad_list:
                            _replay_new_attn_task(_head, _qj, _kj, _vj, _fvj, _qkpj, _y,
                                                  _qg, _kg, _vg, _pg,
                                                  CS, N_T, args.attn_qk_n_t, scale, use_numba,
                                                  getattr(args, 'PE_only', False),
                                                  freeze_pe))
            elif getattr(args, 'soft_attention', False):
                jV_list  = [be.head[l][h]._v_j  for be in batch_elements]
                jPE_list = [be.head[l][h]._pe_j for be in batch_elements]
                val_j_list      = [be.head[l][h]._val_j             for be in batch_elements]
                score_grad_list = [be.grad_accum.attn_score_grads[l][h] for be in batch_elements]
                val_grad_list   = [be.grad_accum.attn_val_grads[l][h]   for be in batch_elements]
                if update_stats is not None:
                    for val_j, jV, jPE, val_grad, score_grad in zip(
                            val_j_list, jV_list, jPE_list, val_grad_list, score_grad_list):
                        update_stats.record_simple(f'head_{l}_{h}_Value', head.Value, val_j)
                        update_stats.record_simple_coords(f'head_{l}_{h}_Value', head.Value, val_grad)
                        record_attention_update_rows(update_stats, f'head_{l}_{h}_Score',
                                                     head.Score, jV, jPE, args)
                        update_stats.record_attention_scalar_coords(
                            f'head_{l}_{h}_Score', head.Score, score_grad, args)
                work.append(lambda _SS=head.Score.S, _VS=head.Value.S,
                                   _st=head.Score.trees, _vt=head.Value.trees,
                                   _jV=jV_list, _jPE=jPE_list, _vj=val_j_list,
                                   _sg=score_grad_list, _vg=val_grad_list,
                                   _PE=head.Positional_encoding, _pg=pe_grad_list:
                            _replay_soft_attn_task(_SS, _VS, _st, _vt,
                                                   _jV, _jPE, _vj, _sg, _vg,
                                                   _PE, _pg,
                                                   args.shift_qk, args.positional_dim,
                                                   CS, N_T, scale, use_numba, freeze_pe))
            else:
                jV_list  = [be.head[l][h]._v_j  for be in batch_elements]
                jPE_list = [be.head[l][h]._pe_j for be in batch_elements]
                lut_v = head.V
                ygrad_list = [be.grad_accum.attn_y_grads[l] for be in batch_elements]
                if update_stats is not None:
                    for jV, jPE, ygrad in zip(jV_list, jPE_list, ygrad_list):
                        record_attention_update_rows(update_stats, f'head_{l}_{h}_V',
                                                     lut_v, jV, jPE, args)
                        update_stats.record_attention_coords(f'head_{l}_{h}_V', lut_v, ygrad, args)
                work.append(lambda _S=lut_v.S, _jV=jV_list, _jPE=jPE_list,
                                   _y=ygrad_list, _PE=head.Positional_encoding,
                                   _pg=pe_grad_list, _yd=lut_v.y_dim:
                            _replay_attn_task(_S, _jV, _jPE, _y, _PE, _pg,
                                              args.shift_qk, args.positional_dim,
                                              CS, N_T, _yd, scale, use_numba, freeze_pe))

    # Output head: one task per LUT
    if args.factored_output:
        for lut, j_attr, out_attr in [(m.output_head.unembedder_hi, '_hi_j', 'output_hi'),
                                      (m.output_head.unembedder_lo, '_lo_j', 'output_lo')]:
            j_list = [getattr(be.output_head, j_attr) for be in batch_elements]
            out_list = [getattr(be.output_head, out_attr) for be in batch_elements]
            if update_stats is not None:
                for j, output in zip(j_list, out_list):
                    update_stats.record_simple(f'output_{j_attr[1:-2]}', lut, j)
                    update_stats.record_simple_coords(f'output_{j_attr[1:-2]}', lut, output)
            work.append(lambda _l=lut, _j=j_list, _o=out_list:
                        _replay_output_task(_l.S, _l.trees, _j, _o, _l.y_dim,
                                            CS, N_T, scale, use_numba))
    else:
        lut = m.output_head.unembedder
        j_list = [be.output_head._j for be in batch_elements]
        out_list = [be.output_head.output for be in batch_elements]
        if update_stats is not None:
            for j, output in zip(j_list, out_list):
                update_stats.record_simple('output', lut, j)
                update_stats.record_simple_coords('output', lut, output)
        work.append(lambda _l=lut, _j=j_list, _o=out_list:
                    _replay_output_task(_l.S, _l.trees, _j, _o, _l.y_dim,
                                        CS, N_T, scale, use_numba))

    # Dispatch: parallel if pool provided, sequential otherwise
    if pool is not None:
        pool.map(lambda f: f(), work)
    else:
        for f in work:
            f()

    # LayerNorm parameter updates (averaged over batch)
    if getattr(args, 'layernorm', False):
        ln_scale = lr / len(batch_elements)
        for l in range(args.num_layers):
            m.ln_attn_gamma[l] -= ln_scale * sum(be.grad_accum.ln_attn_dgamma[l] for be in batch_elements)
            m.ln_attn_beta[l]  -= ln_scale * sum(be.grad_accum.ln_attn_dbeta[l]  for be in batch_elements)
            m.ln_ffn_gamma[l]  -= ln_scale * sum(be.grad_accum.ln_ffn_dgamma[l]  for be in batch_elements)
            m.ln_ffn_beta[l]   -= ln_scale * sum(be.grad_accum.ln_ffn_dbeta[l]   for be in batch_elements)


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def token_stream_hash(token_ids):
    tokens = np.asarray(token_ids, dtype=np.int32)
    return hashlib.sha256(tokens.tobytes()).hexdigest()[:16]


def pos_cache_path(input_path, token_ids, args):
    token_hash = token_stream_hash(token_ids)
    vocab_id = os.path.basename(args.vocab_file) if args.tokenizer == 'word' else args.tokenizer
    cache_name = f"{os.path.basename(input_path)}.{vocab_id}.tokenized.{token_hash}.pos.npy"
    return os.path.join(args.pos_cache_dir, cache_name)


def load_or_build_pos_encoding(input_path, token_ids, args, name):
    os.makedirs(args.pos_cache_dir, exist_ok=True)
    cache_path = pos_cache_path(input_path, token_ids, args)
    token_count = len(token_ids)

    if os.path.exists(cache_path):
        pos_data = np.load(cache_path)
        if len(pos_data) == token_count:
            print(f"Loaded POS encoding for {name}: {cache_path}")
            return pos_data.astype(np.int32, copy=False)
        print(
            f"Ignoring stale POS cache for {name}: expected {token_count} labels, "
            f"found {len(pos_data)}"
        )

    print(f"Building POS encoding for {name} from tokenized input...")
    pos_data = np.array(args.enc.encode_pos_ids(token_ids), dtype=np.int32)
    if len(pos_data) != token_count:
        print(
            f"POS encoding length mismatch for {name}: "
            f"{len(pos_data)} POS labels vs {token_count} tokens"
        )
        sys.exit(1)
    np.save(cache_path, pos_data)
    print(f"Saved POS encoding for {name}: {cache_path}")
    return pos_data


def load_training_data(training, args):
    
    TESTING_LENGTH = args.testing_length
    enc = args.enc

    # Load and tokenize training data
    try:
        with open(args.training_data, 'r', encoding='utf-8') as f:
            text = f.read()
    except IOError:
        print(f"Error opening training datafile {args.training_data}")
        sys.exit(1)

    tokens = enc.encode(text)
    training.data = np.array(tokens, dtype=np.int32)
    if args.POS_task:
        training.pos_data = load_or_build_pos_encoding(
            args.training_data, training.data, args, "training data")
    training.length = len(training.data) - args.context_size - 1
    print(f"Training data: {len(training.data)} tokens from {args.training_data}")

    # Load or split validation data
    if args.validation_data:
        try:
            with open(args.validation_data, 'r', encoding='utf-8') as f:
                val_text = f.read()
        except IOError:
            print(f"Error opening validation datafile {args.validation_data}")
            sys.exit(1)
        val_tokens = enc.encode(val_text)
        training.val_data = np.array(val_tokens, dtype=np.int32)
        if args.POS_task:
            training.val_pos_data = load_or_build_pos_encoding(
                args.validation_data, training.val_data, args, "validation data")
        training.val_length = len(training.val_data) - args.context_size - 1
        print(f"Validation data: {len(training.val_data)} tokens from {args.validation_data}")
    else:
        # No separate validation file: use random snippets from training data
        training.val_data = training.data
        training.val_pos_data = training.pos_data
        training.val_length = training.length
        print("Validation data: sampled from training data")

    # Sample validation snippet indices from val data
    training.testing_input_data = np.zeros(TESTING_LENGTH, dtype=np.int32)
    for i in range(TESTING_LENGTH):
        training.testing_input_data[i] = random.randint(0, training.val_length - 1)

    print(f"Tokenizer: {args.tokenizer}, vocab_size: {args.vocab_size_input}")


def get_random_training_index(training):
    return random.randint(0, training.length - 1)


def load_snippet(m, data_array, char_start, args, target_array=None):
    CS = args.context_size
    m.tokens[:CS + 1] = data_array[char_start:char_start + CS + 1]
    m.z[:] = m.Token_embedder[m.tokens[:CS]]
    if target_array is None:
        target_array = m.tokens
    else:
        target_array = target_array[char_start:char_start + CS + 1]
    if args.POS_task:
        target_array = np.array(target_array, copy=True)
        target_array[m.tokens[:CS + 1] == UNK_TOKEN_ID] = IGNORE_INDEX
    m.output_head.load_targets(target_array, CS)


def model_inference(m, args):
    model_forward(m, args, training=False)
    return m.output_head.sample_token(args)


def model_pos_response(m, prompt_text, response_length, args,
                       data_array=None, start_idx=None, target_array=None):
    enc = args.enc
    sys.stdout.write(prompt_text[:80])

    if data_array is not None and start_idx is not None and target_array is not None:
        max_response = min(response_length, args.context_size, len(target_array) - start_idx)
        load_snippet(m, data_array, start_idx, args, target_array)
        model_forward(m, args, training=False)
        preds = np.argmax(m.output_head.output[:max_response], axis=1).astype(int).tolist()
        targets = [int(t) for t in m.output_head.tokens[:max_response]]
        target_labels = [
            POS_LABELS[t] if 0 <= t < len(POS_LABELS) else "IGN"
            for t in targets
        ]
        sys.stdout.write("\nPOS pred:   " + enc.decode_pos(preds))
        sys.stdout.write("\nPOS target: " + " ".join(target_labels))
        sys.stdout.flush()
        return

    prompt_tokens = enc.encode(prompt_text)
    if len(prompt_tokens) > args.context_size:
        prompt_tokens = prompt_tokens[:args.context_size]
    else:
        prompt_tokens = [0] * (args.context_size - len(prompt_tokens)) + list(prompt_tokens)
    m.tokens[:args.context_size] = prompt_tokens
    m.z[:] = m.Token_embedder[prompt_tokens]
    model_forward(m, args, training=False)
    pred = int(np.argmax(m.output_head.output[args.context_size - 1]))
    sys.stdout.write("\nnext POS: " + enc.decode_pos([pred]))
    sys.stdout.flush()


def model_text_response(m, prompt_text, response_length, args):
    enc = args.enc
    prompt_tokens = enc.encode(prompt_text)
    if len(prompt_tokens) > args.context_size:
        prompt_tokens = prompt_tokens[:args.context_size]
    else:
        prompt_tokens = [0] * (args.context_size - len(prompt_tokens)) + list(prompt_tokens)

    sys.stdout.write(prompt_text[:80])

    for i in range(response_length):
        m.z[:] = m.Token_embedder[prompt_tokens]
        response = model_inference(m, args)
        sys.stdout.write(enc.decode([response]))

        # Shift context window and append new token
        prompt_tokens = prompt_tokens[1:] + [response]

    sys.stdout.flush()


def model_prompt_response(m, prompt_text, response_length, args,
                          data_array=None, start_idx=None, target_array=None):
    if args.POS_task:
        model_pos_response(m, prompt_text, response_length, args,
                           data_array=data_array,
                           start_idx=start_idx,
                           target_array=target_array)
    else:
        model_text_response(m, prompt_text, response_length, args)


def print_model_stats(m, args):
    """Print model parameter count and memory footprint."""
    total_params = 0
    total_bytes = 0

    def count_array(arr):
        nonlocal total_params, total_bytes
        total_params += arr.size
        total_bytes += arr.nbytes

    # Token embedder
    count_array(m.Token_embedder)

    # Per-layer
    for l in range(args.num_layers):
        # FFN LUT
        count_array(m.FFN[l].S)
        # Attention heads
        for h in range(args.num_heads):
            head = m.head[l][h]
            if getattr(args, 'soft_attention', False):
                count_array(head.Score.S)
                count_array(head.Value.S)
            elif getattr(args, 'new_attention', False):
                count_array(head.Q.S)
                count_array(head.K.S)
                count_array(head.V.S)
                count_array(head.Final.S)
            else:
                count_array(head.V.S)
            count_array(head.Positional_encoding)

    # Output head
    if args.factored_output:
        count_array(m.output_head.unembedder_hi.S)
        count_array(m.output_head.unembedder_lo.S)
    else:
        count_array(m.output_head.unembedder.S)

    # LayerNorm parameters
    if getattr(args, 'layernorm', False):
        for l in range(args.num_layers):
            count_array(m.ln_attn_gamma[l]); count_array(m.ln_attn_beta[l])
            count_array(m.ln_ffn_gamma[l]);  count_array(m.ln_ffn_beta[l])

    if total_bytes < 1024**2:
        mem_str = f"{total_bytes / 1024:.1f} KB"
    elif total_bytes < 1024**3:
        mem_str = f"{total_bytes / 1024**2:.1f} MB"
    else:
        mem_str = f"{total_bytes / 1024**3:.2f} GB"

    if total_params < 1_000_000:
        param_str = f"{total_params:,}"
    else:
        param_str = f"{total_params / 1_000_000:.1f}M"

    dtype_str = "fp16" if getattr(args, 'fp16', False) else "fp32"
    print(f"Model: {param_str} parameters, {mem_str} memory ({dtype_str})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global learning_rate

    parser = argparse.ArgumentParser(description="SNN Transformer (Python port)")
    parser.add_argument('training_data', type=str, help='Path to training data file')
    parser.add_argument('--validation-data', type=str, default=None,
                        help='Path to validation data file (if not provided, samples from training data)')
    parser.add_argument('--tokenizer', type=str, default='word',
                        help='tiktoken encoding name, or "word" for word-based tokenizer (default: word)')
    parser.add_argument('--vocab-file', type=str, default='word_vocab.txt',
                        help='Vocabulary file for --tokenizer word (built by build_word_vocab.py)')
    parser.add_argument('--context-size', type=int, default=128)
    parser.add_argument('--vocab-size', type=int, default=256,
                        help='(auto-detected from tokenizer, this default is overridden)')
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--positional-dim', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--n-t', type=int, default=16)
    parser.add_argument('--n-c', type=int, default=6)
    parser.add_argument('--weight-sharing', type=int, default=None,
                        help='With --new-attention, use N physical Final LUT trees shared by logical --n-t comparator slots')
    parser.add_argument('--lut-comparison-mode', choices=['random', 'no-overlap', 'overlap'],
                        default='random',
                        help='How LUT a/b comparison indices are assigned (default: random)')
    parser.add_argument('--attn_qk_dim', type=int, default=8)
    parser.add_argument('--attn_qk_n_c', type=int, default=3)
    parser.add_argument('--attn_qk_n_t', type=int, default=4)
    parser.add_argument('--testing-length', type=int, default=10000)
    parser.add_argument('--max-steps', type=int, default=100000000)
    parser.add_argument('--validation-interval', type=int, default=10000)
    parser.add_argument('--temperature', type=float, default=0.4)
    parser.add_argument('--loss-file', type=str, default='loss.csv')
    parser.add_argument('--factored-output', action='store_true',
                        help='Use factored base-256 decomposition for unembedder (faster for large vocabs)')
    parser.add_argument('--fp16', action='store_true',
                        help='Use float16 for LUT S tables (halves memory, may reduce precision)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Number of sequences per training step (default: 1)')
    parser.add_argument('--cpu-cores', type=int, default=1,
                        help='CPU cores allocated to this run, e.g. $SLURM_CPUS_PER_TASK (default: 1)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Batch worker threads for --batch-size > 1 (0 = derive from --cpu-cores)')
    parser.add_argument('--numba-threads', type=int, default=0,
                        help='Numba threads per worker/kernel (0 = derive from --cpu-cores)')
    parser.add_argument('--replay-workers', type=int, default=0,
                        help='Worker threads for batched sparse-gradient replay (0 = derive from --cpu-cores)')
    parser.add_argument('--soft-attention', action='store_true',
                        help='Decomposed Score+Value LUTs with softmax weighting (transformer-style)')
    parser.add_argument('--layernorm', action='store_true',
                        help='Pre-norm LayerNorm before attention and FFN (off by default)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate applied to residual deltas (0.0 = disabled)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for Python and NumPy RNGs (default: no seeding)')
    parser.add_argument('--save-model', type=str, default=None,
                        help='Path to save model weights (.npz) at each validation interval and end of training')
    parser.add_argument('--save-multiple-checkpoints', action='store_true',
                        help='With --save-model, keep initial and step-tagged validation checkpoints instead of overwriting the same file')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to load model weights (.npz) before training begins')
    parser.add_argument('--lut-update-stats-file', type=str, default=None,
                        help='Optional .npz path for per-LUT table-row update counts')
    parser.add_argument('--lut-update-stats-snapshot-steps', type=str, default=None,
                        help='Comma-separated 1-based training steps at which to save step-tagged LUT update stats snapshots')
    parser.add_argument('--wandb', action='store_true',
                        help='Log training and validation metrics to Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='lut',
                        help='Weights & Biases project name (default: lut)')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Optional Weights & Biases run name')
    parser.add_argument('--wandb-group', type=str, default=None,
                        help='Optional Weights & Biases group name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='Optional Weights & Biases entity/team')
    parser.add_argument('--wandb-tags', nargs='*', default=None,
                        help='Optional space-separated Weights & Biases tags')
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online',
                        help='Weights & Biases mode (default: online)')
    parser.add_argument('--wandb-log-interval', type=int, default=100,
                        help='Training metric logging interval in steps (default: 1)')
    parser.add_argument('--POS-task', '--pos-task', dest='POS_task', action='store_true',
                        help='Whether to run the POS tagging task instead of language modeling')
    parser.add_argument('--pos-cache-dir', type=str, default='.pos_cache',
                        help='Directory for cached POS encodings used by --POS-task')
    parser.add_argument('--new-attention', action='store_true',
                        help='Use new attention implementation (default: off)')
    parser.add_argument('--PE-only', dest='PE_only', action='store_true',
                        help='With --new-attention, zero Q/K embeddings and train PE+V+Final only')
    parser.add_argument('--freeze-PE', dest='freeze_PE', action='store_true',
                        help='Treat positional encodings as frozen constants after initialization/load')
    parser.add_argument('--PE-init-scale', dest='PE_init_scale', type=float, default=1.0,
                        help='Scale for frozen PE random magnitudes sampled from [0, scale]')
    parser.add_argument('--qkpe-direct-sign', action='store_true',
                        help='With --new-attention, build qkpe bins from qkpe[c] > 0 instead of qkpe[a] > qkpe[b]')
    parser.add_argument('--normalize-qk', action='store_true',
                        help='With --new-attention, normalize Q+K to PE init statistics before summing with raw PE')
    parser.add_argument('--qk-norm-eps', type=float, default=1e-6,
                        help='Epsilon for --normalize-qk vector variance normalization')
    args = parser.parse_args()
    if args.PE_only and not args.new_attention:
        parser.error('--PE-only requires --new-attention')
    if args.qkpe_direct_sign and not args.new_attention:
        parser.error('--qkpe-direct-sign requires --new-attention')
    if args.qkpe_direct_sign and args.positional_dim > args.attn_qk_dim:
        parser.error('--qkpe-direct-sign requires --positional-dim <= --attn_qk_dim')
    if args.normalize_qk and not args.new_attention:
        parser.error('--normalize-qk requires --new-attention')
    if args.normalize_qk and args.qkpe_direct_sign and args.positional_dim < 2:
        parser.error('--normalize-qk with --qkpe-direct-sign requires --positional-dim >= 2')
    if args.normalize_qk and not args.qkpe_direct_sign and args.attn_qk_dim < 2:
        parser.error('--normalize-qk requires --attn_qk_dim >= 2')
    if args.PE_init_scale < 0.0:
        parser.error('--PE-init-scale must be non-negative')
    if args.qk_norm_eps <= 0.0:
        parser.error('--qk-norm-eps must be positive')
    pe_stat_scale = args.PE_init_scale if args.freeze_PE else 1.0
    args.qk_norm_target_std = np.float32(pe_stat_scale / math.sqrt(3.0))
    if args.wandb_log_interval < 1:
        parser.error('--wandb-log-interval must be at least 1')
    configure_lut_comparison_mode(args, parser)
    configure_weight_sharing(args, parser)

    # Seed RNGs before anything else
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.POS_task and args.tokenizer != 'word':
        print("POS task currently requires --tokenizer word.")
        sys.exit(1)

    # Initialize tokenizer and auto-set vocab size
    if args.tokenizer == 'word':
        enc = WordTokenizer(args.vocab_file)
    else:
        enc = tiktoken.get_encoding(args.tokenizer)
    args.vocab_size_input = enc.n_vocab
    args.vocab_size_output = len(POS_LABELS) if args.POS_task else enc.n_vocab
    args.enc = enc
    if args.POS_task:
        if args.factored_output:
            print("POS task uses the standard output head; disable --factored-output.")
            sys.exit(1)

    # Factored output dimensions
    args.vocab_hi = (args.vocab_size_output + 255) // 256
    args.vocab_lo = 256

    # Pre-compute positional encoding bitmasks and shift constants
    args.pe_bitmasks = (1 << np.arange(args.positional_dim)).astype(np.int32)
    args.shift_qk = args.n_c + args.positional_dim

    configure_cpu_parallelism(args, parser)

    # Print hyperparameters
    print("=" * 50)
    print("Hyperparameters:")
    print(f"  context_size      = {args.context_size}")
    print(f"  vocab_size_input  = {args.vocab_size_input}")
    print(f"  vocab_size_output = {args.vocab_size_output}")
    print(f"  embedding_dim     = {args.embedding_dim}")
    print(f"  positional_dim    = {args.positional_dim}")
    print(f"  num_layers        = {args.num_layers}")
    print(f"  num_heads         = {args.num_heads}")
    print(f"  n_t               = {args.n_t}")
    print(f"  n_c               = {args.n_c}")
    print(f"  weight_sharing    = {args.weight_sharing}")
    print(f"  final_n_t         = {args.final_n_t}")
    print(f"  final_share_m     = {args.final_share_m}")
    print(f"  lut_comparison_mode = {args.lut_comparison_mode}")
    print(f"  attn_qk_dim       = {args.attn_qk_dim}")
    print(f"  attn_qk_n_c       = {args.attn_qk_n_c}")
    print(f"  attn_qk_n_t       = {args.attn_qk_n_t}")
    print(f"  temperature       = {args.temperature}")
    print(f"  factored_output   = {args.factored_output}")
    print(f"  fp16              = {args.fp16}")
    print(f"  batch_size        = {args.batch_size}")
    print(f"  cpu_cores         = {args.cpu_cores}")
    print(f"  num_workers       = {args.num_workers}")
    print(f"  replay_workers    = {args.replay_workers}")
    print(f"  numba_available   = {HAS_NUMBA}")
    print(f"  numba_threads     = {args.numba_threads}")
    print(f"  soft_attention    = {args.soft_attention}")
    print(f"  layernorm         = {args.layernorm}")
    print(f"  dropout           = {args.dropout}")
    print(f"  tokenizer         = {args.tokenizer}")
    print(f"  max_steps         = {args.max_steps}")
    print(f"  validation_interval = {args.validation_interval}")
    print(f"  testing_length    = {args.testing_length}")
    print(f"  training_data     = {args.training_data}")
    print(f"  validation_data   = {args.validation_data}")
    print(f"  loss_file         = {args.loss_file}")
    print(f"  seed              = {args.seed}")
    print(f"  save_model        = {args.save_model}")
    print(f"  save_multiple_checkpoints = {args.save_multiple_checkpoints}")
    print(f"  load_model        = {args.load_model}")
    print(f"  lut_update_stats_file = {args.lut_update_stats_file}")
    print(f"  wandb             = {args.wandb}")
    print(f"  wandb_project     = {args.wandb_project}")
    print(f"  wandb_run_name    = {args.wandb_run_name}")
    print(f"  wandb_group       = {args.wandb_group}")
    print(f"  wandb_entity      = {args.wandb_entity}")
    print(f"  wandb_tags        = {args.wandb_tags}")
    print(f"  wandb_mode        = {args.wandb_mode}")
    print(f"  wandb_log_interval = {args.wandb_log_interval}")
    print(f"  POS_task          = {args.POS_task}")
    print(f"  pos_cache_dir     = {args.pos_cache_dir}")
    print(f"  new_attention     = {args.new_attention}")
    print(f"  PE_only           = {args.PE_only}")
    print(f"  freeze_PE         = {args.freeze_PE}")
    print(f"  PE_init_scale     = {args.PE_init_scale}")
    print(f"  qkpe_direct_sign  = {args.qkpe_direct_sign}")
    print(f"  normalize_qk    = {args.normalize_qk}")
    print(f"  qk_norm_eps     = {args.qk_norm_eps}")
    print(f"  qk_norm_target_std = {float(args.qk_norm_target_std):.6f}")
    print("=" * 50)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError:
            print("Weights & Biases support requires installing wandb. Try: pip install wandb")
            sys.exit(1)

        wandb_config = {
            key: value
            for key, value in vars(args).items()
            if key != 'enc' and isinstance(value, (int, float, bool, str, type(None), list, tuple))
        }
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            group=args.wandb_group,
            entity=args.wandb_entity,
            tags=args.wandb_tags,
            mode=args.wandb_mode,
            config=wandb_config,
        )
        wandb.define_metric("step")
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("val/*", step_metric="step")

    # Initialize loss file with header
    with open(args.loss_file, 'w') as f:
        f.write("step, loss, perplexity, accuracy\n")

    training = TrainingData()
    load_training_data(training, args)

    m = Model(args)
    build_Model(m, args)
    if args.load_model is not None:
        load_model(m, args.load_model, args)
    print_model_stats(m, args)
    update_stats = LUTUpdateStats() if args.lut_update_stats_file is not None else None
    try:
        lut_update_snapshot_steps = parse_lut_update_snapshot_steps(
            args.lut_update_stats_snapshot_steps)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    batch_size = args.batch_size
    if batch_size > 1:
        batch_elements = [BatchElement(m, args) for _ in range(batch_size)]
        pool = ThreadPool(args.num_workers) if args.num_workers > 1 else None
        replay_pool = ThreadPool(args.replay_workers) if args.replay_workers > 1 else None
    else:
        batch_elements = None
        pool = None
        replay_pool = None

    if args.save_model is not None and args.save_multiple_checkpoints:
        saved_model_path = save_training_checkpoint(m, args, 0)
        print(f"Saved initial model checkpoint to {saved_model_path}")

    pbar = tqdm(range(args.max_steps), desc="Training", unit="step")
    last_ppl = None
    last_loss = None
    last_acc = None
    ema_loss = None
    ema_alpha = 0.02

    for t in pbar:
        # Adam learning rate scheduler
        learning_rate = min(1.0 / math.sqrt(1 + t), t / 4000.0 / math.sqrt(4000))

        if batch_size == 1:
            # Fast path: identical to original code (no overhead)
            load_snippet(m, training.data, get_random_training_index(training), args,
                         training.pos_data if args.POS_task else None)
            step_stats = update_stats if learning_rate != 0.0 else None
            loss = model_training_step(m, args, update_stats=step_stats)
            ema_loss = loss if ema_loss is None else ema_alpha * loss + (1 - ema_alpha) * ema_loss
        else:
            # Batched path: parallel forward+backward, then averaged S update
            for b in range(batch_size):
                idx = get_random_training_index(training)
                load_snippet(batch_elements[b], training.data, idx, args,
                             training.pos_data if args.POS_task else None)
                batch_elements[b].grad_accum.zero()

            if pool is not None:
                losses = pool.map(lambda be: _batch_element_step(be, args), batch_elements)
            else:
                losses = [_batch_element_step(be, args) for be in batch_elements]
            step_stats = update_stats if learning_rate != 0.0 else None
            apply_averaged_gradients(m, batch_elements, learning_rate, args,
                                     pool=replay_pool, update_stats=step_stats)
            loss = sum(losses) / batch_size
            ema_loss = loss if ema_loss is None else ema_alpha * loss + (1 - ema_alpha) * ema_loss

        train_ppl = math.exp(min(ema_loss, 20.0)) if ema_loss is not None else None
        if wandb_run is not None and t % args.wandb_log_interval == 0:
            wandb_run.log({
                "step": t,
                "train/loss": loss,
                "train/ema_loss": ema_loss,
                "train/perplexity": train_ppl,
                "train/learning_rate": learning_rate,
            })

        current_step = t + 1
        if update_stats is not None and current_step in lut_update_snapshot_steps:
            saved_stats = save_lut_update_stats(
                update_stats, args.lut_update_stats_file, current_step)
            tqdm.write(
                f"Saved LUT update stats snapshot step {current_step} to "
                f"{saved_stats['snapshot_path']}; base stats refreshed at "
                f"{saved_stats['stats_path']}")

        if t % args.validation_interval == 0:
            pbar.set_description("Validating")

            validation_loss = 0.0
            validation_accuracy = 0.0
            validation_count = 0
            for i in tqdm(range(args.testing_length), desc="  Val", leave=False, unit="snip"):
                load_snippet(m, training.val_data, int(training.testing_input_data[i]), args,
                             training.val_pos_data if args.POS_task else None)
                model_forward(m, args, training=False)
                sample_accuracy = m.output_head.validation_accuracy(args)
                sample_loss = m.output_head.validation_loss(args)
                if sample_loss is not None:
                    if args.POS_task:
                        loss_sum, loss_count = sample_loss
                        correct_count, accuracy_count = sample_accuracy
                        validation_accuracy += correct_count
                        validation_loss += loss_sum
                        validation_count += loss_count
                    else:
                        validation_accuracy += sample_accuracy
                        validation_loss += sample_loss
                        validation_count += 1
            if validation_count > 0:
                validation_loss /= validation_count
                validation_accuracy /= validation_count
            else:
                validation_loss = float('nan')
                validation_accuracy = float('nan')

            perplexity = math.exp(validation_loss)
            last_ppl = perplexity
            last_loss = validation_loss
            last_acc = validation_accuracy

            with open(args.loss_file, 'a') as f:
                f.write(f"{t}, {validation_loss:.6f}, {perplexity:.2f}, {validation_accuracy:.6f}\n")

            if args.save_model is not None:
                saved_model_path = save_training_checkpoint(m, args, current_step)
                tqdm.write(f"Saved model to {saved_model_path}")
            if update_stats is not None:
                saved_stats = save_lut_update_stats(update_stats, args.lut_update_stats_file)
                tqdm.write(
                    f"Saved LUT update stats to {saved_stats['stats_path']}, "
                    f"{saved_stats['summary_path']}, and {saved_stats['hist_path']}; "
                    f"j stats to {saved_stats['j_stats_path']}, "
                    f"{saved_stats['j_summary_path']}, and {saved_stats['j_top_path']}")

            # Print validation result and sample generation
            tqdm.write(
                f"\n--- step {t:,} | ppl={perplexity:.2f} | "
                f"loss={validation_loss:.3f} | acc={validation_accuracy:.3f} ---"
            )
            val_idx = random.randint(0, training.val_length - 1)
            prompt_tokens = training.val_data[val_idx:val_idx + args.context_size].tolist()
            prompt_text = enc.decode(prompt_tokens)
            # Capture generation / POS prediction output
            old_stdout = sys.stdout
            sys.stdout = buf = io.StringIO()
            model_prompt_response(
                m, prompt_text, 80, args,
                data_array=training.val_data if args.POS_task else None,
                start_idx=val_idx if args.POS_task else None,
                target_array=training.val_pos_data if args.POS_task else None,
            )
            sys.stdout = old_stdout
            sample_text = buf.getvalue()
            tqdm.write(sample_text)
            tqdm.write("")

            if wandb_run is not None:
                wandb_run.log({
                    "step": t,
                    "val/loss": validation_loss,
                    "val/perplexity": perplexity,
                    "val/accuracy": validation_accuracy,
                    "val/sample": sample_text,
                })

            pbar.set_description("Training")

        train_ppl_str = f"{train_ppl:.1f}" if train_ppl is not None else "-"
        if last_ppl is not None:
            pbar.set_postfix(train_ppl=train_ppl_str, val_ppl=f"{last_ppl:.2f}",
                             loss=f"{last_loss:.3f}", acc=f"{last_acc:.3f}",
                             lr=f"{learning_rate:.4f}")
        elif ema_loss is not None:
            pbar.set_postfix(train_ppl=train_ppl_str, lr=f"{learning_rate:.4f}")

    if args.save_model is not None:
        save_model(m, args.save_model, args)
        print(f"Saved final model to {model_save_display_path(args.save_model)}")
    if update_stats is not None:
        saved_stats = save_lut_update_stats(update_stats, args.lut_update_stats_file)
        print(
            f"Saved final LUT update stats to {saved_stats['stats_path']}, "
            f"{saved_stats['summary_path']}, and {saved_stats['hist_path']}; "
            f"j stats to {saved_stats['j_stats_path']}, "
            f"{saved_stats['j_summary_path']}, and {saved_stats['j_top_path']}")

    if pool is not None:
        pool.close()
        pool.join()
    if replay_pool is not None:
        replay_pool.close()
        replay_pool.join()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
