"""Model modules: LUT tables, the new-attention head, and the full LUT model.

Construction deliberately follows ``main.build_Model``'s RNG draw order exactly -- Python's
``random`` for comparison pairs, NumPy's for the embedder and PE -- so that seeding both
implementations identically produces identical tables.  ``test_torch_parity.py`` depends on
this; see the note in :mod:`lut_torch.comparisons`.

Everything downstream of a margin tensor is plain torch, which is the point of the port:
QK normalisation, PE addition, direct-sign slicing and the Q/K/PE gradient routing that
``main.new_attention_backward`` writes out by hand all fall out of autograd here.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .comparisons import (
    build_random_comparison_arrays,
    causal_pair_indices,
    final_tree_map,
    initialize_positional_encoding,
    random_vector,
)
from .functional import (
    TABLE_GRAD_AUTO,
    LutGroup,
    lut_lookup,
    lut_lookup_reference,
)


def normalize_to_target(vec, target_std, eps):
    """Mirror of ``main._normalize_vector_to_target`` over the last dimension."""
    mean = vec.mean(dim=-1, keepdim=True)
    centered = vec - mean
    inv_std = torch.rsqrt((centered * centered).mean(dim=-1, keepdim=True) + eps)
    return target_std * centered * inv_std


class LUTTable(nn.Module):
    """One LUT: a table ``S``, its comparison pairs, and its logical-to-physical tree map.

    ``S`` is stored flat as ``(n_physical_trees * n_bins, y_dim)`` because that is the
    layout the lookup wants; :attr:`S` exposes the logical ``(trees, bins, y_dim)`` view
    that ``main.py`` uses.
    """

    def __init__(self, name, n_t, n_c, fan_in, total_n_c, y_dim, input_dim,
                 s_n_t=None, trees=None):
        super().__init__()
        self.name = name
        self.n_t = n_t
        self.n_c = n_c
        self.fan_in = fan_in
        self.total_n_c = total_n_c
        self.y_dim = y_dim
        self.n_bins = 1 << total_n_c
        self.s_n_t = n_t if s_n_t is None else s_n_t

        a_arr, b_arr = build_random_comparison_arrays(n_t, n_c * fan_in, input_dim)
        self.register_buffer('a_arr', torch.tensor(a_arr, dtype=torch.int64))
        self.register_buffer('b_arr', torch.tensor(b_arr, dtype=torch.int64))
        tree_map = np.arange(n_t, dtype=np.int32) if trees is None else np.asarray(trees)
        self.register_buffer('trees', torch.tensor(tree_map, dtype=torch.int64))
        self.S_flat = nn.Parameter(torch.zeros(self.s_n_t * self.n_bins, y_dim))

    @property
    def S(self):
        """Logical ``(physical_trees, bins, y_dim)`` view of the table."""
        return self.S_flat.view(self.s_n_t, self.n_bins, self.y_dim)

    def margins(self, x):
        """``u = x[a] - x[b]`` for every tree, shape ``(..., N_T, n_c * fan_in)``."""
        return x[..., self.a_arr] - x[..., self.b_arr]

    def group(self, bit_offset=0, pair_index=None):
        return LutGroup(n_bits=self.n_c, fan_in=self.fan_in,
                        bit_offset=bit_offset, pair_index=pair_index)

    def forward(self, x, selection, capture=None, table_grad=TABLE_GRAD_AUTO):
        entry = {} if capture is not None else None
        y = lut_lookup(self.S_flat, self.trees, self.n_bins, [self.group()],
                       [self.margins(x)], selection=selection,
                       table_grad=table_grad, capture=entry)
        if capture is not None:
            capture[self.name] = entry
        return y


class NewAttentionHead(nn.Module):
    """One ``--new-attention`` head: Q, K, V, Final and a positional encoding.

    Q/K produce ``attn_qk_dim`` vectors that are combined per causal pair and compared
    against (or added to) the PE to form the low bits of the Final row index; V produces
    the value vector whose comparisons form the high bits.  Only pairs with ``j < i``
    exist, so position 0 receives no attention output.
    """

    def __init__(self, cfg, name, frozen_pe=None):
        super().__init__()
        self.cfg = cfg
        self.name = name
        value_dim = cfg.head_value_dim

        # Draw order below mirrors main.build_Model: PE, then Q, K, V, Final, then the
        # QKPE comparison arrays (which are drawn even under --qkpe-direct-sign).
        if frozen_pe is not None:
            pe = np.array(frozen_pe, dtype=np.float32, copy=True)
        else:
            pe = random_vector(cfg.context_size * cfg.attn_qk_dim, 1.0).reshape(
                cfg.context_size, cfg.attn_qk_dim)
        self.positional_encoding = nn.Parameter(
            torch.tensor(pe, dtype=torch.float32), requires_grad=not cfg.freeze_PE)

        self.Q = LUTTable(f'{name}_Q', cfg.attn_qk_n_t, cfg.attn_qk_n_c,
                          cfg.comparisons_per_bit, cfg.attn_qk_n_c, cfg.attn_qk_dim,
                          cfg.embedding_dim)
        self.K = LUTTable(f'{name}_K', cfg.attn_qk_n_t, cfg.attn_qk_n_c,
                          cfg.comparisons_per_bit, cfg.attn_qk_n_c, cfg.attn_qk_dim,
                          cfg.embedding_dim)
        self.V = LUTTable(f'{name}_V', cfg.n_t, cfg.n_c, cfg.comparisons_per_bit,
                          cfg.n_c, value_dim, cfg.embedding_dim)
        self.Final = LUTTable(f'{name}_Final', cfg.n_t, cfg.n_c, cfg.comparisons_per_bit,
                              cfg.final_total_n_c, value_dim, value_dim,
                              s_n_t=cfg.final_n_t,
                              trees=final_tree_map(cfg.final_n_t, cfg.final_share_m))

        qkpe_a, qkpe_b = build_random_comparison_arrays(
            cfg.n_t, cfg.positional_dim, cfg.attn_qk_dim)
        self.register_buffer('qkpe_a', torch.tensor(qkpe_a, dtype=torch.int64))
        self.register_buffer('qkpe_b', torch.tensor(qkpe_b, dtype=torch.int64))

        tok_i, tok_j = causal_pair_indices(cfg.context_size)
        self.register_buffer('tok_i', torch.tensor(tok_i))
        self.register_buffer('tok_j', torch.tensor(tok_j))
        self.register_buffer('pe_pos', torch.tensor(tok_i - tok_j))

        # CSR inverse of tok_j (pair -> source position), for the fused-kernel backend's
        # per-source-position parallel backward (lut_torch.kernels._k_grad_paired). Built
        # once here rather than per call; unused by the torch backward path.
        tok_j_t = self.tok_j
        order = torch.argsort(tok_j_t, stable=True)
        counts = torch.bincount(tok_j_t, minlength=cfg.context_size)
        pair_ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        self.register_buffer('final_pair_ptr', pair_ptr)
        self.register_buffer('final_pair_list', order)

    def qkpe_margins(self, x, selection, capture):
        """Margins for the positional/low bits of the Final row index."""
        cfg = self.cfg
        active = cfg.qkpe_active_dim
        pe = self.positional_encoding.index_select(0, self.pe_pos)[..., :active]

        if cfg.PE_only:
            combined = pe.unsqueeze(0).expand(x.shape[0], -1, -1)
        else:
            q = self.Q(x, selection, capture, cfg.table_grad)
            k = self.K(x, selection, capture, cfg.table_grad)
            qk = q.index_select(1, self.tok_i) + k.index_select(1, self.tok_j)
            qk = qk[..., :active]
            if cfg.normalize_qk:
                qk = normalize_to_target(qk, cfg.qk_norm_target_std, cfg.qk_norm_eps)
            combined = qk + pe

        if cfg.qkpe_direct_sign:
            # Bit c is just sign(combined[c]); the margin is shared by every tree, so
            # keep the tree axis at size 1 and let the lookup broadcast it.
            return combined[..., :cfg.positional_dim].unsqueeze(-2)
        return combined[..., self.qkpe_a] - combined[..., self.qkpe_b]

    def forward(self, x, selection, capture=None):
        cfg = self.cfg
        v = self.V(x, selection, capture, cfg.table_grad)
        qkpe = self.qkpe_margins(x, selection, capture)
        v_margins = self.Final.margins(v)

        groups = [
            LutGroup(n_bits=cfg.positional_dim, fan_in=1, bit_offset=0),
            LutGroup(n_bits=self.Final.n_c, fan_in=self.Final.fan_in,
                    bit_offset=cfg.positional_dim, pair_index=self.tok_j,
                    pair_ptr=self.final_pair_ptr, pair_list=self.final_pair_list),
        ]
        entry = {} if capture is not None else None
        y_pairs = lut_lookup(self.Final.S_flat, self.Final.trees, self.Final.n_bins,
                             groups, [qkpe, v_margins], selection=selection,
                             table_grad=cfg.table_grad, capture=entry)
        if capture is not None:
            capture[f'{self.name}_Final'] = entry

        out = x.new_zeros(x.shape[0], cfg.context_size, cfg.head_value_dim)
        return out.index_add_(1, self.tok_i, y_pairs)


class LutLayer(nn.Module):
    """Pre-norm attention block plus pre-norm FFN block, both residual."""

    def __init__(self, cfg, index, frozen_pe=None):
        super().__init__()
        self.cfg = cfg
        ed = cfg.embedding_dim

        self.ffn = LUTTable(f'ffn_{index}', cfg.n_t, cfg.n_c, cfg.comparisons_per_bit,
                            cfg.n_c, ed, ed)
        if cfg.new_MHA:
            self.mha_o = LUTTable(f'mha_o_{index}', cfg.n_t, cfg.n_c,
                                  cfg.comparisons_per_bit, cfg.n_c, ed,
                                  cfg.num_heads * cfg.head_value_dim)
        else:
            self.mha_o = None
        self.heads = nn.ModuleList([
            NewAttentionHead(cfg, f'head_{index}_{h}', frozen_pe)
            for h in range(cfg.num_heads)
        ])

        self.ln_attn_gamma = nn.Parameter(torch.ones(ed))
        self.ln_attn_beta = nn.Parameter(torch.zeros(ed))
        self.ln_ffn_gamma = nn.Parameter(torch.ones(ed))
        self.ln_ffn_beta = nn.Parameter(torch.zeros(ed))

    def _residual_dropout(self, z_pre, z):
        if self.cfg.dropout > 0.0 and self.training:
            return z_pre + F.dropout(z - z_pre, p=self.cfg.dropout, training=True)
        return z

    def forward(self, z, selection, capture=None):
        cfg = self.cfg
        ed = cfg.embedding_dim

        x = (F.layer_norm(z, (ed,), self.ln_attn_gamma, self.ln_attn_beta, eps=1e-5)
             if cfg.layernorm else z)
        z_pre = z
        if cfg.new_MHA:
            concat = torch.cat([h(x, selection, capture) for h in self.heads], dim=-1)
            z = z + self.mha_o(concat, selection, capture, cfg.table_grad)
        else:
            for head in self.heads:
                z = z + head(x, selection, capture)
        z = self._residual_dropout(z_pre, z)

        ffn_in = (F.layer_norm(z, (ed,), self.ln_ffn_gamma, self.ln_ffn_beta, eps=1e-5)
                  if cfg.layernorm else z)
        z_pre = z
        z = z + self.ffn(ffn_in, selection, capture, cfg.table_grad)
        return self._residual_dropout(z_pre, z)


class LutModel(nn.Module):
    """The full LUT model: frozen token embedding, N layers, and a LUT output head.

    The token embedding is a buffer rather than a parameter because ``main.py`` never
    updates it (the embedder update is commented out in ``model_backward``).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        embedder = random_vector(cfg.vocab_size_input * cfg.embedding_dim, 1.0).reshape(
            cfg.vocab_size_input, cfg.embedding_dim)
        self.register_buffer('token_embedding', torch.tensor(embedder,
                                                             dtype=torch.float32))

        frozen_pe = None
        if cfg.freeze_PE:
            frozen_pe = initialize_positional_encoding(
                (cfg.context_size, cfg.attn_qk_dim), cfg.PE_init_scale,
                cfg.positional_dim)

        self.layers = nn.ModuleList([
            LutLayer(cfg, l, frozen_pe) for l in range(cfg.num_layers)
        ])
        self.output_head = LUTTable('output', cfg.n_t, cfg.n_c, cfg.comparisons_per_bit,
                                    cfg.n_c, cfg.vocab_size_output, cfg.embedding_dim)

    def forward(self, tokens, capture=None):
        """``tokens``: ``(B, context_size)`` int64. Returns logits ``(B, CS, vocab)``."""
        selection = self.cfg.gradient_pair_selection
        z = self.token_embedding[tokens]
        for layer in self.layers:
            z = layer(z, selection, capture)
        return self.output_head(z, selection, capture, self.cfg.table_grad)

    def table_parameters(self):
        """The sparse-gradient table parameters (``S``), for optimizer grouping."""
        return [m.S_flat for m in self.modules() if isinstance(m, LUTTable)]

    def dense_parameters(self):
        """Everything else that trains: positional encodings and LayerNorm affines."""
        table_ids = {id(p) for p in self.table_parameters()}
        return [p for p in self.parameters()
                if p.requires_grad and id(p) not in table_ids]
