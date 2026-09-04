"""Model configuration: the structural flags, normalised and validated.

Mirrors the validation in ``main.main`` plus ``configure_new_mha`` /
``configure_weight_sharing`` / ``configure_lut_comparison_mode``, minus everything this
port dropped (fp16, factored output, soft/old attention, no-overlap and overlap
comparison layouts, the Numba/thread-pool knobs).

Keeping this separate from the trainer's argparse namespace means the model can be built
from a plain :class:`LutConfig` in tests without going through a CLI.
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

from .functional import (
    SELECTION_CLOSEST,
    SELECTION_MODES,
    TABLE_GRAD_AUTO,
    TABLE_GRAD_MODES,
)


class ConfigError(ValueError):
    """Raised for an invalid structural configuration."""


@dataclass
class LutConfig:
    # --- shape ---
    context_size: int = 32
    embedding_dim: int = 64
    positional_dim: int = 5
    num_layers: int = 6
    num_heads: int = 4
    vocab_size_input: int = 256
    vocab_size_output: int = 18

    # --- LUT capacity ---
    n_t: int = 64
    n_c: int = 6
    comparisons_per_bit: int = 1
    attn_qk_dim: int = 8
    attn_qk_n_c: int = 6
    attn_qk_n_t: int = 64

    # --- architecture ---
    new_MHA: bool = False
    weight_sharing: Optional[int] = None

    # --- PE / QK ablations ---
    PE_only: bool = False
    freeze_PE: bool = False
    PE_init_scale: float = 1.0
    qkpe_direct_sign: bool = False
    normalize_qk: bool = False
    qk_norm_eps: float = 1e-6

    # --- gradients / regularisation ---
    gradient_pair_selection: str = SELECTION_CLOSEST
    table_grad: str = TABLE_GRAD_AUTO
    layernorm: bool = False
    dropout: float = 0.0

    # --- task ---
    POS_task: bool = False

    # --- derived (filled by __post_init__) ---
    mha_head_dim: int = field(init=False, default=0)
    final_n_t: int = field(init=False, default=0)
    final_share_m: int = field(init=False, default=1)
    qk_norm_target_std: float = field(init=False, default=0.0)

    def __post_init__(self):
        self.validate()
        self.mha_head_dim = (self.embedding_dim // self.num_heads
                             if self.new_MHA else self.embedding_dim)
        if self.weight_sharing is None:
            self.final_n_t = self.n_t
            self.final_share_m = 1
        else:
            self.final_n_t = self.weight_sharing
            self.final_share_m = self.n_t // self.weight_sharing
        # Matches main.py: the frozen-PE magnitude scale drives the QK normalisation
        # target so normalised QK and raw PE stay on comparable scales.
        pe_stat_scale = self.PE_init_scale if self.freeze_PE else 1.0
        self.qk_norm_target_std = pe_stat_scale / math.sqrt(3.0)

    def validate(self):
        if self.n_c < 1:
            raise ConfigError('n_c must be at least 1')
        if self.attn_qk_n_c < 1:
            raise ConfigError('attn_qk_n_c must be at least 1')
        if self.comparisons_per_bit < 1:
            raise ConfigError('comparisons_per_bit must be at least 1')
        if self.gradient_pair_selection not in SELECTION_MODES:
            raise ConfigError(
                f'gradient_pair_selection must be one of {SELECTION_MODES}')
        if self.table_grad not in TABLE_GRAD_MODES:
            raise ConfigError(f'table_grad must be one of {TABLE_GRAD_MODES}')
        if self.new_MHA and self.embedding_dim % self.num_heads != 0:
            raise ConfigError('new-MHA requires embedding_dim divisible by num_heads')
        if self.weight_sharing is not None:
            if self.weight_sharing < 1:
                raise ConfigError('weight_sharing must be at least 1')
            if self.weight_sharing > self.n_t:
                raise ConfigError('weight_sharing must be no greater than n_t')
            if self.n_t % self.weight_sharing != 0:
                raise ConfigError('n_t must be divisible by weight_sharing')
        if self.qkpe_direct_sign and self.positional_dim > self.attn_qk_dim:
            raise ConfigError('qkpe_direct_sign requires positional_dim <= attn_qk_dim')
        if self.normalize_qk and self.qkpe_direct_sign and self.positional_dim < 2:
            raise ConfigError('normalize_qk with qkpe_direct_sign requires '
                              'positional_dim >= 2')
        if self.normalize_qk and not self.qkpe_direct_sign and self.attn_qk_dim < 2:
            raise ConfigError('normalize_qk requires attn_qk_dim >= 2')
        if self.PE_init_scale < 0.0:
            raise ConfigError('PE_init_scale must be non-negative')
        if self.qk_norm_eps <= 0.0:
            raise ConfigError('qk_norm_eps must be positive')
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError('dropout must be in [0, 1)')

    @property
    def head_value_dim(self):
        """Output width of one attention head's V and Final tables."""
        return self.mha_head_dim

    @property
    def qkpe_active_dim(self):
        """Number of QK/PE dimensions that actually feed the QKPE comparisons."""
        return self.positional_dim if self.qkpe_direct_sign else self.attn_qk_dim

    @property
    def final_total_n_c(self):
        """Row-index width of the Final table: V bits plus positional bits."""
        return self.n_c + self.positional_dim

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_namespace(cls, args):
        """Build from an argparse namespace, ignoring trainer-only fields."""
        names = {f for f in cls.__dataclass_fields__ if cls.__dataclass_fields__[f].init}
        return cls(**{k: v for k, v in vars(args).items() if k in names})
