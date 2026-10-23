"""Model configuration: the structural flags, normalised and validated.

Mirrors the validation in ``main.main`` plus ``configure_new_mha`` /
``configure_weight_sharing`` / ``configure_lut_comparison_mode``, minus everything this
port dropped (fp16, factored output, soft attention, no-overlap and overlap comparison
layouts, the Numba/thread-pool knobs). ``main.py``'s original hard attention is available
as ``attention='hard'``.

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


# How the output head's low-dimensional code is chosen. ``learned`` makes the class-code
# matrix a dense parameter; ``binary`` freezes it to the +-1 binary expansion of each
# class index. See ``LutModel`` for why both are a plain linear readout.
OUTPUT_CODE_LEARNED = 'learned'
OUTPUT_CODE_BINARY = 'binary'
OUTPUT_CODE_RANDOM = 'random'
OUTPUT_CODE_MODES = (OUTPUT_CODE_LEARNED, OUTPUT_CODE_BINARY, OUTPUT_CODE_RANDOM)
OUTPUT_CODE_FROZEN = (OUTPUT_CODE_BINARY, OUTPUT_CODE_RANDOM)

# Which attention mechanism the heads use. ``new`` is main.py's --new-attention
# (Q/K/V/Final). ``hard`` is main.py's original attention -- what it runs when
# --new-attention is absent: one table per head indexed by (query bits, key bits, PE bits)
# of every causal pair. See OldAttentionHead.
ATTENTION_NEW = 'new'
ATTENTION_HARD = 'hard'
ATTENTION_MODES = (ATTENTION_NEW, ATTENTION_HARD)

# The hard-attention table has 2 ** (2 * n_c + positional_dim) rows, so its size explodes
# with n_c. Refuse configurations whose row index alone needs more bits than this; at 24
# bits a single tree of a 16-wide table is already 1 GiB.
HARD_ATTN_MAX_TOTAL_BITS = 24


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
    attention: str = ATTENTION_NEW
    # Learned bits per side (query and key) of the hard-attention table; None uses n_c,
    # which is what main.py does. Its row index is 2 * hard_attn_n_c + positional_dim bits.
    hard_attn_n_c: Optional[int] = None
    new_MHA: bool = False
    weight_sharing: Optional[int] = None
    # Run a layer's heads as 4 stacked LUT calls instead of 4*H separate ones. Purely a
    # performance choice -- bit-identical either way (test_torch_stacking.py) -- but it
    # changes the parameter layout, so it defaults off and the trainer turns it on. The
    # test suites that reach into per-head modules (test_torch_parity.py above all) build
    # models unstacked and are the reason this is not simply always on.
    stack_heads: bool = False
    # A learned weight per tree slot in every table: y = sum_t w[t] * S[trees[t], j[t]].
    # Initialised to 1, so a fresh model computes exactly what it would without them.
    # Off by default because it adds parameters, which existing checkpoints do not have.
    tree_weights: bool = False

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

    # --- output head ---
    # None keeps the head's y_dim at vocab_size_output (one score per class, the
    # `main.py` behaviour). An int D makes the head emit D numbers that a (V, D) code
    # matrix maps to the V logits, so the table costs n_t * 2**n_c * D instead of
    # n_t * 2**n_c * V.
    output_code_dim: Optional[int] = None
    output_code: str = OUTPUT_CODE_LEARNED
    # Add a per-class bias: logits = y @ C.T + b. Without it every logit vector the model
    # can produce lies in col(C), a D-dimensional subspace -- class priors included. The
    # frozen binary code's column space holds only 20.8% of this corpus's log-prior energy
    # (less than a random D-dim subspace), so that arm cannot express "NOUN is common"
    # except by spending its rank on it. V extra parameters; 18 here, 14,398 at full vocab.
    output_code_bias: bool = False

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
        if self.attention not in ATTENTION_MODES:
            raise ConfigError(f'attention must be one of {ATTENTION_MODES}')
        if self.attention == ATTENTION_HARD:
            self._validate_hard_attention()
        elif self.hard_attn_n_c is not None:
            raise ConfigError('hard_attn_n_c requires attention=hard')
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
        if self.output_code not in OUTPUT_CODE_MODES:
            raise ConfigError(f'output_code must be one of {OUTPUT_CODE_MODES}')
        if self.output_code_bias and self.output_code_dim is None:
            # The uncoded head reaches class priors through the table's own rows, so a
            # bias there would make the control differ from the default head by two
            # things at once.
            raise ConfigError('output_code_bias requires output_code_dim')
        if self.output_code_dim is not None:
            if self.output_code_dim < 1:
                raise ConfigError('output_code_dim must be at least 1')
            if self.output_code_dim > self.vocab_size_output:
                raise ConfigError('output_code_dim must not exceed vocab_size_output; '
                                  'a code wider than the vocabulary saves nothing')
            # Any code needs enough bits to give every class its own codeword.
            need = max(1, (self.vocab_size_output - 1).bit_length())
            if self.output_code in OUTPUT_CODE_FROZEN and self.output_code_dim < need:
                raise ConfigError(
                    f'output_code {self.output_code} needs output_code_dim >= {need} to '
                    f'give {self.vocab_size_output} classes distinct codewords')
            # Above `need` bits, `binary` is a systematic ECOC: the first `need` bits are
            # the class index and the rest are parity checks over it (see ecoc_masks).
            # Padding the index expansion with its own high bits would give constant
            # columns instead, which only shift every logit equally and cancel.

    def _validate_hard_attention(self):
        if self.hard_attn_n_c is not None and self.hard_attn_n_c < 1:
            raise ConfigError('hard_attn_n_c must be at least 1')
        # These all act on the new-attention Q/K/PE path, which hard attention lacks.
        for flag in ('PE_only', 'qkpe_direct_sign', 'normalize_qk'):
            if getattr(self, flag):
                raise ConfigError(f'{flag} requires attention=new')
        if self.gradient_pair_selection != SELECTION_CLOSEST:
            raise ConfigError('attention=hard supports closest gradient-pair selection '
                              'only (main.py\'s cosine-distance variant is not ported)')
        if self.stack_heads:
            raise ConfigError('stack_heads applies to attention=new only')
        if self.hard_attn_total_n_c > HARD_ATTN_MAX_TOTAL_BITS:
            raise ConfigError(
                f'the hard-attention table needs 2 * {self.hard_attn_bits} + '
                f'{self.positional_dim} = {self.hard_attn_total_n_c} row bits, more than '
                f'{HARD_ATTN_MAX_TOTAL_BITS}; lower hard_attn_n_c or positional_dim')

    @property
    def hard_attn_bits(self):
        """Learned bits per side (query, key) of the hard-attention table."""
        return self.n_c if self.hard_attn_n_c is None else self.hard_attn_n_c

    @property
    def hard_attn_total_n_c(self):
        """Row-index width of the hard-attention table: query, key, then PE bits."""
        return 2 * self.hard_attn_bits + self.positional_dim

    @property
    def head_value_dim(self):
        """Output width of one attention head's V and Final (or hard-attention) table."""
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
