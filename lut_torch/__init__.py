"""PyTorch implementation of the LUT (look-up table) model.

This package re-implements the ``--new-attention`` / ``--new-MHA`` architecture from
``main.py`` on top of PyTorch autograd.  ``main.py`` itself is untouched and remains the
reference implementation; ``test_torch_parity.py`` asserts that this package reproduces
its forward values, gradients and post-step tables exactly.

The only genuinely custom piece is :mod:`lut_torch.functional`, which defines how a LUT
lookup differentiates.  Everything else -- optimizers, schedulers, checkpointing, the
gradient routing through Q/K/PE and QK normalisation -- is left to torch.
"""

__all__ = ['comparisons', 'config', 'functional']
