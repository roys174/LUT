"""Convert an unstacked ``state_dict`` into the layout :class:`StackedHeads` expects.

Stacking a layer's heads (see :class:`lut_torch.modules.StackedHeads`) replaces ``4 * H``
per-head tables with three tables that each hold every head's rows in one slab, so a
checkpoint written by an unstacked run does not load into a stacked model as-is.  This
module does that translation, and the trainer applies it automatically on
``--load-model`` so an existing run can be resumed with ``--stack-heads on``.

The reverse direction is deliberately not provided.  Splitting a stacked table back apart
is easy enough, but nothing needs it: the unstacked path exists for the test suites that
inspect per-head modules, and those build their models rather than loading them.
"""

import torch

# Which stacked table each per-head table folds into, and in what slot order.  Q and K
# share a shape, so they occupy one table of ``2H`` slots -- all the Q heads first, then
# all the K heads, matching StackedHeads._add_stack.
_ROLES = (('qk', ('Q', 'K')), ('v', ('V',)), ('final', ('Final',)))

# Per-head leaf -> stacked suffix.  ``S_flat`` concatenates along the row axis (the slabs
# are contiguous); the index buffers stack into a new leading slot axis.
_CONCAT = {'S_flat': 'S_flat'}
_STACK = {'a_arr': 'a', 'b_arr': 'b', 'trees': 'trees'}


def _stacked_layers(model):
    """``{layer key prefix: StackedHeads module}`` for every stacked layer of ``model``."""
    from .modules import StackedHeads
    return {name: module for name, module in model.named_modules()
            if isinstance(module, StackedHeads)}


def stack_state_dict(source, model):
    """Return ``source`` with its per-head entries folded into ``model``'s stacked layout.

    Keys that do not belong to a stacked head block (FFN, MHA_O, the output head,
    LayerNorm affines, the token embedding) pass through untouched, so this is safe to run
    on a dict that is already stacked or on a partial dict -- a mapping of table names to
    gradients, say, which is how ``test_torch_stacking.py`` compares the two paths.
    """
    out = dict(source)
    for prefix, module in _stacked_layers(model).items():
        n_heads = module.cfg.num_heads
        consumed = set()
        for stack_name, roles in _ROLES:
            slots = [f'{prefix}.{head}.{role}'
                     for role in roles for head in range(n_heads)]
            for leaf, suffix in list(_CONCAT.items()) + list(_STACK.items()):
                keys = [f'{slot}.{leaf}' for slot in slots]
                if not all(key in out for key in keys):
                    continue
                parts = [out[key] for key in keys]
                consumed.update(keys)
                target = f'{prefix}.{stack_name}_{suffix}'
                if leaf in _CONCAT:
                    out[target] = torch.cat(parts, dim=0)
                elif leaf == 'trees':
                    # Each slot's rows start at slot * s_n_t, and the tree map is what
                    # carries that offset into the lookup.
                    n_bins = getattr(module, f'{stack_name}_n_bins')
                    s_n_t = parts[0].shape[0] and (
                        getattr(module, f'{stack_name}_S_flat').shape[0]
                        // len(slots) // n_bins)
                    offsets = torch.arange(len(slots), dtype=torch.int64).unsqueeze(1)
                    out[target] = torch.stack(parts) + offsets * s_n_t
                else:
                    out[target] = torch.stack(parts)

        for leaf in ('positional_encoding', 'qkpe_a', 'qkpe_b'):
            keys = [f'{prefix}.{head}.{leaf}' for head in range(n_heads)]
            if all(key in out for key in keys):
                consumed.update(keys)
                out[f'{prefix}.{leaf}'] = torch.stack([out[key] for key in keys])

        for leaf in ('tok_i', 'tok_j', 'pe_pos', 'final_pair_ptr', 'final_pair_list'):
            key = f'{prefix}.0.{leaf}'
            if key in out:
                out[f'{prefix}.{leaf}'] = out[key]
                consumed.update(f'{prefix}.{head}.{leaf}' for head in range(n_heads))

        for key in consumed:
            out.pop(key, None)
    return out


def looks_unstacked(source, model):
    """Whether ``source`` is a per-head checkpoint being loaded into a stacked model."""
    return any(f'{prefix}.0.V.S_flat' in source
               for prefix in _stacked_layers(model))
