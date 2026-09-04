"""Tiny pytest shim so lut_torch tests run on either interpreter here.

The project virtualenv has torch but no pytest; the cluster python has pytest but no
torch (see AGENTS.md).  Rather than install anything, test modules import `parametrize`
from here and end with `run_all(globals())`, so both of these work::

    .venv/bin/python test_torch_lut_primitive.py     # shim runner
    python -m pytest -q test_torch_lut_primitive.py  # real pytest, if available

When pytest is importable this is a thin alias over `pytest.mark.parametrize` and the
runner is never used.
"""

import itertools
import traceback

try:  # pragma: no cover - depends on interpreter
    import pytest

    HAS_PYTEST = True
    parametrize = pytest.mark.parametrize
    skip = pytest.skip
except ImportError:  # pragma: no cover - depends on interpreter
    HAS_PYTEST = False

    class _Skipped(Exception):
        pass

    def skip(reason=''):
        raise _Skipped(reason)

    def parametrize(argnames, argvalues):
        names = [n.strip() for n in argnames.split(',')] \
            if isinstance(argnames, str) else list(argnames)

        def decorator(fn):
            cases = []
            for value in argvalues:
                if len(names) == 1:
                    cases.append({names[0]: value})
                else:
                    cases.append(dict(zip(names, value)))
            # Stacked decorators accumulate; the runner takes the cartesian product.
            fn._lut_params = getattr(fn, '_lut_params', []) + [cases]
            return fn

        return decorator


def _expand(fn):
    groups = getattr(fn, '_lut_params', None)
    if not groups:
        return [{}]
    combos = []
    for combo in itertools.product(*groups):
        merged = {}
        for part in combo:
            merged.update(part)
        combos.append(merged)
    return combos


def run_all(namespace, prefix='test_'):
    """Run every `test_*` callable in `namespace`. Returns a process exit code."""
    if HAS_PYTEST:  # pragma: no cover
        import sys
        return pytest.main(['-q', namespace['__file__']])

    passed = failed = skipped = 0
    failures = []
    for name, fn in sorted(namespace.items()):
        if not name.startswith(prefix) or not callable(fn):
            continue
        for kwargs in _expand(fn):
            label = name + (f"[{kwargs}]" if kwargs else '')
            try:
                fn(**kwargs)
            except _Skipped as exc:
                skipped += 1
                print(f"s {label}: {exc}")
            except Exception:
                failed += 1
                failures.append((label, traceback.format_exc()))
                print(f"F {label}")
            else:
                passed += 1
    for label, tb in failures:
        print(f"\n===== FAILED {label} =====\n{tb}")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0
