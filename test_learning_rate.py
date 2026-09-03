from types import SimpleNamespace

from main import get_learning_rate


def test_optional_learning_rate_halving_boundaries():
    args = SimpleNamespace(
        warmup_steps=1_000,
        max_learning_rate=0.1,
        lr_scheduler='inverse-sqrt',
        inverse_sqrt_alpha=0.0,
        max_steps=100_000,
        cosine_min_lr_ratio=0.01,
        halve_lr_at_20k_40k=True,
    )

    assert get_learning_rate(20_000, args) == 0.1
    assert get_learning_rate(20_001, args) == 0.05
    assert get_learning_rate(40_000, args) == 0.05
    assert get_learning_rate(40_001, args) == 0.025


def test_learning_rate_halving_is_opt_in():
    args = SimpleNamespace(
        warmup_steps=1_000,
        max_learning_rate=0.1,
        lr_scheduler='inverse-sqrt',
        inverse_sqrt_alpha=0.0,
        max_steps=100_000,
        cosine_min_lr_ratio=0.01,
        halve_lr_at_20k_40k=False,
    )

    assert get_learning_rate(40_001, args) == 0.1
