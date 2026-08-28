"""Grant et al. representational-divergence prediction baseline.

The ICLR 2026 Oral evaluates divergence at the cohort level, comparing
natural and intervened activation distributions.  This module preserves that
semantics and deliberately does not expose a per-row OOD classifier.

Primary-source pin:
https://github.com/grantsrb/rep_divergence/tree/f2548d2ea9b4f4b87a87ba5d53db43838d15c521
(`divergence/divergence_utils.py`, especially `sample_emd`,
`collect_divergences`, and `divergences`).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import torch


SinkhornLoss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class GrantDivergencePanel:
    sample_size: int
    split_size: int
    seed: int
    mse: float
    emd: float
    base_emd: float
    cost_cos: float
    base_cost_cos: float
    nn_cos: float
    base_nn_cos: float
    cost_mse: float
    base_cost_mse: float
    nn_mse: float
    base_nn_mse: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_states(
    natural_states: torch.Tensor, intervened_states: torch.Tensor
) -> None:
    if natural_states.ndim != 2 or intervened_states.ndim != 2:
        raise ValueError("states must have shape [sample, feature]")
    if natural_states.shape != intervened_states.shape:
        raise ValueError("natural and intervened states must have equal shape")
    if natural_states.shape[0] < 4 or natural_states.shape[1] < 1:
        raise ValueError("at least four samples and one feature are required")
    if not natural_states.is_floating_point() or not intervened_states.is_floating_point():
        raise ValueError("states must be floating point")
    if not torch.isfinite(natural_states).all() or not torch.isfinite(
        intervened_states
    ).all():
        raise ValueError("states must be finite")


def _feature_zscore(
    states: torch.Tensor, *, unbiased: bool, epsilon: float = 1e-5
) -> torch.Tensor:
    """Match either standardization convention used in the official code."""

    mean = states.mean(dim=0)
    std = states.std(dim=0, unbiased=unbiased)
    std = torch.where(std > 0, std, torch.ones_like(std))
    return (states - mean) / (std + epsilon)


def _load_official_sinkhorn() -> SinkhornLoss:
    try:
        from geomloss import SamplesLoss
    except ImportError as error:  # pragma: no cover - dependency-gate path
        raise RuntimeError(
            "Grant EMD requires geomloss; install it only in the isolated baseline runtime"
        ) from error
    return SamplesLoss(loss="sinkhorn", p=2, blur=0.05)


def normalized_sinkhorn_emd(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    sinkhorn_loss: SinkhornLoss | None = None,
) -> float:
    """Official normalized Sinkhorn/EMD calculation from `sample_emd`."""

    if first.shape != second.shape or first.ndim != 2 or first.shape[0] == 0:
        raise ValueError("EMD inputs must have equal nonempty [sample, feature] shape")
    # `sample_emd` calls torch.std directly, whose correction=1 convention is
    # the sample standard deviation.  The companion correlation code below
    # instead implements a population standard deviation explicitly.
    x = _feature_zscore(first, unbiased=True).float()
    y = _feature_zscore(second, unbiased=True).float()
    loss = _load_official_sinkhorn() if sinkhorn_loss is None else sinkhorn_loss
    value = loss(x, y)
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError("Sinkhorn backend must return one scalar tensor")
    result = float(value.detach().cpu()) / math.sqrt(first.shape[1])
    if not math.isfinite(result):
        raise ValueError("Sinkhorn backend returned a non-finite value")
    return result


def _correlation_cost_matrix(first: torch.Tensor, second: torch.Tensor) -> np.ndarray:
    # Official code transposes [sample, feature] twice around get_cor_mtx;
    # the resulting matrix compares samples after z-scoring across features.
    x = _feature_zscore(first.T, unbiased=False).T
    y = _feature_zscore(second.T, unbiased=False).T
    correlation = (x @ y.T) / first.shape[1]
    return (1.0 - correlation).detach().cpu().numpy()


def _mse_cost_matrix(first: torch.Tensor, second: torch.Tensor) -> np.ndarray:
    return (
        (first[:, None, :] - second[None, :, :]).square().mean(dim=-1).cpu().numpy()
    )


def _assignment_mean(cost: np.ndarray) -> float:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:  # pragma: no cover - dependency-gate path
        raise RuntimeError("Grant matching metrics require scipy") from error
    rows, columns = linear_sum_assignment(cost)
    return float(cost[rows, columns].sum() / max(len(rows), len(columns)))


def _distribution_metrics(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    sinkhorn_loss: SinkhornLoss | None,
) -> dict[str, float]:
    cos = _correlation_cost_matrix(first, second)
    mse = _mse_cost_matrix(first, second)
    return {
        "emd": normalized_sinkhorn_emd(
            first, second, sinkhorn_loss=sinkhorn_loss
        ),
        "cost_cos": _assignment_mean(cos),
        "nn_cos": float(cos.min(axis=1).mean()),
        "cost_mse": _assignment_mean(mse),
        "nn_mse": float(mse.min(axis=1).mean()),
    }


def grant_divergence_panel(
    natural_states: torch.Tensor,
    intervened_states: torch.Tensor,
    *,
    seed: int,
    sample_size: int = 5000,
    sinkhorn_loss: SinkhornLoss | None = None,
) -> GrantDivergencePanel:
    """Compute a deterministic, development-only Grant-style cohort panel.

    The official analysis first samples paired rows, then compares one natural
    half with an independently permuted intervention half and with a second
    natural half.  We preserve those roles while using a local generator so the
    protocol cannot perturb or depend on global RNG state.
    """

    _validate_states(natural_states, intervened_states)
    if sample_size < 4:
        raise ValueError("sample_size must be at least four")
    selected_size = min(sample_size, natural_states.shape[0])
    if selected_size % 2:
        selected_size -= 1
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    selected = torch.randperm(natural_states.shape[0], generator=generator)[
        :selected_size
    ]
    natural = natural_states.detach().cpu()[selected]
    intervened = intervened_states.detach().cpu()[selected]
    split_size = selected_size // 2

    natural_order = torch.randperm(selected_size, generator=generator)
    intervention_order = torch.randperm(selected_size, generator=generator)
    reference_natural = natural[natural_order[:split_size]]
    comparison_intervened = intervened[intervention_order[split_size:]]
    comparison_natural = natural[intervention_order[:split_size]]

    observed = _distribution_metrics(
        reference_natural, comparison_intervened, sinkhorn_loss=sinkhorn_loss
    )
    baseline = _distribution_metrics(
        reference_natural, comparison_natural, sinkhorn_loss=sinkhorn_loss
    )
    paired_mse = float((natural - intervened).square().mean())
    values = {
        "sample_size": selected_size,
        "split_size": split_size,
        "seed": seed,
        "mse": paired_mse,
        **observed,
        **{f"base_{name}": value for name, value in baseline.items()},
    }
    return GrantDivergencePanel(**values)
