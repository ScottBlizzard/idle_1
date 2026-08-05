"""Cross-fitted activation-overlap diagnostics for the July audit.

This module intentionally leaves ``validity_fast.FastSiteReference`` unchanged so
that historical results remain reproducible.  The implementation here fixes two
audit-critical issues:

* PCA and normalization are fit on disjoint samples.
* Held-out queries keep their nearest neighbour; only explicit self-queries drop it.

The returned quantities are overlap diagnostics, not causal-validity labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_2d_f64(x: np.ndarray, name: str, *, min_rows: int = 2) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < min_rows:
        raise ValueError(
            f"{name} must have shape [n, d] with n >= {min_rows}; got {x.shape}"
        )
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return x


def _unique_rows(x: np.ndarray) -> int:
    # Exact duplicates are the quantity relevant to the duplicated-prompt audit.
    return int(np.unique(np.ascontiguousarray(x), axis=0).shape[0])


@dataclass(frozen=True)
class CalibrationStats:
    median: float
    mad_scale: float
    std: float


class CrossFitSiteReference:
    """PCA overlap scorer with an independent same-distribution calibration set."""

    COMPONENTS = ("knn", "recon", "maha")

    def __init__(
        self,
        fit_ref: np.ndarray,
        calibration_ref: np.ndarray,
        *,
        normalization_ref: np.ndarray | None = None,
        target_law: str = "natural_reference",
        knn_k: int = 12,
        proj_rank: int = 32,
        shrinkage: float = 1e-2,
        rank_rtol: float = 1e-7,
    ):
        fit_ref = _as_2d_f64(fit_ref, "fit_ref")
        calibration_ref = _as_2d_f64(calibration_ref, "calibration_ref")
        if normalization_ref is not None:
            normalization_ref = _as_2d_f64(normalization_ref, "normalization_ref")
        widths = [fit_ref.shape[1], calibration_ref.shape[1]]
        if normalization_ref is not None:
            widths.append(normalization_ref.shape[1])
        if len(set(widths)) != 1:
            raise ValueError("all reference splits must have the same width")
        if not isinstance(target_law, str) or not target_law.strip():
            raise ValueError("target_law must be a nonempty declared-law label")
        if knn_k < 1 or proj_rank < 1:
            raise ValueError("knn_k and proj_rank must be positive")
        if normalization_ref is None and len(calibration_ref) < 4:
            raise ValueError(
                "calibration_ref needs at least 4 rows so composite conformal "
                "normalization and final calibration remain independent"
            )
        if normalization_ref is not None and (
            len(normalization_ref) < 2 or len(calibration_ref) < 2
        ):
            raise ValueError(
                "explicit normalization and final calibration splits each need "
                "at least 2 rows"
            )

        self.fit_ref = fit_ref
        self.calibration_ref = calibration_ref
        self.target_law = target_law.strip()
        self.explicit_composite_splits = normalization_ref is not None
        self.mean_orig = fit_ref.mean(axis=0)
        centered = fit_ref - self.mean_orig

        # Full SVD is deterministic. Rank is truncated using the observed numerical
        # rank, preventing the requested PCA dimension from exceeding the support.
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        if singular_values.size == 0 or singular_values[0] <= 0:
            effective_rank = 1
        else:
            effective_rank = int(np.sum(singular_values > singular_values[0] * rank_rtol))
            effective_rank = max(effective_rank, 1)
        selected_rank = min(proj_rank, effective_rank, fit_ref.shape[0] - 1, fit_ref.shape[1])

        self.singular_values = singular_values
        self.effective_rank = effective_rank
        self.selected_rank = selected_rank
        self.basis = vt[:selected_rank]
        self.fit_proj = centered @ self.basis.T
        self.knn_k = min(knn_k, len(fit_ref))

        cov = np.atleast_2d(np.cov(self.fit_proj, rowvar=False))
        trace = float(np.trace(cov))
        ridge = shrinkage * trace / selected_rank if trace > 0 else shrinkage
        self.cov_inv = np.linalg.pinv(cov + ridge * np.eye(selected_rank), hermitian=True)
        self.proj_mean = self.fit_proj.mean(axis=0)

        marginal_ref = (
            calibration_ref
            if normalization_ref is None
            else np.concatenate([normalization_ref, calibration_ref], axis=0)
        )
        calibration_raw = self.raw_metrics(marginal_ref)
        self.marginal_reference = marginal_ref
        self.calibration_raw = calibration_raw
        self.calibration_stats: dict[str, CalibrationStats] = {}
        for name in self.COMPONENTS:
            values = calibration_raw[name]
            median = float(np.median(values))
            mad_scale = float(1.4826 * np.median(np.abs(values - median)))
            self.calibration_stats[name] = CalibrationStats(
                median=median, mad_scale=mad_scale, std=float(values.std())
            )

        # A second split inside the held-out calibration sample is required for a
        # valid *composite* conformal score.  The first half freezes the component
        # normalization.  Only the untouched second half calibrates the final
        # scalar nonconformity.  In contrast, the historical overlap_ecdf below is
        # only a geometric mean of marginal tail probabilities and has no joint
        # finite-sample coverage guarantee.
        if normalization_ref is None:
            normalization_stop = len(calibration_ref) // 2
            self.composite_normalization_ref = calibration_ref[:normalization_stop]
            self.composite_calibration_ref = calibration_ref[normalization_stop:]
            self.composite_normalization_raw = {
                name: values[:normalization_stop]
                for name, values in calibration_raw.items()
            }
            self.composite_calibration_raw = {
                name: values[normalization_stop:]
                for name, values in calibration_raw.items()
            }
        else:
            self.composite_normalization_ref = normalization_ref
            self.composite_calibration_ref = calibration_ref
            normalization_raw = self.raw_metrics(normalization_ref)
            final_calibration_raw = self.raw_metrics(calibration_ref)
            self.composite_normalization_raw = normalization_raw
            self.composite_calibration_raw = final_calibration_raw
        self.composite_stats: dict[str, CalibrationStats] = {}
        for name in self.COMPONENTS:
            values = self.composite_normalization_raw[name]
            median = float(np.median(values))
            mad_scale = float(1.4826 * np.median(np.abs(values - median)))
            self.composite_stats[name] = CalibrationStats(
                median=median,
                mad_scale=mad_scale,
                std=float(values.std()),
            )
        self.composite_calibration_scores = self._composite_nonconformity_from_raw(
            self.composite_calibration_raw
        )

    def _project(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        centered = x - self.mean_orig
        return centered, centered @ self.basis.T

    def _knn_dist(self, x_proj: np.ndarray, *, exclude_self: bool = False) -> np.ndarray:
        ref_sq = np.einsum("ij,ij->i", self.fit_proj, self.fit_proj)
        out = np.empty(len(x_proj), dtype=np.float64)
        start = 1 if exclude_self else 0
        stop = min(start + self.knn_k, len(self.fit_proj))
        if stop <= start:
            raise ValueError("not enough reference samples for requested kNN mode")
        for offset in range(0, len(x_proj), 256):
            xb = x_proj[offset:offset + 256]
            d2 = (
                np.einsum("ij,ij->i", xb, xb)[:, None]
                - 2.0 * xb @ self.fit_proj.T
                + ref_sq[None, :]
            )
            d2 = np.maximum(d2, 0.0)
            nearest = np.partition(d2, kth=stop - 1, axis=1)[:, start:stop]
            out[offset:offset + len(xb)] = np.sqrt(nearest).mean(axis=1)
        return out

    def raw_metrics(self, x: np.ndarray, *, exclude_self: bool = False) -> dict[str, np.ndarray]:
        x = _as_2d_f64(x, "x", min_rows=1)
        if x.shape[1] != self.fit_ref.shape[1]:
            raise ValueError("query width differs from reference width")
        if exclude_self and (x.shape != self.fit_ref.shape or not np.array_equal(x, self.fit_ref)):
            raise ValueError("exclude_self=True is only valid for the exact fit_ref array")
        centered, x_proj = self._project(x)
        reconstruction = x_proj @ self.basis
        diff = x_proj - self.proj_mean
        return {
            "knn": self._knn_dist(x_proj, exclude_self=exclude_self),
            "recon": np.linalg.norm(centered - reconstruction, axis=1),
            "maha": np.sqrt(
                np.einsum("ik,kl,il->i", diff, self.cov_inv, diff).clip(min=0.0)
            ),
        }

    @staticmethod
    def _upper_tail_probability(calibration: np.ndarray, query: np.ndarray) -> np.ndarray:
        # Finite-sample conformal-style upper-tail probability. Large distance/error
        # receives a small probability; the +1 correction avoids exact zero.
        return (1.0 + (calibration[:, None] >= query[None, :]).sum(axis=0)) / (
            len(calibration) + 1.0
        )

    def _composite_nonconformity_from_raw(
        self, raw: dict[str, np.ndarray], *, scale_floor: float = 1e-12
    ) -> np.ndarray:
        """Return a fixed scalar nonconformity learned without final calibration.

        Softplus of each robust standardized distance is monotone, non-negative,
        and avoids cancellation between components.  The precise score controls
        power, not conformal validity: conditional on the fit and normalization
        splits, the final calibration samples and an in-reference query are
        exchangeable.
        """
        if scale_floor <= 0:
            raise ValueError("scale_floor must be positive")
        components = []
        for name in self.COMPONENTS:
            stats = self.composite_stats[name]
            scale = max(
                stats.mad_scale,
                0.05 * abs(stats.median),
                0.5 * stats.std,
                scale_floor,
            )
            z = (np.asarray(raw[name], dtype=np.float64) - stats.median) / scale
            components.append(np.logaddexp(0.0, z))
        return np.mean(np.stack(components, axis=0), axis=0)

    def score(self, x: np.ndarray, *, scale_floor: float = 1e-6) -> dict[str, np.ndarray]:
        if scale_floor <= 0:
            raise ValueError("scale_floor must be positive")
        raw = self.raw_metrics(x)
        result: dict[str, np.ndarray] = dict(raw)
        z_sum = np.zeros(len(x), dtype=np.float64)
        tail_components = []
        for name in self.COMPONENTS:
            stats = self.calibration_stats[name]
            scale = max(
                stats.mad_scale,
                0.05 * abs(stats.median),
                0.5 * stats.std,
                scale_floor,
            )
            z = (raw[name] - stats.median) / scale
            tail = self._upper_tail_probability(self.calibration_raw[name], raw[name])
            result[f"{name}_z"] = z
            result[f"{name}_tail_p"] = tail
            result[f"{name}_scale"] = np.full(len(x), scale)
            z_sum += z
            tail_components.append(tail)

        result["z_sum"] = z_sum
        result["overlap_z"] = 1.0 / (1.0 + np.exp(np.clip(z_sum / 3.0, -30.0, 30.0)))
        # Geometric mean retains a [0,1] scale and does not depend on a raw-unit floor.
        result["overlap_ecdf"] = np.exp(np.mean(np.log(np.stack(tail_components)), axis=0))
        composite = self._composite_nonconformity_from_raw(raw)
        result["composite_nonconformity"] = composite
        result["overlap_conformal"] = self._upper_tail_probability(
            self.composite_calibration_scores, composite
        )
        return result

    def diagnostics(self) -> dict:
        retained = self.singular_values[: self.selected_rank]
        condition_number = (
            float(retained[0] / retained[-1])
            if len(retained) and retained[-1] > 0
            else float("inf")
        )
        return {
            "n_fit": int(len(self.fit_ref)),
            "n_calibration": int(len(self.calibration_ref)),
            "n_marginal_reference": int(len(self.marginal_reference)),
            "n_composite_normalization": int(len(self.composite_normalization_ref)),
            "n_composite_calibration": int(len(self.composite_calibration_ref)),
            "explicit_composite_splits": self.explicit_composite_splits,
            "target_law": self.target_law,
            "unique_fit_activations": _unique_rows(self.fit_ref),
            "unique_calibration_activations": _unique_rows(self.calibration_ref),
            "effective_rank": int(self.effective_rank),
            "selected_rank": int(self.selected_rank),
            "condition_number": condition_number,
            "singular_values": self.singular_values[: min(64, len(self.singular_values))].tolist(),
            "calibration": {
                name: {
                    "median": stats.median,
                    "mad_scale": stats.mad_scale,
                    "std": stats.std,
                }
                for name, stats in self.calibration_stats.items()
            },
            "composite_conformal": {
                "score": "mean softplus of independently normalized component z-scores",
                "normalization": {
                    name: {
                        "median": stats.median,
                        "mad_scale": stats.mad_scale,
                        "std": stats.std,
                    }
                    for name, stats in self.composite_stats.items()
                },
            },
        }
