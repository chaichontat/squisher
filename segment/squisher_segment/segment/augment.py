from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import cv2


@dataclass
class PhotometricConfig:
    # Intensity/gamma jitter (per-channel)
    gain_min: float = 0.8
    gain_max: float = 1.2
    gamma_min: float = 0.8
    gamma_max: float = 1.25

    # Poisson–Gaussian noise
    # Peak photon count used to convert to/from Poisson domain
    poisson_peak_min: float = 500.0
    poisson_peak_max: float = 2000.0
    read_noise_min: float = 1.0  # DN at peak
    read_noise_max: float = 10.0 # DN at peak

    # Blur
    p_motion: float = 0.25
    defocus_sigma_min: float = 0.5
    defocus_sigma_max: float = 2.0
    motion_len_min: int = 3
    motion_len_max: int = 7

    # Vignetting / bias field
    p_bias: float = 0.8
    bias_knots_min: int = 2
    bias_knots_max: int = 4
    bias_range: tuple[float, float] = (0.85, 1.15)


def _rand_uniform(rng: np.random.Generator, low: float, high: float):
    return float(rng.uniform(low, high))


def _apply_intensity_gamma(img: np.ndarray, rng: np.random.Generator, cfg: PhotometricConfig) -> np.ndarray:
    # img: [N, C, H, W]
    n, c, h, w = img.shape
    gains = rng.uniform(cfg.gain_min, cfg.gain_max, size=(n, c, 1, 1)).astype(np.float32)
    gammas = rng.uniform(cfg.gamma_min, cfg.gamma_max, size=(n, c, 1, 1)).astype(np.float32)
    out = img * gains
    # avoid negative and tiny values before gamma
    out = np.maximum(out, 0.0).astype(np.float32, copy=False)
    # gamma transform in linear space
    return np.power(out, gammas, out)


def _apply_poisson_gaussian(img: np.ndarray, rng: np.random.Generator, cfg: PhotometricConfig) -> np.ndarray:
    # Convert each (N,C) slice to Poisson domain by picking a peak count
    n, c, h, w = img.shape
    out = img.astype(np.float32, copy=False)
    # Estimate robust per-(n,c) scale to handle arbitrary input ranges
    reshaped = out.transpose(0, 1, 3, 2).reshape(n * c, -1)
    p99 = np.percentile(reshaped, 99.5, axis=1).astype(np.float32)
    p99 = np.clip(p99, 1e-3, None)
    p99 = p99.reshape(n, c, 1, 1)

    peak = rng.uniform(cfg.poisson_peak_min, cfg.poisson_peak_max, size=(n, c, 1, 1)).astype(np.float32)
    # Normalize to [0, 1] w.r.t p99 and map to counts
    norm = np.clip(out / p99, 0.0, None)
    lam = norm * peak
    # Poisson sampling (vectorized); ensure non-negative
    counts = rng.poisson(lam).astype(np.float32)
    norm_poisson = counts / peak

    # Add read noise in DN relative to peak then denormalize
    sigma_dn = rng.uniform(cfg.read_noise_min, cfg.read_noise_max, size=(n, c, 1, 1)).astype(np.float32)
    sigma_norm = sigma_dn / peak
    noisy = norm_poisson + rng.normal(0.0, 1.0, size=norm_poisson.shape).astype(np.float32) * sigma_norm
    return noisy * p99


def _gaussian_blur_per_channel(img: np.ndarray, sigma: float) -> np.ndarray:
    # Using cv2 GaussianBlur per (N,C)
    n, c, h, w = img.shape
    k = max(3, int(round(sigma * 6)) | 1)
    out = img.copy()
    for i in range(n):
        for j in range(c):
            out[i, j] = cv2.GaussianBlur(out[i, j], (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    return out


def _motion_blur_kernel(length: int, angle_deg: float) -> np.ndarray:
    k = max(3, int(length))
    kernel = np.zeros((k, k), dtype=np.float32)
    cv2.line(kernel, (0, k // 2), (k - 1, k // 2), 1.0, thickness=1)
    # rotate
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    s = kernel.sum()
    return kernel / s if s > 0 else kernel


def _apply_blur(img: np.ndarray, rng: np.random.Generator, cfg: PhotometricConfig) -> np.ndarray:
    if rng.random() < cfg.p_motion:
        length = rng.integers(cfg.motion_len_min, cfg.motion_len_max + 1)
        angle = rng.uniform(0, 180)
        k = _motion_blur_kernel(int(length), float(angle))
        out = img.copy()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                out[i, j] = cv2.filter2D(out[i, j], -1, k, borderType=cv2.BORDER_REFLECT)
        return out
    else:
        sigma = _rand_uniform(rng, cfg.defocus_sigma_min, cfg.defocus_sigma_max)
        return _gaussian_blur_per_channel(img, sigma)


def _apply_bias_field(img: np.ndarray, rng: np.random.Generator, cfg: PhotometricConfig) -> np.ndarray:
    if rng.random() > cfg.p_bias:
        return img
    n, c, h, w = img.shape
    ky = rng.integers(cfg.bias_knots_min, cfg.bias_knots_max + 1)
    kx = rng.integers(cfg.bias_knots_min, cfg.bias_knots_max + 1)
    low, high = cfg.bias_range
    bias_lowres = rng.uniform(low, high, size=(1, 1, ky, kx)).astype(np.float32)
    bias = cv2.resize(bias_lowres[0, 0], (w, h), interpolation=cv2.INTER_CUBIC)[None, None]
    return img * bias.astype(img.dtype)


def build_batch_augmenter(
    cfg: Optional[PhotometricConfig] = None,
    seed: Optional[int] = None,
    *,
    min_ops_per_image: int = 1,
    max_ops_per_image: int = 2,
    op_weights: Optional[dict[str, float]] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a photometric augmenter for (N,C,H,W) batches.

    Picks 1–2 ops per image (configurable) from the pool:
      - intensity_gamma, poisson_gaussian, blur, bias_field
    and applies only those, in a randomized order per image.

    Labels are not touched; call this only on image tensors.
    """
    _cfg = cfg or PhotometricConfig()
    rng = np.random.default_rng(seed)

    ops: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "intensity_gamma": lambda x: _apply_intensity_gamma(x, rng, _cfg),
        "poisson_gaussian": lambda x: _apply_poisson_gaussian(x, rng, _cfg),
        "blur": lambda x: _apply_blur(x, rng, _cfg),
        "bias_field": lambda x: _apply_bias_field(x, rng, _cfg),
    }
    op_names = list(ops.keys())
    weights = np.array(
        [op_weights.get(n, 1.0) if op_weights else 1.0 for n in op_names],
        dtype=np.float32,
    )
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)

    def _choose_ops(k: int) -> list[str]:
        # sample without replacement according to weights
        idx = rng.choice(len(op_names), size=min(k, len(op_names)), replace=False, p=weights)
        chosen = [op_names[i] for i in idx]
        rng.shuffle(chosen)
        return chosen

    def _augment(images: np.ndarray) -> np.ndarray:
        if images.ndim != 4:
            return images
        x = images.astype(np.float32, copy=False)
        n = x.shape[0]
        for i in range(n):
            k = int(rng.integers(min_ops_per_image, max_ops_per_image + 1))
            img = x[i:i+1]
            for name in _choose_ops(k):
                img = ops[name](img)
            x[i:i+1] = np.clip(img, 0, None)
        return x

    return _augment
