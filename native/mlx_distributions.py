
from __future__ import annotations

import math
import mlx.core as mx

def symlog(x: mx.array) -> mx.array:
    
    return mx.sign(x) * mx.log1p(mx.abs(x))

def symexp(x: mx.array) -> mx.array:
    
    return mx.sign(x) * mx.expm1(mx.abs(x))

def masked_mean(values: mx.array, mask: mx.array | None) -> mx.array:
    
    if mask is None:
        return mx.mean(values)
    m = mask.astype(values.dtype)
    while m.ndim < values.ndim:
        m = mx.expand_dims(m, axis=-1)
    denom = mx.maximum(mx.sum(m), mx.array(1.0, dtype=values.dtype))
    return mx.sum(values * m) / denom

def kl(logits_left: mx.array, logits_right: mx.array) -> mx.array:
    
    logits_left  = logits_left.astype(mx.float32)
    logits_right = logits_right.astype(mx.float32)
    log_p = mx.log_softmax(logits_left,  axis=-1)
    log_q = mx.log_softmax(logits_right, axis=-1)
    p     = mx.softmax(logits_left, axis=-1)
    return (p * (log_p - log_q)).sum(axis=-1)

class OneHotDist:
    

    def __init__(self, logits: mx.array, unimix_ratio: float = 0.0):
        logits = logits.astype(mx.float32)
        probs  = mx.softmax(logits, axis=-1)
        if unimix_ratio > 0.0:
            k     = probs.shape[-1]
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / k
        self.logits = mx.log(mx.maximum(probs, 1e-8))
        self._probs = probs

    @property
    def mode(self) -> mx.array:
        idx  = mx.argmax(self.logits, axis=-1)
        hard = mx.eye(self.logits.shape[-1])[idx]
        
        return hard + self.logits - mx.stop_gradient(self.logits)

    def rsample(self, temperature: float = 1.0) -> mx.array:
        g    = mx.random.gumbel(self.logits.shape)
        y    = mx.softmax((self.logits + g) / temperature, axis=-1)
        idx  = mx.argmax(y, axis=-1)
        hard = mx.eye(self.logits.shape[-1])[idx]
        return hard + y - mx.stop_gradient(y)

    def log_prob(self, value: mx.array) -> mx.array:
        log_p = self.logits - mx.logsumexp(self.logits, axis=-1, keepdims=True)
        return (value * log_p).sum(axis=-1)

    def entropy(self) -> mx.array:
        log_p = self.logits - mx.logsumexp(self.logits, axis=-1, keepdims=True)
        return -(self._probs * log_p).sum(axis=-1)

class TwoHot:
    

    def __init__(
        self,
        logits: mx.array,
        bins: mx.array,
        squash=None,
        unsquash=None,
    ):
        self.logits   = logits.astype(mx.float32)
        self.bins     = bins.astype(mx.float32)
        self.probs    = mx.softmax(self.logits, axis=-1)
        self.squash   = squash   if squash   is not None else (lambda x: x)
        self.unsquash = unsquash if unsquash is not None else (lambda x: x)

    def mode(self) -> mx.array:
        return self.unsquash(
            mx.sum(self.probs * self.bins, axis=-1, keepdims=True)
        )

    def log_prob(self, target: mx.array) -> mx.array:
        target    = target.squeeze(-1)
        target_sq = self.squash(mx.stop_gradient(target))

        below = mx.sum(
            (self.bins <= mx.expand_dims(target_sq, -1)).astype(mx.int32),
            axis=-1,
        ) - 1
        above = len(self.bins) - mx.sum(
            (self.bins > mx.expand_dims(target_sq, -1)).astype(mx.int32),
            axis=-1,
        )
        n     = len(self.bins)
        below = mx.clip(below, 0, n - 1)
        above = mx.clip(above, 0, n - 1)
        equal = (below == above)

        bins_below = self.bins[below]
        bins_above = self.bins[above]

        dist_below = mx.where(equal, mx.ones_like(target_sq),  mx.abs(bins_below - target_sq))
        dist_above = mx.where(equal, mx.ones_like(target_sq),  mx.abs(bins_above - target_sq))
        total      = dist_below + dist_above

        weight_below = dist_above / total
        weight_above = dist_below / total

        oh_below = mx.eye(n)[below].astype(mx.float32)
        oh_above = mx.eye(n)[above].astype(mx.float32)

        mixed = (
            oh_below * mx.expand_dims(weight_below, -1)
            + oh_above * mx.expand_dims(weight_above, -1)
        )

        log_pred = self.logits - mx.logsumexp(self.logits, axis=-1, keepdims=True)
        return (mixed * log_pred).sum(axis=-1)

class MSEDist:
    

    def __init__(self, mode: mx.array, agg: str = "sum"):
        self._mode = mode.astype(mx.float32)
        self._agg  = agg

    def mode(self) -> mx.array:  return self._mode
    def mean(self) -> mx.array:  return self._mode

    def log_prob(self, value: mx.array) -> mx.array:
        distance = (self._mode - value.astype(mx.float32)) ** 2
        axes = list(range(2, distance.ndim))
        if not axes:
            return -distance.squeeze(-1) if distance.shape[-1] == 1 else -distance.sum(axis=-1)
        return -(mx.mean(distance, axis=axes) if self._agg == "mean" else mx.sum(distance, axis=axes))

class SymlogDist:
    

    def __init__(
        self,
        mode: mx.array,
        dist: str   = "mse",
        agg:  str   = "sum",
        tol:  float = 1e-8,
    ):
        self._mode = mode.astype(mx.float32)
        self._dist = dist
        self._agg  = agg
        self._tol  = tol

    def mode(self) -> mx.array:  return symexp(self._mode)
    def mean(self) -> mx.array:  return symexp(self._mode)

    def log_prob(self, value: mx.array) -> mx.array:
        value = value.astype(mx.float32)
        if self._dist == "mse":
            d = (self._mode - symlog(value)) ** 2
        elif self._dist == "abs":
            d = mx.abs(self._mode - symlog(value))
        else:
            raise NotImplementedError(self._dist)
        d    = mx.where(d < self._tol, mx.zeros_like(d), d)
        axes = list(range(2, d.ndim))
        if not axes:
            return -d.squeeze(-1) if d.shape[-1] == 1 else -d.sum(axis=-1)
        return -(mx.mean(d, axis=axes) if self._agg == "mean" else mx.sum(d, axis=axes))

class BernoulliDist:
    

    def __init__(self, logits: mx.array):
        self.logits = logits.astype(mx.float32)

    def log_prob(self, value: mx.array) -> mx.array:
        x   = self.logits
        bce = mx.maximum(x, 0) - x * value.astype(mx.float32) + mx.log1p(mx.exp(-mx.abs(x)))
        return -mx.sum(bce, axis=-1)

    def mode(self) -> mx.array:
        return (self.logits > 0).astype(mx.float32)

    def mean(self) -> mx.array:
        return mx.sigmoid(self.logits)

class BoundedNormalDist:
    

    def __init__(
        self,
        x:       mx.array,
        min_std: float = 0.1,
        max_std: float = 1.0,
    ):
        x    = x.astype(mx.float32)
        mean, std_preact = mx.split(x, 2, axis=-1)
        self.std       = (max_std - min_std) * mx.sigmoid(std_preact + 2.0) + min_std
        self.tanh_mean = mx.tanh(mean)

    def mode(self) -> mx.array:
        return self.tanh_mean

    def rsample(self) -> mx.array:
        eps = mx.random.normal(self.tanh_mean.shape)
        return self.tanh_mean + self.std * eps

    def log_prob(self, value: mx.array) -> mx.array:
        value = value.astype(mx.float32)
        var   = self.std ** 2
        lp    = -0.5 * (
            (value - self.tanh_mean) ** 2 / var
            + mx.log(var)
            + math.log(2.0 * math.pi)
        )
        return mx.sum(lp, axis=-1)

    def entropy(self) -> mx.array:
        return mx.sum(
            0.5 + 0.5 * math.log(2.0 * math.pi) + mx.log(self.std),
            axis=-1,
        )

def focal_bce(
    logits:       mx.array,
    targets:      mx.array,
    alpha:        float = 0.8,
    gamma:        float = 2.0,
) -> mx.array:
    
    logits  = logits.astype(mx.float32)
    targets = targets.astype(mx.float32)
    x       = logits
    bce     = mx.maximum(x, 0.0) - x * targets + mx.log1p(mx.exp(-mx.abs(x)))
    p_t     = mx.exp(-bce)
    alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    return mx.mean(alpha_t * ((1.0 - p_t) ** gamma) * bce)

def bounded_normal(x: mx.array, min_std: float = 0.1, max_std: float = 1.0, **_) -> BoundedNormalDist:
    return BoundedNormalDist(x, min_std=min_std, max_std=max_std)

def binary(logits: mx.array, **_) -> BernoulliDist:
    return BernoulliDist(logits)

def symexp_twohot(logits: mx.array, bin_num: int, **_) -> TwoHot:
    
    if bin_num % 2 == 1:
        half = mx.linspace(-20.0, 0.0, (bin_num - 1) // 2 + 1)
        half = symexp(half)
        bins = mx.concatenate([half, mx.flip(-half[:-1], axis=0)])
    else:
        half = mx.linspace(-20.0, 0.0, bin_num // 2)
        half = symexp(half)
        bins = mx.concatenate([half, mx.flip(-half, axis=0)])
    return TwoHot(logits, bins)

def symlog_mse(logits: mx.array, **_) -> SymlogDist:
    return SymlogDist(logits)

def mse(logits: mx.array, **_) -> MSEDist:
    return MSEDist(logits)

def identity(logits: mx.array, **_) -> mx.array:
    return logits

def onehot(logits: mx.array, unimix_ratio: float = 0.0, **_) -> OneHotDist:
    return OneHotDist(logits, unimix_ratio=unimix_ratio)
