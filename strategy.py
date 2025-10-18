"""
CTC Derivatives Trading Game - FFT Strategy (General D, FFT with improved normal fallback, delta-aware)

- Builds a D-length pmf p over faces 1..D from training + current rolls (Dirichlet-smoothed).
- OPTIONS: always quote fair ± spread/2 (no inventory skew).
- FUTURES: quote to reduce portfolio net delta ONLY if |net delta| > delta_thresh.
- Per-product horizon uses *rounds_left* from the product id:
      n_rem = rounds_left * rolls_per_subround
      S_t   = sum(current_rolls)  # sum of all observed rolls so far

Expected product ids:
- Futures: "S,F,ROUNDS_LEFT"
- Calls:   "S,C,STRIKE,ROUNDS_LEFT"
- Puts:    "S,P,STRIKE,ROUNDS_LEFT"
"""

from autograder.sdk.strategy_interface import AbstractTradingStrategy
import numpy as np
import scipy
from typing import Any, Dict, Tuple, List

# ============ Utilities ============

def next_pow2(n: int) -> int:
    """Next power of two >= n."""
    m = 1
    while m < n:
        m <<= 1
    return m

def _ravel_ints(seq: Any) -> np.ndarray:
    """Flatten any array-like into a 1-D numpy array of ints."""
    if seq is None:
        return np.array([], dtype=int)
    return np.asarray(seq, dtype=int).ravel()

def _estimate_pD(training_rolls: Any, current_rolls: Any, D: int, alpha: float = 1.0) -> np.ndarray:
    """
    Dirichlet(alpha)-smoothed pmf over faces 1..D from training + current rolls.
    Rolls outside 1..D are ignored.
    """
    counts = np.full(D, float(alpha), dtype=np.float64)
    for seq in (_ravel_ints(training_rolls), _ravel_ints(current_rolls)):
        if seq.size == 0:
            continue
        mask = (seq >= 1) & (seq <= D)
        if np.any(mask):
            hist = np.bincount(seq[mask], minlength=D + 1)  # 0..D
            counts += hist[1:D + 1]
    s = float(counts.sum())
    if s > 0 and np.isfinite(s):
        return (counts / s).astype(np.float32, copy=False)  # pmf as float32
    return np.full(D, 1.0 / D, dtype=np.float32)

# One-roll central moments from pmf p(1..D)
def _central_moments_one_roll(p: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Returns (mu1, var1, skew1, exkurt1) for the 1-roll distribution on {1..D}.
    """
    D = p.size
    x = np.arange(1, D + 1, dtype=np.float64)
    p64 = p.astype(np.float64, copy=False)
    mu = float(np.dot(x, p64))
    c = x - mu
    m2 = float(np.dot(c * c, p64))
    if m2 <= 0.0:
        return mu, 0.0, 0.0, 0.0
    m3 = float(np.dot(c**3, p64))
    m4 = float(np.dot(c**4, p64))
    skew = m3 / (m2 ** 1.5)
    exkurt = m4 / (m2 * m2) - 3.0
    return mu, m2, skew, exkurt

# ============ FFT path (prefix-sum pricing) ============

def _remaining_sum_pmf_fft(p: np.ndarray, n_rem: int) -> np.ndarray:
    """
    PMF of remaining sum R over n_rem rolls of D-sided die with pmf p (len D).
    Returns f where f[r] = P(R = r) for r = 0..D*n_rem (FFT grid, zero-padded after).
    """
    n_rem = max(0, int(n_rem))
    if n_rem == 0:
        return np.array([1.0], dtype=np.float32)

    p = np.asarray(p, dtype=np.float32).reshape(-1)
    s = float(p.sum())
    if s <= 0 or not np.isfinite(s):
        raise ValueError("invalid probability vector")
    p /= s
    D = p.size

    L = D * n_rem + 1
    M = next_pow2(L)

    # float32 time-domain buffer
    g = np.zeros(M, dtype=np.float32)
    g[1:D+1] = p  # faces 1..D -> indices 1..D

    # complex64 spectrum
    G = np.fft.fft(g)
    if G.dtype != np.complex64:
        G = G.astype(np.complex64, copy=False)

    # convolution by power in freq domain
    np.power(G, n_rem, out=G)

    # back to time domain (float32)
    f = np.fft.ifft(G).real
    if f.dtype != np.float32:
        f = f.astype(np.float32, copy=False)

    # clean & normalize
    f[f < 0] = 0.0
    tot = float(f.sum())
    if tot > 0:
        f /= tot
    else:
        f[:] = 0.0
        f[0] = 1.0
    return f  # float32

def _pmf_triplet(p: np.ndarray, n_rem: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute (f, cdf, pr) where:
      f[r]   = P(R = r)
      cdf[r] = sum_{k<=r} f[k]
      pr[r]  = sum_{k<=r} k * f[k]
    Uses float32 for f, float64 for cumulative arrays for stability.
    """
    f = _remaining_sum_pmf_fft(p, n_rem)
    idx = np.arange(f.size, dtype=np.float64)
    cdf = np.cumsum(f, dtype=np.float64)
    pr = np.cumsum(idx * f.astype(np.float64), dtype=np.float64)
    return f, cdf, pr

def _call_price_from_prefix(S_t: float, K: float, cdf: np.ndarray, pr: np.ndarray) -> float:
    t = int(np.floor(K - S_t))
    t = np.clip(t, -1, cdf.size - 1)
    tail_prob = 1.0 - (0.0 if t < 0 else cdf[t])
    tail_pr   = pr[-1] - (0.0 if t < 0 else pr[t])
    return float((S_t - K) * tail_prob + tail_pr)

def _put_price_from_prefix(S_t: float, K: float, cdf: np.ndarray, pr: np.ndarray) -> float:
    u = int(np.ceil(K - S_t) - 1.0)
    u = np.clip(u, -1, cdf.size - 1)
    left_prob = 0.0 if u < 0 else cdf[u]
    left_pr   = 0.0 if u < 0 else pr[u]
    return float((K - S_t) * left_prob - left_pr)

def _call_delta_from_prefix(S_t: float, K: float, cdf: np.ndarray) -> float:
    t = int(np.floor(K - S_t))
    t = np.clip(t, -1, cdf.size - 1)
    return float(1.0 - (0.0 if t < 0 else cdf[t]))

def _put_delta_from_prefix(S_t: float, K: float, cdf: np.ndarray) -> float:
    u = int(np.ceil(K - S_t) - 1.0)
    u = np.clip(u, -1, cdf.size - 1)
    return float(-(0.0 if u < 0 else cdf[u]))

# ============ Improved Normal fallback path ============

_SQRT2 = np.sqrt(2.0)
_SQRT2PI = np.sqrt(2.0 * np.pi)

def _Phi(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + scipy.special.erf(x / _SQRT2))

def _phi(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / _SQRT2PI

def _edgeworth_Phi(d: float, skew_R: float, ek_R: float) -> float:
    """Edgeworth-corrected CDF Φ_E(d); safe for modest corrections."""
    Phi = 0.5 * (1.0 + scipy.special.erf(d / np.sqrt(2.0)))
    phi = np.exp(-0.5 * d * d) / np.sqrt(2.0 * np.pi)
    c1 = (skew_R / 6.0) * (1.0 - d * d) * phi
    c2 = (ek_R   / 24.0) * (d**3 - 3.0 * d) * phi
    c3 = (skew_R * skew_R / 72.0) * (d**5 - 10.0 * d**3 + 15.0 * d) * phi
    PhiE = Phi + c1 + c2 + c3
    return float(min(1.0, max(0.0, PhiE)))

def _normal_call_put_and_deltas(
    S_t: float, K: float, n_rem: int, mu1: float, var1: float,
    *, D: int = None, continuity: float = 0.5,
    use_edgeworth: bool = False, skew1: float = 0.0, exkurt1: float = 0.0,
    var_inflation: float = 0.0
) -> Tuple[float, float, float, float]:
    """
    Improved normal fallback with continuity correction, support clamps, optional Edgeworth, and variance stabilizers.
    Returns (call, put, call_delta, put_delta).
    """
    n_rem = max(0, int(n_rem))
    m = n_rem * mu1
    v = max(0.0, n_rem * var1) * (1.0 + var_inflation)
    muY = S_t + m

    # bounds of the lattice sum Y in [L, U]
    if D is not None and n_rem > 0:
        L = S_t + n_rem * 1.0
        U = S_t + n_rem * float(D)
        if K >= U:
            # call 0, put linear segment
            return 0.0, max(K - muY, 0.0), 0.0, -1.0
        if K <= L:
            # call linear segment, put 0
            c = muY - K
            return c, 0.0, 1.0, 0.0

    if v <= 1e-12:
        call = max(muY - K, 0.0)
        put  = max(K - muY, 0.0)
        call_delta = 1.0 if muY >= K else 0.0
        put_delta  = call_delta - 1.0
        return float(call), float(put), float(call_delta), float(put_delta)

    sigma = max(np.sqrt(v), 1e-6)

    # continuity correction
    d_call = (muY - (K - continuity)) / sigma
    d_put  = (muY - (K + continuity)) / sigma

    if use_edgeworth and n_rem > 0:
        skew_R = skew1 / np.sqrt(n_rem)
        ek_R   = exkurt1 / max(1, n_rem)
        Phi_c = _edgeworth_Phi(d_call, skew_R, ek_R)
        Phi_p = _edgeworth_Phi(d_put,  skew_R, ek_R)
    else:
        Phi_c = float(_Phi(d_call))
        Phi_p = float(_Phi(d_put))

    phi_c = float(_phi(d_call))
    phi_p = float(_phi(d_put))

    call = (muY - K) * Phi_c + sigma * phi_c
    put  = (K - muY) * (1.0 - Phi_p) + sigma * phi_p
    call_delta = Phi_c
    put_delta  = Phi_p - 1.0

    # final safety clamps against the finite support
    if D is not None and n_rem > 0:
        L = S_t + n_rem * 1.0
        U = S_t + n_rem * float(D)
        call = float(min(max(call, 0.0), U - K))
        put  = float(min(max(put, 0.0), K - L))

    return float(call), float(put), float(call_delta), float(put_delta)

# ============ Dispatcher: choose FFT or Normal based on size ============

def _use_fft(D: int, n_rem: int, max_fft_points: int) -> bool:
    if n_rem <= 0:
        return True
    L = D * n_rem + 1
    M = next_pow2(L)
    return M <= max_fft_points

# Public-like wrappers that may use FFT or improved Normal
def call_price_auto(p: np.ndarray, K: float, S_t: float, n_rem: int,
                    pmf_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                    mu1: float, var1: float, skew1: float, exkurt1: float,
                    D: int, max_fft_points: int, *,
                    continuity: float, use_edgeworth: bool, var_inflation: float) -> float:
    if _use_fft(D, n_rem, max_fft_points):
        if n_rem not in pmf_cache:
            pmf_cache[n_rem] = _pmf_triplet(p, n_rem)
        _, cdf, pr = pmf_cache[n_rem]
        return _call_price_from_prefix(S_t, K, cdf, pr)
    else:
        call, _, _, _ = _normal_call_put_and_deltas(
            S_t, K, n_rem, mu1, var1,
            D=D, continuity=continuity, use_edgeworth=use_edgeworth,
            skew1=skew1, exkurt1=exkurt1, var_inflation=var_inflation
        )
        return call

def put_price_auto(p: np.ndarray, K: float, S_t: float, n_rem: int,
                   pmf_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                   mu1: float, var1: float, skew1: float, exkurt1: float,
                   D: int, max_fft_points: int, *,
                   continuity: float, use_edgeworth: bool, var_inflation: float) -> float:
    if _use_fft(D, n_rem, max_fft_points):
        if n_rem not in pmf_cache:
            pmf_cache[n_rem] = _pmf_triplet(p, n_rem)
        _, cdf, pr = pmf_cache[n_rem]
        return _put_price_from_prefix(S_t, K, cdf, pr)
    else:
        _, put, _, _ = _normal_call_put_and_deltas(
            S_t, K, n_rem, mu1, var1,
            D=D, continuity=continuity, use_edgeworth=use_edgeworth,
            skew1=skew1, exkurt1=exkurt1, var_inflation=var_inflation
        )
        return put

def call_delta_auto(p: np.ndarray, K: float, S_t: float, n_rem: int,
                    pmf_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                    mu1: float, var1: float, skew1: float, exkurt1: float,
                    D: int, max_fft_points: int, *,
                    continuity: float, use_edgeworth: bool, var_inflation: float) -> float:
    if _use_fft(D, n_rem, max_fft_points):
        if n_rem not in pmf_cache:
            pmf_cache[n_rem] = _pmf_triplet(p, n_rem)
        _, cdf, _ = pmf_cache[n_rem]
        return _call_delta_from_prefix(S_t, K, cdf)
    else:
        _, _, cdelta, _ = _normal_call_put_and_deltas(
            S_t, K, n_rem, mu1, var1,
            D=D, continuity=continuity, use_edgeworth=use_edgeworth,
            skew1=skew1, exkurt1=exkurt1, var_inflation=var_inflation
        )
        return cdelta

def put_delta_auto(p: np.ndarray, K: float, S_t: float, n_rem: int,
                   pmf_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                   mu1: float, var1: float, skew1: float, exkurt1: float,
                   D: int, max_fft_points: int, *,
                   continuity: float, use_edgeworth: bool, var_inflation: float) -> float:
    if _use_fft(D, n_rem, max_fft_points):
        if n_rem not in pmf_cache:
            pmf_cache[n_rem] = _pmf_triplet(p, n_rem)
        _, cdf, _ = pmf_cache[n_rem]
        return _put_delta_from_prefix(S_t, K, cdf)
    else:
        _, _, _, pdelta = _normal_call_put_and_deltas(
            S_t, K, n_rem, mu1, var1,
            D=D, continuity=continuity, use_edgeworth=use_edgeworth,
            skew1=skew1, exkurt1=exkurt1, var_inflation=var_inflation
        )
        return pdelta

def futures_price_sum(p: np.ndarray, S_t: float, n_rem: int, multiplier: float = 1.0) -> float:
    """E[final sum] = S_t + n_rem * E[one roll]. (Exact from pmf mean; no FFT needed.)"""
    p = np.asarray(p, dtype=np.float32).reshape(-1)
    p = p / float(p.sum())
    D = p.size
    faces = np.arange(1, D + 1, dtype=np.float64)  # small array; keep float64
    mu = float(np.dot(faces, p.astype(np.float64)))
    return float(multiplier * (S_t + max(0, int(n_rem)) * mu))

# ============ Strategy (delta-aware, FFT with improved Normal fallback) ============

class MyTradingStrategy(AbstractTradingStrategy):
    """
    - D-length pmf from data (Dirichlet-smoothed).
    - OPTIONS: fair ± spread/2 (no inventory skew).
    - FUTURES: hedge net delta only if |net_delta| > delta_thresh.
    """

    def __init__(self):
        self.dice_sides = 10000
        self.spread_width = 10
        self.dirichlet_alpha = 1.0
        self.rolls_per_subround = 2000  # horizon: rounds_left -> rounds_left*rolls_per_subround
        self.delta_thresh = 5
        self.team_name = "Unknown"
        self.last_p = None  # debug

        # FFT cap (tunable): if next_pow2(D*n_rem+1) > cap, use Normal fallback
        self.max_fft_points = 1 << 26  # ≈67M points

        # Normal fallback tuning
        self.continuity_correction = 0.5
        self.var_inflation = 0.0         # e.g., 0.05 to be conservative when data are thin
        self.use_edgeworth_base = True   # allow Edgeworth
        self.edgeworth_nrem_max = 50000  # only use Edgeworth for moderate horizons

    def on_game_start(self, config: Dict[str, Any]) -> None:
        self.dice_sides = int(config.get("dice_sides", self.dice_sides))
        self.team_name = config.get("team_name", "Unknown")
        self.rolls_per_subround = int(config.get("rolls_per_subround", self.rolls_per_subround))
        self.max_fft_points = int(config.get("max_fft_points", self.max_fft_points))
        self.var_inflation = float(config.get("var_inflation", self.var_inflation))
        self.edgeworth_nrem_max = int(config.get("edgeworth_nrem_max", self.edgeworth_nrem_max))
        self.use_edgeworth_base = bool(config.get("use_edgeworth", self.use_edgeworth_base))
        np.random.seed(int(config.get("seed", 42)))
        print(f"[INFO] Strategy {self.team_name}: D={self.dice_sides}, rolls/subround={self.rolls_per_subround}, max_fft_points={self.max_fft_points}")

    def make_market(
        self, *, marketplace: Any, training_rolls: Any, my_trades: Any,
        current_rolls: Any, round_info: Any
    ) -> Dict[str, Tuple[float, float]]:

        D = int(self.dice_sides)
        half = float(self.spread_width) * 0.5

        # pmf p over 1..D from training + live (float32)
        pD = _estimate_pD(training_rolls, current_rolls, D=D, alpha=self.dirichlet_alpha)
        self.last_p = pD

        # one-roll moments (incl. skew/kurtosis) for Improved Normal fallback
        mu1, var1, skew1, exkurt1 = _central_moments_one_roll(pD)

        # state at "now"
        current_arr = _ravel_ints(current_rolls)
        S_now = float(np.sum(current_arr)) if current_arr.size > 0 else 0.0

        # compute (S_t, n_rem) from *rounds_left*
        def St_nrem_from_rounds_left(rounds_left: int) -> Tuple[float, int]:
            rl = max(0, int(rounds_left))
            n_rem_val = rl * self.rolls_per_subround
            return S_now, n_rem_val

        # per-n_rem pmf cache (for FFT path) for this tick
        pmf_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        products = list(marketplace.get_products())
        quotes: Dict[str, Tuple[float, float]] = {}

        for product in products:
            pid = getattr(product, "id", "")
            parts = pid.split(",")
            if len(parts) < 3:
                continue

            kind = parts[1].upper()
            try:
                if kind == "F":
                    # id: "S,F,ROUNDS_LEFT"
                    rounds_left = int(parts[2])
                    S_t, n_rem = St_nrem_from_rounds_left(rounds_left)
                    fair = futures_price_sum(pD, S_t, n_rem)

                    # delta-aware futures quoting
                    net_d = self._expiry_net_delta(products, my_trades, pD, rounds_left,
                                                   St_nrem_from_rounds_left, pmf_cache, mu1, var1,
                                                   skew1, exkurt1)
                    edge = 0.025
                    adj = 0.01
                    if net_d > self.delta_thresh:
                        bid = (1 - edge) * fair
                        ask = (1 + edge - adj) * fair
                    elif net_d < -self.delta_thresh:
                        bid = (1 - edge + adj) * fair
                        ask = (1 + edge) * fair
                    else:
                        bid = (1 - edge) * fair
                        ask = (1 + edge) * fair
                    quotes[pid] = (float(bid), float(ask))

                elif kind in ("C", "P"):
                    # id: "S,C,STRIKE,ROUNDS_LEFT" / "S,P,STRIKE,ROUNDS_LEFT"
                    if len(parts) < 4:
                        continue
                    K = float(parts[2])
                    rounds_left = int(parts[3])

                    S_t, n_rem = St_nrem_from_rounds_left(rounds_left)

                    # decide if we want Edgeworth at this horizon
                    use_edge = self.use_edgeworth_base and (n_rem <= self.edgeworth_nrem_max)

                    if kind == "C":
                        fair = call_price_auto(
                            pD, K, S_t, n_rem, pmf_cache,
                            mu1, var1, skew1, exkurt1, D, self.max_fft_points,
                            continuity=self.continuity_correction,
                            use_edgeworth=use_edge,
                            var_inflation=self.var_inflation
                        )
                    else:
                        fair = put_price_auto(
                            pD, K, S_t, n_rem, pmf_cache,
                            mu1, var1, skew1, exkurt1, D, self.max_fft_points,
                            continuity=self.continuity_correction,
                            use_edgeworth=use_edge,
                            var_inflation=self.var_inflation
                        )

                    edge = 0.025
                    bid = (1 - edge) * fair
                    ask = (1 + edge) * fair
                    if bid >= ask:
                        ask = bid + max(1e-6, self.spread_width * 1e-6)
                    quotes[pid] = (float(bid), float(ask))

            except Exception as e:
                print(f"[WARN] Skipping {pid}: {e}")
                continue

        return quotes

    # ---- Net delta for an expiry (futures + options) ----
    def _expiry_net_delta(
        self,
        products: List[Any],
        my_trades: Any,
        pD: np.ndarray,
        rounds_left: int,  # keyed by rounds_left
        St_nrem_func,      # rounds_left -> (S_t, n_rem)
        pmf_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        mu1: float,
        var1: float,
        skew1: float,
        exkurt1: float,
    ) -> float:
        S_t, n_rem = St_nrem_func(rounds_left)
        net_d = 0.0
        D = pD.size

        # decide if we want Edgeworth here (same rule as pricing)
        use_edge = self.use_edgeworth_base and (n_rem <= self.edgeworth_nrem_max)

        for prod in products:
            pid = getattr(prod, "id", "")
            parts = pid.split(",")
            if len(parts) < 3:
                continue
            typ = parts[1].upper()
            try:
                if typ == "F":
                    rl = int(parts[2])
                    if rl != rounds_left:
                        continue
                    pos = my_trades.get_position(pid)
                    q = 0.0 if pos is None else float(pos.position)
                    net_d += q * 1.0
                elif typ in ("C", "P"):
                    # "S,C,STRIKE,ROUNDS_LEFT" / "S,P,STRIKE,ROUNDS_LEFT"
                    if len(parts) < 4:
                        continue
                    rl = int(parts[3])
                    if rl != rounds_left:
                        continue
                    K = float(parts[2])
                    pos = my_trades.get_position(pid)
                    q = 0.0 if pos is None else float(pos.position)
                    if q == 0.0:
                        continue
                    if typ == "C":
                        d = call_delta_auto(
                            pD, K, S_t, n_rem, pmf_cache,
                            mu1, var1, skew1, exkurt1, D, self.max_fft_points,
                            continuity=self.continuity_correction,
                            use_edgeworth=use_edge,
                            var_inflation=self.var_inflation
                        )
                    else:
                        d = put_delta_auto(
                            pD, K, S_t, n_rem, pmf_cache,
                            mu1, var1, skew1, exkurt1, D, self.max_fft_points,
                            continuity=self.continuity_correction,
                            use_edgeworth=use_edge,
                            var_inflation=self.var_inflation
                        )
                    net_d += q * d
            except Exception:
                continue

        return float(net_d)

    # ---- Debug hooks ----
    def on_round_end(self, result: Dict[str, Any]) -> None:
        pnl = result.get("pnl", 0.0)
        dice = result.get("dice_rolls", [])
        print(f"[ROUND END] PnL: {pnl:.2f}, Dice(first 10): {dice[:10]}")

    def on_game_end(self, summary: Dict[str, Any]) -> None:
        total_pnl = summary.get("total_pnl", 0.0)
        final_score = summary.get("final_score", 0.0)
        print(f"[GAME END] Total PnL: {total_pnl:.2f}, Score: {final_score:.2f}")

Strategy = MyTradingStrategy
STRATEGY_CLASS = MyTradingStrategy
__all__ = ["MyTradingStrategy", "Strategy", "STRATEGY_CLASS"]
