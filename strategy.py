"""
CTC Derivatives Trading Game - Enhanced Strategy with DynamicFutures Support

Features:
- Dirichlet-smoothed PMF estimation from training + live data
- FFT-based exact pricing for small horizons, Normal approximation for large horizons
- DynamicFutures: contracts with expiries 1-9 subrounds, tradeable even after expiry
- Position-aware quoting with inventory management
- Delta-neutral hedging across all products
- Dynamic spreads based on contract state (expired vs unexpired)

Product ID formats:
- Standard Futures: "S,F,ROUNDS_LEFT"
- Standard Calls:   "S,C,STRIKE,ROUNDS_LEFT"
- Standard Puts:    "S,P,STRIKE,ROUNDS_LEFT"
- DynamicFutures:   "S,DF,EXPIRY" (EXPIRY ∈ {1..9} subrounds)
"""

from autograder.sdk.strategy_interface import AbstractTradingStrategy
import numpy as np
import scipy.special
from typing import Any, Dict, Tuple, List
from dataclasses import dataclass
from collections import defaultdict

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
        return (counts / s).astype(np.float32, copy=False)
    return np.full(D, 1.0 / D, dtype=np.float32)

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

    g = np.zeros(M, dtype=np.float32)
    g[1:D+1] = p

    G = np.fft.fft(g)
    if G.dtype != np.complex64:
        G = G.astype(np.complex64, copy=False)

    np.power(G, n_rem, out=G)

    f = np.fft.ifft(G).real
    if f.dtype != np.float32:
        f = f.astype(np.float32, copy=False)

    f[f < 0] = 0.0
    tot = float(f.sum())
    if tot > 0:
        f /= tot
    else:
        f[:] = 0.0
        f[0] = 1.0
    return f

def _pmf_triplet(p: np.ndarray, n_rem: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute (f, cdf, pr) where:
      f[r]   = P(R = r)
      cdf[r] = sum_{k<=r} f[k]
      pr[r]  = sum_{k<=r} k * f[k]
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
    """Edgeworth-corrected CDF."""
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
    Improved normal fallback with continuity correction.
    Returns (call, put, call_delta, put_delta).
    """
    n_rem = max(0, int(n_rem))
    m = n_rem * mu1
    v = max(0.0, n_rem * var1) * (1.0 + var_inflation)
    muY = S_t + m

    if D is not None and n_rem > 0:
        L = S_t + n_rem * 1.0
        U = S_t + n_rem * float(D)
        if K >= U:
            return 0.0, max(K - muY, 0.0), 0.0, -1.0
        if K <= L:
            c = muY - K
            return c, 0.0, 1.0, 0.0

    if v <= 1e-12:
        call = max(muY - K, 0.0)
        put  = max(K - muY, 0.0)
        call_delta = 1.0 if muY >= K else 0.0
        put_delta  = call_delta - 1.0
        return float(call), float(put), float(call_delta), float(put_delta)

    sigma = max(np.sqrt(v), 1e-6)

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

    if D is not None and n_rem > 0:
        L = S_t + n_rem * 1.0
        U = S_t + n_rem * float(D)
        call = float(min(max(call, 0.0), U - K))
        put  = float(min(max(put, 0.0), K - L))

    return float(call), float(put), float(call_delta), float(put_delta)

# ============ Dispatcher: choose FFT or Normal ============

def _use_fft(D: int, n_rem: int, max_fft_points: int) -> bool:
    if n_rem <= 0:
        return True
    L = D * n_rem + 1
    M = next_pow2(L)
    return M <= max_fft_points

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
    """E[final sum] = S_t + n_rem * E[one roll]."""
    p = np.asarray(p, dtype=np.float32).reshape(-1)
    p = p / float(p.sum())
    D = p.size
    faces = np.arange(1, D + 1, dtype=np.float64)
    mu = float(np.dot(faces, p.astype(np.float64)))
    return float(multiplier * (S_t + max(0, int(n_rem)) * mu))

# ============ DynamicFutures Helper Functions ============

def compute_subround_roll_sums(current_rolls: Any, rolls_per_subround: int) -> Dict[int, float]:
    """
    Pre-compute cumulative sums for each subround.
    Returns dict: subround -> sum of first (subround * rolls_per_subround) rolls
    """
    rolls = _ravel_ints(current_rolls)
    if rolls.size == 0:
        return {}

    sums = {}
    for subround in range(1, 10):
        end_idx = subround * rolls_per_subround
        if end_idx <= rolls.size:
            sums[subround] = float(np.sum(rolls[:end_idx]))
    return sums

# ============ Enhanced Strategy with DynamicFutures ============

class MyTradingStrategy(AbstractTradingStrategy):
    """
    Enhanced trading strategy with:
    - DynamicFutures support (expired and unexpired)
    - Position-aware quoting with inventory limits
    - Delta-neutral hedging across all products
    - Dynamic spread management
    """

    def __init__(self):
        # Game parameters
        self.dice_sides = 10000
        self.rolls_per_subround = 2000
        self.team_name = "Unknown"

        # Pricing parameters
        self.dirichlet_alpha = 1.0
        self.max_fft_points = 1 << 24  # Reduced from 26 to 24 for faster computation
        self.continuity_correction = 0.5
        self.var_inflation = 0.0
        self.use_edgeworth_base = False  # Disabled for speed
        self.edgeworth_nrem_max = 50000

        # Trading parameters
        self.position_limit = 25  # Max position per product (increased for more liquidity)
        self.delta_hedge_threshold = 15  # Hedge when |net_delta| > this

        # Spread parameters (tightened for more competitive pricing)
        self.spread_expired_df = 0.001      # 0.1% for expired DynamicFutures (very tight)
        self.spread_unexpired_df = 0.025    # 2.5% for unexpired DynamicFutures
        self.spread_standard_options = 0.035 # 3.5% for standard options
        self.spread_standard_futures = 0.02  # 2% for standard futures

        # Inventory skew parameters
        self.inventory_skew_factor = 0.008  # 0.8% per unit of inventory (reduced for more trades)

        # State tracking
        self.current_subround = 1
        self.subround_sums = {}  # Cache of cumulative sums by subround

        # Performance optimization: cache last PMF to avoid recomputation
        self._last_pmf_hash = None
        self._last_pmf = None

    def on_game_start(self, config: Dict[str, Any]) -> None:
        self.dice_sides = int(config.get("dice_sides", self.dice_sides))
        self.team_name = config.get("team_name", "Unknown")
        self.rolls_per_subround = int(config.get("rolls_per_subround", self.rolls_per_subround))

        # Allow config overrides
        self.max_fft_points = int(config.get("max_fft_points", self.max_fft_points))
        self.var_inflation = float(config.get("var_inflation", self.var_inflation))
        self.position_limit = int(config.get("position_limit", self.position_limit))

        np.random.seed(int(config.get("seed", 42)))
        print(f"[INFO] Enhanced Strategy: team={self.team_name}, D={self.dice_sides}, "
              f"pos_limit={self.position_limit}, rolls/subround={self.rolls_per_subround}")

    def make_market(
        self, *, marketplace: Any, training_rolls: Any, my_trades: Any,
        current_rolls: Any, round_info: Any
    ) -> Dict[str, Tuple[float, float]]:

        # Extract round info
        self.current_subround = int(round_info.current_sub_round)

        # Build PMF from data (with caching for performance)
        D = int(self.dice_sides)
        current_arr = _ravel_ints(current_rolls)

        # Simple hash of current data to detect if we can reuse cached PMF
        data_hash = (len(current_arr), float(np.sum(current_arr)) if current_arr.size > 0 else 0.0)

        if self._last_pmf_hash == data_hash and self._last_pmf is not None:
            # Reuse cached PMF
            pD = self._last_pmf
        else:
            # Compute new PMF
            pD = _estimate_pD(training_rolls, current_rolls, D=D, alpha=self.dirichlet_alpha)
            self._last_pmf = pD
            self._last_pmf_hash = data_hash

        mu1, var1, skew1, exkurt1 = _central_moments_one_roll(pD)

        # Pre-compute subround sums for DynamicFutures pricing
        self.subround_sums = compute_subround_roll_sums(current_rolls, self.rolls_per_subround)

        # Current sum of all rolls observed (already computed current_arr above)
        S_now = float(np.sum(current_arr)) if current_arr.size > 0 else 0.0

        # PMF cache for this tick
        pmf_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        # Get all products
        products = list(marketplace.get_products())
        quotes: Dict[str, Tuple[float, float]] = {}

        # First pass: calculate portfolio delta
        total_delta = self._calculate_portfolio_delta(
            products, my_trades, pD, mu1, var1, skew1, exkurt1,
            S_now, pmf_cache, D
        )

        # Process each product
        for product in products:
            pid = getattr(product, "id", "")
            parts = pid.split(",")
            if len(parts) < 3:
                continue

            try:
                kind = parts[1].upper()

                if kind == "DF":
                    # DynamicFutures: "S,DF,EXPIRY"
                    expiry_subround = int(parts[2])
                    bid, ask = self._price_dynamic_future(
                        pid, expiry_subround, pD, mu1, my_trades, S_now, total_delta
                    )
                    quotes[pid] = (bid, ask)

                elif kind == "F":
                    # Standard Futures: "S,F,ROUNDS_LEFT"
                    rounds_left = int(parts[2])
                    n_rem = rounds_left * self.rolls_per_subround
                    fair = futures_price_sum(pD, S_now, n_rem)

                    # Position-aware spread with delta hedging
                    position = self._get_position(my_trades, pid)
                    spread = self.spread_standard_futures

                    # Apply delta hedging adjustment
                    bid, ask = self._apply_spread_skew_and_hedge(
                        fair, spread, position, total_delta, delta_per_contract=1.0
                    )
                    quotes[pid] = (bid, ask)

                elif kind in ("C", "P"):
                    # Standard Options: "S,C,STRIKE,ROUNDS_LEFT" or "S,P,STRIKE,ROUNDS_LEFT"
                    if len(parts) < 4:
                        continue
                    K = float(parts[2])
                    rounds_left = int(parts[3])
                    n_rem = rounds_left * self.rolls_per_subround

                    use_edge = self.use_edgeworth_base and (n_rem <= self.edgeworth_nrem_max)

                    if kind == "C":
                        fair = call_price_auto(
                            pD, K, S_now, n_rem, pmf_cache,
                            mu1, var1, skew1, exkurt1, D, self.max_fft_points,
                            continuity=self.continuity_correction,
                            use_edgeworth=use_edge,
                            var_inflation=self.var_inflation
                        )
                        delta = call_delta_auto(
                            pD, K, S_now, n_rem, pmf_cache,
                            mu1, var1, skew1, exkurt1, D, self.max_fft_points,
                            continuity=self.continuity_correction,
                            use_edgeworth=use_edge,
                            var_inflation=self.var_inflation
                        )
                    else:
                        fair = put_price_auto(
                            pD, K, S_now, n_rem, pmf_cache,
                            mu1, var1, skew1, exkurt1, D, self.max_fft_points,
                            continuity=self.continuity_correction,
                            use_edgeworth=use_edge,
                            var_inflation=self.var_inflation
                        )
                        delta = put_delta_auto(
                            pD, K, S_now, n_rem, pmf_cache,
                            mu1, var1, skew1, exkurt1, D, self.max_fft_points,
                            continuity=self.continuity_correction,
                            use_edgeworth=use_edge,
                            var_inflation=self.var_inflation
                        )

                    position = self._get_position(my_trades, pid)
                    spread = self.spread_standard_options
                    bid, ask = self._apply_spread_skew_and_hedge(
                        fair, spread, position, total_delta, delta_per_contract=delta
                    )
                    quotes[pid] = (bid, ask)

            except Exception as e:
                # Skip problematic products
                print(f"[WARN] Error pricing {pid}: {e}")
                continue

        return quotes

    def _price_dynamic_future(
        self, pid: str, expiry_subround: int, pD: np.ndarray, mu1: float,
        my_trades: Any, S_now: float, total_delta: float
    ) -> Tuple[float, float]:
        """
        Price a DynamicFutures contract.
        - If expired: price is known exactly
        - If unexpired: price is expected value
        """
        position = self._get_position(my_trades, pid)

        if expiry_subround < self.current_subround:
            # Contract has expired - underlying is deterministic
            if expiry_subround in self.subround_sums:
                fair = self.subround_sums[expiry_subround]
            else:
                # Shouldn't happen, but fallback
                fair = S_now

            # Very tight spread for expired contracts
            spread = self.spread_expired_df
            bid, ask = self._apply_spread_skew_and_hedge(
                fair, spread, position, total_delta, delta_per_contract=1.0
            )

        else:
            # Contract not yet expired
            rolls_observed = (self.current_subround - 1) * self.rolls_per_subround
            rolls_to_expiry = expiry_subround * self.rolls_per_subround
            rolls_remaining = max(0, rolls_to_expiry - rolls_observed)

            # Expected value = current sum + expected remaining
            expected_remaining = rolls_remaining * mu1
            fair = S_now + expected_remaining

            # Wider spread for unexpired contracts
            spread = self.spread_unexpired_df

            # Adjust spread based on time to expiry (tighter as we approach expiry)
            time_factor = rolls_remaining / (self.rolls_per_subround * 9)  # Normalize to [0,1]
            adjusted_spread = spread * (0.3 + 0.7 * time_factor)  # Min 30% of base spread

            bid, ask = self._apply_spread_skew_and_hedge(
                fair, adjusted_spread, position, total_delta, delta_per_contract=1.0
            )

        return bid, ask

    def _calculate_portfolio_delta(
        self, products: List[Any], my_trades: Any, pD: np.ndarray,
        mu1: float, var1: float, skew1: float, exkurt1: float,
        S_now: float, pmf_cache: Dict[int, Tuple], D: int
    ) -> float:
        """
        Calculate total portfolio delta across all positions.
        Delta represents exposure to underlying price movements.
        """
        total_delta = 0.0

        for product in products:
            pid = getattr(product, "id", "")
            parts = pid.split(",")
            if len(parts) < 3:
                continue

            position = self._get_position(my_trades, pid)
            if abs(position) < 1e-6:
                continue

            try:
                kind = parts[1].upper()

                if kind in ("F", "DF"):
                    # Futures have delta = 1.0
                    total_delta += position * 1.0

                elif kind in ("C", "P"):
                    # Options have varying deltas
                    if len(parts) < 4:
                        continue
                    K = float(parts[2])
                    rounds_left = int(parts[3])
                    n_rem = rounds_left * self.rolls_per_subround

                    use_edge = self.use_edgeworth_base and (n_rem <= self.edgeworth_nrem_max)

                    if kind == "C":
                        delta = call_delta_auto(
                            pD, K, S_now, n_rem, pmf_cache,
                            mu1, var1, skew1, exkurt1, D, self.max_fft_points,
                            continuity=self.continuity_correction,
                            use_edgeworth=use_edge,
                            var_inflation=self.var_inflation
                        )
                    else:
                        delta = put_delta_auto(
                            pD, K, S_now, n_rem, pmf_cache,
                            mu1, var1, skew1, exkurt1, D, self.max_fft_points,
                            continuity=self.continuity_correction,
                            use_edgeworth=use_edge,
                            var_inflation=self.var_inflation
                        )

                    total_delta += position * delta

            except Exception:
                continue

        return float(total_delta)

    def _get_position(self, my_trades: Any, pid: str) -> float:
        """Get current position for a product."""
        try:
            pos = my_trades.get_position(pid)
            if pos is None:
                return 0.0
            return float(pos.position)
        except:
            return 0.0

    def _apply_spread_skew_and_hedge(
        self, fair: float, spread_pct: float, position: float,
        total_delta: float, delta_per_contract: float
    ) -> Tuple[float, float]:
        """
        Apply spread, inventory skew, and delta hedging to fair value.

        - spread_pct: base spread as percentage (e.g., 0.03 = 3%)
        - position: current inventory position for this product
        - total_delta: total portfolio delta
        - delta_per_contract: delta contribution of one contract of this product

        Returns (bid, ask)
        """
        # Base spread
        half_spread = fair * spread_pct / 2

        # Inventory skew: make it harder to increase position
        skew = 0.0
        if abs(position) > 0:
            # Skew quotes away from direction we're leaning
            # If long (position > 0), lower both bid and ask
            # If short (position < 0), raise both bid and ask
            skew_magnitude = min(abs(position) / self.position_limit, 1.0)
            skew = -position * fair * self.inventory_skew_factor * skew_magnitude

        # Delta hedging adjustment
        # If we need to reduce delta, incentivize trades that move us toward delta-neutral
        hedge_adjustment = 0.0
        if abs(total_delta) > self.delta_hedge_threshold:
            # Determine which direction we need to trade to reduce delta
            # If total_delta > 0 (long), we want to sell delta (make ask more attractive)
            # If total_delta < 0 (short), we want to buy delta (make bid more attractive)

            hedge_urgency = min(abs(total_delta) / (self.delta_hedge_threshold * 2), 1.0)

            # Calculate the delta impact of buying vs selling this product
            # Buying increases our delta by +delta_per_contract
            # Selling decreases our delta by -delta_per_contract

            if total_delta > 0 and delta_per_contract > 0:
                # Long delta, this product adds delta -> make selling attractive
                hedge_adjustment = -hedge_urgency * fair * 0.005  # Lower ask by up to 0.5%
            elif total_delta < 0 and delta_per_contract > 0:
                # Short delta, this product adds delta -> make buying attractive
                hedge_adjustment = hedge_urgency * fair * 0.005  # Lower bid magnitude (raise bid)
            elif total_delta > 0 and delta_per_contract < 0:
                # Long delta, this product reduces delta -> make buying attractive
                hedge_adjustment = hedge_urgency * fair * 0.005
            elif total_delta < 0 and delta_per_contract < 0:
                # Short delta, this product reduces delta -> make selling attractive
                hedge_adjustment = -hedge_urgency * fair * 0.005

        # Position limit enforcement: widen spread drastically if near limit
        position_ratio = abs(position) / self.position_limit
        if position_ratio > 0.7:
            # Exponentially widen spread as we approach limit
            penalty = (position_ratio - 0.7) / 0.3  # 0 to 1 as we go from 70% to 100%
            half_spread *= (1 + 3 * penalty)  # Up to 4x wider spread

            # If at limit, only quote one side
            if abs(position) >= self.position_limit:
                if position > 0:
                    # Long at limit: only willing to sell
                    return (0.0, fair + half_spread + skew + hedge_adjustment)
                else:
                    # Short at limit: only willing to buy
                    return (fair - half_spread + skew + hedge_adjustment, float('inf'))

        bid = fair - half_spread + skew + hedge_adjustment
        ask = fair + half_spread + skew + hedge_adjustment

        # Ensure well-formed market
        if bid >= ask:
            mid = (bid + ask) / 2
            epsilon = max(fair * 0.0001, 0.01)
            bid = mid - epsilon
            ask = mid + epsilon

        # Ensure non-negative prices
        bid = max(0.0, bid)
        ask = max(bid + 0.01, ask)

        return float(bid), float(ask)

    def on_round_end(self, result: Dict[str, Any]) -> None:
        pnl = result.get("pnl", 0.0)
        print(f"[ROUND END] PnL: {pnl:.2f}")

    def on_game_end(self, summary: Dict[str, Any]) -> None:
        total_pnl = summary.get("total_pnl", 0.0)
        final_score = summary.get("final_score", 0.0)
        num_trades = summary.get("num_trades", 0)
        print(f"[GAME END] Total PnL: {total_pnl:.2f}, Score: {final_score:.2f}, Trades: {num_trades}")

# Export the strategy
Strategy = MyTradingStrategy
STRATEGY_CLASS = MyTradingStrategy
__all__ = ["MyTradingStrategy", "Strategy", "STRATEGY_CLASS"]
