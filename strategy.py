"""
CTC Derivatives Trading Game - Corrected Strategy with Proper DynamicFutures Pricing

KEY FIX: DynamicFutures settle on sum of FIRST (2000 × expiry) rolls, NOT all rolls!

Features:
- Accurate DynamicFutures pricing (expired, current, future expiries)
- Competitive spreads optimized for trade matching
- Simple position limits for risk control
- Fast execution with minimal complexity

Product ID formats:
- DynamicFutures:   "S,DF,EXPIRY" (EXPIRY ∈ {1..9} subrounds)
- Standard Futures: "S,F,ROUNDS_LEFT"
- Standard Calls:   "S,C,STRIKE,ROUNDS_LEFT"
- Standard Puts:    "S,P,STRIKE,ROUNDS_LEFT"
"""

from autograder.sdk.strategy_interface import AbstractTradingStrategy
import numpy as np
import scipy.special
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
    """
    counts = np.full(D, float(alpha), dtype=np.float64)
    for seq in (_ravel_ints(training_rolls), _ravel_ints(current_rolls)):
        if seq.size == 0:
            continue
        mask = (seq >= 1) & (seq <= D)
        if np.any(mask):
            hist = np.bincount(seq[mask], minlength=D + 1)
            counts += hist[1:D + 1]
    s = float(counts.sum())
    if s > 0 and np.isfinite(s):
        return (counts / s).astype(np.float32, copy=False)
    return np.full(D, 1.0 / D, dtype=np.float32)

def _central_moments_one_roll(p: np.ndarray) -> Tuple[float, float, float, float]:
    """Returns (mu1, var1, skew1, exkurt1) for 1-roll distribution."""
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

# ============ FFT-based pricing ============

def _remaining_sum_pmf_fft(p: np.ndarray, n_rem: int) -> np.ndarray:
    """PMF of sum of n_rem rolls using FFT convolution."""
    n_rem = max(0, int(n_rem))
    if n_rem == 0:
        return np.array([1.0], dtype=np.float32)

    p = np.asarray(p, dtype=np.float32).reshape(-1)
    s = float(p.sum())
    if s <= 0 or not np.isfinite(s):
        return np.array([1.0], dtype=np.float32)
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
    """Compute (pmf, cdf, cumsum_weighted) for pricing."""
    f = _remaining_sum_pmf_fft(p, n_rem)
    idx = np.arange(f.size, dtype=np.float64)
    cdf = np.cumsum(f, dtype=np.float64)
    pr = np.cumsum(idx * f.astype(np.float64), dtype=np.float64)
    return f, cdf, pr

def _call_price_from_prefix(S_t: float, K: float, cdf: np.ndarray, pr: np.ndarray) -> float:
    t = int(np.floor(K - S_t))
    t = np.clip(t, -1, cdf.size - 1)
    tail_prob = 1.0 - (0.0 if t < 0 else cdf[t])
    tail_pr = pr[-1] - (0.0 if t < 0 else pr[t])
    return float((S_t - K) * tail_prob + tail_pr)

def _put_price_from_prefix(S_t: float, K: float, cdf: np.ndarray, pr: np.ndarray) -> float:
    u = int(np.ceil(K - S_t) - 1.0)
    u = np.clip(u, -1, cdf.size - 1)
    left_prob = 0.0 if u < 0 else cdf[u]
    left_pr = 0.0 if u < 0 else pr[u]
    return float((K - S_t) * left_prob - left_pr)

# ============ Normal approximation fallback ============

_SQRT2 = np.sqrt(2.0)
_SQRT2PI = np.sqrt(2.0 * np.pi)

def _Phi(x: float) -> float:
    return 0.5 * (1.0 + scipy.special.erf(x / _SQRT2))

def _phi(x: float) -> float:
    return np.exp(-0.5 * x * x) / _SQRT2PI

def _normal_call_put(S_t: float, K: float, n_rem: int, mu1: float, var1: float, D: int = None) -> Tuple[float, float]:
    """Normal approximation for call/put prices."""
    n_rem = max(0, int(n_rem))
    m = n_rem * mu1
    v = max(0.0, n_rem * var1)
    muY = S_t + m

    # Boundary checks
    if D is not None and n_rem > 0:
        L = S_t + n_rem * 1.0
        U = S_t + n_rem * float(D)
        if K >= U:
            return 0.0, max(K - muY, 0.0)
        if K <= L:
            return max(muY - K, 0.0), 0.0

    if v <= 1e-12:
        call = max(muY - K, 0.0)
        put = max(K - muY, 0.0)
        return call, put

    sigma = max(np.sqrt(v), 1e-6)
    d = (muY - K) / sigma

    Phi_d = _Phi(d)
    phi_d = _phi(d)

    call = (muY - K) * Phi_d + sigma * phi_d
    put = (K - muY) * (1.0 - Phi_d) + sigma * phi_d

    # Clamp to valid range
    if D is not None and n_rem > 0:
        L = S_t + n_rem * 1.0
        U = S_t + n_rem * float(D)
        call = min(max(call, 0.0), U - K)
        put = min(max(put, 0.0), K - L)

    return float(call), float(put)

# ============ Auto dispatcher ============

def _use_fft(D: int, n_rem: int, max_fft_points: int) -> bool:
    if n_rem <= 0:
        return True
    L = D * n_rem + 1
    M = next_pow2(L)
    return M <= max_fft_points

def call_price_auto(p: np.ndarray, K: float, S_t: float, n_rem: int,
                    pmf_cache: Dict[int, Tuple], mu1: float, var1: float,
                    D: int, max_fft_points: int) -> float:
    """Auto-select FFT or Normal approximation for call pricing."""
    if _use_fft(D, n_rem, max_fft_points):
        if n_rem not in pmf_cache:
            pmf_cache[n_rem] = _pmf_triplet(p, n_rem)
        _, cdf, pr = pmf_cache[n_rem]
        return _call_price_from_prefix(S_t, K, cdf, pr)
    else:
        call, _ = _normal_call_put(S_t, K, n_rem, mu1, var1, D)
        return call

def put_price_auto(p: np.ndarray, K: float, S_t: float, n_rem: int,
                   pmf_cache: Dict[int, Tuple], mu1: float, var1: float,
                   D: int, max_fft_points: int) -> float:
    """Auto-select FFT or Normal approximation for put pricing."""
    if _use_fft(D, n_rem, max_fft_points):
        if n_rem not in pmf_cache:
            pmf_cache[n_rem] = _pmf_triplet(p, n_rem)
        _, cdf, pr = pmf_cache[n_rem]
        return _put_price_from_prefix(S_t, K, cdf, pr)
    else:
        _, put = _normal_call_put(S_t, K, n_rem, mu1, var1, D)
        return put

def futures_price_sum(p: np.ndarray, S_t: float, n_rem: int) -> float:
    """Expected future sum: S_t + n_rem * E[one roll]."""
    p = np.asarray(p, dtype=np.float32).reshape(-1)
    s = float(p.sum())
    if s <= 0:
        return float(S_t)
    p = p / s
    D = p.size
    faces = np.arange(1, D + 1, dtype=np.float64)
    mu = float(np.dot(faces, p.astype(np.float64)))
    return float(S_t + max(0, int(n_rem)) * mu)

# ============ CORRECTED DynamicFutures Helpers ============

def get_dynamic_future_fair_value(
    expiry_subround: int,
    current_subround: int,
    rolls_array: np.ndarray,
    rolls_per_subround: int,
    mu1: float
) -> float:
    """
    Calculate fair value for DynamicFutures contract.

    CRITICAL: DynamicFutures settle on sum of FIRST (expiry × rolls_per_subround) rolls!
    NOT the sum of all rolls observed!

    Args:
        expiry_subround: When contract expires (1-9)
        current_subround: Current subround (1-10)
        rolls_array: All rolls observed so far
        rolls_per_subround: Number of rolls per subround (2000)
        mu1: Expected value of one roll

    Returns:
        Fair value of the DynamicFutures contract
    """
    total_rolls_at_expiry = expiry_subround * rolls_per_subround
    rolls_observed = len(rolls_array)

    if expiry_subround < current_subround:
        # CONTRACT HAS EXPIRED - value is known exactly
        # Sum of first (expiry × rolls_per_subround) rolls
        if rolls_observed >= total_rolls_at_expiry:
            return float(np.sum(rolls_array[:total_rolls_at_expiry]))
        else:
            # Shouldn't happen but handle gracefully
            return float(np.sum(rolls_array))

    elif expiry_subround == current_subround:
        # CONTRACT EXPIRES THIS SUBROUND
        # Some rolls from expiry subround observed, some not yet
        rolls_observed_in_expiry = max(0, rolls_observed - (expiry_subround - 1) * rolls_per_subround)
        rolls_remaining_in_expiry = rolls_per_subround - rolls_observed_in_expiry

        # Fair value = actual sum so far + expected remaining in this subround
        if rolls_observed >= total_rolls_at_expiry:
            # All rolls for this contract already observed
            return float(np.sum(rolls_array[:total_rolls_at_expiry]))
        else:
            # Partial: use what we have + expect rest
            sum_so_far = float(np.sum(rolls_array[:rolls_observed]))
            expected_remaining = rolls_remaining_in_expiry * mu1
            return sum_so_far + expected_remaining

    else:
        # CONTRACT EXPIRES IN FUTURE
        # We may have seen some subrounds, but not the expiry yet
        rolls_we_need = total_rolls_at_expiry

        if rolls_observed >= rolls_we_need:
            # We've already passed the expiry subrounds somehow (defensive)
            return float(np.sum(rolls_array[:rolls_we_need]))
        else:
            # Use observed + expected for remainder
            sum_observed = float(np.sum(rolls_array[:rolls_observed])) if rolls_observed > 0 else 0.0
            rolls_remaining = rolls_we_need - rolls_observed
            expected_remaining = rolls_remaining * mu1
            return sum_observed + expected_remaining

# ============ Main Strategy ============

class MyTradingStrategy(AbstractTradingStrategy):
    """
    Corrected trading strategy with accurate DynamicFutures pricing.

    Key improvements:
    - Fixed DynamicFutures pricing (was using wrong underlying!)
    - Simplified spread management
    - Removed buggy delta hedging
    - Focus on correctness and trade volume
    """

    def __init__(self):
        # Game parameters
        self.dice_sides = 10000
        self.rolls_per_subround = 2000
        self.team_name = "Unknown"

        # Pricing parameters
        self.dirichlet_alpha = 1.0
        self.max_fft_points = 1 << 22  # Smaller for speed

        # Trading parameters
        self.position_limit = 30  # Max position per product

        # Spread parameters (competitive for high match rate)
        self.spread_df_expired = 0.0005      # 0.05% - nearly risk-free
        self.spread_df_current_expiry = 0.005  # 0.5% - low risk
        self.spread_df_future = 0.015         # 1.5% - moderate risk
        self.spread_standard_futures = 0.02   # 2%
        self.spread_standard_options = 0.03   # 3%

        # Inventory skew
        self.inventory_skew_factor = 0.005  # 0.5% per unit

        # State
        self.current_subround = 1

    def on_game_start(self, config: Dict[str, Any]) -> None:
        self.dice_sides = int(config.get("dice_sides", self.dice_sides))
        self.team_name = config.get("team_name", "Unknown")
        self.rolls_per_subround = int(config.get("rolls_per_subround", self.rolls_per_subround))

        # Allow config overrides
        self.max_fft_points = int(config.get("max_fft_points", self.max_fft_points))
        self.position_limit = int(config.get("position_limit", self.position_limit))

        np.random.seed(int(config.get("seed", 42)))
        print(f"[INFO] Strategy {self.team_name}: D={self.dice_sides}, "
              f"pos_limit={self.position_limit}, rolls/sub={self.rolls_per_subround}")

    def make_market(
        self, *, marketplace: Any, training_rolls: Any, my_trades: Any,
        current_rolls: Any, round_info: Any
    ) -> Dict[str, Tuple[float, float]]:

        # Extract round info
        self.current_subround = int(round_info.get("current_sub_round", 1))

        # Build PMF from data
        D = int(self.dice_sides)
        pD = _estimate_pD(training_rolls, current_rolls, D=D, alpha=self.dirichlet_alpha)
        mu1, var1, _, _ = _central_moments_one_roll(pD)

        # Get current rolls as array for DynamicFutures
        rolls_array = _ravel_ints(current_rolls)
        S_now = float(np.sum(rolls_array)) if rolls_array.size > 0 else 0.0

        # PMF cache for options pricing
        pmf_cache: Dict[int, Tuple] = {}

        # Get all products
        products = list(marketplace.get_products())
        quotes: Dict[str, Tuple[float, float]] = {}

        # Process each product
        for product in products:
            # Get product ID
            if hasattr(product, "product_id"):
                pid = product.product_id
            elif hasattr(product, "id"):
                pid = product.id
            else:
                continue

            if not pid:
                continue

            parts = pid.split(",")
            if len(parts) < 3:
                continue

            try:
                kind = parts[1].upper()

                if kind == "DF":
                    # DynamicFutures: "S,DF,EXPIRY"
                    expiry = int(parts[2])

                    # CORRECTED PRICING
                    fair = get_dynamic_future_fair_value(
                        expiry, self.current_subround, rolls_array,
                        self.rolls_per_subround, mu1
                    )

                    # Determine spread based on contract state
                    if expiry < self.current_subround:
                        spread = self.spread_df_expired
                    elif expiry == self.current_subround:
                        spread = self.spread_df_current_expiry
                    else:
                        spread = self.spread_df_future

                    position = self._get_position(my_trades, pid)
                    bid, ask = self._apply_spread_and_skew(fair, spread, position)
                    quotes[pid] = (bid, ask)

                elif kind == "F":
                    # Standard Futures: "S,F,ROUNDS_LEFT"
                    rounds_left = int(parts[2])
                    n_rem = rounds_left * self.rolls_per_subround
                    fair = futures_price_sum(pD, S_now, n_rem)

                    position = self._get_position(my_trades, pid)
                    bid, ask = self._apply_spread_and_skew(fair, self.spread_standard_futures, position)
                    quotes[pid] = (bid, ask)

                elif kind in ("C", "P"):
                    # Standard Options: "S,C,STRIKE,ROUNDS_LEFT" or "S,P,STRIKE,ROUNDS_LEFT"
                    if len(parts) < 4:
                        continue
                    K = float(parts[2])
                    rounds_left = int(parts[3])
                    n_rem = rounds_left * self.rolls_per_subround

                    if kind == "C":
                        fair = call_price_auto(pD, K, S_now, n_rem, pmf_cache, mu1, var1, D, self.max_fft_points)
                    else:
                        fair = put_price_auto(pD, K, S_now, n_rem, pmf_cache, mu1, var1, D, self.max_fft_points)

                    position = self._get_position(my_trades, pid)
                    bid, ask = self._apply_spread_and_skew(fair, self.spread_standard_options, position)
                    quotes[pid] = (bid, ask)

            except Exception as e:
                # Skip problematic products
                continue

        return quotes

    def _get_position(self, my_trades: Any, pid: str) -> float:
        """Get current position for a product."""
        try:
            pos = my_trades.get_position(pid)
            if pos is None:
                return 0.0
            return float(pos.position)
        except:
            return 0.0

    def _apply_spread_and_skew(self, fair: float, spread_pct: float, position: float) -> Tuple[float, float]:
        """
        Apply spread and simple inventory skew.

        Args:
            fair: Fair value
            spread_pct: Spread as percentage (e.g., 0.02 = 2%)
            position: Current position

        Returns:
            (bid, ask) tuple
        """
        # Base spread
        half_spread = fair * spread_pct / 2

        # Simple inventory skew
        skew = 0.0
        if abs(position) > 0:
            # If long, lower quotes to encourage selling
            # If short, raise quotes to encourage buying
            skew_magnitude = min(abs(position) / self.position_limit, 1.0)
            skew = -position * fair * self.inventory_skew_factor * skew_magnitude

        # Position limit: widen spread if near limit
        position_ratio = abs(position) / self.position_limit
        if position_ratio > 0.75:
            # Widen spread
            penalty = (position_ratio - 0.75) / 0.25
            half_spread *= (1 + 2 * penalty)

            # At limit: only quote one side
            if abs(position) >= self.position_limit:
                if position > 0:
                    # Long: only sell
                    return (0.0, fair + half_spread + skew)
                else:
                    # Short: only buy
                    return (fair - half_spread + skew, float('inf'))

        bid = fair - half_spread + skew
        ask = fair + half_spread + skew

        # Ensure well-formed market
        if bid >= ask:
            mid = (bid + ask) / 2
            epsilon = max(fair * 0.0001, 0.01)
            bid = mid - epsilon
            ask = mid + epsilon

        # Ensure non-negative
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
