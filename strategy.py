"""
CTC Derivatives Trading Game - Debugged Strategy

Focus: Simple, correct pricing with aggressive market-making
Goal: Maximize trades and earn bid-ask spread
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
    """Dirichlet-smoothed pmf over faces 1..D."""
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

def _central_moments_one_roll(p: np.ndarray) -> Tuple[float, float]:
    """Returns (mu1, var1) for 1-roll distribution."""
    D = p.size
    x = np.arange(1, D + 1, dtype=np.float64)
    p64 = p.astype(np.float64, copy=False)
    mu = float(np.dot(x, p64))
    c = x - mu
    m2 = float(np.dot(c * c, p64))
    return mu, m2

# ============ FFT-based pricing ============

def _remaining_sum_pmf_fft(p: np.ndarray, n_rem: int) -> np.ndarray:
    """PMF of sum of n_rem rolls using FFT."""
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
    """Compute (pmf, cdf, cumsum_weighted)."""
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

# ============ Normal approximation ============

_SQRT2 = np.sqrt(2.0)
_SQRT2PI = np.sqrt(2.0 * np.pi)

def _Phi(x: float) -> float:
    return 0.5 * (1.0 + scipy.special.erf(x / _SQRT2))

def _phi(x: float) -> float:
    return np.exp(-0.5 * x * x) / _SQRT2PI

def _normal_call_put(S_t: float, K: float, n_rem: int, mu1: float, var1: float) -> Tuple[float, float]:
    """Normal approximation for call/put."""
    n_rem = max(0, int(n_rem))
    m = n_rem * mu1
    v = max(0.0, n_rem * var1)
    muY = S_t + m

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

    return max(0.0, call), max(0.0, put)

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
    """Auto-select FFT or Normal for call pricing."""
    if _use_fft(D, n_rem, max_fft_points):
        if n_rem not in pmf_cache:
            pmf_cache[n_rem] = _pmf_triplet(p, n_rem)
        _, cdf, pr = pmf_cache[n_rem]
        return _call_price_from_prefix(S_t, K, cdf, pr)
    else:
        call, _ = _normal_call_put(S_t, K, n_rem, mu1, var1)
        return call

def put_price_auto(p: np.ndarray, K: float, S_t: float, n_rem: int,
                   pmf_cache: Dict[int, Tuple], mu1: float, var1: float,
                   D: int, max_fft_points: int) -> float:
    """Auto-select FFT or Normal for put pricing."""
    if _use_fft(D, n_rem, max_fft_points):
        if n_rem not in pmf_cache:
            pmf_cache[n_rem] = _pmf_triplet(p, n_rem)
        _, cdf, pr = pmf_cache[n_rem]
        return _put_price_from_prefix(S_t, K, cdf, pr)
    else:
        _, put = _normal_call_put(S_t, K, n_rem, mu1, var1)
        return put

def futures_price_sum(mu1: float, S_t: float, n_rem: int) -> float:
    """Expected future sum: S_t + n_rem * mu1."""
    return float(S_t + max(0, int(n_rem)) * mu1)

# ============ DynamicFutures Pricing ============

def get_dynamic_future_fair_value(
    expiry_subround: int,
    current_subround: int,
    rolls_array: np.ndarray,
    rolls_per_subround: int,
    mu1: float
) -> float:
    """
    Calculate fair value for DynamicFutures.

    CRITICAL: DF settles on sum of FIRST (expiry × rolls_per_subround) rolls!
    """
    total_rolls_at_expiry = expiry_subround * rolls_per_subround
    rolls_observed = len(rolls_array)

    if expiry_subround < current_subround:
        # Expired: exact value known
        if rolls_observed >= total_rolls_at_expiry:
            return float(np.sum(rolls_array[:total_rolls_at_expiry]))
        else:
            return float(np.sum(rolls_array))

    elif expiry_subround == current_subround:
        # Current expiry subround
        if rolls_observed >= total_rolls_at_expiry:
            return float(np.sum(rolls_array[:total_rolls_at_expiry]))
        else:
            # Partial
            sum_so_far = float(np.sum(rolls_array))
            rolls_remaining = total_rolls_at_expiry - rolls_observed
            return sum_so_far + rolls_remaining * mu1

    else:
        # Future expiry
        if rolls_observed >= total_rolls_at_expiry:
            return float(np.sum(rolls_array[:total_rolls_at_expiry]))
        else:
            sum_observed = float(np.sum(rolls_array)) if rolls_observed > 0 else 0.0
            rolls_remaining = total_rolls_at_expiry - rolls_observed
            return sum_observed + rolls_remaining * mu1

# ============ Main Strategy ============

class MyTradingStrategy(AbstractTradingStrategy):
    """
    Simple, aggressive market-making strategy.

    Key principles:
    - Correct pricing (especially DynamicFutures!)
    - Tight spreads to maximize trades
    - NO position limits (trust our pricing)
    """

    def __init__(self):
        self.dice_sides = 10000
        self.rolls_per_subround = 2000
        self.team_name = "Unknown"
        self.dirichlet_alpha = 1.0
        self.max_fft_points = 1 << 22

        # TIGHT spreads for maximum trade volume
        self.spread_df_expired = 0.001       # 0.1%
        self.spread_df_current = 0.01        # 1%
        self.spread_df_future = 0.02         # 2%
        self.spread_standard_futures = 0.02  # 2%
        self.spread_standard_options = 0.03  # 3%

        self.current_subround = 1

    def on_game_start(self, config: Dict[str, Any]) -> None:
        self.dice_sides = int(config.get("dice_sides", self.dice_sides))
        self.team_name = config.get("team_name", "Unknown")
        self.rolls_per_subround = int(config.get("rolls_per_subround", self.rolls_per_subround))
        np.random.seed(int(config.get("seed", 42)))
        print(f"[INFO] Strategy {self.team_name}: D={self.dice_sides}, rolls/sub={self.rolls_per_subround}")

    def make_market(
        self, *, marketplace: Any, training_rolls: Any, my_trades: Any,
        current_rolls: Any, round_info: Any
    ) -> Dict[str, Tuple[float, float]]:

        try:
            # Extract round info
            self.current_subround = int(round_info.get("current_sub_round", 1))

            # Build PMF
            D = int(self.dice_sides)
            pD = _estimate_pD(training_rolls, current_rolls, D=D, alpha=self.dirichlet_alpha)
            mu1, var1 = _central_moments_one_roll(pD)

            # Get rolls
            rolls_array = _ravel_ints(current_rolls)
            S_now = float(np.sum(rolls_array)) if rolls_array.size > 0 else 0.0

            # PMF cache
            pmf_cache: Dict[int, Tuple] = {}

            # Get products
            products = list(marketplace.get_products())
            quotes: Dict[str, Tuple[float, float]] = {}

            print(f"[DEBUG] Subround {self.current_subround}, S_now={S_now:.2f}, mu1={mu1:.2f}, #products={len(products)}")

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
                        # DynamicFutures
                        expiry = int(parts[2])
                        fair = get_dynamic_future_fair_value(
                            expiry, self.current_subround, rolls_array,
                            self.rolls_per_subround, mu1
                        )

                        # Determine spread
                        if expiry < self.current_subround:
                            spread = self.spread_df_expired
                        elif expiry == self.current_subround:
                            spread = self.spread_df_current
                        else:
                            spread = self.spread_df_future

                        bid = fair * (1 - spread)
                        ask = fair * (1 + spread)

                        print(f"[DEBUG] {pid}: fair={fair:.2f}, spread={spread*100:.1f}%, bid={bid:.2f}, ask={ask:.2f}")

                    elif kind == "F":
                        # Standard Futures
                        rounds_left = int(parts[2])
                        n_rem = rounds_left * self.rolls_per_subround
                        fair = futures_price_sum(mu1, S_now, n_rem)

                        spread = self.spread_standard_futures
                        bid = fair * (1 - spread)
                        ask = fair * (1 + spread)

                    elif kind in ("C", "P"):
                        # Options
                        if len(parts) < 4:
                            continue
                        K = float(parts[2])
                        rounds_left = int(parts[3])
                        n_rem = rounds_left * self.rolls_per_subround

                        if kind == "C":
                            fair = call_price_auto(pD, K, S_now, n_rem, pmf_cache, mu1, var1, D, self.max_fft_points)
                        else:
                            fair = put_price_auto(pD, K, S_now, n_rem, pmf_cache, mu1, var1, D, self.max_fft_points)

                        spread = self.spread_standard_options
                        bid = fair * (1 - spread)
                        ask = fair * (1 + spread)

                    else:
                        continue

                    # Ensure valid market
                    bid = max(0.0, bid)
                    ask = max(bid + 0.01, ask)

                    if bid < ask and ask < 1e15:  # Sanity check
                        quotes[pid] = (float(bid), float(ask))

                except Exception as e:
                    print(f"[ERROR] Failed to price {pid}: {e}")
                    continue

            print(f"[DEBUG] Generated {len(quotes)} quotes")
            return quotes

        except Exception as e:
            print(f"[CRITICAL ERROR in make_market]: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def on_round_end(self, result: Dict[str, Any]) -> None:
        pnl = result.get("pnl", 0.0)
        print(f"[ROUND END] PnL: {pnl:.2f}")
        print(f"[ROUND END] Full result: {result}")

    def on_game_end(self, summary: Dict[str, Any]) -> None:
        total_pnl = summary.get("total_pnl", 0.0)
        final_score = summary.get("final_score", 0.0)
        num_trades = summary.get("num_trades", 0)
        print(f"[GAME END] Total PnL: {total_pnl:.2f}, Score: {final_score:.2f}, Trades: {num_trades}")

# Export
Strategy = MyTradingStrategy
STRATEGY_CLASS = MyTradingStrategy
__all__ = ["MyTradingStrategy", "Strategy", "STRATEGY_CLASS"]
