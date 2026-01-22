import numpy as np
import pandas as pd
from pathlib import Path

def make_fake_data(
    out_dir: str = "data",
    n_skus: int = 40,
    n_days: int = 180,
    start_date: str = "2025-07-01",
    seed: int = 42,
):
    """
    Generates:
      - sales_history.csv with: date, sku, price, units, cost, promo_flag
      - competitors.csv with: date, sku, competitor, comp_price

    The data tries to be plausible:
      - units decrease as price increases (negative elasticity)
      - promo increases units
      - competitor price is correlated with our base price + noise
      - cost is below price
    """
    rng = np.random.default_rng(seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    # SKUs and base economics
    skus = [f"SKU-{i:03d}" for i in range(1, n_skus + 1)]
    base_price = rng.uniform(8, 120, size=n_skus)           # typical retail price range
    base_cost = base_price * rng.uniform(0.45, 0.80, size=n_skus)  # margin 20–55%

    # Demand knobs
    # Elasticity: more negative => units more sensitive to price
    elasticity = rng.uniform(-2.2, -0.6, size=n_skus)
    promo_lift = rng.uniform(0.10, 0.60, size=n_skus)       # 10–60% lift on promo
    base_units = rng.uniform(5, 80, size=n_skus)            # baseline daily volume

    # Seasonality (weekly): weekend bump or dip
    dow = np.array([d.weekday() for d in dates])  # Mon=0..Sun=6
    # Make weekends slightly higher on average
    seasonality = np.where(dow >= 5, 1.12, 1.00)

    sales_rows = []
    comp_rows = []

    competitor_names = ["CompA", "CompB"]

    for s_idx, sku in enumerate(skus):
        bp = float(base_price[s_idx])
        bc = float(base_cost[s_idx])
        el = float(elasticity[s_idx])
        pu = float(promo_lift[s_idx])
        bu = float(base_units[s_idx])

        for d_idx, date in enumerate(dates):
            # Promo probability: occasional promos
            promo_flag = int(rng.random() < 0.18)  # ~18% of days on promo
            promo_discount = rng.uniform(0.05, 0.25) if promo_flag else 0.0

            # Our price: base +/- noise, discounted if promo
            noise = rng.normal(0, 0.03)  # ~3% daily jitter
            price = bp * (1 + noise) * (1 - promo_discount)
            price = max(price, bc * 1.05)  # keep at least 5% above cost

            # Competitor price correlated with our base price
            # Add day-to-day noise and sometimes undercut or overprice slightly
            comp_base = bp * rng.uniform(0.95, 1.05)
            comp_today = comp_base * (1 + rng.normal(0, 0.025))

            # Build units using log model-ish behavior:
            # units = base_units * seasonality * (price/bp)^elasticity * (1+promo_lift*promo_flag) * random_noise
            price_ratio = price / bp
            units_mean = bu * seasonality[d_idx] * (price_ratio ** el) * (1 + pu * promo_flag)

            # Add multiplicative noise, then Poisson-ish rounding
            units_noisy = units_mean * rng.lognormal(mean=0.0, sigma=0.18)
            units = int(max(0, round(units_noisy)))

            sales_rows.append({
                "date": date.date().isoformat(),
                "sku": sku,
                "price": round(float(price), 2),
                "units": units,
                "cost": round(float(bc), 2),
                "promo_flag": promo_flag,
            })

            # Save competitor prices (2 competitors)
            for comp in competitor_names:
                # each competitor differs a bit
                comp_price = comp_today * rng.uniform(0.97, 1.03)
                comp_rows.append({
                    "date": date.date().isoformat(),
                    "sku": sku,
                    "competitor": comp,
                    "comp_price": round(float(comp_price), 2),
                })

    sales_df = pd.DataFrame(sales_rows)
    comp_df = pd.DataFrame(comp_rows)

    sales_csv = out_path / "sales_history.csv"
    comp_csv = out_path / "competitors.csv"
    sales_df.to_csv(sales_csv, index=False)
    comp_df.to_csv(comp_csv, index=False)

    print(f"Wrote: {sales_csv}  ({len(sales_df):,} rows)")
    print(f"Wrote: {comp_csv}  ({len(comp_df):,} rows)")
    print("Done.")

if __name__ == "__main__":
    make_fake_data()
