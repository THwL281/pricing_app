import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def fit_elasticity_model(df: pd.DataFrame) -> dict:
    """
    Fits a simple log-linear demand model:
      log(units) = a + b*log(price) + c*promo_flag + d*log(comp_price) + ...
    Returns model + feature columns used.
    """
    df = df.copy()

    # Basic cleaning
    df = df[(df["units"] > 0) & (df["price"] > 0) & (df["cost"] >= 0)]

    # If comp_price missing, fill with own price (neutral-ish default)
    if "comp_price" not in df.columns:
        df["comp_price"] = df["price"]
    df["comp_price"] = df["comp_price"].where(df["comp_price"] > 0, df["price"])

    # Features
    X = pd.DataFrame({
        "log_price": np.log(df["price"].astype(float)),
        "promo_flag": df["promo_flag"].astype(int),
        "log_comp_price": np.log(df["comp_price"].astype(float)),
    })
    y = np.log(df["units"].astype(float))

    model = LinearRegression()
    model.fit(X, y)

    return {"model": model, "features": list(X.columns)}

def predict_units(model_bundle: dict, price: float, promo_flag: int, comp_price: float) -> float:
    model = model_bundle["model"]
    X = np.array([[np.log(price), int(promo_flag), np.log(comp_price)]], dtype=float)
    y_log = float(model.predict(X)[0])
    return float(np.exp(y_log))

def optimize_price_for_sku(
    model_bundle: dict,
    sku_row_context: dict,
    candidate_prices: np.ndarray,
    allow_promo: bool = True,
    promo_lift_cost: float = 0.0,
    guardrails: dict | None = None,
) -> dict:
    """
    Brute-force search across candidate prices (and optional promo) for max expected profit.
    sku_row_context must include: cost, comp_price
    profit = (price - cost) * predicted_units - promo_lift_cost*promo_flag
    """
    cost = float(sku_row_context["cost"])
    comp_price = float(sku_row_context.get("comp_price", sku_row_context.get("price", 1.0)))

    guardrails = guardrails or {}
    min_margin = float(guardrails.get("min_margin", 0.0))  # e.g. 0.1 means 10% margin floor
    max_price_rel_to_comp = float(guardrails.get("max_price_rel_to_comp", 10.0))  # e.g. 1.2 => 20% above comp

    best = None

    promo_options = [0, 1] if allow_promo else [0]

    for p in candidate_prices:
        p = float(p)

        # Guardrails
        if p <= 0:
            continue
        if p < cost * (1 + min_margin):
            continue
        if comp_price > 0 and p > comp_price * max_price_rel_to_comp:
            continue

        for promo_flag in promo_options:
            units_hat = predict_units(model_bundle, price=p, promo_flag=promo_flag, comp_price=comp_price)
            profit_hat = (p - cost) * units_hat - promo_lift_cost * promo_flag

            row = {
                "price": p,
                "promo_flag": promo_flag,
                "pred_units": units_hat,
                "pred_profit": profit_hat,
                "pred_revenue": p * units_hat,
                "margin_per_unit": p - cost,
            }

            if (best is None) or (profit_hat > best["pred_profit"]):
                best = row

    return best or {
        "price": float(np.nan),
        "promo_flag": 0,
        "pred_units": 0.0,
        "pred_profit": 0.0,
        "pred_revenue": 0.0,
        "margin_per_unit": 0.0,
    }
