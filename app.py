# app.py
# Streamlit UI for simple pricing optimization + action-oriented plan (next N days)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import date, timedelta

from pricing import fit_elasticity_model, optimize_price_for_sku, predict_units


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Simple Pricing & Promo Optimizer", layout="wide")


# ----------------------------
# Data loading
# ----------------------------
@st.cache_data
def load_sales(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "sku", "price", "units", "cost", "promo_flag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sales_history.csv missing columns: {missing}")
    return df


@st.cache_data
def load_comp(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    # expected: date, sku, competitor, comp_price (we'll be tolerant)
    return df


def latest_comp_price(comp_df: pd.DataFrame) -> pd.DataFrame:
    if comp_df is None or comp_df.empty:
        return pd.DataFrame(columns=["sku", "comp_price"])
    comp_df = comp_df.dropna(subset=["comp_price"]).copy()
    if comp_df.empty:
        return pd.DataFrame(columns=["sku", "comp_price"])
    comp_df = comp_df.sort_values("date").groupby("sku").tail(1)
    return comp_df[["sku", "comp_price"]]


# ----------------------------
# Action plan builder
# ----------------------------
def build_action_plan(
    rec_df: pd.DataFrame,
    horizon_days: int,
    max_changes_per_day: int,
    max_abs_price_change: float,
    min_abs_change_to_act: float,
    start_from_tomorrow: bool = True,
) -> pd.DataFrame:
    """
    Converts recommendations into an execution plan over the next N days.

    - Caps per-SKU absolute price change to max_abs_price_change
    - Ignores changes smaller than min_abs_change_to_act
    - Schedules up to max_changes_per_day actions per day, prioritizing highest profit uplift
    """
    if rec_df is None or rec_df.empty:
        return pd.DataFrame(
            columns=[
                "planned_date",
                "sku",
                "current_price",
                "planned_price",
                "change_%",
                "rec_promo",
                "profit_uplift",
                "action",
            ]
        )

    df = rec_df.copy()

    # Compute change %
    df["raw_change_%"] = (df["rec_price"] / df["current_price"] - 1.0)

    # Cap changes
    df["capped_change_%"] = df["raw_change_%"].clip(-max_abs_price_change, max_abs_price_change)

    # Ignore tiny changes
    df = df[df["capped_change_%"].abs() >= min_abs_change_to_act].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "planned_date",
                "sku",
                "current_price",
                "planned_price",
                "change_%",
                "rec_promo",
                "profit_uplift",
                "action",
            ]
        )

    # Recompute recommended price after capping
    df["planned_price"] = (df["current_price"] * (1.0 + df["capped_change_%"])).round(2)

    # Action text
    def action_text(row):
        direction = "Decrease" if row["capped_change_%"] < 0 else "Increase"
        pct = abs(row["capped_change_%"]) * 100.0
        promo_txt = " + Promo" if int(row.get("rec_promo", 0)) == 1 else ""
        return f"{direction} price by {pct:.1f}% to {row['planned_price']:.2f}{promo_txt}"

    df["action"] = df.apply(action_text, axis=1)

    # Prioritize by uplift if present, otherwise by predicted profit
    if "profit_uplift" in df.columns:
        df = df.sort_values("profit_uplift", ascending=False).reset_index(drop=True)
    else:
        df = df.sort_values("pred_profit", ascending=False).reset_index(drop=True)

    # Schedule across next N days
    start = date.today() + timedelta(days=1) if start_from_tomorrow else date.today()
    schedule_dates = [start + timedelta(days=i) for i in range(horizon_days)]

    plan_rows = []
    idx = 0
    for d in schedule_dates:
        day_slice = df.iloc[idx : idx + max_changes_per_day].copy()
        if day_slice.empty:
            break
        day_slice["planned_date"] = pd.to_datetime(str(d))
        plan_rows.append(day_slice)
        idx += len(day_slice)

    if not plan_rows:
        return pd.DataFrame(
            columns=[
                "planned_date",
                "sku",
                "current_price",
                "planned_price",
                "change_%",
                "rec_promo",
                "profit_uplift",
                "action",
            ]
        )

    plan = pd.concat(plan_rows, ignore_index=True)
    plan["change_%"] = (plan["capped_change_%"] * 100.0).round(1)

    cols = [
        "planned_date",
        "sku",
        "current_price",
        "planned_price",
        "change_%",
        "rec_promo",
        "profit_uplift" if "profit_uplift" in plan.columns else "pred_profit",
        "action",
    ]

    # Rename column in display to always be profit_uplift label
    if "profit_uplift" not in plan.columns:
        plan = plan.rename(columns={"pred_profit": "profit_uplift"})
        cols = [
            "planned_date",
            "sku",
            "current_price",
            "planned_price",
            "change_%",
            "rec_promo",
            "profit_uplift",
            "action",
        ]

    return plan[cols].sort_values(["planned_date", "profit_uplift"], ascending=[True, False])


# ----------------------------
# App UI
# ----------------------------
st.title("Simple Pricing Optimization & Promotion Management")

with st.sidebar:
    st.header("Data")
    sales_path = st.text_input("Sales history CSV", "data/sales_history.csv")
    comp_path = st.text_input("Competitor prices CSV", "data/competitors.csv")

    st.divider()
    st.header("Optimization guardrails")
    min_margin = st.slider("Min margin floor (%)", 0, 80, 10) / 100.0
    max_rel_to_comp = st.slider("Max price vs competitor (x)", 1.0, 3.0, 1.2, 0.05)
    allow_promo = st.checkbox("Allow promo recommendation", value=True)
    promo_lift_cost = st.number_input("Promo cost penalty (optional)", value=0.0, step=1.0)

    st.divider()
    st.header("Search space")
    step = st.number_input("Candidate price step", value=1.0, min_value=0.01, step=0.5)
    span = st.slider("Search span around current price (%)", 5, 80, 20) / 100.0

    st.divider()
    st.header("Execution / Action Plan")
    horizon_days = st.slider("Action horizon (days)", 3, 14, 6)
    max_changes_per_day = st.slider("Max changes per day", 1, 50, 8)
    max_abs_price_change = st.slider("Max abs price change per SKU (%)", 1, 50, 12) / 100.0
    min_abs_change_to_act = st.slider("Ignore tiny changes under (%)", 0, 10, 2) / 100.0
    start_from_tomorrow = st.checkbox("Start from tomorrow", value=True)

# Load data
sales = load_sales(sales_path)
comp = load_comp(comp_path)

# Add latest competitor prices onto training set
comp_latest = latest_comp_price(comp)
sales_train = sales.merge(comp_latest, on="sku", how="left")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "SKU Drilldown", "Recommendations", "Action Plan (Next days)"]
)


# ----------------------------
# Tab 1: Overview
# ----------------------------
with tab1:
    colA, colB, colC, colD = st.columns(4)

    total_rev = float((sales["price"] * sales["units"]).sum())
    total_units = float(sales["units"].sum())
    avg_price = float((sales["price"] * sales["units"]).sum() / max(total_units, 1.0))
    promo_share = float(sales["promo_flag"].mean())

    colA.metric("Revenue", f"{total_rev:,.0f}")
    colB.metric("Units", f"{total_units:,.0f}")
    colC.metric("Avg selling price", f"{avg_price:,.2f}")
    colD.metric("Promo share", f"{promo_share*100:.1f}%")

    st.subheader("Revenue over time")
    daily = (
        sales.assign(revenue=sales["price"] * sales["units"])
        .groupby("date", as_index=False)["revenue"]
        .sum()
    )
    fig = px.line(daily, x="date", y="revenue")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top SKUs by revenue")
    top = (
        sales.assign(revenue=sales["price"] * sales["units"])
        .groupby("sku", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
        .head(20)
    )
    st.dataframe(top, use_container_width=True)


# ----------------------------
# Tab 2: SKU Drilldown
# ----------------------------
with tab2:
    skus = sorted(sales["sku"].unique().tolist())
    sku = st.selectbox("Select SKU", skus)

    sku_hist = sales_train[sales_train["sku"] == sku].sort_values("date")
    st.write(f"Rows: {len(sku_hist)}")

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(sku_hist, x="date", y="units", title="Units over time")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.line(sku_hist, x="date", y="price", title="Price over time")
        st.plotly_chart(fig2, use_container_width=True)

    if "comp_price" in sku_hist.columns and sku_hist["comp_price"].notna().any():
        fig3 = px.line(
            sku_hist.dropna(subset=["comp_price"]),
            x="date",
            y="comp_price",
            title="Competitor price (latest source)",
        )
        st.plotly_chart(fig3, use_container_width=True)


# ----------------------------
# Tab 3: Recommendations (with profit uplift)
# ----------------------------
with tab3:
    st.subheader("Train a simple demand model and generate price/promo recommendations")

    model_bundle = fit_elasticity_model(
        sales_train.dropna(subset=["price", "units", "cost", "promo_flag"])
    )

    st.caption("Model coefficients (roughly interpretable: log(units) vs log(price), promo, log(comp))")
    coef = dict(zip(model_bundle["features"], model_bundle["model"].coef_))
    st.json(coef)

    # Latest observation per SKU (acts like today's context)
    latest = sales_train.sort_values("date").groupby("sku").tail(1).copy()

    recs = []
    for _, r in latest.iterrows():
        sku = r["sku"]
        cur_price = float(r["price"])
        cur_cost = float(r["cost"])
        comp_price = float(r["comp_price"]) if pd.notna(r.get("comp_price")) else cur_price

        # Baseline: current price & current promo flag (as observed)
        cur_promo = int(r["promo_flag"])
        base_units = predict_units(model_bundle, price=cur_price, promo_flag=cur_promo, comp_price=comp_price)
        base_profit = (cur_price - cur_cost) * base_units  # baseline profit

        # Candidate search around current price
        lo = max(0.01, cur_price * (1 - span))
        hi = cur_price * (1 + span)
        candidate_prices = np.arange(lo, hi + step, step)

        best = optimize_price_for_sku(
            model_bundle=model_bundle,
            sku_row_context={"cost": cur_cost, "comp_price": comp_price, "price": cur_price},
            candidate_prices=candidate_prices,
            allow_promo=allow_promo,
            promo_lift_cost=promo_lift_cost,
            guardrails={"min_margin": min_margin, "max_price_rel_to_comp": max_rel_to_comp},
        )

        uplift = float(best["pred_profit"] - base_profit)

        recs.append(
            {
                "sku": sku,
                "current_price": cur_price,
                "current_promo": cur_promo,
                "baseline_profit": base_profit,
                "rec_price": float(best["price"]),
                "rec_promo": int(best["promo_flag"]),
                "pred_units": float(best["pred_units"]),
                "pred_revenue": float(best["pred_revenue"]),
                "pred_profit": float(best["pred_profit"]),
                "profit_uplift": uplift,
                "comp_price": comp_price,
            }
        )

    rec_df = pd.DataFrame(recs)
    rec_df["price_change_%"] = (rec_df["rec_price"] / rec_df["current_price"] - 1.0) * 100.0
    rec_df = rec_df.sort_values("profit_uplift", ascending=False)

    # Action filters / highlights
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("SKUs with posgit config --global user.name "Teo Lindroth"
itive uplift", int((rec_df["profit_uplift"] > 0).sum()))
    with c2:
        st.metric("Total uplift (sum)", f"{rec_df['profit_uplift'].sum():,.0f}")
    with c3:
        st.metric("Median suggested change", f"{rec_df['price_change_%'].median():.1f}%")

    st.dataframe(rec_df, use_container_width=True)

    st.subheader("Distribution of recommended price changes")
    fig = px.histogram(rec_df, x="price_change_%")
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# Tab 4: Action Plan
# ----------------------------
with tab4:
    st.subheader(f"Action Plan: next {horizon_days} days")
    st.caption(
        "A simple rollout calendar: highest profit uplift actions first, "
        "capped by daily capacity and max price move per SKU."
    )

    plan_df = build_action_plan(
        rec_df=rec_df,
        horizon_days=horizon_days,
        max_changes_per_day=max_changes_per_day,
        max_abs_price_change=max_abs_price_change,
        min_abs_change_to_act=min_abs_change_to_act,
        start_from_tomorrow=start_from_tomorrow,
    )

    if plan_df.empty:
        st.info(
            "No actions scheduled. Try lowering the minimum change threshold, "
            "increasing max abs price change, or relaxing guardrails."
        )
    else:
        # Quick summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Total actions scheduled", len(plan_df))
        col2.metric("Days with actions", plan_df["planned_date"].nunique())
        col3.metric("Total uplift (scheduled)", f"{plan_df['profit_uplift'].sum():,.0f}")

        # Day-by-day view
        for d, day_df in plan_df.groupby("planned_date"):
            st.markdown(f"### {pd.to_datetime(d).date().isoformat()}")
            st.dataframe(day_df.drop(columns=["planned_date"]).reset_index(drop=True), use_container_width=True)

        st.divider()
        st.subheader("Export")
        csv_bytes = plan_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download action plan CSV",
            data=csv_bytes,
            file_name="action_plan_next_days.csv",
            mime="text/csv",
        )