import streamlit as st
import pandas as pd
st.set_page_config(
    page_title="Walmart M5 Inventory Dashboard",
    layout="wide"
)

st.title("Walmart M5 Inventory Optimization Dashboard")

st.write("Demand forecasting and inventory optimization using the Walmart M5 dataset.")
st.divider()

st.header("Project Overview")

st.write(
    "This dashboard analyzes historical Walmart M5 sales data "
    "to explore demand patterns, support demand forecasting, "
    "and evaluate inventory optimization strategies."
)
st.divider()

st.header("Historical Sales Data")

df = pd.read_csv("walmart_sku_forecasting_data.csv")

st.write("Total daily sales records:", len(df))

st.dataframe(df.head(10))
st.divider()

st.header("Historical Daily Sales Trend")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Average Daily Sales", f"{df['sales'].mean():.2f} units")

with col2:
    st.metric("Maximum Daily Sales", f"{df['sales'].max()} units")

with col3:
    st.metric("Total Units Sold", f"{df['sales'].sum():,}")
df["date"] = pd.to_datetime(df["date"])

st.line_chart(
    df.set_index("date")["sales"]
)

st.divider()

st.header("Monthly Average Sales Trend")

monthly_sales = (
    df.set_index("date")["sales"]
    .resample("MS")
    .mean()
    .reset_index()
)

st.line_chart(
    monthly_sales,
    x="date",
    y="sales"
)

st.caption(
    "Monthly average daily sales help identify longer-term "
    "demand changes while reducing daily fluctuations."
)
st.divider()

st.header("Average Sales by Day of Week")

weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekday_sales = (
    df.groupby("weekday")["sales"]
    .mean()
    .reindex(weekday_order)
)

st.bar_chart(
    weekday_sales.reset_index(),
    x="weekday",
    y="sales",
    sort=False
)

st.caption(
    "This chart compares average daily sales across weekdays "
    "to identify potential weekly demand patterns."
)
st.divider()

st.header("Demand Forecasting — 7-Day Moving Average")

forecast_df = df.sort_values("date").copy()

forecast_df["7_day_moving_avg"] = (
    forecast_df["sales"]
    .rolling(window=7)
    .mean()
)

st.line_chart(
    forecast_df.set_index("date")[["sales", "7_day_moving_avg"]]
)

st.caption(
    "The 7-day moving average smooths daily sales fluctuations "
    "and helps identify short-term demand trends. "
    "It is a historical baseline, not yet a future forecast."
)


st.divider()

st.header("Demand Forecasting — Seasonal Naive Model")

st.write(
    "This model predicts daily sales using the sales "
    "from the same weekday in the previous week."
)

forecast_df = df.sort_values("date").copy()

forecast_df["predicted_sales"] = forecast_df["sales"].shift(7)
# Remove the first 7 days without predictions
evaluation_df = forecast_df.dropna(
    subset=["predicted_sales"]
).copy()

# Calculate forecasting errors
evaluation_df["absolute_error"] = (
    evaluation_df["sales"] - evaluation_df["predicted_sales"]
).abs()

# Calculate Mean Absolute Error (MAE)
mae = evaluation_df["absolute_error"].mean()

st.metric(
    "Seasonal Naive Forecast — Mean Absolute Error (MAE)",
    f"{mae:.2f} units"
)
st.subheader("Actual vs Predicted Daily Sales")

comparison_df = evaluation_df.tail(90).set_index("date")

st.line_chart(
    comparison_df[["sales", "predicted_sales"]]
)

st.caption(
    "Comparison of actual and predicted daily sales "
    "over the last 90 days of historical data."
)
st.subheader("Forecast Model Evaluation")

st.write(
    f"The Seasonal Naive Model has a Mean Absolute Error "
    f"of {mae:.2f} units per day."
)

st.info(
    "This model uses sales from the same weekday one week "
    "earlier as its prediction. It provides a simple baseline "
    "for comparing more advanced forecasting models."
)
st.divider()

st.header("Forecasting Model 2 — 7-Day Moving Average")

# Sort historical observations chronologically
model2_df = df.sort_values("date").copy()

# Predict each day using the previous 7 days of sales
model2_df["ma7_prediction"] = (
    model2_df["sales"]
    .shift(1)
    .rolling(window=7)
    .mean()
)

# Calculate forecasting accuracy
model2_eval = model2_df.dropna(
    subset=["ma7_prediction"]
).copy()

model2_mae = (
    model2_eval["sales"] - model2_eval["ma7_prediction"]
).abs().mean()

st.metric(
    "7-Day Moving Average — MAE",
    f"{model2_mae:.2f} units"
)

st.write(
    "This model predicts daily sales using the average "
    "sales from the previous seven days."
)
st.divider()

st.header("Forecasting Model Comparison")

comparison_results = pd.DataFrame({
    "Model": [
        "Seasonal Naive",
        "7-Day Moving Average"
    ],
    "MAE": [
        mae,
        model2_mae
    ]
})

st.dataframe(
    comparison_results,
    hide_index=True
)
st.divider()

st.header("Forecasting Model Test Evaluation")

# Use the final 90 days for testing
test_data = df.sort_values("date").tail(90)

# Compare predictions for the same test dates
seasonal_test = evaluation_df[
    evaluation_df["date"].isin(test_data["date"])
]

moving_avg_test = model2_eval[
    model2_eval["date"].isin(test_data["date"])
]

# Calculate test-period MAE
seasonal_test_mae = (
    seasonal_test["sales"] - seasonal_test["predicted_sales"]
).abs().mean()

moving_avg_test_mae = (
    moving_avg_test["sales"] - moving_avg_test["ma7_prediction"]
).abs().mean()

st.write("Forecasting accuracy over the last 90 historical days:")

col1, col2 = st.columns(2)

with col1:
    st.metric("Seasonal Naive Test MAE", f"{seasonal_test_mae:.2f}")

with col2:
    st.metric("Moving Average Test MAE", f"{moving_avg_test_mae:.2f}")

st.subheader("Best Forecasting Model")

if moving_avg_test_mae < seasonal_test_mae:
    best_model = "7-Day Moving Average"
    best_mae = moving_avg_test_mae
else:
    best_model = "Seasonal Naive"
    best_mae = seasonal_test_mae

st.success(f"Selected Model: {best_model}")

st.write(
    f"The selected model achieved a test MAE "
    f"of {best_mae:.2f} units per day."
)
st.divider()

st.header("Inventory Optimization")

st.write(
    "This section uses historical demand to estimate "
    "inventory requirements, safety stock, and reorder points."
)

# Calculate average daily demand
average_daily_demand = df["sales"].mean()

st.metric(
    "Average Daily Demand",
    f"{average_daily_demand:.2f} units"
)
st.subheader("Lead-Time Demand")

# Assumed supplier delivery time
lead_time_days = 7

# Calculate expected demand during lead time
lead_time_demand = average_daily_demand * lead_time_days

st.metric(
    "Expected Demand During Lead Time",
    f"{lead_time_demand:.2f} units"
)

st.caption(
    "Assumes a supplier lead time of 7 days."
)
st.divider()

st.subheader("Safety Stock Calculation")

# Calculate variation in daily sales
daily_demand_std = df["sales"].std()

# Z-score for a 95% cycle service level
z_score = 1.65

# Calculate safety stock
safety_stock = (
    z_score * daily_demand_std * (lead_time_days ** 0.5)
)

st.metric(
    "Recommended Safety Stock",
    f"{safety_stock:.0f} units"
)

st.caption(
    "Safety stock assumes a fixed 7-day lead time, "
    "independent daily demand, and a 95% target cycle service level."
)
st.divider()

st.subheader("Reorder Point Calculation")

# Calculate reorder point
reorder_point = lead_time_demand + safety_stock

# Round up to a whole unit
import math
reorder_point = math.ceil(reorder_point)

st.metric(
    "Recommended Reorder Point",
    f"{reorder_point} units"
)

st.write(
    "When the inventory position reaches or falls below "
    "this level, a new stock order should be placed."
)
st.divider()

st.subheader("Inventory Reorder Alert")

current_inventory = st.number_input(
    "Enter Current Inventory Position (units)",
    min_value=0,
    value=300,
    step=1
)

if current_inventory <= reorder_point:
    st.error("Reorder Alert: Inventory has reached the reorder point!")
else:
    st.success("Inventory Status: Stock is above the reorder point.")
    st.divider()

st.subheader("Suggested Replenishment Quantity")

# Target inventory coverage period
coverage_days = 14

# Calculate target inventory position
target_inventory = math.ceil(
    average_daily_demand * coverage_days + safety_stock
)

# Calculate suggested order quantity
if current_inventory <= reorder_point:
    suggested_order = max(
        0, target_inventory - current_inventory
    )
else:
    suggested_order = 0

st.metric(
    "Suggested Order Quantity",
    f"{suggested_order} units"
)

st.caption(
    "Illustrative order-up-to policy using 14 days "
    "of demand coverage plus safety stock."
)
st.divider()

st.subheader("Inventory Decision Summary")

if current_inventory <= reorder_point:
    st.warning(
        f"Replenishment is recommended. "
        f"Current inventory position is {current_inventory} units, "
        f"below the reorder point of {reorder_point} units."
    )

    st.write(
        f"Suggested action: Order {suggested_order} units "
        f"to reach the target inventory position "
        f"of {target_inventory} units."
    )
else:
    st.success(
        "No replenishment order is currently required."
    )
    st.divider()

st.header("Project Summary")

st.write(
    "This Walmart M5 Inventory Optimization Dashboard "
    "analyzes historical sales, compares demand forecasting "
    "models, and provides inventory replenishment recommendations."
)

st.info(
    "Key features include demand visualization, forecasting "
    "accuracy evaluation, safety stock estimation, reorder "
    "point calculation, and interactive replenishment alerts."
)

st.caption(
    "Inventory recommendations are illustrative and based on "
    "historical sales and assumed supplier lead times. "
    "Actual inventory balances are not included in the M5 dataset."
)