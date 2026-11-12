import streamlit as st
import pandas as pd
st.set_page_config(
    page_title="Walmart M5 Inventory Dashboard",
    layout="wide"
)

st.title("Walmart M5 Inventory Optimization Dashboard")

st.write("Demand forecasting and inventory optimization using the Walmart M5 dataset.")
st.divider()

# ==================================================
# DASHBOARD ORGANIZATION
# ==================================================

# Separate Nevia's existing single-SKU analysis
# from Anish's expanded multi-SKU analysis.
tab_nevia, tab_anish = st.tabs([
    "Nevia | Single-SKU Analysis",
    "Anish | Multi-SKU Analysis"
])

# ==================================================
# TAB 1: NEVIA'S ORIGINAL SINGLE-SKU DASHBOARD
# ==================================================

with tab_nevia:

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

# ==================================================
# TAB 2: ANISH'S MULTI-SKU ANALYSIS
# ==================================================

with tab_anish:

    st.header("Multi-SKU Demand Analysis")

    st.write(
        "This section expands the Walmart M5 analysis "
        "from one SKU-store combination to nine "
        "representative combinations across four "
        "demand-frequency segments."
    )

    st.info(
        "Analysis includes data cleaning, demand "
        "segmentation, exploratory analysis, and "
        "preliminary forecasting comparisons."
    )

    # Load the cleaned nine-SKU dataset
    multi_df = pd.read_csv(
        "walmart_m5_nine_sku_cleaned.csv"
    )

    # Prepare date values for time-series charts
    multi_df["date"] = pd.to_datetime(
        multi_df["date"]
    )

    # Display initial dataset metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Daily Observations",
            f"{len(multi_df):,}"
        )

    with col2:
        st.metric(
            "SKU-Store Combinations",
            multi_df[
                ["item_id", "store_id"]
            ].drop_duplicates().shape[0]
        )

    with col3:
        st.metric(
            "Missing Sales Values",
            int(multi_df["sales"].isna().sum())
        )

    st.subheader("Cleaned Modeling Dataset")

    st.dataframe(
        multi_df.head(20),
        use_container_width=True
    )

    st.caption(
        "The nine-SKU dataset includes historical sales, "
        "calendar information, and weekly selling prices. "
        "Pre-price-history observations were excluded "
        "from the modeling dataset."
    )

    # ==================================================
    # DEMAND SEGMENTATION - FULL WALMART M5 DATASET
    # ==================================================

    st.divider()
    st.subheader("Dataset-Wide Demand Segmentation")

    st.write(
        "The original Walmart M5 sales dataset contains "
        "30,490 SKU-store combinations. We classified "
        "these combinations according to the percentage "
        "of active selling days with zero recorded sales."
    )

    # Dataset-wide segmentation results from Colab
    segment_data = pd.DataFrame({
        "Demand Segment": [
            "Consistent sales",
            "Moderate intermittency",
            "High intermittency",
            "Very high intermittency"
        ],
        "SKU-Store Count": [
            1994, 7975, 13111, 7410
        ],
        "Percentage": [
            6.54, 26.16, 43.00, 24.30
        ],
        "Average Daily Sales": [
            6.90, 2.25, 0.62, 0.16
        ],
        "Average CV": [
            0.80, 1.18, 1.79, 3.18
        ]
    })

    st.bar_chart(
        segment_data.set_index(
            "Demand Segment"
        )["SKU-Store Count"]
    )

    st.dataframe(
        segment_data,
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "Key finding: 67.3% of SKU-store combinations "
        "exhibit high or very high sales intermittency. "
        "This supports evaluating forecasting and "
        "replenishment methods across different "
        "demand profiles."
    )

    st.caption(
        "These dataset-wide results were calculated "
        "from the original M5 sales and pricing files "
        "in the Colab analysis. They are not derived "
        "from the nine-SKU modeling CSV."
    )

    # ==================================================
    # NINE-SKU EXPLORATORY DATA ANALYSIS
    # ==================================================

    st.divider()
    st.subheader("Nine-SKU Sales Comparison")

    st.write(
        "Nine SKU-store combinations were selected across "
        "four demand-frequency segments. This analysis "
        "compares average recorded daily sales during "
        "each product's provisional active selling period."
    )

    # Calculate average daily sales directly from
    # Anish's cleaned nine-SKU dataset.
    sku_sales_summary = (
        multi_df
        .groupby(["item_id", "store_id"])["sales"]
        .mean()
        .reset_index(name="Average Daily Sales")
    )

    # Create readable labels for the dashboard
    sku_sales_summary["SKU-Store"] = (
        sku_sales_summary["item_id"]
        + " | "
        + sku_sales_summary["store_id"]
    )

    # Sort products by average daily sales
    sku_sales_summary = sku_sales_summary.sort_values(
        "Average Daily Sales",
        ascending=False
    )

    st.write("#### Average Daily Sales by SKU-Store")

    st.bar_chart(
        sku_sales_summary.set_index("SKU-Store")[
            "Average Daily Sales"
        ]
    )

    # Display exact values
    st.dataframe(
        sku_sales_summary[
            ["SKU-Store", "Average Daily Sales"]
        ].round(2),
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "Nevia's original SKU (FOODS_3_586 at CA_2) "
        "averages approximately 33.19 units per day, "
        "while some highly intermittent products "
        "average fewer than 0.2 units. These differences "
        "support evaluating SKU-specific forecasting "
        "and inventory parameters."
    )

    # ==================================================
    # ZERO-SALES FREQUENCY AND DEMAND VARIABILITY
    # ==================================================

    st.divider()
    st.subheader("Sales Intermittency and Demand Variability")

    st.write(
        "This analysis compares how frequently each "
        "SKU-store combination records zero sales "
        "and how variable its daily sales are."
    )

    # Calculate statistics from the cleaned dataset
    sku_variability = (
        multi_df.groupby(["item_id", "store_id"])["sales"]
        .agg(
            mean_sales="mean",
            std_sales=lambda s: s.std(ddof=0),
            zero_sales_pct=lambda s: (s == 0).mean() * 100
        )
        .reset_index()
    )

    # Calculate coefficient of variation
    sku_variability["cv"] = (
        sku_variability["std_sales"] /
        sku_variability["mean_sales"]
    )

    # Create readable labels
    sku_variability["SKU-Store"] = (
        sku_variability["item_id"] + " | " +
        sku_variability["store_id"]
    )

    # Sort from highest to lowest zero-sales frequency
    sku_variability = sku_variability.sort_values(
        "zero_sales_pct", ascending=False
    )

    # Visualization 1: Zero-sales percentage
    st.write("#### Percentage of Days With Zero Sales")

    st.bar_chart(
        sku_variability,
        x="SKU-Store",
        y="zero_sales_pct",
        horizontal=True
    )

    # Visualization 2: Coefficient of variation
    st.write("#### Relative Demand Variability (CV)")

    st.bar_chart(
        sku_variability,
        x="SKU-Store",
        y="cv",
        horizontal=True
    )

    # Numerical summary
    st.write("#### Demand Characteristics by SKU")

    st.dataframe(
        sku_variability[
            ["SKU-Store", "zero_sales_pct", "cv"]
        ].rename(columns={
            "zero_sales_pct": "Zero-Sales %",
            "cv": "Coefficient of Variation"
        }).round(2),
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "Key finding: Products with more frequent "
        "zero-sales days generally exhibit greater "
        "relative demand variability. This supports "
        "evaluating forecasting methods across "
        "different demand segments."
    )

    # ==================================================
    # INTERACTIVE HISTORICAL SALES TRENDS
    # ==================================================

    st.divider()
    st.subheader("Historical Sales Trends by SKU")

    st.write(
        "Select a SKU-store combination to examine "
        "historical daily sales and a 28-day rolling "
        "average. The rolling average smooths short-term "
        "fluctuations and highlights broader trends."
    )

    # Create a readable SKU-store identifier
    trend_df = multi_df.copy()

    trend_df["SKU-Store"] = (
        trend_df["item_id"]
        + " | "
        + trend_df["store_id"]
    )

    # Build selection options from our cleaned dataset
    sku_options = sorted(
        trend_df["SKU-Store"].unique()
    )

    selected_sku = st.selectbox(
        "Select SKU-Store Combination",
        options=sku_options,
        key="anish_trend_sku"
    )

    # Extract the selected product's history
    selected_history = trend_df[
        trend_df["SKU-Store"] == selected_sku
    ].sort_values("date").copy()

    # Calculate historical 28-day rolling average
    selected_history["28-Day Rolling Average"] = (
        selected_history["sales"]
        .rolling(window=28, min_periods=7)
        .mean()
    )

    # Prepare chart columns
    trend_chart = (
        selected_history[
            ["date", "sales", "28-Day Rolling Average"]
        ]
        .rename(columns={"sales": "Daily Sales"})
        .set_index("date")
    )

    # Interactive time-series visualization
    st.line_chart(trend_chart)

    # Display useful demand statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Daily Sales",
            f"{selected_history['sales'].mean():.2f}"
        )

    with col2:
        st.metric(
            "Maximum Daily Sales",
            int(selected_history["sales"].max())
        )

    with col3:
        st.metric(
            "Zero-Sales Days",
            f"{(selected_history['sales'] == 0).mean() * 100:.2f}%"
        )

    st.caption(
        "Historical sales are used as a proxy for demand. "
        "Zero-sales days may reflect a lack of purchases "
        "or unobserved product availability constraints. "
        "The rolling average is descriptive, not a "
        "future-demand forecast."
    )

    # ==================================================
    # CHRONOLOGICAL DATA PARTITIONING
    # ==================================================

    st.divider()
    st.subheader("Chronological Data Partitioning")

    st.write(
        "The cleaned nine-SKU dataset was divided into "
        "training, validation, and testing periods "
        "using chronological dates rather than "
        "random sampling."
    )

    # Create a separate copy for partition analysis
    partition_df = multi_df.copy()

    # Identify the final historical sales date
    last_date = partition_df["date"].max()

    # Final 180 days: Testing
    test_start = last_date - pd.Timedelta(days=179)

    # Previous 180 days: Validation
    validation_start = test_start - pd.Timedelta(days=180)

    # Assign observations to chronological partitions
    import numpy as np

    partition_df["Partition"] = np.select(
        [
            partition_df["date"] < validation_start,
            partition_df["date"] < test_start
        ],
        ["Train", "Validation"],
        default="Test"
    )

    # Summarize observation counts
    partition_summary = (
        partition_df["Partition"]
        .value_counts()
        .reindex(["Train", "Validation", "Test"])
        .reset_index()
    )

    partition_summary.columns = [
        "Partition", "Observations"
    ]

    # Calculate percentage of observations
    partition_summary["Percentage"] = (
        partition_summary["Observations"]
        / len(partition_df) * 100
    ).round(2)

    # Display summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Training Observations",
            f"{partition_summary.loc[0, 'Observations']:,}"
        )

    with col2:
        st.metric(
            "Validation Observations",
            f"{partition_summary.loc[1, 'Observations']:,}"
        )

    with col3:
        st.metric(
            "Testing Observations",
            f"{partition_summary.loc[2, 'Observations']:,}"
        )

    st.write("#### Dataset Partition Summary")

    st.bar_chart(
        partition_summary.set_index("Partition")[
            "Observations"
        ]
    )

    st.dataframe(
        partition_summary,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        f"Validation period: "
        f"{validation_start.date()} to "
        f"{(test_start - pd.Timedelta(days=1)).date()}. "
        f"Testing period: {test_start.date()} to "
        f"{last_date.date()}. "
        "Earlier observations were used for training."
    )

    st.info(
        "Chronological partitioning preserves the "
        "time-series structure. Validation data support "
        "forecast model selection, while test data "
        "remain reserved for final evaluation."
    )

    # ==================================================
    # FOUR-MODEL FORECASTING COMPARISON
    # ==================================================

    st.divider()
    st.subheader("Preliminary Forecasting Model Comparison")

    st.write(
        "Four forecasting methods were evaluated across "
        "nine selected SKU-store combinations using "
        "a common 180-day validation period. "
        "Mean Absolute Error (MAE) measures average "
        "forecast error in units sold per day."
    )

    # Validation results calculated in Colab
    # These are historical analysis outputs,
    # not forecasts recalculated by Streamlit.
    forecast_results = pd.DataFrame({
        "SKU-Store": [
            "FOODS_3_586 | CA_2",
            "FOODS_3_734 | CA_1",
            "HOBBIES_1_035 | WI_3",
            "HOBBIES_1_078 | TX_3",
            "HOBBIES_1_285 | CA_1",
            "HOUSEHOLD_1_083 | CA_3",
            "HOUSEHOLD_1_226 | WI_3",
            "HOUSEHOLD_1_347 | WI_1",
            "HOUSEHOLD_1_487 | WI_2"
        ],
        "Seasonal Naive": [
            10.444, 2.972, 0.206, 0.689, 0.222,
            5.539, 0.861, 1.267, 0.489
        ],
        "7-Day Moving Average": [
            9.834, 2.231, 0.180, 0.622, 0.226,
            4.291, 0.644, 1.077, 0.389
        ],
        "Zero-Demand": [
            45.094, 4.761, 0.106, 0.494, 0.117,
            10.411, 0.739, 1.206, 0.544
        ],
        "TSB": [
            10.111, 2.235, 0.183, 0.626, 0.218,
            4.330, 0.629, 1.076, 0.387
        ]
    })

    # Determine the lowest-MAE model for each SKU
    model_names = [
        "Seasonal Naive",
        "7-Day Moving Average",
        "Zero-Demand",
        "TSB"
    ]

    forecast_results["Lowest-MAE Model"] = (
        forecast_results[model_names].idxmin(axis=1)
    )

    st.write("#### Forecast Accuracy by SKU-Store")

    st.dataframe(
        forecast_results.round(3),
        hide_index=True,
        use_container_width=True
    )

    # Count how often each method has lowest MAE
    model_wins = (
        forecast_results["Lowest-MAE Model"]
        .value_counts()
        .reindex(model_names, fill_value=0)
        .rename_axis("Forecasting Method")
        .reset_index(name="SKU-Store Wins")
    )

    st.write("#### Models With Lowest Validation MAE")

    st.bar_chart(
        model_wins,
        x="Forecasting Method",
        y="SKU-Store Wins",
        horizontal=True
    )

    st.info(
        "Validation finding: The 7-day moving average, "
        "TSB, and zero-demand baseline each achieved "
        "the lowest MAE for three SKU-store combinations. "
        "No single method performed best across all nine."
    )

    st.warning(
        "A zero-demand forecast can achieve low MAE "
        "for intermittent products while failing to "
        "anticipate occasional purchases. Therefore, "
        "forecast accuracy alone is insufficient for "
        "selecting an inventory replenishment policy."
    )

    st.caption(
        "Results were calculated in Google Colab using "
        "rolling one-day-ahead predictions. The test "
        "partition has not been evaluated."
    )

    # ==================================================
    # EXISTING INVENTORY POLICY AND FUTURE EXTENSION
    # ==================================================

    st.divider()
    st.subheader("Inventory Replenishment Policy: Current Status")

    st.write(
        "The existing single-SKU dashboard developed "
        "by Nevia includes an illustrative inventory "
        "replenishment policy. Our multi-SKU analysis "
        "extends the forecasting and demand evidence "
        "that could inform future improvements."
    )

    st.markdown("#### Existing Policy — Nevia's Implementation")

    st.markdown(
        """
        The current policy uses:

        - **Supplier lead time:** Assumed 7 days.
        - **Safety stock:** 1.65 × daily sales standard
          deviation × square root of lead time.
        - **Reorder point:** Expected lead-time sales
          plus safety stock.
        - **Order-up-to target:** 14 days of average
          sales plus safety stock.
        - **Order quantity:** Raise inventory position
          to the order-up-to target when the reorder
          threshold is reached.

        The interactive recommendation is available
        in the **Nevia | Single-SKU Analysis** tab.
        """
    )

    st.markdown("#### Optimization Model — Proposed Future Work")

    st.write(
        "The existing formula-based policy provides "
        "a baseline for future optimization. We propose "
        "evaluating SKU-specific reorder points and "
        "order-up-to levels rather than creating a "
        "replacement policy at this stage."
    )

    st.markdown(
        """
        **Proposed decision variables**
        - Reorder point (r)
        - Order-up-to level (S)
        - Replenishment quantity (Q), determined by
          inventory position and the policy thresholds

        **Proposed objective**

        Minimize expected holding, ordering, and
        shortage costs.

        **Proposed constraints**
        - Nonnegative replenishment quantities
        - Order-up-to level greater than or equal
          to reorder point
        - Assumed inventory capacity limits
        - Target simulated inventory service level

        **Planned solution strategy**

        Compare alternative policy parameters through
        simulated demand and supplier lead times,
        using the existing policy as a baseline.
        """
    )

    st.warning(
        "Optimization and inventory-policy simulation "
        "have not yet been implemented. Walmart M5 "
        "does not contain historical stock levels, "
        "supplier lead times, or inventory costs. "
        "These inputs would require documented assumptions."
    )

    # ==================================================
    # DATA CLEANING AND QUALITY ASSURANCE
    # ==================================================

    st.divider()
    st.subheader("Data Cleaning and Quality Assurance")

    st.write(
        "Before forecasting, we audited the original "
        "Walmart M5 sales, calendar, and pricing files. "
        "The audit assessed missing values, duplicate "
        "keys, invalid observations, pricing coverage, "
        "and potential sales outliers."
    )

    # Audited findings from our Google Colab notebook
    audit_results = pd.DataFrame({
        "Data Quality Check": [
            "SKU-store combinations",
            "Missing sales values",
            "Missing source pricing values",
            "Missing calendar-event entries",
            "Duplicate source keys",
            "Negative or fractional sales observations",
            "Weekly pricing coverage",
            "Positive sales before first recorded price",
            "Zero-IQR SKU-store combinations",
            "Revised potential IQR outlier flags",
            "Cleaned nine-SKU daily observations"
        ],
        "Result": [
            "30,490",
            "0",
            "0",
            "7,542",
            "0",
            "0",
            "79.27%",
            "0",
            "9,971",
            "1,263,749",
            f"{len(multi_df):,}"
        ]
    })

    st.dataframe(
        audit_results,
        hide_index=True,
        use_container_width=True
    )

    st.markdown("#### Cleaning Decisions")

    st.markdown(
        """
        **Missing calendar events**

        Empty event-name and event-type fields were
        interpreted as no corresponding listed event
        and labeled "No Event" in the modeling dataset.

        **Pre-price selling history**

        Observations before each SKU-store's first
        recorded price were excluded from the modeling
        dataset. No positive sales were identified
        during those periods.

        **Zero-sales observations**

        Zero-sales days within the provisional active
        selling periods were retained.

        **Outlier treatment**

        The initial IQR method overflagged observations
        for intermittent products with zero IQR.

        We therefore separated the 9,971 zero-IQR
        combinations from conventional IQR screening.

        The remaining potential outliers were flagged
        for investigation rather than deleted or
        automatically adjusted.

        **Data integrity**

        Original sales and selling-price values were
        preserved. Source datasets were not overwritten.
        """
    )

    st.info(
        "The resulting nine-SKU modeling dataset "
        "contains 15,509 daily observations, "
        "with no missing sales or selling prices "
        "and no duplicate SKU-store-date records."
    )

    st.caption(
        "The dataset-wide audit results were generated "
        "in Google Colab using the original Walmart M5 "
        "source files. The cleaned CSV used by this "
        "dashboard contains the nine selected SKU-store "
        "combinations."
    )
