import pandas as pd
from brokers.forward_broker import ForwardTradingBroker
import numpy as np
from predict_spot import linear_reg, elastic_net
import streamlit as st
from datetime import date
import matplotlib.pyplot as plt


# Let's apply our strategy from January to March 2023. The data for January will be used
# to forecast the spot price for March.

# We are going to apply our strategy to February's data: buy or sell the forward,
# hoping to make a profit. The profit and loss (PnL) is calculated as the difference
# between the forward price and the spot price 30 days later.

# In this study, we do not take into consideration the discounting effect due to the
# risk-free rate in our PnL calculation.

# A better approach would be to use adaptive regression, either through online regression
# as we move forward, or offline regression. For simplicity, we tune our parameters
# using January's data. The same parameters are then used to predict the spot prices
# for both 1st March and 30th March.


def streamlit_interface(data, start_date, end_date, final_value, news_data):
    st.title("Streamlit Interface")
    st.write("Welcome to your Streamlit app! Use the options below to interact.")

    # Create a single row for inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Select Start Date", value=start_date)
    with col2:
        end_date = st.date_input("Select End Date", value=end_date)
    with col3:
        cash = st.number_input("Initial Cash", value=10, min_value=0)

    # Validate the selected date range
    if end_date <= start_date:
        st.error(
            "End Date must be later than Start Date. Please adjust your selection."
        )
        return

    # Run Linear Regression
    linear_data, new_start_date, new_end_date = linear_reg(
        start_page, end_page, start_date, end_date, news_data
    )
    broker = ForwardTradingBroker(cash)
    linear_pnl = broker.run_backtest(linear_data) - cash

    # Run ElasticNet Regression
    elastic_data, _, _ = elastic_net(
        start_page, end_page, start_date, end_date, news_data
    )
    elastic_pnl = broker.run_backtest(elastic_data) - cash

    st.write(
        f"We will apply this strategy on this date range: {pd.to_datetime(new_start_date).strftime('%Y-%m-%d')} to {pd.to_datetime(new_end_date).strftime('%Y-%m-%d')}"
    )

    # Create two columns for results
    col_left, col_right = st.columns(2)


    # Calculate Linear Regression Returns
    linear_data['Returns'] = linear_data['Cumulative PnL'].pct_change().fillna(0)
    linear_sharpe_ratio = (
        linear_data['Returns'].mean() / linear_data['Returns'].std()
        if linear_data['Returns'].std() != 0
        else 0
    )

    # Calculate ElasticNet Returns
    elastic_data['Returns'] = elastic_data['Cumulative PnL'].pct_change().fillna(0)
    elastic_sharpe_ratio = (
        elastic_data['Returns'].mean() / elastic_data['Returns'].std()
        if elastic_data['Returns'].std() != 0
        else 0
    )

    # Left Column - Linear Regression
    with col_left:
        st.header("Linear Regression Results")
        st.line_chart(linear_data.set_index("Date")["Cumulative PnL"])
        st.metric(
            label="Total PnL (Linear Regression)",
            value=f"${linear_data['Realized PnL'].sum():,.2f}",
        )
        st.metric(
            label="Sharpe Ratio (Linear Regression)",
            value=f"{linear_sharpe_ratio:.2f}",
        )
        st.dataframe(linear_data[["Date", "Behaviour", "Realized PnL"]])

    # Right Column - ElasticNet
    with col_right:
        st.header("Elastic Net Results")
        st.line_chart(elastic_data.set_index("Date")["Cumulative PnL"])
        st.metric(
            label="Total PnL (ElasticNet)",
            value=f"${elastic_data['Realized PnL'].sum():,.2f}",
        )
        st.metric(
            label="Sharpe Ratio (ElasticNet)",
            value=f"{elastic_sharpe_ratio:.2f}",
        )
        st.dataframe(elastic_data[["Date", "Behaviour", "Realized PnL"]])

    return data, start_date, end_date


if __name__ == "__main__":
    start_date = "2024-09-01"
    end_date = "2024-10-01"
    start_page = 1
    end_page = (
        start_page + 1
    )  # You can set a higher number; the algorithm stops automatically when needed
    news_data = []
    data = []
    final_value = 0
    # # 1. Initialiser l'interface
    data, start_date, end_date = streamlit_interface(
        data, start_date, end_date, final_value, news_data
    )

