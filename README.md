How to open this file:
git clone this repo and if there are issues with different versions then type in these commands :
# Project Setup Guide

create a Conda environment by running `conda create --name name numpy=1.26.4 python=3.10`, then activate it with `conda activate name`. Next, install the required Python packages: `pip install - r requirements.txt`. Once these steps are completed, you're ready to explore and run the project!

# python_203
# Python_Programming : Click on launch.bat to open up the streamlit
# Commodities Market Sentiment Analysis and Portfolio Projection

## Project Overview

This project evaluates the performance of the strategy based on a date range chosen by the user through an intuitive interface. For instance, if the user selects a backtesting period from January 1, 2023, to March 1, 2023, the results will showcase the strategy's projected performance for the period from March 1, 2023, to May 1, 2023, as predicted by our linear regression model. It is important to note that due to model constraints, predictions are only available up to one month from the current date.
The models utilize a linear regression and an Elastic Net regression to predict spot prices and evaluate trading strategies based on the Brent Futures market.
---

## Data Sources
1. **Headlines Sentiment Analysis**: Extracted from a specialized website, providing an overall sentiment on the commodities market.
2. **Exchange Rates**:
   - EUR/USD
   - JPY/USD
3. **Brent Futures Prices**: Focus on contracts with 1-month maturity.
4. **Top 10 Oil Companies**: The average performance of the largest oil companies in the market.
5. **CBOE Crude Oil Volatility Index (OVX)**: Represents the market's expectation of 30-day volatility in crude oil prices.

---

## Methodology

### Models
1. **Linear Regression**: A baseline model for predicting spot prices.
2. **Elastic Net Regression**:
   - Combines L1 (Lasso) and L2 (Ridge) regularization to improve model performance and prevent overfitting.
   - Parameters are optimized using Grid Search.

### Trading Strategy
The strategy is based on comparing the predicted spot prices with the Brent Futures price (1-month maturity):

1. **Prediction and Comparison**:
   - For each day within the user-selected backtesting range, predict the spot price using both the linear regression and Elastic Net models.
   - Compare the predicted spot price to the Brent Futures price.

2. **Trading Decision**:
   - **If the predicted spot price > Brent Futures price**: Buy the Brent Futures contract and Sell the Brent Spot in 30 days.
   - **If the predicted spot price < Brent Futures price**: Sell the Brent Futures contract and Buy the Brent Spot in 30 days.

3. **Payoff Calculation**:
   - Track the outcome of each trade 30 days later (maturity).
   - Calculate the actual payoff based on the realized spot price at the end of the maturity period.

### Limitations
- **Backtesting Range**: The user selects the start and end dates for backtesting through an intuitive interface. For example, if the user specifies a backtesting period from January 1, 2023, to March 1, 2023, the model will project the strategy's performance for the period from March 1, 2023, to May 1, 2023, based on the linear regression predictions.
- **Portfolio Projection**:
   - Results cannot display portfolio values for dates beyond one month after the project launch date.

---

## User Interface (Streamlit)
The Streamlit interface allows users to:
1. Select a backtesting time range.
2. View predictions generated by both linear regression and Elastic Net models.
3. Analyze portfolio performance based on the trading strategy.

---

## Optimization
- **Elastic Net Grid Search**: Parameters for Elastic Net regression are optimized using Grid Search, balancing model complexity and performance.
---

## Example Workflow
1. **Backtesting Range**: User selects January 1, 2023, to March 1, 2023.
2. **Model Prediction Exemple**:
   - Linear Regression predicts a spot price higher than the Brent Futures price on April 1, 2023.
   - Decision: Buy the Brent Futures contract and Sell the Spot.
3. **Outcome**:
   - On April the 1st, 2023, the realized spot price is compared to the Futures price.
   - Payoff is calculated and added to the portfolio's performance.

---

## Results and Evaluation
The effectiveness of the trading strategy is evaluated by:
1. Comparing cumulative payoffs between linear regression and Elastic Net models.
2. Analyzing model accuracy and profitability over the user-selected backtesting range.
3. Visualizing portfolio performance trends.

---

## Future Work
- Integrate additional sentiment analysis techniques to refine predictions.
- Expand the dataset to include other commodities and economic indicators.
- Explore advanced machine learning models for improved accuracy and robustness.

--
