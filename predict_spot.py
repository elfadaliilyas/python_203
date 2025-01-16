import yfinance as yf
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from textblob import TextBlob
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from bs4 import BeautifulSoup
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
from pandas.tseries.offsets import BDay


def fetch_brent_news_paged(start_page, end_page, start_date, end_date, news_data):
    base_url = "https://oilprice.com/Latest-Energy-News/World-News/Page-{}.html"
    # Convert date range to datetime objects

    start_date_obj = datetime.strptime(str(start_date), "%Y-%m-%d")
    end_date_obj = datetime.strptime(str(end_date), "%Y-%m-%d")
    for page in range(start_page, end_page):
        url = base_url.format(page)
        response = requests.get(url)
        if response.status_code != 200:
            print(
                f"Failed to fetch data from {url}. HTTP Status Code: {response.status_code}"
            )
            break
        # Parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")
        articles = soup.find_all("div", class_="categoryArticle")
        if not articles:
            print(f"No articles found on page {page}. Stopping.")
            break
        # Extract headlines and dates
        for article in articles:
            title_tag = article.find("h2")
            title = title_tag.text.strip() if title_tag else None
            headline_tag = article.find("p")
            headline = headline_tag.text.strip() if headline_tag else None
            # Extract date
            date_part = headline.split("at")[0].strip()
            # Convert to datetime object
            extracted_date = datetime.strptime(date_part, "%b %d, %Y")
            formatted_date = extracted_date
            # Filter by the specified date range
            if start_date_obj <= formatted_date <= end_date_obj:
                news_data.append({"Date": formatted_date, "Headline": title})
    if formatted_date >= start_date_obj:
        if calculate_month_difference(end_date_obj, formatted_date) > 1:
            start_page += 25
            end_page += 25
        start_page += 1
        end_page += 1
        fetch_brent_news_paged(start_page, end_page, start_date, end_date, news_data)
    return pd.DataFrame(news_data)


def calculate_month_difference(date1, date2):
    # Calculate the difference in years and months
    year_diff = date2.year - date1.year
    month_diff = date2.month - date1.month
    # Combine years and months into total months
    total_months = year_diff * 12 + month_diff
    # Adjust for days if necessary (to count incomplete months)
    if date2.day < date1.day:
        total_months -= 1
    return total_months


def evaluate_headlines_sentiment_with_daily_sentiment(df):
    # Analyze sentiment for each headline
    def analyze_sentiment(headline):
        blob = TextBlob(headline)
        polarity = blob.sentiment.polarity
        return polarity

    df["Polarity"] = df["Headline"].apply(analyze_sentiment)
    df["Sentiment"] = df["Polarity"].apply(
        lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral"
    )
    # Calculate daily average sentiment
    daily_sentiment = df.groupby("Date")["Polarity"].mean().reset_index()
    daily_sentiment["Daily Sentiment"] = daily_sentiment["Polarity"].apply(
        lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral"
    )
    daily_sentiment["Daily Sentiment"] = daily_sentiment["Daily Sentiment"].map(
        {"Positive": 1, "Negative": 0, "Neutral": 0}
    )
    return daily_sentiment


def get_daily_exchange_rates_yfinance(
    base_currencies, target_currency, start_date, end_date
):
    all_data = []  # List to store all dataframes
    for base_currency in base_currencies:
        # Construct the ticker for the currency pair (e.g., EURUSD=X)
        ticker = f"{base_currency}{target_currency}=X"
        data = (
            yf.download(ticker, start=start_date, end=end_date, interval="1d")["Close"]
            .pct_change()
            .dropna()
        )
        data = data.reset_index()
        data = data.rename(columns={"Close": "Exchange Rate to USD"})
        all_data.append(data)
    merged_df = pd.concat(all_data, axis=1)  # Join on Date by default
    # Identify all columns named 'Date'
    date_columns = [col for col in merged_df.columns if col == "Date"]
    # If there are multiple 'Date' columns, drop all but the first one
    if len(date_columns) > 1:
        merged_df = merged_df.loc[
            :, ~((merged_df.columns == "Date") & (merged_df.columns.duplicated()))
        ]
    return merged_df


# Fetch historical data for Brent Crude Oil (BZ=F)
def fetch_historical_data(start_date, end_date):
    brent_data = yf.download("BZ=F", start=start_date, end=end_date)
    return brent_data["Close"]


# Fetch real-time price for Brent Crude Oil
def fetch_real_time_price():
    brent = yf.Ticker("BZ=F")
    return brent.history(period="1d")["Close"].iloc[-1]


def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    return data


def calculate_correlations(data):
    return data.corr()


def check_stationarity(series):
    result = adfuller(series.dropna())  # Drop NaN values for the test
    print("ADF Statistic:", result[0])


def stationary_graph(start_page, end_page, start_date, end_date):
    brent_close_prices = fetch_historical_data(start_date=start_date, end_date=end_date)
    check_stationarity(brent_close_prices)

    ## Brent time series is not stationary at all, we'll differentiate it to make it stationary.
    differenced_series = brent_close_prices.diff()
    check_stationarity(differenced_series)
    differenced_series = differenced_series.reset_index().dropna()
    brent_present = differenced_series.rename(columns={"BZ=F": "Brent Historical"})

    return brent_present


def main_df(start_page, end_page, start_date, end_date, news_data):

    # Fetch and display news
    sentiment_data = fetch_brent_news_paged(
        start_page, end_page, start_date, end_date, news_data
    )
    df_with_sentiment = evaluate_headlines_sentiment_with_daily_sentiment(
        sentiment_data
    )
    df_with_sentiment = df_with_sentiment[["Date", "Daily Sentiment"]]

    brent_present = stationary_graph(start_page, end_page, start_date, end_date)
    brent_futures = fetch_data("BZ=F", start_date=start_date, end_date=end_date)
    brent_futures = brent_futures.reset_index()
    brent_futures = brent_futures.rename(columns={"BZ=F": "Brent Futures"})

    ## Get all the exchange rates from the top 5 different currencies

    top_currencies = ["EUR", "JPY"]
    exchange_rates_df = get_daily_exchange_rates_yfinance(
        base_currencies=top_currencies,
        target_currency="USD",
        start_date=start_date,
        end_date=end_date,
    )

    ## Get all the correlations from the different tickers

    tickers = [
        "XOM",
        "CVX",
        "SHEL",
        "TTE",
        "BP",
        "COP",
        "E",
        "OXY",
        "EQNR",
        "REPYY",
    ]  # Brent, ExxonMobil, Chevron, Shell, Eni,Occidental Petroleum, Conoco, Equinor, Repsol
    price_data = fetch_data(tickers, start_date, end_date)
    daily_pct_changes = price_data.pct_change().dropna()
    # Step 3: Calculate the average performance across all companies
    average_performance = daily_pct_changes.mean(axis=1)
    # Add the average performance to the dataset
    daily_pct_changes["Average Performance Top Oil Companies"] = average_performance
    daily_pct_changes = daily_pct_changes.reset_index()
    daily_pct_changes = daily_pct_changes[
        ["Date", "Average Performance Top Oil Companies"]
    ]

    ## Get the CBOE Crude Oil Volatility Index

    ovx_data = fetch_data("^OVX", start_date, end_date)
    ovx_data = ovx_data.reset_index()
    ovx_data = ovx_data.rename(columns={"^OVX": "OVX Index"})
    exchange_rates_df["Date"] = pd.to_datetime(exchange_rates_df["Date"])
    ovx_data["Date"] = pd.to_datetime(ovx_data["Date"])
    brent_futures["Date"] = pd.to_datetime(brent_futures["Date"])

    # Merge all DataFrames on 'date'
    merged_df = exchange_rates_df.merge(ovx_data, on="Date", how="inner")
    merged_df = merged_df.merge(brent_futures, on="Date", how="inner")
    merged_df = merged_df.merge(brent_present, on="Date", how="inner")
    merged_df = merged_df.merge(df_with_sentiment, on="Date", how="inner")
    merged_df = merged_df.merge(daily_pct_changes, on="Date", how="inner")
    return merged_df


def linear_reg(start_page, end_page, start_date, end_date, news_data):
    merged_df = main_df(start_page, end_page, start_date, end_date, news_data)
    y = merged_df["Brent Historical"]
    X = merged_df.drop(columns=["Brent Historical", "Date"])
    X = sm.add_constant(X)
    # Fit the OLS model
    model = sm.OLS(y, X).fit()
    model_demand = LinearRegression()
    model_demand.fit(X, y)
    predict_spot = model.predict(X)

    # Start from the next trading day
    new_start_date = merged_df["Date"].iloc[-1] + BDay(1)
    # Generate trading days to match the same number of row for historical
    future_dates = pd.date_range(
        start=new_start_date, periods=len(merged_df) + 1, freq=BDay()
    )

    actual_values = fetch_historical_data(
        start_date=new_start_date, end_date=future_dates[-1]
    ).reset_index()

    reconstructed_values = [
        fetch_historical_data(start_date=start_date, end_date=end_date).iloc[-1]
    ]  # Start with the initial value
    for diff in predict_spot:  # Skip the first NaN value
        last_value = reconstructed_values[-1]
        reconstructed_values.append(last_value + diff)

    # Convert to a pandas Series
    original_series = pd.DataFrame(reconstructed_values).reset_index()

    # Create the new DataFrame for futures
    future_df = pd.DataFrame(
        {
            "Date": future_dates,
            "actual_values": actual_values["BZ=F"],
            "predicted_spot_price": original_series["BZ=F"],
            "Brent Futures": merged_df["Brent Futures"],
        }
    ).dropna()

    future_df["Behaviour"] = future_df.apply(
        lambda row: (
            "BUY" if row["predicted_spot_price"] > row["Brent Futures"] else "SELL"
        ),
        axis=1,
    )
    future_df["Realized PnL"] = future_df.apply(calculate_payoff, axis=1)
    future_df["Date"] = pd.to_datetime(
        future_df["Date"].astype(str).str.split(":").str[0]
    ).dt.strftime("%Y-%m-%d")
    future_df["Cumulative PnL"] = future_df["Realized PnL"].cumsum()
    return future_df, new_start_date, future_dates[-1]


def elastic_net(start_page, end_page, start_date, end_date, news_data):
    # Prepare data
    merged_df = main_df(start_page, end_page, start_date, end_date, news_data)
    y = merged_df["Brent Historical"]
    X = merged_df.drop(columns=["Brent Historical", "Date"])

    # Split into train/test sets
    X_train = merged_df.iloc[
        : int(round(len(merged_df) - len(merged_df) * 0.3, 0))
    ].drop(columns=["Brent Historical", "Date"])
    y_train = merged_df.iloc[: int(round(len(merged_df) - len(merged_df) * 0.3, 0))][
        "Brent Historical"
    ]
    X_test = merged_df.iloc[-int(round(len(merged_df) * 0.3, 0)) :].drop(
        columns=["Brent Historical", "Date"]
    )
    y_test = merged_df.iloc[-int(round(len(merged_df) * 0.3, 0)) :]["Brent Historical"]

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Elastic Net parameters
    param_grid = {
        "alpha": [0.01, 0.05, 0.1, 0.15, 0.2, 0.5],
        "l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9, 1],
    }
    elastic_net = ElasticNet(random_state=42)

    # GridSearchCV for Elastic Net
    grid_search = GridSearchCV(
        estimator=elastic_net,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        verbose=1,
    )
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(X_train_scaled)
    y_pred_test = best_model.predict(X_test_scaled)

    # Generate future dates and actual values
    new_start_date = merged_df["Date"].iloc[-1] + BDay(1)
    future_dates = pd.date_range(
        start=new_start_date, periods=len(merged_df) + 1, freq=BDay()
    )
    actual_values = fetch_historical_data(
        start_date=new_start_date, end_date=future_dates[-1]
    ).reset_index()

    # Reconstruct values
    predict_spot = np.concatenate((y_pred_train, y_pred_test))
    reconstructed_values = [
        fetch_historical_data(start_date=start_date, end_date=end_date)
        .reset_index()["BZ=F"]
        .iloc[-1]
    ]
    for diff in predict_spot:
        last_value = reconstructed_values[-1]
        reconstructed_values.append(last_value + diff)
    original_series = pd.DataFrame(reconstructed_values, columns=["BZ=F"])

    # Create the future DataFrames
    future_df = pd.DataFrame(
        {
            "Date": future_dates,
            "actual_values": actual_values["BZ=F"],
            "predicted_spot_price": original_series["BZ=F"],
            "Brent Futures": merged_df["Brent Futures"],
        }
    ).dropna()

    # Add behavior column
    future_df["Behaviour"] = future_df.apply(
        lambda row: (
            "BUY" if row["predicted_spot_price"] > row["Brent Futures"] else "SELL"
        ),
        axis=1,
    )
    future_df["Realized PnL"] = future_df.apply(calculate_payoff, axis=1)
    future_df["Date"] = pd.to_datetime(
        future_df["Date"].astype(str).str.split(":").str[0]
    ).dt.strftime("%Y-%m-%d")
    future_df["Cumulative PnL"] = future_df["Realized PnL"].cumsum()

    return future_df, new_start_date, future_dates[-1]


def calculate_payoff(row):
    if row["Behaviour"] == "BUY":
        return row["actual_values"] - row["Brent Futures"]
    elif row["Behaviour"] == "SELL":
        return row["Brent Futures"] - row["actual_values"]


