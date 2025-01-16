import yfinance as yf
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from bs4 import BeautifulSoup
from textblob import TextBlob




def fetch_brent_news_paged(start_page, end_page, start_date, end_date,news_data):
    base_url = "https://oilprice.com/Latest-Energy-News/World-News/Page-{}.html"
    # Convert date range to datetime objects
    
    start_date_obj = datetime.strptime(str(start_date), "%Y-%m-%d")
    end_date_obj = datetime.strptime(str(end_date), "%Y-%m-%d")
    for page in range(start_page, end_page):
        url = base_url.format(page)
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch data from {url}. HTTP Status Code: {response.status_code}")
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
        if calculate_month_difference(end_date_obj,formatted_date) > 1:
            start_page += 25  
            end_page += 25
        start_page +=1
        end_page +=1
        fetch_brent_news_paged(start_page, end_page, start_date, end_date,news_data)  
    return pd.DataFrame(news_data)


# Fetch historical data for Brent Crude Oil (BZ=F)
def fetch_historical_data(start_date, end_date):
    brent_data = yf.download("BZ=F", start=start_date, end=end_date)
    return brent_data['Close']

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
    print("p-value:", result[1])
    for key, value in result[4].items():
        print(f"Critical Value ({key}): {value}")
    if result[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is NOT stationary.")

def stationary_graph(start_page, end_page, start_date, end_date):
    brent_close_prices = fetch_historical_data(start_date=start_date, end_date=end_date)
    check_stationarity(brent_close_prices)

    ## Brent time series is not stationary at all, we'll differentiate it to make it stationary.
    differenced_series = brent_close_prices.diff()
    check_stationarity(differenced_series)
    differenced_series = differenced_series.reset_index().dropna()
    brent_present = differenced_series.rename(columns={"BZ=F": "Brent Historical"})
 
    return brent_present  

def main_df(start_page, end_page, start_date, end_date,news_data):     
    
    # Fetch and display news
    sentiment_data = fetch_brent_news_paged(start_page, end_page, start_date, end_date,news_data)
    df_with_sentiment = evaluate_headlines_sentiment_with_daily_sentiment(sentiment_data)
    df_with_sentiment = df_with_sentiment[["Date","Daily Sentiment"]]
    
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
        end_date=end_date
    )
    
    ## Get all the correlations from the different tickers
    
    tickers = ["XOM", "CVX", "SHEL","TTE","BP","COP","E", "OXY","EQNR","REPYY"]  # Brent, ExxonMobil, Chevron, Shell, Eni,Occidental Petroleum, Conoco, Equinor, Repsol
    price_data = fetch_data(tickers, start_date, end_date)
    daily_pct_changes = price_data.pct_change().dropna()
    # Step 3: Calculate the average performance across all companies
    average_performance = daily_pct_changes.mean(axis=1)
    # Add the average performance to the dataset
    daily_pct_changes["Average Performance Top Oil Companies"] = average_performance
    daily_pct_changes = daily_pct_changes.reset_index()
    daily_pct_changes = daily_pct_changes[["Date","Average Performance Top Oil Companies"]]

    ## Get the CBOE Crude Oil Volatility Index
    
    ovx_data = fetch_data("^OVX",start_date, end_date)    
    ovx_data = ovx_data.reset_index()
    ovx_data = ovx_data.rename(columns={"^OVX": "OVX Index"})
    exchange_rates_df['Date'] = pd.to_datetime(exchange_rates_df['Date'])
    ovx_data['Date'] = pd.to_datetime(ovx_data['Date'])
    brent_futures['Date'] = pd.to_datetime(brent_futures['Date'])
    
    # Merge all DataFrames on 'date'
    merged_df = exchange_rates_df.merge(ovx_data, on='Date', how='inner')
    merged_df = merged_df.merge(brent_futures, on='Date', how='inner')
    merged_df = merged_df.merge(brent_present, on='Date', how='inner')
    merged_df = merged_df.merge(df_with_sentiment, on='Date', how='inner')
    merged_df = merged_df.merge(daily_pct_changes, on='Date', how='inner')
    return merged_df
 
def linear_reg(start_page, end_page, start_date, end_date,news_data):
    merged_df = main_df(start_page, end_page, start_date, end_date,news_data)
    y = merged_df['Brent Historical'] 
    X = merged_df.drop(columns=['Brent Historical','Date'])
    X = sm.add_constant(X)
    # Fit the OLS model
    model = sm.OLS(y, X).fit()   
    model_demand = LinearRegression()                  
    model_demand.fit(X, y)
    predict_spot = model.predict(X)

    # Start from the next trading day
    new_start_date = merged_df['Date'].iloc[-1] + BDay(1)
    # Generate trading days to match the same number of row for historical
    future_dates = pd.date_range(start=new_start_date, periods=len(merged_df)+1, freq=BDay())
    
    actual_values = fetch_historical_data(start_date=new_start_date, end_date=future_dates[-1]).reset_index()

    reconstructed_values = [fetch_historical_data(start_date=start_date, end_date=end_date).iloc[-1]]  # Start with the initial value
    for diff in predict_spot:  # Skip the first NaN value
        last_value = reconstructed_values[-1]
        reconstructed_values.append(last_value + diff)

