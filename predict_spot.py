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
