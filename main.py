import pandas as pd
from brokers.forward_broker import ForwardTradingBroker
import numpy as np
from predict_spot import linear_reg, elastic_net
import streamlit as st
from datetime import date
import matplotlib.pyplot as plt


def streamlit_interface(data, start_date, end_date, final_value, news_data):

    st.title("Streamlit Interface")
    st.write("Welcome to your Streamlit app! Use the options below to interact.")
    start_date = st.date_input("Select Start Date", value=start_date)
    end_date = st.date_input("Select End Date", value=end_date)
    cash = st.number_input("Initial Cash", value=10, min_value=0)

    # Validate the selected date range
    if end_date <= start_date:
        st.error("End Date must be later than Start Date. Please adjust your selection.")







if __name__ == "__main__":
    start_date = "2024-09-01"
    end_date = "2024-10-01"
    start_page = 1
    end_page = start_page + 1   # You can set a higher number; the algorithm stops automatically when needed
    news_data = []
    data=[]
    final_value = 0
    # # 1. Initialiser l'interface
    data, start_date, end_date = streamlit_interface(data, start_date, end_date, final_value, news_data)

