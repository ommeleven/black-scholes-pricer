import numpy as np
import scipy.stats as stats 
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

# Black-Scholes pricing function
def black_scholes(stock_price, strike_price, time_to_expiry, interest_rate, volatility):
    d1 = (np.log(stock_price / strike_price) + (interest_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    call_price = stock_price * norm.cdf(d1) - strike_price * np.exp(-interest_rate * time_to_expiry) * norm.cdf(d2)
    put_price = strike_price * np.exp(-interest_rate * time_to_expiry) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)
    return call_price, put_price

# Streamlit app setup
st.title("Black-Scholes Options Pricer")
st.sidebar.header("Input Parameters")

# Input fields
stock_price = st.sidebar.number_input("Stock Price", min_value=0.0, value=100.0)
strike_price = st.sidebar.number_input("Strike Price", min_value=0.0, value=100.0)
time_to_expiry = st.sidebar.number_input("Time to Expiry (years)", min_value=0.01, value=1.0)
interest_rate = st.sidebar.number_input("Interest Rate", min_value=0.0, value=0.05)
volatility = st.sidebar.number_input("Volatility", min_value=0.0, value=0.2)

# Calculation and display
call_price, put_price = black_scholes(stock_price, strike_price, time_to_expiry, interest_rate, volatility)
st.write(f"Call Option Price: {call_price}")
st.write(f"Put Option Price: {put_price}")

# Line chart of stock prices vs call and put prices
stock_prices = np.linspace(50, 150, 100)
call_prices = [black_scholes(s, strike_price, time_to_expiry, interest_rate, volatility)[0] for s in stock_prices]
put_prices = [black_scholes(s, strike_price, time_to_expiry, interest_rate, volatility)[1] for s in stock_prices]

plt.figure(figsize=(10, 5))
plt.plot(stock_prices, call_prices, label="Call Price")
plt.plot(stock_prices, put_prices, label="Put Price")
plt.xlabel("Stock Price")
plt.ylabel("Option Price")
plt.legend()
st.pyplot(plt)

# Heatmap for call prices at different volatilities and stock prices
def generate_heatmap(strike_price, time_to_expiry, interest_rate):
    stock_prices = np.linspace(50, 150, 20)
    volatilities = np.linspace(0.1, 0.5, 20)
    call_prices = np.array([[black_scholes(price, strike_price, time_to_expiry, interest_rate, vol)[0] for vol in volatilities] for price in stock_prices])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(call_prices, xticklabels=np.round(volatilities, 2), yticklabels=np.round(stock_prices, 2), cmap="YlGnBu")
    plt.xlabel("Volatility")
    plt.ylabel("Stock Price")
    plt.title("Call Option Price Heatmap")
    st.pyplot(plt)

generate_heatmap(strike_price, time_to_expiry, interest_rate)

# Data storage in SQLite
def store_data(stock_price, strike_price, time_to_expiry, interest_rate, volatility, call_price, put_price):
    conn = sqlite3.connect("options_data.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS options
                      (stock_price REAL, strike_price REAL, time_to_expiry REAL, interest_rate REAL, volatility REAL, call_price REAL, put_price REAL)''')
    cursor.execute("INSERT INTO options VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (stock_price, strike_price, time_to_expiry, interest_rate, volatility, call_price, put_price))
    conn.commit()
    conn.close()

store_data(stock_price, strike_price, time_to_expiry, interest_rate, volatility, call_price, put_price)
