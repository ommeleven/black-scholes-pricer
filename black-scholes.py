import numpy as np
import yfinance as yf
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import plotly.graph_objects as go

# Black-Scholes function
def black_scholes(stock_price, strike_price, time_to_expiry, interest_rate, volatility):
    d1 = (np.log(stock_price / strike_price) + (interest_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    call_price = stock_price * norm.cdf(d1) - strike_price * np.exp(-interest_rate * time_to_expiry) * norm.cdf(d2)
    put_price = strike_price * np.exp(-interest_rate * time_to_expiry) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)
    return call_price, put_price

# Streamlit UI
st.title("3D Black-Scholes Options Pricer with Real-Time Data")
st.sidebar.header("Input Parameters")

# Get stock ticker from user
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()

# Retrieve stock data
ticker = yf.Ticker(ticker_symbol)
stock_info = ticker.history(period="1d")
stock_price = stock_info["Close"].iloc[-1]

# Retrieve option expiry dates
option_expiries = ticker.options
expiry_date = st.sidebar.selectbox("Select Expiry Date", option_expiries)

# Retrieve options chain for selected expiry
option_chain = ticker.option_chain(expiry_date)
option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])

# Select strike prices based on option type
if option_type == "Call":
    strikes = option_chain.calls['strike']
else:
    strikes = option_chain.puts['strike']

# User selects a strike price
strike_price = st.sidebar.selectbox("Select Strike Price", strikes)
interest_rate = st.sidebar.number_input("Interest Rate", min_value=0.0, value=0.05)
volatility = st.sidebar.number_input("Volatility", min_value=0.0, value=0.2)

# Calculate time to expiry in years
expiry_datetime = np.datetime64(expiry_date)
current_datetime = np.datetime64('today')
time_to_expiry = (expiry_datetime - current_datetime).astype('timedelta64[D]').astype(int) / 365

# Display real-time stock and option data
st.write(f"Stock Price ({ticker_symbol}): {stock_price}")
st.write(f"Selected Strike Price: {strike_price}")
st.write(f"Time to Expiry: {time_to_expiry} years")

# Calculate option prices
call_price, put_price = black_scholes(stock_price, strike_price, time_to_expiry, interest_rate, volatility)
st.write(f"Call Option Price: {call_price}" if option_type == "Call" else f"Put Option Price: {put_price}")

# Preparing data for 3D plot
stock_prices = np.linspace(stock_price * 0.5, stock_price * 1.5, 50)
time_to_expiry_range = np.linspace(0.01, 2, 50)
Stock, Time = np.meshgrid(stock_prices, time_to_expiry_range)

# Calculate option prices for the grid
if option_type == "Call":
    Prices = np.array([black_scholes(S, strike_price, T, interest_rate, volatility)[0] for S, T in zip(np.ravel(Stock), np.ravel(Time))]).reshape(Stock.shape)
else:
    Prices = np.array([black_scholes(S, strike_price, T, interest_rate, volatility)[1] for S, T in zip(np.ravel(Stock), np.ravel(Time))]).reshape(Stock.shape)

# Plotting with Plotly
fig = go.Figure()
fig.add_trace(go.Surface(z=Prices, x=Stock, y=Time, colorscale='Viridis', opacity=0.7, name="Option Price"))

# Update layout for clarity and style
fig.update_layout(
    title=f"3D Black-Scholes {option_type} Option Pricing",
    scene=dict(
        xaxis_title="Stock Price",
        yaxis_title="Time to Expiry",
        zaxis_title="Option Price",
        xaxis=dict(nticks=10, range=[stock_price * 0.5, stock_price * 1.5]),
        yaxis=dict(nticks=10, range=[0, 2]),
        zaxis=dict(nticks=10, range=[0, np.max(Prices) + 10]),
    ),
    margin=dict(l=0, r=0, b=0, t=50)
)

st.plotly_chart(fig)


# Heatmap generation function
def generate_heatmap(strike_price, time_to_expiry, interest_rate):
    # Generate ranges for stock prices and volatilities
    stock_prices = np.linspace(stock_price * 0.5, stock_price * 1.5, 20)
    volatilities = np.linspace(0.1, 0.5, 20)
    
    # Create a 2D array of option prices
    prices = np.array([
        [black_scholes(price, strike_price, time_to_expiry, interest_rate, vol)[0 if option_type == "Call" else 1] for vol in volatilities]
        for price in stock_prices
    ])
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(prices, xticklabels=np.round(volatilities, 2), yticklabels=np.round(stock_prices, 2), cmap="YlGnBu")
    plt.xlabel("Volatility")
    plt.ylabel("Stock Price")
    plt.title(f"{option_type} Option Price Heatmap")
    st.pyplot(plt)

# Call the function with input parameters
generate_heatmap(strike_price, time_to_expiry, interest_rate)


# Data storage function
def store_data(stock_price, strike_price, time_to_expiry, interest_rate, volatility, call_price, put_price):
    conn = sqlite3.connect("options_data.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS options
                      (stock_price REAL, strike_price REAL, time_to_expiry REAL, interest_rate REAL, volatility REAL, call_price REAL, put_price REAL)''')
    cursor.execute("INSERT INTO options VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (stock_price, strike_price, time_to_expiry, interest_rate, volatility, call_price, put_price))
    conn.commit()
    conn.close()

# Store the computed option prices in the database
store_data(stock_price, strike_price, time_to_expiry, interest_rate, volatility, call_price, put_price)
