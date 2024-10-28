import numpy as np
import scipy.stats as stats 
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import streamlit as st

# Black-Scholes function
def black_scholes(stock_price, strike_price, time_to_expiry, interest_rate, volatility):
    d1 = (np.log(stock_price / strike_price) + (interest_rate + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    call_price = stock_price * norm.cdf(d1) - strike_price * np.exp(-interest_rate * time_to_expiry) * norm.cdf(d2)
    put_price = strike_price * np.exp(-interest_rate * time_to_expiry) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)
    return call_price, put_price

# Streamlit UI
st.title("3D Black-Scholes Options Pricer")
st.sidebar.header("Input Parameters")

stock_price = st.sidebar.number_input("Stock Price", min_value=0.0, value=100.0)
strike_price = st.sidebar.number_input("Strike Price", min_value=0.0, value=100.0)
interest_rate = st.sidebar.number_input("Interest Rate", min_value=0.0, value=0.05)
volatility = st.sidebar.number_input("Volatility", min_value=0.0, value=0.2)

# Displaying call and put prices
call_price, put_price = black_scholes(stock_price, strike_price, 1, interest_rate, volatility)
st.write(f"Call Option Price: {call_price}")
st.write(f"Put Option Price: {put_price}")

# Preparing data for 3D plot
stock_prices = np.linspace(50, 150, 50)
time_to_expiry = np.linspace(0.01, 2, 50)
Stock, Time = np.meshgrid(stock_prices, time_to_expiry)

# Calculating call and put prices for the grid
Call_prices = np.array([black_scholes(S, strike_price, T, interest_rate, volatility)[0] for S, T in zip(np.ravel(Stock), np.ravel(Time))]).reshape(Stock.shape)
Put_prices = np.array([black_scholes(S, strike_price, T, interest_rate, volatility)[1] for S, T in zip(np.ravel(Stock), np.ravel(Time))]).reshape(Stock.shape)

# Plotting with Plotly
fig = go.Figure()

# Add call price surface
fig.add_trace(go.Surface(z=Call_prices, x=Stock, y=Time, colorscale='Viridis', opacity=0.7, name="Call Option Price"))
# Add put price surface
fig.add_trace(go.Surface(z=Put_prices, x=Stock, y=Time, colorscale='Cividis', opacity=0.7, name="Put Option Price"))

# Update layout for clarity and style
fig.update_layout(
    title="3D Black-Scholes Option Pricing",
    scene=dict(
        xaxis_title="Stock Price",
        yaxis_title="Time to Expiry",
        zaxis_title="Option Price",
        xaxis=dict(nticks=10, range=[50, 150]),
        yaxis=dict(nticks=10, range=[0, 2]),
        zaxis=dict(nticks=10, range=[0, np.max(Call_prices) + 10]),
    ),
    legend=dict(x=0.1, y=0.9),
    margin=dict(l=0, r=0, b=0, t=50)
)

st.plotly_chart(fig)


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
