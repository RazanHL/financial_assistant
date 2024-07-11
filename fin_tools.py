import yfinance as yf
from langchain_community.tools import DuckDuckGoSearchRun
import requests
from bs4 import BeautifulSoup
from datetime import timedelta
import re
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import plotly.graph_objects as go



# Get the ticker name
def get_stock(ticker):
    try:
      stock = yf.Ticker(ticker)
      if stock.history(period="1y").empty:
          ticker=ticker+".F"
          stock = yf.Ticker(ticker)
    except:
      print("Can't find this ticker!")
    return stock

# Fetch stock data from Yahoo Finance
def get_recent_data(ticker, period='5y'):

    stock = get_stock(ticker)
    latest_data =  stock.history(period=period)
    data = latest_data[['Close', 'Volume']]
    # data.index=[str(x).split()[0] for x in list(data.index)]
    # data.index.rename("Date",inplace=True)
    return latest_data #data

def basic_info(symbol):
    stock_data = get_recent_data(symbol)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Close price changes in 5y")
        st.line_chart(stock_data['Close'])

    with col2:
        st.subheader("Volume changes in 5y")
        st.line_chart(stock_data['Volume'])

    # plot some important features
    stock = get_stock(symbol)
    col11, col22, col33 = st.columns(3)

    with col11:
        st.write('**Quarterly cashflow**')
        cashflow_features = ['Operating Cash Flow', 'Capital Expenditure', 'Free Cash Flow', 'Investing Cash Flow', 
                            'Cash Flow From Continuing Financing Activities', 'Net Long Term Debt Issuance']
        cashflow_available_features = []
        for feat in cashflow_features:
            if feat in stock.quarterly_cashflow.index:
                cashflow_available_features.append(feat)


        # st.line_chart(stock.quarterly_cashflow.loc[cashflow_available_features].T)
        fig, axs = plt.subplots(figsize=(10, 10))
        for feat in cashflow_available_features:
            axs.plot(stock.quarterly_cashflow.loc[feat], label=feat)

        plt.ylabel('Quarterly cashflow x100 billions')
        plt.title('Quarterly Cashflow')
        plt.legend()
        st.pyplot(fig)

    with col22:
        st.write('**Quarterly balance sheet**')
        balance_sheet_features = ['Total Assets', 'Total Liabilities Net Minority Interest', 'Stockholders Equity', 
                                'Current Liabilities', 'Working Capital', 'Long Term Debt', 'Total Revenue']
        balance_sheet_available_features = []
        for feat in balance_sheet_features:
            if feat in stock.quarterly_balance_sheet.index:
                balance_sheet_available_features.append(feat)
        fig, axs = plt.subplots(figsize=(10, 10))
        for feat in balance_sheet_available_features:
            axs.plot(stock.quarterly_balance_sheet.loc[feat], label=feat)
        plt.ylabel('Balance Sheet x100 billions')
        plt.title('Balance Sheet')
        plt.legend()
        st.pyplot(fig)

    with col33:
        st.write('**Quarterly financials**')
        financials_features = ['EBIT', 'Net Income', 'Gross Profit', 'Basic EPS', 'Earnings From Equity Interest Net Of Tax']
        financials_available_features = []
        for feat in financials_features:
            if feat in stock.quarterly_financials.index:
                financials_available_features.append(feat)
        fig, axs = plt.subplots(figsize=(10, 10))
        for feat in financials_available_features:
            axs.plot(stock.quarterly_financials.loc[feat], label=feat)

        plt.ylabel('Financials x100 billions')
        plt.title('Financials')

        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.legend()
        st.pyplot(fig)

    return

#valid period = 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max


# def get_stock_price(ticker,history=360):
    
#     stock = get_stock(ticker)
#     df = stock.history(period="max")
#     df=df[["Close","Volume"]]
#     df.index=[str(x).split()[0] for x in list(df.index)]
#     df.index.rename("Date",inplace=True)
#     df=df[-history:]

#     return df #.to_string()

# Script to scrap top5 googgle news for given company name

def creat_candle_chart(ticker):
    df = get_recent_data(ticker)
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

    # fig.show()
    st.plotly_chart(fig)
    return

def google_query(search_term):
    if "news" not in search_term:
        search_term=search_term+ "stock news" 
    url=f"https://www.google.com/search?q={search_term}"
    url=re.sub(r"\s","+",url)
    return url


# @st.cache_resource
def get_recent_stock_news(company_name):
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}

    g_query=google_query("market news and trends"+ company_name)
    res=requests.get(g_query,headers=headers).text
    soup=BeautifulSoup(res,"html.parser")
    news=[]
    for n in soup.find_all("div","kb0PBd cvP2Ce A9Y9g"):
        news.append(n.text)
    for n in soup.find_all("VwiC3b yXK7lf lVm3ye r025kc hJNv6b Hdw6tb"):
        news.append(n.text)

    if len(news)>10:
        news=news[:10]
    else:
        news=news
    news_string=""
    for i,n in enumerate(news):
        news_string+=f"{i}. {n}\n\n"
    top5_news="Recent News:\n\n"+news_string 

    return top5_news

DuckDuck_search=DuckDuckGoSearchRun()

import math

# CLose price forcasting
@st.cache_resource
def arima_forcasting_close(ticker):
    # data = yf.download(ticker, start='2020-01-01', end='2024-07-01')
    data = get_recent_data(ticker)
    decomposition = seasonal_decompose(data['Close'], model='multiplicative', period=int(len(data)/2)) #365)
    fig = decomposition.plot()
    st.pyplot(fig)

    data = data[['Close']]
    data = data.dropna()
    data.index = pd.to_datetime(data.index)

    # fit ARIMA model
    model = ARIMA(data['Close'], order=(5,1,0))
    model_fit = model.fit()

    # Forecast
    forecast_steps = 100
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps+1, inclusive='right')
    forecast_df = pd.DataFrame(forecast.predicted_mean.values, index=forecast_index, columns=['Forecast'])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Close'], label='Actual')
    ax.plot(forecast_df, label='Forecast')
    ax.fill_between(forecast_index,
                    forecast.conf_int().iloc[:, 0],
                    forecast.conf_int().iloc[:, 1],
                    color='k', alpha=.15)
    plt.title(f'{ticker}, Close Forecast')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    st.pyplot(fig)

    return forecast_df[-1:]


def preprocess_data(data, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    return x, y, scaler

@st.cache_resource
def build_train_lstm_model(data, sequence_length=60):
    # data = get_recent_data(ticker)
    data = data[['Close']]

    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    x_train, y_train, scaler = preprocess_data(train_data, sequence_length)
    x_test, y_test, _ = preprocess_data(test_data, sequence_length)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.4))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.4))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    # data = fetch_stock_data(ticker, start_date, end_date)

    # model = build_lstm_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, epochs=50, batch_size=32)

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    rmse = np.sqrt(np.mean(predictions - y_test)**2)

    return model, scaler, rmse, (x_train.shape[1], 1)


def predict_next_stock_price(data, model, scaler, sequence_length=60):

    # data = get_recent_data(ticker)
    data = data[["Close"]] # fetch_stock_data(ticker, start_date, end_date)
    test_data = data[-100:]

    x_test, y_test, _ = preprocess_data(test_data, sequence_length)

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # plot_predictions(test_data[sequence_length:].values, predictions)

    return predictions[-1][0]

def predict_future_prices(model, data, scaler, days=10):
    data = data[['Close']]
    last_index = data.index[-1]
    for _ in range(days):
        pred = predict_next_stock_price(data, model, scaler)
        next_index = data.index[-1]+ timedelta(days=1)
        data.loc[next_index] = {'Close': pred}

    fig, ax = plt.subplots()    
    ax.plot(data.Close, color='blue')
    ax.plot(data.Close[last_index:], color='red')
    st.plotly_chart(fig)

    return data[last_index:]